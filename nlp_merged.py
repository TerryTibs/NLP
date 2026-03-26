"""
Neuro-Symbolic Language Model
==============================
A character-/subword-level language model that routes text through five
independent "psychological submodality" streams (Brightness, Distance, Focus,
Color, Perspective), each backed by a separate VQ-VAE codebook.  The
quantised symbols are averaged and passed through a causally-masked
Transformer for next-token prediction.

Merged best parts from: nlp.py, nlp1–7.py
  - tiktoken GPT-2 tokeniser      (nlp3+)
  - set_seed reproducibility       (nlp4+)
  - Named submodality extractors   (nlp7 — best for explainability)
  - Improved codebook init         (nlp4+: 1/sqrt(dim) uniform)
  - Entropy regularisation         (nlp4+: prevents codebook collapse)
  - Causal attention mask          (nlp4+: correct autoregressive setup)
  - AdamW + gradient clipping      (nlp4+)
  - Perplexity in training logs    (nlp4)
  - Top-k sampling in generate()   (nlp4/nlp6)
  - Rich explainability demo       (nlp3 + nlp7)
  - Separated load_data()          (nlp6/nlp7)
  - Model checkpoint saving        (nlp4+)
"""

import math
import os
import random

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# DEVICE & REPRODUCIBILITY
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# ==========================================================
# POSITIONAL ENCODING
# ==========================================================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, : x.size(1)]


# ==========================================================
# NEURO-SYMBOLIC CODEBOOK  (VQ-VAE style)
# ==========================================================

class NeuroSymbolicCodebook(nn.Module):
    """
    Vector-quantisation codebook that maps a continuous embedding to the
    nearest discrete symbol.

    Improvements over the original version:
      - Better weight initialisation  (1/sqrt(dim) uniform)
      - Entropy monitoring            (tracks codebook utilisation)
      - Straight-through estimator    (gradient flows through quantisation)
    """

    def __init__(
        self,
        num_symbols: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()

        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_symbols, embedding_dim)
        nn.init.uniform_(
            self.codebook.weight,
            -1 / math.sqrt(embedding_dim),
            1 / math.sqrt(embedding_dim),
        )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: [..., embedding_dim]  (any leading batch/seq dims)

        Returns:
            quantized:  same shape as `features`  (straight-through)
            indices:    [...] long tensor of nearest codebook entries
            vq_loss:    scalar — codebook + commitment loss
            entropy:    scalar — codebook utilisation (higher = healthier)
        """
        flat = features.reshape(-1, self.embedding_dim)

        # Squared Euclidean distances to every codebook vector
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            + self.codebook.weight.pow(2).sum(1)
            - 2 * flat @ self.codebook.weight.t()
        )

        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices).view_as(features)

        # VQ-VAE loss: push codebook toward encodings + commitment term
        e_loss = F.mse_loss(quantized.detach(), features)
        q_loss = F.mse_loss(quantized, features.detach())
        vq_loss = q_loss + self.commitment_cost * e_loss

        # Straight-through estimator — gradients bypass the argmin
        quantized = features + (quantized - features).detach()

        # Entropy over codebook usage (no gradient needed)
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.num_symbols).float()
            probs = one_hot.mean(0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()

        indices = indices.view(features.shape[:-1])

        return quantized, indices, vq_loss, entropy


# ==========================================================
# SUBMODALITY FEATURE EXTRACTOR
# ==========================================================

class VisualSubmodalityEncoder(nn.Module):
    """
    Projects a token embedding into five independent psychological
    submodality streams.  Named projections make the model more
    interpretable during the explainability demo.

    Submodalities (inspired by NLP / sensory representation theory):
        Brightness  — how vivid / intense the concept is
        Distance    — psychological proximity (close vs. far)
        Focus       — sharpness / salience of attention
        Color       — affective tone (warm vs. cool)
        Perspective — viewpoint / frame of reference
    """

    SUBMODALITY_NAMES = [
        "Brightness",
        "Distance",
        "Focus",
        "Color",
        "Perspective",
    ]

    def __init__(self, d_model: int):
        super().__init__()

        self.extract_brightness  = nn.Linear(d_model, d_model)
        self.extract_distance    = nn.Linear(d_model, d_model)
        self.extract_focus       = nn.Linear(d_model, d_model)
        self.extract_color       = nn.Linear(d_model, d_model)
        self.extract_perspective = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, 5, d_model]
        """
        return torch.stack(
            [
                self.extract_brightness(x),
                self.extract_distance(x),
                self.extract_focus(x),
                self.extract_color(x),
                self.extract_perspective(x),
            ],
            dim=2,
        )


# ==========================================================
# MAIN MODEL
# ==========================================================

class SubmodalitySymbolicLanguageModel(nn.Module):
    """
    Neuro-Symbolic Language Model pipeline:

        Token IDs
            → Embedding
            → 5 × Submodality Projection
            → 5 × Independent VQ Codebook   (discrete symbols)
            → Mean-aggregate composite token
            → Positional Encoding
            → Causal Transformer Encoder
            → Linear → logits over vocab
    """

    NUM_SUBMODALITIES = 5

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_symbols: int = 256,
        n_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.d_model = d_model

        # ── Embedding ──────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model)

        # ── Submodality streams ────────────────────────────────────────────
        self.sub_encoder = VisualSubmodalityEncoder(d_model)

        # One independent codebook per submodality (prevents collapse)
        self.codebooks = nn.ModuleList(
            [
                NeuroSymbolicCodebook(num_symbols, d_model)
                for _ in range(self.NUM_SUBMODALITIES)
            ]
        )

        # ── Transformer ────────────────────────────────────────────────────
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Output head ────────────────────────────────────────────────────
        self.output_head = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------------------
    # CAUSAL MASK
    # ------------------------------------------------------------------

    def causal_mask(self, seq_len: int, dev: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position i cannot attend to i+1…T."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=dev), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, seq_len]  — token IDs

        Returns:
            logits:          [batch, seq_len, vocab_size]
            symbolic_indices:[batch, seq_len, 5]   — nearest codebook symbol per submodality
            total_vq_loss:   scalar
            total_entropy:   scalar
        """
        embed = self.embedding(x)                     # [B, T, D]
        sub_features = self.sub_encoder(embed)        # [B, T, 5, D]

        symbolic_vectors = []
        symbolic_indices = []
        total_vq_loss = 0.0
        total_entropy = 0.0

        for i in range(self.NUM_SUBMODALITIES):
            quantized, indices, vq_loss, entropy = self.codebooks[i](
                sub_features[:, :, i, :]              # [B, T, D]
            )
            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)
            total_vq_loss += vq_loss
            total_entropy += entropy

        # [B, T, 5, D]  — stable mean aggregation (vs. sum)
        composite = torch.stack(symbolic_vectors, dim=2).mean(dim=2)

        composite = self.pos_encoder(composite)

        mask = self.causal_mask(composite.size(1), x.device)
        transformer_out = self.transformer(composite, mask)

        logits = self.output_head(transformer_out)

        return (
            logits,
            torch.stack(symbolic_indices, dim=2),   # [B, T, 5]
            total_vq_loss,
            total_entropy,
        )

    # ------------------------------------------------------------------
    # GENERATION  (autoregressive, temperature + top-k)
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        seq_len: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressively extend `input_ids` by `max_new_tokens` tokens.

        Args:
            input_ids:       [1, T]  — seed token IDs
            max_new_tokens:  how many tokens to sample
            seq_len:         max context window length
            temperature:     > 1 = more random, < 1 = more peaked
            top_k:           restrict sampling to top-k probability mass

        Returns:
            [1, T + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            context = input_ids[:, -seq_len:]

            with torch.no_grad():
                logits, _, _, _ = self(context)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            if top_k and top_k > 0:
                values, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                threshold = values[:, -1].unsqueeze(1)
                probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ==========================================================
# DATA LOADING
# ==========================================================

def load_data(filepath: str = "data.txt"):
    """
    Load and tokenise a plain-text corpus using the GPT-2 tiktoken tokeniser.

    If `filepath` does not exist, a small Shakespeare excerpt is written
    automatically so the script can still run end-to-end out of the box.

    Returns:
        data:       1-D LongTensor of token IDs
        tokenizer:  tiktoken Encoding object
        vocab_size: int  (50,257 for GPT-2)
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    if not os.path.exists(filepath):
        print(f"[{filepath} not found] Writing a mini Shakespeare excerpt…")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(
                "First Citizen:\n"
                "Before we proceed any further, hear me speak.\n\n"
                "All:\nSpeak, speak.\n\n"
                "First Citizen:\nYou are all resolved rather to die than to famish?\n\n"
                "All:\nResolved. resolved.\n\n"
                "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n\n"
                "All:\nWe know't, we know't.\n"
            )

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(
        f"Loaded {len(data):,} tokens from '{filepath}'  "
        f"(vocab size: {vocab_size:,})"
    )
    return data, tokenizer, vocab_size


# ==========================================================
# TRAINING
# ==========================================================

def train_model(filepath: str = "data.txt"):
    """
    Full training loop.

    Key training choices (merged from nlp4–nlp7):
      - AdamW with weight decay     (better regularisation than Adam)
      - Gradient clipping at 1.0    (stabilises early training)
      - Entropy bonus in loss       (encourages codebook diversity)
      - Perplexity logged each step (human-readable quality metric)
      - Checkpoint saved to disk    (resume / inference later)
    """
    # ── Hyperparameters ───────────────────────────────────────────────
    seq_len        = 64     # Context window in tokens (~50 words)
    batch_size     = 32
    steps          = 1000   # Increase for a larger corpus
    lr             = 3e-4
    weight_decay   = 0.01
    vq_beta        = 0.25   # Weight on codebook loss
    entropy_beta   = 0.01   # Weight on entropy bonus (higher = more diverse symbols)
    grad_clip      = 1.0
    log_every      = 100
    checkpoint     = "neuro_symbolic_model.pt"

    # ── Data ──────────────────────────────────────────────────────────
    data, tokenizer, vocab_size = load_data(filepath)

    # ── Model ─────────────────────────────────────────────────────────
    model = SubmodalitySymbolicLanguageModel(
        vocab_size=vocab_size,
        d_model=128,
        num_symbols=256,
        n_heads=4,
        num_layers=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}  |  Device: {device}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    print("Starting training…\n")

    for step in range(steps):
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)

        optimizer.zero_grad()

        logits, _, vq_loss, entropy = model(x)

        loss_task  = criterion(logits.view(-1, vocab_size), y.view(-1))
        total_loss = loss_task + vq_beta * vq_loss - entropy_beta * entropy

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            ppl = torch.exp(loss_task).item()
            print(
                f"Step {step:4d}/{steps} | "
                f"CE {loss_task.item():.4f} | "
                f"VQ {vq_loss.item():.4f} | "
                f"Entropy {entropy.item():.2f} | "
                f"PPL {ppl:.2f}"
            )

    torch.save(model.state_dict(), checkpoint)
    print(f"\nCheckpoint saved → {checkpoint}")

    return model, tokenizer, seq_len


# ==========================================================
# INFERENCE & EXPLAINABILITY DEMO
# ==========================================================

def evaluate_and_generate(
    model: SubmodalitySymbolicLanguageModel,
    tokenizer,
    seq_len: int,
    seed_text: str = "First Citizen:\nBefore we proceed any further, hear me",
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int = 50,
) -> None:
    """
    Two-part demo:
      1. Explainability trace — shows which discrete symbol each submodality
         selected for the final token of the prompt.
      2. Text generation      — autoregressively samples new tokens and
         decodes them back to readable text.
    """
    print("\n" + "=" * 60)
    print("  NEURO-SYMBOLIC EXPLAINABILITY & GENERATION DEMO")
    print("=" * 60)

    # ── Tokenise seed ─────────────────────────────────────────────────
    encoded_seed = tokenizer.encode(seed_text)
    context = torch.tensor([encoded_seed], dtype=torch.long).to(device)

    # ── 1. Symbolic trace ─────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, symbolic_indices, _, _ = model(context)

    # Indices of the last token in the prompt  →  [5]
    last_token_symbols = symbolic_indices[0, -1]

    print(f"\nSEED PROMPT:\n{'-'*40}\n{seed_text}\n{'-'*40}")
    print(
        "\nHow the model internally represents the final word "
        "before generating:\n"
    )
    for i, name in enumerate(VisualSubmodalityEncoder.SUBMODALITY_NAMES):
        sym_id = last_token_symbols[i].item()
        print(f"  {name:12s} → Symbol #{sym_id:03d}  (from {name}'s codebook)")

    # ── 2. Autoregressive generation ──────────────────────────────────
    print(f"\nGenerating {max_new_tokens} new tokens…\n")

    generated_ids = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        seq_len=seq_len,
        temperature=temperature,
        top_k=top_k,
    )

    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"GENERATED TEXT:\n{'-'*50}")
    print(generated_text)
    print("-" * 50)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    # Train (reads data.txt, creates it if missing)
    trained_model, tokenizer, s_len = train_model()

    # Demonstrate explainability + generation
    evaluate_and_generate(trained_model, tokenizer, s_len)
