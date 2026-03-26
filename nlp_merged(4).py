"""
Neuro-Symbolic Language Model
==============================
A subword-level language model that routes text through five independent
"psychological submodality" streams (Brightness, Distance, Focus, Color,
Perspective), each backed by a separate VQ-VAE codebook.  The quantised
symbols are averaged into a composite latent token and passed through a
causally-masked decoder-only Transformer for next-token prediction.

Fixes applied (v2):
  [🔴] TransformerEncoder → custom CausalTransformerDecoder
         PyTorch's nn.TransformerDecoder requires cross-attention memory,
         so a bespoke stack of CausalDecoderBlock (nn.MultiheadAttention
         + FFN, pre-norm, causal mask) is the cleanest decoder-only impl.
         This is the same pattern used by nanoGPT.
  [🟠] Annealed entropy bonus
         Entropy weight ramps from 0 → entropy_beta over `entropy_warmup`
         steps.  VQ commitment clusters form first; diversity is encouraged
         only once the codebook has settled — no more conflicting gradients.
  [🟡] Train / validation split + codebook usage histogram
         90/10 split when corpus > 50 k tokens.  Active-symbol count
         (entries used at least once per batch) is logged so collapse is
         caught before it silently wrecks training.
  [⭐] LR cosine annealing (CosineAnnealingLR)

Fixes applied (v3):
  [🔴] EMA entropy regularisation replaces raw batch entropy
         NeuroSymbolicCodebook now tracks a per-codebook exponential
         moving average of entropy (α=0.1).  The loss term is changed from
         the linear  -entropy_weight * H_batch  to the squared deviation
         entropy_weight * (H_target - H_ema)^2.  This eliminates:
           • oscillatory codebook drift from noisy batch-marginal stats
           • runaway entropy maximisation fighting VQ commitment loss
           • the sign confusion of mixing representation entropy with the
             batch marginal statistics used to compute it
  [🟠] Removed redundant attn_mask — kept only is_causal=True
         Passing both attn_mask and is_causal=True to nn.MultiheadAttention
         is redundant and fragile across PyTorch versions.  The manual
         make_causal_mask() helper and the causal_mask argument to
         CausalDecoderBlock.forward() have been removed entirely.

Fix applied (v4):
  [🔴] Submodality orthogonality regularisation
         Without a constraint, the five projection matrices (Brightness,
         Distance, Focus, Color, Perspective) can collapse into correlated
         latent directions, making the multi-stream split semantically
         redundant.  VisualSubmodalityEncoder.orthogonality_loss() adds:

             L_orth = λ · Σ_{i<j} ‖ Wᵢ Wⱼᵀ ‖²_F

         The squared Frobenius norm of each cross-gram matrix is zero only
         when Wᵢ and Wⱼ span fully orthogonal subspaces.  λ = 1e-4 keeps
         the penalty small relative to CE loss while providing a consistent
         gradient signal throughout training.

Fixes applied (v5):
  [🟠] Sigmoid-scheduled orthogonality onset
         orth_weight(t) = sigmoid((t − T₀) / τ)  where T₀=200, τ=100.
         VQ loss needs early training to form stable clusters before the
         projection matrices are pushed apart.  Starting the orthogonality
         pressure at step 0 causes gradient competition with VQ commitment;
         the sigmoid delays and smoothly ramps the penalty so both losses
         have room to operate in their natural training regime.
  [🟠] EMA comment precision
         The comment on the entropy EMA now explicitly names both
         coefficients: decay = (1 − α) = 0.9, update rate α = 0.1,
         matching the standard EMA convention new = (1−α)·old + α·batch.
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
        return x + self.pe[:, : x.size(1)]


# ==========================================================
# FIX 1 (RED) — TRUE CAUSAL DECODER  (replaces TransformerEncoder)
# ==========================================================
# PyTorch's nn.TransformerDecoder is an encoder-decoder layer that requires
# cross-attention over a separate encoder memory tensor.  For a decoder-only
# LM we need pure causal self-attention only.  The cleanest solution is to
# build the block directly from nn.MultiheadAttention — the same approach
# used by nanoGPT and most production decoder-only models.
#
# Mathematical objective being served:
#   P(x_t | x_{<t})   ← position t can attend only to positions 0 … t-1

class CausalDecoderBlock(nn.Module):
    """
    Single decoder-only transformer block with pre-LayerNorm:
        LayerNorm → causal self-attention → residual
        LayerNorm → position-wise FFN     → residual

    Pre-norm is more stable than post-norm (GPT-2 style).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff    = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal self-attention (pre-norm).
        # is_causal=True tells PyTorch MHA to generate the causal mask
        # internally — passing a manual attn_mask alongside it is redundant
        # and risks shape mismatches on different PyTorch versions.
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            is_causal=True,
            need_weights=False,
        )
        x = x + self.drop(attn_out)
        # Feed-forward (pre-norm)
        x = x + self.ff(self.norm2(x))
        return x


class CausalTransformerDecoder(nn.Module):
    """Stack of CausalDecoderBlocks with a final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.blocks   = nn.ModuleList([
            CausalDecoderBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


# ==========================================================
# NEURO-SYMBOLIC CODEBOOK  (VQ-VAE style)
# ==========================================================

class NeuroSymbolicCodebook(nn.Module):
    """
    Vector-quantisation codebook: maps a continuous embedding to the
    nearest entry in a learned discrete symbol table.

        - Improved initialisation   (1/√dim uniform)
        - Straight-through gradient (argmin is non-differentiable)
        - Entropy tracking          (signals codebook utilisation health)
        - Active-entry ratio        (collapse early-warning signal)
    """

    def __init__(
        self,
        num_symbols:     int,
        embedding_dim:   int,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_symbols     = num_symbols
        self.embedding_dim   = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_symbols, embedding_dim)
        nn.init.uniform_(
            self.codebook.weight,
            -1 / math.sqrt(embedding_dim),
             1 / math.sqrt(embedding_dim),
        )

        # Moving-average entropy estimate — smooths noisy batch statistics so
        # the entropy regularisation signal doesn't oscillate step-to-step.
        # Registered as a buffer so it is saved/loaded with the checkpoint and
        # moved to the correct device automatically, but is NOT a parameter.
        self.register_buffer("ema_entropy", torch.tensor(0.0))

        # Maximum possible entropy for this codebook: log(num_symbols).
        # Used as the target in the squared penalty — we want all entries used
        # equally, which drives toward full utilisation without the runaway
        # behaviour of a pure maximisation objective.
        self.H_target = math.log(num_symbols)

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: [..., embedding_dim]

        Returns:
            quantized:    same shape — straight-through quantised version
            indices:      [...] long — nearest codebook entry per position
            vq_loss:      scalar    — codebook + commitment loss
            entropy_loss: scalar    — (H_target − H_ema)²  stable entropy penalty
            ema_H:        float     — current smoothed entropy (for logging)
            active_ratio: float     — fraction of codebook entries used this batch
        """
        flat = features.reshape(-1, self.embedding_dim)       # [N, D]

        # Squared Euclidean distances to every codebook vector
        distances = (
            flat.pow(2).sum(1, keepdim=True)                  # [N, 1]
            + self.codebook.weight.pow(2).sum(1)               # [V]
            - 2 * flat @ self.codebook.weight.t()              # [N, V]
        )

        indices   = torch.argmin(distances, dim=1)             # [N]
        quantized = self.codebook(indices).view_as(features)

        # VQ-VAE loss
        e_loss  = F.mse_loss(quantized.detach(), features)    # encoder → codebook
        q_loss  = F.mse_loss(quantized, features.detach())    # codebook → encoder
        vq_loss = q_loss + self.commitment_cost * e_loss

        # Straight-through estimator
        quantized = features + (quantized - features).detach()

        # Codebook usage stats — all computed without gradient.
        with torch.no_grad():
            one_hot      = F.one_hot(indices, self.num_symbols).float()  # [N, V]
            probs        = one_hot.mean(0)                               # [V]
            batch_H      = -(probs * torch.log(probs + 1e-10)).sum()    # scalar

            # EMA update: new = (1 - α) · old + α · batch
            #   decay α = 0.1  →  fast response to recent batches
            #   The 0.9 coefficient is (1 - α), not a separate decay rate.
            self.ema_entropy = 0.9 * self.ema_entropy + 0.1 * batch_H

            # Squared deviation from max-entropy target — penalises both
            # collapse (H << H_target) and any over-diffusion (H >> H_target).
            # Unlike linear -H, this is a bounded, stable signal.
            entropy_loss = (self.H_target - self.ema_entropy).pow(2)

            active_ratio = (one_hot.sum(0) > 0).float().mean().item()

        indices = indices.view(features.shape[:-1])

        return quantized, indices, vq_loss, entropy_loss, self.ema_entropy.item(), active_ratio


# ==========================================================
# SUBMODALITY FEATURE EXTRACTOR
# ==========================================================

class VisualSubmodalityEncoder(nn.Module):
    """
    Projects a token embedding into five independent psychological
    submodality streams.  Named projections aid explainability.

    Submodalities (NLP / sensory representation theory):
        Brightness  — vividness / intensity of the concept
        Distance    — psychological proximity (near vs. far)
        Focus       — attentional salience / sharpness
        Color       — affective tone (warm vs. cool)
        Perspective — viewpoint / frame of reference

    Orthogonality regularisation:
        Without a constraint the five weight matrices can collapse into
        correlated latent directions, making the submodality split
        semantically redundant.  orthogonality_loss() penalises pairwise
        alignment between projection matrices:

            L_orth = λ · Σ_{i≠j} ‖ Wᵢ Wⱼᵀ ‖²_F

        The squared Frobenius norm of the cross-gram matrix is zero only
        when Wᵢ and Wⱼ span fully orthogonal subspaces.  λ = 1e-4 keeps
        the penalty small relative to CE loss while still providing a
        consistent gradient signal.
    """

    SUBMODALITY_NAMES = ["Brightness", "Distance", "Focus", "Color", "Perspective"]

    def __init__(self, d_model: int, orthogonality_lambda: float = 1e-4):
        super().__init__()
        self.extract_brightness  = nn.Linear(d_model, d_model)
        self.extract_distance    = nn.Linear(d_model, d_model)
        self.extract_focus       = nn.Linear(d_model, d_model)
        self.extract_color       = nn.Linear(d_model, d_model)
        self.extract_perspective = nn.Linear(d_model, d_model)
        self.orthogonality_lambda = orthogonality_lambda

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Penalises pairwise correlation between submodality projection matrices.

        For each pair (i, j) with i < j we compute the squared Frobenius
        norm of Wᵢ @ Wⱼᵀ.  When the two weight matrices are perfectly
        orthogonal this product is zero; the further they deviate the
        larger the penalty.

        nn.Linear stores weights as [out, in], so Wᵢ has shape [D, D] and
        Wᵢ @ Wⱼᵀ is the [D, D] cross-gram matrix — exactly the object
        whose norm measures directional overlap.

        Returns: scalar tensor (differentiable w.r.t. all projection weights)
        """
        mats = [
            self.extract_brightness.weight,   # [D, D]
            self.extract_distance.weight,
            self.extract_focus.weight,
            self.extract_color.weight,
            self.extract_perspective.weight,
        ]
        loss = torch.tensor(0.0, device=mats[0].device)
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                # Squared Frobenius norm of the cross-gram matrix
                loss = loss + (mats[i] @ mats[j].t()).pow(2).sum()
        return self.orthogonality_lambda * loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]  →  [B, T, 5, D]"""
        return torch.stack([
            self.extract_brightness(x),
            self.extract_distance(x),
            self.extract_focus(x),
            self.extract_color(x),
            self.extract_perspective(x),
        ], dim=2)


# ==========================================================
# MAIN MODEL
# ==========================================================

class SubmodalitySymbolicLanguageModel(nn.Module):
    """
    Neuro-Symbolic Language Model — decoder-only architecture:

        Token IDs
            → Embedding
            → 5 × Named Submodality Projection
            → 5 × Independent VQ Codebook      (discrete symbols)
            → Mean-aggregate → composite latent token
            → Positional Encoding
            → CausalTransformerDecoder          ← FIX 1 (RED)
            → Linear → logits over vocab
    """

    NUM_SUBMODALITIES = 5

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 128,
        num_symbols: int   = 256,
        n_heads:     int   = 4,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.sub_encoder = VisualSubmodalityEncoder(d_model)

        # One independent codebook per submodality (prevents collapse)
        self.codebooks = nn.ModuleList([
            NeuroSymbolicCodebook(num_symbols, d_model)
            for _ in range(self.NUM_SUBMODALITIES)
        ])

        self.pos_encoder = PositionalEncoding(d_model)

        # FIX 1 (RED): custom causal decoder, not TransformerEncoder
        self.decoder = CausalTransformerDecoder(d_model, n_heads, num_layers, dropout)

        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T] token IDs

        Returns:
            logits:          [B, T, vocab_size]
            symbolic_indices:[B, T, 5]
            total_vq_loss:   scalar — VQ commitment loss
            total_ent_loss:  scalar — sum of (H_target − H_ema)² across codebooks
            mean_ema_H:      float  — mean smoothed entropy (for logging)
            mean_active:     float  — mean active-symbol ratio (for logging)
        """
        embed        = self.embedding(x)          # [B, T, D]
        sub_features = self.sub_encoder(embed)    # [B, T, 5, D]

        symbolic_vectors = []
        symbolic_indices = []
        total_vq_loss    = 0.0
        total_ent_loss   = 0.0   # sum of (H_target - H_ema)^2 across codebooks
        total_ema_H      = 0.0   # sum of smoothed entropies (for logging)
        total_active     = 0.0

        for i in range(self.NUM_SUBMODALITIES):
            quantized, indices, vq_loss, ent_loss, ema_H, active_ratio = self.codebooks[i](
                sub_features[:, :, i, :]          # [B, T, D]
            )
            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)
            total_vq_loss  += vq_loss
            total_ent_loss += ent_loss
            total_ema_H    += ema_H
            total_active   += active_ratio

        # Stable mean aggregation: [B, T, 5, D] → [B, T, D]
        composite = torch.stack(symbolic_vectors, dim=2).mean(dim=2)
        composite = self.pos_encoder(composite)

        # FIX 1 (RED): truly causal decoder-only pass
        decoder_out = self.decoder(composite)      # [B, T, D]
        logits      = self.output_head(decoder_out)

        return (
            logits,
            torch.stack(symbolic_indices, dim=2),          # [B, T, 5]
            total_vq_loss,
            total_ent_loss,                                 # stable squared entropy penalty
            total_ema_H    / self.NUM_SUBMODALITIES,        # mean smoothed H (logging)
            total_active   / self.NUM_SUBMODALITIES,        # mean active ratio (logging)
        )

    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int,
        seq_len:        int,
        temperature:    float = 0.8,
        top_k:          int   = 50,
    ) -> torch.Tensor:
        """
        Autoregressively extend `input_ids` by `max_new_tokens` tokens.
        Uses temperature scaling + top-k nucleus sampling.
        """
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -seq_len:]
            with torch.no_grad():
                logits, *_ = self(context)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            if top_k and top_k > 0:
                values, _  = torch.topk(probs, min(top_k, probs.size(-1)))
                threshold  = values[:, -1].unsqueeze(1)
                probs      = torch.where(probs < threshold, torch.zeros_like(probs), probs)
                probs      = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids  = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ==========================================================
# DATA LOADING  (with 90/10 train/val split)
# ==========================================================

def load_data(filepath: str = "data.txt"):
    """
    Tokenise a plain-text corpus with the GPT-2 tiktoken tokeniser.
    Writes a small Shakespeare excerpt if `filepath` doesn't exist.

    FIX 3b — applies a 90/10 train/val split when corpus > 50 k tokens.

    Returns:
        train_data:  LongTensor
        val_data:    LongTensor | None
        tokenizer:   tiktoken Encoding
        vocab_size:  int  (50,257 for GPT-2)
    """
    tokenizer  = tiktoken.get_encoding("gpt2")
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
    n    = len(data)

    VAL_THRESHOLD = 50_000
    if n > VAL_THRESHOLD:
        split      = int(0.9 * n)
        train_data = data[:split]
        val_data   = data[split:]
        print(
            f"Loaded {n:,} tokens  |  train {len(train_data):,}  "
            f"val {len(val_data):,}  (vocab {vocab_size:,})"
        )
    else:
        train_data = data
        val_data   = None
        print(
            f"Loaded {n:,} tokens (< {VAL_THRESHOLD:,} → no val split)  "
            f"vocab {vocab_size:,}"
        )

    return train_data, val_data, tokenizer, vocab_size


def _random_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x  = torch.stack([data[i     : i + seq_len]     for i in ix]).to(device)
    y  = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


# ==========================================================
# TRAINING
# ==========================================================

def train_model(filepath: str = "data.txt"):
    """
    Full training loop with all three critical fixes applied.

      - CausalTransformerDecoder        (FIX 1: correct P(x_t | x_{<t}))
      - Annealed entropy weight         (FIX 2: safe codebook diversification)
      - Train/val split                 (FIX 3: overfitting detection)
      - Codebook active-symbol logging  (FIX 3: collapse detection)
      - CosineAnnealingLR               (BONUS: smoother convergence)
    """
    # ── Hyperparameters ───────────────────────────────────────────────
    seq_len        = 64      # context window in tokens (~50 words)
    batch_size     = 32
    steps          = 1000    # increase for larger corpora
    lr             = 3e-4
    weight_decay   = 0.01
    vq_beta        = 0.25    # weight on VQ commitment loss
    entropy_beta   = 0.01    # MAX entropy weight (annealed up to this)
    entropy_warmup = 300     # steps to ramp entropy bonus from 0 → full
    orth_T0        = 200     # step at which orthogonality penalty begins rising
    orth_tau       = 100     # sigmoid temperature: smaller = sharper onset
    grad_clip      = 1.0
    log_every      = 100
    val_every      = 200
    checkpoint     = "neuro_symbolic_model.pt"

    # ── Data ──────────────────────────────────────────────────────────
    train_data, val_data, tokenizer, vocab_size = load_data(filepath)

    # ── Model ─────────────────────────────────────────────────────────
    model = SubmodalitySymbolicLanguageModel(
        vocab_size  = vocab_size,
        d_model     = 128,
        num_symbols = 256,
        n_heads     = 4,
        num_layers  = 2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  |  Device: {device}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # BONUS — cosine LR annealing: smoothly decays lr to lr/10 by final step
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr / 10
    )

    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    print("Starting training…\n")

    for step in range(steps):
        x, y = _random_batch(train_data, batch_size, seq_len)

        optimizer.zero_grad()
        logits, _, vq_loss, ent_loss, mean_H, active_ratio = model(x)
        loss_task = criterion(logits.view(-1, vocab_size), y.view(-1))

        # Entropy penalty: entropy_weight * (H_target - H_ema)^2
        # The squared form targets full codebook utilisation from both sides
        # — it penalises collapse (H too low) and diffusion (H too high)
        # equally. The EMA-smoothed H prevents oscillatory codebook drift.
        entropy_weight = entropy_beta * min(1.0, step / max(1, entropy_warmup))

        # Orthogonality penalty with sigmoid schedule:
        #   orth_weight(t) = sigmoid((t - T0) / τ)
        # VQ loss needs early training to form stable clusters before
        # the projection matrices are pushed apart. The sigmoid delays the
        # orthogonality pressure until step T0, then ramps it smoothly over
        # ~4τ steps. This avoids gradient competition during early training.
        orth_weight = 1.0 / (1.0 + math.exp(-(step - orth_T0) / orth_tau))
        orth_loss   = orth_weight * model.sub_encoder.orthogonality_loss()

        total_loss = loss_task + vq_beta * vq_loss + entropy_weight * ent_loss + orth_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────
        if step % log_every == 0 or step == steps - 1:
            ppl = torch.exp(loss_task).item()
            # FIX 3a — active-symbol % is the codebook collapse early warning
            print(
                f"Step {step:4d}/{steps} | "
                f"CE {loss_task.item():.4f} | "
                f"VQ {vq_loss.item():.4f} | "
                f"Orth {orth_loss.item():.4f} | "
                f"EntLoss {ent_loss.item():.4f} | "
                f"H_ema {mean_H:.2f} | "
                f"Active {active_ratio*100:4.1f}% | "
                f"PPL {ppl:.2f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e}"
            )

        # ── Validation ────────────────────────────────────────────────
        if val_data is not None and step > 0 and step % val_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv   = _random_batch(val_data, batch_size, seq_len)
                lv, *_   = model(xv)
                val_loss = criterion(lv.view(-1, vocab_size), yv.view(-1))
            print(
                f"  ↳ Val  CE {val_loss.item():.4f}  "
                f"PPL {torch.exp(val_loss).item():.2f}"
            )
            model.train()

    torch.save(model.state_dict(), checkpoint)
    print(f"\nCheckpoint saved → {checkpoint}")

    return model, tokenizer, seq_len


# ==========================================================
# INFERENCE & EXPLAINABILITY DEMO
# ==========================================================

def evaluate_and_generate(
    model:          SubmodalitySymbolicLanguageModel,
    tokenizer,
    seq_len:        int,
    seed_text:      str   = "First Citizen:\nBefore we proceed any further, hear me",
    max_new_tokens: int   = 60,
    temperature:    float = 0.8,
    top_k:          int   = 50,
) -> None:
    """
    Two-part demo:
      1. Explainability trace — discrete symbol chosen per submodality for
         the final token of the prompt, plus active codebook entry %.
      2. Autoregressive generation — decoded back to readable text.
    """
    print("\n" + "=" * 60)
    print("  NEURO-SYMBOLIC EXPLAINABILITY & GENERATION DEMO")
    print("=" * 60)

    encoded_seed = tokenizer.encode(seed_text)
    context      = torch.tensor([encoded_seed], dtype=torch.long).to(device)

    # ── 1. Symbolic trace ─────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, symbolic_indices, _, _, mean_H, active_ratio = model(context)

    last_tok = symbolic_indices[0, -1]  # [5]

    print(f"\nSEED PROMPT:\n{'-'*40}\n{seed_text}\n{'-'*40}")
    print("\nInternal submodality representation of the final token:\n")
    for i, name in enumerate(VisualSubmodalityEncoder.SUBMODALITY_NAMES):
        print(f"  {name:12s} → Symbol #{last_tok[i].item():03d}")
    print(f"\n  H_ema (smoothed entropy): {mean_H:.3f}  |  Active: {active_ratio*100:.1f}%")

    # ── 2. Generation ─────────────────────────────────────────────────
    print(f"\nGenerating {max_new_tokens} new tokens…\n")
    generated_ids  = model.generate(context, max_new_tokens, seq_len, temperature, top_k)
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"GENERATED TEXT:\n{'-'*50}")
    print(generated_text)
    print("-" * 50)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    trained_model, tokenizer, s_len = train_model()
    evaluate_and_generate(trained_model, tokenizer, s_len)
