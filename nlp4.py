import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random

try:
    import tiktoken
except ImportError:
    print("ERROR: Please install tiktoken by running: pip install tiktoken")
    exit()

# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# POSITIONAL ENCODING
# ==========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ==========================================================
# VECTOR QUANTIZATION (Improved)
# ==========================================================

class NeuroSymbolicCodebook(nn.Module):
    def __init__(self, num_symbols, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_symbols, embedding_dim)

        # Improved initialization
        nn.init.uniform_(self.codebook.weight,
                         -1 / math.sqrt(embedding_dim),
                         1 / math.sqrt(embedding_dim))

    def forward(self, continuous_features):
        flat_inputs = continuous_features.reshape(-1, self.embedding_dim)

        distances = (
            flat_inputs.pow(2).sum(1, keepdim=True)
            + self.codebook.weight.pow(2).sum(1)
            - 2 * flat_inputs @ self.codebook.weight.t()
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices).view_as(continuous_features)

        e_loss = F.mse_loss(quantized.detach(), continuous_features)
        q_loss = F.mse_loss(quantized, continuous_features.detach())
        vq_loss = q_loss + self.commitment_cost * e_loss

        # Straight-through estimator
        quantized = continuous_features + (quantized - continuous_features).detach()

        # Entropy regularization (prevent collapse)
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_symbols).float()
            avg_probs = encodings.mean(0)
            entropy = - (avg_probs * torch.log(avg_probs + 1e-10)).sum()

        encoding_indices = encoding_indices.view(continuous_features.shape[:-1])

        return quantized, encoding_indices, vq_loss, entropy

# ==========================================================
# SUBMODALITY ENCODER
# ==========================================================

class VisualSubmodalityEncoder(nn.Module):
    def __init__(self, input_dim, submodality_dim):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(input_dim, submodality_dim) for _ in range(5)
        ])

    def forward(self, x):
        outputs = [proj(x) for proj in self.projections]
        return torch.stack(outputs, dim=2)

# ==========================================================
# MAIN MODEL (FIXED CAUSAL)
# ==========================================================

class SubmodalitySymbolicLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_symbols=256, n_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model

        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.submodality_encoder = VisualSubmodalityEncoder(d_model, d_model)

        self.codebooks = nn.ModuleList([
            NeuroSymbolicCodebook(num_symbols, d_model) for _ in range(5)
        ])

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        embedded = self.char_embedding(x)
        continuous_sub = self.submodality_encoder(embedded)

        symbolic_vectors = []
        symbolic_indices = []
        total_vq_loss = 0
        total_entropy = 0

        for i in range(5):
            sub_i = continuous_sub[:, :, i, :]
            quantized, indices, vq_loss, entropy = self.codebooks[i](sub_i)

            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)
            total_vq_loss += vq_loss
            total_entropy += entropy

        symbolic_vectors = torch.stack(symbolic_vectors, dim=2)
        symbolic_indices = torch.stack(symbolic_indices, dim=2)

        # Use MEAN instead of SUM (stability)
        composite = symbolic_vectors.mean(dim=2)

        composite = self.pos_encoder(composite)

        seq_len = composite.size(1)
        mask = self.generate_causal_mask(seq_len, x.device)

        transformer_out = self.transformer(composite, mask)

        logits = self.output_head(transformer_out)

        return logits, symbolic_indices, total_vq_loss, total_entropy

    # Improved generation with top-k
    def generate(self, input_ids, max_new_tokens, seq_len, temperature=0.8, top_k=50):
        self.eval()

        for _ in range(max_new_tokens):
            context = input_ids[:, -seq_len:]

            with torch.no_grad():
                logits, _, _, _ = self(context)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            if top_k is not None:
                values, _ = torch.topk(probs, top_k)
                min_values = values[:, -1].unsqueeze(1)
                probs = torch.where(probs < min_values, torch.zeros_like(probs), probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids

# ==========================================================
# TRAINING LOOP (IMPROVED)
# ==========================================================

def train_model():
    seq_len = 128
    batch_size = 32
    d_model = 128
    num_symbols = 256
    training_steps = 5000
    lr = 3e-4
    vq_beta = 0.25
    entropy_beta = 0.01

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = SubmodalitySymbolicLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_symbols=num_symbols
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for step in range(training_steps):

        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)

        optimizer.zero_grad()

        logits, _, vq_loss, entropy = model(x)

        loss_task = criterion(logits.view(-1, vocab_size), y.view(-1))
        total_loss = loss_task + vq_beta * vq_loss - entropy_beta * entropy

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if step % 100 == 0:
            perplexity = torch.exp(loss_task)
            print(f"Step {step} | CE: {loss_task.item():.4f} | "
                  f"VQ: {vq_loss.item():.4f} | "
                  f"Entropy: {entropy.item():.4f} | "
                  f"PPL: {perplexity.item():.2f}")

    torch.save(model.state_dict(), "neuro_symbolic_model.pt")
    return model, tokenizer, seq_len