import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import random
import tiktoken

# ==========================================================
# DEVICE + SEED
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================================
# POSITIONAL ENCODING
# ==========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ==========================================================
# NEURO-SYMBOLIC CODEBOOK
# ==========================================================

class NeuroSymbolicCodebook(nn.Module):

    def __init__(self, num_symbols, embedding_dim, commitment_cost=0.25):

        super().__init__()

        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_symbols, embedding_dim)

        nn.init.uniform_(
            self.codebook.weight,
            -1 / math.sqrt(embedding_dim),
            1 / math.sqrt(embedding_dim)
        )

    def forward(self, features):

        flat = features.reshape(-1, self.embedding_dim)

        distances = (
            flat.pow(2).sum(1, keepdim=True)
            + self.codebook.weight.pow(2).sum(1)
            - 2 * flat @ self.codebook.weight.t()
        )

        indices = torch.argmin(distances, dim=1)

        quantized = self.codebook(indices).view_as(features)

        e_loss = F.mse_loss(quantized.detach(), features)
        q_loss = F.mse_loss(quantized, features.detach())

        vq_loss = q_loss + self.commitment_cost * e_loss

        # Explainability straight-through estimator
        quantized = features + (quantized - features).detach()

        # Entropy monitoring
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.num_symbols).float()
            probs = one_hot.mean(0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()

        indices = indices.view(features.shape[:-1])

        return quantized, indices, vq_loss, entropy

# ==========================================================
# SUBMODALITY FEATURE EXTRACTOR (RESTORED)
# ==========================================================

class VisualSubmodalityEncoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        # Restore named semantic projection intent
        self.extract_brightness = nn.Linear(input_dim, input_dim)
        self.extract_distance = nn.Linear(input_dim, input_dim)
        self.extract_focus = nn.Linear(input_dim, input_dim)
        self.extract_color = nn.Linear(input_dim, input_dim)
        self.extract_perspective = nn.Linear(input_dim, input_dim)

    def forward(self, x):

        brightness = self.extract_brightness(x)
        distance = self.extract_distance(x)
        focus = self.extract_focus(x)
        color = self.extract_color(x)
        perspective = self.extract_perspective(x)

        return torch.stack(
            [brightness, distance, focus, color, perspective],
            dim=2
        )

# ==========================================================
# MAIN MODEL
# ==========================================================

class SubmodalitySymbolicLanguageModel(nn.Module):

    def __init__(self, vocab_size, d_model=128, num_symbols=256):

        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.sub_encoder = VisualSubmodalityEncoder(d_model)

        # Five independent codebooks
        self.codebooks = nn.ModuleList([
            NeuroSymbolicCodebook(num_symbols, d_model)
            for _ in range(5)
        ])

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.output_head = nn.Linear(d_model, vocab_size)

    # ------------------------------------------------------

    def causal_mask(self, seq_len, device):

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        )

        return mask.masked_fill(mask == 1, float("-inf"))

    # ------------------------------------------------------

    def forward(self, x):

        embed = self.embedding(x)

        sub_features = self.sub_encoder(embed)

        symbolic_vectors = []
        symbolic_indices = []

        total_vq_loss = 0
        total_entropy = 0

        # Restore 5 modality pathway logic
        for i in range(5):

            quantized, indices, vq_loss, entropy = self.codebooks[i](
                sub_features[:, :, i, :]
            )

            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)

            total_vq_loss += vq_loss
            total_entropy += entropy

        symbolic_vectors = torch.stack(symbolic_vectors, dim=2)

        # Restore original composite mental state token logic
        composite = symbolic_vectors.mean(dim=2)

        composite = self.pos_encoder(composite)

        mask = self.causal_mask(composite.size(1), x.device)

        transformer_out = self.transformer(composite, mask)

        logits = self.output_head(transformer_out)

        return logits, torch.stack(symbolic_indices, dim=2), total_vq_loss, total_entropy

    # ------------------------------------------------------

    def generate(self, input_ids, max_new_tokens, seq_len, temperature=0.8):

        self.eval()

        for _ in range(max_new_tokens):

            context = input_ids[:, -seq_len:]

            with torch.no_grad():
                logits, _, _, _ = self(context)

            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ==========================================================
# DATA LOADER RESTORATION
# ==========================================================

def load_data():

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

    if not os.path.exists("data.txt"):
        with open("data.txt", "w", encoding="utf-8") as f:
            f.write(
                "First Citizen:\n"
                "Before we proceed any further, hear me speak.\n\n"
                "All:\nSpeak, speak.\n"
            )

    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    return data, tokenizer, vocab_size

# ==========================================================
# TRAINING LOOP
# ==========================================================

def train_model():

    seq_len = 64
    batch_size = 32
    steps = 1000

    lr = 3e-4
    vq_beta = 0.25
    entropy_beta = 0.01

    data, tokenizer, vocab_size = load_data()

    model = SubmodalitySymbolicLanguageModel(vocab_size).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for step in range(steps):

        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))

        x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)

        optimizer.zero_grad()

        logits, _, vq_loss, entropy = model(x)

        loss_task = criterion(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        total_loss = loss_task + vq_beta * vq_loss - entropy_beta * entropy

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss {loss_task.item():.4f}")

    torch.save(model.state_dict(), "model.pt")

    return model, tokenizer, seq_len

# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    model, tokenizer, seq_len = train_model()

    seed_text = "First Citizen:\nBefore we proceed any further"

    context = torch.tensor(
        [tokenizer.encode(seed_text)],
        dtype=torch.long
    ).to(device)

    generated = model.generate(
        context,
        max_new_tokens=50,
        seq_len=seq_len
    )

    print("\nGenerated Text:\n")
    print(tokenizer.decode(generated[0].tolist()))