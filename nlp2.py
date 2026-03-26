import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

# Set device (uses GPU if you have one, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. NEURO-SYMBOLIC ARCHITECTURE COMPONENTS
# ==========================================

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
        return x + self.pe[:, :x.size(1), :]

class NeuroSymbolicCodebook(nn.Module):
    def __init__(self, num_symbols, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.codebook = nn.Embedding(num_symbols, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_symbols, 1.0 / num_symbols)

    def forward(self, continuous_features):
        flat_inputs = continuous_features.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.codebook.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.codebook(encoding_indices).view(continuous_features.shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), continuous_features)
        q_latent_loss = F.mse_loss(quantized, continuous_features.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = continuous_features + (quantized - continuous_features).detach()
        # Squeeze the last dimension so it matches the input shape sequence properly
        encoding_indices = encoding_indices.view(continuous_features.shape[:-1])
        
        return quantized, encoding_indices, vq_loss

class VisualSubmodalityEncoder(nn.Module):
    def __init__(self, input_dim, submodality_dim):
        super().__init__()
        # Translates abstract text embeddings into psychological submodalities
        self.extract_brightness = nn.Linear(input_dim, submodality_dim)
        self.extract_distance   = nn.Linear(input_dim, submodality_dim)
        self.extract_focus      = nn.Linear(input_dim, submodality_dim)
        self.extract_color      = nn.Linear(input_dim, submodality_dim)
        self.extract_perspective= nn.Linear(input_dim, submodality_dim) 

    def forward(self, x):
        brightness  = self.extract_brightness(x)
        distance    = self.extract_distance(x)
        focus       = self.extract_focus(x)
        color       = self.extract_color(x)
        perspective = self.extract_perspective(x)
        
        submodalities = torch.stack([brightness, distance, focus, color, perspective], dim=2)
        return submodalities

class SubmodalitySymbolicLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_symbols=256, n_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        # 1. Read text characters and embed them
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Extract Submodalities from the text
        self.submodality_encoder = VisualSubmodalityEncoder(d_model, d_model)
        
        # 3. INDEPENDENT CODEBOOKS UPGRADE
        # We now give each of the 5 NLP submodalities its own private dictionary
        # This prevents "Codebook Collapse" (everything becoming Symbol 163)
        self.codebooks = nn.ModuleList([
            NeuroSymbolicCodebook(num_symbols, d_model) for _ in range(5)
        ])
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 4. Transformer Reasoner
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Predict the next character in the sequence
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len] of character integer IDs
        embedded_text = self.char_embedding(x)
        
        # Extract visual submodalities from text: Shape [batch, seq_len, 5, d_model]
        continuous_sub = self.submodality_encoder(embedded_text)
        
        symbolic_vectors = []
        symbolic_indices = []
        total_vq_loss = 0
        
        # Route each submodality through its OWN independent psychological dictionary
        for i in range(5):
            # sub_i shape: [batch, seq_len, d_model]
            sub_i = continuous_sub[:, :, i, :]
            quantized, indices, vq_loss = self.codebooks[i](sub_i)
            
            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)
            total_vq_loss += vq_loss
            
        # Re-stack them: Shape [batch, seq_len, 5, d_model] and [batch, seq_len, 5]
        symbolic_vectors = torch.stack(symbolic_vectors, dim=2)
        symbolic_indices = torch.stack(symbolic_indices, dim=2)
        
        # Combine the 5 submodalities into one state per timestep
        composite_symbol_tokens = symbolic_vectors.sum(dim=2) 
        composite_symbol_tokens = self.pos_encoder(composite_symbol_tokens)
        
        # Pass through Transformer
        transformer_out = self.transformer(composite_symbol_tokens)
        
        # We only care about the last timestep to predict the NEXT character
        last_timestep_out = transformer_out[:, -1, :] 
        logits = self.output_head(last_timestep_out)
        
        return logits, symbolic_indices, total_vq_loss

# ==========================================
# 2. FILE HANDLING & TEXT DATA LOADING
# ==========================================

def load_text_data(filepath="data.txt"):
    """Reads raw text (like Tiny Shakespeare) and creates a vocabulary."""
    print(f"Loading text data from {filepath}...")
    if not os.path.exists(filepath):
        print("File not found! Creating a mini data.txt to train on...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n")
            
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Mapping characters to integers and vice versa
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the entire text dataset into integers
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    print(f"Loaded {len(data)} characters. Vocabulary size: {vocab_size} unique characters.")
    return data, vocab_size, stoi, itos

def get_batch(data, batch_size, seq_len):
    """Generates a random batch of sequences and their next-character targets."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.tensor([data[i + seq_len] for i in ix], dtype=torch.long)
    return x.to(device), y.to(device)

# ==========================================
# 3. TRAINING LOOP
# ==========================================

def train_model():
    filepath = "data.txt"
    seq_len = 128      # Look at 32 characters of context to predict the 33rd
    batch_size = 64
    d_model = 64
    num_symbols = 512 # Size of our symbolic dictionary (per submodality)
    training_steps = 10000  # Number of batches to train on
    lr = 0.003

    # Load Text Data
    data, vocab_size, stoi, itos = load_text_data(filepath)

    # Initialize Model
    model = SubmodalitySymbolicLanguageModel(
        vocab_size=vocab_size, d_model=d_model, num_symbols=num_symbols
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    task_criterion = nn.CrossEntropyLoss() 

    print("\nStarting Training Phase...")
    model.train()
    
    for step in range(training_steps):
        batch_x, batch_y = get_batch(data, batch_size, seq_len)
        
        optimizer.zero_grad()
        logits, _, vq_loss = model(batch_x)
        
        t_loss = task_criterion(logits, batch_y)
        total_loss = t_loss + vq_loss 
        
        total_loss.backward()
        optimizer.step()
        
        if step % 100 == 0 or step == training_steps - 1:
            print(f"Step {step:04d}/{training_steps} | CrossEntropy Loss: {t_loss.item():.4f} | Total Codebook Loss: {vq_loss.item():.4f}")

    return model, stoi, itos, seq_len

# ==========================================
# 4. INFERENCE & EXPLAINABILITY DEMO
# ==========================================

def evaluate_and_explain(model, stoi, itos, seq_len):
    print("\n--- NEURO-SYMBOLIC TEXT GENERATION DEMO ---")
    model.eval()
    
    # Provide a seed text for the model to think about
    seed_text = "First Citizen:\n"
    
    # Pad or truncate seed text to match seq_len
    if len(seed_text) < seq_len:
        seed_text = seed_text.rjust(seq_len, ' ')
    else:
        seed_text = seed_text[-seq_len:]
        
    print(f"Feeding the AI the seed text: {repr(seed_text.strip())}\n")
    
    context = torch.tensor([stoi[c] for c in seed_text], dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, symbolic_indices, _ = model(context)
        
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_char = itos[predicted_id]
        
    print(f"Predicted NEXT Character: '{predicted_char}'\n")
    
    print("Tracing the Symbolic Reasoning (What Submodalities did the AI extract from the text?):")
    submodality_names = ["Brightness", "Distance", "Focus", "Color", "Perspective"]
    
    # Let's look at how the model represented the LAST character in the sequence
    sequence_symbols = symbolic_indices[0] 
    last_char_symbols = sequence_symbols[-1]
    
    print(f"To predict '{predicted_char}', the AI generated this internal mental state for the end of the text:")
    for idx, sub_name in enumerate(submodality_names):
        symbol_id = last_char_symbols[idx].item()
        print(f"  - {sub_name:12}: Symbol #{symbol_id:03d} (From {sub_name}'s unique dictionary)")
        
    print("\nBecause the text generated this specific sequence of symbolic submodalities,")
    print("the Transformer applied symbolic logic and concluded the next letter must be:", repr(predicted_char))

if __name__ == "__main__":
    trained_model, stoi, itos, s_len = train_model()
    evaluate_and_explain(trained_model, stoi, itos, s_len)