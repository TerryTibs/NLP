import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

try:
    import tiktoken
except ImportError:
    print("ERROR: Please install tiktoken by running: pip install tiktoken")
    exit()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. NEURO-SYMBOLIC ARCHITECTURE
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
        encoding_indices = encoding_indices.view(continuous_features.shape[:-1])
        
        return quantized, encoding_indices, vq_loss

class VisualSubmodalityEncoder(nn.Module):
    def __init__(self, input_dim, submodality_dim):
        super().__init__()
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
        return torch.stack([brightness, distance, focus, color, perspective], dim=2)

class SubmodalitySymbolicLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_symbols=256, n_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.submodality_encoder = VisualSubmodalityEncoder(d_model, d_model)
        
        # 5 Independent Codebooks (Preventing "Codebook Collapse")
        self.codebooks = nn.ModuleList([
            NeuroSymbolicCodebook(num_symbols, d_model) for _ in range(5)
        ])
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        embedded_text = self.char_embedding(x)
        continuous_sub = self.submodality_encoder(embedded_text)
        
        symbolic_vectors = []
        symbolic_indices = []
        total_vq_loss = 0
        
        for i in range(5):
            sub_i = continuous_sub[:, :, i, :]
            quantized, indices, vq_loss = self.codebooks[i](sub_i)
            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)
            total_vq_loss += vq_loss
            
        symbolic_vectors = torch.stack(symbolic_vectors, dim=2)
        symbolic_indices = torch.stack(symbolic_indices, dim=2)
        
        # Sum the 5 submodalities into a combined mental state token
        composite_symbol_tokens = symbolic_vectors.sum(dim=2) 
        composite_symbol_tokens = self.pos_encoder(composite_symbol_tokens)
        
        transformer_out = self.transformer(composite_symbol_tokens)
        
        # Output logits for all sequence steps
        logits = self.output_head(transformer_out)
        
        return logits, symbolic_indices, total_vq_loss

    def generate(self, input_ids, max_new_tokens, seq_len, temperature=0.8):
        """Autoregressive Generation Loop: Predicts sentences token-by-token"""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max sequence length to avoid out-of-memory or untrained pos embeddings
            context = input_ids[:, -seq_len:]
            
            with torch.no_grad():
                logits, _, _ = self(context)
            
            # Focus only on the last timestep's prediction
            last_logits = logits[:, -1, :] / temperature
            
            # Convert to probabilities
            probs = F.softmax(last_logits, dim=-1)
            
            # Sample the next token based on probabilities
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the running sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
        return input_ids

# ==========================================
# 2. GPT-LEVEL TOKENIZATION & DATA HANDLING
# ==========================================

def load_text_data(filepath="data.txt"):
    """Uses TikToken (GPT-2 tokenizer) for high-efficiency word/subword learning."""
    print(f"Loading text data from {filepath} using TikToken...")
    
    # Initialize GPT-2 standard tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab # 50,257 tokens
    
    if not os.path.exists(filepath):
        print("File not found! Creating a mini data.txt to train on...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n")
            
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Encode the massive text into an array of integers
    encoded_text = tokenizer.encode(text)
    data = torch.tensor(encoded_text, dtype=torch.long)
    
    print(f"Loaded {len(data)} subword tokens. Vocabulary size: {vocab_size}.")
    return data, vocab_size, tokenizer

def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 3. TRAINING LOOP
# ==========================================

def train_model():
    filepath = "data.txt"
    seq_len = 64      # Context window (about ~50 words)
    batch_size = 32
    d_model = 128     # Increased for word-level comprehension
    num_symbols = 256
    training_steps = 1500  # Train a bit longer for subwords
    lr = 0.001

    data, vocab_size, tokenizer = load_text_data(filepath)

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
        
        # Flatten the batches to calculate loss across all sequence tokens
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = batch_y.view(-1)
        
        t_loss = task_criterion(logits_flat, targets_flat)
        total_loss = t_loss + vq_loss 
        
        total_loss.backward()
        optimizer.step()
        
        if step % 100 == 0 or step == training_steps - 1:
            print(f"Step {step:04d}/{training_steps} | CrossEntropy: {t_loss.item():.4f} | VQ Loss: {vq_loss.item():.4f}")

    return model, tokenizer, seq_len

# ==========================================
# 4. INFERENCE, GENERATION & EXPLAINABILITY
# ==========================================

def evaluate_and_generate(model, tokenizer, seq_len):
    print("\n--- NEURO-SYMBOLIC TEXT GENERATION DEMO ---")
    
    seed_text = "First Citizen:\nBefore we proceed any further, hear me"
    print(f"SEED PROMPT:\n{'-'*30}\n{seed_text}\n{'-'*30}\n")
    
    # Tokenize seed text
    encoded_seed = tokenizer.encode(seed_text)
    context = torch.tensor([encoded_seed], dtype=torch.long).to(device)
    
    # 1. Tracing the internal logic of the prompt
    model.eval()
    with torch.no_grad():
        _, symbolic_indices, _ = model(context)
    
    submodality_names = ["Brightness", "Distance", "Focus", "Color", "Perspective"]
    last_word_symbols = symbolic_indices[0, -1]
    
    print("How the AI visualizes the final word of the prompt before generating:")
    for idx, sub_name in enumerate(submodality_names):
        symbol_id = last_word_symbols[idx].item()
        print(f"  - {sub_name:12}: Symbol #{symbol_id:03d}")
    print("\nGenerating a few sentences based on these internal submodalities...\n")
    
    # 2. Autoregressive Generation Loop (Outputting paragraphs)
    # Generate 50 new tokens (~35-45 words)
    generated_tokens = model.generate(context, max_new_tokens=50, seq_len=seq_len, temperature=0.8)
    
    # Decode the integers back into raw text
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    
    print(f"AI GENERATED TEXT:\n{'-'*50}")
    print(generated_text)
    print(f"{'-'*50}\n")

if __name__ == "__main__":
    trained_model, tokenizer, s_len = train_model()
    evaluate_and_generate(trained_model, tokenizer, s_len)