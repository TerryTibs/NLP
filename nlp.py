import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
import os

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
        
        submodalities = torch.stack([brightness, distance, focus, color, perspective], dim=2)
        return submodalities

class SubmodalitySymbolicTransformer(nn.Module):
    def __init__(self, input_dim, d_model=16, num_symbols=128, n_heads=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        
        self.submodality_encoder = VisualSubmodalityEncoder(input_dim, d_model)
        self.codebook = NeuroSymbolicCodebook(num_symbols, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        continuous_sub = self.submodality_encoder(x)
        symbolic_vectors, symbolic_indices, vq_loss = self.codebook(continuous_sub)
        
        composite_symbol_tokens = symbolic_vectors.sum(dim=2) 
        composite_symbol_tokens = self.pos_encoder(composite_symbol_tokens)
        
        transformer_out = self.transformer(composite_symbol_tokens)
        pooled_out = transformer_out.mean(dim=1) 
        prediction = self.output_head(pooled_out)
        
        return prediction, symbolic_indices, vq_loss

# ==========================================
# 2. FILE HANDLING & DATA LOADING
# ==========================================

def create_sample_data_file(filepath="data.txt", num_samples=500, seq_len=8, input_dim=16):
    """Generates a dummy data.txt file if one doesn't exist."""
    print(f"[{filepath} not found] -> Auto-generating a sample file for testing...")
    with open(filepath, 'w') as f:
        for _ in range(num_samples):
            # Generate random sensory data for one sequence
            features = torch.randn(seq_len * input_dim).tolist()
            
            # Arbitrary NLP rule: if the sum of features is high, Target = 1 (Intense)
            target = 1.0 if sum(features) > 0 else 0.0
            
            # Write features and target separated by commas
            row = features + [target]
            f.write(','.join(f"{val:.4f}" for val in row) + '\n')
    print(f"Created {filepath} with {num_samples} samples.")

def load_data_from_txt(filepath="data.txt", seq_len=8, input_dim=16):
    """Reads data.txt, parses it, and creates a PyTorch Dataset."""
    print(f"Loading data from {filepath}...")
    X_list = []
    y_list = []
    
    expected_feature_count = seq_len * input_dim
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip(): continue
            
            # Parse comma separated values
            vals = [float(v) for v in line.strip().split(',')]
            
            # The last value is our NLP Emotional Intensity target
            target = vals[-1]
            features = vals[:-1]
            
            if len(features) != expected_feature_count:
                raise ValueError(f"Line {line_num+1}: Expected {expected_feature_count} features, but got {len(features)}. "
                                 f"Check your seq_len ({seq_len}) and input_dim ({input_dim}).")
            
            X_list.append(features)
            y_list.append([target])
            
    # Convert lists to PyTorch Tensors and reshape X to [batch, seq_len, input_dim]
    X_tensor = torch.tensor(X_list, dtype=torch.float32).view(-1, seq_len, input_dim)
    y_tensor = torch.tensor(y_list, dtype=torch.float32)
    
    print(f"Successfully loaded {len(X_tensor)} sequences.")
    return TensorDataset(X_tensor, y_tensor)

# ==========================================
# 3. TRAINING LOOP
# ==========================================

def train_model():
    # Model Hyperparameters
    input_dim = 16   # Dimensionality of the raw sensory input per timestep
    seq_len = 8      # How many timesteps/frames in the mental sequence
    d_model = 16     # Transformer embedding dimension
    num_symbols = 64 # Size of our discrete NLP symbolic dictionary
    batch_size = 16
    epochs = 10
    lr = 0.002
    filepath = "data.txt"

    # 1. Check if file exists, if not, generate it
    if not os.path.exists(filepath):
        create_sample_data_file(filepath, num_samples=500, seq_len=seq_len, input_dim=input_dim)

    # 2. Load the file
    dataset = load_data_from_txt(filepath, seq_len=seq_len, input_dim=input_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Initialize Model
    model = SubmodalitySymbolicTransformer(input_dim=input_dim, d_model=d_model, num_symbols=num_symbols)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    task_criterion = nn.BCELoss() 

    print("\nStarting Training Phase...")
    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_vq_loss = 0, 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions, _, vq_loss = model(batch_x)
            
            t_loss = task_criterion(predictions, batch_y)
            total_loss = t_loss + vq_loss 
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += t_loss.item()
            epoch_vq_loss += vq_loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Emotional Pred Loss: {epoch_loss/len(dataloader):.4f} | VQ NLP Symbol Loss: {epoch_vq_loss/len(dataloader):.4f}")

    return model, input_dim, seq_len

# ==========================================
# 4. INFERENCE DEMO
# ==========================================

def evaluate_and_explain(model, input_dim, seq_len):
    print("\n--- NEURO-SYMBOLIC EXPLAINABILITY DEMO ---")
    model.eval()
    
    # Grab one random sequence to test
    test_input = torch.randn(1, seq_len, input_dim)
    
    with torch.no_grad():
        prediction, symbolic_indices, _ = model(test_input)
    
    intensity = prediction.item()
    print(f"Predicted Emotional Intensity: {intensity:.2f} (0=Calm, 1=Intense)\n")
    
    print("Tracing the Symbolic Reasoning (Explainable AI):")
    submodality_names = ["Brightness", "Distance", "Focus", "Color", "Perspective"]
    
    sequence_symbols = symbolic_indices[0] 
    
    for seq_step in range(sequence_symbols.shape[0]):
        print(f"Time Step {seq_step + 1} Mental Image represented by symbols:")
        for idx, sub_name in enumerate(submodality_names):
            symbol_id = sequence_symbols[seq_step, idx].item()
            print(f"  - {sub_name:12}: Symbol #{symbol_id:03d}")
        print("-" * 35)

if __name__ == "__main__":
    # Train model (reads from data.txt automatically)
    trained_model, i_dim, s_len = train_model()
    
    # Run a test inference
    evaluate_and_explain(trained_model, i_dim, s_len)