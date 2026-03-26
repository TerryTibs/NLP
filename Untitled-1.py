"""
Neuro-Symbolic Language Model
==============================
A subword-level language model that routes text through five independent
"psychological submodality" streams (Brightness, Distance, Focus, Color,
Perspective), each backed by a separate VQ-VAE codebook.  The quantised
symbols are averaged into a composite latent token and passed through a
causally-masked decoder-only Transformer for next-token prediction.
"""

import math
import os
import random
import time

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# ==========================================================
# DEVICE & REPRODUCIBILITY
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN ops — slightly slower but fully reproducible.
    # benchmark=False stops cuDNN from auto-selecting non-deterministic kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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
        self.register_buffer("pe", pe.unsqueeze(0))  #[1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ==========================================================
# CAUSAL DECODER
# ==========================================================

class CausalDecoderBlock(nn.Module):
    """
    Single decoder-only transformer block with pre-LayerNorm.
    Uses F.scaled_dot_product_attention (Flash Attention on CUDA) instead of
    nn.MultiheadAttention — same math, ~2-4x faster and half the memory.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.d_model  = d_model
        self.attn_drop = dropout   # passed to SDPA; updated by set_dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Separate Q/K/V and output projections (no bias — standard practice)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ff   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        normed   = self.norm1(x)

        # Project and reshape to [B, n_heads, T, d_head]
        q = self.q_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Flash Attention (is_causal=True handles the mask internally)
        attn_drop_p = self.attn_drop if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=attn_drop_p, is_causal=True
        )                                              #[B, n_heads, T, d_head]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn_out = self.o_proj(attn_out)

        x = x + self.drop(attn_out)
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

    def add_layer(self, device: torch.device) -> CausalDecoderBlock:
        """
        Append a new transformer block at runtime.
        The new block's weights are uninitialised (PyTorch default).
        Returns the new block so the caller can add its params to the optimizer.
        """
        d_model = self.blocks[0].norm1.normalized_shape[0]
        n_heads = self.blocks[0].attn.num_heads
        dropout = self.blocks[0].drop.p
        new_block = CausalDecoderBlock(d_model, n_heads, dropout).to(device)
        self.blocks.append(new_block)
        return new_block

    def set_dropout(self, p: float) -> None:
        """
        Update dropout probability in all blocks at runtime.
        Called during training when overfitting is detected.
        """
        p = float(max(0.0, min(0.5, p)))
        for block in self.blocks:
            block.drop.p     = p
            block.attn_drop  = p   # Flash Attention dropout
            for m in block.ff.modules():
                if isinstance(m, nn.Dropout):
                    m.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # is_causal=True inside each block tells Flash Attention to apply the
        # causal mask natively in the CUDA kernel — no explicit mask needed.
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
    """

    def __init__(
        self,
        num_symbols:     int,
        embedding_dim:   int,
        commitment_cost: float = 0.25,
        ema_decay:       float = 0.2,   # starts fast; annealed to 0.05 during training
        topk:            int   = 3,     # Top-K symbol routing: K symbols blended per token
        topk_tau:        float = 0.1,   # softmax temperature for blending weights
    ):
        super().__init__()
        self.num_symbols     = num_symbols
        self.embedding_dim   = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay       = ema_decay
        self.topk            = topk     # number of symbols blended per token
        self.topk_tau        = topk_tau # temperature for top-K softmax blending

        self.codebook = nn.Embedding(num_symbols, embedding_dim)
        nn.init.uniform_(
            self.codebook.weight,
            -1 / math.sqrt(embedding_dim),
             1 / math.sqrt(embedding_dim),
        )

        self.residual_alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("ema_entropy", torch.tensor(0.0, dtype=torch.float32))
        self.H_target = 0.9 * math.log(num_symbols)

    def grow_codebook(self, new_num_symbols: int, device: torch.device) -> None:
        """
        Expand the codebook from num_symbols → new_num_symbols at runtime.
        New entries are initialised uniformly in the same range as the original.
        The existing entries and H_target are updated in-place.
        """
        if new_num_symbols <= self.num_symbols:
            return
        n_new = new_num_symbols - self.num_symbols
        new_w = torch.empty(n_new, self.embedding_dim, device=device)
        nn.init.uniform_(new_w, -1 / math.sqrt(self.embedding_dim),
                                 1 / math.sqrt(self.embedding_dim))
        combined = torch.cat([self.codebook.weight.data, new_w], dim=0)
        self.codebook = nn.Embedding(new_num_symbols, self.embedding_dim).to(device)
        self.codebook.weight.data.copy_(combined)
        self.num_symbols = new_num_symbols
        self.H_target    = 0.9 * math.log(new_num_symbols)
        print(f"    ↑ Codebook grown: {self.num_symbols - n_new} → {new_num_symbols} symbols")

    def revive_dead_entries(self, device: torch.device, noise_scale: float = 0.1) -> int:
        """
        Reset dead codebook entries (never selected in recent batch) to
        small random perturbations of a randomly chosen live entry.
        Returns number of entries revived.
        """
        with torch.no_grad():
            W       = self.codebook.weight.data          # [K, D]
            norms   = (W ** 2).sum(dim=1)                # proxy: entries near origin are dead
            # Entries with very small norm vs median are likely unused / collapsed
            med_norm = norms.median()
            dead_mask = norms < (med_norm * 0.1)         # threshold: <10% of median norm
            n_dead   = dead_mask.sum().item()
            if n_dead == 0:
                return 0
            live_idx = (~dead_mask).nonzero(as_tuple=True)[0]
            if len(live_idx) == 0:
                return 0
            # For each dead entry, copy a random live entry + small noise
            for dead_i in dead_mask.nonzero(as_tuple=True)[0]:
                src = live_idx[torch.randint(len(live_idx), (1,)).item()]
                W[dead_i] = W[src] + torch.randn_like(W[src]) * noise_scale
            return int(n_dead)

    def forward(self, features: torch.Tensor):
        flat      = features.reshape(-1, self.embedding_dim)          # [BT, D]
        distances = torch.cdist(flat, self.codebook.weight, p=2)      # [BT, K_sym]

        # ── Top-K Symbol Routing ──────────────────────────────────────────
        k_eff               = min(self.topk, self.num_symbols)
        topk_dists, topk_idx = distances.topk(k_eff, dim=1, largest=False)  # [BT, k_eff]
        topk_weights        = F.softmax(-topk_dists / self.topk_tau, dim=1) # [BT, k_eff]

        # Weighted sum of top-K centroid vectors
        topk_vecs  = self.codebook(topk_idx)                                # [BT, k_eff, D]
        quantized  = (topk_weights.unsqueeze(-1) * topk_vecs).sum(dim=1)   #[BT, D]
        quantized  = quantized.view_as(features)                            #[B, T, D]

        # Commitment losses — encoder side: weighted distance to selected centroids
        e_loss  = (topk_weights.detach() * topk_dists).sum(dim=1).mean()
        q_loss  = F.mse_loss(quantized, features.detach())

        # Within-codebook diversity loss: push centroids apart from each other.
        n_sample   = min(self.num_symbols, 64)
        sample_idx = torch.randperm(self.num_symbols, device=self.codebook.weight.device)[:n_sample]
        W_sample   = self.codebook.weight[sample_idx]
        W_norms    = W_sample.norm(dim=1, keepdim=True).clamp(min=1e-8)  # never divide by 0
        W_sample   = W_sample / W_norms                                   # safe normalise
        cos_sim    = torch.mm(W_sample, W_sample.t())                     # [n_sample, n_sample]
        mask = ~torch.eye(n_sample, dtype=torch.bool, device=cos_sim.device)
        diversity_loss = cos_sim[mask].mean().clamp(min=0)
        
        # Guard: if diversity_loss is somehow NaN (e.g. all-zero codebook), skip it
        if torch.isnan(diversity_loss):
            diversity_loss = torch.zeros(1, device=features.device)[0]
        vq_loss = q_loss + self.commitment_cost * e_loss + 0.01 * diversity_loss

        # Straight-through: gradients flow to features, quantized detached
        quantized_st = features + (quantized - features).detach()
        alpha        = 0.5 * torch.sigmoid(self.residual_alpha_raw)
        quantized    = quantized_st + alpha * features

        with torch.no_grad():
            # Count ALL K selected symbols per token for utilisation tracking
            flat_idx  = topk_idx.flatten()                               # [BT * k_eff]
            one_hot   = F.one_hot(flat_idx, self.num_symbols).float()
            probs     = one_hot.mean(0)
            batch_H   = -(probs * torch.log(probs + 1e-10)).sum()
            self.ema_entropy.mul_(1.0 - self.ema_decay).add_(batch_H * self.ema_decay)
            # active = any symbol touched by ANY of the K selections
            active_ratio = (one_hot.sum(0) > 0).float().mean().item()
            # Top-K blending softness: entropy of the K weights per token, averaged.
            topk_H = -(topk_weights * torch.log(topk_weights + 1e-10)).sum(dim=1).mean().item()

        # Return shape [B, T, k_eff] — downstream histogram flattens all K naturally
        indices = topk_idx.view(*features.shape[:-1], k_eff)
        return quantized, indices, vq_loss, self.ema_entropy.item(), active_ratio, topk_H


# ==========================================================
# SUBMODALITY FEATURE EXTRACTOR
# ==========================================================

class VisualSubmodalityEncoder(nn.Module):
    """
    Projects a token embedding into five independent psychological
    submodality streams.  Named projections aid explainability.
    """

    # Read-only reference — never mutate this directly.
    BASE_SUBMODALITY_NAMES = ["Brightness", "Distance", "Focus", "Color",
                               "Perspective", "Size", "Position"]

    def __init__(self, d_model: int, orthogonality_lambda: float = 0.01):
        super().__init__()
        # Instance-level list — safe to mutate as streams are added at runtime
        self.submodality_names = list(self.BASE_SUBMODALITY_NAMES)
        self.extract_brightness  = nn.Linear(d_model, d_model)
        self.extract_distance    = nn.Linear(d_model, d_model)
        self.extract_focus       = nn.Linear(d_model, d_model)
        self.extract_color       = nn.Linear(d_model, d_model)
        self.extract_perspective = nn.Linear(d_model, d_model)
        self.extract_size        = nn.Linear(d_model, d_model)
        self.extract_position    = nn.Linear(d_model, d_model)
        self.orthogonality_lambda = orthogonality_lambda

    def orthogonality_loss(self) -> torch.Tensor:
        """Weight-space orthogonality."""
        mats =[
            getattr(self, f"extract_{n.lower()}").weight
            for n in self.submodality_names
        ]
        D    = mats[0].shape[0]              # embedding dimension
        loss = torch.tensor(0.0, device=mats[0].device)
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                loss = loss + (mats[i] @ mats[j].t()).pow(2).sum() / (D * D)
        return self.orthogonality_lambda * loss

    def feature_orthogonality_loss(self, sub_features: torch.Tensor) -> torch.Tensor:
        """Feature-space orthogonality."""
        B, T, _, D = sub_features.shape
        N          = B * T                   # total token count in the batch
        loss       = torch.tensor(0.0, device=sub_features.device)

        for i in range(len(self.submodality_names)):
            fi = sub_features[:, :, i, :].reshape(N, D)      # [N, D]
            fi = fi - fi.mean(0, keepdim=True)                # remove mean bias
            fi = F.normalize(fi, dim=1)                       # unit norm → cosine
            for j in range(i + 1, len(self.submodality_names)):
                fj = sub_features[:, :, j, :].reshape(N, D)  # [N, D]
                fj = fj - fj.mean(0, keepdim=True)
                fj = F.normalize(fj, dim=1)
                cross_gram = (fi.t() @ fj) / N               # [D, D]
                loss = loss + cross_gram.pow(2).mean()

        return self.orthogonality_lambda * loss

    def add_stream(self, name: str, d_model: int, device: torch.device) -> nn.Linear:
        """
        Dynamically add a new submodality projection at runtime.
        """
        attr = f"extract_{name.lower()}"
        if hasattr(self, attr):
            raise ValueError(f"Stream '{name}' already exists.")
        new_lin = nn.Linear(d_model, d_model).to(device)
        setattr(self, attr, new_lin)
        self.submodality_names.append(name)
        return new_lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]  →  [B, T, N, D]  where N = NUM_SUBMODALITIES"""
        streams =[getattr(self, f"extract_{n.lower()}")(x)
                   for n in self.submodality_names]
        return torch.stack(streams, dim=2)


# ==========================================================
# MAIN MODEL
# ==========================================================

class SubmodalitySymbolicLanguageModel(nn.Module):
    """
    Neuro-Symbolic Language Model — decoder-only architecture:
    """

    NUM_SUBMODALITIES = 7

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 384,
        num_symbols: int   = 512,
        n_heads:     int   = 4,
        num_layers:  int   = 8,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Fail fast with a clear message rather than crashing deep inside
        if d_model % n_heads != 0:
            valid =[h for h in range(1, d_model + 1) if d_model % h == 0]
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).\n"
                f"Valid n_heads for d_model={d_model}: {valid}"
            )

        self.d_model = d_model

        self.embedding   = nn.Embedding(vocab_size, d_model)
        self.sub_encoder = VisualSubmodalityEncoder(d_model)

        # One independent codebook per submodality (prevents collapse)
        self.codebooks = nn.ModuleList([
            NeuroSymbolicCodebook(num_symbols, d_model)
            for _ in range(self.NUM_SUBMODALITIES)
        ])

        self.pre_vq_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(self.NUM_SUBMODALITIES)
        ])

        self.pos_encoder = PositionalEncoding(d_model)

        self.submodality_gate = nn.Linear(d_model, self.NUM_SUBMODALITIES)
        self.decoder = CausalTransformerDecoder(d_model, n_heads, num_layers, dropout)
        self.output_head = nn.Linear(d_model, vocab_size)

        self.stream_active: list[bool] = [True] * self.NUM_SUBMODALITIES
        self._update_stream_mask()

    def _update_stream_mask(self) -> None:
        """Rebuild the float mask tensor from stream_active list."""
        dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        mask = torch.tensor(
            [1.0 if a else 0.0 for a in self.stream_active],
            dtype=torch.float32, device=dev,
        )
        self.register_buffer("stream_mask", mask)

    def add_submodality(
        self, name: str, device: torch.device, optimizer: torch.optim.Optimizer,
        lr: float, weight_decay: float, num_symbols: int | None = None,
        boost_lr: float | None = None,
    ) -> list:
        """
        Dynamically grow the model by one new submodality stream at runtime.
        """
        if num_symbols is None:
            num_symbols = self.codebooks[0].num_symbols
        d   = self.d_model
        dev = device

        # 1. encoder projection
        new_lin = self.sub_encoder.add_stream(name, d, dev)

        # 2. codebook
        new_cb = NeuroSymbolicCodebook(num_symbols, d).to(dev)
        self.codebooks.append(new_cb)

        # 3. pre-VQ norm
        new_norm = nn.LayerNorm(d).to(dev)
        self.pre_vq_norms.append(new_norm)

        # 4. grow gate
        old_gate = self.submodality_gate
        new_N    = self.NUM_SUBMODALITIES + 1
        new_gate = nn.Linear(d, new_N).to(dev)
        with torch.no_grad():
            new_gate.weight[:self.NUM_SUBMODALITIES] = old_gate.weight
            new_gate.bias[:self.NUM_SUBMODALITIES]   = old_gate.bias
            nn.init.zeros_(new_gate.weight[self.NUM_SUBMODALITIES:])
            new_gate.bias.data[self.NUM_SUBMODALITIES] = 0.5
        self.submodality_gate = new_gate

        # 5. add new params to optimizer and fully clear old gate from state to prevent VRAM leak
        new_params = (
            list(new_lin.parameters())
            + list(new_cb.parameters())
            + list(new_norm.parameters())
            + list(new_gate.parameters())
        )
        
        old_gate_params = list(old_gate.parameters())
        old_gate_ids = {id(p) for p in old_gate_params}
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in old_gate_ids]
        
        # Free Adam momentum buffers for the destroyed gate weights
        for p in old_gate_params:
            optimizer.state.pop(p, None)
            
        optimizer.add_param_group({
            "params":       new_params,
            "lr":           boost_lr if boost_lr is not None else lr,
            "weight_decay": weight_decay,
        })

        # 6. state
        self.NUM_SUBMODALITIES += 1
        self.stream_active.append(True)
        self._update_stream_mask()
        print(f"  ✚ Stream added: '{name}'  "
              f"(total active: {sum(self.stream_active)}/{self.NUM_SUBMODALITIES})")
        return new_params

    def deactivate_submodality(self, idx: int) -> None:
        """
        Deactivate stream `idx` — gate contribution masked to zero.
        """
        name = self.sub_encoder.submodality_names[idx]
        self.stream_active[idx] = False
        self._update_stream_mask()
        active = sum(self.stream_active)
        print(f"  ✖ Stream deactivated: '{name}' (idx {idx})  "
              f"({active}/{self.NUM_SUBMODALITIES} active)")

    def kickstart_worst_pair(
        self,
        i: int, j: int,
        device: torch.device,
        noise_scale: float = 0.15,
        fraction: float = 0.25,
    ) -> int:
        """
        Break codebook symmetry between streams i and j.
        """
        with torch.no_grad():
            Wi = F.normalize(self.codebooks[i].codebook.weight.data, dim=1)  # [Ki, D]
            Wj = self.codebooks[j].codebook.weight.data                       #[Kj, D]
            Wj_norm = F.normalize(Wj, dim=1)
            sim = torch.mm(Wi, Wj_norm.t())
            max_sim_j, _ = sim.max(dim=0)                                     # [Kj]
            k_perturb = max(1, int(fraction * self.codebooks[j].num_symbols))
            _, top_idx = max_sim_j.topk(k_perturb)
            for idx in top_idx:
                Wj[idx] += torch.randn_like(Wj[idx]) * noise_scale
            name_i = self.sub_encoder.submodality_names[i]
            name_j = self.sub_encoder.submodality_names[j]
            print(f"  ⚡ Kickstart ({name_i},{name_j}): perturbed {k_perturb} "
                  f"of {self.codebooks[j].num_symbols} centroids in {name_j}")
            return k_perturb

    def forward(self, x: torch.Tensor):
        embed        = self.embedding(x)          # [B, T, D]
        sub_features = self.sub_encoder(embed)    # [B, T, 5, D]

        symbolic_vectors = []
        symbolic_indices =[]
        total_vq_loss    = 0.0
        total_ema_H      = 0.0
        per_stream_active = []
        per_stream_topk_H =[]

        for i in range(self.NUM_SUBMODALITIES):
            normed_stream = self.pre_vq_norms[i](sub_features[:, :, i, :])
            quantized, indices, vq_loss, ema_H, active_ratio, topk_H = self.codebooks[i](normed_stream)
            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)
            total_vq_loss  += vq_loss
            total_ema_H    += ema_H
            per_stream_active.append(active_ratio)
            per_stream_topk_H.append(topk_H)

        gate_logits  = self.submodality_gate(embed)                     # [B, T, N]
        if getattr(self, "gate_bias_override", None) is not None:
            gate_logits = gate_logits + self.gate_bias_override
        gate_weights = F.softmax(gate_logits, dim=-1)                  # [B, T, N]

        if not all(self.stream_active):
            gate_weights = gate_weights * self.stream_mask          # zero inactive
            gate_weights = gate_weights / (gate_weights.sum(-1, keepdim=True) + 1e-10)
        sym_stack    = torch.stack(symbolic_vectors, dim=2)              #[B, T, 5, D]
        composite    = (sym_stack * gate_weights.unsqueeze(-1)).sum(dim=2)  # [B, T, D]

        # Gate entropy penalty
        log_uniform  = math.log(self.NUM_SUBMODALITIES)
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(-1).mean()
        gate_kl      = log_uniform - gate_entropy          # ≥ 0, zero iff uniform

        if self.training:
            feat_orth_loss = self.sub_encoder.feature_orthogonality_loss(sub_features)

            with torch.no_grad():
                hists =[]
                for i in range(self.NUM_SUBMODALITIES):
                    idx_i = symbolic_indices[i].reshape(-1)          # [B*T*K]
                    K_i   = self.codebooks[i].num_symbols
                    h_i   = F.one_hot(idx_i, K_i).float().mean(0)   # [K_i]
                    hists.append(h_i)
                max_K = max(h.shape[0] for h in hists)
                padded =[F.pad(h, (0, max_K - h.shape[0])) + 1e-10 for h in hists]
                padded = [p / p.sum() for p in padded]

            pair_js_vals  = {}
            for i in range(self.NUM_SUBMODALITIES):
                for j in range(i + 1, self.NUM_SUBMODALITIES):
                    p, q = padded[i], padded[j]
                    m    = 0.5 * (p + q)
                    js   = 0.5 * ((p * torch.log(p / m)).sum() + (q * torch.log(q / m)).sum())
                    pair_js_vals[(i, j)] = math.log(2) - js.item()

            N_s = self.NUM_SUBMODALITIES
            f_streams = [
                self.pre_vq_norms[i](sub_features[:, :, i, :]).reshape(-1, self.d_model)
                for i in range(N_s)
            ]
            d_owns = [
                torch.cdist(f_streams[i], self.codebooks[i].codebook.weight, p=2)
                .min(dim=1).values.mean()
                for i in range(N_s)
            ]

            f_stack = torch.stack(f_streams)
            f_sq    = (f_stack ** 2).sum(-1)

            cross_cols =[]
            for j in range(N_s):
                c_j    = self.codebooks[j].codebook.weight
                dist2_j = (f_sq[:, :, None]
                           + (c_j ** 2).sum(-1)[None, None, :]
                           - 2 * torch.einsum('nbd,kd->nbk', f_stack, c_j))
                dist2_j = dist2_j.clamp(min=1e-8).sqrt()           # avoid NaN
                cross_cols.append(dist2_j.min(dim=2).values.mean(dim=1))
            cross_mean = torch.stack(cross_cols, dim=1)

            pair_grad_terms = {}
            for i in range(N_s):
                for j in range(i + 1, N_s):
                    pair_grad_terms[(i, j)] = (
                        F.relu(d_owns[i] - cross_mean[i, j] + 1.0)
                        + F.relu(d_owns[j] - cross_mean[j, i] + 1.0)
                    ) * 0.5

            n_pairs = max(1, len(pair_js_vals))
            sym_decor_loss = torch.tensor(
                sum(pair_js_vals.values()) / n_pairs, device=x.device)
        else:
            feat_orth_loss = sub_features.new_zeros(1).squeeze()
            sym_decor_loss = sub_features.new_zeros(1).squeeze()

        composite = self.pos_encoder(composite)
        decoder_out = self.decoder(composite)
        logits      = self.output_head(decoder_out)

        return (
            logits,
            torch.stack(symbolic_indices, dim=2),          # [B, T, N]
            total_vq_loss,
            None,                                           # Deprecated Placebo Entropy Loss
            feat_orth_loss,                                 # unscaled; sigmoid applied in loop
            gate_kl,                                        # KL(gate || uniform), unscaled
            sym_decor_loss,                                 # mean JS (logging scalar, no grad)
            gate_weights,                                   # [B, T, N] — reused for balance_loss
            total_ema_H    / self.NUM_SUBMODALITIES,
            per_stream_active,  # list[float], one per stream
            pair_js_vals if self.training else {},          # {(i,j): float} per-pair JS
            pair_grad_terms if self.training else {},       # {(i,j): tensor} per-pair hinge
            sub_features.detach() if self.training else None,  # for corr tracking in loop
            per_stream_topk_H,  # list[float] — Top-K softness per stream
        )

    def generate(
        self,
        input_ids:          torch.Tensor,
        max_new_tokens:     int,
        seq_len:            int,
        temperature:        float = 0.8,
        top_k:              int   = 50,
        top_p:              float = 0.9,
        repetition_penalty: float = 1.15,
    ) -> torch.Tensor:
        self.eval()
        temperature = max(0.5, temperature)
        for _ in range(max_new_tokens):
            context = input_ids[:, -seq_len:]
            with torch.no_grad():
                logits, *_ = self(context)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty > 1.0:
                recent = set(input_ids[0, -64:].tolist())
                penalty = repetition_penalty * 2.0
                for tok in recent:
                    logits[0, tok] = logits[0, tok] - penalty

            if top_k and top_k > 0:
                k            = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                min_val      = topk_vals[:, -1].unsqueeze(1)
                logits       = logits.masked_fill(logits < min_val, float("-inf"))

            if top_p < 1.0:
                probs_sorted, sorted_idx = torch.sort(F.softmax(logits, dim=-1),
                                                      descending=True)
                cumulative = torch.cumsum(probs_sorted, dim=-1)
                remove_mask = cumulative - probs_sorted > top_p
                sorted_logits = logits.gather(1, sorted_idx)
                sorted_logits[remove_mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids  = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ==========================================================
# TOKENIZERS
# ==========================================================

import re as _re

class WordLevelTokenizer:
    PAD   = "<PAD>"
    UNK   = "<UNK>"
    BOS   = "<BOS>"
    EOS   = "<EOS>"

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._built = False

    @staticmethod
    def _split(text: str) -> list[str]:
        return _re.findall(r"\w+|[^\w\s]", text, _re.UNICODE)

    def build_vocab(self, text: str, min_freq: int = 1) -> None:
        from collections import Counter
        tokens  = self._split(text)
        counts  = Counter(tokens)
        specials =[self.PAD, self.UNK, self.BOS, self.EOS]
        vocab   = specials +[t for t, c in counts.most_common() if c >= min_freq]
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self._built   = True

    def encode(self, text: str) -> list[int]:
        unk_id = self.token2id[self.UNK]
        return[self.token2id.get(t, unk_id) for t in self._split(text)]

    def decode(self, ids: list[int]) -> str:
        tokens =[self.id2token.get(i, self.UNK) for i in ids]
        out =[]
        for tok in tokens:
            if out and _re.match(r"^[^\w]", tok):
                out[-1] += tok
            else:
                out.append(tok)
        return " ".join(out)

    @property
    def n_vocab(self) -> int:
        return len(self.token2id)


class FastBPETokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self._tok       = None      # set in train()

    def train(self, text: str) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        print(f"  BPE (fast/C++): training to vocab {self.vocab_size:,}…",
              flush=True)

        tok = Tokenizer(BPE(unk_token=self.UNK))
        tok.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size    = self.vocab_size,
            special_tokens =[self.PAD, self.UNK, self.BOS, self.EOS],
            show_progress  = True,
        )
        tok.train_from_iterator([text], trainer=trainer)
        self._tok = tok
        print(f"  BPE vocab: {self.n_vocab:,} tokens", flush=True)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def n_vocab(self) -> int:
        return self._tok.get_vocab_size()


class BPETokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    EOW = "</w>"   # end-of-word marker

    def __init__(self, vocab_size: int = 4000):
        self.vocab_size    = vocab_size
        self.merges:  list[tuple[str, str]]  =[]
        self.token2id: dict[str, int]         = {}
        self.id2token: dict[int, str]         = {}
        self._built = False

    @staticmethod
    def _word_freqs(text: str) -> dict[tuple[str, ...], int]:
        from collections import Counter
        words = _re.findall(r"\S+", text)
        counts: Counter = Counter(words)
        return {tuple(list(w) + ["</w>"]): c for w, c in counts.items()}

    @staticmethod
    def _get_pairs(vocab: dict[tuple, int]) -> dict[tuple[str, str], int]:
        pairs: dict[tuple[str, str], int] = {}
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: tuple[str, str], vocab: dict) -> dict:
        merged  = "".join(pair)
        pattern = _re.compile(
            r"(?<!\S)" + _re.escape(" ".join(pair)) + r"(?!\S)"
        )
        new_vocab = {}
        for word, freq in vocab.items():
            word_str = " ".join(word)
            word_str = pattern.sub(merged, word_str)
            new_vocab[tuple(word_str.split())] = freq
        return new_vocab

    def train(self, text: str) -> None:
        specials   =[self.PAD, self.UNK, self.BOS, self.EOS]
        word_freqs = self._word_freqs(text)

        base_chars: set[str] = set()
        for word in word_freqs:
            base_chars.update(word)
        vocab_tokens = specials + sorted(base_chars)

        num_merges = max(0, self.vocab_size - len(vocab_tokens))
        print(f"  BPE (pure Python): {len(vocab_tokens)} base chars, "
              f"learning {num_merges} merges…", flush=True)
        report_every = max(1, num_merges // 20)   # print ~20 progress ticks

        for merge_idx in range(num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best       = max(pairs, key=pairs.get)
            word_freqs = self._merge_vocab(best, word_freqs)
            self.merges.append(best)
            vocab_tokens.append("".join(best))

            if (merge_idx + 1) % report_every == 0 or merge_idx == num_merges - 1:
                pct = 100 * (merge_idx + 1) / num_merges
                print(f"    {pct:5.1f}%  merge {merge_idx+1}/{num_merges}  "
                      f"latest: {''.join(best)!r}", flush=True)

        self.token2id = {t: i for i, t in enumerate(vocab_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self._built   = True
        print(f"  BPE vocab: {len(self.token2id):,} tokens  "
              f"({len(self.merges):,} merges learned)")

    def _tokenize_word(self, word: str) -> list[str]:
        symbols = list(word) + [self.EOW]
        for a, b in self.merges:
            merged = a + b
            i = 0
            new_syms =[]
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms
        return symbols

    def encode(self, text: str) -> list[int]:
        unk_id = self.token2id[self.UNK]
        ids    =[]
        for word in _re.findall(r"\S+", text):
            for sym in self._tokenize_word(word):
                ids.append(self.token2id.get(sym, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens =[self.id2token.get(i, self.UNK) for i in ids]
        text   = "".join(tokens)
        text   = text.replace(self.EOW, " ")
        return text.strip()

    @property
    def n_vocab(self) -> int:
        return len(self.token2id)


def _make_bpe_tokenizer(vocab_size: int):
    try:
        import tokenizers  # noqa: F401
        return FastBPETokenizer(vocab_size=vocab_size)
    except ImportError:
        print(
            "  [hint] Install the `tokenizers` package for 100× faster BPE:\n"
            "         pip install tokenizers\n"
            "  Falling back to pure Python BPE (slower but fully functional).",
            flush=True,
        )
        return BPETokenizer(vocab_size=vocab_size)


# ==========================================================
# DATA LOADING
# ==========================================================

def load_data(filepath: str = "data.txt",
              tokenizer_mode: str = "tiktoken",
              bpe_vocab_size: int | None = None):
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

    mode = tokenizer_mode.lower().strip()
    if mode == "tiktoken":
        tokenizer  = tiktoken.get_encoding("gpt2")
        vocab_size = tokenizer.n_vocab
        print(f"  Tokenizer: tiktoken GPT-2 BPE  (vocab {vocab_size:,})")
    elif mode == "word":
        tokenizer = WordLevelTokenizer()
        tokenizer.build_vocab(text)
        vocab_size = tokenizer.n_vocab
        print(f"  Tokenizer: word-level  (vocab {vocab_size:,})")
    elif mode == "bpe":
        try:
            import tokenizers as _hf_tok  # noqa: F401
            _fast_available = True
        except ImportError:
            _fast_available = False

        if bpe_vocab_size is None:
            unique_words   = len(set(_re.findall(r"\S+", text)))
            bpe_vocab_size = int(max(1000, unique_words * 1.5))
            if not _fast_available and bpe_vocab_size > 8000:
                _est_min = bpe_vocab_size // 2000
                print(f"  ⚠ BPE vocab = {bpe_vocab_size:,} with pure Python backend "
                      f"— estimated ~{_est_min}+ min.  "
                      f"Run: pip install tokenizers  to make this instant.")
            print(f"  Tokenizer: BPE — auto vocab size = {bpe_vocab_size:,} "
                  f"(from {unique_words:,} unique words × 1.5, no cap"
                  f"{', fast C++ backend' if _fast_available else ', pure Python'})")
        else:
            print(f"  Tokenizer: BPE — vocab size = {bpe_vocab_size:,} (manual)")
        tokenizer  = _make_bpe_tokenizer(bpe_vocab_size)
        tokenizer.train(text)
        vocab_size = tokenizer.n_vocab
    else:
        raise ValueError(
            f"Unknown tokenizer_mode '{tokenizer_mode}'. "
            "Choose 'tiktoken', 'word', or 'bpe'."
        )

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n    = len(data)

    VAL_THRESHOLD = 50_000
    if n < VAL_THRESHOLD:
        train_data = data
        val_data   = None
        print(f"  Tokens: {n:,} (< {VAL_THRESHOLD:,} → no val split)")
    else:
        if n < 200_000:
            val_frac = 0.05
        elif n < 1_000_000:
            val_frac = 0.10
        else:
            val_frac = 0.20
        split      = int((1.0 - val_frac) * n)
        train_data = data[:split]
        val_data   = data[split:]
        print(
            f"  Tokens: {n:,}  |  train {len(train_data):,}  "
            f"val {len(val_data):,}  ({int(val_frac*100)}% val)"
        )

    return train_data, val_data, tokenizer, vocab_size


def _random_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    max_start = data.shape[0] - seq_len - 1
    ix      = torch.randint(0, max_start, (batch_size,), device=data.device)
    offsets = torch.arange(seq_len, device=data.device)        # [seq_len]
    idx     = ix.unsqueeze(1) + offsets.unsqueeze(0)           # [B, seq_len]
    x = data[idx]                                              # [B, seq_len]
    y = data[idx + 1]                                          # [B, seq_len]
    return x, y


# ==========================================================
# TRAINING
# ==========================================================

def train_model(filepath: str = "data.txt",
                tokenizer_mode: str = "tiktoken",
                bpe_vocab_size: int | None = None):
    # ── Hyperparameters (static base values) ─────────────────────────
    steps          = 30000
    lr             = 3e-4
    lr_warmup      = 100
    weight_decay   = 0.01
    orth_tau       = 100
    log_every      = 100
    checkpoint      = "neuro_symbolic_model.pt"
    best_checkpoint = "neuro_symbolic_model_best.pt"

    # 1. Batch size
    batch_size_tokens = 1024
    batch_size_min    = 4
    batch_size_max    = 64
    batch_size        = 32

    # 2. Gradient clip
    grad_clip         = 1.0
    grad_norm_ema     = 1.0
    grad_norm_alpha   = 0.05
    grad_clip_factor  = 1.5
    grad_clip_min     = 0.3
    grad_clip_max     = 5.0

    # 3. gate_gamma
    gate_gamma        = 0.05
    target_gate_kl    = 0.20
    gate_gamma_lr     = 0.005
    gate_gamma_min    = 0.005
    gate_gamma_max    = 0.5

    # 4. sym_decor_lambda
    sym_decor_lambda  = 0.01
    target_sym_decor  = 0.05  # near-zero: we want streams fully separated → low penalty

    # 5. vq_beta
    vq_beta           = 0.60
    target_active     = 0.85
    vq_beta_lr        = 0.003
    vq_beta_min       = 0.1
    vq_beta_max       = 0.9   # >1.0 means VQ > CE, dominating training

    # 6. H_target annealing
    num_symbols_ref   = None
    h_target_max_frac = 1.0
    h_target_min_frac = 0.7

    # 7. Orth onset (VQ-triggered)
    orth_T0           = steps
    orth_unlocked     = False

    # 8. Dropout
    current_dropout   = 0.1
    dropout_min       = 0.0
    dropout_max       = 0.4
    dropout_lr        = 0.01
    overfit_threshold = 1.15

    # 9. alpha_reg_lambda
    alpha_reg_lambda  = 0.005
    target_alpha      = 0.15   # desired mean residual α (0 = full discrete, 0.5 = max bypass)
    alpha_lr          = 0.005
    alpha_lambda_min  = 0.001
    alpha_lambda_max  = 0.05

    # 10. orth_lambda
    orth_lambda       = 0.01   # mirrors model.sub_encoder.orthogonality_lambda
    target_orth       = 0.05   # desired unscaled feat_orth_loss
    orth_lambda_lr    = 0.005
    orth_lambda_min   = 0.001
    orth_lambda_max   = 0.2

    # 12. val_every — starts frequent, doubles on each LR reduction (up to 1000)
    val_every         = 200
    val_every_max     = 1000

    # 13. Dynamic stream management
    candidate_streams  =["Texture", "Motion", "Valence", "Agency",
                          "Causality", "Certainty", "Temporal"]
    next_candidate_idx = 0
    max_streams        = 12
    stream_check_every = 100        # starts frequent; grows to 2000 later
    stream_check_max   = 2000
    stream_gate_ema    = None
    stream_gate_alpha  = 0.02
    stream_dead_patience = 1000
    stream_dead_counts   = None
    stream_min_dwell     = 500   # min steps with current stream count before adding another
    stream_last_added_step = 0   # step at which last stream was added

    # 14. Dynamic weight_decay
    weight_decay_val    = weight_decay   # current value (updated each val step)
    weight_decay_min    = 0.001
    weight_decay_max    = 0.1
    weight_decay_lr     = 0.01           # fractional update per val step

    lr_warmup_base      = lr_warmup      # will be overridden below

    # 16. Velocity-based self-tuning LR controller
    vel_ema_alpha       = 0.02    # ~50-step effective window (slow, stable signal)
    vel_check_every     = 100     # evaluate velocity this often
    vel_reduce_thresh   = 0.0002  # velocity ≥ this → clearly stalled → reduce
    vel_boost_thresh    = -0.005  # velocity ≤ this → improving fast → small boost
    lr_reduce_factor    = 0.99    # LR multiplier on stall — 1% reduction per trigger
    lr_boost_factor     = 1.03    # LR multiplier when improving fast (gentler)
    vel_cooldown        = 300     # min steps between LR changes
    vel_min_step        = 400     # never reduce LR before this step
    lr_reductions       = 0       # count of LR reductions so far

    stream_lr_reset_fraction = 0.35   # global reset target = 0.35 × peak LR

    # ── TopK temperature annealing ────────────────────────────────────────
    topk_tau_start = 0.1    # cold: weights ≈ one-hot on nearest
    topk_tau_end   = 1.5    # warm: weights spread softly across K

    # ── Rolling stream addition trigger ──────────────────────────────────
    stream_trigger_quorum = 0.6   # fraction of active streams that must be saturated

    # ── Dead codebook revival ─────────────────────────────────────────────
    revival_check_every = 500
    revival_noise_scale = 0.1     # std of perturbation applied to revived entries

    # ── Cross-stream centroid kickstart ───────────────────────────────────
    kickstart_js_thresh  = 0.35   # if worst pair JS below this → consider kickstart
    kickstart_patience   = 800    # steps worst pair must be below threshold before firing
    kickstart_cooldown   = 1000   # min steps between kickstarts for same pair
    kickstart_last       = {}     # {(i,j): last_step_kickstarted}
    kickstart_bad_since  = {}     # {(i,j): step_first_bad}

    # 17. Adam beta2 annealing
    adam_beta2_start    = 0.99
    adam_beta2_end      = 0.999
    adam_beta1          = 0.9

    # 18. seq_len_max curriculum gated by val CE thresholds
    seq_len_stages      =[64, 128, 256, 512]
    seq_len_stage_idx   = 0
    seq_len_max         = seq_len_stages[0]   # overrides the static 512 set above

    # 19. batch_size_tokens doubles on each LR reduction
    batch_tokens_max    = 4096   # hard cap

    # 20. log_every grows gently over time
    log_every_base      = 100
    log_every_max       = 150  # caps at 150 — tied to steps_since_imp (not step count)

    # 21. commitment_cost per-stream
    commit_cost_lr      = 0.005
    commit_cost_min     = 0.05
    commit_cost_max     = 0.25
    target_active_per   = target_active   # reuse global target

    # 23. ema_decay annealing per codebook
    ema_decay_start     = 0.2
    ema_decay_end       = 0.05

    # 25. per-stream codebook growth
    cb_grow_check_every = 200       # steps between codebook grow checks
    cb_grow_active_thresh = 0.95    # must be above this consistently
    cb_grow_patience    = 400       # steps above threshold before growing
    cb_active_ema       = None      # per-stream EMA of active_ratio (init after model)
    cb_active_ema_alpha = 0.05
    cb_grow_counts      = None      # per-stream steps above threshold

    # 26. Per-pair sym_decor lambdas
    pair_sym_lambdas     = {}        # {(i,j): float} — populated after model init
    pair_sym_lambda_init = 0.01
    pair_sym_lambda_lr   = 0.005
    pair_sym_lambda_min  = 0.001
    pair_sym_lambda_max  = 20.0
    target_pair_js       = 0.05     # target overlap per pair (near-zero = separated)

    # ── Growth LR reset ───────────────────────────────────────────────────
    growth_reset_fraction  = 0.25  # fraction of peak LR for new params
    growth_warmup_steps    = 500   # steps to decay back to current global LR

    # 28. H_target scales with batch_tokens
    h_target_batch_scale = True    # enable scaling
    h_target_batch_ref   = 1024   # reference batch_tokens (baseline = no scaling)

    # 29. Orth onset based on feature correlation
    stream_corr_baseline = None    # calibrated mean cross-stream cosine sim
    stream_corr_ema      = 0.0
    stream_corr_alpha    = 0.05
    stream_corr_trigger  = 0.05   # correlation above baseline by this → unlock orth
    orth_fallback_step   = 500    # unlock orth unconditionally at this step if still locked
    stream_corr_calib_done = False

    # 30. Seq_len curriculum by improvement rate
    seq_len_val_ce_history =[]    # val CE readings at current stage
    seq_len_improvement_ema = 0.0
    seq_len_improvement_alpha = 0.3
    seq_len_plateau_thresh = 0.005  # improvement per val step below this → unlock

    # 31. Depth growth by final-layer gradient magnitude
    layer_grad_emas      =[]      # per-layer grad norm EMA (populated after model)
    layer_grad_alpha     = 0.05
    depth_grad_ratio_thresh = 1.5  # last-layer grad / mean-layer grad above this → add
    depth_check_every    = 200     # steps between depth checks
    depth_last_added     = 0       # step of last layer addition
    depth_min_dwell      = 500     # min steps between additions

    # ── Data ──────────────────────────────────────────────────────────
    train_data, val_data, tokenizer, vocab_size = load_data(
        filepath, tokenizer_mode, bpe_vocab_size
    )

    train_data = train_data.to(device)
    if val_data is not None:
        val_data = val_data.to(device)

    n_train = len(train_data)

    if n_train < 512 + 2:
        raise ValueError(
            f"Corpus too small: {n_train} tokens, need at least 514. "
            "Provide a larger data.txt."
        )

    # ── Dynamic num_symbols ───────────────────────────────────────────
    num_symbols_target = max(16, int((vocab_size / max_streams) ** 0.5))
    num_symbols        = max(16, min(32, num_symbols_target))  # start small
    num_symbols_ref    = num_symbols_target   # H_target anchored to final size
    print(f"  num_symbols: {num_symbols} (start, target {num_symbols_target})")

    # ── Dynamic d_model ───────────────────────────────────────────────
    _raw_d  = (n_train ** 0.25) * 8
    d_model = int(max(128, min(768, round(_raw_d / 64) * 64)))
    print(f"  d_model: {d_model}")

    # ── Dynamic n_heads ───────────────────────────────────────────────
    n_heads = 1
    for h in [8, 4, 2, 1]:
        if d_model % h == 0:
            n_heads = h
            break
    print(f"  n_heads: {n_heads}")

    # ── Dynamic num_layers ────────────────────────────────────────────
    num_layers     = int(max(2, min(8, round(math.log2(n_train / 1000)))))
    num_layers_max = 16
    print(f"  num_layers: {num_layers} (start)  max {num_layers_max}")

    # ── Dynamic seq_len_base ──────────────────────────────────────────
    seq_len_base = int(max(8, min(64, round(math.log2(vocab_size) * 2))))
    print(f"  seq_len_base: {seq_len_base}  seq_len_max (curriculum start): {seq_len_max}")

    # ── Model ─────────────────────────────────────────────────────────
    model = SubmodalitySymbolicLanguageModel(
        vocab_size  = vocab_size,
        d_model     = d_model,
        num_symbols = num_symbols,
        n_heads     = n_heads,
        num_layers  = num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  Device: {device}")

    _cuda_cap = (torch.cuda.get_device_capability(device)
                 if device.type == "cuda" else (0, 0))
    _torch_compile_available = (
        hasattr(torch, "compile")
        and device.type == "cuda"
        and _cuda_cap[0] >= 7
    )
    if _torch_compile_available:
        torch._dynamo.config.capture_scalar_outputs = True
        import logging, warnings
        logging.getLogger('torch._inductor.utils').setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message="Online softmax is disabled")
        print("  Compiling model with torch.compile (dynamic=True)…")
        _compiled_model = torch.compile(model, dynamic=True)
        print("  Compile complete.\n")
    else:
        _compiled_model = model
        if device.type == "cuda" and _cuda_cap[0] < 7:
            print(f"  torch.compile skipped — CUDA capability {_cuda_cap[0]}.{_cuda_cap[1]} < 7.0 (Triton requirement)\n")
        else:
            print()

    N = model.NUM_SUBMODALITIES
    stream_gate_ema    = [1.0 / N] * N
    stream_dead_counts = [0] * N
    cb_active_ema      = [0.5] * N
    cb_grow_counts     = [0] * N

    pair_sym_lambdas = {
        (i, j): pair_sym_lambda_init
        for i in range(N) for j in range(i + 1, N)
    }

    layer_grad_emas = [1.0] * num_layers

    def _stream_thresholds(n):
        uniform = 1.0 / n
        return 0.5 * uniform, 0.95 * uniform
    stream_dead_thresh, stream_saturated_thresh = _stream_thresholds(N)

    lr_warmup = max(100, int(lr_warmup_base * (num_layers / 4) * (d_model / 256)))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay = weight_decay,
        betas        = (adam_beta1, adam_beta2_start),
    )

    _use_amp = torch.cuda.is_available()
    scaler   = GradScaler('cuda', enabled=_use_amp)
    lr_ceiling = lr

    growth_warmup_queue =[]
    raw_norm = torch.tensor(0.0)

    lr_patience          = 300
    lr_reduce_factor     = 0.99
    lr_reduce_threshold  = 0.005
    criterion = nn.CrossEntropyLoss()

    best_val_loss    = float("inf")
    last_train_ce    = None
    grad_norm_ema    = grad_clip

    ce_vel_ema       = 0.0
    ce_prev          = None
    vel_last_action  = 0
    lr_min           = lr / 100

    best_ce_ema      = float("inf")
    ce_ema           = None
    ce_ema_alpha     = 0.05
    steps_since_imp  = 0
    lr_patience_base = 400
    lr_patience_grow = 100
    lr_patience      = 400

    model.train()
    print("Starting training…\n")

    def _add_with_growth_warmup(new_params, current_step):
        cur_lr    = optimizer.param_groups[0]["lr"]
        reset_lr  = max(cur_lr * 2.0, lr * growth_reset_fraction)
        group_idx = len(optimizer.param_groups)
        optimizer.add_param_group({
            "params":       new_params,
            "lr":           reset_lr,
            "weight_decay": weight_decay_val,
        })
        growth_warmup_queue.append({
            "group_idx":  group_idx,
            "start_lr":   reset_lr,
            "start_step": current_step,
        })
        print(f"    ↑ Growth LR reset: {reset_lr:.2e} → {cur_lr:.2e} "
              f"over {growth_warmup_steps} steps")

    _train_start   = time.time()
    _step_time     = time.time()
    _last_log_step = 0
    _last_heartbeat = time.time()

    for step in range(steps):

        ema_decay_now = ema_decay_start - (ema_decay_start - ema_decay_end) * (step / max(1, steps - 1))
        for cb in model.codebooks:
            cb.ema_decay = ema_decay_now

        beta2_now = adam_beta2_start + (adam_beta2_end - adam_beta2_start) * (step / max(1, steps - 1))
        for g in optimizer.param_groups:
            g["betas"] = (adam_beta1, beta2_now)

        log_every = min(log_every_max, log_every_base + steps_since_imp // 10)

        seq_len = min(seq_len_max,
                      seq_len_base + int(math.log(step + 1) * 8))
        safe_seq = min(seq_len, len(train_data) - 2)

        batch_size = int(max(batch_size_min,
                             min(batch_size_max, batch_size_tokens // safe_seq)))

        x, y = _random_batch(train_data, batch_size, safe_seq)

        if step < lr_warmup:
            for g in optimizer.param_groups:
                g["lr"] = lr * (step + 1) / lr_warmup

        h_frac       = h_target_max_frac - (h_target_max_frac - h_target_min_frac) * (step / max(1, steps - 1))
        h_base       = h_frac * math.log(num_symbols_ref)
        if h_target_batch_scale and batch_size_tokens > h_target_batch_ref:
            batch_scale = 1.0 + 0.1 * math.log(batch_size_tokens / h_target_batch_ref)
        else:
            batch_scale = 1.0
        h_target_now = h_base * min(batch_scale, 1.3)
        for cb in model.codebooks:
            cb.H_target = min(h_target_now, 0.9 * math.log(max(2, cb.num_symbols)))

        model.sub_encoder.orthogonality_lambda = orth_lambda

        optimizer.zero_grad()
        with autocast('cuda', enabled=_use_amp):
            logits, _, vq_loss, _, feat_orth_loss, gate_kl, sym_decor_loss, \
                gate_weights, mean_H, per_stream_active, pair_js_vals, pair_grad_terms, \
                sub_features, per_stream_topk_H = _compiled_model(x)
            active_ratio = sum(per_stream_active) / len(per_stream_active)
            loss_task = criterion(logits.view(-1, vocab_size), y.view(-1))
        last_train_ce = loss_task.item()

        vq_val = vq_loss.item()

        if orth_unlocked is False:
            with torch.no_grad():
                sub_f = sub_features.detach().reshape(-1, model.NUM_SUBMODALITIES, model.d_model)
                sub_f_n = F.normalize(sub_f, dim=2)
                corr_mat = torch.einsum('bnd,bmd->nm', sub_f_n, sub_f_n) / sub_f_n.shape[0]
                n_s = model.NUM_SUBMODALITIES
                idx_i, idx_j = torch.triu_indices(n_s, n_s, offset=1)
                mean_corr = corr_mat[idx_i, idx_j].mean().item()
            stream_corr_ema = ((1 - stream_corr_alpha) * stream_corr_ema
                               + stream_corr_alpha * mean_corr)
            if not stream_corr_calib_done and step >= 100:
                stream_corr_baseline = stream_corr_ema
                stream_corr_calib_done = True
                print(f"  ✦ Stream correlation baseline calibrated: {stream_corr_baseline:.4f}")
            if (stream_corr_calib_done
                    and stream_corr_ema > stream_corr_baseline + stream_corr_trigger):
                orth_T0       = step
                orth_unlocked = True
                print(f"  ✦ Orth unlocked at step {step}  "
                      f"(corr {stream_corr_ema:.4f} > baseline "
                      f"{stream_corr_baseline:.4f} + {stream_corr_trigger})")
            elif stream_corr_calib_done and step >= orth_fallback_step:
                orth_T0       = step
                orth_unlocked = True
                print(f"  ✦ Orth unlocked at step {step} (fallback — "
                      f"corr {stream_corr_ema:.4f} never rose +{stream_corr_trigger} above baseline)")
        elif stream_corr_calib_done:
            with torch.no_grad():
                sub_f   = sub_features.detach().reshape(-1, model.NUM_SUBMODALITIES, model.d_model)
                sub_f_n = F.normalize(sub_f, dim=2)
                corr_mat = torch.einsum('bnd,bmd->nm', sub_f_n, sub_f_n) / sub_f_n.shape[0]
                n_s = model.NUM_SUBMODALITIES
                idx_i, idx_j = torch.triu_indices(n_s, n_s, offset=1)
                stream_corr_ema = ((1 - stream_corr_alpha) * stream_corr_ema
                                   + stream_corr_alpha * corr_mat[idx_i, idx_j].mean().item())

        if orth_unlocked:
            orth_weight = 0.8 / (1.0 + math.exp(-(step - orth_T0) / orth_tau))
        else:
            orth_weight = 0.0
            
        orth_loss = orth_weight * feat_orth_loss

        gate_kl_val = gate_kl.item()
        gate_gamma  = gate_gamma * (1.0 + gate_gamma_lr * (gate_kl_val - target_gate_kl))
        gate_gamma  = max(gate_gamma_min, min(gate_gamma_max, gate_gamma))
        gate_loss   = gate_gamma * gate_kl

        gate_mean    = gate_weights.mean(dim=(0, 1))
        if all(model.stream_active):
            balance_loss = ((gate_mean - 1.0 / model.NUM_SUBMODALITIES) ** 2).sum()
        else:
            active_mask = model.stream_mask > 0
            n_act = max(1, int(active_mask.sum().item()))
            balance_loss = ((gate_mean[active_mask] - 1.0 / n_act) ** 2).sum()

        sym_dec_val = sym_decor_loss.item()
        pair_sym_loss = torch.tensor(0.0, device=device)
        for (pi, pj), js_val in pair_js_vals.items():
            lam = pair_sym_lambdas.get((pi, pj), pair_sym_lambda_init)
            lam = lam * (1.0 + pair_sym_lambda_lr * (js_val - target_pair_js))
            lam = max(pair_sym_lambda_min, min(pair_sym_lambda_max, lam))
            pair_sym_lambdas[(pi, pj)] = lam
            if (pi, pj) in pair_grad_terms:
                pair_sym_loss = pair_sym_loss + lam * (js_val + pair_grad_terms[(pi, pj)])
        sym_decor_lambda = sum(pair_sym_lambdas.values()) / max(1, len(pair_sym_lambdas))

        vq_beta = vq_beta * (1.0 + vq_beta_lr * (target_active - active_ratio))
        vq_beta = max(vq_beta_min, min(vq_beta_max, vq_beta))

        mean_alpha = sum(
            0.5 * torch.sigmoid(cb.residual_alpha_raw).item()
            for cb in model.codebooks
        ) / model.NUM_SUBMODALITIES
        alpha_reg_lambda = alpha_reg_lambda * (1.0 + alpha_lr * (mean_alpha - target_alpha))
        alpha_reg_lambda = max(alpha_lambda_min, min(alpha_lambda_max, alpha_reg_lambda))
        alpha_reg = alpha_reg_lambda * sum(
            (0.5 * torch.sigmoid(cb.residual_alpha_raw)) ** 2
            for cb in model.codebooks
        )

        orth_raw    = feat_orth_loss.item()
        orth_lambda = orth_lambda * (1.0 + orth_lambda_lr * (orth_raw - target_orth))
        orth_lambda = max(orth_lambda_min, min(orth_lambda_max, orth_lambda))

        total_loss = (loss_task
                      + vq_beta        * vq_loss
                      + orth_loss
                      + gate_loss
                      + gate_gamma     * balance_loss
                      + alpha_reg
                      + pair_sym_loss)

        if not torch.isfinite(total_loss):
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                g["lr"] = max(lr_min, g["lr"] * 0.95)
            ce_prev = None
            if step % log_every == 0:
                print(f"  ⚠ NaN/Inf loss at step {step} — skipping step, LR reduced ×0.95")
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        raw_norm  = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if torch.isfinite(raw_norm):
            grad_norm_ema = (1 - grad_norm_alpha) * grad_norm_ema + grad_norm_alpha * raw_norm.item()
        grad_clip = max(grad_clip_min, min(grad_clip_max, grad_norm_ema * grad_clip_factor))

        if not torch.isfinite(raw_norm):
            if step % log_every == 0:
                print(f"  ⚠ NaN/Inf gradient at step {step} — skipping step, LR ×0.95", flush=True)
            optimizer.zero_grad()
            scaler.update()
            for g in optimizer.param_groups:
                g["lr"] = max(lr_min, g["lr"] * 0.95)
            ce_prev = None
            continue

        scaler.step(optimizer)
        scaler.update()

        if (step >= lr_warmup
                and steps_since_imp >= lr_patience
                and step > vel_last_action + vel_cooldown // 2):
            cur_lr = optimizer.param_groups[0]['lr']
            new_lr = max(lr_min, cur_lr * lr_reduce_factor)
            if new_lr < cur_lr:
                for g in optimizer.param_groups:
                    g['lr'] = new_lr
                lr_reductions   += 1
                lr_ceiling       = new_lr
                lr_patience      = lr_patience_base + lr_reductions * lr_patience_grow
                vel_last_action  = step
                batch_size_tokens = min(batch_tokens_max, batch_size_tokens * 2)
                print(f"  ↓ LR  {cur_lr:.2e} → {new_lr:.2e}  "
                      f"(patience #{lr_reductions}, stalled {steps_since_imp} steps)")
                print(f"  ↑ batch_tokens → {batch_size_tokens}")
            steps_since_imp = 0

        if growth_warmup_queue:
            cur_lr = optimizer.param_groups[0]["lr"]
            done   =[]
            for event in growth_warmup_queue:
                elapsed = step - event["start_step"]
                if elapsed >= growth_warmup_steps:
                    optimizer.param_groups[event["group_idx"]]["lr"] = cur_lr
                    done.append(event)
                else:
                    t      = elapsed / growth_warmup_steps
                    new_lr = event["start_lr"] * (1.0 - t) + cur_lr * t
                    optimizer.param_groups[event["group_idx"]]["lr"] = new_lr
            for event in done:
                growth_warmup_queue.remove(event)

        _tau = topk_tau_start + (topk_tau_end - topk_tau_start) * (step / max(1, steps - 1))
        for _cb in model.codebooks:
            _cb.topk_tau = _tau

        if step >= lr_warmup:
            current_ce = loss_task.item()
            if ce_prev is not None:
                raw_vel   = current_ce - ce_prev        # negative = improving
                ce_vel_ema = ((1 - vel_ema_alpha) * ce_vel_ema
                              + vel_ema_alpha * raw_vel)
            ce_prev = current_ce

            if ce_ema is None:
                ce_ema = current_ce
            ce_ema = (1 - ce_ema_alpha) * ce_ema + ce_ema_alpha * current_ce
            if ce_ema < best_ce_ema * (1 - lr_reduce_threshold):
                best_ce_ema     = ce_ema
                steps_since_imp = 0
            else:
                steps_since_imp += 1

            if (step >= vel_min_step
                    and step % vel_check_every == 0
                    and step > vel_last_action + vel_cooldown):
                cur_lr = optimizer.param_groups[0]["lr"]

                if ce_vel_ema >= vel_reduce_thresh:
                    new_lr = max(lr_min, cur_lr * lr_reduce_factor)
                    if new_lr < cur_lr:
                        for g in optimizer.param_groups:
                            g["lr"] = new_lr
                        lr_ceiling    = new_lr
                        vel_last_action = step
                        lr_reductions  += 1
                        print(f"  ↓ LR {cur_lr:.2e} → {new_lr:.2e}  "
                              f"(vel {ce_vel_ema:+.5f} ≥ {vel_reduce_thresh:+.5f}, "
                              f"reduction #{lr_reductions})")
                        batch_size_tokens = min(batch_tokens_max, batch_size_tokens * 2)
                        print(f"  ↑ batch_tokens → {batch_size_tokens}")
                        val_every = min(val_every_max, val_every * 2)

                elif ce_vel_ema <= vel_boost_thresh:
                    new_lr = min(lr_ceiling, cur_lr * lr_boost_factor)
                    if new_lr > cur_lr * 1.001:
                        for g in optimizer.param_groups:
                            g["lr"] = new_lr
                        vel_last_action = step
                        print(f"  ↑ LR {cur_lr:.2e} → {new_lr:.2e}  "
                              f"(vel {ce_vel_ema:+.5f} ≤ {vel_boost_thresh:+.5f})")

        if (step > 0
                and step % depth_check_every == 0
                and step >= depth_last_added + depth_min_dwell
                and num_layers < num_layers_max):
            with torch.no_grad():
                block_norms =[]
                for blk in model.decoder.blocks:
                    nrm = sum(
                        p.grad.norm().item() ** 2
                        for p in blk.parameters()
                        if p.grad is not None
                    ) ** 0.5
                    block_norms.append(nrm)
                while len(layer_grad_emas) < len(block_norms):
                    layer_grad_emas.append(block_norms[-1])
                for i, nrm in enumerate(block_norms):
                    layer_grad_emas[i] = (
                        (1 - layer_grad_alpha) * layer_grad_emas[i]
                        + layer_grad_alpha * nrm
                    )
                mean_norm = sum(layer_grad_emas) / max(1, len(layer_grad_emas))
                last_norm = layer_grad_emas[-1]
                if mean_norm > 0 and last_norm / mean_norm > depth_grad_ratio_thresh:
                    current_lr = optimizer.param_groups[0]["lr"]
                    new_block  = model.decoder.add_layer(device)
                    optimizer.add_param_group({
                        "params":       list(new_block.parameters()),
                        "lr":           current_lr,
                        "weight_decay": weight_decay_val,
                    })
                    layer_grad_emas.append(last_norm)
                    num_layers += 1
                    depth_last_added = step
                    print(f"  ↑ Layer added (grad-driven, total: {num_layers})  "
                          f"last/mean grad ratio: {last_norm/mean_norm:.2f}")
                    if _torch_compile_available:
                        print("\n[JIT] Recompiling model after layer addition… (may take 1–3 min)", flush=True)
                        torch._dynamo.reset()
                        _compiled_model = torch.compile(model, dynamic=True)

        with torch.no_grad():
            for i, cb in enumerate(model.codebooks):
                stream_ar_i = per_stream_active[i] if i < len(per_stream_active) else active_ratio
                cb.commitment_cost = cb.commitment_cost * (
                    1.0 + commit_cost_lr * (target_active_per - stream_ar_i))
                cb.commitment_cost = max(commit_cost_min, min(commit_cost_max, cb.commitment_cost))

        stream_check_every = min(stream_check_max,
                                 100 + int(step / steps * (stream_check_max - 100)))

        if step > 0 and step % stream_check_every == 0:
            N = model.NUM_SUBMODALITIES
            gate_mean_np = gate_mean.detach().cpu().tolist()
            stream_dead_thresh, stream_saturated_thresh = _stream_thresholds(
                sum(model.stream_active))

            for i in range(N):
                stream_gate_ema[i] = (
                    (1 - stream_gate_alpha) * stream_gate_ema[i]
                    + stream_gate_alpha * gate_mean_np[i]
                )

            for i in range(N):
                if not model.stream_active[i]:
                    continue
                if stream_gate_ema[i] < stream_dead_thresh:
                    stream_dead_counts[i] += stream_check_every
                    if stream_dead_counts[i] >= stream_dead_patience:
                        model.deactivate_submodality(i)
                else:
                    stream_dead_counts[i] = 0

            n_active = sum(model.stream_active)
            _stream_add_eligible = (
                step >= 300
                and step >= stream_last_added_step + stream_min_dwell
            )
            if (_stream_add_eligible
                    and n_active < max_streams
                    and next_candidate_idx < len(candidate_streams)
                    and n_active > 0):
                active_emas = [stream_gate_ema[i]
                               for i in range(N) if model.stream_active[i]]
                n_saturated = sum(1 for e in active_emas if e > stream_saturated_thresh)
                quorum_met  = n_saturated >= math.ceil(stream_trigger_quorum * len(active_emas))
                if quorum_met:
                    new_name = candidate_streams[next_candidate_idx]
                    next_candidate_idx += 1
                    stream_gate_ema.append(1.0 / (N + 1))
                    stream_dead_counts.append(0)
                    cb_active_ema.append(0.5)
                    cb_grow_counts.append(0)
                    new_idx = model.NUM_SUBMODALITIES
                    for _pi in range(new_idx):
                        pair_sym_lambdas[(_pi, new_idx)] = pair_sym_lambda_init
                    _cur_lr   = optimizer.param_groups[0]["lr"]
                    _boost_lr = max(_cur_lr * 2.0, lr * growth_reset_fraction)
                    _new_params = model.add_submodality(
                        new_name, device, optimizer, lr, weight_decay_val,
                        model.codebooks[0].num_symbols,
                        boost_lr=_boost_lr)
                    growth_warmup_queue.append({
                        "group_idx":  len(optimizer.param_groups) - 1,
                        "start_lr":   _boost_lr,
                        "start_step": step,
                    })
                    _global_reset_lr = lr * stream_lr_reset_fraction
                    for g in optimizer.param_groups:
                        if g["lr"] < _global_reset_lr:
                            g["lr"] = _global_reset_lr
                    lr_ceiling      = _global_reset_lr
                    ce_vel_ema      = vel_boost_thresh * 2.0
                    vel_last_action = step + vel_cooldown * 2
                    ce_prev         = None
                    print(f"  ↑ Global LR reset → {_global_reset_lr:.2e}  "
                          f"(stream addition, ceiling reset)")
                    stream_last_added_step = step
                    if _torch_compile_available:
                        print("\n[JIT] Recompiling model after stream addition… (may take 1–3 min)", flush=True)
                        torch._dynamo.reset()
                        _compiled_model = torch.compile(model, dynamic=True)

        for i in range(len(cb_active_ema)):
            ar_i = per_stream_active[i] if i < len(per_stream_active) else active_ratio
            cb_active_ema[i] = ((1 - cb_active_ema_alpha) * cb_active_ema[i]
                                + cb_active_ema_alpha * ar_i)
        if step > 0 and step % cb_grow_check_every == 0:
            for i, cb in enumerate(model.codebooks):
                if cb.num_symbols >= num_symbols_target:
                    cb_grow_counts[i] = 0
                    continue
                if cb_active_ema[i] > cb_grow_active_thresh:
                    cb_grow_counts[i] += cb_grow_check_every
                    if cb_grow_counts[i] >= cb_grow_patience:
                        old_n = cb.num_symbols
                        n_new = max(4, old_n // 2)
                        new_n = min(num_symbols_target, old_n + n_new)
                        if new_n > old_n:
                            # 1. Clear old parameters from optimizer groups and optimizer state!
                            old_cb_params = list(cb.codebook.parameters())
                            old_cb_ids = {id(p) for p in old_cb_params}
                            for group in optimizer.param_groups:
                                group["params"] = [p for p in group["params"] if id(p) not in old_cb_ids]
                            for p in old_cb_params:
                                optimizer.state.pop(p, None)

                            # 2. Grow and add new weights
                            cb.grow_codebook(new_n, device)
                            _add_with_growth_warmup(list(cb.codebook.parameters()), step)
                            
                            if _torch_compile_available:
                                print("\n[JIT] Recompiling model after codebook growth… (may take 1–3 min)", flush=True)
                                torch._dynamo.reset()
                                _compiled_model = torch.compile(model, dynamic=True)
                        cb_grow_counts[i] = 0
                else:
                    cb_grow_counts[i] = 0

        if step > 0 and step % revival_check_every == 0:
            total_revived = 0
            for i, _cb in enumerate(model.codebooks):
                n = _cb.revive_dead_entries(device, revival_noise_scale)
                if n > 0:
                    total_revived += n
                    cb_active_ema[i]  = 0.5
                    cb_grow_counts[i] = 0
            if total_revived > 0:
                print(f"  ♻ Revived {total_revived} dead codebook entries across "
                      f"{model.NUM_SUBMODALITIES} streams  (growth counters reset)")

        if pair_js_vals and step > 0 and step % 100 == 0:
            worst_pair = min(pair_js_vals, key=pair_js_vals.get)
            worst_js   = pair_js_vals[worst_pair]
            pi, pj     = worst_pair
            if worst_js < kickstart_js_thresh:
                if worst_pair not in kickstart_bad_since:
                    kickstart_bad_since[worst_pair] = step
                elif (step - kickstart_bad_since[worst_pair] >= kickstart_patience
                      and step - kickstart_last.get(worst_pair, 0) >= kickstart_cooldown):
                    model.kickstart_worst_pair(pi, pj, device)
                    kickstart_last[worst_pair] = step
                    kickstart_bad_since.pop(worst_pair, None)
            else:
                kickstart_bad_since.pop(worst_pair, None)

        _now = time.time()
        if _now - _last_heartbeat >= 30.0 and step % log_every != 0:
            _elapsed_hb = _now - _train_start
            print(f"  ⏳ step {step}/{steps}  "
                  f"[{time.strftime('%H:%M:%S', time.gmtime(_elapsed_hb))} elapsed]  "
                  f"still running…", flush=True)
            _last_heartbeat = _now

        if step % log_every == 0 or step == steps - 1:
            ppl        = torch.exp(loss_task).item()
            n_active   = sum(model.stream_active)
            avg_cb     = sum(cb.num_symbols for cb in model.codebooks) / model.NUM_SUBMODALITIES
            now        = time.time()
            _last_heartbeat = now
            elapsed    = now - _train_start
            steps_since_last = max(1, step - _last_log_step)
            step_secs  = (now - _step_time) / steps_since_last
            _step_time = now
            _last_log_step = step
            eta_secs   = step_secs * (steps - step)
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str     = time.strftime("%H:%M:%S", time.gmtime(eta_secs))
            vel_str  = f"{ce_vel_ema:+.4f}" if ce_prev is not None else "n/a"
            corr_str = f"{stream_corr_ema:.3f}" if stream_corr_calib_done else "cal"

            names_abbrev = [n[:3] for n in model.sub_encoder.submodality_names]
            gw = gate_weights.mean(dim=(0, 1))
            gate_str = " ".join(
                f"{names_abbrev[i]}={gw[i].item():.2f}" for i in range(len(names_abbrev))
            )

            act_min = min(per_stream_active) * 100
            act_max = max(per_stream_active) * 100
            act_str = f"{active_ratio*100:.1f}%[{act_min:.0f}–{act_max:.0f}]"

            if pair_js_vals:
                worst_pair = min(pair_js_vals, key=pair_js_vals.get)
                worst_js   = pair_js_vals[worst_pair]
                wpi, wpj   = worst_pair
                wn_i = model.sub_encoder.submodality_names[wpi][:3]
                wn_j = model.sub_encoder.submodality_names[wpj][:3]
                worst_str = f"{worst_js:.3f}({wn_i},{wn_j})"
            else:
                worst_str = "n/a"

            mean_topk_H = sum(per_stream_topk_H) / max(1, len(per_stream_topk_H))
            topk_max    = math.log(model.codebooks[0].topk) if model.codebooks else 1.0
            cur_tau     = model.codebooks[0].topk_tau if model.codebooks else 0.0
            topk_str    = f"{mean_topk_H:.3f}/{topk_max:.2f}(τ{cur_tau:.2f})"

            if growth_warmup_queue:
                warmup_str = f"W{len(growth_warmup_queue)}@{min(growth_warmup_steps - (step - e['start_step']) for e in growth_warmup_queue)}"
            else:
                warmup_str = "-"

            gnorm_str = f"{raw_norm.item():.2f}→{grad_clip:.2f}"

            print(
                f"Step {step:5d}/{steps} | "
                f"CE {loss_task.item():.4f} | "
                f"VQ {vq_val:.4f}(β{vq_beta:.2f},cc{model.codebooks[0].commitment_cost:.3f}) | "
                f"Orth {orth_loss.item():.2e}(λ{orth_lambda:.3f},ρ{corr_str}) | "
                f"GKL {gate_kl_val:.3f}(γ{gate_gamma:.4f}) | "
                f"SymD μ{sym_dec_val:.3f} ↓{worst_str} | "
                f"Act {act_str} | "
                f"TopK-H {topk_str} | "
                f"α {mean_alpha:.3f}(λ{alpha_reg_lambda:.4f}) | "
                f"H {mean_H:.2f}→{h_target_now:.2f} | "
                f"PPL {ppl:.2f} | "
                f"BS {batch_size}(t{batch_size_tokens}) | "
                f"∇ {gnorm_str} | "
                f"vel {vel_str} | "
                f"do {current_dropout:.2f} | "
                f"L {num_layers} | "
                f"S {n_active}/{model.NUM_SUBMODALITIES} | "
                f"CB {avg_cb:.0f} | "
                f"Wmup {warmup_str} | "
                f"β2 {beta2_now:.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e} | "
                f"⏱ {elapsed_str} (+{step_secs:.2f}s/step, ETA {eta_str})",
                flush=True,
            )
            if step % (log_every * 5) == 0 or step == steps - 1:
                print(f"  Gates: {gate_str}")
                if per_stream_topk_H:
                    topkh_detail = " ".join(
                        f"{names_abbrev[i]}={per_stream_topk_H[i]:.2f}"
                        for i in range(len(per_stream_topk_H))
                    )
                    print(f"  TopK-H/stream: {topkh_detail}")

        if val_data is not None and step > 0 and step % val_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv   = _random_batch(val_data, batch_size, safe_seq)
                lv, *_   = model(xv)
                val_loss = criterion(lv.view(-1, vocab_size), yv.view(-1))
            val_ce = val_loss.item()
            marker = ""
            if val_ce < best_val_loss:
                best_val_loss = val_ce
                torch.save(model.state_dict(), best_checkpoint)
                marker = "  ✓ best"

            if step > lr_warmup and step > vel_min_step:
                if val_ce > best_val_loss * 1.02:
                    for g in optimizer.param_groups:
                        g["lr"] = max(lr_min, g["lr"] * 0.97)
                    lr_ceiling = optimizer.param_groups[0]["lr"]
                    vel_last_action = step
                elif val_ce >= best_val_loss:
                    for g in optimizer.param_groups:
                        g["lr"] = max(lr_min, g["lr"] * 0.99)
                    lr_ceiling = optimizer.param_groups[0]["lr"]
                    vel_last_action = step

            if last_train_ce is not None:
                ratio = last_train_ce / max(val_ce, 1e-8)
                if ratio > overfit_threshold:
                    current_dropout = current_dropout * (
                        1.0 + dropout_lr * (ratio - overfit_threshold))
                else:
                    current_dropout = current_dropout * (1.0 - dropout_lr * 0.5)
                current_dropout = max(dropout_min, min(dropout_max, current_dropout))
                model.decoder.set_dropout(current_dropout)

                if ratio > overfit_threshold:
                    weight_decay_val = weight_decay_val * (
                        1.0 + weight_decay_lr * (ratio - overfit_threshold))
                else:
                    weight_decay_val = weight_decay_val * (1.0 - weight_decay_lr * 0.5)
                weight_decay_val = max(weight_decay_min, min(weight_decay_max, weight_decay_val))
                for g in optimizer.param_groups:
                    g["weight_decay"] = weight_decay_val

                if seq_len_stage_idx < len(seq_len_stages) - 1:
                    if len(seq_len_val_ce_history) >= 10:
                        seq_len_stage_idx += 1
                        seq_len_max = seq_len_stages[seq_len_stage_idx]
                        seq_len_val_ce_history =[]
                        seq_len_improvement_ema = 0.0
                        print(f"  ↑ seq_len_max unlocked → {seq_len_max}  (max val checks reached)")
                    elif seq_len_val_ce_history:
                        prev_ce = seq_len_val_ce_history[-1]
                        improvement = prev_ce - val_ce
                        seq_len_improvement_ema = (
                            (1 - seq_len_improvement_alpha) * seq_len_improvement_ema
                            + seq_len_improvement_alpha * improvement
                        )
                        if seq_len_improvement_ema < seq_len_plateau_thresh:
                            _imp_snapshot = seq_len_improvement_ema
                            seq_len_stage_idx += 1
                            seq_len_max = seq_len_stages[seq_len_stage_idx]
                            seq_len_val_ce_history =[]
                            seq_len_improvement_ema = 0.0
                            print(f"  ↑ seq_len_max unlocked → {seq_len_max}  "
                                  f"(val CE improvement plateaued: "
                                  f"{_imp_snapshot:.4f} < {seq_len_plateau_thresh})")
                    seq_len_val_ce_history.append(val_ce)

            print(
                f"  ↳ Val CE {val_ce:.4f}  "
                f"PPL {torch.exp(val_loss).item():.2f}  "
                f"do→{current_dropout:.3f}  "
                f"wd→{weight_decay_val:.4f}  "
                f"val_every={val_every}  "
                f"vel={ce_vel_ema:+.5f}{marker}"
            )
            model.train()

    torch.save(model.state_dict(), checkpoint)
    _total = time.time() - _train_start
    print(f"\nFinal checkpoint  → {checkpoint}")
    print(f"Best checkpoint   → {best_checkpoint}  (val CE {best_val_loss:.4f})")
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(_total))}  "
          f"({_total/3600:.2f} hours)")

    return model, tokenizer, seq_len_max


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
    print("\n" + "=" * 60)
    print("  NEURO-SYMBOLIC EXPLAINABILITY & GENERATION DEMO")
    print("=" * 60)

    encoded_seed = tokenizer.encode(seed_text)
    context      = torch.tensor([encoded_seed], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        _, symbolic_indices, _, _, _, _, _, _, mean_H, _ps_active, *_ = model(context)
    active_ratio = sum(_ps_active) / len(_ps_active)

    last_tok = symbolic_indices[0, -1]

    print(f"\nSEED PROMPT:\n{'-'*40}\n{seed_text}\n{'-'*40}")
    print("\nInternal submodality representation of the final token (Top-K routing):\n")
    for i, name in enumerate(model.sub_encoder.submodality_names):
        stream_syms = last_tok[i]
        k = stream_syms.shape[0]
        sym_str = "  ".join(f"#{stream_syms[k_].item():03d}" for k_ in range(k))
        print(f"  {name:12s} → {sym_str}")
    print(f"\n  H_ema (smoothed entropy): {mean_H:.3f}  |  Active: {active_ratio*100:.1f}%")

    print(f"\nGenerating {max_new_tokens} new tokens…\n")
    generated_ids  = model.generate(context, max_new_tokens, seq_len, temperature, top_k)
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"GENERATED TEXT:\n{'-'*50}")
    print(generated_text)
    print("-" * 50)


# ==========================================================
# XAI: "MIND READER" VISUALIZATION (HEATMAP)
# ==========================================================

def visualize_submodality_mind_reader(
    model:     SubmodalitySymbolicLanguageModel,
    tokenizer,
    device:    torch.device,
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        print("ERROR: Please install seaborn: pip install seaborn")
        return
    import numpy as np

    print("\n" + "=" * 60)
    print("  RUNNING XAI 'MIND READER' VISUALIZATION...")
    print("=" * 60)

    model.eval()

    word_categories = {
        "Emotion / Affect":["love", "hate", "rage", "grief", "joy",       "fear"],
        "Spatial / Distance":["far",  "near", "above", "behind", "within",  "beyond"],
        "Visual / Light":["bright", "dark", "shadow", "glow", "shining", "dim"],
        "Abstract / Logic":["therefore", "because", "reason", "logic",    "however", "truth"],
        "Size / Magnitude":   ["vast", "tiny", "enormous", "massive", "little", "great"],
        "Position / Order":   ["first", "last", "before", "after", "middle",  "finally"],
    }

    all_words:   list[str]        = []
    all_weights: list[np.ndarray] = []
    category_boundaries: list[int] =[]

    with torch.no_grad():
        for category, words in word_categories.items():
            category_boundaries.append(len(all_words))
            for word in words:
                token_ids    = tokenizer.encode(" " + word)
                token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

                embed     = model.embedding(token_tensor)
                gates     = F.softmax(model.submodality_gate(embed), dim=-1)
                mean_gate = gates.mean(dim=1).squeeze(0).cpu().numpy()

                all_words.append(word)
                all_weights.append(mean_gate)

    weight_matrix      = np.array(all_weights)
    submodality_names  = model.sub_encoder.submodality_names

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        weight_matrix,
        annot=True,
        fmt=".2f",
        cmap="magma",
        xticklabels=submodality_names,
        yticklabels=all_words,
        cbar_kws={"label": "Submodality Gating Weight"},
        ax=ax,
    )

    for row in category_boundaries[1:]:
        ax.axhline(row, color="white", lw=2)

    cat_names = list(word_categories.keys())
    for k, (start, name) in enumerate(zip(category_boundaries, cat_names)):
        end = category_boundaries[k + 1] if k + 1 < len(category_boundaries) else len(all_words)
        mid = (start + end) / 2
        ax.text(
            5.15, mid, name,
            va="center", ha="left", fontsize=9,
            transform=ax.get_yaxis_transform(),
        )

    ax.set_title(
        "Neuro-Symbolic XAI: How the Model 'Sees' Different Words",
        fontsize=14, pad=20,
    )
    ax.set_xlabel("Psychological Submodalities", fontsize=12)
    ax.set_ylabel("Test Words", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved → {save_path}")

    print("Opening heatmap window… (close the window to end the script)")
    plt.show()


# ==========================================================
# AUTO-LABELER: ORGANIC CONCEPT DISCOVERY
# ==========================================================

def auto_label_buckets(
    model,
    tokenizer,
    device:    torch.device,
    filepath:  str = "data.txt",
    top_k:     int = 20,
) -> None:
    print("\n" + "=" * 60)
    print("  DECODING AI PSYCHOLOGY (AUTO-LABELER)...")
    print("=" * 60)

    if not os.path.exists(filepath):
        print(f"[!] Cannot find {filepath} to run Auto-Labeler.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    unique_tokens = list(set(tokenizer.encode(text)))
    all_tokens    = random.sample(unique_tokens, min(3000, len(unique_tokens)))

    bucket_activations = {i:[] for i in range(model.NUM_SUBMODALITIES)}

    model.eval()
    with torch.no_grad():
        for tok in all_tokens:
            word = tokenizer.decode([tok]).strip()

            if len(word) < 4 or not word.isalpha():
                continue

            tok_tensor   = torch.tensor([[tok]], dtype=torch.long).to(device)
            embed        = model.embedding(tok_tensor)
            gates        = F.softmax(model.submodality_gate(embed), dim=-1)
            gate_weights = gates.squeeze().cpu().numpy()

            for i in range(model.NUM_SUBMODALITIES):
                bucket_activations[i].append((word, gate_weights[i]))

    submodality_names = model.sub_encoder.submodality_names

    for i in range(model.NUM_SUBMODALITIES):
        sorted_words = sorted(bucket_activations[i], key=lambda x: x[1], reverse=True)
        seen      = set()
        top_words =[]
        for word, _ in sorted_words:
            word_lower = word.lower()
            if word_lower not in seen:
                seen.add(word_lower)
                top_words.append(word)
            if len(top_words) >= top_k:
                break

        print(f"\n► BUCKET {i}  (Named: '{submodality_names[i]}')")
        print("  Top Words: " + ", ".join(top_words))

    print("\n" + "=" * 60)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    TOKENIZER_MODE = "bpe"         
    BPE_VOCAB_SIZE = None         

    trained_model, tokenizer, s_len = train_model(
        tokenizer_mode = TOKENIZER_MODE,
        bpe_vocab_size = BPE_VOCAB_SIZE,
    )

    evaluate_and_generate(trained_model, tokenizer, s_len)
    visualize_submodality_mind_reader(trained_model, tokenizer, device)
    auto_label_buckets(trained_model, tokenizer, device)

    input("\nScript complete. Press Enter to exit...")