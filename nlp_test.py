"""
Patched v16 (Full Integrity & Compile-Safe Release):
- Fixes Pre-VQ information leak (retains pre_vq_cross_stream_attn for checkpoint compat, bypassed by default)
- Fixes Top-2 routing regularizers:
    * Applies stream_mask and stream_dropout before Top-K (with vectorized k-stream rescue)
    * Computes gate_kl and balance_loss on active soft_probs (prevents dead-stream collapse & gate_kl == 0)
    * Fixes sparsity_loss sign to Gini impurity (1 - sum(w^2)) and applies late-stage gate specialization
- Fixes VQ norm initialization / grow_codebook scale / spherical pair_grad_vec consistency
- Fixes meta_targets space alignment (computed from meta_in before pos_encoder) and meta_balance_loss
- Fixes phase schedule to run all phases cleanly within steps (including ChunkUp / ChunkDown)
- Fixes weight_decay clobbering on no-decay params and growth warmup param classification
- Fixes kickstart_worst_pair polarity (max overlap > threshold)
- Makes forward() torch.compile-friendly (0-d buffers, pre-registered concept_bank, zero-init add_layer)
- Fixes resume_from ordering & structural reconstruction + real KeyboardInterrupt checkpointing
- Aligns XAI Mind Reader / Auto-Labeler with actual Top-2 routing and fixes heatmap label positioning
"""

import collections
import logging
import math
import os
import random
import re as _re
import sys
import time
import warnings

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# ==========================================================
# DEVICE & REPRODUCIBILITY
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ==========================================================
# POSITIONAL ENCODING
# ==========================================================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ==========================================================
# CAUSAL DECODER
# ==========================================================


class CausalDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.attn_drop = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        normed = self.norm1(x)

        q = self.q_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_drop_p = self.attn_drop if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=attn_drop_p, is_causal=True
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn_out = self.o_proj(attn_out)

        x = x + self.drop(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class CausalTransformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalDecoderBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def add_layer(self, dev: torch.device, zero_init_residual: bool = True) -> CausalDecoderBlock:
        d_model = self.blocks[0].norm1.normalized_shape[0]
        n_heads = self.blocks[0].n_heads
        dropout = self.blocks[0].drop.p
        new_block = CausalDecoderBlock(d_model, n_heads, dropout).to(dev)
        if zero_init_residual:
            with torch.no_grad():
                nn.init.zeros_(new_block.o_proj.weight)
                nn.init.zeros_(new_block.ff[2].weight)
                if new_block.ff[2].bias is not None:
                    nn.init.zeros_(new_block.ff[2].bias)
        self.blocks.append(new_block)
        return new_block

    def set_dropout(self, p: float) -> None:
        p = float(max(0.0, min(0.5, p)))
        for block in self.blocks:
            block.drop.p = p
            block.attn_drop = p
            for m in block.ff.modules():
                if isinstance(m, nn.Dropout):
                    m.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)


# ==========================================================
# CROSS-STREAM ATTENTION
# ==========================================================


class CrossStreamAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads //= 2
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.o_proj.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M, N, _ = x.shape
        normed = self.norm(x)

        q = self.q_proj(normed).view(M, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(normed).view(M, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(normed).view(M, N, self.n_heads, self.d_head).transpose(1, 2)

        attn_drop_p = self.drop.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=attn_drop_p, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(M, N, self.d_model)
        attn_out = self.o_proj(attn_out)

        return x + self.drop(attn_out)


# ==========================================================
# NEURO-SYMBOLIC CODEBOOK  (VQ-VAE style)
# ==========================================================


class NeuroSymbolicCodebook(nn.Module):
    def __init__(
        self,
        num_symbols: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.2,
        topk: int = 3,
        topk_tau: float = 0.1,
    ):
        super().__init__()
        self.num_symbols = num_symbols
        self.embedding_dim = embedding_dim
        self.topk = topk

        self.codebook = nn.Embedding(num_symbols, embedding_dim)
        # Unit-RMS normal init matches LayerNorm output scale (~sqrt(d_model) L2 norm)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)

        self.residual_alpha_raw = nn.Parameter(torch.tensor(-2.0))
        self.register_buffer("ema_entropy", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("commitment_cost_buf", torch.tensor(float(commitment_cost), dtype=torch.float32))
        self.register_buffer("ema_decay_buf", torch.tensor(float(ema_decay), dtype=torch.float32))
        self.register_buffer("topk_tau_buf", torch.tensor(float(topk_tau), dtype=torch.float32))
        self.register_buffer("H_target_buf", torch.tensor(0.9 * math.log(max(2, num_symbols)), dtype=torch.float32))

    @property
    def commitment_cost(self) -> float:
        return float(self.commitment_cost_buf.item())

    @commitment_cost.setter
    def commitment_cost(self, val: float) -> None:
        self.commitment_cost_buf.fill_(float(val))

    @property
    def ema_decay(self) -> float:
        return float(self.ema_decay_buf.item())

    @ema_decay.setter
    def ema_decay(self, val: float) -> None:
        self.ema_decay_buf.fill_(float(val))

    @property
    def topk_tau(self) -> float:
        return float(self.topk_tau_buf.item())

    @topk_tau.setter
    def topk_tau(self, val: float) -> None:
        self.topk_tau_buf.fill_(float(val))

    @property
    def H_target(self) -> float:
        return float(self.H_target_buf.item())

    @H_target.setter
    def H_target(self, val: float) -> None:
        self.H_target_buf.fill_(float(val))

    def grow_codebook(self, new_num_symbols: int, dev: torch.device) -> None:
        if new_num_symbols <= self.num_symbols:
            return
        old_n = self.num_symbols
        n_new = new_num_symbols - old_n
        with torch.no_grad():
            existing_w = self.codebook.weight.data
            existing_norms = existing_w.norm(dim=1)
            med_norm = existing_norms.median().clamp(min=1e-4)
            new_w = torch.randn(n_new, self.embedding_dim, device=dev, dtype=existing_w.dtype)
            new_w = F.normalize(new_w, dim=1) * med_norm
            combined = torch.cat([existing_w, new_w], dim=0)
        self.codebook = nn.Embedding(new_num_symbols, self.embedding_dim).to(dev)
        with torch.no_grad():
            self.codebook.weight.data.copy_(combined)
        self.num_symbols = new_num_symbols
        self.H_target = 0.9 * math.log(max(2, new_num_symbols))
        print(f"    ↑ Codebook grown: {old_n} → {new_num_symbols} symbols")

    def revive_dead_entries(self, dev: torch.device, noise_scale: float = 0.1) -> int:
        with torch.no_grad():
            W = self.codebook.weight.data
            norms_sq = (W ** 2).sum(dim=1)
            med_norm_sq = norms_sq.median()
            dead_mask = norms_sq < (med_norm_sq * 0.1)
            n_dead = int(dead_mask.sum().item())
            if n_dead == 0:
                return 0
            live_idx = (~dead_mask).nonzero(as_tuple=True)[0]
            if len(live_idx) == 0:
                return 0
            for dead_i in dead_mask.nonzero(as_tuple=True)[0]:
                src = live_idx[torch.randint(len(live_idx), (1,), device=dev).item()]
                W[dead_i] = W[src] + torch.randn_like(W[src]) * noise_scale
            return n_dead

    def forward(self, features: torch.Tensor):
        flat = features.reshape(-1, self.embedding_dim).float()
        flat_sphere = F.normalize(flat, dim=1)
        cb_sphere = F.normalize(self.codebook.weight.float(), dim=1)
        distances = torch.cdist(flat_sphere, cb_sphere, p=2)

        tau = self.topk_tau_buf.clone().clamp(min=1e-4)
        h_target = self.H_target_buf.clone().clamp(min=0.01)
        commit_cost = self.commitment_cost_buf.clone()

        # Differentiable Usage Loss
        soft_assign = F.softmax(-distances / tau, dim=-1)
        p_bar = soft_assign.mean(dim=0)
        diff_entropy = -(p_bar * torch.log(p_bar + 1e-10)).sum()
        usage_loss = ((diff_entropy / h_target) - 1.0) ** 2

        k_eff = min(self.topk, self.num_symbols)
        topk_dists, topk_idx = distances.topk(k_eff, dim=1, largest=False)
        topk_weights = F.softmax(-topk_dists / tau, dim=1)

        topk_vecs = self.codebook(topk_idx).float()
        quantized_flat = (topk_weights.unsqueeze(-1) * topk_vecs).sum(dim=1)
        quantized = quantized_flat.view_as(features).to(features.dtype)

        e_loss = (topk_weights.detach() * topk_dists).sum(dim=1).mean()
        q_loss = F.mse_loss(quantized_flat, flat.detach())

        n_sample = min(self.num_symbols, 64)
        sample_idx = torch.randperm(self.num_symbols, device=self.codebook.weight.device)[:n_sample]
        W_sample = F.normalize(self.codebook.weight[sample_idx].float(), dim=1)
        cos_sim = torch.mm(W_sample, W_sample.t())
        mask = ~torch.eye(n_sample, dtype=torch.bool, device=cos_sim.device)
        diversity_loss = cos_sim[mask].mean().clamp(min=0.0) if n_sample > 1 else cos_sim.new_zeros(())

        vq_loss = q_loss + commit_cost * e_loss + 0.01 * diversity_loss + 0.5 * usage_loss

        quantized_st = features + (quantized - features).detach()
        alpha = 0.5 * torch.sigmoid(self.residual_alpha_raw)
        quantized_out = quantized_st + alpha * features

        with torch.no_grad():
            flat_idx = topk_idx.flatten()
            one_hot = F.one_hot(flat_idx, self.num_symbols).float()
            probs = one_hot.mean(0)
            batch_H = -(probs * torch.log(probs + 1e-10)).sum().to(self.ema_entropy.dtype)
            ema_d = self.ema_decay_buf.to(self.ema_entropy.dtype)
            self.ema_entropy.mul_(1.0 - ema_d).add_(batch_H * ema_d)
            active_ratio = (one_hot.sum(0) > 0).float().mean()
            topk_H = -(topk_weights * torch.log(topk_weights + 1e-10)).sum(dim=1).mean()

        indices = topk_idx.view(*features.shape[:-1], k_eff)

        return (
            quantized_out,
            quantized_st,
            indices,
            vq_loss,
            self.ema_entropy.detach(),
            active_ratio,
            topk_H,
        )


# ==========================================================
# SUBMODALITY FEATURE EXTRACTOR
# ==========================================================


class VisualSubmodalityEncoder(nn.Module):
    BASE_SUBMODALITY_NAMES = [
        "Brightness", "Distance", "Focus", "Color",
        "Perspective", "Size", "Position",
    ]

    def __init__(self, d_model: int, orthogonality_lambda: float = 0.01):
        super().__init__()
        self.submodality_names = list(self.BASE_SUBMODALITY_NAMES)
        self.extract_brightness = nn.Linear(d_model, d_model)
        self.extract_distance = nn.Linear(d_model, d_model)
        self.extract_focus = nn.Linear(d_model, d_model)
        self.extract_color = nn.Linear(d_model, d_model)
        self.extract_perspective = nn.Linear(d_model, d_model)
        self.extract_size = nn.Linear(d_model, d_model)
        self.extract_position = nn.Linear(d_model, d_model)

        with torch.no_grad():
            W_b = self.extract_brightness.weight.data
            W_f = self.extract_focus.weight.data
            for row in range(W_f.shape[0]):
                b = W_b[row]
                f = W_f[row]
                f = f - (f @ b / (b @ b + 1e-8)) * b
                W_f[row] = f
            W_f_norms = W_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
            orig_norms = self.extract_brightness.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.extract_focus.weight.data = W_f / W_f_norms * orig_norms

        with torch.no_grad():
            W_per = self.extract_perspective.weight.data
            W_dis = self.extract_distance.weight.data
            W_pos = self.extract_position.weight.data
            orig_pos_norms = W_pos.norm(dim=1, keepdim=True).clamp(min=1e-8)
            for row in range(W_pos.shape[0]):
                p = W_pos[row]
                v = W_per[row]
                p = p - (p @ v / (v @ v + 1e-8)) * v
                d = W_dis[row]
                p = p - (p @ d / (d @ d + 1e-8)) * d
                W_pos[row] = p
            W_pos_norms = W_pos.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.extract_position.weight.data = W_pos / W_pos_norms * orig_pos_norms

        self.register_buffer(
            "_orth_lambda_buf",
            torch.tensor(float(orthogonality_lambda), dtype=torch.float32),
        )

    @property
    def orthogonality_lambda(self) -> float:
        return float(self._orth_lambda_buf.item())

    @orthogonality_lambda.setter
    def orthogonality_lambda(self, val: float) -> None:
        self._orth_lambda_buf.fill_(float(val))

    def feature_orthogonality_loss(self, sub_features: torch.Tensor, scaled: bool = False) -> torch.Tensor:
        B, T, N_s, D = sub_features.shape
        N = B * T
        loss = sub_features.new_zeros((), dtype=torch.float32)

        normed = []
        for i in range(N_s):
            fi = sub_features[:, :, i, :].reshape(N, D).float()
            fi = fi - fi.mean(0, keepdim=True)
            fi = F.normalize(fi, dim=1)
            normed.append(fi)

        for i in range(N_s):
            for j in range(i + 1, N_s):
                cross_gram = (normed[i].t() @ normed[j]) / max(1, N)
                loss = loss + cross_gram.pow(2).sum()

        return (self._orth_lambda_buf.clone() * loss) if scaled else loss

    def add_stream(
        self,
        name: str,
        d_model: int,
        dev: torch.device,
        init_weight: torch.Tensor | None = None,
    ) -> nn.Linear:
        attr = f"extract_{name.lower()}"
        if hasattr(self, attr):
            raise ValueError(f"Stream '{name}' already exists.")
        new_lin = nn.Linear(d_model, d_model).to(dev)
        if init_weight is not None:
            with torch.no_grad():
                new_lin.weight.copy_(init_weight.to(dev).to(new_lin.weight.dtype))
                nn.init.zeros_(new_lin.bias)
        setattr(self, attr, new_lin)
        self.submodality_names.append(name)
        return new_lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        streams = [getattr(self, f"extract_{n.lower()}")(x) for n in self.submodality_names]
        return torch.stack(streams, dim=2)


# ==========================================================
# CHUNKING MODULES  (NLP chunk-up / chunk-down)
# ==========================================================


class ChunkUpCodebook(nn.Module):
    def __init__(self, d_model: int, num_categories: int, commitment_cost: float = 0.10):
        super().__init__()
        self.d_model = d_model
        self.num_categories = num_categories
        self.norm = nn.LayerNorm(d_model)
        self.codebook = NeuroSymbolicCodebook(
            num_symbols=num_categories,
            embedding_dim=d_model,
            commitment_cost=commitment_cost,
            topk=1,
        )
        self.residual_raw = nn.Parameter(torch.tensor(-2.0))

    def forward(self, meta_quantized: torch.Tensor):
        cat_in = self.norm(meta_quantized)

        (cat_q_out, cat_q_st, _idx, cat_vq_loss, cat_ema_H, cat_active, cat_topk_H) = self.codebook(cat_in)

        alpha = 0.5 * torch.sigmoid(self.residual_raw)
        category_quantized = meta_quantized + alpha * cat_q_out

        return category_quantized, cat_q_st, cat_vq_loss, cat_ema_H, cat_active, cat_topk_H


class ChunkDownHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, pure_cat_q: torch.Tensor, meta_quantized_target: torch.Tensor) -> torch.Tensor:
        reconstructed = self.net(pure_cat_q)
        rec_norm = F.normalize(reconstructed.float(), dim=-1)
        tgt_norm = F.normalize(meta_quantized_target.detach().float(), dim=-1)
        return F.mse_loss(rec_norm, tgt_norm)


# ==========================================================
# MAIN MODEL
# ==========================================================


class SubmodalitySymbolicLanguageModel(nn.Module):
    NUM_SUBMODALITIES = 7

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        num_symbols: int = 512,
        n_heads: int = 4,
        num_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        self.sub_encoder = VisualSubmodalityEncoder(d_model)

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

        # Retained in __init__ for checkpoint compatibility; bypassed by default (enable_pre_vq_attn=False)
        self.pre_vq_cross_stream_attn = CrossStreamAttention(
            d_model, n_heads=max(1, n_heads // 2), dropout=dropout
        )
        self.enable_pre_vq_attn: bool = False

        self.cross_stream_attn = CrossStreamAttention(
            d_model, n_heads=max(1, n_heads // 2), dropout=dropout
        )

        self.stream_dropout_p: float = 0.15
        self.gate_temperature: float = 0.5
        self.top_k_streams: int = 2

        self.composite_residual_raw = nn.Parameter(torch.tensor(-1.0))

        self.meta_norm = nn.LayerNorm(d_model)
        self.meta_codebook = NeuroSymbolicCodebook(
            num_symbols=num_symbols,
            embedding_dim=d_model,
            commitment_cost=0.10,
        )
        self.meta_residual_raw = nn.Parameter(torch.tensor(-2.0))
        self.meta_pred_head = nn.Linear(d_model, num_symbols, bias=False)

        num_categories = max(4, num_symbols // 4)
        self.chunk_up = ChunkUpCodebook(
            d_model=d_model,
            num_categories=num_categories,
            commitment_cost=0.10,
        )
        self.category_residual_raw = nn.Parameter(torch.tensor(-3.0))
        self.chunk_down = ChunkDownHead(d_model)

        self.decoder = CausalTransformerDecoder(d_model, n_heads, num_layers, dropout)

        self.register_buffer(
            "concept_bank",
            torch.empty((0, d_model), dtype=torch.float32),
            persistent=False,
        )

        self.stream_active: list[bool] = [True] * self.NUM_SUBMODALITIES
        self._n_active_streams: int = self.NUM_SUBMODALITIES
        self._update_stream_mask()

        # Phase thresholds (overridden proportionally by train_model)
        self.current_step = 0
        self.phase_post_vq_attn_step = 750
        self.phase_pair_decor_step = 900
        self.phase_meta_step = 1500
        self.phase_meta_loss_step = 1650
        self.phase_chunk_step = 2160
        self.phase_chunk_loss_step = 2280

        # Static bool phase toggles updated at transitions to keep torch.compile fast
        self._phase_post_vq_attn = False
        self._phase_pair_decor = False
        self._phase_meta = False
        self._phase_meta_loss = False
        self._phase_chunk = False
        self._phase_chunk_loss = False

    def set_phase(self, step: int) -> None:
        self.current_step = int(step)
        self._phase_post_vq_attn = step >= self.phase_post_vq_attn_step
        self._phase_pair_decor = step >= self.phase_pair_decor_step
        self._phase_meta = step >= self.phase_meta_step
        self._phase_meta_loss = step >= self.phase_meta_loss_step
        self._phase_chunk = step >= self.phase_chunk_step
        self._phase_chunk_loss = step >= self.phase_chunk_loss_step

    def _update_stream_mask(self) -> None:
        dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        mask = torch.tensor(
            [1.0 if a else 0.0 for a in self.stream_active],
            dtype=torch.float32,
            device=dev,
        )
        self.register_buffer("stream_mask", mask)
        self._n_active_streams = max(1, sum(1 for a in self.stream_active if a))

    def compute_gate_weights(
        self,
        embed: torch.Tensor,
        apply_stream_dropout: bool = False,
    ):
        """
        Shared Hard Top-K routing used identically across forward() and XAI diagnostics.
        Returns (gate_weights, soft_probs, keep_bool).
        """
        gate_logits = self.submodality_gate(embed)
        if getattr(self, "gate_bias_override", None) is not None:
            gate_logits = gate_logits + self.gate_bias_override

        B_sz, _, N_s = gate_logits.shape
        k_eff = min(self.top_k_streams, max(1, self._n_active_streams))

        active_mask = (self.stream_mask > 0).view(1, 1, N_s)

        if (
            apply_stream_dropout
            and self.training
            and self.stream_dropout_p > 0.0
            and self._n_active_streams > k_eff
        ):
            keep_prob = 1.0 - self.stream_dropout_p
            keep = torch.bernoulli(
                torch.full((B_sz, 1, N_s), keep_prob, device=gate_logits.device, dtype=torch.float32)
            ) * active_mask.float()
            # Vectorized rescue ensuring >= k_eff active streams survive per batch element
            tie_break = torch.rand_like(keep) * active_mask.float()
            score = keep * 2.0 + tie_break
            _, rescue_idx = torch.topk(score, k=k_eff, dim=-1)
            keep = keep.scatter(-1, rescue_idx, 1.0)
            keep_bool = (keep > 0.5) & active_mask
        else:
            keep_bool = active_mask.expand(B_sz, 1, N_s)

        valid_logits = gate_logits.masked_fill(~keep_bool, float("-inf"))
        tau = self.gate_temperature

        soft_probs = F.softmax(valid_logits / tau, dim=-1)

        _, topk_idx = torch.topk(valid_logits, k=k_eff, dim=-1)
        topk_mask = torch.zeros_like(valid_logits, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_idx, True)
        masked_logits = valid_logits.masked_fill(~topk_mask, float("-inf"))
        gate_weights = F.softmax(masked_logits / tau, dim=-1)

        return gate_weights, soft_probs, keep_bool

    def grow_meta_pred_head(
        self,
        new_num_symbols: int,
        dev: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        old_n = self.meta_pred_head.out_features
        if new_num_symbols <= old_n:
            return
        d = self.meta_pred_head.in_features
        new_head = nn.Linear(d, new_num_symbols, bias=False).to(dev)
        with torch.no_grad():
            new_head.weight[:old_n] = self.meta_pred_head.weight.data
            nn.init.zeros_(new_head.weight[old_n:])
        old_params = list(self.meta_pred_head.parameters())
        old_ids = {id(p) for p in old_params}
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in old_ids]
        for p in old_params:
            optimizer.state.pop(p, None)
        self.meta_pred_head = new_head
        print(f"    ↑ meta_pred_head grown: {old_n} → {new_num_symbols} classes")

    def add_submodality(
        self,
        name: str,
        dev: torch.device,
        optimizer: torch.optim.Optimizer,
        lr: float,
        weight_decay: float,
        num_symbols: int | None = None,
        boost_lr: float | None = None,
        init_weight: torch.Tensor | None = None,
    ) -> list:
        if num_symbols is None:
            num_symbols = self.codebooks[0].num_symbols
        d = self.d_model

        new_lin = self.sub_encoder.add_stream(name, d, dev, init_weight=init_weight)
        new_cb = NeuroSymbolicCodebook(num_symbols, d).to(dev)
        self.codebooks.append(new_cb)

        new_norm = nn.LayerNorm(d).to(dev)
        self.pre_vq_norms.append(new_norm)

        old_gate = self.submodality_gate
        new_N = self.NUM_SUBMODALITIES + 1
        new_gate = nn.Linear(d, new_N).to(dev)
        with torch.no_grad():
            new_gate.weight[: self.NUM_SUBMODALITIES] = old_gate.weight
            new_gate.bias[: self.NUM_SUBMODALITIES] = old_gate.bias
            nn.init.zeros_(new_gate.weight[self.NUM_SUBMODALITIES :])
            new_gate.bias.data[self.NUM_SUBMODALITIES] = 0.5
        self.submodality_gate = new_gate

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
        for p in old_gate_params:
            optimizer.state.pop(p, None)

        _no_decay_keys = ("bias", "norm", "codebook", "residual_alpha_raw", "ema_entropy", "meta_residual_raw")
        _new_decay = [
            p
            for n, p in (list(new_lin.named_parameters()) + list(new_gate.named_parameters()))
            if p.ndim > 1 and not any(k in n for k in _no_decay_keys)
        ]
        _new_nodecay = [p for p in new_params if id(p) not in {id(q) for q in _new_decay}]
        _boost_lr = boost_lr if boost_lr is not None else lr

        if _new_decay:
            optimizer.add_param_group({
                "params": _new_decay,
                "lr": _boost_lr,
                "weight_decay": weight_decay,
                "is_decay": True,
            })
        if _new_nodecay:
            optimizer.add_param_group({
                "params": _new_nodecay,
                "lr": _boost_lr,
                "weight_decay": 0.0,
                "is_decay": False,
            })

        self.NUM_SUBMODALITIES += 1
        self.stream_active.append(True)
        self._update_stream_mask()
        print(f"  ✚ Stream added: '{name}'  (total active: {sum(self.stream_active)}/{self.NUM_SUBMODALITIES})")
        return new_params

    def deactivate_submodality(self, idx: int) -> None:
        name = self.sub_encoder.submodality_names[idx]
        self.stream_active[idx] = False
        self._update_stream_mask()
        active = sum(self.stream_active)
        print(f"  ✖ Stream deactivated: '{name}' (idx {idx})  ({active}/{self.NUM_SUBMODALITIES} active)")

    def kickstart_worst_pair(
        self,
        i: int,
        j: int,
        dev: torch.device,
        noise_scale: float = 0.15,
        fraction: float = 0.25,
    ) -> int:
        with torch.no_grad():
            Wi = F.normalize(self.codebooks[i].codebook.weight.data, dim=1)
            Wj = self.codebooks[j].codebook.weight.data
            Wj_norm = F.normalize(Wj, dim=1)
            sim = torch.mm(Wi, Wj_norm.t())
            max_sim_j, _ = sim.max(dim=0)
            k_perturb = max(1, int(fraction * self.codebooks[j].num_symbols))
            _, top_idx = max_sim_j.topk(k_perturb)
            for idx in top_idx:
                Wj[idx] += torch.randn_like(Wj[idx]) * noise_scale
            name_i = self.sub_encoder.submodality_names[i]
            name_j = self.sub_encoder.submodality_names[j]
            print(f"  ⚡ Kickstart ({name_i},{name_j}): perturbed {k_perturb} of {self.codebooks[j].num_symbols} centroids")
            return k_perturb

    def forward(self, x: torch.Tensor):
        embed = self.embedding(x)
        sub_features = self.sub_encoder(embed)

        # Pre-VQ cross-stream attention is bypassed by default to prevent pre-VQ leakage
        if self.enable_pre_vq_attn:
            B_f, T_f, N_f, D_f = sub_features.shape
            sub_features = self.pre_vq_cross_stream_attn(
                sub_features.reshape(B_f * T_f, N_f, D_f)
            ).reshape(B_f, T_f, N_f, D_f)

        symbolic_vectors = []
        symbolic_indices = []
        total_vq_loss = embed.new_zeros((), dtype=torch.float32)
        total_ema_H = embed.new_zeros((), dtype=torch.float32)
        per_stream_active_list = []
        per_stream_topk_H_list = []

        for i in range(self.NUM_SUBMODALITIES):
            normed_stream = self.pre_vq_norms[i](sub_features[:, :, i, :])
            quantized_out, _, indices, vq_loss, ema_H, active_ratio, topk_H = self.codebooks[i](normed_stream)
            symbolic_vectors.append(quantized_out)
            symbolic_indices.append(indices)
            total_vq_loss = total_vq_loss + vq_loss / self.NUM_SUBMODALITIES
            total_ema_H = total_ema_H + ema_H
            per_stream_active_list.append(active_ratio)
            per_stream_topk_H_list.append(topk_H)

        per_stream_active = torch.stack(per_stream_active_list)
        per_stream_topk_H = torch.stack(per_stream_topk_H_list)

        gate_weights, soft_probs, keep_bool = self.compute_gate_weights(
            embed, apply_stream_dropout=self.training
        )

        # Sparsity = Gini impurity within Top-K weights (0.5 when 50/50 split, 0.0 when one-hot)
        sparsity_loss = 1.0 - torch.mean(torch.sum(gate_weights ** 2, dim=-1))

        sym_stack = torch.stack(symbolic_vectors, dim=2)
        sym_stack = sym_stack * keep_bool.unsqueeze(-1).to(sym_stack.dtype)

        if self._phase_post_vq_attn:
            B_, T_, N_, D_ = sym_stack.shape
            sym_stack = self.cross_stream_attn(
                sym_stack.reshape(B_ * T_, N_, D_)
            ).reshape(B_, T_, N_, D_)

        composite = (sym_stack * gate_weights.unsqueeze(-1)).sum(dim=2)
        composite_alpha = 0.5 * torch.sigmoid(self.composite_residual_raw)
        composite = composite + composite_alpha * embed

        meta_in = self.meta_norm(composite)
        (
            meta_quantized,
            _,
            meta_idx,
            meta_vq_loss,
            meta_ema_H,
            meta_active_ratio,
            meta_topk_H,
        ) = self.meta_codebook(meta_in)

        # Next-position meta-symbol target comes directly from meta_in before pos_encoder
        meta_targets = meta_idx[:, 1:, 0].detach().contiguous()

        meta_alpha = 0.5 * torch.sigmoid(self.meta_residual_raw)
        if self._phase_meta:
            composite = composite + meta_alpha * meta_quantized

        (
            category_quantized,
            pure_cat_q,
            category_vq_loss,
            category_ema_H,
            category_active_ratio,
            category_topk_H,
        ) = self.chunk_up(meta_quantized)

        category_alpha = 0.5 * torch.sigmoid(self.category_residual_raw)
        if self._phase_chunk:
            composite = composite + category_alpha * category_quantized

        chunk_down_loss = self.chunk_down(pure_cat_q, meta_quantized)
        if not self._phase_meta_loss:
            meta_vq_loss = meta_vq_loss * 0.0
        if not self._phase_chunk_loss:
            category_vq_loss = category_vq_loss * 0.0
            chunk_down_loss = chunk_down_loss * 0.0

        # Gate KL measured over active soft_probs so Top-K cap does not zero it out
        log_soft = torch.where(
            soft_probs > 0,
            torch.log(soft_probs.clamp_min(1e-10)),
            torch.zeros_like(soft_probs),
        )
        soft_entropy = -(soft_probs * log_soft).sum(dim=-1).mean()
        n_kept_eff = keep_bool.float().sum(dim=-1).mean().clamp(min=1.0)
        target_entropy = 0.7 * torch.log(n_kept_eff)
        gate_kl = torch.clamp(soft_entropy - target_entropy, min=0.0)

        if self.training:
            feat_orth_loss = self.sub_encoder.feature_orthogonality_loss(sub_features, scaled=False)

            if self._phase_pair_decor:
                N_s = self.NUM_SUBMODALITIES
                with torch.no_grad():
                    hists = []
                    for i in range(N_s):
                        idx_i = symbolic_indices[i].reshape(-1)
                        K_i = self.codebooks[i].num_symbols
                        h_i = F.one_hot(idx_i, K_i).float().mean(0)
                        hists.append(h_i)
                    max_K = max(h.shape[0] for h in hists)
                    padded = [F.pad(h, (0, max_K - h.shape[0])) + 1e-10 for h in hists]
                    padded = torch.stack([p / p.sum() for p in padded], dim=0)

                    iu, ju = torch.triu_indices(N_s, N_s, offset=1, device=x.device)
                    p_i = padded[iu]
                    p_j = padded[ju]
                    m = 0.5 * (p_i + p_j)
                    js = 0.5 * (
                        (p_i * torch.log(p_i / m)).sum(dim=-1)
                        + (p_j * torch.log(p_j / m)).sum(dim=-1)
                    )
                    pair_js_vec = math.log(2) - js
                else_iu, else_ju = iu, ju

                # Spherical distances matching NeuroSymbolicCodebook lookup geometry
                f_streams = [
                    F.normalize(self.pre_vq_norms[i](sub_features[:, :, i, :]).reshape(-1, self.d_model).float(), dim=1)
                    for i in range(N_s)
                ]
                cb_spheres = [
                    F.normalize(self.codebooks[i].codebook.weight.float(), dim=1)
                    for i in range(N_s)
                ]
                d_owns = torch.stack([
                    torch.cdist(f_streams[i], cb_spheres[i], p=2).min(dim=1).values.mean()
                    for i in range(N_s)
                ])
                cross_rows = []
                for i in range(N_s):
                    row_cols = [
                        torch.cdist(f_streams[i], cb_spheres[j], p=2).min(dim=1).values.mean()
                        for j in range(N_s)
                    ]
                    cross_rows.append(torch.stack(row_cols))
                cross_mean = torch.stack(cross_rows, dim=0)

                margin = 0.25
                pair_grad_vec = 0.5 * (
                    F.relu(d_owns[else_iu] - cross_mean[else_iu, else_ju] + margin)
                    + F.relu(d_owns[else_ju] - cross_mean[else_ju, else_iu] + margin)
                )
                sym_decor_loss = pair_js_vec.mean()
            else:
                pair_js_vec = sub_features.new_empty((0,), dtype=torch.float32)
                pair_grad_vec = sub_features.new_empty((0,), dtype=torch.float32)
                sym_decor_loss = sub_features.new_zeros((), dtype=torch.float32)
        else:
            feat_orth_loss = sub_features.new_zeros((), dtype=torch.float32)
            sym_decor_loss = sub_features.new_zeros((), dtype=torch.float32)
            pair_js_vec = sub_features.new_empty((0,), dtype=torch.float32)
            pair_grad_vec = sub_features.new_empty((0,), dtype=torch.float32)

        composite = self.pos_encoder(composite)
        decoder_out = self.decoder(composite)
        logits = F.linear(decoder_out, self.embedding.weight)

        meta_pred_logits = self.meta_pred_head(decoder_out[:, :-1, :])
        if not self._phase_meta_loss:
            meta_pred_logits = meta_pred_logits.detach() + 0.0 * meta_pred_logits

        return (
            logits,
            torch.stack(symbolic_indices, dim=2),
            total_vq_loss,
            feat_orth_loss,
            gate_kl,
            sym_decor_loss,
            gate_weights,
            soft_probs,
            total_ema_H / self.NUM_SUBMODALITIES,
            per_stream_active,
            pair_js_vec,
            pair_grad_vec,
            sub_features.detach() if self.training else None,
            per_stream_topk_H,
            meta_vq_loss,
            meta_ema_H,
            meta_active_ratio,
            meta_topk_H,
            meta_pred_logits,
            meta_targets,
            category_vq_loss,
            category_ema_H,
            category_active_ratio,
            category_topk_H,
            chunk_down_loss,
            sparsity_loss,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        seq_len: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
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
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                min_val = topk_vals[:, -1].unsqueeze(1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            if top_p < 1.0:
                probs_sorted, sorted_idx = torch.sort(F.softmax(logits, dim=-1), descending=True)
                cumulative = torch.cumsum(probs_sorted, dim=-1)
                remove_mask = cumulative - probs_sorted > top_p
                sorted_logits = logits.gather(1, sorted_idx)
                sorted_logits[remove_mask] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ==========================================================
# TOKENIZERS
# ==========================================================


class WordLevelTokenizer:
    PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"

    def __init__(self):
        self.token2id, self.id2token, self._built = {}, {}, False

    @staticmethod
    def _split(text: str) -> list[str]:
        return _re.findall(r"\w+|[^\w\s]", text, _re.UNICODE)

    def build_vocab(self, text: str, min_freq: int = 1) -> None:
        counts = collections.Counter(self._split(text))
        specials = [self.PAD, self.UNK, self.BOS, self.EOS]
        vocab = specials + [t for t, c in counts.most_common() if c >= min_freq]
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self._built = True

    def encode(self, text: str) -> list[int]:
        unk_id = self.token2id[self.UNK]
        return [self.token2id.get(t, unk_id) for t in self._split(text)]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id2token.get(i, self.UNK) for i in ids]
        out = []
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
    PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"

    def __init__(self, vocab_size: int = 4000):
        self.vocab_size, self._tok = vocab_size, None

    def train(self, text: str) -> None:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        print(f"  BPE (fast/C++): training to vocab {self.vocab_size:,}…", flush=True)
        tok = Tokenizer(BPE(unk_token=self.UNK))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[self.PAD, self.UNK, self.BOS, self.EOS],
            show_progress=True,
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
    PAD, UNK, BOS, EOS, EOW = "<PAD>", "<UNK>", "<BOS>", "<EOS>", "</w>"

    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.merges, self.token2id, self.id2token, self._built = [], {}, {}, False

    @staticmethod
    def _word_freqs(text: str) -> dict[tuple[str, ...], int]:
        words = _re.findall(r"\S+", text)
        return {tuple(list(w) + ["</w>"]): c for w, c in collections.Counter(words).items()}

    @staticmethod
    def _get_pairs(vocab: dict[tuple, int]) -> dict[tuple[str, str], int]:
        pairs = {}
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: tuple[str, str], vocab: dict) -> dict:
        merged = "".join(pair)
        pattern = _re.compile(r"(?<!\S)" + _re.escape(" ".join(pair)) + r"(?!\S)")
        new_vocab = {}
        for word, freq in vocab.items():
            word_str = pattern.sub(merged, " ".join(word))
            new_vocab[tuple(word_str.split())] = freq
        return new_vocab

    def train(self, text: str) -> None:
        word_freqs = self._word_freqs(text)
        base_chars = set()
        for word in word_freqs:
            base_chars.update(word)
        vocab_tokens = [self.PAD, self.UNK, self.BOS, self.EOS] + sorted(base_chars)
        num_merges = max(0, self.vocab_size - len(vocab_tokens))
        print(f"  BPE (pure Python): {len(vocab_tokens)} base chars, learning {num_merges} merges…", flush=True)
        report_every = max(1, num_merges // 20)
        for merge_idx in range(num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            word_freqs = self._merge_vocab(best, word_freqs)
            self.merges.append(best)
            vocab_tokens.append("".join(best))
            if (merge_idx + 1) % report_every == 0 or merge_idx == num_merges - 1:
                print(
                    f"    {100 * (merge_idx + 1) / num_merges:5.1f}%  merge {merge_idx+1}/{num_merges}  latest: {''.join(best)!r}",
                    flush=True,
                )
        self.token2id = {t: i for i, t in enumerate(vocab_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self._built = True
        print(f"  BPE vocab: {len(self.token2id):,} tokens  ({len(self.merges):,} merges learned)")

    def _tokenize_word(self, word: str) -> list[str]:
        symbols = list(word) + [self.EOW]
        for a, b in self.merges:
            merged = a + b
            i, new_syms = 0, []
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
        ids = []
        for word in _re.findall(r"\S+", text):
            for sym in self._tokenize_word(word):
                ids.append(self.token2id.get(sym, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id2token.get(i, self.UNK) for i in ids]
        return "".join(tokens).replace(self.EOW, " ").strip()

    @property
    def n_vocab(self) -> int:
        return len(self.token2id)


def _make_bpe_tokenizer(vocab_size: int):
    try:
        import tokenizers  # noqa: F401
        return FastBPETokenizer(vocab_size=vocab_size)
    except ImportError:
        print(
            "  [hint] Install `tokenizers` for 100× faster BPE: pip install tokenizers\n"
            "  Falling back to pure Python BPE.",
            flush=True,
        )
        return BPETokenizer(vocab_size=vocab_size)


# ==========================================================
# DATA LOADING
# ==========================================================


def load_data(filepath: str = "data.txt", tokenizer_mode: str = "tiktoken", bpe_vocab_size: int | None = None):
    if not os.path.exists(filepath):
        print(f"[{filepath} not found] Writing a mini Shakespeare excerpt…")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\n")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    mode = tokenizer_mode.lower().strip()
    if mode == "tiktoken":
        tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = tokenizer.n_vocab
        print(f"  Tokenizer: tiktoken GPT-2 BPE  (vocab {vocab_size:,})")
    elif mode == "word":
        tokenizer = WordLevelTokenizer()
        tokenizer.build_vocab(text)
        vocab_size = tokenizer.n_vocab
        print(f"  Tokenizer: word-level  (vocab {vocab_size:,})")
    elif mode == "bpe":
        if bpe_vocab_size is None:
            unique_words = len(set(_re.findall(r"\S+", text)))
            bpe_vocab_size = int(max(1000, unique_words * 1.5))
        print(f"  Tokenizer: BPE — vocab size = {bpe_vocab_size:,}")
        tokenizer = _make_bpe_tokenizer(bpe_vocab_size)
        tokenizer.train(text)
        vocab_size = tokenizer.n_vocab
    else:
        raise ValueError(f"Unknown tokenizer_mode '{tokenizer_mode}'.")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = len(data)

    if n < 50_000:
        train_data, val_data = data, None
        print(f"  Tokens: {n:,} (< 50k → no val split)")
    else:
        val_frac = 0.05 if n < 200_000 else 0.10 if n < 1_000_000 else 0.20
        split = int((1.0 - val_frac) * n)
        train_data, val_data = data[:split], data[split:]
        print(f"  Tokens: {n:,}  |  train {len(train_data):,}  val {len(val_data):,}  ({int(val_frac*100)}% val)")

    return train_data, val_data, tokenizer, vocab_size


def _random_batch(data: torch.Tensor, batch_size: int, seq_len: int):
    max_start = data.shape[0] - seq_len - 1
    ix = torch.randint(0, max_start, (batch_size,), device=data.device)
    offsets = torch.arange(seq_len, device=data.device)
    idx = ix.unsqueeze(1) + offsets.unsqueeze(0)
    return data[idx], data[idx + 1]


# ==========================================================
# TRAINING
# ==========================================================


def stream_age_weight(stream_idx: int, total_streams: int) -> float:
    age = stream_idx / max(1, total_streams - 1)
    return 1.0 + 3.0 * age


def train_model(
    filepath: str = "data.txt",
    tokenizer_mode: str = "tiktoken",
    bpe_vocab_size: int | None = None,
    steps: int = 3000,
    resume_from: str | None = None,
):
    lr = 3e-4
    lr_warmup_base = 100
    weight_decay = 0.01
    orth_tau = max(40, int(steps * 0.033))
    checkpoint = "neuro_symbolic_model.pt"
    best_checkpoint = "neuro_symbolic_model_best.pt"
    interrupt_checkpoint = "neuro_symbolic_model_interrupt.pt"

    batch_size_tokens = 1024
    batch_size_min = 4
    batch_size_max = 64
    batch_size = 32

    grad_clip = 1.0
    grad_norm_ema = 1.0
    grad_norm_alpha = 0.05
    grad_clip_factor = 1.5
    grad_clip_min = 0.3
    grad_clip_max = 5.0

    gate_gamma = 0.05
    target_gate_kl = 0.15
    gate_gamma_lr = 0.005
    gate_gamma_min = 0.005
    gate_gamma_max = 0.5
    gate_specialize_step = int(steps * 0.70)
    gate_specialize_act_threshold = 0.65
    gate_peak_lambda_max = 0.015
    gate_sparsity_lambda_max = 0.02

    sym_decor_lambda = 0.01

    vq_beta = 0.25
    target_active = 0.72
    vq_beta_lr = 0.01
    vq_beta_min = 0.15
    vq_beta_max = 0.5

    h_target_max_frac = 1.0
    h_target_min_frac = 0.3
    h_target_ramp_steps = max(200, int(steps * 0.66))

    orth_T0 = steps
    orth_unlocked = False

    current_dropout = 0.1
    dropout_min = 0.0
    dropout_max = 0.4
    dropout_lr = 0.01
    overfit_threshold = 1.15

    alpha_reg_lambda = 0.005
    target_alpha = 0.15
    alpha_lr = 0.005
    alpha_lambda_min = 0.001
    alpha_lambda_max = 0.05

    orth_lambda = 2e-5
    target_orth = 0.5
    orth_lambda_lr = 0.005
    orth_lambda_min = 1e-6
    orth_lambda_max = 1e-3

    val_every = 200
    val_every_max = 1000

    candidate_streams = [
        "Texture", "Motion", "Valence", "Agency",
        "Causality", "Certainty", "Temporal",
        "Intent", "Ownership", "Risk", "Hierarchy",
        "Conflict", "Probability", "Obligation",
    ]
    next_candidate_idx = 0
    max_streams = 12
    stream_check_every = 100
    stream_check_max = 2000
    stream_gate_alpha = 0.02
    stream_dead_patience = 1000
    stream_min_dwell = 500
    stream_last_added_step = 0

    weight_decay_val = weight_decay
    weight_decay_min = 0.001
    weight_decay_max = 0.1
    weight_decay_lr = 0.01

    vel_ema_alpha = 0.02
    vel_check_every = 100
    vel_reduce_thresh = 0.0002
    vel_boost_thresh = -0.005
    lr_reduce_factor = 0.99
    lr_boost_factor = 1.03
    vel_cooldown = 300
    vel_min_step = max(300, int(steps * 0.33))
    lr_reductions = 0

    stream_lr_reset_fraction = 0.35

    topk_tau_start = 0.1
    topk_tau_end = 0.5

    meta_topk_tau_start = 0.1
    meta_topk_tau_end = 0.25

    stream_trigger_quorum = 0.6

    revival_check_every = 500
    revival_noise_scale = 0.1

    kickstart_js_thresh = 0.35
    kickstart_patience = 800
    kickstart_cooldown = 1000
    kickstart_last = {}
    kickstart_bad_since = {}

    adam_beta2_start = 0.99
    adam_beta2_end = 0.999
    adam_beta1 = 0.9

    seq_len_stages = [64, 128, 256, 512]
    seq_len_stage_idx = 0
    seq_len_max = seq_len_stages[0]

    batch_tokens_max = 4096

    log_every_base = 100
    log_every_max = 150
    log_every = log_every_base

    commit_cost_lr = 0.005
    commit_cost_min = 0.15
    commit_cost_max = 0.50
    target_active_per = target_active

    meta_vq_beta = 0.10
    meta_vq_beta_min = 0.15
    meta_vq_beta_max = 1.20
    meta_vq_beta_lr = 0.005
    meta_commit_cost_min = 0.15
    meta_commit_cost_max = 0.90
    meta_commit_cost_lr = 0.005
    meta_target_active = 0.72

    meta_pred_lambda = 0.02
    meta_pred_lambda_min = 0.02
    meta_pred_lambda_max = 0.40
    meta_pred_lambda_lr = 0.005
    meta_pred_warmup_steps = int(steps * 0.55)
    meta_pred_acc_ema = 0.0
    meta_pred_acc_alpha = 0.05

    category_vq_beta = 0.10
    category_vq_beta_min = 0.15
    category_vq_beta_max = 0.60
    category_vq_beta_lr = 0.005
    category_target_active = 0.70
    category_active_ema = 0.5

    chunk_down_lambda = 0.05
    chunk_down_lambda_min = 0.01
    chunk_down_lambda_max = 0.20
    chunk_down_lambda_lr = 0.005
    chunk_down_loss_ema = 1.0
    chunk_down_target_mse = 0.005
    chunk_down_warmup_steps = int(steps * 0.76)

    # Proportional developmental schedule so every stage runs inside `steps`
    phase_post_vq_attn_step = int(steps * 0.25)
    phase_pair_decor_step = int(steps * 0.30)
    stream_add_enable_step = int(steps * 0.40)
    codebook_growth_enable_step = int(steps * 0.40)
    revival_enable_step = int(steps * 0.45)
    phase_meta_step = int(steps * 0.50)
    phase_meta_loss_step = int(steps * 0.55)
    meta_growth_enable_step = int(steps * 0.60)
    phase_chunk_step = int(steps * 0.72)
    phase_chunk_loss_step = int(steps * 0.76)
    category_growth_enable_step = int(steps * 0.78)

    ema_decay_start = 0.2
    ema_decay_end = 0.05

    cb_grow_check_every = 200
    cb_grow_active_thresh = 0.95
    cb_grow_patience = 400
    cb_active_ema_alpha = 0.05

    pair_sym_lambda_init = 0.01
    pair_sym_lambda_lr = 0.005
    pair_sym_lambda_min = 0.001
    pair_sym_lambda_max = 20.0
    target_pair_js = 0.05

    growth_reset_fraction = 0.25
    growth_warmup_steps = 500

    h_target_batch_scale = True
    h_target_batch_ref = 1024

    stream_corr_baseline = None
    stream_corr_ema = 0.0
    stream_corr_alpha = 0.05
    stream_corr_trigger = 0.05
    orth_fallback_step = min(1200, int(steps * 0.40))
    stream_corr_calib_done = False

    seq_len_val_ce_history = []
    seq_len_improvement_ema = 0.0
    seq_len_improvement_alpha = 0.3
    seq_len_plateau_thresh = 0.005

    layer_grad_alpha = 0.05
    depth_grad_ratio_thresh = 1.3
    depth_check_every = 200
    depth_last_added = 0
    depth_min_dwell = 500

    train_data, val_data, tokenizer, vocab_size = load_data(filepath, tokenizer_mode, bpe_vocab_size)
    train_data = train_data.to(device)
    if val_data is not None:
        val_data = val_data.to(device)
    n_train = len(train_data)

    if n_train < 512 + 2:
        raise ValueError(f"Corpus too small: {n_train} tokens.")

    num_symbols_target = max(16, int((vocab_size / max_streams) ** 0.5))
    num_symbols = max(16, min(32, num_symbols_target))
    num_symbols_ref = num_symbols_target
    print(f"  num_symbols: {num_symbols} (start, target {num_symbols_target})")

    meta_num_symbols_target = num_symbols_target
    meta_num_symbols_start = max(16, min(32, meta_num_symbols_target))

    _raw_d = (n_train ** 0.25) * 8
    d_model = int(max(128, min(768, round(_raw_d / 64) * 64)))
    print(f"  d_model: {d_model}")

    n_heads = 1
    for h in [8, 4, 2, 1]:
        if d_model % h == 0:
            n_heads = h
            break
    print(f"  n_heads: {n_heads}")

    num_layers = int(max(2, min(8, round(math.log2(n_train / 1000)))))
    num_layers_max = 16
    print(f"  num_layers: {num_layers} (start)  max {num_layers_max}")

    seq_len_base = int(max(8, min(64, round(math.log2(vocab_size) * 2))))
    print(f"  seq_len_base: {seq_len_base}  seq_len_max (curriculum start): {seq_len_max}")

    model = SubmodalitySymbolicLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_symbols=num_symbols,
        n_heads=n_heads,
        num_layers=num_layers,
    ).to(device)

    if meta_num_symbols_start != num_symbols:
        model.meta_codebook.grow_codebook(meta_num_symbols_start, device)
        model.meta_pred_head = nn.Linear(d_model, meta_num_symbols_start, bias=False).to(device)

    meta_cb_active_ema = 0.5
    meta_cb_grow_count = 0
    total_tokens_seen = 0
    growth_warmup_queue = []

    best_val_loss = float("inf")
    val_degrade_count = 0
    val_degrade_limit = 10
    last_train_ce = None
    grad_norm_ema = grad_clip

    ce_vel_ema = 0.0
    ce_prev = None
    vel_last_action = 0
    lr_min = lr / 50
    lr_ceiling = lr

    best_ce_ema = float("inf")
    ce_ema = None
    ce_ema_alpha = 0.05
    steps_since_imp = 0
    lr_patience_base = 400
    lr_patience_grow = 100
    lr_patience = 400

    _ckpt = None
    _resume_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"  ↩ Resuming from {resume_from} ...")
        _ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        _ts = _ckpt["train_state"]

        # 1. Reconstruct dynamically added streams before load_state_dict
        _saved_streams = _ts.get("stream_names", [])
        for s_idx in range(model.NUM_SUBMODALITIES, len(_saved_streams)):
            s_name = _saved_streams[s_idx]
            model.sub_encoder.add_stream(s_name, model.d_model, device)
            model.codebooks.append(NeuroSymbolicCodebook(num_symbols, model.d_model).to(device))
            model.pre_vq_norms.append(nn.LayerNorm(model.d_model).to(device))
            model.NUM_SUBMODALITIES += 1
            model.stream_active.append(True)
        if len(_saved_streams) > 0:
            model.submodality_gate = nn.Linear(model.d_model, model.NUM_SUBMODALITIES).to(device)

        # 2. Reconstruct dynamically added decoder layers
        _saved_layers = _ts.get("num_layers", num_layers)
        for _ in range(len(model.decoder.blocks), _saved_layers):
            model.decoder.add_layer(device)
        num_layers = len(model.decoder.blocks)

        # 3. Reconstruct grown codebooks & meta_pred_head
        for i, target_size in enumerate(_ts["cb_sizes"]):
            if model.codebooks[i].num_symbols < target_size:
                model.codebooks[i].grow_codebook(target_size, device)
        if model.meta_codebook.num_symbols < _ts["meta_cb_size"]:
            model.meta_codebook.grow_codebook(_ts["meta_cb_size"], device)
        if model.meta_pred_head.out_features != _ts["meta_cb_size"]:
            model.meta_pred_head = nn.Linear(model.d_model, _ts["meta_cb_size"], bias=False).to(device)
        if model.chunk_up.codebook.num_symbols < _ts["category_cb_size"]:
            model.chunk_up.codebook.grow_codebook(_ts["category_cb_size"], device)

        model.load_state_dict(_ckpt["model_state"])
        model.stream_active = list(_ts["stream_active"])
        model._update_stream_mask()

    model.phase_post_vq_attn_step = phase_post_vq_attn_step
    model.phase_pair_decor_step = phase_pair_decor_step
    model.phase_meta_step = phase_meta_step
    model.phase_meta_loss_step = phase_meta_loss_step
    model.phase_chunk_step = phase_chunk_step
    model.phase_chunk_loss_step = phase_chunk_loss_step

    N = model.NUM_SUBMODALITIES
    stream_gate_ema = [1.0 / N] * N
    stream_dead_counts = [0] * N
    cb_active_ema = [0.5] * N
    cb_grow_counts = [0] * N
    pair_sym_lambdas = {(i, j): pair_sym_lambda_init for i in range(N) for j in range(i + 1, N)}
    layer_grad_emas = [1.0] * num_layers

    def _stream_thresholds(n):
        return 0.5 / n, 0.95 / n

    lr_warmup = max(100, min(500, int(lr_warmup_base * (num_layers / 4) * (d_model / 256))))

    _no_decay_keys = (
        "bias", "norm", "codebook", "residual_alpha_raw",
        "ema_entropy", "meta_residual_raw", "category_residual_raw",
    )

    def _is_nodecay_param(name: str, p: torch.Tensor) -> bool:
        return p.ndim <= 1 or any(k in name for k in _no_decay_keys)

    if _ckpt is not None and "param_group_specs" in _ckpt["train_state"]:
        _name_to_param = dict(model.named_parameters())
        _rebuilt_groups = []
        for spec in _ckpt["train_state"]["param_group_specs"]:
            _ps = [_name_to_param[nm] for nm in spec["names"] if nm in _name_to_param]
            if _ps:
                _rebuilt_groups.append({
                    "params": _ps,
                    "lr": spec["lr"],
                    "weight_decay": spec["weight_decay"],
                    "is_decay": spec.get("is_decay", spec["weight_decay"] > 0.0),
                })
        optimizer = torch.optim.AdamW(_rebuilt_groups, lr=lr, betas=(adam_beta1, adam_beta2_start))
    else:
        _decay_params = [p for n, p in model.named_parameters() if p.requires_grad and not _is_nodecay_param(n, p)]
        _nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and _is_nodecay_param(n, p)]
        print(f"  Optimizer groups: {len(_decay_params)} decay params, {len(_nodecay_params)} no-decay params")
        optimizer = torch.optim.AdamW(
            [
                {"params": _decay_params, "weight_decay": weight_decay, "is_decay": True},
                {"params": _nodecay_params, "weight_decay": 0.0, "is_decay": False},
            ],
            lr=lr,
            betas=(adam_beta1, adam_beta2_start),
        )

    _use_amp = torch.cuda.is_available()
    scaler = GradScaler(device.type, enabled=_use_amp, init_scale=256)

    def _build_train_state(step):
        id_to_name = {id(p): n for n, p in model.named_parameters()}
        param_group_specs = [
            {
                "names": [id_to_name[id(p)] for p in g["params"] if id(p) in id_to_name],
                "lr": g["lr"],
                "weight_decay": g["weight_decay"],
                "is_decay": g.get("is_decay", g["weight_decay"] > 0.0),
            }
            for g in optimizer.param_groups
        ]
        return {
            "step": step,
            "total_tokens_seen": total_tokens_seen,
            "lr_reductions": lr_reductions,
            "lr_patience": lr_patience,
            "lr_ceiling": lr_ceiling,
            "vel_last_action": vel_last_action,
            "ce_vel_ema": ce_vel_ema,
            "ce_prev": ce_prev,
            "ce_ema": ce_ema,
            "best_ce_ema": best_ce_ema,
            "steps_since_imp": steps_since_imp,
            "grad_norm_ema": grad_norm_ema,
            "grad_clip": grad_clip,
            "best_val_loss": best_val_loss,
            "val_degrade_count": val_degrade_count,
            "vq_beta": vq_beta,
            "meta_vq_beta": meta_vq_beta,
            "category_vq_beta": category_vq_beta,
            "category_active_ema": category_active_ema,
            "gate_gamma": gate_gamma,
            "orth_lambda": orth_lambda,
            "orth_unlocked": orth_unlocked,
            "orth_T0": orth_T0,
            "stream_corr_ema": stream_corr_ema,
            "stream_corr_baseline": stream_corr_baseline,
            "stream_corr_calib_done": stream_corr_calib_done,
            "alpha_reg_lambda": alpha_reg_lambda,
            "pair_sym_lambdas": pair_sym_lambdas,
            "sym_decor_lambda": sym_decor_lambda,
            "meta_pred_lambda": meta_pred_lambda,
            "meta_pred_acc_ema": meta_pred_acc_ema,
            "meta_cb_active_ema": meta_cb_active_ema,
            "meta_cb_grow_count": meta_cb_grow_count,
            "chunk_down_lambda": chunk_down_lambda,
            "chunk_down_loss_ema": chunk_down_loss_ema,
            "current_dropout": current_dropout,
            "weight_decay_val": weight_decay_val,
            "stream_active": list(model.stream_active),
            "stream_names": list(model.sub_encoder.submodality_names),
            "next_candidate_idx": next_candidate_idx,
            "stream_gate_ema": list(stream_gate_ema),
            "stream_dead_counts": list(stream_dead_counts),
            "cb_grow_counts": list(cb_grow_counts),
            "layer_grad_emas": list(layer_grad_emas),
            "growth_warmup_queue": list(growth_warmup_queue),
            "seq_len_stage_idx": seq_len_stage_idx,
            "seq_len_max": seq_len_max,
            "batch_size_tokens": batch_size_tokens,
            "val_every": val_every,
            "num_layers": len(model.decoder.blocks),
            "cb_sizes": [cb.num_symbols for cb in model.codebooks],
            "meta_cb_size": model.meta_codebook.num_symbols,
            "category_cb_size": model.chunk_up.codebook.num_symbols,
            "stream_last_added_step": stream_last_added_step,
            "depth_last_added": depth_last_added,
            "cb_active_ema": list(cb_active_ema),
            "param_group_specs": param_group_specs,
        }

    def _save_full_checkpoint(path, step, tag=""):
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_state": _build_train_state(step),
        }
        torch.save(payload, path)
        if tag:
            print(f"  💾 Checkpoint saved ({tag}): {path}  (step {step})")

    def _restore_train_state(ts):
        nonlocal total_tokens_seen, lr_reductions, lr_patience, lr_ceiling
        nonlocal vel_last_action, ce_vel_ema, ce_prev, ce_ema, best_ce_ema
        nonlocal steps_since_imp, grad_norm_ema, grad_clip, best_val_loss, val_degrade_count
        nonlocal vq_beta, meta_vq_beta, category_vq_beta, category_active_ema
        nonlocal gate_gamma, orth_lambda, orth_unlocked, orth_T0
        nonlocal stream_corr_ema, stream_corr_baseline, stream_corr_calib_done
        nonlocal alpha_reg_lambda, pair_sym_lambdas, sym_decor_lambda
        nonlocal meta_pred_lambda, meta_pred_acc_ema, meta_cb_active_ema, meta_cb_grow_count
        nonlocal chunk_down_lambda, chunk_down_loss_ema, current_dropout, weight_decay_val
        nonlocal stream_last_added_step, depth_last_added, next_candidate_idx
        nonlocal seq_len_stage_idx, seq_len_max, batch_size_tokens, val_every

        total_tokens_seen = ts["total_tokens_seen"]
        lr_reductions = ts["lr_reductions"]
        lr_patience = ts["lr_patience"]
        lr_ceiling = ts["lr_ceiling"]
        vel_last_action = ts["vel_last_action"]
        ce_vel_ema = ts["ce_vel_ema"]
        ce_prev = ts["ce_prev"]
        ce_ema = ts["ce_ema"]
        best_ce_ema = ts["best_ce_ema"]
        steps_since_imp = ts["steps_since_imp"]
        grad_norm_ema = ts["grad_norm_ema"]
        grad_clip = ts["grad_clip"]
        best_val_loss = ts["best_val_loss"]
        val_degrade_count = ts["val_degrade_count"]
        vq_beta = ts["vq_beta"]
        meta_vq_beta = ts["meta_vq_beta"]
        category_vq_beta = ts["category_vq_beta"]
        category_active_ema = ts["category_active_ema"]
        gate_gamma = ts["gate_gamma"]
        orth_lambda = ts["orth_lambda"]
        orth_unlocked = ts["orth_unlocked"]
        orth_T0 = ts["orth_T0"]
        stream_corr_ema = ts["stream_corr_ema"]
        stream_corr_baseline = ts["stream_corr_baseline"]
        stream_corr_calib_done = ts["stream_corr_calib_done"]
        alpha_reg_lambda = ts["alpha_reg_lambda"]
        pair_sym_lambdas = dict(ts["pair_sym_lambdas"])
        sym_decor_lambda = ts["sym_decor_lambda"]
        meta_pred_lambda = ts["meta_pred_lambda"]
        meta_pred_acc_ema = ts["meta_pred_acc_ema"]
        meta_cb_active_ema = ts["meta_cb_active_ema"]
        meta_cb_grow_count = ts["meta_cb_grow_count"]
        chunk_down_lambda = ts["chunk_down_lambda"]
        chunk_down_loss_ema = ts["chunk_down_loss_ema"]
        current_dropout = ts["current_dropout"]
        weight_decay_val = ts.get("weight_decay_val", weight_decay)
        stream_last_added_step = ts.get("stream_last_added_step", 0)
        depth_last_added = ts.get("depth_last_added", 0)
        next_candidate_idx = ts.get("next_candidate_idx", 0)
        seq_len_stage_idx = ts.get("seq_len_stage_idx", 0)
        seq_len_max = ts.get("seq_len_max", seq_len_stages[seq_len_stage_idx])
        batch_size_tokens = ts.get("batch_size_tokens", batch_size_tokens)
        val_every = ts.get("val_every", val_every)

        for dst, key in [
            (cb_active_ema, "cb_active_ema"),
            (stream_gate_ema, "stream_gate_ema"),
            (stream_dead_counts, "stream_dead_counts"),
            (cb_grow_counts, "cb_grow_counts"),
            (layer_grad_emas, "layer_grad_emas"),
            (growth_warmup_queue, "growth_warmup_queue"),
        ]:
            saved = ts.get(key)
            if saved is not None:
                dst[:] = list(saved)

        model.decoder.set_dropout(current_dropout)
        model.cross_stream_attn.drop.p = current_dropout
        model.pre_vq_cross_stream_attn.drop.p = current_dropout
        return ts["step"] + 1

    if _ckpt is not None:
        try:
            optimizer.load_state_dict(_ckpt["optimizer_state"])
        except ValueError as e:
            print(f"  ⚠ Optimizer state mismatch on resume ({e}); continuing with fresh optimizer state.")
        scaler.load_state_dict(_ckpt["scaler_state"])
        _resume_step = _restore_train_state(_ckpt["train_state"])
        print(f"  ✓ Resumed cleanly; continuing at step {_resume_step}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  Device: {device}")

    def _has_working_triton() -> bool:
        try:
            from torch.utils._triton import has_triton
            return bool(has_triton())
        except Exception:
            try:
                import triton  # noqa: F401
                return True
            except ImportError:
                return False

    _cuda_cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
    _torch_compile_available = (
        hasattr(torch, "compile")
        and device.type == "cuda"
        and _cuda_cap[0] >= 7
        and _has_working_triton()
    )

    if _torch_compile_available:
        torch._dynamo.config.capture_scalar_outputs = True
        logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message="Online softmax is disabled")
        print("  Compiling model with torch.compile (dynamic=True)…")
        _compiled_model = torch.compile(model, dynamic=True)
        print("  Compile complete.\n")
    else:
        _compiled_model = model
        if device.type == "cuda" and not _has_working_triton():
            print("  [info] Triton not available — running in eager CUDA mode.\n")
        else:
            print()

    raw_norm = torch.tensor(0.0)
    lr_reduce_threshold = 0.005
    criterion = nn.CrossEntropyLoss()

    def _add_with_growth_warmup(new_params, current_step, force_decay: bool | None = None):
        cur_lr = optimizer.param_groups[0]["lr"]
        reset_lr = max(cur_lr * 2.0, lr * growth_reset_fraction)
        group_idx = len(optimizer.param_groups)
        id_to_name = {id(p): n for n, p in model.named_parameters()}

        if force_decay is True:
            _decay_p = list(new_params)
            _nodecay_p = []
        elif force_decay is False:
            _decay_p = []
            _nodecay_p = list(new_params)
        else:
            _decay_p = [
                p for p in new_params if not _is_nodecay_param(id_to_name.get(id(p), ""), p)
            ]
            _nodecay_p = [
                p for p in new_params if _is_nodecay_param(id_to_name.get(id(p), ""), p)
            ]

        if _decay_p:
            optimizer.add_param_group({
                "params": _decay_p,
                "lr": reset_lr,
                "weight_decay": weight_decay_val,
                "is_decay": True,
            })
        if _nodecay_p:
            optimizer.add_param_group({
                "params": _nodecay_p,
                "lr": reset_lr,
                "weight_decay": 0.0,
                "is_decay": False,
            })

        for idx in range(group_idx, len(optimizer.param_groups)):
            growth_warmup_queue.append({
                "group_idx": idx,
                "start_lr": reset_lr,
                "start_step": current_step,
            })
        print(f"    ↑ Growth LR reset: {reset_lr:.2e} → {cur_lr:.2e} over {growth_warmup_steps} steps")

    model.train()
    print("Starting training…\n")

    _train_start = time.time()
    _step_time = time.time()
    _last_log_step = _resume_step
    _last_heartbeat = time.time()
    last_step = _resume_step

    try:
        for step in range(_resume_step, steps):
            last_step = step

            ema_decay_now = ema_decay_start - (ema_decay_start - ema_decay_end) * (step / max(1, steps - 1))
            for cb in model.codebooks:
                cb.ema_decay = ema_decay_now
            model.chunk_up.codebook.ema_decay = ema_decay_now

            beta2_now = adam_beta2_start + (adam_beta2_end - adam_beta2_start) * (step / max(1, steps - 1))
            for g in optimizer.param_groups:
                g["betas"] = (adam_beta1, beta2_now)

            log_every = min(log_every_max, log_every_base + steps_since_imp // 10)
            seq_len = min(seq_len_max, seq_len_base + int(math.log(step + 1) * 8))
            safe_seq = min(seq_len, len(train_data) - 2)
            batch_size = int(max(batch_size_min, min(batch_size_max, batch_size_tokens // safe_seq)))

            x, y = _random_batch(train_data, batch_size, safe_seq)
            total_tokens_seen += batch_size * safe_seq

            if step < lr_warmup:
                for g in optimizer.param_groups:
                    g["lr"] = lr * (step + 1) / lr_warmup

            h_ramp = min(1.0, step / max(1, h_target_ramp_steps))
            h_frac = h_target_min_frac + (h_target_max_frac - h_target_min_frac) * h_ramp
            h_base = h_frac * math.log(num_symbols_ref)
            batch_scale = (
                1.0 + 0.1 * math.log(batch_size_tokens / h_target_batch_ref)
                if h_target_batch_scale and batch_size_tokens > h_target_batch_ref
                else 1.0
            )
            h_target_now = h_base * min(batch_scale, 1.3)
            for cb in model.codebooks:
                cb.H_target = min(h_target_now, 0.9 * math.log(max(2, cb.num_symbols)))

            _cat_K = model.chunk_up.codebook.num_symbols
            model.chunk_up.codebook.H_target = min(
                h_frac * math.log(max(2, _cat_K)) * min(batch_scale, 1.3),
                0.9 * math.log(max(2, _cat_K)),
            )

            model.sub_encoder.orthogonality_lambda = orth_lambda
            model.set_phase(step)

            # Update concept_bank outside compiled forward to avoid buffer-mutation graph breaks
            with torch.no_grad():
                flat_embeds = model.embedding(x).detach().reshape(-1, model.d_model).float()
                num_samples = min(512, flat_embeds.size(0))
                idx = torch.randperm(flat_embeds.size(0), device=flat_embeds.device)[:num_samples]
                model.concept_bank = torch.cat([model.concept_bank, flat_embeds[idx]], dim=0)[-20000:]

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=_use_amp):
                (
                    logits,
                    _,
                    vq_loss,
                    feat_orth_loss,
                    gate_kl,
                    sym_decor_loss,
                    gate_weights,
                    soft_probs,
                    mean_H_t,
                    per_stream_active_t,
                    pair_js_vec,
                    pair_grad_vec,
                    sub_features,
                    per_stream_topk_H_t,
                    meta_vq_loss,
                    meta_ema_H_t,
                    meta_active_ratio_t,
                    meta_topk_H_t,
                    meta_pred_logits,
                    meta_targets,
                    category_vq_loss,
                    category_ema_H_t,
                    category_active_ratio_t,
                    category_topk_H_t,
                    chunk_down_loss,
                    sparsity_loss,
                ) = _compiled_model(x)

                loss_task = criterion(logits.view(-1, vocab_size), y.view(-1))

            per_stream_active = per_stream_active_t.detach().cpu().tolist()
            per_stream_topk_H = per_stream_topk_H_t.detach().cpu().tolist()
            active_ratio = sum(per_stream_active) / max(1, len(per_stream_active))
            mean_H = float(mean_H_t.item())
            meta_ema_H = float(meta_ema_H_t.item())
            meta_active_ratio = float(meta_active_ratio_t.item())
            meta_topk_H = float(meta_topk_H_t.item())
            category_ema_H = float(category_ema_H_t.item())
            category_active_ratio = float(category_active_ratio_t.item())
            category_topk_H = float(category_topk_H_t.item())

            last_train_ce = loss_task.item()
            vq_val = vq_loss.item()

            if not orth_unlocked:
                with torch.no_grad():
                    sub_f = sub_features.detach().reshape(-1, model.NUM_SUBMODALITIES, model.d_model)
                    sub_f_n = F.normalize(sub_f, dim=2)
                    corr_mat = torch.einsum("bnd,bmd->nm", sub_f_n, sub_f_n) / sub_f_n.shape[0]
                    n_s = model.NUM_SUBMODALITIES
                    idx_i, idx_j = torch.triu_indices(n_s, n_s, offset=1)
                    mean_corr = corr_mat[idx_i, idx_j].mean().item()
                stream_corr_ema = (1 - stream_corr_alpha) * stream_corr_ema + stream_corr_alpha * mean_corr
                if not stream_corr_calib_done and step >= 100:
                    stream_corr_baseline = stream_corr_ema
                    stream_corr_calib_done = True
                    print(f"  ✦ Stream correlation baseline calibrated: {stream_corr_baseline:.4f}")
                if stream_corr_calib_done and stream_corr_ema > stream_corr_baseline + stream_corr_trigger:
                    orth_T0, orth_unlocked = step, True
                    print(
                        f"  ✦ Orth unlocked at step {step}  "
                        f"(corr {stream_corr_ema:.4f} > baseline {stream_corr_baseline:.4f} + {stream_corr_trigger})"
                    )
                elif stream_corr_calib_done and step >= orth_fallback_step:
                    orth_T0, orth_unlocked = step, True
                    print(f"  ✦ Orth unlocked at step {step} (fallback)")
            elif stream_corr_calib_done:
                with torch.no_grad():
                    sub_f = sub_features.detach().reshape(-1, model.NUM_SUBMODALITIES, model.d_model)
                    sub_f_n = F.normalize(sub_f, dim=2)
                    corr_mat = torch.einsum("bnd,bmd->nm", sub_f_n, sub_f_n) / sub_f_n.shape[0]
                    n_s = model.NUM_SUBMODALITIES
                    idx_i, idx_j = torch.triu_indices(n_s, n_s, offset=1)
                    stream_corr_ema = (
                        (1 - stream_corr_alpha) * stream_corr_ema
                        + stream_corr_alpha * corr_mat[idx_i, idx_j].mean().item()
                    )

            orth_weight = 0.8 / (1.0 + math.exp(-(step - orth_T0) / max(1, orth_tau))) if orth_unlocked else 0.0
            orth_raw = float(feat_orth_loss.item())
            corr_factor = (
                1.0 + 5.0 * max(0.0, stream_corr_ema - stream_corr_baseline)
                if stream_corr_calib_done and stream_corr_baseline is not None
                else 1.0
            )
            orth_lambda = max(
                orth_lambda_min,
                min(orth_lambda_max, orth_lambda * (1.0 + orth_lambda_lr * (orth_raw - target_orth) * corr_factor)),
            )
            orth_loss = orth_weight * orth_lambda * feat_orth_loss

            gate_kl_val = gate_kl.item()
            gate_gamma = gate_gamma * (1.0 + gate_gamma_lr * (gate_kl_val - target_gate_kl))
            gate_gamma = max(gate_gamma_min, min(gate_gamma_max, gate_gamma))

            # Global load balance computed over soft_probs so unselected streams still receive gradients
            soft_mean = soft_probs.mean(dim=(0, 1))
            gate_mean = gate_weights.mean(dim=(0, 1))
            late_gate_specialize = step >= gate_specialize_step and active_ratio > gate_specialize_act_threshold
            balance_mult = 0.10 if late_gate_specialize else 0.50
            if all(model.stream_active):
                balance_loss = balance_mult * ((soft_mean - 1.0 / model.NUM_SUBMODALITIES) ** 2).sum()
            else:
                active_mask = model.stream_mask > 0
                n_act = max(1, int(active_mask.sum().item()))
                balance_loss = balance_mult * ((soft_mean[active_mask] - 1.0 / n_act) ** 2).sum()

            gate_token_entropy = -(
                gate_weights.clamp_min(1e-9) * gate_weights.clamp_min(1e-9).log()
            ).sum(dim=-1).mean()
            if late_gate_specialize:
                spec_ramp = min(1.0, (step - gate_specialize_step) / max(1.0, steps * 0.13))
                gate_peak_lambda = gate_peak_lambda_max * spec_ramp
                gate_sparsity_lambda = gate_sparsity_lambda_max * spec_ramp
                gate_peak_loss = gate_peak_lambda * gate_token_entropy
                gate_sparsity_loss = gate_sparsity_lambda * sparsity_loss
            else:
                gate_peak_loss = loss_task.new_zeros(())
                gate_sparsity_loss = loss_task.new_zeros(())

            sym_dec_val = sym_decor_loss.item()
            _sym_decor_active = step >= phase_pair_decor_step
            num_s = model.NUM_SUBMODALITIES
            if _sym_decor_active and pair_js_vec.numel() > 0:
                pair_js_list = pair_js_vec.tolist()
                pair_idx_pairs = [(i, j) for i in range(num_s) for j in range(i + 1, num_s)]
                pair_weights = []
                for (pi, pj), js_val in zip(pair_idx_pairs, pair_js_list):
                    lam = pair_sym_lambdas.get((pi, pj), pair_sym_lambda_init)
                    lam = lam * (1.0 + pair_sym_lambda_lr * (js_val - target_pair_js))
                    lam = max(pair_sym_lambda_min, min(pair_sym_lambda_max, lam))
                    pair_sym_lambdas[(pi, pj)] = lam
                    age_weight = max(stream_age_weight(pi, num_s), stream_age_weight(pj, num_s))
                    pair_weights.append(lam * age_weight)
                pair_w_tensor = torch.tensor(pair_weights, device=device, dtype=pair_grad_vec.dtype)
                pair_sym_loss = (pair_w_tensor * (pair_js_vec + pair_grad_vec)).sum()
                pair_js_vals = dict(zip(pair_idx_pairs, pair_js_list))
            else:
                pair_sym_loss = loss_task.new_zeros(())
                pair_js_vals = {}
            sym_decor_lambda = sum(pair_sym_lambdas.values()) / max(1, len(pair_sym_lambdas))

            vq_beta = vq_beta * (1.0 + vq_beta_lr * (target_active - active_ratio))
            vq_beta = max(vq_beta_min, min(vq_beta_max, vq_beta))

            # Average probabilities (not raw logits) before KL divergence
            _meta_usage_grad = F.softmax(meta_pred_logits, dim=-1).mean(dim=(0, 1))
            _meta_uniform = torch.full_like(_meta_usage_grad, 1.0 / _meta_usage_grad.shape[0])
            _meta_log_usage = torch.log(_meta_usage_grad.clamp_min(1e-10))
            meta_balance_loss = F.kl_div(
                _meta_log_usage, _meta_uniform, reduction="sum", log_target=False
            ).clamp(max=20.0)
            meta_balance_gamma = 0.01

            _meta_collapsed = (step >= phase_meta_loss_step) and (meta_cb_active_ema < 0.15)
            if _meta_collapsed:
                meta_vq_beta = max(meta_vq_beta_min, meta_vq_beta * 0.99)
                meta_commit_cost_next = max(
                    meta_commit_cost_min, model.meta_codebook.commitment_cost * 0.99
                )
            else:
                meta_vq_beta = max(
                    meta_vq_beta_min,
                    min(meta_vq_beta_max, meta_vq_beta * (1.0 + meta_vq_beta_lr * (meta_target_active - meta_active_ratio))),
                )
                meta_commit_cost_next = max(
                    meta_commit_cost_min,
                    min(
                        meta_commit_cost_max,
                        model.meta_codebook.commitment_cost * (1.0 + meta_commit_cost_lr * (meta_target_active - meta_active_ratio)),
                    ),
                )

            meta_K = meta_pred_logits.shape[-1]
            meta_pred_loss = F.cross_entropy(meta_pred_logits.reshape(-1, meta_K), meta_targets.reshape(-1))

            with torch.no_grad():
                pred_cls = meta_pred_logits.reshape(-1, meta_K).argmax(dim=1)
                tgt_flat = meta_targets.reshape(-1)
                acc_now = (pred_cls == tgt_flat).float().mean().item()
                meta_pred_acc_ema = (1 - meta_pred_acc_alpha) * meta_pred_acc_ema + meta_pred_acc_alpha * acc_now

            target_meta_acc = 0.70
            if step >= max(meta_pred_warmup_steps, phase_meta_loss_step):
                meta_pred_lambda = max(
                    meta_pred_lambda_min,
                    min(
                        meta_pred_lambda_max,
                        max(meta_pred_lambda, meta_pred_lambda_min)
                        * (1.0 + meta_pred_lambda_lr * (meta_pred_acc_ema - target_meta_acc)),
                    ),
                )

            mean_alpha = (
                sum(0.5 * torch.sigmoid(cb.residual_alpha_raw).item() for cb in model.codebooks)
                / model.NUM_SUBMODALITIES
            )
            alpha_reg_lambda = max(
                alpha_lambda_min,
                min(alpha_lambda_max, alpha_reg_lambda * (1.0 + alpha_lr * (mean_alpha - target_alpha))),
            )
            alpha_reg = alpha_reg_lambda * sum((0.5 * torch.sigmoid(cb.residual_alpha_raw)) ** 2 for cb in model.codebooks)

            category_active_ema = 0.9 * category_active_ema + 0.1 * category_active_ratio
            category_vq_beta = max(
                category_vq_beta_min,
                min(
                    category_vq_beta_max,
                    category_vq_beta * (1.0 + category_vq_beta_lr * (category_target_active - category_active_ema)),
                ),
            )

            with torch.no_grad():
                chunk_down_loss_ema = 0.95 * chunk_down_loss_ema + 0.05 * chunk_down_loss.item()
            if step >= max(chunk_down_warmup_steps, phase_chunk_loss_step):
                chunk_down_lambda = max(
                    chunk_down_lambda_min,
                    min(
                        chunk_down_lambda_max,
                        max(chunk_down_lambda, chunk_down_lambda_min)
                        * (1.0 + chunk_down_lambda_lr * (chunk_down_loss_ema - chunk_down_target_mse)),
                    ),
                )
                chunk_down_lambda_eff = chunk_down_lambda
            else:
                chunk_down_lambda_eff = 0.0

            meta_vq_beta_eff = meta_vq_beta if (step >= phase_meta_loss_step and active_ratio > 0.45) else 0.0
            meta_pred_lambda_eff = (
                meta_pred_lambda
                if (step >= max(meta_pred_warmup_steps, phase_meta_loss_step) and active_ratio > 0.45)
                else 0.0
            )
            category_vq_beta_eff = category_vq_beta if step >= phase_chunk_loss_step else 0.0
            gate_gamma_eff = gate_gamma if step >= phase_post_vq_attn_step else 0.0
            pair_sym_loss_eff = (pair_sym_loss * 1.4) if step >= phase_pair_decor_step else pair_sym_loss * 0.0

            total_loss = (
                loss_task
                + vq_beta * vq_loss
                + meta_vq_beta_eff * meta_vq_loss
                + meta_pred_lambda_eff * meta_pred_loss
                + (meta_balance_gamma * meta_balance_loss if step >= max(meta_pred_warmup_steps, phase_meta_loss_step) else 0.0)
                + category_vq_beta_eff * category_vq_loss
                + chunk_down_lambda_eff * chunk_down_loss
                + orth_loss
                + gate_gamma_eff * gate_kl
                + (balance_loss if step >= phase_post_vq_attn_step else 0.0)
                + gate_peak_loss
                + gate_sparsity_loss
                + alpha_reg
                + pair_sym_loss_eff
            )

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

            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

            if step >= lr_warmup and steps_since_imp >= lr_patience and step > vel_last_action + vel_cooldown // 2:
                cur_lr = optimizer.param_groups[0]["lr"]
                new_lr = max(lr_min, cur_lr * lr_reduce_factor)
                if new_lr < cur_lr:
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    lr_reductions += 1
                    lr_ceiling = new_lr
                    lr_patience = lr_patience_base + lr_reductions * lr_patience_grow
                    vel_last_action = step
                    batch_size_tokens = min(batch_tokens_max, batch_size_tokens * 2)
                    print(f"  ↓ LR  {cur_lr:.2e} → {new_lr:.2e}  (patience #{lr_reductions}, stalled {steps_since_imp} steps)")
                    print(f"  ↑ batch_tokens → {batch_size_tokens}")
                steps_since_imp = 0

            if growth_warmup_queue:
                cur_lr = optimizer.param_groups[0]["lr"]
                done = []
                for event in growth_warmup_queue:
                    elapsed = step - event["start_step"]
                    if elapsed >= growth_warmup_steps:
                        optimizer.param_groups[event["group_idx"]]["lr"] = cur_lr
                        done.append(event)
                    else:
                        t = elapsed / growth_warmup_steps
                        new_lr = event["start_lr"] * (1.0 - t) + cur_lr * t
                        optimizer.param_groups[event["group_idx"]]["lr"] = new_lr
                for event in done:
                    growth_warmup_queue.remove(event)

            _tau = topk_tau_start + (topk_tau_end - topk_tau_start) * (step / max(1, steps - 1))
            for _cb in model.codebooks:
                _cb.topk_tau = _tau
            _meta_tau = meta_topk_tau_start + (meta_topk_tau_end - meta_topk_tau_start) * (step / max(1, steps - 1))
            model.meta_codebook.topk_tau = _meta_tau

            if step >= lr_warmup:
                current_ce = loss_task.item()
                if ce_prev is not None:
                    raw_vel = current_ce - ce_prev
                    ce_vel_ema = (1 - vel_ema_alpha) * ce_vel_ema + vel_ema_alpha * raw_vel
                ce_prev = current_ce

                if ce_ema is None:
                    ce_ema = current_ce
                ce_ema = (1 - ce_ema_alpha) * ce_ema + ce_ema_alpha * current_ce
                if ce_ema < best_ce_ema * (1 - lr_reduce_threshold):
                    best_ce_ema = ce_ema
                    steps_since_imp = 0
                else:
                    steps_since_imp += 1

                if step >= vel_min_step and step % vel_check_every == 0 and step > vel_last_action + vel_cooldown:
                    cur_lr = optimizer.param_groups[0]["lr"]

                    if ce_vel_ema >= vel_reduce_thresh:
                        new_lr = max(lr_min, cur_lr * lr_reduce_factor)
                        if new_lr < cur_lr:
                            for g in optimizer.param_groups:
                                g["lr"] = new_lr
                            lr_ceiling = new_lr
                            vel_last_action = step
                            lr_reductions += 1
                            print(
                                f"  ↓ LR {cur_lr:.2e} → {new_lr:.2e}  "
                                f"(vel {ce_vel_ema:+.5f} ≥ {vel_reduce_thresh:+.5f}, reduction #{lr_reductions})"
                            )
                            batch_size_tokens = min(batch_tokens_max, batch_size_tokens * 2)
                            print(f"  ↑ batch_tokens → {batch_size_tokens}")
                            val_every = min(val_every_max, val_every * 2)

                    elif ce_vel_ema <= vel_boost_thresh:
                        new_lr = min(lr_ceiling, cur_lr * lr_boost_factor)
                        if new_lr > cur_lr * 1.001:
                            for g in optimizer.param_groups:
                                g["lr"] = new_lr
                            vel_last_action = step
                            print(f"  ↑ LR {cur_lr:.2e} → {new_lr:.2e}  (vel {ce_vel_ema:+.5f} ≤ {vel_boost_thresh:+.5f})")

            if (
                step > 0
                and step % depth_check_every == 0
                and step >= depth_last_added + depth_min_dwell
                and num_layers < num_layers_max
            ):
                with torch.no_grad():
                    block_norms = []
                    for blk in model.decoder.blocks:
                        nrm = sum(p.grad.norm().item() ** 2 for p in blk.parameters() if p.grad is not None) ** 0.5
                        block_norms.append(nrm)
                    while len(layer_grad_emas) < len(block_norms):
                        layer_grad_emas.append(block_norms[-1])
                    for i, nrm in enumerate(block_norms):
                        layer_grad_emas[i] = (1 - layer_grad_alpha) * layer_grad_emas[i] + layer_grad_alpha * nrm
                    mean_norm = sum(layer_grad_emas) / max(1, len(layer_grad_emas))
                    last_norm = layer_grad_emas[-1]
                    if mean_norm > 0 and last_norm / mean_norm > depth_grad_ratio_thresh:
                        new_block = model.decoder.add_layer(device, zero_init_residual=True)
                        _add_with_growth_warmup(list(new_block.parameters()), step)
                        layer_grad_emas.append(last_norm)
                        num_layers += 1
                        depth_last_added = step
                        print(f"  ↑ Layer added (grad-driven, total: {num_layers})  last/mean grad ratio: {last_norm/mean_norm:.2f}")
                        if _torch_compile_available:
                            print("\n[JIT] Recompiling model after layer addition…", flush=True)
                            torch._dynamo.reset()
                            _compiled_model = torch.compile(model, dynamic=True)

            with torch.no_grad():
                for i, cb in enumerate(model.codebooks):
                    stream_ar_i = per_stream_active[i] if i < len(per_stream_active) else active_ratio
                    cb.commitment_cost = max(
                        commit_cost_min,
                        min(commit_cost_max, cb.commitment_cost * (1.0 + commit_cost_lr * (target_active_per - stream_ar_i))),
                    )
                model.meta_codebook.commitment_cost = meta_commit_cost_next

            stream_check_every = min(stream_check_max, 100 + int(step / max(1, steps) * (stream_check_max - 100)))

            if step > 0 and step % stream_check_every == 0:
                N = model.NUM_SUBMODALITIES
                gate_mean_np = gate_mean.detach().cpu().tolist()
                stream_dead_thresh, stream_saturated_thresh = _stream_thresholds(sum(model.stream_active))

                for i in range(N):
                    stream_gate_ema[i] = (1 - stream_gate_alpha) * stream_gate_ema[i] + stream_gate_alpha * gate_mean_np[i]

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
                    step >= stream_add_enable_step
                    and step >= stream_last_added_step + stream_min_dwell
                )

                if (
                    _stream_add_eligible
                    and n_active < max_streams
                    and next_candidate_idx < len(candidate_streams)
                    and n_active > 0
                ):
                    active_emas = [stream_gate_ema[i] for i in range(N) if model.stream_active[i]]
                    n_saturated = sum(1 for e in active_emas if e > stream_saturated_thresh)
                    quorum_met = n_saturated >= math.ceil(stream_trigger_quorum * len(active_emas))
                    if quorum_met:
                        new_name = candidate_streams[next_candidate_idx]
                        next_candidate_idx += 1

                        try:
                            with torch.no_grad():
                                if model.concept_bank.size(0) > 1000:
                                    bank = model.concept_bank.float()
                                    bank = bank - bank.mean(dim=0, keepdim=True)
                                    actual_q = min(bank.size(0), model.d_model)
                                    _, _, V = torch.pca_lowrank(bank, q=actual_q, center=False)
                                    init_weight = V.t()
                                    if init_weight.size(0) < model.d_model:
                                        pad_size = model.d_model - init_weight.size(0)
                                        pad_w = torch.randn(pad_size, model.d_model, device=init_weight.device) * 0.02
                                        init_weight = torch.cat([init_weight, pad_w], dim=0)
                                else:
                                    init_weight = None
                        except Exception as e:
                            print(f"  ⚠ PCA failed: {e}. Falling back to random init.")
                            init_weight = None

                        stream_gate_ema.append(1.0 / (N + 1))
                        stream_dead_counts.append(0)
                        cb_active_ema.append(0.5)
                        cb_grow_counts.append(0)
                        new_idx = model.NUM_SUBMODALITIES
                        for _pi in range(new_idx):
                            pair_sym_lambdas[(_pi, new_idx)] = pair_sym_lambda_init

                        _cur_lr = optimizer.param_groups[0]["lr"]
                        _boost_lr = max(_cur_lr * 2.0, lr * growth_reset_fraction)
                        _pg_before = len(optimizer.param_groups)

                        model.add_submodality(
                            new_name,
                            device,
                            optimizer,
                            lr,
                            weight_decay_val,
                            model.codebooks[0].num_symbols,
                            boost_lr=_boost_lr,
                            init_weight=init_weight,
                        )

                        for _pg_idx in range(_pg_before, len(optimizer.param_groups)):
                            growth_warmup_queue.append({
                                "group_idx": _pg_idx,
                                "start_lr": _boost_lr,
                                "start_step": step,
                            })

                        _global_reset_lr = lr * stream_lr_reset_fraction
                        for g in optimizer.param_groups:
                            if g["lr"] < _global_reset_lr:
                                g["lr"] = _global_reset_lr
                        lr_ceiling = _global_reset_lr
                        ce_vel_ema = vel_boost_thresh * 2.0
                        vel_last_action = step + vel_cooldown * 2
                        ce_prev = None
                        print(f"  ↑ Global LR reset → {_global_reset_lr:.2e}  (stream addition, ceiling reset)")
                        stream_last_added_step = step
                        if _torch_compile_available:
                            print("\n[JIT] Recompiling model after stream addition…", flush=True)
                            torch._dynamo.reset()
                            _compiled_model = torch.compile(model, dynamic=True)

            for i in range(len(cb_active_ema)):
                ar_i = per_stream_active[i] if i < len(per_stream_active) else active_ratio
                cb_active_ema[i] = (1 - cb_active_ema_alpha) * cb_active_ema[i] + cb_active_ema_alpha * ar_i

            meta_cb_active_ema = (1 - cb_active_ema_alpha) * meta_cb_active_ema + cb_active_ema_alpha * meta_active_ratio

            if step >= codebook_growth_enable_step and step > 0 and step % cb_grow_check_every == 0:
                grew_any_cb = False
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
                                old_cb_params = list(cb.codebook.parameters())
                                old_cb_ids = {id(p) for p in old_cb_params}
                                for group in optimizer.param_groups:
                                    group["params"] = [p for p in group["params"] if id(p) not in old_cb_ids]
                                for p in old_cb_params:
                                    optimizer.state.pop(p, None)

                                cb.grow_codebook(new_n, device)
                                _add_with_growth_warmup(list(cb.codebook.parameters()), step, force_decay=False)
                                grew_any_cb = True
                            cb_grow_counts[i] = 0
                    else:
                        cb_grow_counts[i] = 0
                if grew_any_cb and _torch_compile_available:
                    print("\n[JIT] Recompiling model after codebook growth…", flush=True)
                    torch._dynamo.reset()
                    _compiled_model = torch.compile(model, dynamic=True)

            if step >= meta_growth_enable_step and step > 0 and step % cb_grow_check_every == 0:
                meta_cb = model.meta_codebook
                if meta_cb.num_symbols < meta_num_symbols_target:
                    if meta_cb_active_ema > cb_grow_active_thresh:
                        meta_cb_grow_count += cb_grow_check_every
                        if meta_cb_grow_count >= cb_grow_patience:
                            old_n = meta_cb.num_symbols
                            n_new = max(4, old_n // 2)
                            new_n = min(meta_num_symbols_target, old_n + n_new)
                            if new_n > old_n:
                                old_meta_params = list(meta_cb.codebook.parameters())
                                old_meta_ids = {id(p) for p in old_meta_params}
                                for group in optimizer.param_groups:
                                    group["params"] = [p for p in group["params"] if id(p) not in old_meta_ids]
                                for p in old_meta_params:
                                    optimizer.state.pop(p, None)
                                meta_cb.grow_codebook(new_n, device)
                                _add_with_growth_warmup(list(meta_cb.codebook.parameters()), step, force_decay=False)
                                model.grow_meta_pred_head(new_n, device, optimizer)
                                _add_with_growth_warmup(list(model.meta_pred_head.parameters()), step, force_decay=True)
                                print(f"  ↑ Meta-codebook grown: {old_n} → {new_n} symbols")
                                if _torch_compile_available:
                                    torch._dynamo.reset()
                                    _compiled_model = torch.compile(model, dynamic=True)
                            meta_cb_grow_count = 0
                    else:
                        meta_cb_grow_count = 0

            if step >= category_growth_enable_step and step > 0 and step % cb_grow_check_every == 0:
                _cat_target = max(4, model.codebooks[0].num_symbols // 4)
                if model.chunk_up.codebook.num_symbols < _cat_target:
                    _old_cat = model.chunk_up.codebook.num_symbols
                    _old_cat_params = list(model.chunk_up.codebook.codebook.parameters())
                    _old_cat_ids = {id(p) for p in _old_cat_params}
                    for group in optimizer.param_groups:
                        group["params"] = [p for p in group["params"] if id(p) not in _old_cat_ids]
                    for p in _old_cat_params:
                        optimizer.state.pop(p, None)
                    model.chunk_up.codebook.grow_codebook(_cat_target, device)
                    _add_with_growth_warmup(
                        list(model.chunk_up.codebook.codebook.parameters()), step, force_decay=False
                    )
                    print(f"  ↑ Category codebook grown: {_old_cat} → {_cat_target} symbols (follows level-1 at 4:1 ratio)")
                    if _torch_compile_available:
                        print("\n[JIT] Recompiling model after category codebook growth…", flush=True)
                        torch._dynamo.reset()
                        _compiled_model = torch.compile(model, dynamic=True)

            if step >= revival_enable_step and step > 0 and step % revival_check_every == 0:
                total_revived = 0
                for i, _cb in enumerate(model.codebooks):
                    n = _cb.revive_dead_entries(device, revival_noise_scale)
                    if n > 0:
                        total_revived += n
                        cb_active_ema[i] = 0.5
                        cb_grow_counts[i] = 0
                meta_revived = model.meta_codebook.revive_dead_entries(device, revival_noise_scale)
                if meta_revived > 0:
                    total_revived += meta_revived
                    meta_cb_active_ema = 0.5
                    meta_cb_grow_count = 0
                if total_revived > 0:
                    print(f"  ♻ Revived {total_revived} dead codebook entries across {model.NUM_SUBMODALITIES} streams + meta")

            # Overlap metric js_val = log(2) - JS(p, q): HIGHER = worse overlap, LOWER = more disjoint
            if pair_js_vals and step > 0 and step % 100 == 0:
                worst_pair = max(pair_js_vals, key=pair_js_vals.get)
                worst_js = pair_js_vals[worst_pair]
                pi, pj = worst_pair
                if worst_js > kickstart_js_thresh:
                    if worst_pair not in kickstart_bad_since:
                        kickstart_bad_since[worst_pair] = step
                    elif (
                        step - kickstart_bad_since[worst_pair] >= kickstart_patience
                        and step - kickstart_last.get(worst_pair, 0) >= kickstart_cooldown
                    ):
                        model.kickstart_worst_pair(pi, pj, device)
                        kickstart_last[worst_pair] = step
                        kickstart_bad_since.pop(worst_pair, None)
                else:
                    kickstart_bad_since.pop(worst_pair, None)

            _now = time.time()
            if _now - _last_heartbeat >= 30.0 and step % log_every != 0:
                _elapsed_hb = _now - _train_start
                print(
                    f"  ⏳ step {step}/{steps}[{time.strftime('%H:%M:%S', time.gmtime(_elapsed_hb))} elapsed]  still running…",
                    flush=True,
                )
                _last_heartbeat = _now

            if step % log_every == 0 or step == steps - 1:
                ppl = torch.exp(loss_task).item()
                n_active = sum(model.stream_active)
                avg_cb = sum(cb.num_symbols for cb in model.codebooks) / model.NUM_SUBMODALITIES
                now = time.time()
                _last_heartbeat = now
                elapsed = now - _train_start
                steps_since_last = max(1, step - _last_log_step)
                step_secs = (now - _step_time) / steps_since_last
                _step_time = now
                _last_log_step = step
                eta_secs = step_secs * (steps - step)
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))
                vel_str = f"{ce_vel_ema:+.4f}" if ce_prev is not None else "n/a"
                corr_str = f"{stream_corr_ema:.3f}" if stream_corr_calib_done else "cal"

                names_abbrev = [n[:3] for n in model.sub_encoder.submodality_names]
                gw = gate_mean.detach()
                n_gw = gw.shape[0]
                gate_str = " ".join(
                    f"{names_abbrev[i]}={gw[i].item():.2f}" for i in range(min(len(names_abbrev), n_gw))
                )

                act_min = min(per_stream_active) * 100
                act_max = max(per_stream_active) * 100
                act_str = f"{active_ratio*100:.1f}%[{act_min:.0f}–{act_max:.0f}]"

                if pair_js_vals:
                    worst_pair = max(pair_js_vals, key=pair_js_vals.get)
                    worst_js = pair_js_vals[worst_pair]
                    wpi, wpj = worst_pair
                    wn_i = model.sub_encoder.submodality_names[wpi][:3]
                    wn_j = model.sub_encoder.submodality_names[wpj][:3]
                    worst_str = f"{worst_js:.3f}({wn_i},{wn_j})"
                else:
                    worst_str = "n/a"

                mean_topk_H = sum(per_stream_topk_H) / max(1, len(per_stream_topk_H))
                topk_max = math.log(max(2, model.codebooks[0].topk)) if model.codebooks else 1.0
                cur_tau = model.codebooks[0].topk_tau if model.codebooks else 0.0
                topk_str = f"{mean_topk_H:.3f}/{topk_max:.2f}(τ{cur_tau:.2f})"

                if growth_warmup_queue:
                    warmup_str = (
                        f"W{len(growth_warmup_queue)}@"
                        f"{min(growth_warmup_steps - (step - e['start_step']) for e in growth_warmup_queue)}"
                    )
                else:
                    warmup_str = "-"

                gnorm_str = f"{raw_norm.item():.2f}→{grad_clip:.2f}"
                vram_str = f"{torch.cuda.memory_reserved() / 1024**3:.1f}G" if torch.cuda.is_available() else "CPU"
                tok_per_sec = (batch_size * safe_seq) / max(step_secs, 1e-6)
                thr_str = f"{tok_per_sec / 1000:.1f}k"

                seen_m = total_tokens_seen / 1_000_000
                epochs = total_tokens_seen / max(1, n_train)
                seen_str = f"{seen_m:.1f}M({epochs:.1f}x)"

                alphas = [0.5 * torch.sigmoid(cb.residual_alpha_raw).item() for cb in model.codebooks]
                a_min, a_max = min(alphas), max(alphas)
                alpha_str = f"μ{mean_alpha:.3f}[{a_min:.2f}–{a_max:.2f}]"

                print(
                    f"Step {step:5d}/{steps} | "
                    f"CE {loss_task.item():.4f} | "
                    f"PPL {ppl:.2f} | "
                    f"VQ {vq_val:.4f}(β{vq_beta:.2f},cc{model.codebooks[0].commitment_cost:.3f}) | "
                    f"MetaCB Act{meta_active_ratio:.2f} H{meta_ema_H:.2f}→{model.meta_codebook.H_target:.2f} β{meta_vq_beta:.2f} cc{model.meta_codebook.commitment_cost:.3f} TopKH{meta_topk_H:.2f} | "
                    f"MetaPred CE{meta_pred_loss.item():.3f} Acc{meta_pred_acc_ema:.2f} λ{meta_pred_lambda:.3f} Bal{meta_balance_loss.item():.3f} | "
                    f"ChunkUp Act{category_active_ratio:.2f} H{category_ema_H:.2f}→{model.chunk_up.codebook.H_target:.2f} β{category_vq_beta:.2f} | "
                    f"ChunkDn MSE{chunk_down_loss.item():.3f} λ{chunk_down_lambda:.3f} | "
                    f"Orth {orth_loss.item():.2e}(λ{orth_lambda:.2e},ρ{corr_str}) | "
                    f"GKL {gate_kl_val:.3f}(γ{gate_gamma:.4f}) | "
                    f"SymD μ{sym_dec_val:.3f} ↑{worst_str} | "
                    f"Act {act_str} | "
                    f"TopK-H {topk_str} | "
                    f"α {alpha_str}(λ{alpha_reg_lambda:.4f}) | "
                    f"H {mean_H:.2f}→{h_target_now:.2f} | "
                    f"BS {batch_size}x{safe_seq}(t{batch_size_tokens}) | "
                    f"Seen {seen_str} | "
                    f"∇ {gnorm_str} | "
                    f"vel {vel_str} | "
                    f"do {current_dropout:.2f} | "
                    f"L {num_layers} | "
                    f"S {n_active}/{model.NUM_SUBMODALITIES} | "
                    f"CB {avg_cb:.0f} | "
                    f"Wmup {warmup_str} | "
                    f"VRAM {vram_str} | "
                    f"Thr {thr_str} t/s | "
                    f"β2 {beta2_now:.4f} | "
                    f"LR {optimizer.param_groups[0]['lr']:.2e} | "
                    f"⏱ {elapsed_str} (+{step_secs:.2f}s/step, ETA {eta_str})",
                    flush=True,
                )
                if step % (log_every * 5) == 0 or step == steps - 1:
                    print(f"  Gates: {gate_str}")
                    if per_stream_topk_H:
                        topkh_detail = " ".join(
                            f"{names_abbrev[i]}={per_stream_topk_H[i]:.2f}" for i in range(len(per_stream_topk_H))
                        )
                        print(f"  TopK-H/stream: {topkh_detail}")

            if val_data is not None and step > 0 and step % val_every == 0:
                model.eval()
                _val_batches = 8
                _val_losses = []
                with torch.no_grad():
                    for _ in range(_val_batches):
                        xv, yv = _random_batch(val_data, batch_size, safe_seq)
                        lv, *_ = model(xv)
                        _val_losses.append(criterion(lv.view(-1, vocab_size), yv.view(-1)).item())
                val_ce = sum(_val_losses) / len(_val_losses)
                marker = ""
                if val_ce < best_val_loss:
                    best_val_loss = val_ce
                    _save_full_checkpoint(best_checkpoint, step, tag="best val")
                    marker = "  ✓ best"
                    val_degrade_count = 0
                else:
                    val_degrade_count += 1
                    if val_degrade_count >= val_degrade_limit:
                        print(
                            f"  ⏹ Early stop: val CE degraded {val_degrade_count}× consecutively "
                            f"(best {best_val_loss:.4f} → current {val_ce:.4f})"
                        )
                        break

                if step > lr_warmup and step > vel_min_step:
                    if val_ce > best_val_loss * 1.02:
                        for g in optimizer.param_groups:
                            g["lr"] = max(lr_min, g["lr"] * 0.97)
                        lr_ceiling = optimizer.param_groups[0]["lr"]
                        vel_last_action = step
                    elif val_ce > best_val_loss:
                        for g in optimizer.param_groups:
                            g["lr"] = max(lr_min, g["lr"] * 0.99)
                        lr_ceiling = optimizer.param_groups[0]["lr"]
                        vel_last_action = step

                if last_train_ce is not None:
                    ratio = last_train_ce / max(val_ce, 1e-8)
                    if ratio > overfit_threshold:
                        current_dropout = current_dropout * (1.0 + dropout_lr * (ratio - overfit_threshold))
                    else:
                        current_dropout = current_dropout * (1.0 - dropout_lr * 0.5)
                    current_dropout = max(dropout_min, min(dropout_max, current_dropout))
                    model.decoder.set_dropout(current_dropout)
                    model.cross_stream_attn.drop.p = current_dropout
                    model.pre_vq_cross_stream_attn.drop.p = current_dropout

                    if ratio > overfit_threshold:
                        weight_decay_val = weight_decay_val * (1.0 + weight_decay_lr * (ratio - overfit_threshold))
                    else:
                        weight_decay_val = weight_decay_val * (1.0 - weight_decay_lr * 0.5)
                    weight_decay_val = max(weight_decay_min, min(weight_decay_max, weight_decay_val))
                    # Only adjust weight_decay on decay groups (never overwrite no-decay codebooks/norms/biases)
                    for g in optimizer.param_groups:
                        if g.get("is_decay", True):
                            g["weight_decay"] = weight_decay_val

                    if seq_len_stage_idx < len(seq_len_stages) - 1:
                        if len(seq_len_val_ce_history) >= 10:
                            seq_len_stage_idx += 1
                            seq_len_max = seq_len_stages[seq_len_stage_idx]
                            seq_len_val_ce_history = []
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
                                seq_len_val_ce_history = []
                                seq_len_improvement_ema = 0.0
                                print(
                                    f"  ↑ seq_len_max unlocked → {seq_len_max}  "
                                    f"(val CE improvement plateaued: {_imp_snapshot:.4f} < {seq_len_plateau_thresh})"
                                )
                        seq_len_val_ce_history.append(val_ce)

                print(
                    f"  ↳ Val CE {val_ce:.4f}  "
                    f"PPL {math.exp(min(val_ce, 20)):.2f}  "
                    f"do→{current_dropout:.3f}  "
                    f"wd→{weight_decay_val:.4f}  "
                    f"val_every={val_every}  "
                    f"vel={ce_vel_ema:+.5f}{marker}"
                )
                model.train()

    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted! Saving emergency checkpoint…")
        _save_full_checkpoint(interrupt_checkpoint, last_step, tag="interrupt")
        raise

    _save_full_checkpoint(checkpoint, last_step, tag="final")
    _total = time.time() - _train_start
    print(f"\nFinal checkpoint  → {checkpoint}")
    print(f"Best checkpoint   → {best_checkpoint}  (val CE {best_val_loss:.4f})")
    print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(_total))}  ({_total/3600:.2f} hours)")

    return model, tokenizer, seq_len_max


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
    print("\n" + "=" * 60)
    print("  NEURO-SYMBOLIC EXPLAINABILITY & GENERATION DEMO")
    print("=" * 60)

    encoded_seed = tokenizer.encode(seed_text)
    context = torch.tensor([encoded_seed], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        _, symbolic_indices, _, _, _, _, _, _, mean_H_t, _ps_active_t, *_ = model(context)
    mean_H = float(mean_H_t.item())
    active_ratio = float(_ps_active_t.mean().item())

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
    generated_ids = model.generate(context, max_new_tokens, seq_len, temperature, top_k)
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"GENERATED TEXT:\n{'-'*50}")
    print(generated_text)
    print("-" * 50)


# ==========================================================
# XAI: "MIND READER" VISUALIZATION (HEATMAP)
# ==========================================================


def visualize_submodality_mind_reader(
    model: SubmodalitySymbolicLanguageModel,
    tokenizer,
    dev: torch.device,
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import seaborn as sns
    except ImportError:
        print("ERROR: Please install seaborn: pip install seaborn")
        return

    print("\n" + "=" * 60)
    print("  RUNNING XAI 'MIND READER' VISUALIZATION...")
    print("=" * 60)

    model.eval()

    word_categories = {
        "Emotion / Affect": ["love", "hate", "rage", "grief", "joy", "fear"],
        "Spatial / Distance": ["far", "near", "above", "behind", "within", "beyond"],
        "Visual / Light": ["bright", "dark", "shadow", "glow", "shining", "dim"],
        "Abstract / Logic": ["therefore", "because", "reason", "logic", "however", "truth"],
        "Size / Magnitude": ["vast", "tiny", "enormous", "massive", "little", "great"],
        "Position / Order": ["first", "last", "before", "after", "middle", "finally"],
    }

    all_words: list[str] = []
    all_weights: list[np.ndarray] = []
    category_boundaries: list[int] = []

    with torch.no_grad():
        for _, words in word_categories.items():
            category_boundaries.append(len(all_words))
            for word in words:
                token_ids = tokenizer.encode(" " + word)
                if not token_ids:
                    continue
                token_tensor = torch.tensor([token_ids], dtype=torch.long).to(dev)

                embed = model.embedding(token_tensor)
                gates, _, _ = model.compute_gate_weights(embed, apply_stream_dropout=False)
                mean_gate = gates.mean(dim=1).squeeze(0).cpu().numpy()

                all_words.append(word)
                all_weights.append(mean_gate)

    if not all_weights:
        print("  ⚠ No tokens available for Mind Reader visualization.")
        return

    weight_matrix = np.array(all_weights)
    submodality_names = model.sub_encoder.submodality_names

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        weight_matrix,
        annot=True,
        fmt=".2f",
        cmap="magma",
        xticklabels=submodality_names,
        yticklabels=all_words,
        cbar_kws={"label": "Submodality Gating Weight", "pad": 0.18},
        ax=ax,
    )

    for row in category_boundaries[1:]:
        ax.axhline(row, color="white", lw=2)

    cat_names = list(word_categories.keys())
    for k, (start, name) in enumerate(zip(category_boundaries, cat_names)):
        end = category_boundaries[k + 1] if k + 1 < len(category_boundaries) else len(all_words)
        mid = (start + end) / 2
        ax.text(
            1.02,
            mid,
            name,
            va="center",
            ha="left",
            fontsize=9,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

    ax.set_title(
        "Neuro-Symbolic XAI: How the Model 'Sees' Different Words",
        fontsize=14,
        pad=20,
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
    model: SubmodalitySymbolicLanguageModel,
    tokenizer,
    dev: torch.device,
    filepath: str = "data.txt",
    top_k: int = 20,
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
    all_tokens = random.sample(unique_tokens, min(3000, len(unique_tokens)))

    bucket_activations = {i: [] for i in range(model.NUM_SUBMODALITIES)}

    model.eval()
    with torch.no_grad():
        for tok in all_tokens:
            word = tokenizer.decode([tok]).strip()

            if len(word) < 4 or not word.isalpha():
                continue

            tok_tensor = torch.tensor([[tok]], dtype=torch.long).to(dev)
            embed = model.embedding(tok_tensor)
            gates, _, _ = model.compute_gate_weights(embed, apply_stream_dropout=False)
            gate_weights = gates.squeeze().cpu().numpy()

            for i in range(model.NUM_SUBMODALITIES):
                w = float(gate_weights[i])
                if w > 0.0:
                    bucket_activations[i].append((word, w))

    submodality_names = model.sub_encoder.submodality_names

    for i in range(model.NUM_SUBMODALITIES):
        sorted_words = sorted(bucket_activations[i], key=lambda x: x[1], reverse=True)
        seen = set()
        top_words = []
        for word, _ in sorted_words:
            word_lower = word.lower()
            if word_lower not in seen:
                seen.add(word_lower)
                top_words.append(word)
            if len(top_words) >= top_k:
                break

        print(f"\n► BUCKET {i}  (Named: '{submodality_names[i]}')")
        print("  Top Words: " + (", ".join(top_words) if top_words else "(no routed words)"))

    print("\n" + "=" * 60)


# ==========================================================
# SYMBOL MUTUAL INFORMATION DIAGNOSTIC
# ==========================================================


def symbol_mutual_information(
    model: SubmodalitySymbolicLanguageModel,
    tokenizer,
    dev: torch.device,
    n_samples: int = 8000,
    seq_len: int = 64,
    filepath: str = "data.txt",
) -> None:
    print("\n" + "=" * 60)
    print("  SYMBOL MUTUAL INFORMATION DIAGNOSTIC")
    print("=" * 60)
    print("  Estimating MI(symbol ; token) per submodality stream…")
    print(f"  Sampling {n_samples} tokens from {filepath}.\n")

    if not os.path.exists(filepath):
        print(f"  ⚠ Cannot find {filepath} for MI estimate — skipping.")
        return

    model.eval()
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()
    all_ids = list(tokenizer.encode(raw_text))

    if len(all_ids) < seq_len + 1:
        print("  ⚠ Corpus too small for MI estimate — skipping.")
        return

    rng = torch.Generator()
    rng.manual_seed(42)
    starts = torch.randint(
        0,
        len(all_ids) - seq_len,
        (n_samples // seq_len + 1,),
        generator=rng,
    ).tolist()

    sample_ids = []
    for s in starts:
        sample_ids.extend(all_ids[s : s + seq_len])
        if len(sample_ids) >= n_samples:
            break
    sample_ids = sample_ids[:n_samples]

    N = model.NUM_SUBMODALITIES
    joint = [collections.Counter() for _ in range(N)]
    tok_ct = collections.Counter()

    batch = 32
    with torch.no_grad():
        for b in range(0, len(sample_ids) - seq_len + 1, batch * seq_len):
            chunk = sample_ids[b : b + batch * seq_len]
            n_rows = len(chunk) // seq_len
            if n_rows == 0:
                break
            chunk = chunk[: n_rows * seq_len]

            x = torch.tensor(chunk, dtype=torch.long, device=dev).view(n_rows, seq_len)

            embed = model.embedding(x)
            sub_features = model.sub_encoder(embed)
            if getattr(model, "enable_pre_vq_attn", False):
                B_f, T_f, N_f, D_f = sub_features.shape
                sub_features = model.pre_vq_cross_stream_attn(
                    sub_features.reshape(B_f * T_f, N_f, D_f)
                ).reshape(B_f, T_f, N_f, D_f)

            toks_flat = x.reshape(-1).cpu().tolist()
            tok_ct.update(toks_flat)

            for i in range(min(N, len(model.codebooks))):
                normed = model.pre_vq_norms[i](sub_features[:, :, i, :])
                flat = normed.reshape(-1, model.codebooks[i].embedding_dim).float()
                flat_s = F.normalize(flat, dim=1)
                cb_s = F.normalize(model.codebooks[i].codebook.weight.float(), dim=1)
                dists = torch.cdist(flat_s, cb_s, p=2)
                syms = dists.argmin(dim=1).cpu().tolist()
                for tok, sym in zip(toks_flat, syms):
                    joint[i][(tok, sym)] += 1

    total = sum(tok_ct.values())
    if total == 0:
        print("  ⚠ No tokens sampled for MI estimate.")
        model.train()
        return
    p_tok = {t: c / total for t, c in tok_ct.items()}

    results = []
    for i in range(N):
        stream_total = sum(joint[i].values())
        if stream_total == 0:
            results.append((model.sub_encoder.submodality_names[i], 0.0))
            continue

        sym_ct = collections.Counter()
        for (_, sym), cnt in joint[i].items():
            sym_ct[sym] += cnt
        p_sym = {s: c / stream_total for s, c in sym_ct.items()}

        mi = 0.0
        for (tok, sym), cnt in joint[i].items():
            p_ts = cnt / stream_total
            p_t = p_tok.get(tok, 1e-12)
            p_s = p_sym.get(sym, 1e-12)
            mi += p_ts * math.log(p_ts / (p_t * p_s) + 1e-12)

        results.append((model.sub_encoder.submodality_names[i], max(0.0, mi)))

    mean_mi = sum(mi for _, mi in results) / max(1, len(results))

    print(f"  {'Stream':<14} {'MI(sym;tok)':>12}   Quality")
    print("  " + "-" * 42)
    for name, mi in results:
        if mi >= 0.7:
            quality = "✓ high     — symbols encode meaningful structure"
        elif mi >= 0.4:
            quality = "⚠ moderate — some structure; codebook still forming"
        else:
            quality = "✗ low      — symbols may be noise"
        print(f"  {name:<14} {mi:>12.3f}   {quality}")

    print(f"\n  Mean MI across streams: {mean_mi:.3f}")
    threshold = 0.60
    if mean_mi >= threshold:
        verdict = "GO  ✓  — level-1 symbols are semantically meaningful."
        advice = "       Meta-codebook experiment is well-motivated."
    else:
        verdict = "HOLD ✗  — level-1 symbols need more training first."
        advice = "       Run more steps or investigate low-MI streams before adding meta-VQ."

    print(f"\n  Meta-codebook decision: {verdict}")
    print(f"  {advice}")
    print("=" * 60 + "\n")

    model.train()


if __name__ == "__main__":
    FILEPATH = "data.txt"
    TOKENIZER_MODE = "bpe"
    BPE_VOCAB_SIZE = None

    try:
        trained_model, tokenizer, s_len = train_model(
            filepath=FILEPATH,
            tokenizer_mode=TOKENIZER_MODE,
            bpe_vocab_size=BPE_VOCAB_SIZE,
        )
    except KeyboardInterrupt:
        print("💾 Emergency checkpoint saved. Exiting cleanly.")
        sys.exit(0)

    evaluate_and_generate(trained_model, tokenizer, s_len)
    visualize_submodality_mind_reader(trained_model, tokenizer, device)
    auto_label_buckets(trained_model, tokenizer, device, filepath=FILEPATH)
    symbol_mutual_information(trained_model, tokenizer, device, filepath=FILEPATH)

    input("\nScript complete. Press Enter to exit...")