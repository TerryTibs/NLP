"""
Patched v18 (Self-Tuning, Self-Optimising & GPU-Accelerated):
- Fixes all v16/v17 integrity, routing, compile, schedule, and GPU controller logic
- Auto-sizes BPE vocab cap (50k), steps from corpus size, and VRAM batch limits
- Auto-selects bfloat16 (Ampere+) / float16 AMP + fused AdamW + TF32 + Triton fusion
- Adds usage-based dead-symbol revival (splits busiest clusters), readiness-gated phases,
  closed-loop gate_temperature / stream_dropout / topk_tau / balance_mult / orth_lambda,
  proactive VRAM batch scaling, OOM auto-backoff, fixed val/train ratio, rolling 3-check
  seq_len curriculum, bottleneck stream growth, overfit-guarded layer addition,
  and automatic per-run output folder (.log, .csv, checkpoints, phase snapshots, text, heatmap)
"""

import os

os.environ["TRITON_DEFAULT_FP_FUSION"] = "1"  # "1" = FP fusion enabled, "0" = disabled

import collections
import csv
import importlib.util
import logging
import math
import random
import re as _re
import shutil
import sys
import time
import warnings
from datetime import datetime

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

# ==========================================================
# AUTOMATIC PER-RUN OUTPUT FOLDER & LOGGING
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_OUTPUT_DIR = os.path.join(SCRIPT_DIR, SCRIPT_NAME, f"{SCRIPT_NAME}_{RUN_TIMESTAMP}")
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)


def out_path(relative_path: str) -> str:
    full = os.path.join(RUN_OUTPUT_DIR, relative_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    return full


class _TeeStream:
    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file

    def write(self, data):
        self.primary.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self):
        self.primary.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.primary, "encoding", "utf-8")


# ==========================================================
# DEVICE & REPRODUCIBILITY
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


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
        p = round(float(max(0.0, min(0.5, p))) * 20.0) / 20.0
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
        self.register_buffer(
            "ema_usage",
            torch.full((num_symbols,), 1.0 / max(1, num_symbols), dtype=torch.float32),
        )
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
            new_u = torch.full((n_new,), 1.0 / max(1, new_num_symbols), device=dev, dtype=self.ema_usage.dtype)
            self.ema_usage = torch.cat([self.ema_usage, new_u], dim=0)
            self.ema_usage = self.ema_usage / self.ema_usage.sum().clamp_min(1e-8)
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
            usage_thresh = 0.05 / max(1, self.num_symbols)
            dead_mask = (norms_sq < (med_norm_sq * 0.1)) | (self.ema_usage < usage_thresh)
            n_dead = int(dead_mask.sum().item())
            if n_dead == 0:
                return 0
            live_idx = (~dead_mask).nonzero(as_tuple=True)[0]
            if len(live_idx) == 0:
                return 0
            live_usage = self.ema_usage[live_idx].clamp_min(1e-8)
            busiest = torch.multinomial(live_usage, num_samples=n_dead, replacement=True)
            src_sel = live_idx[busiest]
            dead_idx = dead_mask.nonzero(as_tuple=True)[0]
            live_std = W[live_idx].std(dim=0).mean().clamp(min=0.05) if len(live_idx) > 1 else W.new_tensor(1.0)
            W[dead_idx] = W[src_sel] + torch.randn_like(W[src_sel]) * (noise_scale * live_std)
            self.ema_usage[dead_idx] = 1.0 / max(1, self.num_symbols)
            self.ema_usage[src_sel] *= 0.5
            return n_dead

    def forward(self, features: torch.Tensor):
        flat = features.reshape(-1, self.embedding_dim).float()
        flat_sphere = F.normalize(flat, dim=1)
        cb_sphere = F.normalize(self.codebook.weight.float(), dim=1)

        # Spherical L2 distance via cuBLAS GEMM (uses Tensor Cores / TF32 on CUDA)
        dot = torch.mm(flat_sphere, cb_sphere.t())
        distances = (2.0 - 2.0 * dot).clamp(min=1e-12).sqrt()

        tau = self.topk_tau_buf.clone().clamp(min=1e-4)
        h_target = self.H_target_buf.clone().clamp(min=0.01)
        commit_cost = self.commitment_cost_buf.clone()

        # Differentiable Usage Loss (one-sided floor: never penalize high diversity)
        soft_assign = F.softmax(-distances / tau, dim=-1)
        p_bar = soft_assign.mean(dim=0)
        diff_entropy = -(p_bar * torch.log(p_bar + 1e-10)).sum()
        usage_loss = F.relu(1.0 - (diff_entropy / h_target)) ** 2

        k_eff = min(self.topk, self.num_symbols)
        topk_dists, topk_idx = distances.topk(k_eff, dim=1, largest=False)
        topk_weights = F.softmax(-topk_dists / tau, dim=1)

        topk_vecs = self.codebook(topk_idx).float()
        quantized_flat = (topk_weights.unsqueeze(-1) * topk_vecs).sum(dim=1)
        quantized = quantized_flat.view_as(features).to(features.dtype)

        e_loss = (topk_weights.detach() * topk_dists).sum(dim=1).mean()
        q_loss = F.mse_loss(quantized_flat, flat.detach())

        if self.num_symbols <= 64:
            n_sample = self.num_symbols
            W_sample = cb_sphere
        else:
            n_sample = 64
            sample_idx = torch.randint(0, self.num_symbols, (n_sample,), device=cb_sphere.device)
            W_sample = cb_sphere[sample_idx]
        cos_sim = torch.mm(W_sample, W_sample.t())
        mask = ~torch.eye(n_sample, dtype=torch.bool, device=cos_sim.device)
        diversity_loss = cos_sim[mask].mean().clamp(min=0.0) if n_sample > 1 else cos_sim.new_zeros(())

        vq_loss = q_loss + commit_cost * e_loss + 0.01 * diversity_loss + 0.5 * usage_loss

        quantized_st = features + (quantized - features).detach()
        alpha = 0.5 * torch.sigmoid(self.residual_alpha_raw)
        quantized_out = quantized_st + alpha * features

        with torch.no_grad():
            flat_idx = topk_idx.flatten()
            counts = torch.bincount(flat_idx, minlength=self.num_symbols).float()
            probs = counts / max(1, flat_idx.numel())
            nz = probs > 0
            batch_H = -torch.where(nz, probs * torch.log(probs.clamp_min(1e-10)), probs.new_zeros(())).sum()
            batch_H = batch_H.to(self.ema_entropy.dtype)
            ema_d = self.ema_decay_buf.to(self.ema_entropy.dtype)
            self.ema_entropy.mul_(1.0 - ema_d).add_(batch_H * ema_d)
            self.ema_usage.mul_(1.0 - ema_d).add_(probs.to(self.ema_usage.dtype) * ema_d)
            active_ratio = nz.float().mean()
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
        self.d_model = d_model
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

        self._stream_modules = [getattr(self, f"extract_{n.lower()}") for n in self.submodality_names]

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
        """Vectorized cross-stream Gram penalty computed in 2 GPU GEMMs (no Python loops)."""
        B, T, N_s, D = sub_features.shape
        N = max(1, B * T)
        f = sub_features.reshape(B * T, N_s, D).float()
        f = f - f.mean(dim=0, keepdim=True)
        f = F.normalize(f, dim=2)

        f_flat = f.reshape(B * T, N_s * D)
        C = (f_flat.t() @ f_flat) / N
        f_perm = f.permute(1, 0, 2)  # (N_s, B*T, D)
        diag_blocks = torch.bmm(f_perm.transpose(1, 2), f_perm) / N
        loss = 0.5 * (C.pow(2).sum() - diag_blocks.pow(2).sum()).clamp(min=0.0)

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
        self._stream_modules.append(new_lin)
        return new_lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single fused GPU GEMM across all streams while preserving per-stream weights."""
        W = torch.cat([lin.weight for lin in self._stream_modules], dim=0)
        b = torch.cat([lin.bias for lin in self._stream_modules], dim=0)
        B, T, D = x.shape
        return F.linear(x, W, b).view(B, T, len(self._stream_modules), D)


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

        self.register_buffer("_stream_dropout_buf", torch.tensor(0.15, dtype=torch.float32))
        self.register_buffer("_gate_temp_buf", torch.tensor(0.50, dtype=torch.float32))
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

        # Fixed-capacity GPU ring buffer (avoids per-step torch.cat allocations)
        self.register_buffer(
            "concept_bank_buf",
            torch.zeros((20000, d_model), dtype=torch.float32),
            persistent=False,
        )
        self._concept_bank_len: int = 0
        self._concept_bank_ptr: int = 0

        self.stream_active: list[bool] = [True] * self.NUM_SUBMODALITIES
        self._n_active_streams: int = self.NUM_SUBMODALITIES
        self._update_stream_mask()

        self.current_step = 0
        self.phase_hard_gate_step = 600
        self.phase_post_vq_attn_step = 750
        self.phase_pair_decor_step = 900
        self.phase_meta_step = 1500
        self.phase_meta_loss_step = 1650
        self.phase_chunk_step = 2160
        self.phase_chunk_loss_step = 2280

        self._phase_hard_gate = True
        self._phase_post_vq_attn = False
        self._phase_pair_decor = False
        self._phase_meta = False
        self._phase_meta_loss = False
        self._phase_chunk = False
        self._phase_chunk_loss = False

    @property
    def stream_dropout_p(self) -> float:
        return float(self._stream_dropout_buf.item())

    @stream_dropout_p.setter
    def stream_dropout_p(self, val: float) -> None:
        self._stream_dropout_buf.fill_(float(val))

    @property
    def gate_temperature(self) -> float:
        return float(self._gate_temp_buf.item())

    @gate_temperature.setter
    def gate_temperature(self, val: float) -> None:
        self._gate_temp_buf.fill_(float(val))

    def set_phase(self, step: int) -> None:
        self.current_step = int(step)
        self._phase_hard_gate = step >= self.phase_hard_gate_step
        self._phase_post_vq_attn = step >= self.phase_post_vq_attn_step
        self._phase_pair_decor = step >= self.phase_pair_decor_step
        self._phase_meta = step >= self.phase_meta_step
        self._phase_meta_loss = step >= self.phase_meta_loss_step
        self._phase_chunk = step >= self.phase_chunk_step
        self._phase_chunk_loss = step >= self.phase_chunk_loss_step

    def update_phases(
        self,
        step: int,
        hard_gate: bool,
        post_vq_attn: bool,
        pair_decor: bool,
        meta: bool,
        meta_loss: bool,
        chunk: bool,
        chunk_loss: bool,
    ) -> None:
        self.current_step = int(step)
        self._phase_hard_gate = bool(hard_gate)
        self._phase_post_vq_attn = bool(post_vq_attn)
        self._phase_pair_decor = bool(pair_decor)
        self._phase_meta = bool(meta)
        self._phase_meta_loss = bool(meta_loss)
        self._phase_chunk = bool(chunk)
        self._phase_chunk_loss = bool(chunk_loss)

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
        gate_logits = self.submodality_gate(embed)
        if getattr(self, "gate_bias_override", None) is not None:
            gate_logits = gate_logits + self.gate_bias_override

        B_sz, _, N_s = gate_logits.shape
        k_eff = min(self.top_k_streams, max(1, self._n_active_streams))
        active_mask = (self.stream_mask > 0).view(1, 1, N_s)

        if apply_stream_dropout and self.training and self._n_active_streams > k_eff:
            keep_prob = (1.0 - self._stream_dropout_buf.clone()).clamp(0.50, 1.0)
            keep = torch.bernoulli(
                torch.ones((B_sz, 1, N_s), device=gate_logits.device, dtype=torch.float32) * keep_prob
            ) * active_mask.float()
            tie_break = torch.rand_like(keep) * active_mask.float()
            score = keep * 2.0 + tie_break
            _, rescue_idx = torch.topk(score, k=k_eff, dim=-1)
            keep = keep.scatter(-1, rescue_idx, 1.0)
            keep_bool = (keep > 0.5) & active_mask
        else:
            keep_bool = active_mask.expand(B_sz, 1, N_s)

        valid_logits = gate_logits.masked_fill(~keep_bool, float("-inf"))
        tau = self._gate_temp_buf.clone().clamp(min=0.05)

        soft_probs = F.softmax(valid_logits / tau, dim=-1)

        if self._phase_hard_gate:
            _, topk_idx = torch.topk(valid_logits, k=k_eff, dim=-1)
            topk_mask = torch.zeros_like(valid_logits, dtype=torch.bool)
            topk_mask.scatter_(-1, topk_idx, True)
            masked_logits = valid_logits.masked_fill(~topk_mask, float("-inf"))
            gate_weights = F.softmax(masked_logits / tau, dim=-1)
        else:
            gate_weights = soft_probs

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
            Wj[top_idx] += torch.randn_like(Wj[top_idx]) * noise_scale
            name_i = self.sub_encoder.submodality_names[i]
            name_j = self.sub_encoder.submodality_names[j]
            print(f"  ⚡ Kickstart ({name_i},{name_j}): perturbed {k_perturb} of {self.codebooks[j].num_symbols} centroids")
            return k_perturb

    def forward(self, x: torch.Tensor):
        embed = self.embedding(x)
        sub_features = self.sub_encoder(embed)

        if self.enable_pre_vq_attn:
            B_f, T_f, N_f, D_f = sub_features.shape
            sub_features = self.pre_vq_cross_stream_attn(
                sub_features.reshape(B_f * T_f, N_f, D_f)
            ).reshape(B_f, T_f, N_f, D_f)

        normed_streams = []
        symbolic_vectors = []
        symbolic_indices = []
        total_vq_loss = embed.new_zeros((), dtype=torch.float32)
        total_ema_H = embed.new_zeros((), dtype=torch.float32)
        per_stream_active_list = []
        per_stream_topk_H_list = []

        for i in range(self.NUM_SUBMODALITIES):
            normed_stream = self.pre_vq_norms[i](sub_features[:, :, i, :])
            normed_streams.append(normed_stream)
            quantized_out, _, indices, vq_loss, ema_H, active_ratio, topk_H = self.codebooks[i](normed_stream)
            symbolic_vectors.append(quantized_out)
            symbolic_indices.append(indices)
            total_vq_loss = total_vq_loss + vq_loss / self.NUM_SUBMODALITIES
            total_ema_H = total_ema_H + ema_H
            per_stream_active_list.append(active_ratio)
            per_stream_topk_H_list.append(topk_H)

        per_stream_active = torch.stack(per_stream_active_list)
        per_stream_topk_H = torch.stack(per_stream_topk_H_list)
        symbolic_indices_t = torch.stack(symbolic_indices, dim=2)

        gate_weights, soft_probs, keep_bool = self.compute_gate_weights(
            embed, apply_stream_dropout=self.training
        )

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
                max_K = max(cb.num_symbols for cb in self.codebooks)

                with torch.no_grad():
                    idx_mat = symbolic_indices_t.permute(2, 0, 1, 3).reshape(N_s, -1)
                    hists = torch.zeros(N_s, max_K, device=x.device, dtype=torch.float32)
                    hists.scatter_add_(1, idx_mat, torch.ones_like(idx_mat, dtype=torch.float32))
                    padded = hists / max(1, idx_mat.shape[1]) + 1e-10
                    padded = padded / padded.sum(dim=1, keepdim=True)

                    iu, ju = torch.triu_indices(N_s, N_s, offset=1, device=x.device)
                    p_i = padded[iu]
                    p_j = padded[ju]
                    m = 0.5 * (p_i + p_j)
                    js = 0.5 * (
                        (p_i * torch.log(p_i / m)).sum(dim=-1)
                        + (p_j * torch.log(p_j / m)).sum(dim=-1)
                    )
                    pair_js_vec = math.log(2) - js

                f_stack = F.normalize(
                    torch.stack(normed_streams, dim=0).reshape(N_s, -1, self.d_model).float(),
                    dim=2,
                )
                cb_padded = []
                cb_valid_mask = []
                for cb in self.codebooks:
                    w_n = F.normalize(cb.codebook.weight.float(), dim=1)
                    k_i = cb.num_symbols
                    if k_i < max_K:
                        w_n = F.pad(w_n, (0, 0, 0, max_K - k_i))
                    cb_padded.append(w_n)
                    cb_valid_mask.append(torch.arange(max_K, device=x.device) < k_i)
                cb_flat = torch.stack(cb_padded, dim=0).reshape(N_s * max_K, self.d_model)
                cb_mask = torch.stack(cb_valid_mask, dim=0).unsqueeze(0)

                cross_rows = []
                for i in range(N_s):
                    sim_i = (f_stack[i] @ cb_flat.t()).view(-1, N_s, max_K)
                    sim_i = sim_i.masked_fill(~cb_mask, -2.0)
                    max_cos_i = sim_i.max(dim=-1).values
                    min_dist_i = (2.0 - 2.0 * max_cos_i).clamp(min=1e-8).sqrt()
                    cross_rows.append(min_dist_i.mean(dim=0))
                cross_mean = torch.stack(cross_rows, dim=0)
                d_owns = cross_mean.diag()

                margin = 0.25
                pair_grad_vec = 0.5 * (
                    F.relu(d_owns[iu] - cross_mean[iu, ju] + margin)
                    + F.relu(d_owns[ju] - cross_mean[ju, iu] + margin)
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
            symbolic_indices_t,
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
            bpe_vocab_size = int(min(50_000, max(1000, unique_words * 1.5)))
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
    steps: int | None = None,
    resume_from: str | None = None,
):
    lr = 3e-4
    lr_warmup_base = 100
    weight_decay = 0.01
    checkpoint = out_path("neuro_symbolic_model.pt")
    best_checkpoint = out_path("neuro_symbolic_model_best.pt")
    interrupt_checkpoint = out_path("neuro_symbolic_model_interrupt.pt")
    metrics_csv_path = out_path("neuro_symbolic_training_metrics.csv")
    phase_dir = os.path.join(RUN_OUTPUT_DIR, "phase_checkpoints")
    os.makedirs(phase_dir, exist_ok=True)

    batch_size_tokens = 1024
    batch_size_min = 4
    batch_size = 32

    grad_clip = 1.0
    grad_norm_alpha = 0.05
    grad_clip_factor = 1.5
    grad_clip_min = 0.3
    grad_clip_max = 5.0

    gate_gamma = 0.05
    target_gate_kl = 0.15
    gate_gamma_lr = 0.005
    gate_gamma_min = 0.005
    gate_gamma_max = 0.5
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
    h_target_min_frac = 0.65
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

    orth_lambda = 1e-3
    target_corr = 0.20
    orth_lambda_lr = 0.01
    orth_lambda_min = 1e-5
    orth_lambda_max = 0.50

    val_every = 200
    val_every_max = 400

    candidate_streams = [
        "Texture", "Motion", "Valence", "Agency",
        "Causality", "Certainty", "Temporal",
        "Intent", "Ownership", "Risk", "Hierarchy",
        "Conflict", "Probability", "Obligation",
    ]
    next_candidate_idx = 0
    max_streams = 12
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
    vel_reduce_thresh = 0.0005
    vel_boost_thresh = -0.005
    lr_reduce_factor = 0.85
    lr_boost_factor = 1.05
    vel_cooldown = 300
    lr_reductions = 0

    stream_lr_reset_fraction = 0.35

    topk_tau_start = 0.20
    meta_topk_tau_start = 0.15
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

    if device.type == "cuda":
        _vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if _vram_gb >= 15.0:
            batch_tokens_max, batch_size_max = 16384, 128
        elif _vram_gb >= 7.5:
            batch_tokens_max, batch_size_max = 8192, 64
        else:
            batch_tokens_max, batch_size_max = 4096, 64
    else:
        _vram_gb = 0.0
        batch_tokens_max, batch_size_max = 2048, 32

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
    meta_pred_acc_alpha = 0.05

    category_vq_beta = 0.10
    category_vq_beta_min = 0.15
    category_vq_beta_max = 0.60
    category_vq_beta_lr = 0.005
    category_target_active = 0.70

    chunk_down_lambda = 0.05
    chunk_down_lambda_min = 0.01
    chunk_down_lambda_max = 0.20
    chunk_down_lambda_lr = 0.005
    chunk_down_target_mse = 0.005

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
    stream_corr_alpha = 0.05
    stream_corr_trigger = 0.05
    stream_corr_calib_done = False

    seq_len_val_ce_history = []
    seq_len_improvement_ema = 0.0
    seq_len_improvement_alpha = 0.3

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

    if steps is None:
        _target_epochs = max(1.5, min(12.0, 10.0 * (1_000_000 / max(n_train, 100_000)) ** 0.5))
        _est_tok_step = 3500.0 if device.type == "cuda" else 1024.0
        _max_auto_steps = 40000 if _vram_gb >= 15.0 else 30000
        steps = int(max(3000, min(_max_auto_steps, round((n_train * _target_epochs) / _est_tok_step / 500) * 500)))
        print(f"  Auto-tuned steps: {steps:,} (~{_target_epochs:.1f} epochs, VRAM {_vram_gb:.1f}G, max_bt {batch_tokens_max})")

    orth_tau = max(40, int(steps * 0.033))
    orth_T0 = steps
    orth_fallback_step = min(1200, int(steps * 0.40))
    h_target_ramp_steps = min(2000, max(200, int(steps * 0.25)))
    vel_min_step = min(1500, max(300, int(steps * 0.20)))
    gate_specialize_step = min(8000, int(steps * 0.60))
    phase_hard_gate_step = min(1000, int(steps * 0.15))
    phase_post_vq_attn_step = min(1500, int(steps * 0.20))
    phase_pair_decor_step = min(2000, int(steps * 0.25))
    stream_add_enable_step = min(2500, int(steps * 0.30))
    codebook_growth_enable_step = min(2500, int(steps * 0.30))
    revival_enable_step = min(2800, int(steps * 0.35))
    phase_meta_step = min(5000, int(steps * 0.45))
    phase_meta_loss_step = min(5500, int(steps * 0.50))
    meta_growth_enable_step = min(6500, int(steps * 0.55))
    phase_chunk_step = min(9000, int(steps * 0.68))
    phase_chunk_loss_step = min(10000, int(steps * 0.72))
    category_growth_enable_step = min(10500, int(steps * 0.75))
    phase_snapshots_saved = 0

    unlocked_phases = {
        "hard_gate": False,
        "post_vq_attn": False,
        "pair_decor": False,
        "meta": False,
        "meta_loss": False,
        "chunk": False,
        "chunk_loss": False,
    }
    phase_unlock_steps = {}
    is_overfitting = False
    last_val_ce = None

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

    total_tokens_seen = 0
    growth_warmup_queue = []

    best_val_loss = float("inf")
    val_degrade_count = 0
    val_degrade_limit = 15

    vel_last_action = 0
    lr_min = lr / 50
    lr_ceiling = lr

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

        _saved_layers = _ts.get("num_layers", num_layers)
        for _ in range(len(model.decoder.blocks), _saved_layers):
            model.decoder.add_layer(device)
        num_layers = len(model.decoder.blocks)

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

    model.phase_hard_gate_step = phase_hard_gate_step
    model.phase_post_vq_attn_step = phase_post_vq_attn_step
    model.phase_pair_decor_step = phase_pair_decor_step
    model.phase_meta_step = phase_meta_step
    model.phase_meta_loss_step = phase_meta_loss_step
    model.phase_chunk_step = phase_chunk_step
    model.phase_chunk_loss_step = phase_chunk_loss_step

    N = model.NUM_SUBMODALITIES
    stream_gate_ema = [1.0 / N] * N
    stream_dead_counts = [0] * N
    cb_grow_counts = [0] * N
    pair_sym_lambdas = {(i, j): pair_sym_lambda_init for i in range(N) for j in range(i + 1, N)}
    layer_grad_emas = [1.0] * num_layers

    # GPU-resident controller tensors (zero per-step CPU syncs)
    vq_beta_t = torch.tensor(vq_beta, device=device, dtype=torch.float32)
    gate_gamma_t = torch.tensor(gate_gamma, device=device, dtype=torch.float32)
    orth_lambda_t = torch.tensor(orth_lambda, device=device, dtype=torch.float32)
    alpha_reg_lambda_t = torch.tensor(alpha_reg_lambda, device=device, dtype=torch.float32)
    meta_vq_beta_t = torch.tensor(meta_vq_beta, device=device, dtype=torch.float32)
    meta_commit_cost_t = torch.tensor(model.meta_codebook.commitment_cost, device=device, dtype=torch.float32)
    meta_pred_lambda_t = torch.tensor(meta_pred_lambda, device=device, dtype=torch.float32)
    meta_pred_acc_ema_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    category_vq_beta_t = torch.tensor(category_vq_beta, device=device, dtype=torch.float32)
    category_active_ema_t = torch.tensor(0.5, device=device, dtype=torch.float32)
    chunk_down_lambda_t = torch.tensor(chunk_down_lambda, device=device, dtype=torch.float32)
    chunk_down_loss_ema_t = torch.tensor(1.0, device=device, dtype=torch.float32)
    stream_corr_ema_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    grad_norm_ema_t = torch.tensor(grad_clip, device=device, dtype=torch.float32)
    cb_active_ema_t = torch.full((N,), 0.5, device=device, dtype=torch.float32)
    cb_commit_costs_t = torch.tensor(
        [cb.commitment_cost for cb in model.codebooks], device=device, dtype=torch.float32
    )
    meta_cb_active_ema_t = torch.tensor(0.5, device=device, dtype=torch.float32)
    ce_vel_ema_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    ce_ema_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    ce_prev_t = torch.tensor(0.0, device=device, dtype=torch.float32)
    best_ce_ema_t = torch.tensor(float("inf"), device=device, dtype=torch.float32)
    balance_mult_t = torch.tensor(0.50, device=device, dtype=torch.float32)
    topk_tau_t = torch.tensor(topk_tau_start, device=device, dtype=torch.float32)
    meta_topk_tau_t = torch.tensor(meta_topk_tau_start, device=device, dtype=torch.float32)
    gate_temperature_t = torch.tensor(0.50, device=device, dtype=torch.float32)
    stream_dropout_t = torch.tensor(0.15, device=device, dtype=torch.float32)
    for _cb in model.codebooks:
        _cb.topk_tau = topk_tau_start
    model.meta_codebook.topk_tau = meta_topk_tau_start
    model._gate_temp_buf.copy_(gate_temperature_t)
    model._stream_dropout_buf.copy_(stream_dropout_t)
    ce_ema_alpha = 0.05
    steps_since_imp_t = torch.tensor(0, device=device, dtype=torch.long)
    ce_ema_initialized = False
    ce_prev_initialized = False

    def _rebuild_pair_tensors(n_streams: int, pair_dict: dict):
        pairs = [(i, j) for i in range(n_streams) for j in range(i + 1, n_streams)]
        lams = [pair_dict.get(p, pair_sym_lambda_init) for p in pairs]
        ages = [max(stream_age_weight(pi, n_streams), stream_age_weight(pj, n_streams)) for pi, pj in pairs]
        return (
            pairs,
            torch.tensor(lams, device=device, dtype=torch.float32),
            torch.tensor(ages, device=device, dtype=torch.float32),
        )

    pair_idx_pairs, pair_lam_t, pair_age_weights_t = _rebuild_pair_tensors(N, pair_sym_lambdas)

    def _stream_thresholds(n):
        return 0.5 / n, 0.95 / n

    lr_warmup = max(100, min(500, int(lr_warmup_base * (num_layers / 4) * (d_model / 256))))

    _no_decay_keys = (
        "bias", "norm", "codebook", "residual_alpha_raw",
        "ema_entropy", "ema_usage", "meta_residual_raw", "category_residual_raw",
    )

    def _is_nodecay_param(name: str, p: torch.Tensor) -> bool:
        return p.ndim <= 1 or any(k in name for k in _no_decay_keys)

    def _make_adamw(param_groups):
        if device.type == "cuda":
            try:
                return torch.optim.AdamW(param_groups, lr=lr, betas=(adam_beta1, adam_beta2_start), fused=True)
            except (TypeError, RuntimeError):
                pass
        return torch.optim.AdamW(param_groups, lr=lr, betas=(adam_beta1, adam_beta2_start))

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
        optimizer = _make_adamw(_rebuilt_groups)
    else:
        _decay_params = [p for n, p in model.named_parameters() if p.requires_grad and not _is_nodecay_param(n, p)]
        _nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and _is_nodecay_param(n, p)]
        print(f"  Optimizer groups: {len(_decay_params)} decay params, {len(_nodecay_params)} no-decay params")
        optimizer = _make_adamw([
            {"params": _decay_params, "weight_decay": weight_decay, "is_decay": True},
            {"params": _nodecay_params, "weight_decay": 0.0, "is_decay": False},
        ])

    _cuda_cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
    _use_amp = device.type == "cuda"
    _amp_dtype = (
        torch.bfloat16
        if (_use_amp and _cuda_cap[0] >= 8 and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    _use_scaler = _use_amp and (_amp_dtype == torch.float16)
    scaler = GradScaler(device.type, enabled=_use_scaler, init_scale=256)

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
        synced_pair_dict = dict(zip(pair_idx_pairs, pair_lam_t.detach().cpu().tolist()))
        return {
            "step": step,
            "total_tokens_seen": total_tokens_seen,
            "lr_reductions": lr_reductions,
            "lr_patience": lr_patience,
            "lr_ceiling": lr_ceiling,
            "vel_last_action": vel_last_action,
            "ce_vel_ema": float(ce_vel_ema_t.item()),
            "ce_prev": float(ce_prev_t.item()) if ce_prev_initialized else None,
            "ce_ema": float(ce_ema_t.item()) if ce_ema_initialized else None,
            "best_ce_ema": float(best_ce_ema_t.item()),
            "steps_since_imp": int(steps_since_imp_t.item()),
            "grad_norm_ema": float(grad_norm_ema_t.item()),
            "grad_clip": grad_clip,
            "best_val_loss": best_val_loss,
            "val_degrade_count": val_degrade_count,
            "vq_beta": float(vq_beta_t.item()),
            "meta_vq_beta": float(meta_vq_beta_t.item()),
            "category_vq_beta": float(category_vq_beta_t.item()),
            "category_active_ema": float(category_active_ema_t.item()),
            "gate_gamma": float(gate_gamma_t.item()),
            "orth_lambda": float(orth_lambda_t.item()),
            "orth_unlocked": orth_unlocked,
            "orth_T0": orth_T0,
            "stream_corr_ema": float(stream_corr_ema_t.item()),
            "stream_corr_baseline": stream_corr_baseline,
            "stream_corr_calib_done": stream_corr_calib_done,
            "alpha_reg_lambda": float(alpha_reg_lambda_t.item()),
            "pair_sym_lambdas": synced_pair_dict,
            "sym_decor_lambda": float(pair_lam_t.mean().item()) if pair_lam_t.numel() > 0 else sym_decor_lambda,
            "meta_pred_lambda": float(meta_pred_lambda_t.item()),
            "meta_pred_acc_ema": float(meta_pred_acc_ema_t.item()),
            "meta_cb_active_ema": float(meta_cb_active_ema_t.item()),
            "meta_cb_grow_count": meta_cb_grow_count,
            "chunk_down_lambda": float(chunk_down_lambda_t.item()),
            "chunk_down_loss_ema": float(chunk_down_loss_ema_t.item()),
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
            "batch_tokens_max": batch_tokens_max,
            "balance_mult": float(balance_mult_t.item()),
            "topk_tau": float(topk_tau_t.item()),
            "meta_topk_tau": float(meta_topk_tau_t.item()),
            "gate_temperature": float(gate_temperature_t.item()),
            "stream_dropout_p": float(stream_dropout_t.item()),
            "unlocked_phases": dict(unlocked_phases),
            "phase_unlock_steps": dict(phase_unlock_steps),
            "is_overfitting": is_overfitting,
            "last_val_ce": last_val_ce,
            "val_every": val_every,
            "num_layers": len(model.decoder.blocks),
            "cb_sizes": [cb.num_symbols for cb in model.codebooks],
            "meta_cb_size": model.meta_codebook.num_symbols,
            "category_cb_size": model.chunk_up.codebook.num_symbols,
            "stream_last_added_step": stream_last_added_step,
            "depth_last_added": depth_last_added,
            "cb_active_ema": cb_active_ema_t.detach().cpu().tolist(),
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
        nonlocal vel_last_action, steps_since_imp, grad_clip, best_val_loss, val_degrade_count
        nonlocal orth_unlocked, orth_T0, stream_corr_baseline, stream_corr_calib_done
        nonlocal sym_decor_lambda, meta_cb_grow_count, current_dropout, weight_decay_val
        nonlocal stream_last_added_step, depth_last_added, next_candidate_idx
        nonlocal seq_len_stage_idx, seq_len_max, batch_size_tokens, batch_tokens_max, val_every
        nonlocal pair_sym_lambdas, pair_idx_pairs, pair_lam_t, pair_age_weights_t
        nonlocal ce_ema_initialized, ce_prev_initialized, cb_active_ema_t, cb_commit_costs_t
        nonlocal is_overfitting, last_val_ce

        total_tokens_seen = ts["total_tokens_seen"]
        lr_reductions = ts["lr_reductions"]
        lr_patience = ts["lr_patience"]
        lr_ceiling = ts["lr_ceiling"]
        vel_last_action = ts["vel_last_action"]
        steps_since_imp = ts["steps_since_imp"]
        grad_clip = ts["grad_clip"]
        best_val_loss = ts["best_val_loss"]
        val_degrade_count = ts["val_degrade_count"]
        orth_unlocked = ts["orth_unlocked"]
        orth_T0 = ts["orth_T0"]
        stream_corr_baseline = ts["stream_corr_baseline"]
        stream_corr_calib_done = ts["stream_corr_calib_done"]
        pair_sym_lambdas = dict(ts["pair_sym_lambdas"])
        sym_decor_lambda = ts["sym_decor_lambda"]
        meta_cb_grow_count = ts["meta_cb_grow_count"]
        current_dropout = ts["current_dropout"]
        weight_decay_val = ts.get("weight_decay_val", weight_decay)
        stream_last_added_step = ts.get("stream_last_added_step", 0)
        depth_last_added = ts.get("depth_last_added", 0)
        next_candidate_idx = ts.get("next_candidate_idx", 0)
        seq_len_stage_idx = ts.get("seq_len_stage_idx", 0)
        seq_len_max = ts.get("seq_len_max", seq_len_stages[seq_len_stage_idx])
        batch_size_tokens = ts.get("batch_size_tokens", batch_size_tokens)
        if "batch_tokens_max" in ts:
            batch_tokens_max = min(batch_tokens_max, ts["batch_tokens_max"])
        if "balance_mult" in ts:
            balance_mult_t.fill_(float(ts["balance_mult"]))
        if "topk_tau" in ts:
            topk_tau_t.fill_(float(ts["topk_tau"]))
            for _cb in model.codebooks:
                _cb.topk_tau_buf.copy_(topk_tau_t)
        if "meta_topk_tau" in ts:
            meta_topk_tau_t.fill_(float(ts["meta_topk_tau"]))
            model.meta_codebook.topk_tau_buf.copy_(meta_topk_tau_t)
        if "gate_temperature" in ts:
            gate_temperature_t.fill_(float(ts["gate_temperature"]))
            model._gate_temp_buf.copy_(gate_temperature_t)
        if "stream_dropout_p" in ts:
            stream_dropout_t.fill_(float(ts["stream_dropout_p"]))
            model._stream_dropout_buf.copy_(stream_dropout_t)
        if "unlocked_phases" in ts:
            unlocked_phases.update(ts["unlocked_phases"])
        if "phase_unlock_steps" in ts:
            phase_unlock_steps.update(ts["phase_unlock_steps"])
        is_overfitting = ts.get("is_overfitting", False)
        last_val_ce = ts.get("last_val_ce", None)
        val_every = ts.get("val_every", val_every)

        ce_vel_ema_t.fill_(float(ts["ce_vel_ema"]))
        if ts["ce_prev"] is not None:
            ce_prev_t.fill_(float(ts["ce_prev"]))
            ce_prev_initialized = True
        if ts["ce_ema"] is not None:
            ce_ema_t.fill_(float(ts["ce_ema"]))
            ce_ema_initialized = True
        best_ce_ema_t.fill_(float(ts["best_ce_ema"]))
        steps_since_imp_t.fill_(int(ts["steps_since_imp"]))
        grad_norm_ema_t.fill_(float(ts["grad_norm_ema"]))
        vq_beta_t.fill_(float(ts["vq_beta"]))
        meta_vq_beta_t.fill_(float(ts["meta_vq_beta"]))
        meta_commit_cost_t.fill_(float(model.meta_codebook.commitment_cost))
        category_vq_beta_t.fill_(float(ts["category_vq_beta"]))
        category_active_ema_t.fill_(float(ts["category_active_ema"]))
        gate_gamma_t.fill_(float(ts["gate_gamma"]))
        orth_lambda_t.fill_(float(ts["orth_lambda"]))
        stream_corr_ema_t.fill_(float(ts["stream_corr_ema"]))
        alpha_reg_lambda_t.fill_(float(ts["alpha_reg_lambda"]))
        meta_pred_lambda_t.fill_(float(ts["meta_pred_lambda"]))
        meta_pred_acc_ema_t.fill_(float(ts["meta_pred_acc_ema"]))
        meta_cb_active_ema_t.fill_(float(ts["meta_cb_active_ema"]))
        chunk_down_lambda_t.fill_(float(ts["chunk_down_lambda"]))
        chunk_down_loss_ema_t.fill_(float(ts["chunk_down_loss_ema"]))

        if ts.get("cb_active_ema"):
            cb_active_ema_t = torch.tensor(ts["cb_active_ema"], device=device, dtype=torch.float32)
        cb_commit_costs_t = torch.tensor(
            [cb.commitment_cost for cb in model.codebooks], device=device, dtype=torch.float32
        )
        pair_idx_pairs, pair_lam_t, pair_age_weights_t = _rebuild_pair_tensors(
            model.NUM_SUBMODALITIES, pair_sym_lambdas
        )

        for dst, key in [
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
        model.cross_stream_attn.drop.p = round(current_dropout * 20.0) / 20.0
        model.pre_vq_cross_stream_attn.drop.p = round(current_dropout * 20.0) / 20.0
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

    def _can_torch_compile() -> bool:
        if os.name == "nt" and shutil.which("cl") is None:
            return False
        if steps < 1500:
            return False
        try:
            from torch.utils._triton import has_triton
            return bool(has_triton())
        except Exception:
            return importlib.util.find_spec("triton") is not None

    _torch_compile_available = (
        hasattr(torch, "compile")
        and device.type == "cuda"
        and _cuda_cap[0] >= 7
        and _can_torch_compile()
    )
    _amp_label = str(_amp_dtype).replace("torch.", "") if _use_amp else "fp32"

    if _torch_compile_available:
        torch._dynamo.config.capture_scalar_outputs = True
        logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message="Online softmax is disabled")
        print(f"  AMP: {_amp_label} | Compiling model with torch.compile (dynamic=True)…")
        _compiled_model = torch.compile(model, dynamic=True)
        print("  Compile wrapper ready.\n")
    else:
        _compiled_model = model
        print(f"  AMP: {_amp_label} | Running in eager mode.\n")

    raw_norm = torch.tensor(0.0, device=device)
    lr_reduce_threshold = 0.005
    criterion = nn.CrossEntropyLoss()
    meta_cb_grow_count = 0

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
            _decay_p = [p for p in new_params if not _is_nodecay_param(id_to_name.get(id(p), ""), p)]
            _nodecay_p = [p for p in new_params if _is_nodecay_param(id_to_name.get(id(p), ""), p)]

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

    _metrics_file = open(metrics_csv_path, "w", newline="", encoding="utf-8")
    _metrics_writer = csv.writer(_metrics_file)
    _metrics_writer.writerow([
        "step", "event", "train_ce", "train_ppl", "vq_loss", "vq_beta", "cc",
        "meta_act", "meta_H", "meta_pred_ce", "meta_pred_acc",
        "chunk_up_act", "chunk_dn_mse", "orth_loss", "orth_lambda", "stream_corr",
        "gate_kl", "gate_gamma", "sym_decor", "active_ratio", "topk_H",
        "mean_alpha", "mean_H", "h_target", "batch_size", "seq_len", "tokens_seen",
        "grad_norm", "grad_clip", "ce_vel", "dropout", "num_layers",
        "active_streams", "avg_cb_size", "lr", "val_ce", "val_ppl",
        "best_val_loss", "elapsed_sec", "tok_per_sec",
    ])
    _metrics_file.flush()
    print(f"  Metrics CSV → {metrics_csv_path}")

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

            if step == _resume_step or step % 10 == 0:
                ema_decay_now = ema_decay_start - (ema_decay_start - ema_decay_end) * (step / max(1, steps - 1))
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
                    cb.ema_decay = ema_decay_now
                    cb.H_target = min(h_target_now, 0.9 * math.log(max(2, cb.num_symbols)))
                model.chunk_up.codebook.ema_decay = ema_decay_now
                _cat_K = model.chunk_up.codebook.num_symbols
                model.chunk_up.codebook.H_target = min(
                    h_frac * math.log(max(2, _cat_K)) * min(batch_scale, 1.3),
                    0.9 * math.log(max(2, _cat_K)),
                )

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

            # Readiness-gated phase transitions (checked every 25 steps to avoid per-step sync)
            if step == _resume_step or step % 25 == 0:
                _cur_act = float(cb_active_ema_t.mean().item())
                _cur_meta_act = float(meta_cb_active_ema_t.item())
                _new_unlocks = []
                if not unlocked_phases["hard_gate"] and (
                    (step >= 250 and _cur_act >= 0.65) or step >= phase_hard_gate_step
                ):
                    unlocked_phases["hard_gate"] = True
                    phase_unlock_steps["hard_gate"] = step
                    _new_unlocks.append("hard_gate")
                if not unlocked_phases["post_vq_attn"] and (
                    (unlocked_phases["hard_gate"] and step >= phase_unlock_steps.get("hard_gate", step) + 150)
                    or step >= phase_post_vq_attn_step
                ):
                    unlocked_phases["post_vq_attn"] = True
                    phase_unlock_steps["post_vq_attn"] = step
                    _new_unlocks.append("post_vq_attn")
                if not unlocked_phases["pair_decor"] and (
                    (unlocked_phases["post_vq_attn"] and step >= phase_unlock_steps.get("post_vq_attn", step) + 150)
                    or step >= phase_pair_decor_step
                ):
                    unlocked_phases["pair_decor"] = True
                    phase_unlock_steps["pair_decor"] = step
                    _new_unlocks.append("pair_decor")
                if not unlocked_phases["meta"] and (
                    (unlocked_phases["pair_decor"] and _cur_act >= 0.65 and step >= int(steps * 0.30))
                    or step >= phase_meta_step
                ):
                    unlocked_phases["meta"] = True
                    phase_unlock_steps["meta"] = step
                    _new_unlocks.append("meta")
                if not unlocked_phases["meta_loss"] and (
                    (unlocked_phases["meta"] and step >= phase_unlock_steps.get("meta", step) + 250)
                    or step >= phase_meta_loss_step
                ):
                    unlocked_phases["meta_loss"] = True
                    phase_unlock_steps["meta_loss"] = step
                    _new_unlocks.append("meta_loss")
                if not unlocked_phases["chunk"] and (
                    (unlocked_phases["meta_loss"] and _cur_meta_act >= 0.60 and step >= phase_unlock_steps.get("meta_loss", step) + 400)
                    or step >= phase_chunk_step
                ):
                    unlocked_phases["chunk"] = True
                    phase_unlock_steps["chunk"] = step
                    _new_unlocks.append("chunk")
                if not unlocked_phases["chunk_loss"] and (
                    (unlocked_phases["chunk"] and step >= phase_unlock_steps.get("chunk", step) + 250)
                    or step >= phase_chunk_loss_step
                ):
                    unlocked_phases["chunk_loss"] = True
                    phase_unlock_steps["chunk_loss"] = step
                    _new_unlocks.append("chunk_loss")

                if _new_unlocks:
                    _snap_path = os.path.join(phase_dir, f"step_{step:06d}_{'_'.join(_new_unlocks)}.pt")
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "step": step,
                            "unlocked": _new_unlocks,
                            "stream_names": list(model.sub_encoder.submodality_names),
                            "stream_active": list(model.stream_active),
                        },
                        _snap_path,
                    )
                    phase_snapshots_saved += 1
                    print(f"  ✦ Phase unlocked ({', '.join(_new_unlocks)}) @ step {step} → {_snap_path}")

            model.update_phases(
                step,
                hard_gate=unlocked_phases["hard_gate"],
                post_vq_attn=unlocked_phases["post_vq_attn"],
                pair_decor=unlocked_phases["pair_decor"],
                meta=unlocked_phases["meta"],
                meta_loss=unlocked_phases["meta_loss"],
                chunk=unlocked_phases["chunk"],
                chunk_loss=unlocked_phases["chunk_loss"],
            )

            # Update fixed GPU ring buffer periodically with zero allocation
            if step % 8 == 0 or model._concept_bank_len < 1000:
                with torch.no_grad():
                    flat_embeds = model.embedding(x).detach().reshape(-1, model.d_model).float()
                    num_samples = min(512, flat_embeds.size(0))
                    idx = torch.randint(0, flat_embeds.size(0), (num_samples,), device=flat_embeds.device)
                    samples = flat_embeds[idx]
                    cap = model.concept_bank_buf.size(0)
                    end = model._concept_bank_ptr + num_samples
                    if end <= cap:
                        model.concept_bank_buf[model._concept_bank_ptr : end] = samples
                    else:
                        first = cap - model._concept_bank_ptr
                        model.concept_bank_buf[model._concept_bank_ptr :] = samples[:first]
                        model.concept_bank_buf[: num_samples - first] = samples[first:]
                    model._concept_bank_ptr = end % cap
                    model._concept_bank_len = min(cap, model._concept_bank_len + num_samples)

            def _run_fwd():
                with autocast(device_type=device.type, dtype=_amp_dtype, enabled=_use_amp):
                    out = _compiled_model(x)
                    lt = criterion(out[0].view(-1, vocab_size), y.view(-1))
                return out, lt

            optimizer.zero_grad(set_to_none=True)
            while True:
                try:
                    _fwd_out, loss_task = _run_fwd()
                    break
                except Exception as _fwd_err:
                    _ename = type(_fwd_err).__name__
                    _emsg = str(_fwd_err).lower()
                    if _compiled_model is not model and (
                        "inductor" in _ename.lower()
                        or "compiler" in _ename.lower()
                        or "triton" in _ename.lower()
                        or "cl is not found" in _emsg
                    ):
                        print(f"  ⚠ torch.compile fallback ({_ename}); switching to eager CUDA.")
                        _torch_compile_available = False
                        _compiled_model = model
                        continue
                    if ("out of memory" in _emsg or "OutOfMemoryError" in _ename) and batch_size > batch_size_min:
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        _old_bs = batch_size
                        batch_size_tokens = max(batch_size_min * safe_seq, batch_size_tokens // 2)
                        batch_tokens_max = max(batch_size_tokens, batch_tokens_max // 2)
                        batch_size = int(max(batch_size_min, min(batch_size_max, batch_size_tokens // safe_seq)))
                        total_tokens_seen += (batch_size - _old_bs) * safe_seq
                        x, y = _random_batch(train_data, batch_size, safe_seq)
                        print(f"  ⚠ CUDA OOM — auto-backing off batch_tokens → {batch_size_tokens} (cap {batch_tokens_max})")
                        continue
                    raise

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
            ) = _fwd_out

            active_ratio_t = per_stream_active_t.detach().mean()

            # All controller math stays on GPU (zero per-step syncs)
            if not orth_unlocked or stream_corr_calib_done:
                with torch.no_grad():
                    sub_f = sub_features.detach().reshape(-1, model.NUM_SUBMODALITIES, model.d_model)
                    sub_f_n = F.normalize(sub_f, dim=2)
                    corr_mat = torch.einsum("bnd,bmd->nm", sub_f_n, sub_f_n) / max(1, sub_f_n.shape[0])
                    n_s = model.NUM_SUBMODALITIES
                    idx_i, idx_j = torch.triu_indices(n_s, n_s, offset=1, device=device)
                    mean_corr_t = corr_mat[idx_i, idx_j].mean()
                    stream_corr_ema_t = (1.0 - stream_corr_alpha) * stream_corr_ema_t + stream_corr_alpha * mean_corr_t

                if not orth_unlocked:
                    if not stream_corr_calib_done and step >= 100:
                        stream_corr_baseline = float(stream_corr_ema_t.item())
                        stream_corr_calib_done = True
                        print(f"  ✦ Stream correlation baseline calibrated: {stream_corr_baseline:.4f}")
                    elif stream_corr_calib_done and step % 20 == 0:
                        corr_val = float(stream_corr_ema_t.item())
                        if corr_val > stream_corr_baseline + stream_corr_trigger:
                            orth_T0, orth_unlocked = step, True
                            print(
                                f"  ✦ Orth unlocked at step {step}  "
                                f"(corr {corr_val:.4f} > baseline {stream_corr_baseline:.4f} + {stream_corr_trigger})"
                            )
                        elif step >= orth_fallback_step:
                            orth_T0, orth_unlocked = step, True
                            print(f"  ✦ Orth unlocked at step {step} (fallback)")

            orth_weight = 0.8 / (1.0 + math.exp(-(step - orth_T0) / max(1, orth_tau))) if orth_unlocked else 0.0
            orth_lambda_t = (
                orth_lambda_t * (1.0 + orth_lambda_lr * (stream_corr_ema_t - target_corr))
            ).clamp(orth_lambda_min, orth_lambda_max)
            orth_loss = orth_weight * orth_lambda_t * feat_orth_loss

            gate_gamma_t = (
                gate_gamma_t * (1.0 + gate_gamma_lr * (gate_kl.detach() - target_gate_kl))
            ).clamp(gate_gamma_min, gate_gamma_max)

            soft_mean = soft_probs.mean(dim=(0, 1))
            if step >= gate_specialize_step:
                spec_active_f = (active_ratio_t > gate_specialize_act_threshold).float()
                spec_ramp = min(1.0, (step - gate_specialize_step) / max(1.0, steps * 0.13))
            else:
                spec_active_f = loss_task.new_zeros(())
                spec_ramp = 0.0

            with torch.no_grad():
                _, inv_idx, tok_counts = torch.unique(
                    x.reshape(-1), return_inverse=True, return_counts=True
                )
                uniq_w = 1.0 / tok_counts[inv_idx].float()
                uniq_w = uniq_w / uniq_w.sum().clamp_min(1.0)
            uniq_soft_mean = (soft_probs.reshape(-1, model.NUM_SUBMODALITIES) * uniq_w.unsqueeze(1)).sum(dim=0)

            p_gate_mix = 0.5 * soft_mean.detach() + 0.5 * uniq_soft_mean.detach()
            if all(model.stream_active):
                target_p = 1.0 / model.NUM_SUBMODALITIES
                p_act_g = p_gate_mix
                n_act_g = model.NUM_SUBMODALITIES
                balance_loss = balance_mult_t * (
                    0.5 * ((soft_mean - target_p) ** 2).sum()
                    + 0.5 * ((uniq_soft_mean - target_p) ** 2).sum()
                )
            else:
                active_mask = model.stream_mask > 0
                n_act_g = max(1, int(model._n_active_streams))
                target_p = 1.0 / n_act_g
                p_act_g = p_gate_mix[active_mask]
                balance_loss = balance_mult_t * (
                    0.5 * ((soft_mean[active_mask] - target_p) ** 2).sum()
                    + 0.5 * ((uniq_soft_mean[active_mask] - target_p) ** 2).sum()
                )
            p_act_n = p_act_g / p_act_g.sum().clamp_min(1e-9)
            gate_bal_H = -(p_act_n * torch.log(p_act_n.clamp_min(1e-10))).sum()
            target_bal_H = 0.92 * math.log(max(2, n_act_g))
            bal_min = 0.15 if step >= gate_specialize_step else 0.30
            balance_mult_t = (balance_mult_t * (1.0 + 0.02 * (target_bal_H - gate_bal_H))).clamp(bal_min, 1.00)

            gate_token_entropy = -(
                gate_weights.clamp_min(1e-9) * gate_weights.clamp_min(1e-9).log()
            ).sum(dim=-1).mean()
            gate_peak_loss = spec_active_f * (gate_peak_lambda_max * spec_ramp) * gate_token_entropy
            gate_sparsity_loss = spec_active_f * (gate_sparsity_lambda_max * spec_ramp) * sparsity_loss

            _sym_decor_active = unlocked_phases["pair_decor"]
            if _sym_decor_active and pair_js_vec.numel() > 0:
                pair_lam_t = (
                    pair_lam_t * (1.0 + pair_sym_lambda_lr * (pair_js_vec - target_pair_js))
                ).clamp(pair_sym_lambda_min, pair_sym_lambda_max)
                pair_sym_loss = (pair_lam_t * pair_age_weights_t * (pair_js_vec + pair_grad_vec)).sum()
            else:
                pair_sym_loss = loss_task.new_zeros(())

            vq_beta_t = (vq_beta_t * (1.0 + vq_beta_lr * (target_active - active_ratio_t))).clamp(
                vq_beta_min, vq_beta_max
            )

            _meta_usage_grad = F.softmax(meta_pred_logits, dim=-1).mean(dim=(0, 1))
            _meta_uniform = torch.full_like(_meta_usage_grad, 1.0 / _meta_usage_grad.shape[0])
            _meta_log_usage = torch.log(_meta_usage_grad.clamp_min(1e-10))
            meta_balance_loss = F.kl_div(
                _meta_log_usage, _meta_uniform, reduction="sum", log_target=False
            ).clamp(max=20.0)
            meta_balance_gamma = 0.01

            meta_cb_active_ema_t = (
                (1.0 - cb_active_ema_alpha) * meta_cb_active_ema_t
                + cb_active_ema_alpha * meta_active_ratio_t.detach()
            )
            if unlocked_phases["meta_loss"]:
                meta_collapsed = meta_cb_active_ema_t < 0.15
                meta_vq_beta_t = torch.where(
                    meta_collapsed,
                    (meta_vq_beta_t * 0.99).clamp(min=meta_vq_beta_min),
                    (meta_vq_beta_t * (1.0 + meta_vq_beta_lr * (meta_target_active - meta_active_ratio_t.detach()))).clamp(
                        meta_vq_beta_min, meta_vq_beta_max
                    ),
                )
                meta_commit_cost_next_t = torch.where(
                    meta_collapsed,
                    (meta_commit_cost_t * 0.99).clamp(min=meta_commit_cost_min),
                    (
                        meta_commit_cost_t
                        * (1.0 + meta_commit_cost_lr * (meta_target_active - meta_active_ratio_t.detach()))
                    ).clamp(meta_commit_cost_min, meta_commit_cost_max),
                )
            else:
                meta_vq_beta_t = (
                    meta_vq_beta_t * (1.0 + meta_vq_beta_lr * (meta_target_active - meta_active_ratio_t.detach()))
                ).clamp(meta_vq_beta_min, meta_vq_beta_max)
                meta_commit_cost_next_t = (
                    meta_commit_cost_t
                    * (1.0 + meta_commit_cost_lr * (meta_target_active - meta_active_ratio_t.detach()))
                ).clamp(meta_commit_cost_min, meta_commit_cost_max)

            meta_K = meta_pred_logits.shape[-1]
            meta_pred_loss = F.cross_entropy(meta_pred_logits.reshape(-1, meta_K), meta_targets.reshape(-1))

            with torch.no_grad():
                acc_now_t = (meta_pred_logits.detach().argmax(dim=-1) == meta_targets).float().mean()
                meta_pred_acc_ema_t = (1.0 - meta_pred_acc_alpha) * meta_pred_acc_ema_t + meta_pred_acc_alpha * acc_now_t

            if unlocked_phases["meta_loss"]:
                _target_mp_lam = (
                    0.05 * loss_task.detach() / meta_pred_loss.detach().clamp(min=0.2)
                ).clamp(meta_pred_lambda_min, meta_pred_lambda_max)
                meta_pred_lambda_t = 0.95 * meta_pred_lambda_t + 0.05 * _target_mp_lam

            alphas_t = 0.5 * torch.sigmoid(torch.stack([cb.residual_alpha_raw for cb in model.codebooks]))
            mean_alpha_t = alphas_t.mean().detach()
            _alpha_prog = min(1.0, step / max(1, int(steps * 0.70)))
            _target_alpha_now = target_alpha - 0.09 * _alpha_prog
            alpha_reg_lambda_t = (
                alpha_reg_lambda_t * (1.0 + alpha_lr * (mean_alpha_t - _target_alpha_now))
            ).clamp(alpha_lambda_min, alpha_lambda_max)
            alpha_reg = alpha_reg_lambda_t * alphas_t.pow(2).sum()

            category_active_ema_t = 0.9 * category_active_ema_t + 0.1 * category_active_ratio_t.detach()
            category_vq_beta_t = (
                category_vq_beta_t * (1.0 + category_vq_beta_lr * (category_target_active - category_active_ema_t))
            ).clamp(category_vq_beta_min, category_vq_beta_max)

            chunk_down_loss_ema_t = 0.95 * chunk_down_loss_ema_t + 0.05 * chunk_down_loss.detach()
            if unlocked_phases["chunk_loss"]:
                chunk_down_lambda_t = (
                    chunk_down_lambda_t.clamp(min=chunk_down_lambda_min)
                    * (1.0 + chunk_down_lambda_lr * (chunk_down_loss_ema_t - chunk_down_target_mse))
                ).clamp(chunk_down_lambda_min, chunk_down_lambda_max)
                chunk_down_lambda_eff = chunk_down_lambda_t
            else:
                chunk_down_lambda_eff = 0.0

            if unlocked_phases["meta_loss"]:
                meta_guard = (active_ratio_t > 0.45).float()
                meta_vq_beta_eff = meta_vq_beta_t * meta_guard
                meta_pred_lambda_eff = meta_pred_lambda_t * meta_guard
            else:
                meta_vq_beta_eff = 0.0
                meta_pred_lambda_eff = 0.0

            category_vq_beta_eff = category_vq_beta_t if unlocked_phases["chunk_loss"] else 0.0
            gate_gamma_eff = gate_gamma_t if unlocked_phases["post_vq_attn"] else 0.0
            pair_sym_loss_eff = (pair_sym_loss * 1.4) if _sym_decor_active else pair_sym_loss * 0.0

            total_loss = (
                loss_task
                + vq_beta_t * vq_loss
                + meta_vq_beta_eff * meta_vq_loss
                + meta_pred_lambda_eff * meta_pred_loss
                + (meta_balance_gamma * meta_balance_loss if unlocked_phases["meta_loss"] else 0.0)
                + category_vq_beta_eff * category_vq_loss
                + chunk_down_lambda_eff * chunk_down_loss
                + orth_loss
                + gate_gamma_eff * gate_kl
                + balance_loss
                + gate_peak_loss
                + gate_sparsity_loss
                + alpha_reg
                + pair_sym_loss_eff
            )

            if not _use_scaler and device.type == "cpu" and not torch.isfinite(total_loss):
                optimizer.zero_grad(set_to_none=True)
                for g in optimizer.param_groups:
                    g["lr"] = max(lr_min, g["lr"] * 0.95)
                ce_prev_initialized = False
                if step % log_every == 0:
                    print(f"  ⚠ NaN/Inf loss at step {step} — skipping step, LR reduced ×0.95")
                continue

            try:
                scaler.scale(total_loss).backward()
            except Exception as _bwd_err:
                if ("out of memory" in str(_bwd_err).lower() or "OutOfMemoryError" in type(_bwd_err).__name__) and batch_size > batch_size_min:
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    batch_size_tokens = max(batch_size_min * safe_seq, batch_size_tokens // 2)
                    batch_tokens_max = max(batch_size_tokens, batch_tokens_max // 2)
                    print(f"  ⚠ CUDA OOM in backward — auto-backing off batch_tokens → {batch_size_tokens} (cap {batch_tokens_max})")
                    continue
                raise

            scaler.unscale_(optimizer)

            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            is_fin_grad = torch.isfinite(raw_norm)
            is_fin_loss = torch.isfinite(loss_task.detach()) & is_fin_grad
            grad_norm_ema_t = torch.where(
                is_fin_grad,
                (1.0 - grad_norm_alpha) * grad_norm_ema_t + grad_norm_alpha * raw_norm.detach(),
                grad_norm_ema_t,
            )

            if not _use_scaler and device.type == "cpu" and not is_fin_grad:
                if step % log_every == 0:
                    print(f"  ⚠ NaN/Inf gradient at step {step} — skipping step, LR ×0.95", flush=True)
                optimizer.zero_grad(set_to_none=True)
                for g in optimizer.param_groups:
                    g["lr"] = max(lr_min, g["lr"] * 0.95)
                ce_prev_initialized = False
                continue

            scaler.step(optimizer)
            scaler.update()

            _cos_start = int(steps * 0.65)
            if step >= _cos_start and step > lr_warmup:
                _prog = (step - _cos_start) / max(1, steps - _cos_start)
                _cos_cap = lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * _prog))
                lr_ceiling = min(lr_ceiling, _cos_cap)
                for g in optimizer.param_groups:
                    if g["lr"] > lr_ceiling:
                        g["lr"] = max(lr_min, lr_ceiling)

            if step % 20 == 0:
                grad_clip = max(grad_clip_min, min(grad_clip_max, float(grad_norm_ema_t.item()) * grad_clip_factor))

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

            # Closed-loop GPU controllers (async GPU->GPU copies, zero CPU syncs)
            with torch.no_grad():
                _topk_target_H = 0.55 * math.log(max(2, model.codebooks[0].topk))
                topk_tau_t = (
                    topk_tau_t * (1.0 + 0.01 * (_topk_target_H - per_stream_topk_H_t.detach().mean()))
                ).clamp(0.04, 0.25)
                for _cb in model.codebooks:
                    _cb.topk_tau_buf.copy_(topk_tau_t)

                _meta_target_H = 0.55 * math.log(max(2, model.meta_codebook.topk))
                meta_topk_tau_t = (
                    meta_topk_tau_t * (1.0 + 0.01 * (_meta_target_H - meta_topk_H_t.detach()))
                ).clamp(0.04, 0.20)
                model.meta_codebook.topk_tau_buf.copy_(meta_topk_tau_t)

                if unlocked_phases["hard_gate"]:
                    _hg_start = phase_unlock_steps.get("hard_gate", step)
                    _gt_prog = min(1.0, (step - _hg_start) / max(1, int(steps * 0.50)))
                    _target_gate_H = (0.65 - 0.35 * _gt_prog) * math.log(2.0)
                    gate_temperature_t = (
                        gate_temperature_t * (1.0 + 0.01 * (_target_gate_H - gate_token_entropy.detach()))
                    ).clamp(0.15, 0.80)
                    model._gate_temp_buf.copy_(gate_temperature_t)

                _sd_target = (0.05 + 0.40 * (stream_corr_ema_t - 0.15).clamp(min=0.0)).clamp(0.05, 0.25)
                if is_overfitting:
                    _sd_target = (_sd_target + 0.05).clamp(max=0.25)
                stream_dropout_t = 0.98 * stream_dropout_t + 0.02 * _sd_target
                model._stream_dropout_buf.copy_(stream_dropout_t)

            if device.type == "cuda" and step >= max(400, lr_warmup) and step % 400 == 0 and batch_size_tokens < batch_tokens_max:
                _total_vram = torch.cuda.get_device_properties(device).total_memory
                _res_vram = torch.cuda.memory_reserved(device)
                if _res_vram < 0.55 * _total_vram:
                    batch_size_tokens = min(batch_tokens_max, batch_size_tokens * 2)
                    print(f"  ↑ Proactive batch_tokens → {batch_size_tokens} (VRAM {_res_vram/1024**3:.1f}/{_total_vram/1024**3:.1f}G)")

            if step >= lr_warmup:
                cur_ce_t = loss_task.detach()
                if ce_prev_initialized:
                    raw_vel_t = cur_ce_t - ce_prev_t
                    ce_vel_ema_t = torch.where(
                        is_fin_loss,
                        (1.0 - vel_ema_alpha) * ce_vel_ema_t + vel_ema_alpha * raw_vel_t,
                        ce_vel_ema_t,
                    )
                ce_prev_t = torch.where(is_fin_loss, cur_ce_t, ce_prev_t)
                ce_prev_initialized = True

                if not ce_ema_initialized:
                    ce_ema_t = torch.where(is_fin_loss, cur_ce_t, ce_ema_t)
                    ce_ema_initialized = True
                else:
                    ce_ema_t = torch.where(
                        is_fin_loss,
                        (1.0 - ce_ema_alpha) * ce_ema_t + ce_ema_alpha * cur_ce_t,
                        ce_ema_t,
                    )
                improved_t = is_fin_loss & (ce_ema_t < best_ce_ema_t * (1.0 - lr_reduce_threshold))
                best_ce_ema_t = torch.where(improved_t, ce_ema_t, best_ce_ema_t)
                steps_since_imp_t = torch.where(
                    is_fin_loss,
                    torch.where(improved_t, torch.zeros_like(steps_since_imp_t), steps_since_imp_t + 1),
                    steps_since_imp_t,
                )

                if step % 20 == 0:
                    steps_since_imp = int(steps_since_imp_t.item())
                    if steps_since_imp >= lr_patience and step > vel_last_action + vel_cooldown // 2:
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
                            print(
                                f"  ↓ LR  {cur_lr:.2e} → {new_lr:.2e}  "
                                f"(patience #{lr_reductions}, stalled {steps_since_imp} steps)"
                            )
                            print(f"  ↑ batch_tokens → {batch_size_tokens}")
                        steps_since_imp = 0
                        steps_since_imp_t.zero_()

                if step >= vel_min_step and step % vel_check_every == 0 and step > vel_last_action + vel_cooldown:
                    ce_vel_val = float(ce_vel_ema_t.item())
                    cur_lr = optimizer.param_groups[0]["lr"]

                    if ce_vel_val >= vel_reduce_thresh:
                        new_lr = max(lr_min, cur_lr * lr_reduce_factor)
                        if new_lr < cur_lr:
                            for g in optimizer.param_groups:
                                g["lr"] = new_lr
                            lr_ceiling = new_lr
                            vel_last_action = step
                            lr_reductions += 1
                            print(
                                f"  ↓ LR {cur_lr:.2e} → {new_lr:.2e}  "
                                f"(vel {ce_vel_val:+.5f} ≥ {vel_reduce_thresh:+.5f}, reduction #{lr_reductions})"
                            )
                            batch_size_tokens = min(batch_tokens_max, batch_size_tokens * 2)
                            print(f"  ↑ batch_tokens → {batch_size_tokens}")
                            val_every = min(val_every_max, val_every * 2)
                    elif ce_vel_val <= vel_boost_thresh:
                        new_lr = min(lr_ceiling, cur_lr * lr_boost_factor)
                        if new_lr > cur_lr * 1.001:
                            for g in optimizer.param_groups:
                                g["lr"] = new_lr
                            vel_last_action = step
                            print(f"  ↑ LR {cur_lr:.2e} → {new_lr:.2e}  (vel {ce_vel_val:+.5f} ≤ {vel_boost_thresh:+.5f})")

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
                    if not is_overfitting and mean_norm > 0 and last_norm / mean_norm > depth_grad_ratio_thresh:
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
                cb_commit_costs_t = (
                    cb_commit_costs_t * (1.0 + commit_cost_lr * (target_active_per - per_stream_active_t.detach()))
                ).clamp(commit_cost_min, commit_cost_max)
                for i, cb in enumerate(model.codebooks):
                    cb.commitment_cost_buf.copy_(cb_commit_costs_t[i])
                meta_commit_cost_t.copy_(meta_commit_cost_next_t)
                model.meta_codebook.commitment_cost_buf.copy_(meta_commit_cost_t)

            cb_active_ema_t = (
                (1.0 - cb_active_ema_alpha) * cb_active_ema_t + cb_active_ema_alpha * per_stream_active_t.detach()
            )

            stream_check_every = min(400, 100 + int(step / max(1, steps) * 300))

            if step > 0 and step % stream_check_every == 0:
                N = model.NUM_SUBMODALITIES
                gate_mean_np = gate_weights.detach().mean(dim=(0, 1)).cpu().tolist()
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
                    _mean_cb_act = float(cb_active_ema_t.mean().item())
                    _mean_cb_sz = sum(cb.num_symbols for cb in model.codebooks) / max(1, len(model.codebooks))
                    _capacity_bottleneck = (
                        _mean_cb_act > 0.82
                        and _mean_cb_sz >= num_symbols_target * 0.85
                        and float(stream_corr_ema_t.item()) < 0.30
                    )
                    quorum_met = _capacity_bottleneck or (
                        n_saturated >= math.ceil(stream_trigger_quorum * len(active_emas)) and _mean_cb_act > 0.75
                    )
                    if quorum_met:
                        new_name = candidate_streams[next_candidate_idx]
                        next_candidate_idx += 1

                        try:
                            with torch.no_grad():
                                if model._concept_bank_len > 1000:
                                    bank = model.concept_bank_buf[: model._concept_bank_len].float()
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
                        cb_grow_counts.append(0)

                        pair_sym_lambdas = dict(zip(pair_idx_pairs, pair_lam_t.detach().cpu().tolist()))
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

                        cb_active_ema_t = torch.cat([
                            cb_active_ema_t, torch.tensor([0.5], device=device, dtype=torch.float32)
                        ])
                        cb_commit_costs_t = torch.cat([
                            cb_commit_costs_t,
                            torch.tensor([model.codebooks[-1].commitment_cost], device=device, dtype=torch.float32),
                        ])
                        pair_idx_pairs, pair_lam_t, pair_age_weights_t = _rebuild_pair_tensors(
                            model.NUM_SUBMODALITIES, pair_sym_lambdas
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
                        ce_vel_ema_t.fill_(vel_boost_thresh * 2.0)
                        vel_last_action = step + vel_cooldown * 2
                        ce_prev_initialized = False
                        print(f"  ↑ Global LR reset → {_global_reset_lr:.2e}  (stream addition, ceiling reset)")
                        stream_last_added_step = step
                        if _torch_compile_available:
                            print("\n[JIT] Recompiling model after stream addition…", flush=True)
                            torch._dynamo.reset()
                            _compiled_model = torch.compile(model, dynamic=True)

            if step >= codebook_growth_enable_step and step > 0 and step % cb_grow_check_every == 0:
                cb_active_list = cb_active_ema_t.detach().cpu().tolist()
                grew_any_cb = False
                for i, cb in enumerate(model.codebooks):
                    if cb.num_symbols >= num_symbols_target:
                        cb_grow_counts[i] = 0
                        continue
                    if cb_active_list[i] > cb_grow_active_thresh:
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
                    if float(meta_cb_active_ema_t.item()) > cb_grow_active_thresh:
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
                        cb_active_ema_t[i] = 0.5
                        cb_grow_counts[i] = 0
                meta_revived = model.meta_codebook.revive_dead_entries(device, revival_noise_scale)
                if meta_revived > 0:
                    total_revived += meta_revived
                    meta_cb_active_ema_t.fill_(0.5)
                    meta_cb_grow_count = 0
                if total_revived > 0:
                    print(f"  ♻ Revived {total_revived} dead codebook entries across {model.NUM_SUBMODALITIES} streams + meta")

            if _sym_decor_active and step > 0 and step % 100 == 0 and pair_js_vec.numel() > 0:
                worst_idx = int(torch.argmax(pair_js_vec).item())
                worst_js = float(pair_js_vec[worst_idx].item())
                median_js = float(pair_js_vec.median().item())
                worst_pair = pair_idx_pairs[worst_idx]
                pi, pj = worst_pair
                if worst_js > max(kickstart_js_thresh, median_js + 0.12):
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
                loss_val = float(loss_task.detach().item())
                ppl = math.exp(min(loss_val, 20.0))
                vq_val = float(vq_loss.detach().item())
                per_stream_active = per_stream_active_t.detach().cpu().tolist()
                per_stream_topk_H = per_stream_topk_H_t.detach().cpu().tolist()
                active_ratio = sum(per_stream_active) / max(1, len(per_stream_active))
                mean_H = float(mean_H_t.item())
                meta_ema_H = float(meta_ema_H_t.item())
                meta_active_ratio = float(meta_active_ratio_t.item())
                meta_topk_H = float(meta_topk_H_t.item())
                category_ema_H = float(category_ema_H_t.item())
                category_active_ratio = float(category_active_ratio_t.item())

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
                vel_str = f"{float(ce_vel_ema_t.item()):+.4f}" if ce_prev_initialized else "n/a"
                corr_str = f"{float(stream_corr_ema_t.item()):.3f}" if stream_corr_calib_done else "cal"

                names_abbrev = [n[:3] for n in model.sub_encoder.submodality_names]
                gw = gate_weights.detach().mean(dim=(0, 1)).cpu()
                n_gw = gw.shape[0]
                gate_str = " ".join(
                    f"{names_abbrev[i]}={gw[i].item():.2f}" for i in range(min(len(names_abbrev), n_gw))
                )

                act_min = min(per_stream_active) * 100
                act_max = max(per_stream_active) * 100
                act_str = f"{active_ratio*100:.1f}%[{act_min:.0f}–{act_max:.0f}]"

                if _sym_decor_active and pair_js_vec.numel() > 0:
                    w_idx = int(torch.argmax(pair_js_vec).item())
                    worst_js = float(pair_js_vec[w_idx].item())
                    wpi, wpj = pair_idx_pairs[w_idx]
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

                gnorm_str = f"{float(raw_norm.detach().item()):.2f}→{grad_clip:.2f}"
                vram_str = f"{torch.cuda.memory_reserved() / 1024**3:.1f}G" if torch.cuda.is_available() else "CPU"
                tok_per_sec = (batch_size * safe_seq) / max(step_secs, 1e-6)
                thr_str = f"{tok_per_sec / 1000:.1f}k"

                seen_m = total_tokens_seen / 1_000_000
                epochs = total_tokens_seen / max(1, n_train)
                seen_str = f"{seen_m:.1f}M({epochs:.1f}x)"

                alphas_list = alphas_t.detach().cpu().tolist()
                a_min, a_max = min(alphas_list), max(alphas_list)
                alpha_str = f"μ{float(mean_alpha_t.item()):.3f}[{a_min:.2f}–{a_max:.2f}]"

                print(
                    f"Step {step:5d}/{steps} | "
                    f"CE {loss_val:.4f} | "
                    f"PPL {ppl:.2f} | "
                    f"VQ {vq_val:.4f}(β{float(vq_beta_t.item()):.2f},cc{model.codebooks[0].commitment_cost:.3f}) | "
                    f"MetaCB Act{meta_active_ratio:.2f} H{meta_ema_H:.2f}→{model.meta_codebook.H_target:.2f} "
                    f"β{float(meta_vq_beta_t.item()):.2f} cc{model.meta_codebook.commitment_cost:.3f} TopKH{meta_topk_H:.2f} | "
                    f"MetaPred CE{float(meta_pred_loss.detach().item()):.3f} Acc{float(meta_pred_acc_ema_t.item()):.2f} "
                    f"λ{float(meta_pred_lambda_t.item()):.3f} Bal{float(meta_balance_loss.detach().item()):.3f} | "
                    f"ChunkUp Act{category_active_ratio:.2f} H{category_ema_H:.2f}→{model.chunk_up.codebook.H_target:.2f} "
                    f"β{float(category_vq_beta_t.item()):.2f} | "
                    f"ChunkDn MSE{float(chunk_down_loss.detach().item()):.3f} λ{float(chunk_down_lambda_t.item()):.3f} | "
                    f"Orth {float(orth_loss.detach().item()):.2e}(λ{float(orth_lambda_t.item()):.2e},ρ{corr_str}) | "
                    f"GKL {float(gate_kl.detach().item()):.3f}(γ{float(gate_gamma_t.item()):.4f},gt{float(gate_temperature_t.item()):.2f}) | "
                    f"SymD μ{float(sym_decor_loss.detach().item()):.3f} ↑{worst_str} | "
                    f"Act {act_str} | "
                    f"TopK-H {topk_str} | "
                    f"α {alpha_str}(λ{float(alpha_reg_lambda_t.item()):.4f}) | "
                    f"H {mean_H:.2f}→{model.codebooks[0].H_target:.2f} | "
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
                    print(f"  Gates: {gate_str}  (bal_m={float(balance_mult_t.item()):.2f}, sdo={float(stream_dropout_t.item()):.2f})")
                    if per_stream_topk_H:
                        topkh_detail = " ".join(
                            f"{names_abbrev[i]}={per_stream_topk_H[i]:.2f}" for i in range(len(per_stream_topk_H))
                        )
                        print(f"  TopK-H/stream: {topkh_detail}")

                _metrics_writer.writerow([
                    step, "train", f"{loss_val:.6f}", f"{ppl:.4f}", f"{vq_val:.6f}",
                    f"{float(vq_beta_t.item()):.4f}", f"{model.codebooks[0].commitment_cost:.4f}",
                    f"{meta_active_ratio:.4f}", f"{meta_ema_H:.4f}",
                    f"{float(meta_pred_loss.detach().item()):.6f}", f"{float(meta_pred_acc_ema_t.item()):.4f}",
                    f"{category_active_ratio:.4f}", f"{float(chunk_down_loss.detach().item()):.6f}",
                    f"{float(orth_loss.detach().item()):.6e}", f"{float(orth_lambda_t.item()):.6e}",
                    f"{float(stream_corr_ema_t.item()):.4f}",
                    f"{float(gate_kl.detach().item()):.6f}", f"{float(gate_gamma_t.item()):.6f}",
                    f"{float(sym_decor_loss.detach().item()):.6f}", f"{active_ratio:.4f}",
                    f"{mean_topk_H:.4f}", f"{float(mean_alpha_t.item()):.4f}",
                    f"{mean_H:.4f}", f"{model.codebooks[0].H_target:.4f}",
                    batch_size, safe_seq, total_tokens_seen,
                    f"{float(raw_norm.detach().item()):.4f}", f"{grad_clip:.4f}",
                    f"{float(ce_vel_ema_t.item()):.6f}" if ce_prev_initialized else "",
                    f"{current_dropout:.4f}", num_layers, n_active, f"{avg_cb:.2f}",
                    f"{optimizer.param_groups[0]['lr']:.6e}",
                    "", "", f"{best_val_loss:.6f}" if math.isfinite(best_val_loss) else "",
                    f"{elapsed:.2f}", f"{tok_per_sec:.1f}",
                ])
                _metrics_file.flush()

            if val_data is not None and step > 0 and (step % val_every == 0 or step == steps - 1):
                last_train_ce = float(loss_task.detach().item())
                model.eval()
                _val_batches = 24
                _val_rng = torch.Generator().manual_seed(12345)
                _val_losses = []
                with torch.no_grad():
                    _max_sv = val_data.shape[0] - safe_seq - 1
                    _off_v = torch.arange(safe_seq, device=val_data.device)
                    for _ in range(_val_batches):
                        _ix_v = torch.randint(0, _max_sv, (batch_size,), generator=_val_rng).to(val_data.device)
                        _idx_v = _ix_v.unsqueeze(1) + _off_v.unsqueeze(0)
                        xv, yv = val_data[_idx_v], val_data[_idx_v + 1]
                        lv, *_ = model(xv)
                        _val_losses.append(criterion(lv.view(-1, vocab_size), yv.view(-1)).item())
                val_ce = sum(_val_losses) / len(_val_losses)
                last_val_ce = val_ce
                marker = ""
                if val_ce < best_val_loss:
                    best_val_loss = val_ce
                    _save_full_checkpoint(best_checkpoint, step, tag="best val")
                    marker = "  ✓ best"
                    val_degrade_count = 0
                else:
                    if not unlocked_phases["chunk_loss"]:
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
                            g["lr"] = max(lr_min, g["lr"] * 0.90)
                        lr_ceiling = optimizer.param_groups[0]["lr"]
                        vel_last_action = step
                    elif val_ce > best_val_loss:
                        for g in optimizer.param_groups:
                            g["lr"] = max(lr_min, g["lr"] * 0.97)
                        lr_ceiling = optimizer.param_groups[0]["lr"]
                        vel_last_action = step

                ratio = val_ce / max(last_train_ce, 1e-8)
                is_overfitting = ratio > 1.10
                if ratio > overfit_threshold:
                    current_dropout = current_dropout * (1.0 + dropout_lr * (ratio - overfit_threshold))
                    weight_decay_val = weight_decay_val * (1.0 + weight_decay_lr * (ratio - overfit_threshold))
                    val_every = max(200, val_every // 2)
                else:
                    current_dropout = current_dropout * (1.0 - dropout_lr * 0.5)
                    weight_decay_val = weight_decay_val * (1.0 - weight_decay_lr * 0.5)
                    if ratio < 1.05:
                        val_every = min(val_every_max, val_every + 100)
                current_dropout = max(dropout_min, min(dropout_max, current_dropout))
                weight_decay_val = max(weight_decay_min, min(weight_decay_max, weight_decay_val))

                _snapped_do = round(current_dropout * 20.0) / 20.0
                model.decoder.set_dropout(_snapped_do)
                model.cross_stream_attn.drop.p = _snapped_do
                model.pre_vq_cross_stream_attn.drop.p = _snapped_do
                for g in optimizer.param_groups:
                    if g.get("is_decay", True):
                        g["weight_decay"] = weight_decay_val

                if seq_len_stage_idx < len(seq_len_stages) - 1:
                    seq_len_val_ce_history.append(val_ce)
                    if len(seq_len_val_ce_history) >= 8:
                        seq_len_stage_idx += 1
                        seq_len_max = seq_len_stages[seq_len_stage_idx]
                        seq_len_val_ce_history = []
                        seq_len_improvement_ema = 0.0
                        print(f"  ↑ seq_len_max unlocked → {seq_len_max}  (max val checks reached)")
                    elif len(seq_len_val_ce_history) >= 3:
                        prev_ce = seq_len_val_ce_history[-2]
                        rel_imp = (prev_ce - val_ce) / max(prev_ce, 1e-8)
                        seq_len_improvement_ema = (
                            (1 - seq_len_improvement_alpha) * seq_len_improvement_ema
                            + seq_len_improvement_alpha * rel_imp
                        )
                        if seq_len_improvement_ema < 0.003:
                            _imp_snapshot = seq_len_improvement_ema
                            seq_len_stage_idx += 1
                            seq_len_max = seq_len_stages[seq_len_stage_idx]
                            seq_len_val_ce_history = []
                            seq_len_improvement_ema = 0.0
                            print(
                                f"  ↑ seq_len_max unlocked → {seq_len_max}  "
                                f"(rolling rel Val CE improvement {_imp_snapshot:.4f} < 0.003)"
                            )

                print(
                    f"  ↳ Val CE {val_ce:.4f}  "
                    f"PPL {math.exp(min(val_ce, 20)):.2f}  "
                    f"do→{current_dropout:.3f}  "
                    f"wd→{weight_decay_val:.4f}  "
                    f"ratio→{ratio:.2f}  "
                    f"val_every={val_every}  "
                    f"vel={float(ce_vel_ema_t.item()):+.5f}{marker}"
                )
                _metrics_writer.writerow([
                    step, "val", f"{last_train_ce:.6f}", "", "", "", "", "", "", "", "", "", "",
                    "", "", "", "", "", "", "", "", "", "", "", batch_size, safe_seq,
                    total_tokens_seen, "", "", f"{float(ce_vel_ema_t.item()):.6f}",
                    f"{current_dropout:.4f}", num_layers, sum(model.stream_active), "",
                    f"{optimizer.param_groups[0]['lr']:.6e}",
                    f"{val_ce:.6f}", f"{math.exp(min(val_ce, 20)):.4f}",
                    f"{best_val_loss:.6f}" if math.isfinite(best_val_loss) else "",
                    f"{time.time() - _train_start:.2f}", "",
                ])
                _metrics_file.flush()
                model.train()

    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted! Saving emergency checkpoint…")
        _save_full_checkpoint(interrupt_checkpoint, last_step, tag="interrupt")
        raise
    finally:
        _metrics_file.close()

    _save_full_checkpoint(checkpoint, last_step, tag="final")
    _total = time.time() - _train_start
    print(f"\nFinal checkpoint  → {checkpoint}")
    print(f"Training metrics  → {metrics_csv_path}")
    print(f"Phase snapshots   → {phase_dir} ({phase_snapshots_saved} saved)")
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
    gen_path = out_path("generated_text.txt")
    with open(gen_path, "w", encoding="utf-8") as f:
        f.write(f"SEED PROMPT:\n{seed_text}\n\nGENERATED TEXT:\n{generated_text}\n")
    print(f"Generated text saved → {gen_path}")


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
        pad=12,
    )
    ax.set_xlabel("Psychological Submodalities", fontsize=12)
    ax.set_ylabel("Test Words", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

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

    filtered = []
    for tok in all_tokens:
        word = tokenizer.decode([tok]).strip()
        if len(word) >= 4 and word.isalpha():
            filtered.append((tok, word))

    bucket_activations = {i: [] for i in range(model.NUM_SUBMODALITIES)}

    if filtered:
        model.eval()
        tok_ids = torch.tensor([t for t, _ in filtered], dtype=torch.long, device=dev).unsqueeze(1)
        all_gates = []
        with torch.no_grad():
            for s in range(0, tok_ids.size(0), 512):
                emb = model.embedding(tok_ids[s : s + 512])
                gates, _, _ = model.compute_gate_weights(emb, apply_stream_dropout=False)
                all_gates.append(gates.squeeze(1).cpu())
        gate_mat = torch.cat(all_gates, dim=0).numpy()

        for idx_w, (_, word) in enumerate(filtered):
            for i in range(model.NUM_SUBMODALITIES):
                w = float(gate_mat[idx_w, i])
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
    all_tok_chunks = []
    stream_sym_chunks = [[] for _ in range(N)]

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

            all_tok_chunks.append(x.reshape(-1))

            for i in range(min(N, len(model.codebooks))):
                normed = model.pre_vq_norms[i](sub_features[:, :, i, :])
                flat = normed.reshape(-1, model.codebooks[i].embedding_dim).float()
                flat_s = F.normalize(flat, dim=1)
                cb_s = F.normalize(model.codebooks[i].codebook.weight.float(), dim=1)
                dot = torch.mm(flat_s, cb_s.t())
                syms = dot.argmax(dim=1)
                stream_sym_chunks[i].append(syms)

    if not all_tok_chunks:
        print("  ⚠ No tokens sampled for MI estimate.")
        model.train()
        return

    all_toks_gpu = torch.cat(all_tok_chunks, dim=0)
    total = all_toks_gpu.numel()
    uniq_toks, tok_counts = torch.unique(all_toks_gpu, return_counts=True)
    p_tok_map = dict(zip(uniq_toks.cpu().tolist(), (tok_counts.float() / total).cpu().tolist()))

    results = []
    for i in range(N):
        if not stream_sym_chunks[i]:
            results.append((model.sub_encoder.submodality_names[i], 0.0))
            continue

        syms_gpu = torch.cat(stream_sym_chunks[i], dim=0)
        K_i = max(1, model.codebooks[i].num_symbols)
        uniq_syms, sym_counts = torch.unique(syms_gpu, return_counts=True)
        p_sym_map = dict(zip(uniq_syms.cpu().tolist(), (sym_counts.float() / total).cpu().tolist()))

        pair_keys = all_toks_gpu * K_i + syms_gpu
        uniq_pairs, pair_counts = torch.unique(pair_keys, return_counts=True)
        pair_toks = (uniq_pairs // K_i).cpu().tolist()
        pair_syms = (uniq_pairs % K_i).cpu().tolist()
        pair_probs = (pair_counts.float() / total).cpu().tolist()

        mi = 0.0
        for tok, sym, p_ts in zip(pair_toks, pair_syms, pair_probs):
            p_t = p_tok_map.get(tok, 1e-12)
            p_s = p_sym_map.get(sym, 1e-12)
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
    STEPS = None  # None = auto-size from corpus & VRAM (or set e.g. 6000)

    _log_path = out_path("neuro_symbolic_v17_4_training.log")
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    _log_file = open(_log_path, "w", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = _TeeStream(_orig_stdout, _log_file)
    sys.stderr = _TeeStream(_orig_stderr, _log_file)

    try:
        print("=" * 72)
        print(f"Neuro-Symbolic v18 run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output folder: {RUN_OUTPUT_DIR}")
        print(f"Terminal output is also being saved to: {_log_path}")
        print("=" * 72)

        try:
            trained_model, tokenizer, s_len = train_model(
                filepath=FILEPATH,
                tokenizer_mode=TOKENIZER_MODE,
                bpe_vocab_size=BPE_VOCAB_SIZE,
                steps=STEPS,
            )
        except KeyboardInterrupt:
            print("💾 Emergency checkpoint saved. Exiting cleanly.")
            sys.exit(0)

        evaluate_and_generate(trained_model, tokenizer, s_len)
        visualize_submodality_mind_reader(
            trained_model,
            tokenizer,
            device,
            save_path=out_path("stream_routing_heatmap.png"),
        )
        auto_label_buckets(trained_model, tokenizer, device, filepath=FILEPATH)
        symbol_mutual_information(trained_model, tokenizer, device, filepath=FILEPATH)

        input("\nScript complete. Press Enter to exit...")
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        _log_file.close()