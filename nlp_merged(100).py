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
  [⭐] Adaptive LR reduction (replaces cosine annealing)

Fixes applied (v3):[🔴] EMA entropy regularisation replaces raw batch entropy
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

Fixes applied (v6):
  [🔴] Orthogonality loss normalised by D²
         Without normalisation, ‖Wᵢ Wⱼᵀ‖²_F scales with D² and weight
         variance, silently inflating the penalty at larger d_model and
         overwhelming CE loss.  Dividing by D*D makes the magnitude
         invariant to embedding dimension.
  [🔴] torch.cdist replaces manual distance expansion
         The manual ‖a‖² + ‖b‖² − 2aᵀb expansion suffers catastrophic
         cancellation when ‖a‖ ≈ ‖b‖, causing precision drift in long
         training runs.  torch.cdist provides better numerical conditioning
         and dispatches to optimised GPU kernels automatically.
  [🔴] In-place EMA buffer update (.mul_().add_())
         Rebinding a registered buffer inside forward() with = breaks
         TorchScript compilation and DistributedDataParallel buffer sync.
         In-place ops update the tensor without rebinding the Python name.
  [🟠] Orthogonality loss moved before model(x)
         orth_loss depends only on projection weight matrices, not on batch
         data.  Computing it before the forward pass makes this explicit,
         keeps it out of the batch computation graph, and avoids any
         entanglement with VQ backward graph recomputation.

Fixes applied (v7):
  [🔴] commitment_cost 0.25 → 0.15
         With entropy and orthogonality losses also active, the VQ
         commitment cost at 0.25 overconstrained the representation space,
         causing gradient starvation.  0.15 reduces the per-codebook
         commitment pressure while still keeping quantisation stable.
  [🔴] Learned submodality gating replaces fixed mean aggregation
         A plain mean over the five quantised streams assumes equal
         informativeness for every token.  A learned softmax gate
         conditioned on the token embedding lets the model weight streams
         dynamically: e.g. up-weighting Brightness for vivid words,
         Perspective for spatial descriptions.
  [🔴] Feature-conditioned orthogonality (feature_orthogonality_loss)
         Weight-space orthogonality (Wᵢ Wⱼᵀ) can be zero while the actual
         features are still correlated through biases or non-linearities.
         The new method computes the cross-gram on the batch-projected
         features Fᵢᵀ Fⱼ / N, conditioning the penalty on the true
         representational geometry the network produces this step.
  [🟠] Curriculum sequence length
         seq_len grows logarithmically from seq_len_base=16 to
         seq_len_max=64 over training.  Short early contexts let the model
         build reliable local representations before integrating long-range
         dependencies — a standard curriculum practice for LMs.
  [🟠] Top-k masking on logits before softmax
         The old approach zeroed probabilities after softmax and
         renormalised, introducing a discontinuity and a numerical divide.
         Masking to −inf before softmax lets softmax renormalise in one
         pass, producing a clean probability distribution.

Fixes applied (v8):
  [🔴] vq_beta 0.25 → 0.40
         commitment_cost was lowered to 0.15 in v7 but vq_beta was not
         adjusted, leaving effective commitment pressure at 0.25×0.15=0.0375
         — far below the original 0.25×0.25=0.0625.  Setting vq_beta=0.40
         restores effective pressure to 0.40×0.15=0.06, close to baseline.
  [🔴] Scale-invariant entropy penalty: (H_ema/H_target − 1)²
         The old (H_target − H_ema)² had magnitude ∝ H_target², so larger
         codebooks (higher num_symbols) produced disproportionate penalties
         for the same fractional divergence.  The relative form normalises
         by H_target so the signal is always in [0,1] regardless of K.
  [🔴] L2-normalise features before cross-gram
         Without normalisation, growing embedding norms during training
         inflate the feature orthogonality penalty implicitly, making the
         effective λ non-stationary.  F.normalize(fi, dim=1) before the
         cross-gram makes the penalty purely geometry-based (cosine
         similarity) regardless of feature magnitude.
  [🟠] LR warmup (100 steps) + cuDNN determinism flags
         Without warmup, the full lr=3e-4 is applied on step 0 before VQ
         has formed any clusters, destabilising early quantisation.  A
         linear ramp over 100 steps lets the codebook initialise before
         gradients are large.  cudnn.deterministic=True added to set_seed.
  [🟢] Explicit dtype in register_buffer + temperature clamp
         ema_entropy buffer now explicitly typed float32 for dtype
         consistency.  generate() clamps temperature ≥ 0.5 to prevent
         logit explosion at very low values.
  [🟢] Corpus size guard
         _random_batch crashes with a cryptic error if the corpus is
         smaller than seq_len_max + 2.  A ValueError is now raised before
         training begins, with a clear diagnostic message.

Fixes applied (v9):
  [🔴] H_target = 0.9 · log(K) instead of log(K)
         Language symbol usage is naturally non-uniform.  Forcing full
         log(K) uniformity fights natural clustering via the entropy penalty.
         0.9 · log(K) allows ~10% headroom for frequency skew while still
         strongly penalising collapse.
  [🔴] Residual bypass around quantisation: Q(z) + α·z
         Hard VQ destroys continuous information not captured by the nearest
         codebook entry.  A learnable scalar α (init 0.1) lets the network
         preserve fine-grained signal that doesn't fit the discrete manifold.
         The residual starts near-zero so training begins with almost-pure
         quantisation and the bypass opens only as needed.
  [🔴] LayerNorm before each VQ codebook (pre_vq_norms)
         Unnormalised features cause some codebook entries to be permanently
         unreachable (too far from all inputs) while others are always
         selected.  A per-stream LayerNorm before each codebook stabilises
         the scale entering the distance computation.
  [🔴] Attention mask cache + dtype match in CausalTransformerDecoder
         Curriculum training generates a new seq_len each step; without
         caching, make_causal_mask() allocates a new tensor on every forward
         pass.  Masks are now cached by (seq_len, device, dtype).  The dtype
         parameter also ensures the mask matches the model dtype for
         mixed-precision correctness.
  [🟠] orth_weight capped at 0.8
         0.8 * sigmoid(...) keeps the orthogonality pressure bounded while
         still providing a meaningful push once streams start converging.
  [🟠] Validation uses curriculum seq_len
         Validating at seq_len_max while training at a shorter curriculum
         length creates a distribution mismatch.  Validation now uses the
         same seq_len as the current training step.

Fixes applied (v10):
  [🔴] residual_alpha clamped to [0, 0.5]
         An unconstrained α is an escape route: the model can raise it to
         satisfy CE loss while the codebook receives negligible gradient and
         its entries go undertrained ("escape-route collapse").  Clamping
         at 0.5 ensures the discrete component always contributes at least
         as much as the continuous bypass.  Zero new hyperparameters.
  [🟠] feature_orthogonality_loss skipped during inference
         The O(25·N·D) cross-gram computation is pure waste at inference
         time — its result is only used by orth_loss in the training loop.
         Guarded by self.training; a zero scalar is returned during eval
         so all call sites remain identical.

Fixes applied (v11):
  [🔴] Mean-center features before cross-gram
         Without centering, the cross-gram fi.t() @ fj includes the outer
         product of the stream means, so the penalty measures angular
         distance from the origin rather than true covariance.  Subtracting
         fi.mean(0) before L2-normalise makes the penalty enforce
         Cov(fᵢ, fⱼ) ≈ 0 — genuine decorrelation, not just angular bias.[🔴] Sigmoid reparameterisation of residual_alpha
         Hard clamping truncates gradients to zero at the boundary
         (∂L/∂a = 0 when a ≥ max), creating a dead zone in parameter space.
         α = 0.5·sigmoid(a) maps a ∈ ℝ → α ∈ (0, 0.5) with full gradient
         everywhere.  sigmoid(0) = 0.5 gives α_init ≈ 0.25 — moderate
         bypass from the start; the [0, 0.5] range is preserved.
  [🔴] Gate entropy penalty: γ · KL(gate || uniform)
         Without regularisation, softmax positive feedback can collapse the
         gate onto one dominant stream early in training, leaving the other
         four streams without gradient signal and their codebooks stagnant.
         KL(g || u) = log(5) − H(g) is zero iff the gate is uniform and
         grows as it concentrates.  γ=0.01 keeps the pressure light so the
         gate remains free to specialise, but not to collapse.
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

        # Project and reshape to[B, n_heads, T, d_head]
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
        n_heads = self.blocks[0].n_heads   # CausalDecoderBlock uses SDPA — no self.attn attribute
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
# CROSS-STREAM ATTENTION
# ==========================================================
# After VQ quantisation each stream holds an independent discrete
# symbol vector.  Streams are then combined with a gated weighted sum,
# but that sum has no way to model *relationships* between streams.
# For example, "fear" activates Valence (negative) AND Brightness (dark)
# together — a weighted sum treats those independently.
#
# CrossStreamAttention inserts one round of self-attention OVER the N stream
# vectors for each token independently, treating streams as a short sequence
# of length N rather than as a bag to be pooled.  Because N is tiny (7-12)
# this is negligible compute: one SDPA call on N=7 vectors vs T=1024 for the
# main decoder.
#
# Shape flow:
#   sym_stack  [B, T, N, D]
#   → reshape  [B*T, N, D]   (treat each token's N streams as a N-token sequence)
#   → SDPA (is_causal=False — streams are unordered)
#   → reshape  [B, T, N, D]
#   → gate + weighted sum (unchanged)

class CrossStreamAttention(nn.Module):
    """
    Single-layer self-attention over the N submodality stream vectors
    for each token position independently.

    Pre-LayerNorm residual block, identical pattern to CausalDecoderBlock
    but with is_causal=False (streams are unordered) and no FFN (cheap).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # n_heads must divide d_model; fall back to 1 if needed
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads //= 2
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.d_model  = d_model

        self.norm   = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # Zero-init output projection — makes the residual x + drop(o_proj(...))
        # a pure identity at step 0.  Without this, the randomly scaled attention
        # output adds large noise to the stream vectors and causes gradient explosion
        # on the first backward pass.  Same technique used by GPT-2 and nanoGPT.
        nn.init.zeros_(self.o_proj.weight)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [M, N, D]  where M = B*T, N = num_streams
        Returns: [M, N, D]
        """
        M, N, _ = x.shape
        normed   = self.norm(x)

        q = self.q_proj(normed).view(M, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(normed).view(M, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(normed).view(M, N, self.n_heads, self.d_head).transpose(1, 2)

        # is_causal=False — streams are an unordered set, not a sequence
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
    """
    Vector-quantisation codebook: maps a continuous embedding to the
    nearest entry in a learned discrete symbol table.

        - Improved initialisation   (1/√dim uniform)
        - Straight-through gradient (argmin is non-differentiable)
        - Entropy tracking          (signals codebook utilisation health)
        - Active-entry ratio        (collapse early-warning signal)
        - Dynamic commitment_cost   (raised when active% is low)
        - Dynamic ema_decay         (fast early, slow late)
        - grow_codebook()           (expand num_symbols at runtime)
    """

    def __init__(
        self,
        num_symbols:     int,
        embedding_dim:   int,
        commitment_cost: float = 0.25,
        ema_decay:       float = 0.2,   # starts fast; annealed to 0.05 during training
        topk:            int   = 3,     # Top-K symbol routing: K symbols blended per token
        topk_tau:        float = 0.1,   # softmax temperature for blending weights
                                        # Low tau (0.1) → sharp/hard routing (1 symbol dominates)
                                        # High tau (2.0+) → soft/uniform routing (all K equal)
                                        # Annealed cold→warm during training so model first learns
                                        # which symbols matter, then blends them smoothly.
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

        self.residual_alpha_raw = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12 → α≈6% bypass at init; forces model to earn the bypass
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
        Relies on ema_entropy: if a symbol is never touched its one-hot prob=0,
        but we track which are dead via an external active-ratio mask passed in.
        Called by training loop with a per-stream active mask.
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

        # Spherical VQ: project features and codebook onto the unit sphere before
        # computing distances.  Raw features exit LayerNorm with L2 norm ≈ √D ≈ 16;
        # codebook entries initialise with L2 norm ≈ 0.33.  With a 48× radius gap,
        # every centroid is ≈16.0 units away and the distances are functionally
        # identical — tau has no effect and TopK softmax collapses to [0.33,0.33,0.33].
        # Projecting both to the unit sphere constrains distances to [0, 2], making
        # tau=0.10 genuinely sharp and restoring meaningful Top-K routing.
        flat_sphere = F.normalize(flat, dim=1)
        cb_sphere   = F.normalize(self.codebook.weight, dim=1)
        distances   = torch.cdist(flat_sphere, cb_sphere, p=2)        # [BT, K_sym], strictly ∈ [0, 2]

        # ── Top-K Symbol Routing ──────────────────────────────────────────
        # Select the K nearest centroids per token and blend them via softmax
        # weights derived from their negative distances.  This gives:
        #   (1) Dramatically larger representational capacity:  C(K_sym, topk)^N_streams
        #   (2) Smoother gradients: multiple symbols receive signal per step
        #   (3) Better codebook utilisation: K entries updated per token not 1
        k_eff               = min(self.topk, self.num_symbols)
        topk_dists, topk_idx = distances.topk(k_eff, dim=1, largest=False)  # [BT, k_eff]
        # Divide distances by tau before softmax.
        # Small tau (cold) → weights concentrate on nearest → hard routing
        # Large tau (warm) → weights spread evenly → soft blending
        topk_weights        = F.softmax(-topk_dists / self.topk_tau, dim=1) #[BT, k_eff]

        # Weighted sum of top-K centroid vectors
        topk_vecs  = self.codebook(topk_idx)                                # [BT, k_eff, D]
        quantized  = (topk_weights.unsqueeze(-1) * topk_vecs).sum(dim=1)   # [BT, D]
        quantized  = quantized.view_as(features)                            #[B, T, D]

        # Commitment losses — encoder side: weighted distance to selected centroids
        e_loss  = (topk_weights.detach() * topk_dists).sum(dim=1).mean()
        q_loss  = F.mse_loss(quantized, features.detach())

        # Within-codebook diversity loss: push centroids apart from each other.
        # Sample a random subset of centroid pairs each step (O(sample²) not O(K²)).
        # Mean pairwise cosine similarity → penalise when centroids cluster together.
        # Weight is small (0.01) so it nudges without dominating the VQ signal.
        # IMPORTANT: use eps in normalize — zero-norm entries (post-revival) produce
        # NaN without it, which poisons the entire loss and causes training collapse.
        n_sample   = min(self.num_symbols, 64)
        sample_idx = torch.randperm(self.num_symbols, device=self.codebook.weight.device)[:n_sample]
        W_sample   = self.codebook.weight[sample_idx]
        W_norms    = W_sample.norm(dim=1, keepdim=True).clamp(min=1e-8)  # never divide by 0
        W_sample   = W_sample / W_norms                                   # safe normalise
        cos_sim    = torch.mm(W_sample, W_sample.t())                     #[n_sample, n_sample]
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
            # Near log(K)=1.10 → genuinely soft (all K symbols used equally).
            # Near 0 → effectively hard routing (1 symbol dominates).
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

    Submodalities (NLP / sensory representation theory):
        Brightness  — vividness / intensity of the concept
        Distance    — psychological proximity (near vs. far)
        Focus       — attentional salience / sharpness
        Color       — affective tone (warm vs. cool)
        Perspective — viewpoint / frame of reference
        Size        — magnitude / scale of the concept (vast vs. tiny)
        Position    — narrative / spatial anchoring (before/after, above/below)

    Orthogonality regularisation:
        Without a constraint the seven weight matrices can collapse into
        correlated latent directions, making the submodality split
        semantically redundant.  orthogonality_loss() penalises pairwise
        alignment between projection matrices:

            L_orth = λ · Σ_{i≠j} ‖ Wᵢ Wⱼᵀ ‖²_F

        The squared Frobenius norm of the cross-gram matrix is zero only
        when Wᵢ and Wⱼ span fully orthogonal subspaces.  λ = 1e-4 keeps
        the penalty small relative to CE loss while still providing a
        consistent gradient signal.
    """

    # Read-only reference — never mutate this directly.
    # Instance attribute `submodality_names` is initialised in __init__.
    BASE_SUBMODALITY_NAMES =["Brightness", "Distance", "Focus", "Color",
                               "Perspective", "Size", "Position"]

    def __init__(self, d_model: int, orthogonality_lambda: float = 0.01):
        # Raised from 1e-4 → 0.01.  The old value divided by D²=16384 which
        # made Orth ~1e-9 — genuinely invisible to the optimiser.  0.01 with
        # .mean() instead of .sum()/D² puts Orth in the 0.01–0.1 range where
        # it actually pushes streams apart.
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

        # Focus collapses toward Brightness repeatedly in training — they share
        # vividness/intensity semantics and the default random init gives them
        # ~0 cosine similarity which the optimizer immediately reduces.
        # Pre-orthogonalize Focus weights against Brightness at init so the
        # optimizer starts from a genuinely distinct direction and must work to
        # collapse them rather than drifting naturally.
        with torch.no_grad():
            W_b = self.extract_brightness.weight.data   # [D, D]
            W_f = self.extract_focus.weight.data        # [D, D]
            # Gram-Schmidt: remove the Brightness component from each Focus row
            for row in range(W_f.shape[0]):
                b = W_b[row]
                f = W_f[row]
                f = f - (f @ b / (b @ b + 1e-8)) * b
                W_f[row] = f
            # Re-normalise rows to preserve original weight magnitude
            W_f_norms = W_f.norm(dim=1, keepdim=True).clamp(min=1e-8)
            orig_norms = self.extract_brightness.weight.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.extract_focus.weight.data = W_f / W_f_norms * orig_norms
        self.orthogonality_lambda = orthogonality_lambda

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Weight-space orthogonality: penalises pairwise correlation between
        projection matrices.  Fast to compute (no batch data needed) but
        only approximates the true feature-space geometry — weights can be
        orthogonal while features are still correlated via bias terms.

        Used during initialisation / early training via the sigmoid schedule.
        Prefer feature_orthogonality_loss() for the data-conditioned version.
        """
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
        """
        Feature-space orthogonality: penalises pairwise correlation between
        the actual projected features produced on this batch, rather than
        between the raw weight matrices.

        This is mathematically stronger than weight-space orthogonality:
        weights can be orthogonal while features are still correlated through
        biases or upstream non-linearities.  Conditioning on the batch data
        distribution ensures the penalty acts on the geometry the network
        actually uses during training.

        The cross-gram matrix Fᵢᵀ Fⱼ / N has entries
            (1/N) Σ_n fᵢ(xₙ) · fⱼ(xₙ)
        which is zero only when the two feature streams are uncorrelated
        across the training distribution.

        Args:
            sub_features:[B, T, N, D] — output of VisualSubmodalityEncoder.forward()
                          N = NUM_SUBMODALITIES (starts at 7, grows during training)

        Returns: scalar tensor, differentiable w.r.t. projection weights
        """
        B, T, _, D = sub_features.shape
        N          = B * T                   # total token count in the batch
        loss       = torch.tensor(0.0, device=sub_features.device)

        for i in range(len(self.submodality_names)):
            fi = sub_features[:, :, i, :].reshape(N, D)      #[N, D]
            # Mean-center then L2-normalise.  Without centering, any shared
            # mean bias leaks into the cross-gram: fi.t() @ fj picks up the
            # outer product of the means, so the penalty measures angular
            # distance from the origin rather than true covariance.  Centering
            # first makes the cross-gram estimate the sample cross-covariance,
            # so the penalty is zero only when the two streams are truly
            # uncorrelated (Cov(fᵢ, fⱼ) ≈ 0) across the training batch.
            fi = fi - fi.mean(0, keepdim=True)                # remove mean bias
            fi = F.normalize(fi, dim=1)                       # unit norm → cosine
            for j in range(i + 1, len(self.submodality_names)):
                fj = sub_features[:, :, j, :].reshape(N, D)  # [N, D]
                fj = fj - fj.mean(0, keepdim=True)
                fj = F.normalize(fj, dim=1)
                # Cross-gram: [D, D]; .mean() normalises by D² implicitly and
                # keeps the penalty in a stable range regardless of dimension.
                # Previously used .sum()/(D*D) which was identical mathematically
                # but confusing — .mean() is clearer and equally correct.
                cross_gram = (fi.t() @ fj) / N               #[D, D]
                loss = loss + cross_gram.pow(2).sum()        # Frobenius energy: sum not mean
                # .mean() divides by D² (65536 for D=256), pushing already-small
                # correlations into 1e-8 range where the optimizer cannot see them.
                # .sum() gives the true Frobenius norm energy — naturally at the
                # right scale alongside CE/VQ without any compensating ×100 hack.

        return self.orthogonality_lambda * loss  # ×100 removed — Frobenius sum is naturally visible

    def add_stream(self, name: str, d_model: int, device: torch.device) -> nn.Linear:
        """
        Dynamically add a new submodality projection at runtime.
        The new Linear is registered as an attribute so the optimizer
        can be updated with its parameters.  Returns the new module.
        """
        attr = f"extract_{name.lower()}"
        if hasattr(self, attr):
            raise ValueError(f"Stream '{name}' already exists.")
        new_lin = nn.Linear(d_model, d_model).to(device)
        setattr(self, attr, new_lin)
        self.submodality_names.append(name)
        return new_lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x:[B, T, D]  →  [B, T, N, D]  where N = NUM_SUBMODALITIES"""
        streams =[getattr(self, f"extract_{n.lower()}")(x)
                   for n in self.submodality_names]
        return torch.stack(streams, dim=2)


# ==========================================================
# MAIN MODEL
# ==========================================================

class SubmodalitySymbolicLanguageModel(nn.Module):
    """
    Neuro-Symbolic Language Model — decoder-only architecture:

        Token IDs
            → Embedding
            → N × Named Submodality Projection  (starts at 7, grows to 12)
            → N × Independent VQ Codebook       (discrete symbols)
            → Learned gated aggregation          (replaces naïve mean)
            → Positional Encoding
            → CausalTransformerDecoder
            → Linear → logits over vocab

    Learned gating:
        The seven quantised streams are aggregated via a softmax-gated
        weighted sum rather than a plain mean.  The gate is conditioned on
        the token embedding, so the model can learn that Brightness is more
        informative for vivid emotional words, Size for magnitude words,
        and Position for narrative sequencing words.
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
        # PyTorch's MultiheadAttention.  Each head gets d_model/n_heads dims;
        # that must be a whole number.
        if d_model % n_heads != 0:
            valid =[h for h in range(1, d_model + 1) if d_model % h == 0]
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).\n"
                f"Valid n_heads for d_model={d_model}: {valid}"
            )

        self.d_model = d_model

        self.embedding   = nn.Embedding(vocab_size, d_model)
        # Default Normal(0,1) init causes logit std ≈ √D when weights are tied
        # (F.linear output std ≈ √D × input std).  For D=128 that's std≈11,
        # making softmax near-deterministic at init and CE >> log(vocab).
        # Rescale to std = 1/√D, matching codebook init and GPT-2 practice.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        self.sub_encoder = VisualSubmodalityEncoder(d_model)

        # One independent codebook per submodality (prevents collapse)
        self.codebooks = nn.ModuleList([
            NeuroSymbolicCodebook(num_symbols, d_model)
            for _ in range(self.NUM_SUBMODALITIES)
        ])

        # Per-stream LayerNorm applied immediately before each VQ codebook.
        # VQ distance computation is sensitive to feature scale: unnormalised
        # features cause some codebook entries to be permanently unreachable
        # (too far from all inputs) while others are always selected.
        # LayerNorm stabilises the scale entering each codebook independently.
        self.pre_vq_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(self.NUM_SUBMODALITIES)
        ])

        self.pos_encoder = PositionalEncoding(d_model)

        # Learned gate: produces per-token softmax weights over the 5
        # submodality streams.  Conditioned on the token embedding so the
        # model can learn that different submodalities are more informative
        # for different semantic contexts.  Output dim = NUM_SUBMODALITIES.
        self.submodality_gate = nn.Linear(d_model, self.NUM_SUBMODALITIES)

        # Pre-VQ cross-stream attention — streams inform each other BEFORE
        # committing to discrete codes.  This mirrors early cortical cross-talk
        # (thalamus, superior colliculus) where depth influences colour perception,
        # motion influences focus, etc.  Streams adjust their continuous features
        # based on their neighbours BEFORE the hard quantisation decision is made,
        # producing richer, more context-aware discrete symbols.
        # Placed: sub_encoder output → pre_vq_cross_stream_attn → pre_vq_norms → VQ
        self.pre_vq_cross_stream_attn = CrossStreamAttention(
            d_model, n_heads=max(1, n_heads // 2), dropout=dropout
        )

        # Post-VQ cross-stream attention — lets quantised symbols interact before
        # gated aggregation.  Placed after VQ, before the gate sum.
        # Placed: VQ output → cross_stream_attn → gate → composite
        self.cross_stream_attn = CrossStreamAttention(d_model, n_heads=max(1, n_heads // 2), dropout=dropout)

        # Stream dropout probability (applied during training only).
        # Forces the model to learn redundant representations so no single stream
        # is a hard dependency.  Each stream is independently zeroed with this prob.
        self.stream_dropout_p: float = 0.15

        # Composite residual bypass: adds a fraction of the raw token embedding
        # directly to the composite symbolic vector before the decoder.
        # α = 0.5 * sigmoid(raw) ∈ (0, 0.5); prevents catastrophic symbolic bottleneck
        # by preserving fine-grained embedding signal not captured by any codebook.
        self.composite_residual_raw = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ≈ 0.27 → α ≈ 0.13

        # ── Meta-codebook (level-2 VQ) ────────────────────────────────────
        # Quantises the cross-stream composite vector to a single meta-symbol.
        # Architecture:  composite → meta_norm → meta_codebook → residual-add
        #
        # This is NOT stacked VQ (which would stack two straight-through
        # estimators, turning gradients to noise).  Instead the meta-codebook
        # operates on the already-formed composite — a gated cross-stream
        # embedding that is stable and semantically rich.  Only one ST
        # estimator is in the meta path; level-1 gradients are unaffected.
        #
        # meta_residual_raw: learned scale for the additive correction.
        #   alpha = 0.5 * sigmoid(raw) ∈ (0, 0.5)
        #   Init at -2.0 → alpha ≈ 0.06 (very gentle, grows as the meta-
        #   codebook proves useful).
        self.meta_norm         = nn.LayerNorm(d_model)
        self.meta_codebook     = NeuroSymbolicCodebook(
            num_symbols     = num_symbols,
            embedding_dim   = d_model,
            commitment_cost = 0.10,   # softer start: composite already stable
        )
        self.meta_residual_raw = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12 → α≈6%

        # FIX 1 (RED): custom causal decoder, not TransformerEncoder
        self.decoder = CausalTransformerDecoder(d_model, n_heads, num_layers, dropout)

        # Weight-tied output projection — shares the embedding matrix.
        # Enforces consistency between the input token space and output prediction
        # space and halves the vocab-sized parameters (~12M for tiktoken vocab).
        # Applied as F.linear(decoder_out, self.embedding.weight) in forward().\
        # No separate output_head parameter is needed.

        # Stream activity tracking — inactive streams get ~zero gate weight
        # via a mask applied after softmax.  Inactive streams still run through
        # VQ (avoids tensor surgery) but contribute nothing to composite.\
        self.stream_active: list[bool] = [True] * self.NUM_SUBMODALITIES
        self._update_stream_mask()

    def _update_stream_mask(self) -> None:
        """Rebuild the float mask tensor from stream_active list."""
        # Use the model's current device — avoids CPU/GPU mismatch when called
        # after the model has been moved to CUDA (e.g. during stream addition).
        dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        mask = torch.tensor([1.0 if a else 0.0 for a in self.stream_active],
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

        Steps:
          1. Add projection linear to sub_encoder
          2. Add new VQ codebook
          3. Add new pre-VQ LayerNorm
          4. Grow gate linear (d_model → N+1), preserving existing weights
          5. Add new params to optimizer as a new param group
          6. Update NUM_SUBMODALITIES and stream_active

        Returns the list of new parameters so the caller can register a
        growth LR warmup event if desired.
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
            # Gentle positive bias — gives new stream a slight head start without
            # dominating. 0.5 gives ~1.6x uniform weight after softmax, not ~50%.
            # log(N-1) was tested and caused Texture to monopolise 34% of gate.
            new_gate.bias.data[self.NUM_SUBMODALITIES] = 0.5
        self.submodality_gate = new_gate

        # 5. add new params to optimizer
        new_params = (
            list(new_lin.parameters())
            + list(new_cb.parameters())
            + list(new_norm.parameters())
            + list(new_gate.parameters())   # full gate replaces old, add all
        )
        # Remove old gate params from existing groups to avoid duplicates
        old_gate_params = list(old_gate.parameters())
        old_gate_ids = {id(p) for p in old_gate_params}
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"]
                               if id(p) not in old_gate_ids]
        # Free Adam momentum/variance buffers — without this the tensors stay
        # in optimizer.state forever even though they receive no more gradients.
        for p in old_gate_params:
            optimizer.state.pop(p, None)
        # Split new params into decay / no-decay (codebook, biases, norms)
        # to match the same weight-decay exclusion as the original optimizer groups.
        _no_decay_keys = ('bias', 'norm', 'codebook', 'residual_alpha_raw', 'ema_entropy', 'meta_residual_raw')
        _new_decay   = [p for n, p in (list(new_lin.named_parameters())
                                       + list(new_gate.named_parameters()))
                        if not any(k in n for k in _no_decay_keys)]
        _new_nodecay = new_params  # everything else (codebook, norms, biases) → no decay
        _new_nodecay = [p for p in new_params if id(p) not in {id(q) for q in _new_decay}]
        _boost_lr = boost_lr if boost_lr is not None else lr
        if _new_decay:
            optimizer.add_param_group({
                "params": _new_decay, "lr": _boost_lr, "weight_decay": weight_decay,
            })
        if _new_nodecay:
            optimizer.add_param_group({
                "params": _new_nodecay, "lr": _boost_lr, "weight_decay": 0.0,
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
        Params remain in model/optimizer (avoids optimizer surgery).
        The stream can be inspected post-hoc but has no effect on output.
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
        Break codebook symmetry between streams i and j by identifying the
        most-similar centroid pairs across the two codebooks and perturbing
        the stream-j copies with random noise.  This forces the pair to
        explore different regions without discarding what stream i learned.

        fraction: fraction of centroids to perturb (those with highest cosine sim)
        Returns number of centroids perturbed.
        """
        with torch.no_grad():
            Wi = F.normalize(self.codebooks[i].codebook.weight.data, dim=1)  #[Ki, D]
            Wj = self.codebooks[j].codebook.weight.data                       #[Kj, D]
            Wj_norm = F.normalize(Wj, dim=1)
            # Cosine similarity matrix [Ki, Kj]
            sim = torch.mm(Wi, Wj_norm.t())
            # For each centroid in j, find its highest cosine sim with any centroid in i
            max_sim_j, _ = sim.max(dim=0)                                     # [Kj]
            # Perturb the top `fraction` most-similar ones
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
        """
        Args:
            x: [B, T] token IDs

        Returns:
            logits:          [B, T, vocab_size]
            symbolic_indices:[B, T, N, K] — top-K symbol indices per stream
            total_vq_loss:   scalar — VQ commitment loss
            feat_orth_loss:  scalar — feature-space orthogonality penalty (unscaled)
            gate_kl:         scalar — KL(gate || uniform), unscaled
            sym_decor_loss:  scalar — mean JS divergence across stream pairs (logging only)
            gate_weights:    [B, T, N] — softmax gate weights
            mean_ema_H:      float  — mean smoothed entropy across streams (for logging)
            per_stream_active: list[float] — active-symbol ratio per stream
            pair_js_vals:    dict{(i,j): float} — per-pair JS divergence (training only)
            pair_grad_terms: dict{(i,j): tensor} — per-pair hinge gradient (training only)
            sub_features:[B, T, N, D] detached (training only, else None)
            per_stream_topk_H: list[float] — Top-K blending softness per stream
        """
        embed        = self.embedding(x)          # [B, T, D]
        sub_features = self.sub_encoder(embed)    # [B, T, N, D]

        # ── Pre-VQ Cross-Stream Attention ─────────────────────────────────
        # Streams inform each other BEFORE committing to discrete symbols.
        # Each token's N stream vectors attend to each other so e.g. the
        # Distance stream can modulate the Focus stream before both are
        # independently quantised.  This makes the discrete codes richer
        # because the quantisation decision is context-aware across streams.
        # Reshape [B, T, N, D] → [B*T, N, D] → attend → reshape back.
        B_f, T_f, N_f, D_f = sub_features.shape
        sub_features = self.pre_vq_cross_stream_attn(
            sub_features.reshape(B_f * T_f, N_f, D_f)
        ).reshape(B_f, T_f, N_f, D_f)

        symbolic_vectors =[]
        symbolic_indices =[]
        total_vq_loss    = 0.0
        total_ema_H      = 0.0
        per_stream_active =[]  # per-stream active-entry fraction
        per_stream_topk_H =[]  # per-stream Top-K blending softness

        for i in range(self.NUM_SUBMODALITIES):
            # Normalise feature scale before quantisation — prevents codebook
            # entries from being permanently unreachable due to scale mismatch.
            normed_stream = self.pre_vq_norms[i](sub_features[:, :, i, :])
            quantized, indices, vq_loss, ema_H, active_ratio, topk_H = self.codebooks[i](
                normed_stream                     # [B, T, D]  (scale-stabilised)
            )
            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)
            total_vq_loss  += vq_loss / self.NUM_SUBMODALITIES  # normalise: keep VQ proportional to CE regardless of N streams
            total_ema_H    += ema_H
            per_stream_active.append(active_ratio)
            per_stream_topk_H.append(topk_H)

        # Learned gating: softmax weights over the N submodality streams,
        # conditioned on the token embedding.  [B, T, N] →[B, T, N, 1]
        # so it broadcasts cleanly over the D dimension.
        gate_logits  = self.submodality_gate(embed)                     # [B, T, N]
        # Psychology override — only active when set externally (e.g. chat script).
        # Has no effect during normal training since gate_bias_override is never set.
        if getattr(self, "gate_bias_override", None) is not None:
            gate_logits = gate_logits + self.gate_bias_override
        gate_weights = F.softmax(gate_logits, dim=-1)                  # [B, T, N]

        # Apply stream activity mask: inactive streams → zero, then renormalise
        # so active streams still sum to 1.  This happens after softmax so
        # the gate gradient still flows to active streams normally.
        if not all(self.stream_active):
            gate_weights = gate_weights * self.stream_mask          # zero inactive
            gate_weights = gate_weights / (gate_weights.sum(-1, keepdim=True) + 1e-10)

        sym_stack = torch.stack(symbolic_vectors, dim=2)              # [B, T, N, D]

        # ── Stream Dropout (training only) ───────────────────────────────
        # Randomly zero out entire streams with prob stream_dropout_p.
        # Forces each stream to learn representations that are useful even
        # when some streams are absent — prevents over-specialisation collapse.
        # At least 1 stream always survives to keep the forward pass valid.
        n_active_for_kl = self.NUM_SUBMODALITIES  # used for gate KL target below
        if self.training and self.stream_dropout_p > 0.0:
            B_sz = sym_stack.shape[0]
            # Sample one dropout mask per batch item: [B, 1, N, 1] → broadcasts over T, D
            keep_prob = 1.0 - self.stream_dropout_p
            drop_mask = torch.bernoulli(
                torch.full((B_sz, 1, self.NUM_SUBMODALITIES, 1), keep_prob, device=sym_stack.device)
            )
            # Guarantee at least one stream survives per sample
            # (if all dropped, restore a random one)
            all_dead = (drop_mask.squeeze(-1).squeeze(1).sum(-1) < 1)  # [B]
            if all_dead.any():
                rescue = torch.randint(self.NUM_SUBMODALITIES, (all_dead.sum().item(),),
                                       device=sym_stack.device)
                dead_idx = all_dead.nonzero(as_tuple=True)[0]
                drop_mask[dead_idx, 0, rescue, 0] = 1.0
            sym_stack    = sym_stack * drop_mask  # zero dropped streams
            gate_weights = gate_weights * drop_mask.squeeze(-1)   # [B, T, N]
            gate_weights = gate_weights / (gate_weights.sum(-1, keepdim=True) + 1e-10)
            # KL target: uniform over the mean number of surviving streams
            n_active_for_kl = max(1, round(drop_mask.squeeze().mean(0).sum().item()))

        # ── Cross-Stream Attention ────────────────────────────────────────
        # Let streams interact before aggregation so the model can learn
        # relationships like (Valence ↔ Brightness) or (Distance ↔ Position).
        # Reshape to [B*T, N, D], attend, reshape back.
        B_, T_, N_, D_ = sym_stack.shape
        sym_stack_csa = self.cross_stream_attn(
            sym_stack.reshape(B_ * T_, N_, D_)
        ).reshape(B_, T_, N_, D_)

        # Residual: blend CSA-enriched streams with original (safety valve for early training)
        sym_stack = sym_stack_csa  # use CSA output

        composite  = (sym_stack * gate_weights.unsqueeze(-1)).sum(dim=2)  # [B, T, D]

        # ── Composite Residual Bypass ─────────────────────────────────────
        # Add a fraction of the raw token embedding directly to the composite.
        # Preserves fine-grained token information not captured by any codebook
        # — prevents catastrophic information loss at the symbolic bottleneck.
        # α = 0.5 * sigmoid(raw) ∈ (0, 0.5), so the discrete path always
        # contributes at least as much as the bypass.
        composite_alpha = 0.5 * torch.sigmoid(self.composite_residual_raw)
        composite = composite + composite_alpha * embed

        # ── Meta-codebook (level-2 VQ) ────────────────────────────────────
        # Normalise composite → quantise to meta-symbol → residual-add back.
        # The straight-through estimator here is independent of the N level-1
        # ST estimators — gradients flow cleanly to sub_encoder + codebooks
        # through the composite path, and to meta_codebook through the meta path.
        meta_in = self.meta_norm(composite)
        (meta_quantized, _meta_idx, meta_vq_loss,
         meta_ema_H, meta_active_ratio, meta_topk_H) = self.meta_codebook(meta_in)
        meta_alpha  = 0.5 * torch.sigmoid(self.meta_residual_raw)
        composite   = composite + meta_alpha * meta_quantized

        # Gate entropy penalty: KL(gate || uniform) per token, averaged over
        # the batch.  Without this, softmax positive feedback can cause the
        # gate to collapse onto one dominant stream early in training —
        # gradients to the other four streams decay, their codebooks stagnate,
        # and the multi-stream structure becomes effectively single-stream.\
        # KL(g || u) = log(N) + Σᵢ gᵢ log gᵢ, minimised when g is uniform.
        # When stream dropout is active, use the actual mean surviving stream count
        # so the KL target isn't artificially inflated for a reduced stream set.
        log_uniform  = math.log(max(1, n_active_for_kl))
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(-1).mean()
        gate_kl      = torch.clamp(log_uniform - gate_entropy, min=0.0)  # clamp: dropout math can give tiny negatives

        # Feature-conditioned orthogonality — only needed during training.
        # At inference this O(25·N·D) computation is pure waste: the result
        # goes into orth_loss which is never used outside the training loop.
        # self.training is False during model.eval() and generate() calls.
        if self.training:
            feat_orth_loss = self.sub_encoder.feature_orthogonality_loss(sub_features)

            # Cross-stream symbol decorrelation using HARD VQ assignments.
            #
            # Previous approach: soft distances with tau=sqrt(D).  Problem: with
            # 189 symbols and tau=16, softmax(-d/tau) is nearly uniform for every
            # stream → all JS divergences = 0 → penalty = log(2) = 0.693, stuck
            # at ceiling from step 0.  The gradient existed but the signal was
            # always identical regardless of what the model actually learned.
            #
            # Current approach: empirical usage histograms from the hard argmin
            # indices already computed by the VQ forward pass.  These are exact,
            # free (no extra distance computation), and produce real divergence
            # whenever streams make genuinely different routing decisions.
            #
            # Gradient path: sym_decor_loss flows back through pre_vq_norms and
            # sub_encoder projections via a soft re-parameterisation: we use
            # straight-through — the loss is computed on detached indices but
            # we add a soft auxiliary term that carries gradients.
            #
            # Algorithm:
            #   For each stream i:
            #     hist_i = fraction of batch tokens routed to each symbol  [K_i]
            #   Pad all hists to max_K so JS is well-defined across streams.
            #   penalty_ij = log(2) - JS(hist_i, hist_j)  ∈[0, log(2)]
            #   sym_decor_loss = mean over all pairs (normalised by n_pairs)
            #
            # Gradient re-attachment: because one_hot(argmin) is not
            # differentiable, we attach a soft hinge that pulls each stream's
            # mean distance to its own nearest centroid DOWN relative to
            # cross-stream centroids.  This gives a gradient that actually
            # pushes streams toward different codebook regions.
            with torch.no_grad():
                hists =[]
                for i in range(self.NUM_SUBMODALITIES):
                    idx_i = symbolic_indices[i].reshape(-1)          # [B*T*K] — all K symbols per token
                    K_i   = self.codebooks[i].num_symbols
                    h_i   = F.one_hot(idx_i, K_i).float().mean(0)   # [K_i]  empirical freq
                    hists.append(h_i)
                max_K = max(h.shape[0] for h in hists)
                # Pad shorter histograms with zeros so JS is defined across streams
                padded = [F.pad(h, (0, max_K - h.shape[0])) + 1e-10 for h in hists]
                padded =[p / p.sum() for p in padded]   # renormalise after padding

            # JS divergence on detached histograms — per-pair values returned
            # so the training loop can apply independent per-pair lambdas.
            pair_js_vals  = {}   # {(i,j): float} detached, for lambda tracking
            for i in range(self.NUM_SUBMODALITIES):
                for j in range(i + 1, self.NUM_SUBMODALITIES):
                    p, q = padded[i], padded[j]
                    m    = 0.5 * (p + q)
                    js   = 0.5 * ((p * torch.log(p / m)).sum()
                                + (q * torch.log(q / m)).sum())
                    pair_js_vals[(i, j)] = math.log(2) - js.item()

            # Batched pairwise hinge — replaces O(N²) separate cdist calls
            # with two vectorised ops regardless of stream count.
            #
            # Step 1: normalise each stream's features once
            N_s = self.NUM_SUBMODALITIES
            f_streams = [
                self.pre_vq_norms[i](sub_features[:, :, i, :]).reshape(-1, self.d_model)
                for i in range(N_s)
            ]   # list of [B*T, D]

            # Step 2: per-stream d_own — distance from each token to its
            # nearest own-codebook centroid.  Still one cdist per stream but
            # only N calls total (was already N, unchanged).
            d_owns = [
                torch.cdist(f_streams[i], self.codebooks[i].codebook.weight, p=2)
                .min(dim=1).values.mean()
                for i in range(N_s)
            ]

            # Step 3: cross-stream distances — N loops, not N² loops.
            # f_stack:[N, B*T, D]
            f_stack = torch.stack(f_streams)
            f_sq    = (f_stack ** 2).sum(-1)          # [N, B*T]

            # Build cross_mean[i, j] = mean min-dist from stream-i tokens to codebook-j.
            # IMPORTANT: must use torch.stack() not in-place assignment into zeros.
            # In-place assignment (cross_mean[:, j] = ...) strips grad_fn, so the
            # hinge gradient would never flow back through cross_mean to f_stack —
            # streams would only be pulled toward own centroids, never pushed away
            # from each other's centroids.  torch.stack() preserves the graph.
            cross_cols =[]
            for j in range(N_s):
                c_j    = self.codebooks[j].codebook.weight          # [K_j, D]
                # dist²:[N, B*T, K_j]
                dist2_j = (f_sq[:, :, None]
                           + (c_j ** 2).sum(-1)[None, None, :]
                           - 2 * torch.einsum('nbd,kd->nbk', f_stack, c_j))
                dist2_j = dist2_j.clamp(min=1e-8).sqrt()           #[N, B*T, K_j] — clamp>0 avoids ∇√0=∞ NaN
                cross_cols.append(dist2_j.min(dim=2).values.mean(dim=1))  # [N]
            cross_mean = torch.stack(cross_cols, dim=1)             # [N, N] with grad

            pair_grad_terms = {}
            for i in range(N_s):
                for j in range(i + 1, N_s):
                    pair_grad_terms[(i, j)] = (
                        F.relu(d_owns[i] - cross_mean[i, j] + 1.0)
                        + F.relu(d_owns[j] - cross_mean[j, i] + 1.0)
                    ) * 0.5

            # sym_decor_loss = mean JS (logging scalar only, no grad).
            # The backward signal is built in the training loop per pair.
            n_pairs = max(1, len(pair_js_vals))
            sym_decor_loss = torch.tensor(
                sum(pair_js_vals.values()) / n_pairs, device=x.device)
        else:
            feat_orth_loss = sub_features.new_zeros(1).squeeze()  # zero scalar
            sym_decor_loss = sub_features.new_zeros(1).squeeze()

        composite = self.pos_encoder(composite)
        decoder_out = self.decoder(composite)      # [B, T, D]
        logits      = F.linear(decoder_out, self.embedding.weight)  # tied weights [B, T, vocab]

        return (
            logits,
            torch.stack(symbolic_indices, dim=2),          # [B, T, N]
            total_vq_loss,
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
            meta_vq_loss,       # meta-codebook commitment + quantisation loss
            meta_ema_H,         # meta codebook entropy (EMA)
            meta_active_ratio,  # fraction of meta symbols used this batch
            meta_topk_H,        # Top-K blending softness for meta codebook
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
        """
        Autoregressively extend `input_ids` by `max_new_tokens` tokens.
        Uses temperature scaling + top-k + nucleus (top-p) sampling,
        with a repetition penalty on recently generated tokens.
        """
        self.eval()
        # Clamp temperature to prevent logit explosion: at very low values
        # logits / temperature → ±∞, causing softmax to produce a one-hot
        # distribution or NaN.  0.5 is a safe lower bound for generation.
        temperature = max(0.5, temperature)
        for _ in range(max_new_tokens):
            context = input_ids[:, -seq_len:]
            with torch.no_grad():
                logits, *_ = self(context)
            logits = logits[:, -1, :] / temperature

            # Repetition penalty: subtract a flat penalty for recently seen tokens.
            # Flat subtraction (not division) works cleanly for both positive and
            # negative logits — division can amplify negative logits perversely.
            if repetition_penalty > 1.0:
                recent = set(input_ids[0, -64:].tolist())   # look back 64 tokens
                penalty = repetition_penalty * 2.0
                for tok in recent:
                    logits[0, tok] = logits[0, tok] - penalty

            # Top-k: mask everything below the k-th highest logit.
            if top_k and top_k > 0:
                k            = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                min_val      = topk_vals[:, -1].unsqueeze(1)
                logits       = logits.masked_fill(logits < min_val, float("-inf"))

            # Nucleus (top-p): keep the smallest set of tokens whose cumulative
            # probability exceeds top_p, discard the long tail.
            if top_p < 1.0:
                probs_sorted, sorted_idx = torch.sort(F.softmax(logits, dim=-1),
                                                      descending=True)
                cumulative = torch.cumsum(probs_sorted, dim=-1)
                # Remove tokens once cumulative prob exceeds top_p (shift by 1 to
                # keep the token that first crosses the threshold).
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
    """
    Simple word-level tokenizer built from the training corpus.

    Splits on whitespace while preserving punctuation as separate tokens
    (e.g. "don't" →["don", "'", "t"]).  Vocabulary is derived entirely
    from the corpus so vocab_size is proportional to corpus diversity —
    typically 5k–20k for novel-length text vs GPT-2's fixed 50k.

    Tradeoffs vs subword:
      + Every token is a whole readable word → heatmap words map 1-to-1
      + Smaller vocab → codebooks cover more of the space per symbol
      − Out-of-vocab words on new data become <UNK>
      − Morphological variants (run/runs/running) are separate entries
    """

    PAD   = "<PAD>"
    UNK   = "<UNK>"
    BOS   = "<BOS>"
    EOS   = "<EOS>"

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._built = False

    # ── tokenisation ──────────────────────────────────────────────────

    @staticmethod
    def _split(text: str) -> list[str]:
        """Split text into word + punctuation tokens."""
        # Keep contractions together, split on whitespace and punctuation
        return _re.findall(r"\w+|[^\w\s]", text, _re.UNICODE)

    # ── vocabulary ────────────────────────────────────────────────────

    def build_vocab(self, text: str, min_freq: int = 1) -> None:
        """Build vocabulary from corpus text."""
        from collections import Counter
        tokens  = self._split(text)
        counts  = Counter(tokens)
        specials = [self.PAD, self.UNK, self.BOS, self.EOS]
        vocab   = specials +[t for t, c in counts.most_common() if c >= min_freq]
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self._built   = True

    # ── encode / decode ───────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        unk_id = self.token2id[self.UNK]
        return[self.token2id.get(t, unk_id) for t in self._split(text)]

    def decode(self, ids: list[int]) -> str:
        tokens =[self.id2token.get(i, self.UNK) for i in ids]
        # Re-join: attach punctuation to the preceding word without a space
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
    """
    BPE tokenizer backed by the HuggingFace `tokenizers` C++ library.
    Trains in seconds even on large corpora — typically 100× faster than
    the pure Python fallback.

    Same public interface as BPETokenizer (.encode / .decode / .n_vocab)
    so the rest of the codebase is unaffected regardless of which backend
    is active.

    Requires:  pip install tokenizers
    """

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
            special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS],
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
    """
    Byte-Pair Encoding tokenizer — pure Python fallback.
    Used automatically when the `tokenizers` package is not installed.

    Starts with a character-level vocabulary and iteratively merges the
    most frequent adjacent pair until `vocab_size` is reached.  This
    produces subword tokens that balance coverage (rare words split into
    known pieces) and compactness (common words stay whole).

    Tradeoffs vs word-level:
      + No <UNK> for in-distribution text — every character is representable
      + Better compression of common words and morphemes
      − Tokens are subwords, not whole words → heatmap words may split
      − Training is O(vocab_size × corpus_length) — slow on large corpora
        (use FastBPETokenizer / install `tokenizers` to avoid this)

    Args:
        vocab_size: target vocabulary size including base characters
                    and special tokens.  Typical: 1000–8000.
    """

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

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _word_freqs(text: str) -> dict[tuple[str, ...], int]:
        """Split corpus into words and count frequencies."""
        from collections import Counter
        words = _re.findall(r"\S+", text)
        counts: Counter = Counter(words)
        # Represent each word as a tuple of characters + EOW marker
        return {tuple(list(w) + ["</w>"]): c for w, c in counts.items()}

    @staticmethod
    def _get_pairs(vocab: dict[tuple, int]) -> dict[tuple[str, str], int]:
        """Count all adjacent symbol pairs across the vocabulary."""
        pairs: dict[tuple[str, str], int] = {}
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: tuple[str, str], vocab: dict) -> dict:
        """Merge all occurrences of `pair` in vocabulary."""
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

    # ── training ──────────────────────────────────────────────────────

    def train(self, text: str) -> None:
        """Learn BPE merges from corpus text."""
        specials   =[self.PAD, self.UNK, self.BOS, self.EOS]
        word_freqs = self._word_freqs(text)

        # Base character vocabulary (all unique chars + EOW)
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

    # ── encode / decode ───────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply learned BPE merges to a single word."""
        symbols = list(word) +[self.EOW]
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
        # Merge tokens back to text: EOW marks word boundaries
        text   = "".join(tokens)
        text   = text.replace(self.EOW, " ")
        return text.strip()

    @property
    def n_vocab(self) -> int:
        return len(self.token2id)


def _make_bpe_tokenizer(vocab_size: int):
    """
    Return the fastest available BPE tokenizer for this environment.

    Tries to import the HuggingFace `tokenizers` C++ library first.
    If it is installed, returns a FastBPETokenizer (trains in seconds).
    If not, falls back to pure Python BPETokenizer with a one-time
    install hint, and keeps running without interruption.
    """
    try:
        import tokenizers  # noqa: F401 — just checking availability
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
# DATA LOADING  (with 90/10 train/val split)
# ==========================================================

def load_data(filepath: str = "data.txt",
              tokenizer_mode: str = "tiktoken",
              bpe_vocab_size: int | None = None):
    """
    Tokenise a plain-text corpus.  Three modes are supported:

        "tiktoken"  — GPT-2 BPE via tiktoken (default, 50,257-token vocab)
        "word"      — word-level tokenizer built from the corpus (~5–20k vocab)
        "bpe"       — BPE tokenizer trained from scratch on the corpus

    FIX 3b — applies a 90/10 train/val split when corpus > 50 k tokens.

    Args:
        filepath:       path to plain-text corpus
        tokenizer_mode: "tiktoken" | "word" | "bpe"
        bpe_vocab_size: target BPE vocab size, or None (default) to
                        auto-calculate from corpus.  Auto formula:
                        max(1000, unique_words × 1.5) — no upper cap.
                        Size is fully driven by corpus vocabulary size.

    Returns:
        train_data:  LongTensor
        val_data:    LongTensor | None
        tokenizer:   tokenizer object with .encode() / .decode() / .n_vocab
        vocab_size:  int
    """
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

    # ── build tokenizer ───────────────────────────────────────────────
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
        # Auto vocab size: fully derived from corpus with no upper cap.
        # Formula: unique_words × 1.5  (floor 1000).
        # × 1.5 gives enough merges to represent whole common words while
        # still splitting rare ones into known subword pieces.
        # The floor prevents a trivially tiny vocab on micro-corpora.
        # No upper cap — the right vocab size is whatever the corpus needs.
        #
        # Speed note for pure Python fallback:
        #   Training time is O(merges × unique_words).
        #   At 10k merges: ~1 min.  At 32k: ~15 min.  At 80k+: hours.
        #   Install `tokenizers` (pip install tokenizers) for C++ speed
        #   that makes even 100k merges take seconds.
        try:
            import tokenizers as _hf_tok  # noqa: F401
            _fast_available = True
        except ImportError:
            _fast_available = False

        if bpe_vocab_size is None:
            unique_words   = len(set(_re.findall(r"\S+", text)))
            bpe_vocab_size = int(max(1000, unique_words * 1.5))
            if not _fast_available and bpe_vocab_size > 8000:
                _est_min = bpe_vocab_size // 2000          # rough minutes estimate
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

    # ── encode corpus ─────────────────────────────────────────────────
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n    = len(data)

    # Dynamic split ratio: hold out more for validation when corpus is large
    # enough that a bigger val set gives a more reliable signal.
    # Small corpus (< 50k)  → no split (all data for training)
    # Medium (50k–200k)     → 95/5  (protect scarce training tokens)
    # Large (200k–1M)       → 90/10
    # Very large (> 1M)     → 80/20 (val signal more valuable than extra train)
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
    """Vectorised GPU batch sampler — no Python loops, no CPU→GPU copies per step.
    Requires `data` to already live on the target device (moved once at startup).
    """
    max_start = data.shape[0] - seq_len - 1
    ix      = torch.randint(0, max_start, (batch_size,), device=data.device)
    offsets = torch.arange(seq_len, device=data.device)        # [seq_len]
    idx     = ix.unsqueeze(1) + offsets.unsqueeze(0)           #[B, seq_len]
    x = data[idx]                                              # [B, seq_len]
    y = data[idx + 1]                                          # [B, seq_len]
    return x, y


# ==========================================================
# TRAINING
# ==========================================================

def train_model(filepath: str = "data.txt",
                tokenizer_mode: str = "tiktoken",
                bpe_vocab_size: int | None = None):
    """
    Full training loop with all three critical fixes applied.

      - CausalTransformerDecoder        (FIX 1: correct P(x_t | x_{<t}))
      - Annealed entropy weight         (FIX 2: safe codebook diversification)
      - Train/val split                 (FIX 3: overfitting detection)
      - Codebook active-symbol logging  (FIX 3: collapse detection)
      - Adaptive LR reduction           (BONUS: responds to actual training dynamics)
    """
    # ── Hyperparameters (static base values) ─────────────────────────
    steps          = 30000
    lr             = 3e-4
    lr_warmup      = 100
    weight_decay   = 0.01
    orth_tau       = 100
    log_every      = 100
    checkpoint      = "neuro_symbolic_model.pt"
    best_checkpoint = "neuro_symbolic_model_best.pt"

    # ── Dynamic parameter initial values ──────────────────────────────

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
    sym_decor_lr      = 0.005
    sym_decor_min     = 0.001
    sym_decor_max     = 2.0   # lambda can grow large if streams are consistently overlapping

    # 5. vq_beta
    vq_beta           = 0.25   # was 0.60 — VQ loss is now per-stream-normalised, so CE dominates early
    target_active     = 0.72   # lowered: 0.85 pushed Act% to 100% killing dropout pressure
    vq_beta_lr        = 0.01   # was 0.003 — faster adaptation now that scale is stable
    vq_beta_min       = 0.15   # floor raised: 0.05 let symbolic bottleneck collapse
    vq_beta_max       = 0.5    # was 0.9 — >0.5 would let VQ dominate CE again

    # 6. H_target annealing (num_symbols_ref set after tokenization)
    num_symbols_ref   = None
    h_target_max_frac = 1.0    # full entropy pressure — reached after ramp-in
    h_target_min_frac = 0.3    # starting pressure — low so grammar learning comes first
    h_target_ramp_steps = 2000  # steps to ramp entropy pressure from min → max

    # 7. Orth onset (VQ-triggered)
    orth_T0           = steps
    orth_unlocked     = False

    # 8. Dropout
    current_dropout   = 0.1
    dropout_min       = 0.0
    dropout_max       = 0.4
    dropout_lr        = 0.01
    overfit_threshold = 1.15

    # 9. alpha_reg_lambda — controller around target mean bypass α
    alpha_reg_lambda  = 0.005
    target_alpha      = 0.15   # desired mean residual α (0 = full discrete, 0.5 = max bypass)
    alpha_lr          = 0.005
    alpha_lambda_min  = 0.001
    alpha_lambda_max  = 0.05

    # 10. orth_lambda — controller around target feature cross-covariance
    orth_lambda       = 1e-5   # starting lambda — Frobenius sum is naturally large (D²× correlations)
    target_orth       = 0.5    # desired feat_orth_loss: Frobenius sum on [D,D] with small corr ≈ 0.1–2.0
                               # was 0.05 tuned for the .mean()×100 hack; now reflects true Frobenius scale
    orth_lambda_lr    = 0.005
    orth_lambda_min   = 1e-6   # floor: keep some pressure even when streams are very orthogonal
    orth_lambda_max   = 1e-3   # ceiling: Frobenius sum is ~65536× larger than old .mean(), so lambda must be ~65536× smaller

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

    # 14. Dynamic weight_decay — complements dropout at weight level
    #     High train/val gap → raise; gap closing → relax
    weight_decay_val    = weight_decay   # current value (updated each val step)
    weight_decay_min    = 0.001
    weight_decay_max    = 0.1
    weight_decay_lr     = 0.01           # fractional update per val step

    # 15. Dynamic lr_warmup — scales with model size, fixed once at build time
    #     (set after model is created)
    lr_warmup_base      = lr_warmup      # will be overridden below

    # 16. Velocity-based self-tuning LR controller
    #
    #     Replaces the old patience counter with a continuous EMA of CE velocity
    #     (d_CE/d_step).  Negative velocity = improving.  The controller fires
    #     every vel_check_every steps and has a cooldown to prevent thrashing.
    #
    #     vel_reduce_thresh: if velocity ≥ this (stalled/worsening) → reduce LR
    #     vel_boost_thresh:  if velocity ≤ this (improving fast) → small boost
    #     vel_ema_alpha:     smoothing — smaller = smoother but slower to react
    #     vel_cooldown:      minimum steps between any LR change
    #     vel_check_every:   how often to evaluate and potentially act
    #
    vel_ema_alpha       = 0.02    # ~50-step effective window (slow, stable signal)
    vel_check_every     = 100     # evaluate velocity this often (was 50 — too frequent)
    vel_reduce_thresh   = 0.0002  # velocity ≥ this → clearly stalled → reduce
                                  # (was -0.0003: triggered on batch noise in warmup;
                                  #  0.0002 means loss must be *rising* before cutting)
    vel_boost_thresh    = -0.005  # velocity ≤ this → improving fast → small boost
    lr_reduce_factor    = 0.99    # LR multiplier on stall — 1% reduction per trigger
    lr_boost_factor     = 1.03    # LR multiplier when improving fast (gentler)
    vel_cooldown        = 300     # min steps between LR changes (was 150)
    vel_min_step        = 1000    # never reduce LR before this step — was 400, raised to
                                  # protect early training: codebooks need ~1000 steps to
                                  # stabilise before LR cuts are useful
    lr_reductions       = 0       # count of LR reductions so far

    # Stream-addition global LR reset
    #     When a new stream is added ALL existing params need to rewire around it.
    #     Reset the global LR to stream_lr_reset_fraction × peak_lr so the whole
    #     model (not just new params) gets a meaningful learning signal.
    #     lr_ceiling also resets so cosine boost can work from the new level.
    stream_lr_reset_fraction = 0.35   # global reset target = 0.35 × peak LR

    # ── TopK temperature annealing ────────────────────────────────────────
    # Start cold (hard routing: 1 symbol dominates) → anneal to warm (soft blend).
    # The model first learns WHICH symbols are meaningful, then learns to blend them.
    # tau_start=0.1 → near-argmin at init, tau_end=1.5 → genuinely soft by end.
    topk_tau_start = 0.1    # cold: weights ≈ one-hot on nearest
    topk_tau_end   = 0.5    # warm: soft blend but never fully uniform (log(3)=1.10 would be flat average)
    # Annealing is linear in step; each codebook's .topk_tau is updated each step.

    # Meta-codebook tau — anneals more slowly so meta routing stays sharp longer.
    # Level-1 reached τ≈0.40 by step 11k and TopKH→1.10 (fully uniform).
    # Meta tau starts at the same cold value but ends much cooler, keeping
    # routing decisive throughout training.  Meta semantics need sharp routing
    # to form clean concept clusters; blurring K symbols together loses the
    # discrete structure the meta-layer is supposed to provide.
    meta_topk_tau_start = 0.1   # same cold start
    meta_topk_tau_end   = 0.25  # half the warmth of level-1 — stays meaningfully sharp

    # ── Rolling stream addition trigger ──────────────────────────────────
    # Old trigger: wait for ALL streams to saturate simultaneously (too conservative).
    # New trigger: add a stream whenever the MEDIAN gate EMA exceeds threshold,
    # i.e. more than half the existing streams are saturated.
    # This gets late streams (Valence, Motion) in ~3-5× earlier.
    stream_trigger_quorum = 0.6   # fraction of active streams that must be saturated

    # ── Dead codebook revival ─────────────────────────────────────────────
    # Every revival_check_every steps, scan each codebook for near-zero entries.
    # Reset them to perturbed copies of live entries.
    revival_check_every = 500
    revival_noise_scale = 0.1     # std of perturbation applied to revived entries

    # ── Cross-stream centroid kickstart ───────────────────────────────────
    # If the worst JS pair hasn't improved above kickstart_js_thresh after
    # kickstart_patience steps, perturb the more-correlated stream's centroids.
    kickstart_js_thresh  = 0.35   # if worst pair JS below this → consider kickstart
    kickstart_patience   = 800    # steps worst pair must be below threshold before firing
    kickstart_cooldown   = 1000   # min steps between kickstarts for same pair
    kickstart_last       = {}     # {(i,j): last_step_kickstarted}
    kickstart_bad_since  = {}     # {(i,j): step_first_bad}

    # 17. Adam beta2 annealing — 0.99 (fast adapt early) → 0.999 (stable late)
    adam_beta2_start    = 0.99
    adam_beta2_end      = 0.999
    adam_beta1          = 0.9

    # 18. seq_len_max curriculum gated by val CE thresholds
    #     Expands 64→128→256→512 as val CE crosses thresholds
    seq_len_stages      = [64, 128, 256, 512]
    seq_len_stage_idx   = 0
    seq_len_max         = seq_len_stages[0]   # overrides the static 512 set above

    # 19. batch_size_tokens doubles on each LR reduction
    batch_tokens_max    = 4096   # hard cap — 8192 OOMs at seq_len=512 with 12 streams

    # 20. log_every grows gently over time (less noise when training is stable)
    log_every_base      = 100
    log_every_max       = 150  # caps at 150 — tied to steps_since_imp (not step count)

    # 21. commitment_cost per-stream — raised when active% low, relaxed when high
    commit_cost_lr      = 0.005
    commit_cost_min     = 0.15  # floor raised: 0.05 let cc collapse alongside beta
    commit_cost_max     = 0.50  # ceiling must be > init (0.25) so controller can raise cc when Act% is low
                                # was 0.25 = same as init → controller was a no-op, cc stuck at 0.250 all run
    target_active_per   = target_active   # reuse global target

    # ── Meta-codebook (level-2 VQ) hyperparameters ────────────────────
    # Independent controllers: composite is already stable so less pressure
    # needed.  Same floor (0.15) to prevent late-training collapse.
    meta_vq_beta        = 0.10          # softer start than level-1 (was 0.25)
    meta_vq_beta_min    = 0.15          # same floor — prevent meta bottleneck collapse
    meta_vq_beta_max    = 0.70          # raised from 0.40: 0.40 ceiling hit but Act stayed at 0.31
    meta_vq_beta_lr     = 0.005         # same controller rate as level-1
    meta_commit_cost_min = 0.15
    meta_commit_cost_max = 0.60         # raised from 0.40: same reason as beta ceiling
    meta_commit_cost_lr  = 0.005
    meta_target_active   = 0.72         # same target as level-1


    # 23. ema_decay annealing per codebook — fast (0.2) early, slow (0.05) late
    ema_decay_start     = 0.2
    ema_decay_end       = 0.05

    # 24. stream_check_every — frequent early, sparse late
    # (managed via stream_check_every variable already declared, updated below)

    # 25. per-stream codebook growth — grow when active% hits 100% repeatedly
    cb_grow_check_every = 200       # steps between codebook grow checks
    cb_grow_active_thresh = 0.95    # must be above this consistently
    cb_grow_patience    = 400       # steps above threshold before growing
    cb_active_ema       = None      # per-stream EMA of active_ratio (init after model)
    cb_active_ema_alpha = 0.05
    cb_grow_counts      = None      # per-stream steps above threshold

    # 26. Per-pair sym_decor lambdas — each stream pair gets its own controller.
    #     Pairs that keep overlapping get stronger pressure; separated pairs relax.
    #     Initialised after model creation once we know which pairs exist.
    pair_sym_lambdas     = {}        # {(i,j): float} — populated after model init
    pair_sym_lambda_init = 0.01
    pair_sym_lambda_lr   = 0.005
    pair_sym_lambda_min  = 0.001
    pair_sym_lambda_max  = 20.0  # raised from 5.0; streams need stronger push
    target_pair_js       = 0.05     # target overlap per pair (near-zero = separated)

    # ── Growth LR reset ───────────────────────────────────────────────────
    # When a new stream or codebook is added, new params are randomly
    # initialised but the global LR may have decayed to near lr_min, giving
    # them almost no signal.  Instead of a factor-of-current multiplier
    # (which is tiny when LR is already tiny), always reset new params to a
    # fixed fraction of the original peak LR, then linearly decay back to
    # whatever the current global LR is over growth_warmup_steps.
    #
    # Example: peak LR=3e-04, growth_reset_fraction=0.25 → new params always
    # start at 7.5e-05, regardless of when growth happens.  This is 25x the
    # final lr_min=3e-06, giving late-added streams a real learning signal.
    growth_reset_fraction  = 0.25  # fraction of peak LR for new params
    growth_warmup_steps    = 500   # steps to decay back to current global LR

    # 28. H_target scales with batch_tokens — bigger batches can support
    #     more symbol diversity because the histogram is estimated over more tokens.
    h_target_batch_scale = True    # enable scaling
    h_target_batch_ref   = 1024   # reference batch_tokens (baseline = no scaling)

    # 29. Orth onset based on feature correlation — trigger when cross-stream
    #     correlation rises above a calibrated baseline rather than a fixed VQ thresh.
    stream_corr_baseline = None    # calibrated mean cross-stream cosine sim
    stream_corr_ema      = 0.0
    stream_corr_alpha    = 0.05
    stream_corr_trigger  = 0.05   # correlation above baseline by this → unlock orth
    orth_fallback_step   = 1500   # unlock orth unconditionally at this step if correlation
                                   # trigger hasn't fired — was 500, too early for grammar to form
    stream_corr_calib_done = False

    # 30. Seq_len curriculum by improvement rate — unlock next seq_len stage when
    #     val CE improvement per step has plateaued rather than at a fixed threshold.
    seq_len_val_ce_history =[]    # val CE readings at current stage
    seq_len_improvement_ema = 0.0
    seq_len_improvement_alpha = 0.3
    seq_len_plateau_thresh = 0.005  # improvement per val step below this → unlock

    # 31. Depth growth by final-layer gradient magnitude — add a layer when the
    #     last decoder block's grad norm is high relative to the average, signalling
    #     that the model wants more depth, not when LR happens to reduce.
    layer_grad_emas      =[]      # per-layer grad norm EMA (populated after model)
    layer_grad_alpha     = 0.05
    depth_grad_ratio_thresh = 1.3  # lowered from 1.5: never triggered in 30k steps at 1.5
    depth_check_every    = 200     # steps between depth checks
    depth_last_added     = 0       # step of last layer addition
    depth_min_dwell      = 500     # min steps between additions

    # ── Data ──────────────────────────────────────────────────────────
    train_data, val_data, tokenizer, vocab_size = load_data(
        filepath, tokenizer_mode, bpe_vocab_size
    )

    # Move entire dataset to GPU once — eliminates per-step CPU→GPU transfer.
    # On 8 GB VRAM: 663k tokens × 8 bytes = ~5 MB, negligible.
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
    # Each stream covers its share of vocabulary space, not all of it.
    # Formula: sqrt(vocab_size / max_streams) — scales automatically with
    # both corpus size and how many streams will share the representational load.
    # Total capacity = max_streams × target = max_streams^0.5 × vocab^0.5
    # which still grows with vocab but keeps each codebook manageable.
    num_symbols_target = max(16, int((vocab_size / max_streams) ** 0.5))
    num_symbols        = max(16, min(32, num_symbols_target))  # start small
    num_symbols_ref    = num_symbols_target   # H_target anchored to final size
    print(f"  num_symbols: {num_symbols} (start, target {num_symbols_target})  "
          f"(vocab={vocab_size:,}/{max_streams}streams → √{vocab_size//max_streams} → {num_symbols_target})")

    # Meta-codebook target: same formula — it compresses the composite over
    # the same token distribution, so the same capacity is appropriate.
    meta_num_symbols_target = num_symbols_target
    meta_num_symbols_start  = max(16, min(32, meta_num_symbols_target))

    # ── Dynamic d_model ───────────────────────────────────────────────
    # Scale model width with corpus size: more data → richer representation.
    # Formula: n_train^0.25 × 8, rounded to nearest 64, clamped[128, 768].
    # Examples:
    #   50k tokens  → 50000^0.25 × 8 ≈ 119 → 128
    #   200k tokens → 200000^0.25 × 8 ≈ 169 → 192
    #   500k tokens → 500000^0.25 × 8 ≈ 211 → 192
    #   1M tokens   → 1000000^0.25 × 8 ≈ 253 → 256
    #   5M tokens   → 5000000^0.25 × 8 ≈ 358 → 384
    _raw_d  = (n_train ** 0.25) * 8
    d_model = int(max(128, min(768, round(_raw_d / 64) * 64)))
    print(f"  d_model: {d_model}  "
          f"(n_train={n_train:,}^0.25 × 8 ≈ {_raw_d:.0f} → nearest 64)")

    # ── Dynamic n_heads ───────────────────────────────────────────────
    # Largest power-of-2 divisor of d_model, capped at 8.
    # Guarantees d_model % n_heads == 0 without manual coordination.
    n_heads = 1
    for h in[8, 4, 2, 1]:
        if d_model % h == 0:
            n_heads = h
            break
    print(f"  n_heads: {n_heads}  (largest pow2 divisor of {d_model} ≤ 8)")

    # ── Dynamic num_layers ────────────────────────────────────────────
    # Start proportional to corpus size; grows during training on LR reductions.
    # Formula: log2(n_train / 1000), clamped [2, 8].
    # Examples:
    #   10k tokens  → log2(10)  ≈ 3.3 → 3
    #   50k tokens  → log2(50)  ≈ 5.6 → 6
    #   500k tokens → log2(500) ≈ 8.96 → 8 (capped)
    num_layers     = int(max(2, min(8, round(math.log2(n_train / 1000)))))
    num_layers_max = 16
    print(f"  num_layers: {num_layers} (start)  max {num_layers_max}  "
          f"(grows on LR reductions)")

    # ── Dynamic seq_len_base ──────────────────────────────────────────
    # Larger vocab → tokens carry more info → need fewer tokens for context.
    # Smaller vocab (char-level) → need more tokens for a meaningful chunk.
    # Formula: clamp(log2(vocab) × 2, 8, 64)
    seq_len_base = int(max(8, min(64, round(math.log2(vocab_size) * 2))))
    # seq_len_max already set to seq_len_stages[0] by curriculum init -- do NOT reset.
    print(f"  seq_len_base: {seq_len_base}  seq_len_max (curriculum start): {seq_len_max}")

    # ── Model ─────────────────────────────────────────────────────────
    model = SubmodalitySymbolicLanguageModel(
        vocab_size  = vocab_size,
        d_model     = d_model,
        num_symbols = num_symbols,
        n_heads     = n_heads,
        num_layers  = num_layers,
    ).to(device)

    # Resize meta_codebook to correct start size (model __init__ uses num_symbols
    # for both level-1 and meta; they share the same start value so this is a no-op
    # unless meta_num_symbols_start diverges in future).
    if meta_num_symbols_start != num_symbols:
        model.meta_codebook.grow_codebook(meta_num_symbols_start, device)

    # Meta-codebook tracking (mirrors per-stream cb_ vars)
    meta_cb_active_ema  = 0.5   # EMA of meta active_ratio
    meta_cb_grow_count  = 0     # steps above threshold

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  Device: {device}")

    # torch.compile — traces the forward pass and fuses ops into optimised
    # CUDA kernels.  dynamic=True handles variable batch/seq lengths.
    # Structural changes (add_layer, add_submodality) require a recompile;
    # the _compiled_model variable is updated at those points below.
    # torch.compile requires Triton which needs CUDA capability >= 7.0.
    # GTX 1070 Ti and older cards are 6.1 — skip compile on those.
    _cuda_cap = (torch.cuda.get_device_capability(device)
                 if device.type == "cuda" else (0, 0))
    _torch_compile_available = (
        hasattr(torch, "compile")
        and device.type == "cuda"
        and _cuda_cap[0] >= 7
    )
    if _torch_compile_available:
        # capture_scalar_outputs: allows .item() calls inside compiled graphs
        # without breaking out of the traced region.  Without this, every
        # active_ratio = (...).item() in NeuroSymbolicCodebook causes a graph
        # break and a full recompile — negating most of the compile speedup.
        torch._dynamo.config.capture_scalar_outputs = True
        # Suppress the 'not enough SMs for max_autotune_gemm' warning —
        # it is purely informational and fires on every matmul on GPUs with
        # fewer than ~108 SMs (RTX 20/30 mid-range and below).
        import logging, warnings
        logging.getLogger('torch._inductor.utils').setLevel(logging.ERROR)
        # Suppress "Online softmax disabled" — Inductor informational only
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

    # Initialise per-stream tracking state now that model exists
    N = model.NUM_SUBMODALITIES
    stream_gate_ema    = [1.0 / N] * N
    stream_dead_counts = [0] * N
    cb_active_ema      = [0.5] * N      # per-stream active_ratio EMA
    cb_grow_counts     = [0] * N        # steps above cb_grow_active_thresh

    # 26. Initialise per-pair sym_decor lambdas
    pair_sym_lambdas = {
        (i, j): pair_sym_lambda_init
        for i in range(N) for j in range(i + 1, N)
    }

    # 31. Initialise per-layer gradient norm EMAs
    layer_grad_emas = [1.0] * num_layers  # one per decoder block

    # Per-stream dead/saturated thresholds derived from uniform gate
    def _stream_thresholds(n):
        uniform = 1.0 / n
        return 0.5 * uniform, 0.95 * uniform   # dead_thresh, saturated_thresh
    stream_dead_thresh, stream_saturated_thresh = _stream_thresholds(N)

    # 15. Dynamic lr_warmup — proportional to model depth × width
    #     Bigger models need longer warmup for VQ codebooks to form
    lr_warmup = max(500, int(lr_warmup_base * (num_layers / 4) * (d_model / 256)))

    # Weight decay should NOT apply to codebook entries, biases, or LayerNorm
    # parameters — standard transformer practice, and critical for codebooks
    # (AdamW wd pulls codebook toward origin, fighting the VQ commitment loss).
    _no_decay_keys = ('bias', 'norm', 'codebook', 'residual_alpha_raw', 'ema_entropy', 'meta_residual_raw')
    _decay_params   = [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(k in n for k in _no_decay_keys)]
    _nodecay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and     any(k in n for k in _no_decay_keys)]
    print(f"  Optimizer groups: {len(_decay_params)} decay params, "
          f"{len(_nodecay_params)} no-decay params")
    optimizer = torch.optim.AdamW([
        {'params': _decay_params,   'weight_decay': weight_decay},
        {'params': _nodecay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=(adam_beta1, adam_beta2_start))

    # Mixed precision scaler — fp16 on GPU, disabled on CPU.
    # Cuts VRAM usage ~50% with negligible accuracy impact.
    _use_amp = torch.cuda.is_available()
    scaler   = GradScaler('cuda', enabled=_use_amp)

    # lr_ceiling: cosine nudge can only boost LR up to this value.
    # Decreases with every plateau reduction so boosts cannot undo reductions.
    lr_ceiling = lr

    # Queue of active growth warmup events for new param groups.
    # Each entry: {'group_idx': int, 'start_lr': float, 'start_step': int}
    growth_warmup_queue =[]
    raw_norm = torch.tensor(0.0)  # gradient norm before clipping; updated each step

    # ── Dynamic LR ────────────────────────────────────────────────────
    # After warmup, LR is reduced automatically when training stalls.
    # An EMA of training CE is tracked; if it hasn't improved by at least
    # lr_reduce_threshold (fractional) over lr_patience steps, the LR is
    # multiplied by lr_reduce_factor.  This repeats until lr_min is reached.
    # More responsive than cosine (which decays regardless of actual progress)
    # and fully transparent — every reduction is printed with ↓ LR.
    lr_reduce_threshold  = 0.005  # minimum fractional improvement required for patience controller
    criterion = nn.CrossEntropyLoss()

    best_val_loss    = float("inf")
    last_train_ce    = None
    grad_norm_ema    = grad_clip

    # Velocity-based LR controller state
    ce_vel_ema       = 0.0       # smoothed dCE/dstep (negative = loss improving)
    ce_prev          = None      # previous step's CE for velocity calculation
    vel_last_action  = 0         # step when LR was last changed (cooldown guard)
    lr_min           = lr / 50   # hard floor (6e-6 for lr=3e-4) — was /100; floor was hit
                                  # too early, leaving model under-powered for last ~20k steps

    # Val-patience LR controller state (from v52) — fires when training CE stalls
    best_ce_ema      = float("inf")  # best smoothed training CE seen so far
    ce_ema           = None           # EMA of training CE for patience detection
    ce_ema_alpha     = 0.05           # slow EMA — stable, low-noise signal
    steps_since_imp  = 0              # steps elapsed since last CE improvement
    lr_patience_base = 400            # base steps before LR reduction fires
    lr_patience_grow = 100            # patience grows by this after each reduction
    lr_patience      = 400            # current patience threshold (grows over time)

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    print("Starting training…\n")

    def _add_with_growth_warmup(new_params, current_step):
        """
        Add new parameters to the optimizer with a reset LR = peak * growth_reset_fraction,
        decaying linearly back to the current global LR over growth_warmup_steps steps.
        Using a fixed fraction of peak LR (not a multiple of current LR) means
        late-added params always get a meaningful learning rate regardless of how
        far the global LR has decayed.
        Codebook params are added with weight_decay=0 to match the main optimizer groups.
        """
        cur_lr    = optimizer.param_groups[0]["lr"]
        reset_lr  = max(cur_lr * 2.0, lr * growth_reset_fraction)  # at least 2× current
        group_idx = len(optimizer.param_groups)
        _no_decay_keys = ('bias', 'norm', 'codebook', 'residual_alpha_raw', 'ema_entropy', 'meta_residual_raw')
        # new_params is a flat list — infer no-decay by checking registered buffer names
        # We can't use named_parameters here, so use a heuristic: ema_entropy buffer
        # and codebook.weight are registered buffers/params in NeuroSymbolicCodebook.
        # The safest approach: add one group; for codebooks the pull toward 0 is minor
        # compared to VQ gradient, but still worth zeroing for correctness.
        # We split by checking if the param is a codebook weight (num_symbols × d).
        _is_codebook = lambda p: (p.dim() == 2 and p.shape[0] <= 2048
                                  and p.requires_grad
                                  and p.shape[0] != p.shape[1])  # non-square → likely codebook
        _decay_p   = [p for p in new_params if not _is_codebook(p)]
        _nodecay_p = [p for p in new_params if     _is_codebook(p)]
        if _decay_p:
            optimizer.add_param_group({
                "params": _decay_p, "lr": reset_lr, "weight_decay": weight_decay_val,
            })
        if _nodecay_p:
            optimizer.add_param_group({
                "params": _nodecay_p, "lr": reset_lr, "weight_decay": 0.0,
            })

        # Register all newly added groups in the warmup queue so their LRs
        # decay back to the current global LR over growth_warmup_steps steps.
        for idx in range(group_idx, len(optimizer.param_groups)):
            growth_warmup_queue.append({
                "group_idx":  idx,
                "start_lr":   reset_lr,
                "start_step": current_step,
            })
        print(f"    ↑ Growth LR reset: {reset_lr:.2e} → {cur_lr:.2e} "
              f"over {growth_warmup_steps} steps")
    _train_start   = time.time()
    _step_time     = time.time()
    _last_log_step = 0  # track actual steps elapsed for correct step_secs
    _last_heartbeat = time.time()  # for between-log heartbeat prints
    total_tokens_seen = 0  # cumulative tokens processed across all steps

    for step in range(steps):

        # ── 23. Dynamic ema_decay annealing ──────────────────────────
        # Linearly anneal from fast (0.2) → slow (0.05) over all steps.
        # Fast early: noisy entropy estimates are fine when codebook is forming.
        # Slow late: stable EMA needed for reliable H_target tracking.
        ema_decay_now = ema_decay_start - (ema_decay_start - ema_decay_end) * (step / max(1, steps - 1))
        for cb in model.codebooks:
            cb.ema_decay = ema_decay_now

        # ── 17. Adam beta2 annealing ──────────────────────────────────
        beta2_now = adam_beta2_start + (adam_beta2_end - adam_beta2_start) * (step / max(1, steps - 1))
        for g in optimizer.param_groups:
            g["betas"] = (adam_beta1, beta2_now)

        # ── 20. Dynamic log_every ─────────────────────────────────────
        # log_every grows gently over time — frequent early, sparse late
        # log_every grows when training stalls (steps_since_imp) — dense when active,
        # sparse on plateau. Capped at 150 to prevent freezes.
        log_every = min(log_every_max, log_every_base + steps_since_imp // 10)

        # ── 1. Dynamic seq_len (with curriculum gating) ───────────────
        seq_len = min(seq_len_max,
                      seq_len_base + int(math.log(step + 1) * 8))
        safe_seq = min(seq_len, len(train_data) - 2)

        # ── 2. Dynamic batch size ─────────────────────────────────────
        batch_size = int(max(batch_size_min,
                             min(batch_size_max, batch_size_tokens // safe_seq)))

        x, y = _random_batch(train_data, batch_size, safe_seq)
        total_tokens_seen += batch_size * safe_seq

        # ── LR warmup ─────────────────────────────────────────────────
        if step < lr_warmup:
            for g in optimizer.param_groups:
                g["lr"] = lr * (step + 1) / lr_warmup

        # ── 6. H_target annealing + batch_tokens scaling ─────────────
        # Ramp entropy pressure from h_target_min_frac → h_target_max_frac over
        # the first h_target_ramp_steps, then hold at max.
        # Early training: low pressure lets the model learn token co-occurrence
        # and grammar structure without being forced to simultaneously maximise
        # codebook diversity.  After ~2000 steps the structure is established and
        # full diversity pressure helps prevent late-stage codebook collapse.
        h_ramp    = min(1.0, step / max(1, h_target_ramp_steps))
        h_frac    = h_target_min_frac + (h_target_max_frac - h_target_min_frac) * h_ramp
        h_base    = h_frac * math.log(num_symbols_ref)
        if h_target_batch_scale and batch_size_tokens > h_target_batch_ref:
            batch_scale = 1.0 + 0.1 * math.log(batch_size_tokens / h_target_batch_ref)
        else:
            batch_scale = 1.0
        h_target_now = h_base * min(batch_scale, 1.3)  # cap at 30% boost
        for cb in model.codebooks:
            # Cap H_target by what's achievable with this codebook's current size.
            # num_symbols starts at 32; driving toward log(308) is impossible early.
            cb.H_target = min(h_target_now, 0.9 * math.log(max(2, cb.num_symbols)))

        # ── 10. Sync orth_lambda into model ───────────────────────────
        model.sub_encoder.orthogonality_lambda = orth_lambda

        optimizer.zero_grad()
        with autocast('cuda', enabled=_use_amp):
            logits, _, vq_loss, feat_orth_loss, gate_kl, sym_decor_loss, \
                gate_weights, mean_H, per_stream_active, pair_js_vals, pair_grad_terms, \
                sub_features, per_stream_topk_H, \
                meta_vq_loss, meta_ema_H, meta_active_ratio, meta_topk_H = _compiled_model(x)
            active_ratio = sum(per_stream_active) / len(per_stream_active)
            loss_task = criterion(logits.view(-1, vocab_size), y.view(-1))
        last_train_ce = loss_task.item()

        vq_val = vq_loss.item()

        # ── 7. Orth onset by feature correlation ─────────────────────
        # Rather than triggering when VQ loss crosses an arbitrary threshold,
        # we measure the actual mean cosine similarity between stream feature
        # pairs.  Once correlation starts rising above its calibrated baseline
        # the streams are starting to converge — that's when orth pressure helps.
        if orth_unlocked is False:
            with torch.no_grad():
                # Batched pairwise cosine similarity — one matmul instead of
                # N*(N-1)/2 separate dot products.
                sub_f = sub_features.detach().reshape(-1, model.NUM_SUBMODALITIES, model.d_model)
                # Normalise all streams at once:[B*T, N, D]
                sub_f_n = F.normalize(sub_f, dim=2)
                # Correlation matrix: [N, N] via mean over B*T tokens
                corr_mat = torch.einsum('bnd,bmd->nm', sub_f_n, sub_f_n) / sub_f_n.shape[0]
                # Upper-triangle mean (exclude diagonal)
                n_s = model.NUM_SUBMODALITIES
                idx_i, idx_j = torch.triu_indices(n_s, n_s, offset=1)
                mean_corr = corr_mat[idx_i, idx_j].mean().item()
            stream_corr_ema = ((1 - stream_corr_alpha) * stream_corr_ema
                               + stream_corr_alpha * mean_corr)
            if not stream_corr_calib_done and step >= 100:
                stream_corr_baseline = stream_corr_ema
                stream_corr_calib_done = True
                print(f"  ✦ Stream correlation baseline calibrated: {stream_corr_baseline:.4f}")
            # Correlation-based trigger
            if (stream_corr_calib_done
                    and stream_corr_ema > stream_corr_baseline + stream_corr_trigger):
                orth_T0       = step
                orth_unlocked = True
                print(f"  ✦ Orth unlocked at step {step}  "
                      f"(corr {stream_corr_ema:.4f} > baseline "
                      f"{stream_corr_baseline:.4f} + {stream_corr_trigger})")
            # Fallback: unlock after fixed steps regardless of correlation
            elif stream_corr_calib_done and step >= orth_fallback_step:
                orth_T0       = step
                orth_unlocked = True
                print(f"  ✦ Orth unlocked at step {step} (fallback — "
                      f"corr {stream_corr_ema:.4f} never rose +{stream_corr_trigger} above baseline)")
        elif stream_corr_calib_done:
            # Keep tracking correlation even after orth is on (for logging)
            with torch.no_grad():
                sub_f   = sub_features.detach().reshape(-1, model.NUM_SUBMODALITIES, model.d_model)
                sub_f_n = F.normalize(sub_f, dim=2)
                corr_mat = torch.einsum('bnd,bmd->nm', sub_f_n, sub_f_n) / sub_f_n.shape[0]
                n_s = model.NUM_SUBMODALITIES
                idx_i, idx_j = torch.triu_indices(n_s, n_s, offset=1)
                stream_corr_ema = ((1 - stream_corr_alpha) * stream_corr_ema
                                   + stream_corr_alpha * corr_mat[idx_i, idx_j].mean().item())

        # Guard: orth_T0 defaults to `steps`, so the sigmoid would fire
        # near the end of training even if orth never unlocked.
        if orth_unlocked:
            orth_weight = 0.8 / (1.0 + math.exp(-(step - orth_T0) / orth_tau))
        else:
            orth_weight = 0.0
        orth_loss      = orth_weight * feat_orth_loss

        # ── 3. Dynamic gate_gamma ─────────────────────────────────────
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

        # ── 4. Per-pair sym_decor lambdas ────────────────────────────
        # Each stream pair has its own lambda controller.  Pairs that are
        # already well-separated (low JS penalty) relax toward zero pressure;
        # pairs that keep overlapping get stronger push.
        sym_dec_val = sym_decor_loss.item()   # mean JS for logging
        pair_sym_loss = torch.tensor(0.0, device=device)
        for (pi, pj), js_val in pair_js_vals.items():
            # Update this pair's lambda toward target_pair_js
            lam = pair_sym_lambdas.get((pi, pj), pair_sym_lambda_init)
            lam = lam * (1.0 + pair_sym_lambda_lr * (js_val - target_pair_js))
            lam = max(pair_sym_lambda_min, min(pair_sym_lambda_max, lam))
            pair_sym_lambdas[(pi, pj)] = lam
            # Add differentiable hinge for this pair weighted by its lambda
            if (pi, pj) in pair_grad_terms:
                pair_sym_loss = pair_sym_loss + lam * (js_val + pair_grad_terms[(pi, pj)])
        # Keep backward-compatible sym_decor_lambda for logging as mean
        sym_decor_lambda = sum(pair_sym_lambdas.values()) / max(1, len(pair_sym_lambdas))

        # ── 5. Dynamic vq_beta ────────────────────────────────────────
        vq_beta = vq_beta * (1.0 + vq_beta_lr * (target_active - active_ratio))
        vq_beta = max(vq_beta_min, min(vq_beta_max, vq_beta))

        # ── 5b. Dynamic meta_vq_beta ──────────────────────────────────
        # Independent controller: uses meta_active_ratio, own β bounds.
        meta_vq_beta = meta_vq_beta * (1.0 + meta_vq_beta_lr * (meta_target_active - meta_active_ratio))
        meta_vq_beta = max(meta_vq_beta_min, min(meta_vq_beta_max, meta_vq_beta))

        # Per-step meta commit_cost update (mirrors per-stream cc controller)
        model.meta_codebook.commitment_cost = max(
            meta_commit_cost_min,
            min(meta_commit_cost_max,
                model.meta_codebook.commitment_cost * (
                    1.0 + meta_commit_cost_lr * (meta_target_active - meta_active_ratio)
                )
            ),
        )

        # ── 9. Dynamic alpha_reg_lambda ───────────────────────────────
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

        # ── 10. Adaptive orth_lambda (similarity-aware) ───────────────
        # Two signals drive the lambda:
        #   (a) loss-error: orth_raw > target_orth → raise; below → lower
        #   (b) stream similarity: when streams are more correlated than their
        #       calibrated baseline, amplify the pressure proportionally.
        #       corr_factor is 1.0 at baseline, ~1.5× when corr rises 0.1 above it.
        #       This prevents the controller from becoming inert when streams drift
        #       together slowly (loss-error signal lags; similarity signal is instant).
        # orth_raw must be raw Frobenius (before lambda) so the controller
        # compares against target_orth at the correct scale.
        # feat_orth_loss already has orthogonality_lambda baked in (returns lambda*raw),
        # so divide by lambda to recover the unscaled Frobenius value.
        orth_raw     = feat_orth_loss.item() / max(orth_lambda, 1e-12)
        if stream_corr_calib_done and stream_corr_baseline is not None:
            corr_excess  = max(0.0, stream_corr_ema - stream_corr_baseline)
            corr_factor  = 1.0 + 5.0 * corr_excess   # e.g. +0.1 corr → 1.5× pressure
        else:
            corr_factor  = 1.0
        orth_lambda = orth_lambda * (1.0 + orth_lambda_lr * (orth_raw - target_orth) * corr_factor)
        orth_lambda = max(orth_lambda_min, min(orth_lambda_max, orth_lambda))

        total_loss = (loss_task
                      + vq_beta        * vq_loss
                      + meta_vq_beta   * meta_vq_loss
                      + orth_loss
                      + gate_loss
                      + gate_gamma     * balance_loss
                      + alpha_reg
                      + pair_sym_loss)

        # ── NaN guard ─────────────────────────────────────────────────────
        # If any component produced NaN (e.g. zero-norm codebook entry in
        # diversity loss, or overflow in orth/gate), skip this step entirely
        # rather than propagating NaN through gradients and killing the run.
        if not torch.isfinite(total_loss):
            optimizer.zero_grad()
            # Reduce LR slightly to try to recover stability
            for g in optimizer.param_groups:
                g["lr"] = max(lr_min, g["lr"] * 0.95)
            ce_prev = None  # discard stale velocity
            if step % log_every == 0:
                print(f"  ⚠ NaN/Inf loss at step {step} — skipping step, LR reduced ×0.95")
            continue

        scaler.scale(total_loss).backward()

        # Unscale before grad clip and cosine similarity so they see real magnitudes
        scaler.unscale_(optimizer)

        # ── 2. Dynamic grad clip ──────────────────────────────────────
        raw_norm  = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Guard: only update EMA when norm is finite — inf/NaN would poison
        # grad_norm_ema permanently, pegging clip threshold at its ceiling.
        if torch.isfinite(raw_norm):
            grad_norm_ema = (1 - grad_norm_alpha) * grad_norm_ema + grad_norm_alpha * raw_norm.item()
        grad_clip = max(grad_clip_min, min(grad_clip_max, grad_norm_ema * grad_clip_factor))


        # Skip optimizer step if gradients are non-finite.  raw_norm from
        # clip_grad_norm_ is already computed on the GPU and equals NaN/Inf
        # whenever any gradient is non-finite — no per-parameter loop needed.
        if not torch.isfinite(raw_norm):
            if step % log_every == 0:
                print(f"  ⚠ NaN/Inf gradient at step {step} — skipping step, LR ×0.95", flush=True)
            optimizer.zero_grad()
            scaler.update()  # must call update() even when skipping step
            for g in optimizer.param_groups:
                g["lr"] = max(lr_min, g["lr"] * 0.95)
            ce_prev = None
            continue

        scaler.step(optimizer)
        scaler.update()

        # ── Val-patience LR reduction (from v52) ─────────────────────────────
        # Fires when training CE EMA hasn't improved for lr_patience steps.
        # Complements velocity controller: vel catches fast stalls, patience
        # catches slow plateaus that vel_ema misses due to its smoothing.
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

        # ── Growth LR warmup decay ────────────────────────────────────────
        # Linearly decay boosted new-param LRs back toward the current global LR.
        # Once a warmup finishes, pin the group to the global LR so it follows
        # future plateau reductions naturally.
        if growth_warmup_queue:
            cur_lr = optimizer.param_groups[0]["lr"]
            done   = []
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

        # ── TopK temperature annealing ────────────────────────────────────
        # Linear cold→warm: tau_start at step 0, tau_end at final step.
        # Applied to every level-1 codebook every step.
        _tau = topk_tau_start + (topk_tau_end - topk_tau_start) * (step / max(1, steps - 1))
        for _cb in model.codebooks:
            _cb.topk_tau = _tau
        # Meta-codebook: independent slower annealing — stays sharper throughout
        _meta_tau = meta_topk_tau_start + (meta_topk_tau_end - meta_topk_tau_start) * (step / max(1, steps - 1))
        model.meta_codebook.topk_tau = _meta_tau
            
        # ── Velocity-based self-tuning LR ────────────────────────────
        # Track smoothed dCE/dstep.  Negative = loss falling = healthy.
        # Every vel_check_every steps: if velocity ≥ reduce_thresh (stalled)
        # → cut LR; if velocity ≤ boost_thresh (fast improvement) → nudge up.
        # Cooldown and vel_min_step prevent thrashing and warmup interference.
        if step >= lr_warmup:
            current_ce = loss_task.item()
            if ce_prev is not None:
                raw_vel   = current_ce - ce_prev        # negative = improving
                ce_vel_ema = ((1 - vel_ema_alpha) * ce_vel_ema
                              + vel_ema_alpha * raw_vel)
            ce_prev = current_ce

            # Val-patience: track smoothed CE for patience-based LR reduction
            if ce_ema is None:
                ce_ema = current_ce
            ce_ema = (1 - ce_ema_alpha) * ce_ema + ce_ema_alpha * current_ce
            if ce_ema < best_ce_ema * (1 - lr_reduce_threshold):
                best_ce_ema     = ce_ema
                steps_since_imp = 0
            else:
                steps_since_imp += 1

            if (step >= vel_min_step                    # never fire during warmup
                    and step % vel_check_every == 0
                    and step > vel_last_action + vel_cooldown):
                cur_lr = optimizer.param_groups[0]["lr"]

                if ce_vel_ema >= vel_reduce_thresh:
                    # Stalled or worsening → reduce
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
                        # ── 19. batch_size_tokens doubles on LR reduction ─
                        # Hard cap at 4096. At seq_len=512 that is ~8 seqs per
                        # batch (~2s/step but safe). 8192 tokens OOMs with
                        # 12 streams at seq_len=512.
                        batch_size_tokens = min(batch_tokens_max, batch_size_tokens * 2)
                        print(f"  ↑ batch_tokens → {batch_size_tokens}")
                        # ── val_every doubles on LR reduction ─────────────
                        val_every = min(val_every_max, val_every * 2)

                elif ce_vel_ema <= vel_boost_thresh:
                    # Improving fast → small boost (capped by lr_ceiling)
                    new_lr = min(lr_ceiling, cur_lr * lr_boost_factor)
                    if new_lr > cur_lr * 1.001:   # only print if meaningful
                        for g in optimizer.param_groups:
                            g["lr"] = new_lr
                        vel_last_action = step
                        print(f"  ↑ LR {cur_lr:.2e} → {new_lr:.2e}  "
                              f"(vel {ce_vel_ema:+.5f} ≤ {vel_boost_thresh:+.5f})")

        # ── 31. Depth growth by final-layer gradient magnitude ────────
        # Decouple layer addition from LR reductions.  Instead, add a layer
        # when the last decoder block's gradient norm is consistently high
        # relative to the mean across all blocks — signalling that the model
        # is bottlenecked at its current depth and wants more capacity.
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
                # Grow layer_grad_emas list if new streams added layers
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
                        
        # Individual active_ratio per codebook — run without grad
        with torch.no_grad():
            for i, cb in enumerate(model.codebooks):
                # active_ratio is already computed globally; for per-stream
                # we re-examine the per-stream last forward's indices via EMA
                # We use the global active_ratio as a proxy (per-stream requires
                # storing indices per stream — done in grow check below)
                stream_ar_i = per_stream_active[i] if i < len(per_stream_active) else active_ratio
                cb.commitment_cost = cb.commitment_cost * (
                    1.0 + commit_cost_lr * (target_active_per - stream_ar_i))
                cb.commitment_cost = max(commit_cost_min, min(commit_cost_max, cb.commitment_cost))

        # ── 24. stream_check_every grows over time ────────────────────
        stream_check_every = min(stream_check_max,
                                 100 + int(step / steps * (stream_check_max - 100)))

        # ── 13. Dynamic stream management ────────────────────────────
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
            # Only add streams after orth has unlocked (codebooks stable) and
            # after a minimum dwell time with the current stream count.
            # Without this guard, near-uniform gates at initialisation look
            # "saturated" and trigger a flood of new streams before step 500.
            # Stream additions: require step >= 300 (flood guard) and
            # dwell time, but NOT orth_unlocked — orth gating was blocking
            # all additions when correlation trigger failed to fire.
            _stream_add_eligible = (
                step >= 300
                and step >= stream_last_added_step + stream_min_dwell
            )
            if (_stream_add_eligible
                    and n_active < max_streams
                    and next_candidate_idx < len(candidate_streams)
                    and n_active > 0):
                active_emas =[stream_gate_ema[i]
                               for i in range(N) if model.stream_active[i]]
                # Rolling quorum trigger: add stream when enough existing streams
                # are saturated — not all of them.  This gets later streams
                # (Motion, Valence) added 3-5× earlier in training.
                n_saturated = sum(1 for e in active_emas if e > stream_saturated_thresh)
                quorum_met  = n_saturated >= math.ceil(stream_trigger_quorum * len(active_emas))
                if quorum_met:
                    new_name = candidate_streams[next_candidate_idx]
                    next_candidate_idx += 1
                    stream_gate_ema.append(1.0 / (N + 1))
                    stream_dead_counts.append(0)
                    cb_active_ema.append(0.5)
                    cb_grow_counts.append(0)
                    # Add per-pair lambdas for all pairs involving the new stream
                    new_idx = model.NUM_SUBMODALITIES  # index before add_submodality increments
                    for _pi in range(new_idx):
                        pair_sym_lambdas[(_pi, new_idx)] = pair_sym_lambda_init
                    _cur_lr   = optimizer.param_groups[0]["lr"]
                    _boost_lr = max(_cur_lr * 2.0, lr * growth_reset_fraction)
                    _pg_before = len(optimizer.param_groups)
                    _new_params = model.add_submodality(
                        new_name, device, optimizer, lr, weight_decay_val,
                        model.codebooks[0].num_symbols,
                        boost_lr=_boost_lr)
                    # Register all new param groups (may be 2 due to decay/no-decay split)
                    for _pg_idx in range(_pg_before, len(optimizer.param_groups)):
                        growth_warmup_queue.append({
                            "group_idx":  _pg_idx,
                            "start_lr":   _boost_lr,
                            "start_step": step,
                        })
                    # ── Global LR reset on stream addition ────────────────
                    # All existing params (gate, transformer, encoder) need
                    # to rewire to use the new stream.  Reset global LR to
                    # stream_lr_reset_fraction × peak so everything relearns.
                    _global_reset_lr = lr * stream_lr_reset_fraction
                    for g in optimizer.param_groups:
                        # Never lower a group that already has a higher LR
                        # (the new stream's warmup group should keep its boost)
                        if g["lr"] < _global_reset_lr:
                            g["lr"] = _global_reset_lr
                    lr_ceiling      = _global_reset_lr   # cosine ceiling resets too
                    ce_vel_ema      = vel_boost_thresh * 2.0  # signal "improving" to block immediate reduction
                    # Extended cooldown after stream reset: give the model vel_cooldown*3
                    # steps before the velocity controller can fire again.  Without this,
                    # the spike in loss from new-stream disruption triggers a reduction
                    # within 150 steps, undoing the reset entirely.
                    vel_last_action = step + vel_cooldown * 2  # extra 2× buffer
                    ce_prev         = None                      # discard stale velocity
                    print(f"  ↑ Global LR reset → {_global_reset_lr:.2e}  "
                          f"(stream addition, ceiling reset)")
                    stream_last_added_step = step
                    if _torch_compile_available:
                        print("\n[JIT] Recompiling model after stream addition… (may take 1–3 min)", flush=True)
                        torch._dynamo.reset()
                        _compiled_model = torch.compile(model, dynamic=True)
                        
        # Update EMA every step so it converges quickly.
        # (Gating the update inside the check interval meant the EMA
        #  needed 63k steps to reach 0.98 from 0.5 — never fired.)
        for i in range(len(cb_active_ema)):
            ar_i = per_stream_active[i] if i < len(per_stream_active) else active_ratio
            cb_active_ema[i] = ((1 - cb_active_ema_alpha) * cb_active_ema[i]
                                + cb_active_ema_alpha * ar_i)

        # Meta-codebook active EMA update
        meta_cb_active_ema = ((1 - cb_active_ema_alpha) * meta_cb_active_ema
                              + cb_active_ema_alpha * meta_active_ratio)
        if step > 0 and step % cb_grow_check_every == 0:
            for i, cb in enumerate(model.codebooks):
                # Never exceed the computed target size
                if cb.num_symbols >= num_symbols_target:
                    cb_grow_counts[i] = 0
                    continue
                if cb_active_ema[i] > cb_grow_active_thresh:
                    cb_grow_counts[i] += cb_grow_check_every
                    if cb_grow_counts[i] >= cb_grow_patience:
                        old_n = cb.num_symbols
                        # Grow by 50% but never exceed target
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
                            _add_with_growth_warmup(
                                list(cb.codebook.parameters()), step)
                            if _torch_compile_available:
                                print("\n[JIT] Recompiling model after codebook growth… (may take 1–3 min)", flush=True)
                                torch._dynamo.reset()
                                _compiled_model = torch.compile(model, dynamic=True)
                        cb_grow_counts[i] = 0
                else:
                    cb_grow_counts[i] = 0

        # ── Meta-codebook growth (same patience logic as level-1) ─────────
        if step > 0 and step % cb_grow_check_every == 0:
            meta_cb = model.meta_codebook
            if meta_cb.num_symbols < meta_num_symbols_target:
                if meta_cb_active_ema > cb_grow_active_thresh:
                    meta_cb_grow_count += cb_grow_check_every
                    if meta_cb_grow_count >= cb_grow_patience:
                        old_n  = meta_cb.num_symbols
                        n_new  = max(4, old_n // 2)
                        new_n  = min(meta_num_symbols_target, old_n + n_new)
                        if new_n > old_n:
                            old_meta_params = list(meta_cb.codebook.parameters())
                            old_meta_ids    = {id(p) for p in old_meta_params}
                            for group in optimizer.param_groups:
                                group["params"] = [p for p in group["params"]
                                                   if id(p) not in old_meta_ids]
                            for p in old_meta_params:
                                optimizer.state.pop(p, None)
                            meta_cb.grow_codebook(new_n, device)
                            _add_with_growth_warmup(list(meta_cb.codebook.parameters()), step)
                            print(f"  ↑ Meta-codebook grown: {old_n} → {new_n} symbols")
                            if _torch_compile_available:
                                torch._dynamo.reset()
                                _compiled_model = torch.compile(model, dynamic=True)
                        meta_cb_grow_count = 0
                else:
                    meta_cb_grow_count = 0
        # Periodically scan each codebook for near-zero-norm entries (dead
        # symbols that are never selected).  Reset them to perturbed copies
        # of live entries so the capacity is not wasted.
        # IMPORTANT: after revival, reset cb_active_ema and grow_counts for
        # that stream — revived entries will look "active" immediately and
        # would otherwise trigger a spurious codebook growth cascade.
        if step > 0 and step % revival_check_every == 0:
            total_revived = 0
            for i, _cb in enumerate(model.codebooks):
                n = _cb.revive_dead_entries(device, revival_noise_scale)
                if n > 0:
                    total_revived += n
                    cb_active_ema[i]  = 0.5   # reset EMA — revived entries not yet proven
                    cb_grow_counts[i] = 0      # restart patience counter
            # Meta-codebook revival
            meta_revived = model.meta_codebook.revive_dead_entries(device, revival_noise_scale)
            if meta_revived > 0:
                total_revived    += meta_revived
                meta_cb_active_ema = 0.5
                meta_cb_grow_count = 0
            if total_revived > 0:
                print(f"  ♻ Revived {total_revived} dead codebook entries across "
                      f"{model.NUM_SUBMODALITIES} streams + meta  (growth counters reset)")

        # ── Cross-stream centroid kickstart ────────────────────────────────
        # If the worst JS pair has been stuck below kickstart_js_thresh for
        # kickstart_patience steps AND is past its cooldown, perturb the
        # more-correlated stream to break the symmetry.
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
                kickstart_bad_since.pop(worst_pair, None)  # reset if improved

        # ── Heartbeat: print every 30 s between log steps to confirm alive ─
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

            # ── Per-stream gate weights (compact) ─────────────────────
            names_abbrev = [n[:3] for n in model.sub_encoder.submodality_names]
            gw = gate_weights.mean(dim=(0, 1))  # [N] mean over batch+time
            # Guard: a stream may be added after gate_weights was sampled this
            # step, so gw.shape[0] can be one less than len(names_abbrev).
            n_gw = gw.shape[0]
            gate_str = " ".join(
                f"{names_abbrev[i]}={gw[i].item():.2f}" for i in range(min(len(names_abbrev), n_gw))
            )

            # ── Per-stream active ratio: mean, min, max ────────────────
            act_min = min(per_stream_active) * 100
            act_max = max(per_stream_active) * 100
            act_str = f"{active_ratio*100:.1f}%[{act_min:.0f}–{act_max:.0f}]"

            # ── Worst-pair JS divergence ───────────────────────────────
            if pair_js_vals:
                worst_pair = min(pair_js_vals, key=pair_js_vals.get)
                worst_js   = pair_js_vals[worst_pair]
                wpi, wpj   = worst_pair
                wn_i = model.sub_encoder.submodality_names[wpi][:3]
                wn_j = model.sub_encoder.submodality_names[wpj][:3]
                worst_str = f"{worst_js:.3f}({wn_i},{wn_j})"
            else:
                worst_str = "n/a"

            # ── Mean Top-K blending softness ──────────────────────────
            # log(3)=1.099 is max for K=3 (fully uniform blend)
            mean_topk_H = sum(per_stream_topk_H) / max(1, len(per_stream_topk_H))
            topk_max    = math.log(model.codebooks[0].topk) if model.codebooks else 1.0
            cur_tau     = model.codebooks[0].topk_tau if model.codebooks else 0.0
            topk_str    = f"{mean_topk_H:.3f}/{topk_max:.2f}(τ{cur_tau:.2f})"

            # ── Active growth warmups ──────────────────────────────────
            if growth_warmup_queue:
                warmup_str = f"W{len(growth_warmup_queue)}@{min(growth_warmup_steps - (step - e['start_step']) for e in growth_warmup_queue)}"
            else:
                warmup_str = "-"

            # ── Raw gradient norm (before clipping) ───────────────────
            gnorm_str = f"{raw_norm.item():.2f}→{grad_clip:.2f}"

            # ── VRAM usage ────────────────────────────────────────────
            # memory_reserved() reports the full block PyTorch holds from the
            # driver (allocated tensors + cached-but-free pages).  This matches
            # what nvidia-smi and task manager show for the process, whereas
            # max_memory_allocated() only counts live tensor bytes (~half the
            # true figure once the caching allocator has warmed up).
            if torch.cuda.is_available():
                vram_str = f"{torch.cuda.memory_reserved() / 1024**3:.1f}G"
            else:
                vram_str = "CPU"

            # ── True throughput (tokens/sec) ──────────────────────────
            # step_secs is per-step average over this log window; safe_seq is
            # the sequence length actually used (clamped to corpus size).
            tok_per_sec = (batch_size * safe_seq) / max(step_secs, 1e-6)
            thr_str = f"{tok_per_sec / 1000:.1f}k"

            # ── Tokens seen / epoch equivalent ────────────────────────
            seen_m  = total_tokens_seen / 1_000_000
            epochs  = total_tokens_seen / max(1, n_train)
            seen_str = f"{seen_m:.1f}M({epochs:.1f}x)"

            # ── Per-stream alpha range (min/max bypass) ────────────────
            alphas  = [0.5 * torch.sigmoid(cb.residual_alpha_raw).item()
                       for cb in model.codebooks]
            a_min, a_max = min(alphas), max(alphas)
            alpha_str = f"μ{mean_alpha:.3f}[{a_min:.2f}–{a_max:.2f}]"

            print(
                f"Step {step:5d}/{steps} | "
                f"CE {loss_task.item():.4f} | "
                f"PPL {ppl:.2f} | "
                f"VQ {vq_val:.4f}(β{vq_beta:.2f},cc{model.codebooks[0].commitment_cost:.3f}) | "
                f"MetaCB Act{meta_active_ratio:.2f} H{meta_ema_H:.2f} β{meta_vq_beta:.2f} cc{model.meta_codebook.commitment_cost:.3f} TopKH{meta_topk_H:.2f} | "
                f"Orth {orth_loss.item():.2e}(λ{orth_lambda:.2e},ρ{corr_str}) | "
                f"GKL {gate_kl_val:.3f}(γ{gate_gamma:.4f}) | "
                f"SymD μ{sym_dec_val:.3f} ↓{worst_str} | "
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
            # ── Per-stream gate detail (every 5th log or on stream change) ──
            if step % (log_every * 5) == 0 or step == steps - 1:
                print(f"  Gates: {gate_str}")
                if per_stream_topk_H:
                    topkh_detail = " ".join(
                        f"{names_abbrev[i]}={per_stream_topk_H[i]:.2f}"
                        for i in range(len(per_stream_topk_H))
                    )
                    print(f"  TopK-H/stream: {topkh_detail}")

        # ── Validation + dynamic dropout + weight_decay ───────────────
        if val_data is not None and step > 0 and step % val_every == 0:
            model.eval()
            # Average over several batches to get a stable val CE estimate.
            # A single batch at batch_size=16, seq_len~30 is only ~480 tokens —
            # high variance, so "✓ best" would fire on lucky draws rather than
            # genuine improvement.  8 batches → ~3840 tokens, variance ~3× lower.
            _val_batches = 8
            _val_losses  = []
            with torch.no_grad():
                for _ in range(_val_batches):
                    xv, yv = _random_batch(val_data, batch_size, safe_seq)
                    lv, *_ = model(xv)
                    _val_losses.append(criterion(lv.view(-1, vocab_size), yv.view(-1)).item())
            val_ce = sum(_val_losses) / len(_val_losses)
            marker = ""
            if val_ce < best_val_loss:
                best_val_loss = val_ce
                torch.save(model.state_dict(), best_checkpoint)
                marker = "  ✓ best"

            # ── Val-CE plateau LR reduction ──────────────────────────
            # Worsening val CE (overfitting signal): reduce LR by 3%.
            # Plateau (no improvement): reduce LR by 1%.
            # NOTE: best_val_loss is already updated above, so we must use
            # strict > here — >= would fire even on a new best (val_ce == best_val_loss).
            if step > lr_warmup and step > vel_min_step:
                if val_ce > best_val_loss * 1.02:  # actively getting worse
                    for g in optimizer.param_groups:
                        g["lr"] = max(lr_min, g["lr"] * 0.97)
                    lr_ceiling = optimizer.param_groups[0]["lr"]  # prevent velocity from undoing this
                    vel_last_action = step                        # reset cooldown timer
                elif val_ce > best_val_loss:  # plateau (strictly worse than best, but within 2%)
                    for g in optimizer.param_groups:
                        g["lr"] = max(lr_min, g["lr"] * 0.99)
                    lr_ceiling = optimizer.param_groups[0]["lr"]  # prevent velocity from undoing this
                    vel_last_action = step                        # reset cooldown timer

            # ── 8. Dynamic dropout ────────────────────────────────────
            if last_train_ce is not None:
                ratio = last_train_ce / max(val_ce, 1e-8)
                if ratio > overfit_threshold:
                    current_dropout = current_dropout * (
                        1.0 + dropout_lr * (ratio - overfit_threshold))
                else:
                    current_dropout = current_dropout * (1.0 - dropout_lr * 0.5)
                current_dropout = max(dropout_min, min(dropout_max, current_dropout))
                model.decoder.set_dropout(current_dropout)
                model.cross_stream_attn.drop.p = current_dropout
                model.pre_vq_cross_stream_attn.drop.p = current_dropout

                # ── 14. Dynamic weight_decay ──────────────────────────
                if ratio > overfit_threshold:
                    weight_decay_val = weight_decay_val * (
                        1.0 + weight_decay_lr * (ratio - overfit_threshold))
                else:
                    weight_decay_val = weight_decay_val * (1.0 - weight_decay_lr * 0.5)
                weight_decay_val = max(weight_decay_min, min(weight_decay_max, weight_decay_val))
                for g in optimizer.param_groups:
                    g["weight_decay"] = weight_decay_val

                # ── 18. seq_len_max curriculum by improvement rate ────
                # Unlock next stage when val CE improvement per step has
                # plateaued at the current seq_len, rather than at a fixed
                # absolute CE threshold.  This adapts to any corpus difficulty.
                if seq_len_stage_idx < len(seq_len_stages) - 1:
                    # Force unlock after 10 val checks at same stage (prevents getting stuck)
                    if len(seq_len_val_ce_history) >= 10:
                        seq_len_stage_idx += 1
                        seq_len_max = seq_len_stages[seq_len_stage_idx]
                        seq_len_val_ce_history =[]
                        seq_len_improvement_ema = 0.0
                        print(f"  ↑ seq_len_max unlocked → {seq_len_max}  (max val checks reached)")
                    elif seq_len_val_ce_history:
                        prev_ce = seq_len_val_ce_history[-1]
                        improvement = prev_ce - val_ce   # positive = improving
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
                f"PPL {math.exp(min(val_ce, 20)):.2f}  "
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
        _, symbolic_indices, _, _, _, _, _, mean_H, _ps_active, *_ = model(context)
    active_ratio = sum(_ps_active) / len(_ps_active)

    last_tok = symbolic_indices[0, -1]  # [N, K] — top-K symbol indices per stream

    print(f"\nSEED PROMPT:\n{'-'*40}\n{seed_text}\n{'-'*40}")
    print("\nInternal submodality representation of the final token (Top-K routing):\n")
    for i, name in enumerate(model.sub_encoder.submodality_names):
        stream_syms = last_tok[i]  # [K] indices
        # Reconstruct approximate weights: uniform display (true weights not stored at eval)
        k = stream_syms.shape[0]
        sym_str = "  ".join(f"#{stream_syms[k_].item():03d}" for k_ in range(k))
        print(f"  {name:12s} → {sym_str}")
    print(f"\n  H_ema (smoothed entropy): {mean_H:.3f}  |  Active: {active_ratio*100:.1f}%")

    # ── 2. Generation ─────────────────────────────────────────────────
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
    """
    Extracts the learned submodality gating weights for specific words and
    renders a heatmap showing how the model routes different semantic
    categories (emotion, space, light, logic) through its five submodality
    streams.

    The gate is computed directly from the token embedding — no forward pass
    through the VQ codebooks or decoder is needed, so this is fast even for
    large word lists.

    Args:
        model:     trained SubmodalitySymbolicLanguageModel
        tokenizer: tiktoken encoding (GPT-2)
        device:    torch device the model lives on
        save_path: if given, save the figure to this path instead of
                   (or in addition to) showing it interactively
    """
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

    # Words organised by semantic category.  Leading space matters for GPT-2
    # tokenisation: " love" and "love" tokenise differently.
    word_categories = {
        "Emotion / Affect":["love", "hate", "rage", "grief", "joy",       "fear"],
        "Spatial / Distance":["far",  "near", "above", "behind", "within",  "beyond"],
        "Visual / Light":["bright", "dark", "shadow", "glow", "shining", "dim"],
        "Abstract / Logic":   ["therefore", "because", "reason", "logic",    "however", "truth"],
        "Size / Magnitude":   ["vast", "tiny", "enormous", "massive", "little", "great"],
        "Position / Order":["first", "last", "before", "after", "middle",  "finally"],
    }

    all_words:   list[str]        =[]
    all_weights: list[np.ndarray] = []
    category_boundaries: list[int] =[]   # row indices where a new category starts

    with torch.no_grad():
        for category, words in word_categories.items():
            category_boundaries.append(len(all_words))
            for word in words:
                token_ids    = tokenizer.encode(" " + word)
                token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

                # Gate depends only on the embedding — no VQ / decoder needed.
                embed     = model.embedding(token_tensor)                      # [1, S, D]
                gates     = F.softmax(model.submodality_gate(embed), dim=-1)  # [1, S, 5]
                mean_gate = gates.mean(dim=1).squeeze(0).cpu().numpy()        # [5]

                all_words.append(word)
                all_weights.append(mean_gate)

    weight_matrix      = np.array(all_weights)                # [N_words, 5]
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

    # Draw white separator lines between categories.
    # Skip index 0 (start of the first category — no line needed there).
    for row in category_boundaries[1:]:
        ax.axhline(row, color="white", lw=2)

    # Category labels in the right margin
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
    """
    Scans up to 3,000 unique tokens from the corpus and ranks each one by
    its gate weight for each submodality stream.  The top-k words per stream
    are printed, revealing what semantic concept each stream has gravitated
    toward organically — without any hand-labelled supervision.

    Only words ≥ 4 alphabetic characters are included to filter punctuation
    and short function words that would otherwise dominate every bucket.
    """
    print("\n" + "=" * 60)
    print("  DECODING AI PSYCHOLOGY (AUTO-LABELER)...")
    print("=" * 60)

    if not os.path.exists(filepath):
        print(f"[!] Cannot find {filepath} to run Auto-Labeler.")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Sample up to 3,000 unique tokens from the corpus so the labeller is
    # fast but representative.  random.sample without replacement ensures
    # no duplicates even on small corpora.
    unique_tokens = list(set(tokenizer.encode(text)))
    all_tokens    = random.sample(unique_tokens, min(3000, len(unique_tokens)))

    bucket_activations = {i:[] for i in range(model.NUM_SUBMODALITIES)}

    model.eval()
    with torch.no_grad():
        for tok in all_tokens:
            word = tokenizer.decode([tok]).strip()

            # Filter: only meaningful words — no punctuation, numbers, or
            # short prepositions that carry no submodality signal.
            if len(word) < 4 or not word.isalpha():
                continue

            tok_tensor   = torch.tensor([[tok]], dtype=torch.long).to(device)
            embed        = model.embedding(tok_tensor)                       # [1, 1, D]
            gates        = F.softmax(model.submodality_gate(embed), dim=-1) #[1, 1, 5]
            gate_weights = gates.squeeze().cpu().numpy()                     # [5]

            for i in range(model.NUM_SUBMODALITIES):
                bucket_activations[i].append((word, gate_weights[i]))

    submodality_names = model.sub_encoder.submodality_names

    for i in range(model.NUM_SUBMODALITIES):
        # Sort by gate weight descending, deduplicate case-insensitively.
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
# SYMBOL MUTUAL INFORMATION DIAGNOSTIC
# ==========================================================

def symbol_mutual_information(
    model:     "SubmodalitySymbolicLanguageModel",
    tokenizer,
    device:    torch.device,
    n_samples: int = 8000,
    seq_len:   int = 64,
) -> None:
    """
    Estimates I(symbol ; token) per stream using empirical discrete MI.

    For each submodality stream we ask: does knowing which VQ symbol was
    chosen tell us anything about which token was being processed?

    High MI (> 0.7)  → symbols carve meaningful semantic territory.
                       Meta-codebook experiment: GO.
    Mid  MI (0.4–0.7)→ some structure; codebook may still be forming.
                       Meta-codebook: WAIT for another run.
    Low  MI (< 0.4)  → symbols are nearly random given the token.
                       Meta-codebook would amplify noise.

    This is a diagnostic, not a training signal.  We use a random sample
    of the training corpus to build the joint (token, symbol) histogram,
    then compute:

        MI = Σ_{t,s} P(t,s) log[ P(t,s) / (P(t) P(s)) ]

    Because vocab is large (~50k) we cap at the top-K tokens by frequency
    to keep the estimate tractable and reduce sparsity bias.
    """
    import math, collections

    print("\n" + "=" * 60)
    print("  SYMBOL MUTUAL INFORMATION DIAGNOSTIC")
    print("=" * 60)
    print("  Estimating MI(symbol ; token) per submodality stream…")
    print(f"  Sampling {n_samples} tokens from corpus.\n")

    model.eval()

    # ── Build token corpus sample ──────────────────────────────────────
    if hasattr(tokenizer, "encode"):
        raw_text = open("data.txt", "r", encoding="utf-8", errors="replace").read()
        all_ids  = tokenizer.encode(raw_text)
    else:
        raw_text = open("data.txt", "r", encoding="utf-8", errors="replace").read()
        all_ids  = tokenizer.encode(raw_text)

    all_ids = list(all_ids)
    if len(all_ids) < seq_len + 1:
        print("  ⚠ Corpus too small for MI estimate — skipping.")
        return

    # Random subsequences
    rng       = torch.Generator()
    rng.manual_seed(42)
    starts    = torch.randint(0, len(all_ids) - seq_len, (n_samples // seq_len + 1,),
                              generator=rng).tolist()

    sample_ids = []
    for s in starts:
        sample_ids.extend(all_ids[s : s + seq_len])
        if len(sample_ids) >= n_samples:
            break
    sample_ids = sample_ids[:n_samples]

    # ── Collect (token, symbol_per_stream) pairs ───────────────────────
    # joint[stream] = Counter{(token_id, symbol_idx): count}
    N      = model.NUM_SUBMODALITIES
    joint  = [collections.Counter() for _ in range(N)]
    tok_ct = collections.Counter()

    batch  = 32
    with torch.no_grad():
        for b in range(0, len(sample_ids) - seq_len, batch * seq_len):
            chunk = sample_ids[b : b + batch * seq_len]
            if len(chunk) < seq_len:
                break
            x = torch.tensor(chunk, dtype=torch.long, device=device)
            x = x.view(-1, seq_len)          # [B, T]

            # Forward through sub_encoder + pre_vq_cross + pre_vq_norm + codebook
            embed        = model.embedding(x)                     # [B, T, D]
            sub_features = model.sub_encoder(embed)               # [B, T, N, D]
            B_f, T_f, N_f, D_f = sub_features.shape
            sub_features = model.pre_vq_cross_stream_attn(
                sub_features.reshape(B_f * T_f, N_f, D_f)
            ).reshape(B_f, T_f, N_f, D_f)

            toks_flat = x.reshape(-1).cpu().tolist()              # [B*T]
            tok_ct.update(toks_flat)

            for i in range(min(N, len(model.codebooks))):
                normed = model.pre_vq_norms[i](sub_features[:, :, i, :])  # [B, T, D]
                flat   = normed.reshape(-1, model.codebooks[i].embedding_dim)
                flat_s = F.normalize(flat, dim=1)
                cb_s   = F.normalize(model.codebooks[i].codebook.weight, dim=1)
                dists  = torch.cdist(flat_s, cb_s, p=2)
                syms   = dists.argmin(dim=1).cpu().tolist()       # [B*T]
                for tok, sym in zip(toks_flat, syms):
                    joint[i][(tok, sym)] += 1

    # ── Compute MI per stream ──────────────────────────────────────────
    total      = sum(tok_ct.values())
    p_tok      = {t: c / total for t, c in tok_ct.items()}

    results    = []
    for i in range(N):
        stream_total = sum(joint[i].values())
        if stream_total == 0:
            results.append((model.sub_encoder.submodality_names[i], 0.0))
            continue

        # P(symbol) marginal
        sym_ct = collections.Counter()
        for (_, sym), cnt in joint[i].items():
            sym_ct[sym] += cnt
        p_sym  = {s: c / stream_total for s, c in sym_ct.items()}

        # MI = Σ P(t,s) log[P(t,s) / P(t)P(s)]
        mi = 0.0
        for (tok, sym), cnt in joint[i].items():
            p_ts = cnt / stream_total
            p_t  = p_tok.get(tok, 1e-12)
            p_s  = p_sym.get(sym, 1e-12)
            mi  += p_ts * math.log(p_ts / (p_t * p_s) + 1e-12)

        results.append((model.sub_encoder.submodality_names[i], max(0.0, mi)))

    # ── Report ─────────────────────────────────────────────────────────
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
        advice  = "       Meta-codebook experiment is well-motivated."
    else:
        verdict = "HOLD ✗  — level-1 symbols need more training first."
        advice  = "       Run more steps or investigate low-MI streams before adding meta-VQ."

    print(f"\n  Meta-codebook decision: {verdict}")
    print(f"  {advice}")
    print("=" * 60 + "\n")

    model.train()



if __name__ == "__main__":
    # ── Choose your tokenizer here ────────────────────────────────────
    # "tiktoken" — GPT-2 BPE, vocab ~50k, no training needed (default)
    # "word"     — word-level, vocab built from corpus (~5–20k)
    # "bpe"      — BPE trained on corpus, vocab size set by bpe_vocab_size
    TOKENIZER_MODE = "bpe"         # ← "tiktoken" | "word" | "bpe"
    BPE_VOCAB_SIZE = None         # None = auto-calculate from corpus; set an int to override

    # 1. Train the model
    trained_model, tokenizer, s_len = train_model(
        tokenizer_mode = TOKENIZER_MODE,
        bpe_vocab_size = BPE_VOCAB_SIZE,
    )

    # 2. Test text generation
    evaluate_and_generate(trained_model, tokenizer, s_len)

    # 3. RUN THE MIND-READER VISUALIZATION!
    visualize_submodality_mind_reader(trained_model, tokenizer, device)

    # 4. RUN THE AUTO-LABELER! (prints after you close the heatmap window)
    auto_label_buckets(trained_model, tokenizer, device)

    # 5. SYMBOL MUTUAL INFORMATION — meta-codebook GO / HOLD decision
    symbol_mutual_information(trained_model, tokenizer, device)

    input("\nScript complete. Press Enter to exit...")
