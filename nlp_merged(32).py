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
  [🟠] orth_weight capped at 0.5
         A full sigmoid asymptotes to 1.0, so in runs > 1k steps orth_loss
         would eventually dominate CE loss.  0.5 * sigmoid(...) keeps the
         orthogonality pressure bounded at half the λ ceiling.
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
         Cov(fᵢ, fⱼ) ≈ 0 — genuine decorrelation, not just angular bias.
  [🔴] Sigmoid reparameterisation of residual_alpha
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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Causal self-attention (pre-norm).
        # PyTorch's MHA behaviour for is_causal=True varies across versions:
        # some require an explicit attn_mask alongside the flag, others don't.
        # Passing both is the correct cross-version approach — the mask is
        # generated once per sequence length in CausalTransformerDecoder and
        # reused across all blocks, so there is no extra allocation per block.
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=attn_mask,
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
        # Cache masks by (seq_len, device_str) so we only allocate once per
        # unique length.  Sequence length varies with curriculum, so without
        # caching we'd allocate a new tensor on every forward pass.
        self._mask_cache: dict = {}

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Upper-triangular −∞ mask cast to the model's active dtype."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )

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
        self._mask_cache.clear()   # seq-len masks are invalidated by structural change
        return new_block

    def set_dropout(self, p: float) -> None:
        """
        Update dropout probability in all blocks at runtime.
        Called during training when overfitting is detected.
        Touches: attention residual drop, FF drop, and MHA internal drop.
        """
        p = float(max(0.0, min(0.5, p)))
        for block in self.blocks:
            block.drop.p = p
            for m in block.ff.modules():
                if isinstance(m, nn.Dropout):
                    m.p = p
            block.attn.dropout = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        # Cache key includes device and dtype so mixed-precision and
        # multi-device runs each get their own correctly-typed mask.
        cache_key = (seq_len, x.device, x.dtype)
        if cache_key not in self._mask_cache:
            self._mask_cache[cache_key] = self.make_causal_mask(seq_len, x.device, x.dtype)
        mask = self._mask_cache[cache_key]
        for block in self.blocks:
            x = block(x, mask)
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
        commitment_cost: float = 0.25,  # raised from 0.15 → 0.25: forces encoder
    ):                                  # to respect codebook geometry more strongly
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

        # Residual bypass: quantized_out = Q(z) + α(a) · z
        # α is reparameterised as 0.5·sigmoid(a) where a is the raw parameter.
        # This is smoother than clamping: clamp truncates gradients to zero at
        # the boundary (∂L/∂a = 0 when a ≥ clamp_max), whereas sigmoid
        # always produces a nonzero gradient that decays gracefully toward
        # the asymptote.  The 0.5 factor preserves the [0, 0.5] range from
        # before so the discrete component always dominates the bypass.
        self.residual_alpha_raw = nn.Parameter(torch.tensor(0.0))

        # Moving-average entropy estimate — smooths noisy batch statistics so
        # the entropy regularisation signal doesn't oscillate step-to-step.
        # Registered as a buffer so it is saved/loaded with the checkpoint and
        # moved to the correct device automatically, but is NOT a parameter.
        self.register_buffer("ema_entropy", torch.tensor(0.0, dtype=torch.float32))

        # Entropy target: 0.9 · log(K) rather than full log(K).
        # Language symbol usage is naturally non-uniform — forcing perfect
        # uniformity slightly over-regularises and fights the clustering
        # pressure from VQ loss.  0.9 allows ~10% headroom for natural
        # frequency skew while still strongly resisting collapse.
        self.H_target = 0.9 * math.log(num_symbols)

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

        # torch.cdist computes pairwise Euclidean distances with better
        # numerical conditioning than the manual expansion
        #   ‖a − b‖² = ‖a‖² + ‖b‖² − 2aᵀb
        # which suffers from catastrophic cancellation when ‖a‖ ≈ ‖b‖.
        # It also dispatches to optimised GPU kernels automatically.
        distances = torch.cdist(flat, self.codebook.weight, p=2)  # [N, V]

        indices   = torch.argmin(distances, dim=1)             # [N]
        quantized = self.codebook(indices).view_as(features)

        # VQ-VAE loss
        e_loss  = F.mse_loss(quantized.detach(), features)    # encoder → codebook
        q_loss  = F.mse_loss(quantized, features.detach())    # codebook → encoder
        vq_loss = q_loss + self.commitment_cost * e_loss

        # Straight-through estimator
        quantized_st = features + (quantized - features).detach()

        # Residual bypass: α = 0.5·σ(a) maps raw param a ∈ ℝ → α ∈ (0, 0.5).
        # sigmoid(0.0) = 0.5 → α_init ≈ 0.25 (moderate bypass at start).
        # Smooth gradient throughout: no truncation at the boundary.
        alpha     = 0.5 * torch.sigmoid(self.residual_alpha_raw)
        quantized = quantized_st + alpha * features

        # Codebook usage stats — all computed without gradient.
        with torch.no_grad():
            one_hot      = F.one_hot(indices, self.num_symbols).float()  # [N, V]
            probs        = one_hot.mean(0)                               # [V]
            batch_H      = -(probs * torch.log(probs + 1e-10)).sum()    # scalar

            # In-place EMA: new = (1 − α) · old + α · batch  (α = 0.1)
            # In-place ops (.mul_ / .add_) update the registered buffer
            # without rebinding the Python name, which keeps TorchScript
            # compilation and DistributedDataParallel buffer sync correct.
            self.ema_entropy.mul_(0.9).add_(batch_H * 0.1)

            # Scale-invariant squared penalty: (H_ema / H_target − 1)²
            # The old form (H_target − H_ema)² has magnitude proportional to
            # H_target², so larger codebooks (higher K, higher log K) produced
            # larger raw penalties for the same fractional divergence.  The
            # relative form normalises by H_target so the signal is always in
            # the same range regardless of num_symbols, and the penalty is zero
            # only when H_ema == H_target (perfect uniform usage).
            entropy_loss = (self.ema_entropy / self.H_target - 1.0).pow(2)

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

    SUBMODALITY_NAMES  = ["Brightness", "Distance", "Focus", "Color", "Perspective", "Size", "Position"]
    NUM_SUBMODALITIES  = 7

    def __init__(self, d_model: int, orthogonality_lambda: float = 0.01):
        # Raised from 1e-4 → 0.01.  The old value divided by D²=16384 which
        # made Orth ~1e-9 — genuinely invisible to the optimiser.  0.01 with
        # .mean() instead of .sum()/D² puts Orth in the 0.01–0.1 range where
        # it actually pushes streams apart.
        super().__init__()
        self.extract_brightness  = nn.Linear(d_model, d_model)
        self.extract_distance    = nn.Linear(d_model, d_model)
        self.extract_focus       = nn.Linear(d_model, d_model)
        self.extract_color       = nn.Linear(d_model, d_model)
        self.extract_perspective = nn.Linear(d_model, d_model)
        self.extract_size        = nn.Linear(d_model, d_model)
        self.extract_position    = nn.Linear(d_model, d_model)
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
        mats = [
            getattr(self, f"extract_{n.lower()}").weight
            for n in self.SUBMODALITY_NAMES
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
            sub_features: [B, T, 5, D] — output of VisualSubmodalityEncoder.forward()

        Returns: scalar tensor, differentiable w.r.t. projection weights
        """
        B, T, _, D = sub_features.shape
        N          = B * T                   # total token count in the batch
        loss       = torch.tensor(0.0, device=sub_features.device)

        for i in range(self.NUM_SUBMODALITIES):
            fi = sub_features[:, :, i, :].reshape(N, D)      # [N, D]
            # Mean-center then L2-normalise.  Without centering, any shared
            # mean bias leaks into the cross-gram: fi.t() @ fj picks up the
            # outer product of the means, so the penalty measures angular
            # distance from the origin rather than true covariance.  Centering
            # first makes the cross-gram estimate the sample cross-covariance,
            # so the penalty is zero only when the two streams are truly
            # uncorrelated (Cov(fᵢ, fⱼ) ≈ 0) across the training batch.
            fi = fi - fi.mean(0, keepdim=True)                # remove mean bias
            fi = F.normalize(fi, dim=1)                       # unit norm → cosine
            for j in range(i + 1, self.NUM_SUBMODALITIES):
                fj = sub_features[:, :, j, :].reshape(N, D)  # [N, D]
                fj = fj - fj.mean(0, keepdim=True)
                fj = F.normalize(fj, dim=1)
                # Cross-gram: [D, D]; .mean() normalises by D² implicitly and
                # keeps the penalty in a stable range regardless of dimension.
                # Previously used .sum()/(D*D) which was identical mathematically
                # but confusing — .mean() is clearer and equally correct.
                cross_gram = (fi.t() @ fj) / N               # [D, D]
                loss = loss + cross_gram.pow(2).mean()

        return self.orthogonality_lambda * loss

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
        # Instance-level lists shadow the class-level defaults after first mutation
        if not isinstance(self.SUBMODALITY_NAMES, list):
            self.SUBMODALITY_NAMES = list(self.SUBMODALITY_NAMES)
        self.SUBMODALITY_NAMES.append(name)
        self.NUM_SUBMODALITIES = len(self.SUBMODALITY_NAMES)
        return new_lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]  →  [B, T, N, D]  where N = NUM_SUBMODALITIES"""
        streams = [getattr(self, f"extract_{n.lower()}")(x)
                   for n in self.SUBMODALITY_NAMES]
        return torch.stack(streams, dim=2)


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
            → Learned gated aggregation         (replaces naïve mean)
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
            valid = [h for h in range(1, d_model + 1) if d_model % h == 0]
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

        # FIX 1 (RED): custom causal decoder, not TransformerEncoder
        self.decoder = CausalTransformerDecoder(d_model, n_heads, num_layers, dropout)

        self.output_head = nn.Linear(d_model, vocab_size)

        # Stream activity tracking — inactive streams get ~zero gate weight
        # via a mask applied after softmax.  Inactive streams still run through
        # VQ (avoids tensor surgery) but contribute nothing to composite.
        self.stream_active: list[bool] = [True] * self.NUM_SUBMODALITIES
        self._update_stream_mask()

    def _update_stream_mask(self) -> None:
        """Rebuild the float mask tensor from stream_active list."""
        mask = torch.tensor(
            [1.0 if a else 0.0 for a in self.stream_active],
            dtype=torch.float32
        )
        # Register as buffer so it moves with .to(device) automatically
        self.register_buffer("stream_mask", mask)

    def add_submodality(
        self, name: str, device: torch.device, optimizer: torch.optim.Optimizer,
        lr: float, weight_decay: float, num_symbols: int | None = None
    ) -> None:
        """
        Dynamically grow the model by one new submodality stream at runtime.

        Steps:
          1. Add projection linear to sub_encoder
          2. Add new VQ codebook
          3. Add new pre-VQ LayerNorm
          4. Grow gate linear (d_model → N+1), preserving existing weights
          5. Add new params to optimizer as a new param group
          6. Update NUM_SUBMODALITIES and stream_active
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
            # New stream starts with small negative bias → gate ~1/N after softmax
            nn.init.zeros_(new_gate.weight[self.NUM_SUBMODALITIES:])
            new_gate.bias.data[self.NUM_SUBMODALITIES] = 0.0
        self.submodality_gate = new_gate

        # 5. add new params to optimizer
        new_params = (
            list(new_lin.parameters())
            + list(new_cb.parameters())
            + list(new_norm.parameters())
            + list(new_gate.parameters())   # full gate replaces old, add all
        )
        # Remove old gate params from existing groups to avoid duplicates
        old_gate_ids = {id(p) for p in old_gate.parameters()}
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"]
                               if id(p) not in old_gate_ids]
        optimizer.add_param_group({
            "params":       new_params,
            "lr":           lr,
            "weight_decay": weight_decay,
        })

        # 6. state
        self.NUM_SUBMODALITIES += 1
        self.stream_active.append(True)
        self._update_stream_mask()
        print(f"  ✚ Stream added: '{name}'  "
              f"(total active: {sum(self.stream_active)}/{self.NUM_SUBMODALITIES})")

    def deactivate_submodality(self, idx: int) -> None:
        """
        Deactivate stream `idx` — gate contribution masked to zero.
        Params remain in model/optimizer (avoids optimizer surgery).
        The stream can be inspected post-hoc but has no effect on output.
        """
        name = self.sub_encoder.SUBMODALITY_NAMES[idx]
        self.stream_active[idx] = False
        self._update_stream_mask()
        active = sum(self.stream_active)
        print(f"  ✖ Stream deactivated: '{name}' (idx {idx})  "
              f"({active}/{self.NUM_SUBMODALITIES} active)")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T] token IDs

        Returns:
            logits:          [B, T, vocab_size]
            symbolic_indices:[B, T, 5]
            total_vq_loss:   scalar — VQ commitment loss
            total_ent_loss:  scalar — sum of (H_target − H_ema)² across codebooks
            feat_orth_loss:  scalar — feature-space orthogonality penalty (unscaled)
            mean_ema_H:      float  — mean smoothed entropy (for logging)
            mean_active:     float  — mean active-symbol ratio (for logging)
        """
        embed        = self.embedding(x)          # [B, T, D]
        sub_features = self.sub_encoder(embed)    # [B, T, 5, D]

        symbolic_vectors = []
        symbolic_indices = []
        total_vq_loss    = 0.0
        total_ent_loss   = 0.0
        total_ema_H      = 0.0
        total_active     = 0.0

        for i in range(self.NUM_SUBMODALITIES):
            # Normalise feature scale before quantisation — prevents codebook
            # entries from being permanently unreachable due to scale mismatch.
            normed_stream = self.pre_vq_norms[i](sub_features[:, :, i, :])
            quantized, indices, vq_loss, ent_loss, ema_H, active_ratio = self.codebooks[i](
                normed_stream                     # [B, T, D]  (scale-stabilised)
            )
            symbolic_vectors.append(quantized)
            symbolic_indices.append(indices)
            total_vq_loss  += vq_loss
            total_ent_loss += ent_loss
            total_ema_H    += ema_H
            total_active   += active_ratio

        # Learned gating: softmax weights over the N submodality streams,
        # conditioned on the token embedding.  [B, T, N] → [B, T, N, 1]
        # so it broadcasts cleanly over the D dimension.
        gate_weights = F.softmax(self.submodality_gate(embed), dim=-1)  # [B, T, N]

        # Apply stream activity mask: inactive streams → zero, then renormalise
        # so active streams still sum to 1.  This happens after softmax so
        # the gate gradient still flows to active streams normally.
        if not all(self.stream_active):
            gate_weights = gate_weights * self.stream_mask          # zero inactive
            gate_weights = gate_weights / (gate_weights.sum(-1, keepdim=True) + 1e-10)
        sym_stack    = torch.stack(symbolic_vectors, dim=2)              # [B, T, 5, D]
        composite    = (sym_stack * gate_weights.unsqueeze(-1)).sum(dim=2)  # [B, T, D]

        # Gate entropy penalty: KL(gate || uniform) per token, averaged over
        # the batch.  Without this, softmax positive feedback can cause the
        # gate to collapse onto one dominant stream early in training —
        # gradients to the other four streams decay, their codebooks stagnate,
        # and the multi-stream structure becomes effectively single-stream.
        # KL(g || u) = log(5) + Σᵢ gᵢ log gᵢ, minimised when g is uniform.
        log_uniform  = math.log(self.NUM_SUBMODALITIES)
        gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(-1).mean()
        gate_kl      = log_uniform - gate_entropy          # ≥ 0, zero iff uniform

        # Feature-conditioned orthogonality — only needed during training.
        # At inference this O(25·N·D) computation is pure waste: the result
        # goes into orth_loss which is never used outside the training loop.
        # self.training is False during model.eval() and generate() calls.
        if self.training:
            feat_orth_loss = self.sub_encoder.feature_orthogonality_loss(sub_features)

            # Cross-stream symbol decorrelation: penalises streams for selecting
            # the same codebook entries.  Feature orthogonality pushes the
            # projection geometry apart; this pushes the discrete routing apart.
            # For each stream compute a soft usage distribution over V symbols
            # via negative distance (sharper than uniform; avoids one-hot issues).
            # Then penalise pairwise JS divergence — symmetric, bounded [0, log2].
            # We want streams to route *different* tokens to *different* symbols.
            # NOTE: no torch.no_grad() here — gradients must flow through
            # stream_probs back into pre_vq_norms and sub_encoder projections.
            # The previous version wrapped this in no_grad(), making sym_decor_loss
            # numerically present but gradient-dead — a silent no-op.
            sym_decor_loss = torch.tensor(0.0, device=x.device)
            stream_probs   = []
            for i in range(self.NUM_SUBMODALITIES):
                flat_i  = self.pre_vq_norms[i](
                    sub_features[:, :, i, :]).reshape(-1, self.d_model)
                dists_i = torch.cdist(flat_i, self.codebooks[i].codebook.weight, p=2)
                # Temperature-scaled soft assignment: softmax(-d / τ) where
                # τ = √D normalises for embedding dimension.
                # Without temperature, softmax(-distances) favours small-norm
                # embeddings — streams can inflate their norms to artificially
                # push all distances large, making every distribution look
                # uniform and gaming the JS penalty without genuine separation.
                # Dividing by √D makes the effective scale invariant to d_model.
                tau    = math.sqrt(self.d_model)
                soft_i = F.softmax(-dists_i / tau, dim=1).mean(0)  # [V]
                stream_probs.append(soft_i)
            # Pairwise JS divergence between stream usage distributions
            for i in range(self.NUM_SUBMODALITIES):
                for j in range(i + 1, self.NUM_SUBMODALITIES):
                    m = 0.5 * (stream_probs[i] + stream_probs[j])
                    js = 0.5 * (
                        (stream_probs[i] * torch.log(stream_probs[i] / (m + 1e-10) + 1e-10)).sum()
                      + (stream_probs[j] * torch.log(stream_probs[j] / (m + 1e-10) + 1e-10)).sum()
                    )
                    sym_decor_loss = sym_decor_loss + (math.log(2) - js)  # penalise low divergence
        else:
            feat_orth_loss = sub_features.new_zeros(1).squeeze()  # zero scalar
            sym_decor_loss = sub_features.new_zeros(1).squeeze()

        composite = self.pos_encoder(composite)
        decoder_out = self.decoder(composite)      # [B, T, D]
        logits      = self.output_head(decoder_out)

        return (
            logits,
            torch.stack(symbolic_indices, dim=2),          # [B, T, 5]
            total_vq_loss,
            total_ent_loss,
            feat_orth_loss,                                 # unscaled; sigmoid applied in loop
            gate_kl,                                        # KL(gate || uniform), unscaled
            sym_decor_loss,                                 # cross-stream symbol decorrelation
            gate_weights,                                   # [B, T, 5] — reused for balance_loss
            total_ema_H    / self.NUM_SUBMODALITIES,
            total_active   / self.NUM_SUBMODALITIES,
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
        # Clamp temperature to prevent logit explosion: at very low values
        # logits / temperature → ±∞, causing softmax to produce a one-hot
        # distribution or NaN.  0.5 is a safe lower bound for generation.
        temperature = max(0.5, temperature)
        for _ in range(max_new_tokens):
            context = input_ids[:, -seq_len:]
            with torch.no_grad():
                logits, *_ = self(context)
            logits = logits[:, -1, :] / temperature

            # Apply top-k by masking logits to -inf before softmax, rather
            # than zeroing probabilities after.  This avoids a renormalisation
            # discontinuity (zeroed probs don't sum to exactly 1 before the
            # divide) and lets softmax renormalise cleanly in one pass.
            if top_k and top_k > 0:
                k               = min(top_k, logits.size(-1))
                topk_vals, _    = torch.topk(logits, k)
                min_val         = topk_vals[:, -1].unsqueeze(1)
                logits          = logits.masked_fill(logits < min_val, float("-inf"))

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
    (e.g. "don't" → ["don", "'", "t"]).  Vocabulary is derived entirely
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
        vocab   = specials + [t for t, c in counts.most_common() if c >= min_freq]
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self._built   = True

    # ── encode / decode ───────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        unk_id = self.token2id[self.UNK]
        return [self.token2id.get(t, unk_id) for t in self._split(text)]

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id2token.get(i, self.UNK) for i in ids]
        # Re-join: attach punctuation to the preceding word without a space
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
        self.merges:  list[tuple[str, str]]  = []
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
        specials   = [self.PAD, self.UNK, self.BOS, self.EOS]
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
        symbols = list(word) + [self.EOW]
        for a, b in self.merges:
            merged = a + b
            i = 0
            new_syms = []
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
        ids    = []
        for word in _re.findall(r"\S+", text):
            for sym in self._tokenize_word(word):
                ids.append(self.token2id.get(sym, unk_id))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id2token.get(i, self.UNK) for i in ids]
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
                        clamp(unique_words × 1.5, 1000, 32000).
                        Smaller corpora get smaller vocabs automatically.

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
        # Auto vocab size: derived from corpus, capped based on backend speed.
        # Pure Python BPE is O(merges × vocab) — 32k merges on a large corpus
        # takes 30–60 minutes.  The HuggingFace C++ backend does 32k in seconds.
        # Detect which backend will be used and set the cap accordingly.
        try:
            import tokenizers as _hf_tok  # noqa: F401
            _fast_available = True
        except ImportError:
            _fast_available = False

        _cap = 32_000 if _fast_available else 8_000

        if bpe_vocab_size is None:
            unique_words   = len(set(_re.findall(r"\S+", text)))
            bpe_vocab_size = int(max(1000, min(_cap, unique_words * 1.5)))
            _cap_note      = f"cap {_cap:,} — {'fast C++ backend' if _fast_available else 'pure Python, install tokenizers for larger cap'}"
            print(f"  Tokenizer: BPE — auto vocab size = {bpe_vocab_size:,} "
                  f"(from {unique_words:,} unique words × 1.5, {_cap_note})")
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
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x  = torch.stack([data[i     : i + seq_len]     for i in ix]).to(device)
    y  = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
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
    steps          = 10000
    lr             = 3e-4
    lr_warmup      = 100
    weight_decay   = 0.01
    entropy_warmup = 300
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
    target_sym_decor  = 1.0
    sym_decor_lr      = 0.005
    sym_decor_min     = 0.001
    sym_decor_max     = 0.1

    # 5. vq_beta
    vq_beta           = 0.60
    target_active     = 0.85
    vq_beta_lr        = 0.003
    vq_beta_min       = 0.2
    vq_beta_max       = 1.5

    # 6. H_target annealing (num_symbols_ref set after tokenization)
    num_symbols_ref   = None
    h_target_max_frac = 1.0
    h_target_min_frac = 0.7

    # 7. Orth onset (VQ-triggered)
    orth_T0           = steps
    vq_unlock_thresh  = 1.0
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
    orth_lambda       = 0.01   # mirrors model.sub_encoder.orthogonality_lambda
    target_orth       = 0.05   # desired unscaled feat_orth_loss
    orth_lambda_lr    = 0.005
    orth_lambda_min   = 0.001
    orth_lambda_max   = 0.2

    # 11. entropy_beta — controller around H_ema vs H_target
    entropy_beta      = 0.01
    ent_beta_lr       = 0.005
    ent_beta_min      = 0.001
    ent_beta_max      = 0.1

    # 12. val_every — starts frequent, doubles on each LR reduction (up to 1000)
    val_every         = 200
    val_every_max     = 1000

    # 13. Dynamic stream management
    # Candidate names for new streams added at runtime
    candidate_streams  = ["Texture", "Motion", "Valence", "Agency",
                          "Causality", "Certainty", "Temporal"]
    next_candidate_idx = 0          # index into candidate_streams
    max_streams        = 12         # never exceed this many streams
    stream_check_every = 500        # steps between stream health checks
    # Per-stream gate EMA for detecting dead / saturated streams
    stream_gate_ema    = None       # initialised after model is built
    stream_gate_alpha  = 0.02       # slow EMA for stability
    stream_dead_thresh = 0.03       # gate EMA below this → candidate for deactivation
    stream_dead_patience = 1000     # must be dead for this many steps before removal
    stream_dead_counts   = None     # initialised after model is built
    stream_saturated_thresh = 0.90  # if all streams' gate EMA > this × (1/N), consider adding

    # ── Data ──────────────────────────────────────────────────────────
    train_data, val_data, tokenizer, vocab_size = load_data(
        filepath, tokenizer_mode, bpe_vocab_size
    )

    n_train = len(train_data)

    if n_train < 512 + 2:
        raise ValueError(
            f"Corpus too small: {n_train} tokens, need at least 514. "
            "Provide a larger data.txt."
        )

    # ── Dynamic num_symbols ───────────────────────────────────────────
    _raw_symbols  = vocab_size ** 0.6
    _pow2         = 2 ** round(math.log2(_raw_symbols))
    num_symbols   = int(max(64, min(2048, _pow2)))
    num_symbols_ref = num_symbols
    print(f"  num_symbols: {num_symbols}  "
          f"(vocab={vocab_size:,}^0.6 → {_raw_symbols:.0f} → pow2)")

    # ── Dynamic d_model ───────────────────────────────────────────────
    # Scale model width with corpus size: more data → richer representation.
    # Formula: n_train^0.25 × 8, rounded to nearest 64, clamped [128, 768].
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
    for h in [8, 4, 2, 1]:
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
    seq_len_max  = 512
    print(f"  seq_len_base: {seq_len_base}  seq_len_max: {seq_len_max}")

    # ── Model ─────────────────────────────────────────────────────────
    model = SubmodalitySymbolicLanguageModel(
        vocab_size  = vocab_size,
        d_model     = d_model,
        num_symbols = num_symbols,
        n_heads     = n_heads,
        num_layers  = num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  Device: {device}\n")

    # Initialise per-stream tracking state now that model exists
    stream_gate_ema   = [1.0 / model.NUM_SUBMODALITIES] * model.NUM_SUBMODALITIES
    stream_dead_counts = [0] * model.NUM_SUBMODALITIES

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # ── Dynamic LR ────────────────────────────────────────────────────
    # After warmup, LR is reduced automatically when training stalls.
    # An EMA of training CE is tracked; if it hasn't improved by at least
    # lr_reduce_threshold (fractional) over lr_patience steps, the LR is
    # multiplied by lr_reduce_factor.  This repeats until lr_min is reached.
    # More responsive than cosine (which decays regardless of actual progress)
    # and fully transparent — every reduction is printed with ↓ LR.
    lr_patience          = 300    # steps of no improvement before reducing
    lr_reduce_factor     = 0.5    # multiply LR by this on plateau
    lr_reduce_threshold  = 0.005  # minimum fractional improvement required
    lr_min               = lr / 100  # floor: never go below this

    criterion = nn.CrossEntropyLoss()

    best_val_loss    = float("inf")
    last_train_ce    = None
    grad_norm_ema    = grad_clip

    # Adaptive LR state
    ce_ema           = None
    ce_ema_alpha     = 0.05
    best_ce_ema      = float("inf")
    steps_since_imp  = 0
    lr_patience      = 300
    lr_reduce_factor = 0.5
    lr_reduce_threshold = 0.005
    lr_min           = lr / 100

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    print("Starting training…\n")

    for step in range(steps):

        # ── 1. Dynamic seq_len ────────────────────────────────────────
        seq_len = min(seq_len_max,
                      seq_len_base + int(math.log(step + 1) * 8))

        # Guard: corpus must be larger than the window we're sampling
        safe_seq = min(seq_len, len(train_data) - 2)

        # ── 2. Dynamic batch size ─────────────────────────────────────
        batch_size = int(max(batch_size_min,
                             min(batch_size_max, batch_size_tokens // safe_seq)))

        x, y = _random_batch(train_data, batch_size, safe_seq)

        # ── LR warmup ─────────────────────────────────────────────────
        if step < lr_warmup:
            for g in optimizer.param_groups:
                g["lr"] = lr * (step + 1) / lr_warmup

        # ── 6. H_target annealing ─────────────────────────────────────
        h_frac       = h_target_max_frac - (h_target_max_frac - h_target_min_frac) * (step / max(1, steps - 1))
        h_target_now = h_frac * math.log(num_symbols_ref)
        for cb in model.codebooks:
            cb.H_target = h_target_now

        # ── 10. Sync orth_lambda into model ───────────────────────────
        model.sub_encoder.orthogonality_lambda = orth_lambda

        optimizer.zero_grad()
        logits, _, vq_loss, ent_loss, feat_orth_loss, gate_kl, sym_decor_loss, \
            gate_weights, mean_H, active_ratio = model(x)
        loss_task = criterion(logits.view(-1, vocab_size), y.view(-1))
        last_train_ce = loss_task.item()

        # ── 7. Orth onset trigger ─────────────────────────────────────
        if not orth_unlocked and vq_loss.item() < vq_unlock_thresh:
            orth_T0       = step
            orth_unlocked = True
            print(f"  ✦ Orth unlocked at step {step}  "
                  f"(VQ {vq_loss.item():.4f} < {vq_unlock_thresh})")

        orth_weight    = 0.8 / (1.0 + math.exp(-(step - orth_T0) / orth_tau))
        entropy_weight = entropy_beta * min(1.0, step / max(1, entropy_warmup))
        orth_loss      = orth_weight * feat_orth_loss

        # ── 3. Dynamic gate_gamma ─────────────────────────────────────
        gate_kl_val = gate_kl.item()
        gate_gamma  = gate_gamma * (1.0 + gate_gamma_lr * (gate_kl_val - target_gate_kl))
        gate_gamma  = max(gate_gamma_min, min(gate_gamma_max, gate_gamma))
        gate_loss   = gate_gamma * gate_kl

        gate_mean    = gate_weights.mean(dim=(0, 1))       # [N]
        balance_loss = ((gate_mean - 1.0 / model.NUM_SUBMODALITIES) ** 2).sum()

        # ── 4. Dynamic sym_decor_lambda ───────────────────────────────
        sym_dec_val      = sym_decor_loss.item()
        sym_decor_lambda = sym_decor_lambda * (1.0 + sym_decor_lr * (sym_dec_val - target_sym_decor))
        sym_decor_lambda = max(sym_decor_min, min(sym_decor_max, sym_decor_lambda))

        # ── 5. Dynamic vq_beta ────────────────────────────────────────
        vq_beta = vq_beta * (1.0 + vq_beta_lr * (target_active - active_ratio))
        vq_beta = max(vq_beta_min, min(vq_beta_max, vq_beta))

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

        # ── 10. Dynamic orth_lambda ───────────────────────────────────
        orth_raw = feat_orth_loss.item()
        orth_lambda = orth_lambda * (1.0 + orth_lambda_lr * (orth_raw - target_orth))
        orth_lambda = max(orth_lambda_min, min(orth_lambda_max, orth_lambda))

        # ── 11. Dynamic entropy_beta ──────────────────────────────────
        ent_error    = h_target_now - mean_H      # positive = below target (needs more)
        entropy_beta = entropy_beta * (1.0 + ent_beta_lr * ent_error)
        entropy_beta = max(ent_beta_min, min(ent_beta_max, entropy_beta))

        total_loss = (loss_task
                      + vq_beta       * vq_loss
                      + entropy_weight * ent_loss
                      + orth_loss
                      + gate_loss
                      + gate_gamma    * balance_loss
                      + alpha_reg
                      + sym_decor_lambda * sym_decor_loss)

        total_loss.backward()

        # ── 2. Dynamic grad clip ──────────────────────────────────────
        raw_norm      = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm_ema = (1 - grad_norm_alpha) * grad_norm_ema + grad_norm_alpha * raw_norm.item()
        grad_clip     = max(grad_clip_min, min(grad_clip_max, grad_norm_ema * grad_clip_factor))

        optimizer.step()

        # ── Adaptive LR (post-warmup) ─────────────────────────────────
        if step >= lr_warmup:
            current_ce = loss_task.item()
            if ce_ema is None:
                ce_ema = current_ce
            ce_ema = (1 - ce_ema_alpha) * ce_ema + ce_ema_alpha * current_ce
            if ce_ema < best_ce_ema * (1 - lr_reduce_threshold):
                best_ce_ema     = ce_ema
                steps_since_imp = 0
            else:
                steps_since_imp += 1
            if steps_since_imp >= lr_patience:
                current_lr = optimizer.param_groups[0]["lr"]
                new_lr     = max(lr_min, current_lr * lr_reduce_factor)
                if new_lr < current_lr:
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    print(f"  ↓ LR  {current_lr:.2e} → {new_lr:.2e}")

                    # ── num_layers growth on LR reduction ─────────────
                    if num_layers < num_layers_max:
                        new_block = model.decoder.add_layer(device)
                        optimizer.add_param_group({
                            "params":       list(new_block.parameters()),
                            "lr":           new_lr,
                            "weight_decay": weight_decay,
                        })
                        num_layers += 1
                        print(f"  ↑ Layer added  (total: {num_layers})")

                    # ── val_every doubles on LR reduction ─────────────
                    val_every = min(val_every_max, val_every * 2)

                steps_since_imp = 0

        # ── 13. Dynamic stream management ────────────────────────────
        if step > 0 and step % stream_check_every == 0:
            gate_mean_np = gate_mean.detach().cpu().tolist()

            # Update per-stream gate EMA
            for i in range(model.NUM_SUBMODALITIES):
                stream_gate_ema[i] = (
                    (1 - stream_gate_alpha) * stream_gate_ema[i]
                    + stream_gate_alpha * gate_mean_np[i]
                )

            # Dead stream detection
            for i in range(model.NUM_SUBMODALITIES):
                if not model.stream_active[i]:
                    continue
                if stream_gate_ema[i] < stream_dead_thresh:
                    stream_dead_counts[i] += stream_check_every
                    if stream_dead_counts[i] >= stream_dead_patience:
                        model.deactivate_submodality(i)
                else:
                    stream_dead_counts[i] = 0

            # New stream: add if all active streams are saturated and budget allows
            n_active = sum(model.stream_active)
            if (n_active < max_streams
                    and next_candidate_idx < len(candidate_streams)
                    and n_active > 0):
                uniform = 1.0 / n_active
                active_emas = [stream_gate_ema[i]
                               for i in range(model.NUM_SUBMODALITIES)
                               if model.stream_active[i]]
                if all(e > stream_saturated_thresh * uniform for e in active_emas):
                    new_name = candidate_streams[next_candidate_idx]
                    next_candidate_idx += 1
                    # Extend tracking arrays for new stream
                    stream_gate_ema.append(1.0 / (model.NUM_SUBMODALITIES + 1))
                    stream_dead_counts.append(0)
                    model.add_submodality(
                        new_name, device, optimizer, lr, weight_decay, num_symbols
                    )

        # ── Logging ───────────────────────────────────────────────────
        if step % log_every == 0 or step == steps - 1:
            ppl = torch.exp(loss_task).item()
            n_active = sum(model.stream_active)
            print(
                f"Step {step:5d}/{steps} | "
                f"CE {loss_task.item():.4f} | "
                f"VQ {vq_loss.item():.4f}(β{vq_beta:.2f}) | "
                f"Orth {orth_loss.item():.2e}(λ{orth_lambda:.3f}) | "
                f"GKL {gate_kl_val:.3f}(γ{gate_gamma:.4f}) | "
                f"SymD {sym_dec_val:.3f}(λ{sym_decor_lambda:.4f}) | "
                f"Act {active_ratio*100:4.1f}% | "
                f"α {mean_alpha:.3f}(λ{alpha_reg_lambda:.4f}) | "
                f"H {mean_H:.2f}→{h_target_now:.2f}(β{entropy_beta:.4f}) | "
                f"PPL {ppl:.2f} | "
                f"BS {batch_size} | "
                f"Clip {grad_clip:.2f} | "
                f"do {current_dropout:.2f} | "
                f"L {num_layers} | "
                f"S {n_active}/{model.NUM_SUBMODALITIES} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e}"
            )

        # ── Validation + dynamic dropout ──────────────────────────────
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

            # ── 8. Dynamic dropout ────────────────────────────────────
            if last_train_ce is not None:
                ratio = last_train_ce / max(val_ce, 1e-8)
                if ratio > overfit_threshold:
                    current_dropout = current_dropout * (1.0 + dropout_lr * (ratio - overfit_threshold))
                else:
                    current_dropout = current_dropout * (1.0 - dropout_lr * 0.5)
                current_dropout = max(dropout_min, min(dropout_max, current_dropout))
                model.decoder.set_dropout(current_dropout)

            print(
                f"  ↳ Val CE {val_ce:.4f}  "
                f"PPL {torch.exp(val_loss).item():.2f}  "
                f"dropout→{current_dropout:.3f}  "
                f"val_every={val_every}{marker}"
            )
            model.train()

    torch.save(model.state_dict(), checkpoint)
    print(f"\nFinal checkpoint  → {checkpoint}")
    print(f"Best checkpoint   → {best_checkpoint}  (val CE {best_val_loss:.4f})")

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
        _, symbolic_indices, _, _, _, _, _, _, mean_H, active_ratio = model(context)

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
        "Emotion / Affect":   ["love", "hate", "rage", "grief", "joy",       "fear"],
        "Spatial / Distance": ["far",  "near", "above", "behind", "within",  "beyond"],
        "Visual / Light":     ["bright", "dark", "shadow", "glow", "shining", "dim"],
        "Abstract / Logic":   ["therefore", "because", "reason", "logic",    "however", "truth"],
        "Size / Magnitude":   ["vast", "tiny", "enormous", "massive", "little", "great"],
        "Position / Order":   ["first", "last", "before", "after", "middle",  "finally"],
    }

    all_words:   list[str]        = []
    all_weights: list[np.ndarray] = []
    category_boundaries: list[int] = []   # row indices where a new category starts

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
    submodality_names  = VisualSubmodalityEncoder.SUBMODALITY_NAMES

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

    bucket_activations = {i: [] for i in range(model.NUM_SUBMODALITIES)}

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
            gates        = F.softmax(model.submodality_gate(embed), dim=-1) # [1, 1, 5]
            gate_weights = gates.squeeze().cpu().numpy()                     # [5]

            for i in range(model.NUM_SUBMODALITIES):
                bucket_activations[i].append((word, gate_weights[i]))

    submodality_names = VisualSubmodalityEncoder.SUBMODALITY_NAMES

    for i in range(model.NUM_SUBMODALITIES):
        # Sort by gate weight descending, deduplicate case-insensitively.
        sorted_words = sorted(bucket_activations[i], key=lambda x: x[1], reverse=True)
        seen      = set()
        top_words = []
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

    input("\nScript complete. Press Enter to exit...")
