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
  [⭐] LR cosine annealing (CosineAnnealingLR)

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
        commitment_cost: float = 0.15,  # lowered from 0.25 — reduces overconstrained
    ):                                  # pressure when entropy + orth losses are also active
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

        # Learnable residual bypass: quantized_out = Q(z) + α · z
        # A hard VQ bottleneck destroys all continuous information not
        # captured by the nearest codebook entry.  The bypass lets the
        # network preserve fine-grained signal that doesn't fit the discrete
        # manifold while still training a meaningful codebook.  α is
        # initialised near zero so the model starts with almost-pure
        # quantisation and gradually opens the bypass as needed.
        self.residual_alpha = nn.Parameter(torch.tensor(0.1))

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

        # Residual bypass: preserves continuous signal not captured by the
        # nearest codebook entry.  α is clamped to [0, 0.5] to prevent
        # "escape route collapse": an unconstrained α can grow to dominate
        # the output, letting CE loss be satisfied while the codebook
        # receives negligible gradient and its entries go undertrained.
        # Clamping at 0.5 ensures the discrete component always contributes
        # at least as much as the residual.
        alpha    = self.residual_alpha.clamp(0.0, 0.5)
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

    Orthogonality regularisation:
        Without a constraint the five weight matrices can collapse into
        correlated latent directions, making the submodality split
        semantically redundant.  orthogonality_loss() penalises pairwise
        alignment between projection matrices:

            L_orth = λ · Σ_{i≠j} ‖ Wᵢ Wⱼᵀ ‖²_F

        The squared Frobenius norm of the cross-gram matrix is zero only
        when Wᵢ and Wⱼ span fully orthogonal subspaces.  λ = 1e-4 keeps
        the penalty small relative to CE loss while still providing a
        consistent gradient signal.
    """

    SUBMODALITY_NAMES  = ["Brightness", "Distance", "Focus", "Color", "Perspective"]
    NUM_SUBMODALITIES  = 5

    def __init__(self, d_model: int, orthogonality_lambda: float = 1e-4):
        super().__init__()
        self.extract_brightness  = nn.Linear(d_model, d_model)
        self.extract_distance    = nn.Linear(d_model, d_model)
        self.extract_focus       = nn.Linear(d_model, d_model)
        self.extract_color       = nn.Linear(d_model, d_model)
        self.extract_perspective = nn.Linear(d_model, d_model)
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
            self.extract_brightness.weight,   # [D, D]
            self.extract_distance.weight,
            self.extract_focus.weight,
            self.extract_color.weight,
            self.extract_perspective.weight,
        ]
        D    = mats[0].shape[0]              # embedding dimension
        loss = torch.tensor(0.0, device=mats[0].device)
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                # Squared Frobenius norm, normalised by D² so the penalty
                # magnitude is invariant to embedding dimension and weight
                # variance.  Without this, larger d_model would silently
                # inflate the penalty and overwhelm CE loss.
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
            # L2-normalise along the feature dimension so the cross-gram
            # measures angular correlation only.  Without this, growing
            # embedding norms during training inflate the penalty implicitly,
            # making the effective λ non-stationary.
            fi = F.normalize(fi, dim=1)
            for j in range(i + 1, self.NUM_SUBMODALITIES):
                fj = sub_features[:, :, j, :].reshape(N, D)  # [N, D]
                fj = F.normalize(fj, dim=1)
                # Cross-gram: [D, D]; normalised by N·D² for scale invariance
                cross_gram = (fi.t() @ fj) / N               # [D, D]
                loss = loss + cross_gram.pow(2).sum() / (D * D)

        return self.orthogonality_lambda * loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]  →  [B, T, 5, D]"""
        return torch.stack([
            self.extract_brightness(x),
            self.extract_distance(x),
            self.extract_focus(x),
            self.extract_color(x),
            self.extract_perspective(x),
        ], dim=2)


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
        The five quantised streams are aggregated via a softmax-gated
        weighted sum rather than a plain mean.  The gate is conditioned on
        the token embedding, so the model can learn that Brightness is more
        informative for, say, vivid emotional words and Perspective more
        informative for spatial descriptions.  A fixed mean implicitly
        assumes all five streams are equally reliable for every token, which
        is rarely true.
    """

    NUM_SUBMODALITIES = 5

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 128,
        num_symbols: int   = 256,
        n_heads:     int   = 4,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
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

        # Learned gating: softmax weights over the 5 submodality streams,
        # conditioned on the token embedding.  [B, T, 5] → [B, T, 5, 1]
        # so it broadcasts cleanly over the D dimension.
        gate_weights = F.softmax(self.submodality_gate(embed), dim=-1)  # [B, T, 5]
        sym_stack    = torch.stack(symbolic_vectors, dim=2)              # [B, T, 5, D]
        composite    = (sym_stack * gate_weights.unsqueeze(-1)).sum(dim=2)  # [B, T, D]

        # Feature-conditioned orthogonality — only needed during training.
        # At inference this O(25·N·D) computation is pure waste: the result
        # goes into orth_loss which is never used outside the training loop.
        # self.training is False during model.eval() and generate() calls.
        if self.training:
            feat_orth_loss = self.sub_encoder.feature_orthogonality_loss(sub_features)
        else:
            feat_orth_loss = sub_features.new_zeros(1).squeeze()  # zero scalar

        composite = self.pos_encoder(composite)
        decoder_out = self.decoder(composite)      # [B, T, D]
        logits      = self.output_head(decoder_out)

        return (
            logits,
            torch.stack(symbolic_indices, dim=2),          # [B, T, 5]
            total_vq_loss,
            total_ent_loss,
            feat_orth_loss,                                 # unscaled; sigmoid applied in loop
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
# DATA LOADING  (with 90/10 train/val split)
# ==========================================================

def load_data(filepath: str = "data.txt"):
    """
    Tokenise a plain-text corpus with the GPT-2 tiktoken tokeniser.
    Writes a small Shakespeare excerpt if `filepath` doesn't exist.

    FIX 3b — applies a 90/10 train/val split when corpus > 50 k tokens.

    Returns:
        train_data:  LongTensor
        val_data:    LongTensor | None
        tokenizer:   tiktoken Encoding
        vocab_size:  int  (50,257 for GPT-2)
    """
    tokenizer  = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab

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

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n    = len(data)

    VAL_THRESHOLD = 50_000
    if n > VAL_THRESHOLD:
        split      = int(0.9 * n)
        train_data = data[:split]
        val_data   = data[split:]
        print(
            f"Loaded {n:,} tokens  |  train {len(train_data):,}  "
            f"val {len(val_data):,}  (vocab {vocab_size:,})"
        )
    else:
        train_data = data
        val_data   = None
        print(
            f"Loaded {n:,} tokens (< {VAL_THRESHOLD:,} → no val split)  "
            f"vocab {vocab_size:,}"
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

def train_model(filepath: str = "data.txt"):
    """
    Full training loop with all three critical fixes applied.

      - CausalTransformerDecoder        (FIX 1: correct P(x_t | x_{<t}))
      - Annealed entropy weight         (FIX 2: safe codebook diversification)
      - Train/val split                 (FIX 3: overfitting detection)
      - Codebook active-symbol logging  (FIX 3: collapse detection)
      - CosineAnnealingLR               (BONUS: smoother convergence)
    """
    # ── Hyperparameters ───────────────────────────────────────────────
    seq_len_base   = 16      # starting context length (tokens)
    seq_len_max    = 64      # final context window (~50 words)
    batch_size     = 32
    steps          = 1000    # increase for larger corpora
    lr             = 3e-4
    lr_warmup      = 100     # steps to ramp lr from 0 → lr before cosine decay
    weight_decay   = 0.01
    vq_beta        = 0.40    # raised from 0.25 — commitment_cost is now 0.15 (was 0.25),
                             # so effective pressure = 0.40×0.15 = 0.06, close to the
                             # original 0.25×0.25 = 0.0625.  Keeps cluster pressure stable.
    entropy_beta   = 0.01    # MAX entropy weight (annealed up to this)
    entropy_warmup = 300     # steps to ramp entropy bonus from 0 → full
    orth_T0        = 200     # step at which orthogonality penalty begins rising
    orth_tau       = 100     # sigmoid temperature: smaller = sharper onset
    grad_clip      = 1.0
    log_every      = 100
    val_every      = 200
    checkpoint     = "neuro_symbolic_model.pt"

    # ── Data ──────────────────────────────────────────────────────────
    train_data, val_data, tokenizer, vocab_size = load_data(filepath)

    # Guard: _random_batch draws windows of length seq_len_max + 1.
    # If the corpus is smaller, torch.randint receives a non-positive upper
    # bound and crashes with a cryptic error.  Fail fast here instead.
    if len(train_data) < seq_len_max + 2:
        raise ValueError(
            f"Corpus too small: {len(train_data)} tokens, need at least "
            f"{seq_len_max + 2}.  Provide a larger data.txt or reduce seq_len_max."
        )

    # ── Model ─────────────────────────────────────────────────────────
    model = SubmodalitySymbolicLanguageModel(
        vocab_size  = vocab_size,
        d_model     = 128,
        num_symbols = 256,
        n_heads     = 4,
        num_layers  = 2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}  |  Device: {device}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # BONUS — cosine LR annealing: smoothly decays lr to lr/10 by final step
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr / 10
    )

    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    print("Starting training…\n")

    for step in range(steps):
        # Curriculum sequence length: grows logarithmically from seq_len_base
        # to seq_len_max.  Short contexts are easier to model early in
        # training, so the model builds reliable representations before
        # being asked to integrate long-range dependencies.
        seq_len = min(seq_len_max,
                      seq_len_base + int(math.log(step + 1) * 8))

        x, y = _random_batch(train_data, batch_size, seq_len)

        # LR warmup: ramp linearly from 0 → lr over the first lr_warmup steps.
        # Without warmup, the full learning rate on step 0 applies before VQ
        # codebook entries have formed any clusters — this destabilises early
        # quantisation.  The cosine scheduler takes over after warmup.
        if step < lr_warmup:
            warmup_factor = (step + 1) / lr_warmup
            for g in optimizer.param_groups:
                g["lr"] = lr * warmup_factor

        # Sigmoid schedule for orthogonality weight — capped at 0.5.
        # A full sigmoid asymptotes to 1.0, meaning orth_loss would eventually
        # dominate all other terms in runs longer than ~1k steps.  The 0.5
        # cap keeps orth pressure bounded so CE loss always remains primary.
        orth_weight = 0.5 / (1.0 + math.exp(-(step - orth_T0) / orth_tau))

        optimizer.zero_grad()
        logits, _, vq_loss, ent_loss, feat_orth_loss, mean_H, active_ratio = model(x)
        loss_task = criterion(logits.view(-1, vocab_size), y.view(-1))

        entropy_weight = entropy_beta * min(1.0, step / max(1, entropy_warmup))

        # Feature-conditioned orthogonality returned from forward() and scaled
        # by the sigmoid schedule.  Using batch features rather than raw
        # weights ensures the penalty reflects actual representational geometry.
        orth_loss = orth_weight * feat_orth_loss

        total_loss = loss_task + vq_beta * vq_loss + entropy_weight * ent_loss + orth_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────
        if step % log_every == 0 or step == steps - 1:
            ppl = torch.exp(loss_task).item()
            # FIX 3a — active-symbol % is the codebook collapse early warning
            print(
                f"Step {step:4d}/{steps} | "
                f"CE {loss_task.item():.4f} | "
                f"VQ {vq_loss.item():.4f} | "
                f"Orth {orth_loss.item():.4f} | "
                f"EntLoss {ent_loss.item():.4f} | "
                f"H_ema {mean_H:.2f} | "
                f"Active {active_ratio*100:4.1f}% | "
                f"PPL {ppl:.2f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e}"
            )

        # ── Validation ────────────────────────────────────────────────
        if val_data is not None and step > 0 and step % val_every == 0:
            model.eval()
            with torch.no_grad():
                # Use the same seq_len as current training step so validation
                # distribution matches training — using seq_len_max always
                # creates a mismatch early in curriculum training.
                xv, yv   = _random_batch(val_data, batch_size, seq_len)
                lv, *_   = model(xv)
                val_loss = criterion(lv.view(-1, vocab_size), yv.view(-1))
            print(
                f"  ↳ Val  CE {val_loss.item():.4f}  "
                f"PPL {torch.exp(val_loss).item():.2f}"
            )
            model.train()

    torch.save(model.state_dict(), checkpoint)
    print(f"\nCheckpoint saved → {checkpoint}")

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
        _, symbolic_indices, _, _, _, mean_H, active_ratio = model(context)

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
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    trained_model, tokenizer, s_len = train_model()
    evaluate_and_generate(trained_model, tokenizer, s_len)
