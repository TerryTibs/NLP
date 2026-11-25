nlp_test1.py
  
  Tokenizer: BPE — vocab size = 83,308
  BPE (fast/C++): training to vocab 83,308…
[00:00:00] Pre-processing sequences       ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 0        /        0
[00:00:00] Tokenize words                 ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 26935    /    26935
[00:00:00] Count pairs                    ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 26935    /    26935
[00:00:00] Compute merges                 ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 35884    /    35884
  BPE vocab: 35,972 tokens
  Tokens: 737,729  |  train 663,956  val 73,773  (10% val)
  num_symbols: 32 (start, target 54)
  d_model: 256
  n_heads: 8
  num_layers: 8 (start)  max 16
  seq_len_base: 30  seq_len_max (curriculum start): 64
  Optimizer groups: 68 decay params, 104 no-decay params
  Parameters: 16,850,196  |  Device: cuda
  Compiling model with torch.compile (dynamic=True)…
  Compile complete.

Starting training…

Step     0/3000 | CE 11.0115 | PPL 60568.31 | VQ 3.1867(β0.25,cc0.250) | MetaCB Act1.00 H0.64→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE3.729 Acc0.00 λ0.020 Bal0.115 | ChunkUp Act1.00 H0.35→0.62 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 0.00e+00(λ2.00e-05,ρcal) | GKL 0.517(γ0.0501) | SymD μ0.000 ↑n/a | Act 88.4%[84–97] | TopK-H 1.085/1.10(τ0.10) | α μ0.060[0.06–0.06](λ0.0050) | H 0.53→1.20 | BS 34x30(t1024) | Seen 0.0M(0.0x) | ∇ 4.52→1.76 | vel n/a | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 0.8G | Thr 0.1k t/s | β2 0.9900 | LR 1.50e-06 | ⏱ 00:00:13 (+13.01s/step, ETA 10:50:22)
  Gates: Bri=0.05 Dis=0.19 Foc=0.05 Col=0.03 Per=0.25 Siz=0.07 Pos=0.36
  TopK-H/stream: Bri=1.09 Dis=1.07 Foc=1.09 Col=1.09 Per=1.09 Siz=1.09 Pos=1.09
  ✦ Stream correlation baseline calibrated: 0.0035
Step   100/3000 | CE 7.7326 | PPL 2281.42 | VQ 2.5874(β0.22,cc0.229) | MetaCB Act1.00 H3.16→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.682 Acc0.03 λ0.020 Bal0.211 | ChunkUp Act1.00 H1.76→0.70 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 0.00e+00(λ1.85e-05,ρ0.003) | GKL 0.526(γ0.0601) | SymD μ0.000 ↑n/a | Act 79.9%[69–91] | TopK-H 1.082/1.10(τ0.11) | α μ0.060[0.06–0.06](λ0.0048) | H 2.44→1.34 | BS 16x64(t1024) | Seen 0.1M(0.2x) | ∇ 0.98→1.89 | vel n/a | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.4G | Thr 18.4k t/s | β2 0.9903 | LR 1.51e-04 | ⏱ 00:00:18 (+0.06s/step, ETA 00:02:41)
Step   200/3000 | CE 6.6729 | PPL 790.65 | VQ 1.6762(β0.23,cc0.246) | MetaCB Act1.00 H3.01→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.516 Acc0.08 λ0.020 Bal0.171 | ChunkUp Act1.00 H1.64→0.77 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 0.00e+00(λ1.76e-05,ρ0.037) | GKL 0.520(γ0.0723) | SymD μ0.000 ↑n/a | Act 56.2%[34–69] | TopK-H 0.865/1.10(τ0.13) | α μ0.061[0.06–0.06](λ0.0046) | H 1.85→1.48 | BS 16x64(t1024) | Seen 0.2M(0.3x) | ∇ 1.04→1.48 | vel +0.0000 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.4G | Thr 23.0k t/s | β2 0.9906 | LR 3.00e-04 | ⏱ 00:00:23 (+0.04s/step, ETA 00:02:04)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 200)
  ↳ Val CE 6.8679  PPL 960.93  do→0.100  wd→0.0100  val_every=200  vel=+0.00000  ✓ best
  ✦ Orth unlocked at step 240  (corr 0.0593 > baseline 0.0035 + 0.05)
Step   300/3000 | CE 6.2281 | PPL 506.81 | VQ 1.1948(β0.28,cc0.291) | MetaCB Act1.00 H2.97→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.650 Acc0.03 λ0.020 Bal0.132 | ChunkUp Act1.00 H1.66→0.84 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 1.11e-05(λ2.21e-05,ρ0.080) | GKL 0.504(γ0.0865) | SymD μ0.000 ↑n/a | Act 48.7%[34–69] | TopK-H 0.681/1.10(τ0.14) | α μ0.062[0.06–0.06](λ0.0044) | H 1.80→1.62 | BS 16x64(t1024) | Seen 0.3M(0.5x) | ∇ 1.11→1.61 | vel -0.0060 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 16.5k t/s | β2 0.9909 | LR 3.00e-04 | ⏱ 00:00:29 (+0.06s/step, ETA 00:02:47)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 400)
  ↳ Val CE 6.5735  PPL 715.87  do→0.099  wd→0.0099  val_every=200  vel=-0.00460  ✓ best
  ⏳ step 401/3000[00:01:00 elapsed]  still running…
Step   416/3000 | CE 5.9516 | PPL 384.39 | VQ 1.1226(β0.36,cc0.340) | MetaCB Act1.00 H3.05→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.797 Acc0.02 λ0.020 Bal0.121 | ChunkUp Act1.00 H1.67→0.93 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 3.01e-05(λ3.76e-05,ρ0.116) | GKL 0.501(γ0.1061) | SymD μ0.000 ↑n/a | Act 50.9%[28–72] | TopK-H 0.686/1.10(τ0.15) | α μ0.064[0.06–0.07](λ0.0042) | H 1.86→1.77 | BS 16x64(t1024) | Seen 0.4M(0.6x) | ∇ 1.28→1.73 | vel -0.0073 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 3.8k t/s | β2 0.9912 | LR 3.00e-04 | ⏱ 00:01:00 (+0.27s/step, ETA 00:11:45)
Step   500/3000 | CE 6.0422 | PPL 420.84 | VQ 1.1162(β0.42,cc0.373) | MetaCB Act1.00 H3.09→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.729 Acc0.02 λ0.020 Bal0.104 | ChunkUp Act1.00 H1.69→0.99 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 6.43e-05(λ6.51e-05,ρ0.143) | GKL 0.521(γ0.1229) | SymD μ0.000 ↑n/a | Act 50.4%[31–78] | TopK-H 0.719/1.10(τ0.17) | α μ0.065[0.06–0.07](λ0.0040) | H 1.88→1.90 | BS 16x64(t1024) | Seen 0.5M(0.8x) | ∇ 1.14→1.83 | vel -0.0023 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 22.6k t/s | β2 0.9915 | LR 3.00e-04 | ⏱ 00:01:04 (+0.05s/step, ETA 00:01:53)
  Gates: Bri=0.05 Dis=0.12 Foc=0.24 Col=0.01 Per=0.22 Siz=0.06 Pos=0.29
  TopK-H/stream: Bri=0.49 Dis=0.47 Foc=1.07 Col=0.43 Per=0.66 Siz=0.84 Pos=1.08
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 600)
  ↳ Val CE 6.3948  PPL 598.73  do→0.099  wd→0.0099  val_every=200  vel=-0.00504  ✓ best
  ⏳ step 609/3000[00:01:34 elapsed]  still running…
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 800)
  ↳ Val CE 6.2549  PPL 520.54  do→0.098  wd→0.0098  val_every=200  vel=+0.00063  ✓ best
  ⏳ step 801/3000[00:02:14 elapsed]  still running…
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 1000)
  ↳ Val CE 6.1228  PPL 456.12  do→0.098  wd→0.0098  val_every=200  vel=-0.00316  ✓ best
  ⏳ step 1001/3000[00:02:50 elapsed]  still running…
  ↓ LR 3.00e-04 → 2.97e-04  (vel +0.00119 ≥ +0.00020, reduction #1)
  ↑ batch_tokens → 2048
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 1200)
  ↳ Val CE 6.0237  PPL 413.11  do→0.097  wd→0.0097  val_every=400  vel=-0.00087  ✓ best
  ⏳ step 1201/3000[00:03:29 elapsed]  still running…
  ↓ LR 2.97e-04 → 2.94e-04  (vel +0.00257 ≥ +0.00020, reduction #2)
  ↑ batch_tokens → 4096
Step  1600/3000 | CE 5.3113 | PPL 202.62 | VQ 1.0634(β0.50,cc0.500) | MetaCB Act1.00 H3.29→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.691 Acc0.03 λ0.020 Bal0.050 | ChunkUp Act1.00 H1.81→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 1.20e-03(λ1.00e-03,ρ0.319) | GKL 0.000(γ0.1910) | SymD μ0.169 ↑0.361(Dis,Pos) | Act 73.7%[34–100] | TopK-H 0.882/1.10(τ0.31) | α μ0.082[0.08–0.08](λ0.0026) | H 2.03→3.12 | BS 64x64(t4096) | Seen 2.4M(3.5x) | ∇ 1.11→1.59 | vel -0.0020 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 4.4G | Thr 26.1k t/s | β2 0.9948 | LR 2.94e-04 | ⏱ 00:03:57 (+0.16s/step, ETA 00:03:39)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 1600)
  ↳ Val CE 5.8561  PPL 349.37  do→0.097  wd→0.0097  val_every=800  vel=-0.00201  ✓ best
W1125 14:42:57.159000 18332 .venv\Lib\site-packages\torch\_dynamo\convert_frame.py:2061] [11/8] torch._dynamo hit config.recompile_limit (8)
W1125 14:42:57.159000 18332 .venv\Lib\site-packages\torch\_dynamo\convert_frame.py:2061] [11/8]    function: 'forward' (C:\Users\Justi\OneDrive\Desktop\Drone Control System\nlp_test1.py:153)
W1125 14:42:57.159000 18332 .venv\Lib\site-packages\torch\_dynamo\convert_frame.py:2061] [11/8]    last reason: 11/7: self._modules['blocks']._modules['0'].attn_drop == 0.09703725093562657  # attn_out = F.scaled_dot_product_attention(  # nlp_test1.py:109 in forward (_dynamo\utils.py:4549 in run_node)
W1125 14:42:57.159000 18332 .venv\Lib\site-packages\torch\_dynamo\convert_frame.py:2061] [11/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
W1125 14:42:57.159000 18332 .venv\Lib\site-packages\torch\_dynamo\convert_frame.py:2061] [11/8] To diagnose recompilation issues, see https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html
Step  1700/3000 | CE 5.3443 | PPL 209.41 | VQ 1.0708(β0.49,cc0.500) | MetaCB Act1.00 H3.31→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE2.871 Acc0.17 λ0.020 Bal0.175 | ChunkUp Act1.00 H1.79→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 1.16e-03(λ1.00e-03,ρ0.332) | GKL 0.000(γ0.1773) | SymD μ0.178 ↑0.375(Dis,Pos) | Act 72.8%[34–100] | TopK-H 0.894/1.10(τ0.33) | α μ0.084[0.08–0.09](λ0.0025) | H 2.07→3.12 | BS 64x64(t4096) | Seen 2.8M(4.2x) | ∇ 1.15→1.70 | vel -0.0001 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 4.9G | Thr 37.5k t/s | β2 0.9951 | LR 2.94e-04 | ⏱ 00:04:07 (+0.11s/step, ETA 00:02:22)
    ↑ Codebook grown: 32 → 48 symbols
    ↑ Growth LR reset: 5.88e-04 → 2.94e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
  ⏳ step 1801/3000[00:05:38 elapsed]  still running…
Step  1818/3000 | CE 5.2961 | PPL 199.55 | VQ 1.0914(β0.47,cc0.500) | MetaCB Act1.00 H3.30→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE2.750 Acc0.21 λ0.020 Bal0.168 | ChunkUp Act1.00 H1.81→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 1.10e-03(λ1.00e-03,ρ0.353) | GKL 0.000(γ0.1624) | SymD μ0.185 ↑0.375(Dis,Pos) | Act 75.0%[38–100] | TopK-H 0.911/1.10(τ0.34) | α μ0.086[0.08–0.09](λ0.0024) | H 2.12→3.12 | BS 64x64(t4096) | Seen 3.2M(4.9x) | ∇ 1.10→1.85 | vel +0.0002 | do 0.10 | L 8 | S 7/7 | CB 34 | Wmup W1@482 | VRAM 4.9G | Thr 5.2k t/s | β2 0.9955 | LR 2.94e-04 | ⏱ 00:05:40 (+0.78s/step, ETA 00:15:25)
Step  1919/3000 | CE 5.1733 | PPL 176.50 | VQ 1.1163(β0.45,cc0.499) | MetaCB Act1.00 H3.28→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE2.689 Acc0.23 λ0.020 Bal0.170 | ChunkUp Act1.00 H1.81→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 1.04e-03(λ1.00e-03,ρ0.369) | GKL 0.000(γ0.1506) | SymD μ0.200 ↑0.376(Dis,Pos) | Act 77.1%[34–100] | TopK-H 0.925/1.10(τ0.35) | α μ0.088[0.08–0.09](λ0.0024) | H 2.17→3.12 | BS 64x64(t4096) | Seen 3.7M(5.5x) | ∇ 1.27→1.80 | vel -0.0010 | do 0.10 | L 8 | S 7/7 | CB 34 | Wmup W1@381 | VRAM 4.9G | Thr 42.7k t/s | β2 0.9958 | LR 2.94e-04 | ⏱ 00:05:50 (+0.10s/step, ETA 00:01:43)
Step  1976/3000 | CE 5.1968 | PPL 180.69 | VQ 1.1150(β0.43,cc0.493) | MetaCB Act1.00 H3.28→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE2.697 Acc0.23 λ0.020 Bal0.190 | ChunkUp Act1.00 H1.83→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 1.05e-03(λ1.00e-03,ρ0.377) | GKL 0.000(γ0.1443) | SymD μ0.202 ↑0.382(Dis,Pos) | Act 76.3%[34–97] | TopK-H 0.927/1.10(τ0.36) | α μ0.090[0.09–0.09](λ0.0023) | H 2.19→3.12 | BS 64x64(t4096) | Seen 3.9M(5.9x) | ∇ 1.28→1.78 | vel -0.0000 | do 0.10 | L 8 | S 7/7 | CB 34 | Wmup W1@324 | VRAM 4.9G | Thr 42.7k t/s | β2 0.9959 | LR 2.94e-04 | ⏱ 00:05:55 (+0.10s/step, ETA 00:01:38)
    ↑ Codebook grown: 32 → 48 symbols
    ↑ Growth LR reset: 5.88e-04 → 2.94e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
    ↑ Codebook grown: 32 → 48 symbols
    ↑ Growth LR reset: 5.88e-04 → 2.94e-04 over 500 steps
    ↑ meta_pred_head grown: 32 → 48 classes
    ↑ Growth LR reset: 5.88e-04 → 2.94e-04 over 500 steps
  ↑ Meta-codebook grown: 32 → 48 symbols
Step  2000/3000 | CE 5.1431 | PPL 171.24 | VQ 1.1190(β0.43,cc0.489) | MetaCB Act1.00 H3.28→3.48 β0.15 cc0.150 TopKH1.09 | MetaPred CE2.643 Acc0.23 λ0.020 Bal0.186 | ChunkUp Act1.00 H1.83→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 1.04e-03(λ1.00e-03,ρ0.381) | GKL 0.000(γ0.1417) | SymD μ0.207 ↑0.382(Dis,Pos) | Act 77.4%[34–98] | TopK-H 0.930/1.10(τ0.37) | α μ0.090[0.09–0.09](λ0.0023) | H 2.20→3.12 | BS 64x64(t4096) | Seen 4.0M(6.0x) | ∇ 1.14→1.82 | vel -0.0008 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W4@300 | VRAM 4.9G | Thr 42.3k t/s | β2 0.9960 | LR 2.94e-04 | ⏱ 00:05:57 (+0.10s/step, ETA 00:01:36)
  Gates: Bri=0.15 Dis=0.12 Foc=0.16 Col=0.13 Per=0.11 Siz=0.13 Pos=0.20
  TopK-H/stream: Bri=0.88 Dis=0.89 Foc=1.03 Col=0.77 Per=0.95 Siz=0.99 Pos=1.00
  ⏳ step 2182/3000[00:06:27 elapsed]  still running…
  ↓ LR 2.94e-04 → 2.91e-04  (vel +0.00212 ≥ +0.00020, reduction #3)
  ↑ batch_tokens → 4096
    ↑ Codebook grown: 48 → 54 symbols
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
Step  2244/3000 | CE 5.0763 | PPL 160.18 | VQ 1.1517(β0.36,cc0.436) | MetaCB Act0.96 H3.46→3.48 β0.15 cc0.150 TopKH1.09 | MetaPred CE2.713 Acc0.25 λ0.020 Bal0.305 | ChunkUp Act1.00 H1.85→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 9.94e-04(λ1.00e-03,ρ0.415) | GKL 0.000(γ0.1181) | SymD μ0.222 ↑0.370(Bri,Pos) | Act 79.9%[38–98] | TopK-H 0.955/1.10(τ0.40) | α μ0.095[0.09–0.10](λ0.0022) | H 2.35→3.12 | BS 64x64(t4096) | Seen 5.0M(7.5x) | ∇ 1.33→1.91 | vel +0.0002 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W5@56 | VRAM 4.9G | Thr 17.5k t/s | β2 0.9967 | LR 2.91e-04 | ⏱ 00:06:55 (+0.23s/step, ETA 00:02:57)
    ↑ Codebook grown: 48 → 54 symbols
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps
    ↑ meta_pred_head grown: 48 → 54 classes
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps
  ↑ Meta-codebook grown: 48 → 54 symbols
  ⏳ step 2420/3000[00:07:25 elapsed]  still running…
Step  2448/3000 | CE 5.0486 | PPL 155.80 | VQ 1.1757(β0.31,cc0.369) | MetaCB Act0.87 H3.49→3.59 β0.15 cc0.150 TopKH1.09 | MetaPred CE2.773 Acc0.26 λ0.020 Bal0.307 | ChunkUp Act0.88 H1.16→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.29e-04(λ1.00e-03,ρ0.436) | GKL 0.012(γ0.1013) | SymD μ0.226 ↑0.378(Bri,Pos) | Act 79.9%[38–94] | TopK-H 0.974/1.10(τ0.43) | α μ0.100[0.10–0.10](λ0.0020) | H 2.41→3.12 | BS 64x64(t4096) | Seen 5.8M(8.8x) | ∇ 1.34→1.88 | vel +0.0018 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W6@52 | VRAM 4.9G | Thr 25.5k t/s | β2 0.9973 | LR 2.91e-04 | ⏱ 00:07:27 (+0.16s/step, ETA 00:01:28)
Step  2530/3000 | CE 4.8209 | PPL 124.08 | VQ 1.1825(β0.28,cc0.340) | MetaCB Act0.91 H3.51→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.711 Acc0.26 λ0.020 Bal0.309 | ChunkUp Act0.75 H0.93→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.41e-04(λ1.00e-03,ρ0.443) | GKL 0.000(γ0.0953) | SymD μ0.234 ↑0.402(Bri,Pos) | Act 81.7%[38–96] | TopK-H 0.977/1.10(τ0.44) | α μ0.101[0.10–0.11](λ0.0020) | H 2.44→3.12 | BS 64x64(t4096) | Seen 6.2M(9.3x) | ∇ 1.25→1.95 | vel -0.0023 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W3@170 | VRAM 4.9G | Thr 42.2k t/s | β2 0.9976 | LR 2.91e-04 | ⏱ 00:07:35 (+0.10s/step, ETA 00:00:45)
Step  2575/3000 | CE 4.7823 | PPL 119.38 | VQ 1.1924(β0.27,cc0.324) | MetaCB Act0.87 H3.51→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.682 Acc0.26 λ0.020 Bal0.297 | ChunkUp Act0.75 H0.92→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.15e-04(λ1.00e-03,ρ0.446) | GKL 0.000(γ0.0921) | SymD μ0.237 ↑0.411(Bri,Pos) | Act 84.0%[41–100] | TopK-H 0.982/1.10(τ0.44) | α μ0.102[0.10–0.11](λ0.0020) | H 2.46→3.12 | BS 64x64(t4096) | Seen 6.3M(9.6x) | ∇ 1.28→1.94 | vel -0.0028 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W3@125 | VRAM 4.9G | Thr 42.3k t/s | β2 0.9977 | LR 2.91e-04 | ⏱ 00:07:40 (+0.10s/step, ETA 00:00:41)
  Gates: Bri=0.14 Dis=0.13 Foc=0.16 Col=0.11 Per=0.14 Siz=0.12 Pos=0.21
  TopK-H/stream: Bri=0.95 Dis=0.97 Foc=1.04 Col=0.87 Per=1.00 Siz=1.03 Pos=1.02
Step  2600/3000 | CE 4.7325 | PPL 113.58 | VQ 1.1779(β0.27,cc0.315) | MetaCB Act0.89 H3.51→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.688 Acc0.26 λ0.020 Bal0.298 | ChunkUp Act0.75 H0.92→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.47e-04(λ1.00e-03,ρ0.448) | GKL 0.000(γ0.0904) | SymD μ0.240 ↑0.412(Bri,Pos) | Act 83.0%[41–96] | TopK-H 0.977/1.10(τ0.45) | α μ0.103[0.10–0.11](λ0.0020) | H 2.47→3.12 | BS 64x64(t4096) | Seen 6.5M(9.7x) | ∇ 1.41→2.01 | vel -0.0035 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W3@100 | VRAM 4.9G | Thr 42.1k t/s | β2 0.9978 | LR 2.91e-04 | ⏱ 00:07:42 (+0.10s/step, ETA 00:00:38)
Step  2678/3000 | CE 4.8042 | PPL 122.02 | VQ 1.2041(β0.24,cc0.289) | MetaCB Act0.93 H3.52→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.693 Acc0.25 λ0.020 Bal0.287 | ChunkUp Act0.88 H0.99→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.05e-04(λ1.00e-03,ρ0.453) | GKL 0.000(γ0.0853) | SymD μ0.244 ↑0.409(Bri,Pos) | Act 84.2%[41–98] | TopK-H 0.987/1.10(τ0.46) | α μ0.105[0.10–0.11](λ0.0019) | H 2.49→3.12 | BS 64x64(t4096) | Seen 6.8M(10.2x) | ∇ 1.27→1.94 | vel -0.0013 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W3@22 | VRAM 4.9G | Thr 42.5k t/s | β2 0.9980 | LR 2.91e-04 | ⏱ 00:07:50 (+0.10s/step, ETA 00:00:31)
Step  2730/3000 | CE 4.8772 | PPL 131.27 | VQ 1.2102(β0.23,cc0.273) | MetaCB Act0.93 H3.52→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.662 Acc0.25 λ0.020 Bal0.284 | ChunkUp Act0.75 H1.05→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.12e-04(λ1.00e-03,ρ0.456) | GKL 0.000(γ0.0820) | SymD μ0.254 ↑0.411(Bri,Pos) | Act 83.0%[41–97] | TopK-H 0.990/1.10(τ0.46) | α μ0.106[0.10–0.11](λ0.0019) | H 2.51→3.12 | BS 64x64(t4096) | Seen 7.0M(10.5x) | ∇ 1.32→1.96 | vel +0.0005 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W2@170 | VRAM 4.9G | Thr 42.4k t/s | β2 0.9982 | LR 2.91e-04 | ⏱ 00:07:55 (+0.10s/step, ETA 00:00:26)
  ↓ LR 2.91e-04 → 2.88e-04  (vel +0.00116 ≥ +0.00020, reduction #4)
  ↑ batch_tokens → 4096
    ↑ Codebook grown: 48 → 54 symbols
    ↑ Growth LR reset: 5.76e-04 → 2.88e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
Step  2856/3000 | CE 4.7168 | PPL 111.81 | VQ 1.2131(β0.20,cc0.236) | MetaCB Act0.89 H3.53→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.668 Acc0.25 λ0.020 Bal0.260 | ChunkUp Act0.62 H1.16→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.16e-04(λ1.00e-03,ρ0.465) | GKL 0.000(γ0.0746) | SymD μ0.248 ↑0.409(Bri,Pos) | Act 83.1%[47–97] | TopK-H 0.993/1.10(τ0.48) | α μ0.109[0.10–0.12](λ0.0019) | H 2.55→3.12 | BS 64x64(t4096) | Seen 7.5M(11.3x) | ∇ 1.29→2.01 | vel -0.0016 | do 0.10 | L 8 | S 7/7 | CB 38 | Wmup W3@44 | VRAM 4.9G | Thr 21.0k t/s | β2 0.9986 | LR 2.88e-04 | ⏱ 00:08:19 (+0.20s/step, ETA 00:00:28)
  ⚡ Kickstart (Brightness,Position): perturbed 8 of 32 centroids
Step  2912/3000 | CE 4.8391 | PPL 126.36 | VQ 1.2280(β0.19,cc0.220) | MetaCB Act0.89 H3.54→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.750 Acc0.25 λ0.020 Bal0.244 | ChunkUp Act0.75 H1.20→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.01e-04(λ1.00e-03,ρ0.467) | GKL 0.000(γ0.0715) | SymD μ0.257 ↑0.420(Bri,Pos) | Act 82.3%[44–94] | TopK-H 0.999/1.10(τ0.49) | α μ0.110[0.11–0.12](λ0.0018) | H 2.56→3.12 | BS 64x64(t4096) | Seen 7.7M(11.6x) | ∇ 1.30→2.08 | vel +0.0013 | do 0.10 | L 8 | S 7/7 | CB 38 | Wmup W1@388 | VRAM 4.9G | Thr 42.4k t/s | β2 0.9987 | LR 2.88e-04 | ⏱ 00:08:25 (+0.10s/step, ETA 00:00:08)
Step  2987/3000 | CE 4.7000 | PPL 109.95 | VQ 1.2431(β0.17,cc0.201) | MetaCB Act0.91 H3.55→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.666 Acc0.25 λ0.020 Bal0.261 | ChunkUp Act0.75 H1.23→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 8.83e-04(λ1.00e-03,ρ0.472) | GKL 0.000(γ0.0676) | SymD μ0.256 ↑0.418(Bri,Pos) | Act 84.9%[47–97] | TopK-H 1.004/1.10(τ0.50) | α μ0.112[0.11–0.12](λ0.0018) | H 2.57→3.12 | BS 64x64(t4096) | Seen 8.0M(12.1x) | ∇ 1.36→1.99 | vel -0.0012 | do 0.10 | L 8 | S 7/7 | CB 38 | Wmup W1@313 | VRAM 4.9G | Thr 42.2k t/s | β2 0.9990 | LR 2.88e-04 | ⏱ 00:08:32 (+0.10s/step, ETA 00:00:01)
Step  2999/3000 | CE 4.7316 | PPL 113.47 | VQ 1.2422(β0.17,cc0.198) | MetaCB Act0.89 H3.54→3.59 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.646 Acc0.26 λ0.020 Bal0.257 | ChunkUp Act0.62 H1.23→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 9.30e-04(λ1.00e-03,ρ0.472) | GKL 0.000(γ0.0670) | SymD μ0.259 ↑0.407(Bri,Pos) | Act 83.6%[50–97] | TopK-H 1.001/1.10(τ0.50) | α μ0.112[0.11–0.12](λ0.0018) | H 2.58→3.12 | BS 64x64(t4096) | Seen 8.1M(12.2x) | ∇ 1.28→1.99 | vel -0.0004 | do 0.10 | L 8 | S 7/7 | CB 38 | Wmup W1@301 | VRAM 4.9G | Thr 42.2k t/s | β2 0.9990 | LR 2.88e-04 | ⏱ 00:08:33 (+0.10s/step, ETA 00:00:00)
  Gates: Bri=0.12 Dis=0.16 Foc=0.17 Col=0.12 Per=0.13 Siz=0.12 Pos=0.18
  TopK-H/stream: Bri=0.98 Dis=0.99 Foc=1.04 Col=0.92 Per=1.01 Siz=1.04 Pos=1.03
  💾 Checkpoint saved (final): neuro_symbolic_model.pt  (step 2999)

Final checkpoint  → neuro_symbolic_model.pt
Best checkpoint   → neuro_symbolic_model_best.pt  (val CE 5.8561)
Total training time: 00:08:33  (0.14 hours)

============================================================
  NEURO-SYMBOLIC EXPLAINABILITY & GENERATION DEMO
============================================================

SEED PROMPT:
----------------------------------------
First Citizen:
Before we proceed any further, hear me
----------------------------------------

Internal submodality representation of the final token (Top-K routing):

  Brightness   → #018  #028  #007
  Distance     → #005  #036  #031
  Focus        → #011  #008  #031
  Color        → #019  #016  #023
  Perspective  → #015  #004  #021
  Size         → #025  #006  #010
  Position     → #000  #022  #001

  H_ema (smoothed entropy): 2.561  |  Active: 34.8%

Generating 60 new tokens…

GENERATED TEXT:
--------------------------------------------------
First Citizen : Before we proceed any further , hear me and then . MARCIUS : ' Twas not my heart but what I have told of him . Second Senator : No , nor the queen of God it is slain - And thou too long in my heart , if they be more to see The very time hath no way than this man would die . But now
--------------------------------------------------

============================================================
  RUNNING XAI 'MIND READER' VISUALIZATION...
============================================================
Opening heatmap window… (close the window to end the script)

============================================================
  DECODING AI PSYCHOLOGY (AUTO-LABELER)...
============================================================

► BUCKET 0  (Named: 'Brightness')
  Top Words: thou, more, always, change, certain, didst, hold, made, refrain, stand, less, greater, find, report, Apollo

► BUCKET 1  (Named: 'Distance')
  Top Words: labour, fresh, majesty, weight, fruit, champion, foot, road, Plato, wine, GREMIO, field, lamb, heard, Tower, arms, sake, wolves, feast, report

► BUCKET 2  (Named: 'Focus')
  Top Words: less, since, gods, thee

► BUCKET 3  (Named: 'Color')
  Top Words: lashing, unanimity, Anacreon, cylinders, PEACE, newest, exordium, wiggling, assures, livid, relative, barriers, adopting, brothels, sneaks, succumb, seawater, supplanted, brays, Miltiades

► BUCKET 4  (Named: 'Perspective')
  Top Words: Apollo, thee, HENRY, When, STREPSIADES, hence, Even, find, Which, Pray, strophe, From, greater, XANTHIAS, PHIDIPPIDES, CLYTEMNESTRA, help, Make, uncle, lips

► BUCKET 5  (Named: 'Size')
  Top Words: past, early, upon, human, sweet, PRAXAGORA, just, From, When

► BUCKET 6  (Named: 'Position')
  Top Words: made, upon, royal, help, Which, more, thou

============================================================

============================================================
  SYMBOL MUTUAL INFORMATION DIAGNOSTIC
============================================================
  Estimating MI(symbol ; token) per submodality stream…
  Sampling 8000 tokens from data.txt.

  Stream          MI(sym;tok)   Quality
  ------------------------------------------
  Brightness            1.565   ✓ high     — symbols encode meaningful structure
  Distance              1.699   ✓ high     — symbols encode meaningful structure
  Focus                 2.019   ✓ high     — symbols encode meaningful structure
  Color                 1.055   ✓ high     — symbols encode meaningful structure
  Perspective           1.847   ✓ high     — symbols encode meaningful structure
  Size                  1.579   ✓ high     — symbols encode meaningful structure
  Position              2.519   ✓ high     — symbols encode meaningful structure

  Mean MI across streams: 1.755

  Meta-codebook decision: GO  ✓  — level-1 symbols are semantically meaningful.
         Meta-codebook experiment is well-motivated.
============================================================

nlp_test2.py

Tokenizer: BPE — vocab size = 83,308
  BPE (fast/C++): training to vocab 83,308…
[00:00:00] Pre-processing sequences       ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 0        /        0
[00:00:00] Tokenize words                 ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 26935    /    26935
[00:00:00] Count pairs                    ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 26935    /    26935
[00:00:00] Compute merges                 ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 35884    /    35884
  BPE vocab: 35,972 tokens
  Tokens: 737,729  |  train 663,956  val 73,773  (10% val)
  num_symbols: 32 (start, target 54)
  d_model: 256
  n_heads: 8
  num_layers: 8 (start)  max 16
  seq_len_base: 30  seq_len_max (curriculum start): 64
  Optimizer groups: 68 decay params, 104 no-decay params
  Parameters: 16,850,196  |  Device: cuda
  Compiling model with torch.compile (dynamic=True)…
  Compile complete.

Starting training…

Step     0/3000 | CE 11.0134 | PPL 60679.31 | VQ 3.1867(β0.25,cc0.250) | MetaCB Act1.00 H0.62→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.768 Acc0.00 λ0.020 Bal0.150 | ChunkUp Act1.00 H0.32→0.62 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 0.00e+00(λ1.00e-03,ρcal) | GKL 0.517(γ0.0501) | SymD μ0.000 ↑n/a | Act 88.4%[84–97] | TopK-H 1.085/1.10(τ0.20) | α μ0.060[0.06–0.06](λ0.0050) | H 0.53→1.20 | BS 34x30(t1024) | Seen 0.0M(0.0x) | ∇ 4.81→1.79 | vel n/a | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 0.8G | Thr 0.1k t/s | β2 0.9900 | LR 1.50e-06 | ⏱ 00:00:18 (+18.61s/step, ETA 15:30:27)
  Gates: Bri=0.14 Dis=0.16 Foc=0.14 Col=0.13 Per=0.15 Siz=0.14 Pos=0.15
  TopK-H/stream: Bri=1.09 Dis=1.07 Foc=1.09 Col=1.09 Per=1.09 Siz=1.09 Pos=1.09
  ✦ Stream correlation baseline calibrated: 0.0094
Step   100/3000 | CE 7.7226 | PPL 2258.82 | VQ 2.6565(β0.21,cc0.227) | MetaCB Act1.00 H3.03→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.863 Acc0.03 λ0.020 Bal0.210 | ChunkUp Act1.00 H1.67→0.70 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 0.00e+00(λ1.22e-03,ρ0.009) | GKL 0.527(γ0.0601) | SymD μ0.000 ↑n/a | Act 83.9%[75–91] | TopK-H 1.094/1.10(τ0.20) | α μ0.060[0.06–0.06](λ0.0048) | H 2.57→1.34 | BS 16x64(t1024) | Seen 0.1M(0.2x) | ∇ 0.98→1.89 | vel n/a | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.5G | Thr 12.5k t/s | β2 0.9903 | LR 1.51e-04 | ⏱ 00:00:26 (+0.08s/step, ETA 00:03:57)
  ✦ Orth unlocked at step 180  (corr 0.0973 > baseline 0.0094 + 0.05)
Step   200/3000 | CE 6.6852 | PPL 800.47 | VQ 2.1855(β0.20,cc0.217) | MetaCB Act1.00 H2.81→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.928 Acc0.02 λ0.020 Bal0.173 | ChunkUp Act1.00 H1.73→0.77 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.26e-04(λ1.80e-03,ρ0.140) | GKL 0.523(γ0.0723) | SymD μ0.000 ↑n/a | Act 71.9%[66–78] | TopK-H 1.085/1.10(τ0.19) | α μ0.061[0.06–0.06](λ0.0046) | H 2.37→1.48 | BS 16x64(t1024) | Seen 0.2M(0.3x) | ∇ 1.16→1.45 | vel +0.0000 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.5G | Thr 15.8k t/s | β2 0.9906 | LR 3.00e-04 | ⏱ 00:00:33 (+0.06s/step, ETA 00:03:01)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 200)
  ↳ Val CE 6.8763  PPL 969.06  do→0.100  wd→0.0100  val_every=200  vel=+0.00000  ✓ best
Step   300/3000 | CE 6.2712 | PPL 529.13 | VQ 1.8855(β0.21,cc0.221) | MetaCB Act1.00 H2.84→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.885 Acc0.02 λ0.020 Bal0.128 | ChunkUp Act1.00 H1.78→0.84 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 3.51e-03(λ6.72e-03,ρ0.257) | GKL 0.514(γ0.0869) | SymD μ0.000 ↑n/a | Act 65.6%[53–75] | TopK-H 1.069/1.10(τ0.19) | α μ0.063[0.06–0.06](λ0.0044) | H 2.35→1.62 | BS 16x64(t1024) | Seen 0.3M(0.5x) | ∇ 1.86→1.76 | vel -0.0053 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 16.6k t/s | β2 0.9909 | LR 3.00e-04 | ⏱ 00:00:39 (+0.06s/step, ETA 00:02:46)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 400)
  ↳ Val CE 6.5603  PPL 706.50  do→0.099  wd→0.0099  val_every=200  vel=-0.00515  ✓ best
Step   412/3000 | CE 6.2805 | PPL 534.07 | VQ 1.6959(β0.23,cc0.230) | MetaCB Act1.00 H2.90→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.758 Acc0.03 λ0.020 Bal0.101 | ChunkUp Act1.00 H1.83→0.93 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 3.65e-02(λ5.15e-02,ρ0.317) | GKL 0.503(γ0.1063) | SymD μ0.000 ↑n/a | Act 63.4%[56–69] | TopK-H 1.059/1.10(τ0.18) | α μ0.064[0.06–0.06](λ0.0042) | H 2.37→1.77 | BS 16x64(t1024) | Seen 0.4M(0.6x) | ∇ 1.47→1.81 | vel -0.0006 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 19.3k t/s | β2 0.9912 | LR 3.00e-04 | ⏱ 00:00:45 (+0.05s/step, ETA 00:02:17)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 600)
  ↳ Val CE 6.4261  PPL 617.78  do→0.099  wd→0.0099  val_every=200  vel=-0.00383  ✓ best
Step   636/3000 | CE 5.6064 | PPL 272.15 | VQ 1.4984(β0.31,cc0.258) | MetaCB Act1.00 H3.22→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.635 Acc0.03 λ0.020 Bal0.081 | ChunkUp Act1.00 H1.90→1.09 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 6.03e-02(λ1.00e-01,ρ0.368) | GKL 0.524(γ0.1593) | SymD μ0.000 ↑n/a | Act 60.7%[50–69] | TopK-H 1.043/1.10(τ0.17) | α μ0.067[0.07–0.07](λ0.0038) | H 2.37→2.09 | BS 16x64(t1024) | Seen 0.7M(1.0x) | ∇ 1.31→1.99 | vel -0.0083 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 19.9k t/s | β2 0.9919 | LR 3.00e-04 | ⏱ 00:00:56 (+0.05s/step, ETA 00:02:01)
Step   800/3000 | CE 5.9071 | PPL 367.63 | VQ 1.4446(β0.38,cc0.262) | MetaCB Act1.00 H3.24→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.641 Acc0.03 λ0.020 Bal0.065 | ChunkUp Act1.00 H1.88→1.21 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.22e-02(λ1.00e-01,ρ0.389) | GKL 0.527(γ0.2140) | SymD μ0.000 ↑n/a | Act 61.2%[44–78] | TopK-H 1.029/1.10(τ0.17) | α μ0.070[0.07–0.07](λ0.0035) | H 2.40→2.32 | BS 16x64(t1024) | Seen 0.8M(1.2x) | ∇ 1.38→2.04 | vel +0.0004 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 15.4k t/s | β2 0.9924 | LR 3.00e-04 | ⏱ 00:01:07 (+0.07s/step, ETA 00:02:26)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 800)
  ↳ Val CE 6.2260  PPL 505.72  do→0.098  wd→0.0098  val_every=200  vel=+0.00038  ✓ best
Step   808/3000 | CE 5.9918 | PPL 400.14 | VQ 1.4414(β0.38,cc0.262) | MetaCB Act1.00 H3.24→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.652 Acc0.03 λ0.020 Bal0.071 | ChunkUp Act1.00 H1.87→1.21 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.19e-02(λ1.00e-01,ρ0.388) | GKL 0.512(γ0.2170) | SymD μ0.000 ↑n/a | Act 59.8%[47–75] | TopK-H 1.027/1.10(τ0.17) | α μ0.070[0.07–0.07](λ0.0035) | H 2.40→2.32 | BS 16x64(t1024) | Seen 0.8M(1.2x) | ∇ 1.35→2.04 | vel +0.0021 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 8.4k t/s | β2 0.9924 | LR 3.00e-04 | ⏱ 00:01:08 (+0.12s/step, ETA 00:04:26)
Step   824/3000 | CE 5.7025 | PPL 299.61 | VQ 1.4249(β0.39,cc0.261) | MetaCB Act1.00 H3.23→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.686 Acc0.02 λ0.020 Bal0.081 | ChunkUp Act1.00 H1.90→1.23 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.46e-02(λ1.00e-01,ρ0.388) | GKL 0.496(γ0.2233) | SymD μ0.000 ↑n/a | Act 58.0%[41–72] | TopK-H 1.018/1.10(τ0.16) | α μ0.070[0.07–0.07](λ0.0035) | H 2.40→2.35 | BS 16x64(t1024) | Seen 0.8M(1.3x) | ∇ 1.51→2.07 | vel -0.0036 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 21.0k t/s | β2 0.9925 | LR 3.00e-04 | ⏱ 00:01:09 (+0.05s/step, ETA 00:01:45)
Step   909/3000 | CE 5.9417 | PPL 380.59 | VQ 1.4020(β0.43,cc0.259) | MetaCB Act1.00 H3.25→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE3.711 Acc0.02 λ0.020 Bal0.070 | ChunkUp Act1.00 H1.90→1.29 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.01e-02(λ1.00e-01,ρ0.380) | GKL 0.471(γ0.2579) | SymD μ0.246 ↑0.396(Bri,Pos) | Act 58.9%[41–72] | TopK-H 1.011/1.10(τ0.16) | α μ0.071[0.07–0.07](λ0.0034) | H 2.38→2.47 | BS 16x64(t1024) | Seen 0.9M(1.4x) | ∇ 1.39→2.08 | vel +0.0027 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 20.7k t/s | β2 0.9927 | LR 3.00e-04 | ⏱ 00:01:13 (+0.05s/step, ETA 00:01:43)
Step   927/3000 | CE 5.9068 | PPL 367.53 | VQ 1.3469(β0.44,cc0.259) | MetaCB Act1.00 H3.22→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE3.648 Acc0.02 λ0.020 Bal0.074 | ChunkUp Act1.00 H1.88→1.30 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.39e-02(λ1.00e-01,ρ0.372) | GKL 0.449(γ0.2652) | SymD μ0.223 ↑0.367(Bri,Pos) | Act 57.1%[41–66] | TopK-H 0.977/1.10(τ0.16) | α μ0.072[0.07–0.07](λ0.0034) | H 2.33→2.49 | BS 16x64(t1024) | Seen 0.9M(1.4x) | ∇ 1.47→2.07 | vel +0.0023 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 17.1k t/s | β2 0.9928 | LR 3.00e-04 | ⏱ 00:01:14 (+0.06s/step, ETA 00:02:03)
Step   945/3000 | CE 5.7334 | PPL 309.03 | VQ 1.2957(β0.46,cc0.261) | MetaCB Act1.00 H3.21→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE3.664 Acc0.02 λ0.020 Bal0.071 | ChunkUp Act1.00 H1.88→1.31 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.22e-02(λ1.00e-01,ρ0.356) | GKL 0.432(γ0.2722) | SymD μ0.208 ↑0.348(Bri,Pos) | Act 57.6%[47–66] | TopK-H 0.939/1.10(τ0.16) | α μ0.072[0.07–0.07](λ0.0033) | H 2.27→2.52 | BS 16x64(t1024) | Seen 1.0M(1.5x) | ∇ 1.47→2.07 | vel -0.0014 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 17.1k t/s | β2 0.9928 | LR 3.00e-04 | ⏱ 00:01:15 (+0.06s/step, ETA 00:02:02)
Step   963/3000 | CE 5.6565 | PPL 286.14 | VQ 1.2679(β0.47,cc0.264) | MetaCB Act1.00 H3.18→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE3.617 Acc0.03 λ0.020 Bal0.070 | ChunkUp Act1.00 H1.88→1.33 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.52e-02(λ1.00e-01,ρ0.342) | GKL 0.420(γ0.2789) | SymD μ0.211 ↑0.330(Bri,Pos) | Act 56.7%[50–62] | TopK-H 0.915/1.10(τ0.16) | α μ0.072[0.07–0.07](λ0.0033) | H 2.22→2.55 | BS 16x64(t1024) | Seen 1.0M(1.5x) | ∇ 1.39→2.14 | vel -0.0028 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 17.3k t/s | β2 0.9929 | LR 3.00e-04 | ⏱ 00:01:16 (+0.06s/step, ETA 00:02:00)
Step   981/3000 | CE 5.5320 | PPL 252.64 | VQ 1.2473(β0.48,cc0.267) | MetaCB Act1.00 H3.19→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE3.623 Acc0.03 λ0.020 Bal0.059 | ChunkUp Act1.00 H1.86→1.34 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.60e-02(λ1.00e-01,ρ0.332) | GKL 0.383(γ0.2853) | SymD μ0.198 ↑0.317(Bri,Pos) | Act 54.9%[47–66] | TopK-H 0.900/1.10(τ0.16) | α μ0.073[0.07–0.07](λ0.0033) | H 2.20→2.58 | BS 16x64(t1024) | Seen 1.0M(1.5x) | ∇ 1.58→2.19 | vel -0.0053 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 1.6G | Thr 17.1k t/s | β2 0.9929 | LR 3.00e-04 | ⏱ 00:01:18 (+0.06s/step, ETA 00:02:01)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 1000)
  ↳ Val CE 6.1129  PPL 451.64  do→0.098  wd→0.0098  val_every=200  vel=-0.00331  ✓ best
  ↓ LR 3.00e-04 → 2.97e-04  (vel +0.00165 ≥ +0.00020, reduction #1)
  ↑ batch_tokens → 2048
Step  1200/3000 | CE 5.6020 | PPL 270.98 | VQ 1.2633(β0.50,cc0.311) | MetaCB Act1.00 H3.12→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE3.568 Acc0.03 λ0.020 Bal0.049 | ChunkUp Act1.00 H1.77→1.61 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 5.05e-02(λ1.00e-01,ρ0.331) | GKL 0.000(γ0.2991) | SymD μ0.187 ↑0.317(Per,Pos) | Act 57.1%[47–66] | TopK-H 0.922/1.10(τ0.15) | α μ0.076[0.07–0.08](λ0.0030) | H 2.13→3.09 | BS 32x64(t2048) | Seen 1.3M(2.0x) | ∇ 1.20→1.84 | vel -0.0004 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 2.2G | Thr 32.4k t/s | β2 0.9936 | LR 2.97e-04 | ⏱ 00:01:31 (+0.06s/step, ETA 00:01:53)
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 1200)
  ↳ Val CE 5.9934  PPL 400.76  do→0.097  wd→0.0097  val_every=400  vel=-0.00041  ✓ best
Step  1300/3000 | CE 5.5083 | PPL 246.73 | VQ 1.2864(β0.50,cc0.330) | MetaCB Act1.00 H3.15→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE3.551 Acc0.03 λ0.020 Bal0.054 | ChunkUp Act1.00 H1.80→1.69 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.98e-02(λ1.00e-01,ρ0.329) | GKL 0.000(γ0.2778) | SymD μ0.190 ↑0.326(Per,Pos) | Act 60.7%[56–81] | TopK-H 0.935/1.10(τ0.14) | α μ0.077[0.08–0.08](λ0.0029) | H 2.14→3.12 | BS 32x64(t2048) | Seen 1.5M(2.3x) | ∇ 1.19→2.08 | vel -0.0011 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 2.4G | Thr 26.9k t/s | β2 0.9939 | LR 2.97e-04 | ⏱ 00:01:39 (+0.08s/step, ETA 00:02:09)
  ↓ LR 2.97e-04 → 2.94e-04  (vel +0.00226 ≥ +0.00020, reduction #2)
  ↑ batch_tokens → 4096
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 1600)
  ↳ Val CE 5.8535  PPL 348.46  do→0.097  wd→0.0097  val_every=400  vel=-0.00191  ✓ best
Step  1616/3000 | CE 5.3114 | PPL 202.63 | VQ 1.3296(β0.49,cc0.406) | MetaCB Act1.00 H3.21→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE3.635 Acc0.03 λ0.020 Bal0.048 | ChunkUp Act1.00 H1.88→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.54e-02(λ1.00e-01,ρ0.339) | GKL 0.000(γ0.2197) | SymD μ0.200 ↑0.327(Per,Pos) | Act 75.4%[56–97] | TopK-H 0.955/1.10(τ0.13) | α μ0.083[0.08–0.08](λ0.0026) | H 2.19→3.12 | BS 64x64(t4096) | Seen 2.4M(3.6x) | ∇ 1.17→1.66 | vel -0.0011 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 4.4G | Thr 46.1k t/s | β2 0.9948 | LR 2.94e-04 | ⏱ 00:02:07 (+0.09s/step, ETA 00:02:02)
Step  1768/3000 | CE 5.2937 | PPL 199.08 | VQ 1.3391(β0.46,cc0.440) | MetaCB Act1.00 H3.25→3.12 β0.15 cc0.150 TopKH1.09 | MetaPred CE2.787 Acc0.22 λ0.020 Bal0.161 | ChunkUp Act1.00 H1.85→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.40e-02(λ1.00e-01,ρ0.348) | GKL 0.000(γ0.1962) | SymD μ0.206 ↑0.322(Per,Pos) | Act 78.1%[56–97] | TopK-H 0.963/1.10(τ0.12) | α μ0.086[0.08–0.09](λ0.0025) | H 2.21→3.12 | BS 64x64(t4096) | Seen 3.0M(4.6x) | ∇ 1.24→1.80 | vel +0.0001 | do 0.10 | L 8 | S 7/7 | CB 32 | Wmup - | VRAM 4.4G | Thr 52.4k t/s | β2 0.9953 | LR 2.94e-04 | ⏱ 00:02:19 (+0.08s/step, ETA 00:01:36)
    ↑ Codebook grown: 32 → 48 symbols
    ↑ Growth LR reset: 5.88e-04 → 2.94e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
Step  1836/3000 | CE 5.2202 | PPL 184.97 | VQ 1.3293(β0.43,cc0.451) | MetaCB Act1.00 H3.25→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.775 Acc0.21 λ0.020 Bal0.160 | ChunkUp Act1.00 H1.82→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.54e-02(λ1.00e-01,ρ0.354) | GKL 0.000(γ0.1865) | SymD μ0.213 ↑0.324(Per,Pos) | Act 79.2%[56–98] | TopK-H 0.958/1.10(τ0.12) | α μ0.087[0.09–0.09](λ0.0024) | H 2.24→3.12 | BS 64x64(t4096) | Seen 3.3M(5.0x) | ∇ 1.40→1.81 | vel -0.0004 | do 0.10 | L 8 | S 7/7 | CB 34 | Wmup W1@464 | VRAM 4.4G | Thr 14.8k t/s | β2 0.9955 | LR 2.94e-04 | ⏱ 00:02:38 (+0.28s/step, ETA 00:05:21)
Step  1890/3000 | CE 5.3305 | PPL 206.54 | VQ 1.3328(β0.41,cc0.458) | MetaCB Act1.00 H3.26→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.799 Acc0.21 λ0.020 Bal0.166 | ChunkUp Act1.00 H1.82→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.56e-02(λ1.00e-01,ρ0.360) | GKL 0.000(γ0.1792) | SymD μ0.218 ↑0.329(Per,Pos) | Act 83.6%[56–100] | TopK-H 0.956/1.10(τ0.12) | α μ0.089[0.09–0.09](λ0.0024) | H 2.26→3.12 | BS 64x64(t4096) | Seen 3.5M(5.3x) | ∇ 1.32→1.90 | vel +0.0023 | do 0.10 | L 8 | S 7/7 | CB 34 | Wmup W1@410 | VRAM 4.4G | Thr 41.7k t/s | β2 0.9957 | LR 2.94e-04 | ⏱ 00:02:43 (+0.10s/step, ETA 00:01:48)
Step  1919/3000 | CE 5.2952 | PPL 199.38 | VQ 1.3371(β0.40,cc0.460) | MetaCB Act1.00 H3.24→3.12 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.762 Acc0.21 λ0.020 Bal0.175 | ChunkUp Act1.00 H1.83→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.48e-02(λ1.00e-01,ρ0.362) | GKL 0.011(γ0.1754) | SymD μ0.217 ↑0.335(Bri,Pos) | Act 82.3%[56–100] | TopK-H 0.959/1.10(τ0.12) | α μ0.089[0.09–0.09](λ0.0024) | H 2.26→3.12 | BS 64x64(t4096) | Seen 3.7M(5.5x) | ∇ 1.21→1.91 | vel +0.0019 | do 0.10 | L 8 | S 7/7 | CB 34 | Wmup W1@381 | VRAM 4.4G | Thr 41.6k t/s | β2 0.9958 | LR 2.94e-04 | ⏱ 00:02:46 (+0.10s/step, ETA 00:01:46)
    ↑ Codebook grown: 32 → 48 symbols
    ↑ Growth LR reset: 5.88e-04 → 2.94e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
    ↑ Codebook grown: 32 → 48 symbols
    ↑ Growth LR reset: 5.88e-04 → 2.94e-04 over 500 steps
    ↑ meta_pred_head grown: 32 → 48 classes
    ↑ Growth LR reset: 5.88e-04 → 2.94e-04 over 500 steps
  ↑ Meta-codebook grown: 32 → 48 symbols
  💾 Checkpoint saved (best val): neuro_symbolic_model_best.pt  (step 2000)
  ↳ Val CE 5.7805  PPL 323.93  do→0.096  wd→0.0096  val_every=400  vel=-0.00147  ✓ best
  ⏳ step 2029/3000[00:03:16 elapsed]  still running…
Step  2100/3000 | CE 5.0994 | PPL 163.93 | VQ 1.3228(β0.32,cc0.460) | MetaCB Act0.98 H3.41→3.48 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.873 Acc0.23 λ0.020 Bal0.260 | ChunkUp Act1.00 H1.87→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.42e-02(λ1.00e-01,ρ0.377) | GKL 0.003(γ0.1533) | SymD μ0.225 ↑0.348(Bri,Pos) | Act 85.0%[56–100] | TopK-H 0.952/1.10(τ0.11) | α μ0.093[0.09–0.10](λ0.0023) | H 2.31→3.12 | BS 64x64(t4096) | Seen 4.4M(6.6x) | ∇ 1.33→1.97 | vel -0.0000 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W4@200 | VRAM 4.4G | Thr 19.9k t/s | β2 0.9963 | LR 2.94e-04 | ⏱ 00:03:23 (+0.21s/step, ETA 00:03:04)
  Gates: Bri=0.18 Dis=0.07 Foc=0.32 Col=0.05 Per=0.08 Siz=0.25 Pos=0.05
  TopK-H/stream: Bri=0.96 Dis=0.94 Foc=0.97 Col=0.96 Per=0.96 Siz=0.91 Pos=0.95
Step  2142/3000 | CE 5.0491 | PPL 155.89 | VQ 1.3209(β0.31,cc0.454) | MetaCB Act0.98 H3.42→3.48 β0.15 cc0.150 TopKH1.08 | MetaPred CE2.812 Acc0.23 λ0.020 Bal0.253 | ChunkUp Act1.00 H1.90→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.38e-02(λ1.00e-01,ρ0.379) | GKL 0.000(γ0.1486) | SymD μ0.230 ↑0.350(Bri,Pos) | Act 84.7%[56–100] | TopK-H 0.951/1.10(τ0.11) | α μ0.094[0.09–0.10](λ0.0022) | H 2.32→3.12 | BS 64x64(t4096) | Seen 4.6M(6.9x) | ∇ 1.24→1.91 | vel -0.0006 | do 0.10 | L 8 | S 7/7 | CB 37 | Wmup W4@158 | VRAM 4.4G | Thr 43.5k t/s | β2 0.9964 | LR 2.94e-04 | ⏱ 00:03:27 (+0.09s/step, ETA 00:01:20)
  ↓ LR 2.94e-04 → 2.91e-04  (vel +0.00062 ≥ +0.00020, reduction #3)
  ↑ batch_tokens → 4096
    ↑ Codebook grown: 48 → 54 symbols
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps
    ↑ Codebook grown: 32 → 48 symbols
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
Step  2200/3000 | CE 5.0746 | PPL 159.91 | VQ 1.3201(β0.28,cc0.444) | MetaCB Act0.96 H3.43→3.48 β0.15 cc0.150 TopKH1.07 | MetaPred CE2.812 Acc0.23 λ0.020 Bal0.247 | ChunkUp Act1.00 H1.92→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.45e-02(λ1.00e-01,ρ0.383) | GKL 0.000(γ0.1423) | SymD μ0.238 ↑0.357(Bri,Pos) | Act 86.2%[56–100] | TopK-H 0.946/1.10(τ0.10) | α μ0.095[0.09–0.10](λ0.0022) | H 2.34→3.12 | BS 64x64(t4096) | Seen 4.8M(7.2x) | ∇ 1.30→1.96 | vel +0.0006 | do 0.10 | L 8 | S 7/7 | CB 40 | Wmup W6@100 | VRAM 4.4G | Thr 51.0k t/s | β2 0.9966 | LR 2.91e-04 | ⏱ 00:03:32 (+0.08s/step, ETA 00:01:04)
Step  2244/3000 | CE 5.0359 | PPL 153.83 | VQ 1.3220(β0.27,cc0.437) | MetaCB Act0.96 H3.45→3.48 β0.15 cc0.150 TopKH1.07 | MetaPred CE2.852 Acc0.24 λ0.020 Bal0.239 | ChunkUp Act1.00 H1.93→1.87 β0.15 | ChunkDn MSE0.000 λ0.050 | Orth 4.42e-02(λ1.00e-01,ρ0.387) | GKL 0.000(γ0.1376) | SymD μ0.234 ↑0.353(Bri,Pos) | Act 86.3%[56–100] | TopK-H 0.946/1.10(τ0.10) | α μ0.096[0.09–0.10](λ0.0022) | H 2.36→3.12 | BS 64x64(t4096) | Seen 5.0M(7.5x) | ∇ 1.24→2.03 | vel +0.0003 | do 0.10 | L 8 | S 7/7 | CB 40 | Wmup W6@56 | VRAM 4.4G | Thr 10.3k t/s | β2 0.9967 | LR 2.91e-04 | ⏱ 00:03:49 (+0.40s/step, ETA 00:05:00)
Step  2346/3000 | CE 5.0383 | PPL 154.21 | VQ 1.3272(β0.23,cc0.414) | MetaCB Act0.98 H3.47→3.48 β0.15 cc0.150 TopKH1.07 | MetaPred CE2.826 Acc0.23 λ0.020 Bal0.229 | ChunkUp Act1.00 H1.60→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 4.33e-02(λ1.00e-01,ρ0.392) | GKL 0.000(γ0.1275) | SymD μ0.238 ↑0.368(Bri,Pos) | Act 85.9%[56–98] | TopK-H 0.943/1.10(τ0.10) | α μ0.099[0.10–0.10](λ0.0021) | H 2.36→3.12 | BS 64x64(t4096) | Seen 5.4M(8.1x) | ∇ 1.42→2.12 | vel +0.0012 | do 0.10 | L 8 | S 7/7 | CB 40 | Wmup W5@154 | VRAM 4.4G | Thr 38.5k t/s | β2 0.9970 | LR 2.91e-04 | ⏱ 00:04:00 (+0.11s/step, ETA 00:01:09)
    ↑ Codebook grown: 32 → 48 symbols
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps
    ↑ Codebook grown: 48 → 54 symbols
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
    ↑ Codebook grown: 48 → 54 symbols
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps
    ↑ meta_pred_head grown: 48 → 54 classes
    ↑ Growth LR reset: 5.82e-04 → 2.91e-04 over 500 steps
  ↑ Meta-codebook grown: 48 → 54 symbols
  ↳ Val CE 5.8356  PPL 342.27  do→0.096  wd→0.0096  val_every=400  vel=-0.00002
  ⏳ step 2499/3000[00:04:30 elapsed]  still running…
Step  2678/3000 | CE 4.7496 | PPL 115.54 | VQ 1.3366(β0.15,cc0.323) | MetaCB Act0.91 H3.51→3.59 β0.15 cc0.150 TopKH1.06 | MetaPred CE2.799 Acc0.23 λ0.020 Bal0.244 | ChunkUp Act1.00 H0.86→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 4.32e-02(λ1.00e-01,ρ0.406) | GKL 0.000(γ0.0994) | SymD μ0.247 ↑0.367(Bri,Pos) | Act 83.4%[56–96] | TopK-H 0.919/1.10(τ0.08) | α μ0.106[0.10–0.11](λ0.0019) | H 2.41→3.12 | BS 64x64(t4096) | Seen 6.8M(10.2x) | ∇ 1.34→2.21 | vel -0.0018 | do 0.10 | L 8 | S 7/7 | CB 43 | Wmup W6@22 | VRAM 4.4G | Thr 26.7k t/s | β2 0.9980 | LR 2.88e-04 | ⏱ 00:04:51 (+0.15s/step, ETA 00:00:49)
Step  2725/3000 | CE 4.6352 | PPL 103.05 | VQ 1.3617(β0.15,cc0.310) | MetaCB Act0.93 H3.51→3.59 β0.15 cc0.150 TopKH1.06 | MetaPred CE2.814 Acc0.24 λ0.020 Bal0.238 | ChunkUp Act1.00 H0.92→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 4.14e-02(λ1.00e-01,ρ0.411) | GKL 0.000(γ0.0960) | SymD μ0.253 ↑0.381(Bri,Pos) | Act 85.2%[56–96] | TopK-H 0.927/1.10(τ0.08) | α μ0.107[0.10–0.11](λ0.0019) | H 2.42→3.12 | BS 64x64(t4096) | Seen 7.0M(10.5x) | ∇ 1.68→2.13 | vel -0.0036 | do 0.10 | L 8 | S 7/7 | CB 43 | Wmup W4@175 | VRAM 4.4G | Thr 34.5k t/s | β2 0.9982 | LR 2.88e-04 | ⏱ 00:04:56 (+0.12s/step, ETA 00:00:32)
  Gates: Bri=0.13 Dis=0.09 Foc=0.33 Col=0.04 Per=0.09 Siz=0.27 Pos=0.05
  TopK-H/stream: Bri=0.93 Dis=0.93 Foc=0.93 Col=0.94 Per=0.93 Siz=0.90 Pos=0.93
    ↑ Codebook grown: 48 → 54 symbols
    ↑ Growth LR reset: 5.76e-04 → 2.88e-04 over 500 steps

[JIT] Recompiling model after codebook growth…
  ↳ Val CE 5.7867  PPL 325.93  do→0.095  wd→0.0095  val_every=400  vel=-0.00345
Step  2828/3000 | CE 4.6966 | PPL 109.57 | VQ 1.3556(β0.15,cc0.281) | MetaCB Act0.89 H3.52→3.59 β0.15 cc0.150 TopKH1.05 | MetaPred CE2.852 Acc0.24 λ0.020 Bal0.230 | ChunkUp Act1.00 H1.01→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 4.24e-02(λ1.00e-01,ρ0.414) | GKL 0.000(γ0.0888) | SymD μ0.248 ↑0.379(Bri,Pos) | Act 86.5%[56–94] | TopK-H 0.910/1.10(τ0.08) | α μ0.110[0.11–0.11](λ0.0019) | H 2.44→3.12 | BS 64x64(t4096) | Seen 7.4M(11.1x) | ∇ 1.47→2.21 | vel -0.0014 | do 0.10 | L 8 | S 7/7 | CB 44 | Wmup W5@72 | VRAM 4.4G | Thr 17.0k t/s | β2 0.9985 | LR 2.85e-04 | ⏱ 00:05:21 (+0.24s/step, ETA 00:00:41)
Step  2889/3000 | CE 4.7410 | PPL 114.54 | VQ 1.3563(β0.15,cc0.265) | MetaCB Act0.87 H3.53→3.59 β0.15 cc0.150 TopKH1.05 | MetaPred CE2.836 Acc0.23 λ0.020 Bal0.224 | ChunkUp Act1.00 H1.05→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 4.41e-02(λ1.00e-01,ρ0.415) | GKL 0.000(γ0.0848) | SymD μ0.252 ↑0.392(Bri,Pos) | Act 86.2%[56–100] | TopK-H 0.899/1.10(τ0.08) | α μ0.111[0.11–0.12](λ0.0019) | H 2.44→3.12 | BS 64x64(t4096) | Seen 7.6M(11.5x) | ∇ 1.41→2.26 | vel -0.0003 | do 0.10 | L 8 | S 7/7 | CB 44 | Wmup W5@11 | VRAM 4.4G | Thr 40.5k t/s | β2 0.9987 | LR 2.85e-04 | ⏱ 00:05:27 (+0.10s/step, ETA 00:00:11)
Step  2929/3000 | CE 4.5962 | PPL 99.10 | VQ 1.3725(β0.15,cc0.254) | MetaCB Act0.85 H3.53→3.59 β0.15 cc0.150 TopKH1.05 | MetaPred CE2.850 Acc0.23 λ0.020 Bal0.212 | ChunkUp Act1.00 H1.06→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 4.18e-02(λ1.00e-01,ρ0.417) | GKL 0.000(γ0.0823) | SymD μ0.252 ↑0.372(Bri,Pos) | Act 88.2%[56–97] | TopK-H 0.905/1.10(τ0.07) | α μ0.112[0.11–0.12](λ0.0019) | H 2.44→3.12 | BS 64x64(t4096) | Seen 7.8M(11.7x) | ∇ 1.48→2.30 | vel -0.0026 | do 0.10 | L 8 | S 7/7 | CB 44 | Wmup W1@371 | VRAM 4.4G | Thr 34.9k t/s | β2 0.9988 | LR 2.85e-04 | ⏱ 00:05:32 (+0.12s/step, ETA 00:00:08)
Step  2996/3000 | CE 4.6782 | PPL 107.58 | VQ 1.3809(β0.15,cc0.236) | MetaCB Act0.87 H3.54→3.59 β0.15 cc0.150 TopKH1.05 | MetaPred CE2.875 Acc0.23 λ0.020 Bal0.216 | ChunkUp Act1.00 H1.09→1.87 β0.15 | ChunkDn MSE0.004 λ0.050 | Orth 4.21e-02(λ1.00e-01,ρ0.419) | GKL 0.000(γ0.0783) | SymD μ0.254 ↑0.383(Bri,Pos) | Act 87.5%[56–98] | TopK-H 0.899/1.10(τ0.07) | α μ0.114[0.11–0.12](λ0.0018) | H 2.45→3.12 | BS 64x64(t4096) | Seen 8.1M(12.2x) | ∇ 1.44→2.26 | vel -0.0008 | do 0.10 | L 8 | S 7/7 | CB 44 | Wmup W1@304 | VRAM 4.4G | Thr 35.0k t/s | β2 0.9990 | LR 2.85e-04 | ⏱ 00:05:40 (+0.12s/step, ETA 00:00:00)
Step  2999/3000 | CE 4.5950 | PPL 98.99 | VQ 1.3813(β0.15,cc0.235) | MetaCB Act0.85 H3.53→3.59 β0.15 cc0.150 TopKH1.05 | MetaPred CE2.834 Acc0.23 λ0.020 Bal0.226 | ChunkUp Act1.00 H1.09→1.87 β0.15 | ChunkDn MSE0.003 λ0.050 | Orth 4.25e-02(λ1.00e-01,ρ0.419) | GKL 0.000(γ0.0781) | SymD μ0.257 ↑0.386(Bri,Pos) | Act 85.6%[56–96] | TopK-H 0.899/1.10(τ0.07) | α μ0.114[0.11–0.12](λ0.0018) | H 2.45→3.12 | BS 64x64(t4096) | Seen 8.1M(12.2x) | ∇ 1.49→2.26 | vel -0.0024 | do 0.10 | L 8 | S 7/7 | CB 44 | Wmup W1@301 | VRAM 4.4G | Thr 34.8k t/s | β2 0.9990 | LR 2.85e-04 | ⏱ 00:05:40 (+0.12s/step, ETA 00:00:00)
  Gates: Bri=0.14 Dis=0.09 Foc=0.35 Col=0.03 Per=0.09 Siz=0.24 Pos=0.06
  TopK-H/stream: Bri=0.91 Dis=0.90 Foc=0.90 Col=0.91 Per=0.91 Siz=0.87 Pos=0.89
  ↑ seq_len_max unlocked → 128  (max val checks reached)
  ↳ Val CE 5.8147  PPL 335.20  do→0.095  wd→0.0095  val_every=400  vel=-0.00237
  💾 Checkpoint saved (final): neuro_symbolic_model.pt  (step 2999)

Final checkpoint  → neuro_symbolic_model.pt
Best checkpoint   → neuro_symbolic_model_best.pt  (val CE 5.7805)
Total training time: 00:05:41  (0.09 hours)

============================================================
  NEURO-SYMBOLIC EXPLAINABILITY & GENERATION DEMO
============================================================

SEED PROMPT:
----------------------------------------
First Citizen:
Before we proceed any further, hear me
----------------------------------------

Internal submodality representation of the final token (Top-K routing):

  Brightness   → #023  #018  #000
  Distance     → #005  #029  #026
  Focus        → #011  #008  #000
  Color        → #016  #027  #019
  Perspective  → #002  #015  #004
  Size         → #010  #006  #030
  Position     → #001  #026  #000

  H_ema (smoothed entropy): 2.473  |  Active: 31.1%

Generating 60 new tokens…

GENERATED TEXT:
--------------------------------------------------
First Citizen : Before we proceed any further , hear me this . Third Servingman : So did the duke call to go on my master ' s grace in his heart and not for him ; I am going to be at once that time of the king . Second Watchman : My lord is the queen with you ; And he hath sent a most less true fools too
--------------------------------------------------

============================================================
  RUNNING XAI 'MIND READER' VISUALIZATION...
============================================================
Opening heatmap window… (close the window to end the script)

============================================================
  DECODING AI PSYCHOLOGY (AUTO-LABELER)...
============================================================

► BUCKET 0  (Named: 'Brightness')
  Top Words: Duke, refrain, Lucentio, XANTHIAS, patience, more, HENRY, horse, DUCHESS, find, occasion, fearful, heard, fine, report, greater, Pray, certain, gentleman, sleep

► BUCKET 1  (Named: 'Distance')
  Top Words: ignorantly, scatter, torches, sheath, pays, cherish, motherland, struck, nigh, dreaded, stopp, tent, repast, hindrance, relief, crouch, tusks, flashing, obeying, gale

► BUCKET 2  (Named: 'Focus')
  Top Words: foot, envy, blame, just, Which, hence, arms, human, since, made, less, because, gods, sweet, Apollo, stand, hold, thee, help, along

► BUCKET 3  (Named: 'Color')
  Top Words: serious, during, cost, view, addressed, taking, deserved, laugh, trial, suspicion, discourse, virtue, valour, seen, pleasant, taught, white, dozen, feast, Cypris

► BUCKET 4  (Named: 'Perspective')
  Top Words: persecution, NICHOLAS, Aeaea, earning, cannibally, scourged, legislating, spire, Shepherding, lashing, Mardian, culprits, Advocate, salter, Chremylus, Stoicism, glint, brothels, Geusistrate, Sooth

► BUCKET 5  (Named: 'Size')
  Top Words: upon, From, STREPSIADES, fire, past, below, Hera, along, When, PRAXAGORA, always, singing, Even, suddenly, weary, thither, Apollo, royal, gods, because

► BUCKET 6  (Named: 'Position')
  Top Words: field, loud, lips, uncle, didst, beloved, strike, wonder, sire, receive, judges, madness, brothers, destruction, wide, nought, swiftly, burning, fresh, sovereign

============================================================

============================================================
  SYMBOL MUTUAL INFORMATION DIAGNOSTIC
============================================================
  Estimating MI(symbol ; token) per submodality stream…
  Sampling 8000 tokens from data.txt.

  Stream          MI(sym;tok)   Quality
  ------------------------------------------
  Brightness            2.310   ✓ high     — symbols encode meaningful structure
  Distance              2.445   ✓ high     — symbols encode meaningful structure
  Focus                 2.394   ✓ high     — symbols encode meaningful structure
  Color                 2.325   ✓ high     — symbols encode meaningful structure
  Perspective           2.321   ✓ high     — symbols encode meaningful structure
  Size                  2.313   ✓ high     — symbols encode meaningful structure
  Position              2.323   ✓ high     — symbols encode meaningful structure

  Mean MI across streams: 2.347

  Meta-codebook decision: GO  ✓  — level-1 symbols are semantically meaningful.
         Meta-codebook experiment is well-motivated.
============================================================
