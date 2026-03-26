# `train_gpt.py` — Comprehensive Documentation

> A deep-dive into every component of the parameter-golf GPT training script.
> This document follows the same section order as the source code for easy cross-referencing.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Script Flow](#2-script-flow)
3. [Hyperparameters](#3-hyperparameters)
4. [Muon Optimizer](#4-muon-optimizer)
5. [Tokenizer-Agnostic Evaluation (BPB)](#5-tokenizer-agnostic-evaluation-bpb)
6. [Post-Training Quantization](#6-post-training-quantization)
7. [Data Loading](#7-data-loading)
8. [Transformer Modules](#8-transformer-modules)
   - [RMSNorm](#81-rmsnorm)
   - [CastedLinear](#82-castedlinear)
   - [Rotary Position Embeddings (RoPE)](#83-rotary-position-embeddings-rope)
   - [CausalSelfAttention](#84-causalselfAttention)
   - [MLP](#85-mlp)
   - [Block](#86-block)
   - [GPT (Full Model)](#87-gpt-full-model)
9. [Training (`main()`)](#9-training-main)
   - [Distributed + CUDA Setup](#91-distributed--cuda-setup)
   - [Tokenizer + Validation Setup](#92-tokenizer--validation-setup)
   - [Model + Optimizer Setup](#93-model--optimizer-setup)
   - [Compilation Warmup](#94-compilation-warmup)
   - [Learning Rate Schedule](#95-learning-rate-schedule)
   - [Main Training Loop](#96-main-training-loop)
   - [Serialization + Round-Trip Validation](#97-serialization--round-trip-validation)
10. [Appendix: Hyperparameter Reference Table](#10-appendix-hyperparameter-reference-table)

---

## 1. Overview

`train_gpt.py` is a self-contained script that **defines, trains, quantizes, and serializes** a small GPT language model for the _parameter-golf_ challenge. The challenge constraints are:

| Constraint                   | Default                                       |
| ---------------------------- | --------------------------------------------- |
| **Model + code size**        | ≤ 16 MB (int8 + zlib compressed)              |
| **Wall-clock training time** | ≤ 10 minutes                                  |
| **Evaluation metric**        | Bits-per-byte (BPB) on FineWeb validation set |

The script is intentionally kept under 1500 lines to remain readable for newcomers.

### Key Design Choices

- **Architecture**: A GPT-2-style decoder-only transformer with modern enhancements — RoPE, GQA, RMSNorm, U-Net skip connections, logit softcapping, and squared-ReLU MLP.
- **Optimizer**: A split strategy — the **Muon** optimizer (Newton-Schulz–orthogonalized SGD with momentum) for 2D weight matrices, and **Adam** for embeddings, scalars, and control parameters.
- **Precision**: Mixed bf16/fp32 training. Weights stored in fp32 for optimizer quality, cast to bf16 at matmul time.
- **Serialization**: Post-training int8 quantization with per-row scales + zlib compression to fit under the size cap.

---

## 2. Script Flow

```mermaid
flowchart TD
    A["🚀 main()"] --> B["Parse Hyperparameters"]
    B --> C["torch.compile Newton-Schulz"]
    C --> D["Distributed + CUDA Setup"]
    D --> E["Load Tokenizer + Validation Data"]
    E --> F["Build BPB Lookup Tables"]
    F --> G["Initialize GPT Model"]
    G --> H["Set Up Optimizers\n(Adam × 2-3 + Muon × 1)"]
    H --> I{"warmup_steps > 0?"}
    I -- Yes --> J["Compilation Warmup\n(run N steps, then restore initial state)"]
    J --> K["Main Training Loop"]
    I -- No --> K
    K --> L{"Every val_loss_every steps\nor last step?"}
    L -- Yes --> M["Run Validation\n(val_loss + val_bpb)"]
    L -- No --> N["Forward + Backward\n(gradient accumulation)"]
    M --> N
    N --> O["Update LR schedule\n(wallclock-aware warmdown)"]
    O --> P["Optimizer Steps"]
    P --> Q{"Wallclock cap\nreached?"}
    Q -- No --> K
    Q -- Yes --> R["Final Validation"]
    R --> S["Save raw model\n(final_model.pt)"]
    S --> T["Int8 Quantize + zlib Compress\n(final_model.int8.ptz)"]
    T --> U["Round-trip Validation\n(dequantize → eval → report)"]
    U --> V["🏁 Done"]

    style A fill:#4CAF50,color:white
    style V fill:#4CAF50,color:white
    style T fill:#FF9800,color:white
    style M fill:#2196F3,color:white
```

---

## 3. Hyperparameters

The `Hyperparameters` class (lines 39–90) centralizes every tunable knob. **All values are overridable via environment variables**, making it easy to sweep without editing the script.

### 3.1 Data & I/O

| Parameter        | Default                                    | Description                         |
| ---------------- | ------------------------------------------ | ----------------------------------- |
| `data_path`      | `./data/datasets/fineweb10B_sp1024`        | Root directory for tokenized shards |
| `train_files`    | `{data_path}/fineweb_train_*.bin`          | Glob for training shards            |
| `val_files`      | `{data_path}/fineweb_val_*.bin`            | Glob for validation shards          |
| `tokenizer_path` | `./data/tokenizers/fineweb_1024_bpe.model` | SentencePiece `.model` file         |
| `run_id`         | random UUID                                | Unique identifier for logging       |
| `seed`           | `1337`                                     | Random seed for reproducibility     |

### 3.2 Validation & Logging

| Parameter         | Default   | Description                     |
| ----------------- | --------- | ------------------------------- |
| `val_batch_size`  | `524,288` | Tokens per validation pass      |
| `val_loss_every`  | `1,000`   | Validate every N training steps |
| `train_log_every` | `200`     | Log training loss every N steps |

### 3.3 Training Length

| Parameter               | Default   | Description                                              |
| ----------------------- | --------- | -------------------------------------------------------- |
| `iterations`            | `20,000`  | Maximum training steps                                   |
| `warmdown_iters`        | `1,200`   | Steps over which LR decays to 0                          |
| `warmup_steps`          | `20`      | `torch.compile` warmup iterations (state restored after) |
| `train_batch_tokens`    | `524,288` | Global tokens per step (across all GPUs)                 |
| `train_seq_len`         | `1,024`   | Sequence length                                          |
| `max_wallclock_seconds` | `600.0`   | Hard wall-clock cap (10 minutes)                         |
| `qk_gain_init`          | `1.5`     | Initial value for learnable Q scaling                    |

### 3.4 Model Shape

| Parameter        | Default    | Description                                   |
| ---------------- | ---------- | --------------------------------------------- |
| `vocab_size`     | `1,024`    | Vocabulary size (must match tokenizer)        |
| `num_layers`     | `9`        | Total transformer blocks                      |
| `num_kv_heads`   | `4`        | Key/Value heads (GQA)                         |
| `model_dim`      | `512`      | Hidden dimension                              |
| `num_heads`      | `8`        | Query heads                                   |
| `mlp_mult`       | `2`        | MLP expansion factor (hidden = dim × mult)    |
| `tie_embeddings` | `True`     | Share embedding and output projection weights |
| `rope_base`      | `10,000.0` | RoPE frequency base                           |
| `logit_softcap`  | `30.0`     | Softcap value for output logits               |

### 3.5 Optimizer

| Parameter                    | Default        | Description                             |
| ---------------------------- | -------------- | --------------------------------------- |
| `embed_lr`                   | `0.6`          | LR for untied embedding (Adam)          |
| `head_lr`                    | `0.008`        | LR for untied lm_head (Adam)            |
| `tied_embed_lr`              | `0.05`         | LR for tied embedding (Adam)            |
| `tied_embed_init_std`        | `0.005`        | Std-dev for tied embedding init         |
| `matrix_lr`                  | `0.04`         | LR for 2D weight matrices (Muon)        |
| `scalar_lr`                  | `0.04`         | LR for 1D/scalar control params (Adam)  |
| `muon_momentum`              | `0.95`         | Target Muon momentum                    |
| `muon_backend_steps`         | `5`            | Newton-Schulz iteration count           |
| `muon_momentum_warmup_start` | `0.85`         | Muon momentum at step 0                 |
| `muon_momentum_warmup_steps` | `500`          | Steps to ramp momentum from 0.85 → 0.95 |
| `beta1` / `beta2`            | `0.9` / `0.95` | Adam betas                              |
| `adam_eps`                   | `1e-8`         | Adam epsilon                            |
| `grad_clip_norm`             | `0.0`          | Global gradient clipping (0 = disabled) |

---

## 4. Muon Optimizer

**Muon** (Momentum + Orthogonalization) is a specialized optimizer for matrix-shaped parameters borrowed from [modded-nanogpt](https://kellerjordan.github.io/posts/muon/). Instead of Adam's adaptive per-element scaling, Muon uses the **Newton-Schulz iteration** to orthogonalize the gradient matrix, then applies it as a steepest-descent step on the Stiefel manifold.

### 4.1 `zeropower_via_newtonschulz5`

This function computes an approximate **matrix polar decomposition** — given a matrix $G$, it finds the nearest orthogonal matrix $U$ such that $G = US$ where $U^TU = I$.

**Algorithm** (5th-order Newton-Schulz):

$$X_0 = \frac{G}{\|G\|_F}$$

$$A_k = X_k X_k^T$$

$$B_k = bA_k + cA_k^2$$

$$X_{k+1} = aX_k + B_k X_k$$

where $a = 3.4445$, $b = -4.7750$, $c = 2.0315$ are pre-tuned coefficients for fast convergence.

Key implementation details:

- **bf16 arithmetic** — the iteration runs entirely in bfloat16 for speed
- **Transpose trick** — if `rows > cols`, transposes before iterating and transposes back, keeping the inner matrix square-ish
- **5 iterations** (default `backend_steps=5`) suffice for adequate convergence

### 4.2 `Muon` Optimizer Class

```mermaid
flowchart LR
    G["Gradient g"] --> MOM["Momentum Buffer\nbuf = μ·buf + g"]
    MOM --> NEST{"Nesterov?"}
    NEST -- Yes --> NEST_G["g = g + μ·buf"]
    NEST -- No --> NS
    NEST_G --> NS["Newton-Schulz\nOrthogonalize g"]
    NS --> SCALE["Scale Correction\ng *= √(max(1, rows/cols))"]
    SCALE --> FLAT["Pack into flat buffer"]
    FLAT --> AR{"Distributed?"}
    AR -- Yes --> ALL["all_reduce(SUM)"]
    AR -- No --> UPD
    ALL --> UPD["θ = θ - lr · g"]

    style NS fill:#FF9800,color:white
    style ALL fill:#2196F3,color:white
```

**Distributed work splitting**: In multi-GPU training, each rank only processes parameters where `param_index % world_size == rank`. After the Newton-Schulz step, all ranks contribute their updates into a single flat buffer and call `all_reduce(SUM)` to synchronize.

**Scale correction**: After orthogonalization, the update is scaled by $\sqrt{\max(1, \text{rows}/\text{cols})}$. This compensates for the aspect ratio of non-square weight matrices, ensuring the effective step size is comparable across layers with different shapes.

---

## 5. Tokenizer-Agnostic Evaluation (BPB)

The challenge allows participants to bring their own tokenizer. To compare fairly, the evaluation metric is **bits per byte (BPB)** rather than per-token loss. This is tokenizer-agnostic because it measures compression efficiency on raw UTF-8 bytes.

### 5.1 BPB Formula

$$\text{BPB} = \frac{\text{bits per token} \times \text{tokens}}{\text{bytes}} = \frac{H / \ln 2 \times N_{\text{tokens}}}{N_{\text{bytes}}}$$

where $H$ is the mean cross-entropy loss (in nats).

### 5.2 `build_sentencepiece_luts`

Builds three lookup tables (LUTs) indexed by token ID that let us count bytes efficiently on GPU:

| LUT                     | Shape                 | Description                                                                                   |
| ----------------------- | --------------------- | --------------------------------------------------------------------------------------------- |
| `base_bytes_lut`        | `(vocab_size,)` int16 | UTF-8 byte length of each token's text piece                                                  |
| `has_leading_space_lut` | `(vocab_size,)` bool  | Whether the token starts with the SentencePiece `▁` (space) marker                            |
| `is_boundary_token_lut` | `(vocab_size,)` bool  | True for control/unknown/unused tokens (boundaries that don't contribute leading-space bytes) |

**The leading-space correction**: SentencePiece's `▁` marker represents a space that was in the original text, but the `▁` character itself is not one UTF-8 byte — the original space was. The LUT adds 1 byte for `▁` tokens only when the previous token is _not_ a boundary token, correctly counting the space byte.

### 5.3 `eval_val`

```mermaid
flowchart TD
    VT["Validation Tokens"] --> SPLIT["Split by rank\n(disjoint ranges)"]
    SPLIT --> BATCH["Iterate in local batches"]
    BATCH --> FWD["Forward pass\n(autocast bf16)"]
    FWD --> LOSS["Accumulate\nweighted loss sum"]
    FWD --> BYTES["Count bytes via LUTs\n(base_bytes + leading_space correction)"]
    LOSS --> AR["all_reduce SUM\n(loss_sum, token_count, byte_count)"]
    BYTES --> AR
    AR --> CALC["val_loss = loss_sum / token_count\nbpb = (loss / ln2) × (tokens / bytes)"]
    CALC --> OUT["Return (val_loss, val_bpb)"]

    style FWD fill:#2196F3,color:white
    style AR fill:#FF9800,color:white
```

Key details:

- Uses `torch.inference_mode()` for no-grad evaluation
- Validation data is split across ranks for parallel evaluation, then `all_reduce`d
- Both `val_loss` (cross-entropy in nats) and `val_bpb` (bits per byte) are returned
- All accumulation uses `float64` to avoid precision issues over large validation sets

---

## 6. Post-Training Quantization

The model trains in bf16/fp32 but is serialized as **int8 + zlib** to fit under the 16 MB size cap. This is a _post-training_ quantization — no quantization-aware training is involved.

### 6.1 Quantization Strategy

```mermaid
flowchart TD
    SD["state_dict"] --> CHECK{"For each tensor"}
    CHECK --> NF{"Non-float?"}
    NF -- Yes --> PASS1["Passthrough\n(exact copy)"]
    CHECK --> SMALL{"numel ≤ 65,536?"}
    SMALL -- Yes --> CTRL{"Control tensor?\n(attn_scale, mlp_scale,\nresid_mix, q_gain, skip_weights)"}
    CTRL -- Yes --> FP32["Keep as fp32"]
    CTRL -- No --> FP16["Downcast to fp16"]
    CHECK --> LARGE{"Large float tensor"}
    LARGE --> DIM{"ndim == 2?"}
    DIM -- Yes --> ROW["Per-row int8\n(one scale per output channel)"]
    DIM -- No --> TENSOR["Per-tensor int8\n(single global scale)"]

    ROW --> PACK["Pack into dict:\nquantized + scales + dtypes + passthrough"]
    TENSOR --> PACK
    FP32 --> PACK
    FP16 --> PACK
    PASS1 --> PACK
    PACK --> SAVE["torch.save → zlib.compress(level=9)\n→ final_model.int8.ptz"]

    style ROW fill:#FF9800,color:white
    style SAVE fill:#4CAF50,color:white
```

### 6.2 Percentile Clipping

Before quantizing, tensor values are clipped at the **99.99984th percentile** of absolute values. This removes extreme outliers that would otherwise waste quantization range:

$$\text{clip\_abs} = \text{quantile}(|W|, 0.9999984)$$

$$\text{scale} = \max\left(\frac{\text{clip\_abs}}{127}, \frac{1}{127}\right)$$

$$W_q = \text{clamp}\left(\text{round}\left(\frac{\text{clamp}(W, -\text{clip\_abs}, \text{clip\_abs})}{\text{scale}}\right), -127, 127\right)$$

### 6.3 Per-Row vs Per-Tensor Scales

| Tensor Type                  | Quantization       | Scale Shape    | Rationale                                              |
| ---------------------------- | ------------------ | -------------- | ------------------------------------------------------ |
| 2D matrices (weights)        | Per-row int8       | `(rows,)` fp16 | Output channels often have very different value ranges |
| 1D vectors / scalars         | Per-tensor int8    | scalar fp32    | Single scale suffices for small tensors                |
| Small tensors (≤ 65K params) | Passthrough (fp16) | —              | Quantization overhead would exceed savings             |
| Control tensors              | Passthrough (fp32) | —              | Precision-sensitive (scales, gains, mix weights)       |

### 6.4 Dequantization (`dequantize_state_dict_int8`)

Round-trip dequantization reverses the process:

- **Per-row**: $W = W_q \cdot \text{scale}[\text{row}]$, broadcast across columns
- **Per-tensor**: $W = W_q \cdot \text{scale}$
- **Passthrough**: restore original dtype from saved metadata

### 6.5 Serialization Format

The final artifact is `final_model.int8.ptz`:

1. Build a Python dict with `quantized`, `scales`, `dtypes`, `passthrough`, `qmeta`, `passthrough_orig_dtypes`
2. `torch.save()` to an in-memory `BytesIO` buffer
3. `zlib.compress(level=9)` the buffer
4. Write the compressed blob to disk

The format identifier is `int8_clean_per_row_v1`.

---

## 7. Data Loading

### 7.1 Shard Format (`load_data_shard`)

Each binary shard file has:

| Offset       | Content                  | Type                 |
| ------------ | ------------------------ | -------------------- |
| 0–1023 bytes | 256-integer header       | little-endian int32  |
| `header[0]`  | Magic number: `20240520` | int32                |
| `header[1]`  | Version: `1`             | int32                |
| `header[2]`  | Number of tokens         | int32                |
| 1024+ bytes  | Token data               | little-endian uint16 |

### 7.2 `TokenStream`

```mermaid
flowchart LR
    subgraph TokenStream
        S0["Shard 0"] --> S1["Shard 1"] --> S2["Shard 2"] --> DOTS["..."] --> SN["Shard N"]
        SN -. "wrap around" .-> S0
    end

    TS["take(n)"] --> POS["Read from current\nposition in current shard"]
    POS --> ENOUGH{"Got n tokens?"}
    ENOUGH -- No --> ADV["Advance to next shard\n(wrap at end)"]
    ADV --> POS
    ENOUGH -- Yes --> RET["Return concatenated tokens"]
```

- Reads shards **sequentially** (no shuffling, no random access)
- Wraps around to the first shard when all shards are consumed → infinite stream
- Deterministic behavior: same tokens in same order every epoch

### 7.3 `DistributedTokenLoader`

```mermaid
flowchart TD
    CALL["next_batch(global_tokens=524288,\nseq_len=1024, grad_accum_steps)"] --> CALC["local_tokens = global / (world_size × accum)\nper_rank_span = local + 1"]
    CALC --> TAKE["stream.take(\nper_rank_span × world_size)"]
    TAKE --> SLICE["Slice rank's disjoint span:\nchunk[rank × span : (rank+1) × span]"]
    SLICE --> SHIFT["x = local[:-1]  (input)\ny = local[1:]   (target)"]
    SHIFT --> RESHAPE["Reshape to\n(batch, seq_len)"]
    RESHAPE --> GPU["Move to GPU\n(non_blocking)"]
```

The `+1` token trick: by taking one extra token per rank, the script creates input/target pairs by shifting: `x = tokens[:-1]`, `y = tokens[1:]`. This avoids wasting a full sequence boundary between batches.

---

## 8. Transformer Modules

### 8.1 RMSNorm

**Root Mean Square Layer Normalization** — a simpler alternative to LayerNorm that skips the mean-subtraction step:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}$$

Unlike standard LayerNorm, this implementation has **no learnable affine parameters** (no gain/bias). It uses PyTorch's built-in `F.rms_norm` for efficiency.

**Used in 5 locations**:

1. Post-embedding normalization (in `GPT.forward`)
2. QK normalization (in `CausalSelfAttention.forward`, applied to Q and K separately)
3. Pre-attention norm (in `Block`, via `attn_norm`)
4. Pre-MLP norm (in `Block`, via `mlp_norm`)
5. Final norm before output projection (in `GPT`, via `final_norm`)

### 8.2 CastedLinear

A subclass of `nn.Linear` that **stores weights in fp32** but **casts them to the input dtype** (typically bf16) at forward time:

```
weight (stored fp32) ──cast to bf16──→ F.linear(x_bf16, weight_bf16)
```

This gives the optimizer fp32-quality weight updates while keeping the forward/backward pass in bf16 for speed. The `_zero_init` attribute flags certain layers (output projections) for zero-initialization.

### 8.3 Rotary Position Embeddings (RoPE)

RoPE encodes positional information by **rotating** query and key vectors in 2D subspaces, indexed by position.

#### Frequency Computation

$$\theta_i = \frac{1}{\text{base}^{2i/d}}, \quad i = 0, 1, \ldots, d/2 - 1$$

where `base = 10,000` and `d = head_dim = 64`.

#### Rotation Application

For a vector $x = [x_1, x_2]$ split into two halves at position $t$:

$$\text{RoPE}(x, t) = \begin{bmatrix} x_1 \cos(t\theta) + x_2 \sin(t\theta) \\ -x_1 \sin(t\theta) + x_2 \cos(t\theta) \end{bmatrix}$$

```mermaid
flowchart LR
    subgraph "Rotary Module"
        INV["inv_freq = 1 / (base^(2i/d))"] --> FREQ["freqs = outer(positions, inv_freq)"]
        FREQ --> COS["cos_cache = cos(freqs)"]
        FREQ --> SIN["sin_cache = sin(freqs)"]
    end

    subgraph "apply_rotary_emb(x, cos, sin)"
        X["x"] --> SPLIT["x₁ = x[..., :half]\nx₂ = x[..., half:]"]
        SPLIT --> ROT["out = concat(\n  x₁·cos + x₂·sin,\n  -x₁·sin + x₂·cos\n)"]
    end

    COS --> ROT
    SIN --> ROT
```

The `Rotary` module **caches** cos/sin tables keyed by sequence length and device to avoid recomputation. Shape after expanding: `(1, 1, seq_len, head_dim/2)` — broadcasts over batch and head dimensions.

### 8.4 CausalSelfAttention

The attention module implements **Grouped Query Attention (GQA)** with QK normalization and learnable query scaling.

```mermaid
flowchart TD
    X["Input x\n(batch, seq, dim=512)"] --> Q_PROJ["c_q: Linear(512→512)\n→ reshape to (B, 8 heads, S, 64)"]
    X --> K_PROJ["c_k: Linear(512→256)\n→ reshape to (B, 4 kv_heads, S, 64)"]
    X --> V_PROJ["c_v: Linear(512→256)\n→ reshape to (B, 4 kv_heads, S, 64)"]

    Q_PROJ --> Q_NORM["RMSNorm(Q)\nper-head normalization"]
    K_PROJ --> K_NORM["RMSNorm(K)\nper-head normalization"]

    Q_NORM --> Q_ROPE["apply_rotary_emb(Q)"]
    K_NORM --> K_ROPE["apply_rotary_emb(K)"]

    Q_ROPE --> Q_GAIN["Q = Q × q_gain\n(learnable, per-head,\ninit=1.5)"]

    Q_GAIN --> SDPA["F.scaled_dot_product_attention\n(causal=True, enable_gqa=True)"]
    K_ROPE --> SDPA
    V_PROJ --> SDPA

    SDPA --> PROJ["proj: Linear(512→512)\n(zero-initialized)"]
    PROJ --> OUT["Output\n(batch, seq, 512)"]

    style Q_GAIN fill:#FF9800,color:white
    style SDPA fill:#2196F3,color:white
```

#### Grouped Query Attention (GQA)

With `num_heads=8` and `num_kv_heads=4`, each KV head is shared by 2 Q heads. This reduces KV memory and compute by 2× while retaining most of the attention capacity.

| Component | Heads | Dim per head | Total dim |
| --------- | ----- | ------------ | --------- |
| Q         | 8     | 64           | 512       |
| K         | 4     | 64           | 256       |
| V         | 4     | 64           | 256       |

PyTorch's `F.scaled_dot_product_attention` handles the GQA broadcasting automatically when `enable_gqa=True`.

#### QK Normalization + Q Gain

Before RoPE, both Q and K are **RMS-normalized per head**. This stabilizes attention logits regardless of activation magnitudes. After normalization and RoPE:

$$\text{attn}(Q, K) = \frac{(\gamma_Q \cdot \hat{Q}_{\text{rope}}) \cdot \hat{K}_{\text{rope}}^T}{\sqrt{d_k}}$$

The `q_gain` ($\gamma_Q$) is a **learnable, per-head scalar** initialized to 1.5. Scaling only Q (not K) is sufficient because:

$$\gamma_Q \hat{Q} \cdot \hat{K}^T = \hat{Q} \cdot (\gamma_Q \hat{K})^T$$

Mathematically, scaling Q by $\gamma$ is equivalent to scaling K by $\gamma$ — they both multiply the dot product by $\gamma$. Applying it to one side avoids redundancy.

#### Zero-Init Output Projection

The `proj` layer is zero-initialized (`_zero_init = True`). At the start of training, the attention block contributes nothing to the residual stream, making the initial model effectively a simple embedding → output projection.

### 8.5 MLP

A two-layer feedforward network with **squared ReLU** activation (from the modded-nanogpt):

$$\text{MLP}(x) = W_{\text{proj}} \cdot \text{ReLU}(W_{\text{fc}} \cdot x)^2$$

```mermaid
flowchart LR
    X["x (dim=512)"] --> FC["fc: Linear(512→1024)\n(bias=False)"]
    FC --> RELU["ReLU"]
    RELU --> SQ["Square (x²)"]
    SQ --> PROJ["proj: Linear(1024→512)\n(zero-init, bias=False)"]
    PROJ --> OUT["output (dim=512)"]

    style SQ fill:#FF9800,color:white
```

The squared ReLU ($\text{ReLU}(x)^2$) provides:

- **Sparsity**: Like ReLU, negative values are zeroed
- **Smoothness**: The squaring makes the activation differentiable at 0
- **Sharpness**: Larger activations are amplified quadratically, creating sparser effective representations

Hidden dimension = `mlp_mult × dim = 2 × 512 = 1024`.

### 8.6 Block

A single transformer layer combining attention and MLP with several modern techniques.

```mermaid
flowchart TD
    X_IN["x (from previous block)"] --> MIX["Residual Mix with x0:\nx = mix[0]·x + mix[1]·x0"]
    X0["x0 (original embedding)"] --> MIX

    MIX --> AN["attn_norm (RMSNorm)"]
    AN --> ATTN["CausalSelfAttention"]
    ATTN --> AS["× attn_scale\n(learnable, per-dim)"]
    AS --> ADD1["x = x + scaled_attn"]
    MIX --> ADD1

    ADD1 --> MN["mlp_norm (RMSNorm)"]
    MN --> MLP2["MLP (relu²)"]
    MLP2 --> MS["× mlp_scale\n(learnable, per-dim)"]
    MS --> ADD2["x = x + scaled_mlp"]
    ADD1 --> ADD2

    ADD2 --> X_OUT["Output x"]

    style MIX fill:#9C27B0,color:white
    style AS fill:#FF9800,color:white
    style MS fill:#FF9800,color:white
```

#### Residual Mix (`resid_mix`)

Each block blends the current hidden state `x` with the **original post-embedding representation** `x0`:

$$x' = \alpha \odot x + \beta \odot x_0$$

where $\alpha$ = `resid_mix[0]` (init: all ones) and $\beta$ = `resid_mix[1]` (init: all zeros). Per-dimension learnable parameters (shape `(2, dim)`).

This provides a **direct gradient highway** from any block back to the input embedding, combating vanishing gradients in deep networks. At initialization, $\alpha=1, \beta=0$, so the block starts as a standard residual.

#### Learnable Residual Scales

Both `attn_scale` and `mlp_scale` are per-dimension vectors (init: all ones) that scale the sub-layer outputs before adding to the residual:

$$x = x + \text{attn\_scale} \odot \text{Attn}(\text{Norm}(x))$$
$$x = x + \text{mlp\_scale} \odot \text{MLP}(\text{Norm}(x))$$

These allow the model to learn per-dimension importance for each sub-layer's contribution.

### 8.7 GPT (Full Model)

The top-level model class implements the full forward pass with **U-Net skip connections**.

#### U-Net Architecture

With `num_layers=9`:

- **Encoder**: first `9 // 2 = 4` blocks (indices 0–3)
- **Decoder**: remaining `9 - 4 = 5` blocks (indices 4–8)
- **Skip connections**: 4 (one per encoder block)

```mermaid
flowchart TD
    TOK["Token Embedding\n+ RMSNorm → x0"] --> E0

    subgraph "Encoder (blocks 0-3)"
        E0["Block 0"] --> E1["Block 1"]
        E1 --> E2["Block 2"]
        E2 --> E3["Block 3"]
    end

    E0 -. "skip[0]" .-> D3
    E1 -. "skip[1]" .-> D2
    E2 -. "skip[2]" .-> D1
    E3 -. "skip[3]" .-> D0

    E3 --> D0

    subgraph "Decoder (blocks 4-8)"
        D0["+ skip[3] × w₃\n→ Block 4"] --> D1["+ skip[2] × w₂\n→ Block 5"]
        D1 --> D2["+ skip[1] × w₁\n→ Block 6"]
        D2 --> D3["+ skip[0] × w₀\n→ Block 7"]
        D3 --> D4["Block 8\n(no skip — ran out)"]
    end

    D4 --> NORM["final_norm (RMSNorm)"]
    NORM --> LM{"tie_embeddings?"}
    LM -- Yes --> TIED["F.linear(x, tok_emb.weight)"]
    LM -- No --> HEAD["lm_head(x)"]
    TIED --> CAP["Logit Softcap:\n30 × tanh(logits / 30)"]
    HEAD --> CAP
    CAP --> CE["cross_entropy(logits, targets)"]

    style TOK fill:#4CAF50,color:white
    style CAP fill:#FF9800,color:white
```

Encoder blocks store their outputs onto a stack; decoder blocks pop from the stack (LIFO = reverse order). Each skip connection has a **learnable per-dimension weight** `skip_weights[i]`:

$$x = x + \text{skip\_weight}_i \odot \text{skip}_i$$

Since encoder has 4 blocks and decoder has 5, the last decoder block (Block 8) runs without a skip connection.

#### Logit Softcapping

Output logits are clamped into $[-30, +30]$ via:

$$\text{logits} = 30 \cdot \tanh\left(\frac{\text{raw\_logits}}{30}\right)$$

This prevents extreme logit values from destabilizing training and is borrowed from Gemma-2/PaLM architectures.

#### Weight Initialization

- **Tied embeddings**: `Normal(0, 0.005)` — small std because tied weights serve double duty
- **Zero-init projections**: All layers with `_zero_init = True` (`attn.proj`, `mlp.proj`, optional `lm_head`) are zero-initialized, making the initial model a near-identity function through the residual path

---

## 9. Training (`main()`)

### 9.1 Distributed + CUDA Setup

```mermaid
flowchart TD
    START["main()"] --> CHECK{"RANK & WORLD_SIZE\nin env?"}
    CHECK -- Yes --> DDP_INIT["dist.init_process_group(nccl)"]
    CHECK -- No --> SINGLE["Single-GPU mode"]
    DDP_INIT --> DEV["device = cuda:LOCAL_RANK"]
    SINGLE --> DEV
    DEV --> ACCUM["grad_accum_steps = 8 / world_size"]
    ACCUM --> TF32["Enable TF32 matmul & cuDNN"]
    TF32 --> SDP["Flash Attention ONLY\n(disable cuDNN, math, mem_efficient)"]
```

Key points:

- **`grad_accum_steps = 8 // world_size`**: With 8 GPUs, no accumulation; with 1 GPU, 8 micro-steps per optimization step. The "8" is hardcoded so `world_size` must divide 8.
- **TF32**: Enabled for both matmul and cuDNN — provides ~2× speedup on Ampere+ GPUs with negligible accuracy loss.
- **SDP backend**: Only **Flash Attention** is enabled; cuDNN, math, and memory-efficient backends are disabled. Flash Attention is fastest for causal masking.

### 9.2 Tokenizer + Validation Setup

1. Seeds all RNGs (Python `random`, NumPy, PyTorch CPU + CUDA) with `seed=1337`
2. Loads the SentencePiece tokenizer and validates vocab size matches
3. Loads the full validation token set (pre-tokenized binary shards)
4. Builds the 3 BPB lookup tables on GPU

### 9.3 Model + Optimizer Setup

```mermaid
flowchart TD
    subgraph "Model Precision Setup"
        CREATE["GPT(...).to(device).bfloat16()"]
        CREATE --> CAST_LIN["CastedLinear modules → .float()\n(weights stored fp32)"]
        CAST_LIN --> RESTORE["restore_low_dim_params_to_fp32()\n(1D + control params → fp32)"]
        RESTORE --> COMPILE["torch.compile(fullgraph=True)"]
        COMPILE --> DDP_WRAP{"Distributed?"}
        DDP_WRAP -- Yes --> DDP["DistributedDataParallel"]
        DDP_WRAP -- No --> MODEL["compiled_model"]
    end

    subgraph "Optimizer Split"
        direction LR
        EMB["tok_emb.weight\n→ Adam (lr=0.05*)"]
        MAT["2D block weights\n(not control)\n→ Muon (lr=0.04)"]
        SCA["1D params + control +\nskip_weights\n→ Adam (lr=0.04)"]
        LMH["lm_head.weight**\n→ Adam (lr=0.008)"]
    end

    MODEL --> EMB
    MODEL --> MAT
    MODEL --> SCA
    MODEL --> LMH

    style CREATE fill:#4CAF50,color:white
    style MAT fill:#FF9800,color:white
```

_\* `tied_embed_lr=0.05` when tied, `embed_lr=0.6` when untied_
_\*\* Only exists when `tie_embeddings=False`_

**Parameter routing rules**:

| Parameter Type        | Criterion                          | Optimizer    | LR                         |
| --------------------- | ---------------------------------- | ------------ | -------------------------- |
| Token embedding       | `tok_emb.weight`                   | Adam (fused) | 0.05 (tied) / 0.6 (untied) |
| LM head               | `lm_head.weight` (if exists)       | Adam (fused) | 0.008                      |
| Block matrices        | `ndim == 2` and not control tensor | Muon         | 0.04                       |
| Block scalars/vectors | `ndim < 2` or control tensor name  | Adam (fused) | 0.04                       |
| Skip weights          | `skip_weights`                     | Adam (fused) | 0.04                       |

Control tensor names matching: `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weight[s]`.

### 9.4 Compilation Warmup

`torch.compile` has non-trivial first-run overhead (tracing, code generation). The warmup phase:

1. **Save** initial model state + optimizer states to CPU
2. **Run** `warmup_steps=20` full training steps (forward + backward + optimizer)
3. **Restore** the saved initial state — weights and optimizer states reset to their original values
4. **Recreate** the data loader so token ordering is deterministic from the true start

This ensures the measured training time doesn't include compilation overhead, which can be 30+ seconds.

### 9.5 Learning Rate Schedule

The LR schedule has **no warmup ramp** — it starts at full LR and applies a **linear warmdown** at the end of training:

```mermaid
flowchart LR
    subgraph "lr_mul(step, elapsed_ms)"
        CHECK{"warmdown_iters > 0?"}
        CHECK -- No --> FLAT["return 1.0\n(constant LR)"]
        CHECK -- Yes --> WC{"wallclock cap set?"}
        WC -- No --> ITER["Iteration-based warmdown:\nlinear decay over last\n1200 steps"]
        WC -- Yes --> TIME["Time-based warmdown:\nestimate remaining time,\nlinear decay when\nremaining ≤ warmdown_iters × step_time"]
    end
```

**Time-based warmdown** (the default path since `max_wallclock_seconds=600`):

$$\text{step\_time} = \frac{\text{elapsed\_ms}}{\text{step}}$$
$$\text{warmdown\_ms} = \text{warmdown\_iters} \times \text{step\_time}$$
$$\text{remaining\_ms} = \text{max\_wallclock\_ms} - \text{elapsed\_ms}$$

$$\text{lr\_mul} = \begin{cases} 1.0 & \text{if remaining > warmdown\_ms} \\ \frac{\text{remaining\_ms}}{\text{warmdown\_ms}} & \text{otherwise (linear decay to 0)} \end{cases}$$

This is adaptive: the warmdown region expands or contracts based on the actual step time, ensuring the LR reaches ~0 right as the wall-clock cap is hit.

### 9.6 Main Training Loop

```mermaid
flowchart TD
    INIT["step = 0\ntraining_time_ms = 0"] --> LOOP["while True"]
    LOOP --> LAST{"last step?\n(step == iterations OR\nstop_after_step reached)"}

    LAST --> VAL{"Should validate?\n(last_step OR\nstep % 1000 == 0)"}
    VAL -- Yes --> EVAL["eval_val()\nLog val_loss, val_bpb"]
    VAL -- No --> CONT

    EVAL --> DONE{"last_step?"}
    DONE -- Yes --> BREAK["break → Serialization"]
    DONE -- No --> CONT

    CONT --> LR["Compute LR scale\n(wallclock-aware warmdown)"]
    LR --> ZERO["zero_grad_all()"]
    ZERO --> MICRO["For micro_step in\nrange(grad_accum_steps)"]
    MICRO --> SYNC{"DDP: only sync\ngrads on last\nmicro-step"}
    SYNC --> BATCH["Load batch\n(x, y)"]
    BATCH --> FWD["Forward (autocast bf16)\nloss = model(x, y)"]
    FWD --> BWD["(loss × grad_scale).backward()"]
    BWD --> ACC["train_loss += loss.detach()"]
    ACC --> MICRO

    MICRO -- done --> MUON_MOM["Update Muon momentum\n(linear warmup 0.85 → 0.95\nover 500 steps)"]
    MUON_MOM --> LR_SET["Set lr = base_lr × scale\nfor all optimizer groups"]
    LR_SET --> CLIP{"grad_clip_norm > 0?"}
    CLIP -- Yes --> CLIPG["clip_grad_norm_()"]
    CLIP -- No --> OPT
    CLIPG --> OPT["All optimizers .step()"]
    OPT --> ZEROF["zero_grad_all()"]
    ZEROF --> INC["step += 1"]
    INC --> LOG{"Log train loss?"}
    LOG --> CAP{"Wallclock cap\nreached?"}
    CAP -- Yes --> STOP["stop_after_step = step"]
    CAP -- No --> LOOP
    STOP --> LOOP

    style FWD fill:#2196F3,color:white
    style OPT fill:#FF9800,color:white
    style BREAK fill:#4CAF50,color:white
```

#### Gradient Accumulation

With `train_batch_tokens=524,288`, `seq_len=1024`, and `grad_accum_steps=8` (1 GPU):

$$\text{micro-batch} = \frac{524{,}288}{1 \times 8} = 65{,}536 \text{ tokens} = 64 \text{ sequences}$$

Gradients sync only on the **last micro-step** (`require_backward_grad_sync = micro_step == grad_accum_steps - 1`), reducing DDP communication by 8×.

#### Muon Momentum Warmup

Muon's momentum ramps linearly from 0.85 to 0.95 over the first 500 steps:

$$\mu_t = (1 - f) \times 0.85 + f \times 0.95, \quad f = \min\left(\frac{t}{500}, 1\right)$$

#### Distributed Wallclock Sync

When any rank hits the 10-minute cap, all ranks must agree to stop at the _same step_. This is achieved via `all_reduce(MAX)` on a flag tensor — if any rank sets it to 1, all ranks see 1 and set `stop_after_step`, which takes effect after the next validation.

### 9.7 Serialization + Round-Trip Validation

After training ends:

1. **Raw save**: `final_model.pt` — the standard PyTorch state dict (useful for debugging)
2. **Quantized save**: `final_model.int8.ptz` — int8 quantized + zlib level-9 compressed
3. **Round-trip validation**: Load `.int8.ptz` from disk → decompress → dequantize → load into model → run full validation → report `val_loss` and `val_bpb`

The round-trip step verifies the quantized model actually works and reports the final challenge score.

```mermaid
flowchart LR
    SD["state_dict (bf16/fp32)"] --> RAW["torch.save → final_model.pt"]
    SD --> Q8["quantize_state_dict_int8()"]
    Q8 --> SAVE["torch.save → BytesIO"]
    SAVE --> ZLIB["zlib.compress(level=9)"]
    ZLIB --> DISK["Write final_model.int8.ptz"]
    DISK --> READ["Read from disk"]
    READ --> DECOMP["zlib.decompress"]
    DECOMP --> LOAD["torch.load"]
    LOAD --> DEQ["dequantize_state_dict_int8()"]
    DEQ --> LOADSD["model.load_state_dict()"]
    LOADSD --> EVAL2["eval_val() → final BPB"]

    style EVAL2 fill:#4CAF50,color:white
    style ZLIB fill:#FF9800,color:white
```

---

## 10. Appendix: Hyperparameter Reference Table

Complete table of all hyperparameters with their environment variable names and defaults:

| Env Variable                 | Field                        | Default                                    | Type  | Category   |
| ---------------------------- | ---------------------------- | ------------------------------------------ | ----- | ---------- |
| `DATA_PATH`                  | `data_path`                  | `./data/datasets/fineweb10B_sp1024`        | str   | Data       |
| `TOKENIZER_PATH`             | `tokenizer_path`             | `./data/tokenizers/fineweb_1024_bpe.model` | str   | Data       |
| `RUN_ID`                     | `run_id`                     | random UUID                                | str   | I/O        |
| `SEED`                       | `seed`                       | 1337                                       | int   | I/O        |
| `VAL_BATCH_SIZE`             | `val_batch_size`             | 524,288                                    | int   | Validation |
| `VAL_LOSS_EVERY`             | `val_loss_every`             | 1,000                                      | int   | Validation |
| `TRAIN_LOG_EVERY`            | `train_log_every`            | 200                                        | int   | Logging    |
| `ITERATIONS`                 | `iterations`                 | 20,000                                     | int   | Training   |
| `WARMDOWN_ITERS`             | `warmdown_iters`             | 1,200                                      | int   | Training   |
| `WARMUP_STEPS`               | `warmup_steps`               | 20                                         | int   | Training   |
| `TRAIN_BATCH_TOKENS`         | `train_batch_tokens`         | 524,288                                    | int   | Training   |
| `TRAIN_SEQ_LEN`              | `train_seq_len`              | 1,024                                      | int   | Training   |
| `MAX_WALLCLOCK_SECONDS`      | `max_wallclock_seconds`      | 600.0                                      | float | Training   |
| `QK_GAIN_INIT`               | `qk_gain_init`               | 1.5                                        | float | Model      |
| `VOCAB_SIZE`                 | `vocab_size`                 | 1,024                                      | int   | Model      |
| `NUM_LAYERS`                 | `num_layers`                 | 9                                          | int   | Model      |
| `NUM_KV_HEADS`               | `num_kv_heads`               | 4                                          | int   | Model      |
| `MODEL_DIM`                  | `model_dim`                  | 512                                        | int   | Model      |
| `NUM_HEADS`                  | `num_heads`                  | 8                                          | int   | Model      |
| `MLP_MULT`                   | `mlp_mult`                   | 2                                          | int   | Model      |
| `TIE_EMBEDDINGS`             | `tie_embeddings`             | 1 (True)                                   | bool  | Model      |
| `ROPE_BASE`                  | `rope_base`                  | 10,000.0                                   | float | Model      |
| `LOGIT_SOFTCAP`              | `logit_softcap`              | 30.0                                       | float | Model      |
| `EMBED_LR`                   | `embed_lr`                   | 0.6                                        | float | Optimizer  |
| `HEAD_LR`                    | `head_lr`                    | 0.008                                      | float | Optimizer  |
| `TIED_EMBED_LR`              | `tied_embed_lr`              | 0.05                                       | float | Optimizer  |
| `TIED_EMBED_INIT_STD`        | `tied_embed_init_std`        | 0.005                                      | float | Optimizer  |
| `MATRIX_LR`                  | `matrix_lr`                  | 0.04                                       | float | Optimizer  |
| `SCALAR_LR`                  | `scalar_lr`                  | 0.04                                       | float | Optimizer  |
| `MUON_MOMENTUM`              | `muon_momentum`              | 0.95                                       | float | Optimizer  |
| `MUON_BACKEND_STEPS`         | `muon_backend_steps`         | 5                                          | int   | Optimizer  |
| `MUON_MOMENTUM_WARMUP_START` | `muon_momentum_warmup_start` | 0.85                                       | float | Optimizer  |
| `MUON_MOMENTUM_WARMUP_STEPS` | `muon_momentum_warmup_steps` | 500                                        | int   | Optimizer  |
| `BETA1`                      | `beta1`                      | 0.9                                        | float | Optimizer  |
| `BETA2`                      | `beta2`                      | 0.95                                       | float | Optimizer  |
| `ADAM_EPS`                   | `adam_eps`                   | 1e-8                                       | float | Optimizer  |
| `GRAD_CLIP_NORM`             | `grad_clip_norm`             | 0.0                                        | float | Optimizer  |

---

_Generated from `train_gpt.py` — parameter-golf project._
