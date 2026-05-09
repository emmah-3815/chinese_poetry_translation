# Methods (for Midterm Report)

## Section Outline

```
3. Methods
   3.1 Problem Formulation
   3.2 Parameter-Efficient Fine-Tuning via LoRA
   3.3 Path 1: Qwen2.5-1.5B + QLoRA
       3.3.1 Model Architecture
       3.3.2 QLoRA Quantization
   3.4 Path 2: mT5-base + LoRA
       3.4.1 Model Architecture
       3.4.2 LoRA on Encoder-Decoder Attention
   3.5 Training Procedure
   3.6 Evaluation Metrics (BLEU, ROUGE, BERTScore)
```

## 3.1 Problem Formulation

Let a classical Chinese poem be a sequence of tokens:

$$x = (x_1, x_2, \dots, x_m)$$

and its English translation target:

$$y = (y_1, y_2, \dots, y_n)$$

The goal is to learn a conditional distribution $P_\theta(y \mid x)$ that maximizes translation quality across **adequacy, fluency, and poetic elegance** (criteria from Chen et al. 2025).

Training minimizes the standard **negative log-likelihood (NLL)** loss over a parallel corpus $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}$:

$$\mathcal{L}(\theta) = -\sum_{i=1}^{|\mathcal{D}|} \sum_{t=1}^{n} \log P_\theta(y_t^{(i)} \mid y_{<t}^{(i)}, x^{(i)})$$

## 3.2 Parameter-Efficient Fine-Tuning via LoRA

Both paths use **Low-Rank Adaptation (LoRA)** (Hu et al., 2021). The pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ is frozen, and a trainable low-rank decomposition is injected:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$.

The forward pass becomes:

$$h = W_0 x + \frac{\alpha}{r} BAx$$

where $\alpha$ is a scaling hyperparameter. Only $A$ and $B$ are updated, reducing trainable parameters from $d \times k$ to $r(d + k)$.

A full fine-tune of Qwen2.5-1.5B updates ~1.5B parameters. With LoRA at $r = 16$, only ~2–5M parameters are updated — a **>99% reduction**.

## 3.3 Path 1: Qwen2.5-1.5B + QLoRA

### Architecture
Qwen2.5-1.5B is a **decoder-only transformer** that models $P_\theta(y \mid x)$ autoregressively. The input is formatted as a prompt:

```
Translate the following classical Chinese poem into English:
[Chinese poem]
English:
```

The model generates tokens left-to-right conditioning on all previous context.

### QLoRA Extension
QLoRA (Dettmers et al., 2023) adds **4-bit NormalFloat (NF4) quantization** on top of LoRA. Base weights are quantized as:

$$W_0^{q} = \text{quantize}_{NF4}(W_0)$$

stored at 4-bit precision, while LoRA adapters $A, B$ remain in full BFloat16. The forward pass dequantizes on-the-fly:

$$h = \text{dequantize}(W_0^q) x + \frac{\alpha}{r} BAx$$

This combines two savings: 4-bit quantization reduces base model VRAM by ~75%, and LoRA restricts gradient updates to low-rank matrices. A 1.5B model becomes feasible on a single 16GB consumer GPU.

## 3.4 Path 2: mT5-base + LoRA

### Architecture
mT5-base is a **multilingual encoder-decoder transformer** pretrained on 101 languages with span-corruption.

- **Encoder:** $H = \text{Encoder}(x)$, where $H \in \mathbb{R}^{m \times d}$
- **Decoder:** autoregressively generates conditioning on both $H$ and previous tokens:

$$P_\theta(y_t \mid y_{<t}, x) = \text{softmax}(W_o \cdot \text{Decoder}(y_{<t}, H))$$

### LoRA on Encoder-Decoder Attention
Adapters are applied to:
- Self-attention $\{W_q, W_v\}$ in encoder and decoder
- **Cross-attention** $\{W_q, W_v\}$ in decoder

Cross-attention adaptation is critical for poetry: it governs how the model attends to specific Chinese source tokens when generating each English word — directly affecting poetic fidelity.

## 3.5 Training Procedure

```
Algorithm: Two-Path PEFT Training for Poetry Translation

Input:  Parallel corpus D = {(x_i, y_i)}, pretrained model M, rank r, epochs E
Output: Fine-tuned adapter weights {A, B}

1. Freeze all original weights W_0 of M
2. Initialize LoRA matrices: A ~ N(0, σ²), B = 0
   (B = 0 ensures ΔW = 0 at start, preserving pretrained behavior)
3. [Path 1 only] Quantize W_0 to NF4 4-bit precision
4. For epoch = 1 to E:
     For each batch (x, y) in D:
       a. Compute forward pass with W = W_0 + (α/r)·BA
       b. Compute NLL loss
       c. Compute gradients only w.r.t. {A, B}
       d. Update {A, B} with AdamW
5. Return trained {A, B}
```

The key theoretical property: **gradient flow never touches $W_0$**, preserving general language knowledge while specializing the adapter to poetry style.

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| LoRA rank $r$ | 16 (ablate: 8, 32) |
| LoRA $\alpha$ | 32 |
| LoRA target modules | `q_proj, v_proj` |
| Quantization (Path 1) | NF4 4-bit |
| Learning rate | 2e-4 |
| Optimizer | AdamW |
| LR schedule | Cosine |

## Theoretical Analysis

### Why Low-Rank Works
The hypothesis (Hu et al.) is that fine-tuning weight updates have a **low intrinsic rank** — task-specific adaptation lies in a low-dimensional subspace. For poetry translation specifically, the stylistic shift from generic language to poetic language may be even more compact than general domain adaptation, making LoRA an especially strong fit.

### Decoder-only vs. Encoder-Decoder Tradeoff

| Property | Qwen2.5-1.5B (Decoder-only) | mT5-base (Encoder-Decoder) |
|---|---|---|
| Translation inductive bias | Low — uses prompted generation | High — built for seq2seq |
| Cross-lingual pretraining | Primarily Chinese/English | 101 languages via mC4 |
| Parameter count | ~1.5B | ~580M |
| LoRA targets | Self-attention only | Self + Cross attention |
| Inference | Autoregressive from prompt | Encoder outputs reused |

This architectural comparison is itself a **novel contribution** — isolating whether quality gains come from architecture vs. PEFT adaptation, which existing poetry translation work has not cleanly separated.

## Evaluation Metrics

- **BLEU** — n-gram overlap (literal accuracy)
- **ROUGE** — recall-oriented overlap (content coverage)
- **BERTScore** — contextual semantic similarity (better suited for poetry's loose word-level mappings)
- **Qualitative human review** — for elegance, following the adequacy/fluency/elegance framework from Chen et al. 2025
