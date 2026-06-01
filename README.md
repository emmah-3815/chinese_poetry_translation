# Classical Chinese Poetry Translation via Parameter-Efficient Fine-Tuning

## Course & Team
- **Course:** ECE175B (UCSD)
- **Team:** Juqy Chen (juc052@ucsd.edu), Emma Huang (emh002@ucsd.edu)

## Research Problem
Build efficient, high-quality English translations of classical Chinese poetry using small or medium-sized models. The challenge is not only to preserve the poem's literal meaning, but also to retain its cultural references, imagery, conciseness, and poetic style in English.

## Why It Matters
Classical Chinese poetry is highly compressed, heavily allusive, and stylistically distinct. General-purpose LLMs produce fluent literal translations but fail to preserve adequacy, historical context, and poetic elegance. This limits accessibility for students and researchers studying Chinese literature.

## Two-Path Framework

### Path 1: Qwen2.5-1.5B + QLoRA (decoder-only)
Lightweight decoder-only baseline using prompted generation. QLoRA adds 4-bit NF4 quantization on top of LoRA, making the 1.5B model trainable on a single consumer GPU.

### Path 2: mT5-base + LoRA (encoder-decoder)
Tests whether an encoder-decoder model is inherently better suited for poetry translation than a decoder-only model of similar scale. mT5 was pretrained on 101 languages and has explicit translation inductive bias.

Standard LoRA is used here (not QLoRA). mT5-base at ~580M parameters occupies ~1.2GB in BF16 — well within GPU VRAM without quantization. QLoRA's 4-bit NF4 compression is designed for models too large to fit in VRAM (7B+); applying it to mT5-base would introduce quantization error with no memory benefit. At this scale, quantization error is proportionally larger than in large models because there are fewer parameters to absorb the approximation, so it would hurt translation quality. QLoRA is applied to Path 1 (Qwen2.5-1.5B) where it is necessary, and to Path 3 (Qwen2.5-14B) where it is essential.

**Note:** mT5-base was pre-trained only on unsupervised span-corruption (not translation), so its ZH→EN ability before fine-tuning is essentially zero. Fine-tuning on the small PoetMT corpus (581 poems) makes learning challenging.

### Path 2b: opus-mt-zh-en + LoRA (pre-trained ZH→EN encoder-decoder)
`Helsinki-NLP/opus-mt-zh-en` is a MarianMT model (~74M params) already trained on OPUS Chinese→English parallel data. It starts as a working translator — fine-tuning on poetry domain data specializes it for classical style rather than teaching translation from scratch.

Early results (1-epoch smoke test): **48 sec/epoch, BLEU=0.43** vs mT5-base's 19.5 min/epoch, BLEU=0.01. Full 15-epoch run estimated at ~12 min locally, ~8 min on free Colab T4.

The comparison **isolates whether translation gains come from architecture choice, pre-training domain, vs. parameter-efficient adaptation** — a separation no prior work cleanly addresses for poetry.

## Novelty & Significance

- Frames translation as **stylistic cloning via parameter-efficient fine-tuning**, not just linguistic equivalence
- Moves RAT's knowledge supplementation ability into the **training stage** while keeping LoRA's low-cost adaptation at deployment time
- Provides a **scalable, low-compute framework** that could generalize to other low-resource or highly stylized translation tasks (ancient prose, religious texts, indigenous storytelling)
- Introduces an **annotation ablation** (with vs. without 注释) as a clean test of whether baking domain knowledge into training improves elegance

## Existing Work & Limitations

| Work | Contribution | Limitation |
|---|---|---|
| Gao et al. 2024 | ChatGPT vs Google Translate vs DeepL on poetry | Generic LLMs lack poetic elegance |
| Chen et al. 2025 (RATs + PoetMT) | Retrieval-augmented pipeline + benchmark | Heavy inference cost, complex deployment |
| Xie 2025 (PoetryQwen) | LoRA fine-tune of Qwen2.5-14B on CCPoetry-49K | Focuses on poetry *appreciation*, not CN→EN translation |

## References
- Gao, R. et al. (2024). *Machine translation of Chinese classical poetry: ChatGPT, Google Translate, DeepL*. Humanities and Social Sciences Communications.
- Gao, L. (2010). *On English translation of classical Chinese poetry: A perspective from skopos theory*.
- Chen, A. et al. (2025). *Benchmarking LLMs for translating classical Chinese poetry*. EMNLP 2025.
- Xie, H. (2025). *System report for CCL25-eval task 5*. CCL 2025.
- Xue, L. et al. (2021). *mT5: A massively multilingual pre-trained text-to-text transformer*. NAACL 2021.

---

## Methods

### Section Outline

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


# How To Get Started
Run these in your terminal to set up the environment
1. Create the environment
conda create --name qwen_poetry python=3.12 -y
2. Activate it
conda activate qwen_poetry

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

3. Core fine-tuning libraries
pip install unsloth "trl<0.12.0" peft accelerate bitsandbytes

4. Transformers and data handling
pip install transformers datasets sentencepiece protobuf


5. install unsloth_zoo
pip install unsloth_zoo

6. Optional: WandB for tracking your poetry translation metrics
pip install wandb

#### Run the qwen model tester
python qwen_test.py


### 3.1 Problem Formulation

Let a classical Chinese poem be a sequence of tokens:

$$x = (x_1, x_2, \dots, x_m)$$

and its English translation target:

$$y = (y_1, y_2, \dots, y_n)$$

The goal is to learn a conditional distribution $P_\theta(y \mid x)$ that maximizes translation quality across **adequacy, fluency, and poetic elegance** (criteria from Chen et al. 2025).

Training minimizes the standard **negative log-likelihood (NLL)** loss over a parallel corpus

$\mathcal{D} = \{(x^{(i)}, y^{(i)})\}$: $$\mathcal{L}(\theta) = -\sum_{i=1}^{|\mathcal{D}|} \sum_{t=1}^{n} \log P_\theta(y_t^{(i)} \mid y_{<t}^{(i)}, x^{(i)})$$

### 3.2 Parameter-Efficient Fine-Tuning via LoRA

Both paths use **Low-Rank Adaptation (LoRA)** (Hu et al., 2021). The pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ is frozen, and a trainable low-rank decomposition is injected:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$.

The forward pass becomes:

$$h = W_0 x + \frac{\alpha}{r} BAx$$

where $\alpha$ is a scaling hyperparameter. Only $A$ and $B$ are updated, reducing trainable parameters from $d \times k$ to $r(d + k)$.

A full fine-tune of Qwen2.5-1.5B updates ~1.5B parameters. With LoRA at $r = 16$, only ~2–5M parameters are updated — a **>99% reduction**.

### 3.3 Path 1: Qwen2.5-1.5B + QLoRA

#### Architecture
Qwen2.5-1.5B is a **decoder-only transformer** that models $P_\theta(y \mid x)$ autoregressively. The input is formatted as a prompt:

```
Translate the following classical Chinese poem into English:
[Chinese poem]
English:
```

The model generates tokens left-to-right conditioning on all previous context.

#### QLoRA Extension
QLoRA (Dettmers et al., 2023) adds **4-bit NormalFloat (NF4) quantization** on top of LoRA. Base weights are quantized as:

$$W_0^{q} = \text{quantize}_{NF4}(W_0)$$

stored at 4-bit precision, while LoRA adapters $A, B$ remain in full BFloat16. The forward pass dequantizes on-the-fly: $$h = \text{dequantize}(W_0^q) x + \frac{\alpha}{r} BAx$$

This combines two savings: 4-bit quantization reduces base model VRAM by ~75%, and LoRA restricts gradient updates to low-rank matrices. A 1.5B model becomes feasible on a single 16GB consumer GPU.

### 3.4 Path 2: mT5-base + LoRA

#### Why LoRA, not QLoRA
QLoRA's value proposition is enabling training on a model that would not otherwise fit in GPU VRAM. mT5-base (~580M parameters, ~1.2GB in BF16) fits comfortably in under 5GB including optimizer state and activations — no quantization needed. Applying 4-bit NF4 quantization to a model this small would compress a weight space that is already constrained, introducing approximation error proportionally larger than in a 7B+ model where the same error averages out across far more parameters. The result is a quality regression with no hardware benefit. QLoRA is therefore reserved for paths where it is necessary: Qwen2.5-1.5B (borderline) and Qwen2.5-14B (required).

#### Architecture
mT5-base is a **multilingual encoder-decoder transformer** pretrained on 101 languages with span-corruption.

- **Encoder:** $H = \text{Encoder}(x)$, where $H \in \mathbb{R}^{m \times d}$
- **Decoder:** autoregressively generates conditioning on both $H$ and previous tokens: $$P_\theta(y_t \mid y_{<t}, x) = \text{softmax}(W_o \cdot \text{Decoder}(y_{<t}, H))$$

#### LoRA on Encoder-Decoder Attention
Adapters are applied to:
- Self-attention $\{W_q, W_v\}$ in encoder and decoder
- **Cross-attention** $\{W_q, W_v\}$ in decoder

Cross-attention adaptation is critical for poetry: it governs how the model attends to specific Chinese source tokens when generating each English word — directly affecting poetic fidelity.

#### E2 Implementation (`train_e2_mt5.py`)

Input is prefixed with `"translate classical Chinese to English: "` followed by the classical Chinese poem text (first, for truncation safety) then context (title, poet, modern_zh, annotations). CCPM auxiliary samples are excluded — E2 trains on PoetMT translation pairs only (filters `task == "translation"` from `data/combined/`).

#### Path 2b Implementation (`pipelines/opus_mt/train_opus_mt.py`)

No task prefix — MarianMT already knows the ZH→EN direction from OPUS pre-training. Same data format and LoRA configuration as E2, but targets `q_proj`/`v_proj` (MarianMT attention module names) and uses `Helsinki-NLP/opus-mt-zh-en` as the base model.

| Hyperparameter | E2 Value |
|---|---|
| LoRA rank $r$ | 16 |
| LoRA $\alpha$ | 32 |
| LoRA target modules | `q`, `v` |
| LoRA dropout | 0.05 |
| Precision | bf16 (Ampere GPU), fp32 fallback |
| Learning rate | 3e-4 |
| Batch size (per device) | 8 |
| Gradient accumulation | 4 steps (effective batch 32) |
| Warmup steps | 100 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Max source length | 512 tokens |
| Max target length | 256 tokens |
| Beam size (eval) | 4 |
| Early stopping patience | 2 epochs |
| Checkpoint metric | BLEU |

### 3.5 Training Procedure

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

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| LoRA rank $r$ | 16 (ablate: 8, 32) |
| LoRA $\alpha$ | 32 |
| LoRA target modules | `q_proj, v_proj` |
| Quantization (Path 1) | NF4 4-bit |
| Learning rate | 2e-4 |
| Optimizer | AdamW |
| LR schedule | Cosine |

### Theoretical Analysis

#### Why Low-Rank Works
The hypothesis (Hu et al.) is that fine-tuning weight updates have a **low intrinsic rank** — task-specific adaptation lies in a low-dimensional subspace. For poetry translation specifically, the stylistic shift from generic language to poetic language may be even more compact than general domain adaptation, making LoRA an especially strong fit.

#### Decoder-only vs. Encoder-Decoder Tradeoff

| Property | Qwen2.5-1.5B (Decoder-only) | mT5-base (Encoder-Decoder) |
|---|---|---|
| Translation inductive bias | Low — uses prompted generation | High — built for seq2seq |
| Cross-lingual pretraining | Primarily Chinese/English | 101 languages via mC4 |
| Parameter count | ~1.5B | ~580M |
| PEFT method | QLoRA (4-bit NF4, VRAM-constrained) | LoRA (BF16, fits in VRAM without quantization) |
| LoRA targets | Self-attention only | Self + Cross attention |
| Inference | Autoregressive from prompt | Encoder outputs reused |

This architectural comparison is itself a **novel contribution** — isolating whether quality gains come from architecture vs. PEFT adaptation, which existing poetry translation work has not cleanly separated.

### Evaluation Metrics

- **BLEU** — n-gram overlap (literal accuracy)
- **ROUGE** — recall-oriented overlap (content coverage)
- **BERTScore** — contextual semantic similarity (better suited for poetry's loose word-level mappings)
- **Qualitative human review** — for elegance, following the adequacy/fluency/elegance framework from Chen et al. 2025
