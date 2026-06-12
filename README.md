# Classical Chinese Poetry Translation via Parameter-Efficient Fine-Tuning

## Course & Team
- **Course:** ECE175B (UCSD)
- **Team:** Juqy Chen (juc052@ucsd.edu), Emma Huang (emh002@ucsd.edu)
- **Code:** https://github.com/emmah-3815/chinese_poetry_translation

## Research Problem
This project investigates whether model architecture or instruction-tuned decoder-only generation is more effective for low-resource classical Chinese poetry translation. We compare Qwen2.5-Instruct (decoder-only, instruction-tuned) against two encoder-decoder models — opus-mt-zh-en (translation-pretrained, ≈74M parameters) and mT5-base (general multilingual, ≈580M parameters) — all fine-tuned with parameter-efficient LoRA or QLoRA adapters on the same dataset.

The primary comparison is **Qwen2.5-0.5B-Instruct + QLoRA** (≈500M parameters) vs **opus-mt-zh-en + LoRA** (≈74M parameters), holding fine-tuning method and dataset fixed while varying architecture and pretraining domain.

## Why It Matters
Classical Chinese poetry is highly compressed, heavily allusive, and stylistically distinct. General-purpose LLMs produce fluent literal translations but fail to preserve adequacy, historical context, and poetic elegance. This limits accessibility for students and researchers studying Chinese literature. Current approaches either rely on expensive retrieval-augmented inference pipelines or focus on poetry appreciation rather than translation.

## Experiment Design

| Exp | Model | PEFT | Role |
|---|---|---|---|
| E1 | Qwen2.5-0.5B-Instruct | QLoRA | Decoder-only, **primary comparison** |
| E2 | opus-mt-zh-en | LoRA | Encoder-decoder, translation-pretrained, **primary comparison** |
| E3 | Qwen2.5-1.5B-Instruct | QLoRA | Decoder-only, scale ablation |
| E4 | mT5-base | LoRA | Encoder-decoder, general multilingual |
| E5 | Qwen2.5-0.5B-Instruct | LoRA | QLoRA vs LoRA comparison (0.5B) |
| E6 | Qwen2.5-1.5B-Instruct | LoRA | QLoRA vs LoRA comparison (1.5B) |

**E1 vs. E2** is the primary architectural comparison (decoder-only vs. encoder-decoder). **E1 vs. E3** ablates scale within the decoder-only family. **E2 vs. E4** isolates the effect of translation-specific pretraining (opus-mt) versus general multilingual pretraining (mT5). **E1 vs. E5** and **E3 vs. E6** isolate the effect of quantization.

## Model Architectures

### Qwen2.5-0.5B / 1.5B-Instruct + QLoRA (Decoder-Only)
Instruction-tuned decoder-only transformer. The Instruct variant already follows translation prompts without bespoke prompt engineering; QLoRA specializes this aligned behavior toward poetic elegance. QLoRA adds 4-bit NF4 quantization on top of LoRA, yielding an 80–85% VRAM reduction relative to full fine-tuning and keeping Qwen2.5-1.5B within 16 GB on a single consumer GPU.

**Adapter targets:** `q_proj`, `k_proj`, `v_proj`, `o_proj` (all attention projections) + `gate_proj`, `up_proj`, `down_proj` (MLP layers). Adapting both attention and MLP captures stylistic shifts in both where the model attends and how it generates English vocabulary.

### opus-mt-zh-en + LoRA (Encoder-Decoder, Translation-Pretrained)
`Helsinki-NLP/opus-mt-zh-en` is a MarianMT model (~74M params) pretrained end-to-end on Chinese→English OPUS parallel data. It enters fine-tuning as a working translator; LoRA specializes it for classical poetic style rather than teaching translation from scratch. Standard LoRA (no quantization) is used — the model fits in VRAM without memory compression.

**Adapter targets:** `q_proj`, `k_proj`, `v_proj`, `out_proj` (all attention in encoder/decoder self-attention and decoder cross-attention) + `fc1`, `fc2` (feed-forward layers). No task prefix at inference — MarianMT already encodes the ZH→EN direction from OPUS pretraining.

### mT5-base + LoRA (Encoder-Decoder, General Multilingual)
mT5-base (~580M params) was pretrained on 101 languages with span-corruption — it never produces a full translated sentence before fine-tuning, so LoRA must learn the cross-lingual mapping from scratch on the small PoetMT corpus. Input is prefixed with `"translate classical Chinese to English: "`.

**Adapter targets:** `q`, `v` projections in all self-attention layers (encoder, decoder, and cross-attention). More conservative configuration; key/output projections and FFN left in pretrained state.

**Note:** mT5-base failed to produce coherent English output in experiments (BLEU-4 ≈ 0) and is not a viable deployment option at this data scale.

## Novelty & Significance

- **Size-controlled architectural comparison** — decoder-only (Qwen2.5-Instruct) vs. encoder-decoder under an identical LoRA/QLoRA fine-tuning regime; no prior work has cleanly isolated this variable for Chinese poetry translation
- **opus-mt baseline** — direct comparison between translation-pretrained encoder-decoder (opus-mt-zh-en) and instruction-tuned decoder-only (Qwen2.5) under the same regime, isolating the contribution of translation-specific pretraining
- **Stylistic cloning via PEFT** — frames translation as stylistic cloning rather than pure linguistic equivalence; moves contextual knowledge (modern paraphrases, creation background) from retrieval into training prompts, eliminating per-query retrieval cost at deployment
- **Scalable low-compute framework** that could generalize to other highly stylized or low-resource translation tasks (ancient prose, religious texts, indigenous storytelling)

## Existing Work & Limitations

| Work | Contribution | Limitation |
|---|---|---|
| Gao et al. 2024 | ChatGPT vs Google Translate vs DeepL on poetry | Generic LLMs lack poetic elegance |
| Chen et al. 2025 (RATs + PoetMT) | Retrieval-augmented pipeline + benchmark | Heavy inference cost, complex deployment |
| Xie 2025 (PoetryQwen) | LoRA fine-tune of Qwen2.5-14B on CCPoetry-49K | Focuses on poetry *appreciation*, not CN→EN translation |

## References
- Gao, R. et al. (2024). *Machine translation of Chinese classical poetry: ChatGPT, Google Translate, DeepL*. Humanities and Social Sciences Communications. https://doi.org/10.1057/s41599-024-03363-0
- Gao, L. (2010). *On English translation of classical Chinese poetry: A perspective from skopos theory*. Journal of Language Teaching and Research.
- Chen, A. et al. (2025). *Benchmarking LLMs for translating classical Chinese poetry: Evaluating adequacy, fluency, and elegance*. EMNLP 2025.
- Dettmers, T. et al. (2023). *QLoRA: Efficient finetuning of quantized LLMs*. https://arxiv.org/abs/2305.14314
- Hu, E. J. et al. (2021). *LoRA: Low-rank adaptation of large language models*. arXiv:2106.09685.
- Xie, H. (2025). *System report for CCL25-eval task 5*. CCL 2025.
- Xue, L. et al. (2021). *mT5: A massively multilingual pre-trained text-to-text transformer*. NAACL 2021.
- Li, W. et al. (2021). *CCPM: A Chinese classical poetry matching dataset*. arXiv:2106.01979.
- Qwen Team. (2024). *Qwen2.5: A party of foundation models*. https://qwenlm.github.io/blog/qwen2.5/

---

## How To Get Started

Run these in your terminal to set up the environment.

```bash
# 1. Create and activate the environment
conda create --name qwen_poetry python=3.12 -y
conda activate qwen_poetry

# 2. PyTorch (CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 3. Core fine-tuning libraries
pip install unsloth "trl<0.12.0" peft accelerate bitsandbytes

# 4. Transformers and data handling
pip install transformers datasets sentencepiece protobuf

# 5. Unsloth zoo
pip install unsloth_zoo

# 6. Optional: WandB for experiment tracking
pip install wandb
```

#### Run the Qwen model tester
```bash
python qwen_test.py
```

---

## Methods

### Section Outline

```
3. Method
   3.1 LoRA and QLoRA Implementation
   3.2 Model Architectures
       3.2.1 Qwen2.5-Instruct (Decoder-Only) + QLoRA
       3.2.2 mT5-base (Encoder-Decoder) + LoRA
       3.2.3 opus-mt-zh-en (Encoder-Decoder, Translation-Pretrained) + LoRA
   3.3 Training Objective
   3.4 Adapter Placement
4. Theoretical Analysis
   4.1 Why Low-Rank Adaptation Works
   4.2 Why Instruct over Base
   4.3 Decoder-Only vs. Encoder-Decoder Trade-off
5. Experiment Design
   5.1 Hyperparameters
   5.2 Dataset
   5.3 Evaluation Metrics
```

### 3.1 Problem Formulation

Let a classical Chinese poem be a sequence of tokens:

$$x = (x_1, x_2, \dots, x_m)$$

and its English translation target:

$$y = (y_1, y_2, \dots, y_n)$$

The goal is to learn a conditional distribution $P_\theta(y \mid x)$ that maximizes translation quality across **adequacy, fluency, and poetic elegance** (criteria from Chen et al. 2025).

Training minimizes the standard **negative log-likelihood (NLL)** loss over a parallel corpus $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}$:

$$\mathcal{L}(\theta) = -\sum_{i=1}^{|\mathcal{D}|} \sum_{t=1}^{n} \log P_\theta(y_t^{(i)} \mid y_{<t}^{(i)}, x^{(i)})$$

### 3.2 Parameter-Efficient Fine-Tuning via LoRA

All models use **Low-Rank Adaptation (LoRA)** (Hu et al., 2021). The pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ is frozen, and a trainable low-rank decomposition is injected:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$. At initialization, $A \sim \mathcal{N}(0, \sigma^2)$ and $B = 0$, ensuring the adapter contributes nothing at the start of training. The forward pass becomes:

$$h = W_0 x + \frac{\alpha}{r} BAx$$

where $\alpha$ is a scaling hyperparameter. Only $A$ and $B$ are updated, reducing trainable parameters from $d \times k$ to $r(d + k)$.

**QLoRA** (used for Qwen2.5) additionally quantizes $W_0$ to 4-bit NF4:

$$h = \text{dequantize}(W_0^{NF4}) x + \frac{\alpha}{r} BAx$$

storing base weights at 4-bit precision while LoRA adapters remain in BFloat16. This yields an 80–85% VRAM reduction relative to full fine-tuning.

**Standard LoRA** (used for mT5-base and opus-mt-zh-en) keeps $W_0$ in bf16/fp32 — both models fit in VRAM without quantization, and applying NF4 compression at this scale would introduce proportionally larger approximation error with no memory benefit.

### 3.3 Training Objective

The training objective is cross-entropy loss between model-generated tokens and expert translations. For QLoRA (Qwen2.5), the frozen weights are quantized to 4-bit NF4:

$$Y = (W^{NF4} \cdot c_1 \cdot c_2 + L_A L_B) X$$

where $c_1, c_2$ are quantization constants and $L_A, L_B$ are the low-rank adapter matrices. For standard LoRA (mT5-base, opus-mt-zh-en), frozen weights remain in bf16/fp32 and the same cross-entropy loss applies without quantization constants.

### 3.4 Training Procedure

```
Algorithm: Multi-Experiment PEFT Training for Poetry Translation

Input:  Parallel corpus D = {(x_i, y_i)}, pretrained model M, rank r, epochs E
Output: Fine-tuned adapter weights {A, B}

1. Freeze all original weights W_0 of M
2. Initialize LoRA matrices: A ~ N(0, σ²), B = 0
   (B = 0 ensures ΔW = 0 at start, preserving pretrained behavior)
3. [Qwen only] Quantize W_0 to NF4 4-bit precision
4. For epoch = 1 to E:
     For each batch (x, y) in D:
       a. Compute forward pass with W = W_0 + (α/r)·BA
       b. Compute cross-entropy loss against expert translation y
       c. Compute gradients only w.r.t. {A, B}
       d. Update {A, B} with AdamW
5. Return trained {A, B}
```

Gradient flow never touches $W_0$, preserving general language knowledge while specializing the adapter to poetic style.

### Hyperparameters

| Hyperparameter | Qwen2.5-0.5B / 1.5B (QLoRA) | opus-mt-zh-en (LoRA) | mT5-base (LoRA) |
|---|---|---|---|
| LoRA rank $r$ | 32 | 32 | 16 |
| LoRA $\alpha$ | 64 | 64 | 32 |
| LoRA target modules | `q/k/v/o_proj`, `gate/up/down_proj` | `q/k/v/out_proj`, `fc1/fc2` | `q`, `v` |
| LoRA dropout | 0.05 | 0.05 | 0.05 |
| Quantization | NF4 4-bit | — | — |
| Learning rate | 2e-4 | 2e-4 | 2e-4 |
| Optimizer | AdamW | AdamW | AdamW |
| Gradient accumulation | 16 steps | 4 steps | 4 steps |
| Early stopping | None | Patience 2–3 epochs (BLEU) | Patience 2–3 epochs (BLEU) |

### Dataset

**PoetMT** (Chen et al. 2025) is the primary parallel corpus: ~790 classical Chinese poems with expert English translations spanning Tang, Song, and Yuan dynasties. Each poem is paired with a modern Chinese paraphrase (*fanyi*) and creation context (*shangxi*) used to enrich the training prompt.

**Preprocessing:** length bounds (classical 4–300 chars; English 5–2,000 chars); language-ratio guards (≥30% Chinese in source, ≥30% Latin in target); deduplication; rejection of identical source–target pairs.

| Split | Count |
|---|---|
| Train | 621 |
| Valid | 68 |
| Test | 78 |

A fixed 78-poem canonical test set is shared by all models; no test poem appears in any training split.

**Prompt format (Qwen / E1, E3):** Each training sample includes poem title, poet and dynasty, modern Chinese paraphrase, character annotations (*zhùshì*), and creation background. Classical Chinese text is placed first to avoid truncation.

**Prompt format (opus-mt / E2, mT5 / E4):** Encoder receives structured content (title, poet, dynasty, classical poem). mT5 is additionally prefixed with `"translate classical Chinese to English: "`. At inference, E2 and E4 are evaluated on bare Chinese text only, testing generalization beyond the training distribution.

### Theoretical Analysis

#### Why Low-Rank Works
Hu et al. hypothesize that fine-tuning weight updates have a **low intrinsic rank** — task-specific adaptation lies in a low-dimensional subspace. For classical poetry translation, the stylistic shift from generic prose to poetic English verse may be even more compact than general-domain adaptation because the target distribution is narrow and highly structured, making LoRA an especially strong fit.

#### Why Instruct over Base
The base Qwen2.5 model is a continuation model: given a classical Chinese poem, it is liable to continue generating Chinese text rather than translating. The Instruct variant has been aligned via instruction tuning to follow explicit translation prompts, so QLoRA specializes already-aligned behavior toward poetic elegance rather than first re-teaching the model what "translate to English" means.

#### Decoder-Only vs. Encoder-Decoder Trade-off

| Property | Qwen2.5-Instruct (Decoder-Only) | opus-mt-zh-en (Encoder-Decoder) | mT5-base (Encoder-Decoder) |
|---|---|---|---|
| Translation prior | Strong (instruction-aligned) | Strong (OPUS parallel data) | None (span-corruption only) |
| Cross-lingual pretraining | Chinese/English + broad web | ZH→EN parallel corpus | 101 languages via mC4 |
| Parameter count | 500M / 1.5B | ~74M | ~580M |
| PEFT method | QLoRA (4-bit NF4) | LoRA (bf16) | LoRA (bf16) |
| LoRA targets | Self-attention + MLP | All attention + FFN | Self-attention q/v only |
| Inference | Autoregressive from prompt | Encoder outputs reused | Encoder outputs reused |

### Evaluation Metrics

- **BLEU-4** — 4-gram overlap with reference translations (standard MT benchmark; known weak signal for poetry where paraphrase is desirable)
- **ROUGE-L** — longest common subsequence overlap; captures structural similarity beyond exact n-gram matches
- **BERTScore** — contextual semantic similarity via BERT embeddings; more robust to paraphrase and better suited to poetry
- **Human evaluation (15 poems)** — Adequacy, Fluency, and Poeticness rated on a 1–5 Likert scale (5 poems per dynasty: Tang, Song, Yuan), following the framework of Chen et al. 2025
- **Memory efficiency** — peak GPU VRAM measured via `torch.cuda.max_memory_reserved() / 1024**3` at end of training and inference

---

## Results

### Automatic Metrics

| Model | BLEU-4 | ΔBLEU-4 | Brevity | ROUGE-L | BERTScore |
|---|---|---|---|---|---|
| Qwen1.5B-Instruct + QLoRA | 2.8582 | +1.1123 | 1.0257 | 0.2151 | **0.8643** |
| Qwen1.5B-Instruct + LoRA | **2.8991** | +1.1532 | 1.0070 | 0.2036 | 0.8621 |
| Qwen1.5B-Instruct (baseline) | 1.7459 | — | 1.9129 | 0.1625 | 0.8403 |
| opus-mt-zh-en + LoRA | 2.1371 | **+1.3460** | 0.7551 | 0.2031 | 0.8574 |
| opus-mt-zh-en (baseline) | 0.7911 | — | 0.5696 | 0.1593 | 0.8364 |
| Qwen0.5B-Instruct + QLoRA | 2.0236 | +1.2781 | 0.9098 | 0.1823 | 0.8599 |
| Qwen0.5B-Instruct + LoRA | 1.6477 | +0.9022 | 1.0780 | 0.1788 | 0.8552 |
| Qwen0.5B-Instruct (baseline) | 0.7455 | — | 2.3022 | 0.1326 | 0.8231 |
| mT5-base + LoRA | 0.2035 | +0.2030 | N/A | 0.0868 | 0.8074 |
| mT5-base (baseline) | 0.0005 | — | N/A | 0.0043 | 0.7508 |

Key findings:
- **Qwen2.5-1.5B+QLoRA achieves BLEU-4 2.86**, surpassing GPT-4+RAT (2.2, Chen et al. 2025) despite being orders of magnitude smaller
- Every fine-tuned model substantially outperformed its untuned baseline; opus-mt+LoRA achieved the largest absolute BLEU gain (Δ+1.35)
- **Scale-dependent crossover:** at 0.5B, opus-mt+LoRA (74M params) outperforms Qwen0.5B+QLoRA (500M params), showing translation-specific pretraining compensates for a 7× parameter disadvantage; at 1.5B, Qwen overtakes opus-mt by +0.75 BLEU-4
- mT5-base failed to produce coherent English output and is not a viable deployment option at this data scale

### Human Evaluation (15 poems, 1–5 Likert scale)

| Model | Adequacy | Fluency | Poeticness | Overall |
|---|---|---|---|---|
| Qwen0.5B-Instruct + QLoRA | **2.73** | **4.40** | **3.80** | **3.64** |
| Qwen0.5B-Instruct (baseline) | 2.13 | 3.47 | 2.27 | 2.62 |
| opus-mt-zh-en + LoRA | 2.20 | 2.27 | 1.67 | 2.04 |
| opus-mt-zh-en (baseline) | 1.53 | 1.80 | 1.13 | 1.49 |

QLoRA fine-tuning yields the largest per-dimension gains for Qwen0.5B: Adequacy (+0.60), Fluency (+0.93), Poeticness (+1.53). The weakest dimension diverges by family: opus-mt variants lag on Poeticness (1.13–1.67), reflecting the gap between semantic adequacy and genuine poetic register; Qwen variants lag on Adequacy, suggesting instruction-tuned pretraining captures fluency and register more readily than precise semantic fidelity.

### Memory Efficiency

| Model | Train Peak VRAM (GB) | Inference VRAM (GB) |
|---|---|---|
| Qwen1.5B-Instruct + QLoRA | **8.42** | 1.91 |
| Qwen1.5B-Instruct + LoRA | 12.73 | 1.91 |
| Qwen1.5B-Instruct (baseline) | N/A | 1.55 |
| Qwen0.5B-Instruct + QLoRA | 7.43 | 0.90 |
| Qwen0.5B-Instruct + LoRA | 8.83 | 0.90 |
| Qwen0.5B-Instruct (baseline) | N/A | 0.66 |
| opus-mt-zh-en + LoRA | 7.23 | **0.39** |
| opus-mt-zh-en (baseline) | N/A | 0.50 |

All training VRAM peaks remain within 16 GB — every model can be fine-tuned on a single consumer GPU. QLoRA on Qwen2.5-1.5B reduces training peak from 12.73 GB (LoRA) to 8.42 GB, a 34% reduction.

---

## Discussion

### Scale-Dependent Architectural Trade-off
The results reveal a scale-dependent crossover. At 1.5B parameters, decoder-only Qwen2.5-1.5B (BLEU-4 2.90) comfortably outperforms opus-mt-zh-en+LoRA (BLEU-4 2.14). At smaller scale the result reverses: opus-mt-zh-en+LoRA (74M, BLEU-4 2.14) outperforms Qwen2.5-0.5B+QLoRA (500M, BLEU-4 2.02) despite a 7× parameter disadvantage. Translation-specific pretraining proves highly parameter-efficient; only when the decoder-only model reaches 1.5B does its broader pretraining corpus and instruction alignment overcome this structural advantage.

Contributing factors: Qwen's tokenizer was trained on high-density Chinese text including classical forms, compactly tokenizing 4-character compounds, while mT5/MarianMT SentencePiece vocabularies optimized for modern web-prose may over-segment classical characters. Qwen's instruction alignment also means QLoRA is nudging a model that already knows what elegant translation looks like — the adapter only shifts style — whereas encoder-decoder models must simultaneously learn the translation mapping and the stylistic shift from a small corpus.

### Limitations
- **Metric sensitivity:** BLEU and ROUGE-L correlate poorly with human judgments of poeticness; human evaluation covers only 15 poems on a 1–5 Likert scale
- **Compute budget:** hyperparameter sweeps are limited within a quarter-long project
- **Generalizability:** the framework is validated only on classical Chinese p
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
