# Project Brief

## Title
Classical Chinese Poetry Translation via Parameter-Efficient Fine-Tuning

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

The comparison **isolates whether translation gains come from architecture choice vs. parameter-efficient adaptation** — a separation no prior work cleanly addresses for poetry.

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

## Cited References
- Gao, R. et al. (2024). *Machine translation of Chinese classical poetry: ChatGPT, Google Translate, DeepL*. Humanities and Social Sciences Communications.
- Gao, L. (2010). *On English translation of classical Chinese poetry: A perspective from skopos theory*.
- Chen, A. et al. (2025). *Benchmarking LLMs for translating classical Chinese poetry*. EMNLP 2025.
- Xie, H. (2025). *System report for CCL25-eval task 5*. CCL 2025.
- Xue, L. et al. (2021). *mT5: A massively multilingual pre-trained text-to-text transformer*. NAACL 2021.
