# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECE175B research project (UCSD, Juqy Chen & Emma Huang): fine-tuning small LLMs for **classical Chinese poetry → English translation**, preserving adequacy, fluency, and poetic elegance.

Training paths under investigation:
- **Path 1 (E1):** Qwen2.5-1.5B + QLoRA (decoder-only) — **Emma's work; all variants evaluated (see `qwen` branch)**
- **Path 2 (E2):** mT5-base + LoRA (encoder-decoder) — pipeline complete, full retrain on `poetmt_compact` pending
- **Path 2b (opus-mt):** Helsinki-NLP/opus-mt-zh-en + LoRA (MarianMT, pre-trained ZH→EN) — **new; smoke test shows 24× faster than mT5-base with BLEU=0.43 at epoch 1 vs mT5's 0.01**
- **Path 3:** Qwen2.5-14B + QLoRA (larger decoder-only baseline) — planned

## Repository Layout

```
build_dataset.py                    # Combined dataset builder (PoetMT + CCPM) → data/combined/
build_dataset_poetmt.py             # PoetMT-only builder for E2/opus-mt → data/poetmt_compact/
train_e2_mt5.py                     # E2: mT5-base + LoRA training pipeline
eval_e2_mt5.py                      # Evaluation: mT5/opus-mt adapter or baseline
convert_preds_for_poetry_eval.py    # Convert eval_e2_mt5.py preds → evaluate_poetry.py format
evaluate_poetry.py                  # Emma's eval script (E1 Qwen variants)
qwen_test.py                        # Qwen model tester (inference only)
split_metadata.json                 # Canonical test split info (seed, sizes, titles)
split_alignment_status.md           # Split mismatch analysis + canonical test set recommendation
pipelines/
  opus_mt/
    train_opus_mt.py                # Path 2b: opus-mt-zh-en + LoRA training
    train_opus_mt.ipynb             # Colab notebook (Drive mount, pip install)
data/
  PoetMT-main/            # Raw PoetMT source data
  CCPM-master/            # Raw CCPM source data
  combined/               # Output of build_dataset.py (train/valid/test.jsonl)
  poetmt_compact/         # Output of build_dataset_poetmt.py — CORRECTED format
                          #   (poem-first, compact metadata, no raw-dict backgrounds)
                          #   train=581, valid=72, test=72
models/                   # Saved LoRA adapter checkpoints
```

## Dataset Pipeline

Two dataset builders exist depending on which training path you're preparing for.

### `build_dataset_poetmt.py` — PoetMT only (used by E2 / opus-mt)

Outputs to `data/poetmt_compact/{train,valid,test}.jsonl` (default).

**Important fixes applied:** Classical Chinese poem appears FIRST in the user message (truncation-safe), Background field is cleaned (no raw Python dict), metadata capped at MAX_NOTES_CHARS=300 / MAX_MODERN_CHARS=360.

```bash
python build_dataset_poetmt.py --inspect   # print field names from raw files
python build_dataset_poetmt.py             # build the dataset → data/poetmt_compact/
```

### `build_dataset.py` — Combined (PoetMT + CCPM)

Outputs to `data/combined/{train,valid,test}.jsonl`. Includes auxiliary `understanding` task records from CCPM in train/valid (not test).

```bash
python build_dataset.py --inspect
python build_dataset.py
# optional overrides:
python build_dataset.py --poetmt_dir data/PoetMT-main/PoetMT-main/all_poems --ccpm_dir data/CCPM-master --output_dir data/combined
```

## Data Sources

**PoetMT** (`data/PoetMT-main/PoetMT-main/all_poems/`) — primary translation data
- `tang.jsonl`, `song.jsonl`, `yuan.jsonl` — parallel classical ZH / English poem pairs
- `tang-background.jsonl`, `song-background.jsonl`, `yuan-background.jsonl` — enrichment data (modern ZH translation, annotations, historical background), joined to poems by title

**CCPM** (`data/CCPM-master/`) — auxiliary semantic understanding data
- `train.jsonl`, `valid.jsonl` — modern Chinese description → correct classical line (multiple-choice format; the correct choice is extracted as a pair)
- `test_public.jsonl` is skipped (no answer labels)

## Dataset Architecture

Both builders produce chat-style `messages` arrays (system / user / assistant).

`build_dataset.py` produces two task types:

| `task` field | Source | Input → Output |
|---|---|---|
| `translation` | PoetMT | classical_zh (+ optional context) → English |
| `auxiliary_understanding` | CCPM | modern_zh → classical_zh |

**Split strategy:** Test is carved deterministically from the tail of PoetMT before any shuffling, ensuring it is always non-empty. Test contains **translation-only** samples for clean BLEU/BERTScore evaluation.

**Field resolution:** Raw JSONL keys vary across files. `POETMT_FIELD_MAP` and `BACKGROUND_FIELD_MAP` in each builder define candidate key lists tried in order. Run `--inspect` if field names change.

## Training

### E2: mT5-base + LoRA (`train_e2_mt5.py`)

Reads from `data/poetmt_compact/`. Saves LoRA adapter to `models/e2-mt5-fp32-compact/lora_adapter/`.

```bash
python train_e2_mt5.py
python train_e2_mt5.py --epochs 15 --early_stopping_patience 4 --output_dir models/e2-mt5-compact-v1
```

Key hyperparameters: LoRA r=16, α=32, target modules `q`/`v`, lr=3e-4, batch=8, grad_accum=4, beam=4, early stopping patience=4 (metric: BLEU, `greater_is_better=True`).

Input format: TASK_PREFIX + classical Chinese poem (FIRST, truncation-safe) + title/poet/modern_zh/annotations.

**Known issue with old runs:** `e2-mt5-fp32-v2` was trained on `data/poetmt` (old buggy dataset — raw dict backgrounds, poem at end of prompt). BLEU-4=0.20 on test is essentially 0 with degenerate outputs. The `poetmt_compact` retrain is the correct experiment.

### Path 2b: opus-mt-zh-en + LoRA (`pipelines/opus_mt/train_opus_mt.py`)

Uses `Helsinki-NLP/opus-mt-zh-en` (MarianMT, ~74M params) — already pre-trained on OPUS ZH→EN corpus. Dramatically better starting point than mT5-base.

```bash
python pipelines/opus_mt/train_opus_mt.py
python pipelines/opus_mt/train_opus_mt.py --epochs 15 --output_dir models/opus-mt-poetry
# For Colab: open pipelines/opus_mt/train_opus_mt.ipynb
```

Key hyperparameters: LoRA r=16, α=32, target `q_proj`/`v_proj`, lr=3e-4, batch=8, grad_accum=4, beam=4, early stopping patience=3 (metric: BLEU). **No TASK_PREFIX** (Marian already knows the task).

Smoke test results (1 epoch): 48 sec/epoch, eval_bleu=0.43 — vs mT5-base 19.5 min/epoch, bleu=0.01. Estimated full run (~15 epochs): ~12 min locally, ~8 min on Colab free T4.

### Environment Setup

```bash
conda create --name qwen_poetry python=3.12 -y
conda activate qwen_poetry
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install unsloth "trl<0.12.0" peft accelerate bitsandbytes
pip install transformers datasets sentencepiece protobuf
pip install unsloth_zoo
pip install wandb        # optional, for experiment tracking
pip install evaluate sacrebleu  # for BLEU metric in train_e2_mt5.py
```

## Evaluation

```bash
# Evaluate any fine-tuned adapter (mT5 or opus-mt)
python eval_e2_mt5.py --adapter_dir models/opus-mt-poetry/lora_adapter

# Evaluate mT5-base with no adapter (baseline)
python eval_e2_mt5.py --baseline --output_dir models/mt5-base-baseline/eval_results

# Cross-eval: convert E2 predictions to evaluate_poetry.py format
python convert_preds_for_poetry_eval.py
python evaluate_poetry.py --test_file data/poetmt_compact/test_flat.jsonl \
    --predictions_file models/.../predictions_converted.jsonl \
    --skip_human --exp_id E2
```

Metrics: BLEU-4 (sacrebleu), ROUGE-L, BERTScore F1, qualitative human review (adequacy / fluency / elegance per Chen et al. 2025).

**Cross-eval confirmed:** `eval_e2_mt5.py` and `evaluate_poetry.py` produce identical BLEU-4, ROUGE-L, and BERTScore on the same predictions (delta = 0 for all metrics). See `models/e2-mt5-fp32-v2/cross_eval/cross_eval_comparison.md`.

**Test set alignment:** Juqy's test set (72 poems, `data/poetmt_compact/test.jsonl`) ≠ Emma's test set (78 poems, `results/test.jsonl` on `qwen` branch). See `split_alignment_status.md` for analysis. Canonical test set for paper Table 1 = Emma's 78-poem flat-format set.

## Current Results (as of May 2026)

| Experiment | Model | Test BLEU-4 | ROUGE-L | BERTScore-F | Test Set |
|---|---|:---:|:---:|:---:|---|
| E0 (baseline) | Qwen2.5-1.5B (no adapter) | ~? | ~? | ~? | Emma's 78 |
| E1 | Qwen2.5-1.5B + QLoRA | **2.86** | 0.2151 | 0.8643 | Emma's 78 |
| E2 (old, buggy) | mT5-base + LoRA (old dataset) | 0.20 | 0.087 | 0.807 | Juqy's 72 |
| E2 (pending) | mT5-base + LoRA (poetmt_compact) | — | — | — | — |
| opus-mt (pending) | opus-mt-zh-en + LoRA | — | — | — | — |

Note: E2 "old, buggy" result is on the old `data/poetmt` dataset with prompt format bugs; effectively ~0 BLEU.

## Paper

`neurips_2026.tex` — not yet created. Will use the NeurIPS 2026 LaTeX template.
