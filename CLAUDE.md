# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECE175B research project (UCSD, Juqy Chen & Emma Huang): fine-tuning small LLMs for **classical Chinese poetry → English translation**, preserving adequacy, fluency, and poetic elegance.

Two training paths under investigation:
- **Path 1:** Qwen2.5-1.5B + QLoRA (decoder-only, parameter-efficient)
- **Path 2:** mT5-base + LoRA (encoder-decoder, architecture better suited for translation)

## Dataset Pipeline

The only code so far is `build_dataset.py`, which merges two source datasets into train/valid/test splits.

**Run dataset inspection first** (prints field names from raw files to verify FIELD_MAP constants):
```bash
python build_dataset.py --inspect
```

**Build the final dataset** (outputs to `data/combined/{train,valid,test}.jsonl`):
```bash
python build_dataset.py
# optional overrides:
python build_dataset.py --poetmt_dir PoetMT-main/PoetMT-main/all_poems --ccpm_dir CCPM-master --output_dir data/combined
```

## Data Sources

**PoetMT** (`PoetMT-main/PoetMT-main/all_poems/`) — primary translation data
- `tang.jsonl`, `song.jsonl`, `yuan.jsonl` — parallel classical ZH / English poem pairs
- `tang-background.jsonl`, `song-background.jsonl`, `yuan-background.jsonl` — enrichment data (modern ZH translation, annotations, historical background), joined to poems by title

**CCPM** (`CCPM-master/`) — auxiliary semantic understanding data
- `train.jsonl`, `valid.jsonl` — modern Chinese description → correct classical line (multiple-choice format; the correct choice is extracted as a pair)
- `test_public.jsonl` is skipped (no answer labels)

## Dataset Architecture

`build_dataset.py` produces two task types per record:

| `task` field | Source | Input → Output |
|---|---|---|
| `translation` | PoetMT | classical_zh (+ optional context) → English |
| `auxiliary_understanding` | CCPM | modern_zh → classical_zh |

Records are formatted as chat-style `messages` arrays (system / user / assistant).

**Split strategy:** Test is carved deterministically from the tail of PoetMT before any shuffling, ensuring it is always non-empty. Test contains **translation-only** samples for clean BLEU/BERTScore evaluation — no CCPM auxiliary samples in test.

**Field resolution:** Raw JSONL keys vary across files. `POETMT_FIELD_MAP` and `BACKGROUND_FIELD_MAP` in `build_dataset.py` define candidate key lists tried in order. If field names in the raw data don't match, update these maps after running `--inspect`.

## Paper

`neurips_2026.tex` — paper draft using the NeurIPS 2026 LaTeX template. Currently contains the blank template; actual content TBD.

Compile with:
```bash
pdflatex neurips_2026.tex
```
