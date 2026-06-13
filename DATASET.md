## Datasets

### Sources

#### PoetMT (primary — CN→EN parallel)
- From Chen et al. 2025 (EMNLP), `andongBlue/PoetMT` on GitHub
- ~790 classical poems with human English translations across **Tang, Song, Yuan** dynasties
- Each poem has matching `-background.jsonl` files with modern Chinese 译文 and creation context

#### CCPM (auxiliary — classical ↔ modern Chinese)
- From `THUNLP-AIPoet/CCPM` on GitHub
- ~24,000 (classical line ↔ modern Chinese paraphrase) pairs
- **Important:** CCPM has **no English translations**. Originally designed as a multiple-choice matching task, not translation
- Used only for auxiliary classical Chinese understanding training, not for translation evaluation

### Why Both?

PoetMT alone is small (~790 pairs). CCPM provides large-scale semantic grounding in classical Chinese, helping the model understand source-language meaning before it ever sees the translation task. This mirrors the proposal's stated goal: combining "a small amount of high-quality Chinese-English classical poetry translation data and NiuTrans / CCLUE / CCPM-style auxiliary understanding data for semantic support."

### PoetMT Field Mapping (verified from inspection)

| Field in raw JSONL | Meaning |
|---|---|
| `src` | Classical Chinese poem text |
| `ref` | English human translation |
| `note` | 注释 — Chinese annotations |
| `english_note` | English explanation of annotations |
| `title` / `english_title` | Poem title (Chinese / English) |
| `author` / `english_author` | Poet (Chinese / English) |
| `background.fanyi` | Modern Chinese 译文 (in nested dict) |
| `background.about` | Creation context / 创作背景 (in nested dict) |

Background files are joined to poem files by `title`.

### Dataset Design Decisions

#### 注释 (Annotations) Are Included as Training Context
- 注释 contain exactly the kind of background knowledge RATs retrieved externally at inference time
- Including them in the training prompt directly supports the proposal's goal: *"moving RAT's knowledge supplementation ability into the training stage"*
- Becomes a novel contribution and supports an **annotation ablation** (with vs. without 注释)
- **Caveat for the report:** the original English translations were produced without 注释 visible to translators, so there's a slight input-output asymmetry. Worth one acknowledgment line in the methods section.

#### Test Set Is Translation-Only
- No CCPM auxiliary samples mixed into test
- Reason: BLEU/BERTScore only make sense on Chinese→English pairs
- Test set carved out **before shuffling** for reproducibility
- 78 test samples is acceptable for poetry translation — this domain emphasizes qualitative evaluation alongside BLEU

#### CCPM `test_public.jsonl` Excluded
- Public competition file with no `answer` field provided
- Only `train.jsonl` and `valid.jsonl` are used

#### Input Prompt Includes Rich Context
Each PoetMT training sample's user message includes (when available):
- Title (Chinese)
- Poet + dynasty
- Annotations / 注释 (527/621 train, 59/68 valid, 72/78 test)
- Creation background (309/621 train, 33/68 valid, 40/78 test)
- Classical Chinese poem text

Modern Chinese 译文 and English notes are **not** currently included in the prompt despite being present in the raw PoetMT files.

This is a meaningful contrast to standard MT setups — the model is told *who, when, and what to watch for* before producing the translation.

### E2 / opus-mt Dataset

For E2 (mT5-base + LoRA) and Path 2b (opus-mt + LoRA), CCPM is excluded at training time. Both scripts read from `data/combined/` and filter to `task == "translation"` records only (single builder: `build_dataset.py`).

**Prompt format:** Classical Chinese poem appears FIRST in the user message so it is never truncated at MAX_SRC_LEN=512. Metadata (title, poet, modern_zh, annotations) follows. The mT5 encoder input is prefixed with `"translate classical Chinese to English: "`; opus-mt uses raw source text with no prefix.

| Split | Translation samples | Source |
|---|---|---|
| Train | 621 | `data/combined/train.jsonl` filtered to `task == "translation"` |
| Valid | 68 | `data/combined/valid.jsonl` filtered to `task == "translation"` |
| Test (local) | 78 | `data/combined/test.jsonl` (chat-format, Juqy's split) |
| Test (canonical) | 78 | `data/combined/test_canonical.jsonl` (flat-format, Emma's set) |

### Canonical Test Set

`data/combined/test_canonical.jsonl` — Emma's 78-poem flat-format set sourced from `origin/qwen:results/test.jsonl`. Fields: `chinese`, `english`, `title`, `author`, `dynasty`, `fanyi`, `shangxi`, `about`.

**This is the paper Table 1 test set.** Use `--flat_test` in `eval_e2_mt5.py` to evaluate against it:
```bash
python eval_e2_mt5.py --adapter_dir models/<run>/lora_adapter --flat_test
```

See `split_alignment_status.md` for the historical root-cause analysis of the 72 vs 78 discrepancy.

### E1+E2 Combined Dataset Statistics (latest run)

| Split | Total | Translation | Auxiliary (CCPM) |
|---|---|---|---|
| Train | 21,712 | 621 | 21,091 |
| Valid | 2,638 | 68 | 2,570 |
| Test | 78 | 78 | 0 |

- **Annotated (with 注释):** 527 train / 59 valid / 72 test
- **With creation background:** 309 train / 33 valid / 40 test
- **With 译文 (modern ZH):** 0 — not currently included in prompt messages

### Cleaning Filters Applied

- Length bounds (separate ranges for classical / English / modern Chinese)
- Language ratio checks — guards against swapped or garbled fields
- Deduplication on both source and target
- Modern paraphrase must be longer than classical line (CCPM sanity check)
- Reject identical source = target

### Open Items

- Could pull additional bilingual poetry from sources like NiuTrans Poetry to grow the parallel set
- Annotation ablation: train one model with `note` in prompt, one without, compare BLEU/BERTScore
