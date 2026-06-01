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
- 73 test samples is acceptable for poetry translation — this domain emphasizes qualitative evaluation alongside BLEU

#### CCPM `test_public.jsonl` Excluded
- Public competition file with no `answer` field provided
- Only `train.jsonl` and `valid.jsonl` are used

#### Input Prompt Includes Rich Context
Each PoetMT training sample's input prompt includes (when available):
- English title (with Chinese in parens)
- Poet + dynasty
- Modern Chinese meaning (from `background.fanyi`)
- Annotations / 注释
- English note
- Creation background (from `background.about`)
- Classical Chinese poem text

This is a meaningful contrast to standard MT setups — the model is told *who, when, what it means, and what to watch for* before producing the translation.

### E2 / opus-mt Dataset (PoetMT-only, no CCPM)

For E2 (mT5-base + LoRA) and Path 2b (opus-mt + LoRA), CCPM is excluded. Built with `build_dataset_poetmt.py` into `data/poetmt_compact/`.

**Prompt format (corrected):** Classical Chinese poem appears FIRST in the user message so it is never truncated at MAX_SRC_LEN=512. Metadata (title, poet, modern_zh, annotations) follows. Background field is sanitized (no raw Python dicts). The mT5 encoder input is prefixed with `"translate classical Chinese to English: "`; opus-mt uses raw source text with no prefix.

| Split | Samples | Notes |
|---|---|---|
| Train | 581 | PoetMT translation pairs, full context |
| Valid | 72 | PoetMT translation pairs, full context |
| Test | 72 | Carved deterministically from tail before shuffle |

### Test Set Alignment Issue

Juqy's test set (72 poems, `data/poetmt_compact/`) and Emma's test set (78 poems, `results/test.jsonl` on `qwen` branch) **differ in size** despite using the same seed=42 + tail-first split logic. Root cause: Emma's combined `build_dataset.py` loads ~780 total PoetMT poems vs our ~725, producing a larger test slice.

**Canonical test set for paper Table 1:** Emma's 78-poem flat-format `results/test.jsonl`. All E1 variants are already evaluated on it. E2/opus-mt should be re-evaluated on this set. See `split_alignment_status.md` for details.

### E1+E2 Combined Dataset Statistics (latest run)

| Split | Total | Translation | Auxiliary (CCPM) |
|---|---|---|---|
| Train | 21,669 | 578 | 21,091 |
| Valid | 2,642 | 72 | 2,570 |
| Test | ~73 (carved deterministically) | ~73 | 0 |

- **Annotated (with 注释):** 578 train / 72 valid / 73 test
- **With 译文 (modern ZH from background):** ~261 train / ~29 valid / ~28 test

### Cleaning Filters Applied

- Length bounds (separate ranges for classical / English / modern Chinese)
- Language ratio checks — guards against swapped or garbled fields
- Deduplication on both source and target
- Modern paraphrase must be longer than classical line (CCPM sanity check)
- Reject identical source = target

### Open Items

- Could pull additional bilingual poetry from sources like NiuTrans Poetry to grow the parallel set
- Annotation ablation: train one model with `note` in prompt, one without, compare BLEU/BERTScore
