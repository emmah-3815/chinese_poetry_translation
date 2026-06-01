# Dataset Split Alignment Status

## Finding

Both `build_dataset_poetmt.py` (Juqy/E2) and `build_dataset.py` (Emma/E1) use **identical split logic**:
- `random.seed(42)`
- Test carved from tail: `records[-test_size:]` before any shuffle
- `test_size = max(50, int(n * 0.1))`, capped at `n // 5`

However, they load a **different total number of PoetMT poems**, producing different test sizes:

| Partner | Script | Total PoetMT | Test Size | Format |
|---------|--------|:---:|:---:|--------|
| Juqy (E2) | `build_dataset_poetmt.py` | 725 | **72** | chat `messages` |
| Emma (E1) | `build_dataset.py` | 780 | **78** | flat `{chinese, english, ...}` |

The 55-poem discrepancy comes from different filtering/enrichment logic in the two builders. Emma's combined builder includes poems that Juqy's compact builder filtered or compacted away.

## Impact

Emma has **already evaluated all E1 variants** (E0, E0_05b, E1, E1_lora, E1_05b, E1_05b_lora) on her 78-poem test set. Re-running all those experiments would be wasteful.

## Recommendation: Adopt Emma's test set as canonical

1. **Emma:** No changes needed. Her `results/test.jsonl` (78 poems, flat format) is the reference test set.

2. **Juqy (E2):** Re-evaluate the mT5/opus-mt models on Emma's 78-poem test set.
   - The test.jsonl is available at `results/test.jsonl` on the `qwen` branch.
   - `eval_e2_mt5.py` currently expects chat-messages format. Two options:
     - **Option A (recommended):** Add a `--flat_test` mode to `eval_e2_mt5.py` that loads Emma's flat format directly.
     - **Option B:** Run `build_dataset.py` (Emma's script) to produce an 780-poem messages-format dataset, then use the last 78 as test.

3. **Paper Table 1:** Report E1 and E2 metrics both on the 78-poem canonical test set.

## Current Status

- [ ] E2 re-evaluated on 78-poem canonical test set
- [x] E1 evaluated on 78-poem test set (Emma's qwen branch, all variants complete)
- [x] Cross-eval confirmed: `eval_e2_mt5.py` and `evaluate_poetry.py` produce identical metrics on same predictions (see `models/e2-mt5-fp32-v2/cross_eval/cross_eval_comparison.md`)

## Emma's E1 Results (for reference)

From `results/E1_auto_metrics.json` on `qwen` branch:

| Metric | E1 (Qwen2.5-1.5B + QLoRA) |
|--------|:---:|
| BLEU-4 | 2.86 |
| ROUGE-L F | 0.2151 |
| BERTScore F | 0.8643 |

Compare to E2 mT5-base (72-poem test, old dataset): BLEU-4=0.20, ROUGE-L=0.087, BERTScore-F=0.807.
E2 needs re-evaluation on clean compact dataset + 78-poem canonical test.
