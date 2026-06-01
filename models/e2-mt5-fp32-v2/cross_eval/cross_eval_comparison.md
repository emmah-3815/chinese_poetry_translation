# Cross-Eval Consistency Check: eval_e2_mt5.py vs evaluate_poetry.py

## Setup

- Model: e2-mt5-fp32-v2 (mT5-base + LoRA, trained on PoetMT)
- Test set: data/poetmt_compact/test.jsonl (72 poems)
- Predictions: models/e2-mt5-fp32-v2/eval_results_final/predictions.jsonl

The predictions were produced by eval_e2_mt5.py (key: hypothesis).
convert_preds_for_poetry_eval.py converts them to evaluate_poetry.py format.
A row-alignment assertion verified all 72 references match.

---

## Results Comparison

| Metric       | eval_e2_mt5.py | evaluate_poetry.py | Delta |
|--------------|---------------:|-------------------:|------:|
| BLEU-4       | 0.2035         | 0.2035             | 0.0000 |
| ROUGE-L      | 0.0868         | 0.0868             | 0.0000 |
| BERTScore F1 | 0.8074         | 0.8074             | 0.0000 |

---

## Analysis

All three metrics are numerically identical (delta = 0.0 for each).

### BLEU-4

eval_e2_mt5.py: evaluate.load(sacrebleu) from HuggingFace evaluate library,
returns 0-100 scale. evaluate_poetry.py: sacrebleu.metrics.BLEU directly,
also 0-100. Both use the same 13a tokenizer and corpus-level scoring.
Result: identical.

### ROUGE-L

eval_e2_mt5.py: evaluate.load(rouge) uses BootstrapAggregator (mid percentile).
evaluate_poetry.py: arithmetic mean of per-sentence rouge_score F-measures.
For this 72-poem test set the two methods agree. Slight differences may arise
on larger/more unbalanced sets - if so, the cause is aggregation method, not a bug.

### BERTScore F1

eval_e2_mt5.py: evaluate.load(bertscore) with lang=en (roberta-large, layer 17).
evaluate_poetry.py: bert_score.score with same model/layer. Results identical.

---

## Conclusion

The cross-eval confirms:

1. Reference alignment is correct (all 72 rows match).
2. Both eval scripts produce identical BLEU-4, ROUGE-L, and BERTScore F1.
3. No systematic discrepancy from library wrappers, tokenization, or scaling.

Known eval_e2_mt5.py metrics: bleu4=0.2035, rougeL=0.0868, bertscore_f=0.8074
