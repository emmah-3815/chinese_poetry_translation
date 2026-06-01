"""
E2 evaluation: mT5-base (baseline or + LoRA) on the PoetMT test set.

Outputs per run:
  <output_dir>/metrics.json      — BLEU-4, ROUGE-L, BERTScore F1
  <output_dir>/predictions.jsonl — source / reference / hypothesis per sample

Usage:
  # Fine-tuned adapter
  python eval_e2_mt5.py --adapter_dir models/e2-mt5-compact-v1/lora_adapter

  # Pretrained baseline (no adapter)
  python eval_e2_mt5.py --baseline --output_dir models/mt5-base-baseline/eval_results
"""

import json
import sys
import argparse
from pathlib import Path

import torch

# Windows consoles default to GBK; force UTF-8 so Chinese source text prints cleanly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import evaluate

TASK_PREFIX  = "translate classical Chinese poetry to English: "
MAX_SRC_LEN  = 512
MAX_TGT_LEN  = 256
BEAM_SIZE    = 4
BASE_MODEL   = "google/mt5-base"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def messages_to_pair(record: dict) -> dict | None:
    if record.get("task") not in (None, "translation"):
        return None
    msgs = record.get("messages", [])
    src  = next((m["content"] for m in msgs if m["role"] == "user"),      None)
    tgt  = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    if not src or not tgt:
        return None
    return {"source": TASK_PREFIX + src, "target": tgt, "raw_source": src}


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_translations(model, tokenizer, sources: list[str], batch_size: int = 8) -> list[str]:
    model.eval()
    device = next(model.parameters()).device
    results = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=MAX_SRC_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=MAX_TGT_LEN,
                num_beams=BEAM_SIZE,
                early_stopping=True,
                bad_words_ids=[[250099]],  # block <extra_id_0>
            )
        results.extend(tokenizer.batch_decode(out_ids, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(sources))}/{len(sources)}", end="\r")
    print()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(hypotheses: list[str], references: list[str]) -> dict:
    sacrebleu = evaluate.load("sacrebleu")
    rouge     = evaluate.load("rouge")

    bleu_score = sacrebleu.compute(
        predictions=hypotheses,
        references=[[r] for r in references],
    )["score"]

    rouge_scores = rouge.compute(
        predictions=hypotheses,
        references=references,
    )

    metrics = {
        "bleu4":       round(bleu_score, 4),
        "rouge1":      round(rouge_scores["rouge1"], 4),
        "rouge2":      round(rouge_scores["rouge2"], 4),
        "rougeL":      round(rouge_scores["rougeL"], 4),
        "n_samples":   len(hypotheses),
    }
    try:
        bertscore = evaluate.load("bertscore")
        bs = bertscore.compute(
            predictions=hypotheses,
            references=references,
            lang="en",
            verbose=False,
        )
        metrics.update({
            "bertscore_p": round(sum(bs["precision"]) / len(bs["precision"]), 4),
            "bertscore_r": round(sum(bs["recall"])    / len(bs["recall"]),    4),
            "bertscore_f": round(sum(bs["f1"])        / len(bs["f1"]),        4),
        })
    except Exception as exc:
        print(f"[WARN] BERTScore unavailable: {exc}")
        print("[WARN] Install with: pip install bert_score")
        metrics.update({
            "bertscore_p": None,
            "bertscore_r": None,
            "bertscore_f": None,
        })
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    test_file = Path(args.test_file)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    if args.baseline:
        print(f"Loading pretrained baseline: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, dtype=dtype)
    else:
        adapter_dir = Path(args.adapter_dir)
        print(f"Loading adapter from {adapter_dir} ...")
        tokenizer  = AutoTokenizer.from_pretrained(str(adapter_dir))
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, dtype=dtype)
        model      = PeftModel.from_pretrained(base_model, str(adapter_dir))

    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Loading test set from {test_file} ...")
    records = [messages_to_pair(r) for r in load_jsonl(test_file)]
    records = [r for r in records if r]
    if not records:
        raise ValueError("No translation examples loaded from the test file.")
    sources    = [r["source"]     for r in records]
    references = [r["target"]     for r in records]
    raw_srcs   = [r["raw_source"] for r in records]
    print(f"  {len(records)} test samples")

    print("Generating translations ...")
    hypotheses = generate_translations(model, tokenizer, sources, batch_size=args.batch_size)

    print("\n── Sample Translations ───────────────────────────")
    for i in range(min(5, len(records))):
        print(f"\n[{i+1}] Source:     {raw_srcs[i]}")
        print(f"    Reference:  {references[i]}")
        print(f"    mT5 output: {hypotheses[i]}")
    print("\n─────────────────────────────────────────────────")

    preds_path = out_dir / "predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for src, hyp, ref in zip(raw_srcs, hypotheses, references):
            f.write(json.dumps({"source": src, "hypothesis": hyp, "reference": ref}, ensure_ascii=False) + "\n")
    print(f"Predictions saved -> {preds_path}")

    print("\nComputing metrics ...")
    metrics = compute_all_metrics(hypotheses, references)

    print("\n── Test Results ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")
    print("─────────────────────────────────────────────────\n")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",    action="store_true",
                        help="Evaluate raw mT5-base with no adapter (pretrained baseline)")
    parser.add_argument("--adapter_dir", default="models/e2-mt5-compact-v1/lora_adapter")
    parser.add_argument("--test_file",   default="data/poetmt_compact/test.jsonl")
    parser.add_argument("--output_dir",  default="models/e2-mt5-compact-v1/eval_results")
    parser.add_argument("--batch_size",  type=int, default=8)
    main(parser.parse_args())
