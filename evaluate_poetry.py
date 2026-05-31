"""
evaluate_poetry.py
──────────────────
Automatic + human evaluation of poetry translation models on the PoetMT test set.

Automatic metrics  : BLEU-4, ROUGE-L, BERTScore
Human evaluation   : Adequacy / Fluency / Poeticness on a 1–5 Likert scale
                     (15 poems sampled to cover Tang, Song, and Yuan dynasties)

Experiments supported
─────────────────────
  E0           Qwen2.5-1.5B base (no adapter)
  E0_05b       Qwen2.5-0.5B base (no adapter)
  E1           Qwen2.5-1.5B + QLoRA
  E1_lora      Qwen2.5-1.5B + LoRA (full precision)
  E1_05b       Qwen2.5-0.5B + QLoRA
  E1_05b_lora  Qwen2.5-0.5B + LoRA (full precision)
  E2           mT5-base + QLoRA
  E3           Qwen2.5-14B + QLoRA (scale reference)
  E4           Best E1/E2 + CCPM aux loss

Usage
─────
  pip install transformers peft bitsandbytes sacrebleu rouge-score bert-score

  # Baselines (no adapter needed)
    python evaluate_poetry.py --test_file ./results/test.jsonl --exp_id E0              # Qwen 1.5B base
    python evaluate_poetry.py --test_file ./results/test.jsonl --exp_id E0_05b          # Qwen 0.5B base

  # Fine-tuned variants
    python evaluate_poetry.py --test_file ./results/test.jsonl --exp_id E1     --adapter ./qwen15b_qlora    # 1.5B QLoRA
    python evaluate_poetry.py --test_file ./results/test.jsonl --exp_id E1_lora     --adapter ./qwen15b_lora     # 1.5B LoRA
    python evaluate_poetry.py --test_file ./results/test.jsonl --exp_id E1_05b      --adapter ./qwen05b_qlora    # 0.5B QLoRA
    python evaluate_poetry.py --test_file ./results/test.jsonl --exp_id E1_05b_lora --adapter ./qwen05b_lora     # 0.5B LoRA
  
  # Evaluate a single fine-tuned model
  python evaluate_poetry.py \\
      --test_file  ./qwen_poetry_lora/test.jsonl \\
      --adapter    ./qwen_poetry_lora \\
      --exp_id     E1 \\
      --output_dir ./results

  # Compare two models side-by-side (E1 vs E2)
  python evaluate_poetry.py \\
      --test_file  ./qwen_poetry_lora/test.jsonl \\
      --adapter    ./qwen_poetry_lora \\
      --exp_id     E1 \\
      --compare_file ./results/E2_predictions.jsonl \\
      --output_dir ./results

  # Run only the human evaluation CLI (no GPU needed)
  python evaluate_poetry.py \\
      --test_file  ./qwen_poetry_lora/test.jsonl \\
      --predictions_file ./results/E1_predictions.jsonl \\
      --human_only \\
      --output_dir ./results
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Pastel colours (matches poetry_ui.py)
# ──────────────────────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
def _fg(n): return f"\033[38;5;{n}m"
LAVENDER = _fg(183)
PEACH    = _fg(223)
SAGE     = _fg(151)
ROSE     = _fg(218)
SKY      = _fg(153)
SAND     = _fg(187)
BLUSH    = _fg(210)
def c(text, *codes): return "".join(codes) + str(text) + RESET

WIDTH = 72

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a literary translator specialising in classical Chinese poetry. "
    "Translate the given Chinese poem into elegant English verse, preserving "
    "its meaning, imagery, and poetic quality."
)

DYNASTY_MAP = {"唐代": "Tang", "宋代": "Song", "元代": "Yuan"}


def load_test_set(path: str) -> list[dict]:
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    print(c(f"  Loaded {len(pairs)} test poems from {path}", SAND))
    return pairs


def sample_human_eval_poems(test_pairs: list[dict], n: int = 15,
                             seed: int = 42) -> list[dict]:
    """
    Sample n poems for human evaluation, covering all available dynasties.
    Tries to spread evenly; falls back to random if dynasties are uneven.
    """
    rng = random.Random(seed)

    by_dynasty: dict[str, list] = defaultdict(list)
    for p in test_pairs:
        tag = DYNASTY_MAP.get(p.get("dynasty", ""), p.get("dynasty", "unknown"))
        by_dynasty[tag].append(p)

    dynasties = sorted(by_dynasty.keys())
    per = n // len(dynasties)
    extra = n % len(dynasties)

    sampled = []
    for i, d in enumerate(dynasties):
        take = per + (1 if i < extra else 0)
        pool = by_dynasty[d]
        rng.shuffle(pool)
        sampled.extend(pool[:take])

    # If any dynasty had fewer poems than requested, top up randomly
    if len(sampled) < n:
        remaining = [p for p in test_pairs if p not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[: n - len(sampled)])

    return sampled[:n]


# ──────────────────────────────────────────────────────────────────────────────
# 2. INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(example: dict, tokenizer) -> str:
    title   = example.get("title",   "")
    author  = example.get("author",  "")
    dynasty = example.get("dynasty", "")
    parts   = []
    if title and author and dynasty:
        parts.append(f"Poem: \u300a{title}\u300b by {author} ({dynasty})")
    elif title and author:
        parts.append(f"Poem: \u300a{title}\u300b by {author}")
    parts.append(
        f"Translate the following classical Chinese poem into English:\n\n{example['chinese']}"
    )
    fanyi   = example.get("fanyi",   "")
    about   = example.get("about",   "")
    shangxi = example.get("shangxi", "")
    ctx = []
    if fanyi:   ctx.append(f"Modern Chinese paraphrase: {fanyi[:200].strip()}…")
    if about:   ctx.append(f"Composition background: {about[:150].strip()}…")
    if shangxi: ctx.append(f"Literary notes: {shangxi[:150].strip()}…")
    if ctx:
        parts.append("\nContext (for reference, do not translate):\n" + "\n".join(ctx))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "\n".join(parts)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_predictions(test_pairs: list[dict], model, tokenizer,
                          max_new_tokens: int = 256,
                          batch_size: int = 4) -> tuple[list[str], dict]:
    """
    Run batched inference over the full test set.
    Returns (predictions, perf_stats) where perf_stats contains:
      - inference_total_s   : wall time for all predictions
      - inference_per_poem_s: mean wall time per poem
      - inference_per_poem_ms: mean wall time per poem in ms
      - peak_vram_gb        : peak VRAM during inference
    """
    import torch, time
    predictions  = []
    total        = len(test_pairs)
    batch_times  = []   # wall time per batch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for start in range(0, total, batch_size):
        batch = test_pairs[start: start + batch_size]
        prompts = [build_prompt(p, tokenizer) for p in batch]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=768
        ).to(model.device)

        # Sync GPU before timing so we measure actual compute, not queue lag
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - t0, len(batch)))

        for output_ids in out:
            new_ids = output_ids[inputs["input_ids"].shape[-1]:]
            pred = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            predictions.append(pred)

        done = min(start + batch_size, total)
        sys.stdout.write(f"\r  Generating … {done}/{total}")
        sys.stdout.flush()

    print()

    total_s      = sum(t for t, _ in batch_times)
    per_poem_s   = sum(t for t, _ in batch_times) / total if total else 0
    vram_bytes   = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0

    perf_stats = {
        "inference_total_s":    round(total_s, 3),
        "inference_per_poem_s": round(per_poem_s, 4),
        "inference_per_poem_ms":round(per_poem_s * 1000, 2),
        "peak_vram_gb":         round(vram_bytes / 1024 ** 3, 2),
        "n_poems":              total,
    }
    return predictions, perf_stats


def load_model(model_name: str, adapter_path: str | None,
               hf_token: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto",
        token=hf_token, trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for batched generation
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# 3. AUTOMATIC METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    """BLEU-4 via sacrebleu (corpus-level)."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU(max_ngram_order=4)
    result = bleu.corpus_score(predictions, [references])
    return {
        "bleu4":      round(result.score, 4),
        "bp":         round(result.bp,    4),
        "brevity":    round(result.sys_len / result.ref_len, 4) if result.ref_len else 0,
    }


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """ROUGE-L via rouge-score."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred)["rougeL"]
              for pred, ref in zip(predictions, references)]
    n = len(scores)
    return {
        "rougeL_f":  round(sum(s.fmeasure  for s in scores) / n, 4),
        "rougeL_p":  round(sum(s.precision for s in scores) / n, 4),
        "rougeL_r":  round(sum(s.recall    for s in scores) / n, 4),
    }


def compute_bertscore(predictions: list[str],
                      references:  list[str]) -> dict:
    """
    BERTScore using roberta-large (layer 17), the standard model for English
    MT evaluation.

    Two known issues with roberta-large and bert_score, both fixed here:

    1. UNEXPECTED key warnings (lm_head, pooler) — harmless. bert_score only
       uses encoder layers for embeddings; extra heads are ignored.

    2. OverflowError: int too big to convert — the roberta-large *fast*
       tokenizer stores its max_length as a ~1e28 sentinel value that
       overflows the Rust tokenizer backend when bert_score tries to set
       truncation. Fix: use_fast_tokenizer=False forces the slow (pure
       Python) tokenizer which has no such sentinel.
    """
    from bert_score import score as bscore
    P, R, F = bscore(
        predictions, references,
        lang="en",
        model_type="roberta-large",
        num_layers=17,
        use_fast_tokenizer=False,   # avoids Rust overflow on max_length sentinel
        verbose=False,
        batch_size=8,
    )
    return {
        "bertscore_f":  round(F.mean().item(),  4),
        "bertscore_p":  round(P.mean().item(),  4),
        "bertscore_r":  round(R.mean().item(),  4),
    }


def run_automatic_metrics(predictions: list[str],
                           references:  list[str],
                           exp_id: str) -> dict:
    print(c(f"\n  Computing BLEU-4 …", SAND))
    bleu = compute_bleu(predictions, references)

    print(c(f"  Computing ROUGE-L …", SAND))
    rouge = compute_rouge(predictions, references)

    print(c(f"  Computing BERTScore …", SAND))
    bert = compute_bertscore(predictions, references)

    results = {"exp_id": exp_id, **bleu, **rouge, **bert}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. HUMAN EVALUATION CLI
# ──────────────────────────────────────────────────────────────────────────────

HUMAN_CRITERIA = ["adequacy", "fluency", "poeticness"]
CRITERIA_DESC  = {
    "adequacy":   "Does the translation preserve the meaning of the original?",
    "fluency":    "Does the English read naturally and grammatically?",
    "poeticness": "Does the translation feel like a poem (imagery, rhythm, elegance)?",
}


def _divider(label=""):
    if label:
        pad = (WIDTH - len(label) - 2) // 2
        print(c("·" * pad + f" {label} " + "·" * pad, LAVENDER))
    else:
        print(c("·" * WIDTH, SAND))


def _ask_score(criterion: str) -> int:
    desc = CRITERIA_DESC[criterion]
    while True:
        try:
            raw = input(
                c(f"  {criterion.capitalize():12s}", PEACH + BOLD) +
                c(f"  {desc}\n  Score (1–5): ", SAND)
            ).strip()
            val = int(raw)
            if 1 <= val <= 5:
                return val
            print(c("  Please enter a number between 1 and 5.", BLUSH))
        except ValueError:
            print(c("  Please enter a number between 1 and 5.", BLUSH))


def run_human_evaluation(human_poems:  list[dict],
                         predictions:  list[str],
                         exp_id:       str,
                         output_dir:   Path) -> list[dict]:
    """
    Interactive CLI for rating 15 poems on Adequacy / Fluency / Poeticness.
    Ratings are saved after each poem so progress is never lost.
    """
    save_path = output_dir / f"{exp_id}_human_eval.jsonl"
    # Resume: skip already-rated poems
    rated_titles: set[str] = set()
    if save_path.exists():
        for line in save_path.read_text().splitlines():
            if line.strip():
                rated_titles.add(json.loads(line)["title"])

    print(c("\n" + "·" * WIDTH, LAVENDER))
    print(c(f"  Human Evaluation — {exp_id}  ({len(human_poems)} poems)", LAVENDER + BOLD))
    print(c("  Rate each poem 1 (poor) → 5 (excellent) on three criteria.", SAND))
    print(c("·" * WIDTH, LAVENDER))

    all_ratings = []
    for idx, (poem, pred) in enumerate(zip(human_poems, predictions), 1):
        title = poem.get("title", f"Poem {idx}")
        if title in rated_titles:
            print(c(f"  [{idx:02d}/{len(human_poems)}] {title} — already rated, skipping.", DIM))
            continue

        dynasty_en = DYNASTY_MAP.get(poem.get("dynasty", ""), poem.get("dynasty", ""))
        print()
        _divider(f"{idx:02d} / {len(human_poems)}  {title}  {dynasty_en}")

        # Show original + reference + prediction
        print(c("  Original", SAND + BOLD))
        print(f"  {poem['chinese']}\n")

        print(c("  Reference translation", SAND + BOLD))
        print(f"  {poem['english']}\n")

        print(c(f"  Model translation  [{exp_id}]", SAGE + BOLD))
        print(f"  {pred}\n")

        _divider("Rate this translation")
        scores = {crit: _ask_score(crit) for crit in HUMAN_CRITERIA}
        avg = round(sum(scores.values()) / len(scores), 2)

        record = {
            "exp_id":  exp_id,
            "idx":     idx,
            "title":   title,
            "author":  poem.get("author", ""),
            "dynasty": poem.get("dynasty", ""),
            **scores,
            "average": avg,
        }
        all_ratings.append(record)
        rated_titles.add(title)

        # Append immediately so progress survives interruptions
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(c(f"\n  Saved  ·  average this poem: {avg:.2f}", SAGE))

    # Summary
    if all_ratings:
        print(c("\n" + "·" * WIDTH, LAVENDER))
        print(c(f"  Human Evaluation Summary — {exp_id}", LAVENDER + BOLD))
        _divider()
        for crit in HUMAN_CRITERIA:
            vals = [r[crit] for r in all_ratings]
            print(c(f"  {crit.capitalize():12s}", PEACH) +
                  f"  mean={sum(vals)/len(vals):.2f}  "
                  f"min={min(vals)}  max={max(vals)}")
        avgs = [r["average"] for r in all_ratings]
        print(c(f"\n  Overall average:  {sum(avgs)/len(avgs):.2f}", SAGE + BOLD))
        print(c("·" * WIDTH, LAVENDER))

    return all_ratings


# ──────────────────────────────────────────────────────────────────────────────
# 5. RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]):
    """Pretty-print a side-by-side comparison table."""
    header = f"{'Exp':6s}  {'BLEU-4':>8s}  {'ROUGE-L':>8s}  {'BERTScore':>10s}"
    print(c("\n" + "·" * WIDTH, LAVENDER))
    print(c("  Automatic Metrics Summary", LAVENDER + BOLD))
    _divider_fn = lambda: print(c("·" * WIDTH, SAND))
    _divider_fn()
    print(c("  " + header, SAND + BOLD))
    _divider_fn()
    for r in results:
        row = (f"  {r['exp_id']:6s}  "
               f"{r['bleu4']:8.4f}  "
               f"{r['rougeL_f']:8.4f}  "
               f"{r['bertscore_f']:10.4f}")
        print(c(row, SAGE if r.get("best") else PEACH))
    _divider_fn()


def save_results(results: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(c(f"  Results saved → {path}", SAND))


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate poetry translation: BLEU-4, ROUGE-L, BERTScore + human eval"
    )
    parser.add_argument("--test_file",        required=True,
                        help="Path to test.jsonl (saved by train_qwen_poetry.py)")
    parser.add_argument("--exp_id",           default="E1",
                        choices=["E0", "E0_05b", "E1", "E1_lora", "E1_05b", "E1_05b_lora",
                                 "E2", "E3", "E4"],
                        help=(
                            "Experiment ID:\n"
                            "  E0           Qwen2.5-1.5B base (no adapter)\n"
                            "  E0_05b       Qwen2.5-0.5B base (no adapter)\n"
                            "  E1           Qwen2.5-1.5B + QLoRA\n"
                            "  E1_lora      Qwen2.5-1.5B + LoRA (full precision)\n"
                            "  E1_05b       Qwen2.5-0.5B + QLoRA\n"
                            "  E1_05b_lora  Qwen2.5-0.5B + LoRA (full precision)\n"
                            "  E2           mT5-base + QLoRA\n"
                            "  E3           Qwen2.5-14B + QLoRA (scale reference)\n"
                            "  E4           Best E1/E2 + CCPM aux loss"
                        ))
    parser.add_argument("--output_dir",       default="./results")

    # Model (skip if --human_only or --predictions_file given)
    parser.add_argument("--model_name",       default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter",          default=None,
                        help="Path to QLoRA adapter directory")
    parser.add_argument("--max_new_tokens",   type=int, default=256)
    parser.add_argument("--batch_size",       type=int, default=4)
    parser.add_argument("--hf_token",         default=os.environ.get("HF_TOKEN"))

    # Pre-computed predictions (skip generation)
    parser.add_argument("--predictions_file", default=None,
                        help="JSONL file with {title, prediction} — skips generation")

    # Comparison
    parser.add_argument("--compare_file",     default=None,
                        help="Predictions JSONL for a second model (for E1 vs E2 table)")

    # Flags
    parser.add_argument("--skip_auto",        action="store_true",
                        help="Skip automatic metrics, run human eval only")
    parser.add_argument("--human_only",       action="store_true",
                        help="Alias for --skip_auto (no GPU needed)")
    parser.add_argument("--human_n",          type=int, default=15,
                        help="Number of poems for human evaluation")
    parser.add_argument("--skip_human",       action="store_true",
                        help="Skip human evaluation")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    skip_auto = args.skip_auto or args.human_only
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Resolve model + adapter from exp_id ───────────────────────────────────
    # Each entry: (model_name, use_adapter)
    # use_adapter=True  → load --adapter path (error if not provided)
    # use_adapter=False → base model, ignore --adapter
    EXP_CONFIGS = {
        "E0":          ("Qwen/Qwen2.5-1.5B-Instruct", False),
        "E0_05b":      ("Qwen/Qwen2.5-0.5B-Instruct", False),
        "E1":          ("Qwen/Qwen2.5-1.5B-Instruct", True),
        "E1_lora":     ("Qwen/Qwen2.5-1.5B-Instruct", True),
        "E1_05b":      ("Qwen/Qwen2.5-0.5B-Instruct", True),
        "E1_05b_lora": ("Qwen/Qwen2.5-0.5B-Instruct", True),
        "E2":          ("google/mt5-base",              True),
        "E3":          ("Qwen/Qwen2.5-14B-Instruct",   True),
        "E4":          ("Qwen/Qwen2.5-1.5B-Instruct",  True),
    }

    cfg_model, cfg_use_adapter = EXP_CONFIGS[args.exp_id]

    # --model_name overrides the default for the experiment if explicitly passed
    resolved_model = args.model_name if args.model_name != "Qwen/Qwen2.5-1.5B-Instruct"                      else cfg_model

    if cfg_use_adapter and not args.adapter and not args.predictions_file             and not skip_auto and not pred_path.exists():
        print(c(f"  ✗ {args.exp_id} requires --adapter <path>", BLUSH))
        sys.exit(1)

    resolved_adapter = args.adapter if cfg_use_adapter else None
    if not cfg_use_adapter and args.adapter:
        print(c(f"  {args.exp_id} is a base model — ignoring --adapter.", SAND))

    # ── Load test set ──────────────────────────────────────────────────────────
    print(c("\n" + "·" * WIDTH, LAVENDER))
    print(c(f"  Poetry Evaluation  ·  {args.exp_id}", LAVENDER + BOLD))
    print(c(f"  Model  : {resolved_model}", SAND))
    print(c(f"  Adapter: {resolved_adapter or '(none — base model)'}", SAND))
    print(c("·" * WIDTH, LAVENDER))

    test_pairs = load_test_set(args.test_file)
    references = [p["english"] for p in test_pairs]

    # ── Get predictions ────────────────────────────────────────────────────────
    pred_path = out / f"{args.exp_id}_predictions.jsonl"

    if args.predictions_file:
        # Load pre-computed
        preds_raw = [json.loads(l) for l in
                     Path(args.predictions_file).read_text().splitlines() if l.strip()]
        predictions = [p["prediction"] for p in preds_raw]
        print(c(f"  Loaded {len(predictions)} pre-computed predictions.", SAND))

    elif pred_path.exists() and not skip_auto:
        # Resume from previous run
        preds_raw = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]
        predictions = [p["prediction"] for p in preds_raw]
        print(c(f"  Resuming: loaded {len(predictions)} existing predictions.", SAND))

    elif not skip_auto:
        print(c(f"\n[1/3] Generating predictions …", LAVENDER + BOLD))
        if cfg_use_adapter and not resolved_adapter:
            print(c(f"  ✗ {args.exp_id} requires --adapter <path>", BLUSH))
            sys.exit(1)
        model, tokenizer = load_model(resolved_model, resolved_adapter, args.hf_token)
        predictions, infer_perf = generate_predictions(
            test_pairs, model, tokenizer, args.max_new_tokens, args.batch_size
        )
        print(c(f"  Inference time : {infer_perf['inference_total_s']:.1f}s total  |  "
                f"{infer_perf['inference_per_poem_ms']:.1f} ms/poem", SAND))
        print(c(f"  Peak VRAM      : {infer_perf['peak_vram_gb']:.2f} GB", SAND))

        # Save perf stats alongside predictions
        import json as _json
        _json.dump(infer_perf, open(out / f"{args.exp_id}_infer_perf.json", "w"), indent=2)

        # Save predictions
        with open(pred_path, "w", encoding="utf-8") as f:
            for poem, pred in zip(test_pairs, predictions):
                f.write(json.dumps({
                    "title":      poem.get("title", ""),
                    "author":     poem.get("author", ""),
                    "dynasty":    poem.get("dynasty", ""),
                    "chinese":    poem["chinese"],
                    "reference":  poem["english"],
                    "prediction": pred,
                }, ensure_ascii=False) + "\n")
        print(c(f"  Predictions saved → {pred_path}", SAND))
    else:
        predictions = [""] * len(test_pairs)   # dummy for human-only mode

    # ── Automatic metrics ──────────────────────────────────────────────────────
    all_auto_results = []

    if not skip_auto:
        print(c(f"\n[2/3] Automatic metrics …", LAVENDER + BOLD))
        auto = run_automatic_metrics(predictions, references, args.exp_id)
        all_auto_results.append(auto)

        # Optional comparison model
        if args.compare_file:
            cmp_preds_raw = [json.loads(l) for l in
                             Path(args.compare_file).read_text().splitlines() if l.strip()]
            cmp_predictions = [p["prediction"] for p in cmp_preds_raw]
            cmp_id = Path(args.compare_file).stem.split("_")[0]
            cmp_auto = run_automatic_metrics(cmp_predictions, references, cmp_id)
            all_auto_results.append(cmp_auto)

            # Mark the better BERTScore for the table
            best = max(all_auto_results, key=lambda r: r["bertscore_f"])
            best["best"] = True

        print_results_table(all_auto_results)

        # Attach inference perf to saved results if we just ran generation
        perf_to_save = infer_perf if "infer_perf" in dir() else {}
        save_results(
            {"automatic": all_auto_results, "inference_perf": perf_to_save},
            out / f"{args.exp_id}_auto_metrics.json"
        )

    # ── Human evaluation ───────────────────────────────────────────────────────
    if not args.skip_human:
        print(c(f"\n[3/3] Human evaluation …", LAVENDER + BOLD))
        human_poems = sample_human_eval_poems(test_pairs, n=args.human_n, seed=args.seed)

        # Match predictions to human subset by index in test_pairs
        title_to_pred = dict(zip(
            [p.get("title", "") for p in test_pairs], predictions
        ))
        human_predictions = [
            title_to_pred.get(p.get("title", ""), "(no prediction)") for p in human_poems
        ]

        human_ratings = run_human_evaluation(
            human_poems, human_predictions, args.exp_id, out
        )

        if human_ratings:
            summary = {
                crit: round(sum(r[crit] for r in human_ratings) / len(human_ratings), 3)
                for crit in HUMAN_CRITERIA
            }
            summary["average"] = round(
                sum(summary.values()) / len(HUMAN_CRITERIA), 3
            )
            save_results(
                {"human_eval_summary": summary, "ratings": human_ratings},
                out / f"{args.exp_id}_human_eval_summary.json"
            )

    print(c(f"\n  Done. All results saved to {args.output_dir}/\n", SAGE + BOLD))


if __name__ == "__main__":
    main()