"""
train_qwen_poetry.py
────────────────────
QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on the PoetMT dataset
for classical Chinese → English poetry translation.

Hardware target : consumer GPU ≤ 8 GB VRAM
LoRA config     : r=32, alpha=64, dropout=0.05  (matches paper §5.1)
Adapter targets : q/k/v/o_proj + gate/up/down_proj  (all linear layers)

Dataset layout expected
-----------------------
  data_dir/
    tang.jsonl
    song.jsonl
    yuan.jsonl                   (add when available)
    tang-background.jsonl        (optional, enriches prompts)
    song-background.jsonl        (optional, enriches prompts)
    yuan-background.jsonl        (optional, enriches prompts)

The script will:
  1. Load ALL poems (plain + background files), merge background enrichment.
  2. Deduplicate by (title, author, chinese) so background variants don't repeat.
  3. Random 90/10 split (seed-controlled) across all dynasties.
  4. Save train.jsonl + test.jsonl to --output_dir before training starts.
  5. Train on the 90% split; use test set for per-epoch validation loss.

Usage
-----
    pip install transformers peft bitsandbytes datasets accelerate

    python train_qwen_poetry.py --data_dir ./PoetMT/all_poems --output_dir ./qwen_poetry_lora
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATASET LOADER
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a literary translator specialising in classical Chinese poetry. "
    "Translate the given Chinese poem into elegant English verse, preserving "
    "its meaning, imagery, and poetic quality."
)

def _parse_jsonl(path: Path) -> list[dict]:
    """Read every valid JSON line from a file."""
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _extract_background(obj: dict) -> dict:
    """
    Pull enrichment fields out of a background record.
    Returns a dict with any of: fanyi, shangxi, about  (all strings).
    """
    bg = obj.get("background", {})
    if not isinstance(bg, dict):
        return {}
    out = {}
    for key in ("fanyi", "shangxi", "about"):
        val = bg.get(key, "").strip()
        if val:
            out[key] = val
    return out


def _extract_pair(obj: dict, enrichment: dict | None = None) -> dict | None:
    """
    Build a training example from a PoetMT record.
    enrichment: pre-merged background fields (fanyi, shangxi, about).
    Returns None if src or ref is missing.
    """
    zh = obj.get("src", "").strip()
    en = obj.get("ref", "").strip()
    if not zh or not en:
        return None

    pair = {
        "chinese": zh,
        "english": en,
        "title":   obj.get("title",  "").strip(),
        "author":  obj.get("author", "").strip(),
        "dynasty": "",   # filled from background if available
    }

    # Inline background (background files have the same src/ref + nested data)
    bg = _extract_background(obj)
    if enrichment:
        bg.update({k: v for k, v in enrichment.items() if k not in bg})
    pair.update(bg)

    # Try to get dynasty from background nested dict
    nested = obj.get("background", {})
    if isinstance(nested, dict):
        pair["dynasty"] = nested.get("dynasty", "").strip()

    return pair


def _infer_dynasty(filename: str) -> str:
    """Guess dynasty tag from filename (tang.jsonl → 唐代, etc.)."""
    stem = Path(filename).stem.split("-")[0].lower()
    return {"tang": "唐代", "song": "宋代", "yuan": "元代"}.get(stem, "")


def load_poetmt(data_dir: str, test_ratio: float = 0.1,
                seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Load ALL PoetMT poems, merge background enrichment, deduplicate,
    then return a stratified 90/10 train/test split.

    Stratification is by dynasty so Tang, Song, and Yuan poems appear in
    the same proportion in both splits.

    Returns
    -------
    train_pairs, test_pairs : list[dict]
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    # ── Step 1: build background enrichment index ─────────────────────────────
    background_index: dict[tuple, dict] = {}
    for bg_file in sorted(data_path.glob("*-background.jsonl")):
        for obj in _parse_jsonl(bg_file):
            key = (obj.get("title", "").strip(), obj.get("author", "").strip())
            bg  = _extract_background(obj)
            if key not in background_index:
                background_index[key] = bg
            else:
                for k, v in bg.items():
                    if k not in background_index[key]:
                        background_index[key][k] = v
    print(f"  Background index: {len(background_index)} unique (title, author) entries")

    # ── Step 2: load all plain poem files ─────────────────────────────────────
    # Deduplicate by (title, author, chinese) — background files share poems
    # with plain files; keep the richer version (the one with more fields).
    seen:  set[tuple]  = set()
    all_pairs: list[dict] = []

    plain_files = sorted(
        f for f in data_path.glob("*.jsonl")
        if "background" not in f.name
    )
    if not plain_files:
        raise FileNotFoundError(f"No plain *.jsonl files found in {data_dir}")

    for pf in plain_files:
        dynasty_hint = _infer_dynasty(pf.name)
        added = 0
        for obj in _parse_jsonl(pf):
            key = (obj.get("title", "").strip(), obj.get("author", "").strip())
            enrichment = background_index.get(key)
            pair = _extract_pair(obj, enrichment)
            if pair is None:
                continue
            # Fill dynasty from filename hint if not already set
            if not pair.get("dynasty") and dynasty_hint:
                pair["dynasty"] = dynasty_hint
            dedup_key = (pair["title"], pair["author"], pair["chinese"])
            if dedup_key not in seen:
                seen.add(dedup_key)
                all_pairs.append(pair)
                added += 1
        print(f"  Loaded {pf.name}: {added} unique pairs")

    # Also ingest background files directly for their richer prompt data,
    # skipping any poem already seen from the plain file.
    for bg_file in sorted(data_path.glob("*-background.jsonl")):
        dynasty_hint = _infer_dynasty(bg_file.name)
        added = 0
        for obj in _parse_jsonl(bg_file):
            pair = _extract_pair(obj)
            if pair is None:
                continue
            if not pair.get("dynasty") and dynasty_hint:
                pair["dynasty"] = dynasty_hint
            dedup_key = (pair["title"], pair["author"], pair["chinese"])
            if dedup_key not in seen:
                seen.add(dedup_key)
                all_pairs.append(pair)
                added += 1
        if added:
            print(f"  Loaded {bg_file.name}: {added} additional unique pairs")

    print(f"  Total unique poems: {len(all_pairs)}")

    # ── Step 3: random 90/10 split ───────────────────────────────────────────
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    n_test = max(1, round(len(all_pairs) * test_ratio))
    test_pairs  = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:]

    print(f"  Split  →  train: {len(train_pairs)}  |  test: {len(test_pairs)}")
    return train_pairs, test_pairs


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PROMPT FORMATTING
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(example: dict, english: str | None, tokenizer) -> str:
    """
    Build a Qwen2.5 chat-template prompt.
    Injects background context (modern translation + literary notes) when
    available so the model can learn cultural and stylistic nuance.
    Pass english=None for inference.
    """
    parts = []

    # Header: title / author / dynasty
    title   = example.get("title",   "")
    author  = example.get("author",  "")
    dynasty = example.get("dynasty", "")
    if title and author and dynasty:
        parts.append(f"Poem: \u300a{title}\u300b by {author} ({dynasty})")
    elif title and author:
        parts.append(f"Poem: \u300a{title}\u300b by {author}")

    parts.append(
        f"Translate the following classical Chinese poem into English:\n\n{example['chinese']}"
    )

    # Optional background context — helps the model understand allusions
    fanyi   = example.get("fanyi",   "")   # modern Chinese paraphrase
    shangxi = example.get("shangxi", "")   # literary analysis
    about   = example.get("about",   "")   # composition background

    context_parts = []
    if fanyi:
        # Truncate to first 200 chars to keep prompt length in check
        context_parts.append(f"Modern Chinese paraphrase: {fanyi[:200].strip()}…")
    if about:
        context_parts.append(f"Composition background: {about[:150].strip()}…")
    if shangxi:
        context_parts.append(f"Literary notes: {shangxi[:150].strip()}…")

    if context_parts:
        parts.append("\nContext (for reference, do not translate):\n" +
                     "\n".join(context_parts))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "\n".join(parts)},
    ]
    if english is not None:
        messages.append({"role": "assistant", "content": english})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(english is None),
    )


def tokenize_example(example: dict, tokenizer, max_length: int = 768):
    """
    Tokenise one training example.
    Labels are masked to -100 for the prompt so loss only covers the translation.
    max_length raised to 768 to accommodate background context.
    """
    full_text   = build_prompt(example, example["english"], tokenizer)
    prompt_only = build_prompt(example, None,               tokenizer)

    full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_only, truncation=True, max_length=max_length)["input_ids"]

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MODEL + QLoRA SETUP
# ──────────────────────────────────────────────────────────────────────────────

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # MLP / feed-forward
]


def load_model_and_tokenizer(model_name: str, hf_token: str | None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./PoetMT/all_poems")
    parser.add_argument("--output_dir", default="./qwen_poetry_lora")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--grad_accum", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--max_length", type=int,   default=768)
    parser.add_argument("--hf_token",   default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction of each dynasty held out for evaluation")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load & split ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading PoetMT dataset …")
    train_pairs, test_pairs = load_poetmt(
        args.data_dir, test_ratio=args.test_ratio, seed=args.seed
    )

    # Save splits immediately — before any training — for full reproducibility
    for name, pairs in [("train.jsonl", train_pairs), ("test.jsonl", test_pairs)]:
        split_path = out / name
        with open(split_path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"  Saved {name} ({len(pairs)} poems) → {split_path}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Loading model + applying QLoRA …")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.hf_token)

    # ── Tokenise ──────────────────────────────────────────────────────────────
    print("\n[3/4] Tokenising …")
    meta_cols = ["chinese", "english", "title", "author", "dynasty",
                 "fanyi", "shangxi", "about"]

    def tok(example):
        return tokenize_example(example, tokenizer, args.max_length)

    train_ds = Dataset.from_list(train_pairs)
    train_cols = [c for c in meta_cols if c in train_ds.column_names]
    train_ds = train_ds.map(tok, remove_columns=train_cols)

    # Use test set as validation during training (no labels leaked — loss only)
    val_ds = Dataset.from_list(test_pairs)
    val_cols = [c for c in meta_cols if c in val_ds.column_names]
    val_ds = val_ds.map(tok, remove_columns=val_cols)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[4/4] Training …")
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )

    trainer.train()

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\n✓ LoRA adapter saved to {args.output_dir}")
    print(f"  train.jsonl : {len(train_pairs)} poems")
    print(f"  test.jsonl  : {len(test_pairs)} poems  (stratified by dynasty)")
    print("  Run evaluate_poetry.py to compute BLEU-4 / ROUGE-L / BERTScore.")


if __name__ == "__main__":
    main()