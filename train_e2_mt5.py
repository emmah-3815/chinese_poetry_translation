"""
E2: mT5-base + LoRA — classical Chinese poetry -> English translation.
Data: data/combined/{train,valid,test}.jsonl  (build_dataset.py output)

Usage:
  python train_e2_mt5.py
  python train_e2_mt5.py --data_dir data/combined --output_dir models/e2-mt5-combined --epochs 5
  python train_e2_mt5.py --precision bf16 --output_dir models/e2-mt5-bf16-combined
"""

import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME  = "google/mt5-base"   # overridden by --base_model arg
MAX_SRC_LEN = 512
MAX_TGT_LEN = 256
TASK_PREFIX = "translate classical Chinese to English: "

LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA ADAPTER  (chat-messages format -> seq2seq pairs)
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def messages_to_pair(record: dict) -> dict | None:
    if record.get("task") not in (None, "translation"):
        return None
    msgs     = record.get("messages", [])
    src_text = next((m["content"] for m in msgs if m["role"] == "user"),      None)
    tgt_text = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    if not src_text or not tgt_text:
        return None
    return {"source": TASK_PREFIX + src_text, "target": tgt_text}

def load_split(path: Path) -> Dataset:
    pairs = [messages_to_pair(r) for r in load_jsonl(path)]
    return Dataset.from_list([p for p in pairs if p])

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────────────────────────────────────

def make_tokenize_fn(tokenizer):
    def tokenize(batch):
        inputs = tokenizer(
            batch["source"],
            max_length=MAX_SRC_LEN,
            truncation=True,
            padding=False,
        )
        targets = tokenizer(
            text_target=batch["target"],
            max_length=MAX_TGT_LEN,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs
    return tokenize

# ─────────────────────────────────────────────────────────────────────────────
# METRICS  (BLEU-4)
# ─────────────────────────────────────────────────────────────────────────────

def make_compute_metrics(tokenizer):
    sacrebleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = [
            [tok if tok != -100 else tokenizer.pad_token_id for tok in seq]
            for seq in labels
        ]
        decoded_preds  = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        decoded_labels = [[l.strip()] for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]

        return {
            "bleu": round(sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)["score"], 2),
        }
    return compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    base_model_name = args.base_model
    use_bf16 = args.precision == "bf16"
    use_fp16 = args.precision == "fp16"

    if use_bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise ValueError(
            "This GPU does not report bfloat16 support. Use --precision fp32, "
            "or only try bf16 on Ampere/Ada/Hopper-class NVIDIA GPUs."
        )

    train_ds = load_split(data_dir / "train.jsonl")
    valid_ds = load_split(data_dir / "valid.jsonl")
    print(f"Train: {len(train_ds):,}  |  Valid: {len(valid_ds):,}")
    if len(train_ds) == 0 or len(valid_ds) == 0:
        raise ValueError("No translation examples loaded. Check --data_dir points to data/combined.")

    tokenizer  = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )
    model = get_peft_model(base_model, LORA_CONFIG)
    model.print_trainable_parameters()

    tok_fn    = make_tokenize_fn(tokenizer)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["source", "target"])
    valid_tok = valid_ds.map(tok_fn, batched=True, remove_columns=["source", "target"])

    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        warmup_steps=100,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=MAX_TGT_LEN,
        generation_num_beams=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    import time
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t_start

    # Peak VRAM via reserved bytes / GiB — matches the E1 (Qwen) measurement method
    peak_vram_gb = (
        round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        if torch.cuda.is_available() else None
    )
    per_epoch_s = elapsed / args.epochs
    print(f"Train time: {elapsed:.1f}s ({per_epoch_s:.1f}s/epoch) | Peak VRAM: {peak_vram_gb} GB")

    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    (output_dir / "train_perf.json").write_text(
        json.dumps({
            "model":          base_model_name,
            "training_s":     round(elapsed, 2),
            "per_epoch_s":    round(per_epoch_s, 2),
            "peak_vram_gb":   peak_vram_gb,
            "epochs":         args.epochs,
            "train_examples": len(train_ds),
        }, indent=2), encoding="utf-8"
    )
    print(f"\nAdapter saved -> {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  default=MODEL_NAME,
                        help="HF model ID or local path to merged Stage-1 model")
    parser.add_argument("--data_dir",   default="data/combined")
    parser.add_argument("--output_dir", default="models/e2-mt5-combined")
    parser.add_argument("--epochs",                   type=int,  default=5)
    parser.add_argument("--batch_size",               type=int,  default=8)
    parser.add_argument("--resume_from_checkpoint",   default=None)
    parser.add_argument("--early_stopping_patience",  type=int,  default=2)
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Training precision. bf16 is faster on supported GPUs; fp32 is safest.",
    )
    main(parser.parse_args())
