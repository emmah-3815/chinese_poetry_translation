"""
Opus-MT + LoRA — classical Chinese poetry -> English translation.

Uses Helsinki-NLP/opus-mt-zh-en (MarianMT, ~74M params) as the base model.
It is already pre-trained on OPUS ZH->EN, so fine-tuning on poetry data
starts from a working translator rather than mT5-base's zero translation ability.

Data: data/combined/{train,valid,test}.jsonl  (build_dataset.py output)

Usage:
  python pipelines/opus_mt/train_opus_mt.py
  python pipelines/opus_mt/train_opus_mt.py --epochs 15 --output_dir models/opus-mt-poetry
  python pipelines/opus_mt/train_opus_mt.py --precision bf16 --output_dir models/opus-mt-poetry-bf16

Estimated training time (15 epochs, translation-only subset):
  ~20-35 min on Colab free T4, ~50-70 min locally
"""

import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
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

MODEL_NAME  = "Helsinki-NLP/opus-mt-zh-en"
MAX_SRC_LEN = 512
MAX_TGT_LEN = 256

LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=32,
    lora_alpha=64,                          # keep alpha = 2*r scaling
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # full attention (q/k/v/out) + FFN (fc1/fc2) in encoder & decoder for more adapt capacity
    lora_dropout=0.05,
    bias="none",
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA ADAPTER  (chat-messages format -> seq2seq pairs)
# Identical to train_e2_mt5.py — same data format.
# No TASK_PREFIX: Marian already knows the task from pre-training.
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
    return {"source": src_text, "target": tgt_text}

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
        preds = [
            [tok if tok >= 0 else tokenizer.pad_token_id for tok in seq]
            for seq in preds
        ]
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
    use_bf16 = args.precision == "bf16"
    use_fp16 = args.precision == "fp16"

    # Auto-fallback: bf16 requires Ampere+ (A100/H100); free Colab T4 is Turing (fp16 only)
    if use_bf16:
        if not torch.cuda.is_available():
            print("[WARN] No GPU found, falling back to fp32")
            use_bf16 = False
        elif not torch.cuda.is_bf16_supported():
            print("[WARN] GPU does not support bf16 (need Ampere+), falling back to fp16")
            use_bf16 = False
            use_fp16 = True
    print(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

    train_ds = load_split(data_dir / "train.jsonl")
    valid_ds = load_split(data_dir / "valid.jsonl")
    print(f"Train: {len(train_ds):,}  |  Valid: {len(valid_ds):,}")
    if len(train_ds) == 0 or len(valid_ds) == 0:
        raise ValueError("No translation examples loaded. Check --data_dir points to data/combined.")

    tokenizer  = MarianTokenizer.from_pretrained(MODEL_NAME)
    base_model = MarianMTModel.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, LORA_CONFIG)
    model.print_trainable_parameters()

    tok_fn    = make_tokenize_fn(tokenizer)
    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["source", "target"])
    valid_tok = valid_ds.map(tok_fn, batched=True, remove_columns=["source", "target"])

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        warmup_steps=50,
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
        logging_steps=20,
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

    trainer.train()

    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved -> {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",                  default="data/combined")
    parser.add_argument("--output_dir",                default="models/opus-mt-poetry")
    parser.add_argument("--epochs",                    type=int,  default=15)
    parser.add_argument("--batch_size",                type=int,  default=8)
    parser.add_argument("--early_stopping_patience",   type=int,  default=3)
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Training precision. bf16 is faster on supported GPUs.",
    )
    main(parser.parse_args())
