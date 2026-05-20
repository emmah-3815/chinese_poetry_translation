"""
E2: mT5-base + LoRA — classical Chinese poetry -> English translation.
Data: data/poetmt/{train,valid,test}.jsonl  (build_dataset_poetmt.py output)

Usage:
  python train_e2_mt5.py
  python train_e2_mt5.py --data_dir data/poetmt --output_dir models/e2-mt5 --epochs 5
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

MODEL_NAME  = "google/mt5-base"
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

    train_ds = load_split(data_dir / "train.jsonl")
    valid_ds = load_split(data_dir / "valid.jsonl")
    print(f"Train: {len(train_ds):,}  |  Valid: {len(valid_ds):,}")

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
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
        fp16=False,                       # mT5 has fp16 instability
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    adapter_path = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved -> {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/poetmt")
    parser.add_argument("--output_dir", default="models/e2-mt5")
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    main(parser.parse_args())
