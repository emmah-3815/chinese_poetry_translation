"""
Dataset Pipeline: PoetMT only (CN→EN translation) for E1 / E2.

Directory structure expected:
  PoetMT/all_poems/
    tang.jsonl / tang-background.jsonl
    song.jsonl / song-background.jsonl
    yuan.jsonl / yuan-background.jsonl

Usage:
  python build_dataset_poetmt.py --inspect          # inspect field names first
  python build_dataset_poetmt.py                    # build final dataset
"""

import json
import re
import argparse
import random
from pathlib import Path
from collections import Counter

random.seed(42)

SCRIPT_DIR = Path(__file__).parent.resolve()

# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

ZH_CHARS = re.compile(r'[一-鿿]')
EN_CHARS = re.compile(r'[a-zA-Z]')

# ══════════════════════════════════════════════════════════════════════════════
# FILE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

DYNASTY_FILES = {
    "tang": "tang.jsonl",
    "song": "song.jsonl",
    "yuan": "yuan.jsonl",
}
BACKGROUND_FILES = {
    "tang": "tang-background.jsonl",
    "song": "song-background.jsonl",
    "yuan": "yuan-background.jsonl",
}

# ══════════════════════════════════════════════════════════════════════════════
# FIELD MAPS  ← edit these after running --inspect
# ══════════════════════════════════════════════════════════════════════════════

POETMT_FIELD_MAP = {
    "title":        ["title", "poem_title", "name", "题目"],
    "author":       ["author", "poet", "author_name", "作者"],
    "dynasty":      ["dynasty", "era", "period", "朝代"],
    "classical_zh": ["src", "content", "lines", "chinese", "poem", "original",
                     "classical", "text", "原文", "诗句"],
    "english":      ["ref", "translation", "english", "en", "english_translation",
                     "reference", "human_translation"],
    "modern_zh":    ["modern_chinese", "modern", "modern_zh",
                     "vernacular", "paraphrase", "译文", "白话"],
}

BACKGROUND_FIELD_MAP = {
    "join_key":    ["title", "poem_title", "id", "poem_id", "题目"],
    "annotations": ["note", "注释", "annotation", "notes", "commentary", "注解"],
    "modern_zh":   ["modern_chinese", "modern", "vernacular", "白话", "译文"],
    "background":  ["background", "author_intro", "history", "context", "背景"],
}

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

TRANSLATION_INSTRUCTION = (
    "You are a skilled literary translator. "
    "Translate the following classical Chinese poem into English. "
    "Preserve the poem's imagery, cultural depth, and poetic elegance."
)

# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_poetmt(folder: str) -> tuple[list, list]:
    folder = (SCRIPT_DIR / folder).resolve()
    print(f"[PoetMT] Looking in: {folder}")
    poems, backgrounds = [], []

    for dynasty, fname in DYNASTY_FILES.items():
        p = folder / fname
        if not p.exists():
            print(f"[WARN] Missing: {p}")
            continue
        recs = load_jsonl(p)
        for r in recs:
            r["_dynasty"] = dynasty
        poems.extend(recs)
        print(f"[PoetMT] {fname}: {len(recs):,} records")

    for dynasty, fname in BACKGROUND_FILES.items():
        p = folder / fname
        if not p.exists():
            print(f"[WARN] Missing: {p}")
            continue
        recs = load_jsonl(p)
        for r in recs:
            r["_dynasty"] = dynasty
        backgrounds.extend(recs)
        print(f"[PoetMT] {fname}: {len(recs):,} records")

    print(f"\n[PoetMT] Total poems: {len(poems):,} | background entries: {len(backgrounds):,}")
    return poems, backgrounds

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

def inspect_schema(records: list, name: str, n: int = 3) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  —  first {n} records")
    print(f"{'='*60}")
    all_keys: Counter = Counter()
    for r in records:
        all_keys.update(r.keys())
    print(f"  Keys (frequency): {dict(all_keys.most_common())}")
    for i, r in enumerate(records[:n]):
        print(f"\n  Record {i}:")
        for k, v in r.items():
            preview = str(v)[:120].replace("\n", " ")
            print(f"    {k!r:25s}: {preview}")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_field(record: dict, candidates: list) -> str | None:
    for key in candidates:
        if key in record and record[key]:
            return record[key]
    return None

def normalize_lines(value) -> str:
    if isinstance(value, list):
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()

def count_zh(text: str) -> int:
    return ZH_CHARS.subn("", text)[1]

def count_en(text: str) -> int:
    return EN_CHARS.subn("", text)[1]

# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def build_background_index(background_records: list) -> dict:
    index = {}
    for rec in background_records:
        key = resolve_field(rec, BACKGROUND_FIELD_MAP["join_key"])
        if key:
            index[str(key).strip()] = rec
    print(f"[PoetMT] Background index: {len(index):,} entries")
    return index

def enrich_poem(poem: dict, bg_index: dict) -> dict:
    join_key = resolve_field(poem, POETMT_FIELD_MAP["title"])
    if not join_key:
        return poem
    bg = bg_index.get(str(join_key).strip())
    if not bg:
        return poem
    poem["annotations"] = normalize_lines(
        resolve_field(bg, BACKGROUND_FIELD_MAP["annotations"]) or "")
    poem["background"]  = normalize_lines(
        resolve_field(bg, BACKGROUND_FIELD_MAP["background"])  or "")
    if not poem.get("modern_zh"):
        poem["modern_zh"] = normalize_lines(
            resolve_field(bg, BACKGROUND_FIELD_MAP["modern_zh"]) or "")
    return poem

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT & CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def extract_poetmt(records: list) -> list:
    extracted, skipped = [], Counter()
    for rec in records:
        classical = resolve_field(rec, POETMT_FIELD_MAP["classical_zh"])
        english   = resolve_field(rec, POETMT_FIELD_MAP["english"])
        if not classical or not english:
            skipped["missing_required_field"] += 1
            continue
        extracted.append({
            "source":       "poetmt",
            "title":        normalize_lines(resolve_field(rec, POETMT_FIELD_MAP["title"])  or ""),
            "author":       normalize_lines(resolve_field(rec, POETMT_FIELD_MAP["author"]) or ""),
            "dynasty":      rec.get("_dynasty", normalize_lines(
                                resolve_field(rec, POETMT_FIELD_MAP["dynasty"]) or "")),
            "classical_zh": normalize_lines(classical),
            "modern_zh":    normalize_lines(resolve_field(rec, POETMT_FIELD_MAP["modern_zh"]) or ""),
            "annotations":  normalize_lines(rec.get("annotations", "")),
            "background":   normalize_lines(rec.get("background",  "")),
            "english":      normalize_lines(english),
        })
    print(f"[PoetMT] Extracted {len(extracted):,} | skipped: {dict(skipped)}")
    return extracted

def clean_poetmt(records: list) -> list:
    seen_classical, seen_english = set(), set()
    cleaned, skipped = [], Counter()
    for p in records:
        c, e = p["classical_zh"], p["english"]
        if not (4 <= len(c) <= 300):
            skipped["classical_length"] += 1; continue
        if not (5 <= len(e) <= 800):
            skipped["english_length"] += 1; continue
        if count_zh(c) / max(len(c), 1) < 0.3:
            skipped["classical_not_zh"] += 1; continue
        if count_en(e) / max(len(e), 1) < 0.3:
            skipped["english_not_en"] += 1; continue
        if c.strip() == e.strip():
            skipped["identical_src_tgt"] += 1; continue
        c_norm = " ".join(c.split())
        e_norm = " ".join(e.split())
        if c_norm in seen_classical:
            skipped["dup_classical"] += 1; continue
        if e_norm in seen_english:
            skipped["dup_english"] += 1; continue
        seen_classical.add(c_norm)
        seen_english.add(e_norm)
        p["classical_zh"] = c_norm
        p["english"]      = e_norm
        cleaned.append(p)
    print(f"[PoetMT] After cleaning: {len(cleaned):,} kept | {dict(skipped)}")
    return cleaned

# ══════════════════════════════════════════════════════════════════════════════
# FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def format_translation(p: dict) -> dict:
    context = ""
    if p.get("title"):
        context += f"Title: {p['title']}\n"
    if p.get("author") and p.get("dynasty"):
        context += f"Poet: {p['author']} ({p['dynasty'].capitalize()} Dynasty)\n"
    elif p.get("author"):
        context += f"Poet: {p['author']}\n"
    if p.get("modern_zh"):
        context += f"Modern Chinese meaning: {p['modern_zh']}\n"
    if p.get("annotations"):
        context += f"Annotations (注释): {p['annotations']}\n"
    if p.get("background"):
        context += f"Background: {p['background']}\n"

    user_content = context + f"\nClassical Chinese:\n{p['classical_zh']}"
    return {
        "source":         p["source"],
        "task":           "translation",
        "has_annotation": bool(p.get("annotations")),
        "messages": [
            {"role": "system",    "content": TRANSLATION_INSTRUCTION},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": p["english"]},
        ],
        "classical_zh": p["classical_zh"],
        "english":      p["english"],
    }

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def split_poetmt(records: list,
                 train_ratio: float = 0.8,
                 val_ratio:   float = 0.1) -> tuple[list, list, list]:
    """
    Test is carved from the tail FIRST (before shuffling) so it is always
    non-empty and deterministic.  Minimum test size = max(50, 10% of total),
    capped at 20% so the training set stays large.
    """
    n = len(records)
    if n == 0:
        return [], [], []

    test_size = max(50, int(n * (1 - train_ratio - val_ratio)))
    test_size = min(test_size, n // 5)
    remaining = records[:-test_size]
    test      = records[-test_size:]

    random.shuffle(remaining)
    val_size = int(len(remaining) * val_ratio / (train_ratio + val_ratio))
    val_size = max(val_size, 20)

    valid = remaining[:val_size]
    train = remaining[val_size:]

    print(f"[Split]  train={len(train):,}  valid={len(valid):,}  test={len(test):,}")
    return train, valid, test

# ══════════════════════════════════════════════════════════════════════════════
# SAVE & STATS
# ══════════════════════════════════════════════════════════════════════════════

def save_jsonl(records: list, path: str) -> None:
    out = (SCRIPT_DIR / path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} → {out}")

def print_stats(splits: dict) -> None:
    print("\n" + "="*60)
    print("  FINAL DATASET STATISTICS")
    print("="*60)
    for split_name, records in splits.items():
        annotated = sum(1 for r in records if r.get("has_annotation"))
        print(f"\n  {split_name.upper()} ({len(records):,} total)")
        if annotated:
            print(f"    With 注释: {annotated:,}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    poem_records, background_records = load_poetmt(args.poetmt_dir)

    if args.inspect:
        inspect_schema(poem_records,       "PoetMT POEMS")
        inspect_schema(background_records, "PoetMT BACKGROUND")
        print("\nEdit POETMT_FIELD_MAP / BACKGROUND_FIELD_MAP to match your keys.")
        return

    bg_index     = build_background_index(background_records)
    poem_records = [enrich_poem(p, bg_index) for p in poem_records]
    matched_modern = sum(1 for p in poem_records if p.get("modern_zh"))
    matched_about  = sum(1 for p in poem_records if p.get("background"))
    print(f"[PoetMT] {matched_modern:,}/{len(poem_records):,} poems enriched with 译文")
    print(f"[PoetMT] {matched_about:,}/{len(poem_records):,} poems enriched with 创作背景")

    pairs   = extract_poetmt(poem_records)
    cleaned = clean_poetmt(pairs)

    formatted        = [format_translation(p) for p in cleaned]
    train, valid, test = split_poetmt(formatted)

    print(f"\nSaving to {args.output_dir}/")
    splits = {"train": train, "valid": valid, "test": test}
    for split_name, records in splits.items():
        save_jsonl(records, f"{args.output_dir}/{split_name}.jsonl")

    print_stats(splits)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poetmt_dir", default="data/PoetMT-main/PoetMT-main/all_poems")
    parser.add_argument("--output_dir", default="data/poetmt")
    parser.add_argument("--inspect",    action="store_true")
    main(parser.parse_args())
