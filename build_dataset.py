"""
Dataset Pipeline: PoetMT (CN→EN) + CCPM (auxiliary) for Path 1 (Qwen2.5-1.5B + QLoRA)

Directory structure expected:
  PoetMT/all_poems/
    tang.jsonl
    tang-background.jsonl
    song.jsonl
    song-background.jsonl
    yuan.jsonl
    yuan-background.jsonl
  CCPM/
    train.jsonl
    valid.jsonl
    test_public.jsonl

Usage:
  python build_dataset.py --inspect                  # inspect field names first
  python build_dataset.py                            # build final dataset
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

ZH_CHARS      = re.compile(r'[\u4e00-\u9fff]')
EN_CHARS      = re.compile(r'[a-zA-Z]')
ZH_PUNCT = r'，。！？、；：""（）【】…—～\s'
CLASSICAL_PAT = re.compile(rf'^[\u4e00-\u9fff{ZH_PUNCT}]+$')
MODERN_PAT    = re.compile(rf'^[\u4e00-\u9fffa-zA-Z0-9{ZH_PUNCT}]+$')

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
    "classical_zh": ["content", "lines", "chinese", "poem", "original",
                     "classical", "text", "原文", "诗句"],
    "english":      ["translation", "english", "en", "english_translation",
                     "reference", "human_translation"],
    "modern_zh":    ["modern_chinese", "modern", "modern_zh",
                     "vernacular", "paraphrase", "译文", "白话"],
}

BACKGROUND_FIELD_MAP = {
    "join_key":    ["title", "poem_title", "id", "poem_id", "题目"],
    "annotations": ["注释", "annotation", "notes", "commentary", "注解"],
    "modern_zh":   ["modern_chinese", "modern", "vernacular", "白话", "译文"],
    "background":  ["background", "author_intro", "history", "context", "背景"],
}

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

TRANSLATION_INSTRUCTION = (
    "You are a skilled literary translator. "
    "Translate the following classical Chinese poem into English. "
    "Preserve the poem's imagery, cultural depth, and poetic elegance."
)

AUXILIARY_INSTRUCTION = (
    "You are an expert in classical Chinese literature. "
    "Given a modern Chinese paraphrase, recover the original classical Chinese poem line."
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

def load_ccpm_splits(folder: str) -> dict:
    folder = (SCRIPT_DIR / folder).resolve()
    print(f"[CCPM]   Looking in: {folder}")
    splits = {}
    # Only load train and valid — test_public.jsonl has no answer field
    for name, fname in [("train", "train.jsonl"),
                         ("valid", "valid.jsonl")]:
        p = folder / fname
        if p.exists():
            splits[name] = load_jsonl(p)
            print(f"[CCPM]   {fname}: {len(splits[name]):,} records")
        else:
            print(f"[WARN] Missing: {p}")
    return splits

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
    return len(ZH_CHARS.findall(text))

def count_en(text: str) -> int:
    return len(EN_CHARS.findall(text))

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
# EXTRACTORS
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
            "source":      "poetmt",
            "title":       normalize_lines(resolve_field(rec, POETMT_FIELD_MAP["title"])   or ""),
            "author":      normalize_lines(resolve_field(rec, POETMT_FIELD_MAP["author"])  or ""),
            "dynasty":     rec.get("_dynasty", normalize_lines(
                               resolve_field(rec, POETMT_FIELD_MAP["dynasty"]) or "")),
            "classical_zh": normalize_lines(classical),
            "modern_zh":   normalize_lines(resolve_field(rec, POETMT_FIELD_MAP["modern_zh"]) or ""),
            "annotations": normalize_lines(rec.get("annotations", "")),
            "background":  normalize_lines(rec.get("background",  "")),
            "english":     normalize_lines(english),
        })
    print(f"[PoetMT] Extracted {len(extracted):,} | skipped: {dict(skipped)}")
    return extracted

def extract_ccpm(splits: dict) -> list:
    all_pairs, skipped = [], Counter()
    for split_name, records in splits.items():
        for rec in records:
            modern  = rec.get("translation", "").strip()
            choices = rec.get("choices", [])
            answer  = rec.get("answer")
            if not isinstance(answer, int) or not (0 <= answer < len(choices)):
                skipped["bad_answer"] += 1
                continue
            classical = choices[answer].strip()
            if not classical or not modern:
                skipped["empty_field"] += 1
                continue
            all_pairs.append({
                "source":       "ccpm",
                "split":        split_name,
                "classical_zh": classical,
                "modern_zh":    modern,
            })
    print(f"[CCPM]   Extracted {len(all_pairs):,} | skipped: {dict(skipped)}")
    return all_pairs

# ══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def is_valid_poetmt(p: dict) -> tuple[bool, str]:
    c, e = p["classical_zh"], p["english"]
    if not (4 <= len(c) <= 300):                          return False, "classical_length"
    if not (5 <= len(e) <= 800):                          return False, "english_length"
    if count_zh(c) / max(len(c), 1) < 0.3:               return False, "classical_not_zh"
    if count_en(e) / max(len(e), 1) < 0.3:               return False, "english_not_en"
    if c.strip() == e.strip():                            return False, "identical_src_tgt"
    return True, "ok"

def is_valid_ccpm(p: dict) -> tuple[bool, str]:
    c, m = p["classical_zh"], p["modern_zh"]
    if not (4 <= len(c) <= 40):       return False, "classical_length"
    if not (5 <= len(m) <= 80):       return False, "modern_length"
    if not CLASSICAL_PAT.match(c):    return False, "classical_not_zh"
    if not MODERN_PAT.match(m):       return False, "modern_not_zh"
    if len(m) < len(c):               return False, "modern_shorter"
    return True, "ok"

def clean_poetmt(records: list) -> list:
    seen_classical, seen_english = set(), set()
    cleaned, skipped = [], Counter()
    for p in records:
        valid, reason = is_valid_poetmt(p)
        if not valid:
            skipped[reason] += 1
            continue
        c = " ".join(p["classical_zh"].split())
        e = " ".join(p["english"].split())
        if c in seen_classical: skipped["dup_classical"] += 1; continue
        if e in seen_english:   skipped["dup_english"]   += 1; continue
        seen_classical.add(c)
        seen_english.add(e)
        p["classical_zh"] = c
        p["english"]      = e
        cleaned.append(p)
    print(f"[PoetMT] After cleaning: {len(cleaned):,} kept | {dict(skipped)}")
    return cleaned

def clean_ccpm(records: list) -> list:
    seen_classical, seen_modern = set(), set()
    cleaned, skipped = [], Counter()
    for p in records:
        valid, reason = is_valid_ccpm(p)
        if not valid:
            skipped[reason] += 1
            continue
        c, m = p["classical_zh"].strip(), p["modern_zh"].strip()
        if c in seen_classical: skipped["dup_classical"] += 1; continue
        if m in seen_modern:    skipped["dup_modern"]    += 1; continue
        seen_classical.add(c)
        seen_modern.add(m)
        p["classical_zh"] = c
        p["modern_zh"]    = m
        cleaned.append(p)
    print(f"[CCPM]   After cleaning: {len(cleaned):,} kept | {dict(skipped)}")
    return cleaned

# ══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION FORMATTING
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

def format_auxiliary(p: dict) -> dict:
    return {
        "source": p["source"],
        "task":   "auxiliary_understanding",
        "messages": [
            {"role": "system",    "content": AUXILIARY_INSTRUCTION},
            {"role": "user",      "content": p["modern_zh"]},
            {"role": "assistant", "content": p["classical_zh"]},
        ],
        "classical_zh": p["classical_zh"],
        "modern_zh":    p["modern_zh"],
    }

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def split_poetmt(records: list,
                 train_ratio: float = 0.8,
                 val_ratio:   float = 0.1) -> tuple[list, list, list]:
    """
    Split PoetMT records into train / valid / test.
    Test is carved out FIRST (before shuffling) so it is always non-empty.
    Minimum test size is 50 or 10% of total, whichever is larger.
    """
    n = len(records)
    if n == 0:
        return [], [], []

    # Carve test out first from the END (deterministic, no shuffle yet)
    test_size  = max(50, int(n * (1 - train_ratio - val_ratio)))
    test_size  = min(test_size, n // 5)   # cap at 20% so train stays large
    remaining  = records[:-test_size]
    test       = records[-test_size:]

    # Now shuffle only the remaining pool
    random.shuffle(remaining)
    val_size   = int(len(remaining) * val_ratio / (train_ratio + val_ratio))
    val_size   = max(val_size, 20)        # at least 20 val samples

    valid = remaining[:val_size]
    train = remaining[val_size:]

    print(f"[Split]  train={len(train):,}  valid={len(valid):,}  test={len(test):,}")
    return train, valid, test

def build_final_splits(poetmt_clean: list, ccpm_by_split: dict) -> dict:
    pm_formatted = [format_translation(p) for p in poetmt_clean]
    print(f"[Build]  PoetMT formatted: {len(pm_formatted):,} translation samples")

    cc_train = [format_auxiliary(p) for p in ccpm_by_split.get("train", [])]
    cc_valid = [format_auxiliary(p) for p in ccpm_by_split.get("valid", [])]

    pm_train, pm_valid, pm_test = split_poetmt(pm_formatted)

    # Test set is translation-only (PoetMT) for clean BLEU/BERTScore evaluation
    train = pm_train + cc_train
    valid = pm_valid + cc_valid
    test  = pm_test             # no auxiliary samples in test

    random.shuffle(train)
    return {"train": train, "valid": valid, "test": test}

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
        task_counts = Counter(r["task"] for r in records)
        src_counts  = Counter(r["source"] for r in records)
        annotated   = sum(1 for r in records if r.get("has_annotation"))
        print(f"\n  {split_name.upper()} ({len(records):,} total)")
        print(f"    By task:    {dict(task_counts)}")
        print(f"    By source:  {dict(src_counts)}")
        if annotated:
            print(f"    With 注释:  {annotated:,}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    poem_records, background_records = load_poetmt(args.poetmt_dir)
    ccpm_splits = load_ccpm_splits(args.ccpm_dir)

    if args.inspect:
        inspect_schema(poem_records,       "PoetMT POEMS")
        inspect_schema(background_records, "PoetMT BACKGROUND")
        inspect_schema(list(ccpm_splits.values())[0], "CCPM")
        print("\nEdit POETMT_FIELD_MAP / BACKGROUND_FIELD_MAP to match your keys.")
        return

    # Enrich poems with background knowledge (modern ZH translation + creation context)
    bg_index     = build_background_index(background_records)
    poem_records = [enrich_poem(p, bg_index) for p in poem_records]
    matched_modern = sum(1 for p in poem_records if p.get("modern_zh"))
    matched_about  = sum(1 for p in poem_records if p.get("about"))
    print(f"[PoetMT] {matched_modern:,}/{len(poem_records):,} poems enriched with 译文 (modern ZH)")
    print(f"[PoetMT] {matched_about:,}/{len(poem_records):,} poems enriched with 创作背景")

    # Extract → clean → format → split
    poetmt_pairs = extract_poetmt(poem_records)
    ccpm_all     = extract_ccpm(ccpm_splits)

    poetmt_clean = clean_poetmt(poetmt_pairs)
    ccpm_clean   = clean_ccpm(ccpm_all)

    ccpm_by_split = {
        split: [p for p in ccpm_clean if p["split"] == split]
        for split in ["train", "valid", "test"]
    }

    final_splits = build_final_splits(poetmt_clean, ccpm_by_split)

    print(f"\nSaving to {args.output_dir}/")
    for split_name, records in final_splits.items():
        save_jsonl(records, f"{args.output_dir}/{split_name}.jsonl")

    print_stats(final_splits)
    print("\nDone. Run --inspect first if field names need adjusting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poetmt_dir",  default="PoetMT-main/PoetMT-main/all_poems")
    parser.add_argument("--ccpm_dir",    default="CCPM-master")
    parser.add_argument("--output_dir",  default="data/combined")
    parser.add_argument("--inspect",     action="store_true")
    main(parser.parse_args())