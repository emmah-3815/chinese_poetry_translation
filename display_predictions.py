"""Display predictions.jsonl in formatted terminal output matching the slide style."""

import io
import json
import sys
import textwrap

# Force UTF-8 stdout so Chinese/unicode renders correctly on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PRED_FILE = "eval_results/opus-mt-poetry/eval_results_no_metadata/predictions.jsonl"
CANONICAL_FILE = "data/combined/test_canonical.jsonl"
MODEL_LABEL = "opus-mt + LoRA"
WIDTH = 72

# ANSI codes
BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[36m"
RESET = "\033[0m"

DYNASTY_MAP = {
    "唐代": "Tang", "宋代": "Song", "元代": "Yuan",
    "明代": "Ming", "清代": "Qing", "五代": "Five Dynasties",
    "Tang Dynasty": "Tang", "Song Dynasty": "Song",
}


def load_canonical_lookup() -> dict:
    """Return {normalised_chinese: {title, dynasty}} from the canonical test set."""
    lookup = {}
    try:
        with open(CANONICAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = rec.get("chinese", "").replace(" ", "").replace("\n", "")
                if key:
                    lookup[key] = {
                        "title":   rec.get("title", ""),
                        "dynasty": DYNASTY_MAP.get(rec.get("dynasty", ""), rec.get("dynasty", "")),
                    }
    except FileNotFoundError:
        pass
    return lookup


def parse_source(source: str, canonical: dict):
    poem = source.strip()
    key = poem.replace(" ", "").replace("\n", "")
    meta = canonical.get(key, {})
    title   = meta.get("title", "")
    dynasty = meta.get("dynasty", "")
    return poem, title, dynasty


def wrap_text(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    wrapped = textwrap.fill(text, width=WIDTH - indent,
                            initial_indent=prefix, subsequent_indent=prefix)
    return wrapped


def render(idx: int, total: int, poem: str, title: str, dynasty: str,
           reference: str, hypothesis: str):
    # ── header ──────────────────────────────────────────────────────────
    tag = f"  {idx:02d} / {total:02d}  {title}  {dynasty}  "
    dots = "·" * ((WIDTH - len(tag)) // 2)
    header = f"{dots}{tag}{dots}"
    # pad to exact WIDTH
    header += "·" * (WIDTH - len(header))

    print()
    print(BOLD + CYAN + header + RESET)

    # ── Original ─────────────────────────────────────────────────────────
    print()
    print(BOLD + "  Original" + RESET)
    for line in poem.split("\n"):
        print("  " + line)

    # ── Reference ────────────────────────────────────────────────────────
    print()
    print(BOLD + "  Reference translation" + RESET)
    print(wrap_text(reference, indent=4))

    # ── Model ────────────────────────────────────────────────────────────
    print()
    print(BOLD + f"  Model translation  [{MODEL_LABEL}]" + RESET)
    print(wrap_text(hypothesis, indent=4))
    print()


def main():
    # Usage: display_predictions.py [title_filter [display_idx display_total]]
    # e.g.   display_predictions.py 长相思 3 15
    filter_title   = sys.argv[1] if len(sys.argv) > 1 else None
    override_idx   = int(sys.argv[2]) if len(sys.argv) > 2 else None
    override_total = int(sys.argv[3]) if len(sys.argv) > 3 else None

    canonical = load_canonical_lookup()

    entries = []
    with open(PRED_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        poem, title, dynasty = parse_source(entry["source"], canonical)
        if filter_title and filter_title not in title and filter_title not in poem:
            continue
        display_idx   = override_idx   if override_idx   is not None else idx
        display_total = override_total if override_total is not None else total
        render(display_idx, display_total, poem, title, dynasty,
               entry["reference"], entry["hypothesis"])


if __name__ == "__main__":
    main()
