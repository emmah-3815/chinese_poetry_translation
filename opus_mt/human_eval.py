"""
Human evaluation: Adequacy / Fluency / Poeticness (1-5 Likert) for 15 poems.

Usage:
    python human_eval.py               # start / resume
    python human_eval.py --summary     # print summary of completed ratings

Results are saved to human_eval_results.json after each poem and loaded
automatically on the next run so you can pause and resume at any time.

──────────────────────────────────────────────────────────────────────────
CONFIG: add / remove models here.  Each entry is a display label → path to
a predictions.jsonl file with {source, hypothesis, reference} rows.
Qwen predictions can be added later once Emma's eval files are available.
──────────────────────────────────────────────────────────────────────────
"""

import io
import json
import random
import re
import sys
import textwrap
import argparse
from collections import defaultdict
from pathlib import Path

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these paths to match your local layout
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()

MODELS: dict[str, str] = {
    "opus-mt + LoRA":      "eval_results/opus-mt-poetry/eval_results_no_metadata/predictions.jsonl",
    "opus-mt baseline":    "eval_results/opus-mt-baseline/eval_results/predictions.jsonl",
    # Add Qwen predictions below once available:
    # "Qwen0.5B + QLoRA":  "eval_results/qwen-0.5b-qlora/predictions.jsonl",
    # "Qwen1.5B + QLoRA":  "eval_results/qwen-1.5b-qlora/predictions.jsonl",
}

CANONICAL_TEST   = "data/combined/test_canonical.jsonl"
RESULTS_FILE     = "human_eval_results.json"
N_POEMS          = 15          # total poems to rate
N_PER_DYNASTY    = 5           # poems per dynasty
RANDOM_SEED      = 42

# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"
WIDTH = 78


def hr(char="─"):
    print(char * WIDTH)


def bold(s):  return BOLD + s + RESET
def dim(s):   return DIM  + s + RESET
def cyan(s):  return CYAN + s + RESET
def green(s): return GREEN + s + RESET


def wrap(text: str, indent: int = 4) -> str:
    prefix = " " * indent
    return textwrap.fill(
        text, width=WIDTH - indent,
        initial_indent=prefix, subsequent_indent=prefix,
        break_long_words=False, break_on_hyphens=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & matching
# ─────────────────────────────────────────────────────────────────────────────

DYNASTY_MAP = {"唐代": "Tang", "宋代": "Song", "元代": "Yuan"}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_chinese_from_source(source: str) -> str:
    """
    Pull just the Chinese poem lines out of a prediction source field.
    Both opus-mt (Chinese first) and mT5 (task-prefix first) are handled.
    Lines starting with known metadata prefixes are stripped.
    """
    meta_prefixes = (
        "translate classical", "Title:", "Poet:", "Modern Chinese",
        "Annotations", "Background:", "Classical Chinese:",
    )
    lines = source.strip().split("\n")
    poem_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in meta_prefixes):
            continue
        # Skip lines that are entirely ASCII (task prefix fragments)
        if stripped.isascii():
            continue
        poem_lines.append(stripped)
    return "".join(poem_lines).replace(" ", "")


def build_lookup(pred_path: Path) -> dict[str, str]:
    """Return {normalised_chinese -> hypothesis}."""
    rows = load_jsonl(pred_path)
    lookup: dict[str, str] = {}
    for row in rows:
        key = extract_chinese_from_source(row["source"])
        if key:
            lookup[key] = row["hypothesis"]
    return lookup


def build_poem_list() -> list[dict]:
    """
    Load canonical test, join each model's prediction, return only poems
    where every configured model has a prediction.
    """
    canonical = load_jsonl(SCRIPT_DIR / CANONICAL_TEST)

    # Build per-model lookups
    lookups: dict[str, dict[str, str]] = {}
    for label, rel_path in MODELS.items():
        full = SCRIPT_DIR / rel_path
        if not full.exists():
            print(f"[WARN] prediction file not found, skipping model: {label}\n  {full}")
            continue
        lookups[label] = build_lookup(full)

    if not lookups:
        sys.exit("No model prediction files found. Check MODELS config paths.")

    poems = []
    for rec in canonical:
        chinese_raw = rec.get("chinese", "").strip()
        key = chinese_raw.replace(" ", "").replace("\n", "")
        dynasty_raw = rec.get("dynasty", "")
        dynasty = DYNASTY_MAP.get(dynasty_raw, dynasty_raw)

        # Check every model has a prediction for this poem
        preds: dict[str, str] = {}
        for label, lkp in lookups.items():
            # exact match first
            hyp = lkp.get(key)
            if hyp is None:
                # fuzzy: try partial key
                for k, v in lkp.items():
                    if key and (key in k or k in key):
                        hyp = v
                        break
            if hyp is None:
                break          # skip poem if any model missing
            preds[label] = hyp
        else:
            poems.append({
                "chinese":   chinese_raw,
                "english":   rec.get("english", ""),
                "title":     rec.get("title", ""),
                "author":    rec.get("author", ""),
                "dynasty":   dynasty,
                "dynasty_raw": dynasty_raw,
                "predictions": preds,
            })

    return poems


def sample_poems(poems: list[dict], n_per_dynasty: int, seed: int) -> list[dict]:
    """
    Sample poems for human evaluation, spreading evenly across dynasties.
    Matches evaluate_poetry.py's sample_human_eval_poems logic.
    """
    n = n_per_dynasty * 3   # total target (3 dynasties: Tang, Song, Yuan)
    rng = random.Random(seed)

    by_dynasty: dict[str, list[dict]] = defaultdict(list)
    for p in poems:
        by_dynasty[p["dynasty"]].append(p)

    dynasties = sorted(by_dynasty.keys())
    per   = n // len(dynasties)
    extra = n % len(dynasties)

    selected: list[dict] = []
    for i, dynasty in enumerate(dynasties):
        take = per + (1 if i < extra else 0)
        pool = list(by_dynasty[dynasty])
        rng.shuffle(pool)
        selected.extend(pool[:take])

    # Top up from remainder if any dynasty had fewer poems than requested
    if len(selected) < n:
        remaining = [p for p in poems if p not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Progress / persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"poems": [], "ratings": {}}


def save_results(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive rating
# ─────────────────────────────────────────────────────────────────────────────

DIMENSIONS = [
    ("adequacy",   "Adequacy",   "How well does the translation preserve the poem's meaning?"),
    ("fluency",    "Fluency",    "How natural and grammatically correct is the English?"),
    ("poeticness", "Poeticness", "How poetic and elegant does the translation read?"),
]


_STDIN_EOF = False   # set True on first EOFError so ask_score can bail


def safe_input(prompt: str) -> str:
    """input() that sets _STDIN_EOF on first EOFError instead of crashing."""
    global _STDIN_EOF
    try:
        return input(prompt)
    except EOFError:
        _STDIN_EOF = True
        print()
        return ""


def ask_score(prompt_label: str, description: str) -> int:
    scale = dim("  1=poor  2=fair  3=good  4=very good  5=excellent")
    while True:
        if _STDIN_EOF:
            print("\n  stdin is not interactive. Run this script in a real terminal, not a pipe.")
            sys.exit(1)
        try:
            raw = safe_input(f"  {bold(prompt_label)} {dim(description)}\n{scale}\n  → ").strip()
            if _STDIN_EOF or not raw:
                continue
            val = int(raw)
            if 1 <= val <= 5:
                return val
            print("  Please enter a number from 1 to 5.")
        except ValueError:
            print("  Invalid input — enter a digit 1–5.")
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Progress saved.")
            sys.exit(0)


def display_poem(idx: int, total: int, poem: dict) -> None:
    print()
    hr("═")
    header = f"  POEM {idx}/{total}  ·  {poem['title']}  ·  {poem['author']}  ·  {poem['dynasty']} Dynasty"
    print(cyan(bold(header)))
    hr("═")

    print()
    print(bold("  Original Chinese"))
    for line in poem["chinese"].split("\n"):
        if line.strip():
            print("  " + line.strip())

    print()
    print(bold("  Reference translation (human expert)"))
    print(wrap(poem["english"], indent=4))


# Metadata keyword patterns — all require a colon to avoid false-positive on
# words like "poet" in a real translation.  Covers garbled variants the
# baseline model produces (e.g. "Tile:" for "Title:", "Poot:" for "Poet:").
_META_KW = (
    r"Ti[tle]{1,3}:"           # Title: / Tile: / Titl: / Tite:
    r"|Po[oe]t:"              # Poet: / Poot:
    r"|Annotations?\s*[\(（]"
    r"|Background:"
    r"|translate\s+classical"
    r"|注释[：:]"
)

# Matches metadata at line start OR preceded by whitespace/sentence boundary.
# Group 1 captures the translation prefix before the keyword.
_INLINE_META_RE = re.compile(
    r"(.*?)"
    r"(?:[\s.]|(?<=\w))"
    r"(?:" + _META_KW + r")"
    r".*$",
    re.IGNORECASE | re.DOTALL,
)
_LINE_START_META_RE = re.compile(
    r"^\s*(?:" + _META_KW + r")",
    re.IGNORECASE,
)


# Matches a bare author-name echo at the very start of a hypothesis.
# Handles three model output patterns:
#   - "Li Honghing (Tang Dae)Translation…"  — multi-token + (Dynasty)
#   - "Wu Wen-young (Omong)Translation…"    — hyphenated name
#   - "Qingqing(Tunding)Translation…"       — single name directly before (Dynasty)
#   - "Su Jijiu:Translation…"               — author + colon, no paren
_LEADING_AUTHOR_RE = re.compile(
    r"^"
    r"(?:"
        r"(?:[A-Z][a-zA-Z-]+[ \t]+){1,3}[A-Z][a-zA-Z-]+"  # 2-4 tokens (hyphens ok)
        r"|"
        r"[A-Z][a-zA-Z-]{3,}"                               # single token ≥4 chars
    r")"
    r"\s*"
    r"(?:\([A-Z][A-Za-z ]{1,28}\)|:)"   # (Capitalised Dynasty) or bare colon
    r"[ \t]*",
)


def clean_hypothesis(hyp: str) -> tuple[str, bool]:
    """Strip echoed input-prompt metadata from a model output.

    Handles whole-line echoes, inline echoes (mid-sentence), and bare
    author-name prefixes like \"Li XingXuanin (Tang Dong)\".
    Returns (cleaned_text, was_modified).
    """
    modified = False

    # Strip bare "Author Name (Dynasty)" echo at the very start of the hypothesis
    m0 = _LEADING_AUTHOR_RE.match(hyp)
    if m0:
        hyp = hyp[m0.end():]
        modified = True

    lines = hyp.split("\n")
    out_lines = []
    for line in lines:
        # Drop entire line if it starts with a metadata keyword
        if _LINE_START_META_RE.match(line):
            modified = True
            continue
        # Truncate line at an inline metadata keyword
        m = _INLINE_META_RE.match(line)
        if m and m.group(1).strip():
            out_lines.append(m.group(1).rstrip(" .,;"))
            modified = True
        elif m:
            # Whole line was metadata after the regex matched empty prefix — drop it
            modified = True
        else:
            out_lines.append(line)
    cleaned = "\n".join(out_lines).strip()
    return cleaned, modified


def display_and_rate_model(label: str, hypothesis: str) -> dict[str, int]:
    cleaned, had_echo = clean_hypothesis(hypothesis)
    print()
    hr()
    print(bold(f"  Model: {label}"))
    if had_echo:
        print(RED + "  [Note: model echoed input metadata — stripped for display]" + RESET)
    hr()
    print(wrap(cleaned if cleaned else hypothesis, indent=4))
    print()
    ratings: dict[str, int] = {}
    for key, name, desc in DIMENSIONS:
        ratings[key] = ask_score(name, f"({desc})")
    return ratings


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(data: dict) -> None:
    ratings_by_model: dict[str, dict[str, list[int]]] = {}
    for poem_key, poem_ratings in data["ratings"].items():
        for model_label, scores in poem_ratings.items():
            if model_label not in ratings_by_model:
                ratings_by_model[model_label] = {d[0]: [] for d in DIMENSIONS}
            for dim_key, _, _ in DIMENSIONS:
                if dim_key in scores:
                    ratings_by_model[model_label][dim_key].append(scores[dim_key])

    if not ratings_by_model:
        print("No ratings recorded yet.")
        return

    print()
    hr("═")
    print(cyan(bold("  HUMAN EVALUATION SUMMARY")))
    hr("═")

    col_w = 12
    dim_names = [d[1] for d in DIMENSIONS]
    header = f"  {'Model':<28}" + "".join(f"{n:>{col_w}}" for n in dim_names) + f"{'Avg':>{col_w}}"
    print(bold(header))
    hr()

    for model_label, dim_scores in sorted(ratings_by_model.items()):
        avgs = []
        row = f"  {model_label:<28}"
        for dim_key, _, _ in DIMENSIONS:
            vals = dim_scores[dim_key]
            avg = sum(vals) / len(vals) if vals else 0.0
            avgs.append(avg)
            row += f"{avg:>{col_w}.2f}"
        overall = sum(avgs) / len(avgs) if avgs else 0.0
        row += f"{overall:>{col_w}.2f}"
        print(row)

    hr()
    n_poems = len(data["ratings"])
    print(f"  Poems rated: {n_poems}")
    print()

    # Per-poem breakdown (brief)
    print(bold("  Per-poem breakdown"))
    hr()
    for poem_key, poem_ratings in data["ratings"].items():
        # Extract title from stored poem list
        title = poem_key
        for p in data.get("poems", []):
            if p["chinese"][:10].replace(" ", "") in poem_key:
                title = f"{p.get('title','')} ({p.get('dynasty','')})"
                break
        print(f"  {title}")
        for model_label, scores in poem_ratings.items():
            vals = " / ".join(f"{scores.get(d[0],'?')}" for d in DIMENSIONS)
            print(f"    {model_label:<30} {vals}  (A/F/P)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def poem_key(poem: dict) -> str:
    return poem["chinese"][:20].replace(" ", "").replace("\n", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Human evaluation for poetry translation")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n", type=int, default=N_PER_DYNASTY,
                        help="Poems per dynasty to sample (default 5)")
    args = parser.parse_args()

    results_path = SCRIPT_DIR / RESULTS_FILE

    if args.summary:
        data = load_results(results_path)
        print_summary(data)
        return

    print()
    print(cyan(bold("  Classical Chinese Poetry — Human Evaluation")))
    print(dim("  Rating scale: 1=poor  2=fair  3=good  4=very good  5=excellent"))
    print(dim("  Dimensions: Adequacy · Fluency · Poeticness"))
    print(dim("  Press Ctrl-C at any time to pause; progress is saved automatically."))
    print()

    # Load or build poem list
    data = load_results(results_path)

    if data.get("poems"):
        poems_to_rate = data["poems"]
        print(f"  Resuming from saved session ({len(data['ratings'])} poems already rated).")
    else:
        print("  Building poem list and sampling...")
        all_poems = build_poem_list()
        print(f"  Found {len(all_poems)} poems with predictions from all {len(MODELS)} model(s).")
        dynasty_counts = {}
        for p in all_poems:
            dynasty_counts[p["dynasty"]] = dynasty_counts.get(p["dynasty"], 0) + 1
        print(f"  Dynasty distribution: {dynasty_counts}")

        poems_to_rate = sample_poems(all_poems, args.n, args.seed)
        print(f"  Sampled {len(poems_to_rate)} poems.")
        data["poems"] = poems_to_rate
        save_results(data, results_path)

    already_done = set(data.get("ratings", {}).keys())
    remaining = [p for p in poems_to_rate if poem_key(p) not in already_done]

    if not remaining:
        print(green("  All poems rated!"))
        print_summary(data)
        return

    total = len(poems_to_rate)
    done_count = total - len(remaining)

    import sys as _sys
    if not _sys.stdin.isatty():
        print("\n  ERROR: Run this script in a real terminal (not a pipe or IDE run panel).")
        print("  Example:  python human_eval.py")
        _sys.exit(1)

    print(f"\n  {done_count}/{total} poems already rated. Starting from poem {done_count + 1}.\n")
    safe_input(dim("  Press Enter to begin..."))

    for poem in remaining:
        idx = done_count + 1
        display_poem(idx, total, poem)

        poem_ratings: dict[str, dict[str, int]] = {}
        for label in poem["predictions"]:
            hypothesis = poem["predictions"][label]
            poem_ratings[label] = display_and_rate_model(label, hypothesis)
            print(green(f"  Saved ratings for {label}."))

        key = poem_key(poem)
        data.setdefault("ratings", {})[key] = poem_ratings
        save_results(data, results_path)
        done_count += 1

        print()
        print(green(f"  ✓ Poem {idx}/{total} rated and saved."))
        if done_count < total:
            try:
                safe_input(dim("  Press Enter for next poem (Ctrl-C to pause)..."))
            except KeyboardInterrupt:
                print("\n  Progress saved. Run again to continue.")
                sys.exit(0)

    print()
    hr("═")
    print(green(bold("  All poems rated! Generating summary...")))
    hr("═")
    print_summary(data)
    print(f"  Full results saved to: {results_path}")


if __name__ == "__main__":
    main()
