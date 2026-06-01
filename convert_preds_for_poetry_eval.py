
"""
convert_preds_for_poetry_eval.py
---------------------------------
Prepare data for cross-eval consistency check between eval_e2_mt5.py and
evaluate_poetry.py.

Steps:
  a) Read data/poetmt_compact/test.jsonl (chat-message format) and write
     data/poetmt_compact/test_flat.jsonl with flat {chinese, english, title, dynasty}.

  b) Read models/e2-mt5-fp32-v2/eval_results_final/predictions.jsonl and write
     models/e2-mt5-fp32-v2/eval_results_final/predictions_converted.jsonl, renaming
     "hypothesis" -> "prediction".

  c) Assert reference alignment: predictions[i]["reference"] must equal
     test[i]["english"] for all rows. Any mismatch is reported and the script
     exits with code 1.

Usage:
  python convert_preds_for_poetry_eval.py
  python convert_preds_for_poetry_eval.py --root /path/to/project
"""

import argparse
import json
import re
import sys
from pathlib import Path


def read_jsonl(path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def extract_flat(record):
    """
    Extract classical_zh, english, title, dynasty from a chat-format test record.

    The record has top-level fields:
      - classical_zh  (the classical Chinese poem text)
      - english       (the English reference translation)
      - title, dynasty (if present)
    And a messages array where:
      - messages[user].content: contains "Title: ...
Poet: ... (Dynasty)"
      - messages[assistant].content: the English reference translation
    """
    # Use top-level fields as the primary source -- guaranteed clean.
    classical_zh = record.get("classical_zh", "")
    english      = record.get("english",      "")

    title   = record.get("title",   "")
    dynasty = record.get("dynasty", "")

    if not title or not dynasty:
        user_content = next(
            (m["content"] for m in record.get("messages", []) if m["role"] == "user"),
            ""
        )
        if not title:
            m_title = re.search(r"^Title:\s*(.+)$", user_content, re.MULTILINE)
            if m_title:
                title = m_title.group(1).strip()
        if not dynasty:
            m_poet = re.search(
                r"^Poet:\s*.+\(([^)]+Dynasty)\)", user_content, re.MULTILINE
            )
            if m_poet:
                dynasty = m_poet.group(1).strip()

    # Fallback: extract classical_zh from user message if top-level is missing.
    if not classical_zh:
        user_content = next(
            (m["content"] for m in record.get("messages", []) if m["role"] == "user"),
            ""
        )
        m_zh = re.search(
            r"Classical Chinese:\n(.*?)(?:\n\n|$)", user_content, re.DOTALL
        )
        if m_zh:
            classical_zh = m_zh.group(1).strip()

    # Fallback: pull english from assistant message if top-level is missing.
    if not english:
        english = next(
            (m["content"] for m in record.get("messages", []) if m["role"] == "assistant"),
            ""
        )

    return {
        "chinese": classical_zh,
        "english": english,
        "title":   title,
        "dynasty": dynasty,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert eval_e2_mt5.py outputs for use with evaluate_poetry.py"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root directory. Defaults to the directory containing this script.",
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).resolve()
    else:
        root = Path(__file__).resolve().parent

    test_chat_path = root / "data" / "poetmt_compact" / "test.jsonl"
    test_flat_path = root / "data" / "poetmt_compact" / "test_flat.jsonl"
    preds_in_path  = root / "models" / "e2-mt5-fp32-v2" / "eval_results_final" / "predictions.jsonl"
    preds_out_path = root / "models" / "e2-mt5-fp32-v2" / "eval_results_final" / "predictions_converted.jsonl"

    print(f"Reading {test_chat_path} ...")
    if not test_chat_path.exists():
        print(f"ERROR: {test_chat_path} not found.", file=sys.stderr)
        sys.exit(1)

    test_chat    = read_jsonl(test_chat_path)
    flat_records = [extract_flat(r) for r in test_chat]

    test_flat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(test_flat_path, "w", encoding="utf-8") as f:
        for rec in flat_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(flat_records)} flat test records -> {test_flat_path}")

    print(f"\nReading {preds_in_path} ...")
    if not preds_in_path.exists():
        print(f"ERROR: {preds_in_path} not found.", file=sys.stderr)
        sys.exit(1)

    preds = read_jsonl(preds_in_path)

    preds_out_path.parent.mkdir(parents=True, exist_ok=True)
    converted = []
    for p in preds:
        rec = dict(p)
        if "hypothesis" in rec:
            rec["prediction"] = rec.pop("hypothesis")
        converted.append(rec)

    with open(preds_out_path, "w", encoding="utf-8") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(converted)} converted predictions -> {preds_out_path}")

    print(
        "\nVerifying reference alignment "
        "(predictions[i].reference == test[i].english) ..."
    )
    if len(flat_records) != len(preds):
        print(
            f"ERROR: row count mismatch -- test has {len(flat_records)} rows, "
            f"predictions has {len(preds)} rows.",
            file=sys.stderr,
        )
        sys.exit(1)

    mismatches = []
    for i, (flat, pred) in enumerate(zip(flat_records, preds)):
        if flat["english"] != pred.get("reference", ""):
            mismatches.append(i)

    if mismatches:
        print(
            f"ERROR: reference mismatch at {len(mismatches)} row(s): "
            f"{mismatches[:10]}",
            file=sys.stderr,
        )
        for i in mismatches[:3]:
            print(f"  row {i}:")
            print(f"    test english:   {flat_records[i]['english'][:80]!r}")
            print(f"    pred reference: {preds[i].get('reference', '')[:80]!r}")
        sys.exit(1)
    else:
        print(f"  All {len(flat_records)} rows aligned -- references match.")

    print("\nDone.")


if __name__ == "__main__":
    main()
