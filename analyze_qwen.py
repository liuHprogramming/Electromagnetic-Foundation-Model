"""Deterministic first-pass error categories for the ten raw Qwen replies."""
import argparse
import json
from collections import Counter
from pathlib import Path


def error_type(row):
    expected, answer = row["expected_answer"], row["parsed_answer"]
    if answer == "unparseable":
        return "unparseable_or_format_failure"
    if expected == "insufficient":
        return "false_certainty_on_short_clip" if answer != "insufficient" else "correct_abstention"
    if answer == "insufficient":
        return "unnecessary_abstention"
    return "correct_condition" if answer == expected else "wrong_condition"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 10 or len({x["case_id"] for x in rows}) != 10:
        raise ValueError("Exactly ten unique raw records are required")
    cases = [{"case_id": x["case_id"], "expected": x["expected_answer"],
              "parsed": x["parsed_answer"], "error_type": error_type(x)} for x in rows]
    result = {"counts": dict(Counter(x["error_type"] for x in cases)), "cases": cases,
              "scope": "String-based categories only; specific fault hallucinations require manual review of raw replies."}
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
