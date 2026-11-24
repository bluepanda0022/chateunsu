"""사용법: .venv/bin/python scripts/run_research.py [--input 자료.json]"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_tokenizer import experiment, pilot_data


def main():
    parser = argparse.ArgumentParser(description="은수 BPE 비교 예비 실험")
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("research/results/pilot.json")
    )
    args = parser.parse_args()
    data = (
        json.loads(args.input.read_text(encoding="utf-8"))
        if args.input
        else pilot_data()
    )
    result = experiment(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "condition",
        "pair",
        "language",
        "text",
        "token_count",
        "characters",
        "utf8_bytes",
        "tokens_per_character",
        "roundtrip",
    ]
    with args.output.with_suffix(".csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])
    for condition in result["conditions"]:
        print(condition)
    print(f"JSON과 CSV 저장: {args.output}")


if __name__ == "__main__":
    main()
