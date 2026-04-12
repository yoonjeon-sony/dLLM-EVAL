#!/usr/bin/env python3

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path(
    # "/home/yoonjeon.kim/dLLM-RL/train_sft/outputs/image_gen_usebboxTrue_tok512_blk256_step256_t0"
    "/home/yoonjeon.kim/dLLM-RL/train_sft/outputs/image_gen_usebboxFalse_tok512_blk256_step256_t0"
    # "/home/yoonjeon.kim/dLLM-RL/train_sft/testing/image_gen_usebboxFalse_tok512_blk256_step256_t0"
)


def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def is_correct(extracted: str, target: str) -> bool:
    extracted_norm = normalize_text(extracted)
    target_norm = normalize_text(target)
    if not extracted_norm or not target_norm:
        return False
    return extracted_norm == target_norm or extracted_norm in target_norm


@dataclass
class FileStats:
    path: Path
    total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def evaluate_file(path: Path, write_details: bool) -> FileStats:
    stats = FileStats(path=path)
    details_path = path.with_name(f"{path.stem}_xml_eval.jsonl")
    if "VisPuzzle" in path.stem:
        target_map = {
            "Part 1 should be to the right of Part 2": "A",
            "Part 1 should be to the left of Part 2": "B",
        }
    writer = details_path.open("w", encoding="utf-8") if write_details else None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                filtered_resps = coerce_text(row["resps"][0][0]["text_gen_output"])
                target = coerce_text(row.get("target", ""))
                if "VisPuzzle" in path.stem:
                    target = target_map.get(target, target)

                extracted = extract_xml_answer(filtered_resps)
                if extracted == "":
                    extracted = filtered_resps
                correct = is_correct(extracted, target)

                stats.total += 1
                stats.correct += int(correct)

                if writer is not None:
                    detail = {
                        "doc_id": row.get("doc_id"),
                        "line_number": line_number,
                        "target": target,
                        "filtered_resps": filtered_resps,
                        "xml_extracted": extracted,
                        "is_correct": correct,
                    }
                    writer.write(json.dumps(detail, ensure_ascii=True) + "\n")
    finally:
        if writer is not None:
            writer.close()

    return stats


def collect_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("*/*.jsonl")
        if not path.name.endswith("_xml_eval.jsonl")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate filtered_resps XML answers against target for all JSONL files "
            "under outputs/eval_generate_logs/text_gen_tok512_blk256_step256_t0."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root directory to scan. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--write-details",
        action="store_true",
        help="Write per-row evaluation JSONL next to each input file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    files = collect_files(root)
    if not files:
        raise SystemExit(f"No JSONL files found under: {root}")

    overall_total = 0
    overall_correct = 0

    print(f"Evaluating {len(files)} files under {root}")
    for path in files:
        stats = evaluate_file(path, write_details=args.write_details)
        overall_total += stats.total
        overall_correct += stats.correct
        print(
            f"{path.relative_to(root)}: "
            f"{stats.correct}/{stats.total} "
            f"({stats.accuracy:.2%})"
        )

    overall_accuracy = (overall_correct / overall_total) if overall_total else 0.0
    print(
        f"OVERALL: {overall_correct}/{overall_total} "
        f"({overall_accuracy:.2%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
