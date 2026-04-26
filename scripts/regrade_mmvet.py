#!/usr/bin/env python3
"""
Re-grade every mmvet.jsonl under /scratch2/yoonjeon.kim/outputs/image_gen_usebboxFalse_default
using lmms-eval's official GPT judge (mmvet_process_results), caching per-record
scores into <ckpt_dir>/mmvet.regraded.jsonl so that we don't re-pay for the API.

Set OPENAI_API_KEY in env before running.
"""
import json
import os
import sys
import time
from pathlib import Path

# Allow `import lmms_eval.tasks.mmvet.utils`
sys.path.insert(0, "/home/yoonjeon.kim/dLLM-EVAL/lmms-eval")

from lmms_eval.tasks.mmvet.utils import mmvet_process_results  # noqa: E402

ROOT = Path("/scratch2/yoonjeon.kim/outputs/image_gen_usebboxFalse_default")
PER_CALL_SLEEP = float(os.environ.get("REGRADE_SLEEP", "1.5"))  # seconds; avoids 429s on tight tiers


def regrade(jsonl_path: Path):
    out_path = jsonl_path.with_suffix(".regraded.jsonl")

    # Load anything already regraded — keyed by question_id for idempotency
    cached = {}
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cached[rec["question_id"]] = rec

    rows = []
    with jsonl_path.open() as f:
        for line in f:
            rows.append(json.loads(line))

    new_records = []
    for row in rows:
        doc = row["doc"]
        qid = doc["question_id"]
        if qid in cached:
            new_records.append(cached[qid])
            continue
        # Our resps is [[{...,"text_gen_output":"..."}]] — extract the pred string.
        pred = row["resps"][0][0]["text_gen_output"]
        out = mmvet_process_results(doc, [pred])
        rec = out["gpt_eval_score"]
        new_records.append(rec)
        # Append immediately — robust to interruption
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  graded {qid}: {rec['score']}", flush=True)
        # Throttle to avoid 429s — lmms-eval's internal retry only has patience=3
        if PER_CALL_SLEEP > 0:
            time.sleep(PER_CALL_SLEEP)

    # Compute overall and per-capability
    n = len(new_records)
    total = sum(r["score"] for r in new_records)
    overall = total / n if n else 0
    by_cap = {}
    for r in new_records:
        for cap in r["capabilities"].split(","):
            cap = cap.strip()
            if not cap:
                continue
            by_cap.setdefault(cap, []).append(r["score"])
    cap_scores = {c: sum(s) / len(s) for c, s in by_cap.items()}
    return {"n": n, "overall": overall, "by_cap": cap_scores, "out_path": str(out_path)}


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY in env first")
    summary = {}
    for jsonl in sorted(ROOT.glob("*/mmvet.jsonl")):
        ckpt = jsonl.parent.name
        print(f"\n=== {ckpt} === ({jsonl})", flush=True)
        s = regrade(jsonl)
        summary[ckpt] = s
        print(f"  overall={s['overall']:.4f}  n={s['n']}  cap={s['by_cap']}", flush=True)
    print("\n=== summary ===")
    for ckpt, s in summary.items():
        print(f"{ckpt}: {s['overall']*100:.2f}%  ({s['n']} samples)")


if __name__ == "__main__":
    main()
