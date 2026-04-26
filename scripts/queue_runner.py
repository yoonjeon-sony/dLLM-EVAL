#!/usr/bin/env python3
"""Orchestrator: queues every (ckpt × task) cell, dispatches one GPU per cell,
monitors via tqdm log parsing, posts to Slack on state changes, and rescore +
reports after each completion. Crash-safe: all state lives in a JSON lock file
guarded by fcntl.flock.

Usage
-----
  # Normal run (uses Slack webhook, all GPUs, full queue)
  python scripts/queue_runner.py

  # Dry run — build lock from existing jsonls, write report, no launches
  DRY_RUN=1 python scripts/queue_runner.py

  # Subset filters (comma-separated)
  QUEUE_FILTER_TASKS=blink_jigsaw QUEUE_FILTER_CKPTS=Unified-cp50 \
      python scripts/queue_runner.py

  # Mute Slack
  SLACK_MUTE=1 python scripts/queue_runner.py
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import queue_lib as ql                       # noqa: E402
import rescore_all                            # noqa: E402

# ──────────────────────────── configuration ────────────────────────────

CKPTS = [
    # (label, full path or HF repo)
    ("Unified-cp50",          "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-Unified-LavidaO/checkpoint-50"),
    ("region-edit-cp50",      "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-region-edit-LavidaO/checkpoint-50"),
    ("answer-LavidaO-ckpt50", "yjyjyj98/thinkmorph_answer-LavidaO-ckpt50"),
    ("edit-LavidaO-ckpt50",   "yjyjyj98/thinkmorph_edit-LavidaO-ckpt50"),
    ("interleave-cp50",       "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-LavidaO/checkpoint-50"),
]

TASKS = [
    "mmvet", "mmstar", "mmmu_val", "vstar_bench", "cv_bench_reasoning",
    "chartqa", "blink_jigsaw", "VisualPuzzles_cot",
]

BASE_OUT      = Path("/scratch2/yoonjeon.kim/outputs")
DEFAULT_ROOT  = BASE_OUT / "image_gen_usebboxFalse_default"
LOG_DIR       = ql.LOG_DIR
TMP_DIR       = Path("/tmp/queue_runner")
SUMMARY_EVERY = int(os.environ.get("QUEUE_SUMMARY_EVERY_S", 1800))   # 30 min
TICK_S        = int(os.environ.get("QUEUE_TICK_S", 15))
DRY_RUN       = os.environ.get("DRY_RUN", "").lower() in ("1", "true")
SLACK_MUTE    = os.environ.get("SLACK_MUTE", "").lower() in ("1", "true")
TEXT_BUDGET   = os.environ.get("TEXT_BATCH_BUDGET", "32768")

# Optional subset filters
FILTER_TASKS  = set(filter(None, os.environ.get("QUEUE_FILTER_TASKS", "").split(",")))
FILTER_CKPTS  = set(filter(None, os.environ.get("QUEUE_FILTER_CKPTS", "").split(",")))

# blink_jigsaw override prompt
BLINK_PRE_PROMPT = "Think step-by-step and answer the question. "

LMMS_EVAL_BIN = ["python", "-u", "-m", "lmms_eval"]


# ──────────────────────────── cell construction ────────────────────────────

def ckpt_dirname(ckpt_path: str) -> str:
    """Mirror MODEL_NAME from run_lmms-eval.sh: basename(dirname)-basename."""
    p = ckpt_path.rstrip("/")
    parent = os.path.basename(os.path.dirname(p))
    leaf   = os.path.basename(p)
    return f"{parent}-{leaf}"


def jsonl_path_for(ckpt_path: str, task: str) -> Path:
    return DEFAULT_ROOT / ckpt_dirname(ckpt_path) / f"{task}.jsonl"


def cell_filtered(c: dict) -> bool:
    """True iff this cell is excluded by FILTER_* env vars (filters only narrow dispatch)."""
    if FILTER_CKPTS and c["ckpt_label"] not in FILTER_CKPTS:
        return True
    if FILTER_TASKS and c["task"] not in FILTER_TASKS:
        return True
    return False


def build_cells() -> list[dict]:
    cells = []
    for ck_label, ck_path in CKPTS:
        for task in TASKS:
            cells.append({
                "id":               f"{ck_label}__{task}",
                "ckpt_path":        ck_path,
                "ckpt_label":       ck_label,
                "task":             task,
                "status":           "pending",
                "pid":              None,
                "gpu_id":           None,
                "started_at":       None,
                "ended_at":         None,
                "retries":          0,
                "max_retries":      2,
                "text_batch_scale": 1.0,
                "log_path":         None,
                "jsonl_path":       str(jsonl_path_for(ck_path, task)),
                "robust_score":     None,
                "n_records":        None,
                "tqdm_elapsed_s":   None,
                "tqdm_remaining_s": None,
                "tqdm_pct":         None,
                "error_excerpt":    None,
            })
    return cells


def mark_already_done(cells: list[dict]) -> int:
    n = 0
    for c in cells:
        if c["status"] == "done":
            continue
        p = Path(c["jsonl_path"])
        if p.exists() and p.stat().st_size > 0:
            c["status"] = "done"
            c["ended_at"] = c["ended_at"] or ql.utc_now()
            stats = rescore_all.score_jsonl_path(p, c["task"])
            if stats:
                c["robust_score"] = stats["robust_acc"]
                c["n_records"]    = stats["n"]
            n += 1
    return n


def merge_cells(existing: list[dict], target: list[dict]) -> list[dict]:
    """Keep state for cells already in lock; add new cells from target."""
    by_id = {c["id"]: c for c in existing}
    out = []
    for t in target:
        if t["id"] in by_id:
            out.append(by_id[t["id"]])
        else:
            out.append(t)
    return out


# ──────────────────────────── launch / monitor ────────────────────────────

def write_blink_overlay(job_dir: Path) -> Path:
    """Write an --include_path overlay yaml that swaps blink_jigsaw's pre_prompt."""
    job_dir.mkdir(parents=True, exist_ok=True)
    yaml_text = f"""# Generated by scripts/queue_runner.py for the queue
include: {REPO}/lmms-eval/lmms_eval/tasks/blink/_default_template_yaml
dataset_name: "Jigsaw"
tag: "blink_groups"
task: "blink_jigsaw"
test_split: val
lmms_eval_specific_kwargs:
  default:
    pre_prompt: "{BLINK_PRE_PROMPT}"
    post_prompt: ""
"""
    path = job_dir / "blink_jigsaw.yaml"
    path.write_text(yaml_text)
    return path


def build_command(cell: dict, gpu: int) -> tuple[list[str], dict, Path]:
    """Return (argv, env, log_path) for the lmms-eval invocation."""
    job_dir = TMP_DIR / cell["id"]
    job_dir.mkdir(parents=True, exist_ok=True)

    out_dir = DEFAULT_ROOT / ckpt_dirname(cell["ckpt_path"])
    gen_img = out_dir / "gen_imgs"

    model_args = (
        f"pretrained={cell['ckpt_path']},"
        f"conv_template=llada,"
        f"model_name=llava_llada,"
        f"chat_mode=image_gen,"
        f"use_bbox=False,"
        f"gen_img_dir={gen_img}"
    )

    argv = LMMS_EVAL_BIN + [
        "--model", "llava_llada",
        "--model_args", model_args,
        "--tasks", cell["task"],
        "--gen_kwargs", "prefix_lm=True",
        "--log_samples",
        "--log_samples_suffix", "llava_llada",
        "--output_path", str(out_dir),
        "--wandb_args", f"project=lmms-eval,job_type=eval,name=queue_{cell['id']}",
    ]

    if cell["task"] == "blink_jigsaw":
        write_blink_overlay(job_dir)
        argv += ["--include_path", str(job_dir)]

    env = os.environ.copy()
    # Mirror run_lmms-eval.sh's NUM_GPUS=1 branch: unset distributed env vars so
    # accelerate runs in single-process mode without torch.distributed rendezvous.
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
              "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    env.update({
        "CUDA_VISIBLE_DEVICES":   str(gpu),
        "NOT_ALWASY_DO_2DPOOL":   "1",
        "DEBUG_PRINT_IMAGE_RES":  "1",
        "DEBUG_FIX_PADDING":      "1",
        "TEXT_BATCH_BUDGET":      TEXT_BUDGET,
        "TEXT_BATCH_SCALE":       str(cell["text_batch_scale"]),
    })

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cell['id']}.log"
    return argv, env, log_path


def launch(cell: dict, gpu: int) -> None:
    argv, env, log_path = build_command(cell, gpu)
    print(f"[launch] {cell['id']} on GPU {gpu} → {log_path}")
    print("         " + " ".join(shlex.quote(x) for x in argv))
    fh = log_path.open("ab")
    fh.write(f"\n=== queue_runner launch at {ql.utc_now()} | gpu={gpu} | scale={cell['text_batch_scale']} ===\n".encode())
    fh.flush()
    proc = subprocess.Popen(
        argv, env=env, stdout=fh, stderr=fh,
        cwd=str(REPO), start_new_session=True,
    )
    cell["pid"]        = proc.pid
    cell["gpu_id"]     = gpu
    cell["status"]     = "running"
    cell["started_at"] = ql.utc_now()
    cell["log_path"]   = str(log_path)
    ql.slack_post(
        f"▶ `{cell['ckpt_label']}` · `{cell['task']}` on GPU {gpu} (pid {proc.pid}) — log `{log_path.name}`, scale={cell['text_batch_scale']}",
        mute=SLACK_MUTE,
    )


def update_progress(cell: dict) -> None:
    if not cell["log_path"]:
        return
    parsed = ql.parse_tqdm_tail(Path(cell["log_path"]))
    if parsed:
        cell["tqdm_elapsed_s"]   = parsed.get("elapsed_s")
        cell["tqdm_remaining_s"] = parsed.get("remaining_s")
        cell["tqdm_pct"]         = parsed.get("pct")


def finalize(cell: dict, pool: ql.GPUPool) -> bool:
    """Process exited; classify result. Returns True if cell terminal."""
    log_path = Path(cell["log_path"]) if cell["log_path"] else None
    jsonl    = Path(cell["jsonl_path"])
    cell["ended_at"] = ql.utc_now()

    if jsonl.exists() and jsonl.stat().st_size > 0:
        cell["status"] = "done"
        stats = rescore_all.score_jsonl_path(jsonl, cell["task"])
        if stats:
            cell["robust_score"] = stats["robust_acc"]
            cell["n_records"]    = stats["n"]
        ql.slack_post(
            f"✅ `{cell['ckpt_label']}` · `{cell['task']}` — N={stats['n'] if stats else '?'}, "
            f"robust acc={(cell['robust_score'] or 0)*100:.2f}%",
            mute=SLACK_MUTE,
        )
        if cell["gpu_id"] is not None:
            pool.release(cell["gpu_id"])
        return True

    excerpt = "\n".join(ql.tail_lines(log_path, 30)) if log_path else ""
    if log_path and ql.looks_like_oom(ql.tail_lines(log_path, 200)):
        if cell["retries"] < cell["max_retries"]:
            cell["retries"]         += 1
            cell["text_batch_scale"] *= 0.5
            cell["status"]           = "pending"
            cell["pid"]              = None
            cell["started_at"]       = None
            cell["ended_at"]         = None
            cell["error_excerpt"]    = "OOM — auto-retry"
            ql.slack_post(
                f"♻ `{cell['ckpt_label']}` · `{cell['task']}` OOM, retry "
                f"{cell['retries']}/{cell['max_retries']} with TEXT_BATCH_SCALE={cell['text_batch_scale']}",
                mute=SLACK_MUTE,
            )
            if cell["gpu_id"] is not None:
                pool.release(cell["gpu_id"])
                cell["gpu_id"] = None
            return False     # not terminal — back to pending
        cell["status"] = "error_oom"
    else:
        cell["status"] = "error_other"

    cell["error_excerpt"] = excerpt[-3000:]
    ql.slack_post(
        f"❌ `{cell['ckpt_label']}` · `{cell['task']}` failed ({cell['status']})\n"
        f"```\n{excerpt[-1500:]}\n```",
        mute=SLACK_MUTE,
    )
    if cell["gpu_id"] is not None:
        pool.release(cell["gpu_id"])
    return True


# ──────────────────────────── report writing ────────────────────────────

STATUS_GLYPH = {
    "pending":     "⏳",
    "running":     "🏃",
    "done":        "✅",
    "error_oom":   "♻❌",
    "error_other": "❌",
    "skipped":     "·",
}


def render_top_section(cells: list[dict]) -> str:
    rows = {ck for ck, _ in CKPTS}
    by   = {(c["ckpt_label"], c["task"]): c for c in cells}
    n_done    = sum(1 for c in cells if c["status"] == "done")
    n_running = sum(1 for c in cells if c["status"] == "running")
    n_err     = sum(1 for c in cells if c["status"].startswith("error"))
    n_pend    = sum(1 for c in cells if c["status"] == "pending")
    total     = len(cells)
    eta_remaining = sum(
        (c["tqdm_remaining_s"] or 0)
        for c in cells if c["status"] == "running"
    )

    out = ["", "## Queue progress (live)", ""]
    out.append(
        f"Updated {ql.utc_now()}. **{n_done} / {total}** done · "
        f"{n_running} running · {n_pend} pending · {n_err} errored. "
        f"Aggregate running ETA ≈ **{ql.hms(eta_remaining)}**."
    )
    out.append("")
    out.append("|             | " + " | ".join(f"`{t}`" for t in TASKS) + " |")
    out.append("|---|" + "|".join([":-:"] * len(TASKS)) + "|")
    for ck_label, _ in CKPTS:
        if ck_label not in rows:
            continue
        cells_in_row = [by.get((ck_label, t)) for t in TASKS]
        line = [f"`{ck_label}`"]
        for c in cells_in_row:
            if c is None:
                line.append("·")
            elif c["status"] == "done":
                line.append(f"✅ {(c['robust_score'] or 0)*100:.2f}%")
            elif c["status"] == "running":
                pct = f"{c['tqdm_pct']:.0f}%" if c["tqdm_pct"] is not None else "—"
                line.append(f"🏃 {pct}")
            elif c["status"] == "pending":
                line.append("⏳")
            elif c["status"] == "error_oom":
                line.append(f"♻ retry{c['retries']}")
            elif c["status"] == "error_other":
                line.append("❌")
            else:
                line.append(c["status"])
        out.append("| " + " | ".join(line) + " |")
    out.append("")
    out.append("Legend: ✅ done · 🏃 running (% from tqdm) · ⏳ pending · ♻ retrying · ❌ failed")
    out.append("")

    # Per-job ETA table
    running = [c for c in cells if c["status"] == "running"]
    if running:
        out.append("### Per-job ETA")
        out.append("")
        out.append("| ckpt | task | gpu | started | elapsed | tqdm % | remaining (ETA) |")
        out.append("|---|---|---:|---|---:|---:|---:|")
        for c in running:
            elapsed = ql.hms(c["tqdm_elapsed_s"])
            remain  = ql.hms(c["tqdm_remaining_s"])
            pct     = f"{c['tqdm_pct']:.0f}%" if c["tqdm_pct"] is not None else "—"
            out.append(
                f"| `{c['ckpt_label']}` | `{c['task']}` | {c['gpu_id']} | "
                f"{c['started_at'] or '—'} | {elapsed} | {pct} | {remain} |"
            )
        out.append("")
    out.append("---")
    out.append("")
    return "\n".join(out)


def write_report(cells: list[dict]) -> None:
    """Rebuild the full report: top section (live matrix + ETA) + body tables."""
    matrix = rescore_all.build_full_matrix()
    rescore_all.write_aggregate_report(matrix, prefix_md=render_top_section(cells))


# ──────────────────────────── main loop ────────────────────────────

_last_summary_ts = 0.0


def maybe_summary(cells: list[dict]) -> None:
    global _last_summary_ts
    if SLACK_MUTE: return
    now = time.time()
    if now - _last_summary_ts < SUMMARY_EVERY:
        return
    _last_summary_ts = now
    n_done    = sum(1 for c in cells if c["status"] == "done")
    n_running = sum(1 for c in cells if c["status"] == "running")
    n_err     = sum(1 for c in cells if c["status"].startswith("error"))
    eta = sum((c["tqdm_remaining_s"] or 0) for c in cells if c["status"] == "running")
    ql.slack_post(
        f"⏳ Queue: {n_done}/{len(cells)} done · {n_running} running · {n_err} errored · ETA ≈ {ql.hms(eta)}",
    )


def main() -> int:
    target_cells = build_cells()
    lock = ql.load_lock()
    existing = lock.get("cells", [])
    cells = merge_cells(existing, target_cells)
    skipped = mark_already_done(cells)
    print(f"[queue] {len(cells)} cells, {skipped} already complete")

    lock = {
        "version":     1,
        "started_at":  lock.get("started_at") or ql.utc_now(),
        "updated_at":  ql.utc_now(),
        "cells":       cells,
    }
    ql.save_lock(lock)
    write_report(cells)

    if DRY_RUN:
        print("[queue] DRY_RUN — exiting after lock + report.")
        return 0

    gpus = ql.detect_gpus()
    if not gpus:
        print("[queue] no GPUs visible. Exiting.")
        return 1
    pool = ql.GPUPool(gpus)
    # Re-attach: any cell already running with valid pid stays running and reserves its GPU
    for c in cells:
        if c["status"] == "running" and ql.pid_alive(c.get("pid")) and c.get("gpu_id") is not None:
            pool.reserve(c["gpu_id"])
    print(f"[queue] GPU pool: {gpus} (free={pool.free}, busy={pool.busy})")

    ql.slack_post(
        f"🚀 queue_runner started on `{ql.host()}` — "
        f"{len(cells)} cells, {skipped} already done, GPUs={gpus}",
        mute=SLACK_MUTE,
    )

    try:
        while True:
            # Dispatch (filters narrow which pending cells are picked up)
            for c in cells:
                if c["status"] != "pending" or cell_filtered(c):
                    continue
                gpu = pool.acquire()
                if gpu is None:
                    break
                launch(c, gpu)
                ql.save_lock(lock)
                write_report(cells)

            # Monitor
            for c in cells:
                if c["status"] != "running":
                    continue
                update_progress(c)
                if not ql.pid_alive(c.get("pid")):
                    finalize(c, pool)
                    ql.save_lock(lock)
                    write_report(cells)

            # Heartbeat
            ql.save_lock(lock)
            write_report(cells)
            maybe_summary(cells)

            # Termination — finished if every IN-FILTER cell is terminal
            in_scope = [c for c in cells if not cell_filtered(c)]
            if all(c["status"] in ("done", "error_oom", "error_other", "skipped") for c in in_scope):
                break
            time.sleep(TICK_S)

        n_done = sum(1 for c in cells if c["status"] == "done")
        n_err  = sum(1 for c in cells if c["status"].startswith("error"))
        ql.slack_post(
            f"🏁 queue_runner finished — {n_done}/{len(cells)} done · {n_err} errored",
            mute=SLACK_MUTE,
        )
    except KeyboardInterrupt:
        ql.slack_post("⚠ queue_runner interrupted (Ctrl-C). Running cells left in place.", mute=SLACK_MUTE)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
