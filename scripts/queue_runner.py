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
    # (label, path or HF repo, allowed_tasks or None for "all TASKS")
    ("Unified-cp50",          "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-Unified-LavidaO/checkpoint-50", None),
    ("region-edit-cp50",      "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-region-edit-LavidaO/checkpoint-50", None),
    ("answer-LavidaO-ckpt50", "yjyjyj98/thinkmorph_answer-LavidaO-ckpt50", None),
    ("edit-LavidaO-ckpt50",   "yjyjyj98/thinkmorph_edit-LavidaO-ckpt50", None),
    ("interleave-cp50",       "/scratch2/yoonjeon.kim/rl-lavidao-thinkmorph/thinkmorph_interleave-LavidaO/checkpoint-50", None),
    # Newly queued at user's request: only the 4 reasoning-style tasks for these.
    ("LaViDa-O",              "/scratch2/yoonjeon.kim/LaViDa-O",
        ["mmstar", "mmmu_val", "cv_bench_reasoning", "vstar_bench"]),
    ("sft-zebracot",          "/scratch2/yoonjeon.kim/sft_LaViDa-O-thinkmorph_zebracot-step9000",
        ["mmstar", "mmmu_val", "cv_bench_reasoning", "vstar_bench"]),
]

TASKS = [
    "mmvet", "mmstar", "mmmu_val", "vstar_bench", "cv_bench_reasoning",
    "chartqa", "blink_jigsaw",
    # VisualPuzzles_cot removed at user's request 2026-04-26 — too long-running
    # for the current queue, and an aborted run is preferable to wasting
    # compute on it.
]

# Per-task TEXT_BATCH_BUDGET overrides. mmmu_val OOM'd with the global default
# of 2**15 / max_new_tokens — halve the budget for that task so each text batch
# is smaller and fits in 143 GB H200 memory.
TASK_TEXT_BATCH_BUDGET = {
    "mmmu_val": 2 ** 11,   # 2048 tokens of budget → text_batch_size = 2 at max_new_tokens=1024.
                           # batch=4 still OOM'd even with --num_processes=2 (DP doesn't reduce
                           # per-rank memory pressure — each rank still allocates the full
                           # text-gen activations). Manual diagnostic at batch=2 ran cleanly
                           # with ~55 GB headroom on the H200/143GB.
}

# Per-task GPU count. Tasks with count > 1 are launched via `accelerate launch
# --num_processes=N`, halving the per-rank dataset slice and reducing peak
# memory pressure. Default is 1 (plain `python -m lmms_eval`).
TASK_GPU_COUNT = {
    "mmmu_val": 2,        # data-parallel across 2 GPUs to dodge OOMs at batch=4
}

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
    for entry in CKPTS:
        ck_label, ck_path = entry[0], entry[1]
        allowed = entry[2] if len(entry) > 2 else None
        ck_tasks = allowed if allowed else TASKS
        for task in ck_tasks:
            cells.append({
                "id":               f"{ck_label}__{task}",
                "ckpt_path":        ck_path,
                "ckpt_label":       ck_label,
                "task":             task,
                "status":           "pending",
                "pid":              None,
                "gpu_id":           None,                     # primary GPU (= gpu_ids[0]); kept for display + back-compat
                "gpu_ids":          [],                       # all GPUs reserved for this cell (1..N)
                "gpu_count":        TASK_GPU_COUNT.get(task, 1),
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


def cell_gpus(cell: dict) -> list[int]:
    """Return the cell's GPU id list, normalising legacy single-GPU records."""
    g = cell.get("gpu_ids") or ([] if cell.get("gpu_id") is None else [cell["gpu_id"]])
    return list(g)


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
    """Keep STATE for cells already in lock; refresh CONFIG fields from target.
    Add fresh cells from target. Drops cells that are no longer in target."""
    by_id = {c["id"]: c for c in existing}
    # Config fields (path-/policy-derived) — always taken from target so a
    # config change in queue_runner.py propagates on next restart.
    config_keys = ("gpu_count", "ckpt_path", "jsonl_path", "max_retries")
    out = []
    for t in target:
        if t["id"] in by_id:
            merged = {**by_id[t["id"]]}
            for k in config_keys:
                merged[k] = t[k]
            out.append(merged)
        else:
            out.append(t)
    return out


# ──────────────────────────── launch / monitor ────────────────────────────

def build_command(cell: dict, gpus: list[int]) -> tuple[list[str], dict, Path]:
    """Return (argv, env, log_path) for the lmms-eval invocation. When more
    than one GPU is allocated, use `accelerate launch --num_processes=N` so
    lmms-eval data-parallelises across the ranks (mirrors run_lmms-eval.sh
    NUM_GPUS>1 branch)."""
    assert gpus, "build_command needs at least one GPU"
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

    n = len(gpus)
    if n == 1:
        launch_cmd = LMMS_EVAL_BIN[:]
    else:
        # Random port to avoid collisions if multiple multi-GPU cells run concurrently.
        master_port = 10000 + os.getpid() % 50000 + n
        launch_cmd = [
            "accelerate", "launch",
            "--num_machines=1", "--machine_rank=0",
            f"--main_process_ip=127.0.0.1",
            f"--main_process_port={master_port}",
            f"--num_processes={n}",
            "-m", "lmms_eval",
        ]

    argv = launch_cmd + [
        "--model", "llava_llada",
        "--model_args", model_args,
        "--tasks", cell["task"],
        "--gen_kwargs", "prefix_lm=True",
        "--log_samples",
        "--log_samples_suffix", "llava_llada",
        "--output_path", str(out_dir),
        "--wandb_args", f"project=lmms-eval,job_type=eval,name=queue_{cell['id']}",
    ]

    env = os.environ.copy()
    # Mirror run_lmms-eval.sh's NUM_GPUS=1 branch: unset distributed env vars so
    # accelerate runs in single-process mode without torch.distributed rendezvous.
    # For NUM_GPUS>1 accelerate sets these itself via --main_process_port etc.
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
              "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    budget = TASK_TEXT_BATCH_BUDGET.get(cell["task"], int(TEXT_BUDGET))
    env.update({
        "CUDA_VISIBLE_DEVICES":   ",".join(str(g) for g in gpus),
        "NOT_ALWASY_DO_2DPOOL":   "1",
        "DEBUG_PRINT_IMAGE_RES":  "1",
        "DEBUG_FIX_PADDING":      "1",
        "TEXT_BATCH_BUDGET":      str(budget),
        "TEXT_BATCH_SCALE":       str(cell["text_batch_scale"]),
        # expandable_segments lets PyTorch reuse its reserved-but-unallocated
        # cache (~21 GB at the mmmu_val OOM site) for larger allocations.
        # The PyTorch OOM error itself recommends this for fragmentation-driven
        # failures (the 34.6 GB allocation that crashes mmmu_val is the same
        # size at batch=4 and batch=2, so it's not a text-batch-scaling issue).
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cell['id']}.log"
    return argv, env, log_path


def launch(cell: dict, gpus: list[int]) -> None:
    argv, env, log_path = build_command(cell, gpus)
    gpu_str = ",".join(str(g) for g in gpus)
    print(f"[launch] {cell['id']} on GPUs {gpu_str} → {log_path}")
    print("         " + " ".join(shlex.quote(x) for x in argv))
    fh = log_path.open("ab")
    fh.write(f"\n=== queue_runner launch at {ql.utc_now()} | gpus={gpu_str} | scale={cell['text_batch_scale']} ===\n".encode())
    fh.flush()
    proc = subprocess.Popen(
        argv, env=env, stdout=fh, stderr=fh,
        cwd=str(REPO), start_new_session=True,
    )
    cell["pid"]        = proc.pid
    cell["gpu_ids"]    = list(gpus)
    cell["gpu_id"]     = gpus[0]
    cell["status"]     = "running"
    cell["started_at"] = ql.utc_now()
    cell["log_path"]   = str(log_path)
    gpu_disp = gpu_str if len(gpus) > 1 else f"GPU {gpus[0]}"
    ql.slack_post(
        f"▶ `{cell['ckpt_label']}` · `{cell['task']}` on {gpu_disp} (pid {proc.pid}) — log `{log_path.name}`, scale={cell['text_batch_scale']}",
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


def _safe_release(cell: dict, pool: ql.GPUPool, all_cells: list[dict]) -> None:
    """Release each GPU held by this cell back to the pool, but only if no OTHER
    running cell is still holding it. Prevents double-allocation when one cell's
    finalize races with another cell's launch on the same GPU id."""
    my_gpus = cell_gpus(cell)
    if not my_gpus:
        return
    for gpu in my_gpus:
        others = [
            c for c in all_cells
            if c is not cell
            and c.get("status") == "running"
            and gpu in cell_gpus(c)
        ]
        if others:
            continue
        pool.release(gpu)


def finalize(cell: dict, pool: ql.GPUPool, all_cells: list[dict] | None = None) -> bool:
    """Process exited; classify result. Returns True if cell terminal.
    Pass `all_cells` so the GPU release can check for other current owners."""
    if all_cells is None:
        all_cells = [cell]
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
        _safe_release(cell, pool, all_cells)
        return True

    excerpt = "\n".join(ql.tail_lines(log_path, 30)) if log_path else ""
    is_oom = bool(log_path) and ql.looks_like_oom(ql.tail_lines(log_path, 200))
    if is_oom:
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
            _safe_release(cell, pool, all_cells)
            cell["gpu_id"]  = None
            cell["gpu_ids"] = []
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
    _safe_release(cell, pool, all_cells)
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
    rows = {entry[0] for entry in CKPTS}
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
    for entry in CKPTS:
        ck_label = entry[0]
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
                f"| `{c['ckpt_label']}` | `{c['task']}` | {','.join(str(g) for g in cell_gpus(c)) or '—'} | "
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
    # Re-attach pass 1: any cell with a still-alive PID keeps ALL its GPUs reserved.
    for c in cells:
        if c["status"] == "running" and ql.pid_alive(c.get("pid")):
            for g in cell_gpus(c):
                pool.reserve(g)
    # Re-attach pass 2: any cell whose PID is dead (orchestrator restart, manual
    # kill, OOM, crash, etc.) gets finalized BEFORE the dispatch loop. This
    # prevents a window where pool.release() in the OOM-retry path collides with
    # an already-allocated GPU.
    for c in cells:
        if c["status"] == "running" and not ql.pid_alive(c.get("pid")):
            print(f"[queue] startup-finalize dead cell {c['id']} (pid {c.get('pid')})")
            finalize(c, pool, cells)
    ql.save_lock(lock)
    write_report(cells)
    print(f"[queue] GPU pool: {gpus} (free={pool.free}, busy={pool.busy})")

    ql.slack_post(
        f"🚀 queue_runner started on `{ql.host()}` — "
        f"{len(cells)} cells, {skipped} already done, GPUs={gpus}",
        mute=SLACK_MUTE,
    )

    try:
        while True:
            # Dispatch (filters narrow which pending cells are picked up).
            # Multi-GPU cells (e.g. mmmu_val with gpu_count=2) need the full
            # bundle acquired atomically; if the pool can't satisfy that yet,
            # leave the cell pending and try again next tick.
            for c in cells:
                if c["status"] != "pending" or cell_filtered(c):
                    continue
                need = c.get("gpu_count", 1)
                if len(pool.free) < need:
                    continue
                acquired = [pool.acquire() for _ in range(need)]
                if any(g is None for g in acquired):
                    # shouldn't happen given the len check above, but be safe
                    for g in acquired:
                        if g is not None:
                            pool.release(g)
                    continue
                launch(c, acquired)
                ql.save_lock(lock)
                write_report(cells)

            # Monitor
            for c in cells:
                if c["status"] != "running":
                    continue
                update_progress(c)
                if not ql.pid_alive(c.get("pid")):
                    finalize(c, pool, cells)
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
