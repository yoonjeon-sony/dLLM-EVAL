#!/usr/bin/env python3
"""Anole orchestrator — parallel of scripts/queue_runner.py for the anole model.

Mirrors `run_anole.sh`'s invocation:

  python -m lmms_eval --model anole \
    --model_args pretrained=$CKPT,gen_img_dir=…,temperature=…,max_seq_len=… \
    --batch_size $BATCH_SIZE \
    --tasks <task> --log_samples --log_samples_suffix anole --output_path …

Anole-specific contract (vs. lavida/mmada):
  - batch_size is a SEPARATE --batch_size CLI flag, never inside --model_args
    (the framework auto-injects it via create_from_arg_string() and would
    collide with a duplicate kwarg — see warning at top of run_anole.sh).
  - No --gen_kwargs prefix_lm=True (lavida/llada-only).
  - No chat_mode / use_bbox / conv_template / model_name in model_args.
  - The lavida-style env vars (NOT_ALWASY_DO_2DPOOL, DEBUG_FIX_PADDING,
    DEBUG_PRINT_IMAGE_RES, TEXT_BATCH_BUDGET, TEXT_BATCH_SCALE) are not read
    by the Anole code path, so they are intentionally omitted.

Writes to a separate lock file (`scheduled_tasks_anole.lock`) and per-cell log
dir (`_queue_logs_anole/`) so it can run concurrently with the llava_llada and
mmada orchestrators (different GPU pool — set CUDA_VISIBLE_DEVICES at launch
to carve the GPUs between the queues).

Usage
-----
  python scripts/queue_runner_anole.py
  DRY_RUN=1 python scripts/queue_runner_anole.py
  QUEUE_FILTER_TASKS=chartqa python scripts/queue_runner_anole.py
  CUDA_VISIBLE_DEVICES=0,1 python scripts/queue_runner_anole.py   # carve GPUs
  BATCH_SIZE=4 TEMPERATURE=0 python scripts/queue_runner_anole.py
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Set lock + log dir BEFORE importing queue_lib (it reads these at import time).
os.environ.setdefault(
    "QUEUE_LOCK_PATH",
    "/scratch2/yoonjeon.kim/.claude/scheduled_tasks_anole.lock",
)
os.environ.setdefault(
    "QUEUE_LOG_DIR",
    "/scratch2/yoonjeon.kim/outputs/_queue_logs_anole",
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import queue_lib as ql                       # noqa: E402
import rescore_all                            # noqa: E402

# ──────────────────────────── configuration ────────────────────────────

CKPTS = [
    ("Anole-7b-v0.1", "/scratch2/yoonjeon.kim/Anole-7b-v0.1"),
]

TASKS = [
    "mmvet", "mmstar", "mmmu_val", "vstar_bench", "cv_bench_reasoning",
    "chartqa", "blink_jigsaw",
]

# Per-task BATCH_SIZE overrides. Empty by default; populate after first run if
# any task OOMs (e.g. {"mmmu_val": 1} if mmmu_val refuses to fit at BATCH_SIZE).
TASK_BATCH_SIZE: dict[str, int] = {}

BASE_OUT     = Path(os.environ.get("BASE_OUT", "/scratch2/yoonjeon.kim/outputs"))
DEFAULT_ROOT = BASE_OUT / "anole"   # mirrors run_anole.sh's BASE_DIR/anole
LOG_DIR      = ql.LOG_DIR
TMP_DIR      = Path("/tmp/queue_runner_anole")
SUMMARY_EVERY = int(os.environ.get("QUEUE_SUMMARY_EVERY_S", 1800))
TICK_S        = int(os.environ.get("QUEUE_TICK_S", 15))
DRY_RUN       = os.environ.get("DRY_RUN", "").lower() in ("1", "true")
SLACK_MUTE    = os.environ.get("SLACK_MUTE", "").lower() in ("1", "true")

# Anole adapter knobs (see lmms-eval/lmms_eval/models/anole.py constructor).
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "1"))
TEMPERATURE = os.environ.get("TEMPERATURE", "0")          # "0" → greedy
MAX_SEQ_LEN = os.environ.get("MAX_SEQ_LEN", "4096")
CFG_IMAGE   = os.environ.get("CFG_IMAGE", "")             # optional, omit if blank
CFG_TEXT    = os.environ.get("CFG_TEXT", "")
LIMIT       = os.environ.get("LIMIT", "")                 # smoke-run cap; "" or "none" = full set

FILTER_TASKS  = set(filter(None, os.environ.get("QUEUE_FILTER_TASKS", "").split(",")))
FILTER_CKPTS  = set(filter(None, os.environ.get("QUEUE_FILTER_CKPTS", "").split(",")))

LMMS_EVAL_BIN = ["python", "-u", "-m", "lmms_eval"]


# ──────────────────────────── cell construction ────────────────────────────

def ckpt_dirname(ckpt_path: str) -> str:
    """Mirror MODEL_NAME from run_lmms-eval.sh: basename(dirname)-basename.

    Intentionally diverges from run_anole.sh's ${CKPT_NAME//\\//__} so jsonl
    paths line up with the lavida and mmada queue runners' convention."""
    p = ckpt_path.rstrip("/")
    parent = os.path.basename(os.path.dirname(p))
    leaf   = os.path.basename(p)
    return f"{parent}-{leaf}"


def jsonl_path_for(ckpt_path: str, task: str) -> Path:
    return DEFAULT_ROOT / ckpt_dirname(ckpt_path) / f"{task}.jsonl"


def cell_filtered(c: dict) -> bool:
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
                "batch_size":       TASK_BATCH_SIZE.get(task, BATCH_SIZE),
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
    by_id = {c["id"]: c for c in existing}
    out = []
    for t in target:
        if t["id"] in by_id:
            out.append(by_id[t["id"]])
        else:
            out.append(t)
    return out


# ──────────────────────────── launch / monitor ────────────────────────────

def build_command(cell: dict, gpu: int) -> tuple[list[str], dict, Path]:
    job_dir = TMP_DIR / cell["id"]
    job_dir.mkdir(parents=True, exist_ok=True)

    out_dir = DEFAULT_ROOT / ckpt_dirname(cell["ckpt_path"])
    gen_img = out_dir / "gen_imgs"

    # IMPORTANT: do NOT include batch_size= in model_args — it is passed via
    # the separate --batch_size CLI flag and would collide otherwise.
    parts = [
        f"pretrained={cell['ckpt_path']}",
        f"gen_img_dir={gen_img}",
        f"temperature={TEMPERATURE}",
        f"max_seq_len={MAX_SEQ_LEN}",
    ]
    if CFG_IMAGE:
        parts.append(f"cfg_image={CFG_IMAGE}")
    if CFG_TEXT:
        parts.append(f"cfg_text={CFG_TEXT}")
    model_args = ",".join(parts)

    argv = LMMS_EVAL_BIN + [
        "--model", "anole",
        "--model_args", model_args,
        "--batch_size", str(cell["batch_size"]),
        "--tasks", cell["task"],
        "--log_samples",
        "--log_samples_suffix", "anole",
        "--output_path", str(out_dir),
        "--wandb_args", f"project=lmms-eval,job_type=eval,name=anole_queue_{cell['id']}",
    ]
    if LIMIT and LIMIT.lower() != "none":
        argv += ["--limit", LIMIT]
    # NOTE: no --gen_kwargs prefix_lm=True (lavida/llada-only).

    env = os.environ.copy()
    # Single-process mode: unset distributed env vars so accelerate doesn't try
    # to init torch.distributed.
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
              "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Intentionally omit NOT_ALWASY_DO_2DPOOL / DEBUG_FIX_PADDING /
    # DEBUG_PRINT_IMAGE_RES / TEXT_BATCH_BUDGET / TEXT_BATCH_SCALE — those are
    # lavida/mmada-only and unread by the anole adapter.

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cell['id']}.log"
    return argv, env, log_path


def launch(cell: dict, gpu: int) -> None:
    argv, env, log_path = build_command(cell, gpu)
    print(f"[anole-launch] {cell['id']} on GPU {gpu} → {log_path}")
    print("              " + " ".join(shlex.quote(x) for x in argv))
    fh = log_path.open("ab")
    fh.write(f"\n=== queue_runner_anole launch at {ql.utc_now()} | gpu={gpu} | bs={cell['batch_size']} ===\n".encode())
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
        f"▶ [anole] `{cell['ckpt_label']}` · `{cell['task']}` on GPU {gpu} (pid {proc.pid}) — log `{log_path.name}`, bs={cell['batch_size']}",
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
    gpu = cell.get("gpu_id")
    if gpu is None:
        return
    others = [
        c for c in all_cells
        if c is not cell and c.get("status") == "running" and c.get("gpu_id") == gpu
    ]
    if others:
        return
    pool.release(gpu)


def finalize(cell: dict, pool: ql.GPUPool, all_cells: list[dict] | None = None) -> bool:
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
            f"✅ [anole] `{cell['ckpt_label']}` · `{cell['task']}` — N={stats['n'] if stats else '?'}, "
            f"robust acc={(cell['robust_score'] or 0)*100:.2f}%",
            mute=SLACK_MUTE,
        )
        _safe_release(cell, pool, all_cells)
        return True

    excerpt = "\n".join(ql.tail_lines(log_path, 30)) if log_path else ""
    is_oom = bool(log_path) and ql.looks_like_oom(ql.tail_lines(log_path, 200))
    if is_oom:
        # Halve batch_size (clamp to 1) and retry. If we're already at bs=1,
        # there's no smaller value to try → terminal.
        if cell["retries"] < cell["max_retries"] and cell["batch_size"] > 1:
            cell["retries"]      += 1
            cell["batch_size"]    = max(1, cell["batch_size"] // 2)
            cell["status"]        = "pending"
            cell["pid"]           = None
            cell["started_at"]    = None
            cell["ended_at"]      = None
            cell["error_excerpt"] = "OOM — auto-retry"
            ql.slack_post(
                f"♻ [anole] `{cell['ckpt_label']}` · `{cell['task']}` OOM, retry "
                f"{cell['retries']}/{cell['max_retries']} with batch_size={cell['batch_size']}",
                mute=SLACK_MUTE,
            )
            _safe_release(cell, pool, all_cells)
            cell["gpu_id"] = None
            return False
        cell["status"] = "error_oom"
    else:
        cell["status"] = "error_other"

    cell["error_excerpt"] = excerpt[-3000:]
    ql.slack_post(
        f"❌ [anole] `{cell['ckpt_label']}` · `{cell['task']}` failed ({cell['status']})\n"
        f"```\n{excerpt[-1500:]}\n```",
        mute=SLACK_MUTE,
    )
    _safe_release(cell, pool, all_cells)
    return True


# ──────────────────────────── report writing ────────────────────────────

REPORT_PATH = REPO / "report_all_tasks_anole.md"


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

    out = ["# Anole queue progress (live)", ""]
    out.append(
        f"Updated {ql.utc_now()}. **{n_done} / {total}** done · "
        f"{n_running} running · {n_pend} pending · {n_err} errored. "
        f"Aggregate running ETA ≈ **{ql.hms(eta_remaining)}**."
    )
    out.append(f"Output root: `{DEFAULT_ROOT}`")
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

    running = [c for c in cells if c["status"] == "running"]
    if running:
        out.append("## Per-job ETA")
        out.append("")
        out.append("| ckpt | task | gpu | bs | started | elapsed | tqdm % | remaining (ETA) |")
        out.append("|---|---|---:|---:|---|---:|---:|---:|")
        for c in running:
            elapsed = ql.hms(c["tqdm_elapsed_s"])
            remain  = ql.hms(c["tqdm_remaining_s"])
            pct     = f"{c['tqdm_pct']:.0f}%" if c["tqdm_pct"] is not None else "—"
            out.append(
                f"| `{c['ckpt_label']}` | `{c['task']}` | {c['gpu_id']} | {c['batch_size']} | "
                f"{c['started_at'] or '—'} | {elapsed} | {pct} | {remain} |"
            )
        out.append("")
    return "\n".join(out)


def write_report(cells: list[dict]) -> None:
    """Self-contained anole report: live matrix + ETA. The body per-ROOT tables
    in rescore_all are hard-coded for the lavida output tree, so we don't reuse
    write_aggregate_report here (same approach as queue_runner_mmada.py)."""
    REPORT_PATH.write_text(render_top_section(cells))


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
        f"⏳ [anole] Queue: {n_done}/{len(cells)} done · {n_running} running · {n_err} errored · ETA ≈ {ql.hms(eta)}",
    )


def main() -> int:
    target_cells = build_cells()
    lock = ql.load_lock()
    existing = lock.get("cells", [])
    cells = merge_cells(existing, target_cells)
    skipped = mark_already_done(cells)
    print(f"[anole-queue] {len(cells)} cells, {skipped} already complete")

    lock = {
        "version":     1,
        "model":       "anole",
        "started_at":  lock.get("started_at") or ql.utc_now(),
        "updated_at":  ql.utc_now(),
        "cells":       cells,
    }
    ql.save_lock(lock)
    write_report(cells)

    if DRY_RUN:
        print("[anole-queue] DRY_RUN — exiting after lock + report.")
        return 0

    gpus = ql.detect_gpus()
    if not gpus:
        print("[anole-queue] no GPUs visible. Exiting.")
        return 1
    pool = ql.GPUPool(gpus)
    for c in cells:
        if c["status"] == "running" and ql.pid_alive(c.get("pid")) and c.get("gpu_id") is not None:
            pool.reserve(c["gpu_id"])
    for c in cells:
        if c["status"] == "running" and not ql.pid_alive(c.get("pid")):
            print(f"[anole-queue] startup-finalize dead cell {c['id']} (pid {c.get('pid')})")
            finalize(c, pool, cells)
    ql.save_lock(lock)
    write_report(cells)
    print(f"[anole-queue] GPU pool: {gpus} (free={pool.free}, busy={pool.busy})")

    ql.slack_post(
        f"🚀 [anole] queue_runner started on `{ql.host()}` — "
        f"{len(cells)} cells, {skipped} already done, GPUs={gpus}",
        mute=SLACK_MUTE,
    )

    try:
        while True:
            for c in cells:
                if c["status"] != "pending" or cell_filtered(c):
                    continue
                gpu = pool.acquire()
                if gpu is None:
                    break
                launch(c, gpu)
                ql.save_lock(lock)
                write_report(cells)

            for c in cells:
                if c["status"] != "running":
                    continue
                update_progress(c)
                if not ql.pid_alive(c.get("pid")):
                    finalize(c, pool, cells)
                    ql.save_lock(lock)
                    write_report(cells)

            ql.save_lock(lock)
            write_report(cells)
            maybe_summary(cells)

            in_scope = [c for c in cells if not cell_filtered(c)]
            if all(c["status"] in ("done", "error_oom", "error_other", "skipped") for c in in_scope):
                break
            time.sleep(TICK_S)

        n_done = sum(1 for c in cells if c["status"] == "done")
        n_err  = sum(1 for c in cells if c["status"].startswith("error"))
        ql.slack_post(
            f"🏁 [anole] queue_runner finished — {n_done}/{len(cells)} done · {n_err} errored",
            mute=SLACK_MUTE,
        )
    except KeyboardInterrupt:
        ql.slack_post("⚠ [anole] queue_runner interrupted (Ctrl-C). Running cells left in place.", mute=SLACK_MUTE)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
