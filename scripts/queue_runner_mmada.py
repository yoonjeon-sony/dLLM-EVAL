#!/usr/bin/env python3
"""MMaDA orchestrator — parallel of scripts/queue_runner.py for the mmada model.

Mirrors `run_mmada.sh`'s invocation:

  python -m lmms_eval --model mmada \
    --model_args pretrained=$CKPT,gen_img_dir=…,chat_mode=image_gen \
    --tasks <task> --gen_kwargs prefix_lm=True --log_samples \
    --log_samples_suffix mmada --output_path …

Writes to a separate lock file (`scheduled_tasks_mmada.lock`) and per-cell log
dir (`_queue_logs_mmada/`) so it can run concurrently with the llava_llada
orchestrator (different GPU pool — set CUDA_VISIBLE_DEVICES at launch to carve
the GPUs between the two queues).

Usage
-----
  python scripts/queue_runner_mmada.py
  DRY_RUN=1 python scripts/queue_runner_mmada.py
  QUEUE_FILTER_TASKS=blink_jigsaw QUEUE_FILTER_CKPTS=MMaDA-PM \
      python scripts/queue_runner_mmada.py
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/queue_runner_mmada.py   # half the GPUs
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
    "/scratch2/yoonjeon.kim/.claude/scheduled_tasks_mmada.lock",
)
os.environ.setdefault(
    "QUEUE_LOG_DIR",
    "/scratch2/yoonjeon.kim/outputs/_queue_logs_mmada",
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import queue_lib as ql                       # noqa: E402
import rescore_all                            # noqa: E402

# ──────────────────────────── configuration ────────────────────────────

CKPTS = [
    # (label, path or HF repo)
    ("MMaDA-PM",                 "tyfeld/MMaDA-Parallel-M"),
    ("sft-PM-zebracot",          "yjyjyj98/sft_MMaDA-PM-thinkmorph_zebracot-ckpt8000"),
    ("answer-MMaDA-ckpt50",      "yjyjyj98/thinkmorph_answer-MMaDA-ckpt50"),
    ("edit-MMaDA-ckpt50",        "yjyjyj98/thinkmorph_edit-MMaDA-ckpt50"),
    ("Unified-MMaDA-cp50",       "/scratch2/yoonjeon.kim/rl-mmadaMixCoT-thinkmorph/thinkmorph_interleave-Unified-MMaDA-MixCoT/checkpoint-50"),
]

TASKS = [
    "mmvet", "mmstar", "mmmu_val", "vstar_bench", "cv_bench_reasoning",
    "chartqa", "blink_jigsaw",
]

# Per-task TEXT_BATCH_BUDGET overrides (same rationale as the llava_llada
# orchestrator — mmmu_val tends to OOM at the global default).
TASK_TEXT_BATCH_BUDGET = {
    "mmmu_val": 2 ** 14,
}

CHAT_MODE   = os.environ.get("CHAT_MODE", "image_gen")
USE_BBOX    = os.environ.get("USE_BBOX", "False")          # parity with run_mmada.sh
BASE_OUT    = Path(os.environ.get("BASE_OUT", "/scratch2/yoonjeon.kim/outputs"))
DEFAULT_ROOT = BASE_OUT / f"mmada_{CHAT_MODE}_usebbox{USE_BBOX}"
LOG_DIR     = ql.LOG_DIR
TMP_DIR     = Path("/tmp/queue_runner_mmada")
SUMMARY_EVERY = int(os.environ.get("QUEUE_SUMMARY_EVERY_S", 1800))
TICK_S        = int(os.environ.get("QUEUE_TICK_S", 15))
DRY_RUN       = os.environ.get("DRY_RUN", "").lower() in ("1", "true")
SLACK_MUTE    = os.environ.get("SLACK_MUTE", "").lower() in ("1", "true")
TEXT_BUDGET   = os.environ.get("TEXT_BATCH_BUDGET", "32768")

FILTER_TASKS  = set(filter(None, os.environ.get("QUEUE_FILTER_TASKS", "").split(",")))
FILTER_CKPTS  = set(filter(None, os.environ.get("QUEUE_FILTER_CKPTS", "").split(",")))

LMMS_EVAL_BIN = ["python", "-u", "-m", "lmms_eval"]


# ──────────────────────────── cell construction ────────────────────────────

def ckpt_dirname(ckpt_path: str) -> str:
    """Mirror MODEL_NAME from run_mmada.sh: basename(dirname)-basename."""
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

    # Mirror run_mmada.sh's --model_args: pretrained, gen_img_dir, chat_mode.
    # No conv_template / use_bbox — those are llava_llada-only.
    model_args = (
        f"pretrained={cell['ckpt_path']},"
        f"gen_img_dir={gen_img},"
        f"chat_mode={CHAT_MODE}"
    )

    argv = LMMS_EVAL_BIN + [
        "--model", "mmada",
        "--model_args", model_args,
        "--tasks", cell["task"],
        "--gen_kwargs", "prefix_lm=True",
        "--log_samples",
        "--log_samples_suffix", "mmada",
        "--output_path", str(out_dir),
        "--wandb_args", f"project=lmms-eval,job_type=eval,name=mmada_queue_{cell['id']}",
    ]

    env = os.environ.copy()
    # Single-process mode: unset distributed env vars so accelerate doesn't try
    # to init torch.distributed.
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
              "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    budget = TASK_TEXT_BATCH_BUDGET.get(cell["task"], int(TEXT_BUDGET))
    env.update({
        "CUDA_VISIBLE_DEVICES":   str(gpu),
        "NOT_ALWASY_DO_2DPOOL":   "1",
        "DEBUG_PRINT_IMAGE_RES":  "1",
        "DEBUG_FIX_PADDING":      "1",
        "TEXT_BATCH_BUDGET":      str(budget),
        "TEXT_BATCH_SCALE":       str(cell["text_batch_scale"]),
    })

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cell['id']}.log"
    return argv, env, log_path


def launch(cell: dict, gpu: int) -> None:
    argv, env, log_path = build_command(cell, gpu)
    print(f"[mmada-launch] {cell['id']} on GPU {gpu} → {log_path}")
    print("              " + " ".join(shlex.quote(x) for x in argv))
    fh = log_path.open("ab")
    fh.write(f"\n=== queue_runner_mmada launch at {ql.utc_now()} | gpu={gpu} | scale={cell['text_batch_scale']} ===\n".encode())
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
        f"▶ [mmada] `{cell['ckpt_label']}` · `{cell['task']}` on GPU {gpu} (pid {proc.pid}) — log `{log_path.name}`, scale={cell['text_batch_scale']}",
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
            f"✅ [mmada] `{cell['ckpt_label']}` · `{cell['task']}` — N={stats['n'] if stats else '?'}, "
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
                f"♻ [mmada] `{cell['ckpt_label']}` · `{cell['task']}` OOM, retry "
                f"{cell['retries']}/{cell['max_retries']} with TEXT_BATCH_SCALE={cell['text_batch_scale']}",
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
        f"❌ [mmada] `{cell['ckpt_label']}` · `{cell['task']}` failed ({cell['status']})\n"
        f"```\n{excerpt[-1500:]}\n```",
        mute=SLACK_MUTE,
    )
    _safe_release(cell, pool, all_cells)
    return True


# ──────────────────────────── report writing ────────────────────────────

REPORT_PATH = REPO / "report_all_tasks_mmada.md"


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

    out = ["# MMaDA queue progress (live)", ""]
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
    return "\n".join(out)


def write_report(cells: list[dict]) -> None:
    """Write a self-contained mmada report with just the live matrix + ETA.
    Body per-ROOT tables from rescore_all.py are shared with the llava_llada
    queue (they live in `report_all_tasks.md`); this report is just the mmada
    queue's live status snapshot."""
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
        f"⏳ [mmada] Queue: {n_done}/{len(cells)} done · {n_running} running · {n_err} errored · ETA ≈ {ql.hms(eta)}",
    )


def main() -> int:
    target_cells = build_cells()
    lock = ql.load_lock()
    existing = lock.get("cells", [])
    cells = merge_cells(existing, target_cells)
    skipped = mark_already_done(cells)
    print(f"[mmada-queue] {len(cells)} cells, {skipped} already complete")

    lock = {
        "version":     1,
        "model":       "mmada",
        "started_at":  lock.get("started_at") or ql.utc_now(),
        "updated_at":  ql.utc_now(),
        "cells":       cells,
    }
    ql.save_lock(lock)
    write_report(cells)

    if DRY_RUN:
        print("[mmada-queue] DRY_RUN — exiting after lock + report.")
        return 0

    gpus = ql.detect_gpus()
    if not gpus:
        print("[mmada-queue] no GPUs visible. Exiting.")
        return 1
    pool = ql.GPUPool(gpus)
    for c in cells:
        if c["status"] == "running" and ql.pid_alive(c.get("pid")) and c.get("gpu_id") is not None:
            pool.reserve(c["gpu_id"])
    for c in cells:
        if c["status"] == "running" and not ql.pid_alive(c.get("pid")):
            print(f"[mmada-queue] startup-finalize dead cell {c['id']} (pid {c.get('pid')})")
            finalize(c, pool, cells)
    ql.save_lock(lock)
    write_report(cells)
    print(f"[mmada-queue] GPU pool: {gpus} (free={pool.free}, busy={pool.busy})")

    ql.slack_post(
        f"🚀 [mmada] queue_runner started on `{ql.host()}` — "
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
            f"🏁 [mmada] queue_runner finished — {n_done}/{len(cells)} done · {n_err} errored",
            mute=SLACK_MUTE,
        )
    except KeyboardInterrupt:
        ql.slack_post("⚠ [mmada] queue_runner interrupted (Ctrl-C). Running cells left in place.", mute=SLACK_MUTE)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
