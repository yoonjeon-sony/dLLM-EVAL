#!/usr/bin/env python3
"""MMaDA-Parallel-M (mmada_m) orchestrator — parallel of scripts/queue_runner.py.

Mirrors a stripped-down `run_mmadaM.sh` invocation for the M variant.
- model: mmada_m

Writes to a separate lock file and per-cell log dir so it can run alongside
the legacy mmada and llava_llada queues; carve GPUs via CUDA_VISIBLE_DEVICES.

Usage
-----
  python scripts/queue_runner_mmada.py
  DRY_RUN=1 python scripts/queue_runner_mmada.py
  LIMIT=2 QUEUE_FILTER_TASKS=mmvet QUEUE_FILTER_CKPTS=sft-zebracot-ckpt8000 \
      python scripts/queue_runner_mmada.py     # smoke test
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/queue_runner_mmada.py
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
    "/scratch2/yoonjeon.kim/.claude/scheduled_tasks_mmada_m.lock",
)
os.environ.setdefault(
    "QUEUE_LOG_DIR",
    "/scratch2/yoonjeon.kim/outputs/_queue_logs_mmada_m",
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import queue_lib as ql                       # noqa: E402
import rescore_all                           # noqa: E402

# ──────────────────────────── configuration ────────────────────────────

# (label, ckpt path or HF repo). Labels are used for QUEUE_FILTER_CKPTS,
# slack, and lock-file IDs; the on-disk output dir uses basename(ckpt).
CKPTS = [
    ("sft-zebracot-ckpt8000",       "yjyjyj98/sft_MMaDA-PM-thinkmorph_zebracot-ckpt8000"),
    ("MMaDA-Parallel-M",            "tyfeld/MMaDA-Parallel-M"),
    # ("answer-MMaDA-ckpt50",         "yjyjyj98/thinkmorph_answer-MMaDA-ckpt50"),
    # ("edit-MMaDA-ckpt50",           "yjyjyj98/thinkmorph_edit-MMaDA-ckpt50"),
    ("Separate-MMaDA-MixCoT-cp50",  "yjyjyj98/thinkmorph_interleave-MMaDA-MixCoT-ckpt50"),
    ("Unified-MMaDA-MixCoT-cp50",   "/scratch2/yoonjeon.kim/rl-mmadaMixCoT-thinkmorph/thinkmorph_interleave-Unified-MMaDA-MixCoT/checkpoint-50"),
]

# Order matters: build_cells() materializes cells in this order, so launches
# happen in this order too. mmmu_val is intentionally last; it gates on all
# other tasks finishing before it claims its 2-GPU slot.
# cv_bench_reasoning removed at user's request — its 2638-example set with
# max_new=1024 + 512 diffusion steps was a ~25h-per-cell bottleneck even at
# bs=16 (compute-bound, not memory-bound, so bs increases didn't help).
TASKS = [
    "mmvet", "mmstar", "vstar_bench",
    "chartqa", "blink_jigsaw", "mmmu_val",
    "blink",
]

# Task-group tasks that lmms_eval fans out into multiple per-subtask jsonls
# instead of writing a single ``<task>.jsonl``. Map each group to a distinct
# *canary* sub-task whose jsonl serves as the completion marker. We avoid
# matching ``blink_jigsaw.jsonl`` (that runs as its own standalone cell) by
# pinning to ``blink_relative_depth`` — only emitted when the ``blink`` group
# is run.
TASK_GROUP_CANARY: dict[str, str] = {
    "blink": "blink_relative_depth.jsonl",
}

# Every cell is accelerate-launched on 2 GPUs by default — matches
# run_mmadaM.sh's NUM_GPUS=2 default. Per-task overrides go in this dict.
TASK_NEEDS_GPUS: dict[str, int] = {}

# Per-task --gen_kwargs overrides. Mirrors run_mmadaM.sh, which always sets
# prefix_lm=True. Override values replace the default entirely.
DEFAULT_GEN_KWARGS = "prefix_lm=True"
TASK_GEN_KWARGS: dict[str, str] = {}

# Per-task --batch_size, derived empirically from a sweep on ckpt 4
# (Unified-MMaDA-MixCoT-cp50) on 2 H200 GPUs (143 GiB each):
#   chartqa @ bs=16 → 97 GiB peak  (max_new=16 group: chartqa, mmmu_val, vstar_bench)
#   mmstar  @ bs=16 → 118 GiB peak (max_new=256 — close to ceiling, do not raise)
#   mmvet   @ bs=4  → 48 GiB peak  (max_new=1024 group: mmvet, cv_bench_reasoning,
#                                    blink_jigsaw — has ~95 GiB headroom)
# A user-supplied BATCH_SIZE env still wins (override below).
DEFAULT_BATCH_SIZE = 16
TASK_BATCH_SIZES: dict[str, int] = {
    # All sizes doubled from the prior verified config — 16-token group
    # was at 97 GiB peak / mmstar at 118 GiB / 1024-group at 48 GiB on
    # the bs=4/16 sweep. Doubling pushes the 16-token and mmstar groups
    # close to or above the 143 GiB ceiling and may OOM; bs=16 for the
    # 1024-group still has ~50 GiB headroom by the linear projection.
    "chartqa":            32,    # verified at 138.9 GiB peak — ok
    "mmmu_val":           16,    # bs=32 OOMed (mmu_val has longer A/B/C/D prefix)
    "vstar_bench":        32,    # verified at MMaDA-Parallel-M run — ok
    "mmstar":             16,    # bs=32 OOMed (max_new=256 + bs=32 = ~219 GiB projected)
    "mmvet":              16,
    "blink_jigsaw":       16,
    "blink":              16,    # fan-out group (14 sub-tasks, all max_new=1024)
}

BASE_OUT     = Path(os.environ.get("BASE_OUT", "/scratch2/yoonjeon.kim/outputs"))
DEFAULT_ROOT = BASE_OUT / "MMaDA-PM"

# Fine-tuned MMaDA-PM ckpts ship weights but not the tokenizer's
# chat_template; the M reference script loads the tokenizer from the base
# ckpt regardless of which CKPT runs. Mirror that — `mmada_m`'s
# `tokenizer_path` knob points at the base for every cell.
TOKENIZER_BASE = os.environ.get("TOKENIZER_BASE", "tyfeld/MMaDA-Parallel-M")
LOG_DIR      = ql.LOG_DIR
TMP_DIR      = Path("/tmp/queue_runner_mmada_m")
SUMMARY_EVERY = int(os.environ.get("QUEUE_SUMMARY_EVERY_S", 1800))
TICK_S        = int(os.environ.get("QUEUE_TICK_S", 15))
DRY_RUN       = os.environ.get("DRY_RUN", "").lower() in ("1", "true")
SLACK_MUTE    = os.environ.get("SLACK_MUTE", "").lower() in ("1", "true")
LIMIT         = os.environ.get("LIMIT", "")          # smoke-run cap; "" or "none" = full set
BATCH_SIZE    = os.environ.get("BATCH_SIZE", "")     # passed through to --batch_size when set

FILTER_TASKS  = set(filter(None, os.environ.get("QUEUE_FILTER_TASKS", "").split(",")))
FILTER_CKPTS  = set(filter(None, os.environ.get("QUEUE_FILTER_CKPTS", "").split(",")))

PYTHON_BIN    = os.environ.get("PYTHON_BIN", sys.executable)
LMMS_EVAL_ARGS = ["-u", "-m", "lmms_eval"]


# ──────────────────────────── cell construction ────────────────────────────

# Per-ckpt output-dir override. The default (basename of the ckpt path)
# is too generic for paths like ".../checkpoint-50" (clashes across
# unrelated runs). Map ambiguous paths to a curated label here.
CKPT_OUTDIR_OVERRIDE: dict[str, str] = {
    "/scratch2/yoonjeon.kim/rl-mmadaMixCoT-thinkmorph/thinkmorph_interleave-Unified-MMaDA-MixCoT/checkpoint-50":
        "Unified-MMaDA-ckpt50",
}


def ckpt_basename(ckpt_path: str) -> str:
    """Subdir under outputs/MMaDA-PM/ for this ckpt's results.

    Honors ``CKPT_OUTDIR_OVERRIDE`` first (explicit curated name);
    otherwise falls back to ``basename(ckpt)``."""
    cleaned = ckpt_path.rstrip("/")
    if cleaned in CKPT_OUTDIR_OVERRIDE:
        return CKPT_OUTDIR_OVERRIDE[cleaned]
    return os.path.basename(cleaned)


def jsonl_path_for(ckpt_path: str, task: str) -> Path:
    return DEFAULT_ROOT / ckpt_basename(ckpt_path) / f"{task}.jsonl"


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
                "needs_gpus":       TASK_NEEDS_GPUS.get(task, 2),
                "status":           "pending",
                "pid":              None,
                "gpu_ids":          None,         # list[int]
                "started_at":       None,
                "ended_at":         None,
                "retries":          0,
                "max_retries":      0,            # mmada_m has no auto-retry knob
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
        if not (p.exists() and p.stat().st_size > 0):
            # Task-group fan-outs (e.g. blink) write per-sub-task jsonls
            # instead of a single <task>.jsonl. Treat the cell as done if
            # the canary sub-task jsonl is present.
            canary = TASK_GROUP_CANARY.get(c["task"])
            if canary:
                cf = p.parent / canary
                if cf.exists() and cf.stat().st_size > 0:
                    c["status"] = "done"
                    c["ended_at"] = c["ended_at"] or ql.utc_now()
                    n += 1
            continue
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
    # config change in this file propagates on next restart. Mirrors the
    # llava_llada queue_runner.py pattern.
    config_keys = ("needs_gpus", "ckpt_path", "jsonl_path", "max_retries")
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
    """Build argv + env + log_path for a cell.

    For 1-GPU cells: `python -m lmms_eval ...`.
    For N-GPU cells (mmmu_val): `accelerate launch --num_processes=N -m lmms_eval ...`.
    """
    job_dir = TMP_DIR / cell["id"]
    job_dir.mkdir(parents=True, exist_ok=True)

    out_dir = DEFAULT_ROOT / ckpt_basename(cell["ckpt_path"])
    gen_img = out_dir / "gen_imgs"

    model_args = (
        f"pretrained={cell['ckpt_path']},"
        f"tokenizer_path={TOKENIZER_BASE},"
        f"gen_img_dir={gen_img}"
    )

    base_argv = [
        "--model", "mmada_m",
        "--model_args", model_args,
        "--tasks", cell["task"],
        "--log_samples",
        "--log_samples_suffix", "mmada_m",
        "--output_path", str(out_dir),
        "--wandb_args", f"project=lmms-eval,job_type=eval,name=mmada_m_queue_{cell['id']}",
    ]
    gen_kwargs_str = TASK_GEN_KWARGS.get(cell["task"], DEFAULT_GEN_KWARGS)
    if gen_kwargs_str:
        base_argv.extend(["--gen_kwargs", gen_kwargs_str])
    if LIMIT and LIMIT.lower() != "none":
        base_argv.extend(["--limit", LIMIT])
    bs = BATCH_SIZE or str(TASK_BATCH_SIZES.get(cell["task"], DEFAULT_BATCH_SIZE))
    base_argv.extend(["--batch_size", bs])

    nproc = cell["needs_gpus"]
    if nproc == 1:
        argv = [PYTHON_BIN] + LMMS_EVAL_ARGS + base_argv
    else:
        # accelerate launch handles distributed init; lmms_eval picks up
        # local_rank via env. Mirrors run_mmada.sh's multi-GPU branch.
        master_port = str(10000 + (hash(cell["id"]) % 50000))
        argv = [
            PYTHON_BIN, "-m", "accelerate.commands.launch",
            "--num_machines=1", "--machine_rank=0",
            "--main_process_ip=127.0.0.1",
            f"--main_process_port={master_port}",
            f"--num_processes={nproc}",
            "-m", "lmms_eval",
        ] + base_argv

    env = os.environ.copy()
    # Single-process mode: clear distributed env vars so accelerate (when
    # absent) doesn't try to init torch.distributed inadvertently.
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
              "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    env.update({
        "CUDA_VISIBLE_DEVICES":   ",".join(str(g) for g in gpus),
        # Cap CPU threads per process so concurrent cells don't oversubscribe
        # the host. Each lmms_eval worker otherwise defaults torch / OMP / MKL
        # to num_cores, and N concurrent workers would each spawn that many
        # compute threads — pegging the box even though the heavy work is on
        # GPU. 4 is plenty for tokenization + dataloader on H200-class GPUs.
        "OMP_NUM_THREADS":         os.environ.get("OMP_NUM_THREADS", "4"),
        "MKL_NUM_THREADS":         os.environ.get("MKL_NUM_THREADS", "4"),
        "OPENBLAS_NUM_THREADS":    os.environ.get("OPENBLAS_NUM_THREADS", "4"),
        "TOKENIZERS_PARALLELISM":  os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        # Match queue_runner.py: lets PyTorch reuse reserved-but-unallocated
        # cache for larger allocations, mitigating fragmentation-driven OOMs.
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    })

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cell['id']}.log"
    return argv, env, log_path


def launch(cell: dict, gpus: list[int]) -> None:
    argv, env, log_path = build_command(cell, gpus)
    gpu_str = ",".join(str(g) for g in gpus)
    print(f"[mmada_m-launch] {cell['id']} on GPUs [{gpu_str}] → {log_path}")
    print("                 " + " ".join(shlex.quote(x) for x in argv))
    fh = log_path.open("ab")
    fh.write(
        f"\n=== queue_runner_mmada_m launch at {ql.utc_now()} | gpus={gpu_str} ===\n".encode()
    )
    fh.flush()
    proc = subprocess.Popen(
        argv, env=env, stdout=fh, stderr=fh,
        cwd=str(REPO), start_new_session=True,
    )
    cell["pid"]        = proc.pid
    cell["gpu_ids"]    = list(gpus)
    cell["status"]     = "running"
    cell["started_at"] = ql.utc_now()
    cell["log_path"]   = str(log_path)
    ql.slack_post(
        f"▶ [mmada_m] `{cell['ckpt_label']}` · `{cell['task']}` on GPUs [{gpu_str}] "
        f"(pid {proc.pid}) — log `{log_path.name}`",
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


def _release_all(cell: dict, pool: ql.GPUPool, all_cells: list[dict]) -> None:
    gpus = cell.get("gpu_ids") or []
    for gpu in gpus:
        others = [
            c for c in all_cells
            if c is not cell and c.get("status") == "running"
            and gpu in (c.get("gpu_ids") or [])
        ]
        if others:
            continue
        pool.release(gpu)


def finalize(cell: dict, pool: ql.GPUPool, all_cells: list[dict] | None = None) -> bool:
    if all_cells is None:
        all_cells = [cell]
    log_path = Path(cell["log_path"]) if cell["log_path"] else None
    jsonl    = Path(cell["jsonl_path"])
    cell["ended_at"] = ql.utc_now()

    # Task-group fan-outs (e.g. blink) write per-sub-task jsonls instead of a
    # single <task>.jsonl; treat the cell as done if the canary sub-task
    # jsonl is present. No robust_score (rescore_all has no per-group aggregator).
    canary = TASK_GROUP_CANARY.get(cell["task"])
    if canary:
        canary_path = jsonl.parent / canary
        if canary_path.exists() and canary_path.stat().st_size > 0:
            cell["status"] = "done"
            n_sub = len(list(jsonl.parent.glob("blink_*.jsonl"))) if cell["task"] == "blink" else 1
            ql.slack_post(
                f"✅ [mmada_m] `{cell['ckpt_label']}` · `{cell['task']}` (group) — "
                f"{n_sub} sub-task jsonls written",
                mute=SLACK_MUTE,
            )
            _release_all(cell, pool, all_cells)
            return True

    if jsonl.exists() and jsonl.stat().st_size > 0:
        cell["status"] = "done"
        stats = rescore_all.score_jsonl_path(jsonl, cell["task"])
        if stats:
            cell["robust_score"] = stats["robust_acc"]
            cell["n_records"]    = stats["n"]
        ql.slack_post(
            f"✅ [mmada_m] `{cell['ckpt_label']}` · `{cell['task']}` — "
            f"N={stats['n'] if stats else '?'}, "
            f"robust acc={(cell['robust_score'] or 0)*100:.2f}%",
            mute=SLACK_MUTE,
        )
        _release_all(cell, pool, all_cells)
        return True

    excerpt = "\n".join(ql.tail_lines(log_path, 30)) if log_path else ""
    is_oom = bool(log_path) and ql.looks_like_oom(ql.tail_lines(log_path, 200))
    cell["status"] = "error_oom" if is_oom else "error_other"
    cell["error_excerpt"] = excerpt[-3000:]
    ql.slack_post(
        f"❌ [mmada_m] `{cell['ckpt_label']}` · `{cell['task']}` failed ({cell['status']})\n"
        f"```\n{excerpt[-1500:]}\n```",
        mute=SLACK_MUTE,
    )
    _release_all(cell, pool, all_cells)
    return True


# ──────────────────────────── scheduling helpers ────────────────────────────

def is_gating_task(task: str) -> bool:
    """True for tasks that block lower-priority cells until they finish.

    Every cell defaults to 2 GPUs, so this is currently a no-op gate; kept
    so per-task overrides in TASK_NEEDS_GPUS can flag a task as needing
    everything else to drain first by setting it strictly higher than the
    default.
    """
    return TASK_NEEDS_GPUS.get(task, 2) > 2


def can_launch(cell: dict, all_cells: list[dict], pool: ql.GPUPool) -> bool:
    """Two gates:

    1. GPU budget: free GPUs >= cell['needs_gpus'].
    2. Priority: a multi-GPU (gating) cell must wait until every single-GPU
       cell that's in scope is settled (done / errored). This implements
       "mmvet → ... → chartqa first, mmmu_val last".
    """
    if len(pool.free) < cell["needs_gpus"]:
        return False
    if cell["needs_gpus"] > 1:
        # block until all 1-GPU cells in scope are settled
        for other in all_cells:
            if cell_filtered(other):
                continue
            if other is cell:
                continue
            if other["needs_gpus"] > 1:
                continue
            if other["status"] in ("pending", "running"):
                return False
    return True


# ──────────────────────────── report writing ────────────────────────────

REPORT_PATH = REPO / "report_all_tasks_mmada_m.md"


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

    out = ["# MMaDA-M queue progress (live)", ""]
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
                line.append("♻ OOM")
            elif c["status"] == "error_other":
                line.append("❌")
            else:
                line.append(c["status"])
        out.append("| " + " | ".join(line) + " |")
    out.append("")
    out.append("Legend: ✅ done · 🏃 running (% from tqdm) · ⏳ pending · ❌ failed")
    out.append("")

    running = [c for c in cells if c["status"] == "running"]
    if running:
        out.append("## Per-job ETA")
        out.append("")
        out.append("| ckpt | task | gpus | started | elapsed | tqdm % | remaining (ETA) |")
        out.append("|---|---|---:|---|---:|---:|---:|")
        for c in running:
            elapsed = ql.hms(c["tqdm_elapsed_s"])
            remain  = ql.hms(c["tqdm_remaining_s"])
            pct     = f"{c['tqdm_pct']:.0f}%" if c["tqdm_pct"] is not None else "—"
            gpu_str = ",".join(str(g) for g in (c.get("gpu_ids") or []))
            out.append(
                f"| `{c['ckpt_label']}` | `{c['task']}` | {gpu_str or '—'} | "
                f"{c['started_at'] or '—'} | {elapsed} | {pct} | {remain} |"
            )
        out.append("")
    return "\n".join(out)


def write_report(cells: list[dict]) -> None:
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
        f"⏳ [mmada_m] Queue: {n_done}/{len(cells)} done · "
        f"{n_running} running · {n_err} errored · ETA ≈ {ql.hms(eta)}",
    )


def main() -> int:
    target_cells = build_cells()
    lock = ql.load_lock()
    existing = lock.get("cells", [])
    cells = merge_cells(existing, target_cells)
    skipped = mark_already_done(cells)
    print(f"[mmada_m-queue] {len(cells)} cells, {skipped} already complete")

    lock = {
        "version":     1,
        "model":       "mmada_m",
        "started_at":  lock.get("started_at") or ql.utc_now(),
        "updated_at":  ql.utc_now(),
        "cells":       cells,
    }
    ql.save_lock(lock)
    write_report(cells)

    if DRY_RUN:
        print("[mmada_m-queue] DRY_RUN — exiting after lock + report.")
        return 0

    gpus = ql.detect_gpus()
    if not gpus:
        print("[mmada_m-queue] no GPUs visible. Exiting.")
        return 1
    pool = ql.GPUPool(gpus)
    for c in cells:
        if c["status"] == "running" and ql.pid_alive(c.get("pid")):
            for g in (c.get("gpu_ids") or []):
                pool.reserve(g)
    for c in cells:
        if c["status"] == "running" and not ql.pid_alive(c.get("pid")):
            print(f"[mmada_m-queue] startup-finalize dead cell {c['id']} (pid {c.get('pid')})")
            finalize(c, pool, cells)
    ql.save_lock(lock)
    write_report(cells)
    print(f"[mmada_m-queue] GPU pool: {gpus} (free={pool.free}, busy={pool.busy})")

    ql.slack_post(
        f"🚀 [mmada_m] queue_runner started on `{ql.host()}` — "
        f"{len(cells)} cells, {skipped} already done, GPUs={gpus}",
        mute=SLACK_MUTE,
    )

    try:
        while True:
            for c in cells:
                if c["status"] != "pending" or cell_filtered(c):
                    continue
                if not can_launch(c, cells, pool):
                    continue
                acquired: list[int] = []
                for _ in range(c["needs_gpus"]):
                    g = pool.acquire()
                    if g is None:
                        break
                    acquired.append(g)
                if len(acquired) < c["needs_gpus"]:
                    for g in acquired:
                        pool.release(g)
                    continue
                launch(c, acquired)
                ql.save_lock(lock)
                write_report(cells)
                # Stagger launches: only one new cell per tick. Each cell
                # deserialises an 8B MMaDA-Parallel-M checkpoint at startup,
                # which pegs CPU + disk for ~30s. Letting the next tick handle
                # the next cell keeps simultaneous model loads to one.
                break

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
            f"🏁 [mmada_m] queue_runner finished — {n_done}/{len(cells)} done · {n_err} errored",
            mute=SLACK_MUTE,
        )
    except KeyboardInterrupt:
        ql.slack_post(
            "⚠ [mmada_m] queue_runner interrupted (Ctrl-C). Running cells left in place.",
            mute=SLACK_MUTE,
        )
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
