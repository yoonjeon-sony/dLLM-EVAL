"""Helpers for scripts/queue_runner.py — lock file IO, Slack webhook, tqdm parser, GPU pool."""

from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import subprocess
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

LOCK_PATH = Path(os.environ.get(
    "QUEUE_LOCK_PATH",
    "/scratch2/yoonjeon.kim/.claude/scheduled_tasks.lock",
))
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
LOG_DIR = Path(os.environ.get(
    "QUEUE_LOG_DIR",
    "/scratch2/yoonjeon.kim/outputs/_queue_logs",
))


# ──────────────────────────── lock-file IO ────────────────────────────

@contextmanager
def locked(path: Path, mode: str = "r+"):
    """Open with fcntl flock (exclusive). Caller is responsible for fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("{}")
    fh = path.open(mode)
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield fh
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def load_lock() -> dict[str, Any]:
    if not LOCK_PATH.exists():
        return {}
    with locked(LOCK_PATH, "r") as fh:
        try:
            return json.loads(fh.read() or "{}")
        except json.JSONDecodeError:
            return {}


def save_lock(data: dict[str, Any]) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = utc_now()
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=str(LOCK_PATH.parent), delete=False, suffix=".tmp"
    )
    json.dump(data, tmp, indent=2, sort_keys=False)
    tmp.flush()
    os.fsync(tmp.fileno())
    tmp.close()
    os.replace(tmp.name, LOCK_PATH)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ──────────────────────────── Slack ────────────────────────────

def slack_post(text: str, *, blocks: list | None = None, mute: bool = False) -> None:
    if mute or not SLACK_WEBHOOK_URL:
        return
    body = {"text": text}
    if blocks:
        body["blocks"] = blocks
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as exc:  # network noise — don't crash the orchestrator
        print(f"[queue_lib] slack_post failed: {exc}")


# ──────────────────────────── tqdm parsing ────────────────────────────

# Outer per-task progress bar: "Model Responding: 41%|...| 12/29 [57:33<1:20:05, 282.67s/it]"
# Inner per-batch bars (e.g. "Region editing 8 images: 26/64 [01:36<02:21]") are
# image-gen sub-steps and would mislead the ETA, so we ignore them.
_TQDM_LINE_RE = re.compile(
    r"Model Responding:\s*(?P<pct>\d+(?:\.\d+)?)\s*%[^\[]*"
    r"\[(?P<elapsed>\d+(?::\d+){1,2})<(?P<remaining>\d+(?::\d+){1,2})"
)
# Fallback for runs that haven't emitted Model Responding yet (e.g. mid-load):
# we'll fall back to None so the report shows pct=— rather than a misleading
# image-gen bar value.


def _hms_to_seconds(s: str) -> int:
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return h * 3600 + m * 60 + sec


def parse_tqdm_tail(log_path: Path, tail_bytes: int = 64 * 1024) -> dict | None:
    """Return {elapsed_s, remaining_s, pct} from the most recent `Model Responding`
    bar in the last N bytes of the log. Returns None if no such bar exists yet
    — image-gen sub-bars are intentionally ignored so they don't pollute the ETA."""
    if not log_path.exists():
        return None
    size = log_path.stat().st_size
    with log_path.open("rb") as fh:
        if size > tail_bytes:
            fh.seek(size - tail_bytes)
        chunk = fh.read().decode("utf-8", errors="ignore")
    last_match = None
    for line in re.split(r"[\r\n]", chunk):
        m = _TQDM_LINE_RE.search(line)
        if m:
            last_match = m
    if last_match is None:
        return None
    return {
        "elapsed_s":   _hms_to_seconds(last_match.group("elapsed")),
        "remaining_s": _hms_to_seconds(last_match.group("remaining")),
        "pct":         float(last_match.group("pct")),
    }


# ──────────────────────────── GPU pool ────────────────────────────

class GPUPool:
    """Round-robin pool of GPU IDs. Acquire/release semantics."""

    def __init__(self, ids: list[int]):
        self._free  = list(ids)
        self._busy  = set()

    def acquire(self) -> int | None:
        if not self._free:
            return None
        gpu = self._free.pop(0)
        self._busy.add(gpu)
        return gpu

    def release(self, gpu: int) -> None:
        if gpu in self._busy:
            self._busy.remove(gpu)
            if gpu not in self._free:
                self._free.append(gpu)

    def reserve(self, gpu: int) -> None:
        """Mark a GPU busy without acquiring (for crash recovery)."""
        if gpu in self._free:
            self._free.remove(gpu)
        self._busy.add(gpu)

    @property
    def free(self) -> list[int]:
        return list(self._free)

    @property
    def busy(self) -> list[int]:
        return sorted(self._busy)


def detect_gpus() -> list[int]:
    """Return GPU indices visible to the host. Honors CUDA_VISIBLE_DEVICES if set."""
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv:
        return [int(x) for x in cv.split(",") if x.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        return [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
    except Exception:
        return []


# ──────────────────────────── misc ────────────────────────────

def pid_alive(pid: int | None) -> bool:
    """True iff the process exists AND is not a zombie. `os.kill(pid, 0)` alone
    returns True for zombies (defunct procs whose parent never reaped them),
    which is too lax — we need the /proc state byte to distinguish."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Process exists; check it's not a zombie via /proc/<pid>/stat field 3.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            stat = fh.read()
        # Field 3 is the state char; format is "<pid> (<comm>) <state> ..."
        # comm can contain spaces/parens, so split from the right.
        state = stat.rsplit(")", 1)[1].strip().split()[0]
        return state != "Z"
    except (FileNotFoundError, IndexError):
        # /proc entry vanished between os.kill and open — treat as dead.
        return False


def tail_lines(path: Path, n: int = 30) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    chunk = min(size, 64 * 1024)
    with path.open("rb") as fh:
        fh.seek(size - chunk)
        data = fh.read().decode("utf-8", errors="ignore")
    return data.splitlines()[-n:]


_OOM_PATTERNS = [
    "torch.cuda.OutOfMemoryError",
    "CUDA out of memory",
    "OutOfMemoryError",
]


def looks_like_oom(lines: list[str]) -> bool:
    blob = "\n".join(lines).lower()
    return any(p.lower() in blob for p in _OOM_PATTERNS)


def host() -> str:
    return socket.gethostname()


def hms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"
