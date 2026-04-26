#!/usr/bin/env python3


"""
Re-score completed lmms-eval *.jsonl outputs with a more robust answer-extraction
pipeline. Compares against the default lmms-eval scoring.

Patterns tried, in order:
  1. <answer>...</answer>     (case-insensitive)
  2. \\boxed{...}             (handles nested braces 1 level)
  3. "(?:therefore )?the answer is X"  (with/without trailing dot)
  4. "Answer:\\s*X"
  5. Trailing letter at end of text (multi-choice)
  6. Whole text (fallback)

Multi-choice tasks (cv_bench, blink_jigsaw, vstar_bench): the extracted span is
then narrowed to a single letter A-E.

Free-form tasks (chartqa): the extracted span is compared with the target via
the relaxed_correctness rule from lmms-eval.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/yoonjeon.kim/dLLM-EVAL/outputs/image_gen_usebboxFalse_default")

# (ckpt_dir, task_basename) -> task_kind
TASKS = {
    "chartqa":      "free",
    "cv_bench":     "letter",
    "blink_jigsaw": "letter",
    "vstar_bench":  "letter",
}

# ────────────────────────────── extraction ──────────────────────────────

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE      = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
THE_ANSWER_RE = re.compile(
    r"(?:therefore[\s,]+)?(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)?\s*[:]?\s*"
    r"(?:\$)?([^.\n\$]+?)\s*[\.\n\$]",
    re.IGNORECASE,
)
ANSWER_COLON_RE = re.compile(r"\banswer\s*[:\-]\s*([^\n.]+?)(?=[.\n]|$)", re.IGNORECASE)
TRAILING_LETTER_RE = re.compile(r"(?:^|[^A-Za-z])([A-Ea-e])\s*[.)\]]?\s*$")

LETTER_FROM_SPAN = re.compile(r"\(?\s*([A-Ea-e])\s*\)?", re.IGNORECASE)

NUMERIC_TOKEN_RE = re.compile(r"-?\d+(?:[.,]\d+)?%?")


def _strip(s: str) -> str:
    return s.strip().strip("`*\"' \t\n")


def extract_answer_span(text: str) -> tuple[str, str]:
    """Return (extracted_span, which_pattern)."""
    if text is None:
        return "", "empty"
    t = text.strip()
    if not t:
        return "", "empty"

    m = ANSWER_TAG_RE.search(t)
    if m:
        return _strip(m.group(1)), "answer_tag"

    m = BOXED_RE.search(t)
    if m:
        return _strip(m.group(1)), "boxed"

    # "the answer is X." (with optional therefore/final/etc); take the LAST match
    matches = list(THE_ANSWER_RE.finditer(t + "\n"))
    if matches:
        return _strip(matches[-1].group(1)), "the_answer_is"

    matches = list(ANSWER_COLON_RE.finditer(t))
    if matches:
        return _strip(matches[-1].group(1)), "answer_colon"

    m = TRAILING_LETTER_RE.search(t)
    if m:
        return m.group(1).upper(), "trailing_letter"

    return _strip(t), "raw"


def to_letter(span: str) -> str | None:
    """Narrow an extracted span to a single A-E letter (or None)."""
    if not span:
        return None
    s = span.strip().strip("()[]{}.,*").strip()
    if len(s) == 1 and s.upper() in "ABCDE":
        return s.upper()
    # First letter A-E (avoid words like "A red ball" -> 'A')
    m = re.match(r"\s*\(?\s*([A-E])\s*\)?\s*[\.\):,\-]?\s*$", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Patterns inside the span
    m = re.search(r"\(([A-E])\)", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-E])\b", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def to_letter_from_full(text: str) -> str | None:
    """Last-resort scan over the entire raw text for a letter answer."""
    if not text:
        return None
    # Prefer a final-position letter pattern
    m = TRAILING_LETTER_RE.search(text)
    if m:
        return m.group(1).upper()
    # Letters in parens anywhere
    matches = re.findall(r"\(([A-E])\)", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


# ────────────────────────────── default lmms-eval parsers ──────────────────────────────

def default_extract_xml_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def default_extract_letter_cvblink(text: str) -> str:
    """The cv_bench / blink _extract_answer_letter (after _extract_xml_answer)."""
    text = text.strip()
    m = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def default_extract_letter_vstar(response: str) -> str | None:
    """Reproduce vstar_bench.utils.extract_answer_letter exactly."""
    response = default_extract_xml_answer(response)
    response = response.strip().upper()
    patterns = [
        r"^([A-D])\s*[\.)\]]*",
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",
        r"\(([A-D])\)",
        r"([A-D])\s*(?:\.|\)|])",
        r"(?:^|\s)([A-D])(?:\s|$)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    letters = re.findall(r"[A-D]", response)
    if len(letters) == 1:
        return letters[0]
    if response and response[0] in "ABCD":
        return response[0]
    return None


# ────────────────────────────── scoring ──────────────────────────────

def relaxed_correctness(prediction: str, target: str, max_rel: float = 0.05) -> bool:
    def _f(s):
        s = s.strip()
        try:
            if s.endswith("%"):
                return float(s.rstrip("%")) / 100.0
            return float(s.replace(",", ""))
        except ValueError:
            return None
    pf, tf = _f(prediction), _f(target)
    if pf is not None and tf:
        return abs(pf - tf) / abs(tf) <= max_rel
    return prediction.strip().lower() == target.strip().lower()


def chartqa_pred_robust(text: str) -> str:
    span, pat = extract_answer_span(text)
    # If model produced verbose output, span is already the cleaned answer.
    # For chartqa we may also try to extract a numeric token if span has a number.
    if span and NUMERIC_TOKEN_RE.search(span):
        m = NUMERIC_TOKEN_RE.search(span)
        return m.group(0)
    return span


def letter_pred_robust(text: str, n_choices: int = 5) -> str | None:
    span, _ = extract_answer_span(text)
    letter = to_letter(span)
    if letter is None:
        letter = to_letter_from_full(text)
    if letter is None:
        return None
    # Clamp to allowed range
    if n_choices < 5 and letter > chr(ord("A") + n_choices - 1):
        return letter  # don't filter — caller compares directly
    return letter


# ────────────────────────────── runners ──────────────────────────────

def score_chartqa(records):
    d_correct = r_correct = 0
    examples_helped = []
    pat_hits = defaultdict(int)
    empty_raw = 0
    for r in records:
        target = (r.get("target") or "").strip()
        raw = r["resps"][0][0]["text_gen_output"]
        if not raw.strip(): empty_raw += 1
        _, pat = extract_answer_span(raw)
        pat_hits[pat] += 1
        d_pred = default_extract_xml_answer(raw)
        d_ok = relaxed_correctness(d_pred, target)
        r_pred = chartqa_pred_robust(raw)
        r_ok = relaxed_correctness(r_pred, target)
        if d_ok: d_correct += 1
        if r_ok: r_correct += 1
        if r_ok and not d_ok and len(examples_helped) < 4:
            examples_helped.append({"target": target, "raw": raw, "d_pred": d_pred, "r_pred": r_pred})
    n = len(records)
    return {
        "n": n,
        "default_correct": d_correct,
        "robust_correct": r_correct,
        "default_acc": d_correct / n if n else 0,
        "robust_acc":  r_correct / n if n else 0,
        "examples_helped": examples_helped,
        "pat_hits": dict(pat_hits),
        "empty_raw": empty_raw,
    }


def score_letter(records, target_kind: str, vstar_style: bool = False):
    d_correct = r_correct = 0
    examples_helped = []
    pat_hits = defaultdict(int)
    empty_raw = 0
    for r in records:
        if target_kind == "vstar":
            target = (r.get("target") or "").strip().upper()
        else:
            target = (r.get("target") or "").strip().strip("()")
        raw = r["resps"][0][0]["text_gen_output"]
        if not raw.strip(): empty_raw += 1
        _, pat = extract_answer_span(raw)
        pat_hits[pat] += 1

        if vstar_style:
            d_pred = default_extract_letter_vstar(raw) or ""
        else:
            d_pred = default_extract_letter_cvblink(default_extract_xml_answer(raw))
        d_ok = (d_pred == target) and bool(target)

        r_pred = letter_pred_robust(raw) or ""
        r_ok = (r_pred == target) and bool(target)

        if d_ok: d_correct += 1
        if r_ok: r_correct += 1
        if r_ok and not d_ok and len(examples_helped) < 4:
            examples_helped.append({"target": target, "raw": raw, "d_pred": d_pred, "r_pred": r_pred})
    n = len(records)
    return {
        "n": n,
        "default_correct": d_correct,
        "robust_correct": r_correct,
        "default_acc": d_correct / n if n else 0,
        "robust_acc":  r_correct / n if n else 0,
        "examples_helped": examples_helped,
        "pat_hits": dict(pat_hits),
        "empty_raw": empty_raw,
    }


SCORE_FN = {
    ("chartqa",      "free"):   lambda recs: score_chartqa(recs),
    ("cv_bench",     "letter"): lambda recs: score_letter(recs, "cvblink"),
    ("blink_jigsaw", "letter"): lambda recs: score_letter(recs, "cvblink"),
    ("vstar_bench",  "letter"): lambda recs: score_letter(recs, "vstar", vstar_style=True),
}


def load_jsonl(path: Path):
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ckpts = [
        ("yoonjeon.kim-LaViDa-O", "LaViDa-O (ckpt 0)"),
        ("thinkmorph_interleave-Unified-LavidaO-checkpoint-50",
         "interleave-Unified-LavidaO/cp50 (ckpt 3)"),
    ]
    results = {}  # ckpt -> task -> stats

    for ckpt_dir, _ in ckpts:
        results[ckpt_dir] = {}
        for task, kind in TASKS.items():
            f = ROOT / ckpt_dir / f"{task}.jsonl"
            if not f.exists():
                continue
            records = load_jsonl(f)
            stats = SCORE_FN[(task, kind)](records)
            results[ckpt_dir][task] = stats

    # ────────────────────────────── markdown ──────────────────────────────
    out = []
    out.append("# lmms-eval re-scoring with robust answer extraction\n")
    out.append("Re-scored completed `text_gen_output` using an extended parser that "
               "tries — in order — `<answer>…</answer>`, `\\boxed{…}`, "
               "`the answer is X`, `Answer: X`, a trailing letter, then the raw "
               "text. The default scorer is the unmodified lmms-eval logic for each "
               "task.\n")
    out.append("Scores below are **accuracy** (correct / N).\n")

    out.append("## Summary\n")
    out.append("| ckpt | task | N | default | robust | Δ |")
    out.append("|---|---|---:|---:|---:|---:|")
    for ckpt_dir, label in ckpts:
        for task, kind in TASKS.items():
            s = results[ckpt_dir].get(task)
            if not s:
                continue
            out.append(
                f"| `{label}` | `{task}` | {s['n']} | "
                f"{s['default_acc']*100:.2f}% | {s['robust_acc']*100:.2f}% | "
                f"{(s['robust_acc']-s['default_acc'])*100:+.2f} pp |"
            )
    out.append("")

    out.append("## Per-task detail\n")
    PAT_LABEL = {
        "answer_tag": "<answer>…</answer>", "boxed": "\\boxed{…}",
        "the_answer_is": "the answer is X", "answer_colon": "Answer: X",
        "trailing_letter": "trailing letter", "raw": "raw text",
        "empty": "(empty output)",
    }
    for ckpt_dir, label in ckpts:
        out.append(f"### `{label}`\n")
        for task in TASKS:
            s = results[ckpt_dir].get(task)
            if not s:
                continue
            out.append(f"**`{task}`** — N={s['n']}, default={s['default_correct']} "
                       f"({s['default_acc']*100:.2f}%), robust={s['robust_correct']} "
                       f"({s['robust_acc']*100:.2f}%), "
                       f"Δ=**{(s['robust_acc']-s['default_acc'])*100:+.2f}pp** "
                       f"(+{s['robust_correct']-s['default_correct']} samples), "
                       f"empty `text_gen_output` = {s['empty_raw']} ({s['empty_raw']/s['n']*100:.1f}%)\n")
            out.append("Pattern that fired during extraction:")
            out.append("")
            out.append("| pattern | n | % |")
            out.append("|---|---:|---:|")
            for pat, n_hit in sorted(s['pat_hits'].items(), key=lambda kv: -kv[1]):
                out.append(f"| {PAT_LABEL.get(pat, pat)} | {n_hit} | {n_hit/s['n']*100:.1f}% |")
            out.append("")
            if s['examples_helped']:
                out.append("Examples the robust parser rescued:\n")
                out.append("| target | raw text_gen_output | default pred | robust pred |")
                out.append("|---|---|---|---|")
                for ex in s['examples_helped']:
                    raw = ex['raw'].replace("|", "\\|").replace("\n", " ")
                    if len(raw) > 120: raw = raw[:117] + "..."
                    out.append(
                        f"| `{ex['target']}` | {raw or '(empty)'} | "
                        f"`{ex['d_pred'] or '(empty)'}` | `{ex['r_pred'] or '(empty)'}` |"
                    )
                out.append("")
        out.append("")

    md = "\n".join(out)
    print(md)


if __name__ == "__main__":
    main()
