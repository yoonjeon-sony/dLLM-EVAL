#!/usr/bin/env python3
"""
Re-score every completed (ROOT × ckpt × task) jsonl under

  /scratch2/yoonjeon.kim/outputs/<ROOT>/<ckpt_dir>/<task>.jsonl

with both the default lmms-eval per-task scoring logic AND a robust extended
parser. Iterates over all four sampling-config ROOTs and the full TASK_KIND
table. Writes:

  - report_all_tasks.md   : aggregate (ckpt × task) accuracy matrices per ROOT
  - parser_report.md      : first-32 per-sample dump (response, parsed, target,
                             default ✓/✗, robust ✓/✗) for each non-empty cell

Record schemas:
  ROOT :  rec["resps"][0][0]  is a dict   {text_gen_input, text_gen_output, ...}
  ROOT2 / ROOT3 :  rec["resps"][0][0]  is a string (the model's text directly)
  rec["target"] is a string in every case.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

# ────────────────────────────── configuration ──────────────────────────────

ROOTS = {
    "ROOT  (default)":          Path("/scratch2/yoonjeon.kim/outputs/image_gen_usebboxFalse_default"),
    "ROOT2 (tok128_blk128_step64)": Path("/scratch2/yoonjeon.kim/outputs/image_gen_usebboxFalse_tok128_blk128_step64_t0"),
    "ROOT3 (tok256_blk128_step64)": Path("/scratch2/yoonjeon.kim/outputs/image_gen_usebboxFalse_tok256_blk128_step64_t0"),
}

CKPT_DIRS = {
    "LaViDa-O (base)":        "yoonjeon.kim-LaViDa-O",
    "sft-zebracot":           "yoonjeon.kim-sft_LaViDa-O-thinkmorph_zebracot-step9000",
    "Unified-cp50":           "thinkmorph_interleave-Unified-LavidaO-checkpoint-50",
    "region-edit-cp50":       "thinkmorph_interleave-region-edit-LavidaO-checkpoint-50",
    "answer-LavidaO-ckpt50":  "yjyjyj98-thinkmorph_answer-LavidaO-ckpt50",
    "edit-LavidaO-ckpt50":    "yjyjyj98-thinkmorph_edit-LavidaO-ckpt50",
    "interleave-cp50":        "thinkmorph_interleave-LavidaO-checkpoint-50",
}

# task -> kind (drives both default and robust scoring)
TASK_KIND = {
    "chartqa":                              "chartqa",
    "cv_bench":                             "cv_bench",
    "cv_bench_reasoning":                   "cv_bench",   # judge-based default; robust = letter
    "blink_jigsaw":                         "letter",
    "vstar_bench":                          "vstar",
    "VisPuzzle_direct":                     "vispuzzle_direct",
    "mmstar":                               "mmstar",
    "mmmu_val":                             "mmmu",
    "mmvet":                                "mmvet",
    "ai2d_lite":                            "ai2d",
    "mathverse_testmini_vision_dominant":   "mathverse",
    "mathvista_testmini_format":            "mathvista",
    "scienceqa_img":                        "scienceqa",
    "VisualPuzzles_cot":                    "vispuzzle_cot",
}

PARSER_SAMPLES = 32   # first N records per (ROOT × ckpt × task) cell in parser_report.md

OUT_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = OUT_DIR / "report_all_tasks.md"
PARSER_REPORT_PATH = OUT_DIR / "parser_report.md"


# ────────────────────────────── extraction primitives ──────────────────────────────

ANSWER_TAG_RE      = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE           = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
THE_ANSWER_RE      = re.compile(
    r"(?:therefore[\s,]+)?(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)?\s*[:]?\s*"
    r"(?:\$)?([^.\n\$]+?)\s*[\.\n\$]",
    re.IGNORECASE,
)
ANSWER_COLON_RE    = re.compile(r"\banswer\s*[:\-]\s*([^\n.]+?)(?=[.\n]|$)", re.IGNORECASE)
TRAILING_LETTER_RE = re.compile(r"(?:^|[^A-Za-z])([A-Ea-e])\s*[.)\]]?\s*$")
NUMERIC_TOKEN_RE   = re.compile(r"-?\d+(?:[.,]\d+)?%?")


def _strip(s: str) -> str:
    return s.strip().strip("`*\"' \t\n")


def extract_answer_span(text: str) -> tuple[str, str]:
    """Return (extracted_span, pattern_name)."""
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


def to_letter(span: str, options: str = "ABCDE") -> str | None:
    if not span:
        return None
    s = span.strip().strip("()[]{}.,*").strip()
    if len(s) == 1 and s.upper() in options:
        return s.upper()
    m = re.match(rf"\s*\(?\s*([{options}])\s*\)?\s*[\.\):,\-]?\s*$", s, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(rf"\(([{options}])\)", s, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(rf"\b([{options}])\b", s, re.IGNORECASE)
    if m: return m.group(1).upper()
    return None


def to_letter_full(text: str, options: str = "ABCDE") -> str | None:
    if not text:
        return None
    pat = re.compile(rf"(?:^|[^A-Za-z])([{options}{options.lower()}])\s*[.)\]]?\s*$")
    m = pat.search(text)
    if m: return m.group(1).upper()
    matches = re.findall(rf"\(([{options}])\)", text, re.IGNORECASE)
    if matches: return matches[-1].upper()
    return None


def relaxed_correctness(prediction: str, target: str, max_rel: float = 0.05) -> bool:
    def _f(s):
        if not isinstance(s, str): return None
        s = s.strip()
        try:
            if s.endswith("%"): return float(s.rstrip("%")) / 100.0
            return float(s.replace(",", ""))
        except ValueError:
            return None
    pf, tf = _f(prediction), _f(target)
    if pf is not None and tf:
        return abs(pf - tf) / abs(tf) <= max_rel
    return prediction.strip().lower() == target.strip().lower()


def numeric_or_substring_match(pred: str, target: str, max_rel: float = 0.05) -> bool:
    """Heuristic: numeric within 5%, exact lower-case match, or target as substring of pred."""
    if not pred or not target: return False
    p, t = pred.strip().lower(), target.strip().lower()
    if p == t: return True
    if t in p: return True
    try:
        pf, tf = float(p), float(t)
        if tf and abs(pf - tf) / abs(tf) <= max_rel: return True
    except ValueError:
        pass
    return False


# ────────────────────────────── default scorers (mirror lmms-eval) ──────────────────────────────

def default_extract_xml(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def default_letter_cvbench(text: str) -> str:
    """cv_bench / blink first-char letter extraction (after _extract_xml)."""
    text = text.strip()
    m = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", text, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def default_letter_vstar(response: str) -> str:
    response = default_extract_xml(response).strip().upper()
    patterns = [
        r"^([A-D])\s*[\.)\]]*",
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+([A-D])",
        r"\(([A-D])\)",
        r"([A-D])\s*(?:\.|\)|])",
        r"(?:^|\s)([A-D])(?:\s|$)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m: return m.group(1).upper()
    letters = re.findall(r"[A-D]", response)
    if len(letters) == 1: return letters[0]
    if response and response[0] in "ABCD": return response[0]
    return ""


def default_mmstar(pred: str, gt: str) -> bool:
    answer = (gt or "").lower().replace("\n", " ").strip()
    p = (pred or "").lower().replace("\n", " ").strip()
    try:
        if answer == p[0]: return True
        if p[0] == "(" and answer == p[1]: return True
        if p[0:7] == "option " and answer == p[7]: return True
        if p[0:14] == "the answer is " and answer == p[14]: return True
    except Exception:
        return False
    return False


def default_scienceqa(pred: str, gt: str) -> bool:
    """Mirror sqa_process_results."""
    target = (gt or "").strip().lower()
    p = (pred or "").strip().lower()
    if p == target: return True
    if len(p) >= 2 and p[0].isupper() and p[1] == ".":
        return p[0] == target
    # case-insensitive variant since lmms-eval also does .lower()
    if len(p) >= 2 and p[1] == "." and p[0] == target:
        return True
    return False


def default_ai2d_lite(pred: str, target: str) -> bool:
    """ai2d_lite uses the MultiChoiceRegexFilter that matches '^[A-Z]\\.', then exact_match."""
    text = (pred or "").strip()
    m = re.match(r"^\s*([A-Z])\.", text)
    if m:
        text = m.group(1)
    return text.strip().lower() == (target or "").strip().lower()


def default_visualpuzzles(pred_raw: str, target: str, options) -> bool:
    """Mirror VisualPuzzles_process_result: extract <answer>…</answer> first, then parse_response."""
    extracted = default_extract_xml(pred_raw)
    if not extracted:
        return False
    all_choices = ["A", "B", "C", "D"]
    index2ans = None
    if options is not None and isinstance(options, list) and len(options) >= 4:
        index2ans = {all_choices[i]: options[i] for i in range(4)}
    pred = _vispuzzle_parse_response(extracted, all_choices, index2ans)
    return pred is not None and pred.lower() == (target or "").lower()


def _vispuzzle_parse_response(response: str, all_choices, index2ans) -> str | None:
    """Mirror parse_response from VisualPuzzles utils, minus the random-fallback."""
    pats = [
        r"Answer:\s*\(([A-Za-z])\)",
        r"(?<!Final )Answer:\s*([A-Za-z])",
        r"Answer:\s*([A-Za-z])",
        r"\s*\(([A-Za-z])\)",
        r"\s*([A-Za-z])\)",
        r"\s*\{([A-Za-z])\}",
        r"\s*\$([A-Za-z])\$",
    ]
    for pat in pats:
        for m in reversed(re.findall(pat, response)):
            if m in all_choices or m.upper() in all_choices:
                return m.upper()
    response2 = " " + response.strip()
    for pat in [r" ([A-Da-d])\.", r" ([A-Da-d])"]:
        ms = re.findall(pat, response2)
        if ms and (pat == r" ([A-Da-d])\." or len(response2) <= 5):
            for m in reversed(ms):
                if m.upper() in all_choices:
                    return m.upper()
    if index2ans is not None:
        for idx in all_choices:
            ans = index2ans[idx]
            if f"answer: {ans}".lower() in response.lower(): return idx
            if f"answer:{ans}".lower() in response.lower(): return idx
    return None


# ────────────────────────────── robust scorers ──────────────────────────────

def chartqa_robust_pred(text: str) -> str:
    span, _ = extract_answer_span(text)
    if span and NUMERIC_TOKEN_RE.search(span):
        m = NUMERIC_TOKEN_RE.search(span); return m.group(0)
    return span


def letter_robust_pred(text: str, opts: str = "ABCDE") -> str:
    span, _ = extract_answer_span(text)
    return to_letter(span, opts) or to_letter_full(text, opts) or ""


def mmvet_robust_match(pred: str, target: str) -> bool:
    span, _ = extract_answer_span(pred)
    cand = (span or pred).strip().lower()
    tgt = (target or "").strip().lower()
    if not cand or not tgt: return False
    if cand == tgt: return True
    if tgt in cand: return True
    try:
        c, t = float(cand), float(tgt)
        if t and abs(c - t) / abs(t) <= 0.05: return True
    except ValueError:
        pass
    if "<or>" in tgt:
        for opt in tgt.split("<or>"):
            if opt.strip() and opt.strip() in cand: return True
    if "<and>" in tgt:
        # require all parts present
        parts = [p.strip() for p in tgt.split("<and>") if p.strip()]
        if parts and all(p in cand for p in parts): return True
    return False


def mathverse_robust(pred_raw: str, target: str, qtype: str) -> bool:
    span, _ = extract_answer_span(pred_raw)
    cand = span or pred_raw
    tgt = (target or "").strip()
    if qtype == "multi-choice":
        L = letter_robust_pred(pred_raw, opts="ABCDE") or ""
        return L == tgt.upper()
    return numeric_or_substring_match(cand, tgt)


def mathvista_robust(pred_raw: str, target: str, qtype: str, answer_type: str) -> bool:
    span, _ = extract_answer_span(pred_raw)
    cand = span or pred_raw
    tgt = (target or "").strip()
    if qtype == "multi_choice":
        L = letter_robust_pred(pred_raw, opts="ABCDE") or ""
        return L == tgt.upper()
    # free_form: numeric (with precision) or substring
    if answer_type in ("integer", "float"):
        # extract first numeric token from candidate
        m = NUMERIC_TOKEN_RE.search(cand or "")
        if m:
            try:
                pf = float(m.group(0).replace(",", "").rstrip("%"))
                tf = float(tgt)
                if tf == 0: return abs(pf - tf) < 1e-6
                return abs(pf - tf) / abs(tf) <= 0.05
            except ValueError:
                pass
    return numeric_or_substring_match(cand, tgt)


# ────────────────────────────── per-record scoring dispatch ──────────────────────────────

def get_raw(rec) -> str:
    """Pull the model's text response, handling both record schemas."""
    r = rec.get("resps")
    if not r: return ""
    inner = r[0]
    if isinstance(inner, list) and inner:
        x = inner[0]
        if isinstance(x, dict):
            return x.get("text_gen_output") or ""
        if isinstance(x, str):
            return x
    return ""


def score_record(rec, kind):
    """Return (default_correct, robust_correct, default_pred_str, robust_pred_str, pattern)."""
    raw = get_raw(rec)
    target = (rec.get("target") or "").strip()
    doc = rec.get("doc") or {}
    _, pat = extract_answer_span(raw)

    if kind == "chartqa":
        d_pred = default_extract_xml(raw)
        d_ok = relaxed_correctness(d_pred, target)
        r_pred = chartqa_robust_pred(raw)
        r_ok = relaxed_correctness(r_pred, target)

    elif kind == "cv_bench":
        # default = _extract_xml then first-letter; target may be wrapped in ()
        tgt = target.strip("()")
        d_pred = default_letter_cvbench(default_extract_xml(raw))
        d_ok = bool(tgt) and d_pred == tgt
        r_pred = letter_robust_pred(raw, opts="ABCDE")
        r_ok = bool(tgt) and r_pred == tgt

    elif kind == "letter":  # blink_jigsaw
        tgt = target.strip("()").upper()
        d_pred = default_letter_cvbench(default_extract_xml(raw))
        d_ok = bool(tgt) and d_pred == tgt
        r_pred = letter_robust_pred(raw, opts="ABCDE")
        r_ok = bool(tgt) and r_pred == tgt

    elif kind == "vstar":
        tgt = target.upper()
        d_pred = default_letter_vstar(raw)
        d_ok = bool(tgt) and d_pred == tgt
        r_pred = letter_robust_pred(raw, opts="ABCD")
        r_ok = bool(tgt) and r_pred == tgt

    elif kind == "vispuzzle_direct":
        # lmms-eval normalises with case/punct ignored exact_match; target is free-form sentence
        def _norm(x): return re.sub(r"[^a-z0-9 ]", " ", (x or "").lower()).strip()
        tgt_n = _norm(target)
        d_pred = raw.strip()
        d_ok = bool(tgt_n) and _norm(d_pred) == tgt_n
        span, _ = extract_answer_span(raw)
        cand = (span or raw).strip()
        r_pred = cand[:80]
        r_ok = bool(tgt_n) and (_norm(cand) == tgt_n or tgt_n in _norm(cand))

    elif kind == "mmstar":
        d_ok = default_mmstar(raw, target)
        d_pred = "(first-char check)"
        r_pred = letter_robust_pred(raw, opts="ABCDE")
        r_ok = bool(target) and r_pred == target.strip().upper()

    elif kind == "mmvet":
        d_pred = "(needs gpt-4 judge)"
        d_ok = False
        r_pred = "(heuristic substring)"
        r_ok = mmvet_robust_match(raw, target)

    elif kind == "mmmu":
        # Multi-choice (most of MMMU val) → letter parser; open-ended → substring.
        d_pred = "(needs gpt-4 judge)"
        d_ok = False
        tgt = target.strip().upper()
        if tgt and len(tgt) == 1 and tgt in "ABCDEFGHIJ":
            r_pred = letter_robust_pred(raw, opts="ABCDEFGHIJ")
            r_ok = r_pred == tgt
        else:
            r_pred = "(substring)"
            r_ok = numeric_or_substring_match(raw, target)

    elif kind == "ai2d":
        d_ok = default_ai2d_lite(raw, target)
        d_pred = (raw or "").strip()[:30]
        r_pred = letter_robust_pred(raw, opts="ABCDE")
        r_ok = bool(target) and r_pred == target.strip().upper()

    elif kind == "scienceqa":
        d_ok = default_scienceqa(raw, target)
        d_pred = (raw or "").strip()[:30]
        r_pred = letter_robust_pred(raw, opts="ABCDE")
        r_ok = bool(target) and r_pred == target.strip().upper()

    elif kind == "vispuzzle_cot":
        d_ok = default_visualpuzzles(raw, target, doc.get("options"))
        d_pred = "(parse_response on <answer>…)"
        r_pred = letter_robust_pred(raw, opts="ABCD")
        r_ok = bool(target) and r_pred == target.strip().upper()

    elif kind == "mathverse":
        qtype = doc.get("question_type", "multi-choice")
        d_pred = "(needs gpt-4 judge)"
        d_ok = False  # default uses GPT-4 judge
        r_pred = letter_robust_pred(raw, "ABCDE") if qtype == "multi-choice" else (extract_answer_span(raw)[0] or "")[:30]
        r_ok = mathverse_robust(raw, target, qtype)

    elif kind == "mathvista":
        qtype = doc.get("question_type", "multi_choice")
        atype = doc.get("answer_type", "text")
        d_pred = "(needs gpt-4 judge)"
        d_ok = False
        if qtype == "multi_choice":
            r_pred = letter_robust_pred(raw, "ABCDE")
        else:
            span, _ = extract_answer_span(raw)
            cand = span or raw
            m = NUMERIC_TOKEN_RE.search(cand or "")
            r_pred = (m.group(0) if m else (cand or "").strip())[:30]
        r_ok = mathvista_robust(raw, target, qtype, atype)

    else:
        d_pred, d_ok, r_pred, r_ok = "", False, "", False

    return bool(d_ok), bool(r_ok), str(d_pred), str(r_pred), pat


def score_one(records, kind):
    n = len(records)
    d_ok = r_ok = empty = 0
    pat = Counter()
    rescued = []
    for rec in records:
        raw = get_raw(rec)
        if not raw.strip(): empty += 1
        dc, rc, dp, rp, p = score_record(rec, kind)
        pat[p] += 1
        if dc: d_ok += 1
        if rc: r_ok += 1
        if rc and not dc and len(rescued) < 4:
            rescued.append({"target": (rec.get("target") or "").strip(),
                            "raw": raw, "d": dp, "r": rp})
    span_used = pat["answer_tag"] + pat["boxed"] + pat["the_answer_is"] + pat["answer_colon"] + pat["trailing_letter"]
    return {
        "n": n,
        "default_correct": d_ok,
        "robust_correct":  r_ok,
        "default_acc":     d_ok / n if n else 0.0,
        "robust_acc":      r_ok / n if n else 0.0,
        "patterns":        dict(pat),
        "structured":      span_used,
        "raw_fallback":    pat.get("raw", 0),
        "empty":           empty,
        "rescued":         rescued,
    }


# ────────────────────────────── load + render ──────────────────────────────

def load_jsonl(path: Path):
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("|", "\\|").replace("\n", " ⏎ ")
    return s if len(s) <= n else s[: n - 3] + "..."


def render_aggregate_body_md(matrix) -> str:
    """Return the per-ROOT body tables as a markdown string. Pure, no I/O."""
    out = []
    for root_label, root in ROOTS.items():
        out.append(f"\n## `{root_label}` — `{root.name}`\n")
        out.append("Each cell: `default% → robust% (Δ pp, correct/N)`. `—` = file missing.\n")
        out.append("| task | " + " | ".join(CKPT_DIRS.keys()) + " |")
        out.append("|---|" + "|".join(["---"] * len(CKPT_DIRS)) + "|")
        for task in TASK_KIND:
            cells = []
            for ckpt_label in CKPT_DIRS:
                s = matrix[root_label][task].get(ckpt_label)
                if s is None:
                    cells.append("—")
                elif s["n"] == 0:
                    cells.append("(empty)")
                else:
                    d = s["default_acc"] * 100; r = s["robust_acc"] * 100
                    cells.append(f"{d:.2f}% → **{r:.2f}%** (Δ {r-d:+.2f}pp, {s['robust_correct']}/{s['n']})")
            out.append(f"| `{task}` | " + " | ".join(cells) + " |")
        out.append("")

        out.append("### Parser pattern share (% of records)\n")
        out.append("| task | ckpt | N | empty% | structured% | raw% | top patterns |")
        out.append("|---|---|---:|---:|---:|---:|---|")
        for task in TASK_KIND:
            for ckpt_label in CKPT_DIRS:
                s = matrix[root_label][task].get(ckpt_label)
                if s is None or s["n"] == 0: continue
                n = s["n"]
                e_p = s["empty"]/n*100
                s_p = s["structured"]/n*100
                r_p = s["raw_fallback"]/n*100
                top = ", ".join(f"{k}:{v}" for k, v in
                                sorted(s["patterns"].items(), key=lambda kv: -kv[1])[:3])
                out.append(f"| `{task}` | {ckpt_label} | {n} | {e_p:.2f} | {s_p:.2f} | {r_p:.2f} | {top} |")
        out.append("")
    return "\n".join(out)


def build_full_matrix() -> dict:
    """Walk every ROOT × CKPT × TASK file on disk and score it. Returns the matrix dict."""
    matrix = {}
    for root_label, root in ROOTS.items():
        matrix[root_label] = {t: {} for t in TASK_KIND}
        for ckpt_label, ckpt_dir in CKPT_DIRS.items():
            for task, kind in TASK_KIND.items():
                path = root / ckpt_dir / f"{task}.jsonl"
                if not path.exists():
                    matrix[root_label][task][ckpt_label] = None
                    continue
                matrix[root_label][task][ckpt_label] = score_one(load_jsonl(path), kind)
    return matrix


def score_jsonl_path(path: Path, task: str) -> dict | None:
    """Score one jsonl. Returns the same shape as score_one(), or None if file missing."""
    if not path.exists():
        return None
    kind = TASK_KIND.get(task)
    if kind is None:
        return None
    return score_one(load_jsonl(path), kind)


def write_aggregate_report(matrix, *, prefix_md: str = "") -> None:
    """Write the full report. `prefix_md` is prepended (used by queue_runner for the
    live progress matrix + per-job ETA section)."""
    header = ["# Re-scored evaluation matrix (4 ROOTs × ckpts × tasks)\n",
           "Re-scored with both lmms-eval's per-task default scorer (mirrored from "
           "`lmms-eval/lmms_eval/tasks/<task>/utils.py`) and an extended robust parser "
           "(`<answer>…</answer>` → `\\boxed{…}` → 'the answer is X' → 'Answer: X' → "
           "trailing letter → raw fallback).\n",
           "Tasks needing a GPT-4 judge (`mmvet`, `mathverse`, `mathvista`) report "
           "default = `n/a` and rely on the robust heuristic match (numeric within 5% "
           "or substring of gold answer).\n",
           "Schema note: ROOT2/ROOT3 records store `resps[0][0]` as a plain string, "
           "while ROOT wrap it in a dict with `text_gen_output`. The unified "
           "`get_raw(rec)` handles both.\n"]
    parts = ["\n".join(header)]
    if prefix_md:
        parts.append(prefix_md)
    parts.append(render_aggregate_body_md(matrix))
    REPORT_PATH.write_text("\n".join(parts))


def write_parser_report(matrix_records):
    """matrix_records[root_label][ckpt_label][task] = list of per-record dicts (first 32)."""
    out = [f"# Parser robustness report — first {PARSER_SAMPLES} samples per cell\n",
           "Each table row is one record. Columns:\n",
           "- `target` — gold answer (`rec.target`)\n",
           "- `model response` — raw `text_gen_output` (truncated to 140 chars)\n",
           "- `default pred` — what the lmms-eval default scorer would parse\n",
           "- `robust pred` — what the extended robust parser extracts\n",
           "- `default ✓` / `robust ✓` — evaluation result\n",
           "- `pattern` — which extraction rule fired in the robust parser\n"]

    for root_label, root in ROOTS.items():
        any_root = any(matrix_records[root_label][c] for c in matrix_records[root_label])
        if not any_root: continue
        out.append(f"\n## `{root_label}` — `{root.name}`\n")
        for ckpt_label in CKPT_DIRS:
            tasks_here = matrix_records[root_label].get(ckpt_label, {})
            if not tasks_here: continue
            out.append(f"\n### ckpt: `{ckpt_label}`\n")
            for task in TASK_KIND:
                rows = tasks_here.get(task)
                if not rows: continue
                out.append(f"\n#### task: `{task}` (showing {len(rows)} of first {PARSER_SAMPLES})\n")
                out.append("| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |")
                out.append("|--:|---|---|---|---|:--:|:--:|---|")
                for i, r in enumerate(rows):
                    out.append(
                        f"| {i} | `{_truncate(r['target'], 40)}` "
                        f"| {_truncate(r['raw'], 140) or '(empty)'} "
                        f"| `{_truncate(r['d_pred'], 30)}` "
                        f"| `{_truncate(r['r_pred'], 30)}` "
                        f"| {'✅' if r['d_ok'] else '❌'} "
                        f"| {'✅' if r['r_ok'] else '❌'} "
                        f"| {r['pattern']} |"
                    )
                out.append("")
    PARSER_REPORT_PATH.write_text("\n".join(out))


def main():
    matrix = {}          # [root_label][task][ckpt_label] -> stats
    samples = {}         # [root_label][ckpt_label][task] -> list of per-record dicts
    for root_label, root in ROOTS.items():
        matrix[root_label] = {t: {} for t in TASK_KIND}
        samples[root_label] = {c: {} for c in CKPT_DIRS}
        for ckpt_label, ckpt_dir in CKPT_DIRS.items():
            for task, kind in TASK_KIND.items():
                path = root / ckpt_dir / f"{task}.jsonl"
                if not path.exists():
                    matrix[root_label][task][ckpt_label] = None
                    continue
                records = load_jsonl(path)
                matrix[root_label][task][ckpt_label] = score_one(records, kind)
                # per-sample dump: first N records
                rows = []
                for rec in records[:PARSER_SAMPLES]:
                    raw = get_raw(rec)
                    dc, rc, dp, rp, p = score_record(rec, kind)
                    rows.append({
                        "target": (rec.get("target") or "").strip(),
                        "raw": raw, "d_pred": dp, "r_pred": rp,
                        "d_ok": dc, "r_ok": rc, "pattern": p,
                    })
                if rows:
                    samples[root_label][ckpt_label][task] = rows

    write_aggregate_report(matrix)
    write_parser_report(samples)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {PARSER_REPORT_PATH}")


if __name__ == "__main__":
    main()
