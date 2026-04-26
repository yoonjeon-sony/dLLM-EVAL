# Re-scored evaluation matrix (4 ROOTs × ckpts × tasks)

Re-scored with both lmms-eval's per-task default scorer (mirrored from `lmms-eval/lmms_eval/tasks/<task>/utils.py`) and an extended robust parser (`<answer>…</answer>` → `\boxed{…}` → 'the answer is X' → 'Answer: X' → trailing letter → raw fallback).

Tasks needing a GPT-4 judge (`mmvet`, `mathverse`, `mathvista`) report default = `n/a` and rely on the robust heuristic match (numeric within 5% or substring of gold answer).

Schema note: ROOT2/ROOT3 records store `resps[0][0]` as a plain string, while ROOT wrap it in a dict with `text_gen_output`. The unified `get_raw(rec)` handles both.


## Queue progress (live)

Updated 2026-04-26T15:49:46Z. **18 / 40** done · 8 running · 14 pending · 0 errored. Aggregate running ETA ≈ **5m12s**.

|             | `mmvet` | `mmstar` | `mmmu_val` | `vstar_bench` | `cv_bench_reasoning` | `chartqa` | `blink_jigsaw` | `VisualPuzzles_cot` |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `Unified-cp50` | ✅ 18.81% | ✅ 44.60% | 🏃 56% | ✅ 33.84% | 🏃 100% | ✅ 62.40% | ✅ 47.33% | 🏃 100% |
| `region-edit-cp50` | ✅ 20.64% | ✅ 44.40% | 🏃 45% | ✅ 29.32% | 🏃 100% | ✅ 63.36% | ✅ 47.33% | 🏃 100% |
| `answer-LavidaO-ckpt50` | ✅ 17.43% | 🏃 67% | ⏳ | ✅ 36.13% | ⏳ | ⏳ | ✅ 47.33% | ⏳ |
| `edit-LavidaO-ckpt50` | ✅ 18.35% | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | 🏃 100% | ⏳ |
| `interleave-cp50` | ✅ 19.72% | ✅ 41.93% | ⏳ | ⏳ | ⏳ | ✅ 66.16% | ✅ 48.00% | ⏳ |

Legend: ✅ done · 🏃 running (% from tqdm) · ⏳ pending · ♻ retrying · ❌ failed

### Per-job ETA

| ckpt | task | gpu | started | elapsed | tqdm % | remaining (ETA) |
|---|---|---:|---|---:|---:|---:|
| `Unified-cp50` | `mmmu_val` | 1 | 2026-04-26T14:48:06Z | 2m14s | 56% | 1m44s |
| `Unified-cp50` | `cv_bench_reasoning` | 2 | 2026-04-26T14:48:07Z | 3m19s | 100% | 0m00s |
| `Unified-cp50` | `VisualPuzzles_cot` | 3 | 2026-04-26T14:48:09Z | 4m03s | 100% | 0m00s |
| `region-edit-cp50` | `mmmu_val` | 4 | 2026-04-26T14:48:10Z | 1m48s | 45% | 2m10s |
| `region-edit-cp50` | `cv_bench_reasoning` | 5 | 2026-04-26T14:48:12Z | 3m19s | 100% | 0m00s |
| `region-edit-cp50` | `VisualPuzzles_cot` | 6 | 2026-04-26T14:48:13Z | 4m04s | 100% | 0m00s |
| `answer-LavidaO-ckpt50` | `mmstar` | 7 | 2026-04-26T14:48:15Z | 2m40s | 67% | 1m18s |
| `edit-LavidaO-ckpt50` | `blink_jigsaw` | 0 | 2026-04-26T14:37:18Z | 3m26s | 100% | 0m00s |

---


## `ROOT  (default)` — `image_gen_usebboxFalse_default`

Each cell: `default% → robust% (Δ pp, correct/N)`. `—` = file missing.

| task | LaViDa-O (base) | sft-zebracot | Unified-cp50 | region-edit-cp50 | answer-LavidaO-ckpt50 | edit-LavidaO-ckpt50 | interleave-cp50 |
|---|---|---|---|---|---|---|---|
| `chartqa` | 0.00% → **4.55%** (Δ +4.55pp, 117/2569) | 0.28% → **7.16%** (Δ +6.88pp, 179/2500) | 0.00% → **62.40%** (Δ +62.40pp, 1605/2572) | 0.00% → **63.36%** (Δ +63.36pp, 1584/2500) | — | — | 0.00% → **66.16%** (Δ +66.16pp, 1654/2500) |
| `cv_bench` | 0.00% → **16.09%** (Δ +16.09pp, 425/2642) | — | 0.00% → **64.48%** (Δ +64.48pp, 1701/2638) | 0.00% → **65.20%** (Δ +65.20pp, 1720/2638) | — | — | — |
| `cv_bench_reasoning` | — | — | — | — | — | — | — |
| `blink_jigsaw` | 0.00% → **25.32%** (Δ +25.32pp, 39/154) | 0.00% → **25.33%** (Δ +25.33pp, 38/150) | 0.00% → **47.33%** (Δ +47.33pp, 71/150) | 0.00% → **47.33%** (Δ +47.33pp, 71/150) | 0.00% → **47.33%** (Δ +47.33pp, 71/150) | — | 0.00% → **48.00%** (Δ +48.00pp, 72/150) |
| `vstar_bench` | — | — | 0.00% → **33.84%** (Δ +33.84pp, 89/263) | 0.00% → **29.32%** (Δ +29.32pp, 56/191) | 0.00% → **36.13%** (Δ +36.13pp, 69/191) | — | — |
| `VisPuzzle_direct` | 0.00% → **0.00%** (Δ +0.00pp, 0/400) | — | 0.00% → **0.00%** (Δ +0.00pp, 0/400) | 0.00% → **0.00%** (Δ +0.00pp, 0/400) | — | — | — |
| `mmstar` | — | 2.60% → **3.13%** (Δ +0.53pp, 47/1500) | 33.40% → **44.60%** (Δ +11.20pp, 669/1500) | 31.27% → **44.40%** (Δ +13.13pp, 666/1500) | — | — | 29.60% → **41.93%** (Δ +12.33pp, 629/1500) |
| `mmmu_val` | — | — | — | — | — | — | — |
| `mmvet` | — | 0.00% → **1.83%** (Δ +1.83pp, 4/218) | 0.00% → **18.81%** (Δ +18.81pp, 41/218) | 0.00% → **20.64%** (Δ +20.64pp, 45/218) | 0.00% → **17.43%** (Δ +17.43pp, 38/218) | 0.00% → **18.35%** (Δ +18.35pp, 40/218) | 0.00% → **19.72%** (Δ +19.72pp, 43/218) |
| `ai2d_lite` | — | — | — | — | — | — | — |
| `mathverse_testmini_vision_dominant` | — | — | — | — | — | — | — |
| `mathvista_testmini_format` | — | — | — | — | — | — | — |
| `scienceqa_img` | — | — | — | — | — | — | — |
| `VisualPuzzles_cot` | — | — | — | — | — | — | — |

### Parser pattern share (% of records)

| task | ckpt | N | empty% | structured% | raw% | top patterns |
|---|---|---:|---:|---:|---:|---|
| `chartqa` | LaViDa-O (base) | 2569 | 91.94 | 0.08 | 7.98 | empty:2362, raw:205, trailing_letter:2 |
| `chartqa` | sft-zebracot | 2500 | 87.36 | 3.48 | 9.16 | empty:2184, raw:229, the_answer_is:76 |
| `chartqa` | Unified-cp50 | 2572 | 0.89 | 0.43 | 98.68 | raw:2538, empty:23, trailing_letter:6 |
| `chartqa` | region-edit-cp50 | 2500 | 0.76 | 0.48 | 98.76 | raw:2469, empty:19, trailing_letter:10 |
| `chartqa` | interleave-cp50 | 2500 | 0.60 | 0.48 | 98.92 | raw:2473, empty:15, trailing_letter:11 |
| `cv_bench` | LaViDa-O (base) | 2642 | 77.37 | 22.60 | 0.04 | empty:2044, trailing_letter:597, raw:1 |
| `cv_bench` | Unified-cp50 | 2638 | 0.00 | 99.05 | 0.95 | trailing_letter:2613, raw:25 |
| `cv_bench` | region-edit-cp50 | 2638 | 0.00 | 99.09 | 0.91 | trailing_letter:2614, raw:24 |
| `blink_jigsaw` | LaViDa-O (base) | 154 | 52.60 | 47.40 | 0.00 | empty:81, trailing_letter:73 |
| `blink_jigsaw` | sft-zebracot | 150 | 52.67 | 47.33 | 0.00 | empty:79, trailing_letter:71 |
| `blink_jigsaw` | Unified-cp50 | 150 | 0.00 | 100.00 | 0.00 | trailing_letter:150 |
| `blink_jigsaw` | region-edit-cp50 | 150 | 0.00 | 100.00 | 0.00 | trailing_letter:150 |
| `blink_jigsaw` | answer-LavidaO-ckpt50 | 150 | 0.00 | 100.00 | 0.00 | trailing_letter:150 |
| `blink_jigsaw` | interleave-cp50 | 150 | 0.00 | 100.00 | 0.00 | trailing_letter:150 |
| `vstar_bench` | Unified-cp50 | 263 | 0.00 | 89.35 | 10.65 | the_answer_is:148, trailing_letter:87, raw:28 |
| `vstar_bench` | region-edit-cp50 | 191 | 0.00 | 91.10 | 8.90 | trailing_letter:104, the_answer_is:70, raw:17 |
| `vstar_bench` | answer-LavidaO-ckpt50 | 191 | 0.00 | 97.91 | 2.09 | trailing_letter:127, the_answer_is:60, raw:4 |
| `VisPuzzle_direct` | LaViDa-O (base) | 400 | 70.00 | 29.00 | 1.00 | empty:280, trailing_letter:103, the_answer_is:13 |
| `VisPuzzle_direct` | Unified-cp50 | 400 | 0.00 | 99.75 | 0.25 | the_answer_is:397, raw:1, answer_tag:1 |
| `VisPuzzle_direct` | region-edit-cp50 | 400 | 0.00 | 100.00 | 0.00 | the_answer_is:398, answer_tag:2 |
| `mmstar` | sft-zebracot | 1500 | 92.93 | 6.47 | 0.60 | empty:1394, the_answer_is:54, trailing_letter:36 |
| `mmstar` | Unified-cp50 | 1500 | 0.00 | 93.60 | 6.40 | the_answer_is:730, trailing_letter:506, answer_tag:166 |
| `mmstar` | region-edit-cp50 | 1500 | 0.00 | 92.93 | 7.07 | the_answer_is:722, trailing_letter:490, answer_tag:181 |
| `mmstar` | interleave-cp50 | 1500 | 0.00 | 90.07 | 9.93 | trailing_letter:612, the_answer_is:530, answer_tag:202 |
| `mmvet` | sft-zebracot | 218 | 76.61 | 14.68 | 8.72 | empty:167, the_answer_is:23, raw:19 |
| `mmvet` | Unified-cp50 | 218 | 0.46 | 57.34 | 42.20 | the_answer_is:121, raw:92, trailing_letter:2 |
| `mmvet` | region-edit-cp50 | 218 | 0.46 | 60.55 | 38.99 | the_answer_is:125, raw:85, answer_tag:7 |
| `mmvet` | answer-LavidaO-ckpt50 | 218 | 0.00 | 45.87 | 54.13 | raw:118, the_answer_is:99, trailing_letter:1 |
| `mmvet` | edit-LavidaO-ckpt50 | 218 | 0.46 | 66.97 | 32.57 | the_answer_is:137, raw:71, answer_tag:8 |
| `mmvet` | interleave-cp50 | 218 | 0.00 | 57.80 | 42.20 | the_answer_is:121, raw:92, answer_tag:3 |


## `ROOT2 (tok128_blk128_step64)` — `image_gen_usebboxFalse_tok128_blk128_step64_t0`

Each cell: `default% → robust% (Δ pp, correct/N)`. `—` = file missing.

| task | LaViDa-O (base) | sft-zebracot | Unified-cp50 | region-edit-cp50 | answer-LavidaO-ckpt50 | edit-LavidaO-ckpt50 | interleave-cp50 |
|---|---|---|---|---|---|---|---|
| `chartqa` | — | — | — | — | — | — | — |
| `cv_bench` | — | — | — | — | — | — | — |
| `cv_bench_reasoning` | — | — | — | — | — | — | — |
| `blink_jigsaw` | 0.00% → **47.33%** (Δ +47.33pp, 71/150) | 0.00% → **47.33%** (Δ +47.33pp, 71/150) | — | — | — | — | — |
| `vstar_bench` | 0.00% → **45.03%** (Δ +45.03pp, 86/191) | 0.00% → **40.84%** (Δ +40.84pp, 78/191) | — | — | — | — | — |
| `VisPuzzle_direct` | — | — | — | — | — | — | — |
| `mmstar` | — | — | — | — | — | — | — |
| `mmmu_val` | 0.00% → **44.11%** (Δ +44.11pp, 397/900) | 0.00% → **43.89%** (Δ +43.89pp, 395/900) | — | — | — | — | — |
| `mmvet` | — | — | — | — | — | — | — |
| `ai2d_lite` | — | — | — | — | — | — | — |
| `mathverse_testmini_vision_dominant` | — | — | — | — | — | — | — |
| `mathvista_testmini_format` | — | — | — | — | — | — | — |
| `scienceqa_img` | — | — | — | — | — | — | — |
| `VisualPuzzles_cot` | — | — | — | — | — | — | — |

### Parser pattern share (% of records)

| task | ckpt | N | empty% | structured% | raw% | top patterns |
|---|---|---:|---:|---:|---:|---|
| `blink_jigsaw` | LaViDa-O (base) | 150 | 0.00 | 100.00 | 0.00 | trailing_letter:150 |
| `blink_jigsaw` | sft-zebracot | 150 | 0.00 | 100.00 | 0.00 | trailing_letter:150 |
| `vstar_bench` | LaViDa-O (base) | 191 | 0.00 | 100.00 | 0.00 | trailing_letter:191 |
| `vstar_bench` | sft-zebracot | 191 | 0.00 | 100.00 | 0.00 | trailing_letter:191 |
| `mmmu_val` | LaViDa-O (base) | 900 | 0.00 | 95.33 | 4.67 | trailing_letter:707, the_answer_is:151, raw:42 |
| `mmmu_val` | sft-zebracot | 900 | 0.00 | 95.67 | 4.33 | trailing_letter:812, raw:39, the_answer_is:28 |


## `ROOT3 (tok256_blk128_step64)` — `image_gen_usebboxFalse_tok256_blk128_step64_t0`

Each cell: `default% → robust% (Δ pp, correct/N)`. `—` = file missing.

| task | LaViDa-O (base) | sft-zebracot | Unified-cp50 | region-edit-cp50 | answer-LavidaO-ckpt50 | edit-LavidaO-ckpt50 | interleave-cp50 |
|---|---|---|---|---|---|---|---|
| `chartqa` | 0.00% → **47.60%** (Δ +47.60pp, 1190/2500) | 1.16% → **27.32%** (Δ +26.16pp, 683/2500) | — | — | — | — | — |
| `cv_bench` | 0.00% → **66.34%** (Δ +66.34pp, 1750/2638) | 4.78% → **57.70%** (Δ +52.92pp, 1522/2638) | — | — | — | — | — |
| `cv_bench_reasoning` | — | — | — | — | — | — | — |
| `blink_jigsaw` | — | — | — | — | — | — | — |
| `vstar_bench` | — | — | — | — | — | — | — |
| `VisPuzzle_direct` | — | — | — | — | — | — | — |
| `mmstar` | 53.27% → **53.27%** (Δ +0.00pp, 799/1500) | 41.53% → **50.33%** (Δ +8.80pp, 755/1500) | — | — | — | — | — |
| `mmmu_val` | — | — | — | — | — | — | — |
| `mmvet` | 0.00% → **22.94%** (Δ +22.94pp, 50/218) | 0.00% → **23.85%** (Δ +23.85pp, 52/218) | — | — | — | — | — |
| `ai2d_lite` | 71.40% → **71.40%** (Δ +0.00pp, 357/500) | — | — | — | — | — | — |
| `mathverse_testmini_vision_dominant` | 0.00% → **31.60%** (Δ +31.60pp, 249/788) | 0.00% → **29.31%** (Δ +29.31pp, 231/788) | — | — | — | — | — |
| `mathvista_testmini_format` | 0.00% → **18.40%** (Δ +18.40pp, 184/1000) | 0.00% → **16.40%** (Δ +16.40pp, 164/1000) | — | — | — | — | — |
| `scienceqa_img` | 84.28% → **84.28%** (Δ +0.00pp, 1700/2017) | 79.72% → **83.44%** (Δ +3.72pp, 1683/2017) | — | — | — | — | — |
| `VisualPuzzles_cot` | 0.00% → **30.39%** (Δ +30.39pp, 355/1168) | 26.88% → **28.60%** (Δ +1.71pp, 334/1168) | — | — | — | — | — |

### Parser pattern share (% of records)

| task | ckpt | N | empty% | structured% | raw% | top patterns |
|---|---|---:|---:|---:|---:|---|
| `chartqa` | LaViDa-O (base) | 2500 | 0.00 | 0.08 | 99.92 | raw:2498, trailing_letter:2 |
| `chartqa` | sft-zebracot | 2500 | 0.00 | 23.48 | 76.52 | raw:1913, the_answer_is:490, answer_tag:95 |
| `cv_bench` | LaViDa-O (base) | 2638 | 0.00 | 87.95 | 12.05 | trailing_letter:2100, raw:318, the_answer_is:220 |
| `cv_bench` | sft-zebracot | 2638 | 0.00 | 92.57 | 7.43 | the_answer_is:1277, trailing_letter:909, raw:196 |
| `mmstar` | LaViDa-O (base) | 1500 | 0.00 | 100.00 | 0.00 | trailing_letter:1118, the_answer_is:382 |
| `mmstar` | sft-zebracot | 1500 | 0.00 | 99.80 | 0.20 | trailing_letter:823, the_answer_is:450, answer_tag:224 |
| `mmvet` | LaViDa-O (base) | 218 | 0.00 | 54.59 | 45.41 | the_answer_is:119, raw:99 |
| `mmvet` | sft-zebracot | 218 | 0.00 | 60.55 | 39.45 | raw:86, the_answer_is:77, answer_tag:54 |
| `ai2d_lite` | LaViDa-O (base) | 500 | 0.00 | 100.00 | 0.00 | trailing_letter:500 |
| `mathverse_testmini_vision_dominant` | LaViDa-O (base) | 788 | 0.00 | 90.10 | 9.90 | trailing_letter:392, the_answer_is:318, raw:78 |
| `mathverse_testmini_vision_dominant` | sft-zebracot | 788 | 0.00 | 98.98 | 1.02 | answer_tag:629, the_answer_is:131, trailing_letter:20 |
| `mathvista_testmini_format` | LaViDa-O (base) | 1000 | 0.00 | 59.30 | 40.70 | trailing_letter:506, raw:407, the_answer_is:87 |
| `mathvista_testmini_format` | sft-zebracot | 1000 | 0.00 | 81.00 | 19.00 | answer_tag:393, trailing_letter:290, raw:190 |
| `scienceqa_img` | LaViDa-O (base) | 2017 | 0.00 | 100.00 | 0.00 | trailing_letter:2017 |
| `scienceqa_img` | sft-zebracot | 2017 | 0.00 | 99.70 | 0.30 | trailing_letter:1914, answer_tag:68, the_answer_is:29 |
| `VisualPuzzles_cot` | LaViDa-O (base) | 1168 | 0.00 | 98.12 | 1.88 | the_answer_is:1138, raw:22, trailing_letter:8 |
| `VisualPuzzles_cot` | sft-zebracot | 1168 | 0.00 | 99.23 | 0.77 | answer_tag:1102, the_answer_is:52, raw:9 |
