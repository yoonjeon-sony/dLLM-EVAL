# Parser robustness report — first 32 samples per cell

Each table row is one record. Columns:

- `target` — gold answer (`rec.target`)

- `model response` — raw `text_gen_output` (truncated to 140 chars)

- `default pred` — what the lmms-eval default scorer would parse

- `robust pred` — what the extended robust parser extracts

- `default ✓` / `robust ✓` — evaluation result

- `pattern` — which extraction rule fired in the robust parser


## `ROOT  (default)` — `image_gen_usebboxFalse_default`


### ckpt: `LaViDa-O (base)`


#### task: `chartqa` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `14` | 12 | `` | `12` | ❌ | ❌ | raw |
| 1 | `3` | (empty) | `` | `` | ❌ | ❌ | empty |
| 2 | `0.57` | 0 | `` | `0` | ❌ | ❌ | raw |
| 3 | `No` | (empty) | `` | `` | ❌ | ❌ | empty |
| 4 | `14` | (empty) | `` | `` | ❌ | ❌ | empty |
| 5 | `3` | (empty) | `` | `` | ❌ | ❌ | empty |
| 6 | `23` | (empty) | `` | `` | ❌ | ❌ | empty |
| 7 | `62` | (empty) | `` | `` | ❌ | ❌ | empty |
| 8 | `Inspired` | Lonely | `` | `Lonely` | ❌ | ❌ | raw |
| 9 | `1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 10 | `17` | (empty) | `` | `` | ❌ | ❌ | empty |
| 11 | `2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 12 | `2014` | (empty) | `` | `` | ❌ | ❌ | empty |
| 13 | `1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 14 | `green line` | (empty) | `` | `` | ❌ | ❌ | empty |
| 15 | `29` | (empty) | `` | `` | ❌ | ❌ | empty |
| 16 | `Yes` | (empty) | `` | `` | ❌ | ❌ | empty |
| 17 | `80` | (empty) | `` | `` | ❌ | ❌ | empty |
| 18 | `0.6` | 23 | `` | `23` | ❌ | ❌ | raw |
| 19 | `13` | (empty) | `` | `` | ❌ | ❌ | empty |
| 20 | `0.57` | (empty) | `` | `` | ❌ | ❌ | empty |
| 21 | `No` | (empty) | `` | `` | ❌ | ❌ | empty |
| 22 | `6` | (empty) | `` | `` | ❌ | ❌ | empty |
| 23 | `Yes` | (empty) | `` | `` | ❌ | ❌ | empty |
| 24 | `0.03` | (empty) | `` | `` | ❌ | ❌ | empty |
| 25 | `0.08` | (empty) | `` | `` | ❌ | ❌ | empty |
| 26 | `21.6` | (empty) | `` | `` | ❌ | ❌ | empty |
| 27 | `2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 28 | `2011` | (empty) | `` | `` | ❌ | ❌ | empty |
| 29 | `No` | (empty) | `` | `` | ❌ | ❌ | empty |
| 30 | `2008` | (empty) | `` | `` | ❌ | ❌ | empty |
| 31 | `1.216666667` | (empty) | `` | `` | ❌ | ❌ | empty |


#### task: `cv_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(C)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 1 | `(C)` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 2 | `(E)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 3 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 4 | `(C)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 5 | `(C)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 6 | `(D)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 7 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 8 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 9 | `(D)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 10 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 11 | `(E)` | E | `` | `E` | ❌ | ✅ | trailing_letter |
| 12 | `(D)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 13 | `(E)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 14 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 15 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 16 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 17 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 18 | `(D)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 19 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 20 | `(E)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 21 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 22 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 23 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 24 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 25 | `(C)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 26 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 27 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 28 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 29 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 30 | `(F)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 31 | `(D)` | (empty) | `` | `` | ❌ | ❌ | empty |


#### task: `blink_jigsaw` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 1 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 2 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 3 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 6 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 7 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 8 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 9 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 10 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 11 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 12 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 13 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 14 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 15 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 16 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 17 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 18 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 19 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 20 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 21 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 22 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 23 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 24 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 25 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 26 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 27 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 28 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 29 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 31 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |


#### task: `VisPuzzle_direct` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `Part 1 should be to the right of Part 2` | B | `B` | `B` | ❌ | ❌ | trailing_letter |
| 1 | `Part 2 should be to the right of Part 1` | B | `B` | `B` | ❌ | ❌ | trailing_letter |
| 2 | `Part 2 should be to the right of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 3 | `Part 1 should be to the right of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 4 | `Part 2 should be to the left of Part 1` | B | `B` | `B` | ❌ | ❌ | trailing_letter |
| 5 | `Part 1 should be to the right of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 6 | `Part 2 should be to the left of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 7 | `Part 2 should be to the left of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 8 | `Part 2 should be to the left of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 9 | `Part 2 should be to the right of Part 1` | The answer is B | `The answer is B` | `B` | ❌ | ❌ | the_answer_is |
| 10 | `Part 1 should be to the right of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 11 | `Part 1 should be to the right of Part 2` | B | `B` | `B` | ❌ | ❌ | trailing_letter |
| 12 | `Part 1 should be to the right of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 13 | `Part 1 should be to the left of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 14 | `Part 1 should be to the right of Part 2` | A | `A` | `A` | ❌ | ❌ | trailing_letter |
| 15 | `Part 2 should be to the left of Part 1` | The answer is correct | `The answer is correct` | `correct` | ❌ | ❌ | the_answer_is |
| 16 | `Part 1 should be to the left of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 17 | `Part 1 should be to the right of Part 2` | The answer is correct | `The answer is correct` | `correct` | ❌ | ❌ | the_answer_is |
| 18 | `Part 2 should be to the right of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 19 | `Part 1 should be to the left of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 20 | `Part 2 should be to the left of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 21 | `Part 1 should be to the left of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 22 | `Part 1 should be to the left of Part 2` | A | `A` | `A` | ❌ | ❌ | trailing_letter |
| 23 | `Part 1 should be to the left of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 24 | `Part 2 should be to the right of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 25 | `Part 1 should be to the left of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 26 | `Part 1 should be to the right of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 27 | `Part 2 should be to the left of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 28 | `Part 2 should be to the left of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 29 | `Part 2 should be to the right of Part 1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 30 | `Part 1 should be to the left of Part 2` | A | `A` | `A` | ❌ | ❌ | trailing_letter |
| 31 | `Part 1 should be to the right of Part 2` | (empty) | `` | `` | ❌ | ❌ | empty |


### ckpt: `sft-zebracot`


#### task: `chartqa` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `14` | (empty) | `` | `` | ❌ | ❌ | empty |
| 1 | `3` | (empty) | `` | `` | ❌ | ❌ | empty |
| 2 | `23` | (empty) | `` | `` | ❌ | ❌ | empty |
| 3 | `62` | (empty) | `` | `` | ❌ | ❌ | empty |
| 4 | `Inspired` | (empty) | `` | `` | ❌ | ❌ | empty |
| 5 | `1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 6 | `17` | (empty) | `` | `` | ❌ | ❌ | empty |
| 7 | `2` | (empty) | `` | `` | ❌ | ❌ | empty |
| 8 | `2014` | 2014 | `` | `2014` | ❌ | ✅ | raw |
| 9 | `1` | (empty) | `` | `` | ❌ | ❌ | empty |
| 10 | `green line` | (empty) | `` | `` | ❌ | ❌ | empty |
| 11 | `29` | (empty) | `` | `` | ❌ | ❌ | empty |
| 12 | `Yes` | (empty) | `` | `` | ❌ | ❌ | empty |
| 13 | `80` | answer is 80. <answer> 80 </answer | `` | `80` | ❌ | ✅ | the_answer_is |
| 14 | `0.6` | 23 | `` | `23` | ❌ | ❌ | raw |
| 15 | `13` | (empty) | `` | `` | ❌ | ❌ | empty |
| 16 | `24` | (empty) | `` | `` | ❌ | ❌ | empty |
| 17 | `Italy` | (empty) | `` | `` | ❌ | ❌ | empty |
| 18 | `33` | (empty) | `` | `` | ❌ | ❌ | empty |
| 19 | `4` | (empty) | `` | `` | ❌ | ❌ | empty |
| 20 | `92` | 92 | `` | `92` | ❌ | ✅ | raw |
| 21 | `21` | (empty) | `` | `` | ❌ | ❌ | empty |
| 22 | `2009` | (empty) | `` | `` | ❌ | ❌ | empty |
| 23 | `68` | (empty) | `` | `` | ❌ | ❌ | empty |
| 24 | `83` | (empty) | `` | `` | ❌ | ❌ | empty |
| 25 | `Simulation` | (empty) | `` | `` | ❌ | ❌ | empty |
| 26 | `Yes` | (empty) | `` | `` | ❌ | ❌ | empty |
| 27 | `Yes` | (empty) | `` | `` | ❌ | ❌ | empty |
| 28 | `Democrat (scores 60 to 100)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 29 | `Libya` | (empty) | `` | `` | ❌ | ❌ | empty |
| 30 | `28` | 28 | `` | `28` | ❌ | ✅ | raw |
| 31 | `UK` | (empty) | `` | `` | ❌ | ❌ | empty |


#### task: `blink_jigsaw` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 1 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 2 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 3 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 6 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 7 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 8 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 9 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 10 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 11 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 12 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 13 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 14 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 15 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 16 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 17 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 18 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 19 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 20 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 21 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 22 | `(A)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 23 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 24 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 25 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 26 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 27 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 28 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |
| 29 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `(B)` | (empty) | `` | `` | ❌ | ❌ | empty |


#### task: `mmstar` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 1 | `D` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 2 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 3 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 4 | `B` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 5 | `D` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 6 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 7 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 8 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 9 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 10 | `B` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 11 | `C` | The answer is A. | `(first-char check)` | `A` | ❌ | ❌ | the_answer_is |
| 12 | `C` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 13 | `B` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 14 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 15 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 16 | `D` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 17 | `D` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 18 | `B` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 19 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 20 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 21 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 22 | `D` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 23 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 24 | `B` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 25 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 26 | `C` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 27 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 28 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 29 | `B` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 30 | `D` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |
| 31 | `A` | (empty) | `(first-char check)` | `` | ❌ | ❌ | empty |


#### task: `mmvet` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `-1<AND>-5` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 1 | `0.75<OR>=0.75<OR>3/4` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 2 | `7<OR>=7` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 3 | `5<OR>=5` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 4 | `14<OR>=14` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 5 | `0.11` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 6 | `344.97` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 7 | `25.8<OR>25,8` | 0> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 8 | `4.299<OR>4.3` | The answer is 5.25. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 9 | `conditioner` | The answer is bottle.> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 10 | `above the man<OR>top left` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 11 | `31` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 12 | `orange` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 13 | `Australia` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 14 | `16` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 15 | `same` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 16 | `warm place<AND>1 hour` | > make bread. </> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 17 | `tilt pan slightly` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 18 | `milk` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 19 | `nose` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 20 | `identify your audience` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 21 | `76.55<OR>76.56` | 06.55 </answer> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 22 | `packaging` | the category that did the worst job in plastic recycling is Construction. Therefore the answer is Construction. < <answer> Construction </> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 23 | `271` | 78.9p. ⏎  <answer> 78.9p </0 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 24 | `north` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 25 | `double garage` | <answer> living room. </answer> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 26 | `replace blub` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |
| 27 | `15 min` | step. ⏎  ⏎ The average wait time to climb out of bed is 3. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 28 | `6,233.15 billion U.S. dollars` | 2 is 5173.19. ⏎  <answer> 5173.16. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 29 | `Administration` | 0, is in the $3,000. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 30 | `India` | below answer is South Korea. <answer> South </answer> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 31 | `Feb` | (empty) | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | empty |


### ckpt: `Unified-cp50`


#### task: `chartqa` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `14` | 12 | `` | `12` | ❌ | ❌ | raw |
| 1 | `3` | 3 | `` | `3` | ❌ | ✅ | raw |
| 2 | `0.57` | 0.67 | `` | `0.67` | ❌ | ❌ | raw |
| 3 | `No` | No | `` | `No` | ❌ | ✅ | raw |
| 4 | `14` | 12 | `` | `12` | ❌ | ❌ | raw |
| 5 | `3` | 3 | `` | `3` | ❌ | ✅ | raw |
| 6 | `0.57` | 0.67 | `` | `0.67` | ❌ | ❌ | raw |
| 7 | `No` | No | `` | `No` | ❌ | ✅ | raw |
| 8 | `14` | 12 | `` | `12` | ❌ | ❌ | raw |
| 9 | `3` | 3 | `` | `3` | ❌ | ✅ | raw |
| 10 | `23` | 68 | `` | `68` | ❌ | ❌ | raw |
| 11 | `62` | 62 | `` | `62` | ❌ | ✅ | raw |
| 12 | `Inspired` | Depressed | `` | `Depressed` | ❌ | ❌ | raw |
| 13 | `1` | 1 | `` | `1` | ❌ | ✅ | raw |
| 14 | `17` | 17 | `` | `17` | ❌ | ✅ | raw |
| 15 | `2` | 2 | `` | `2` | ❌ | ✅ | raw |
| 16 | `2014` | 2014 | `` | `2014` | ❌ | ✅ | raw |
| 17 | `1` | 5 | `` | `5` | ❌ | ❌ | raw |
| 18 | `green line` | Red | `` | `Red` | ❌ | ❌ | raw |
| 19 | `29` | 29 | `` | `29` | ❌ | ✅ | raw |
| 20 | `Yes` | Yes | `` | `Yes` | ❌ | ✅ | raw |
| 21 | `80` | 80 | `` | `80` | ❌ | ✅ | raw |
| 22 | `0.6` | 63 | `` | `63` | ❌ | ❌ | raw |
| 23 | `13` | 12 | `` | `12` | ❌ | ❌ | raw |
| 24 | `24` | 0.46 | `` | `0.46` | ❌ | ❌ | raw |
| 25 | `Italy` | Spain | `` | `Spain` | ❌ | ❌ | raw |
| 26 | `33` | 33 | `` | `33` | ❌ | ✅ | raw |
| 27 | `4` | 4 | `` | `4` | ❌ | ✅ | raw |
| 28 | `92` | 92 | `` | `92` | ❌ | ✅ | raw |
| 29 | `21` | 21 | `` | `21` | ❌ | ✅ | raw |
| 30 | `2009` | 2009 | `` | `2009` | ❌ | ✅ | raw |
| 31 | `68` | 68 | `` | `68` | ❌ | ✅ | raw |


#### task: `cv_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(C)` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 1 | `(C)` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 2 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 3 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 6 | `(B)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 7 | `(E)` | E | `` | `E` | ❌ | ✅ | trailing_letter |
| 8 | `(D)` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 9 | `(E)` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 10 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 11 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 12 | `(B)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 13 | `(A)` | E | `` | `E` | ❌ | ❌ | trailing_letter |
| 14 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 15 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 16 | `(E)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 17 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 18 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 19 | `(B)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 20 | `(A)` | F | `` | `` | ❌ | ❌ | raw |
| 21 | `(C)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 22 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 23 | `(A)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 24 | `(B)` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 25 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 26 | `(F)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 27 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 28 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 29 | `(C)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 30 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 31 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |


#### task: `blink_jigsaw` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 1 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 2 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 3 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 6 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 7 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 8 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 9 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 10 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 11 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 12 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 13 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 14 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 15 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 16 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 17 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 18 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 19 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 20 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 21 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 22 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 23 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 24 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 25 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 26 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 27 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 28 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 29 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |


#### task: `vstar_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 1 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 2 | `C` | The color of the dustpan is red. Therefore the answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |
| 3 | `B` | The answer is B. | `` | `B` | ❌ | ✅ | the_answer_is |
| 4 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 5 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 6 | `C` | The color of the dustpan is red. Therefore the answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |
| 7 | `B` | The answer is B. | `` | `B` | ❌ | ✅ | the_answer_is |
| 8 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 9 | `A` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 10 | `C` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 11 | `B` | The answer is B. | `` | `B` | ❌ | ✅ | the_answer_is |
| 12 | `B` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 13 | `C` | The man in the image is wearing a helmet. | `` | `A` | ❌ | ❌ | raw |
| 14 | `C` | The slippers in the image are red. Therefore, the answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |
| 15 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 16 | `B` | The answer is B. | `` | `B` | ❌ | ✅ | the_answer_is |
| 17 | `C` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 18 | `C` | The answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |
| 19 | `C` | The answer is C. | `` | `C` | ❌ | ✅ | the_answer_is |
| 20 | `B` | The answer is A. | `` | `A` | ❌ | ❌ | the_answer_is |
| 21 | `B` | There is no parachute in the image, so the answer is A. | `` | `A` | ❌ | ❌ | the_answer_is |
| 22 | `C` | The answer is C. | `` | `C` | ❌ | ✅ | the_answer_is |
| 23 | `A` | The answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |
| 24 | `B` | The water bottle in the image is blue, which corresponds to option D. | `` | `D` | ❌ | ❌ | trailing_letter |
| 25 | `A` | The image does not contain a helmet. Therefore, the answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 26 | `D` | The color of the flower is red. Therefore the correct answer is C. | `` | `C` | ❌ | ❌ | the_answer_is |
| 27 | `B` | The answer is B. | `` | `B` | ❌ | ✅ | the_answer_is |
| 28 | `B` | There is no motorcycle in the image. | `` | `` | ❌ | ❌ | raw |
| 29 | `A` | The is no van in the image, so its color cannot be determined. | `` | `` | ❌ | ❌ | raw |
| 30 | `D` | The toothbrush in the image is clearly red. Therefore the answer is D. | `` | `D` | ❌ | ✅ | the_answer_is |
| 31 | `A` | The man in the image is wearing a cap that the color is red. | `` | `A` | ❌ | ✅ | raw |


#### task: `VisPuzzle_direct` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 1 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 2 | `Part 2 should be to the right of Part 1` | <answer </answer> | `<answer </answer>` | `</answer>` | ❌ | ❌ | the_answer_is |
| 3 | `Part 1 should be to the right of Part 2` | <answer </answer> | `<answer </answer>` | `</answer>` | ❌ | ❌ | the_answer_is |
| 4 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 5 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 6 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 7 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 8 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 9 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 10 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 11 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 12 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 13 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 14 | `Part 1 should be to the right of Part 2` | <\| </answer> | `<\| </answer>` | `>` | ❌ | ❌ | the_answer_is |
| 15 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 16 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 17 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 18 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 19 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 20 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 21 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 22 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 23 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 24 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 25 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 26 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 27 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 28 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 29 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 30 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 31 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |


#### task: `mmstar` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 1 | `D` | The answer is B. | `(first-char check)` | `B` | ❌ | ❌ | the_answer_is |
| 2 | `A` | The answer is D. | `(first-char check)` | `D` | ❌ | ❌ | the_answer_is |
| 3 | `A` | The answer is C. | `(first-char check)` | `C` | ❌ | ❌ | the_answer_is |
| 4 | `B` | The image shows a man with a geometric overlay on his face. <answer> D </answer> | `(first-char check)` | `D` | ❌ | ❌ | answer_tag |
| 5 | `D` | The answer is B. | `(first-char check)` | `B` | ❌ | ❌ | the_answer_is |
| 6 | `A` | The main focus of the image is option D. | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 7 | `A` | The main subject of the image is the sun. <answer> C | `(first-char check)` | `C` | ❌ | ❌ | the_answer_is |
| 8 | `A` | The image is B. | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 9 | `A` | The image displays a feeling of D. | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 10 | `B` | The primary focus of the image is option D. | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 11 | `C` | The answer is A. | `(first-char check)` | `A` | ❌ | ❌ | the_answer_is |
| 12 | `C` | The image shows a group of people marching. | `(first-char check)` | `A` | ❌ | ❌ | raw |
| 13 | `B` | The main focus of the image is <answer></answer> | `(first-char check)` | `` | ❌ | ❌ | answer_tag |
| 14 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 15 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 16 | `D` | The answer is C. | `(first-char check)` | `C` | ❌ | ❌ | the_answer_is |
| 17 | `D` | The dominant color in the image is blue. The answer is B. | `(first-char check)` | `B` | ❌ | ❌ | the_answer_is |
| 18 | `B` | The answer is C. | `(first-char check)` | `C` | ❌ | ❌ | the_answer_is |
| 19 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 20 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 21 | `A` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 22 | `D` | The answer is D. | `(first-char check)` | `D` | ✅ | ✅ | the_answer_is |
| 23 | `A` | To determine which image has the highest sharpness, let's examine each section closely. | `(first-char check)` | `` | ❌ | ❌ | raw |
| 24 | `B` | The main focus of the image is on the shoes. <answer> B | `(first-char check)` | `B` | ❌ | ✅ | the_answer_is |
| 25 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 26 | `C` | The answer is A. | `(first-char check)` | `A` | ❌ | ❌ | the_answer_is |
| 27 | `A` | The answer is B. | `(first-char check)` | `B` | ❌ | ❌ | the_answer_is |
| 28 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 29 | `B` | The image is a <answer> B </answer> | `(first-char check)` | `B` | ❌ | ✅ | answer_tag |
| 30 | `D` | The main color scheme of the image is brown and black. | `(first-char check)` | `` | ❌ | ❌ | raw |
| 31 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |


#### task: `mmvet` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `-1<AND>-5` | The problem asks for the value of x in the given equation. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 1 | `0.75<OR>=0.75<OR>3/4` | The answer is 3. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 2 | `7<OR>=7` | The answer is 11. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 3 | `5<OR>=5` | The answer is 5. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 4 | `14<OR>=14` | The answer is 9. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 5 | `0.11` | The answer is 0.11. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 6 | `344.97` | The answer is 397. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 7 | `25.8<OR>25,8` | The answer is 22. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 8 | `4.299<OR>4.3` | The answer is 6.25. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 9 | `conditioner` | The answer is conditioner. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 10 | `above the man<OR>top left` | The frisbee is above the man. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 11 | `31` | The answer is 26. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 12 | `orange` | The answer is orange. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 13 | `Australia` | The answer is Australia. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 14 | `16` | The answer is 7. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 15 | `same` | The answer is bottom jet. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 16 | `warm place<AND>1 hour` | The answer is bread. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 17 | `tilt pan slightly` | Chill the knife. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 18 | `milk` | C | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | trailing_letter |
| 19 | `nose` | The problem asks for the fourth step in the sequence. ⏎  ⏎ Draw a nose. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 20 | `identify your audience` | Identify the audience. <answer> Identify your audience. </answer> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | answer_tag |
| 21 | `76.55<OR>76.56` | The answer is 91.65. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 22 | `packaging` | The answer is Text Packaging. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 23 | `271` | 280.95 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 24 | `north` | The answer is east. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 25 | `double garage` | The answer is living room. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 26 | `replace blub` | Replace the bulb. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 27 | `15 min` | The answer is 3 minutes. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 28 | `6,233.15 billion U.S. dollars` | 5235.14 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 29 | `Administration` | The answer is Finance. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 30 | `India` | The answer is India. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 31 | `Feb` | The answer is June. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |


### ckpt: `region-edit-cp50`


#### task: `chartqa` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `14` | 12 | `` | `12` | ❌ | ❌ | raw |
| 1 | `3` | 3 | `` | `3` | ❌ | ✅ | raw |
| 2 | `23` | 62 | `` | `62` | ❌ | ❌ | raw |
| 3 | `62` | 62 | `` | `62` | ❌ | ✅ | raw |
| 4 | `Inspired` | Connected | `` | `Connected` | ❌ | ❌ | raw |
| 5 | `1` | 1 | `` | `1` | ❌ | ✅ | raw |
| 6 | `17` | 17 | `` | `17` | ❌ | ✅ | raw |
| 7 | `2` | 2 | `` | `2` | ❌ | ✅ | raw |
| 8 | `2014` | 2014 | `` | `2014` | ❌ | ✅ | raw |
| 9 | `1` | 5 | `` | `5` | ❌ | ❌ | raw |
| 10 | `green line` | Red | `` | `Red` | ❌ | ❌ | raw |
| 11 | `29` | 29 | `` | `29` | ❌ | ✅ | raw |
| 12 | `Yes` | Yes | `` | `Yes` | ❌ | ✅ | raw |
| 13 | `80` | 0.60 | `` | `0.60` | ❌ | ❌ | raw |
| 14 | `0.6` | 23 | `` | `23` | ❌ | ❌ | raw |
| 15 | `13` | 13 | `` | `13` | ❌ | ✅ | raw |
| 16 | `24` | 21 | `` | `21` | ❌ | ❌ | raw |
| 17 | `Italy` | Spain | `` | `Spain` | ❌ | ❌ | raw |
| 18 | `33` | 33 | `` | `33` | ❌ | ✅ | raw |
| 19 | `4` | 5 | `` | `5` | ❌ | ❌ | raw |
| 20 | `92` | 92 | `` | `92` | ❌ | ✅ | raw |
| 21 | `21` | 21 | `` | `21` | ❌ | ✅ | raw |
| 22 | `2009` | 2009 | `` | `2009` | ❌ | ✅ | raw |
| 23 | `68` | 68 | `` | `68` | ❌ | ✅ | raw |
| 24 | `83` | 85 | `` | `85` | ❌ | ✅ | raw |
| 25 | `Simulation` | Simulation | `` | `Simulation` | ❌ | ✅ | raw |
| 26 | `Yes` | No | `` | `No` | ❌ | ❌ | raw |
| 27 | `Yes` | Yes | `` | `Yes` | ❌ | ✅ | raw |
| 28 | `Democrat (scores 60 to 100)` | Republican | `` | `Republican` | ❌ | ❌ | raw |
| 29 | `Libya` | Libya | `` | `Libya` | ❌ | ✅ | raw |
| 30 | `28` | 29 | `` | `29` | ❌ | ✅ | raw |
| 31 | `UK` | UK | `` | `UK` | ❌ | ✅ | raw |


#### task: `cv_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(C)` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 1 | `(C)` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 2 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 3 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 6 | `(B)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 7 | `(E)` | E | `` | `E` | ❌ | ✅ | trailing_letter |
| 8 | `(D)` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 9 | `(E)` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 10 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 11 | `(A)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 12 | `(B)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 13 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 14 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 15 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 16 | `(E)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 17 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 18 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 19 | `(B)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 20 | `(A)` | F | `` | `` | ❌ | ❌ | raw |
| 21 | `(C)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 22 | `(A)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 23 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 24 | `(B)` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 25 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 26 | `(F)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 27 | `(D)` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 28 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 29 | `(C)` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 30 | `(A)` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 31 | `(A)` | A | `` | `A` | ❌ | ✅ | trailing_letter |


#### task: `blink_jigsaw` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 1 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 2 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 3 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 6 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 7 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 8 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 9 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 10 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 11 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 12 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 13 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 14 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 15 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 16 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 17 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 18 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 19 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 20 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 21 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 22 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 23 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 24 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 25 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 26 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 27 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 28 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 29 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |


#### task: `vstar_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 1 | `C` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 2 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 3 | `B` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 4 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 5 | `D` | The color of the cart is B. | `` | `B` | ❌ | ❌ | trailing_letter |
| 6 | `B` | The answer is A. | `` | `A` | ❌ | ❌ | the_answer_is |
| 7 | `A` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 8 | `B` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 9 | `D` | There is no tissue box in the image. | `` | `` | ❌ | ❌ | raw |
| 10 | `C` | The man in the image is wearing a helmet. | `` | `A` | ❌ | ❌ | raw |
| 11 | `B` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 12 | `C` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 13 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 14 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 15 | `D` | There is no suitcase in the image. | `` | `` | ❌ | ❌ | raw |
| 16 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 17 | `D` | The answer is D. | `` | `D` | ❌ | ✅ | the_answer_is |
| 18 | `C` | The answer is A. | `` | `A` | ❌ | ❌ | the_answer_is |
| 19 | `A` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 20 | `C` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 21 | `A` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 22 | `C` | The answer is C. | `` | `C` | ❌ | ✅ | the_answer_is |
| 23 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 24 | `B` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 25 | `D` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 26 | `B` | There is no parachute in the image, so the answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 27 | `B` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 28 | `C` | The answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |
| 29 | `B` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 30 | `A` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 31 | `C` | The answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |


#### task: `VisPuzzle_direct` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 1 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 2 | `Part 2 should be to the right of Part 1` | The answer is A. | `The answer is A.` | `A` | ❌ | ❌ | the_answer_is |
| 3 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 4 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 5 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 6 | `Part 2 should be to the left of Part 1` | <answer </answer> | `<answer </answer>` | `</answer>` | ❌ | ❌ | the_answer_is |
| 7 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 8 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 9 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 10 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 11 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 12 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 13 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 14 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 15 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 16 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 17 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 18 | `Part 2 should be to the right of Part 1` | The answer is A. | `The answer is A.` | `A` | ❌ | ❌ | the_answer_is |
| 19 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 20 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 21 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 22 | `Part 1 should be to the left of Part 2` | The answer is A. | `The answer is A.` | `A` | ❌ | ❌ | the_answer_is |
| 23 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 24 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 25 | `Part 1 should be to the left of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 26 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 27 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 28 | `Part 2 should be to the left of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 29 | `Part 2 should be to the right of Part 1` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |
| 30 | `Part 1 should be to the left of Part 2` | The answer is A. | `The answer is A.` | `A` | ❌ | ❌ | the_answer_is |
| 31 | `Part 1 should be to the right of Part 2` | <\|reservedanswer> | `<\|reservedanswer>` | `>` | ❌ | ❌ | the_answer_is |


#### task: `mmstar` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 1 | `D` | The answer is B. | `(first-char check)` | `B` | ❌ | ❌ | the_answer_is |
| 2 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 3 | `A` | The answer is C. | `(first-char check)` | `C` | ❌ | ❌ | the_answer_is |
| 4 | `B` | The image is a <answer> | `(first-char check)` | `` | ❌ | ❌ | the_answer_is |
| 5 | `D` | The answer is D. | `(first-char check)` | `D` | ✅ | ✅ | the_answer_is |
| 6 | `A` | The main focus of this image is option D. D. | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 7 | `A` | The main subject of the image is the sun. Therefore correct answer is C. | `(first-char check)` | `C` | ❌ | ❌ | the_answer_is |
| 8 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 9 | `A` | The image conveys a sense of C. | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 10 | `B` | The primary focus of the image is option D. | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 11 | `C` | The main color scheme of the image is dominated by shades of blue. | `(first-char check)` | `` | ❌ | ❌ | raw |
| 12 | `C` | The image shows a group of people <answer </answer | `(first-char check)` | `` | ❌ | ❌ | the_answer_is |
| 13 | `B` | The main focus of the image is <answer></answer> A | `(first-char check)` | `A` | ❌ | ❌ | answer_tag |
| 14 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 15 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 16 | `D` | The answer is C. | `(first-char check)` | `C` | ❌ | ❌ | the_answer_is |
| 17 | `D` | The dominant color in the image is blue. The answer is B. | `(first-char check)` | `B` | ❌ | ❌ | the_answer_is |
| 18 | `B` | The answer is C. | `(first-char check)` | `C` | ❌ | ❌ | the_answer_is |
| 19 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 20 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 21 | `A` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 22 | `D` | The answer is D. | `(first-char check)` | `D` | ✅ | ✅ | the_answer_is |
| 23 | `A` | To determine which image has the highest sharpness, let's examine each section visually. | `(first-char check)` | `` | ❌ | ❌ | raw |
| 24 | `B` | The answer is B. | `(first-char check)` | `B` | ✅ | ✅ | the_answer_is |
| 25 | `A` | To determine which image is more colorful, I will compare the two images. <answer> B </answer> | `(first-char check)` | `B` | ❌ | ❌ | answer_tag |
| 26 | `C` | The answer is A. | `(first-char check)` | `A` | ❌ | ❌ | the_answer_is |
| 27 | `A` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 28 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |
| 29 | `B` | The answer is A. | `(first-char check)` | `A` | ❌ | ❌ | the_answer_is |
| 30 | `D` | The main color scheme of this image is Red and black. | `(first-char check)` | `` | ❌ | ❌ | raw |
| 31 | `A` | The answer is A. | `(first-char check)` | `A` | ✅ | ✅ | the_answer_is |


#### task: `mmvet` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `-1<AND>-5` | The problem asks for the value of x in the given equation. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 1 | `0.75<OR>=0.75<OR>3/4` | The problem asks | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 2 | `7<OR>=7` | The answer is 11 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 3 | `5<OR>=5` | The problem asks for the value of y in the last row of the matrix. ⏎  ⏎ answer> 5 </answer> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 4 | `14<OR>=14` | The answer is 9. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 5 | `0.11` | The answer is 0.11. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 6 | `344.97` | The answer is 419.97. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 7 | `25.8<OR>25,8` | The answer is 24. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 8 | `4.299<OR>4.3` | The answer is 2.85. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 9 | `conditioner` | The answer is conditioner. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 10 | `above the man<OR>top left` | The frisbee is above the man. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 11 | `31` | The answer is 26. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 12 | `orange` | The answer is orange. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 13 | `Australia` | The answer is Russia. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 14 | `16` | The answer is 7. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 15 | `same` | The answer is larger. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 16 | `warm place<AND>1 hour` | rise the dough. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 17 | `tilt pan slightly` | Wash the knife thoroughly. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 18 | `milk` | The answer is egg. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 19 | `nose` | The problem asks for the fourth step in the sequence. ⏎  ⏎ Based on the sequence of instructions, the fourth step is to draw the nose. ⏎ ... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | answer_tag |
| 20 | `identify your audience` | Identify your audience. <answer> Identify your audience </answer> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | answer_tag |
| 21 | `76.55<OR>76.56` | <\|_fake\|> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 22 | `packaging` | The answer is Packaging. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 23 | `271` | The answer is 78.9. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 24 | `north` | The answer is South. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 25 | `double garage` | The answer is living room. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 26 | `replace blub` | Replace the bulb. <answer> Replace the bulb </answer> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 27 | `15 min` | The answer is 3 minutes. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 28 | `6,233.15 billion U.S. dollars` | 5235.14 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 29 | `Administration` | The answer is Finance. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 30 | `India` | The answer is India. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 31 | `Feb` | The answer is August. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |


### ckpt: `answer-LavidaO-ckpt50`


#### task: `blink_jigsaw` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 1 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 2 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 3 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 6 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 7 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 8 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 9 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 10 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 11 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 12 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 13 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 14 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 15 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 16 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 17 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 18 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 19 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 20 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 21 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 22 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 23 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 24 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 25 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 26 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 27 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 28 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 29 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |


#### task: `vstar_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 1 | `C` | The answer is C. | `` | `C` | ❌ | ✅ | the_answer_is |
| 2 | `A` | The answer is A. | `` | `A` | ❌ | ✅ | the_answer_is |
| 3 | `B` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 4 | `C` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 5 | `D` | The answer is A. | `` | `A` | ❌ | ❌ | the_answer_is |
| 6 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 7 | `A` | The answer is C. | `` | `C` | ❌ | ❌ | the_answer_is |
| 8 | `B` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 9 | `D` | The answer is D. | `` | `D` | ❌ | ✅ | the_answer_is |
| 10 | `C` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 11 | `B` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 12 | `C` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 13 | `A` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 14 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 15 | `D` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 16 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 17 | `D` | The answer is D. | `` | `D` | ❌ | ✅ | the_answer_is |
| 18 | `C` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 19 | `A` | The answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |
| 20 | `C` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 21 | `A` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 22 | `C` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 23 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 24 | `B` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 25 | `D` | The answer is D. | `` | `D` | ❌ | ✅ | the_answer_is |
| 26 | `B` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |
| 27 | `B` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 28 | `C` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 29 | `B` | The answer is B. | `` | `B` | ❌ | ✅ | the_answer_is |
| 30 | `A` | The answer is B. | `` | `B` | ❌ | ❌ | the_answer_is |
| 31 | `C` | The answer is D. | `` | `D` | ❌ | ❌ | the_answer_is |


#### task: `mmvet` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `-1<AND>-5` | The problem is a 3. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 1 | `0.75<OR>=0.75<OR>3/4` | The answer is 3. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 2 | `7<OR>=7` | The answer is 11. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 3 | `5<OR>=5` | The answer is 39. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 4 | `14<OR>=14` | 0 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 5 | `0.11` | 0.11. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 6 | `344.97` | The answer is 100. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 7 | `25.8<OR>25,8` | The answer is 24.00. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 8 | `4.299<OR>4.3` | The answer is 2.50. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 9 | `conditioner` | The answer is conditioner. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 10 | `above the man<OR>top left` | above the man | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 11 | `31` | The answer is 2.. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 12 | `orange` | The answer is orange. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 13 | `Australia` | The answer is Australia. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 14 | `16` | The answer is 3.. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 15 | `same` | The answer is larger. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 16 | `warm place<AND>1 hour` | rise the dough. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 17 | `tilt pan slightly` | Wash the knife thoroughly. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 18 | `milk` | C | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | trailing_letter |
| 19 | `nose` | The answer is nose. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 20 | `identify your audience` | Identify your audience. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 21 | `76.55<OR>76.56` | <\|reservedake\|> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 22 | `packaging` | The answer is Packaging. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 23 | `271` | The answer is 280.. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 24 | `north` | The answer is east. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 25 | `double garage` | living room | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 26 | `replace blub` | Replace the bulb. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 27 | `15 min` | The answer is 3 times. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 28 | `6,233.15 billion U.S. dollars` | 130.15 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 29 | `Administration` | The answer is Finance. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 30 | `India` | The answer is India. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 31 | `Feb` | The answer is August. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |


## `ROOT2 (tok128_blk128_step64)` — `image_gen_usebboxFalse_tok128_blk128_step64_t0`


### ckpt: `LaViDa-O (base)`


#### task: `blink_jigsaw` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 1 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 2 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 3 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 6 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 7 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 8 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 9 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 10 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 11 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 12 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 13 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 14 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 15 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 16 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 17 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 18 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 19 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 20 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 21 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 22 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 23 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 24 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 25 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 26 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 27 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 28 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 29 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |


#### task: `vstar_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 1 | `C` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 2 | `A` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 3 | `B` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 4 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 5 | `D` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 6 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 7 | `A` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 8 | `B` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 9 | `D` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 10 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 11 | `B` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 12 | `C` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 13 | `A` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 14 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 15 | `D` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 16 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 17 | `D` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 18 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 19 | `A` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 20 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 21 | `A` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 22 | `C` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 23 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 24 | `B` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 25 | `D` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 26 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 27 | `B` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 28 | `C` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 29 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `A` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `C` | B | `` | `B` | ❌ | ❌ | trailing_letter |


### ckpt: `sft-zebracot`


#### task: `blink_jigsaw` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 1 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 2 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 3 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 4 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 5 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 6 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 7 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 8 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 9 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 10 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 11 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 12 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 13 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 14 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 15 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 16 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 17 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 18 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 19 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 20 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 21 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 22 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 23 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 24 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 25 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 26 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 27 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 28 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 29 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `(A)` | B | `` | `B` | ❌ | ❌ | trailing_letter |


#### task: `vstar_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 1 | `C` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 2 | `A` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 3 | `B` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 4 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 5 | `D` | D | `` | `D` | ❌ | ✅ | trailing_letter |
| 6 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 7 | `A` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 8 | `B` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 9 | `D` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 10 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 11 | `B` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 12 | `C` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 13 | `A` | A | `` | `A` | ❌ | ✅ | trailing_letter |
| 14 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 15 | `D` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 16 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 17 | `D` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 18 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 19 | `A` | B | `` | `B` | ❌ | ❌ | trailing_letter |
| 20 | `C` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 21 | `A` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 22 | `C` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 23 | `B` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 24 | `B` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 25 | `D` | C | `` | `C` | ❌ | ❌ | trailing_letter |
| 26 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 27 | `B` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 28 | `C` | C | `` | `C` | ❌ | ✅ | trailing_letter |
| 29 | `B` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 30 | `A` | D | `` | `D` | ❌ | ❌ | trailing_letter |
| 31 | `C` | B | `` | `B` | ❌ | ❌ | trailing_letter |


## `ROOT3 (tok256_blk128_step64)` — `image_gen_usebboxFalse_tok256_blk128_step64_t0`


### ckpt: `LaViDa-O (base)`


#### task: `chartqa` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `14` | 10 | `` | `10` | ❌ | ❌ | raw |
| 1 | `0.57` | 56.34 | `` | `56.34` | ❌ | ❌ | raw |
| 2 | `3` | 3 | `` | `3` | ❌ | ✅ | raw |
| 3 | `No` | No | `` | `No` | ❌ | ✅ | raw |
| 4 | `23` | 23 | `` | `23` | ❌ | ✅ | raw |
| 5 | `6` | 6 | `` | `6` | ❌ | ✅ | raw |
| 6 | `62` | 62 | `` | `62` | ❌ | ✅ | raw |
| 7 | `Yes` | Yes. | `` | `Yes.` | ❌ | ❌ | raw |
| 8 | `Inspired` | Depry | `` | `Depry` | ❌ | ❌ | raw |
| 9 | `0.03` | 3 | `` | `3` | ❌ | ❌ | raw |
| 10 | `1` | 1.17 | `` | `1.17` | ❌ | ❌ | raw |
| 11 | `0.08` | 0.08 | `` | `0.08` | ❌ | ✅ | raw |
| 12 | `17` | 17 | `` | `17` | ❌ | ✅ | raw |
| 13 | `21.6` | 19.3 | `` | `19.3` | ❌ | ❌ | raw |
| 14 | `2` | 2 | `` | `2` | ❌ | ✅ | raw |
| 15 | `2` | 2 | `` | `2` | ❌ | ✅ | raw |
| 16 | `2014` | 2014 | `` | `2014` | ❌ | ✅ | raw |
| 17 | `2011` | 2014 | `` | `2014` | ❌ | ✅ | raw |
| 18 | `1` | 5 | `` | `5` | ❌ | ❌ | raw |
| 19 | `No` | Yes | `` | `Yes` | ❌ | ❌ | raw |
| 20 | `green line` | Blue | `` | `Blue` | ❌ | ❌ | raw |
| 21 | `2008` | 2006 | `` | `2006` | ❌ | ✅ | raw |
| 22 | `29` | 29 | `` | `29` | ❌ | ✅ | raw |
| 23 | `1.216666667` | 1.1528 | `` | `1.1528` | ❌ | ❌ | raw |
| 24 | `Yes` | Yes | `` | `Yes` | ❌ | ✅ | raw |
| 25 | `Yes` | Yes | `` | `Yes` | ❌ | ✅ | raw |
| 26 | `80` | 80 | `` | `80` | ❌ | ✅ | raw |
| 27 | `77` | 77 | `` | `77` | ❌ | ✅ | raw |
| 28 | `0.6` | 23 | `` | `23` | ❌ | ❌ | raw |
| 29 | `61` | 59 | `` | `59` | ❌ | ✅ | raw |
| 30 | `13` | 10 | `` | `10` | ❌ | ❌ | raw |
| 31 | `5` | 5 | `` | `5` | ❌ | ✅ | raw |


#### task: `cv_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(C)` | To determine the number of organs in the image, we need to carefully observe the objects present.  ⏎  ⏎ 1. **Identify the Objects**: The ... | `` | `C` | ❌ | ✅ | the_answer_is |
| 1 | `(E)` | To determine the number of cushions in the image, we need to carefully observe the bed and count each cushion present. ⏎  ⏎ 1. Identify t... | `` | `B` | ❌ | ❌ | the_answer_is |
| 2 | `(C)` | To determine the number of table lamps in the image, we need to carefully observe the room depicted. | `` | `` | ❌ | ❌ | raw |
| 3 | `(D)` | To determine the number of curtains in the image, we need to carefully observe the room. | `` | `` | ❌ | ❌ | raw |
| 4 | `(D)` | To determine the number of pictures in the image, we need to carefully examine the visual content. | `` | `` | ❌ | ❌ | raw |
| 5 | `(B)` | To determine the number of beds in the image, we need to carefully observe the objects present.  ⏎  ⏎ 1. Identify the objects in the imag... | `` | `A` | ❌ | ❌ | the_answer_is |
| 6 | `(A)` | To determine the number of walls in the image, we need to carefully observe the structure and boundaries depicted. ⏎  ⏎ 1. **Identify the... | `` | `A` | ❌ | ✅ | raw |
| 7 | `(B)` | To determine the number of televisions in the image, we need to carefully examine the room depicted. | `` | `` | ❌ | ❌ | raw |
| 8 | `(B)` | To determine the number of chests in the image, we need to carefully observe the objects present.  ⏎  ⏎ 1. Identify the objects in the im... | `` | `B` | ❌ | ✅ | the_answer_is |
| 9 | `(E)` | To determine the number of walls in the image, we need to carefully observe the structure and layout of the room. ⏎  ⏎ 1. **Identify the ... | `` | `C` | ❌ | ❌ | the_answer_is |
| 10 | `(D)` | To determine the number of windows in the image, we need to carefully examine the room depicted. | `` | `` | ❌ | ❌ | raw |
| 11 | `(D)` | To determine the number of hats in the image, we need to carefully examine the objects depicted. | `` | `` | ❌ | ❌ | raw |
| 12 | `(B)` | To determine the number of curtains in the image, we need to carefully observe the visible depicted. | `` | `` | ❌ | ❌ | raw |
| 13 | `(B)` | To determine the number of pendant lamps in the image, we need to carefully observe the scene.. | `` | `` | ❌ | ❌ | raw |
| 14 | `(E)` | To determine the number of windows in the image, we need to carefully observe the scene. | `` | `` | ❌ | ❌ | raw |
| 15 | `(B)` | To determine the number of candle holders in the image, we need to carefully observe the objects present.  ⏎  ⏎ 1. **Identify the candle**: | `` | `` | ❌ | ❌ | raw |
| 16 | `(D)` | To determine the number of glasses in the image, we need to carefully examine the objects depicted.  ⏎  ⏎ 1. **Identify objects in the im... | `` | `` | ❌ | ❌ | the_answer_is |
| 17 | `(A)` | To determine the number of pictures in the image, we need to carefully examine the visual content. | `` | `` | ❌ | ❌ | raw |
| 18 | `(E)` | To determine the number of armchairs in the image, we need to carefully observe the seating arrangement.  ⏎  ⏎ 1. Identify the seating op... | `` | `E` | ❌ | ✅ | the_answer_is |
| 19 | `(A)` | To Answer: A | `` | `A` | ❌ | ✅ | the_answer_is |
| 20 | `(A)` | To determine the number of buckets in the image, we need to carefully observe the objects present.  ⏎  ⏎ 1. Identify the objects: | `` | `` | ❌ | ❌ | raw |
| 21 | `(B)` | To determine the number of blinds in the image, we need to carefully observe the windows. | `` | `` | ❌ | ❌ | raw |
| 22 | `(A)` | To determine the number of walls in the image, we need to carefully observe the structure and count the distinct walls visible. ⏎  ⏎ 1. *... | `` | `` | ❌ | ❌ | raw |
| 23 | `(C)` | To determine the number of curtains in the image, we need to carefully observe the window area. There are two curtains visible on the win... | `` | `C` | ❌ | ✅ | the_answer_is |
| 24 | `(B)` | B | `` | `B` | ❌ | ✅ | trailing_letter |
| 25 | `(D)` | A | `` | `A` | ❌ | ❌ | trailing_letter |
| 26 | `(A)` | To determine the number of walls in the image, we need to carefully observe the structure and count the visible walls. ⏎  ⏎ 1. **Identify... | `` | `A` | ❌ | ✅ | raw |
| 27 | `(B)` | To determine the number of light switches in the image, we need to carefully examine the kitchen areas. | `` | `` | ❌ | ❌ | raw |
| 28 | `(D)` | To determine the number of blinds in the image, we need to carefully observe the window area. | `` | `` | ❌ | ❌ | raw |
| 29 | `(C)` | To determine the number of ranges in the image, we need to carefully observe the kitchen layout.  ⏎  ⏎ 1. **Identify the Range**: The ran... | `` | `` | ❌ | ❌ | raw |
| 30 | `(A)` | To determine the number of dishwashers in the image, we need to carefully examine the kitchen area.  ⏎  ⏎ 1. **Observation**: the image s... | `` | `C` | ❌ | ❌ | the_answer_is |
| 31 | `(B)` | To determine the number of windows in the image, we need to carefully observe the kitchen layout. | `` | `` | ❌ | ❌ | raw |


#### task: `mmstar` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 1 | `B` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 2 | `D` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 3 | `C` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 4 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 5 | `D` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 6 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 7 | `D` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 8 | `B` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 9 | `D` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 10 | `D` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 11 | `D` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 12 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 13 | `B` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 14 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 15 | `C` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 16 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 17 | `C` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 18 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 19 | `B` | B | `(first-char check)` | `B` | ✅ | ✅ | trailing_letter |
| 20 | `B` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 21 | `B` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 22 | `C` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 23 | `A` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 24 | `C` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 25 | `B` | B | `(first-char check)` | `B` | ✅ | ✅ | trailing_letter |
| 26 | `B` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 27 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 28 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 29 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 30 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 31 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |


#### task: `mmvet` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `-1<AND>-5` | The answer is -5. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 1 | `1.25<OR>=1.25<OR>5/4` | The answer is d | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 2 | `0.75<OR>=0.75<OR>3/4` | The answer is 3. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 3 | `12<OR>=12` | The answer is 5. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 4 | `7<OR>=7` | The answer is 9 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 5 | `2<OR>=2` | The answer is 1. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 6 | `5<OR>=5` | 5 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 7 | `13<OR>=13` | The answer is 1. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 8 | `14<OR>=14` | The answer is 9. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 9 | `13.6<OR>13.7` | To Answer: | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 10 | `0.11` | To find the price difference between diesel and regular gasoline, we need to subtract the price of regular gasoline from the price of die... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 11 | `249.98` | To find the total price for a bottle of Merlot and a bottle of Cabernet, we need to add the prices of each wine together. ⏎  ⏎ The price ... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 12 | `344.97` | To find the cost of three bottles of Moet champagne, we need to determine the cost of one bottle and then multiply it by three. ⏎  ⏎ Step... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 13 | `14.8<OR>14,8` | The answer is 14.00 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 14 | `25.8<OR>25,8` | To Answer: A | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 15 | `14` | To find the original price of the pork belly before the discount, we can to follow the steps below ⏎  ⏎ Step 1: Identify the discounted p... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 16 | `4.299<OR>4.3` | To determine the gasoline price, we need to analyze the information provided in the image. The image shows a gas pump with the price disp... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 17 | `blue` | The answer is blue. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 18 | `conditioner` | The answer is conditioner. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 19 | `mousse` | To determine what is in the rightmost bottle, we need to carefully examine the label on the bottle. The label clearly states "MOISTURIZING | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 20 | `above the man<OR>top left` | To determine the spatial relationship between the frisbee and the man, we need to analyze their positions relative to each other. ⏎  ⏎ 1.... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 21 | `no<OR>empty` | To determine which car is in parking spot number 33, we need to consider the the following steps. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 22 | `31` | To determine which parking spot the red car is parked in, we need to carefully examine the image and follow the steps below: ⏎  ⏎ 1. **Id... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 23 | `eight<OR>8.0` | To determine the price of tomatoes, we need to carefully examine the image provided. The image shows a variety of fruits and vegetables w... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 24 | `orange` | The answer is orange. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 25 | `35%` | Therefore, the total international awareness of Tina Fey is 64%. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 26 | `Australia` | To determine the country where Tina Fey has the highest level of total awareness, we need to examine the "Total" column in the image. ⏎  ... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 27 | `mantle` | The answer is Mant. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 28 | `16` | To determine the number displayed on the motorcycle ridden by the motorcyclist wearing a yellow suit, we need to closely examine the imag... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 29 | `4` | The answer is 7. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 30 | `same` | To answer is jet. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 31 | `no` | To determine if the person is bigger than the car, we need to compare their sizes.  ⏎  ⏎ 1. **Observation the Person**: The person is sta... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |


#### task: `ai2d_lite` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 1 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 2 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 3 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 4 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 5 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 6 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 7 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 8 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 9 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 10 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 11 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 12 | `B` | A | `A` | `A` | ❌ | ❌ | trailing_letter |
| 13 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 14 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 15 | `C` | D | `D` | `D` | ❌ | ❌ | trailing_letter |
| 16 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 17 | `D` | A | `A` | `A` | ❌ | ❌ | trailing_letter |
| 18 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 19 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 20 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 21 | `C` | D | `D` | `D` | ❌ | ❌ | trailing_letter |
| 22 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 23 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 24 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 25 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 26 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 27 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 28 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 29 | `D` | A | `A` | `A` | ❌ | ❌ | trailing_letter |
| 30 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 31 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |


#### task: `mathverse_testmini_vision_dominant` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `D` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 1 | `C` | C | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | trailing_letter |
| 2 | `A` | A | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | trailing_letter |
| 3 | `D` | D | `(needs gpt-4 judge)` | `D` | ❌ | ✅ | trailing_letter |
| 4 | `A` | D | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | trailing_letter |
| 5 | `C` | C | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | trailing_letter |
| 6 | `B` | A | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 7 | `C` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 8 | `A` | C | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | trailing_letter |
| 9 | `A` | A | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | trailing_letter |
| 10 | `A` | A | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | trailing_letter |
| 11 | `A` | A | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | trailing_letter |
| 12 | `C` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 13 | `B` | B | `(needs gpt-4 judge)` | `B` | ❌ | ✅ | trailing_letter |
| 14 | `D` | D | `(needs gpt-4 judge)` | `D` | ❌ | ✅ | trailing_letter |
| 15 | `A` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 16 | `C` | C | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | trailing_letter |
| 17 | `C` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 18 | `C` | D | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | trailing_letter |
| 19 | `A` | A | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | trailing_letter |
| 20 | `D` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 21 | `C` | C | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | trailing_letter |
| 22 | `C` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 23 | `B` | B | `(needs gpt-4 judge)` | `B` | ❌ | ✅ | trailing_letter |
| 24 | `D` | D | `(needs gpt-4 judge)` | `D` | ❌ | ✅ | trailing_letter |
| 25 | `C` | C | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | trailing_letter |
| 26 | `A` | A | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | trailing_letter |
| 27 | `B` | C | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | trailing_letter |
| 28 | `C` | C | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | trailing_letter |
| 29 | `C` | C | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | trailing_letter |
| 30 | `B` | B | `(needs gpt-4 judge)` | `B` | ❌ | ✅ | trailing_letter |
| 31 | `C` | C | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | trailing_letter |


#### task: `mathvista_testmini_format` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `1.2` | 0.1 m | `(needs gpt-4 judge)` | `0.1` | ❌ | ❌ | raw |
| 1 | `1000` | The answer is 250. | `(needs gpt-4 judge)` | `250` | ❌ | ❌ | the_answer_is |
| 2 | `145°` | C | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | trailing_letter |
| 3 | `4` | 4 | `(needs gpt-4 judge)` | `4` | ❌ | ✅ | raw |
| 4 | `97` | C | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | trailing_letter |
| 5 | `6cm` | D | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | trailing_letter |
| 6 | `quarter` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 7 | `Yes` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 8 | `8` | D | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | trailing_letter |
| 9 | `E` | D | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | trailing_letter |
| 10 | `5` | 7 | `(needs gpt-4 judge)` | `7` | ❌ | ❌ | raw |
| 11 | `Yes` | A. | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 12 | `0` | zero. | `(needs gpt-4 judge)` | `zero.` | ❌ | ❌ | raw |
| 13 | `Yes` | A. | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 14 | `Common water flea` | D | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | trailing_letter |
| 15 | `2` | two. | `(needs gpt-4 judge)` | `two.` | ❌ | ❌ | raw |
| 16 | `5` | A | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 17 | `Does Linda's snowboard slide down a h...` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 18 | `400` | 600. | `(needs gpt-4 judge)` | `600` | ❌ | ❌ | raw |
| 19 | `No` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 20 | `2` | 2 | `(needs gpt-4 judge)` | `2` | ❌ | ✅ | raw |
| 21 | `0` | 0. | `(needs gpt-4 judge)` | `0` | ❌ | ✅ | raw |
| 22 | `equal to` | A | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 23 | `no` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 24 | `no` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 25 | `no` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 26 | `11` | 23. | `(needs gpt-4 judge)` | `23` | ❌ | ❌ | raw |
| 27 | `8.5` | C | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | trailing_letter |
| 28 | `4` | 10. | `(needs gpt-4 judge)` | `10` | ❌ | ❌ | raw |
| 29 | `70°` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 30 | `No` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `70°` | D | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | trailing_letter |


#### task: `scienceqa_img` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 1 | `B` | C | `C` | `C` | ❌ | ❌ | trailing_letter |
| 2 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 3 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 4 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 5 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 6 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 7 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 8 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 9 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 10 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 11 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 12 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 13 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 14 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 15 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 16 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 17 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 18 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 19 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 20 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 21 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 22 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 23 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 24 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 25 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 26 | `A` | B | `B` | `B` | ❌ | ❌ | trailing_letter |
| 27 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 28 | `A` | B | `B` | `B` | ❌ | ❌ | trailing_letter |
| 29 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 30 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 31 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |


#### task: `VisualPuzzles_cot` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `D` | The answer is C | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | the_answer_is |
| 1 | `B` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ✅ | the_answer_is |
| 2 | `C` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |
| 3 | `B` | The answer is C. | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | the_answer_is |
| 4 | `C` | The answer is D | `(parse_response on <answer>…)` | `D` | ❌ | ❌ | the_answer_is |
| 5 | `C` | The Answer is A | `(parse_response on <answer>…)` | `A` | ❌ | ❌ | the_answer_is |
| 6 | `D` | The answer is D. | `(parse_response on <answer>…)` | `D` | ❌ | ✅ | the_answer_is |
| 7 | `A` | The answer is C. | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | the_answer_is |
| 8 | `C` | To answer: A | `(parse_response on <answer>…)` | `A` | ❌ | ❌ | the_answer_is |
| 9 | `A` | The Answer is A | `(parse_response on <answer>…)` | `A` | ❌ | ✅ | the_answer_is |
| 10 | `D` | The answer is D | `(parse_response on <answer>…)` | `D` | ❌ | ✅ | the_answer_is |
| 11 | `B` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ✅ | the_answer_is |
| 12 | `A` | The Answer is A | `(parse_response on <answer>…)` | `A` | ❌ | ✅ | the_answer_is |
| 13 | `C` | The Answer is A | `(parse_response on <answer>…)` | `A` | ❌ | ❌ | the_answer_is |
| 14 | `D` | The answer is C | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | the_answer_is |
| 15 | `D` | The answer is C | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | the_answer_is |
| 16 | `D` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |
| 17 | `A` | The answer is C. | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | the_answer_is |
| 18 | `A` | The answer is C | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | the_answer_is |
| 19 | `B` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ✅ | the_answer_is |
| 20 | `A` | The answer is D. | `(parse_response on <answer>…)` | `D` | ❌ | ❌ | the_answer_is |
| 21 | `D` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |
| 22 | `D` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |
| 23 | `B` | The answer is A | `(parse_response on <answer>…)` | `A` | ❌ | ❌ | the_answer_is |
| 24 | `D` | The answer is C | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | the_answer_is |
| 25 | `A` | The Answer is A | `(parse_response on <answer>…)` | `A` | ❌ | ✅ | the_answer_is |
| 26 | `B` | The answer is A | `(parse_response on <answer>…)` | `A` | ❌ | ❌ | the_answer_is |
| 27 | `B` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ✅ | the_answer_is |
| 28 | `C` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |
| 29 | `A` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |
| 30 | `A` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |
| 31 | `D` | The answer is B | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |


### ckpt: `sft-zebracot`


#### task: `chartqa` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `14` | 11. | `` | `11` | ❌ | ❌ | raw |
| 1 | `0.57` | To determine the difference in value attributed to Lamb and Corn, I need to carefully examine the values presented in the chart. ⏎  ⏎ Fir... | `` | `103.91` | ❌ | ❌ | raw |
| 2 | `3` | 3 | `` | `3` | ❌ | ✅ | raw |
| 3 | `No` | No | `` | `No` | ❌ | ✅ | raw |
| 4 | `23` | 23 | `` | `23` | ❌ | ✅ | raw |
| 5 | `6` | 0.06 | `` | `0.06` | ❌ | ❌ | raw |
| 6 | `62` | 62 | `` | `62` | ❌ | ✅ | raw |
| 7 | `Yes` | No. | `` | `No.` | ❌ | ❌ | raw |
| 8 | `Inspired` | Depressed. | `` | `Depressed.` | ❌ | ❌ | raw |
| 9 | `0.03` | 0.03 | `` | `0.03` | ❌ | ✅ | raw |
| 10 | `1` | 1 is the value. <answer>1 </answer> | `1` | `1` | ✅ | ✅ | answer_tag |
| 11 | `0.08` | 0.02 | `` | `0.02` | ❌ | ❌ | raw |
| 12 | `17` | 17 | `` | `17` | ❌ | ✅ | raw |
| 13 | `21.6` | 19.33 | `` | `19.33` | ❌ | ❌ | raw |
| 14 | `2` | 2 | `` | `2` | ❌ | ✅ | raw |
| 15 | `2` | 2 | `` | `2` | ❌ | ✅ | raw |
| 16 | `2014` | 2014 | `` | `2014` | ❌ | ✅ | raw |
| 17 | `2011` | 2014 | `` | `2014` | ❌ | ✅ | raw |
| 18 | `1` | The answer is 5 colors. | `` | `5` | ❌ | ❌ | the_answer_is |
| 19 | `No` | Yes | `` | `Yes` | ❌ | ❌ | raw |
| 20 | `green line` | dark line | `` | `dark line` | ❌ | ❌ | raw |
| 21 | `2008` | 2006 | `` | `2006` | ❌ | ✅ | raw |
| 22 | `29` | 29 | `` | `29` | ❌ | ✅ | raw |
| 23 | `1.216666667` | 0.7036 | `` | `0.7036` | ❌ | ❌ | raw |
| 24 | `Yes` | Yes. | `` | `Yes.` | ❌ | ❌ | raw |
| 25 | `Yes` | No. | `` | `No.` | ❌ | ❌ | raw |
| 26 | `80` | 80 | `` | `80` | ❌ | ✅ | raw |
| 27 | `77` | 57 | `` | `57` | ❌ | ❌ | raw |
| 28 | `0.6` | 20 | `` | `20` | ❌ | ❌ | raw |
| 29 | `61` | 60 | `` | `60` | ❌ | ✅ | raw |
| 30 | `13` | 15. | `` | `15` | ❌ | ❌ | raw |
| 31 | `5` | 9 | `` | `9` | ❌ | ❌ | raw |


#### task: `cv_bench` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `(C)` | To determine the number of arens in the image, we need to carefully examine the visual elements present. The image shows a close-up view ... | `` | `A` | ❌ | ❌ | raw |
| 1 | `(E)` | To determine the number of cushions in the image, let's carefully observe the bed and the items on it. ⏎  ⏎ First, count the the pillows ... | `` | `B` | ❌ | ❌ | the_answer_is |
| 2 | `(C)` | To determine the number of table lamps in the image, let's carefully examine the scene. ⏎  ⏎ First, identify the objects present in the i... | `` | `C` | ❌ | ✅ | the_answer_is |
| 3 | `(D)` | To determine the number of curtains in the image, let's carefully observe the scene: ⏎  ⏎ 1. Look at the window on the right side of the ... | `` | `D` | ❌ | ✅ | the_answer_is |
| 4 | `(D)` | To determine the number of pictures in the image, let's carefully examine the visual details. ⏎  ⏎ First, observe the the wall above the ... | `` | `D` | ❌ | ✅ | the_answer_is |
| 5 | `(B)` | To determine the number of beds in the image, let's carefully observe the visual details: ⏎  ⏎ - On the left side of the image, there is ... | `` | `A` | ❌ | ❌ | the_answer_is |
| 6 | `(A)` | To determine the number of walls in the image, let's carefully observe the visual details: ⏎  ⏎ - The image shows a room with three windo... | `` | `A` | ❌ | ✅ | the_answer_is |
| 7 | `(B)` | To determine the number of televisions in the image, let's carefully examine the scene. ⏎  ⏎ First, look at the area of the the room wher... | `` | `` | ❌ | ❌ | the_answer_is |
| 8 | `(B)` | To determine the number of chests in the image, let's carefully examine the objects present: ⏎  ⏎ First, observe the nightstand on the le... | `` | `B` | ❌ | ✅ | the_answer_is |
| 9 | `(E)` | To determine the number of walls in the image, let's carefully observe the classroom: ⏎  ⏎ 1. The classroom has a long blue with a border... | `` | `C` | ❌ | ❌ | the_answer_is |
| 10 | `(D)` | To determine the number of windows in the image, let's carefully examine the scene. ⏎  ⏎  ⏎ First, look at the right side of the image. T... | `` | `D` | ❌ | ✅ | the_answer_is |
| 11 | `(D)` | To determine the number of hats in the image, let's carefully examine the scene. ⏎  ⏎ - First, look at the left side of the image. There ... | `` | `B` | ❌ | ❌ | the_answer_is |
| 12 | `(B)` | To determine the number of curtains in the image, let's carefully observe the scene. ⏎  ⏎ First, look at the left side of the image. Ther... | `` | `B` | ❌ | ✅ | the_answer_is |
| 13 | `(B)` | To determine the number of pendant lamps in the image, let's carefully observe the scene: ⏎  ⏎ 1. **Identify the pendant lamps**: Look la... | `` | `` | ❌ | ❌ | boxed |
| 14 | `(E)` | To determine the number of windows in the image, let's carefully observe the visual details: ⏎  ⏎ 1. Look at the upper part of the image,... | `` | `` | ❌ | ❌ | raw |
| 15 | `(B)` | To determine the number of candle holders in the image, let's carefully examine the scene: ⏎  ⏎ 1. **Identify the candle holders:** Look ... | `` | `B` | ❌ | ✅ | boxed |
| 16 | `(D)` | To determine the number of glasses in the image, let's carefully examine the scene: ⏎  ⏎ 1. Look at the table: In the center of the image... | `` | `A` | ❌ | ❌ | the_answer_is |
| 17 | `(A)` | To determine how many pictures are in the image, let's carefully examine the visual content. The image shows a room with a wooden table a... | `` | `A` | ❌ | ✅ | the_answer_is |
| 18 | `(E)` | To determine the number of armchairs in the image, let's carefully examine the scene.  ⏎  ⏎ First, identify the types of chairs present. ... | `` | `A` | ❌ | ❌ | raw |
| 19 | `(A)` | To determine the number of light switches in the image, let's carefully examine the scene. ⏎  ⏎ First, look at the cabinets and the wall.... | `` | `B` | ❌ | ❌ | the_answer_is |
| 20 | `(A)` | To determine the number of buckets in the image, let's carefully examine the scene. ⏎  ⏎ First, look for the typical that of a bucket: a ... | `` | `A` | ❌ | ✅ | the_answer_is |
| 21 | `(B)` | To determine the number of blinds in the image, let's carefully observe the scene. ⏎  ⏎  ⏎ On the left side, near the window, the blinds ... | `` | `C` | ❌ | ❌ | the_answer_is |
| 22 | `(A)` | To determine the number of walls in the image, let's carefully observe the visual details: ⏎  ⏎ 1. The image shows a room with a bed, nig... | `` | `B` | ❌ | ❌ | the_answer_is |
| 23 | `(C)` | To determine the number of curtains in the image, let's carefully observe the visual details. ⏎  ⏎ First, look at the window area the mos... | `` | `C` | ❌ | ✅ | the_answer_is |
| 24 | `(B)` | To determine the number of pillows in the image, let's carefully observe the bed on the right side of the image. ⏎  ⏎ First, identify the... | `` | `D` | ❌ | ❌ | the_answer_is |
| 25 | `(D)` | To determine the number of folders in the image, let's carefully observe the objects on the table. ⏎  ⏎ First, the table is a dark, dark ... | `D` | `D` | ✅ | ✅ | answer_tag |
| 26 | `(A)` | To determine the number of walls in the image, let's carefully observe the scene: ⏎  ⏎ 1. The image shows a kitchen area with a counterto... | `` | `A` | ❌ | ✅ | the_answer_is |
| 27 | `(B)` | To determine the number of light switches in the image, let's carefully examine the scene.  ⏎  ⏎ First, look at the walls and the the fro... | `` | `B` | ❌ | ✅ | the_answer_is |
| 28 | `(D)` | To determine the number of blinds in the image, let's carefully observe the scene. ⏎  ⏎ First, look at the window in the image. The windo... | `` | `D` | ❌ | ✅ | boxed |
| 29 | `(C)` | To determine the number of ranges in the image, let's carefully observe the kitchen layout. ⏎  ⏎ First, identify the main components of a... | `` | `C` | ❌ | ✅ | the_answer_is |
| 30 | `(A)` | To determine the number of dishwashers in the image, let's carefully examine the kitchen scene. ⏎  ⏎ First, look the right side of the ki... | `` | `C` | ❌ | ❌ | the_answer_is |
| 31 | `(B)` | To determine the number of windows in the image, let's carefully examine the kitchen scene. ⏎  ⏎ First, look at the right side of the ima... | `` | `A` | ❌ | ❌ | raw |


#### task: `mmstar` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 1 | `B` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 2 | `D` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 3 | `C` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 4 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 5 | `D` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 6 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 7 | `D` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 8 | `B` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 9 | `D` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 10 | `D` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 11 | `D` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 12 | `A` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 13 | `B` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 14 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 15 | `C` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 16 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 17 | `C` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 18 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 19 | `B` | B | `(first-char check)` | `B` | ✅ | ✅ | trailing_letter |
| 20 | `B` | B | `(first-char check)` | `B` | ✅ | ✅ | trailing_letter |
| 21 | `B` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 22 | `C` | C | `(first-char check)` | `C` | ✅ | ✅ | trailing_letter |
| 23 | `A` | D | `(first-char check)` | `D` | ❌ | ❌ | trailing_letter |
| 24 | `C` | B | `(first-char check)` | `B` | ❌ | ❌ | trailing_letter |
| 25 | `B` | B | `(first-char check)` | `B` | ✅ | ✅ | trailing_letter |
| 26 | `B` | A | `(first-char check)` | `A` | ❌ | ❌ | trailing_letter |
| 27 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 28 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 29 | `A` | C | `(first-char check)` | `C` | ❌ | ❌ | trailing_letter |
| 30 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |
| 31 | `A` | A | `(first-char check)` | `A` | ✅ | ✅ | trailing_letter |


#### task: `mmvet` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `-1<AND>-5` | The problem asks for the value of x in the equation (x + 3)2 = 4. ⏎  ⏎ To, solve this, first, the left of the equation is a perfect squar... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 1 | `1.25<OR>=1.25<OR>5/4` | The problem asks for the value of 'd' in the second equation. ⏎  ⏎ Let's isolate the term with 'd' on one side of the equation. ⏎  ⏎ Subt... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 2 | `0.75<OR>=0.75<OR>3/4` | The problem asks for the value of x in the equation 8x + 1 = 4x + 14. ⏎  ⏎ To isolate the, first subtract 4x from both sides. ⏎  ⏎ This s... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 3 | `12<OR>=12` | The problem asks for the result of the calculation. ⏎  <answer> B </answer> | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 4 | `7<OR>=7` | The problem asks for the answer to the second equation on the right side of the image. ⏎  ⏎ The answer is 11 | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 5 | `2<OR>=2` | The problem asks for the answer to the second last equation on the right. Scanning the grid, the second last equation on the right is 34 ... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 6 | `5<OR>=5` | The problem asks for the value of y in the last row. ⏎  ⏎ The answer is 5. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 7 | `13<OR>=13` | The problem requires a pattern of the way the "X" operator is defineddefined. the pattern is be found in the previous images. ⏎  ⏎ The fi... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 8 | `14<OR>=14` | The problem asks for the answer that the girl on the right will write on the board. ⏎  ⏎ Looking at the image, the girl on the right is f... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 9 | `13.6<OR>13.7` | To determine how many gallons of supreme gasoline you can get for $50, we first need to identify the price per gallon of supreme gasoline... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 10 | `0.11` | To determine the price difference between diesel and regular gasoline, we need to carefully examine the prices displayed on the gas pump.... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 11 | `249.98` | To determine the total price for a bottle of Merlot and a bottle of Cabernet, we need to carefully examine the prices listed on the bottl... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 12 | `344.97` | To determine the cost of three bottles of Moet champagne, we need to carefully examine the price labels on the bottles in the image. ⏎  ⏎... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 13 | `14.8<OR>14,8` | To problem asks for the total cost of one one funghi pizza and one prosciutto pizza. ⏎  ⏎ From the menu the priceghi pizza is 14.00. ⏎  ⏎... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 14 | `25.8<OR>25,8` | To determine the total cost, we need to identify the prices of the items listed in the menu. ⏎  ⏎ First,herita pizza is priced at 5.00 ea... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 15 | `14` | To determine the original price for pork belly, we need to analyze the information provided in the image. ⏎  ⏎ The image shows a sale sig... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 16 | `4.299<OR>4.3` | To determine the gasoline price, we need to carefully examine the information presented in the image. The image shows a digital display w... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 17 | `blue` | To determine the color of the hat worn by the person in the front left, we need to observe the image. | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 18 | `conditioner` | To answer the question, to the right of the shampoo, I need to identify the arrangement of the products in the image. ⏎  ⏎ First, I locat... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 19 | `mousse` | To determine what is in the rightmost bottle, let's carefully examine the image. There are three distinct bottles, each labeled with diff... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 20 | `above the man<OR>top left` | To determine the spatial relationship between the frisbee and the man, let's analyze the image: ⏎  ⏎ - The frisbee is visible in the air,... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 21 | `no<OR>empty` | To determine which car is in parking spot 33, we need to carefully examine the image and identify the relevant details. ⏎  ⏎ First, let's... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | the_answer_is |
| 22 | `31` | To determine which parking spot the red car is parked in, we need to carefully observe the image and identify the relevant details. ⏎  ⏎ ... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 23 | `eight<OR>8.0` | To determine the price for tomatoes, the first step is to locate the tomatoes in the image. The tomatoes are visible in the lower left se... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 24 | `orange` | To determine what fruit is to the right of the pl, let's first locate the plums in the image. The plums are the dark purple fruits situat... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |
| 25 | `35%` | To determine the total international awareness of Tina Fey, I need to locate the relevant data in the image. The image presents a table w... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 26 | `Australia` | To determine the country where Tina Fey has the highest level of total awareness, I need to examine the "Total" column in the table for e... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | answer_tag |
| 27 | `mantle` | The problem asks for the part of the Earth's interior indicated by the number 2 in the provided image. ⏎  ⏎ Looking at the image, the num... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | raw |
| 28 | `16` | To determine the number displayed on the motorcycle of  the motorcyclist wearing a yellow suit, we need to carefully observe the image. ⏎... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 29 | `4` | To determine the number displayed on the motorcycle on the right right, we need to closely observe the image. The motorcycle on the far r... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | raw |
| 30 | `same` | To determine which jet is larger, size comparison must be made. ⏎  ⏎ Observing the image, the first two jets are visible in the foregroun... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ❌ | answer_tag |
| 31 | `no` | To determine whether the person is bigger than the car, let's analyze the image and compare ⏎ the sizes of the person and the car. ⏎  ⏎ F... | `(needs gpt-4 judge)` | `(heuristic substring)` | ❌ | ✅ | the_answer_is |


#### task: `mathverse_testmini_vision_dominant` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `D` | The problem shows a triangle ABC with a line segment DE parallel to the base BC. We need to find the measure of angle CED. ⏎  ⏎ Since DE ... | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | answer_tag |
| 1 | `C` | The problem asks for the measure of angle 2. the diagram shows several lines and angles. with the given information, we can use the prope... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |
| 2 | `A` | The problem requires finding the measure of angle CDB. the given conditions are that BD bisects angle ABC, CD is parallel to AB, and the ... | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | answer_tag |
| 3 | `D` | The problem shows a circle with a tangent line AB at point B and a line segment OC inside the circle. The need to find the measure of ang... | `(needs gpt-4 judge)` | `D` | ❌ | ✅ | the_answer_is |
| 4 | `A` | The problem shows that the two lines 'a' and 'b' are parallel, and there is a transversal line intersecting them. We need to find the mea... | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | answer_tag |
| 5 | `C` | The problem shows that the lines AB and CD are parallel, and there is a transversal line intersecting them. We need to find the measure o... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |
| 6 | `B` | The problem requires finding the measure of angle EAC. the given shows that triangles and D are are congruent. ⏎  ⏎ From the congruence o... | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | answer_tag |
| 7 | `C` | The problem gives us the diagram of triangle ABC and triangle DEF, which are stated to be congruent. We need to find the length of EC. ⏎ ... | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | answer_tag |
| 8 | `A` | The problem requires finding the perimeter of triangle DBE. the given shows a triangle with specific lengths and relationships. ⏎  ⏎ Firs... | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | answer_tag |
| 9 | `A` | The problem presents a triangle ABC with specific conditions. We need to find the measure of angle BDC. ⏎  ⏎ Since AB is the to the left ... | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | answer_tag |
| 10 | `A` | The problem asks for the measure of angle AFC. We need to analyze the given diagram and apply the properties of angles. ⏎  ⏎ First the di... | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | answer_tag |
| 11 | `A` | The problem provides a the geometric diagram with two lines and several angles. We need to find the measure of angle 2. ⏎  ⏎ First, obser... | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | answer_tag |
| 12 | `C` | The problem shows a the diagram with a set of lines and angles. We need to find the measure of angle ACD. ⏎  ⏎ From the diagram, we obser... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |
| 13 | `B` | The problem asks for the measure of angle ACB in the given circle. We need to understand the relationship between the circle's properties... | `(needs gpt-4 judge)` | `B` | ❌ | ✅ | answer_tag |
| 14 | `D` | The problem asks for the rotation of line line b around point A so that it becomes parallel to line a. ⏎  ⏎ To determine the rotation,, f... | `(needs gpt-4 judge)` | `D` | ❌ | ✅ | answer_tag |
| 15 | `A` | The problem shows a circle with a central angle and a chord. The goal is to find the measure of angle OAB. ⏎  ⏎ First, observe the geomet... | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | answer_tag |
| 16 | `C` | The problem presents a the intersection of a parallelogram's diagonals and asks us to find the measure of one of. ⏎  ⏎ Using the properti... | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | answer_tag |
| 17 | `C` | The problem asks for the measure of angle 4. the image shows two lines with angles a and b intersecting. ⏎  ⏎ To find the measure of angl... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |
| 18 | `C` | The problem shows two the parallel lines, a and b, and a transversal line intersecting. We need to find the measure of angle 3. ⏎  ⏎ Sinc... | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | answer_tag |
| 19 | `A` | The problem shows a circle with a central angle and an inscribed angle. We need to find the measure of the central angle. ⏎  ⏎ The centra... | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | answer_tag |
| 20 | `D` | The problem asks for the measure of angle 4. the image shows a set of intersecting and parallel lines. ⏎  ⏎ First, identify the relations... | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | answer_tag |
| 21 | `C` | The problem involves a rhombus with diagonals, and we need to find the measure of angle OBC. ⏎  ⏎ In rhombus ABCD, the diagonals bisect e... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |
| 22 | `C` | The problem shows a circle with two tangents PA and PB from from an external point P. The circle at point A is angle BOC and the circle a... | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | answer_tag |
| 23 | `B` | The problem shows two parallel lines, a and b, with a transversal c intersecting them. We need to find the measure of angle 2. ⏎  ⏎ Since... | `(needs gpt-4 judge)` | `B` | ❌ | ✅ | answer_tag |
| 24 | `D` | The problem shows a triangle with a angle bisector and a line. We need to find the measure of angle C. ⏎  ⏎ Since EF is parallel to BC, a... | `(needs gpt-4 judge)` | `D` | ❌ | ✅ | trailing_letter |
| 25 | `C` | The problem requires finding the height of a flagpole. the image shows a flagpole with a shadow and a given ratio. ⏎  ⏎ To find the heigh... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |
| 26 | `A` | The problem asks for the slope distance between two adjacent trees. We need to find the horizontal distance and the vertical distance bet... | `(needs gpt-4 judge)` | `A` | ❌ | ✅ | answer_tag |
| 27 | `B` | The problem requires finding the size of angle CAB when the length of CD is the largest. ⏎ To maximize the length of CD, we need to consi... | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | answer_tag |
| 28 | `C` | The problem asks for the measure of angle D, which is the intersection of the angle bisectors BD and CD in triangle ABC. ⏎  ⏎ First, we t... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |
| 29 | `C` | The problem gives us the lengths of the sides OA, OB, and OC, and asks for the measure of angle AOB. ⏎  ⏎ Since OA = OB = OC, triangle OB... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |
| 30 | `B` | The problem shows a the diagram with lines a and b, which are parallel, and a line line segment. There are also angles marked. ⏎  ⏎ To fi... | `(needs gpt-4 judge)` | `B` | ❌ | ✅ | answer_tag |
| 31 | `C` | The problem shows a the incircle of triangle ABC with center I. The center I is the intersection of the angle bisectors of the triangle. ... | `(needs gpt-4 judge)` | `C` | ❌ | ✅ | answer_tag |


#### task: `mathvista_testmini_format` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `1.2` | 0.0126 | `(needs gpt-4 judge)` | `0.0126` | ❌ | ❌ | raw |
| 1 | `1000` | To determine the total volume of the measuring cup, we need to carefully observe the markings on the cup. The cup cup has measurement mar... | `(needs gpt-4 judge)` | `250` | ❌ | ❌ | the_answer_is |
| 2 | `145°` | The problem asks for the measure of ∠BOC, given that ∠A is 110° and OB and OC are the angle bisectors of ∠ABC and ∠ACB respectively. ⏎  ⏎... | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | the_answer_is |
| 3 | `4` | The initial subtraction problem asks us to identify the number of objects remaining after removing all large rubber spheres and all big s... | `(needs gpt-4 judge)` | `3` | ❌ | ❌ | answer_tag |
| 4 | `97` | The problem asks for the measure of angle H in the given pentagon. The provided are the for the angles and the shape of the pentagon. ⏎  ... | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | answer_tag |
| 5 | `6cm` | The problem presents a the length of a line segment based on the given conditions. We need to analyze the relationship between the points... | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | answer_tag |
| 6 | `quarter` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 7 | `Yes` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 8 | `8` | The problem asks for the area of the largest square in the tree, which the letter C represents. ⏎  ⏎ We need to find the side lengths of ... | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | answer_tag |
| 9 | `E` | C | `(needs gpt-4 judge)` | `C` | ❌ | ❌ | trailing_letter |
| 10 | `5` | 6ed the problem. | `(needs gpt-4 judge)` | `6` | ❌ | ❌ | raw |
| 11 | `Yes` | A. | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 12 | `0` | The problem asks for many objects. | `(needs gpt-4 judge)` | `The problem asks for many obje` | ❌ | ❌ | raw |
| 13 | `Yes` | A. | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 14 | `Common water flea` | D | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | trailing_letter |
| 15 | `2` | The problem asks us to determine how many items sold less than 5 units in at least one store. We need to examine the sales for each item ... | `(needs gpt-4 judge)` | `3` | ❌ | ❌ | answer_tag |
| 16 | `5` | The problem presents a right triangle ABC with ∠ACB = 90°. Point D is the midpoint of theenuse AB, and the need to find the length of CD.... | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | answer_tag |
| 17 | `Does Linda's snowboard slide down a h...` | B | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 18 | `400` | To determine the highest amount this class can measure, we need to carefully examine the markings on the vessel. The ⏎ image shows the ve... | `(needs gpt-4 judge)` | `400` | ❌ | ✅ | the_answer_is |
| 19 | `No` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 20 | `2` | 2 | `(needs gpt-4 judge)` | `2` | ❌ | ✅ | raw |
| 21 | `0` | 0. | `(needs gpt-4 judge)` | `0` | ❌ | ✅ | raw |
| 22 | `equal to` | The problem asks for the comparison of the derivative of the function at two points, x=2 and x=5. ⏎ First, I will find the slope of the f... | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 23 | `no` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 24 | `no` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 25 | `no` | A. | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | trailing_letter |
| 26 | `11` | To determine the age gap between the two individuals in the image, we start by observing their physical appearances. The man on the left ... | `(needs gpt-4 judge)` | `70` | ❌ | ❌ | raw |
| 27 | `8.5` | The problem shows a circle with a diameter and a triangle. We need to find the radius of the circle. ⏎  ⏎ The the problem shows the trian... | `(needs gpt-4 judge)` | `A` | ❌ | ❌ | answer_tag |
| 28 | `4` | 20. | `(needs gpt-4 judge)` | `20` | ❌ | ❌ | raw |
| 29 | `70°` | The problem shows two intersecting chords in a circle. We need to find the measure of angle C. ⏎  ⏎ First the image shows the angles form... | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | answer_tag |
| 30 | `No` | B. | `(needs gpt-4 judge)` | `B` | ❌ | ❌ | trailing_letter |
| 31 | `70°` | The problem presents a the calculation of a angle in a geometric figure. The key details are the equality of the sides and the given angl... | `(needs gpt-4 judge)` | `D` | ❌ | ❌ | the_answer_is |


#### task: `scienceqa_img` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 1 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 2 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 3 | `B` | C | `C` | `C` | ❌ | ❌ | trailing_letter |
| 4 | `C` | A | `A` | `A` | ❌ | ❌ | trailing_letter |
| 5 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 6 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 7 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 8 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 9 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 10 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 11 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 12 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 13 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 14 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 15 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 16 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 17 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 18 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 19 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 20 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 21 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 22 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 23 | `A` | A. | `A.` | `A` | ✅ | ✅ | trailing_letter |
| 24 | `D` | D | `D` | `D` | ✅ | ✅ | trailing_letter |
| 25 | `C` | C | `C` | `C` | ✅ | ✅ | trailing_letter |
| 26 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 27 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 28 | `A` | B | `B` | `B` | ❌ | ❌ | trailing_letter |
| 29 | `B` | B | `B` | `B` | ✅ | ✅ | trailing_letter |
| 30 | `A` | A | `A` | `A` | ✅ | ✅ | trailing_letter |
| 31 | `C` | The answer is C | `The answer is C` | `C` | ❌ | ✅ | the_answer_is |


#### task: `VisualPuzzles_cot` (showing 32 of first 32)

| # | target | model response | default pred | robust pred | default ✓ | robust ✓ | pattern |
|--:|---|---|---|---|:--:|:--:|---|
| 0 | `D` | The problem asks for the correct grouping of the six figures based on a visual pattern. ⏎ To determine the correct grouping, analyze the ... | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | answer_tag |
| 1 | `B` | The problem asks for the correct option to fill the gap mark in the image. ⏎  ⏎  ⏎  Therefore the answer is C. <answer> C </answer> | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 2 | `C` | The problem asks for the maximum number of pieces that Ronnie can be visited, starting from the green square and ending at the red square... | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | the_answer_is |
| 3 | `B` | The problem asks us to find the shape that cannot be formed using the given building blocks. We need to examine the shapes and consider t... | `(parse_response on <answer>…)` | `D` | ❌ | ❌ | answer_tag |
| 4 | `C` | The problem asks for a word cipher using a transfair cipher with on a board.. ⏎  ⏎ The problem image shows the transfair cipher board. We... | `(parse_response on <answer>…)` | `C` | ✅ | ✅ | answer_tag |
| 5 | `C` | The problem asks for the image that best completes the analogy. I need to identify the relationship between the original image and the pr... | `(parse_response on <answer>…)` | `C` | ✅ | ✅ | answer_tag |
| 6 | `D` | The problem asks us to find the pattern that by placing two transparent patterned sheets on top of each other. We need to visualize the s... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 7 | `A` | The problem asks to find the correct that can be made from the given cube shape. ⏎  ⏎ Option C is the correct shape. <answer> A </answer> | `(parse_response on <answer>…)` | `A` | ✅ | ✅ | answer_tag |
| 8 | `C` | The problem asks for the kite with the longest string among the options provided. I need to examine the image to identify the kites and t... | `(parse_response on <answer>…)` | `` | ❌ | ❌ | answer_tag |
| 9 | `A` | The problem asks for the image that best completes the analogy. I need to identify the relationship between the original image and the pr... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 10 | `D` | The problem asks us to determine the number of white blocks that are not visible in the provided image. ⏎  need to count the total number... | `(parse_response on <answer>…)` | `` | ❌ | ❌ | the_answer_is |
| 11 | `B` | The problem asks for the arrangement of the chairs in the image. I need to identify the chairs and their positions relative to each other... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 12 | `A` | The problem asks for the image that best completes the analogy. I need to identify the relationship between the original image and the pr... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 13 | `C` | The problem asks for the image that best completes the analogy. I need to identify the relationship between the original image and the pr... | `(parse_response on <answer>…)` | `C` | ✅ | ✅ | answer_tag |
| 14 | `D` | The problem asks for the correct grouping of the six figures based on a visual pattern. ⏎ To determine the correct grouping, analyze the ... | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | answer_tag |
| 15 | `D` | The problem asks for the correct grouping of the six figures based on a visual pattern. ⏎ To determine the correct grouping, analyze the ... | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | answer_tag |
| 16 | `D` | The problem asks for the arrangement of the chairs in the image. I need to identify the chairs and their positions relative to each other... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 17 | `A` | The problem asks to find the correct cube formed by arranging the 9 identical building blocks. ⏎  ⏎ To determine how the blocks fit toget... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 18 | `A` | To problem,, we first identify the the distances between the locations mentioned: ⏎ - The's a 4-km from Rebecca's place to Kathryn's plac... | `(parse_response on <answer>…)` | `D` | ❌ | ❌ | boxed |
| 19 | `B` | The problem asks us to identify the configuration of shapes in the image that represents the "Kangaroo constellation." We need to find th... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 20 | `A` | The problem asks for the piece that does not fit the the jigsaw in the second image. ⏎ To determine the correct piece, I need to see the ... | `(parse_response on <answer>…)` | `A` | ✅ | ✅ | answer_tag |
| 21 | `D` | The problem asks for the figure that represents a rotation of the object shown in the original image. ⏎  ⏎ Let's identify the object in t... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 22 | `D` | The problem asks for the arrangement of the chairs in the image. I need to identify the chairs and their positions relative to each other... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 23 | `B` | The problem requires filling a 3x3 grid with numbers such that the sum of the numbers in any two cells sharing an edge is constant. We ar... | `(parse_response on <answer>…)` | `A` | ❌ | ❌ | answer_tag |
| 24 | `D` | The problem asks for the correct grouping of the six figures based on a visual pattern. ⏎ To determine the correct grouping, analyze the ... | `(parse_response on <answer>…)` | `B` | ❌ | ❌ | answer_tag |
| 25 | `A` | The problem asks for the image that best completes the analogy. I need to identify the relationship between the original image and the pr... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 26 | `B` | The problem asks for a word represented by a sequence of letters. I need to decode the pattern shown in the image to find the word. ⏎  ⏎ ... | `(parse_response on <answer>…)` | `B` | ✅ | ✅ | answer_tag |
| 27 | `B` | The problem asks for the correct option to fill the gap mark in the image. ⏎  ⏎  ⏎  Therefore the answer is C. <answer> C </answer> | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 28 | `C` | The problem asks for the arrangement of the chairs in the image. I need to identify the chairs and their positions relative to each other... | `(parse_response on <answer>…)` | `C` | ✅ | ✅ | answer_tag |
| 29 | `A` | The problem asks for the arrangement of the chairs in the image. I need to identify the chairs and their positions relative to each other... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 30 | `A` | The problem asks for the arrangement of the chairs in the image. I need to identify the chairs and their positions relative to each other... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
| 31 | `D` | The problem asks for the image that completes the pattern shown in the grid. ⏎  ⏎ Observe the provided image closely. ⏎  <answer> C </ans... | `(parse_response on <answer>…)` | `C` | ❌ | ❌ | answer_tag |
