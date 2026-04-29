"""lmms_eval adapter for the Anole interleaved-generation model.

Mirrors the skeleton of ``lmms_eval/models/llava_llada.py``'s ``generate_until``
but swaps the LLaDA / LavidaO machinery for ``ChameleonInferenceModel`` from
the local ``anole/`` source tree (`anole/chameleon/inference/chameleon.py`).

Per-sample flow:
  1. Build an Anole ``prompt_ui`` list ``[image?, text, sentinel(<START-OF-IMAGE>)]``.
     The trailing sentinel forces the model to emit a 1024-token image segment
     first, then a text answer.
  2. ``ChameleonInferenceModel.generate(batch_prompt_ui=..., options=...)`` runs
     the whole batch as a single synchronized stream.
  3. We prepend a leading ``begin_image`` token (consumed at prompt time, absent
     from the output stream) so ``split_token_sequence`` can cleanly peel out
     image vs text segments.
  4. Image segments are decoded to PIL via ``model.decode_image`` and saved to
     ``gen_img_dir``; the path is registered on the doc as ``gen_img_path`` so
     downstream ``process_results`` callbacks can read it.
  5. Text segments are decoded and returned (concatenated).

Module-import side effects:
  - inserts ``<repo>/anole`` into ``sys.path``
  - changes the process working directory to ``<repo>/anole`` (the chameleon
    package has relative-import quirks that require CWD == anole/)

Build ``ChameleonInferenceModel`` once in ``__init__`` — it spawns subprocess
workers bound to the active CUDA_VISIBLE_DEVICES and is not thread-safe.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")

# Locate the anole/ source tree at repo_root/anole. lmms-eval lives at
# repo_root/lmms-eval/lmms_eval/models/anole.py — four parents up from this
# file is repo_root.
_ANOLE_DIR = Path(__file__).resolve().parents[3] / "anole"
if str(_ANOLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ANOLE_DIR))
os.chdir(_ANOLE_DIR)

from chameleon.inference.chameleon import ChameleonInferenceModel, Options  # noqa: E402
from interleaved_generation import split_token_sequence  # noqa: E402


@register_model("anole")
class Anole(lmms):
    """Anole interleaved-generation adapter.

    Args (via ``--model_args key=val,...``):
        pretrained: chameleon-native checkpoint dir containing
            ``models/7b/consolidated.pth`` and
            ``tokenizer/{text_tokenizer.json,vqgan.yaml,vqgan.ckpt}``.
        batch_size: outer chunk size handed to ``model.generate``.
        device: torch device string (single-GPU setup).
        temperature: text sampling temperature; ``0`` → greedy.
        cfg_image / cfg_text: optional CFG overrides for image generation.
        max_seq_len: cap for ``Options.max_seq_len`` (default 4096).
        gen_img_dir: directory to write generated image segments to. If unset,
            images are still decoded but not persisted.
    """

    def __init__(
        self,
        pretrained: str,
        batch_size: Union[int, str] = 1,
        device: str = "cuda:0",
        temperature: float = 1.0,
        cfg_image: Optional[float] = None,
        cfg_text: Optional[float] = None,
        max_seq_len: int = 4096,
        gen_img_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            eval_logger.warning(f"[anole] ignoring unexpected kwargs: {kwargs}")

        ckpt_root = Path(pretrained)
        model_dir = ckpt_root / "models" / "7b"
        text_tok = ckpt_root / "tokenizer" / "text_tokenizer.json"
        vqgan_cfg = ckpt_root / "tokenizer" / "vqgan.yaml"
        vqgan_ckpt = ckpt_root / "tokenizer" / "vqgan.ckpt"
        for p in (model_dir, text_tok, vqgan_cfg, vqgan_ckpt):
            if not p.exists():
                raise FileNotFoundError(
                    f"[anole] missing required asset under pretrained={pretrained!r}: {p}"
                )

        self.pretrained = pretrained
        self._device = torch.device(device)
        self.temperature = float(temperature)
        self.cfg_image = None if cfg_image is None else float(cfg_image)
        self.cfg_text = None if cfg_text is None else float(cfg_text)
        self.max_seq_len = int(max_seq_len)
        self.gen_img_dir = gen_img_dir
        self.batch_size_per_gpu = int(batch_size)
        self._rank = 0
        self._world_size = 1
        self.datetime_str = None  # set by evaluator after model creation

        eval_logger.info(
            f"[anole] loading ChameleonInferenceModel from {model_dir} "
            f"(batch_size={self.batch_size_per_gpu}, temperature={self.temperature}, "
            f"cfg_image={self.cfg_image}, cfg_text={self.cfg_text}, "
            f"max_seq_len={self.max_seq_len}, gen_img_dir={self.gen_img_dir})"
        )
        self._model = ChameleonInferenceModel(
            model_dir.as_posix(),
            text_tok.as_posix(),
            vqgan_cfg.as_posix(),
            vqgan_ckpt.as_posix(),
        )

    # ---------- properties expected by the lmms base ----------

    @property
    def config(self):
        return None  # no HF config; chameleon-native checkpoint

    @property
    def tokenizer(self):
        return self._model.token_manager.tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._model.vocab.eos_id

    @property
    def max_length(self):
        return self.max_seq_len

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    # ---------- light tokenizer helpers ----------

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        ids = self._model.token_manager.tokenize_text(string)
        if left_truncate_len:
            ids = ids[-left_truncate_len:]
        return ids

    def tok_decode(self, tokens) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._model.decode_text([list(tokens)])[0]

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    # ---------- abstract surface ----------

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # Tasks that need loglikelihood will fail loudly when this returns None;
        # Anole-style interleaved generation only supports generate_until.
        pass

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for Anole.")

    # ---------- main entry point ----------

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            batch_size = len(batched_contexts)

            batch_pil_images = [
                doc_to_visual(self.task_dict[task_name][split_name][doc_id])
                for doc_to_visual, task_name, split_name, doc_id in zip(
                    batched_doc_to_visual, batched_task, batched_split, batched_doc_id
                )
            ]  # List[List[PIL.Image]]

            # gen_kwargs sourced from the task yaml; temperature stays at the
            # script-level default (constructor arg).
            max_new_tokens = int(gen_kwargs.get("max_new_tokens", 256))

            # Anole "chat template" = prompt_ui list. tokens_from_ui accepts PIL
            # directly so we skip the file:/temp-PNG dance.
            batch_prompt_ui: List[List[dict]] = []
            for ctx, images in zip(batched_contexts, batch_pil_images):
                parts: List[dict] = []
                if images:
                    parts.append({"type": "image", "value": images[0]})
                parts.append({"type": "text", "value": ctx})
                parts.append({"type": "sentinel", "value": "<START-OF-IMAGE>"})
                batch_prompt_ui.append(parts)

            options = Options()
            options.max_seq_len = self.max_seq_len
            # Prompt forces a 1024-token image segment first via the <START-OF-IMAGE>
            # sentinel, so the task's max_new_tokens must be in addition to it.
            options.max_gen_len = max_new_tokens + 1024
            if self.temperature == 0:
                options.txt.greedy = True
            else:
                options.txt.temp = self.temperature
            if self.cfg_image is not None:
                options.img.cfg.guidance_scale_image = self.cfg_image
            if self.cfg_text is not None:
                options.img.cfg.guidance_scale_text = self.cfg_text

            tokens: torch.LongTensor = self._model.generate(
                batch_prompt_ui=batch_prompt_ui,
                options=options,
            )  # [B, gen_len]

            # Prompt's trailing <START-OF-IMAGE> was consumed; the returned
            # stream begins with 1024 image tokens but no leading boi. Prepend
            # one so split_token_sequence can detect the opening image segment.
            boi = self._model.vocab.begin_image
            eoi = self._model.vocab.end_image
            leading = torch.full(
                (tokens.shape[0], 1), boi, dtype=tokens.dtype, device=tokens.device
            )
            tokens = torch.cat([leading, tokens], dim=1)

            text_outputs: List[str] = []
            for b_idx in range(batch_size):
                segs = split_token_sequence(tokens[b_idx : b_idx + 1], boi, eoi)
                text_parts: List[str] = []
                img_save_path: Optional[str] = None
                for seg_id, (seg_type, seg_tokens) in enumerate(segs):
                    if seg_type == "image_seg" and seg_tokens.shape[1] == 1024:
                        pil = self._model.decode_image(seg_tokens)[0]
                        if self.gen_img_dir:
                            os.makedirs(self.gen_img_dir, exist_ok=True)
                            img_save_path = os.path.join(
                                self.gen_img_dir,
                                f"{batched_task[b_idx]}_{batched_doc_id[b_idx]}.png",
                            )
                            pil.save(img_save_path)
                    elif seg_type == "text_seg":
                        text_parts.append(self._model.decode_text(seg_tokens)[0])

                text_outputs.append("".join(text_parts).strip())

                if img_save_path is not None:
                    self.task_dict[batched_task[b_idx]][batched_split[b_idx]][
                        batched_doc_id[b_idx]
                    ]["gen_img_path"] = img_save_path

            res.extend(text_outputs)
            for b_ctx, b_output in zip(batched_contexts, text_outputs):
                self.cache_hook.add_partial("generate_until", (b_ctx, gen_kwargs), b_output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res
