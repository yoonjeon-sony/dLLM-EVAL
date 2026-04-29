"""lmms_eval adapter for MMaDA-Parallel-A interleaved generation.

Replaces the legacy M-variant adapter. Mirrors the skeleton of
``lmms_eval/models/anole.py``'s ``generate_until`` but swaps the chameleon
generator for ``generate_ti2ti`` from MMaDA-Parallel-A.

Per-sample flow (for each request in a chunk):
  1. Build A's text-image-to-text-image prompt:
       ``<system>{SYSTEM}</system><user>{ctx}</user>``      (conditional)
       ``<system>{SYSTEM}</system><user><uncondition></user>`` (uncond CFG)
  2. Center-crop the input PIL, encode it via the diffusers ``VQModel``
     (``encode_img_with_breaks``).
  3. Splice ``con_input_list`` = prompt_ids + image_ids + masked region
     ``[BOA, BOI, <masked image grid>, EOI, <masked text>, </answer>]``.
  4. Call ``generate_ti2ti`` (single-sample only — its image-gen branch
     hardcodes batch index ``[0]``) to fill in the masked image+text spans.
  5. ``decode_vq_to_image`` → save PNG to ``gen_img_dir``; concat any
     decoded text → return string.

Module-import side effects:
  - inserts ``<repo>/MMaDA-Parallel/MMaDA-Parallel-A`` into ``sys.path`` so
    A's ``model``, ``generators``, ``utils`` packages resolve.
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")

# Locate the MMaDA-Parallel-A source tree so its absolute imports resolve.
# IMPORTANT: the repo root has its own `model/` package (LavidaO/LLaDA from
# llava_llada). Python's `python -m lmms_eval` adds cwd to sys.path[0],
# which would shadow MMaDA-Parallel-A/model with the repo-root one. We drop
# the cwd entry and any stale `model.*` cached modules before importing.
_MMADA_A_DIR = Path(__file__).resolve().parents[3] / "MMaDA-Parallel" / "MMaDA-Parallel-A"
_REPO_ROOT = str(_MMADA_A_DIR.parents[1])  # /home/.../dLLM-EVAL
# Drop both the empty cwd entry and the absolute repo-root path that
# `python -m lmms_eval` injects — either would let `model/` (LavidaO) shadow
# MMaDA-Parallel-A's `model/` package.
sys.path = [p for p in sys.path if p not in ("", _REPO_ROOT)]
if str(_MMADA_A_DIR) not in sys.path:
    sys.path.insert(0, str(_MMADA_A_DIR))
for _stale in [k for k in list(sys.modules) if k == "model" or k.startswith("model.")]:
    del sys.modules[_stale]

from model import LLaDAForMultiModalGeneration  # noqa: E402
from generators.parallel_generator import generate_ti2ti  # noqa: E402
from utils.image_utils import (  # noqa: E402
    add_break_line,
    calculate_vq_params,
    decode_vq_to_image,
    encode_img_with_breaks,
    generate_crop_size_list,
    var_center_crop,
)
from utils.prompt_utils import generate_text_image_to_text_image_prompt  # noqa: E402
from utils.generation_utils import setup_seed  # noqa: E402


# Special-token IDs copied from MMaDA-Parallel-A/inference.py:22-31.
SPECIAL_TOKENS = {
    "mask_token": 126336,
    "newline_token": 126084,
    "image_token_offset": 126356,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,
    "eoi": 126350,
    "uncondition": 126351,
}

DEFAULT_SYSTEM_PROMPT = (
    "Generate an image applying the following editing instruction "
    "based on the original image."
)


@register_model("mmada")
class Mmada(lmms):
    """MMaDA-Parallel-A interleaved-generation adapter.

    Args (via ``--model_args key=val,...``):
        pretrained: HF dir (or local) for ``LLaDAForMultiModalGeneration``.
            Must include the model weights/config; tokenizer + vqvae can
            optionally be inherited from a base ckpt (see below).
        vae_ckpt: HF/local path with a ``vqvae`` subfolder for diffusers
            ``VQModel`` (typically ``tyfeld/MMaDA-Parallel-A``). Optional —
            falls back to ``pretrained`` when omitted, which is correct for
            the upstream base ckpt but NOT for fine-tunes that don't ship a
            ``vqvae/`` subfolder; pass the base ckpt explicitly there.
        tokenizer_path: HF/local path holding ``tokenizer.json`` /
            ``tokenizer_config.json``. Optional — falls back to
            ``pretrained`` when omitted. Useful for fine-tunes that drop the
            tokenizer files: point this at the base ckpt.
        batch_size: outer Collator chunk size. Inner generation loops one
            sample at a time (``generate_ti2ti`` is single-sample only).
        device: torch device string.
        temperature: sampling temperature for the IMAGE branch.
        text_temperature: sampling temperature for the TEXT branch
            (overridden by ``gen_kwargs.temperature`` from the task yaml).
        cfg_scale / cfg_img: classifier-free guidance scales for text/image.
        text_steps / text_block_length / timesteps: diffusion-step knobs.
        remasking: ``"low_confidence"`` or ``"random"``.
        output_height / output_width: generated image resolution.
        gen_img_dir: where to dump generated PNGs.
        system_prompt: prepended to every prompt; defaults to inference.py's
            image-editing prompt — override for Q&A tasks via ``--model_args``.
        seed: 0 → unseeded, otherwise calls ``setup_seed``.
    """

    def __init__(
        self,
        pretrained: str,
        vae_ckpt: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        batch_size: Union[int, str] = 1,
        device: str = "cuda:0",
        temperature: float = 1.0,
        text_temperature: float = 0.7,
        cfg_scale: float = 2.5,
        cfg_img: float = 4.0,
        text_steps: int = 256,
        text_block_length: int = 32,
        timesteps: int = 64,
        remasking: str = "low_confidence",
        output_height: int = 512,
        output_width: int = 512,
        gen_img_dir: Optional[str] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            eval_logger.warning(f"[mmada] ignoring unexpected kwargs: {kwargs}")

        self.pretrained = pretrained
        # Fine-tune ckpts often ship only weights+config (no tokenizer, no
        # vqvae/ subfolder). Fall back to pretrained when not overridden;
        # callers should pass an explicit base ckpt for fine-tunes.
        self.vae_ckpt = vae_ckpt or pretrained
        self.tokenizer_path = tokenizer_path or pretrained

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            self.accelerator = accelerator
        else:
            self._device = torch.device(device)
            self._rank = 0
            self._world_size = 1
            self.accelerator = None
        self.temperature = float(temperature)
        self.text_temperature = float(text_temperature)
        self.cfg_scale = float(cfg_scale)
        self.cfg_img = float(cfg_img)
        self.text_steps = int(text_steps)
        self.text_block_length = int(text_block_length)
        self.timesteps = int(timesteps)
        self.remasking = remasking
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.gen_img_dir = gen_img_dir
        self.system_prompt = system_prompt
        self.seed = int(seed)
        self.batch_size_per_gpu = int(batch_size)
        self.datetime_str = None

        if self.seed != 0:
            setup_seed(self.seed)

        eval_logger.info(
            f"[mmada] loading LLaDAForMultiModalGeneration from {pretrained}"
        )
        eval_logger.info(f"[mmada] loading tokenizer from {self.tokenizer_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True, padding_side="left"
        )
        self._model = LLaDAForMultiModalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            device_map=str(self._device),
        )
        self._model.eval()

        # diffusers is imported lazily so a misconfigured env produces the
        # informative ImportError only when --model mmada is actually invoked.
        from diffusers import VQModel  # noqa: E402

        eval_logger.info(f"[mmada] loading VQModel from {self.vae_ckpt} (subfolder=vqvae)")
        self._vqvae = VQModel.from_pretrained(self.vae_ckpt, subfolder="vqvae").to(self._device)
        self._vqvae.eval()
        self._vae_scale = 2 ** (len(self._vqvae.config.block_out_channels) - 1)
        eval_logger.info(
            f"[mmada] vae_scale={self._vae_scale}, output={self.output_height}x{self.output_width}, "
            f"batch_size={self.batch_size_per_gpu}, temperature={self.temperature}, "
            f"text_temperature={self.text_temperature}, cfg=(text {self.cfg_scale}, img {self.cfg_img}), "
            f"steps=(text {self.text_steps}, img {self.timesteps})"
        )

    # ---------- properties expected by the lmms base ----------

    @property
    def config(self):
        return self._model.config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self._model.config, "max_position_embeddings", 4096)

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
        ids = self._tokenizer(string)["input_ids"]
        if left_truncate_len:
            ids = ids[-left_truncate_len:]
        return ids

    def tok_decode(self, tokens) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return self._tokenizer.decode(tokens)

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
        # Diffusion-style A doesn't expose token logprobs.
        pass

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for mmada-A.")

    # ---------- per-sample preprocessing ----------

    def _build_sample_inputs(self, ctx: str, pil_image, text_gen_length: int) -> dict:
        """Build the (con_input, uncon_text, uncon_image, positions) tuple
        for a single sample using A's inference.py recipe."""
        # 1. prompt → conditional + unconditional strings
        input_prompt, uncon_text = generate_text_image_to_text_image_prompt(ctx, self.system_prompt)
        prompt_ids = self._tokenizer(input_prompt)["input_ids"]
        uncon_text_ids = self._tokenizer(uncon_text)["input_ids"]

        # 2. image encode via VAE (with break-line tokens between rows)
        img = pil_image.convert("RGB")
        crop_list = generate_crop_size_list((512 // 32) ** 2, 32)
        img = var_center_crop(img, crop_size_list=crop_list)
        input_img_token = encode_img_with_breaks(img, self._vqvae)

        # 3. assemble cond / uncond input sequences
        con_input_list = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
        uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]
        uncon_input_image = list(prompt_ids)

        # 4. masked region for the model to fill: [BOA, BOI, <img mask>, EOI, <text mask>, </answer>]
        seq_len, newline_every, gh, gw = calculate_vq_params(
            self.output_height, self.output_width, self._vae_scale
        )
        img_mask_token = add_break_line(
            [SPECIAL_TOKENS["mask_token"]] * seq_len,
            gh, gw,
            new_number=SPECIAL_TOKENS["newline_token"],
        )
        text_mask_tokens = [SPECIAL_TOKENS["mask_token"]] * text_gen_length
        end_token_ids = self._tokenizer("</answer>", add_special_tokens=False).input_ids
        pred_token = (
            [SPECIAL_TOKENS["answer_start"], SPECIAL_TOKENS["boi"]]
            + img_mask_token
            + [SPECIAL_TOKENS["eoi"]]
            + text_mask_tokens
            + end_token_ids
        )
        full_input_ids = con_input_list + pred_token

        # 5. position bookkeeping (mirrors inference.py:152-156)
        image_start = len(con_input_list) + 2  # skip [BOA, BOI]
        image_end = image_start + len(img_mask_token)
        text_start = image_end + 1  # skip EOI
        text_end = text_start + text_gen_length

        return {
            "con_input": torch.tensor(full_input_ids, device=self._device).unsqueeze(0),
            "uncon_text": torch.tensor(uncon_input_text, device=self._device).unsqueeze(0),
            "uncon_image": torch.tensor(uncon_input_image, device=self._device).unsqueeze(0),
            "text_start": text_start,
            "text_end": text_end,
            "image_start": image_start,
            "seq_len": seq_len,
            "newline_every": newline_every,
        }

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

            # Task-yaml gen_kwargs override script defaults where applicable.
            text_gen_length = int(gen_kwargs.get("max_new_tokens", 256))
            text_temperature = float(gen_kwargs.get("temperature", self.text_temperature))

            # Build all per-sample tensors first (cheap; no GPU forwards yet).
            per_sample_inputs = []
            for ctx, images in zip(batched_contexts, batch_pil_images):
                if not images:
                    raise ValueError(
                        "[mmada] generate_ti2ti requires an input image; got empty visual list."
                    )
                per_sample_inputs.append(
                    self._build_sample_inputs(ctx, images[0], text_gen_length)
                )

            # generate_ti2ti is single-sample only (image-gen branch hardcodes [0]).
            text_outputs: List[str] = []
            for b_idx, sd in enumerate(per_sample_inputs):
                output_tokens, generated_text = generate_ti2ti(
                    model=self._model,
                    input_ids=sd["con_input"],
                    text_start=sd["text_start"],
                    text_end=sd["text_end"],
                    image_start=sd["image_start"],
                    seq_len=sd["seq_len"],
                    newline_every=sd["newline_every"],
                    text_steps=self.text_steps,
                    text_gen_length=sd["text_end"] - sd["text_start"],
                    text_block_length=self.text_block_length,
                    timesteps=self.timesteps,
                    temperature=self.temperature,
                    text_temperature=text_temperature,
                    cfg_scale=self.cfg_scale,
                    cfg_img=self.cfg_img,
                    uncon_text=sd["uncon_text"],
                    uncon_image=sd["uncon_image"],
                    tokenizer=self._tokenizer,
                    remasking=self.remasking,
                )
                text_outputs.append((generated_text or "").strip())

                if self.gen_img_dir and output_tokens:
                    os.makedirs(self.gen_img_dir, exist_ok=True)
                    img_save_path = os.path.join(
                        self.gen_img_dir,
                        f"{batched_task[b_idx]}_{batched_doc_id[b_idx]}.png",
                    )
                    output_tokens_t = torch.tensor(
                        output_tokens, dtype=torch.long, device=self._device
                    ).unsqueeze(0)
                    decode_vq_to_image(
                        output_tokens_t,
                        save_path=img_save_path,
                        image_height=self.output_height,
                        image_width=self.output_width,
                        vqvae=self._vqvae,
                    )
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
