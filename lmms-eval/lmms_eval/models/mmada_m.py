"""lmms_eval adapter for MMaDA-Parallel-M (two-stage rollout).

Wraps the rollout half of ``MMaDAGRPOTrainer._generate_and_score_completions``
(see ``MMaDA-Parallel/MMaDA-Parallel-M/mmada_grpo_trainer.py``) behind the
lmms ``generate_until`` interface. Per request:

  1. Image rollout — ``model.t2i_generate`` over ``f"{edit_prompt} {ctx}"``,
     batched in chunks of ``image_edit_batch_size``.
  2. Text rollout — ``model.mmu_generate`` conditioned on the original input
     image AND the rollout-generated image, batched in chunks of
     ``batch_size``.

Reference: ``tmp/mmada-parallel-m-sft/chartqa_interleave_generate.py``.

Module-import side effects:
  - inserts ``<repo>/MMaDA-Parallel/MMaDA-Parallel-M`` into ``sys.path`` so
    M's ``models`` and ``training`` packages resolve.
  - drops the empty cwd entry and the absolute repo-root entry that
    ``python -m lmms_eval`` injects, so the repo-root ``model/`` /
    ``models/`` packages can't shadow M's ``models/`` package.
"""

import contextlib
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")

# Locate the MMaDA-Parallel-M source tree at repo_root/MMaDA-Parallel/MMaDA-Parallel-M.
# lmms-eval lives at repo_root/lmms-eval/lmms_eval/models/mmada_m.py — four parents
# up from this file is repo_root.
_MMADA_M_DIR = Path(__file__).resolve().parents[3] / "MMaDA-Parallel" / "MMaDA-Parallel-M"
_REPO_ROOT = str(_MMADA_M_DIR.parents[1])
sys.path = [p for p in sys.path if p not in ("", _REPO_ROOT)]
if str(_MMADA_M_DIR) not in sys.path:
    sys.path.insert(0, str(_MMADA_M_DIR))
for _stale in [k for k in list(sys.modules) if k == "models" or k.startswith("models.")]:
    del sys.modules[_stale]
for _stale in [k for k in list(sys.modules) if k == "training" or k.startswith("training.")]:
    del sys.modules[_stale]

from models import MAGVITv2, MMadaModelLM, get_mask_schedule  # noqa: E402
from models import modeling_mmada as _modeling_mmada  # noqa: E402
from training.prompting_utils import UniversalPrompting  # noqa: E402
from training.utils import image_transform_squash  # noqa: E402


# Trainer / inference defaults; mirror constants from the reference script.
DEFAULT_RESOLUTION = 512
DEFAULT_NUM_VQ_TOKENS = 1024
DEFAULT_CODEBOOK_SIZE = 8192
DEFAULT_MASK_TOKEN_ID = 126336

DEFAULT_EDIT_PROMPT = (
    "Edit the region where auxiliary line, box, or drawing could help solve "
    "the following problem."
)
DEFAULT_COT_PROMPT = ""

# Llama-3-style chat template shipped by tyfeld/MMaDA-Parallel-M. Some
# fine-tune ckpts (RL/SFT) save it as a sidecar ``chat_template.jinja`` instead
# of embedding it in ``tokenizer_config.json``; older ``transformers`` releases
# don't auto-load the sidecar, leaving ``tokenizer.chat_template = None``. We
# fall back to this string in that case.
_FALLBACK_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
)

_UNI_SPECIAL_TOKENS = (
    "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
    "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
)


@contextlib.contextmanager
def _softmax_fp64_to_fp32():
    # bf16 autocast on this model can produce fp64 softmax inputs that crash
    # without an explicit downcast. Lifted from the reference script.
    orig = _modeling_mmada.F.softmax

    def patched(x, *args, **kwargs):
        if x.dtype == torch.float64:
            x = x.to(torch.float32)
        return orig(x, *args, **kwargs)

    _modeling_mmada.F.softmax = patched
    try:
        yield
    finally:
        _modeling_mmada.F.softmax = orig


@register_model("mmada_m")
class MmadaM(lmms):
    """MMaDA-Parallel-M two-stage interleaved-generation adapter.

    Args (via ``--model_args key=val,...``):
        pretrained: HF/local ckpt for ``MMadaModelLM``
            (e.g. ``tyfeld/MMaDA-Parallel-M``).
        vq_model: HF/local ckpt for ``MAGVITv2`` (default
            ``showlab/magvitv2``).
        tokenizer_path: HF/local path holding tokenizer files. Falls back to
            ``pretrained``; useful for fine-tunes that drop tokenizer assets.
        batch_size: outer Collator chunk size. Inner image / text rollouts
            sub-chunk by ``image_edit_batch_size`` / ``batch_size``.
        device: torch device string.
        image_edit_batch_size / image_guidance_scale / image_timesteps /
            image_temperature / resolution / num_vq_tokens / codebook_size /
            mask_token_id: image-rollout knobs (mirror trainer defaults).
        batch_size / text_temperature / text_cfg_scale /
            text_remasking: text-rollout knobs (mirror trainer defaults;
            ``diffusion_steps`` and ``block_length`` are derived per request
            from ``gen_kwargs.max_new_tokens``).
        edit_prompt: prepended to the question for the image rollout.
        cot_prompt: prepended to the question for the text rollout.
        gen_img_dir: directory to write rollout-generated PNGs to (parity
            with the ``mmada`` / ``anole`` adapters).
        seed: RNG seed for the image-rollout sampler.

    Notes:
        * ``UniversalPrompting``'s ``max_text_len`` is intentionally left at
          its library default — no instruction-side cap is imposed here.
        * The text-rollout answer-length budget (``max_seq_length``) is read
          per-request from ``gen_kwargs["max_new_tokens"]``. Tasks must set
          it via their generation_kwargs; the adapter raises if missing.
    """

    def __init__(
        self,
        pretrained: str,
        vq_model: str = "showlab/magvitv2",
        tokenizer_path: Optional[str] = None,
        batch_size: Union[int, str] = 1,
        device: str = "cuda:0",
        # image rollout
        image_edit_batch_size: int = 2,
        image_guidance_scale: float = 0.0,
        image_timesteps: int = 10,
        image_temperature: float = 1.0,
        resolution: int = DEFAULT_RESOLUTION,
        num_vq_tokens: int = DEFAULT_NUM_VQ_TOKENS,
        codebook_size: int = DEFAULT_CODEBOOK_SIZE,
        mask_token_id: int = DEFAULT_MASK_TOKEN_ID,
        # text rollout
        text_temperature: float = 0.0,
        text_cfg_scale: float = 0.0,
        text_remasking: str = "low_confidence",
        # prompts
        edit_prompt: str = DEFAULT_EDIT_PROMPT,
        cot_prompt: str = DEFAULT_COT_PROMPT,
        # I/O
        gen_img_dir: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()

        if kwargs:
            eval_logger.warning(f"[mmada_m] ignoring unexpected kwargs: {kwargs}")

        self.pretrained = pretrained
        self.vq_model_path = vq_model
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

        self.image_edit_batch_size = int(image_edit_batch_size)
        self.image_guidance_scale = float(image_guidance_scale)
        self.image_timesteps = int(image_timesteps)
        self.image_temperature = float(image_temperature)
        self.resolution = int(resolution)
        self.num_vq_tokens = int(num_vq_tokens)
        self.codebook_size = int(codebook_size)
        self.mask_token_id = int(mask_token_id)

        self.batch_size_per_gpu = int(batch_size)
        self.text_temperature = float(text_temperature)
        self.text_cfg_scale = float(text_cfg_scale)
        self.text_remasking = text_remasking

        self.edit_prompt = edit_prompt
        self.cot_prompt = cot_prompt

        self.gen_img_dir = gen_img_dir
        self.seed = int(seed)
        self.datetime_str = None

        eval_logger.info(f"[mmada_m] loading tokenizer from {self.tokenizer_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, padding_side="left"
        )
        if not getattr(self._tokenizer, "chat_template", None):
            sidecar = Path(self.tokenizer_path) / "chat_template.jinja"
            if sidecar.is_file():
                eval_logger.warning(
                    f"[mmada_m] tokenizer at {self.tokenizer_path} has no chat_template; "
                    f"loading sidecar {sidecar}"
                )
                self._tokenizer.chat_template = sidecar.read_text()
            else:
                eval_logger.warning(
                    f"[mmada_m] tokenizer at {self.tokenizer_path} has no chat_template "
                    "and no sidecar chat_template.jinja; falling back to the canonical "
                    "MMaDA-Parallel-M template."
                )
                self._tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE
        # Leave max_text_len at the UniversalPrompting class default (8000).
        self._uni_prompting = UniversalPrompting(
            self._tokenizer,
            special_tokens=_UNI_SPECIAL_TOKENS,
            ignore_id=-100,
            cond_dropout_prob=0.1,
            use_reserved_token=True,
        )

        eval_logger.info(f"[mmada_m] loading VQ model from {self.vq_model_path}")
        # low_cpu_mem_usage=False forces full RAM allocation up front; the
        # meta-device path otherwise calls diffusers' load_model_dict_into_meta
        # with a device= kwarg that recent diffusers releases removed.
        self._vq_model = MAGVITv2.from_pretrained(
            self.vq_model_path, low_cpu_mem_usage=False
        ).to(self._device)
        self._vq_model.requires_grad_(False)
        self._vq_model.eval()

        eval_logger.info(f"[mmada_m] loading MMadaModelLM from {pretrained}")
        self._model = MMadaModelLM.from_pretrained(
            pretrained, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(self._device)
        self._model.eval()

        self._mask_schedule = get_mask_schedule("cosine")
        self._generator = torch.Generator(device=self._device).manual_seed(self.seed)

        eval_logger.info(
            f"[mmada_m] image_bs={self.image_edit_batch_size} "
            f"text_bs={self.batch_size} resolution={self.resolution} "
            f"seed={self.seed} gen_img_dir={self.gen_img_dir}"
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
        # Diffusion-style M doesn't expose token logprobs.
        pass

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for mmada_m.")

    # ---------- internal helpers ----------

    def _build_min_config(self, max_seq_length: int):
        return OmegaConf.create(
            {
                "model": {
                    "mmada": {
                        "num_vq_tokens": self.num_vq_tokens,
                        "codebook_size": self.codebook_size,
                    }
                },
                "dataset": {"preprocessing": {"max_seq_length": max_seq_length}},
            }
        )

    def _load_image_tensor(self, img_pil) -> torch.Tensor:
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
        return image_transform_squash(img_pil, resolution=self.resolution).to(self._device)

    def _decode_vq_batch(self, output_image_ids: torch.Tensor) -> List[Image.Image]:
        output_image_ids = torch.clamp(output_image_ids, 0, self.codebook_size - 1).to(torch.long)
        images = self._vq_model.decode_code(output_image_ids)
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0) * 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return [Image.fromarray(arr) for arr in images]

    def _rollout_image_edit(self, examples: List[dict], cfg) -> List[dict]:
        """Mirrors trainer's ``_rollout_image_edit_latents``; chunked by
        ``image_edit_batch_size``."""
        results: List[Optional[dict]] = [None] * len(examples)
        tokenizer = self._uni_prompting.text_tokenizer

        for s in range(0, len(examples), self.image_edit_batch_size):
            chunk = examples[s : s + self.image_edit_batch_size]
            B = len(chunk)

            pixel_batch = torch.stack(
                [self._load_image_tensor(ex["image"]) for ex in chunk], dim=0
            )
            # input_image_tokens_shifted is unused downstream here (we don't
            # score), but the reference encodes them anyway — keep the call
            # for parity (warms the VQ encoder + matches RNG ordering).
            _ = self._vq_model.get_code(pixel_batch) + len(tokenizer)

            prompts = [ex["edit_instruction"] for ex in chunk]
            masked_image_tokens = torch.full(
                (B, self.num_vq_tokens),
                self.mask_token_id,
                dtype=torch.long,
                device=self._device,
            )
            input_ids, attention_mask = self._uni_prompting(
                (prompts, masked_image_tokens, None, 0.0), "t2i_gen",
            )
            if self.image_guidance_scale > 0:
                uncond_input_ids, uncond_attention_mask = self._uni_prompting(
                    ([""] * B, masked_image_tokens, None, 0.0), "t2i_gen",
                )
            else:
                uncond_input_ids = None
                uncond_attention_mask = None

            with torch.no_grad():
                output_image_ids = self._model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    uncond_attention_mask=uncond_attention_mask,
                    guidance_scale=self.image_guidance_scale,
                    temperature=self.image_temperature,
                    timesteps=self.image_timesteps,
                    noise_schedule=self._mask_schedule,
                    seq_len=self.num_vq_tokens,
                    mask_token_id=self.mask_token_id,
                    resolution=self.resolution,
                    codebook_size=self.codebook_size,
                    uni_prompting=self._uni_prompting,
                    config=cfg,
                    generator=self._generator,
                )

            output_image_ids = torch.clamp(
                output_image_ids, 0, self.codebook_size - 1
            ).to(torch.long)
            decoded_images = self._decode_vq_batch(output_image_ids)
            for off in range(B):
                results[s + off] = {"decoded_image": decoded_images[off]}

        return results

    def _rollout_text_gen(
        self,
        examples: List[dict],
        max_seq_length: int,
        diffusion_steps: int,
        block_length: int,
    ) -> List[str]:
        """Mirrors trainer's ``_rollout_multimodal_text_gen`` with
        ``text_rollout_use_gen_image=True``; chunked by
        ``batch_size``."""
        results: List[Optional[str]] = [None] * len(examples)
        tokenizer = self._uni_prompting.text_tokenizer
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        mmu_tok = int(self._uni_prompting.sptids_dict["<|mmu|>"])
        soi = int(self._uni_prompting.sptids_dict["<|soi|>"])
        eoi = int(self._uni_prompting.sptids_dict["<|eoi|>"])

        for s in range(0, len(examples), self.batch_size):
            chunk = examples[s : s + self.batch_size]
            B = len(chunk)

            input_pixel_batch = torch.stack(
                [self._load_image_tensor(ex["image"]) for ex in chunk], dim=0
            )
            gen_pixel_batch = torch.stack(
                [self._load_image_tensor(ex["gen_image"]) for ex in chunk], dim=0
            )
            # Chunk the VQ encode by image_edit_batch_size — at large
            # text-rollout batch sizes (e.g. 128+) running the MAGVITv2
            # encoder over the full pixel batch in one shot OOMs at 140 GiB
            # on H200s. Tokens are still concatenated into a single
            # [B, num_vq_tokens] tensor afterwards.
            def _chunked_vq_encode(pixel_batch: torch.Tensor) -> torch.Tensor:
                step = max(1, self.image_edit_batch_size)
                pieces = [
                    self._vq_model.get_code(pixel_batch[i : i + step])
                    for i in range(0, pixel_batch.shape[0], step)
                ]
                return torch.cat(pieces, dim=0)

            input_image_tokens_shifted = (
                _chunked_vq_encode(input_pixel_batch) + len(tokenizer)
            )
            gen_image_tokens_shifted = (
                _chunked_vq_encode(gen_pixel_batch) + len(tokenizer)
            )

            text_token_lists = []
            for ex in chunk:
                messages = [{"role": "user", "content": str(ex["answer_instruction"])}]
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                text_token_lists.append(list(ids))
            max_text = max(len(ids) for ids in text_token_lists)
            padded = [[pad_id] * (max_text - len(ids)) + ids for ids in text_token_lists]
            text_batch = torch.tensor(padded, dtype=torch.long, device=self._device)

            mmu_col = torch.full((B, 1), mmu_tok, dtype=torch.long, device=self._device)
            soi_col = torch.full((B, 1), soi, dtype=torch.long, device=self._device)
            eoi_col = torch.full((B, 1), eoi, dtype=torch.long, device=self._device)
            input_ids = torch.cat(
                [
                    mmu_col, soi_col, input_image_tokens_shifted, eoi_col,
                    soi_col, gen_image_tokens_shifted, eoi_col,
                    text_batch,
                ],
                dim=1,
            )

            prefix_len = (
                3
                + input_image_tokens_shifted.shape[1]
                + 2
                + gen_image_tokens_shifted.shape[1]
            )
            prefix_mask = torch.ones((B, prefix_len), dtype=torch.long, device=self._device)
            text_mask = (text_batch != pad_id).long()
            gen_mask = torch.ones((B, max_seq_length), dtype=torch.long, device=self._device)
            attention_mask = torch.cat([prefix_mask, text_mask, gen_mask], dim=1)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), _softmax_fp64_to_fp32():
                output_ids = self._model.mmu_generate(
                    input_ids,
                    max_new_tokens=max_seq_length,
                    steps=diffusion_steps,
                    block_length=block_length,
                    temperature=self.text_temperature,
                    cfg_scale=self.text_cfg_scale,
                    remasking=self.text_remasking,
                    attention_mask=attention_mask,
                    mask_id=self.mask_token_id,
                )

            gen_ids = output_ids[:, input_ids.shape[1]:]
            decoded_texts = tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for off, txt in enumerate(decoded_texts):
                results[s + off] = txt

        return results

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

            batch_pil_images = [
                doc_to_visual(self.task_dict[task_name][split_name][doc_id])
                for doc_to_visual, task_name, split_name, doc_id in zip(
                    batched_doc_to_visual, batched_task, batched_split, batched_doc_id
                )
            ]  # List[List[PIL.Image]]

            max_seq_length = int(gen_kwargs.get("max_new_tokens", 256))
            diffusion_steps = max(1, max_seq_length // 2)
            block_length = max(1, max_seq_length // 4)
            cfg = self._build_min_config(max_seq_length)

            examples: List[dict] = []
            for ctx, images in zip(batched_contexts, batch_pil_images):
                if not images:
                    raise ValueError(
                        "[mmada_m] generate_until requires an input image; "
                        "got empty visual list."
                    )
                examples.append(
                    {
                        "image": images[0],
                        "question": ctx,
                        "edit_instruction": f"{self.edit_prompt} {ctx}",
                        "answer_instruction": (f"{self.cot_prompt} {ctx}").strip(),
                    }
                )

            image_contexts = self._rollout_image_edit(examples, cfg)
            for ex, img_ctx in zip(examples, image_contexts):
                ex["gen_image"] = img_ctx["decoded_image"]

            text_outputs = self._rollout_text_gen(
                examples, max_seq_length, diffusion_steps, block_length
            )
            text_outputs = [(t or "").strip() for t in text_outputs]

            if self.gen_img_dir:
                os.makedirs(self.gen_img_dir, exist_ok=True)
                for b_idx, ex in enumerate(examples):
                    img_save_path = os.path.join(
                        self.gen_img_dir,
                        f"{batched_task[b_idx]}_{batched_doc_id[b_idx]}.png",
                    )
                    ex["gen_image"].save(img_save_path)
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
