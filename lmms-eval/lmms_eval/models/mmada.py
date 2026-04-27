import copy
import logging
import os
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")
eval_logger = logging.getLogger("lmms-eval")
torch.backends.cuda.matmul.allow_tf32 = True


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG_PRINT_OUTPUT = _env_flag("DEBUG_PRINT_OUTPUT")
LOG_BATCH_TIMING = _env_flag("LOG_BATCH_TIMING", default=True)


class _DotDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


def _to_dot(value):
    if isinstance(value, dict):
        return _DotDict({k: _to_dot(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_dot(v) for v in value]
    return value


_MMADA_REPO = Path(__file__).resolve().parents[3] / "MMaDA-Parallel-M"
if str(_MMADA_REPO) not in sys.path:
    sys.path.insert(0, str(_MMADA_REPO))

from data_utils import COT_PROMPT, EDIT_PROMPT
from models import MAGVITv2, MMadaModelLM, get_mask_schedule
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform_squash


_RESOLUTION = 512
_NUM_VQ_TOKENS = 1024
_CODEBOOK_SIZE = 8192
# Length of the input-text slot in the model's training format. Both the
# t2i_gen prompt template (UniversalPrompting.t2i_gen_prompt) and the
# interleave training format right-pad the input text to this length, so
# this is structural — not a generation cap. Output length comes from
# `gen_kwargs["max_new_tokens"]` (task default) at call time.
_INPUT_TEXT_LEN = 256
_MASK_TOKEN_ID = 126336
_VQ_MODEL_NAME = "showlab/magvitv2"

_RESERVED_TOKENS = {
    "<|soi|>": 126084,
    "<|eoi|>": 126085,
    "<|sov|>": 126086,
    "<|eov|>": 126087,
    "<|t2i|>": 126088,
    "<|mmu|>": 126089,
    "<|t2v|>": 126090,
    "<|v2v|>": 126091,
    "<|lvg|>": 126092,
    "[iPAD]": 126093,
    "<|r2i|>": 126094,
    "<|interleave|>": 126095,
}


@register_model("mmada")
class Mmada(lmms):
    """
    MMaDA-Parallel-M adapter for lmms-eval. Always runs the image_gen → text_gen
    pipeline: t2i_generate produces an intermediate edit image, then mmu_generate
    answers the question conditioned on (original image, intermediate image, prompt).
    """

    def __init__(
        self,
        pretrained: str = "tyfeld/MMaDA-Parallel-M",
        device: Optional[str] = "cuda:0",
        device_map: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        vq_model_name: str = _VQ_MODEL_NAME,
        img_gen_resolution: int = _RESOLUTION,
        img_gen_n_steps: int = 20,
        img_gen_temperature: float = 0.2,
        img_gen_guidance_scale: float = 0.0,
        img_gen_seed_ratio: float = 0.0,
        img_gen_mask_schedule: str = "cosine",
        text_cfg_scale: float = 0.0,
        text_remasking: str = "low_confidence",
        gen_img_dir: Optional[str] = None,
        chat_mode: Optional[str] = None,
        use_bbox: bool = False,
        t2i_chunk_size: int = 8,
        mmu_chunk_size: Optional[int] = None,
        interleave_chunk_size: int = 8,
        interleave_text_cfg: float = 2.5,
        interleave_image_cfg: float = 4.0,
        interleave_text_steps: int = 128,
        interleave_image_steps: int = 30,
        **kwargs,
    ) -> None:
        super().__init__()

        VALID_CHAT_MODES = (None, "text_gen", "image_gen")
        if chat_mode not in VALID_CHAT_MODES:
            raise ValueError(f"Invalid chat_mode={chat_mode!r}. Must be one of {VALID_CHAT_MODES}")
        self.chat_mode = "image_gen" if chat_mode is None else chat_mode
        self.use_bbox = bool(use_bbox)

        self.img_gen_resolution = int(img_gen_resolution)
        self.img_gen_n_steps = int(img_gen_n_steps)
        self.img_gen_temperature = float(img_gen_temperature)
        self.img_gen_guidance_scale = float(img_gen_guidance_scale)
        self.img_gen_seed_ratio = float(img_gen_seed_ratio)
        self.img_gen_mask_schedule = str(img_gen_mask_schedule)

        self.text_cfg_scale = float(text_cfg_scale)
        self.text_remasking = str(text_remasking)

        self.gen_img_dir = gen_img_dir
        self.datetime_str = None
        self.t2i_chunk_size = max(1, int(t2i_chunk_size))
        self.mmu_chunk_size = max(1, int(mmu_chunk_size)) if mmu_chunk_size is not None else None
        self.interleave_chunk_size = max(1, int(interleave_chunk_size))
        self.interleave_text_cfg = float(interleave_text_cfg)
        self.interleave_image_cfg = float(interleave_image_cfg)
        self.interleave_text_steps = int(interleave_text_steps)
        self.interleave_image_steps = int(interleave_image_steps)

        if kwargs:
            eval_logger.warning(f"Unexpected kwargs (ignored): {kwargs}")

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.batch_size_per_gpu = int(batch_size)

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        self._uni_prompting = UniversalPrompting(
            self._tokenizer,
            max_text_len=_INPUT_TEXT_LEN,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>",
                "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.0,
            use_reserved_token=True,
        )

        self._vq_model = MAGVITv2.from_pretrained(vq_model_name, low_cpu_mem_usage=False).to(self._device)
        self._vq_model.requires_grad_(False)
        self._vq_model.eval()

        self._model = MMadaModelLM.from_pretrained(pretrained, torch_dtype=torch.bfloat16)
        self._model.eval()
        self._model.requires_grad_(False)

        self._cfg = _to_dot({
            "model": {"mmada": {"num_vq_tokens": _NUM_VQ_TOKENS, "codebook_size": _CODEBOOK_SIZE}},
            "dataset": {"preprocessing": {"max_seq_length": _INPUT_TEXT_LEN, "resolution": self.img_gen_resolution}},
            "training": {
                "guidance_scale": self.img_gen_guidance_scale,
                "generation_timesteps": self.img_gen_n_steps,
                "generation_temperature": self.img_gen_temperature,
                "cond_dropout_prob": 0.0,
                "noise_type": "mask",
            },
            "mask_schedule": {"schedule": self.img_gen_mask_schedule},
        })

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], \
                "Unsupported distributed type. Only DDP/FSDP/DeepSpeed are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **ds_kwargs)
                eval_logger.info("Detected DistributedType.DEEPSPEED — set zero stage to 0.")

            self._model.to(self._device)
            self._model = accelerator.prepare(self._model)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

        self._max_length = _INPUT_TEXT_LEN
        self._config = self._model.config

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

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

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self._tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self._tokenizer.decode(tokens)
        except Exception:
            return self._tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        pass

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def _pixels_from_pils(self, pil_list_per_sample, take_first: bool = True):
        """Convert a list[PIL.Image] (or list[list[PIL.Image]]) to a (B, 3, H, H) tensor."""
        device = self._device
        pixels = []
        for item in pil_list_per_sample:
            img = item[0] if take_first and isinstance(item, list) else item
            if img.mode != "RGB":
                img = img.convert("RGB")
            pixels.append(image_transform_squash(img, resolution=self.img_gen_resolution).to(device))
        return torch.stack(pixels, dim=0)

    def _decode_vq(self, token_ids: torch.Tensor) -> List[Image.Image]:
        token_ids = torch.clamp(token_ids, 0, _CODEBOOK_SIZE - 1).to(torch.long)
        images = self._vq_model.decode_code(token_ids)
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        images = (images * 255.0).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        return [Image.fromarray(img) for img in images]

    def _interleave_rollout(
        self, edit_prompts: List[str], batch_pil_images, gen_kwargs: dict
    ) -> Tuple[List[Image.Image], List[str]]:
        images: List[Image.Image] = []
        texts: List[str] = []
        chunk = self.interleave_chunk_size
        for i in range(0, len(edit_prompts), chunk):
            sub_prompts = edit_prompts[i:i + chunk]
            sub_pils = batch_pil_images[i:i + chunk]
            sub_imgs, sub_txts = self._interleave_rollout_chunk(sub_prompts, sub_pils, gen_kwargs)
            images.extend(sub_imgs)
            texts.extend(sub_txts)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return images, texts

    def _interleave_rollout_chunk(
        self, edit_prompts: List[str], batch_pil_images, gen_kwargs: dict
    ) -> Tuple[List[Image.Image], List[str]]:
        device = self._device
        B = len(edit_prompts)
        tok = self._uni_prompting.text_tokenizer

        pixel_batch = self._pixels_from_pils(batch_pil_images, take_first=True)
        image_tokens_shifted = self._vq_model.get_code(pixel_batch) + len(tok)
        uncond_image_tokens = torch.zeros_like(image_tokens_shifted)

        bos = tok.bos_token_id
        eos = tok.eos_token_id

        cond_lists: List[List[int]] = []
        uncond_lists: List[List[int]] = []
        for prompt in edit_prompts:
            ids = tok(prompt)["input_ids"]
            ids = list(ids)
            if not ids or ids[0] != bos:
                ids = [bos] + ids
            ids = ids + [eos]
            cond_lists.append(ids)

            u = list(tok("")["input_ids"])
            if not u or u[0] != bos:
                u = [bos] + u
            u = u + [eos]
            uncond_lists.append(u)

        max_len = max(len(ids) for ids in cond_lists)
        for i in range(B):
            cond_lists[i] = cond_lists[i] + [eos] * (max_len - len(cond_lists[i]))
            uncond_lists[i] = uncond_lists[i] + [eos] * (max_len - len(uncond_lists[i]))

        text_ids = torch.tensor(cond_lists, dtype=torch.long, device=device)
        uncond_text_ids = torch.tensor(uncond_lists, dtype=torch.long, device=device)

        interleave_col = torch.full((B, 1), _RESERVED_TOKENS["<|interleave|>"], dtype=torch.long, device=device)
        soi_col = torch.full((B, 1), _RESERVED_TOKENS["<|soi|>"], dtype=torch.long, device=device)
        eoi_col = torch.full((B, 1), _RESERVED_TOKENS["<|eoi|>"], dtype=torch.long, device=device)

        input_ids = torch.cat([interleave_col, soi_col, image_tokens_shifted, eoi_col, text_ids], dim=1)
        uncond_input_ids = torch.cat(
            [interleave_col, soi_col, uncond_image_tokens, eoi_col, uncond_text_ids], dim=1
        )

        max_new_tokens = self._max_new_tokens_from_kwargs(gen_kwargs)
        text_temperature = float(gen_kwargs.get("temperature", 0.0)) if gen_kwargs.get("temperature") is not None else 0.0
        # interleave_generate reads cfg.dataset.preprocessing.max_seq_length to size
        # the output text slot, so override it for this call.
        prev_max_seq = self._cfg.dataset.preprocessing.max_seq_length
        self._cfg.dataset.preprocessing.max_seq_length = int(max_new_tokens)

        schedule = get_mask_schedule(self.img_gen_mask_schedule)
        try:
            with torch.no_grad():
                output_image_ids, output_text_ids = self.model.interleave_generate(
                    input_ids,
                    uncond_input_ids,
                    text_cfg=self.interleave_text_cfg,
                    image_cfg=self.interleave_image_cfg,
                    noise_schedule=schedule,
                    text_steps=self.interleave_text_steps,
                    image_steps=self.interleave_image_steps,
                    text_temperature=text_temperature,
                    reserved_token_mapping=_RESERVED_TOKENS,
                    uni_prompting=self._uni_prompting,
                    config=self._cfg,
                )
        finally:
            self._cfg.dataset.preprocessing.max_seq_length = prev_max_seq

        output_image_ids = torch.clamp(output_image_ids, 0, _CODEBOOK_SIZE - 1).to(torch.long)
        pil_images = self._decode_vq(output_image_ids)
        output_texts = tok.batch_decode(output_text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return pil_images, output_texts

    def _t2i_rollout(self, edit_prompts: List[str], batch_pil_images) -> List[Image.Image]:
        results: List[Image.Image] = []
        chunk = self.t2i_chunk_size
        for i in range(0, len(edit_prompts), chunk):
            sub_prompts = edit_prompts[i:i + chunk]
            sub_pils = batch_pil_images[i:i + chunk]
            results.extend(self._t2i_rollout_chunk(sub_prompts, sub_pils))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results

    def _t2i_rollout_chunk(self, edit_prompts: List[str], batch_pil_images) -> List[Image.Image]:
        device = self._device
        B = len(edit_prompts)
        tok = self._uni_prompting.text_tokenizer

        pixel_batch = self._pixels_from_pils(batch_pil_images, take_first=True)
        input_image_tokens_shifted = self._vq_model.get_code(pixel_batch) + len(tok)

        masked = torch.full((B, _NUM_VQ_TOKENS), _MASK_TOKEN_ID, dtype=torch.long, device=device)
        ref_ids = input_image_tokens_shifted if self.img_gen_seed_ratio > 0 else None

        input_ids, attn = self._uni_prompting(
            (edit_prompts, masked, ref_ids, self.img_gen_seed_ratio), "t2i_gen"
        )
        if self.img_gen_guidance_scale > 0:
            u_ids, u_attn = self._uni_prompting(
                ([""] * B, masked, ref_ids, self.img_gen_seed_ratio), "t2i_gen"
            )
        else:
            u_ids, u_attn = None, None

        schedule = get_mask_schedule(self.img_gen_mask_schedule)
        with torch.no_grad():
            out = self.model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=u_ids,
                attention_mask=attn,
                uncond_attention_mask=u_attn,
                guidance_scale=self.img_gen_guidance_scale,
                temperature=self.img_gen_temperature,
                timesteps=self.img_gen_n_steps,
                noise_schedule=schedule,
                seq_len=_NUM_VQ_TOKENS,
                mask_token_id=_MASK_TOKEN_ID,
                resolution=self.img_gen_resolution,
                codebook_size=_CODEBOOK_SIZE,
                uni_prompting=self._uni_prompting,
                config=self._cfg,
            )

        return self._decode_vq(out)

    def _mmu_rollout(
        self,
        und_prompts: List[str],
        batch_pil_images,
        intermediate_images: List[Image.Image],
        gen_kwargs: dict,
    ) -> List[str]:
        chunk = self.mmu_chunk_size if self.mmu_chunk_size is not None else len(und_prompts)
        if chunk >= len(und_prompts):
            return self._mmu_rollout_chunk(und_prompts, batch_pil_images, intermediate_images, gen_kwargs)
        results: List[str] = []
        for i in range(0, len(und_prompts), chunk):
            sub_prompts = und_prompts[i:i + chunk]
            sub_pils = batch_pil_images[i:i + chunk]
            sub_intermediate = intermediate_images[i:i + chunk]
            results.extend(self._mmu_rollout_chunk(sub_prompts, sub_pils, sub_intermediate, gen_kwargs))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results

    def _mmu_rollout_chunk(
        self,
        und_prompts: List[str],
        batch_pil_images,
        intermediate_images: List[Image.Image],
        gen_kwargs: dict,
    ) -> List[str]:
        """
        Build the interleave-format prefix exactly as the model was trained:
            [<|interleave|>, <|soi|>, in_img, <|eoi|>, in_text_padded, <|soi|>, gen_img, <|eoi|>]
        Then mmu_generate appends max_new_tokens of MASK tokens (corresponding to the
        out_text slot in training) and unmasks them via the diffusion loop.
        Right-pads in_text to a fixed max_text_len with EOS to match training, and
        skips the broken attention_mask path (interleave training uses no attention
        mask, padding is fixed-length per sample so batch composition does not change
        per-doc inputs).
        """
        device = self._device
        B = len(und_prompts)
        tok = self._uni_prompting.text_tokenizer
        bos = tok.bos_token_id
        eos = tok.eos_token_id
        input_pad_len = _INPUT_TEXT_LEN

        in_pix = self._pixels_from_pils(batch_pil_images, take_first=True)
        gen_pix = self._pixels_from_pils(intermediate_images, take_first=False)
        in_tok = self._vq_model.get_code(in_pix) + len(tok)
        gen_tok = self._vq_model.get_code(gen_pix) + len(tok)

        raw_ids_list = tok(und_prompts)["input_ids"]
        text_ids_padded = []
        for ids in raw_ids_list:
            ids = list(ids)
            if not ids or ids[0] != bos:
                ids = [bos] + ids
            if not ids or ids[-1] != eos:
                ids = ids + [eos]
            if len(ids) > input_pad_len:
                ids = ids[:input_pad_len - 1] + [eos]
            else:
                ids = ids + [eos] * (input_pad_len - len(ids))
            text_ids_padded.append(ids)
        text_batch = torch.tensor(text_ids_padded, dtype=torch.long, device=device)

        interleave_tok = _RESERVED_TOKENS["<|interleave|>"]
        soi = _RESERVED_TOKENS["<|soi|>"]
        eoi = _RESERVED_TOKENS["<|eoi|>"]
        col = lambda v: torch.full((B, 1), v, dtype=torch.long, device=device)

        input_ids = torch.cat(
            [col(interleave_tok), col(soi), in_tok, col(eoi), text_batch, col(soi), gen_tok, col(eoi)],
            dim=1,
        )

        max_new = self._max_new_tokens_from_kwargs(gen_kwargs)
        block_len = max(1, max_new // 4)
        if "block_length" in gen_kwargs and gen_kwargs["block_length"]:
            block_len = max(1, int(gen_kwargs["block_length"]))
        num_blocks = max(1, max_new // block_len)
        steps = max(1, max_new // 2)
        if "step_per_block" in gen_kwargs and gen_kwargs["step_per_block"]:
            steps = int(gen_kwargs["step_per_block"]) * num_blocks
        if max_new % block_len != 0:
            block_len = max(1, max_new // num_blocks)
            max_new = block_len * num_blocks
        if steps % num_blocks != 0:
            steps = max(num_blocks, (steps // num_blocks) * num_blocks)

        text_temperature = float(gen_kwargs.get("temperature", 0.0)) if gen_kwargs.get("temperature") is not None else 0.0
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else _NullCtx()
        with torch.no_grad(), ctx:
            out = self.model.mmu_generate(
                input_ids,
                max_new_tokens=max_new,
                steps=steps,
                block_length=block_len,
                temperature=text_temperature,
                cfg_scale=self.text_cfg_scale,
                remasking=self.text_remasking,
                mask_id=_MASK_TOKEN_ID,
            )

        gen_ids = out[:, input_ids.shape[1]:]
        return tok.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    @staticmethod
    def _max_new_tokens_from_kwargs(gen_kwargs: dict) -> int:
        v = gen_kwargs.get("max_new_tokens") if isinstance(gen_kwargs, dict) else None
        if v is None:
            raise ValueError(
                "Mmada requires `max_new_tokens` in the task's generation_kwargs "
                "(or via --gen_kwargs); none was supplied."
            )
        return int(v)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        if DEBUG_PRINT_OUTPUT:
            re_ords = utils.Collator([reg.args for reg in requests], lambda x: x[-3], grouping=True)
        else:
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (len(requests) + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        delta_t = 0.0
        num_generated = 0

        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            gen_kwargs = dict(all_gen_kwargs[0])
            batch_size = len(batched_contexts)

            batch_pil_images = [
                doc_to_visual(self.task_dict[task_name][split_name][doc_id])
                for doc_to_visual, task_name, split_name, doc_id in zip(
                    batched_doc_to_visual, batched_task, batched_split, batched_doc_id
                )
            ]

            edit_prompts = [f"{EDIT_PROMPT} {ctx}" for ctx in batched_contexts]
            und_prompts = [f"{ctx}" for ctx in batched_contexts]

            t0 = time.time()
            if self.chat_mode == "image_gen":
                t_t2i0 = time.time()
                intermediate_images, text_outputs = self._interleave_rollout(edit_prompts, batch_pil_images, gen_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device=self._device)
                t_t2i1 = time.time()
                t_mmu0 = t_mmu1 = t_t2i1
            else:
                t_t2i0 = time.time()
                intermediate_images = self._t2i_rollout(edit_prompts, batch_pil_images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device=self._device)
                t_t2i1 = time.time()

                t_mmu0 = time.time()
                text_outputs = self._mmu_rollout(und_prompts, batch_pil_images, intermediate_images, gen_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device=self._device)
                t_mmu1 = time.time()

            text_outputs = [t.lstrip("!").strip() for t in text_outputs]

            t1 = time.time()
            delta_t += t1 - t0
            num_generated += batch_size
            chunk_total = t1 - t0
            if LOG_BATCH_TIMING:
                eval_logger.info(
                    f"[stage_timing] rank={self.rank} bs={batch_size} "
                    f"t2i={t_t2i1 - t_t2i0:.3f}s mmu={t_mmu1 - t_mmu0:.3f}s "
                    f"chunk_total={chunk_total:.3f}s per_sample={chunk_total / max(batch_size, 1):.3f}s"
                )

            img_save_paths: List[Optional[str]] = [None] * batch_size
            if self.gen_img_dir and self.rank == 0:
                os.makedirs(self.gen_img_dir, exist_ok=True)
            if self.gen_img_dir:
                for b_idx, gen_img in enumerate(intermediate_images):
                    if gen_img is None:
                        continue
                    task_name = batched_task[b_idx]
                    split_name = batched_split[b_idx]
                    doc_id = batched_doc_id[b_idx]
                    img_save_path = os.path.join(self.gen_img_dir, f"{task_name}_{doc_id}.png")
                    try:
                        gen_img.save(img_save_path)
                    except Exception as e:
                        eval_logger.warning(f"Failed to save gen image to {img_save_path}: {e}")
                        continue
                    img_save_paths[b_idx] = img_save_path
                    self.task_dict[task_name][split_name][doc_id]["gen_img_path"] = img_save_path

            if self.chat_mode == "image_gen":
                for b_idx in range(len(text_outputs)):
                    output = {
                        "image_gen_input": edit_prompts[b_idx],
                        "text_gen_input": und_prompts[b_idx],
                        "text_gen_output": text_outputs[b_idx],
                        "image_gen_output_path": img_save_paths[b_idx],
                    }
                    res.append(output)
            else:
                res.extend(text_outputs)

            for b_ctx, b_output in zip(batched_contexts, text_outputs):
                self.cache_hook.add_partial("generate_until", (b_ctx, gen_kwargs), b_output)
            pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Mmada")


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
