# ADOBE CONFIDENTIAL
# Copyright 2025 Adobe
# All Rights Reserved.
# NOTICE: All information contained herein is, and remains
# the property of Adobe and its suppliers, if any. The intellectual
# and technical concepts contained herein are proprietary to Adobe
# and its suppliers and are protected by all applicable intellectual
# property laws, including trade secret and copyright laws.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Adobe.

import copy
import json
import logging
import math
import re
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import os
import PIL
import torch
import accelerate
import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from decord import VideoReader, cpu
from packaging import version
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav
import time
# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG_PRINT_OUTPUT = _env_flag("DEBUG_PRINT_OUTPUT")
LOG_BATCH_TIMING = _env_flag("LOG_BATCH_TIMING", default=True)

from constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from conversation import conv_templates
from mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    pad_to_square_and_resize,
)
from llava.model.builder import load_pretrained_model
from llava.model.utils import pad_along_last_dim
from input_processor import LavidaOProcessor
from interleaved_inferencer import InterleavedInferencer

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_llada")
class Llava_Llada(lmms):
    """
    Llava Model
    """
    def __init__(
        self,
        pretrained: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "llava_llada",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord",
        mc_num=16,
        chat_mode: Optional[str] = None,
        use_bbox: Optional[bool] = True,
        img_gen_guidance_scale: float = 1.2,
        img_gen_guidance_scale_image: float = 1.4,
        img_gen_conf_policy: str = "stratified",
        img_gen_edit_mode: int = 1,
        img_gen_n_steps: int = 64,
        img_gen_temperature: float = 0.8,
        img_gen_enable_stratified: bool = False,
        img_gen_resolution: int = 512,
        gen_img_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.gen_img_dir = gen_img_dir
        # Validate and store chat_mode
        VALID_CHAT_MODES = (None, "text_gen", "image_gen")
        if chat_mode not in VALID_CHAT_MODES:
            raise ValueError(f"Invalid chat_mode={chat_mode!r}. Must be one of {VALID_CHAT_MODES}")
        self.chat_mode = "text_gen" if chat_mode is None else chat_mode
        self.use_bbox = use_bbox
        # Store image generation parameters with explicit type casts
        self.img_gen_guidance_scale = float(img_gen_guidance_scale)
        self.img_gen_guidance_scale_image = float(img_gen_guidance_scale_image)
        self.img_gen_conf_policy = str(img_gen_conf_policy)
        self.img_gen_edit_mode = int(img_gen_edit_mode)
        self.img_gen_n_steps = int(img_gen_n_steps)
        self.img_gen_temperature = float(img_gen_temperature)
        self.img_gen_enable_stratified = bool(img_gen_enable_stratified)
        self.img_gen_resolution = int(img_gen_resolution)
        self.datetime_str = None  # set by evaluator after model creation

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
        self.mc_num = mc_num
        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = 'llava_llada'# if model_name is not None else get_model_name_from_path(pretrained)
        self.overwrite_image_aspect = os.environ.get("LLAVA_OVERWRITE_IMAGE_ASPECT", None)
        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        llava_model_args["overwrite_config"] = overwrite_config

        vision_kwargs = dict(
            mm_vision_tower="google/siglip-so400m-patch14-384",
            mm_resampler_type=None,
            mm_projector_type='mlp2x_gelu',
            mm_hidden_size=1152,
            use_mm_proj=True
        )

        resize_embeddings = True # default behavior
            
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args,vision_kwargs=vision_kwargs,resize_embeddings=resize_embeddings)

        assert self._tokenizer is not None

        self._config = self._model.config
        self.processing_class = LavidaOProcessor(self._model, self._tokenizer, self._image_processor)
        self.model.eval()
        self.model.tie_weights()
        self.model.model.set_activation_checkpointing(None)
        self.model.requires_grad_(False)
        self.inferencer = InterleavedInferencer(self.model)
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # Image generation modes now support batched inference via text_to_image_batch

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            self.model.to(self._device).to(torch.bfloat16)
            self._model.model.transformer = accelerator.prepare(self.model.model.transformer)
        
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
            self.model.to(self._device).to(torch.bfloat16)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

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

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def _pad_image_for_gen(self, pil_image):
        """Pad image to square and resize to configured generation resolution."""
        return self.model.pad_image(pil_image, image_resolution=self.img_gen_resolution)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            # breakpoint()
            return -len(toks), x[0]

        metadata = requests[0].metadata
        if DEBUG_PRINT_OUTPUT:
            # do not sort by length, instead using lambda x:x[-3]
            re_ords = utils.Collator([reg.args for reg in requests], lambda x:x[-3], grouping=True)
        else:
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        origin_image_aspect_ratio = getattr(self._config, "image_aspect_ratio", None)
        delta_t = 0
        num_generated = 0
        expected_bs = self.batch_size_per_gpu
        # Set up generation kwargs
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            batch_size = len(batched_contexts)
            if LOG_BATCH_TIMING:
                eval_logger.info(
                    f"[batch_check] rank={self.rank} chunk_bs={batch_size} "
                    f"expected_bs={expected_bs} (mismatch={batch_size != expected_bs})"
                )
            batch_pil_images = [
                doc_to_visual(self.task_dict[task_name][split_name][doc_id]) # List[PIL.Image.Image]
                for doc_to_visual, task_name, split_name, doc_id in zip(
                    batched_doc_to_visual, batched_task, batched_split, batched_doc_id
                )
            ]  # List[List[PIL.Image.Image]]
            # assert type(batch_pil_images) == list and type(batch_pil_images[0]) == list and type(batch_pil_images[0][0]) == PIL.Image.Image, f"{type(batch_pil_images)}!=list {type(batch_pil_images[0])}!=list {type(batch_pil_images[0][0])}!=PIL.Image.Image"
            task_type = "image"

            image_gen_post_prompt = gen_kwargs.pop("image_gen_post_prompt", "")

            t0 = time.time()

            needs_image_gen = self.chat_mode == "image_gen"

            # --- Extract text_gen_kwargs from gen_kwargs ---
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 256
            schedule_kwargs = {}
            for key in list(gen_kwargs.keys()):
                if key.startswith('schedule__'):
                    value = gen_kwargs.pop(key)
                    schedule_kwargs[key.replace('schedule__', '')] = value
            if schedule_kwargs:
                gen_kwargs['schedule_kwargs'] = schedule_kwargs
            if 'block_length' not in gen_kwargs:
                if gen_kwargs["max_new_tokens"] < 128:
                    gen_kwargs['block_length'] = gen_kwargs["max_new_tokens"]
                else:
                    gen_kwargs['block_length'] = gen_kwargs["max_new_tokens"] // 2
            if 'step_per_block' not in gen_kwargs and 'step_ratio' not in gen_kwargs:
                gen_kwargs['step_per_block'] = max(1, gen_kwargs['block_length'] // 2)

            text_gen_kwargs = {
                "max_new_tokens": gen_kwargs.get("max_new_tokens", 256),
                "block_length": gen_kwargs.get("block_length", 64),
                "step_per_block": gen_kwargs.get("step_per_block", 32),
            }
            if "schedule_kwargs" in gen_kwargs:
                text_gen_kwargs["schedule_kwargs"] = gen_kwargs["schedule_kwargs"]

            device = self.model.get_model().device
            mask_id = 126336

            def _copy_conv():
                if "llama_3" in self.conv_template or "llada" in self.conv_template:
                    return copy.deepcopy(conv_templates[self.conv_template])
                return conv_templates[self.conv_template].copy()

            def _build_text_prompt(ctx, num_images):
                if num_images > 0 and DEFAULT_IMAGE_TOKEN not in ctx:
                    image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)
                    prompts_input = image_tokens + f"\n {ctx}"
                else:
                    prompts_input = f"{ctx}"
                conv = _copy_conv()
                conv.append_message(conv.roles[0], prompts_input)
                conv.append_message(conv.roles[1], None)
                return conv.get_prompt()

            def _build_grounding_prompt(ctx, num_images):
                from data_utils import GROUNDING_PROMPT
                ctx = ctx or ""
                if num_images > 0 and DEFAULT_IMAGE_TOKEN not in ctx:
                    prompts_input = f"{DEFAULT_IMAGE_TOKEN}\n{GROUNDING_PROMPT} {ctx}"
                else:
                    prompts_input = f"{GROUNDING_PROMPT} {ctx}"
                conv = _copy_conv()
                conv.append_message(conv.roles[0], prompts_input)
                conv.append_message(conv.roles[1], None)
                return conv.get_prompt() + "\n\n" + "<LOC_BEGIN><|mdm_mask|><|mdm_mask|><|mdm_mask|><|mdm_mask|><LOC_END><|eot_id|>"

            def _build_edit_prompt(ctx):
                from data_utils import EDIT_PROMPT
                edit_prompt = f"{EDIT_PROMPT} {ctx}"
                return edit_prompt

            def _parse_bbox_or_full(text, size_hw):
                w, h = size_hw
                loc_vals = re.findall(r"<LOC_([0-9]+)>", text or "")
                vals = None
                if len(loc_vals) >= 4:
                    vals = [float(v) for v in loc_vals[:4]]
                else:
                    num_vals = re.findall(r"-?\d+(?:\.\d+)?", text or "")
                    if len(num_vals) >= 4:
                        vals = [float(v) for v in num_vals[:4]]

                if vals is None:
                    return (0.0, 0.0, float(w), float(h))

                x0, y0, x1, y1 = vals
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                x0 = min(max(x0, 0.0), float(w))
                y0 = min(max(y0, 0.0), float(h))
                x1 = min(max(x1, 0.0), float(w))
                y1 = min(max(y1, 0.0), float(h))
                if x1 <= x0:
                    x1 = min(float(w), x0 + 1.0)
                if y1 <= y0:
                    y1 = min(float(h), y0 + 1.0)
                return (x0, y0, x1, y1)

            
            t_prompt0 = time.time()
            prompt_texts = [_build_text_prompt(ctx, len(images)) for ctx, images in zip(batched_contexts, batch_pil_images)]
            grounding_prompts = [_build_grounding_prompt(ctx, 1) for ctx, images in zip(batched_contexts, batch_pil_images)]
            edit_prompts = [_build_edit_prompt(ctx) for ctx in batched_contexts]
            t_prompt1 = time.time()

            t_proc0 = time.time()
            batch_inputs = self.processing_class(
                texts=prompt_texts,
                grounding_texts=grounding_prompts,
                edit_texts=edit_prompts,
                images=batch_pil_images,
                edit_mode=0,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                device=device,
                dtype=torch.bfloat16,
                mode=self.chat_mode,
                do_cfg=False,
            )
            t_proc1 = time.time()

            # Forward img_gen_* params as kwargs so _generate_image picks them up
            image_gen_kwargs = {
                "guidance_scale": 0,
                "guidance_scale_image": 0,
                "edit_mode": 0,
                "confidence_policy": "stratified",
                "enable_stratified": True,
                "image_resolution": 1024,
                "n_tokens": 4096,
                "n_steps": 64,
                "shift": 5,
                "schedule": "shift",
                "alg_temp": 5,
                "dynamic_temperature": True,
                "temperature": 0.8,
                "schedule_temp": "cosine2",
                "min_temperature": 0.5,
            }
            # Create a collage of all images in the list "images", then pad and resize to square
            # Each element in batch_pil_images is a list of images for that sample
            # init_images = []
            # for images in batch_pil_images:
            #     single_img = images[0].convert("RGB")
            #     padded = pad_to_square_and_resize(single_img, 1024)
            #     init_images.append(padded)

            num_blocks = gen_kwargs["max_new_tokens"] // gen_kwargs["block_length"]
            # In image_gen, image and text rollouts run as separate phases so they can
            # be batched differently: image uses the outer chunk (batch_size), text
            # scales inversely with max_new_tokens.
            text_batch_size = max(1, (2 ** 15) // gen_kwargs["max_new_tokens"])
            image_batch_size = batch_size
            t_gen0 = time.time()
            gen_result = self.inferencer._generate_mode(
                gen_type=self.chat_mode,
                tokenizer=self.processing_class.tokenizer,
                steps=int(gen_kwargs["step_per_block"] * num_blocks),
                gen_length=gen_kwargs["max_new_tokens"],
                block_length=gen_kwargs["block_length"],
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=mask_id,
                generation_batch_size=batch_size,
                image_batch_size=image_batch_size,
                text_batch_size=text_batch_size,
                image_gen_kwargs=image_gen_kwargs,
                processing_class=self.processing_class,
                device=device,
                answer_prompts=prompt_texts,
                use_bbox=self.use_bbox,
                input_images=batch_pil_images,
                **batch_inputs,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=self._device)
            t_gen1 = time.time()
            completion_ids = gen_result["completion_ids"]
            text_outputs = [
                txt.lstrip("!").strip()
                for txt in self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            ]
            generated_images_list = gen_result.get("edited_images", [None] * batch_size)

            t1 = time.time()
            delta_t += t1 - t0
            num_generated += batch_size
            chunk_total = t1 - t0
            print(f"Avg Latency (of {num_generated}): {delta_t/num_generated}")
            if LOG_BATCH_TIMING:
                eval_logger.info(
                    f"[stage_timing] rank={self.rank} bs={batch_size} "
                    f"prompt_build={t_prompt1 - t_prompt0:.3f}s "
                    f"processing={t_proc1 - t_proc0:.3f}s "
                    f"generate_mode={t_gen1 - t_gen0:.3f}s "
                    f"chunk_total={chunk_total:.3f}s "
                    f"per_sample={chunk_total / max(batch_size, 1):.3f}s"
                )

            # --- Save generated images and update doc metadata ---
            img_save_paths = [None] * batch_size
            if needs_image_gen and generated_images_list:
                os.makedirs(self.gen_img_dir, exist_ok=True)
                for b_idx, gen_img in enumerate(generated_images_list):
                    if gen_img is None:
                        continue
                    task_name = batched_task[b_idx]
                    split_name = batched_split[b_idx]
                    doc_id = batched_doc_id[b_idx]
                    img_save_path = os.path.join(self.gen_img_dir, f"{task_name}_{doc_id}.png")
                    gen_img.save(img_save_path)
                    img_save_paths[b_idx] = img_save_path
                    self.task_dict[task_name][split_name][doc_id]["gen_img_path"] = img_save_path

            if needs_image_gen:
                for b_idx in range(len(text_outputs)):
                    task_name = batched_task[b_idx]
                    split_name = batched_split[b_idx]
                    doc_id = batched_doc_id[b_idx]
                    doc = self.task_dict[task_name][split_name][doc_id]
                    output = {
                        "image_gen_input": edit_prompts[b_idx],
                        "text_gen_input": prompt_texts[b_idx],
                        "text_gen_output": text_outputs[b_idx],
                        "image_gen_output_path": img_save_paths[b_idx],
                    }
                    res.append(output)
            else:
                res.extend(text_outputs)

            for b_ctx, b_output in zip(batched_contexts, text_outputs):
                self.cache_hook.add_partial("generate_until", (b_ctx, gen_kwargs), b_output)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LlavaLLaDA")
