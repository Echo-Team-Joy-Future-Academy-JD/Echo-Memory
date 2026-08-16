"""ComfyUI custom nodes for Echo-Memory (Wan 2.1 1.3B memory rows)."""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

_COMFY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_COMFY_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.unified_inference import ALL_MEMORY_TYPES

_PIPE_CACHE: dict = {}

_ACTION_PRESETS = {
    "left_45": os.path.join(_REPO_ROOT, "env", "action_rotation_left_45.json"),
    "right_45": os.path.join(_REPO_ROOT, "env", "action_rotation_right_45.json"),
}

_MEMORY_CHOICES = [name for name in ALL_MEMORY_TYPES if name != "auto"] or ["context_k1"]


def _tensor_to_pil(image):
    import numpy as np
    from PIL import Image

    tensor = image[0] if getattr(image, "dim", lambda: 3)() == 4 else image
    array = (tensor.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    return Image.fromarray(array, mode="RGB")


def _frames_to_tensor(frames, width, height):
    import numpy as np
    import torch
    from env.run_replay_loop_two_chunk import _frame_to_pil

    images = [_frame_to_pil(frame, width, height) for frame in frames]
    stacked = np.stack(
        [np.asarray(image, dtype=np.float32) / 255.0 for image in images],
        axis=0,
    )
    return torch.from_numpy(stacked)


def _cached_bundle(base_model, ckpt, memory_type, tokenizer_path):
    key = (
        os.path.abspath(base_model),
        os.path.abspath(ckpt),
        memory_type,
        tokenizer_path or "",
    )
    cached = _PIPE_CACHE.get(key)
    if cached is not None:
        return cached

    from inference.unified_inference import apply_profile_to_pipe, resolve_memory_profile
    from env.loop_utils import load_pipeline_and_ckpt

    dit_path = os.path.join(base_model, "diffusion_pytorch_model.safetensors")
    text_encoder_path = os.path.join(base_model, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = os.path.join(base_model, "Wan2.1_VAE.pth")
    tokenizer = tokenizer_path or os.path.join(base_model, "google", "umt5-xxl")
    if not os.path.isdir(tokenizer):
        tokenizer = None

    for path in (dit_path, text_encoder_path, vae_path, ckpt):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Echo-Memory path not found: {path}")

    pipe = load_pipeline_and_ckpt(
        ckpt_path=ckpt,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        vae_path=vae_path,
        device="cuda",
        add_action_attn=False,
        action_use_temporal_attention=True,
        tokenizer_path=tokenizer,
    )
    profile = resolve_memory_profile(memory_type, ckpt)
    apply_profile_to_pipe(pipe, profile)
    bundle = {"pipe": pipe, "profile": profile, "memory_type": memory_type}
    _PIPE_CACHE[key] = bundle
    return bundle


class EchoMemoryLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wan_base_model": ("STRING", {"default": "", "multiline": False}),
                "ckpt": ("STRING", {"default": "", "multiline": False}),
                "memory_type": (_MEMORY_CHOICES, {"default": "context_k1"}),
            },
            "optional": {
                "tokenizer_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("ECHO_MEMORY_PIPE",)
    RETURN_NAMES = ("echo_pipe",)
    FUNCTION = "load"
    CATEGORY = "Echo-Memory"

    def load(self, wan_base_model, ckpt, memory_type, tokenizer_path=""):
        if not wan_base_model or not ckpt:
            raise ValueError("Set wan_base_model (Wan2.1-T2V-1.3B dir) and ckpt.")
        return (_cached_bundle(wan_base_model, ckpt, memory_type, tokenizer_path),)


class EchoMemoryCameraAction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (["left_45", "right_45", "custom_path"], {"default": "left_45"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("ECHO_MEMORY_ACTION",)
    RETURN_NAMES = ("action_path",)
    FUNCTION = "resolve"
    CATEGORY = "Echo-Memory"

    def resolve(self, preset, custom_path=""):
        if preset == "custom_path":
            if not custom_path or not os.path.isfile(custom_path):
                raise FileNotFoundError(
                    "preset=custom_path requires an existing 81-frame action JSON."
                )
            return (os.path.abspath(custom_path),)
        path = _ACTION_PRESETS[preset]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Bundled action JSON missing: {path}")
        return (path,)


class EchoMemoryGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "echo_pipe": ("ECHO_MEMORY_PIPE",),
                "first_frame": ("IMAGE",),
                "action_path": ("ECHO_MEMORY_ACTION",),
                "prompt": (
                    "STRING",
                    {
                        "default": "A toy bear on a table, the camera rotates around it",
                        "multiline": True,
                    },
                ),
                "num_chunks": ("INT", {"default": 1, "min": 1, "max": 8}),
                "num_frames": ("INT", {"default": 81, "min": 5, "max": 81}),
                "height": ("INT", {"default": 352, "min": 64, "max": 720, "step": 8}),
                "width": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "sigma_shift": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 30.0}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "Echo-Memory"

    def generate(
        self,
        echo_pipe,
        first_frame,
        action_path,
        prompt,
        num_chunks,
        num_frames,
        height,
        width,
        seed,
        num_inference_steps,
        sigma_shift,
        cfg_scale,
        negative_prompt="",
    ):
        from inference.unified_inference import run_unified_inference

        first = _tensor_to_pil(first_frame).resize((width, height))
        with tempfile.TemporaryDirectory(prefix="echo-memory-comfy-") as tmp:
            image_path = os.path.join(tmp, "first_frame.png")
            first.save(image_path)
            args = SimpleNamespace(
                ckpt="",
                prompt=prompt,
                output_path=None,
                memory_type=echo_pipe["memory_type"],
                base_model="",
                tokenizer_path=None,
                context_image=image_path,
                geometry_memory_video=None,
                action_path=action_path,
                height=height,
                width=width,
                num_frames=num_frames,
                num_chunks=num_chunks,
                seed=seed,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt or None,
                fps=15,
                pipe=echo_pipe["pipe"],
                profile=echo_pipe["profile"],
            )
            frames = run_unified_inference(args)
        return (_frames_to_tensor(frames, width, height),)


NODE_CLASS_MAPPINGS = {
    "EchoMemoryLoader": EchoMemoryLoader,
    "EchoMemoryCameraAction": EchoMemoryCameraAction,
    "EchoMemoryGenerate": EchoMemoryGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EchoMemoryLoader": "Echo-Memory Loader",
    "EchoMemoryCameraAction": "Echo-Memory Camera Action",
    "EchoMemoryGenerate": "Echo-Memory Generate",
}
