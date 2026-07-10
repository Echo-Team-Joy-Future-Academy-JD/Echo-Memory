import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader
from ..models.memory.framepack_length import (
    framepack_align_context_actions_to_latents,
    framepack_length_compress_context_latents,
)
from ..models.memory.framepack_weight import apply_framepack_token_weights
from ..models.memory.spatial_grid_memory import (
    apply_spatial_cross_attn_readout,
    inject_spatial_memory,
)


def pad_actions_for_condition(actions, condition_length, condition_actions=None):
    """Pad actions list to account for appended condition latents (VWM-style).

    When condition latents are concatenated along the time axis, the actions
    sequence must be extended by ``condition_length`` entries so that the
    frame count matches the total latent length (target + condition).
    """
    if actions is None or condition_length <= 0:
        return actions
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().tolist()
    elif isinstance(actions, np.ndarray):
        actions = actions.tolist()
    else:
        actions = list(actions)
    if actions and isinstance(actions[0], (list, tuple)) and actions[0] and isinstance(actions[0][0], (list, tuple)):
        actions = actions[0]
    if len(actions) == 0:
        return actions

    appended_actions = []
    if condition_actions is not None:
        if isinstance(condition_actions, torch.Tensor):
            condition_actions = condition_actions.detach().cpu().tolist()
        elif isinstance(condition_actions, np.ndarray):
            condition_actions = condition_actions.tolist()
        else:
            condition_actions = list(condition_actions)
        if (
            condition_actions
            and isinstance(condition_actions[0], (list, tuple))
            and condition_actions[0]
            and isinstance(condition_actions[0][0], (list, tuple))
        ):
            condition_actions = condition_actions[0]
        appended_actions = [list(action) for action in condition_actions[:condition_length]]

    zero_action = [0.0] * len(actions[0])
    while len(appended_actions) < condition_length:
        appended_actions.append(zero_action[:])
    return actions + appended_actions


class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self


    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames


    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video


    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video


    def load_models_to_device(self, model_names=[]):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)


    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise


    def enable_cpu_offload(self):
        warnings.warn("`enable_cpu_offload` will be deprecated. Please use `enable_vram_management`.")
        self.vram_management_enabled = True
        
        
    def get_vram(self):
        return torch.cuda.mem_get_info(self.device if (isinstance(self.device, torch.device) and self.device.index is not None) else 0)[1] / (1024 ** 3)
    
    
    def freeze_except(self, model_names):
        for name, model in self.named_children():
            if name in model_names:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    skip_download: bool = False

    def download_if_necessary(self, local_model_path="./models", skip_download=False, use_usp=False):
        if self.path is None:
            # Check model_id and origin_file_pattern
            if self.model_id is None:
                raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.""")
            
            # Skip if not in rank 0
            if use_usp:
                import torch.distributed as dist
                skip_download = dist.get_rank() != 0
                
            # Check whether the origin path is a folder
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                allow_file_pattern = None
                is_folder = True
            elif isinstance(self.origin_file_pattern, str) and self.origin_file_pattern.endswith("/"):
                allow_file_pattern = self.origin_file_pattern + "*"
                is_folder = True
            else:
                allow_file_pattern = self.origin_file_pattern
                is_folder = False
            
            # Download
            skip_download = skip_download or self.skip_download
            if not skip_download:
                downloaded_files = glob.glob(self.origin_file_pattern, root_dir=os.path.join(local_model_path, self.model_id))
                snapshot_download(
                    self.model_id,
                    local_dir=os.path.join(local_model_path, self.model_id),
                    allow_file_pattern=allow_file_pattern,
                    ignore_file_pattern=downloaded_files,
                    local_files_only=False
                )
            
            # Let rank 1, 2, ... wait for rank 0
            if use_usp:
                import torch.distributed as dist
                dist.barrier(device_ids=[dist.get_rank()])
                
            # Return downloaded files
            if is_folder:
                self.path = os.path.join(local_model_path, self.model_id, self.origin_file_pattern)
            else:
                self.path = glob.glob(os.path.join(local_model_path, self.model_id, self.origin_file_pattern))
            if isinstance(self.path, list) and len(self.path) == 1:
                self.path = self.path[0]


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedder(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video
        
    
    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

        
    def training_loss(self, **inputs):
        # Use len(self.scheduler.timesteps) instead of num_train_timesteps to avoid index out of bounds
        # The timesteps array may have fewer elements than num_train_timesteps
        num_timesteps = len(self.scheduler.timesteps) if hasattr(self.scheduler, 'timesteps') and self.scheduler.timesteps is not None else self.scheduler.num_train_timesteps
        timestep_id = torch.randint(0, num_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        # Context Memory mode: handle context latents separately
        context_latents = inputs.get("context_latents", None)
        num_context_frames = inputs.get("num_context_frames", 0)
        training_mode = inputs.get("training_mode", "context")  # "predict", "context", or "condition"
        # Experiment 1_2: allow context to be concatenated at the END (suffix) instead of the beginning (prefix)
        import os
        context_position = inputs.get("context_position", os.environ.get("CONTEXT_POSITION", "prefix"))
        if context_position not in ("prefix", "suffix"):
            context_position = "prefix"
        
        # Optional: Frame-level FramePack compression on context latents (K -> K')
        # Use pipeline attributes as the primary source (so inference/multichunk can align),
        # but also allow per-call override via inputs.
        use_framepack_len = bool(
            inputs.get("use_framepack_length_compress", False) or getattr(self, "use_framepack_length_compress", False)
        )
        framepack_ratio = int(
            inputs.get("framepack_ratio", getattr(self, "framepack_ratio", 1)) or 1
        )
        framepack_strategy = str(
            inputs.get("framepack_length_strategy", getattr(self, "framepack_length_strategy", "distance_merge")) or "distance_merge"
        ).lower()
        framepack_recent_keep_ratio = float(
            inputs.get("framepack_recent_keep_ratio", getattr(self, "framepack_recent_keep_ratio", 0.5)) or 0.5
        )
        framepack_multiscale_w2 = float(
            inputs.get("framepack_multiscale_w2", getattr(self, "framepack_multiscale_w2", 0.25)) or 0.25
        )
        framepack_multiscale_w4 = float(
            inputs.get("framepack_multiscale_w4", getattr(self, "framepack_multiscale_w4", 0.15)) or 0.15
        )
        if context_latents is not None and num_context_frames > 0 and use_framepack_len and framepack_ratio > 1:
            # context_latents: (B, C, K, H, W) -> (B, C, K', H, W); non-overlapping r-frames mean pool
            # Pad: replicate last latent until K divisible by r; context_actions get same pad + same mean (train = infer)
            r = int(framepack_ratio)
            Kc0 = int(context_latents.shape[2])
            pad0 = (r - (Kc0 % r)) % r
            K_after_pad = Kc0 + pad0
            if inputs.get("context_actions") is not None:
                inputs["context_actions"] = framepack_align_context_actions_to_latents(
                    inputs["context_actions"],
                    K_orig_latent=Kc0,
                    K_after_pad=K_after_pad,
                    framepack_ratio=r,
                    device=context_latents.device,
                    dtype=context_latents.dtype,
                    strategy=framepack_strategy,
                    recent_keep_ratio=framepack_recent_keep_ratio,
                )
            context_latents, new_k, _, _ = framepack_length_compress_context_latents(
                context_latents,
                r,
                strategy=framepack_strategy,
                recent_keep_ratio=framepack_recent_keep_ratio,
                multiscale_w2=framepack_multiscale_w2,
                multiscale_w4=framepack_multiscale_w4,
            )
            num_context_frames = int(new_k)
            inputs["context_latents"] = context_latents
            inputs["num_context_frames"] = num_context_frames
        
        # Context Noise Augmentation: add noise to context latents during training
        # Experiment 7: Align training-inference noise distribution
        # Use fixed small noise (sigma=0.1) to match inference strategy (Zero-shot ICL standard trick)
        context_noise_prob = inputs.get("context_noise_prob", 0.0)  # Probability of adding noise (legacy, for backward compatibility)
        context_noise_std = inputs.get("context_noise_std", 0.02)  # Noise standard deviation (legacy)
        context_fixed_noise_std = inputs.get("context_fixed_noise_std", None)  # Fixed noise std (Experiment 7: sigma=0.1)
        
        if context_latents is not None and num_context_frames > 0:
            if training_mode == "condition":
                # Condition mode (Experiment 17): Full video prediction with condition frames
                # Formula: (N-1)/4+1 where N is number of frames
                # - Condition: First 5 frames → 2 Latent Tokens ((5-1)/4+1 = 2)
                # - Target: Full 81 frames → 21 Latent Tokens ((81-1)/4+1 = 21)
                # - Concatenation: [Condition (2 tokens), Target (21 tokens)] = 23 tokens total
                # - Loss: Compute on Target (21 tokens), Condition (2 tokens) is used as condition only
                
                # input_latents shape: (B, C, 21, H, W) - 81 frames encoded as 21 latent tokens
                input_latents = inputs["input_latents"]
                noise = inputs["noise"]
                
                # Condition latents: 5 frames encoded as 2 latent tokens (clean, from context_latents parameter)
                # Shape: (B, C, 2, H, W) where 2 is the number of latent tokens
                condition_frames = context_latents  # Already clean, no noise added
                
                # Get actual dimensions (in latent tokens, not frames)
                condition_latent_tokens = condition_frames.shape[2] if len(condition_frames.shape) > 2 else num_context_frames
                target_latent_tokens = input_latents.shape[2] if len(input_latents.shape) > 2 else 21
                
                # Validate dimensions match expected values (in latent tokens)
                # Formula: (N-1)/4+1 where N is number of frames
                # Expected: 5 frames → 2 latent tokens ((5-1)/4+1 = 2)
                expected_condition_tokens = 2
                if condition_latent_tokens != expected_condition_tokens:
                    import warnings
                    warnings.warn(
                        f"Condition latent tokens mismatch: expected {expected_condition_tokens} "
                        f"(from 5 frames), got {condition_latent_tokens}. Using actual value."
                    )
                    num_context_frames = condition_latent_tokens
                else:
                    num_context_frames = expected_condition_tokens
                
                # Target latents: 81 frames encoded as 21 latent tokens (will be noised)
                # Shape: (B, C, 21, H, W)
                # Formula: (81-1)/4+1 = 21 latent tokens
                target_latents = input_latents  # 81 frames → 21 latent tokens
                target_noise = noise  # Noise for 21 latent tokens
                
                # Ensure noise matches target_latents shape (in latent tokens)
                # Formula: (N-1)/4+1 where N is number of frames
                expected_target_tokens = 21  # (81-1)/4+1 = 21 tokens
                if target_noise.shape[2] != target_latent_tokens:
                    import warnings
                    warnings.warn(
                        f"Noise latent tokens mismatch: expected {target_latent_tokens} "
                        f"(from 81 frames), got {target_noise.shape[2]}. Truncating or padding noise."
                    )
                    if target_noise.shape[2] > target_latent_tokens:
                        target_noise = target_noise[:, :, :target_latent_tokens, :, :]
                    else:
                        # Pad noise if needed (shouldn't happen normally)
                        padding = target_latent_tokens - target_noise.shape[2]
                        target_noise = torch.cat([
                            target_noise,
                            torch.zeros_like(target_noise[:, :, :padding, :, :])
                        ], dim=2)
                
                # Add noise to target latents (81 frames)
                noisy_target_latents = self.scheduler.add_noise(target_latents, target_noise, timestep)
                training_target = self.scheduler.training_target(target_latents, target_noise, timestep)
            
                # Experiment 7: Fixed noise strategy (align with inference)
                # If context_fixed_noise_std is set, always add fixed noise (matching inference)
                # Otherwise, use legacy probabilistic noise augmentation
                if context_fixed_noise_std is not None and context_fixed_noise_std > 0.0:
                    # Always add fixed small noise to condition (matching inference strategy)
                    # This ensures training-inference distribution alignment
                    condition_noise = torch.randn_like(condition_frames) * context_fixed_noise_std
                    condition_frames = condition_frames + condition_noise
                elif context_noise_prob > 0.0 and torch.rand(1).item() < context_noise_prob:
                    # Legacy: Randomly add noise to condition latents (probabilistic augmentation)
                    condition_noise = torch.randn_like(condition_frames) * context_noise_std
                    condition_frames = condition_frames + condition_noise
                
                # Concatenate clean condition latents with noisy target latents
                # condition_frames: (B, C, 2, H, W), noisy_target_latents: (B, C, 21, H, W)
                # Result: (B, C, 2+21, H, W) = (B, C, 23, H, W)
                # Ensure dimensions match before concatenation (except in dimension 2 - latent tokens)
                if condition_frames.shape[:2] != noisy_target_latents.shape[:2] or \
                   condition_frames.shape[3:] != noisy_target_latents.shape[3:]:
                    raise RuntimeError(
                        f"Dimension mismatch when concatenating condition and target latents:\n"
                        f"  condition_frames shape: {condition_frames.shape}\n"
                        f"  noisy_target_latents shape: {noisy_target_latents.shape}\n"
                        f"  Expected same shape except in dimension 2 (latent tokens).\n"
                        f"  Condition tokens: {condition_latent_tokens}, Target tokens: {target_latent_tokens}"
                    )
                
                # Experiment 1_2: if suffix => [Target, Context], else [Context, Target]
                if context_position == "suffix":
                    latents = torch.cat([noisy_target_latents, condition_frames], dim=2)
                else:
                    latents = torch.cat([condition_frames, noisy_target_latents], dim=2)
                
                # Store concatenated latents and condition info for model forward
                inputs["latents"] = latents
                inputs["num_context_frames"] = num_context_frames
                if "actions" in inputs and inputs["actions"] is not None:
                    inputs["actions"] = pad_actions_for_condition(
                        inputs["actions"],
                        num_context_frames,
                        condition_actions=inputs.get("context_actions"),
                    )
                
                noise_pred = self.model_fn(**inputs, timestep=timestep)
                
                if context_position == "suffix":
                    target_noise_pred = noise_pred[:, :, :target_latent_tokens, :, :]
                else:
                    target_noise_pred = noise_pred[:, :, num_context_frames:, :, :]
                
                loss = torch.nn.functional.mse_loss(target_noise_pred.float(), training_target.float())
                loss = loss * self.scheduler.training_weight(timestep)
            else:  # training_mode == "context" (default, Experiment 16)
                # Context as Memory mode (Experiment 16): Inpainting mode
                # - Context: First K frames (4 frames, 1 Latent Token) - clean, no noise
                # - Target: Last 77 frames (20 Latent Tokens, num_frames % 4 == 1) - will be noised
                # - Concatenation: [Context (K), Target (77)] = 81 frames total (21 Latent Tokens)
                # - Loss: Only compute on Target (77 frames, 20 Latent Tokens), Context is masked out
                
                # input_latents shape: (B, C, 77, H, W) - target frames only (last 77 frames)
                input_latents = inputs["input_latents"]
                noise = inputs["noise"]
                
                # Context latents: First K frames (clean, from context_latents parameter)
                # Shape: (B, C, K, H, W) where K=4 frames
                context_frames = context_latents  # Already clean, no noise added
                
                # Target latents: Last 77 frames (will be noised)
                # Shape: (B, C, 77, H, W)
                target_latents = input_latents  # Last 77 frames
                target_noise = noise  # Noise for 77 frames (matches num_frames % 4 == 1 requirement)
                
                # Add noise to target latents (77 frames)
                noisy_target_latents = self.scheduler.add_noise(target_latents, target_noise, timestep)
                training_target = self.scheduler.training_target(target_latents, target_noise, timestep)
                
                # Experiment 7: Fixed noise strategy (align with inference)
                # If context_fixed_noise_std is set, always add fixed noise (matching inference)
                # Otherwise, use legacy probabilistic noise augmentation
                if context_fixed_noise_std is not None and context_fixed_noise_std > 0.0:
                    # Always add fixed small noise to context (matching inference strategy)
                    # This ensures training-inference distribution alignment
                    context_noise = torch.randn_like(context_frames) * context_fixed_noise_std
                    context_frames = context_frames + context_noise
                elif context_noise_prob > 0.0 and torch.rand(1).item() < context_noise_prob:
                    # Legacy: Randomly add noise to context latents (probabilistic augmentation)
                    context_noise = torch.randn_like(context_frames) * context_noise_std
                    context_frames = context_frames + context_noise
                
                # Concatenate clean context latents with noisy target latents
                # context_frames: (B, C, K, H, W), noisy_target_latents: (B, C, 77, H, W)
                # Result: (B, C, K+77, H, W) = (B, C, 81, H, W)
                
                # Debug: Check dimensions before concatenation
                context_frames_dim = context_frames.shape[2] if len(context_frames.shape) > 2 else 0
                target_latents_dim = noisy_target_latents.shape[2] if len(noisy_target_latents.shape) > 2 else 0
                
                # Ensure dimensions match (except in dimension 2)
                if context_frames.shape[:2] != noisy_target_latents.shape[:2] or \
                   context_frames.shape[3:] != noisy_target_latents.shape[3:]:
                    raise RuntimeError(
                        f"Dimension mismatch when concatenating context and target latents:\n"
                        f"  context_frames shape: {context_frames.shape}\n"
                        f"  noisy_target_latents shape: {noisy_target_latents.shape}\n"
                        f"  Expected same shape except in dimension 2 (frames).\n"
                        f"  Context frames: {context_frames_dim}, Target frames: {target_latents_dim}"
                    )
                
                # Experiment 1_2: if suffix => [Target, Context], else [Context, Target]
                if context_position == "suffix":
                    latents = torch.cat([noisy_target_latents, context_frames], dim=2)
                else:
                    latents = torch.cat([context_frames, noisy_target_latents], dim=2)
                
                # Store concatenated latents and context info for model forward
                inputs["latents"] = latents
                inputs["num_context_frames"] = num_context_frames
                if "actions" in inputs and inputs["actions"] is not None:
                    inputs["actions"] = pad_actions_for_condition(
                        inputs["actions"],
                        num_context_frames,
                        condition_actions=inputs.get("context_actions"),
                    )
                
                noise_pred = self.model_fn(**inputs, timestep=timestep)
                
                if context_position == "suffix":
                    target_noise_pred = noise_pred[:, :, :target_latents.shape[2], :, :]
                else:
                    target_noise_pred = noise_pred[:, :, num_context_frames:, :, :]
                
                loss = torch.nn.functional.mse_loss(target_noise_pred.float(), training_target.float())
                loss = loss * self.scheduler.training_weight(timestep)
        else:
            # Standard mode: add noise to all latents
            inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
            training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
            
            noise_pred = self.model_fn(**inputs, timestep=timestep)
            
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
            loss = loss * self.scheduler.training_weight(timestep)
        
        return loss


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        local_model_path: str = "./models",
        skip_download: bool = False,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(local_model_path, skip_download=skip_download, use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        pipe.dit = model_manager.fetch_model("wan_video_dit")
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary(local_model_path, skip_download=skip_download)
            pipe.prompter.fetch_models(pipe.text_encoder)
            pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        else:
            pipe.prompter.fetch_models(pipe.text_encoder)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # Explicit camera trajectory control (per-frame camera params)
        # Each entry should be a list/tuple like CameraCtrl format (len>=19):
        #   [frame_id, fx, fy, cx, cy, *, *, w2c(12 vals)]
        camera_control_poses: Optional[list] = None,
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        action_path=None,
        # Context Memory parameters
        context_latents: Optional[torch.Tensor] = None,
        num_context_frames: int = 0,
        enable_context_memory: bool = False,
        cfg_target_only: bool = False,  # Only apply CFG to target frames (not context)
        inference_noise_level: Optional[float] = None,  # Experiment 19: Add small noise to context latents during inference for distribution alignment
        context_actions: Optional[torch.Tensor] = None,  # RT poses for context frames [N_ctx, 12]
        context_position: Optional[str] = None,  # "suffix" = context at end (exp1_4_2), "prefix" = context at start; default from env CONTEXT_POSITION
        cam_pose_actions=None,  # VWM-style: list of 21 relative RT vectors (12 floats each), already latent-frame-aligned
        geometry_memory_latents: Optional[torch.Tensor] = None,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "camera_control_poses": camera_control_poses,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "geometry_memory_latents": geometry_memory_latents,
        }
        if geometry_memory_latents is not None:
            inputs_shared["use_geometry_spatial_memory"] = bool(
                getattr(self, "use_geometry_spatial_memory", False)
            )
            inputs_shared["geometry_spatial_memory_module"] = getattr(
                self,
                "geometry_spatial_memory_module",
                None,
            )
            inputs_shared["geometry_spatial_memory_inject_mode"] = getattr(
                self,
                "geometry_spatial_memory_inject_mode",
                "concat_text",
            )
            inputs_shared["geometry_spatial_memory_readout_module"] = getattr(
                self,
                "geometry_spatial_memory_readout_module",
                None,
            )
        
        if action_path is not None:
            import json
            with open(action_path, "r") as f:
                data = json.load(f)
            action_seq = data.get("actions", data)
            if isinstance(action_seq, dict):
                sorted_items = sorted(action_seq.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else float('inf'))
            else:
                sorted_items = list(enumerate(action_seq))
            first_val = sorted_items[0][1] if sorted_items else None
            use_rt_format = isinstance(first_val, (list, tuple)) and len(first_val) == 12
            actions = []
            for _, val in sorted_items:
                if use_rt_format:
                    actions.append(list(val) if isinstance(val, (list, tuple)) else val)
                else:
                    action_dict = val
                    action = [0] * (2 + 2 + 3 + 1 + 2)
                    if action_dict.get('ws') == 1:
                        action[0] = 1
                    elif action_dict.get('ws') == 2:
                        action[1] = 1
                    if action_dict.get('ad') == 1:
                        action[2] = 1
                    elif action_dict.get('ad') == 2:
                        action[3] = 1
                    if action_dict.get('scs') == 1 and action_dict.get('jump_invalid', 0) == 0:
                        action[4] = 1
                    elif action_dict.get('scs') == 2:
                        action[5] = 1
                    elif action_dict.get('scs') == 3:
                        action[6] = 1
                    if action_dict.get("collision") == 1:
                        action[7] = 1
                        action[0] = action[1] = action[2] = action[3] = 0
                    action[8] = action_dict.get('pre_pitch', 0)
                    action[9] = action_dict.get('pre_yaw', 0)
                    actions.append(action)
            if use_rt_format:
                # 12-dim RT actions: align with training (CamVideoDataset uses
                # pose_indices = list(range(0, num_frames, 4)) to pick one RT per
                # latent frame, since the Wan VAE compresses temporally by 4×).
                # The sampling JSON written by _build_gt_action_json contains one RT
                # per pixel frame (e.g. 81 entries for num_frames=81), so we must
                # subsample to (num_frames - 1) // 4 + 1 latent-frame entries here.
                # Otherwise DiTBlock_w_Action will try to reshape spatial tokens
                # by a frame count that does not divide them, and crash with
                # "shape '[1, 81, -1, 1536]' is invalid for input of size ...".
                _stride = 4
                _n_pixel = int(num_frames) if num_frames else len(actions)
                _n_target_latent = (_n_pixel - 1) // _stride + 1 if _n_pixel > 0 else len(actions)
                if len(actions) >= _n_pixel and _n_target_latent > 0:
                    actions = [actions[i * _stride] for i in range(_n_target_latent)]
                elif len(actions) > _n_target_latent:
                    # JSON is shorter than _n_pixel but still longer than _n_target_latent.
                    # Re-stride uniformly across what we have to land on _n_target_latent samples.
                    step = max(1, len(actions) // _n_target_latent)
                    actions = [actions[min(i * step, len(actions) - 1)] for i in range(_n_target_latent)]
                elif 0 < len(actions) < _n_target_latent:
                    # Pad by repeating the last RT to keep the expected latent length.
                    actions = actions + [list(actions[-1])] * (_n_target_latent - len(actions))
            else:
                # Legacy non-RT (binary action) format: keep original truncation.
                actions = actions[:80]
            inputs_shared["actions"] = actions
        elif cam_pose_actions is not None:
            inputs_shared["actions"] = cam_pose_actions

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        
        # Context Memory mode: prepare context latents
        # Concatenation (default): [target(noise), context(clean)] 与训练一致，衔接正确；Repainting 仅保留兼容
        use_concatenation_inference = os.environ.get("USE_CONCATENATION_INFERENCE", "true").lower() == "true"
        # Experiment 1_2: optionally place context at END (suffix) instead of BEGIN (prefix); prefer explicit arg over env
        _context_position = (context_position or os.environ.get("CONTEXT_POSITION", "prefix")).lower()
        if _context_position not in ("prefix", "suffix"):
            _context_position = "prefix"
        context_position = _context_position
        
        if enable_context_memory and context_latents is not None and num_context_frames > 0:
            # NOTE: num_context_frames is treated as *latent token count* in this codebase.
            # Callers must provide context_latents already encoded by VAE (token-level).
            context_latents_clean = context_latents.to(dtype=self.torch_dtype, device=self.device)  # (B, C, K, H, W)

            # Experiment 19: Add small noise to context latents during inference for distribution alignment
            # This aligns inference distribution with training (where context has noise)
            # Prevents OOD issues and breaks "perfect match" to increase dynamicity
            if inference_noise_level is None:
                # Default: No noise (0.0) unless explicitly set
                # For Experiment 19, training script will automatically set inference_noise_level from context_fixed_noise_std
                inference_noise_level = 0.0

            if inference_noise_level > 0.0:
                # Add small Gaussian noise to context latents (only to context, not to main latents)
                # This noise is added before concatenation, so it's part of the condition input
                context_noise = torch.randn_like(context_latents_clean) * inference_noise_level
                context_latents_clean = context_latents_clean + context_noise
            
            # Context Memory: runtime memory baselines (must be wired for inference/multichunk)
            inputs_shared["use_framepack_memory"] = getattr(self, "use_framepack_memory", False)
            inputs_shared["context_temporal_decay"] = float(getattr(self, "context_temporal_decay", 1.0) or 1.0)
            inputs_shared["context_attention_weight"] = float(getattr(self, "context_attention_weight", 1.0) or 1.0)
            inputs_shared["use_spatial_memory"] = getattr(self, "use_spatial_memory", False)
            inputs_shared["spatial_memory_tokens"] = int(getattr(self, "spatial_memory_tokens", 64) or 64)
            inputs_shared["use_spatial_memory_legacy"] = bool(getattr(self, "use_spatial_memory_legacy", False))
            inputs_shared["spatial_memory_module"] = getattr(self, "spatial_memory_module", None)
            inputs_shared["spatial_memory_inject_mode"] = getattr(self, "spatial_memory_inject_mode", "concat_text")
            inputs_shared["spatial_memory_readout_module"] = getattr(self, "spatial_memory_readout_module", None)

            # Frame-level FramePack length compression for inference: same as training_loss (shared helpers)
            use_framepack_len = bool(getattr(self, "use_framepack_length_compress", False))
            framepack_ratio = int(getattr(self, "framepack_ratio", 1) or 1)
            framepack_strategy = str(getattr(self, "framepack_length_strategy", "distance_merge") or "distance_merge").lower()
            framepack_recent_keep_ratio = float(getattr(self, "framepack_recent_keep_ratio", 0.5) or 0.5)
            framepack_multiscale_w2 = float(getattr(self, "framepack_multiscale_w2", 0.25) or 0.25)
            framepack_multiscale_w4 = float(getattr(self, "framepack_multiscale_w4", 0.15) or 0.15)
            if use_framepack_len and framepack_ratio > 1 and context_latents_clean is not None and num_context_frames > 0:
                r = int(framepack_ratio)
                Kc0 = int(context_latents_clean.shape[2])
                pad0 = (r - (Kc0 % r)) % r
                K_after_pad = Kc0 + pad0
                if context_actions is not None:
                    context_actions = framepack_align_context_actions_to_latents(
                        context_actions,
                        K_orig_latent=Kc0,
                        K_after_pad=K_after_pad,
                        framepack_ratio=r,
                        device=context_latents_clean.device,
                        dtype=context_latents_clean.dtype,
                        strategy=framepack_strategy,
                        recent_keep_ratio=framepack_recent_keep_ratio,
                    )
                context_latents_clean, new_k, _, _ = framepack_length_compress_context_latents(
                    context_latents_clean,
                    r,
                    strategy=framepack_strategy,
                    recent_keep_ratio=framepack_recent_keep_ratio,
                    multiscale_w2=framepack_multiscale_w2,
                    multiscale_w4=framepack_multiscale_w4,
                )
                num_context_frames = int(new_k)

            if use_concatenation_inference:
                # Concatenation Strategy (Experiment 12/16): Direct input concatenation
                # After CM training, model can handle Clean Context + Noisy Target directly
                # Experiment 16: Context (4 frames) + Target (77 frames) = 81 frames total
                # Concatenation: [Context (4), Target (77)] = 81 frames total
                target_latents_init = inputs_shared["latents"]  # (B, C, 77, H, W) - target frames noise
                # Direct concatenation: [Clean Context (4), Random Noise (77)] OR [Random Noise (77), Clean Context (4)]
                # Model has been trained to handle this distribution (Experiment 16: 81 frames)
                # Experiment 1_2 inference: if suffix => [Target(noise), Context(clean)] (context at end)
                if context_position == "suffix":
                    latents_concatenated = torch.cat([target_latents_init, context_latents_clean], dim=2)
                else:
                    latents_concatenated = torch.cat([context_latents_clean, target_latents_init], dim=2)
                inputs_shared["latents"] = latents_concatenated
                inputs_shared["num_context_frames"] = num_context_frames
                inputs_shared["context_position"] = context_position
                if "actions" in inputs_shared and inputs_shared["actions"] is not None:
                    inputs_shared["actions"] = pad_actions_for_condition(
                        inputs_shared["actions"],
                        num_context_frames,
                        condition_actions=context_actions,
                    )
                if getattr(self, "use_moc", False) and getattr(self, "moc_module", None) is not None:
                    inputs_shared["moc_module"] = self.moc_module
                    inputs_shared["use_moc"] = True
            else:
                # Repainting Strategy (Experiment 11): Initialize all latents as noise, then repaint context in loop
                # Store clean context latents for repainting
                # Initialize all latents as random noise (including context part)
                # Experiment 1_2: layout must match training — suffix => [target, context], prefix => [context, target]
                target_latents_init = inputs_shared["latents"]  # (B, C, T, H, W)
                batch_size, channels = target_latents_init.shape[0], target_latents_init.shape[1]
                height, width = target_latents_init.shape[3], target_latents_init.shape[4]
                context_noise_init = torch.randn(
                    batch_size, channels, num_context_frames, height, width,
                    dtype=self.torch_dtype, device=self.device
                )
                if context_position == "suffix":
                    # [target, context] so that last K frames are context (chunk1 last frame etc.), first T are generated
                    latents_full = torch.cat([target_latents_init, context_noise_init], dim=2)  # (B, C, T+K, H, W)
                else:
                    latents_full = torch.cat([context_noise_init, target_latents_init], dim=2)  # (B, C, K+T, H, W)
                inputs_shared["latents"] = latents_full
        else:
            context_latents_clean = None
            use_concatenation_inference = False
        if enable_context_memory and num_context_frames > 0 and getattr(self, "use_moc", False) and getattr(self, "moc_module", None) is not None:
            inputs_shared["moc_module"] = self.moc_module
            inputs_shared["use_moc"] = True
        if enable_context_memory and num_context_frames > 0 and getattr(self, "use_unified_implicit", False) and getattr(self, "unified_implicit_encoder", None) is not None and getattr(self, "context_learning_injector", None) is not None:
            inputs_shared["use_unified_implicit"] = True
            inputs_shared["unified_implicit_encoder"] = self.unified_implicit_encoder
            inputs_shared["context_learning_injector"] = self.context_learning_injector

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Context Memory: Choose inference method
            if enable_context_memory and context_latents_clean is not None:
                if use_concatenation_inference:
                    # Concatenation Strategy (Experiment 12): No special handling needed
                    # Context was already concatenated at input stage, just pass through
                    inputs_shared["num_context_frames"] = num_context_frames
                else:
                    # Repainting Strategy (Experiment 11): Replace context part in each step
                    # --- DEBUG START: Latent Statistics Check (Experiment 11) ---
                    # Log latent statistics to file for analysis (no console print to avoid log file clutter)
                    import json
                    import time
                    debug_log_path = os.environ.get("CONTEXT_DEBUG_LOG", "logs/context_latent_stats.jsonl")
                    
                    # Get current latents (all noisy at this point, before repainting)
                    latents_current = inputs_shared["latents"]  # (B, C, K+T, H, W)
                    if context_position == "suffix":
                        target_latents_current = latents_current[:, :, :-num_context_frames, :, :]  # (B, C, T, H, W)
                    else:
                        target_latents_current = latents_current[:, :, num_context_frames:, :, :]  # (B, C, T, H, W)
                    
                    # Calculate statistics before repainting
                    target_mean = target_latents_current.mean().item()
                    target_std = target_latents_current.std().item()
                    target_min = target_latents_current.min().item()
                    target_max = target_latents_current.max().item()
                    
                    context_mean = context_latents_clean.mean().item()
                    context_std = context_latents_clean.std().item()
                    context_min = context_latents_clean.min().item()
                    context_max = context_latents_clean.max().item()
                    
                    ratio = context_std / target_std if target_std > 0 else float('inf')
                    context_mean_abs = abs(context_mean)
                    
                    # Log to file
                    debug_entry = {
                        "timestamp": time.time(),
                        "timestep_id": progress_id,
                        "total_timesteps": len(self.scheduler.timesteps),
                        "target_latents": {
                            "mean": target_mean,
                            "std": target_std,
                            "min": target_min,
                            "max": target_max
                        },
                        "context_latents_clean": {
                            "mean": context_mean,
                            "std": context_std,
                            "min": context_min,
                            "max": context_max
                        },
                        "ratio": {
                            "std_ratio": ratio,
                            "mean_abs": context_mean_abs
                        },
                        "warnings": []
                    }
                    
                    # Add warnings
                    if ratio < 0.2 or ratio > 5.0:
                        debug_entry["warnings"].append(f"Large std ratio: {ratio:.4f} (Context/Target)")
                    if context_mean_abs > 2.0:
                        debug_entry["warnings"].append(f"Context mean far from 0: {context_mean:.4f}")
                    if context_std < 0.2:
                        debug_entry["warnings"].append(f"Very small context std: {context_std:.4f}")
                    if context_std > 5.0:
                        debug_entry["warnings"].append(f"Very large context std: {context_std:.4f}")
                    
                    # Write to log file only (no console print)
                    try:
                        os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
                        with open(debug_log_path, "a") as f:
                            f.write(json.dumps(debug_entry) + "\n")
                    except Exception as e:
                        pass  # Silently fail if logging fails
                    # --- DEBUG END ---
                    
                    # Repainting: Add noise to clean context to match current timestep
                    # Generate noise for context (can be deterministic if using same seed)
                    noise_for_context = torch.randn_like(context_latents_clean)
                    
                    # Add noise to clean context using scheduler (forward diffusion)
                    # This creates noisy context at the current timestep level
                    context_latents_noisy = self.scheduler.add_noise(
                        context_latents_clean,
                        noise_for_context,
                        timestep.unsqueeze(0)
                    )
                    
                    # Hard Replacement: Replace context part in latents
                    # latents_current: (B, C, K+T, H, W)
                    # Prefix: context = first K frames; Suffix: context = last K (must match concat layout)
                    latents_repainted = latents_current.clone()
                    if context_position == "suffix":
                        latents_repainted[:, :, -num_context_frames:, :, :] = context_latents_noisy
                    else:
                        latents_repainted[:, :, :num_context_frames, :, :] = context_latents_noisy
                    
                    inputs_shared["latents"] = latents_repainted
                inputs_shared["num_context_frames"] = num_context_frames
                inputs_shared["context_position"] = context_position
            else:
                inputs_shared.pop("num_context_frames", None)  # Remove if exists
                inputs_shared.pop("context_position", None)

            # Inference
            try:
                noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            except Exception as e:
                raise
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                
                # CFG only for target frames (Experiment 7)
                if cfg_target_only and enable_context_memory and context_latents_clean is not None and num_context_frames > 0:
                    # Split context and target parts (handle suffix mode)
                    if context_position == "suffix":
                        # Experiment 1_2: target is FRONT, context is BACK
                        target_posi = noise_pred_posi[:, :, :-num_context_frames, :, :]
                        context_posi = noise_pred_posi[:, :, -num_context_frames:, :, :]
                        target_nega = noise_pred_nega[:, :, :-num_context_frames, :, :]
                        context_nega = noise_pred_nega[:, :, -num_context_frames:, :, :]
                    else:
                        # Prefix mode: context is FRONT, target is BACK
                        context_posi = noise_pred_posi[:, :, :num_context_frames, :, :]
                        target_posi = noise_pred_posi[:, :, num_context_frames:, :, :]
                        context_nega = noise_pred_nega[:, :, :num_context_frames, :, :]
                        target_nega = noise_pred_nega[:, :, num_context_frames:, :, :]
                    
                    # Apply CFG only to target part
                    target_cfg = target_nega + cfg_scale * (target_posi - target_nega)
                    
                    # Keep context part unchanged (use positive prediction, or average)
                    # Using positive prediction for context to maintain consistency
                    context_cfg = context_posi
                    
                    # Concatenate back (handle suffix mode)
                    if context_position == "suffix":
                        noise_pred = torch.cat([target_cfg, context_cfg], dim=2)
                    else:
                        noise_pred = torch.cat([context_cfg, target_cfg], dim=2)
                else:
                    # Standard CFG: apply to all frames
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler Step
            if enable_context_memory and context_latents_clean is not None:
                if use_concatenation_inference:
                    # Concatenation Strategy: Update all latents, but lock Context to prevent corruption
                    # CRITICAL FIX: Model was trained with Loss only on Target frames, so noise_pred
                    # for Context frames is undefined/random. We must lock Context after each step
                    # to prevent it from being corrupted by garbage predictions.
                    inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
                    # Lock Context: Reset Context part to original Clean Context after scheduler step
                    # This prevents Context from being corrupted by model's undefined predictions on Context frames
                    # Experiment 1_2: if suffix => lock BACK, else lock FRONT
                    if context_position == "suffix":
                        inputs_shared["latents"][:, :, -num_context_frames:, :, :] = context_latents_clean
                    else:
                        inputs_shared["latents"][:, :, :num_context_frames, :, :] = context_latents_clean
                else:
                    # Repainting Strategy: Update all latents, but context will be repainted in next step
                    inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            else:
                # Standard mode: update all latents
                inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
        
        # Final latents: Extract target part if context memory was used
        if enable_context_memory and context_latents_clean is not None:
            # After denoising, latents contain both context and target
            # Extract only target part for final output (context was only for conditioning)
            # Experiment 16: Extract full 81 frames (including reconstruction of first 4 frames)
            # Experiment 1_2: if suffix => extract FRONT, else extract BACK
            if context_position == "suffix":
                inputs_shared["latents"] = inputs_shared["latents"][:, :, :-num_context_frames, :, :]  # (B, C, 81, H, W)
            else:
                inputs_shared["latents"] = inputs_shared["latents"][:, :, num_context_frames:, :, :]  # (B, C, 81, H, W)
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video



class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names


    def process(self, pipe: WanVideoPipeline, inputs: dict, positive=True, **kwargs) -> dict:
        raise NotImplementedError("`process` is not implemented.")



class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: WanVideoPipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image", "batch_size"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image, batch_size=None):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        batch_size = int(batch_size or 1)
        noise = pipe.generate_noise((batch_size, 16, length, height//8, width//8), seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        # Support batch: input_video can be list of lists (batch of videos) or list of PIL (single video)
        is_batch = isinstance(input_video, (list, tuple)) and len(input_video) > 0 and isinstance(input_video[0], (list, tuple))
        if is_batch:
            # Batch of videos: each element is list of PIL images; preprocess_video returns (1, C, T, H, W)
            videos_tensor = torch.cat([
                pipe.preprocess_video(v) for v in input_video
            ], dim=0)  # (B, C, T, H, W)
            input_video = videos_tensor
        else:
            input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        # Some base T2V checkpoints do not provide an image encoder.
        # In that case, image conditioning is not supported — safely skip.
        if getattr(pipe, "image_encoder", None) is None:
            return {}
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}

        
 
class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y"),
            onload_model_names=("vae")
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -16:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae")
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        # Some base T2V checkpoints do not provide an image encoder.
        # In that case, reference-image conditioning is not supported — safely skip.
        if getattr(pipe, "image_encoder", None) is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "camera_control_poses", "latents", "input_image")
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, camera_control_poses, latents, input_image):
        if getattr(pipe.dit, "control_adapter", None) is None:
            return {}
        if camera_control_direction is None and camera_control_poses is None:
            return {}
        if camera_control_poses is not None:
            # Build plucker embedding from explicit trajectory
            try:
                from ..models.wan_video_camera_controller import process_pose_file  # type: ignore
                camera_control_plucker_embedding = process_pose_file(camera_control_poses, width=width, height=height, device="cpu")
            except Exception as e:
                # Fallback: ignore camera control if malformed
                return {}
        else:
            camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
                camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    actions = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)

    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    # Build time position encoding with negative indices for context frames
    # Get num_context_frames from kwargs (passed from pipeline)
    num_context_frames = kwargs.get("num_context_frames", 0)
    import os
    context_position = kwargs.get("context_position", os.environ.get("CONTEXT_POSITION", "prefix"))
    if isinstance(context_position, str):
        context_position = context_position.lower()
    if context_position not in ("prefix", "suffix"):
        context_position = "prefix"
    
    # MoC [2508.21058]: reweight context chunks by query–chunk similarity before blocks
    use_moc = kwargs.get("use_moc", False)
    moc_module = kwargs.get("moc_module", None)
    if use_moc and moc_module is not None and num_context_frames > 0 and f > num_context_frames:
        x = moc_module(x, num_context_frames, int(f), int(h), int(w), context_position)

    # Geometry-grounded Spatial Memory: encode a TSDF/point-cloud-rendered
    # condition video independently from ordinary context tokens.
    use_geometry_spatial_memory = bool(kwargs.get("use_geometry_spatial_memory", False))
    geometry_memory_latents = kwargs.get("geometry_memory_latents", None)
    geometry_spatial_memory_module = kwargs.get("geometry_spatial_memory_module", None)
    geometry_inject_mode = str(
        kwargs.get("geometry_spatial_memory_inject_mode", "concat_text") or "concat_text"
    )
    geometry_readout_module = kwargs.get("geometry_spatial_memory_readout_module", None)
    if (
        use_geometry_spatial_memory
        and geometry_memory_latents is not None
        and geometry_spatial_memory_module is not None
        and geometry_inject_mode != "none"
    ):
        geometry_memory_latents = geometry_memory_latents.to(device=x.device, dtype=x.dtype)
        mem = geometry_spatial_memory_module(geometry_memory_latents)
        if mem.shape[0] != x.shape[0]:
            if x.shape[0] % mem.shape[0] != 0:
                raise ValueError(
                    f"Geometry memory batch mismatch: memory={mem.shape[0]} model={x.shape[0]}"
                )
            mem = mem.repeat(x.shape[0] // mem.shape[0], 1, 1)
        if geometry_inject_mode == "cross_attn_readout":
            target_frames = int(f) - min(int(num_context_frames), int(f))
            target_tokens = target_frames * int(h) * int(w)
            if target_tokens > 0 and target_tokens <= x.shape[1]:
                if context_position == "suffix":
                    x_target, x_other = x[:, :target_tokens], x[:, target_tokens:]
                    x_target = apply_spatial_cross_attn_readout(x_target, mem, geometry_readout_module)
                    x = torch.cat([x_target, x_other], dim=1)
                else:
                    x_other, x_target = x[:, :-target_tokens], x[:, -target_tokens:]
                    x_target = apply_spatial_cross_attn_readout(x_target, mem, geometry_readout_module)
                    x = torch.cat([x_other, x_target], dim=1)
        else:
            context = inject_spatial_memory(context, mem, "concat_text")

    # FramePack-Weight (FAR-style): see diffsynth.models.memory.framepack_weight
    use_framepack_memory = kwargs.get("use_framepack_memory", False)
    context_temporal_decay = float(kwargs.get("context_temporal_decay", 1.0) or 1.0)
    context_attention_weight = float(kwargs.get("context_attention_weight", 1.0) or 1.0)
    x = apply_framepack_token_weights(
        x,
        num_context_frames=num_context_frames,
        f=int(f),
        h=int(h),
        w=int(w),
        context_position=context_position,
        use_framepack_memory=bool(use_framepack_memory),
        context_temporal_decay=context_temporal_decay,
        context_attention_weight=context_attention_weight,
    )

    # Spatial memory: legacy pool OR SpatialGridMemory; inject to text context (concat_text)
    # or directly read memory from target tokens (cross_attn_readout).
    use_spatial_memory = kwargs.get("use_spatial_memory", False)
    spatial_memory_tokens = int(kwargs.get("spatial_memory_tokens", 64) or 64)
    use_spatial_memory_legacy = bool(kwargs.get("use_spatial_memory_legacy", False))
    spatial_memory_module = kwargs.get("spatial_memory_module", None)
    spatial_inject_mode = str(kwargs.get("spatial_memory_inject_mode", "concat_text") or "concat_text")
    spatial_readout_module = kwargs.get("spatial_memory_readout_module", None)
    if (
        use_spatial_memory
        and spatial_memory_tokens > 0
        and num_context_frames > 0
        and f > num_context_frames
    ):
        spatial_per_frame = int(h) * int(w)
        context_tokens = num_context_frames * spatial_per_frame
        if context_position == "suffix":
            x_context = x[:, -context_tokens:, :]
        else:
            x_context = x[:, :context_tokens, :]
        if spatial_memory_module is not None and not use_spatial_memory_legacy:
            mem = spatial_memory_module(x_context, num_context_frames, int(h), int(w))
        else:
            x_ctx = x_context.reshape(x_context.shape[0], num_context_frames, spatial_per_frame, x_context.shape[-1])
            mem = x_ctx.mean(dim=1)
            if mem.shape[1] != spatial_memory_tokens:
                mem_t = mem.transpose(1, 2)  # (B, D, S)
                mem_t = torch.nn.functional.adaptive_avg_pool1d(mem_t, spatial_memory_tokens)
                mem = mem_t.transpose(1, 2)
        if spatial_inject_mode == "none":
            pass
        elif spatial_inject_mode == "cross_attn_readout":
            target_tokens = (int(f) - int(num_context_frames)) * int(h) * int(w)
            if target_tokens > 0 and target_tokens < x.shape[1]:
                if context_position == "suffix":
                    x_target = x[:, :target_tokens, :]
                    x_context = x[:, target_tokens:, :]
                    x_target = apply_spatial_cross_attn_readout(x_target, mem, spatial_readout_module)
                    x = torch.cat([x_target, x_context], dim=1)
                else:
                    x_context = x[:, :x.shape[1] - target_tokens, :]
                    x_target = x[:, x.shape[1] - target_tokens:, :]
                    x_target = apply_spatial_cross_attn_readout(x_target, mem, spatial_readout_module)
                    x = torch.cat([x_context, x_target], dim=1)
            else:
                context = inject_spatial_memory(context, mem, "concat_text")
        else:
            context = inject_spatial_memory(context, mem, spatial_inject_mode)
    
    if num_context_frames > 0 and f > num_context_frames:
        # Context frames: use negative temporal indices [-k, ..., -1] to represent "past" frames (prefix mode)
        # OR use positive indices [target_frames, ..., target_frames+k-1] to represent "future" frames (suffix mode)
        # Target frames: use positive temporal indices [0, ..., N] to represent "current/future" frames
        target_frames = f - num_context_frames
        
        # Construct positive position indices for target: [0, ..., target_frames-1]
        target_ids = torch.arange(0, target_frames, device=x.device, dtype=torch.float64)
        
        # Experiment 1_2: if context is concatenated at END (suffix), use positive indices after target
        if context_position == "suffix":
            # Suffix mode: Context is after target, so use positive indices [target_frames, ..., target_frames+num_context_frames-1]
            # Example: target_frames=21, num_context_frames=2 => context_ids = [21, 22]
            context_ids = torch.arange(target_frames, target_frames + num_context_frames, device=x.device, dtype=torch.float64)
            t_ids = torch.cat([target_ids, context_ids])
        else:
            # Prefix mode: Context is before target, so use negative indices [-num_context_frames, ..., -1]
            # Example: num_context_frames=2 => context_ids = [-2, -1]
            context_ids = torch.arange(-num_context_frames, 0, device=x.device, dtype=torch.float64)
            t_ids = torch.cat([context_ids, target_ids])
        
        # Generate RoPE frequencies for custom position indices (including negative)
        # dit.freqs[0] is precomputed RoPE frequencies (complex64), we need to generate for custom positions
        # Use the same logic as precompute_freqs_cis but for arbitrary positions
        # Get the frequency dimension from precomputed freqs
        time_freq_dim_complex = dit.freqs[0].shape[-1]  # This is the complex dimension: (head_dim - 2*(head_dim//3)) // 2
        time_freq_dim_original = time_freq_dim_complex * 2  # Original dimension: head_dim - 2*(head_dim//3)
        # Compute base frequencies (same as in precompute_freqs_cis)
        theta = 10000.0
        base_freqs = 1.0 / (theta ** (torch.arange(0, time_freq_dim_original, 2, device=x.device, dtype=torch.float64)
                                      [:time_freq_dim_complex].double() / time_freq_dim_original))
        # Compute frequencies for each position: outer product of positions and base_freqs
        t_freqs = torch.outer(t_ids, base_freqs)  # (f, freq_dim_complex)
        # Convert to complex form (RoPE format): polar(ones, freqs)
        t_freqs_cis = torch.polar(torch.ones_like(t_freqs), t_freqs)  # complex64, (f, freq_dim_complex)
        
        # Reshape and expand to match spatial dimensions: (f, 1, 1, freq_dim) -> (f, h, w, freq_dim)
        freqs_t = t_freqs_cis.view(f, 1, 1, -1).expand(f, h, w, -1).to(x.device)
    else:
        # Standard mode: use precomputed frequencies for all frames
        freqs_t = dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1).to(x.device)
    
    # Spatial frequencies remain the same
    freqs_h = dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1).to(x.device)
    freqs_w = dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1).to(x.device)
    
    # Concatenate all frequencies (all tensors are now on the same device)
    freqs = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1).reshape(f * h * w, 1, -1)

    # Unified implicit (memory_design_advanced): single ctx-learning injection into t_mod
    # When use_non_fov_compressed: compressor input = non_fov_context_tokens (prev chunk non-FOV frames).
    # Otherwise: compressor input = x_ctx (explicit FOV context tokens).
    use_unified_implicit = kwargs.get("use_unified_implicit", False)
    unified_encoder = kwargs.get("unified_implicit_encoder", None)
    context_learning_injector = kwargs.get("context_learning_injector", None)
    use_non_fov_compressed = kwargs.get("use_non_fov_compressed", False)
    non_fov_context_tokens = kwargs.get("non_fov_context_tokens", None)
    if use_unified_implicit and unified_encoder is not None and context_learning_injector is not None and f > num_context_frames:
        if use_non_fov_compressed and non_fov_context_tokens is not None:
            z = unified_encoder(non_fov_context_tokens)
        elif num_context_frames > 0:
            spatial_per_frame = int(h) * int(w)
            context_tokens = num_context_frames * spatial_per_frame
            if context_position == "suffix":
                x_ctx = x[:, -context_tokens:, :]
            else:
                x_ctx = x[:, :context_tokens, :]
            z = unified_encoder(x_ctx)
        else:
            z = None
        if z is not None:
            delta_t_mod = context_learning_injector(z)
            t_mod = t_mod + delta_t_mod

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        if actions is not None:
            actions = torch.tensor(actions, device=x.device, dtype=x.dtype)
            actions = actions.unsqueeze(0)

        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    if actions is not None:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, actions,
                            use_reentrant=False,
                        )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
            elif use_gradient_checkpointing:
                if actions is not None:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, actions,
                        use_reentrant=False,
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                if actions is not None:
                    x = block(x, context, t_mod, freqs, actions)
                else:
                    x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x
