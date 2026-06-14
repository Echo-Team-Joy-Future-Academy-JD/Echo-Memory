import hashlib
import json
import logging
import os
import random
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file as safe_load_file

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from src.model_training.transformers_compat import patch_transformers_hybrid_cache

patch_transformers_hybrid_cache()
from diffsynth.trainers.utils import DiffusionTrainingModule
from src.model_training.fov_retrieval import flip_yaw_rt_list

logger = logging.getLogger(__name__)


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        timestep_shift=1.0,
        resume_from_checkpoint=None,
        dataset_base_path: Optional[str] = None,
        enable_context_memory=False,
        context_memory_frames=8,
        training_mode="context",  # "context" mode for Context Memory (inpainting)
        context_drop_prob: float = 0.0,
        context_drop_seed: int = 42,
        omit_context_actions: bool = False,  # Context-as-Memory: no context RT injection
        context_noise_prob=0.0,
        context_noise_std=0.02,
        context_fixed_noise_std=None,  # Experiment 7: Fixed noise std (e.g., 0.1) to align training-inference
        teacher_forcing_prob=0.0,
        yaw_flip_aug: bool = False,  # 50% prob flip yaw (ACTION_FOLLOWING direction sensitivity)
        context_per_frame_vae: bool = False,  # Encode each context frame separately (1 latent per raw frame)
        context_source: str = "fov",  # fov | replay | prev_chunk_tail (multichunk-aligned context construction)
        use_framepack_memory: bool = False,
        context_temporal_decay: float = 1.0,
        context_attention_weight: float = 1.0,
        use_framepack_length_compress: bool = False,
        framepack_ratio: int = 2,
        framepack_length_strategy: str = "distance_merge",
        framepack_recent_keep_ratio: float = 0.5,
        framepack_multiscale_w2: float = 0.25,
        framepack_multiscale_w4: float = 0.15,
        use_spatial_memory: bool = False,
        use_spatial_memory_legacy: bool = False,
        spatial_memory_tokens: int = 64,
        spatial_memory_inject_mode: str = "concat_text",
        # Note: Self-forcing parameters removed - using standard training only
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        from_pretrained_kw = {"torch_dtype": torch.bfloat16, "device": "cpu", "model_configs": model_configs}
        if tokenizer_path:
            from_pretrained_kw["tokenizer_config"] = ModelConfig(path=tokenizer_path)
        self.pipe = WanVideoPipeline.from_pretrained(**from_pretrained_kw)
        
        # Store timestep_shift for later use (e.g., after video sampling)
        self.timestep_shift = timestep_shift
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=timestep_shift)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
            # Load checkpoint if provided
            if resume_from_checkpoint is not None:
                logger.info(f"Loading LoRA checkpoint from: {resume_from_checkpoint}")
                if not os.path.exists(resume_from_checkpoint):
                    raise FileNotFoundError(f"Checkpoint file not found: {resume_from_checkpoint}")
                checkpoint_state_dict = safe_load_file(resume_from_checkpoint)
                logger.info(f"Checkpoint contains {len(checkpoint_state_dict)} parameters")
                # The checkpoint was saved with remove_prefix_in_ckpt, so keys don't have the prefix
                # The model (pipe.dit) state_dict keys also don't have the prefix, so they should match
                # Use strict=False to allow partial loading
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"{len(missing_keys)} keys were missing when loading checkpoint")
                    if len(missing_keys) <= 10:
                        logger.debug(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"{len(unexpected_keys)} unexpected keys in checkpoint (will be ignored)")
                    if len(unexpected_keys) <= 10:
                        logger.debug(f"Unexpected keys: {unexpected_keys}")
                loaded_count = len(checkpoint_state_dict) - len(missing_keys) - len(unexpected_keys)
                logger.info(f"Successfully loaded {loaded_count} parameters from checkpoint!")
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.dataset_base_path = dataset_base_path
        
        # Context Memory (Context as Memory) configuration
        self.enable_context_memory = enable_context_memory
        self.context_memory_frames = context_memory_frames
        self.training_mode = training_mode  # "predict", "context", or "condition"
        self.context_drop_prob = float(context_drop_prob or 0.0)
        self.context_drop_seed = int(context_drop_seed or 42)
        self.omit_context_actions = bool(omit_context_actions)
        self.context_per_frame_vae = bool(context_per_frame_vae)
        self.context_source = (context_source or "fov").strip().lower()
        if self.context_source not in ("fov", "replay", "prev_chunk_tail"):
            self.context_source = "fov"
        self.context_noise_prob = context_noise_prob
        self.context_noise_std = context_noise_std
        self.context_fixed_noise_std = context_fixed_noise_std  # Experiment 7: Fixed noise for training-inference alignment
        self.teacher_forcing_prob = teacher_forcing_prob
        self.teacher_forcing_enabled = teacher_forcing_prob > 0.0
        self.yaw_flip_aug = bool(yaw_flip_aug)
        # Memory baselines runtime flags (train + sampling path shared).
        self.use_framepack_memory = bool(use_framepack_memory)
        self.context_temporal_decay = float(context_temporal_decay or 1.0)
        self.context_attention_weight = float(context_attention_weight or 1.0)
        self.use_framepack_length_compress = bool(use_framepack_length_compress)
        self.framepack_ratio = int(framepack_ratio or 2)
        self.framepack_length_strategy = str(framepack_length_strategy or "distance_merge").lower()
        self.framepack_recent_keep_ratio = float(framepack_recent_keep_ratio or 0.5)
        self.framepack_multiscale_w2 = float(framepack_multiscale_w2 or 0.25)
        self.framepack_multiscale_w4 = float(framepack_multiscale_w4 or 0.15)
        # Mirror key flags to pipe for inference-time sampling monitor.
        self.pipe.use_framepack_memory = self.use_framepack_memory
        self.pipe.context_temporal_decay = self.context_temporal_decay
        self.pipe.context_attention_weight = self.context_attention_weight
        self.pipe.use_framepack_length_compress = self.use_framepack_length_compress
        self.pipe.framepack_ratio = self.framepack_ratio
        self.pipe.framepack_length_strategy = self.framepack_length_strategy
        self.pipe.framepack_recent_keep_ratio = self.framepack_recent_keep_ratio
        self.pipe.framepack_multiscale_w2 = self.framepack_multiscale_w2
        self.pipe.framepack_multiscale_w4 = self.framepack_multiscale_w4
        self.pipe.use_spatial_memory = bool(use_spatial_memory)
        self.pipe.use_spatial_memory_legacy = bool(use_spatial_memory_legacy)
        self.pipe.spatial_memory_tokens = int(spatial_memory_tokens or 64)
        self.pipe.spatial_memory_inject_mode = str(spatial_memory_inject_mode or "concat_text")
        # Note: Self-forcing removed - using standard training only
        self.current_step = 0  # Track current training step (for logging/debugging)
    
    def _forward_preprocess_batch(self, samples: list) -> dict:
        """Batch preprocessing for Stage 1 Interactive (no context). data is list of sample dicts."""
        if not samples:
            raise ValueError("samples cannot be empty in _forward_preprocess_batch")
        batch_size = len(samples)
        prompts = []
        video_frames_list = []
        actions_list = []
        for s in samples:
            p = s.get("prompt")
            if p is None:
                raise ValueError("sample['prompt'] is missing or None")
            prompts.append(str(p) if not isinstance(p, str) else p)
            video_frames_list.append(s["video"])
            if "actions" in s and s["actions"] is not None:
                acts = s["actions"]
                if getattr(self, 'yaw_flip_aug', False) and isinstance(acts, list) and len(acts) > 0 and isinstance(acts[0], (list, tuple)) and len(acts[0]) >= 12 and random.random() < 0.5:
                    acts = flip_yaw_rt_list(acts)
                if isinstance(acts, torch.Tensor):
                    actions_list.append(acts)
                elif isinstance(acts, list) and len(acts) > 0:
                    actions_list.append(torch.tensor(acts, dtype=torch.float32))
                else:
                    actions_list.append(None)
            else:
                actions_list.append(None)
        
        # input_video: list of lists (each inner list = PIL images for one video)
        input_video = video_frames_list
        first = samples[0]
        h, w = first["video"][0].size[1], first["video"][0].size[0]
        num_frames = len(first["video"])
        
        inputs_posi = {"prompt": prompts}
        inputs_nega = {}
        inputs_shared = {
            "input_video": input_video,
            "height": h,
            "width": w,
            "num_frames": num_frames,
            "batch_size": batch_size,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        
        ref_action = next((a for a in actions_list if a is not None), None)
        if ref_action is not None and batch_size == 1:
            inputs_shared["actions"] = ref_action.detach().cpu().tolist() if isinstance(ref_action, torch.Tensor) else ref_action
        elif ref_action is not None:
            device = self.pipe.device
            dtype = ref_action.dtype
            stacked = []
            for a in actions_list:
                if a is not None:
                    stacked.append(a.to(device=device))
                else:
                    stacked.append(torch.zeros_like(ref_action, device=device, dtype=dtype))
            inputs_shared["actions"] = torch.stack(stacked)
        else:
            inputs_shared["actions"] = None
        
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    def _build_context_with_anchor(self, context_frames, context_actions=None, expected_k=None):
        """Training-side anchor helper: keep last frame as mandatory anchor and keep action length aligned."""
        frames = list(context_frames or [])
        actions = list(context_actions or []) if context_actions is not None else []
        if not frames or not getattr(self, "use_anchor_frame", False):
            return frames, actions
        k = int(expected_k) if (expected_k is not None and int(expected_k) > 0) else len(frames)
        if len(frames) > k:
            frames = frames[-k:]
            if actions:
                actions = actions[-k:]
        if actions:
            if len(actions) < len(frames):
                actions = actions + [actions[-1]] * (len(frames) - len(actions))
            elif len(actions) > len(frames):
                actions = actions[:len(frames)]
        return frames, actions

    def _forward_preprocess_batch_context(self, samples: list) -> dict:
        """Batch preprocessing for Stage 2 Context Memory. Batch-level drop: if drop, all samples get no context."""
        if not samples:
            raise ValueError("samples cannot be empty in _forward_preprocess_batch_context")
        batch_size = len(samples)
        first = samples[0]
        
        def _should_drop_context(_data) -> bool:
            p = float(getattr(self, "context_drop_prob", 0.0) or 0.0)
            if p <= 0.0:
                return False
            if p >= 1.0:
                return True
            vn = str(_data.get("video_name", ""))
            sf = str(_data.get("start_frame", ""))
            key = f"{int(getattr(self, 'context_drop_seed', 42))}|{vn}|{sf}"
            h = hashlib.md5(key.encode("utf-8")).hexdigest()
            u = int(h[:8], 16) / 0xFFFFFFFF
            return u < p
        
        # Batch-level drop: use first sample to decide for whole batch
        dropped_context = _should_drop_context(first)
        # IMPORTANT (DDP safety): ensure all ranks make the same drop decision.
        # If some ranks drop context while others keep it, modules conditioned on context
        # (e.g. implicit encoder / compressor) become unused on a subset of ranks and can
        # deadlock gradient sync / trigger NCCL watchdog timeouts.
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor([1 if dropped_context else 0], device=self.pipe.device, dtype=torch.int64)
                dist.broadcast(flag, src=0)
                dropped_context = bool(int(flag.item()))
        except Exception:
            pass
        
        prompts = []
        video_frames_list = []
        actions_list = []
        context_latents_list = []
        context_actions_list = []
        expected_k = self.context_memory_frames
        training_mode = getattr(self, 'training_mode', 'context')
        
        target_h = first["video"][0].size[1]
        target_w = first["video"][0].size[0]
        num_frames = len(first["video"])
        
        from PIL import Image
        
        for s in samples:
            p = s.get("prompt")
            if p is None:
                raise ValueError("sample['prompt'] is missing or None")
            prompts.append(str(p) if not isinstance(p, str) else p)
            video_frames_list.append(s["video"])
            
            if "actions" in s and s["actions"] is not None:
                acts = s["actions"]
                if getattr(self, 'yaw_flip_aug', False) and isinstance(acts, list) and len(acts) > 0 and isinstance(acts[0], (list, tuple)) and len(acts[0]) >= 12 and random.random() < 0.5:
                    acts = flip_yaw_rt_list(acts)
                if isinstance(acts, torch.Tensor):
                    actions_list.append(acts)
                elif isinstance(acts, list) and len(acts) > 0:
                    actions_list.append(torch.tensor(acts, dtype=torch.float32))
                else:
                    actions_list.append(None)
            else:
                actions_list.append(None)
            
            if dropped_context:
                context_latents_list.append(None)
                context_actions_list.append(None)
                continue
            
            ctx_frames = s.get("context_frames") or []
            ctx_actions = [] if getattr(self, "omit_context_actions", False) else (s.get("context_actions") or [])  # ctx=1: no context action
            context_indices = s.get("context_frame_indices", [])
            start_frame = s.get("start_frame", None)
            end_frame = s.get("end_frame", None)
            
            if ctx_frames and context_indices and start_frame is not None and end_frame is not None:
                filtered_frames, filtered_actions = [ctx_frames[0]], []
                if ctx_actions:
                    filtered_actions.append(ctx_actions[0])
                for i in range(1, len(ctx_frames)):
                    idx = context_indices[i] if i < len(context_indices) else None
                    if idx is None or idx < start_frame or idx > end_frame:
                        filtered_frames.append(ctx_frames[i])
                        if ctx_actions and i < len(ctx_actions):
                            filtered_actions.append(ctx_actions[i])
                ctx_frames, ctx_actions = filtered_frames, filtered_actions if filtered_actions else ctx_actions
            
            if not ctx_frames and len(s["video"]) > expected_k:
                ctx_frames = s["video"][:expected_k]
                if s.get("actions") and len(s["actions"]) >= expected_k:
                    ctx_actions = s["actions"][:expected_k]
            
            if not ctx_frames:
                context_latents_list.append(None)
                context_actions_list.append(None)
                continue
            
            resized = []
            for f in ctx_frames:
                if hasattr(f, 'resize') and hasattr(f, 'size'):
                    w, h = f.size
                    if h != target_h or w != target_w:
                        f = f.resize((target_w, target_h), Image.Resampling.LANCZOS)
                resized.append(f)
            ctx_frames = resized
            
            if len(ctx_frames) < expected_k:
                last = ctx_frames[-1] if ctx_frames else Image.new('RGB', (target_w, target_h), (0, 0, 0))
                ctx_frames = ctx_frames + [last] * (expected_k - len(ctx_frames))
                if ctx_actions:
                    ctx_actions = ctx_actions + [ctx_actions[-1]] * (expected_k - len(ctx_actions))
            elif len(ctx_frames) > expected_k:
                ctx_frames = ctx_frames[:expected_k]
                ctx_actions = ctx_actions[:expected_k] if ctx_actions else []

            ctx_frames, ctx_actions = self._build_context_with_anchor(
                ctx_frames,
                context_actions=ctx_actions,
                expected_k=expected_k,
            )
            
            with torch.no_grad():
                if getattr(self, "context_per_frame_vae", False):
                    # Each context frame -> 1 latent token (no temporal downsample); context_actions remain one per raw frame
                    context_latents_per_sample = []
                    for f in ctx_frames:
                        frame_video = self.pipe.preprocess_video([f])  # (1, C, 1, H, W)
                        frame_sq = frame_video.squeeze(0)  # (C, 1, H, W)
                        lat_one = self.pipe.vae.encode([frame_sq], device=self.pipe.device, tiled=False, tile_size=None, tile_stride=None)
                        context_latents_per_sample.append(lat_one)
                    lat = torch.cat(context_latents_per_sample, dim=2)  # (1, C, K, H//8, W//8)
                else:
                    ctx_video = self.pipe.preprocess_video(ctx_frames)
                    if ctx_video.dim() == 4:
                        ctx_video = ctx_video.unsqueeze(0)
                    lat = self.pipe.vae.encode([ctx_video[i] for i in range(ctx_video.shape[0])], device=self.pipe.device, tiled=False, tile_size=None, tile_stride=None)
            context_latents_list.append(lat.to(dtype=self.pipe.torch_dtype, device=self.pipe.device))
            
            if ctx_actions:
                if isinstance(ctx_actions[0], (list, tuple)):
                    context_actions_list.append(torch.tensor(ctx_actions, dtype=torch.float32))
                else:
                    context_actions_list.append(torch.tensor(ctx_actions, dtype=torch.float32))
            else:
                context_actions_list.append(None)
        
        input_video = video_frames_list
        inputs_posi = {"prompt": prompts}
        inputs_nega = {}
        inputs_shared = {
            "input_video": input_video,
            "height": target_h,
            "width": target_w,
            "num_frames": num_frames,
            "batch_size": batch_size,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        
        # DDP safety: ensure *all* ranks either have context (and thus use context-conditioned modules)
        # or all ranks drop it. Using an all-reduce MIN means if any rank lacks context, we drop globally.
        has_context_step = (not dropped_context) and any(x is not None for x in context_latents_list)
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor([1 if has_context_step else 0], device=self.pipe.device, dtype=torch.int64)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                has_context_step = bool(int(flag.item()))
        except Exception:
            pass
        if not has_context_step:
            dropped_context = True

        if not dropped_context and any(x is not None for x in context_latents_list):
            valid = [x for x in context_latents_list if x is not None]
            if valid:
                ref = valid[0]
                device, dtype = self.pipe.device, ref.dtype
                stacked_ctx = []
                for x in context_latents_list:
                    if x is not None:
                        stacked_ctx.append(x.to(device=device))
                    else:
                        stacked_ctx.append(torch.zeros_like(ref, device=device, dtype=dtype))
                inputs_shared["context_latents"] = torch.cat(stacked_ctx, dim=0)
                inputs_shared["num_context_frames"] = ref.shape[2]
                inputs_shared["training_mode"] = training_mode
                inputs_shared["context_noise_prob"] = getattr(self, 'context_noise_prob', 0.0)
                inputs_shared["context_noise_std"] = getattr(self, 'context_noise_std', 0.02)
                if self.context_fixed_noise_std is not None:
                    inputs_shared["context_fixed_noise_std"] = self.context_fixed_noise_std
                inputs_shared["context_position"] = os.environ.get("CONTEXT_POSITION", "suffix")
                inputs_shared["omit_context_actions"] = getattr(self, "omit_context_actions", False)
                inputs_shared["context_attention_weight"] = getattr(self, "context_attention_weight", 1.0)
                inputs_shared["use_anchor_frame"] = getattr(self, "use_anchor_frame", False)
                inputs_shared["context_temporal_decay"] = getattr(self, "context_temporal_decay", 1.0)
                inputs_shared["use_spatial_memory"] = getattr(self.pipe, "use_spatial_memory", False)
                inputs_shared["spatial_memory_tokens"] = int(getattr(self.pipe, "spatial_memory_tokens", 64) or 64)
                inputs_shared["use_spatial_memory_legacy"] = bool(getattr(self.pipe, "use_spatial_memory_legacy", False))
                inputs_shared["spatial_memory_module"] = getattr(self.pipe, "spatial_memory_module", None)
                inputs_shared["spatial_memory_inject_mode"] = getattr(self.pipe, "spatial_memory_inject_mode", "concat_text")
                inputs_shared["spatial_memory_readout_module"] = getattr(self.pipe, "spatial_memory_readout_module", None)
                inputs_shared["use_framepack_memory"] = bool(getattr(self, "use_framepack_memory", False))
                nf_list = [s.get("non_fov_frames") or [] for s in samples]
                if any(nf for nf in nf_list):
                    inputs_shared["non_fov_frames_list"] = nf_list

                ctx_acts_valid = [a for a in context_actions_list if a is not None]
                if not getattr(self, "omit_context_actions", False) and ctx_acts_valid:
                    ref_act = ctx_acts_valid[0]
                    target_len = ref_act.shape[0]  # num_context_frames (K)
                    stacked_ca = []
                    for a in context_actions_list:
                        if a is not None:
                            a = a.to(device=device)
                            if a.shape[0] != target_len:
                                if a.shape[0] > target_len:
                                    a = a[:target_len]
                                else:
                                    pad = a.new_zeros(target_len - a.shape[0], a.shape[-1])
                                    a = torch.cat([a, pad], dim=0)
                            stacked_ca.append(a)
                        else:
                            stacked_ca.append(torch.zeros_like(ref_act, device=device, dtype=ref_act.dtype))
                    inputs_shared["context_actions"] = torch.stack(stacked_ca)
        
        ref_action = next((a for a in actions_list if a is not None), None)
        if ref_action is not None and batch_size == 1:
            inputs_shared["actions"] = ref_action.detach().cpu().tolist() if isinstance(ref_action, torch.Tensor) else ref_action
        elif ref_action is not None:
            device = self.pipe.device
            dtype = ref_action.dtype
            stacked = []
            for a in actions_list:
                if a is not None:
                    stacked.append(a.to(device=device))
                else:
                    stacked.append(torch.zeros_like(ref_action, device=device, dtype=dtype))
            inputs_shared["actions"] = torch.stack(stacked)
        else:
            inputs_shared["actions"] = None
        
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
        
    @staticmethod
    def _translate_condition_keys(d):
        """Map VWM CamVideoDataset condition_* keys to context-memory keys."""
        if not isinstance(d, dict):
            return d
        if "condition_frames" in d and "context_frames" not in d:
            d["context_frames"] = d.pop("condition_frames")
        if "condition_actions" in d and "context_actions" not in d:
            d["context_actions"] = d.pop("condition_actions")
        if "condition_frame_indices" in d and "context_frame_indices" not in d:
            d["context_frame_indices"] = d.pop("condition_frame_indices")
        if "use_condition_context_frames" in d:
            d.pop("use_condition_context_frames")
        if "condition_source" in d:
            d.pop("condition_source", None)
        return d

    def forward_preprocess(self, data):
        if data is None:
            raise ValueError("data cannot be None in forward_preprocess")
        samples = data if isinstance(data, list) else [data]
        samples = [self._translate_condition_keys(d) for d in samples]
        if self.enable_context_memory:
            return self._forward_preprocess_batch_context(samples)
        return self._forward_preprocess_batch(samples)

    def _ensure_input_latents(self, inputs: Dict[str, Any], *, strict: bool = False) -> Dict[str, Any]:
        if "input_latents" in inputs:
            return inputs
        import warnings
        video_obj = inputs.get("input_video", None)
        if video_obj is None:
            video_obj = inputs.get("video", None)
        vae = getattr(self.pipe, "vae", None)
        if video_obj is not None and vae is not None and hasattr(vae, "encode"):
            try:
                if isinstance(video_obj, list):
                    video_tensor = self.pipe.preprocess_video(video_obj)
                else:
                    video_tensor = video_obj
                if hasattr(video_tensor, "dim"):
                    video_sq = video_tensor.squeeze(0) if video_tensor.dim() == 5 else video_tensor
                    with torch.no_grad():
                        try:
                            lat = vae.encode(video_tensor, device=self.pipe.device, tiled=False, tile_size=None, tile_stride=None)
                        except Exception:
                            lat = vae.encode([video_sq], device=self.pipe.device, tiled=False, tile_size=None, tile_stride=None)
                    if isinstance(lat, (list, tuple)):
                        lat = lat[0]
                    if hasattr(lat, "dim") and lat.dim() == 4:
                        lat = lat.unsqueeze(0)
                    inputs["input_latents"] = lat.to(dtype=torch.bfloat16, device=self.pipe.device)
                    return inputs
            except Exception as e:
                warnings.warn(f"Failed to rebuild input_latents: {e}")
        msg = (
            "input_latents missing and auto-rebuild failed. "
            f"available input keys={sorted(list(inputs.keys()))}"
        )
        if strict:
            raise KeyError(msg)
        warnings.warn(msg)
        return inputs
    
    
    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        if self.enable_context_memory and "context_latents" in inputs:
            return self._training_loss_with_context(**models, **inputs)
        inputs = self._ensure_input_latents(inputs, strict=True)
        return self.pipe.training_loss(**models, **inputs)
    
    def _training_loss_with_context(self, **kwargs):
        context_latents = kwargs.pop("context_latents", None)
        num_context_frames = kwargs.pop("num_context_frames", 0)
        models = {k: v for k, v in kwargs.items() if k in self.pipe.in_iteration_models}
        inputs = {k: v for k, v in kwargs.items() if k not in self.pipe.in_iteration_models}
        if context_latents is not None:
            inputs.update({
                "context_latents": context_latents,
                "num_context_frames": num_context_frames,
                "context_noise_prob": self.context_noise_prob,
                "context_noise_std": self.context_noise_std,
                "context_attention_weight": getattr(self, "context_attention_weight", 1.0),
                "use_anchor_frame": getattr(self, "use_anchor_frame", False),
                "context_temporal_decay": getattr(self, "context_temporal_decay", 1.0),
                "use_spatial_memory": getattr(self.pipe, "use_spatial_memory", False),
                "spatial_memory_tokens": int(getattr(self.pipe, "spatial_memory_tokens", 64) or 64),
                "use_spatial_memory_legacy": bool(getattr(self.pipe, "use_spatial_memory_legacy", False)),
                "spatial_memory_module": getattr(self.pipe, "spatial_memory_module", None),
                "spatial_memory_inject_mode": getattr(self.pipe, "spatial_memory_inject_mode", "concat_text"),
                "spatial_memory_readout_module": getattr(self.pipe, "spatial_memory_readout_module", None),
                "use_framepack_memory": bool(getattr(self, "use_framepack_memory", False)),
            })
            if self.context_fixed_noise_std is not None:
                inputs["context_fixed_noise_std"] = self.context_fixed_noise_std
        inputs = self._ensure_input_latents(inputs, strict=True)
        return self.pipe.training_loss(**models, **inputs)

