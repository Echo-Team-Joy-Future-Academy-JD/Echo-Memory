import os, json, sys, re, hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice
from typing import Any, Dict, Optional
import importlib
import logging

logger = logging.getLogger(__name__)

_rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or os.environ.get("ACCELERATE_PROCESS_INDEX") or "0"
try:
    _rank = int(str(_rank_env))
except Exception:
    _rank = 0
_level = logging.INFO if _rank == 0 else logging.WARNING
try:
    logging.basicConfig(
        level=_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
except TypeError:
    logging.basicConfig(
        level=_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger.setLevel(_level)

current_file_abs = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_abs)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

modules_to_clear = [
    'diffsynth.models.memory.framepack_length',
    'diffsynth.models.memory.framepack_weight',
    'diffsynth.models.memory.spatial_grid_memory',
    'diffsynth.models.memory.videossm_hybrid',
    'diffsynth.models.memory.block_wise_ssm',
    'diffsynth.models.memory',
    'diffsynth.pipelines.wan_video_new',
    'diffsynth.trainers.utils',
    'diffsynth.models.wan_video_dit',
    'diffsynth.lora.flux_lora',
    'diffsynth.lora',
    'diffsynth.configs.model_config',
    'diffsynth.configs',
    'diffsynth.pipelines',
    'diffsynth.trainers',
    'diffsynth.models',
    'diffsynth',
]

for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]

importlib.invalidate_caches()

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

try:
    import transformers
    if not hasattr(transformers, "HybridCache") and hasattr(transformers, "DynamicCache"):
        transformers.HybridCache = transformers.DynamicCache
except Exception:
    pass

from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger as BaseModelLogger, VideoDataset, CamVideoDataset, wan_parser
from diffsynth.models.wan_video_dit import SelfAttention, CrossAttention, GateModule, modulate
from diffsynth.models.memory.videossm_hybrid import HybridStateSpaceMemory
from diffsynth.models.memory.block_wise_ssm import BlockWiseStateSpaceMemory

try:
    import diffsynth.trainers.utils as utils_module
    utils_file = utils_module.__file__ if hasattr(utils_module, '__file__') else 'unknown'
    is_local = 'site-packages' not in utils_file
    if is_local:
        logger.info(f"[VERIFIED] Using LOCAL diffsynth code from: {utils_file}")
    else:
        logger.warning(f"Using INSTALLED diffsynth package from: {utils_file}")
except Exception as e:
    logger.error(f"Failed to verify code location: {e}")

from accelerate import Accelerator
import wandb
from tqdm import tqdm
import torch.distributed as dist
import random
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from safetensors.torch import load_file as safe_load_file

TrainingMemoryBank = None  # optional: set if memory_bank module exists
try:
    from .memory_bank import TrainingMemoryBank
except ImportError:
    try:
        import os as _os
        _current_dir = _os.path.dirname(_os.path.abspath(__file__))
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)
        from memory_bank import TrainingMemoryBank
    except ImportError:
        pass  # memory_bank not used in this run
try:
    from .fov_retrieval import FOVMemoryRetriever, create_fov_retriever
    from .fov_training_integration import retrieve_fov_context_frames, setup_fov_retriever_for_training
    from .context_retrieval import retrieve_context_frames_advanced
except ImportError:
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from fov_retrieval import FOVMemoryRetriever, create_fov_retriever
    from fov_training_integration import retrieve_fov_context_frames, setup_fov_retriever_for_training
    from context_retrieval import retrieve_context_frames_advanced


class MLP_Action(nn.Module):
    def __init__(self, out_dim, sliding_window_size=3, r=4):
        super().__init__()
        self.proj_action = nn.Linear(r * sliding_window_size * 10, out_dim)
        nn.init.zeros_(self.proj_action.weight)
        nn.init.zeros_(self.proj_action.bias)
        self.sliding_window_size = sliding_window_size
        self.r = r

    def forward(self, x):
        bs, nr, act_dim = x.shape
        r = self.r
        n = nr // r
        actions = x.reshape(bs, n, r, act_dim)
        actions = F.pad(actions, (0, 0, 0, 0, self.sliding_window_size - 1, 1), mode="replicate")
        action_windows = []
        for i in range(self.sliding_window_size):
            action_windows.append(actions[:, i:i + n + 1])
        actions = torch.cat(action_windows, dim=2)
        actions = actions.reshape(bs, n + 1, -1)
        actions = self.proj_action(actions)
        return actions


class MLP_CamPose(nn.Module):
    def __init__(self, out_dim, pose_dim=12):
        super().__init__()
        self.proj = nn.Linear(pose_dim, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class DiTBlock_w_Action(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int,
                 eps: float = 1e-6, add_action_attn=False,
                 action_use_temporal_attention: bool = True, use_cam_pose: bool = False,
                 use_block_wise_ssm: bool = False, use_videossm_hybrid: bool = False,
                 videossm_kernel_size: int = 3, videossm_expand: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        if add_action_attn:
            self.self_attn_with_action = SelfAttention(dim, num_heads, eps)
            nn.init.zeros_(self.self_attn_with_action.o.weight)
            nn.init.zeros_(self.self_attn_with_action.o.bias)
        if use_cam_pose:
            self.action_mlp = MLP_CamPose(dim)
        else:
            self.action_mlp = MLP_Action(dim)

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.action_use_temporal_attention = action_use_temporal_attention
        self.use_block_wise_ssm = bool(use_block_wise_ssm)
        self.use_videossm_hybrid = bool(use_videossm_hybrid)
        if use_block_wise_ssm:
            self.block_wise_ssm = BlockWiseStateSpaceMemory(dim)
        if use_videossm_hybrid:
            self.videossm_hybrid = HybridStateSpaceMemory(
                dim, kernel_size=videossm_kernel_size, expand=videossm_expand
            )

    def forward(self, x, context, t_mod, freqs, actions=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        num_frames = None
        if actions is not None:
            original_x = x
            actions = self.action_mlp(actions.to(x.dtype)).to(x.dtype)
            bs, num_frames, dim = actions.shape
            actions = actions.reshape(bs, num_frames, 1, dim)
            x = x.reshape(bs, num_frames, -1, dim)
            x = x + actions
            if hasattr(self, "self_attn_with_action"):
                if not self.action_use_temporal_attention:
                    x = x.reshape(bs, -1, dim)
                    x = original_x + self.self_attn_with_action(x, freqs)
                else:
                    from einops import rearrange
                    x = rearrange(x, "b f p d -> (b p) f d")
                    attn_out = self.self_attn_with_action(x)
                    attn_out = rearrange(attn_out, "(b p) f d -> b f p d", b=bs)
                    x = original_x + attn_out.reshape(bs, -1, dim)
            else:
                x = x.reshape(bs, -1, dim)

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        if num_frames is not None:
            if hasattr(self, "block_wise_ssm"):
                x = self.block_wise_ssm(x, f=num_frames)
            if hasattr(self, "videossm_hybrid"):
                spatial = x.shape[1] // int(num_frames) if int(num_frames) > 0 else 0
                x = self.videossm_hybrid(x, f=num_frames, h=1, w=spatial)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


def set_seed(seed=42):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    logger.info(f"Random seed set to {seed}")


def _log_dit_freeze_summary(dit: torch.nn.Module) -> None:
    by_module: dict[str, tuple[int, bool]] = {}
    for name, p in dit.named_parameters():
        numel = p.numel()
        trainable = p.requires_grad
        parts = name.split(".")
        prefix = ".".join(parts[:-1]) if len(parts) > 1 else name
        if prefix not in by_module:
            by_module[prefix] = (0, False)
        prev_numel, prev_trainable = by_module[prefix]
        by_module[prefix] = (prev_numel + numel, prev_trainable or trainable)
    trainable_list = [(k, v[0]) for k, v in by_module.items() if v[1]]
    frozen_list = [(k, v[0]) for k, v in by_module.items() if not v[1]]
    trainable_list.sort(key=lambda x: x[0])
    frozen_list.sort(key=lambda x: x[0])
    total_trainable = sum(n for _, n in trainable_list)
    total_frozen = sum(n for _, n in frozen_list)
    examples = ", ".join(name for name, _ in trainable_list[:8])
    logger.info(
        f"[DiT freeze] trainable={total_trainable:,} ({len(trainable_list)} groups), "
        f"frozen={total_frozen:,} ({len(frozen_list)} groups), examples=[{examples}]"
    )


set_seed(42)


class ModelLogger(BaseModelLogger):
    """Compatibility wrapper for legacy training scripts."""

    def __init__(
        self,
        output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=lambda x: x,
        wandb_run_name=None,
        ckpt_interval=None,
        resume_step_count=0,
        save_full_model=False,
        context_drop_prob: float = 0.0,
        enable_video_sampling=False,
        sampling_interval_steps: int = 0,
        sampling_two_chunk_memory: bool = False,
        sampling_action_path: Optional[str] = None,
        sampling_two_chunk_action_path: Optional[str] = None,
        sampling_negative_prompt: str = "oversaturated colors, overexposed, static, blurry details",
        sampling_height: int = 352,
        sampling_width: int = 640,
        sampling_num_frames: int = 81,
        sampling_num_inference_steps: int = 50,
        context_memory_frames: int = 1,
        context_source: str = "replay",
        context_per_frame_vae: bool = False,
        **_unused,
    ):
        super().__init__(output_path, remove_prefix_in_ckpt=remove_prefix_in_ckpt, state_dict_converter=state_dict_converter)
        self.wandb_run_name = wandb_run_name
        self.ckpt_interval = int(ckpt_interval) if ckpt_interval else None
        self.step_count = int(resume_step_count or 0)
        self.save_full_model = bool(save_full_model)
        self.total_steps = None
        self.context_drop_prob = float(context_drop_prob or 0.0)
        self.enable_video_sampling = bool(enable_video_sampling)
        self.sampling_interval_steps = int(sampling_interval_steps or 0)
        self.sampling_two_chunk_memory = bool(sampling_two_chunk_memory)
        self.sampling_action_path = sampling_action_path
        self.sampling_two_chunk_action_path = sampling_two_chunk_action_path
        self.sampling_negative_prompt = sampling_negative_prompt
        self.sampling_height = int(sampling_height or 352)
        self.sampling_width = int(sampling_width or 640)
        self.sampling_num_frames = int(sampling_num_frames or 81)
        self.sampling_num_inference_steps = int(sampling_num_inference_steps or 50)
        self.context_memory_frames = int(context_memory_frames or 1)
        self.context_source = (context_source or "replay").strip().lower()
        self.context_per_frame_vae = bool(context_per_frame_vae)
        self.wandb_logger = None
        if self.wandb_run_name:
            try:
                self.wandb_logger = wandb.init(project="wan-cam", name=self.wandb_run_name, reinit=True)
            except Exception as e:
                logger.warning(f"[ModelLogger] wandb init failed: {e}")
                self.wandb_logger = None

    def _save_step_or_epoch_ckpt(self, accelerator, model, path: str):
        state_dict = None
        unwrapped = accelerator.unwrap_model(model)
        if self.save_full_model:
            # Save full DiT (including action/camera/memory modules), not whole pipeline.
            try:
                dit = getattr(getattr(unwrapped, "pipe", None), "dit", None)
                if dit is not None:
                    state_dict = accelerator.get_state_dict(dit)
            except Exception:
                state_dict = None
        if state_dict is None:
            full_state = accelerator.get_state_dict(model)
            state_dict = unwrapped.export_trainable_state_dict(full_state, remove_prefix=self.remove_prefix_in_ckpt)
        state_dict = self.state_dict_converter(state_dict)
        os.makedirs(self.output_path, exist_ok=True)
        accelerator.save(state_dict, path, safe_serialization=True)

    def _maybe_sample_paper_process(self, accelerator=None, model=None, current_batch=None):
        if not (
            self.enable_video_sampling
            and self.sampling_two_chunk_memory
            and self.sampling_interval_steps > 0
            and self.step_count % self.sampling_interval_steps == 0
            and accelerator is not None
            and model is not None
            and current_batch is not None
        ):
            return
        try:
            from diffsynth import save_video
            try:
                from src.model_training.multichunk_sample_utils import (
                    run_two_chunk_memory_monitor,
                    sync_pipe_memory_from_training_module,
                )
            except Exception:
                from multichunk_sample_utils import run_two_chunk_memory_monitor, sync_pipe_memory_from_training_module

            sample = current_batch[0] if isinstance(current_batch, list) else current_batch
            first_frame = (sample.get("video") or [None])[0]
            if first_frame is None:
                return
            unwrapped = accelerator.unwrap_model(model)
            pipe = getattr(unwrapped, "pipe", None)
            if pipe is None:
                return
            sync_pipe_memory_from_training_module(pipe, unwrapped)
            action0 = self.sampling_two_chunk_action_path or self.sampling_action_path
            action1 = self.sampling_action_path
            frames0, frames1, meta = run_two_chunk_memory_monitor(
                pipe,
                prompt=sample.get("prompt") or sample.get("description") or "A scene.",
                negative_prompt=self.sampling_negative_prompt,
                action_path=self.sampling_action_path,
                chunk0_action_path=action0,
                chunk1_action_path=action1,
                first_frame_pil=first_frame,
                context_memory_frames=self.context_memory_frames,
                chunk_frames=self.sampling_num_frames,
                h=self.sampling_height,
                w=self.sampling_width,
                seed=42 + self.step_count + int(getattr(accelerator, "process_index", 0) or 0),
                sigma_shift=5.0,
                num_inference_steps=self.sampling_num_inference_steps,
                cfg_scale=5.0,
                inference_noise_level=0.0,
                omit_context_actions=False,
                context_source=self.context_source,
                context_position=os.environ.get("CONTEXT_POSITION", "suffix"),
                context_per_frame_vae=self.context_per_frame_vae,
                device=pipe.device,
                log_prefix=f"[paper-sampling][step={self.step_count}]",
            )
            out_dir = os.path.join(self.output_path, "paper_process_sampling")
            os.makedirs(out_dir, exist_ok=True)
            rank = int(getattr(accelerator, "process_index", 0) or 0)
            tag = f"step_{self.step_count:07d}_rank{rank}"
            save_video(list(frames0) + list(frames1), os.path.join(out_dir, f"{tag}_pred.mp4"), fps=15, quality=5)
            with open(os.path.join(out_dir, f"{tag}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[paper-sampling] skipped at step {self.step_count}: {type(e).__name__}: {e}")

    def on_step_end(self, loss, accelerator=None, model=None, current_batch=None):
        self.step_count += 1
        if self.wandb_logger is not None:
            try:
                if accelerator is None or accelerator.is_main_process:
                    loss_v = float(loss.detach().float().item()) if hasattr(loss, "detach") else float(loss)
                    self.wandb_logger.log({"train/loss": loss_v, "step": self.step_count})
            except Exception as e:
                logger.debug(f"[ModelLogger] wandb log failed: {e}")
        if accelerator is not None and accelerator.is_main_process:
            self._maybe_sample_paper_process(accelerator, model, current_batch)
        if accelerator is not None and self.enable_video_sampling and self.sampling_two_chunk_memory and self.sampling_interval_steps > 0:
            accelerator.wait_for_everyone()
        if (
            self.ckpt_interval
            and accelerator is not None
            and model is not None
            and (self.step_count % self.ckpt_interval == 0)
        ):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                try:
                    path = os.path.join(self.output_path, f"Step-{self.step_count}.safetensors")
                    self._save_step_or_epoch_ckpt(accelerator, model, path)
                except Exception as e:
                    logger.warning(f"[ModelLogger] step checkpoint save failed: {e}")

    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            try:
                path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
                self._save_step_or_epoch_ckpt(accelerator, model, path)
            except Exception as e:
                logger.warning(f"[ModelLogger] epoch checkpoint save failed: {e}")


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 1,
    seed: int = 42,
    spike_threshold: float = 5.0,
    resume_step_count: int = 0,
    memory_bank: Optional[TrainingMemoryBank] = None,
    use_memory_bank: bool = False,
    memory_retrieve_num: int = 4,
    enable_fov_retrieval: bool = False,
    retrieval_method: str = "fov",  # fov | latent_sim
    latent_retrieval_dir: Optional[str] = None,
    dataset_base_path: str = None,
    fov_retriever: Optional[FOVMemoryRetriever] = None,
    context_memory_frames: int = 5,
    prev_chunk_frames: int = 81,
    fov_top_k: int = 4,  # Number of overlap frames to retrieve. GT frame 0 will be added automatically.
    use_rt_relative: bool = False,  # Experiment 1_4_2: Use RT relative conversion (aligned with Context-as-Memory)
    strict_overlap_context: bool = False,
    fov_vis_interval: int = 0,
    fov_vis_max_saves: int = 0,
    output_path: Optional[str] = None,
    dataset_repeat: int = 1,  # Add dataset_repeat parameter for step calculation
    trainable_dit_modules: Optional[str] = None,  # Comma-separated: only unfreeze these DiT modules (e.g. camera_encoder,block_self_attn). None = train full DiT.
    use_camera_encoder: bool = False,  # exp1_4_3: use CameraEncoder (action_mlp unused -> need find_unused_parameters)
    num_workers: int = 0,  # DataLoader workers: 0=main process, >0=parallel preload (recommend 4 for video)
    context_source: str = "fov",
    max_train_steps: int = 0,
    progress_total_steps: int = 0,
):
    prev_chunk_frames = int(prev_chunk_frames or 81)
    # VideoDataset can return None when file loading fails; keep distributed batches aligned.
    def collate_fn(batch):
        valid_batch = [item for item in batch if item is not None]
        return valid_batch or None
    
    num_workers = max(0, int(num_workers))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        pin_memory=(num_workers > 0 and torch.cuda.is_available()),
    )
    if num_workers > 0:
        logger.info(f"[DataLoader] num_workers={num_workers}, persistent_workers=True, pin_memory={torch.cuda.is_available()} (data preload parallel to GPU)")
    
    import os
    timeout_seconds = int(os.environ.get('TORCH_DISTRIBUTED_DEFAULT_TIMEOUT', 2400))
    os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = str(timeout_seconds)
    logger.info(f"[Timeout Config] Setting TORCH_DISTRIBUTED_DEFAULT_TIMEOUT={timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    
    # Conditional context paths can leave parameters unused on some iterations.
    need_find_unused = bool(use_camera_encoder) or bool(getattr(model_logger, "context_drop_prob", 0.0) > 0.0)
    if need_find_unused:
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
        logger.info("[DDP] find_unused_parameters=True (conditional modules / context_drop_prob enabled)")
    else:
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    if getattr(model_logger, 'enable_video_sampling', False) and model_logger.total_steps is not None:
        dataset_size = len(dataset)
        num_processes = accelerator.num_processes
        effective_dataset_size = dataset_size * dataset_repeat
        total_steps_per_gpu = (effective_dataset_size * num_epochs) // (gradient_accumulation_steps * num_processes * per_device_train_batch_size)
        total_steps_global = total_steps_per_gpu * num_processes
        model_logger.total_steps = total_steps_global
        
        if accelerator.is_main_process:
            logger.info("="*80)
            logger.info("[Step Calculation] Corrected total_steps after accelerator.init")
            logger.info("="*80)
            logger.info(f"  Dataset size (unique samples): {dataset_size}")
            logger.info(f"  Dataset repeat: {dataset_repeat}")
            logger.info(f"  Effective dataset size: {effective_dataset_size} (unique * repeat)")
            logger.info(f"  Number of epochs: {num_epochs}")
            logger.info(f"  Number of GPUs: {num_processes}")
            logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
            logger.info(f"  Per-device batch size: {per_device_train_batch_size}")
            logger.info(f"  Total samples to process: {effective_dataset_size * num_epochs}")
            logger.info(f"  Steps per GPU: ~{total_steps_per_gpu}")
            logger.info(f"  Total steps (global): {total_steps_global}")
            logger.info("")
            logger.info(f"  ✓ Each GPU will process ~{total_steps_per_gpu} steps")
            logger.info(f"  ✓ This ensures traversal of all {effective_dataset_size} samples")
            logger.info(f"    ({dataset_size} unique samples × {dataset_repeat} repeats)")
            logger.info(f"  ✓ Over {num_epochs} epoch(s)")
            logger.info("="*80)
    
    pre_loss = None
    step = resume_step_count
    traj_loss = 0.0
    if resume_step_count > 0:
        adaptation_steps = max(200, resume_step_count // 100)
        spike_detection_start_step = resume_step_count + adaptation_steps
        logger.info(f"Resuming from step {resume_step_count}, spike detection will start at step {spike_detection_start_step} (after {adaptation_steps} adaptation steps)")
    else:
        spike_detection_start_step = 100

    for epoch_id in range(num_epochs):
        epoch_seed = seed + epoch_id
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        
        if resume_step_count > 0 and epoch_id == 0:
            estimated_skip = resume_step_count // gradient_accumulation_steps
            if estimated_skip > 0:
                logger.info(f"Skipping {estimated_skip} data samples to resume from step {resume_step_count}...")
                dataloader_iter = iter(dataloader)
                for _ in tqdm(range(estimated_skip), desc="Skipping data", unit="samples", leave=False):
                    try:
                        next(dataloader_iter)
                    except StopIteration:
                        break
                dataloader = dataloader_iter
                logger.info(f"Successfully skipped {estimated_skip} data samples, resuming training...")
        
        # Track consecutive None data to detect if we're stuck in a loop
        consecutive_none_count = 0
        max_consecutive_none = 100  # If we get 100 consecutive None values, something is wrong
        
        progress_total = int(progress_total_steps or 0)
        if progress_total <= 0:
            try:
                progress_total = len(dataloader)
            except TypeError:
                progress_total = None
        progress_bar = tqdm(
            dataloader,
            total=progress_total,
            initial=resume_step_count if progress_total_steps else 0,
            desc="Training steps",
            unit="step",
        )
        for data_idx, data in enumerate(progress_bar):
            # Handle None data (can happen if all files in batch fail to load)
            if data is None:
                consecutive_none_count += 1
                if consecutive_none_count >= max_consecutive_none:
                    logger.error(f"Received {max_consecutive_none} consecutive None data samples. This suggests a serious dataset issue. Stopping training.")
                    raise ValueError(f"Too many consecutive None data samples ({max_consecutive_none}). Check dataset files.")
                
                # Log warning but continue (will skip this step)
                if consecutive_none_count <= 10 or consecutive_none_count % 10 == 0:
                    logger.warning(f"Received None data at index {data_idx} (consecutive: {consecutive_none_count}). This may indicate missing or corrupted files. Skipping...")
                
                # Still increment step to keep step_count synchronized
                step += 1
                dummy_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=False)
                model_logger.on_step_end(dummy_loss, accelerator, model, current_batch=samples)
                continue
            
            # Reset consecutive None counter when we get valid data
            consecutive_none_count = 0
            
            # Normalize to list of samples for batch processing (per_device_train_batch_size > 1)
            samples = data if isinstance(data, list) else [data]
            
            # Simplified context-based retrieval OR replay/prev_chunk_tail (aligned with multichunk eval)
            context_retrieval_success = True  # Set False if any sample fails (for strict mode)
            _umodel = accelerator.unwrap_model(model)
            _cm_frames = int(getattr(_umodel, "context_memory_frames", context_memory_frames) or context_memory_frames)
            _cs = (context_source or "fov").strip().lower()
            if _cs not in ("fov", "replay", "prev_chunk_tail"):
                _cs = "fov"

            if _cs == "replay" and dataset_base_path:
                try:
                    from .context_chunk_utils import (
                        replay_context_global_indices,
                        replay_context_actions_from_segment_actions,
                        synthetic_replay_context_from_segment,
                    )
                except ImportError:
                    from context_chunk_utils import (
                        replay_context_global_indices,
                        replay_context_actions_from_segment_actions,
                        synthetic_replay_context_from_segment,
                    )
                for d in samples:
                    vf = d.get("video") or []
                    n_seg = min(int(prev_chunk_frames), len(vf)) if vf else 0
                    ctx_pil = synthetic_replay_context_from_segment(vf, n_seg, _cm_frames) if n_seg > 0 else None
                    if not ctx_pil:
                        context_retrieval_success = False
                        break
                    d["context_frames"] = ctx_pil
                    d["context_source"] = "replay_synthetic"
                    acts = d.get("actions")
                    if isinstance(acts, list) and len(acts) >= n_seg:
                        ra = replay_context_actions_from_segment_actions(acts[:n_seg], n_seg, _cm_frames)
                        if ra is not None:
                            d["context_actions"] = ra
                    try:
                        sf = int(d.get("start_frame", 0) or 0)
                    except (TypeError, ValueError):
                        sf = 0
                    idxs = replay_context_global_indices(n_seg, _cm_frames)
                    d["context_frame_indices"] = [sf + int(i) for i in idxs]

            elif _cs == "prev_chunk_tail" and dataset_base_path:
                try:
                    from .context_chunk_utils import load_prev_chunk_tail_from_disk, load_prev_chunk_tail_rt_actions
                except ImportError:
                    from context_chunk_utils import load_prev_chunk_tail_from_disk, load_prev_chunk_tail_rt_actions
                _ctx_pos = os.environ.get("CONTEXT_POSITION", "suffix").strip().lower()
                _nearest_first = (_ctx_pos == "suffix")
                for d in samples:
                    try:
                        sf = int(d.get("start_frame", 0) or 0)
                    except (TypeError, ValueError):
                        sf = 0
                    vn = d.get("video_name", "")
                    pil_list, idxs = load_prev_chunk_tail_from_disk(
                        dataset_base_path, str(vn), sf, _cm_frames, nearest_first=_nearest_first
                    )
                    if not pil_list:
                        context_retrieval_success = False
                        break
                    d["context_frames"] = pil_list
                    d["context_frame_indices"] = list(idxs) if idxs else []
                    d["context_source"] = "prev_chunk_tail"
                    ra, _ = load_prev_chunk_tail_rt_actions(
                        dataset_base_path,
                        str(vn),
                        sf,
                        _cm_frames,
                        use_rt_relative=use_rt_relative,
                        nearest_first=_nearest_first,
                    )
                    if ra:
                        d["context_actions"] = ra

            elif enable_fov_retrieval and dataset_base_path:
                for d in samples:
                    try:
                        if retrieval_method == "latent_sim":
                            (
                                context_frames,
                                context_actions,
                                context_indices,
                                ref_frame_idx,
                                video_name,
                                source,
                            ) = retrieve_context_frames_advanced(
                                data=d,
                                dataset_base_path=dataset_base_path,
                                top_k=fov_top_k,
                                drop_overlap_probability=0.1,
                                use_rt_relative=use_rt_relative,
                                retrieval_method="latent_sim",
                                latent_retrieval_dir=latent_retrieval_dir,
                                strict_overlap_labels=strict_overlap_context,
                            )
                        else:
                            (
                                context_frames,
                                context_actions,
                                context_indices,
                                ref_frame_idx,
                                video_name,
                                source,
                            ) = retrieve_fov_context_frames(
                                data=d,
                                dataset_base_path=dataset_base_path,
                                fov_retriever=fov_retriever,  # unused in simplified retrieval, kept for compat
                                top_k=fov_top_k,  # fov_top_k is number of overlap frames (4), GT frame 0 will be added automatically
                                use_precomputed_overlaps=True,
                                strict_overlap_labels=strict_overlap_context,
                                allow_realtime_fallback=(not strict_overlap_context),
                                allow_segment_fallback=(not strict_overlap_context),
                            )

                        if context_frames and len(context_frames) > 0:
                            # Use retrieved frames as context
                            d["context_frames"] = context_frames
                            if context_actions:
                                d["context_actions"] = context_actions
                            # Store retrieval metadata for visualization/debugging
                            d["context_frame_indices"] = context_indices
                            d["context_ref_frame_idx"] = ref_frame_idx
                            d["context_video_name"] = video_name
                            d["context_source"] = source
                        else:
                            context_retrieval_success = False
                    except Exception as e:
                        context_retrieval_success = False
                        if step % 100 == 0 and accelerator.is_main_process:
                            logger.warning(f"Context retrieval failed for sample: {e}")
                        if accelerator.is_main_process and step == 1 and d is samples[0]:
                            gt_video_name = d.get("video_name", "unknown")
                            start_frame = d.get("start_frame", None)
                            end_frame = d.get("end_frame", None)
                            video_frames_count = len(d.get("video", [])) if isinstance(d.get("video"), list) else 0
                            logger.info("=" * 80)
                            logger.info(f"[STEP 1 CHECK] Context retrieval failed: {e}")
                            logger.info(f"GT Video: {gt_video_name}, start={start_frame}, end={end_frame}, frames={video_frames_count}")
                            logger.info("=" * 80)
                        break
                        
            # Strict mode: if we require context but retrieval failed, skip this step
            _need_ctx_strict = (
                strict_overlap_context
                and (not context_retrieval_success)
                and (
                    enable_fov_retrieval
                    or (context_source or "fov").strip().lower() in ("replay", "prev_chunk_tail")
                )
            )
            if _need_ctx_strict:
                if step % 50 == 0 and accelerator.is_main_process:
                    logger.warning(f"[CONTEXT][STRICT] No context at step={step}, skipping this training sample.")
                step += 1
                dummy_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=False)
                model_logger.on_step_end(dummy_loss, accelerator, model, current_batch=samples)
                continue

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # One forward over full batch: data is list of B dicts when per_device_train_batch_size > 1
                # Main loss on current batch
                loss = model(data)

                step += 1
                if traj_loss == 0.0:
                    traj_loss = loss.item()
                else:
                    alpha = 0.01
                    traj_loss = (1 - alpha) * traj_loss + alpha * loss.item()
                
                if step >= spike_detection_start_step and traj_loss > 0:
                    relative_loss = loss.item() / traj_loss
                    if resume_step_count > 0 and step < resume_step_count + 500:
                        effective_threshold = spike_threshold * 1.5
                    else:
                        effective_threshold = spike_threshold
                    
                    should_skip = relative_loss > effective_threshold
                    # Keep the skip decision identical across ranks to avoid DDP hangs.
                    skip_t = torch.tensor(1.0 if should_skip else 0.0, device=accelerator.device, dtype=torch.float32)
                    if accelerator.num_processes > 1:
                        dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
                    skip_global = skip_t.item() > 0.5
                    
                    if skip_global:
                        if accelerator.is_main_process:
                            logger.warning(f"Spike detected at step {step} (loss={loss.item():.4f}, traj_loss={traj_loss:.4f}, ratio={relative_loss:.2f}), sync skip across all ranks")
                        dummy_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=False)
                        model_logger.on_step_end(dummy_loss, accelerator, model, current_batch=samples)
                        del loss
                        torch.cuda.empty_cache()
                        continue
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(loss, accelerator, model, current_batch=samples)
                scheduler.step()

                if max_train_steps and step >= max_train_steps:
                    if progress_total_steps:
                        progress_bar.n = min(step, progress_bar.total) if progress_bar.total is not None else step
                        progress_bar.refresh()
                    if accelerator.is_main_process:
                        logger.info(f"[TRAIN] Reached max_train_steps={max_train_steps}; stopping without epoch checkpoint.")
                    accelerator.wait_for_everyone()
                    return
            
        model_logger.on_epoch_end(accelerator, model, epoch_id)



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
                    try:
                        from .rt_utils import flip_yaw_rt_list
                    except ImportError:
                        from rt_utils import flip_yaw_rt_list
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
                    try:
                        from .rt_utils import flip_yaw_rt_list
                    except ImportError:
                        from rt_utils import flip_yaw_rt_list
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
                if hasattr(self, 'context_fixed_noise_std') and self.context_fixed_noise_std is not None:
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
            if hasattr(self, "context_fixed_noise_std") and self.context_fixed_noise_std is not None:
                inputs["context_fixed_noise_std"] = self.context_fixed_noise_std
        inputs = self._ensure_input_latents(inputs, strict=True)
        return self.pipe.training_loss(**models, **inputs)


if __name__ == "__main__":
    parser = wan_parser()
    def _add_arg_if_missing(*args, **kwargs):
        if args and args[0] in parser._option_string_actions:
            return
        parser.add_argument(*args, **kwargs)

    for name, kwargs in [
        ("--tokenizer_path", dict(type=str, default=None, help="Local tokenizer path.")),
        ("--wandb_run_name", dict(type=str, default=None)),
        ("--ckpt_interval", dict(type=int, default=None)),
        ("--trainable_dit_modules", dict(type=str, default=None, help="Comma-separated DiT modules to unfreeze.")),
        ("--num_workers", dict(type=int, default=0, help="DataLoader workers.")),
        ("--max_train_steps", dict(type=int, default=0, help="Stop after N optimizer steps.")),
        ("--progress_total_steps", dict(type=int, default=0, help="tqdm total steps override.")),
        ("--resume_from_checkpoint", dict(type=str, default=None)),
        ("--context_memory_frames", dict(type=int, default=8)),
        ("--training_mode", dict(type=str, default="predict", choices=["predict", "context", "condition"])),
        ("--context_drop_prob", dict(type=float, default=0.0)),
        ("--retrieval_method", dict(type=str, default="fov", choices=["fov", "latent_sim"])),
        ("--latent_retrieval_dir", dict(type=str, default=None)),
        ("--fov_top_k", dict(type=int, default=4)),
        ("--context_attention_weight", dict(type=float, default=1.0)),
        ("--context_temporal_decay", dict(type=float, default=1.0)),
        ("--spike_threshold", dict(type=float, default=5.0)),
        ("--spatial_memory_tokens", dict(type=int, default=64)),
        ("--spatial_memory_grid", dict(type=int, default=8)),
        ("--spatial_memory_inject_mode", dict(type=str, default="concat_text", choices=["concat_text", "none", "cross_attn_readout"])),
        ("--framepack_ratio", dict(type=int, default=2)),
        ("--framepack_length_strategy", dict(type=str, default="distance_merge", choices=["distance_merge", "mean", "uniform", "recent_weighted", "weighted_recent", "packed_multiscale"])),
        ("--framepack_recent_keep_ratio", dict(type=float, default=0.5)),
        ("--framepack_multiscale_w2", dict(type=float, default=0.25)),
        ("--framepack_multiscale_w4", dict(type=float, default=0.15)),
        ("--context_source", dict(type=str, default="fov", choices=["fov", "replay", "prev_chunk_tail"])),
        ("--ssm_num_blocks_hint", dict(type=int, default=21)),
        ("--ssm_every_n_blocks", dict(type=int, default=4)),
        ("--videossm_kernel_size", dict(type=int, default=3)),
        ("--videossm_expand", dict(type=int, default=2)),
        ("--videossm_every_n_blocks", dict(type=int, default=4)),
        ("--sampling_interval_steps", dict(type=int, default=0)),
        ("--sampling_negative_prompt", dict(type=str, default="oversaturated colors, overexposed, static, blurry details")),
        ("--sampling_height", dict(type=int, default=352)),
        ("--sampling_width", dict(type=int, default=640)),
        ("--sampling_num_frames", dict(type=int, default=81)),
        ("--sampling_num_inference_steps", dict(type=int, default=50)),
        ("--sampling_action_path", dict(type=str, default=None)),
        ("--sampling_two_chunk_action_path", dict(type=str, default=None)),
        ("--sampling_eval_dataset_base", dict(type=str, default=None)),
        ("--sampling_eval_metadata_path", dict(type=str, default=None)),
        ("--samples_per_epoch", dict(type=int, default=0)),
        ("--camera_encoder_scale", dict(type=float, default=1.0)),
        ("--camera_inject_mode", dict(type=str, default="post", choices=["post", "pre_norm", "pre_qkv", "pre_qkv_post", "pre_modulate", "pre_qkv_gated"])),
    ]:
        _add_arg_if_missing(name, **kwargs)

    for name in [
        "--save_full_model", "--add_action_attn", "--action_use_temporal_attention",
        "--action_inject_after_spatial_attn", "--use_camera_encoder", "--camera_encoder_shallow",
        "--camera_encoder_separate_t_r", "--camera_encoder_explicit_yaw", "--yaw_flip_aug",
        "--camera_encoder_sincos_yaw", "--camera_encoder_r_mlp_no_layernorm",
        "--add_camera_outside_gate", "--no_camera_encoder_zero_init",
        "--camera_encoder_full_zero_init", "--enable_context_memory", "--context_per_frame_vae",
        "--cfg_target_only", "--enable_fov_retrieval", "--use_rt_relative",
        "--strict_overlap_context", "--use_anchor_frame", "--use_spatial_memory",
        "--use_spatial_memory_legacy", "--use_framepack_memory", "--use_framepack_length_compress",
        "--use_block_wise_ssm", "--use_videossm_hybrid", "--sampling_two_chunk_memory",
    ]:
        _add_arg_if_missing(name, action="store_true")

    for name, kwargs in [
        ("--per_device_train_batch_size", dict(type=int, default=None)),
        ("--timestep_shift", dict(type=float, default=1.0)),
        ("--action_base_path", dict(type=str, default=None)),
        ("--ckpt_path", dict(type=str, default=None)),
        ("--cam_position_scale", dict(type=float, default=0.01)),
        ("--resume_from", dict(type=str, default=None)),
        ("--verify_ckpt_step", dict(type=int, default=0)),
        ("--verify_high_noise_first_steps", dict(type=int, default=0)),
        ("--moc_temperature", dict(type=float, default=1.0)),
        ("--moc_top_k", dict(type=int, default=0)),
        ("--prev_chunk_frames", dict(type=int, default=81)),
        ("--implicit_type", dict(type=str, default="summary")),
        ("--context_compressor_ratio", dict(type=int, default=2)),
        ("--episodic_buffer_size", dict(type=int, default=0)),
        ("--episodic_replay_interval", dict(type=int, default=0)),
        ("--episodic_replay_weight", dict(type=float, default=0.0)),
    ]:
        _add_arg_if_missing(name, **kwargs)
    for name in [
        "--enable_video_sampling", "--sampling_atomic_left_right", "--sampling_four_prompts",
        "--sampling_two_prompts", "--train_action_module", "--train_cam_pose",
        "--action_module_only", "--use_moc", "--unified_implicit", "--use_implicit_memory",
        "--use_memory_v2v_compressor", "--use_slow_fast_memory", "--use_entity_memory",
        "--use_episodic_memory",
    ]:
        _add_arg_if_missing(name, action="store_true")
    args = parser.parse_args()
    def _arg(name, default=None):
        return getattr(args, name, default)

    def _normalize_and_validate_args():
        # Backward-compat mappings
        if _arg("per_device_train_batch_size", None) is None:
            args.per_device_train_batch_size = int(_arg("batch_size", 1) or 1)
        if _arg("sampling_atomic_left_right", False) and not _arg("sampling_two_chunk_memory", False):
            # Legacy monitor intent maps to current two-chunk monitor.
            args.sampling_two_chunk_memory = True
        if _arg("enable_video_sampling", False) and int(_arg("sampling_interval_steps", 0) or 0) <= 0:
            args.sampling_interval_steps = 1000

        # Keep paper-style block-wise SSM and legacy VideoSSM hybrid explicitly separated.
        if _arg("use_block_wise_ssm", False) and _arg("use_videossm_hybrid", False):
            raise ValueError(
                "--use_block_wise_ssm and --use_videossm_hybrid are mutually exclusive; "
                "use block-wise SSM for paper-aligned runs or VideoSSM hybrid for legacy baselines."
            )

        # Explicit retrieval strategy visibility: default fov, latent_sim degrades to fov when cache dir is absent.
        if _arg("retrieval_method", "fov") == "latent_sim":
            if not _arg("latent_retrieval_dir", None):
                logger.warning("retrieval_method=latent_sim but latent_retrieval_dir is empty; runtime will fallback to fov retrieval.")
            else:
                logger.info(f"retrieval_method=latent_sim latent_retrieval_dir={args.latent_retrieval_dir}")
        else:
            logger.info("retrieval_method=fov")

        # 2-chunk sampling defaults: keep left/right_45 semantics compatible with existing shell wrappers.
        if _arg("sampling_two_chunk_action_path", None) in (None, ""):
            args.sampling_two_chunk_action_path = _arg("sampling_action_path", None)

    _normalize_and_validate_args()

    resume_step_count = 0
    if args.resume_from_checkpoint is not None:
        if (_arg('trainable_dit_modules', None) or "").strip() or _arg('resume_weights_only', False):
            logger.info("resume_from_checkpoint used for weights only (trainable_dit_modules set or resume_weights_only), step count starts from 0, no skip data")
            resume_step_count = 0
        else:
            checkpoint_filename = os.path.basename(args.resume_from_checkpoint)
            step_match = re.search(r'Step-(\d+)', checkpoint_filename)
            epoch_match = re.search(r'epoch-(\d+)', checkpoint_filename)
            if step_match:
                resume_step_count = int(step_match.group(1))
                logger.info(f"Resuming from step {resume_step_count} (extracted from checkpoint filename)")
            elif epoch_match:
                logger.info(f"Resuming from epoch checkpoint (epoch-{epoch_match.group(1)}), step count will start from 0")
                resume_step_count = 0
            else:
                logger.warning("Could not extract step count from checkpoint filename, starting from step 0")
    
    set_seed(42)
    
    memory_bank = None
    use_memory_bank = False

    import inspect
    try:
        sig = inspect.signature(VideoDataset.__init__)
        sig_params = set(sig.parameters.keys())
    except Exception:
        sig_params = set()
    
    # Build VideoDataset arguments - only include parameters that VideoDataset accepts
    # According to the signature, VideoDataset only accepts: args, base_path, metadata_path, etc.
    # The ICL parameters (enable_icl, icl_num_examples, icl_context_frames) should be passed via args
    # Make sure args has these attributes set
    for _name, _default in (("enable_icl", False), ("icl_num_examples", 2), ("icl_context_frames", 8)):
        if not hasattr(args, _name):
            setattr(args, _name, _default)
    
    if _arg('train_cam_pose', False):
        dataset = CamVideoDataset(args=args)
    else:
        dataset_kwargs = {"args": args}
        if "action_base_path" in sig_params:
            dataset_kwargs["action_base_path"] = getattr(args, "action_base_path", None)
        dataset = VideoDataset(**dataset_kwargs)

    def _log_dataset_validation(ds):
        ds_size = len(ds)
        ds_repeat = _arg('dataset_repeat', 1) or 1
        logger.info(
            f"[Dataset] size={ds_size}, repeat={ds_repeat}, "
            f"epochs={args.num_epochs}, total_samples={ds_size * ds_repeat * args.num_epochs}"
        )

    _log_dataset_validation(dataset)
    
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=_arg('tokenizer_path', None),
        trainable_models=_arg('trainable_models', None),
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        resume_from_checkpoint=args.resume_from_checkpoint,
        dataset_base_path=_arg('dataset_base_path', None),
        enable_context_memory=_arg('enable_context_memory', False),
        context_drop_prob=_arg('context_drop_prob', 0.0),
        context_drop_seed=42,
        omit_context_actions=_arg('omit_context_actions', False) or (_arg('context_memory_frames', 8) == 1),  # ctx=1: no context action injection
        context_noise_prob=_arg('context_noise_prob', 0.0),
        context_noise_std=_arg('context_noise_std', 0.02),
        context_fixed_noise_std=_arg('context_fixed_noise_std', None),
        context_memory_frames=_arg('context_memory_frames', 8),
        context_per_frame_vae=_arg('context_per_frame_vae', False),
        training_mode=_arg('training_mode', 'predict'),
        teacher_forcing_prob=_arg('teacher_forcing_prob', 0.0),
        yaw_flip_aug=_arg('yaw_flip_aug', False),
        context_source=_arg('context_source', 'fov'),
        use_framepack_memory=_arg('use_framepack_memory', False),
        context_temporal_decay=_arg('context_temporal_decay', 1.0),
        context_attention_weight=_arg('context_attention_weight', 1.0),
        use_framepack_length_compress=_arg('use_framepack_length_compress', False),
        framepack_ratio=_arg('framepack_ratio', 2),
        framepack_length_strategy=_arg('framepack_length_strategy', 'distance_merge'),
        framepack_recent_keep_ratio=_arg('framepack_recent_keep_ratio', 0.5),
        framepack_multiscale_w2=_arg('framepack_multiscale_w2', 0.25),
        framepack_multiscale_w4=_arg('framepack_multiscale_w4', 0.15),
        use_spatial_memory=_arg('use_spatial_memory', False),
        use_spatial_memory_legacy=_arg('use_spatial_memory_legacy', False),
        spatial_memory_tokens=_arg('spatial_memory_tokens', 64),
        spatial_memory_inject_mode=_arg('spatial_memory_inject_mode', 'concat_text'),
        timestep_shift=float(_arg('timestep_shift', 1.0) or 1.0),
    )

    # ── VWM-style: Replace DiT blocks with DiTBlock_w_Action ──
    _use_cam_pose = bool(_arg('train_cam_pose', False))
    if _arg('train_action_module', False) or _use_cam_pose:
        dit = model.pipe.dit
        old_blocks = dit.blocks
        has_image_input = getattr(dit, 'has_image_input', False)
        dim = dit.dim
        num_heads = dit.num_heads
        ffn_dim = dit.ffn_dim
        eps = getattr(dit, 'eps', 1e-6)

        block_dtype = None
        for old_block in old_blocks:
            for p in old_block.parameters():
                block_dtype = p.dtype
                break
            if block_dtype is not None:
                break
        if block_dtype is None:
            block_dtype = torch.float32

        use_block_wise_ssm = bool(_arg('use_block_wise_ssm', False))
        use_videossm_hybrid = bool(_arg('use_videossm_hybrid', False))
        ssm_every_n = max(int(_arg('ssm_every_n_blocks', 4) or 4), 1)
        videossm_every_n = max(int(_arg('videossm_every_n_blocks', 4) or 4), 1)

        new_blocks = nn.ModuleList()
        for block_id, old_block in enumerate(old_blocks):
            attach_block_ssm = use_block_wise_ssm and (block_id % ssm_every_n == 0)
            attach_videossm = use_videossm_hybrid and (block_id % videossm_every_n == 0)
            new_block = DiTBlock_w_Action(
                has_image_input=has_image_input,
                dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, eps=eps,
                add_action_attn=_arg('add_action_attn', False),
                action_use_temporal_attention=_arg('action_use_temporal_attention', False),
                use_cam_pose=_use_cam_pose,
                use_block_wise_ssm=attach_block_ssm,
                use_videossm_hybrid=attach_videossm,
                videossm_kernel_size=int(_arg('videossm_kernel_size', 3) or 3),
                videossm_expand=int(_arg('videossm_expand', 2) or 2),
            )
            new_block = new_block.to(dtype=block_dtype, device=next(old_block.parameters()).device)
            for attr in ("self_attn", "cross_attn", "norm1", "norm2", "norm3", "ffn"):
                if hasattr(old_block, attr) and hasattr(new_block, attr):
                    getattr(new_block, attr).load_state_dict(getattr(old_block, attr).state_dict())
            if hasattr(old_block, "modulation") and hasattr(new_block, "modulation"):
                with torch.no_grad():
                    new_block.modulation.copy_(old_block.modulation.to(dtype=block_dtype))
            new_blocks.append(new_block)

        dit.blocks = new_blocks
        _mlp_type = "MLP_CamPose" if _use_cam_pose else "MLP_Action"
        logger.info(f"[VWM-style] Replaced {len(new_blocks)} DiT blocks with DiTBlock_w_Action ({_mlp_type}, zero-init)")
        if use_block_wise_ssm:
            logger.info(f"[Block-wise SSM] attached to every {ssm_every_n} DiT block(s)")
        if use_videossm_hybrid:
            logger.info(f"[VideoSSM hybrid] attached to every {videossm_every_n} DiT block(s)")

        device = next(dit.parameters()).device
        _ckpt_path = _arg('ckpt_path', None) or _arg('resume_from_checkpoint', None)
        if _ckpt_path is not None and os.path.isfile(_ckpt_path):
            ckpt = safe_load_file(_ckpt_path)
            missing, unexpected = dit.load_state_dict(ckpt, strict=False)
            dit.to(device=device)
            logger.info(f"[VWM-style] Loaded ckpt: {len(ckpt)} keys, missing={len(missing)}, unexpected={len(unexpected)}")

        if _arg('action_module_only', False):
            if _arg('add_action_attn', False):
                for block in dit.blocks:
                    for name, param in block.named_parameters():
                        if ("action_mlp" in name) or ("self_attn_with_action" in name) or ("block_wise_ssm" in name) or ("videossm_hybrid" in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
            else:
                for block in dit.blocks:
                    for name, param in block.named_parameters():
                        if "action_mlp" in name or "self_attn" in name or "block_wise_ssm" in name or "videossm_hybrid" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
        else:
            for block in dit.blocks:
                for name, param in block.named_parameters():
                    if "action_mlp" in name or "self_attn_with_action" in name or "block_wise_ssm" in name or "videossm_hybrid" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        _log_dit_freeze_summary(dit)

    _resume_from = _arg('resume_from', None)
    if _resume_from and os.path.isfile(_resume_from):
        logger.info(f"Loading full resume checkpoint: {_resume_from}")
        ckpt = safe_load_file(_resume_from)
        model.pipe.dit.load_state_dict(ckpt, strict=False)
        logger.info(f"Checkpoint loaded, resuming from step {resume_step_count}")

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_run_name=args.wandb_run_name,
        ckpt_interval=args.ckpt_interval,
        resume_step_count=resume_step_count,
        save_full_model=_arg('save_full_model', False),
        context_drop_prob=float(_arg("context_drop_prob", 0.0) or 0.0),
        enable_video_sampling=_arg("enable_video_sampling", False),
        sampling_interval_steps=int(_arg("sampling_interval_steps", 0) or 0),
        sampling_two_chunk_memory=_arg("sampling_two_chunk_memory", False),
        sampling_action_path=_arg("sampling_action_path", None),
        sampling_two_chunk_action_path=_arg("sampling_two_chunk_action_path", None),
        sampling_negative_prompt=_arg("sampling_negative_prompt", ""),
        sampling_height=int(_arg("sampling_height", 352) or 352),
        sampling_width=int(_arg("sampling_width", 640) or 640),
        sampling_num_frames=int(_arg("sampling_num_frames", 81) or 81),
        sampling_num_inference_steps=int(_arg("sampling_num_inference_steps", 50) or 50),
        context_memory_frames=int(_arg("context_memory_frames", 1) or 1),
        context_source=_arg("context_source", "replay"),
        context_per_frame_vae=_arg("context_per_frame_vae", False),
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # Setup FOV retriever for context-based memory training (also for ModelLogger sampling)
    enable_fov_retrieval = _arg('enable_fov_retrieval', False)
    fov_retriever = None
    dataset_base_path = _arg('dataset_base_path', None)
    if enable_fov_retrieval and dataset_base_path:
        fov_retriever = setup_fov_retriever_for_training(
            dataset_base_path=dataset_base_path,
            enable_fov_retrieval=True
        )
    elif enable_fov_retrieval and not dataset_base_path:
        logger.warning("enable_fov_retrieval is True but dataset_base_path is not set, FOV retrieval disabled")
        enable_fov_retrieval = False
    
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=int(_arg("per_device_train_batch_size", 1) or 1),
        spike_threshold=_arg('spike_threshold', 5.0),
        resume_step_count=resume_step_count,
        memory_bank=memory_bank,
        use_memory_bank=use_memory_bank,
        memory_retrieve_num=_arg('memory_retrieve_num', 4),
        enable_fov_retrieval=enable_fov_retrieval,
        retrieval_method=_arg('retrieval_method', 'fov'),
        latent_retrieval_dir=_arg('latent_retrieval_dir', None),
        dataset_base_path=_arg('dataset_base_path', None),
        fov_retriever=fov_retriever,
        context_memory_frames=_arg('context_memory_frames', 8),
        prev_chunk_frames=int(_arg('prev_chunk_frames', 81) or 81),
        fov_top_k=_arg('fov_top_k', 4),  # Number of overlap frames (4), GT frame 0 added automatically
        use_rt_relative=_arg('use_rt_relative', False),  # Experiment 1_4_2: RT relative conversion
        strict_overlap_context=_arg('strict_overlap_context', False),
        fov_vis_interval=_arg('fov_vis_interval', 0),
        fov_vis_max_saves=_arg('fov_vis_max_saves', 0),
        output_path=_arg('output_path', None),
        dataset_repeat=_arg('dataset_repeat', 1),  # Pass dataset_repeat for step calculation
        trainable_dit_modules=_arg('trainable_dit_modules', None),
        use_camera_encoder=_arg('use_camera_encoder', False),  # exp1_4_3: DDP find_unused_parameters
        num_workers=_arg('num_workers', 0),
        context_source=_arg('context_source', 'fov'),
        max_train_steps=int(_arg('max_train_steps', 0) or 0),
        progress_total_steps=int(_arg('progress_total_steps', 0) or 0),
    )

    if model_logger.wandb_logger is not None:
        wandb.finish()