import os, json, sys, re
import torch
import torch.nn as nn
from typing import Optional
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

from src.model_training.transformers_compat import patch_transformers_hybrid_cache

patch_transformers_hybrid_cache()

from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger as BaseModelLogger, VideoDataset, CamVideoDataset, wan_parser

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

try:
    from .fov_retrieval import FOVMemoryRetriever
    from .fov_training_integration import retrieve_fov_context_frames, setup_fov_retriever_for_training
    from .context_retrieval import retrieve_context_frames_advanced
except ImportError:
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from fov_retrieval import FOVMemoryRetriever
    from fov_training_integration import retrieve_fov_context_frames, setup_fov_retriever_for_training
    from context_retrieval import retrieve_context_frames_advanced


from src.model_training.training_modules import DiTBlock_w_Action, WanTrainingModule
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