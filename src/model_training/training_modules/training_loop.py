import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator
from tqdm import tqdm

from src.model_training.transformers_compat import patch_transformers_hybrid_cache

patch_transformers_hybrid_cache()

from diffsynth.trainers.utils import DiffusionTrainingModule
from src.model_training.fov_retrieval import FOVMemoryRetriever
from src.model_training.fov_retrieval import retrieve_context_frames_advanced, retrieve_fov_context_frames
from src.model_training.training_modules.model_logger import ModelLogger

logger = logging.getLogger(__name__)


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
    dataset_repeat: int = 1,  # Add dataset_repeat parameter for step calculation
    use_camera_encoder: bool = False,  # exp1_4_3: use CameraEncoder (action_mlp unused -> need find_unused_parameters)
    num_workers: int = 0,  # DataLoader workers: 0=main process, >0=parallel preload (recommend 4 for video)
    context_source: str = "fov",
    max_train_steps: int = 0,
    progress_total_steps: int = 0,
):
    prev_chunk_frames = int(prev_chunk_frames)
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
    
    timeout_seconds = int(os.environ.get('TORCH_DISTRIBUTED_DEFAULT_TIMEOUT', 2400))
    os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = str(timeout_seconds)
    logger.info(f"[Timeout Config] Setting TORCH_DISTRIBUTED_DEFAULT_TIMEOUT={timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    
    # Conditional context paths can leave parameters unused on some iterations.
    need_find_unused = bool(use_camera_encoder) or model_logger.context_drop_prob > 0.0
    if need_find_unused:
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
        logger.info("[DDP] find_unused_parameters=True (conditional modules / context_drop_prob enabled)")
    else:
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    if model_logger.enable_video_sampling and model_logger.total_steps is not None:
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
        
        progress_total = int(progress_total_steps)
        if progress_total <= 0:
            progress_total = len(dataloader)
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
            _cm_frames = int(_umodel.context_memory_frames)
            _cs = context_source.strip().lower()
            if _cs not in (
                "fov",
                "replay",
                "prev_chunk_tail",
                "causal_prev_prefix",
            ):
                _cs = "fov"

            if _cs == "replay" and dataset_base_path:
                from src.model_training.multichunk_sample_utils import (
                    replay_context_actions_from_segment_actions,
                    replay_context_global_indices,
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
                    sf = int(d.get("start_frame", 0) or 0)
                    idxs = replay_context_global_indices(n_seg, _cm_frames)
                    d["context_frame_indices"] = [sf + int(i) for i in idxs]

            elif _cs == "causal_prev_prefix" and dataset_base_path:
                from src.model_training.multichunk_sample_utils import (
                    load_causal_prefix_context,
                )
                if os.environ.get("CONTEXT_POSITION", "suffix").lower() != "prefix":
                    raise ValueError(
                        "causal_prev_prefix requires CONTEXT_POSITION=prefix"
                    )
                for d in samples:
                    frames, context_actions, target_actions, protocol = (
                        load_causal_prefix_context(
                            dataset_base_path,
                            str(d.get("video_name", "")),
                            int(d.get("start_frame", 0) or 0),
                            _cm_frames,
                            target_num_frames=81,
                            use_rt_relative=use_rt_relative,
                        )
                    )
                    if not frames or not context_actions or not target_actions:
                        context_retrieval_success = False
                        break
                    d["context_frames"] = frames
                    d["context_actions"] = context_actions
                    d["actions"] = target_actions
                    d["context_frame_indices"] = protocol["context_frame_indices"]
                    d["context_source"] = "causal_prev_prefix"

            elif _cs == "prev_chunk_tail" and dataset_base_path:
                from src.model_training.multichunk_sample_utils import load_prev_chunk_tail_from_disk, load_prev_chunk_tail_rt_actions
                _ctx_pos = os.environ.get("CONTEXT_POSITION", "suffix").strip().lower()
                _nearest_first = (_ctx_pos == "suffix")
                for d in samples:
                    sf = int(d.get("start_frame", 0) or 0)
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
                        break
                        
            # Strict mode: if we require context but retrieval failed, skip this step
            _need_ctx_strict = (
                strict_overlap_context
                and (not context_retrieval_success)
                and (
                    enable_fov_retrieval
                    or context_source.strip().lower()
                    in ("replay", "prev_chunk_tail", "causal_prev_prefix")
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
