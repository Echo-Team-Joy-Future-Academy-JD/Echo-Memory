import os, sys, re
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

_rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or os.environ.get("ACCELERATE_PROCESS_INDEX") or "0"
_rank = int(str(_rank_env))
_level = logging.INFO if _rank == 0 else logging.WARNING
logging.basicConfig(
    level=_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger.setLevel(_level)

current_file_abs = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_abs)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model_training.transformers_compat import patch_transformers_hybrid_cache

patch_transformers_hybrid_cache()

from diffsynth.trainers.utils import VideoDataset, CamVideoDataset, wan_parser

import diffsynth.trainers.utils as utils_module

utils_file = utils_module.__file__
if 'site-packages' in utils_file:
    logger.warning(f"Using INSTALLED diffsynth package from: {utils_file}")
else:
    logger.info(f"[VERIFIED] Using LOCAL diffsynth code from: {utils_file}")

import random
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from safetensors.torch import load_file as safe_load_file
from src.model_training.fov_training_integration import setup_fov_retriever_for_training
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


from src.model_training.training_modules.model_logger import ModelLogger
from src.model_training.training_modules.training_loop import launch_training_task


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
            args.per_device_train_batch_size = int(_arg("batch_size", 1))
        if _arg("sampling_atomic_left_right", False) and not _arg("sampling_two_chunk_memory", False):
            # Legacy monitor intent maps to current two-chunk monitor.
            args.sampling_two_chunk_memory = True
        if _arg("enable_video_sampling", False) and int(_arg("sampling_interval_steps", 0)) <= 0:
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
    
    args.enable_icl = False
    args.icl_num_examples = 2
    args.icl_context_frames = 8
    
    if _arg('train_cam_pose', False):
        dataset = CamVideoDataset(args=args)
    else:
        dataset = VideoDataset(args=args, action_base_path=args.action_base_path)

    def _log_dataset_validation(ds):
        ds_size = len(ds)
        ds_repeat = _arg('dataset_repeat', 1)
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
        timestep_shift=float(_arg('timestep_shift', 1.0)),
    )

    # ── VWM-style: Replace DiT blocks with DiTBlock_w_Action ──
    _use_cam_pose = bool(_arg('train_cam_pose', False))
    if _arg('train_action_module', False) or _use_cam_pose:
        dit = model.pipe.dit
        old_blocks = dit.blocks
        has_image_input = dit.has_image_input
        dim = dit.dim
        num_heads = dit.num_heads
        ffn_dim = dit.ffn_dim
        eps = 1e-6

        block_dtype = next(old_blocks[0].parameters()).dtype

        use_block_wise_ssm = bool(_arg('use_block_wise_ssm', False))
        use_videossm_hybrid = bool(_arg('use_videossm_hybrid', False))
        ssm_every_n = max(int(_arg('ssm_every_n_blocks', 4)), 1)
        videossm_every_n = max(int(_arg('videossm_every_n_blocks', 4)), 1)

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
                videossm_kernel_size=int(_arg('videossm_kernel_size', 3)),
                videossm_expand=int(_arg('videossm_expand', 2)),
            )
            new_block = new_block.to(dtype=block_dtype, device=next(old_block.parameters()).device)
            for attr in ("self_attn", "cross_attn", "norm1", "norm2", "norm3", "ffn"):
                getattr(new_block, attr).load_state_dict(getattr(old_block, attr).state_dict())
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
        if _ckpt_path is not None:
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
    if _resume_from:
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
        context_drop_prob=float(_arg("context_drop_prob", 0.0)),
        enable_video_sampling=_arg("enable_video_sampling", False),
        sampling_interval_steps=int(_arg("sampling_interval_steps", 0)),
        sampling_two_chunk_memory=_arg("sampling_two_chunk_memory", False),
        sampling_action_path=_arg("sampling_action_path", None),
        sampling_two_chunk_action_path=_arg("sampling_two_chunk_action_path", None),
        sampling_negative_prompt=_arg("sampling_negative_prompt", ""),
        sampling_height=int(_arg("sampling_height", 352)),
        sampling_width=int(_arg("sampling_width", 640)),
        sampling_num_frames=int(_arg("sampling_num_frames", 81)),
        sampling_num_inference_steps=int(_arg("sampling_num_inference_steps", 50)),
        context_memory_frames=int(_arg("context_memory_frames", 1)),
        context_source=_arg("context_source", "replay"),
        context_per_frame_vae=_arg("context_per_frame_vae", False),
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # Setup FOV retriever for context-based memory training (also for ModelLogger sampling)
    enable_fov_retrieval = _arg('enable_fov_retrieval', False)
    fov_retriever = None
    dataset_base_path = _arg('dataset_base_path', None)
    if enable_fov_retrieval:
        fov_retriever = setup_fov_retriever_for_training(
            dataset_base_path=dataset_base_path,
            enable_fov_retrieval=True
        )
    
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=int(_arg("per_device_train_batch_size", 1)),
        spike_threshold=_arg('spike_threshold', 5.0),
        resume_step_count=resume_step_count,
        enable_fov_retrieval=enable_fov_retrieval,
        retrieval_method=_arg('retrieval_method', 'fov'),
        latent_retrieval_dir=_arg('latent_retrieval_dir', None),
        dataset_base_path=_arg('dataset_base_path', None),
        fov_retriever=fov_retriever,
        context_memory_frames=_arg('context_memory_frames', 8),
        prev_chunk_frames=int(_arg('prev_chunk_frames', 81)),
        fov_top_k=_arg('fov_top_k', 4),  # Number of overlap frames (4), GT frame 0 added automatically
        use_rt_relative=_arg('use_rt_relative', False),  # Experiment 1_4_2: RT relative conversion
        strict_overlap_context=_arg('strict_overlap_context', False),
        dataset_repeat=_arg('dataset_repeat', 1),  # Pass dataset_repeat for step calculation
        use_camera_encoder=_arg('use_camera_encoder', False),  # exp1_4_3: DDP find_unused_parameters
        num_workers=_arg('num_workers', 0),
        context_source=_arg('context_source', 'fov'),
        max_train_steps=int(_arg('max_train_steps', 0)),
        progress_total_steps=int(_arg('progress_total_steps', 0)),
    )

    model_logger.finish()