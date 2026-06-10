import torch, os, json, sys, re, hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice
from typing import Any, Dict, Optional
import importlib
import logging

# Setup logging for sampling
logger = logging.getLogger(__name__)

# Ensure our logs (logger.info/...) are visible.
# Many runs rely on print-heavy libraries; without a handler, our INFO logs can be invisible.
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
    # Python<3.8 fallback (no force=)
    logging.basicConfig(
        level=_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger.setLevel(_level)

# CRITICAL: Ensure we use local code, not installed package
# Add project root to path first to prioritize local code
# Use absolute path to avoid cwd issues
current_file_abs = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_abs)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force reload to avoid cached bytecode issues
# Clear modules to force fresh import from local files
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

# Invalidate import caches to force Python to re-scan
importlib.invalidate_caches()

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger as BaseModelLogger, VideoDataset, CamVideoDataset, wan_parser
from diffsynth.models.wan_video_dit import SelfAttention, CrossAttention, GateModule, modulate
from diffsynth.models.memory.videossm_hybrid import HybridStateSpaceMemory
from diffsynth.models.memory.block_wise_ssm import BlockWiseStateSpaceMemory

# Verify we're using local code (simple check, no blocking operations)
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

# Fix relative import issue: use absolute import or conditional import
TrainingMemoryBank = None  # optional: set if memory_bank module exists
try:
    # Try relative import first (when run as module)
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


# ── MLP_Action / MLP_CamPose / DiTBlock_w_Action（与 VWM/src/model_training/train.py 对齐）──
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


# 固定随机种子
def set_seed(seed=42):
    """设置随机种子以确保训练的可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 确保CUDA操作的确定性
    logger.info(f"Random seed set to {seed}")


def _log_dit_freeze_summary(dit: torch.nn.Module) -> None:
    """校验并打印 DiT 各模块的冻结/解冻状态（按子模块聚合参数量）。"""
    # name 形如 "blocks.0.self_attn.to_q.weight" -> 取前缀 "blocks.0.self_attn"
    by_module: dict[str, tuple[int, bool]] = {}  # prefix -> (numel, any_trainable)
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
    logger.info("=" * 60)
    logger.info("[DiT 冻结/解冻校验]")
    logger.info(f"  TRAINABLE ({total_trainable:,} params):")
    for prefix, numel in trainable_list:
        logger.info(f"    + {prefix}: {numel:,}")
    logger.info(f"  FROZEN ({total_frozen:,} params):")
    for prefix, numel in frozen_list:
        logger.info(f"    - {prefix}: {numel:,}")
    logger.info("=" * 60)


# 设置默认种子
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
        enable_video_sampling=False,
        context_drop_prob: float = 0.0,
        sampling_interval_steps: int = 0,
        sampling_two_chunk_memory: bool = False,
        sampling_two_chunk_action_path: Optional[str] = None,
        sampling_action_path: Optional[str] = None,
        sampling_negative_prompt: str = "oversaturated colors, overexposed, static, blurry details",
        sampling_height: int = 352,
        sampling_width: int = 640,
        sampling_num_frames: int = 81,
        sampling_num_inference_steps: int = 50,
        context_memory_frames: int = 1,
        context_source: str = "fov",
        context_per_frame_vae: bool = False,
        sampling_eval_dataset_base: Optional[str] = None,
        sampling_eval_metadata_path: Optional[str] = None,
    ):
        super().__init__(output_path, remove_prefix_in_ckpt=remove_prefix_in_ckpt, state_dict_converter=state_dict_converter)
        self.wandb_run_name = wandb_run_name
        self.ckpt_interval = int(ckpt_interval) if ckpt_interval else None
        self.step_count = int(resume_step_count or 0)
        self.save_full_model = bool(save_full_model)
        self.enable_video_sampling = bool(enable_video_sampling)
        self.sampling_interval_steps = int(sampling_interval_steps or 0)
        self.sampling_two_chunk_memory = bool(sampling_two_chunk_memory)
        self.sampling_two_chunk_action_path = sampling_two_chunk_action_path
        self.sampling_action_path = sampling_action_path
        self.sampling_negative_prompt = sampling_negative_prompt
        self.sampling_height = int(sampling_height or 352)
        self.sampling_width = int(sampling_width or 640)
        self.sampling_num_frames = int(sampling_num_frames or 81)
        self.sampling_num_inference_steps = int(sampling_num_inference_steps or 50)
        self.context_memory_frames = int(context_memory_frames or 1)
        self.context_source = (context_source or "fov").strip().lower()
        self.context_per_frame_vae = bool(context_per_frame_vae)
        self.total_steps = None
        self.context_drop_prob = float(context_drop_prob or 0.0)
        self.sampling_eval_dataset_base = sampling_eval_dataset_base
        self.sampling_eval_metadata_path = sampling_eval_metadata_path
        self._sampling_eval_candidates = None
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

    def _sampling_ready(self, accelerator=None, model=None, current_batch=None) -> bool:
        return bool(
            self.enable_video_sampling
            and self.sampling_two_chunk_memory
            and self.sampling_interval_steps > 0
            and (self.step_count % self.sampling_interval_steps) == 0
            and accelerator is not None
            and model is not None
            and current_batch is not None
        )

    def _load_sampling_eval_candidates(self):
        if self._sampling_eval_candidates is not None:
            return self._sampling_eval_candidates
        candidates = []
        mp = self.sampling_eval_metadata_path
        if mp and os.path.isfile(mp):
            import csv
            with open(mp, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    vn = row.get("video_name", "")
                    p = row.get("prompt", "A scene.")
                    if vn:
                        candidates.append((vn, p))
        if self.sampling_two_chunk_memory and self.sampling_eval_dataset_base:
            min_frames = self.sampling_num_frames * 2
            filtered = []
            for vn, p in candidates:
                frames_dir = os.path.join(self.sampling_eval_dataset_base, "frames", vn)
                if os.path.isdir(frames_dir):
                    last_frame = os.path.join(frames_dir, f"{min_frames - 1:04d}.png")
                    if os.path.isfile(last_frame):
                        filtered.append((vn, p))
            if filtered:
                logger.info(f"[sampling] filtered eval candidates: {len(filtered)}/{len(candidates)} have {min_frames}+ frames (2 chunks)")
                candidates = filtered
            else:
                logger.warning(f"[sampling] no eval candidates with {min_frames}+ frames; using all {len(candidates)} (GT will be 1 chunk only)")
        self._sampling_eval_candidates = candidates
        return candidates

    def _extract_sampling_prompt_and_first_frame(self, current_batch, rank: int = 0):
        """Returns (prompt, first_frame_pil, video_name_or_None). rank offsets eval sample selection."""
        if self.sampling_eval_dataset_base:
            candidates = self._load_sampling_eval_candidates()
            if candidates:
                idx = (self.step_count + rank) % len(candidates)
                vn, prompt = candidates[idx]
                frame_path = os.path.join(self.sampling_eval_dataset_base, "frames", vn, "0000.png")
                if os.path.isfile(frame_path):
                    from PIL import Image
                    first_frame = Image.open(frame_path).convert("RGB")
                    logger.info(f"[sampling][rank{rank}] using eval sample: {vn} (idx={idx})")
                    return prompt, first_frame, vn
        prompt = current_batch.get("prompt") or current_batch.get("description") or "A scene."
        vf = current_batch.get("video")
        first_frame = vf[0] if isinstance(vf, list) and len(vf) > 0 else None
        return prompt, first_frame, None

    def _build_gt_action_json(self, video_name: str, num_frames: int = 81) -> Optional[str]:
        """Build a temp action JSON from the eval sample's GT camera trajectory (relative RT)."""
        if not self.sampling_eval_dataset_base or not video_name:
            return None
        json_path = os.path.join(self.sampling_eval_dataset_base, "jsons", f"{video_name}.json")
        if not os.path.isfile(json_path):
            return None
        try:
            from src.model_training.fov_retrieval import load_camera_pose
            from src.model_training.rt_utils import pose_to_rt, convert_rt_to_relative
        except ImportError:
            try:
                from model_training.fov_retrieval import load_camera_pose
                from model_training.rt_utils import pose_to_rt, convert_rt_to_relative
            except ImportError:
                return None
        rt_list = []
        for i in range(num_frames):
            pose = load_camera_pose(json_path, i)
            rt = pose_to_rt(pose, constrain_to_xy=True) if pose else None
            if rt is None or len(rt) < 12:
                return None
            rt_list.append(rt)
        ref_rt = rt_list[0]
        rel_rt_list = convert_rt_to_relative(rt_list, ref_rt)
        action_dict = {str(i): rel_rt_list[i] for i in range(len(rel_rt_list))}
        out_dir = os.path.join(self.output_path, "sampling_videos")
        os.makedirs(out_dir, exist_ok=True)
        # `video_name` can contain slashes (e.g. "L1/<uuid>" under mixed datasets).
        # If we use it directly in the filename, it becomes a subdirectory path and sampling will crash.
        safe_video_name = str(video_name).replace("/", "__")
        tmp_path = os.path.join(out_dir, f"_gt_action_{safe_video_name}.json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(action_dict, f, ensure_ascii=False)
        logger.info(f"[sampling] built GT action from {video_name}: yaw range "
                     f"{rel_rt_list[0][3]:.1f}..{rel_rt_list[-1][3]:.1f} (12-dim RT)")
        return tmp_path

    def _resolve_two_chunk_action_paths(self, video_name: Optional[str] = None):
        if video_name and self.sampling_eval_dataset_base:
            gt_path = self._build_gt_action_json(video_name, self.sampling_num_frames * 2)
            if gt_path is None:
                gt_path = self._build_gt_action_json(video_name, self.sampling_num_frames)
            if gt_path:
                return gt_path, gt_path
        action0 = self.sampling_two_chunk_action_path or self.sampling_action_path
        action1 = None
        if action0 and "left_45" in action0:
            cand = action0.replace("left_45", "right_45")
            if os.path.isfile(cand):
                action1 = cand
        if action1 is None:
            action1 = self.sampling_action_path
        return action0, action1

    @staticmethod
    def _load_gt_frames(dataset_base: str, video_name: str, num_frames: int, w: int, h: int,
                        two_chunk: bool = False):
        """Load GT frames from the eval dataset for side-by-side comparison.

        When two_chunk=True, attempts to load num_frames*2 frames so that both
        chunks of the generated video have corresponding GT for comparison.
        Falls back to loading as many frames as exist on disk.
        """
        from PIL import Image
        target = num_frames * 2 if two_chunk else num_frames
        frames = []
        frames_dir = os.path.join(dataset_base, "frames", video_name)
        for i in range(target):
            fp = os.path.join(frames_dir, f"{i:04d}.png")
            if os.path.isfile(fp):
                frames.append(Image.open(fp).convert("RGB").resize((w, h)))
            elif two_chunk and i >= num_frames:
                break
            else:
                frames.append(Image.new("RGB", (w, h), (0, 0, 0)))
        return frames

    @staticmethod
    def _concat_gt_gen_frames(gt_frames, gen_frames, w: int, h: int, chunk_frames: int = 81):
        """Horizontally concat [GT | Gen] for each frame pair, with text labels."""
        from PIL import Image, ImageDraw, ImageFont
        combined = []
        gt_has_two_chunks = len(gt_frames) >= chunk_frames * 2
        for i in range(len(gen_frames)):
            gt_idx = i if i < len(gt_frames) else len(gt_frames) - 1
            gt_f = gt_frames[gt_idx]
            gen_f = gen_frames[i]
            if hasattr(gt_f, "resize"):
                gt_f = gt_f.resize((w, h))
            if hasattr(gen_f, "resize"):
                gen_f = gen_f.resize((w, h))
            canvas = Image.new("RGB", (w * 2, h))
            canvas.paste(gt_f, (0, 0))
            canvas.paste(gen_f, (w, 0))
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            chunk_num = 1 if i < chunk_frames else 2
            gt_label = f"GT (chunk{chunk_num})" if gt_has_two_chunks else "GT"
            if not gt_has_two_chunks and i >= len(gt_frames):
                gt_label = "GT (repeat)"
            label_gen = f"Gen (chunk{chunk_num})"
            draw.rectangle([(0, 0), (160, 24)], fill=(0, 0, 0))
            draw.text((4, 2), gt_label, fill=(0, 255, 0), font=font)
            draw.rectangle([(w, 0), (w + 160, 24)], fill=(0, 0, 0))
            draw.text((w + 4, 2), label_gen, fill=(255, 255, 0), font=font)
            combined.append(canvas)
        return combined

    def _maybe_run_sampling(self, accelerator=None, model=None, current_batch=None):
        if not self._sampling_ready(accelerator=accelerator, model=model, current_batch=current_batch):
            return
        try:
            from diffsynth import save_video
            try:
                from src.model_training.multichunk_sample_utils import run_two_chunk_memory_monitor, sync_pipe_memory_from_training_module
            except Exception:
                from multichunk_sample_utils import run_two_chunk_memory_monitor, sync_pipe_memory_from_training_module
        except Exception as e:
            logger.warning(f"[ModelLogger] sampling import failed: {e}")
            return
        rank = getattr(accelerator, "process_index", 0) or 0
        try:
            unwrapped = accelerator.unwrap_model(model)
            pipe = getattr(unwrapped, "pipe", None)
            if pipe is None:
                return
            _ = sync_pipe_memory_from_training_module(pipe, unwrapped)
            prompt, first_frame, video_name = self._extract_sampling_prompt_and_first_frame(current_batch, rank=rank)
            if first_frame is None:
                return
            action0, action1 = self._resolve_two_chunk_action_paths(video_name)
            frames_ch0, frames_ch1, meta = run_two_chunk_memory_monitor(
                pipe,
                prompt=prompt,
                negative_prompt=self.sampling_negative_prompt,
                action_path=self.sampling_action_path,
                chunk0_action_path=action0,
                chunk1_action_path=action1,
                first_frame_pil=first_frame,
                context_memory_frames=self.context_memory_frames,
                chunk_frames=self.sampling_num_frames,
                h=self.sampling_height,
                w=self.sampling_width,
                seed=42 + self.step_count + rank,
                sigma_shift=5.0,
                num_inference_steps=self.sampling_num_inference_steps,
                cfg_scale=5.0,
                inference_noise_level=0.0,
                omit_context_actions=False,
                context_source=self.context_source,
                context_position=os.environ.get("CONTEXT_POSITION", "suffix"),
                context_per_frame_vae=self.context_per_frame_vae,
                device=pipe.device,
                log_prefix=f"[sampling][step={self.step_count}][rank{rank}]",
            )
            out_dir = os.path.join(self.output_path, "sampling_videos")
            os.makedirs(out_dir, exist_ok=True)
            gen_frames = list(frames_ch0) + list(frames_ch1)
            vn_tag = f"_{str(video_name).replace('/', '__')}" if video_name else ""

            # Save only the generated (predicted) frames — do not include GT.
            out_mp4 = os.path.join(out_dir, f"step_{self.step_count:07d}_rank{rank}{vn_tag}_pred.mp4")
            save_video(gen_frames, out_mp4, fps=15, quality=5)

            meta["rank"] = rank
            meta["video_name"] = video_name
            meta_path = os.path.join(out_dir, f"step_{self.step_count:07d}_rank{rank}{vn_tag}_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[ModelLogger] two-chunk sampling failed at step {self.step_count} rank {rank}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def on_step_end(self, loss, accelerator=None, model=None, current_batch=None):
        self.step_count += 1
        if self.wandb_logger is not None:
            try:
                if accelerator is None or accelerator.is_main_process:
                    loss_v = float(loss.detach().float().item()) if hasattr(loss, "detach") else float(loss)
                    self.wandb_logger.log({"train/loss": loss_v, "step": self.step_count})
            except Exception as e:
                logger.debug(f"[ModelLogger] wandb log failed: {e}")
        self._maybe_run_sampling(accelerator=accelerator, model=model, current_batch=current_batch)
        if self._sampling_ready(accelerator=accelerator, model=model, current_batch=current_batch):
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
    # num_workers>0: 子进程预加载数据，与 GPU 计算并行，减少等待
    # Use drop_last=True to avoid None data in distributed training when dataset size doesn't divide evenly
    # Custom collate_fn that filters out None data and ensures we get valid data
    # This is necessary because VideoDataset.__getitem__ can return None if file loading fails
    def collate_fn(batch):
        # Filter out None values
        valid_batch = [item for item in batch if item is not None]
        if len(valid_batch) == 0:
            # If all items are None, return None (will be handled in training loop)
            return None
        # Real batch: return list of samples; batch_size=1: return single dict (backward compat)
        if per_device_train_batch_size > 1:
            return valid_batch
        return valid_batch[0]
    
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
    
    # Set PyTorch distributed timeout to 40 minutes (2400 seconds) before initializing Accelerator
    # This prevents NCCL timeout during long-running operations like video sampling
    import os
    timeout_seconds = int(os.environ.get('TORCH_DISTRIBUTED_DEFAULT_TIMEOUT', 2400))  # Default to 40 minutes if not set
    # Ensure the environment variable is set (accelerate will use it when initializing process group)
    os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = str(timeout_seconds)
    logger.info(f"[Timeout Config] Setting TORCH_DISTRIBUTED_DEFAULT_TIMEOUT={timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    
    # DDP unused-params:
    # - When context is dropped (context_drop_prob>0) or optional conditioning modules are enabled
    #   (camera_encoder / implicit_memory / memory_v2v_compressor), some parameters may be unused
    # in a given iteration. Enable find_unused_parameters to avoid DDP reduction errors.
    need_find_unused = bool(use_camera_encoder) or bool(getattr(model_logger, "context_drop_prob", 0.0) > 0.0)
    if need_find_unused:
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
        logger.info("[DDP] find_unused_parameters=True (conditional modules / context_drop_prob enabled)")
    else:
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # Recalculate total_steps after accelerator.init to get accurate num_processes
    # and account for dataset_repeat
    if getattr(model_logger, 'enable_video_sampling', False) and model_logger.total_steps is not None:
        dataset_size = len(dataset)
        num_processes = accelerator.num_processes
        
        # Effective dataset size = dataset_size * dataset_repeat
        # Note: VideoDataset may multiply len() by dataset_repeat, but we account for it explicitly here
        # to ensure correct step calculation regardless of VideoDataset implementation
        effective_dataset_size = dataset_size * dataset_repeat
        
        # Total steps per GPU = (effective_dataset_size * num_epochs) / (gradient_accumulation_steps * num_processes * per_device_train_batch_size)
        total_steps_per_gpu = (effective_dataset_size * num_epochs) // (gradient_accumulation_steps * num_processes * per_device_train_batch_size)
        
        # Global total steps
        total_steps_global = total_steps_per_gpu * num_processes
        
        # Update model_logger with corrected total_steps
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
    step = resume_step_count  # 从恢复的步数开始
    traj_loss = 0.0
    # 如果从检查点恢复，需要累积足够的 step 后再启用 spike 检测
    # 恢复训练时，traj_loss 需要重新累积，所以需要更长的适应期
    # 适应期长度：至少 200 个 step，或者恢复步数的 1%（取较大值）
    if resume_step_count > 0:
        adaptation_steps = max(200, resume_step_count // 100)  # 至少 200 步，或恢复步数的 1%
        spike_detection_start_step = resume_step_count + adaptation_steps
        logger.info(f"Resuming from step {resume_step_count}, spike detection will start at step {spike_detection_start_step} (after {adaptation_steps} adaptation steps)")
    else:
        spike_detection_start_step = 100

    for epoch_id in range(num_epochs):
        # 为每个epoch设置不同的种子，但保持可重复性
        epoch_seed = seed + epoch_id
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        
        # 如果从检查点恢复，需要跳过已经处理过的数据
        # 注意：由于数据是随机打乱的，我们无法精确跳到某个数据位置
        # 显示进度条以便用户了解跳过进度
        if resume_step_count > 0 and epoch_id == 0:
            # 估算需要跳过的数据量（假设每个数据对应一个step）
            # 由于有 gradient_accumulation_steps，实际跳过的数据可能更少
            estimated_skip = resume_step_count // gradient_accumulation_steps
            if estimated_skip > 0:
                logger.info(f"Skipping {estimated_skip} data samples to resume from step {resume_step_count}...")
                # 使用迭代器并显示进度条
                dataloader_iter = iter(dataloader)
                # 显示跳过进度条，方便用户了解进度
                for _ in tqdm(range(estimated_skip), desc="Skipping data", unit="samples", leave=False):
                    try:
                        next(dataloader_iter)
                    except StopIteration:
                        break
                # 使用跳过后的迭代器继续训练
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
                model_logger.on_step_end(dummy_loss, accelerator, model)
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
                        
                    # Validation: Check data structure (only on step 1-5 for debugging, first sample only)
                    if step <= 5 and accelerator.is_main_process and d is samples[0] and context_frames and len(context_frames) > 0:
                        start_frame = d.get("start_frame", None)
                        end_frame = d.get("end_frame", None)
                        video_frames_count = len(d.get("video", [])) if isinstance(d.get("video"), list) else 0
                        expected_frames = 81  # Each sample should have 81 frames from start_frame
                        
                        logger.info(f"[Data Validation] Step {step}:")
                        logger.info(f"  video_name: {video_name}")
                        logger.info(f"  start_frame: {start_frame}")
                        logger.info(f"  end_frame: {end_frame}")
                        logger.info(f"  video_frames_count: {video_frames_count} (expected: {expected_frames})")
                        logger.info(f"  context_frames_count: {len(context_frames)} (expected: {fov_top_k + 1} = 1 first + {fov_top_k} overlap)")
                        logger.info(f"  context_indices: {context_indices}")
                        logger.info(f"  context_source: {source}")
                        
                        # Verify frame structure: video should have 81 frames from start_frame
                        if start_frame is not None:
                            if video_frames_count == expected_frames:
                                logger.info(f"  ✓ Video frames: {video_frames_count} frames from start_frame={start_frame} (correct)")
                            else:
                                logger.warning(f"  ✗ Video frames: {video_frames_count} frames (expected {expected_frames} from start_frame={start_frame})")
                            
                            # Verify context frames structure
                            if context_indices:
                                first_context_idx = context_indices[0] if context_indices else None
                                if first_context_idx == start_frame:
                                    logger.info(f"  ✓ First context frame is start_frame={start_frame} (correct)")
                                else:
                                    logger.warning(f"  ✗ First context frame={first_context_idx}, expected start_frame={start_frame}")
                                
                                # Check if overlap frames are outside target segment [start_frame, end_frame]
                                overlap_indices = context_indices[1:] if len(context_indices) > 1 else []
                                if overlap_indices and end_frame is not None:
                                    within_segment = [idx for idx in overlap_indices if start_frame <= idx <= end_frame]
                                    outside_segment = [idx for idx in overlap_indices if idx < start_frame or idx > end_frame]
                                    if within_segment:
                                        logger.warning(f"  ⚠ Some overlap frames are within target segment [start={start_frame}, end={end_frame}]: {within_segment}")
                                    if outside_segment:
                                        logger.info(f"  ✓ {len(outside_segment)} overlap frames are outside target segment (correct)")
                                elif overlap_indices:
                                    all_before_start = all(idx < start_frame for idx in overlap_indices)
                                    if all_before_start:
                                        logger.info(f"  ✓ All {len(overlap_indices)} overlap frames are before start_frame={start_frame} (correct)")
                                    else:
                                        invalid_overlaps = [idx for idx in overlap_indices if idx >= start_frame]
                                        logger.warning(f"  ✗ Some overlap frames are not before start_frame: {invalid_overlaps}")
                        
                        # Verify RT actions
                        if context_actions and context_frames and len(context_frames) > 0:
                            if isinstance(context_actions, list) and len(context_actions) > 0:
                                first_action = context_actions[0]
                                if isinstance(first_action, list):
                                    action_dim = len(first_action)
                                    if action_dim == 12:
                                        logger.info(f"  ✓ Context actions: {len(context_actions)} frames, RT dim={action_dim} (correct)")
                                    else:
                                        logger.warning(f"  ✗ Context actions: RT dim={action_dim}, expected 12")
                                else:
                                    logger.info(f"  ✓ Context actions: {len(context_actions)} frames (tensor format)")
                        else:
                            if context_frames and len(context_frames) > 0:
                                logger.warning(f"  ⚠ Context actions: empty (RT poses not loaded)")
                        
                        # Save sampling result to JSONL for eval consistency
                        if accelerator.is_main_process and context_frames and len(context_frames) > 0 and d is samples[0]:
                            try:
                                try:
                                    from .fov_training_integration import save_sampling_jsonl
                                except ImportError:
                                    from fov_training_integration import save_sampling_jsonl
                                
                                first_frame_index = context_indices[0] if context_indices else ref_frame_idx
                                
                                # Get output directory for saving sampling jsonl
                                # Use function parameter output_path or model_logger.output_path
                                output_path_for_jsonl = output_path if output_path else (model_logger.output_path if hasattr(model_logger, 'output_path') else "./")
                                sampling_jsonl_path = os.path.join(
                                    output_path_for_jsonl,
                                    "sampling_context.jsonl"
                                )
                                
                                # Get prompt and segment info from current training sample (aligned)
                                # CRITICAL FIX: Use current sample's video_name and prompt, not context retrieval's video_name
                                # Context frames may come from different videos, but prompt should match the current sample
                                current_video_name = d.get("video_name", video_name)  # Use current sample's video_name
                                prompt = d.get("prompt", None) or d.get("description", None)  # Get prompt from current sample
                                
                                # CRITICAL FIX: Clean prompt - remove video path prefix if present
                                # CSV prompts may start with "video_name.mp4 " prefix, which should be removed
                                if prompt and isinstance(prompt, str):
                                    import re
                                    # Remove video path prefix pattern: "VideoName/1234_5678.mp4 " or "VideoName.mp4 "
                                    pattern = r'^[A-Za-z0-9_]+(/[A-Za-z0-9_]+)?\.mp4\s+'
                                    prompt = re.sub(pattern, '', prompt)
                                    # Remove trailing "..." if truncated
                                    if prompt.endswith('...'):
                                        prompt = prompt[:-3].rstrip()
                                    prompt = prompt.strip()
                                start_frame = d.get("start_frame", None)
                                end_frame = d.get("end_frame", None)
                                
                                # If video_name contains .mp4 extension, remove it
                                if isinstance(current_video_name, str) and current_video_name.endswith((".mp4", ".avi")):
                                    current_video_name = os.path.splitext(current_video_name)[0]
                                
                                save_sampling_jsonl(
                                    output_path=sampling_jsonl_path,
                                    video_name=current_video_name,  # Use current sample's video_name (aligned with prompt)
                                    frame_index=first_frame_index,
                                    context_indices=context_indices,
                                    prompt=prompt,
                                    start_frame=start_frame,
                                    end_frame=end_frame,
                                    source=source,
                                    append=True,
                                )
                            except Exception as e:
                                logger.warning(f"[ModelLogger] Failed to save sampling jsonl: {e}")
                        
                        # Print detailed info for step 1
                        if accelerator.is_main_process and step == 1 and context_frames and len(context_frames) > 0:
                            import json
                            # Get GT video info
                            gt_video_name = d.get("video_name", "unknown")
                            start_frame = d.get("start_frame", None)
                            end_frame = d.get("end_frame", None)
                            video_frames_count = len(d.get("video", [])) if isinstance(d.get("video"), list) else 0
                            
                            # Prepare JSONL format data
                            step1_info = {
                                "step": step,
                                "gt_video": {
                                    "video_name": gt_video_name,
                                    "start_frame": start_frame,
                                    "end_frame": end_frame,
                                    "total_frames": video_frames_count,
                                    "ref_frame_idx": ref_frame_idx
                                },
                                "context_frames": {
                                    "count": len(context_frames),
                                    "source": source,
                                    "context_video_name": video_name,
                                    "indices": context_indices,
                                    "ref_frame_idx": ref_frame_idx
                                },
                                "context_frame_indices": context_indices  # JSONL format: list of indices
                            }
                            
                            # Print formatted info
                            logger.info("=" * 80)
                            logger.info(f"[STEP 1 CHECK] Training Sample Context Information")
                            logger.info("=" * 80)
                            logger.info(f"GT Video:")
                            logger.info(f"  - Video Name: {gt_video_name}")
                            logger.info(f"  - Start Frame: {start_frame}")
                            logger.info(f"  - End Frame: {end_frame}")
                            logger.info(f"  - Total Frames: {video_frames_count}")
                            logger.info(f"  - Ref Frame Index: {ref_frame_idx}")
                            logger.info(f"Context Frames:")
                            logger.info(f"  - Count: {len(context_frames)}")
                            logger.info(f"  - Source: {source}")
                            logger.info(f"  - Context Video Name: {video_name}")
                            logger.info(f"  - Context Frame Indices: {context_indices}")
                            logger.info(f"JSONL Format (context_frame_indices):")
                            logger.info(json.dumps({"context_frame_indices": context_indices}, ensure_ascii=False))
                            logger.info("=" * 80)
                        
                        # Removed: periodic overlap logging every 100 steps (user request)

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
                model_logger.on_step_end(dummy_loss, accelerator, model)
                continue

            # Optional: save retrieval visualization periodically (rank0 only)
            # CRITICAL: Must synchronize after file I/O to prevent other processes from waiting indefinitely
            should_save_fov_vis = (
                accelerator.is_main_process
                and context_retrieval_success
                and fov_vis_interval
                and fov_vis_interval > 0
                and (step % fov_vis_interval == 0)
                and output_path is not None
            )
            if should_save_fov_vis:
                try:
                    # Respect max saves if set (>0) - use try/except to avoid blocking on file system errors
                    if fov_vis_max_saves and fov_vis_max_saves > 0:
                        vis_root = os.path.join(output_path, "fov_retrieval_vis")
                        if os.path.exists(vis_root):
                            try:
                                existing = [d for d in os.listdir(vis_root) if d.startswith("step_")]
                                if len(existing) >= fov_vis_max_saves:
                                    # Skip saving if max reached (don't raise, just skip)
                                    logger.debug(f"[FOV-VIS] Max saves ({fov_vis_max_saves}) reached, skipping visualization")
                                    should_save_fov_vis = False  # Mark as skipped
                                else:
                                    # Proceed with save (use first sample for batch)
                                    d0 = samples[0]
                                    vis_dir = os.path.join(output_path, "fov_retrieval_vis", f"step_{step:07d}_{d0.get('video_name','')}")
                                    os.makedirs(vis_dir, exist_ok=True)
                                    from PIL import Image as _Image
                                    import json as _json
                                    frames_dir = os.path.join(dataset_base_path, "frames", d0.get("context_video_name", d0.get("video_name", "")))
                                    ref_frame_idx = d0.get("context_ref_frame_idx", d0.get("start_frame", 0))
                                    context_indices = d0.get("context_frame_indices", [])
                                    source = d0.get("context_source", "unknown")

                                    ref_path = os.path.join(frames_dir, f"{int(ref_frame_idx):04d}.png")
                                    if os.path.exists(ref_path):
                                        _Image.open(ref_path).convert("RGB").save(os.path.join(vis_dir, f"ref_{int(ref_frame_idx):04d}.png"))
                                    for idx_i, frame_idx in enumerate(context_indices):
                                        fp = os.path.join(frames_dir, f"{int(frame_idx):04d}.png")
                                        if os.path.exists(fp):
                                            _Image.open(fp).convert("RGB").save(os.path.join(vis_dir, f"ctx_{idx_i:02d}_{int(frame_idx):04d}.png"))
                                    with open(os.path.join(vis_dir, "retrieval.json"), "w", encoding="utf-8") as f:
                                        f.write(_json.dumps({
                                            "video_name": d0.get("context_video_name", d0.get("video_name", "")),
                                            "ref_frame_idx": int(ref_frame_idx),
                                            "context_indices": [int(x) for x in context_indices],
                                            "source": source,
                                        }, ensure_ascii=False, indent=2))
                            except (OSError, IOError) as e:
                                # File system errors shouldn't block training
                                logger.debug(f"[FOV-VIS] Failed to save visualization (non-blocking): {e}")
                                should_save_fov_vis = False  # Mark as failed
                        else:
                            # vis_root doesn't exist yet, create it and proceed
                            os.makedirs(vis_root, exist_ok=True)
                            d0 = samples[0]
                            vis_dir = os.path.join(output_path, "fov_retrieval_vis", f"step_{step:07d}_{d0.get('video_name','')}")
                            os.makedirs(vis_dir, exist_ok=True)
                            from PIL import Image as _Image
                            import json as _json
                            frames_dir = os.path.join(dataset_base_path, "frames", d0.get("context_video_name", d0.get("video_name", "")))
                            ref_frame_idx = d0.get("context_ref_frame_idx", d0.get("start_frame", 0))
                            context_indices = d0.get("context_frame_indices", [])
                            source = d0.get("context_source", "unknown")

                            ref_path = os.path.join(frames_dir, f"{int(ref_frame_idx):04d}.png")
                            if os.path.exists(ref_path):
                                _Image.open(ref_path).convert("RGB").save(os.path.join(vis_dir, f"ref_{int(ref_frame_idx):04d}.png"))
                            for idx_i, frame_idx in enumerate(context_indices):
                                fp = os.path.join(frames_dir, f"{int(frame_idx):04d}.png")
                                if os.path.exists(fp):
                                    _Image.open(fp).convert("RGB").save(os.path.join(vis_dir, f"ctx_{idx_i:02d}_{int(frame_idx):04d}.png"))
                            with open(os.path.join(vis_dir, "retrieval.json"), "w", encoding="utf-8") as f:
                                f.write(_json.dumps({
                                    "video_name": d0.get("context_video_name", d0.get("video_name", "")),
                                    "ref_frame_idx": int(ref_frame_idx),
                                    "context_indices": [int(x) for x in context_indices],
                                    "source": source,
                                }, ensure_ascii=False, indent=2))
                    else:
                        # No max limit, proceed with save
                        d0 = samples[0]
                        vis_dir = os.path.join(output_path, "fov_retrieval_vis", f"step_{step:07d}_{d0.get('video_name','')}")
                        os.makedirs(vis_dir, exist_ok=True)
                        from PIL import Image as _Image
                        import json as _json
                        frames_dir = os.path.join(dataset_base_path, "frames", d0.get("context_video_name", d0.get("video_name", "")))
                        ref_frame_idx = d0.get("context_ref_frame_idx", d0.get("start_frame", 0))
                        context_indices = d0.get("context_frame_indices", [])
                        source = d0.get("context_source", "unknown")

                        ref_path = os.path.join(frames_dir, f"{int(ref_frame_idx):04d}.png")
                        if os.path.exists(ref_path):
                            _Image.open(ref_path).convert("RGB").save(os.path.join(vis_dir, f"ref_{int(ref_frame_idx):04d}.png"))
                        for idx_i, frame_idx in enumerate(context_indices):
                            fp = os.path.join(frames_dir, f"{int(frame_idx):04d}.png")
                            if os.path.exists(fp):
                                _Image.open(fp).convert("RGB").save(os.path.join(vis_dir, f"ctx_{idx_i:02d}_{int(frame_idx):04d}.png"))
                        with open(os.path.join(vis_dir, "retrieval.json"), "w", encoding="utf-8") as f:
                            f.write(_json.dumps({
                                "video_name": d0.get("context_video_name", d0.get("video_name", "")),
                                "ref_frame_idx": int(ref_frame_idx),
                                "context_indices": [int(x) for x in context_indices],
                                "source": source,
                            }, ensure_ascii=False, indent=2))
                except Exception as e:
                    # Any error in visualization should not block training
                    logger.debug(f"[FOV-VIS] Visualization save failed (non-blocking): {type(e).__name__}: {e}")
                    should_save_fov_vis = False  # Mark as failed
            
            # CRITICAL: Synchronize all processes after FOV visualization save (if attempted)
            # This ensures other processes don't wait indefinitely at the next sync point (accelerator.accumulate)
            # Only sync if we're at a step where visualization might be saved (even if skipped due to max_saves)
            if fov_vis_interval and fov_vis_interval > 0 and (step % fov_vis_interval == 0):
                accelerator.wait_for_everyone()
            
            # Memory Bank: retrieve context from memory bank if enabled (fallback or if context retrieval not enabled)
            if (not context_retrieval_success) and use_memory_bank and memory_bank is not None and len(memory_bank) > 0:
                # Retrieve memory frames
                mem_frames_list, mem_actions_list, mem_metadata_list = memory_bank.retrieve_for_training(
                    num_retrieve=memory_retrieve_num
                )
                
                # Flatten memory frames and actions
                context_frames = []
                context_actions = []
                for mem_frames in mem_frames_list:
                    if mem_frames:
                        context_frames.extend(mem_frames)
                for mem_actions in mem_actions_list:
                    if mem_actions is not None:
                        context_actions.extend(mem_actions)
                
                # Add context to data (or each sample when batch)
                if context_frames:
                    for d in samples:
                        d["context_frames"] = context_frames
                        if context_actions and len(context_actions) == len(context_frames):
                            d["context_actions"] = context_actions
            
            # Memory Bank: store original data for memory bank (before adding context frames)
            # We'll add it to memory bank after training, so next iteration can use it
            original_data = None
            if use_memory_bank and memory_bank is not None:
                # Deep copy to avoid modifying original data
                import copy
                original_data = copy.deepcopy(data)
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # Note: Self-forcing removed - using standard training only
                
                # CRITICAL VALIDATION: Check prompt/context/target alignment for first 5 steps
                if step <= 5 and accelerator.is_main_process:
                    try:
                        import json
                        data_log = samples[0]  # Use first sample for alignment logging
                        gt_video_name = data_log.get("video_name", "unknown")
                        context_video_name = data_log.get("context_video_name", data_log.get("video_name", "unknown"))
                        prompt = data_log.get("prompt", None) or data_log.get("description", None)
                        start_frame = data_log.get("start_frame", None)
                        end_frame = data_log.get("end_frame", None)
                        ref_frame_idx = data_log.get("context_ref_frame_idx", data_log.get("start_frame", 0))
                        context_indices = data_log.get("context_frame_indices", [])
                        context_frames = data_log.get("context_frames", [])
                        target_frames = data_log.get("video", [])
                        
                        # Clean prompt if needed
                        if prompt and isinstance(prompt, str):
                            import re
                            pattern = r'^[A-Za-z0-9_]+(/[A-Za-z0-9_]+)?\.mp4\s+'
                            prompt = re.sub(pattern, '', prompt)
                            if prompt.endswith('...'):
                                prompt = prompt[:-3].rstrip()
                            prompt = prompt.strip()
                        
                        # Validation report
                        logger.info("=" * 100)
                        logger.info(f"[ALIGNMENT CHECK] Step {step}: Prompt/Context/Target Alignment Verification")
                        logger.info("=" * 100)
                        
                        # 1. Check prompt and GT video alignment
                        logger.info(f"1. PROMPT ↔ GT VIDEO ALIGNMENT:")
                        logger.info(f"   ✓ GT Video Name: {gt_video_name}")
                        logger.info(f"   ✓ Prompt: {prompt[:100] if prompt and len(prompt) > 100 else prompt}...")
                        logger.info(f"   ✓ Prompt Length: {len(prompt) if prompt else 0} chars")
                        logger.info(f"   {'✓' if prompt else '✗'} Prompt exists: {prompt is not None}")
                        
                        # 2. Check context and GT video alignment
                        logger.info(f"\n2. CONTEXT ↔ GT VIDEO ALIGNMENT:")
                        logger.info(f"   ✓ GT Video Name: {gt_video_name}")
                        logger.info(f"   ✓ Context Video Name: {context_video_name}")
                        logger.info(f"   {'✓' if gt_video_name == context_video_name else '⚠'} Video Name Match: {gt_video_name == context_video_name}")
                        logger.info(f"   ✓ Context Frames Count: {len(context_frames)}")
                        logger.info(f"   ✓ Context Frame Indices: {context_indices[:10]}{'...' if len(context_indices) > 10 else ''}")
                        logger.info(f"   ✓ Ref Frame Index: {ref_frame_idx}")
                        
                        # 3. Check context frame indices validity
                        if start_frame is not None and end_frame is not None:
                            context_before_target = all(idx < start_frame for idx in context_indices)
                            logger.info(f"   {'✓' if context_before_target else '✗'} Context frames before target: {context_before_target}")
                            if not context_before_target:
                                invalid_indices = [idx for idx in context_indices if idx >= start_frame]
                                logger.warning(f"   ⚠ Invalid context indices (>= start_frame={start_frame}): {invalid_indices[:5]}")
                        
                        # 4. Check target frames
                        logger.info(f"\n3. TARGET FRAMES:")
                        logger.info(f"   ✓ Target Frames Count: {len(target_frames) if isinstance(target_frames, list) else 'N/A'}")
                        logger.info(f"   ✓ Start Frame: {start_frame}")
                        logger.info(f"   ✓ End Frame: {end_frame}")
                        if start_frame is not None and end_frame is not None:
                            expected_frames = end_frame - start_frame + 1
                            actual_frames = len(target_frames) if isinstance(target_frames, list) else 0
                            logger.info(f"   {'✓' if actual_frames == expected_frames else '✗'} Frame Count Match: {actual_frames} == {expected_frames}")
                        
                        # 5. Check prompt and context alignment (critical)
                        logger.info(f"\n4. PROMPT ↔ CONTEXT ALIGNMENT (CRITICAL):")
                        if gt_video_name == context_video_name:
                            logger.info(f"   ✓ Context frames from same video as GT: {gt_video_name}")
                            logger.info(f"   ✓ Prompt should describe context frames: VALID")
                        else:
                            logger.warning(f"   ⚠ Context frames from different video!")
                            logger.warning(f"      GT Video: {gt_video_name}")
                            logger.warning(f"      Context Video: {context_video_name}")
                            logger.warning(f"      ⚠ Prompt may not match context frames!")
                        
                        # 6. Check RT / Camera Pose injection (context + target 81-frame)
                        context_actions = data_log.get("context_actions", [])
                        target_actions = data_log.get("actions", [])
                        logger.info(f"\n5. CAMERA POSE (RT) INJECTION:")
                        logger.info(f"   ✓ Context Actions (RT): {len(context_actions)} frames")
                        logger.info(f"   ✓ Target Actions (RT): {len(target_actions)} frames (81-frame camera trajectory)")
                        if context_actions and len(context_actions) == len(context_frames):
                            logger.info(f"   ✓ Context RT poses match context frames count")
                        else:
                            logger.warning(f"   ⚠ Context actions count mismatch: {len(context_actions)} != {len(context_frames)}")
                        if target_actions:
                            first_rt = target_actions[0] if target_actions else []
                            rt_dim = len(first_rt) if isinstance(first_rt, (list, tuple)) else (first_rt.shape[-1] if hasattr(first_rt, 'shape') else 0)
                            if rt_dim == 12:
                                logger.info(f"   ✓ Target RT dim=12 (t_x,t_y,t_z + R_3x3), camera control ready")
                            else:
                                logger.warning(f"   ⚠ Target RT dim={rt_dim}, expected 12")
                        else:
                            logger.warning(f"   ⚠ No target actions loaded - camera control disabled for this sample")
                        
                        # 7. Save detailed alignment info to file
                        alignment_info = {
                            "step": step,
                            "gt_video_name": gt_video_name,
                            "context_video_name": context_video_name,
                            "prompt": prompt,
                            "prompt_length": len(prompt) if prompt else 0,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "ref_frame_idx": ref_frame_idx,
                            "context_indices": context_indices,
                            "context_frames_count": len(context_frames),
                            "target_frames_count": len(target_frames) if isinstance(target_frames, list) else 0,
                            "context_actions_count": len(context_actions),
                            "target_actions_count": len(target_actions),
                            "video_name_match": gt_video_name == context_video_name,
                            "context_before_target": all(idx < start_frame for idx in context_indices) if start_frame is not None else None,
                        }
                        
                        alignment_log_path = os.path.join(
                            output_path if output_path else "./",
                            "alignment_check.jsonl"
                        )
                        try:
                            with open(alignment_log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(alignment_info, ensure_ascii=False) + "\n")
                        except Exception as e:
                            logger.warning(f"   Failed to save alignment info: {e}")
                        
                        logger.info("=" * 100)
                        logger.info("")
                    except Exception as e:
                        logger.warning(f"[ALIGNMENT CHECK] Step {step}: Validation failed: {type(e).__name__}: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                
                # Experiment 1_4_2: Early validation at step 5 to verify RT input is correctly processed
                if step == 5 and accelerator.is_main_process:
                    try:
                        data_log = samples[0]
                        has_context_actions = "context_actions" in data_log and len(data_log.get("context_actions", [])) > 0
                        has_actions = "actions" in data_log and len(data_log.get("actions", [])) > 0
                        
                        if has_context_actions:
                            context_actions = data_log["context_actions"]
                            if isinstance(context_actions, list) and len(context_actions) > 0:
                                # Check first action dimension
                                first_action = context_actions[0]
                                if isinstance(first_action, list):
                                    action_dim = len(first_action)
                                    if action_dim == 12:
                                        logger.info(
                                            f"[RT-VALIDATION] Step {step}: ✓ Context actions (RT poses) detected: "
                                            f"shape={len(context_actions)} frames, dim={action_dim} (expected 12)"
                                        )
                                    else:
                                        logger.warning(
                                            f"[RT-VALIDATION] Step {step}: ✗ Context actions dimension mismatch: "
                                            f"got {action_dim}, expected 12 (RT format)"
                                        )
                                elif hasattr(first_action, 'shape'):
                                    action_dim = first_action.shape[-1] if len(first_action.shape) > 0 else None
                                    logger.info(
                                        f"[RT-VALIDATION] Step {step}: Context actions detected: "
                                        f"shape={first_action.shape}, dim={action_dim}"
                                    )
                        
                        if has_actions:
                            actions = data_log["actions"]
                            if isinstance(actions, list) and len(actions) > 0:
                                first_action = actions[0]
                                if isinstance(first_action, list):
                                    action_dim = len(first_action)
                                    logger.info(
                                        f"[RT-VALIDATION] Step {step}: Target actions detected: "
                                        f"shape={len(actions)} frames, dim={action_dim}"
                                    )
                                elif hasattr(first_action, 'shape'):
                                    logger.info(
                                        f"[RT-VALIDATION] Step {step}: Target actions detected: "
                                        f"shape={first_action.shape}"
                                    )
                        
                        # Verify action_mlp can process the input
                        if hasattr(model, 'pipe') and hasattr(model.pipe, 'dit'):
                            dit = model.pipe.dit
                            if hasattr(dit, 'blocks') and len(dit.blocks) > 0:
                                first_block = dit.blocks[0]
                                if hasattr(first_block, 'action_mlp'):
                                    action_mlp = first_block.action_mlp
                                    expected_dim = getattr(action_mlp, 'action_dim', None)
                                    if expected_dim:
                                        logger.info(
                                            f"[RT-VALIDATION] Step {step}: ✓ action_mlp configured for "
                                            f"{expected_dim}-dim input (RT pose format)"
                                        )
                                    else:
                                        logger.warning(
                                            f"[RT-VALIDATION] Step {step}: action_mlp action_dim not found"
                                        )
                    except Exception as e:
                        logger.warning(f"[RT-VALIDATION] Step {step}: Validation check failed: {type(e).__name__}: {e}")
                
                # One forward over full batch: data is list of B dicts when per_device_train_batch_size > 1
                # Main loss on current batch
                loss = model(data)

                # 先更新 step 和 traj_loss，这样即使跳过训练，traj_loss 也会更新
                # 这对于恢复训练时 traj_loss 的稳定很重要
                step += 1
                # 使用指数移动平均来更新 traj_loss，这样即使跳过某些 step，也能保持更新
                # 使用较小的衰减因子，让 traj_loss 更稳定
                if traj_loss == 0.0:
                    # 第一次，直接使用 loss
                    traj_loss = loss.item()
                else:
                    # 使用指数移动平均：traj_loss = 0.99 * traj_loss + 0.01 * loss
                    # 这样即使跳过某些 step，traj_loss 也会缓慢更新
                    alpha = 0.01  # 较小的学习率，让 traj_loss 更稳定
                    traj_loss = (1 - alpha) * traj_loss + alpha * loss.item()
                
                # 检查 spike，但需要确保 traj_loss 不为 0 且已经累积足够的 step
                # 从检查点恢复时，需要先累积足够的 step 让 traj_loss 稳定
                # 另外，在恢复训练后的适应期内，使用更宽松的阈值
                if step >= spike_detection_start_step and traj_loss > 0:
                    # 计算相对 loss
                    relative_loss = loss.item() / traj_loss
                    # 在恢复训练后的适应期内，使用更宽松的阈值（1.5倍）
                    if resume_step_count > 0 and step < resume_step_count + 500:
                        effective_threshold = spike_threshold * 1.5
                    else:
                        effective_threshold = spike_threshold
                    
                    should_skip = relative_loss > effective_threshold
                    # 多卡同步：必须所有 rank 一致决定 skip，否则部分 rank 不参与 backward 会触发 NCCL 死锁
                    skip_t = torch.tensor(1.0 if should_skip else 0.0, device=accelerator.device, dtype=torch.float32)
                    if accelerator.num_processes > 1:
                        dist.all_reduce(skip_t, op=dist.ReduceOp.MAX)
                    skip_global = skip_t.item() > 0.5
                    
                    if skip_global:
                        if accelerator.is_main_process:
                            logger.warning(f"Spike detected at step {step} (loss={loss.item():.4f}, traj_loss={traj_loss:.4f}, ratio={relative_loss:.2f}), sync skip across all ranks")
                        # 所有 rank 一起 skip：不 backward，避免部分 rank 等待 all-reduce 导致 NCCL 超时
                        dummy_loss = torch.tensor(0.0, device=accelerator.device, requires_grad=False)
                        model_logger.on_step_end(dummy_loss, accelerator, model, current_batch=samples[0] if samples else None)
                        del loss
                        torch.cuda.empty_cache()
                        continue
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(loss, accelerator, model, current_batch=samples[0] if samples else None)
                scheduler.step()

                if max_train_steps and step >= max_train_steps:
                    if progress_total_steps:
                        progress_bar.n = min(step, progress_bar.total) if progress_bar.total is not None else step
                        progress_bar.refresh()
                    if accelerator.is_main_process:
                        logger.info(f"[SMOKE] Reached max_train_steps={max_train_steps}; stopping without epoch checkpoint.")
                    accelerator.wait_for_everyone()
                    return
            
            # Memory Bank: add current sample to memory bank after successful training step
            # When batch_size > 1, original_data is a list; use first sample for memory add
            _mem_data = original_data[0] if isinstance(original_data, list) and len(original_data) > 0 else original_data
            if use_memory_bank and memory_bank is not None and _mem_data is not None and "video" in _mem_data:
                try:
                    # Use original data (without context frames) to avoid storing context in memory
                    video_frames = _mem_data["video"]
                    video_actions = _mem_data.get("actions")
                    video_prompt = _mem_data.get("prompt", "")
                    
                    if video_frames:
                        memory_bank.add_video(
                            frames=video_frames,
                            actions=video_actions,
                            metadata={"prompt": video_prompt, "step": step}
                        )
                except Exception as e:
                    # Skip if adding to memory fails (e.g., out of memory)
                    pass

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
        
        # Stack actions: (B, T, 12), use zeros for samples without actions
        ref_action = next((a for a in actions_list if a is not None), None)
        if ref_action is not None:
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
        if ref_action is not None:
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

        # CamVideoDataset compatibility: translate VWM condition_* keys to context-memory keys.
        if isinstance(data, dict):
            data = self._translate_condition_keys(data)
        elif isinstance(data, list):
            data = [self._translate_condition_keys(d) for d in data]
        
        # Batch mode: data is list of dicts (per_device_train_batch_size > 1)
        # Full batch is used in one forward (no per-sample loop); pipeline gets batched tensors
        is_batch = isinstance(data, list) and len(data) > 0
        if is_batch:
            if self.enable_context_memory:
                return self._forward_preprocess_batch_context(data)
            return self._forward_preprocess_batch(data)
        
        # Single-sample path (per_device_train_batch_size == 1)
        # CFG-sensitive parameters
        # Validate prompt is a string
        prompt = data.get("prompt")
        if prompt is None:
            raise ValueError("data['prompt'] is missing or None")
        if not isinstance(prompt, (str, list)):
            video_name = data.get("video_name", "unknown")
            start_frame = data.get("start_frame", "unknown")
            raise TypeError(
                f"data['prompt'] must be a string or list of strings, but got {type(prompt).__name__}: {prompt}. "
                f"This occurred for video_name={video_name}, start_frame={start_frame}. "
                f"Please check your dataset loading code."
            )
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        
        # Context Memory mode: separate context and target frames
        # In this mode, context frames are kept clean (not noised) and concatenated
        # with noisy target frames before model forward pass
        video_frames = data["video"]
        context_frames = None
        context_actions = None

        def _should_drop_context(_data) -> bool:
            """Deterministic per-sample context drop for ablation.

            Uses (seed, video_name, start_frame) to avoid global RNG side-effects
            and keep results reproducible across DDP workers.
            """
            p = float(getattr(self, "context_drop_prob", 0.0) or 0.0)
            if p <= 0.0:
                return False
            if p >= 1.0:
                return True
            vn = str(_data.get("video_name", ""))
            sf = str(_data.get("start_frame", ""))
            key = f"{int(getattr(self, 'context_drop_seed', 42))}|{vn}|{sf}"
            h = hashlib.md5(key.encode("utf-8")).hexdigest()
            u = int(h[:8], 16) / 0xFFFFFFFF  # in [0, 1]
            return u < p
        
        if self.enable_context_memory:
            dropped_context = _should_drop_context(data)
            # IMPORTANT (DDP safety): keep drop decision consistent across ranks.
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    flag = torch.tensor([1 if dropped_context else 0], device=self.pipe.device, dtype=torch.int64)
                    dist.broadcast(flag, src=0)
                    dropped_context = bool(int(flag.item()))
            except Exception:
                pass
            # Extract context/condition frames from the beginning of the video
            training_mode = getattr(self, 'training_mode', 'context')  # Default to "context" for backward compatibility
            
            if training_mode == "condition":
                # Experiment 17: Condition mode
                # - Condition: First K frames (clean)
                # - Target: Full 81 frames (noisy)
                # - Loss: Computed on full 81 frames (including reconstruction of first K frames)
                # IMPORTANT (Context-as-Memory FOV retrieval alignment):
                # If training loop has provided retrieved context frames (e.g. from overlap_labels/FOV top-k),
                # we should use them as condition, instead of always using the first K frames of the segment.
                if (not dropped_context) and "context_frames" in data and len(data["context_frames"]) > 0:
                    context_frames = data["context_frames"]
                    if len(context_frames) != self.context_memory_frames:
                        import warnings
                        warnings.warn(
                            f"Warning: retrieved context_frames has {len(context_frames)} frames, "
                            f"expected {self.context_memory_frames}. This may cause dimension mismatch."
                        )
                    if "context_actions" in data and len(data["context_actions"]) > 0:
                        context_actions = data["context_actions"]
                elif (not dropped_context) and len(video_frames) > self.context_memory_frames:
                    # Fallback: use prefix frames as condition
                    context_frames = video_frames[:self.context_memory_frames]
                    # video_frames remains full 81 frames for condition mode
                    # Validate that we have enough frames
                    if len(video_frames) < 81:
                        import warnings
                        warnings.warn(
                            f"Warning: video_frames has {len(video_frames)} frames, expected 81. "
                            f"This may cause dimension mismatch errors."
                        )
                    # Also separate actions if available
                    if "actions" in data and len(data["actions"]) > self.context_memory_frames:
                        context_actions = data["actions"][:self.context_memory_frames]
                        # actions remain full length for condition mode
            else:  # training_mode == "context" (default, Experiment 16)
                # Experiment 16: Context Memory mode (Context as Memory)
                # - Context: Additional frames as condition (suffix position, not part of target)
                # - Target: Full 81 frames (from start_frame, always 81 frames)
                # - Context frames are separate conditions, do NOT reduce target frame count
                # Prefer retrieved context if present
                if (not dropped_context) and "context_frames" in data and len(data["context_frames"]) > 0:
                    context_frames = data["context_frames"]
                    context_indices = data.get("context_frame_indices", [])
                    start_frame = data.get("start_frame", None)
                    end_frame = data.get("end_frame", None)
                    
                    # CRITICAL: Filter out overlap frames that are within target segment (FOV mode only)
                    # If overlap frames are in [start_frame, end_frame], they are part of target
                    # and should not be used as context to avoid duplication
                    if (
                        getattr(self, "context_source", "fov") == "fov"
                        and start_frame is not None
                        and end_frame is not None
                        and len(context_indices) > 0
                    ):
                        filtered_context_frames = []
                        filtered_context_indices = []
                        filtered_context_actions = []
                        
                        # First frame is always included (it's the start_frame itself)
                        if len(context_frames) > 0:
                            filtered_context_frames.append(context_frames[0])
                            if len(context_indices) > 0:
                                filtered_context_indices.append(context_indices[0])
                            if "context_actions" in data and len(data["context_actions"]) > 0:
                                filtered_context_actions.append(data["context_actions"][0])
                        
                        # Filter overlap frames: only include those outside target segment
                        for i in range(1, len(context_frames)):
                            frame_idx = context_indices[i] if i < len(context_indices) else None
                            # Include overlap frame only if it's outside target segment
                            if frame_idx is None or frame_idx < start_frame or frame_idx > end_frame:
                                filtered_context_frames.append(context_frames[i])
                                if i < len(context_indices):
                                    filtered_context_indices.append(context_indices[i])
                                if "context_actions" in data and i < len(data["context_actions"]):
                                    filtered_context_actions.append(data["context_actions"][i])
                        
                        # Update context if we filtered any frames
                        if len(filtered_context_frames) < len(context_frames):
                            context_frames = filtered_context_frames
                            data["context_frames"] = filtered_context_frames
                            if filtered_context_indices:
                                data["context_frame_indices"] = filtered_context_indices
                            if filtered_context_actions:
                                data["context_actions"] = filtered_context_actions
                elif (not dropped_context) and len(video_frames) > self.context_memory_frames:
                    context_frames = video_frames[:self.context_memory_frames]
                
                # Target: ALWAYS keep full 81 frames (context frames are separate conditions)
                # Do NOT reduce video_frames count - context is additional condition, not part of target
                # video_frames remains unchanged (81 frames from start_frame)
                
                # Also separate actions if available
                if (not dropped_context) and "context_actions" in data and len(data["context_actions"]) > 0:
                    context_actions = data["context_actions"]
                elif "actions" in data and len(data["actions"]) > 0:
                    # If actions exist but context_actions not set, use first K frames as context actions
                    # But this should not happen if context retrieval worked correctly
                    if (not dropped_context) and len(data["actions"]) >= self.context_memory_frames:
                        context_actions = data["actions"][:self.context_memory_frames]
                    elif not dropped_context:
                        context_actions = data["actions"]

            if (not dropped_context) and context_frames is not None and len(context_frames) > 0:
                context_frames, context_actions = self._build_context_with_anchor(
                    context_frames,
                    context_actions=context_actions,
                    expected_k=self.context_memory_frames,
                )

            # DDP safety: ensure all ranks either have context (and thus use context-conditioned modules)
            # or all ranks drop it. If any rank lacks context, we drop globally to avoid unused-parameter errors.
            has_context_step = (not dropped_context) and (context_frames is not None) and (len(context_frames) > 0)
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
                context_frames = None
                context_actions = None
        # Note: ICL mode removed - only Context Memory mode is supported
        # Context Memory mode: context_frames and video_frames are already separated above
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": video_frames,
            "height": video_frames[0].size[1],
            "width": video_frames[0].size[0],
            "num_frames": len(video_frames),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        
        # Store context frames separately for Context Memory mode
        if self.enable_context_memory and context_frames is not None:
            # CRITICAL: Ensure context frames have the same size as target frames
            # FOV-retrieved frames may come from different video segments with different sizes
            target_height = video_frames[0].size[1]  # PIL Image: (width, height) -> size[1] is height
            target_width = video_frames[0].size[0]   # PIL Image: size[0] is width
            
            # CRITICAL: Resize context frames to match target frame dimensions
            # FOV-retrieved frames may come from different video segments with different sizes
            # This ensures spatial dimension consistency when concatenating latents
            from PIL import Image
            
            resized_context_frames = []
            for ctx_frame in context_frames:
                if hasattr(ctx_frame, 'resize') and hasattr(ctx_frame, 'size'):  # PIL Image
                    ctx_w, ctx_h = ctx_frame.size  # PIL Image.size is (width, height)
                    if ctx_h != target_height or ctx_w != target_width:
                        # Resize to match target dimensions (PIL Image.resize takes (width, height))
                        ctx_frame = ctx_frame.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    resized_context_frames.append(ctx_frame)
                else:
                    # If not PIL Image, keep as-is (shouldn't happen, but be safe)
                    resized_context_frames.append(ctx_frame)
            
            # Ensure context frames count matches expected context_memory_frames
            expected_context_frames = self.context_memory_frames
            if len(resized_context_frames) < expected_context_frames:
                # Pad with zeros or repeat last frame to match expected count
                num_to_pad = expected_context_frames - len(resized_context_frames)
                if resized_context_frames:
                    # Repeat the last frame
                    last_frame = resized_context_frames[-1]
                    for _ in range(num_to_pad):
                        resized_context_frames.append(last_frame)
                else:
                    # If no frames at all, create black frames
                    black_frame = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                    for _ in range(expected_context_frames):
                        resized_context_frames.append(black_frame)
            elif len(resized_context_frames) > expected_context_frames:
                # Truncate to expected count
                resized_context_frames = resized_context_frames[:expected_context_frames]

            inputs_shared["context_frames"] = resized_context_frames
            inputs_shared["num_context_frames"] = len(resized_context_frames)
            inputs_shared["context_position"] = os.environ.get("CONTEXT_POSITION", "suffix")
            if data.get("non_fov_frames"):
                inputs_shared["non_fov_frames"] = data["non_fov_frames"]
        
        # Handle actions: Context Memory mode (context_actions + target actions) or Interactive-only (target actions only)
        # Context-as-Memory [2506.03141]: omit_context_actions -> no context RT injection (use identity for context part)
        if self.enable_context_memory:
            omit = getattr(self, "omit_context_actions", False)
            inputs_shared["omit_context_actions"] = omit
            if not omit and context_actions is not None:
                # Convert list of RT poses to tensor if needed
                if isinstance(context_actions, list) and len(context_actions) > 0:
                    if isinstance(context_actions[0], list):
                        context_actions_tensor = torch.tensor(context_actions, dtype=torch.float32)
                        inputs_shared["context_actions"] = context_actions_tensor
                    else:
                        inputs_shared["context_actions"] = context_actions
                else:
                    inputs_shared["context_actions"] = context_actions

        # Target actions: always add when available (for both context and non-context modes)
        # Stage 1 (Interactive): no context, but needs target actions for RT training
        if "actions" in data:
            actions = data["actions"]
            if getattr(self, 'yaw_flip_aug', False) and isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], (list, tuple)) and len(actions[0]) >= 12:
                import random
                if random.random() < 0.5:
                    try:
                        from .rt_utils import flip_yaw_rt_list
                    except ImportError:
                        from rt_utils import flip_yaw_rt_list
                    actions = flip_yaw_rt_list(actions)
            if isinstance(actions, list) and len(actions) > 0:
                if isinstance(actions[0], list):
                    actions = torch.tensor(actions, dtype=torch.float32)
            inputs_shared["actions"] = actions
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                # Use the first frame of the target video (or context frame if available)
                if self.enable_context_memory and context_frames is not None:
                    inputs_shared["input_image"] = context_frames[0]
                else:
                    inputs_shared["input_image"] = video_frames[0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = video_frames[-1]
            elif extra_input == "camera_control_poses":
                # Build explicit camera trajectory control from dataset jsons/<video_name>.json
                # Requires Camera-Control base model (with control_adapter).
                try:
                    from .fov_retrieval import load_camera_poses_batch  # type: ignore
                except Exception:
                    from fov_retrieval import load_camera_poses_batch  # type: ignore

                def _parse_i(x):
                    try:
                        if x is None:
                            return None
                        if isinstance(x, (int, np.integer)):
                            return int(x)
                        if isinstance(x, (float, np.floating)):
                            return int(round(float(x)))
                        return int(round(float(str(x).strip())))
                    except Exception:
                        return None

                def _euler_to_R(roll, pitch, yaw):
                    rr, pp, yy = np.radians([roll, pitch, yaw])
                    Rx = np.array([[1,0,0],[0,np.cos(rr),-np.sin(rr)],[0,np.sin(rr),np.cos(rr)]], dtype=np.float32)
                    Ry = np.array([[np.cos(pp),0,np.sin(pp)],[0,1,0],[-np.sin(pp),0,np.cos(pp)]], dtype=np.float32)
                    Rz = np.array([[np.cos(yy),-np.sin(yy),0],[np.sin(yy),np.cos(yy),0],[0,0,1]], dtype=np.float32)
                    return (Rz @ Ry @ Rx).astype(np.float32)

                # Defaults from camera controller origin
                fx, fy, cx, cy = 0.532139961, 0.946026558, 0.5, 0.5
                start_frame = _parse_i(data.get("start_frame", None)) or 0
                video_name = data.get("video_name", None)
                if isinstance(video_name, str) and video_name.endswith(".mp4"):
                    video_name = ".".join(video_name.split(".")[:-1])
                poses = []
                if self.dataset_base_path and video_name:
                    json_file = os.path.join(self.dataset_base_path, "jsons", f"{video_name}.json")
                    frame_indices = list(range(start_frame, start_frame + len(video_frames)))
                    poses_batch = load_camera_poses_batch(json_file, frame_indices)
                    last_pose = None
                    for fi, pose in zip(frame_indices, poses_batch):
                        if pose is None:
                            pose = last_pose
                        if pose is None:
                            # identity pose
                            pos = np.zeros(3, dtype=np.float32)
                            R_c2w = np.eye(3, dtype=np.float32)
                        else:
                            pos = np.array(pose.get("position", [0,0,0]), dtype=np.float32)
                            rot = pose.get("rotation", [0,0,0])
                            rot = list(rot) if isinstance(rot, (list, tuple)) else [0,0,0]
                            roll = float(rot[0]) if len(rot) > 0 else 0.0
                            pitch = float(rot[1]) if len(rot) > 1 else 0.0
                            yaw = float(rot[2]) if len(rot) > 2 else 0.0
                            R_c2w = _euler_to_R(roll, pitch, yaw)
                            last_pose = pose
                        R_w2c = R_c2w.T
                        t_w2c = (-R_w2c @ pos.reshape(3, 1)).reshape(3)
                        w2c_3x4 = np.concatenate([R_w2c, t_w2c.reshape(3,1)], axis=1).reshape(-1).tolist()
                        entry = [0.0, float(fx), float(fy), float(cx), float(cy), 0.0, 0.0] + [float(x) for x in w2c_3x4]
                        poses.append(entry)
                inputs_shared["camera_control_poses"] = poses
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

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
        if inputs is None: inputs = self.forward_preprocess(data)
        
        # Note: Self-forcing removed - using standard Context Memory training only
        
        # Context Memory mode: handle context latents separately
        if self.enable_context_memory and "context_frames" in inputs:
            # Extract context frames and process them separately
            context_frames = inputs.pop("context_frames")
            num_context_frames = inputs.pop("num_context_frames", len(context_frames))
            
            # Process context frames through VAE to get clean latents
            # We need to encode context frames to latents
            with torch.no_grad():
                # Get VAE encoder
                vae_encoder = getattr(self.pipe, "vae_encoder", None)
                if vae_encoder is None:
                    # Try to get VAE and use its encoder
                    vae = getattr(self.pipe, "vae", None)
                    if vae is not None and hasattr(vae, "encoder"):
                        vae_encoder = vae.encoder
                
                # Use VAE.encode method for proper frame-to-latent conversion
                # Reference: GF-ICL implementation - encode all frames together as a video
                # VAE.encode handles the conversion: 4 frames = 1 latent token
                # Formula: (N-1)/4+1 where N is number of frames
                # - 5 frames = (5-1)/4+1 = 2 latent tokens
                # - 81 frames = (81-1)/4+1 = 21 latent tokens
                vae = getattr(self.pipe, "vae", None)
                if vae is not None and hasattr(vae, "encode"):
                    with torch.no_grad():
                        if getattr(self, "context_per_frame_vae", False):
                            # Each context frame -> 1 latent token; context_actions already one per raw frame
                            context_latents_list_fwd = []
                            for frame in context_frames:
                                frame_video = self.pipe.preprocess_video([frame])  # (1, C, 1, H, W)
                                frame_sq = frame_video.squeeze(0)  # (C, 1, H, W)
                                latent_one = vae.encode([frame_sq], device=self.pipe.device, tiled=False, tile_size=None, tile_stride=None)
                                context_latents_list_fwd.append(latent_one)
                            context_latents = torch.cat(context_latents_list_fwd, dim=2)  # (1, C, K, H//8, W//8)
                            num_context_frames = context_latents.shape[2]
                            context_latents = context_latents.to(dtype=torch.bfloat16, device=self.pipe.device)
                        else:
                            # Encode all context frames together as a video (temporal downsample: (N-1)//4+1 latent tokens)
                            context_video = self.pipe.preprocess_video(context_frames)  # (1, C, T, H, W)
                            logger.debug(f"Context video shape before encoding: {context_video.shape}, number of context frames: {len(context_frames)}")
                            if len(context_video.shape) == 5:
                                context_video_squeezed = context_video.squeeze(0)
                                context_latents = vae.encode([context_video_squeezed], device=self.pipe.device, tiled=False, tile_size=None, tile_stride=None)
                            elif len(context_video.shape) == 4:
                                context_latents = vae.encode([context_video], device=self.pipe.device, tiled=False, tile_size=None, tile_stride=None)
                            else:
                                context_latents = vae.encode(context_video, device=self.pipe.device, tiled=False, tile_size=None, tile_stride=None)
                            context_latents = context_latents.to(dtype=torch.bfloat16, device=self.pipe.device)
                            actual_latent_tokens = context_latents.shape[2] if len(context_latents.shape) > 2 else 0
                            expected_latent_tokens = (len(context_frames) - 1) // 4 + 1
                            if actual_latent_tokens != expected_latent_tokens:
                                import warnings
                                warnings.warn(
                                    f"Context latent tokens mismatch: expected {expected_latent_tokens} "
                                    f"(from {len(context_frames)} frames using formula (N-1)/4+1), "
                                    f"got {actual_latent_tokens}. Full shape: {context_latents.shape}. "
                                    f"Using actual value: {actual_latent_tokens}"
                                )
                            num_context_frames = actual_latent_tokens
                    inputs["context_latents"] = context_latents
                    inputs["num_context_frames"] = num_context_frames
                    inputs["training_mode"] = getattr(self, 'training_mode', 'context')
                    inputs["context_noise_prob"] = self.context_noise_prob
                    inputs["context_noise_std"] = self.context_noise_std
                    if hasattr(self, 'context_fixed_noise_std') and self.context_fixed_noise_std is not None:
                        inputs["context_fixed_noise_std"] = self.context_fixed_noise_std
                else:
                    # Fallback: use frame-by-frame encoding if VAE.encode not available
                    # This is not ideal but will work
                    import warnings
                    warnings.warn("VAE.encode not available, using frame-by-frame encoding (may not match expected latent tokens)")
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
                    ])
                    context_tensors = torch.stack([
                        transform(frame.convert("RGB")) for frame in context_frames
                    ]).to(self.pipe.device, dtype=torch.bfloat16)
                    
                    context_latents_list = []
                    for frame in context_tensors:
                        frame_batch = frame.unsqueeze(0)
                        with torch.no_grad():
                            latent = vae_encoder(frame_batch)
                        context_latents_list.append(latent)
                    context_latents = torch.cat(context_latents_list, dim=2)
                    # CRITICAL: num_context_frames should be latent tokens, not raw frames
                    # Use actual latent token count from encoded latents shape
                    actual_context_frames = context_latents.shape[2] if len(context_latents.shape) > 2 else len(context_frames)
                    num_context_frames = actual_context_frames
                    
                    # Verify consistency
                    expected_latent_tokens = (len(context_frames) - 1) // 4 + 1
                    if actual_context_frames != expected_latent_tokens:
                        import warnings
                        warnings.warn(
                            f"num_context_frames (latent tokens) mismatch: "
                            f"expected {expected_latent_tokens} tokens from {len(context_frames)} frames "
                            f"(formula: 1 + ({len(context_frames)}-1)//4), "
                            f"got {actual_context_frames} from encoded latents shape. "
                            f"Using actual value: {actual_context_frames}"
                        )
                    
                    inputs["context_latents"] = context_latents
                    inputs["num_context_frames"] = num_context_frames
                    # Pass training mode
                    inputs["training_mode"] = getattr(self, 'training_mode', 'context')
                    # Pass context noise augmentation parameters
                    inputs["context_noise_prob"] = self.context_noise_prob
                    inputs["context_noise_std"] = self.context_noise_std
                    if hasattr(self, 'context_fixed_noise_std') and self.context_fixed_noise_std is not None:
                        inputs["context_fixed_noise_std"] = self.context_fixed_noise_std
                # If VAE not available at all, store context frames for later processing
                if "context_latents" not in inputs:
                    inputs["context_frames"] = context_frames
                    inputs["num_context_frames"] = num_context_frames
        
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        # Check if pipeline supports context_latents parameter
        # If not, we'll need to override training_loss
        if self.enable_context_memory and ("context_latents" in inputs or "context_frames" in inputs):
            # Use custom training loss that handles context latents
            # This handles both cases: when context_latents exist or when context_frames need to be encoded
            loss = self._training_loss_with_context(**models, **inputs)
        else:
            # Standard mode: ensure input_latents exists before calling pipeline training_loss.
            inputs = self._ensure_input_latents(inputs, strict=True)
            loss = self.pipe.training_loss(**models, **inputs)
        return loss
    
    def _training_loss_with_context(self, **kwargs):
        """
        Custom training loss function that handles context latents (Context as Memory).
        
        Core logic:
        1. Only adds noise to target frames (not context frames)
        2. Concatenates clean context latents with noisy target latents before model forward
        3. Computes loss only on target frames (context frames are masked out)
        
        This function intercepts the training loss computation and modifies it to support
        Context Memory mode. The actual implementation depends on the pipeline structure.
        """
        # Extract context information
        context_latents = kwargs.pop("context_latents", None)
        num_context_frames = kwargs.pop("num_context_frames", 0)
        context_frames = kwargs.pop("context_frames", None)
        
        # Separate models and inputs
        models = {k: v for k, v in kwargs.items() if k in self.pipe.in_iteration_models}
        inputs = {k: v for k, v in kwargs.items() if k not in self.pipe.in_iteration_models}
        
        # If we have context frames but not latents, encode them
        if context_frames is not None and context_latents is None:
            # Encode context frames to latents using the same method as in inference
            vae = getattr(self.pipe, "vae", None)
            if vae is not None and hasattr(vae, "encode"):
                try:
                    # Use pipeline's preprocess_video to convert PIL Images to video format
                    # This matches the method used in _sample_video_with_context_memory
                    context_latents_list = []
                    for frame in context_frames:
                        # Preprocess frame: returns (1, C, 1, H, W)
                        frame_video = self.pipe.preprocess_video([frame])  # (1, C, 1, H, W)
                        # Squeeze batch dim: (1, C, 1, H, W) -> (C, 1, H, W)
                        frame_video_squeezed = frame_video.squeeze(0)  # (C, 1, H, W)
                        
                        # Encode: vae.encode expects list of (C, T, H, W) tensors
                        # Returns (B, C, T, H//8, W//8) where B=len(input_list)
                        # CRITICAL FIX (Experiment 15): Remove context_amplify_factor to prevent VAE Decoder overflow
                        with torch.no_grad():
                            latent_batch = vae.encode([frame_video_squeezed], device=self.pipe.device)  # (1, C, 1, H//8, W//8)
                            # Extract single video latent: (1, C, 1, H//8, W//8) -> (1, C, 1, H//8, W//8)
                            latent = latent_batch[0].unsqueeze(0)  # (1, C, 1, H//8, W//8)
                        context_latents_list.append(latent)
                    
                    # Stack along frame dimension: (F, 1, C, 1, H//8, W//8) -> (1, C, F, H//8, W//8)
                    context_latents = torch.cat(context_latents_list, dim=2)
                    # CRITICAL: num_context_frames should be latent tokens, not raw frames
                    # In fallback mode, each frame = 1 latent token (not ideal but will work)
                    # But we should use actual latent token count from shape
                    actual_latent_tokens = context_latents.shape[2] if len(context_latents.shape) > 2 else len(context_frames)
                    num_context_frames = actual_latent_tokens
                except Exception as e:
                    # Encoding failed, log the error but continue without context
                    import warnings
                    warnings.warn(
                        f"Failed to encode context frames to latents: {e}. "
                        f"Will fallback to standard training mode for this batch."
                    )
                    context_latents = None
                    num_context_frames = 0
            else:
                # VAE or encode method not available
                import warnings
                warnings.warn(
                    "VAE encoder not available for encoding context frames. "
                    "Will fallback to standard training mode for this batch."
                )
                context_latents = None
                num_context_frames = 0
        
        # Store context information for pipeline
        if context_latents is not None:
            inputs["context_latents"] = context_latents
            inputs["num_context_frames"] = num_context_frames
            # Pass context noise augmentation parameters
            inputs["context_noise_prob"] = self.context_noise_prob
            inputs["context_noise_std"] = self.context_noise_std
            if hasattr(self, 'context_fixed_noise_std') and self.context_fixed_noise_std is not None:
                inputs["context_fixed_noise_std"] = self.context_fixed_noise_std
            inputs["context_attention_weight"] = getattr(self, "context_attention_weight", 1.0)
            inputs["use_anchor_frame"] = getattr(self, "use_anchor_frame", False)
            inputs["context_temporal_decay"] = getattr(self, "context_temporal_decay", 1.0)
            inputs["use_spatial_memory"] = getattr(self.pipe, "use_spatial_memory", False)
            inputs["spatial_memory_tokens"] = int(getattr(self.pipe, "spatial_memory_tokens", 64) or 64)
            inputs["use_spatial_memory_legacy"] = bool(getattr(self.pipe, "use_spatial_memory_legacy", False))
            inputs["spatial_memory_module"] = getattr(self.pipe, "spatial_memory_module", None)
            inputs["spatial_memory_inject_mode"] = getattr(self.pipe, "spatial_memory_inject_mode", "concat_text")
            inputs["spatial_memory_readout_module"] = getattr(self.pipe, "spatial_memory_readout_module", None)
            inputs["use_framepack_memory"] = bool(getattr(self, "use_framepack_memory", False))
        else:
            # Context memory mode is enabled but context_latents is None
            # This can happen if encoding failed or context_frames were not available
            # In this case, we should fallback to standard training mode
            # Remove any context-related keys and ensure we have input_latents
            inputs.pop("context_latents", None)
            inputs.pop("num_context_frames", None)
            inputs.pop("context_frames", None)
            # Ensure input_latents exists - if not, this will cause an error later
            # but that's expected as we can't train without latents
        
        # Ensure pipeline-required input_latents exists in context mode as well.
        inputs = self._ensure_input_latents(inputs, strict=True)
        
        # Call pipeline's training_loss which supports context_latents
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    def _add_arg_if_missing(*args, **kwargs):
        if args and args[0] in parser._option_string_actions:
            return
        parser.add_argument(*args, **kwargs)

    parser.add_argument("--tokenizer_path", type=str, default=None, help="Local path to tokenizer (e.g. .../Wan2.1-T2V-14B/google/umt5-xxl). If set, uses this instead of downloading from ModelScope. Required for 14B to avoid 1.3B tokenizer download.")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--ckpt_interval", type=int, default=None)
    parser.add_argument("--save_full_model", action="store_true", help="Save full model weights instead of only trainable parameters (LoRA). Default: False (save only LoRA weights)")
    parser.add_argument("--trainable_dit_modules", type=str, default=None, help="Comma-separated modules to unfreeze (rest frozen). Allowed: camera_encoder, block_self_attn, action_mlp, self_attn_with_action, block_ssm, unified_implicit, spatial_memory (SpatialGridMemory on training module). Omit = train full DiT.")
    parser.add_argument("--add_action_attn", action="store_true", help="Add action attention")
    parser.add_argument("--action_use_temporal_attention", action="store_true", help="Use temporal attention for action module")
    parser.add_argument("--action_inject_after_spatial_attn", action="store_true", help="exp1_4_3: Inject RT after spatial attention output (Context-as-Memory [2506.03141] style)")
    parser.add_argument("--use_camera_encoder", action="store_true", help="Use CameraEncoder for RT injection (Context-as-Memory: RT -> MLP -> add to spatial attn output)")
    parser.add_argument("--camera_encoder_shallow", action="store_true", help="Use single-layer CameraEncoder E_c(cam)=Linear(12,D) for context-based ablation")
    parser.add_argument("--camera_encoder_separate_t_r", action="store_true", help="Encode translation and rotation in separate MLPs then add (scale balance)")
    parser.add_argument("--camera_encoder_explicit_yaw", action="store_true", help="Add signed yaw scalar branch for CW/CCW direction sensitivity (Z-only)")
    parser.add_argument("--yaw_flip_aug", action="store_true", help="50%% prob flip yaw of target actions (direction sensitivity aug)")
    parser.add_argument("--camera_encoder_sincos_yaw", action="store_true", help="Add cos(yaw), sin(yaw) branch for direction (scheme C)")
    parser.add_argument("--camera_encoder_scale", type=float, default=1.0, help="Multiply camera embedding by this (e.g. 1.5 for stronger conditioning)")
    parser.add_argument("--camera_encoder_r_mlp_no_layernorm", action="store_true", help="No LayerNorm in R branch (separate_t_r) so yaw sign is not normalized away")
    parser.add_argument("--add_camera_outside_gate", action="store_true", help="Add camera_emb after gate so it is not scaled by gate_msa (direction sensitivity)")
    parser.add_argument(
        "--camera_inject_mode",
        type=str,
        default="post",
        choices=["post", "pre_norm", "pre_qkv", "pre_qkv_post", "pre_modulate", "pre_qkv_gated"],
        help="Camera injection mode (post/pre_norm/pre_qkv/pre_qkv_post/pre_modulate/pre_qkv_gated). pre_modulate: cam before modulate; pre_qkv_gated: input_x+gate*cam, gate 0-init.",
    )
    parser.add_argument("--no_camera_encoder_zero_init", action="store_true", help="Disable 0-init scale for CameraEncoder (ablation: scale init 1 instead of 0)")
    parser.add_argument("--camera_encoder_full_zero_init", action="store_true", help="GF-style: zero-init entire Linear layer (weight+bias) for merged shallow MLP")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for preloading (0=main process). 4 recommended for video I/O to parallelize with GPU.")
    parser.add_argument("--max_train_steps", type=int, default=0, help="Stop after N successful optimizer steps (0 = disabled). Useful for smoke tests.")
    parser.add_argument("--progress_total_steps", type=int, default=0, help="Display this many total steps in tqdm (0 = dataloader length). Useful when smoke tests stop early but should show full training target.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint (optional)")
    # Context memory (Context-as-Memory: clean latents as context)
    parser.add_argument("--enable_context_memory", action="store_true", help="Enable Context Memory mode: use clean VAE latents as context, protect context from noise")
    parser.add_argument("--context_memory_frames", type=int, default=8, help="Number of context frames to use in Context Memory mode")
    parser.add_argument("--context_per_frame_vae", action="store_true", help="Encode each context frame with VAE separately (1 latent per raw frame); default is temporal downsample (N raw -> (N-1)//4+1 latent)")
    parser.add_argument("--training_mode", type=str, default="predict", choices=["predict", "context", "condition"], help="Training mode (context = context-based memory)")
    parser.add_argument("--context_drop_prob", type=float, default=0.0, help="Probability to drop context per sample (0=off)")
    parser.add_argument("--cfg_target_only", action="store_true", help="Apply CFG only to target frames (context-based memory)")
    parser.add_argument("--enable_fov_retrieval", action="store_true", help="Enable FOV-based memory retrieval (context-based memory)")
    parser.add_argument("--retrieval_method", type=str, default="fov", choices=["fov", "latent_sim"], help="Context retrieval: fov (FOV/overlap) or latent_sim (latent similarity to first frame)")
    parser.add_argument("--latent_retrieval_dir", type=str, default=None, help="Per-frame latent dir for latent_sim: latent_dir/video_name/{frame:04d}.pt; omit to fall back to FOV")
    parser.add_argument("--fov_top_k", type=int, default=4, help="FOV overlap frames to retrieve (0=last-k only)")
    parser.add_argument("--use_rt_relative", action="store_true", help="RT relative to segment first frame (context-based memory)")
    parser.add_argument("--strict_overlap_context", action="store_true", help="Require overlap_labels; skip if missing")
    parser.add_argument("--use_anchor_frame", action="store_true", help="Anchor-frame memory: treat context as mandatory anchors (related: MoC mandatory anchors, key-frame persistence)")
    parser.add_argument("--context_attention_weight", type=float, default=1.0, help="Scale context token attention (e.g. <1 compress distant context, >1 emphasize; related: FAR/FramePack-style compression)")
    parser.add_argument("--context_temporal_decay", type=float, default=1.0, help="FAR/FramePack-style: per-frame decay for context (weight_i = decay^distance, 1.0=no decay)")
    parser.add_argument("--spike_threshold", type=float, default=5.0, help="Skip optimizer step when loss/traj_loss exceeds this ratio after warmup.")
    parser.add_argument("--use_spatial_memory", action="store_true", help="Spatial Memory: append context-derived tokens to cross-attn (default: learnable SpatialGridMemory unless --use_spatial_memory_legacy)")
    parser.add_argument("--use_spatial_memory_legacy", action="store_true", help="Non-learnable adaptive pool over context tokens (old baseline; no SpatialGridMemory)")
    parser.add_argument("--spatial_memory_tokens", type=int, default=64, help="Number of spatial memory tokens M (grid output is mixed to M)")
    parser.add_argument("--spatial_memory_grid", type=int, default=8, help="SpatialGridMemory: G×G downsampling after time-mean (default 8)")
    parser.add_argument(
        "--spatial_memory_inject_mode",
        type=str,
        default="concat_text",
        choices=["concat_text", "none", "cross_attn_readout"],
        help="How spatial memory tokens enter the model: concat_text=append to T5 context (default); none=skip injection; cross_attn_readout=target tokens read memory via lightweight cross-attn readout (learnable zero-init gate)",
    )
    parser.add_argument("--use_framepack_memory", action="store_true", help="FAR/FramePack-style baseline: apply temporal decay + weight to context tokens (no length change)")
    parser.add_argument("--use_framepack_length_compress", action="store_true", help="FramePack-style: compress context_latents along time (K->K') before concatenation")
    parser.add_argument("--framepack_ratio", type=int, default=2, help="FramePack temporal ratio r: K' = ceil(K/r)")
    parser.add_argument(
        "--framepack_length_strategy",
        type=str,
        default="distance_merge",
        choices=["distance_merge", "mean", "uniform", "recent_weighted", "weighted_recent", "packed_multiscale"],
        help="FramePack length compression strategy. packed_multiscale approximates base-code 1x/2x/4x packed fusion in current framework.",
    )
    parser.add_argument("--framepack_recent_keep_ratio", type=float, default=0.5, help="When strategy=recent_weighted, larger means stronger bias to recent frames")
    parser.add_argument("--framepack_multiscale_w2", type=float, default=0.25, help="packed_multiscale: weight for 2x pooled branch")
    parser.add_argument("--framepack_multiscale_w4", type=float, default=0.15, help="packed_multiscale: weight for 4x pooled branch")
    parser.add_argument(
        "--context_source",
        type=str,
        default="fov",
        choices=["fov", "replay", "prev_chunk_tail"],
        help="How to build context frames: fov=FOV retrieval (default); replay=virtual chunk1 + same rule as multichunk eval; prev_chunk_tail=strict N frames before start_frame from disk",
    )
    # Block-wise SSM (Long-Context State-Space Video World Models, https://ryanpo.com/ssm_wm/ arXiv:2505.20171)
    parser.add_argument(
        "--use_block_wise_ssm",
        action="store_true",
        help="Paper-style block-wise temporal SSM in DiT (Long-Context SSM, arXiv:2505.20171); preferred over --use_videossm_hybrid for SSM baselines",
    )
    parser.add_argument("--ssm_num_blocks_hint", type=int, default=21, help="Hint for SSM temporal blocks (e.g. num_frames)")
    parser.add_argument("--ssm_every_n_blocks", type=int, default=4, help="Add SSM every N DiT blocks (1=all, 4=~4x faster)")
    # VideoSSM baseline: hybrid state-space memory (separate from block-wise SSM)
    parser.add_argument(
        "--use_videossm_hybrid",
        action="store_true",
        help="Legacy VideoSSM hybrid (depthwise temporal conv in DiT); separate from --use_block_wise_ssm. Prefer block-wise SSM for paper-aligned experiments.",
    )
    parser.add_argument("--videossm_kernel_size", type=int, default=3, help="VideoSSM hybrid: causal temporal kernel size")
    parser.add_argument("--videossm_expand", type=int, default=2, help="VideoSSM hybrid: channel expansion factor")
    parser.add_argument("--videossm_every_n_blocks", type=int, default=4, help="VideoSSM hybrid: apply in every N DiT blocks (1=all)")
    # Video sampling
    parser.add_argument(
        "--sampling_interval_steps",
        type=int,
        default=0,
        help="Fixed sampling interval in steps (0 = auto). Used for --sampling_two_prompts and --sampling_two_chunk_memory.",
    )
    parser.add_argument("--sampling_negative_prompt", type=str, default="oversaturated colors, overexposed, static, blurry details", help="Negative prompt for video sampling")
    parser.add_argument("--sampling_height", type=int, default=352, help="Height for sampled videos")
    parser.add_argument("--sampling_width", type=int, default=640, help="Width for sampled videos")
    parser.add_argument("--sampling_num_frames", type=int, default=81, help="Number of frames for sampled videos")
    parser.add_argument("--sampling_num_inference_steps", type=int, default=50, help="Denoising steps for sampling")
    parser.add_argument("--sampling_action_path", type=str, default=None, help="Action JSON for sampling (e.g. rotation 180)")
    parser.add_argument(
        "--sampling_two_chunk_action_path",
        type=str,
        default=None,
        help="2-chunk sampling action path (left_45). If unset, fallback to --sampling_action_path and infer right_45 by sibling filename.",
    )
    parser.add_argument(
        "--sampling_two_chunk_memory",
        action="store_true",
        help="During sampling intervals, run chunk1->chunk2 monitor aligned with run_replay_loop_two_chunk (mutually prioritized after --sampling_two_prompts)",
    )
    parser.add_argument("--sampling_eval_dataset_base", type=str, default=None, help="Eval dataset base path for in-training 2-chunk sampling (frames/, jsons/, captions.txt)")
    parser.add_argument("--sampling_eval_metadata_path", type=str, default=None, help="Eval metadata CSV for in-training sampling (video_name,prompt columns)")
    parser.add_argument("--samples_per_epoch", type=int, default=0, help="Samples per epoch (0 = use sampling_interval_steps)")
    # Legacy-compatible args kept to avoid breaking historical shell scripts.
    _add_arg_if_missing("--per_device_train_batch_size", type=int, default=None, help="Alias for per-device batch size (legacy scripts).")
    _add_arg_if_missing("--enable_video_sampling", action="store_true", help="Enable periodic video sampling (legacy gate).")
    _add_arg_if_missing("--sampling_atomic_left_right", action="store_true", help="Legacy sampling mode alias.")
    _add_arg_if_missing("--sampling_four_prompts", action="store_true", help="Legacy sampling mode alias.")
    _add_arg_if_missing("--sampling_two_prompts", action="store_true", help="Legacy sampling mode alias.")
    _add_arg_if_missing("--timestep_shift", type=float, default=1.0, help="Scheduler timestep shift.")
    _add_arg_if_missing("--train_action_module", action="store_true", help="Train action module (inject DiTBlock_w_Action into DiT blocks)")
    _add_arg_if_missing("--train_cam_pose", action="store_true", help="Train with CaM camera pose control (MLP_CamPose)")
    _add_arg_if_missing("--add_action_attn", action="store_true", help="Add temporal self-attention for action in DiTBlock_w_Action")
    _add_arg_if_missing("--action_use_temporal_attention", action="store_true", help="Use temporal attention for action module")
    _add_arg_if_missing("--action_base_path", type=str, default=None, help="Base path for game action JSONs")
    _add_arg_if_missing("--action_module_only", action="store_true", help="Only train action module parameters")
    _add_arg_if_missing("--ckpt_path", type=str, default=None, help="Checkpoint path for action/cam module weights")
    _add_arg_if_missing("--cam_position_scale", type=float, default=0.01, help="Scale raw CaM position (cm) to meters")
    _add_arg_if_missing("--resume_from", type=str, default=None, help="Resume full DiT from checkpoint path")
    _add_arg_if_missing("--verify_ckpt_step", type=int, default=0, help="Legacy no-op compatibility flag.")
    _add_arg_if_missing("--verify_high_noise_first_steps", type=int, default=0, help="Legacy no-op compatibility flag.")
    _add_arg_if_missing("--use_moc", action="store_true", help="Legacy compatibility flag.")
    _add_arg_if_missing("--moc_temperature", type=float, default=1.0, help="Legacy compatibility flag.")
    _add_arg_if_missing("--moc_top_k", type=int, default=0, help="Legacy compatibility flag.")
    _add_arg_if_missing("--prev_chunk_frames", type=int, default=81, help="Replay synthetic context segment length.")
    _add_arg_if_missing("--implicit_type", type=str, default="summary", help="Legacy compatibility flag.")
    _add_arg_if_missing("--unified_implicit", action="store_true", help="Legacy compatibility flag.")
    _add_arg_if_missing("--use_implicit_memory", action="store_true", help="Legacy compatibility flag.")
    _add_arg_if_missing("--use_memory_v2v_compressor", action="store_true", help="Legacy compatibility flag.")
    _add_arg_if_missing("--context_compressor_ratio", type=int, default=2, help="Legacy compatibility flag.")
    _add_arg_if_missing("--use_slow_fast_memory", action="store_true", help="Legacy compatibility flag.")
    _add_arg_if_missing("--use_entity_memory", action="store_true", help="Legacy compatibility flag.")
    _add_arg_if_missing("--use_episodic_memory", action="store_true", help="Legacy compatibility flag.")
    _add_arg_if_missing("--episodic_buffer_size", type=int, default=0, help="Legacy compatibility flag.")
    _add_arg_if_missing("--episodic_replay_interval", type=int, default=0, help="Legacy compatibility flag.")
    _add_arg_if_missing("--episodic_replay_weight", type=float, default=0.0, help="Legacy compatibility flag.")
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

    # 从检查点文件名中提取步数（用于 resume 训练）
    # trainable_dit_modules 或 resume_weights_only 时：resume_from_checkpoint 仅用于加载权重，不从 checkpoint 恢复步数，不 skip data
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
    
    # 设置随机种子
    set_seed(42)
    
    # Initialize Memory Bank if enabled (optional: requires memory_bank module)
    memory_bank = None
    use_memory_bank = _arg('use_memory_bank', False)
    if use_memory_bank and TrainingMemoryBank is not None:
        memory_bank = TrainingMemoryBank(
            max_size=_arg('memory_bank_size', 100),
            keyframes_per_entry=_arg('memory_keyframes_per_entry', 4),
            keyframe_selection=_arg('memory_keyframe_selection', 'semantic'),
            memory_update_strategy=_arg('memory_update_strategy', 'fifo'),
        )
        logger.info(f"Memory Bank initialized: max_size={memory_bank.memory_bank.max_size}, "
              f"keyframes_per_entry={memory_bank.keyframes_per_entry}, "
              f"selection={memory_bank.memory_bank.keyframe_selection}")
    elif use_memory_bank and TrainingMemoryBank is None:
        use_memory_bank = False
        logger.info("use_memory_bank requested but memory_bank module not found; disabling Memory Bank.")
    
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
        effective_ds_size = ds_size * ds_repeat

        logger.info("=" * 80)
        logger.info("[Dataset Validation] Dataset loaded successfully")
        logger.info("=" * 80)
        logger.info(f"  Dataset size (len(dataset)): {ds_size}")
        logger.info(f"  Dataset repeat: {ds_repeat}")
        logger.info(f"  Effective dataset size (unique_samples * repeat): {effective_ds_size}")
        logger.info(f"  Number of epochs: {args.num_epochs}")
        logger.info(f"  Total samples to process: {effective_ds_size * args.num_epochs}")

        if ds_size > 0:
            logger.info("\n[Dataset Validation] Sampling first 3 items to verify structure:")
            for i in range(min(3, ds_size)):
                try:
                    sample = ds[i]
                    if sample is None:
                        continue
                    video_name = sample.get("video_name", "unknown")
                    start_frame = sample.get("start_frame", None)
                    end_frame = sample.get("end_frame", None)
                    frame_idx = sample.get("frame_idx", None)
                    num_video_frames = len(sample.get("video", [])) if isinstance(sample.get("video"), list) else 0

                    logger.info(f"  Sample {i}:")
                    logger.info(f"    video_name: {video_name}")
                    logger.info(f"    start_frame: {start_frame}")
                    logger.info(f"    end_frame: {end_frame}")
                    logger.info(f"    frame_idx: {frame_idx}")
                    logger.info(f"    num_video_frames: {num_video_frames}")

                    if start_frame is not None and num_video_frames > 0:
                        expected_frames = 81
                        if num_video_frames == expected_frames:
                            logger.info(f"    ✓ Frame count correct: {num_video_frames} frames (expected {expected_frames})")
                        else:
                            logger.warning(f"    ✗ Frame count mismatch: {num_video_frames} frames (expected {expected_frames})")
                except Exception as e:
                    logger.warning(f"  Sample {i}: Failed to inspect - {type(e).__name__}: {e}")
        logger.info("=" * 80)

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

    total_steps = None

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_run_name=args.wandb_run_name,
        ckpt_interval=args.ckpt_interval,
        resume_step_count=resume_step_count,
        save_full_model=_arg('save_full_model', False),
        enable_video_sampling=bool(_arg('enable_video_sampling', False) or _arg('sampling_two_chunk_memory', False)),
        context_drop_prob=float(_arg("context_drop_prob", 0.0) or 0.0),
        sampling_interval_steps=int(_arg("sampling_interval_steps", 0) or 0),
        sampling_two_chunk_memory=bool(_arg("sampling_two_chunk_memory", False)),
        sampling_two_chunk_action_path=_arg("sampling_two_chunk_action_path", None),
        sampling_action_path=_arg("sampling_action_path", None),
        sampling_negative_prompt=_arg("sampling_negative_prompt", "oversaturated colors, overexposed, static, blurry details"),
        sampling_height=int(_arg("sampling_height", 352) or 352),
        sampling_width=int(_arg("sampling_width", 640) or 640),
        sampling_num_frames=int(_arg("sampling_num_frames", 81) or 81),
        sampling_num_inference_steps=int(_arg("sampling_num_inference_steps", 50) or 50),
        context_memory_frames=int(_arg("context_memory_frames", 1) or 1),
        context_source=_arg("context_source", "fov"),
        context_per_frame_vae=bool(_arg("context_per_frame_vae", False)),
        sampling_eval_dataset_base=_arg("sampling_eval_dataset_base", None),
        sampling_eval_metadata_path=_arg("sampling_eval_metadata_path", None),
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