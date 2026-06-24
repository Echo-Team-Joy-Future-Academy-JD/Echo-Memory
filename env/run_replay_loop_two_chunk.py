#!/usr/bin/env python3
"""
回环：两栏输出 [输入轨迹 | 生成视频]。context 一律放在右侧（suffix），I2V 首帧也放在最右作为 clean 去噪 target。

- 第一帧：从训练数据随机采样一帧作为首 chunk 的 context（I2V，1 帧放右侧）
- 之后每 chunk 的第一帧（context）= 上一 chunk 的 last n 帧，同样放在该段生成视频的右侧
- 右栏视频 = 每段 [生成 81 帧 | context 帧]，context 始终在右

默认回环两场景（每 chunk 45°）：1) 1左1右 2chunk  2) 2左2右 4chunk。--run_legacy_loop 时跑旧三组（roundtrip/onedir/replay）。多卡按 rank 划分样本。

多 chunk 时 RT（相对位姿）与训练一致：
- 训练：ref_rt = rt_list[0]，target_actions = convert_rt_to_relative(rt_list, ref_rt)，即每段内 action 相对本段首帧。
- 推理：每个 chunk 的 action 也是相对该 chunk 的首帧。chunk1 注入 0°→45° 表示本段内转 45°；
  chunk2 的首帧 = chunk1 的末帧（已 45°），chunk2 再注入 0°→45° 表示在 chunk2 局部再转 45°，世界系共 90°。
  因此是 45+45，不是第二个 chunk 直接 90°（直接 90° 表示相对 chunk2 首帧转 90°，世界系会变成 45+90=135°）。

注意：条件以 action + context 为主；prompt 使用 GT 帧对应视频的文案以补充场景描述，利于生成清晰度。
"""

import os
import sys
import json
import argparse
import random
import math
import io
from datetime import datetime
import torch
import numpy as np
from PIL import Image

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
_exp1_4_2_dir = os.path.join(_repo_root, "ab_study", "exp1_4_2_context_suffix_cam_rt_relative")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _exp1_4_2_dir not in sys.path:
    sys.path.insert(0, _exp1_4_2_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import loop_utils as irc
from diffsynth import save_video

from src.model_training.fov_retrieval import compute_rotation_list
from src.model_training.multichunk_sample_utils import (
    context_frames_for_next_chunk,
    replay_context_from_generated_frames,
)


def encode_context_frames_per_frame(pipe, pil_list, device, dtype=torch.bfloat16):
    """与训练 context_per_frame_vae 一致：每帧单独 VAE 编码再在时间维 concat，不做时序降采样。ctx=K → K 个 latent tokens（5/20 等均一帧一过 VAE）。"""
    if not pil_list:
        return None
    encoded = []
    for pil in pil_list:
        frame_video = pipe.preprocess_video([pil]).to(device=device)
        frame_sq = frame_video.squeeze(0) if frame_video.dim() == 5 else frame_video
        if frame_sq.dim() == 3:
            frame_sq = frame_sq.unsqueeze(0)
        lat_one = pipe.vae.encode([frame_sq], device=pipe.device, tiled=False, tile_size=None, tile_stride=None)
        encoded.append(lat_one)
    context_latents = torch.cat(encoded, dim=2).to(dtype=dtype, device=device)
    return context_latents


def _yaw_deg_from_rt(rt_list):
    """从 12 维 RT [t_x,t_y,t_z, R11,R12,...,R33] 提取 yaw（度）。R_z 时 yaw = atan2(R21, R11)."""
    if not rt_list or len(rt_list) < 12:
        return 0.0
    R11, R21 = float(rt_list[3]), float(rt_list[6])
    return math.degrees(math.atan2(R21, R11))


def build_action_ccw_cw(deg: float, chunk_frames: int = 81):
    """R_z(yaw)，相对本段首帧。CCW=0→+deg，CW=0→-deg。匀速。"""
    denom = max(1, chunk_frames - 1)
    actions_ccw = {}
    for i in range(chunk_frames):
        yaw = (i / denom) * deg
        actions_ccw[str(i)] = compute_rotation_list([0.0, 0.0, 0.0, yaw])
    actions_cw = {}
    for i in range(chunk_frames):
        yaw = -(i / denom) * deg
        actions_cw[str(i)] = compute_rotation_list([0.0, 0.0, 0.0, yaw])
    return actions_ccw, actions_cw


def build_action_cw_then_ccw(deg: float, chunk_frames: int = 81):
    """匀速：chunk1 顺时针 0→-deg，chunk2 逆时针 0→+deg（本段局部）。折返：先 CW 30° 再 CCW 30° 回起点。"""
    denom = max(1, chunk_frames - 1)
    ch1 = {}
    for i in range(chunk_frames):
        ch1[str(i)] = compute_rotation_list([0.0, 0.0, 0.0, -(i / denom) * deg])
    ch2 = {}
    for i in range(chunk_frames):
        ch2[str(i)] = compute_rotation_list([0.0, 0.0, 0.0, (i / denom) * deg])
    return ch1, ch2


def build_gt_trajectory_actions(dataset_base, video_name, start_frame, chunk_frames, json_file=None):
    """与 Replay 完全一致：从数据集 json 取同段 GT pose，转成 relative RT 作为 action。若帧不足则返回 None。"""
    if json_file is None:
        json_file = os.path.join(dataset_base, "jsons", f"{video_name}.json")
    if not os.path.isfile(json_file):
        return None
    try:
        rt_list = [irc.load_pose_rt(json_file, start_frame + i) for i in range(chunk_frames)]
        if not rt_list or any(r is None or len(r) < 12 for r in rt_list):
            return None
        ref_rt = rt_list[0]
        rel_actions = {str(i): irc.get_relative_rt(rt_list[i], ref_rt) for i in range(chunk_frames)}
        return rel_actions
    except Exception:
        return None


def build_action_from_gt_yaw_profile(dataset_base, video_name, start_frame, chunk_frames, target_total_deg, clockwise=False, json_file=None):
    """参考训练集方式：用同段 GT 轨迹的 yaw 曲线形状（时间分布）缩放为目标角度，使旋转节奏与训练集一致，而非简单线性。
    若该段无 GT 或总转角过小则返回 None，调用方回退线性合成。"""
    if json_file is None:
        json_file = os.path.join(dataset_base, "jsons", f"{video_name}.json")
    if not os.path.isfile(json_file):
        return None
    try:
        rt_list = [irc.load_pose_rt(json_file, start_frame + i) for i in range(chunk_frames)]
        if not rt_list or any(r is None or len(r) < 12 for r in rt_list):
            return None
        ref_rt = rt_list[0]
        # GT 相对首帧的 yaw 曲线（与训练一致）
        yaw_deg = [_yaw_deg_from_rt(irc.get_relative_rt(rt_list[i], ref_rt)) for i in range(chunk_frames)]
        total_gt_deg = yaw_deg[-1] - yaw_deg[0] if len(yaw_deg) > 1 else 0.0
        if abs(total_gt_deg) < 1e-4:
            return None  # 几乎无旋转，用线性更稳
        # 归一化曲线 t[i] in [0,1]，再缩放到 target_total_deg，保持 GT 的时间节奏
        scale = target_total_deg / total_gt_deg
        sign = -1 if clockwise else 1
        actions = {}
        for i in range(chunk_frames):
            t = (yaw_deg[i] - yaw_deg[0]) / total_gt_deg  # 0 -> 1
            yaw = sign * t * target_total_deg
            actions[str(i)] = compute_rotation_list([0.0, 0.0, 0.0, yaw])
        return actions
    except Exception:
        return None


def build_action_chunk(deg: float, clockwise: bool, chunk_frames: int = 81):
    """单 chunk：0→deg（逆时针）或 0→-deg（顺时针）。相对本 chunk 首帧，与训练一致。
    多 chunk 时：chunk2 的首帧=chunk1 的末帧，所以 chunk2 注入 0→deg 表示在 chunk2 局部再转 deg；
    世界系转角 = chunk1 末帧 yaw + deg。例如 chunk1=45°，chunk2 再 45° 则注入 0→45°（不是 0→90°）。
    返回 (actions_dict, yaw_list_deg)."""
    denom = max(1, chunk_frames - 1)
    actions = {}
    yaw_list = []
    for i in range(chunk_frames):
        yaw = (i / denom) * (-deg if clockwise else deg)
        yaw_list.append(yaw)
        actions[str(i)] = compute_rotation_list([0.0, 0.0, 0.0, yaw])
    return actions, yaw_list


def build_action_translation_only(direction: str, translation_delta: float, chunk_frames: int = 81):
    """纯平移单 chunk：R=I，沿单轴线性位移。与训练一致：fov_retrieval.pose_to_rt(..., constrain_to_xy=True) 仅用 XY，tz=0。
    direction in ('forward','backward','left','right')：forward=+Y, backward=-Y, left=-X, right=+X（XY 平面，无 Z）。
    RT 格式：前 3 维 [tx,ty,tz]，后 9 维 3x3 行优先单位阵；相对本段首帧（与 convert_rt_to_relative(ref=首帧) 一致）。"""
    # 与训练一致：2D 平面位移，z 恒为 0（见 fov_retrieval.pose_to_rt constrain_to_xy=True）
    identity_rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    denom = max(1, chunk_frames - 1)
    actions = {}
    yaw_list = [0.0] * chunk_frames
    for i in range(chunk_frames):
        t = (i / denom) * translation_delta
        if direction == "forward":
            tx, ty, tz = 0.0, t, 0.0
        elif direction == "backward":
            tx, ty, tz = 0.0, -t, 0.0
        elif direction == "left":
            tx, ty, tz = -t, 0.0, 0.0
        elif direction == "right":
            tx, ty, tz = t, 0.0, 0.0
        else:
            tx, ty, tz = 0.0, 0.0, 0.0
        actions[str(i)] = [tx, ty, tz] + identity_rot
    return actions, yaw_list


def _draw_trajectory_pil(yaw_list, plot_width, plot_height, frame_index):
    """PIL-only fallback: draw trajectory 0..frame_index. Returns one PIL Image."""
    pad = 40
    w, h = plot_width - 2 * pad, plot_height - 2 * pad
    if w <= 0 or h <= 0:
        return Image.new("RGB", (plot_width, plot_height), (50, 50, 50))
    img = Image.new("RGB", (plot_width, plot_height), (45, 45, 48))
    end = min(frame_index + 1, len(yaw_list))
    if end == 0:
        return img
    ys = [float(yaw_list[i]) for i in range(end)]
    y_min, y_max = min(ys), max(ys)
    if y_max <= y_min:
        y_min, y_max = y_min - 10.0, y_max + 10.0
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    n_x = max(1, len(yaw_list))
    pts = []
    for i in range(end):
        x = pad + int((i / max(1, n_x - 1)) * w) if n_x > 1 else pad
        yy = (ys[i] - y_min) / max(1e-6, y_max - y_min)
        y = pad + int((1 - yy) * h)
        pts.append((x, y))
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 200, 255), width=2)
    if pts:
        draw.ellipse([pts[-1][0] - 4, pts[-1][1] - 4, pts[-1][0] + 4, pts[-1][1] + 4], fill=(255, 220, 0), outline=(255, 255, 255))
    return img


def draw_trajectory_frames(yaw_history_deg, plot_width, plot_height, total_frames=None):
    """
    yaw_history_deg: list of cumulative yaw (one per frame).
    Returns list of PIL images (one per frame), each showing trajectory from 0 to current frame.
    Uses matplotlib; if output is too dark, falls back to PIL-only drawing.
    """
    yaw_list = [float(y) for y in (yaw_history_deg or [])]
    n = len(yaw_list)
    if total_frames is not None and total_frames > n:
        n = total_frames
    if n == 0:
        return [Image.new("RGB", (plot_width, plot_height), (45, 45, 48))]

    use_pil_fallback = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        use_pil_fallback = True

    if not use_pil_fallback:
        y_min = min(yaw_list) if yaw_list else 0.0
        y_max = max(yaw_list) if yaw_list else 0.0
        if y_max <= y_min:
            y_min, y_max = y_min - 15.0, y_max + 15.0
        margin = max(5, (y_max - y_min) * 0.15)
        y_lo, y_hi = y_min - margin, y_max + margin
        out = []
        for t in range(n):
            fig, ax = plt.subplots(1, 1, figsize=(plot_width / 100.0, plot_height / 100.0), dpi=100)
            fig.patch.set_facecolor("#252525")
            ax.set_facecolor("#252525")
            ax.set_xlim(0, max(n, 1))
            ax.set_ylim(y_lo, y_hi)
            ax.tick_params(colors="0.9", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("0.6")
            end = min(t + 1, len(yaw_list))
            if end > 0:
                x = list(range(end))
                y = [yaw_list[i] for i in range(end)]
                ax.plot(x, y, color="cyan", linewidth=2.0, zorder=2)
                cur_y = yaw_list[min(t, len(yaw_list) - 1)] if t < len(yaw_list) else (yaw_list[-1] if yaw_list else 0)
                ax.scatter([t], [cur_y], color="yellow", s=35, zorder=3)
            ax.set_xlabel("Frame", color="0.9", fontsize=8)
            ax.set_ylabel("Yaw (deg)", color="0.9", fontsize=8)
            ax.set_title("Trajectory (yaw)", color="0.95", fontsize=9)
            fig.tight_layout(pad=0.5)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, facecolor="#252525", edgecolor="none", bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            if img.size != (plot_width, plot_height):
                img = img.resize((plot_width, plot_height), Image.Resampling.LANCZOS)
            # If image is too dark (matplotlib failed in headless?), use PIL fallback for rest
            arr = np.array(img)
            if arr.mean() < 50:
                use_pil_fallback = True
                out = [_draw_trajectory_pil(yaw_list, plot_width, plot_height, t) for t in range(n)]
                break
            out.append(img)
        if not use_pil_fallback:
            return out
    if use_pil_fallback:
        return [_draw_trajectory_pil(yaw_list, plot_width, plot_height, t) for t in range(n)]
    return out


def composite_two_panel(trajectory_frames, right_panel_frames, w_traj=320, h_traj=352, w_gen=640, h_gen=352):
    """左：输入轨迹可视化；右：生成视频（每段后接 context，context 在右）。"""
    n = len(right_panel_frames)
    if not trajectory_frames or len(trajectory_frames) != n:
        traj_frames = draw_trajectory_frames([], w_traj, h_traj, total_frames=n)
    else:
        traj_frames = trajectory_frames
    out = []
    for i in range(n):
        traj = traj_frames[i] if i < len(traj_frames) else traj_frames[-1]
        if traj.size != (w_traj, h_traj):
            traj = traj.resize((w_traj, h_traj))
        gen = _frame_to_pil(right_panel_frames[i], w_gen, h_gen)
        canvas_w = w_traj + w_gen
        canvas_h = max(h_traj, h_gen)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))
        canvas.paste(traj, (0, 0))
        canvas.paste(gen, (w_traj, 0))
        out.append(canvas)
    return out


def load_sample_first_frame(dataset_base, video_name, start_frame, w, h):
    """加载当前样本的首帧（与 Replay 一致：context = 同条轨迹同视频的首帧），避免 loop 用随机帧导致 prompt 与 context 场景不一致而糊。返回 PIL 或 None。"""
    if not video_name or start_frame is None:
        return None
    for fmt in (f"{int(start_frame):04d}.png", f"{int(start_frame)}.png"):
        img_p = os.path.join(dataset_base, "frames", str(video_name), fmt)
        if os.path.isfile(img_p):
            return Image.open(img_p).convert("RGB").resize((w, h))
    return None


def load_gt_frames_at_indices(dataset_base, video_name, frame_indices, w, h):
    """从数据集加载指定帧序号的 GT 图像，用于 chunk 间衔接的 context（与训练一致：context=真实帧，避免用生成帧再编码导致糊）。
    frame_indices: 从左到右使用（如 [80, 79] 表示最后一帧、倒数第二帧）。返回 list[PIL] 若全部存在，否则 None。"""
    if not video_name or not frame_indices:
        return None
    out = []
    base = os.path.join(dataset_base, "frames", str(video_name))
    for idx in frame_indices:
        for fmt in (f"{int(idx):04d}.png", f"{int(idx)}.png"):
            p = os.path.join(base, fmt)
            if os.path.isfile(p):
                out.append(Image.open(p).convert("RGB").resize((w, h)))
                break
        else:
            return None
    return out


def seed_for_sample(base_seed: int, video_name, start_frame) -> int:
    """与 (video_name, start_frame) 一一对应的确定性 seed，使同一样本在 loop_traj_fov_gen 与 evals_ep0 等不同调用下结果一致（避免因 idx/rank 不同导致偏移）。"""
    try:
        h = hash((str(video_name), int(start_frame)))
        return base_seed + (h & 0x7FFFFFFF) % 100000
    except Exception:
        return base_seed


def sample_random_frame_from_dataset(dataset_base, w, h, seed, metadata_path=None):
    """从训练数据中随机采样一帧（用于首 chunk 的 I2V context，放右侧）。返回 PIL 或 None。"""
    import csv
    candidates = []
    if metadata_path and os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vn = row.get("video_name", "").strip()
                    sf = row.get("start_frame", "")
                    if vn and str(sf).strip():
                        try:
                            candidates.append((vn, int(sf)))
                        except ValueError:
                            continue
        except Exception:
            pass
    if not candidates:
        frames_dir = os.path.join(dataset_base, "frames")
        if os.path.isdir(frames_dir):
            for vn in sorted(os.listdir(frames_dir)):
                vd = os.path.join(frames_dir, vn)
                if not os.path.isdir(vd):
                    continue
                pngs = [f for f in os.listdir(vd) if f.endswith(".png")]
                if not pngs:
                    continue
                for p in pngs[:3]:
                    try:
                        frame_idx = int(os.path.splitext(p)[0])
                        candidates.append((vn, frame_idx))
                    except ValueError:
                        continue
    if not candidates:
        return None
    rng = random.Random(seed)
    vn, frame_idx = rng.choice(candidates)
    img_p = os.path.join(dataset_base, "frames", vn, f"{frame_idx:04d}.png")
    if not os.path.isfile(img_p):
        img_p = os.path.join(dataset_base, "frames", vn, f"{frame_idx}.png")
    if os.path.isfile(img_p):
        return Image.open(img_p).convert("RGB").resize((w, h))
    return None


def trim_continuation_first_frame(chunk_frames_list, yaw_history, chunk_frames=81):
    """与训练对齐：训练时 context[0]=target[0]（同一帧），续段首帧=上一段末帧。拼接时去掉续段第 0 帧避免重复/跳帧。
    返回 (trimmed_chunk_frames_list, trimmed_yaw)：chunk0 保留 81 帧，chunk1 及之后只保留 [1:81] 共 80 帧。"""
    if not chunk_frames_list or len(chunk_frames_list) <= 1:
        return chunk_frames_list, yaw_history
    trimmed_chunks = [list(chunk_frames_list[0])]
    trimmed_yaw = []
    offset = 0
    for i, gen_frames in enumerate(chunk_frames_list):
        n_use = len(gen_frames) if i == 0 else max(0, len(gen_frames) - 1)
        if i == 0:
            trimmed_yaw.extend(yaw_history[offset : offset + n_use])
        else:
            trimmed_yaw.extend(yaw_history[offset + 1 : offset + 1 + n_use])
        offset += len(gen_frames)
        if i > 0 and len(gen_frames) > 1:
            trimmed_chunks.append(list(gen_frames[1:]))
        elif i > 0:
            trimmed_chunks.append([])
    return trimmed_chunks, trimmed_yaw


def build_right_panel_and_yaw(chunk_frames_list, yaw_history, context_frames_per_chunk, chunk_frames=81, w=640, h=352, frame_to_pil_fn=None):
    """从每段生成帧 + 每段后的 context 帧拼出右栏；并扩展 yaw 与右栏帧数一致（context 帧沿用上一 yaw）。
    支持变长 chunk（如 trim 后首段 81、续段 80），按 yaw_history 顺序逐帧对齐。"""
    fp = frame_to_pil_fn or _frame_to_pil
    right_frames = []
    yaw_extended = []
    yaw_idx = 0
    for i, gen_frames in enumerate(chunk_frames_list):
        for j, f in enumerate(gen_frames):
            right_frames.append(f)
            yaw_extended.append(yaw_history[yaw_idx] if yaw_idx < len(yaw_history) else (yaw_history[-1] if yaw_history else 0.0))
            yaw_idx += 1
        ctx_list = context_frames_per_chunk[i] if i < len(context_frames_per_chunk) else []
        last_yaw = yaw_extended[-1] if yaw_extended else 0.0
        for ctx in ctx_list:
            right_frames.append(ctx if isinstance(ctx, Image.Image) else fp(ctx, w, h))
            yaw_extended.append(last_yaw)
    return right_frames, yaw_extended


def _frame_to_pil(f, tw, th):
    if hasattr(f, "convert") and hasattr(f, "resize"):
        return f.convert("RGB").resize((tw, th))
    if isinstance(f, np.ndarray):
        if f.dtype != np.uint8:
            f = (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
        return Image.fromarray(f).convert("RGB").resize((tw, th))
    if isinstance(f, torch.Tensor):
        fn = f.cpu().numpy()
        if len(fn.shape) == 3 and fn.shape[0] == 3:
            fn = fn.transpose(1, 2, 0)
        fn = (fn * 255).clip(0, 255).astype(np.uint8) if fn.max() <= 1.0 else fn.clip(0, 255).astype(np.uint8)
        return Image.fromarray(fn).convert("RGB").resize((tw, th))
    return f


def run_one_chunk(
    pipe,
    prompt,
    use_negative_prompt,
    action_path=None,
    cam_pose_actions=None,
    context_latents=None,
    num_context_frames=1,
    context_actions_t=None,
    chunk_frames=81,
    h=352,
    w=640,
    seed=0,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    inference_noise_level=0.0,
    omit_context_actions=False,  # kept for backward compat, no longer used
    log_prefix="[Loop]",
    **_extra,
):
    """Generate one chunk. VWM-aligned action injection via cam_pose_actions (latent-frame-aligned 12-D RT)."""
    device = pipe.device
    kwargs_common = dict(
        prompt=prompt,
        negative_prompt=use_negative_prompt,
        height=h, width=w, num_frames=chunk_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        cfg_scale=cfg_scale,
        sigma_shift=sigma_shift,
        denoising_strength=1.0,
    )
    if action_path is not None:
        kwargs_common["action_path"] = action_path
    elif cam_pose_actions is not None:
        kwargs_common["cam_pose_actions"] = cam_pose_actions

    has_action_mlp = hasattr(pipe.dit.blocks[0], 'action_mlp') if len(pipe.dit.blocks) > 0 else False
    if (action_path or cam_pose_actions is not None) and not has_action_mlp:
        print(f"{log_prefix} 警告: dit.blocks[0] 无 action_mlp，action 可能未注入（请确认 ckpt 含 action_mlp 权重）", flush=True)
    print(f"{log_prefix} run_one_chunk sigma_shift={sigma_shift} steps={num_inference_steps} (ctx={num_context_frames}) inference_noise={inference_noise_level} context_mem={context_latents is not None} action_mlp={has_action_mlp}")
    if context_latents is not None:
        pipe_kw = dict(
            **kwargs_common,
            enable_context_memory=True,
            context_latents=context_latents,
            num_context_frames=num_context_frames,
            context_position="suffix",
            cfg_target_only=True,
            inference_noise_level=inference_noise_level,
        )
        if context_actions_t is not None:
            pipe_kw["context_actions"] = context_actions_t
        with torch.no_grad():
            vid = pipe(**pipe_kw)
    else:
        with torch.no_grad():
            vid = pipe(**kwargs_common, enable_context_memory=False)
    return vid if isinstance(vid, list) else [vid]


def run_roundtrip_2chunk(
    pipe,
    dataset_base,
    output_dir,
    video_name,
    start_frame,
    deg=30.0,
    chunk_frames=81,
    context_frames=1,
    h=352,
    w=640,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    inference_noise_level=0.0,
    metadata_path=None,
    keep_action_jsons=False,
    omit_context_actions=True,
):
    """2 chunk 折返：匀速 顺时针 deg° 再 逆时针 deg°（回起点）。"""
    print(f"[Roundtrip] 匀速 顺时针{deg}° then 逆时针{deg}° sigma_shift={sigma_shift} omit_ctx_act={omit_context_actions}")
    json_file = os.path.join(dataset_base, "jsons", f"{video_name}.json")
    prompt = irc.load_prompt_for_video(dataset_base, video_name) or "A scene."
    use_negative_prompt = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")

    actions_ch1, actions_ch2 = build_action_cw_then_ccw(deg, chunk_frames)
    subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
    os.makedirs(subdir, exist_ok=True)
    path_ch1 = os.path.join(subdir, "_ch1_cw.json")
    path_ch2 = os.path.join(subdir, "_ch2_ccw.json")
    with open(path_ch1, "w") as f:
        json.dump(actions_ch1, f, indent=2)
    with open(path_ch2, "w") as f:
        json.dump(actions_ch2, f, indent=2)
    print(f"[Roundtrip] Chunk1 CW 0->-{deg}° Chunk2 CCW 0->+{deg}° (匀速)")

    denom = max(1, chunk_frames - 1)
    yaw_history = [-(i / denom) * deg for i in range(chunk_frames)]  # chunk1: 0 -> -deg
    last_y1 = -deg
    yaw_history += [last_y1 + (i / denom) * deg for i in range(chunk_frames)]  # chunk2: -deg -> 0

    chunk_frames_list = []
    context_frames_per_chunk = []

    # Chunk 1: context = 当前样本首帧（与 Replay 一致，同视频同场景），避免随机帧导致糊；若无则回退随机帧
    ctx_pil_0 = load_sample_first_frame(dataset_base, video_name, start_frame, w, h)
    used_sample_first = ctx_pil_0 is not None
    if ctx_pil_0 is None:
        ctx_pil_0 = sample_random_frame_from_dataset(dataset_base, w, h, seed, metadata_path)
    if ctx_pil_0 is not None:
        print(f"[Roundtrip] Chunk1 context: {'sample first frame ' + str((video_name, start_frame)) if used_sample_first else 'random frame (fallback)'}")
    if ctx_pil_0 is not None:
        ctx_pil_0 = [ctx_pil_0]
        pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            ctx_latents_0 = encode_context_frames_per_frame(pipe, ctx_pil_0, pipe.device)
        identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ctx_actions_t_0 = torch.tensor([identity_rt], dtype=torch.float32)
        frames_ch1 = run_one_chunk(
            pipe, prompt, use_negative_prompt, path_ch1,
            context_latents=ctx_latents_0,
            num_context_frames=ctx_latents_0.shape[2],
            context_actions_t=ctx_actions_t_0,
            chunk_frames=chunk_frames, h=h, w=w, seed=seed,
            sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
            omit_context_actions=omit_context_actions,
        )
        context_frames_per_chunk.append([ctx_pil_0[0]])
    else:
        frames_ch1 = run_one_chunk(
            pipe, prompt, use_negative_prompt, path_ch1,
            chunk_frames=chunk_frames, h=h, w=w, seed=seed,
            sigma_shift=sigma_shift, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale,
            omit_context_actions=omit_context_actions,
        )
        context_frames_per_chunk.append([_frame_to_pil(frames_ch1[-1], w, h)])
    chunk_frames_list.append(frames_ch1)

    # Chunk 2: condition = 上一 chunk，顺序与训练一致 [last_frame, ctx1, ...] 紧挨噪声后
    n_ctx = min(context_frames, len(frames_ch1))
    prev_frames = replay_context_from_generated_frames(frames_ch1, n_ctx)
    prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        context_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
    num_ctx_tokens = context_latents.shape[2]
    context_actions_t = torch.tensor([identity_rt] * num_ctx_tokens, dtype=torch.float32)
    print(f"[Loop] continuation chunk: len(prev_pil)={len(prev_pil)} num_ctx_tokens={num_ctx_tokens} (应等于 ctx)")
    frames_ch2 = run_one_chunk(
        pipe, prompt, use_negative_prompt, path_ch2,
        context_latents=context_latents,
        num_context_frames=num_ctx_tokens,
        context_actions_t=context_actions_t,
        chunk_frames=chunk_frames, h=h, w=w, seed=seed + 1,
        sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
        omit_context_actions=omit_context_actions,
    )
    chunk_frames_list.append(frames_ch2)
    context_frames_per_chunk.append(list(prev_pil))

    if not keep_action_jsons:
        for p in [path_ch1, path_ch2]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    return chunk_frames_list, yaw_history, context_frames_per_chunk


def run_single_chunk_rotation(
    pipe,
    dataset_base,
    output_dir,
    video_name,
    start_frame,
    deg=45.0,
    clockwise=False,
    chunk_frames=81,
    h=352,
    w=640,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    inference_noise_level=0.0,
    metadata_path=None,
    keep_action_jsons=False,
    sampling_action_dir=None,
    omit_context_actions=True,
    context_image_path=None,
):
    """旋转原子操作：单 chunk 左转(CCW)或右转(CW) deg°。优先使用与训练采样相同的 action JSON。omit_context_actions 与训练 ctx 设置对齐。context_image_path：泛化检查时用指定图片作为首帧 context。"""
    label = "右转(CW)" if clockwise else "左转(CCW)"
    print(f"[SingleChunk] 1 chunk {label} {deg}° sigma_shift={sigma_shift}")
    prompt = irc.load_prompt_for_video(dataset_base, video_name) or "A scene."
    use_negative_prompt = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")

    # 与采样对齐：45°、81 帧时优先用采样同款 JSON（generate_rotation_actions.py 生成），注入方式一致
    path_ch = None
    if deg == 45.0 and chunk_frames == 81:
        action_dir = sampling_action_dir if sampling_action_dir else _script_dir
        if clockwise:
            candidate = os.path.join(action_dir, "action_rotation_right_45.json")
        else:
            candidate = os.path.join(action_dir, "action_rotation_left_45.json")
        if os.path.isfile(candidate):
            path_ch = candidate
            print(f"[SingleChunk] 使用采样同款 action: {os.path.basename(path_ch)} (与训练采样注入一致)", flush=True)

    if path_ch is None:
        actions, yaw_chunk = build_action_chunk(deg, clockwise, chunk_frames)
        yaw0 = _yaw_deg_from_rt(actions["0"])
        yaw_last = _yaw_deg_from_rt(actions[str(chunk_frames - 1)])
        print(f"[SingleChunk] action 首帧 yaw={yaw0:.1f}° 末帧 yaw={yaw_last:.1f}° (预期: 左转 0→+{deg}° 右转 0→-{deg}°)", flush=True)
        subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
        os.makedirs(subdir, exist_ok=True)
        suffix = "cw_right" if clockwise else "ccw_left"
        path_ch = os.path.join(subdir, f"_single_{suffix}.json")
        with open(path_ch, "w") as f:
            json.dump(actions, f, indent=2)
        yaw_history = list(yaw_chunk)
        delete_path_after = not keep_action_jsons
    else:
        with open(path_ch, "r") as f:
            actions = json.load(f)
        yaw_history = [_yaw_deg_from_rt(actions.get(str(i), [0] * 12)) for i in range(min(chunk_frames, len(actions)))]
        if len(yaw_history) < chunk_frames:
            yaw_history += [yaw_history[-1]] * (chunk_frames - len(yaw_history))
        delete_path_after = False

    subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
    os.makedirs(subdir, exist_ok=True)
    yaw_history = yaw_history[:chunk_frames] if len(yaw_history) > chunk_frames else yaw_history
    chunk_frames_list = []
    context_frames_per_chunk = []

    if context_image_path and os.path.isfile(context_image_path):
        ctx_pil_0 = Image.open(context_image_path).convert("RGB").resize((w, h))
        print(f"[SingleChunk] 使用指定 context 图: {context_image_path}", flush=True)
    else:
        ctx_pil_0 = load_sample_first_frame(dataset_base, video_name, start_frame, w, h)
    if ctx_pil_0 is None:
        ctx_pil_0 = sample_random_frame_from_dataset(dataset_base, w, h, seed, metadata_path)
    if ctx_pil_0 is not None:
        ctx_pil_0 = [ctx_pil_0]
        pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            ctx_latents_0 = encode_context_frames_per_frame(pipe, ctx_pil_0, pipe.device)
        identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ctx_actions_t_0 = torch.tensor([identity_rt], dtype=torch.float32)
        frames_ch = run_one_chunk(
            pipe, prompt, use_negative_prompt, path_ch,
            context_latents=ctx_latents_0,
            num_context_frames=ctx_latents_0.shape[2],
            context_actions_t=ctx_actions_t_0,
            chunk_frames=chunk_frames, h=h, w=w, seed=seed,
            sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
            omit_context_actions=omit_context_actions,
        )
        context_frames_per_chunk.append([ctx_pil_0[0]])
    else:
        frames_ch = run_one_chunk(
            pipe, prompt, use_negative_prompt, path_ch,
            chunk_frames=chunk_frames, h=h, w=w, seed=seed,
            sigma_shift=sigma_shift, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale,
            omit_context_actions=omit_context_actions,
        )
        context_frames_per_chunk.append([_frame_to_pil(frames_ch[-1], w, h)])
    chunk_frames_list.append(frames_ch)

    if delete_path_after and os.path.exists(path_ch):
        try:
            os.remove(path_ch)
        except Exception:
            pass
    return chunk_frames_list, yaw_history, context_frames_per_chunk


def run_left_right_2chunk(
    pipe,
    dataset_base,
    output_dir,
    video_name,
    start_frame,
    deg=45.0,
    chunk_frames=81,
    context_frames=1,
    h=352,
    w=640,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    inference_noise_level=0.0,
    metadata_path=None,
    keep_action_jsons=False,
    sampling_action_dir=None,
    omit_context_actions=True,
):
    """回环：1 chunk 左转(CCW) deg°，1 chunk 右转(CW) deg°。45°×81 时优先用采样同款 left_45/right_45.json。"""
    print(f"[LeftRight 2chunk] 左转(CCW){deg}° then 右转(CW){deg}° sigma_shift={sigma_shift} omit_ctx_act={omit_context_actions} context_frames={context_frames} (续段 ctx 数)")
    prompt = irc.load_prompt_for_video(dataset_base, video_name) or "A scene."
    use_negative_prompt = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")

    subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
    os.makedirs(subdir, exist_ok=True)
    action_dir = sampling_action_dir if sampling_action_dir else _script_dir
    path_ch1 = path_ch2 = None
    delete_ch1 = delete_ch2 = True
    if deg == 45.0 and chunk_frames == 81:
        p_left = os.path.join(action_dir, "action_rotation_left_45.json")
        p_right = os.path.join(action_dir, "action_rotation_right_45.json")
        if os.path.isfile(p_left) and os.path.isfile(p_right):
            path_ch1, path_ch2 = p_left, p_right
            delete_ch1 = delete_ch2 = False
            print(f"[LeftRight 2chunk] 使用采样同款 action: left_45 + right_45 (与训练采样注入一致)", flush=True)
    if path_ch1 is None:
        actions_ccw, actions_cw = build_action_ccw_cw(deg, chunk_frames)
        path_ch1 = os.path.join(subdir, "_ch1_ccw_left.json")
        path_ch2 = os.path.join(subdir, "_ch2_cw_right.json")
        with open(path_ch1, "w") as f:
            json.dump(actions_ccw, f, indent=2)
        with open(path_ch2, "w") as f:
            json.dump(actions_cw, f, indent=2)
        delete_ch1 = delete_ch2 = not keep_action_jsons

    if not delete_ch1 and path_ch1 and path_ch2:
        with open(path_ch1, "r") as f:
            a1 = json.load(f)
        with open(path_ch2, "r") as f:
            a2 = json.load(f)
        yaw_left = [_yaw_deg_from_rt(a1.get(str(i), [0] * 12)) for i in range(min(chunk_frames, len(a1)))]
        yaw_right = [_yaw_deg_from_rt(a2.get(str(i), [0] * 12)) for i in range(min(chunk_frames, len(a2)))]
        if len(yaw_left) < chunk_frames:
            yaw_left += [yaw_left[-1]] * (chunk_frames - len(yaw_left))
        if len(yaw_right) < chunk_frames:
            yaw_right += [yaw_right[-1]] * (chunk_frames - len(yaw_right))
        base = yaw_left[-1] if yaw_left else 45.0
        yaw_history = yaw_left[:chunk_frames] + [base + yaw_right[i] for i in range(chunk_frames)]
    else:
        denom = max(1, chunk_frames - 1)
        yaw_history = [(i / denom) * deg for i in range(chunk_frames)]
        yaw_history += [deg - (i / denom) * deg for i in range(chunk_frames)]

    chunk_frames_list = []
    context_frames_per_chunk = []

    ctx_pil_0 = load_sample_first_frame(dataset_base, video_name, start_frame, w, h)
    used_sample_first = ctx_pil_0 is not None
    if ctx_pil_0 is None:
        ctx_pil_0 = sample_random_frame_from_dataset(dataset_base, w, h, seed, metadata_path)
    if ctx_pil_0 is not None:
        ctx_pil_0 = [ctx_pil_0]
        pipe.load_models_to_device(["vae"])
        with torch.no_grad():
            ctx_latents_0 = encode_context_frames_per_frame(pipe, ctx_pil_0, pipe.device)
        identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ctx_actions_t_0 = torch.tensor([identity_rt], dtype=torch.float32)
        frames_ch1 = run_one_chunk(
            pipe, prompt, use_negative_prompt, path_ch1,
            context_latents=ctx_latents_0,
            num_context_frames=ctx_latents_0.shape[2],
            context_actions_t=ctx_actions_t_0,
            chunk_frames=chunk_frames, h=h, w=w, seed=seed,
            sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
            omit_context_actions=omit_context_actions,
        )
        context_frames_per_chunk.append([ctx_pil_0[0]])
    else:
        frames_ch1 = run_one_chunk(
            pipe, prompt, use_negative_prompt, path_ch1,
            chunk_frames=chunk_frames, h=h, w=w, seed=seed,
            sigma_shift=sigma_shift, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale,
            omit_context_actions=omit_context_actions,
        )
        context_frames_per_chunk.append([_frame_to_pil(frames_ch1[-1], w, h)])
    chunk_frames_list.append(frames_ch1)

    n_ctx = min(context_frames, len(frames_ch1))
    prev_frames = replay_context_from_generated_frames(frames_ch1, n_ctx)
    prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        context_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
    num_ctx_tokens = context_latents.shape[2]
    context_actions_t = torch.tensor([identity_rt] * num_ctx_tokens, dtype=torch.float32)
    print(f"[Loop] continuation chunk: len(prev_pil)={len(prev_pil)} num_ctx_tokens={num_ctx_tokens} (应等于 ctx)")
    frames_ch2 = run_one_chunk(
        pipe, prompt, use_negative_prompt, path_ch2,
        context_latents=context_latents,
        num_context_frames=num_ctx_tokens,
        context_actions_t=context_actions_t,
        chunk_frames=chunk_frames, h=h, w=w, seed=seed + 1,
        sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
        omit_context_actions=omit_context_actions,
    )
    chunk_frames_list.append(frames_ch2)
    context_frames_per_chunk.append(list(prev_pil))

    if delete_ch1 and path_ch1 and os.path.exists(path_ch1):
        try:
            os.remove(path_ch1)
        except Exception:
            pass
    if delete_ch2 and path_ch2 and os.path.exists(path_ch2):
        try:
            os.remove(path_ch2)
        except Exception:
            pass
    return chunk_frames_list, yaw_history, context_frames_per_chunk


def run_translation_4chunk(
    pipe,
    dataset_base,
    output_dir,
    video_name,
    start_frame,
    direction="forward",
    translation_delta=0.1,
    chunk_frames=81,
    context_frames=1,
    h=352,
    w=640,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    inference_noise_level=0.0,
    metadata_path=None,
    omit_context_actions=True,
):
    """仅纯平移 4chunk：同一方向（forward/backward/left/right）连续 4 段，R=I，无旋转。仅用于 --run_atomic_translation_4chunk。"""
    _prefix = "[atomic_translation_4chunk]"
    print(f"{_prefix} 纯平移 direction={direction} delta={translation_delta} sigma_shift={sigma_shift} omit_ctx_act={omit_context_actions} context_frames={context_frames} (续段 ctx 数)")
    prompt = irc.load_prompt_for_video(dataset_base, video_name) or "A scene."
    use_negative_prompt = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")

    yaw_history = []
    chunk_frames_list = []
    context_frames_per_chunk = []
    subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
    os.makedirs(subdir, exist_ok=True)
    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    context_latents = None
    context_actions_t = None

    for ch in range(4):
        actions, yaw_chunk = build_action_translation_only(direction, translation_delta, chunk_frames)
        path_ch = os.path.join(subdir, f"_ch{ch}_trans_{direction}.json")
        with open(path_ch, "w") as f:
            json.dump(actions, f, indent=2)
        try:
            if ch == 0:
                ctx_pil_0 = load_sample_first_frame(dataset_base, video_name, start_frame, w, h)
                if ctx_pil_0 is None:
                    ctx_pil_0 = sample_random_frame_from_dataset(dataset_base, w, h, seed, metadata_path)
                if ctx_pil_0 is not None:
                    ctx_pil_0 = [ctx_pil_0]
                    pipe.load_models_to_device(["vae"])
                    with torch.no_grad():
                        context_latents = encode_context_frames_per_frame(pipe, ctx_pil_0, pipe.device)
                    context_actions_t = torch.tensor([identity_rt], dtype=torch.float32)
                    context_frames_per_chunk.append([ctx_pil_0[0]])
                else:
                    context_frames_per_chunk.append([])

            if context_latents is not None and context_actions_t is not None:
                frames_ch = run_one_chunk(
                    pipe, prompt, use_negative_prompt, path_ch,
                    context_latents=context_latents,
                    num_context_frames=context_latents.shape[2],
                    context_actions_t=context_actions_t,
                    chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch,
                    sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
                    cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
                    omit_context_actions=omit_context_actions,
                    log_prefix=_prefix,
                )
            else:
                frames_ch = run_one_chunk(
                    pipe, prompt, use_negative_prompt, path_ch,
                    chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch,
                    sigma_shift=sigma_shift, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale,
                    omit_context_actions=omit_context_actions,
                    log_prefix=_prefix,
                )
                if ch == 0 and not context_frames_per_chunk[0]:
                    context_frames_per_chunk[0] = [_frame_to_pil(frames_ch[-1], w, h)]

            for y in yaw_chunk:
                yaw_history.append(y)
            chunk_frames_list.append(frames_ch)

            if ch < 3:
                n_ctx = min(context_frames, len(frames_ch))
                prev_frames = context_frames_for_next_chunk(frames_ch, n_ctx) if n_ctx else [frames_ch[-1]]
                prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
                context_frames_per_chunk.append(list(prev_pil))
                pipe.load_models_to_device(["vae"])
                with torch.no_grad():
                    context_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
                num_ctx_tokens = context_latents.shape[2]
                context_actions_t = torch.tensor([identity_rt] * num_ctx_tokens, dtype=torch.float32)
                print(f"{_prefix} continuation chunk ch={ch}: len(prev_pil)={len(prev_pil)} num_ctx_tokens={num_ctx_tokens} (应等于 ctx)")
        finally:
            if os.path.exists(path_ch):
                try:
                    os.remove(path_ch)
                except Exception:
                    pass

    return chunk_frames_list, yaw_history, context_frames_per_chunk


def run_left2_right2_4chunk(
    pipe,
    dataset_base,
    output_dir,
    video_name,
    start_frame,
    deg_per_chunk=45.0,
    chunk_frames=81,
    context_frames=1,
    h=352,
    w=640,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    inference_noise_level=0.0,
    metadata_path=None,
    sampling_action_dir=None,
    omit_context_actions=True,
    multi_ctx_all_history=False,
):
    """回环：先左转 2 chunk（各 deg° CCW），再右转 2 chunk（各 deg° CW）。

    默认行为：与 2chunk 回环一致，续段 context 仅来自上一 chunk 的末尾若干帧。
    当 multi_ctx_all_history=True 时：续段 context 第一帧必须为上一 chunk 的最后一帧；
    其余 (ctx-1) 帧从「前面所有已生成帧」均匀采样（保持时间序），用于记忆机制对比。
    """
    print(f"[Left2Right2 4chunk] 左转2chunk then 右转2chunk sigma_shift={sigma_shift} omit_ctx_act={omit_context_actions} context_frames={context_frames} (续段 ctx 数)")
    prompt = irc.load_prompt_for_video(dataset_base, video_name) or "A scene."
    use_negative_prompt = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")

    yaw_history = []
    chunk_frames_list = []
    context_frames_per_chunk = []
    all_prev_frames = []
    subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
    os.makedirs(subdir, exist_ok=True)
    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    cumulative_yaw = 0.0
    context_latents = None
    context_actions_t = None

    # 45°×81 时直接使用采样同款 JSON，多 chunk 多次导入同一文件即可
    action_dir = sampling_action_dir if sampling_action_dir else _script_dir
    path_left = os.path.join(action_dir, "action_rotation_left_45.json")
    path_right = os.path.join(action_dir, "action_rotation_right_45.json")
    use_sampling_files = (deg_per_chunk == 45.0 and chunk_frames == 81 and
                          os.path.isfile(path_left) and os.path.isfile(path_right))
    if use_sampling_files:
        print(f"[Left2Right2 4chunk] 使用采样同款 action: left_45 x2 + right_45 x2 (与训练采样注入一致)", flush=True)
        with open(path_left, "r") as f:
            _actions_left = json.load(f)
        with open(path_right, "r") as f:
            _actions_right = json.load(f)
        _yaw_left = [_yaw_deg_from_rt(_actions_left.get(str(i), [0] * 12)) for i in range(min(chunk_frames, len(_actions_left)))]
        _yaw_right = [_yaw_deg_from_rt(_actions_right.get(str(i), [0] * 12)) for i in range(min(chunk_frames, len(_actions_right)))]
        if len(_yaw_left) < chunk_frames:
            _yaw_left += [_yaw_left[-1]] * (chunk_frames - len(_yaw_left))
        if len(_yaw_right) < chunk_frames:
            _yaw_right += [_yaw_right[-1]] * (chunk_frames - len(_yaw_right))

    for ch in range(4):
        clockwise = ch >= 2
        if use_sampling_files:
            path_ch = path_right if clockwise else path_left
        else:
            actions, yaw_chunk = build_action_chunk(deg_per_chunk, clockwise, chunk_frames)
            path_ch = os.path.join(subdir, f"_ch{ch}_left2right2.json")
            with open(path_ch, "w") as f:
                json.dump(actions, f, indent=2)

        # 与 2chunk 一致：ch=0 用首帧 context；ch>=1 用上一 chunk 输出做 context（上一轮末尾已写好 context_latents/context_actions_t）
        if ch == 0:
            ctx_pil_0 = load_sample_first_frame(dataset_base, video_name, start_frame, w, h)
            if ctx_pil_0 is None:
                ctx_pil_0 = sample_random_frame_from_dataset(dataset_base, w, h, seed, metadata_path)
            if ctx_pil_0 is not None:
                ctx_pil_0 = [ctx_pil_0]
                pipe.load_models_to_device(["vae"])
                with torch.no_grad():
                    context_latents = encode_context_frames_per_frame(pipe, ctx_pil_0, pipe.device)
                context_actions_t = torch.tensor([identity_rt], dtype=torch.float32)
                context_frames_per_chunk.append([ctx_pil_0[0]])
            else:
                context_frames_per_chunk.append([])

        if context_latents is not None and context_actions_t is not None:
            frames_ch = run_one_chunk(
                pipe, prompt, use_negative_prompt, path_ch,
                context_latents=context_latents,
                num_context_frames=context_latents.shape[2],
                context_actions_t=context_actions_t,
                chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch,
                sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
                omit_context_actions=omit_context_actions,
            )
        else:
            frames_ch = run_one_chunk(
                pipe, prompt, use_negative_prompt, path_ch,
                chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch,
                sigma_shift=sigma_shift, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale,
                omit_context_actions=omit_context_actions,
            )
            if ch == 0 and not context_frames_per_chunk[0]:
                context_frames_per_chunk[0] = [_frame_to_pil(frames_ch[-1], w, h)]

        if use_sampling_files:
            yaw_chunk = _yaw_right if clockwise else _yaw_left
        else:
            yaw_chunk = list(yaw_chunk)
        for y in yaw_chunk:
            yaw_history.append(cumulative_yaw + y)
        cumulative_yaw = yaw_history[-1]
        chunk_frames_list.append(frames_ch)
        all_prev_frames.extend(frames_ch)

        # 续段 context：
        # - 默认：与 2chunk 一致，仅用上一 chunk 的 last n 帧
        # - multi_ctx_all_history=True：第一帧必须为上一 chunk 最后一帧；其余 (ctx-1) 帧从「前面所有帧」均匀采样，保持时间序。
        if ch < 3:
            if context_frames <= 0:
                prev_frames = [frames_ch[-1]]
            elif multi_ctx_all_history and len(all_prev_frames) > 0:
                total = len(all_prev_frames)
                if total == 1 or context_frames == 1:
                    prev_frames = [all_prev_frames[-1]]
                else:
                    # 第一帧 = 上一 chunk 最后一帧 (all_prev_frames[-1])；其余 (ctx-1) 帧从 all_prev_frames[0..total-2] 均匀采样
                    n_pick = min(context_frames - 1, total - 1)
                    if n_pick <= 0:
                        prev_frames = [all_prev_frames[-1]]
                    else:
                        pool_size = total - 1  # 可选下标 0..total-2（前面所有帧，不含已单独占位的最后一帧）
                        indices = []
                        for i in range(n_pick):
                            pos = int(round(i * (pool_size - 1) / max(n_pick - 1, 1)))
                            indices.append(max(0, min(pool_size - 1, pos)))
                        uniq_indices = sorted(set(indices))
                        while len(uniq_indices) < n_pick:
                            uniq_indices.append(pool_size - 1 if pool_size > 0 else 0)
                        uniq_indices = sorted(uniq_indices)[:n_pick]
                        chosen = [all_prev_frames[i] for i in uniq_indices]
                        prev_frames = [all_prev_frames[-1]] + chosen
            else:
                n_ctx = min(context_frames, len(frames_ch))
                prev_frames = context_frames_for_next_chunk(frames_ch, n_ctx) if n_ctx else [frames_ch[-1]]

            prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
            context_frames_per_chunk.append(list(prev_pil))
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                context_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
            num_ctx_tokens = context_latents.shape[2]
            context_actions_t = torch.tensor([identity_rt] * num_ctx_tokens, dtype=torch.float32)
            print(
                f"[Loop] continuation chunk ch={ch}: len(prev_pil)={len(prev_pil)} num_ctx_tokens={num_ctx_tokens} "
                f"(multi_ctx_all_history={multi_ctx_all_history})"
            )

        if not use_sampling_files and os.path.exists(path_ch):
            try:
                os.remove(path_ch)
            except Exception:
                pass

    return chunk_frames_list, yaw_history, context_frames_per_chunk


def run_replay_4chunk(
    pipe,
    dataset_base,
    output_dir,
    video_name,
    start_frame,
    chunk_frames=81,
    context_frames=1,
    h=352,
    w=640,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    inference_noise_level=0.0,
    omit_context_actions=True,
):
    """4 chunk 参考训练：每段用同视频连续 4 段的 GT 轨迹，逐 chunk 对比。需 start_frame 起至少 4*81 帧。"""
    print(f"[Replay 4chunk] {video_name} start{start_frame} (4 段 GT 轨迹) context_frames={context_frames} (续段 ctx 数)")
    json_file = os.path.join(dataset_base, "jsons", f"{video_name}.json")
    if not os.path.isfile(json_file):
        return None
    num_chunks = 4
    need_frames = num_chunks * chunk_frames  # 324
    # 检查是否有足够帧（用 pose 存在性粗略判断）
    try:
        last_pose = irc.load_pose_rt(json_file, start_frame + need_frames - 1)
        if last_pose is None or len(last_pose) < 12:
            return None
    except Exception:
        return None
    prompt = irc.load_prompt_for_video(dataset_base, video_name) or "A scene."
    use_negative_prompt = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")
    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
    os.makedirs(subdir, exist_ok=True)
    chunk_frames_list = []
    context_frames_per_chunk = []
    yaw_history = []
    cumulative_yaw = 0.0
    context_latents = None
    context_actions_t = None

    for ch in range(num_chunks):
        seg_start = start_frame + ch * chunk_frames
        ch1_actions = build_gt_trajectory_actions(dataset_base, video_name, seg_start, chunk_frames, json_file)
        if ch1_actions is None:
            return None
        path_ch = os.path.join(subdir, f"_ch{ch}_replay.json")
        with open(path_ch, "w") as f:
            json.dump(ch1_actions, f, indent=2)
        if ch == 0:
            ctx_pil_0 = load_sample_first_frame(dataset_base, video_name, seg_start, w, h)
            if ctx_pil_0 is None:
                return None
            ctx_pil_0 = [ctx_pil_0]
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                context_latents = encode_context_frames_per_frame(pipe, ctx_pil_0, pipe.device)
            context_actions_t = torch.tensor([identity_rt], dtype=torch.float32)
            context_frames_per_chunk.append([ctx_pil_0[0]])
        else:
            # 下一 chunk 的 condition = 上一 chunk 的 last n 帧（生成结果）
            n_ctx = min(context_frames, len(chunk_frames_list[-1]))
            prev_frames = context_frames_for_next_chunk(chunk_frames_list[-1], n_ctx) if n_ctx else [chunk_frames_list[-1][-1]]
            prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
            if ch < num_chunks:
                context_frames_per_chunk.append(list(prev_pil))
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                context_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
            context_actions_t = torch.tensor([identity_rt] * context_latents.shape[2], dtype=torch.float32)
            print(f"[Loop] continuation chunk ch={ch}: len(prev_pil)={len(prev_pil)} num_ctx_tokens={context_latents.shape[2]} (应等于 ctx)")

        frames_ch = run_one_chunk(
            pipe, prompt, use_negative_prompt, path_ch,
            context_latents=context_latents,
            num_context_frames=context_latents.shape[2],
            context_actions_t=context_actions_t,
            chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch * 100,
            sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
            omit_context_actions=omit_context_actions,
        )
        for i in range(chunk_frames):
            yaw_history.append(cumulative_yaw + _yaw_deg_from_rt(ch1_actions[str(i)]))
        cumulative_yaw = yaw_history[-1]
        chunk_frames_list.append(frames_ch)
        if os.path.exists(path_ch):
            try:
                os.remove(path_ch)
            except Exception:
                pass
    return chunk_frames_list, yaw_history, context_frames_per_chunk


def run_one_direction_4chunk(
    pipe,
    dataset_base,
    output_dir,
    video_name,
    start_frame,
    deg_per_chunk=15.0,
    clockwise=True,
    chunk_frames=81,
    context_frames=1,
    h=352,
    w=640,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    inference_noise_level=0.0,
    metadata_path=None,
    omit_context_actions=True,
):
    """4 chunk 单向匀速：共 4*deg_per_chunk 度。omit_context_actions 与训练对齐。"""
    print(f"[Onedir 4chunk] 匀速 总{4*deg_per_chunk:.0f}° {'CW' if clockwise else 'CCW'} sigma_shift={sigma_shift} omit_ctx_act={omit_context_actions} context_frames={context_frames} (续段 ctx 数)")
    prompt = irc.load_prompt_for_video(dataset_base, video_name) or "A scene."
    use_negative_prompt = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")  # 与训练一致

    yaw_history = []
    chunk_frames_list = []
    context_frames_per_chunk = []
    subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
    os.makedirs(subdir, exist_ok=True)
    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    cumulative_yaw_start = 0.0
    context_latents = None
    context_actions_t = None

    for ch in range(4):
        actions, yaw_chunk = build_action_chunk(deg_per_chunk, clockwise, chunk_frames)
        path_ch = os.path.join(subdir, f"_ch{ch}_onedir.json")
        with open(path_ch, "w") as f:
            json.dump(actions, f, indent=2)

        if ch == 0:
            ctx_pil_0 = load_sample_first_frame(dataset_base, video_name, start_frame, w, h)
            if ctx_pil_0 is None:
                ctx_pil_0 = sample_random_frame_from_dataset(dataset_base, w, h, seed, metadata_path)
            if ctx_pil_0 is not None:
                ctx_pil_0 = [ctx_pil_0]
                pipe.load_models_to_device(["vae"])
                with torch.no_grad():
                    context_latents = encode_context_frames_per_frame(pipe, ctx_pil_0, pipe.device)
                context_actions_t = torch.tensor([identity_rt], dtype=torch.float32)
                context_frames_per_chunk.append([ctx_pil_0[0]])
            else:
                context_frames_per_chunk.append([])

        if context_latents is not None and context_actions_t is not None:
            frames_ch = run_one_chunk(
                pipe, prompt, use_negative_prompt, path_ch,
                context_latents=context_latents,
                num_context_frames=context_latents.shape[2],
                context_actions_t=context_actions_t,
                chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch,
                sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
                omit_context_actions=omit_context_actions,
            )
        else:
            frames_ch = run_one_chunk(
                pipe, prompt, use_negative_prompt, path_ch,
                chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch,
                sigma_shift=sigma_shift, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale,
                omit_context_actions=omit_context_actions,
            )
            if ch == 0 and not context_frames_per_chunk[0]:
                context_frames_per_chunk[0] = [_frame_to_pil(frames_ch[-1], w, h)]

        for y in yaw_chunk:
            yaw_history.append(cumulative_yaw_start + y)
        cumulative_yaw_start = yaw_history[-1]
        chunk_frames_list.append(frames_ch)

        # 与 2chunk 一致：仅当有下一 chunk 时用本 chunk 输出做 context（last n 帧 → 逐帧 VAE）
        if ch < 3:
            n_ctx = min(context_frames, len(frames_ch))
            prev_frames = context_frames_for_next_chunk(frames_ch, n_ctx) if n_ctx else [frames_ch[-1]]
            prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
            context_frames_per_chunk.append(list(prev_pil))
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                context_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
            num_ctx_tokens = context_latents.shape[2]
            context_actions_t = torch.tensor([identity_rt] * num_ctx_tokens, dtype=torch.float32)
            print(f"[Loop] next continuation chunk: len(prev_pil)={len(prev_pil)} num_ctx_tokens={num_ctx_tokens} (应等于 ctx)")

        if os.path.exists(path_ch):
            try:
                os.remove(path_ch)
            except Exception:
                pass

    return chunk_frames_list, yaw_history, context_frames_per_chunk


def run_random_4chunk(
    pipe,
    dataset_base,
    output_dir,
    video_name,
    start_frame,
    deg_min=15.0,
    deg_max=30.0,
    chunk_frames=81,
    context_frames=1,
    h=352,
    w=640,
    sigma_shift=15.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    inference_noise_level=0.0,
    metadata_path=None,
    omit_context_actions=True,
):
    """4 chunk 随机。首 chunk context=随机 1 帧；之后 context=上一 chunk last n。"""
    prompt = irc.load_prompt_for_video(dataset_base, video_name) or "A scene."
    use_negative_prompt = getattr(irc, "DEFAULT_NEGATIVE_PROMPT", "oversaturated colors, overexposed, static, blurry details")  # 与训练一致

    rng = random.Random(seed)
    yaw_history = []
    chunk_frames_list = []
    context_frames_per_chunk = []
    subdir = os.path.join(output_dir, f"loop_{video_name}_start{start_frame}")
    os.makedirs(subdir, exist_ok=True)
    identity_rt = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    cumulative_yaw_start = 0.0
    context_latents = None
    context_actions_t = None

    for ch in range(4):
        deg = rng.uniform(deg_min, deg_max)
        clockwise = rng.choice([True, False])
        actions, yaw_chunk = build_action_chunk(deg, clockwise, chunk_frames)
        path_ch = os.path.join(subdir, f"_ch{ch}_rand.json")
        with open(path_ch, "w") as f:
            json.dump(actions, f, indent=2)

        if ch == 0:
            ctx_pil_0 = load_sample_first_frame(dataset_base, video_name, start_frame, w, h)
            if ctx_pil_0 is None:
                ctx_pil_0 = sample_random_frame_from_dataset(dataset_base, w, h, seed, metadata_path)
            if ctx_pil_0 is not None:
                ctx_pil_0 = [ctx_pil_0]
                pipe.load_models_to_device(["vae"])
                with torch.no_grad():
                    context_latents = encode_context_frames_per_frame(pipe, ctx_pil_0, pipe.device)
                context_actions_t = torch.tensor([identity_rt], dtype=torch.float32)
                context_frames_per_chunk.append([ctx_pil_0[0]])
            else:
                context_frames_per_chunk.append([])

        if context_latents is not None and context_actions_t is not None:
            frames_ch = run_one_chunk(
                pipe, prompt, use_negative_prompt, path_ch,
                context_latents=context_latents,
                num_context_frames=context_latents.shape[2],
                context_actions_t=context_actions_t,
                chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch * 7,
                sigma_shift=sigma_shift, num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale, inference_noise_level=inference_noise_level,
                omit_context_actions=omit_context_actions,
            )
        else:
            frames_ch = run_one_chunk(
                pipe, prompt, use_negative_prompt, path_ch,
                chunk_frames=chunk_frames, h=h, w=w, seed=seed + ch * 7,
                sigma_shift=sigma_shift, num_inference_steps=num_inference_steps, cfg_scale=cfg_scale,
                omit_context_actions=omit_context_actions,
            )
            if ch == 0 and not context_frames_per_chunk[0]:
                context_frames_per_chunk[0] = [_frame_to_pil(frames_ch[-1], w, h)]

        for y in yaw_chunk:
            yaw_history.append(cumulative_yaw_start + y)
        cumulative_yaw_start = yaw_history[-1]
        chunk_frames_list.append(frames_ch)

        # 与 2chunk 一致：仅当有下一 chunk 时用本 chunk 输出做 context（last n 帧 → 逐帧 VAE）
        if ch < 3:
            n_ctx = min(context_frames, len(frames_ch))
            prev_frames = context_frames_for_next_chunk(frames_ch, n_ctx) if n_ctx else [frames_ch[-1]]
            prev_pil = [_frame_to_pil(f, w, h) for f in prev_frames]
            context_frames_per_chunk.append(list(prev_pil))
            pipe.load_models_to_device(["vae"])
            with torch.no_grad():
                context_latents = encode_context_frames_per_frame(pipe, prev_pil, pipe.device)
            num_ctx_tokens = context_latents.shape[2]
            context_actions_t = torch.tensor([identity_rt] * num_ctx_tokens, dtype=torch.float32)
            print(f"[Loop] next continuation chunk: len(prev_pil)={len(prev_pil)} num_ctx_tokens={num_ctx_tokens} (应等于 ctx)")

        if os.path.exists(path_ch):
            try:
                os.remove(path_ch)
            except Exception:
                pass

    return chunk_frames_list, yaw_history, context_frames_per_chunk


def main():
    p = argparse.ArgumentParser(description="Loop: 两栏 [输入轨迹|生成视频]。首帧随机采样训练数据，之后每段 context=上一 chunk last n 帧，context 放右侧")
    p.add_argument("--ckpt", required=True, help="Step-15000.safetensors path")
    p.add_argument("--dataset_base", default=os.environ.get("DATASET_BASE_PATH", "data/Context-as-Memory-Dataset"))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk_frames", type=int, default=81)
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--sigma_shift", type=float, default=5.0, help="与训练 timestep_shift 一致：pre_qkv noise_5→5, noise_10→10, noise_3→3")
    p.add_argument("--num_inference_steps", type=int, default=50, help="与训练 sampling_num_inference_steps 一致（如 50）")
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--inference_noise_level", type=float, default=0.0, help="与训练 context_fixed_noise_std 一致（通常 0）")
    p.add_argument("--no_omit_context_actions", action="store_true", help="训练使用 context_actions 时加此参数；默认 omit（与 ctx=1 训练一致）")
    p.add_argument("--deg_per_chunk", type=float, default=45.0, help="回环每 chunk 转角（左转/右转均用此值，默认 45°）")
    p.add_argument("--skip_left_right_2chunk", action="store_true", help="跳过 1左1右 2chunk 回环")
    p.add_argument("--skip_left2_right2_4chunk", action="store_true", help="跳过 2左2右 4chunk 回环")
    p.add_argument("--only_single_left_45", action="store_true", help="仅跑旋转原子：单 chunk 左转 45°")
    p.add_argument("--only_single_right_45", action="store_true", help="仅跑旋转原子：单 chunk 右转 45°")
    p.add_argument("--sampling_action_dir", type=str, default=None, help="与采样一致的 action JSON 目录（含 action_rotation_left_45.json/right_45.json）；默认用本脚本所在目录")
    p.add_argument("--context_image", type=str, default=None, help="泛化检查：用指定图片（如 image.png）作为单帧 context 做旋转原子采样，与训练 action 形式一致")
    p.add_argument("--run_legacy_loop", action="store_true", help="跑旧版三组：roundtrip/onedir/replay，与下面 skip 配合")
    p.add_argument("--roundtrip_deg", type=float, default=30.0, help="[legacy] 2chunk 折返 顺/逆 角度")
    p.add_argument("--onedir_total_deg", type=float, default=60.0, help="[legacy] 4chunk 单向总角度")
    p.add_argument("--keep_action_jsons", action="store_true", help="保留临时 action json 便于检查")
    p.add_argument("--skip_roundtrip", action="store_true", help="[legacy] 跳过 2chunk 折返")
    p.add_argument("--skip_onedir", action="store_true", help="[legacy] 跳过 4chunk 单向 60°")
    p.add_argument("--skip_replay_4chunk", action="store_true", help="[legacy] 跳过 4chunk Replay")
    p.add_argument("--w_traj", type=int, default=320, help="轨迹图宽度")
    p.add_argument("--dataset_metadata_path", type=str, default=None, help="Optional CSV; also used for random first-frame sampling")
    p.add_argument("--context_frames", type=int, default=1, help="后续 chunk 的 context 用上一 chunk 的 last n 帧，n=context_frames；--run_ctx5_multi_chunk 时表示 K（K-1 均匀+最后 1 帧）")
    p.add_argument("--run_ctx5_multi_chunk", action="store_true", help="跑 ctx5 风格：chunk1 无 context，chunk2 用 K-1 均匀+最后 1 帧；多场景 2chunk 前进/后退/平移/左45右45")
    p.add_argument("--run_atomic_translation_4chunk", action="store_true", help="原子操作：仅跑纯平移 4chunk（向前/向后/向左/向右各 4chunk，镜头不旋转）")
    p.add_argument("--translation_delta", type=float, default=0.1, help="纯平移每 chunk 位移量（单轴），默认 0.1")
    # Memory baselines runtime flags (must match training for multichunk consistency)
    p.add_argument("--use_framepack_memory", action="store_true", help="Enable FAR/FramePack-style context reweighting during multichunk inference")
    p.add_argument("--context_temporal_decay", type=float, default=1.0, help="FramePack/FAR: per-frame decay for context")
    p.add_argument("--context_attention_weight", type=float, default=1.0, help="FramePack/FAR: global scale for all context tokens")
    p.add_argument("--use_framepack_length_compress", action="store_true", help="FramePack: compress context length K->K' during multichunk inference")
    p.add_argument("--framepack_ratio", type=int, default=2, help="FramePack temporal ratio r")
    p.add_argument("--use_spatial_memory", action="store_true", help="Enable spatial memory baseline during multichunk inference")
    p.add_argument("--use_spatial_memory_legacy", action="store_true", help="Non-learnable adaptive pool (if no SpatialGridMemory in ckpt, auto-enabled)")
    p.add_argument("--spatial_memory_tokens", type=int, default=64, help="Spatial memory baseline: number of pooled spatial tokens")
    p.add_argument(
        "--spatial_memory_inject_mode",
        type=str,
        default=None,
        choices=("concat_text", "cross_attn_readout", "none"),
        help="Spatial memory inject mode; must match training",
    )
    p.add_argument("--num_gpus", type=int, default=1, help="总 GPU 数，与 --rank 配合做多卡并行：每卡只推理 rank, rank+num_gpus, ... 的样本")
    p.add_argument("--rank", type=int, default=0, help="当前进程的 rank（0 到 num_gpus-1），由启动脚本设 CUDA_VISIBLE_DEVICES=rank 并传 --rank")
    p.add_argument(
        "--multi_ctx_4chunk_all_history",
        action="store_true",
        help="4chunk 回环：续段 ctx 时，除最后 1 帧固定为上一 chunk 的最后一帧，其余 (ctx-1) 帧从所有已生成 chunk 的历史帧中均匀抽取，用于多 chunk 记忆机制实验",
    )
    # Camera encoder 与注入节点：须与训练一致（见 INFERENCE_ALIGNMENT.md）
    p.add_argument("--camera_inject_mode", type=str, default=None, help="与训练一致：post|pre_norm|pre_qkv|pre_qkv_post；不设则从 ckpt 路径推断（如含 pre_qkv_post）")
    p.add_argument("--camera_encoder_shallow", action="store_true", default=None, help="与训练一致：单层 Linear(12,D)。不设则从 ckpt 自动推断")
    p.add_argument("--no_camera_encoder_shallow", action="store_true", help="明确不用 shallow（3 层 MLP）")
    p.add_argument("--camera_encoder_separate_t_r", action="store_true", default=None, help="与训练一致：t/R 分路编码。不设则从 ckpt 自动推断")
    p.add_argument("--no_camera_encoder_separate_t_r", action="store_true", help="明确不用 separate_t_r")
    p.add_argument("--no_camera_encoder_zero_init", action="store_true", help="与训练一致：encoder scale 非 0 初始化时加此参数")
    p.add_argument("--camera_encoder_explicit_yaw", action="store_true", help="与训练一致：显式 yaw 分支（dir_abl_A 等）")
    p.add_argument("--camera_encoder_sincos_yaw", action="store_true", help="与训练一致：sin/cos yaw 分支")
    p.add_argument("--add_camera_outside_gate", action="store_true", help="与训练一致：camera 加在 gate 外（abl 实验）")
    p.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Wan2.1 base model dir; default: $WAN_BASE_MODEL",
    )
    args = p.parse_args()
    # 原子平移评估用独立前缀，与回环 [Loop] 区分
    _log_prefix = "[atomic_translation_4chunk]" if getattr(args, "run_atomic_translation_4chunk", False) else "[Loop]"
    # 与训练 setting 对齐：ctx=1 时训练用 omit_context_actions=True
    args.omit_context_actions = not getattr(args, "no_omit_context_actions", False)
    # context_k* / ctx_*: infer context frames from released HF folders and legacy names.
    ckpt_lower = (args.ckpt or "").lower()
    if "ctx_20" in ckpt_lower or "context_k20" in ckpt_lower or "ctx20" in ckpt_lower:
        if getattr(args, "context_frames", 1) < 20:
            args.context_frames = 20
            print(f"{_log_prefix} 从 ckpt 路径识别 context_k20，将 context_frames 设为 20（续段 20 帧逐帧 VAE）", flush=True)
    elif "ctx_5" in ckpt_lower or "context_k5" in ckpt_lower or "ctx5" in ckpt_lower:
        if getattr(args, "context_frames", 1) < 5:
            args.context_frames = 5
            print(f"{_log_prefix} 从 ckpt 路径识别 context_k5，将 context_frames 设为 5（续段 5 帧）", flush=True)
    # 兼容：run_ctx5_multi_chunk 时若尚未被上面推断，也设为 5
    if getattr(args, "run_ctx5_multi_chunk", False) and getattr(args, "context_frames", 1) == 1:
        args.context_frames = 5
        print(f"{_log_prefix} run_ctx5_multi_chunk 已启用，将 context_frames 设为 5（续段 ctx=5）", flush=True)
    print(f"{_log_prefix} context_frames={args.context_frames}（续段用上一 chunk 的 last n 帧；首 chunk 始终 1 帧）", flush=True)

    # camera_inject_mode：未显式指定时从 ckpt 路径推断（与训练脚本命名一致）
    camera_inject_mode = (getattr(args, "camera_inject_mode", None) or "").strip()
    if not camera_inject_mode:
        env_cam = (os.environ.get("CAMERA_INJECT_MODE") or "").strip().lower()
        if env_cam in ("pre_qkv_post", "pre_qkv", "pre_norm", "post"):
            camera_inject_mode = env_cam
            print(f"{_log_prefix} 从环境 CAMERA_INJECT_MODE 使用 camera_inject_mode={camera_inject_mode}", flush=True)
    if not camera_inject_mode:
        ckpt_lower = (args.ckpt or "").lower()
        for mode in ("pre_qkv_post", "pre_qkv", "pre_norm", "post"):
            if mode.replace("_", "") in ckpt_lower or mode in ckpt_lower:
                camera_inject_mode = mode
                print(f"{_log_prefix} 从 ckpt 路径推断 camera_inject_mode={camera_inject_mode}", flush=True)
                break
        if not camera_inject_mode:
            # 与 memory_baselines_basic 默认训练脚本一致（旧默认 post 易与 pre_qkv 训练 ckpt 错位）
            camera_inject_mode = "pre_qkv"
            print(f"{_log_prefix} camera_inject_mode 未推断，使用默认 pre_qkv（与 train/memory_baselines_basic 对齐）", flush=True)
    # encoder 结构、SSM：均由 load_pipeline_and_ckpt 按 ckpt 自动推断并注入，无需额外参数
    load_kw = dict(
        action_inject_after_spatial_attn=True,
        add_action_attn=True,
        action_use_temporal_attention=True,
        camera_inject_mode=camera_inject_mode,
        add_camera_outside_gate=getattr(args, "add_camera_outside_gate", False),
        no_camera_encoder_zero_init=getattr(args, "no_camera_encoder_zero_init", False),
        camera_encoder_explicit_yaw=getattr(args, "camera_encoder_explicit_yaw", False),
        camera_encoder_sincos_yaw=getattr(args, "camera_encoder_sincos_yaw", False),
    )
    if getattr(args, "camera_encoder_shallow", False):
        load_kw["camera_encoder_shallow"] = True
    if getattr(args, "no_camera_encoder_shallow", False):
        load_kw["camera_encoder_shallow"] = False
    if getattr(args, "camera_encoder_separate_t_r", False):
        load_kw["camera_encoder_separate_t_r"] = True
    if getattr(args, "no_camera_encoder_separate_t_r", False):
        load_kw["camera_encoder_separate_t_r"] = False

    if args.num_gpus > 1:
        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        print(f"{_log_prefix} rank={args.rank} CUDA_VISIBLE_DEVICES={cuda_vis}", flush=True)

    base_model = (
        getattr(args, "base_model", None)
        or os.environ.get("WAN_BASE_MODEL")
    )
    if not base_model:
        raise ValueError("Set --base_model or WAN_BASE_MODEL to the Wan2.1 base model directory.")
    dit_path = os.path.join(base_model, "diffusion_pytorch_model.safetensors")
    text_path = os.path.join(base_model, "models_t5_umt5-xxl-enc-bf16.pth")
    vae_path = os.path.join(base_model, "Wan2.1_VAE.pth")
    for _p in (dit_path, text_path, vae_path):
        if not os.path.isfile(_p):
            raise FileNotFoundError(
                f"Missing Wan2.1 base weight: {_p} (set --base_model or WAN_BASE_MODEL to override)"
            )

    pipe = irc.load_pipeline_and_ckpt(
        args.ckpt,
        dit_path,
        text_path,
        vae_path,
        **load_kw,
    )
    # Runtime memory flags (must align with the checkpoint training flags)
    pipe.use_framepack_memory = bool(getattr(args, "use_framepack_memory", False))
    pipe.context_temporal_decay = float(getattr(args, "context_temporal_decay", 1.0) or 1.0)
    pipe.context_attention_weight = float(getattr(args, "context_attention_weight", 1.0) or 1.0)
    pipe.use_framepack_length_compress = bool(getattr(args, "use_framepack_length_compress", False))
    pipe.framepack_ratio = int(getattr(args, "framepack_ratio", 2) or 2)
    pipe.use_spatial_memory = bool(getattr(args, "use_spatial_memory", False))
    pipe.spatial_memory_tokens = int(getattr(args, "spatial_memory_tokens", 64) or 64)
    if getattr(args, "spatial_memory_inject_mode", None):
        pipe.spatial_memory_inject_mode = str(getattr(args, "spatial_memory_inject_mode"))
    pipe.use_spatial_memory_legacy = bool(getattr(args, "use_spatial_memory_legacy", False))
    # Ckpt may omit SpatialGridMemory weights: fall back to legacy adaptive pool for correct shapes
    if pipe.use_spatial_memory and not pipe.use_spatial_memory_legacy and getattr(pipe, "spatial_memory_module", None) is None:
        pipe.use_spatial_memory_legacy = True
        print(f"{_log_prefix} use_spatial_memory but no spatial_memory_module in ckpt -> use_spatial_memory_legacy=True", flush=True)
    if args.num_gpus > 1 and torch.cuda.is_available():
        print(f"{_log_prefix} rank={args.rank} device_count={torch.cuda.device_count()} current_device={torch.cuda.current_device()}", flush=True)

    if args.dataset_metadata_path and os.path.isfile(args.dataset_metadata_path):
        import csv
        candidates = []
        with open(args.dataset_metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vn = row.get("video_name", "").strip()
                sf = row.get("start_frame", "")
                if vn and str(sf).strip():
                    try:
                        candidates.append((vn, int(sf)))
                    except ValueError:
                        continue
        rng = random.Random(args.seed)
        if len(candidates) <= args.num_samples:
            samples = candidates
        else:
            samples = [candidates[i] for i in rng.sample(range(len(candidates)), args.num_samples)]
    else:
        samples = irc.sample_trajectory_samples_from_dataset(
            args.dataset_base,
            num_samples=args.num_samples,
            num_frames=args.chunk_frames,
            seed=args.seed,
        )
    if not samples:
        print(f"{_log_prefix} No samples from dataset.")
        return

    # 多卡：每卡只跑自己 rank 的样本（样本顺序一致，按 idx % num_gpus == rank 划分）
    if args.num_gpus > 1:
        samples = [s for i, s in enumerate(samples) if i % args.num_gpus == args.rank]
        print(f"{_log_prefix} num_gpus={args.num_gpus} rank={args.rank} -> 本进程跑 {len(samples)} 个样本")
        if not samples:
            print(f"{_log_prefix} 本 rank 无样本，退出")
            return

    os.makedirs(args.output_dir, exist_ok=True)
    h, w = args.height, args.width

    # 日志同时写入 output_dir 下的文件
    log_path = os.path.join(args.output_dir, f"loop_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    _log_file = open(log_path, "w", encoding="utf-8")
    _stdout_orig = sys.stdout

    class _Tee:
        def __init__(self, stream, f):
            self._stream = stream
            self._file = f
        def write(self, data):
            self._stream.write(data)
            self._stream.flush()
            if self._file and not self._file.closed:
                self._file.write(data)
                self._file.flush()
        def flush(self):
            self._stream.flush()
            if self._file and not self._file.closed:
                self._file.flush()

    sys.stdout = _Tee(_stdout_orig, _log_file)
    try:
        print(f"{_log_prefix} Log file: {log_path}")
        if getattr(args, "run_atomic_translation_4chunk", False):
            print(f"{_log_prefix} 模式: 仅原子操作纯平移 4chunk (forward/backward/left/right)，无旋转", flush=True)
        elif getattr(args, "run_ctx5_multi_chunk", False):
            print(f"{_log_prefix} 模式: ctx5 多 chunk 2chunk 多场景 (含旋转)", flush=True)
        else:
            print(f"{_log_prefix} 模式: 回环 1左1右 + 2左2右 旋转 (若只要纯平移请加 --run_atomic_translation_4chunk)", flush=True)

        for idx, (video_name, start_frame) in enumerate(samples):
            subdir = os.path.join(args.output_dir, f"loop_{video_name}_start{start_frame}")
            os.makedirs(subdir, exist_ok=True)
            # 同一样本 (video_name, start_frame) 使用确定性 seed，与 loop_traj_fov_gen / evals_ep0 等调用一致，避免偏移
            sample_seed = seed_for_sample(args.seed, video_name, start_frame)
            print(f"{_log_prefix} [{idx+1}/{len(samples)}] {video_name} start{start_frame} sigma_shift={args.sigma_shift} sample_seed={sample_seed}")

            def save_composite(name, chunk_frames_list, yaw_history, context_frames_per_chunk):
                # 与训练对齐：续段首帧=上一段末帧，拼接时去掉续段第 0 帧，避免重复/跳帧（chunk0 保留 81，chunk1+ 各 80）
                if len(chunk_frames_list) > 1:
                    trim_chunks, trim_yaw = trim_continuation_first_frame(chunk_frames_list, yaw_history, args.chunk_frames)
                else:
                    trim_chunks, trim_yaw = chunk_frames_list, yaw_history
                right_frames, yaw_ext = build_right_panel_and_yaw(
                    trim_chunks, trim_yaw, context_frames_per_chunk,
                    chunk_frames=args.chunk_frames, w=w, h=h,
                )
                traj_frames = draw_trajectory_frames(yaw_ext, args.w_traj, h, total_frames=len(right_frames))
                comp = composite_two_panel(traj_frames, right_frames, w_traj=args.w_traj, h_traj=h, w_gen=w, h_gen=h)
                save_video(comp, os.path.join(subdir, f"{name}_traj_gen.mp4"), fps=15, quality=5)
                gen_only = [f for ch in trim_chunks for f in ch]
                save_video(gen_only, os.path.join(subdir, f"{name}_gen_only.mp4"), fps=15, quality=5)

            meta = getattr(args, "dataset_metadata_path", None)
            try:
                # 原子操作：仅纯平移 4chunk（无旋转），与回环左/右转彻底互斥
                if getattr(args, "run_atomic_translation_4chunk", False):
                    delta = getattr(args, "translation_delta", 0.1)
                    for direction in ("forward", "backward", "left", "right"):
                        chunks, yaw, ctx_per_chunk = run_translation_4chunk(
                            pipe, args.dataset_base, args.output_dir, video_name, start_frame,
                            direction=direction, translation_delta=delta, chunk_frames=args.chunk_frames,
                            context_frames=args.context_frames, h=h, w=w,
                            sigma_shift=args.sigma_shift, num_inference_steps=args.num_inference_steps,
                            cfg_scale=args.cfg_scale, seed=sample_seed, inference_noise_level=args.inference_noise_level,
                            metadata_path=meta, omit_context_actions=args.omit_context_actions,
                        )
                        save_composite(f"{direction}_4chunk", chunks, yaw, ctx_per_chunk)
                # 旋转原子：仅单 chunk 左转 45° 或 右转 45°
                elif getattr(args, "only_single_left_45", False):
                    chunks, yaw, ctx_per_chunk = run_single_chunk_rotation(
                        pipe, args.dataset_base, args.output_dir, video_name, start_frame,
                        deg=args.deg_per_chunk, clockwise=False, chunk_frames=args.chunk_frames,
                        h=h, w=w, sigma_shift=args.sigma_shift, num_inference_steps=args.num_inference_steps,
                        cfg_scale=args.cfg_scale, seed=sample_seed, inference_noise_level=args.inference_noise_level,
                        metadata_path=meta, keep_action_jsons=getattr(args, "keep_action_jsons", False),
                        sampling_action_dir=getattr(args, "sampling_action_dir", None),
                        omit_context_actions=args.omit_context_actions,
                        context_image_path=getattr(args, "context_image", None),
                    )
                    save_composite("single_left_45", chunks, yaw, ctx_per_chunk)
                elif getattr(args, "only_single_right_45", False):
                    chunks, yaw, ctx_per_chunk = run_single_chunk_rotation(
                        pipe, args.dataset_base, args.output_dir, video_name, start_frame,
                        deg=args.deg_per_chunk, clockwise=True, chunk_frames=args.chunk_frames,
                        h=h, w=w, sigma_shift=args.sigma_shift, num_inference_steps=args.num_inference_steps,
                        cfg_scale=args.cfg_scale, seed=sample_seed, inference_noise_level=args.inference_noise_level,
                        metadata_path=meta, keep_action_jsons=getattr(args, "keep_action_jsons", False),
                        sampling_action_dir=getattr(args, "sampling_action_dir", None),
                        omit_context_actions=args.omit_context_actions,
                        context_image_path=getattr(args, "context_image", None),
                    )
                    save_composite("single_right_45", chunks, yaw, ctx_per_chunk)
                # 默认：回环两场景 1左1右(45°)、2左2右(每chunk 45°)。--run_legacy_loop 时跑旧三组
                elif not getattr(args, "run_legacy_loop", False):
                    if not getattr(args, "skip_left_right_2chunk", False):
                        chunks, yaw, ctx_per_chunk = run_left_right_2chunk(
                            pipe, args.dataset_base, args.output_dir, video_name, start_frame,
                            deg=args.deg_per_chunk, chunk_frames=args.chunk_frames, context_frames=args.context_frames,
                            h=h, w=w, sigma_shift=args.sigma_shift, num_inference_steps=args.num_inference_steps,
                            cfg_scale=args.cfg_scale, seed=sample_seed, inference_noise_level=args.inference_noise_level,
                            metadata_path=meta, keep_action_jsons=getattr(args, "keep_action_jsons", False),
                            sampling_action_dir=getattr(args, "sampling_action_dir", None),
                            omit_context_actions=args.omit_context_actions,
                        )
                        save_composite("left_right_2chunk", chunks, yaw, ctx_per_chunk)
                    if not getattr(args, "skip_left2_right2_4chunk", False):
                        chunks, yaw, ctx_per_chunk = run_left2_right2_4chunk(
                            pipe, args.dataset_base, args.output_dir, video_name, start_frame,
                            deg_per_chunk=args.deg_per_chunk, chunk_frames=args.chunk_frames, context_frames=args.context_frames,
                            h=h, w=w, sigma_shift=args.sigma_shift, num_inference_steps=args.num_inference_steps,
                            cfg_scale=args.cfg_scale, seed=sample_seed + 100, inference_noise_level=args.inference_noise_level,
                            metadata_path=meta, sampling_action_dir=getattr(args, "sampling_action_dir", None),
                            omit_context_actions=args.omit_context_actions,
                            multi_ctx_all_history=getattr(args, "multi_ctx_4chunk_all_history", False),
                        )
                        save_composite("left2_right2_4chunk", chunks, yaw, ctx_per_chunk)
                else:
                    if not args.skip_roundtrip:
                        chunks, yaw, ctx_per_chunk = run_roundtrip_2chunk(
                            pipe, args.dataset_base, args.output_dir, video_name, start_frame,
                            deg=args.roundtrip_deg, chunk_frames=args.chunk_frames, context_frames=args.context_frames,
                            h=h, w=w, sigma_shift=args.sigma_shift, num_inference_steps=args.num_inference_steps,
                            cfg_scale=args.cfg_scale, seed=sample_seed, inference_noise_level=args.inference_noise_level,
                            metadata_path=meta, keep_action_jsons=getattr(args, "keep_action_jsons", False),
                            omit_context_actions=args.omit_context_actions,
                        )
                        save_composite("roundtrip_2chunk", chunks, yaw, ctx_per_chunk)
                    if not args.skip_onedir:
                        deg_per = args.onedir_total_deg / 4.0
                        chunks, yaw, ctx_per_chunk = run_one_direction_4chunk(
                            pipe, args.dataset_base, args.output_dir, video_name, start_frame,
                            deg_per_chunk=deg_per, clockwise=True, chunk_frames=args.chunk_frames,
                            context_frames=args.context_frames, h=h, w=w,
                            sigma_shift=args.sigma_shift, num_inference_steps=args.num_inference_steps,
                            cfg_scale=args.cfg_scale, seed=sample_seed + 100, inference_noise_level=args.inference_noise_level,
                            metadata_path=meta, omit_context_actions=args.omit_context_actions,
                        )
                        save_composite("onedir_4chunk_60", chunks, yaw, ctx_per_chunk)
                    if not args.skip_replay_4chunk:
                        result = run_replay_4chunk(
                            pipe, args.dataset_base, args.output_dir, video_name, start_frame,
                            chunk_frames=args.chunk_frames, context_frames=args.context_frames,
                            h=h, w=w, sigma_shift=args.sigma_shift, num_inference_steps=args.num_inference_steps,
                            cfg_scale=args.cfg_scale, seed=sample_seed + 200, inference_noise_level=args.inference_noise_level,
                            omit_context_actions=args.omit_context_actions,
                        )
                        if result is not None:
                            chunks, yaw, ctx_per_chunk = result
                            save_composite("replay_4chunk", chunks, yaw, ctx_per_chunk)
                        else:
                            print(f"[Loop] Skip replay_4chunk for {video_name} start{start_frame} (need 4*81 frames)")

            except Exception as e:
                print(f"[Loop] ERROR {video_name} start{start_frame}: {e}")
                import traceback
                traceback.print_exc()

        if getattr(args, "only_single_left_45", False):
            mode = "旋转原子 单chunk 左转45°"
        elif getattr(args, "only_single_right_45", False):
            mode = "旋转原子 单chunk 右转45°"
        elif getattr(args, "run_legacy_loop", False):
            mode = "legacy"
        else:
            mode = "回环 1左1右 + 2左2右 (每chunk 45°)"
        print(f"{_log_prefix} Done. Mode: {mode}. Outputs under {args.output_dir}")
    finally:
        sys.stdout = _stdout_orig
        _log_file.close()


if __name__ == "__main__":
    main()
