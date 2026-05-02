#!/usr/bin/env python3
"""
按 visual_eval_config 中的 prompt 与首 chunk 预设，批量跑固定首帧 2chunk/4chunk，
输出按 prompt_id 与 first_chunk_id 分目录，便于肉眼对比查看。
用法见 VISUAL_EVAL_DESIGN.md。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_FIRST_FRAME = os.path.join(_EXP_DIR, "train", "ctx_5_20_per_frame_vae", "image.png")
_RUN_GEN_SCRIPT = os.path.join(_EXP_DIR, "run_generalization_fixed_first_frame.py")


def _load_config(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # minimal YAML-like parse for our config (no deps)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # fallback: expect prompts and first_chunk_presets as simple structure; here we only need ids and text/path
        import json
        # Try to find a JSON block or use a simple heuristic - actually better to require PyYAML
        raise SystemExit("pip install PyYAML 后重试，或使用 --prompts/--first_chunks 手动指定。")


def _export_dataset_frame(dataset_base: str, video_name: str, start_frame: int, out_path: str, w: int = 640, h: int = 352) -> bool:
    from PIL import Image
    for suf in (f"{start_frame:04d}.png", f"{start_frame}.png"):
        img_p = os.path.join(dataset_base, "frames", video_name.strip(), suf)
        if os.path.isfile(img_p):
            img = Image.open(img_p).convert("RGB")
            try:
                img = img.resize((w, h), Image.Resampling.LANCZOS)
            except AttributeError:
                img = img.resize((w, h), Image.LANCZOS)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img.save(out_path)
            return True
    return False


def main():
    p = argparse.ArgumentParser(description="按 config 跑多组 prompt × 首帧，便于可视化对比")
    p.add_argument("--ckpt", required=True, help="权重路径")
    p.add_argument("--config", default=None, help="YAML 配置，默认 eval_metrics/visual_eval_config.yaml")
    p.add_argument("--output_root", required=True, help="输出根目录，下建 prompt_<id>_first_<id>")
    p.add_argument("--dataset_base", default=None, help="数据集根目录，用于 dataset_frame 首帧导出")
    p.add_argument("--prompts", nargs="*", default=None, help="只跑这些 prompt id，默认全部")
    p.add_argument("--first_chunks", nargs="*", default=None, help="只跑这些 first_chunk id，默认全部")
    p.add_argument("--use_recommended", action="store_true", help="仅跑 config 里 recommended_pairs 列出的 (prompt_id, first_chunk_id)")
    p.add_argument("--action_dir", default=None, help="action_rotation_*.json 目录，默认 exp 目录")
    p.add_argument("--context_frames", type=int, default=1)
    p.add_argument("--no_camera_encoder_separate_t_r", action="store_true")
    args = p.parse_args()

    config_path = args.config or os.path.join(_SCRIPT_DIR, "visual_eval_config.yaml")
    if not os.path.isfile(config_path):
        print(f"Config 不存在: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = _load_config(config_path)

    prompts_cfg = config.get("prompts") or []
    first_chunk_cfg = config.get("first_chunk_presets") or []
    prompt_map = {x["id"]: x for x in prompts_cfg if x.get("id")}
    first_map = {x["id"]: x for x in first_chunk_cfg if x.get("id")}

    if args.use_recommended and config.get("recommended_pairs"):
        pairs = config["recommended_pairs"]
        prompt_ids = list({p[0] for p in pairs if len(p) >= 2})
        first_ids = list({p[1] for p in pairs if len(p) >= 2})
        run_pairs = [(p[0], p[1]) for p in pairs if len(p) >= 2 and p[0] in prompt_map and p[1] in first_map]
    else:
        prompt_ids = args.prompts or list(prompt_map.keys())
        first_ids = args.first_chunks or list(first_map.keys())
        run_pairs = None  # None = all combinations

    # Resolve first-frame image path for each first_chunk preset
    first_frames_dir = os.path.join(args.output_root, "first_frames")
    first_chunk_to_path = {}
    for fid in first_ids:
        fc = first_map.get(fid)
        if not fc:
            continue
        t = (fc.get("type") or "").strip()
        if t == "fixed_image":
            path = (fc.get("path") or "").strip()
            if not path:
                path = _DEFAULT_FIRST_FRAME
            if os.path.isfile(path):
                first_chunk_to_path[fid] = os.path.abspath(path)
            else:
                print(f"[skip] first_chunk {fid}: 图片不存在 {path}", file=sys.stderr)
        elif t == "dataset_frame":
            if not args.dataset_base:
                print(f"[skip] first_chunk {fid}: dataset_frame 需提供 --dataset_base", file=sys.stderr)
                continue
            vn = (fc.get("video_name") or "").strip()
            sf = int(fc.get("start_frame") or 0)
            out_path = os.path.join(first_frames_dir, f"{fid}.png")
            if _export_dataset_frame(args.dataset_base, vn, sf, out_path):
                first_chunk_to_path[fid] = out_path
            else:
                print(f"[skip] first_chunk {fid}: 无法导出帧 {vn} frame {sf}", file=sys.stderr)
        else:
            print(f"[skip] first_chunk {fid}: 未知 type {t}", file=sys.stderr)

    if not first_chunk_to_path:
        print("没有可用的首帧预设。", file=sys.stderr)
        sys.exit(1)

    action_dir = args.action_dir or _EXP_DIR
    if not os.path.isfile(_RUN_GEN_SCRIPT):
        print(f"未找到 {_RUN_GEN_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    for pid in prompt_ids:
        pr = prompt_map.get(pid)
        if not pr:
            print(f"[skip] prompt {pid} 不在 config 中", file=sys.stderr)
            continue
        text = (pr.get("text") or "A scene.").strip()
        for fid, first_path in first_chunk_to_path.items():
            if run_pairs is not None and (pid, fid) not in run_pairs:
                continue
            out_dir = os.path.join(args.output_root, f"prompt_{pid}_first_{fid}")
            os.makedirs(out_dir, exist_ok=True)
            cmd = [
                sys.executable,
                _RUN_GEN_SCRIPT,
                "--ckpt", args.ckpt,
                "--first_frame_image", first_path,
                "--prompt", text,
                "--output_dir", out_dir,
                "--sampling_action_dir", action_dir,
                "--context_frames", str(args.context_frames),
            ]
            if args.no_camera_encoder_separate_t_r:
                cmd.append("--no_camera_encoder_separate_t_r")
            print(f"[run] prompt={pid} first={fid} -> {out_dir}", file=sys.stderr)
            ret = subprocess.run(cmd, cwd=_EXP_DIR)
            if ret.returncode != 0:
                print(f"[warn] 退出码 {ret.returncode} prompt={pid} first={fid}", file=sys.stderr)

    print(f"Done. 输出根目录: {args.output_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
