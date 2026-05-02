#!/usr/bin/env python3
"""Offline multiview point-cloud rendering/ghosting diagnostics for cross-chunk geometry."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _read_video_rgb(video_path: str, max_frames: int = 0) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    out: List[np.ndarray] = []
    if not cap.isOpened():
        return out
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames > 0 and len(out) >= max_frames:
            break
    cap.release()
    return out


def _read_pose_json(json_path: str) -> Dict[str, Dict]:
    if not os.path.isfile(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "CineCameraActor" in data and isinstance(data["CineCameraActor"], dict):
        return data["CineCameraActor"]
    return data if isinstance(data, dict) else {}


def _pose_to_rt(pose: Dict) -> Tuple[np.ndarray, np.ndarray]:
    # Minimal compatible conversion: translation + yaw-only Z rotation.
    pos = pose.get("position", [0, 0, 0])
    rot = pose.get("rotation", [0, 0, 0])
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0
    yaw = float(rot[2]) if len(rot) > 2 else 0.0
    rad = np.deg2rad(yaw)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    t = np.array([x, y, z], dtype=np.float64).reshape(3, 1)
    return R, t


def _pseudo_depth_from_rgb(rgb: np.ndarray) -> np.ndarray:
    # Placeholder depth: inverse luminance in [0.3, 3.0]
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    d = 0.3 + (1.0 - g) * 2.7
    return d


def _resize_depth(d: np.ndarray, w: int, h: int) -> np.ndarray:
    if d.ndim > 2:
        d = np.squeeze(d)
    if d.shape[:2] == (h, w):
        return d.astype(np.float32)
    return cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def _load_depth_npy(
    depth_dir: str,
    abs_idx: int,
    frame_i: int,
    w: int,
    h: int,
    depth_key: str,
    depth_is_inverse: bool,
    depth_scale: float,
) -> Optional[np.ndarray]:
    for name in (
        f"{abs_idx:04d}.npy",
        f"{abs_idx}.npy",
        f"{frame_i:04d}.npy",
        f"{frame_i}.npy",
    ):
        p = os.path.join(depth_dir, name)
        if os.path.isfile(p):
            d = np.load(p)
            d = _resize_depth(d, w, h)
            if depth_is_inverse:
                d = 1.0 / (np.clip(d, 1e-6, None))
            return np.clip(d * float(depth_scale), 1e-3, 1e4)
    return None


def _load_depth_npz(
    depth_dir: str,
    abs_idx: int,
    frame_i: int,
    w: int,
    h: int,
    depth_key: str,
    depth_is_inverse: bool,
    depth_scale: float,
) -> Optional[np.ndarray]:
    for name in (f"{abs_idx:04d}.npz", f"{abs_idx}.npz", f"{frame_i:04d}.npz", f"{frame_i}.npz"):
        p = os.path.join(depth_dir, name)
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        keys = list(z.files)
        if depth_key in keys:
            d = z[depth_key]
        elif keys:
            d = z[keys[0]]
        else:
            continue
        d = _resize_depth(np.asarray(d), w, h)
        if depth_is_inverse:
            d = 1.0 / (np.clip(d, 1e-6, None))
        return np.clip(d * float(depth_scale), 1e-3, 1e4)
    return None


def _depth_midas(rgb: np.ndarray, device: str) -> Optional[np.ndarray]:
    try:
        import torch
        from PIL import Image
    except Exception:
        return None
    try:
        hub = "intel-isl/MiDaS"
        midas = torch.hub.load(hub, "MiDaS_small", trust_repo=True)
        transforms = torch.hub.load(hub, "transforms", trust_repo=True)
        midas.to(device).eval()
        t = transforms.small_transform
        h0, w0 = rgb.shape[:2]
        batch = t(Image.fromarray(rgb)).to(device)
        with torch.no_grad():
            pred = midas(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(h0, w0),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        out = pred.cpu().numpy().astype(np.float32)
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        return np.clip(0.3 + out * 2.7, 1e-3, 10.0)
    except Exception:
        return None


def resolve_depth(
    depth_mode: str,
    rgb: np.ndarray,
    abs_idx: int,
    frame_i: int,
    depth_dir: Optional[str],
    depth_key: str,
    depth_is_inverse: bool,
    depth_scale: float,
    midas_device: str,
) -> Tuple[np.ndarray, str]:
    """Return depth HxW float32 and a note (empty if ok)."""
    h, w = rgb.shape[:2]
    if depth_mode == "pseudo":
        return _pseudo_depth_from_rgb(rgb), ""

    if depth_mode == "npy_dir":
        if not depth_dir:
            return _pseudo_depth_from_rgb(rgb), "npy_dir: missing --depth_dir; using pseudo"
        d = _load_depth_npy(depth_dir, abs_idx, frame_i, w, h, depth_key, depth_is_inverse, depth_scale)
        if d is None:
            return _pseudo_depth_from_rgb(rgb), f"npy_dir: no file for frame abs_idx={abs_idx}; pseudo"
        return d, ""

    if depth_mode == "npz_dir":
        if not depth_dir:
            return _pseudo_depth_from_rgb(rgb), "npz_dir: missing --depth_dir; using pseudo"
        d = _load_depth_npz(depth_dir, abs_idx, frame_i, w, h, depth_key, depth_is_inverse, depth_scale)
        if d is None:
            return _pseudo_depth_from_rgb(rgb), f"npz_dir: no file for abs_idx={abs_idx}; pseudo"
        return d, ""

    if depth_mode == "midas":
        d = _depth_midas(rgb, midas_device)
        if d is None:
            return _pseudo_depth_from_rgb(rgb), "midas: failed (hub/offline/torch); pseudo"
        return d, ""

    return _pseudo_depth_from_rgb(rgb), f"unknown depth_mode={depth_mode}; pseudo"


def _backproject(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, stride: int = 4):
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[0:h:stride, 0:w:stride].reshape(-1, 1)
    pix = np.stack([xs.reshape(-1), ys.reshape(-1), np.ones(xs.size)], axis=1).astype(np.float64)
    Kinv = np.linalg.inv(K)
    cam = (Kinv @ pix.T).T * z
    # world = R^-1 (cam - t)
    world = (R.T @ (cam.T - t)).T
    cols = rgb[0:h:stride, 0:w:stride].reshape(-1, 3).astype(np.uint8)
    return world, cols


def _write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _project(world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    cam = (R @ world.T) + t
    z = cam[2:3, :]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uv = (K @ cam)[:2, :] / z
    return np.vstack([uv, z])


def _render_simple(world: np.ndarray, rgb: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, w: int, h: int) -> np.ndarray:
    proj = _project(world, K, R, t)
    u = np.round(proj[0]).astype(np.int32)
    v = np.round(proj[1]).astype(np.int32)
    z = proj[2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    zbuf = np.full((h, w), np.inf, dtype=np.float64)
    for i in range(world.shape[0]):
        x, y = u[i], v[i]
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if z[i] > 0 and z[i] < zbuf[y, x]:
            zbuf[y, x] = z[i]
            img[y, x] = rgb[i]
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_video", required=True)
    ap.add_argument("--dataset_base", required=True)
    ap.add_argument("--video_name", required=True)
    ap.add_argument("--start_frame", type=int, required=True)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--sample_stride", type=int, default=4)
    ap.add_argument("--intrinsics_fx", type=float, default=500.0)
    ap.add_argument("--intrinsics_fy", type=float, default=500.0)
    ap.add_argument("--intrinsics_cx", type=float, default=320.0)
    ap.add_argument("--intrinsics_cy", type=float, default=176.0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--depth_mode",
        type=str,
        default="pseudo",
        choices=("pseudo", "npy_dir", "npz_dir", "midas"),
        help="pseudo=luminance; npy_dir/npz_dir=per-frame files under --depth_dir; midas=torch.hub MiDaS_small",
    )
    ap.add_argument("--depth_dir", type=str, default=None, help="Directory of per-frame .npy or .npz depths")
    ap.add_argument("--depth_key", type=str, default="depth", help="npz array key (default depth)")
    ap.add_argument("--depth_is_inverse", action="store_true", help="Treat loaded values as disparity; convert to 1/z")
    ap.add_argument("--depth_scale", type=float, default=1.0, help="Multiply depth after load")
    ap.add_argument("--midas_device", type=str, default="cuda", help="cuda or cpu for depth_mode=midas")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    frames = _read_video_rgb(os.path.abspath(args.gen_video), max_frames=args.num_frames)
    if not frames:
        raise RuntimeError("cannot read generated video frames")

    h, w = frames[0].shape[:2]
    K = np.array(
        [[args.intrinsics_fx, 0.0, args.intrinsics_cx], [0.0, args.intrinsics_fy, args.intrinsics_cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    pose_json = os.path.join(os.path.abspath(args.dataset_base), "jsons", f"{args.video_name}.json")
    poses = _read_pose_json(pose_json)

    depth_dir = os.path.abspath(args.depth_dir) if args.depth_dir else None
    depth_notes: List[str] = []
    xyzs: List[np.ndarray] = []
    rgbs: List[np.ndarray] = []
    frame_stats: List[Dict[str, float]] = []
    for i, fr in enumerate(frames):
        abs_idx = int(args.start_frame + i)
        pose = poses.get(str(abs_idx))
        if not isinstance(pose, dict):
            continue
        R, t = _pose_to_rt(pose)
        depth, note = resolve_depth(
            args.depth_mode,
            fr,
            abs_idx,
            i,
            depth_dir,
            args.depth_key,
            bool(args.depth_is_inverse),
            float(args.depth_scale),
            args.midas_device if args.depth_mode == "midas" else "cpu",
        )
        if note:
            depth_notes.append(note)
        xyz, cols = _backproject(depth, fr, K, R, t, stride=max(1, args.sample_stride))
        xyzs.append(xyz)
        rgbs.append(cols)
        frame_stats.append({"frame": abs_idx, "points": float(xyz.shape[0])})

    if not xyzs:
        raise RuntimeError("no valid points from frames/poses")
    xyz_all = np.concatenate(xyzs, axis=0)
    rgb_all = np.concatenate(rgbs, axis=0)
    ply_path = os.path.join(out_dir, "global_pointcloud.ply")
    _write_ply(ply_path, xyz_all, rgb_all)

    # render at first and last available GT poses
    rendered = []
    for key_name, abs_idx in [("view_first", args.start_frame), ("view_last", args.start_frame + len(frames) - 1)]:
        pose = poses.get(str(abs_idx))
        if not isinstance(pose, dict):
            continue
        R, t = _pose_to_rt(pose)
        img = _render_simple(xyz_all, rgb_all, K, R, t, w, h)
        outp = os.path.join(out_dir, f"{key_name}.png")
        cv2.imwrite(outp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rendered.append(outp)

    # ghosting proxy: per-pixel color variance in rendered-first neighborhood
    ghost_proxy = float(np.var(rgb_all.astype(np.float32), axis=0).mean())
    base_notes = [
        f"depth_mode={args.depth_mode}",
        "depth_dir: set GEOMETRY_DEPTH_DIR + npy_dir/npz_dir for external metric depth.",
        "inverse depth: use --depth_is_inverse if files are disparity.",
    ]
    if args.depth_mode == "pseudo":
        base_notes.insert(0, "Depth is pseudo from luminance (weak geometry).")
    uniq_notes = sorted(set(depth_notes))
    diag = {
        "gen_video": os.path.abspath(args.gen_video),
        "video_name": args.video_name,
        "start_frame": args.start_frame,
        "depth_mode": args.depth_mode,
        "depth_dir": depth_dir,
        "depth_key": args.depth_key,
        "depth_is_inverse": bool(args.depth_is_inverse),
        "depth_scale": float(args.depth_scale),
        "num_frames_used": len(frames),
        "num_points": int(xyz_all.shape[0]),
        "ghosting_proxy_color_var": ghost_proxy,
        "pointcloud_ply": ply_path,
        "rendered_views": rendered,
        "notes": base_notes + uniq_notes,
        "per_frame": frame_stats,
    }
    with open(os.path.join(out_dir, "geometry_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(f"[render_multiview_pointcloud_offline] points={xyz_all.shape[0]} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
