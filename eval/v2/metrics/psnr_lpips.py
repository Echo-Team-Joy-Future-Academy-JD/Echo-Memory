#!/usr/bin/env python3
"""
PSNR + LPIPS for aligned frame pairs.

First version focuses on static consistency revisit:
- Compare first frame vs last frame of a video (revisit proxy).
- Optionally compare symmetric frames if indices provided.

Dependencies:
- PSNR: numpy
- LPIPS: optional (pip install lpips). If missing, lpips will be reported as None.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _read_video_frames(path: str) -> List[np.ndarray]:
    if not HAS_CV2:
        raise RuntimeError("opencv-python required to read mp4")
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    return frames


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape and HAS_CV2:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 0:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _lpips_model(device: str = "cuda"):
    try:
        import torch
        import lpips  # type: ignore
        m = lpips.LPIPS(net="alex").to(device)
        m.eval()
        return m
    except Exception:
        return None


def lpips_distance(img1: np.ndarray, img2: np.ndarray, model, device: str = "cuda") -> float | None:
    if model is None:
        return None
    try:
        import torch
        if img1.shape != img2.shape and HAS_CV2:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
        # [H,W,3] uint8 -> [1,3,H,W] float in [-1,1]
        t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t1 = t1.to(device)
        t2 = t2.to(device)
        with torch.no_grad():
            d = model(t1, t2)
        return float(d.item())
    except Exception:
        return None


def compute_revisit_metrics(video_path: str, device: str = "cuda") -> Dict[str, Any]:
    frames = _read_video_frames(video_path)
    if len(frames) < 2:
        return {"num_frames": len(frames), "psnr": None, "lpips": None}
    first = frames[0]
    last = frames[-1]
    p = psnr(first, last)
    m = _lpips_model(device=device)
    l = lpips_distance(first, last, m, device=device)
    return {"num_frames": len(frames), "psnr": p, "lpips": l}


def main():
    p = argparse.ArgumentParser(description="PSNR+LPIPS for revisit (first vs last frame)")
    p.add_argument("--video", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    res = compute_revisit_metrics(args.video, device=args.device)
    out = json.dumps(res, indent=2)
    print(out)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)


if __name__ == "__main__":
    main()

