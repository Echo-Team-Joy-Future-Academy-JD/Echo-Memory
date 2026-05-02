#!/usr/bin/env python3
"""
Visualize revisit consistency by exporting side-by-side images:
  [reference_frame | revisit_frame]

First version:
- reference = first frame, revisit = last frame.
"""
from __future__ import annotations

import argparse
import os
from typing import List

try:
    import cv2
except ImportError as e:
    raise RuntimeError("opencv-python required") from e

import numpy as np
from PIL import Image


def read_video_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def side_by_side(a: np.ndarray, b: np.ndarray) -> Image.Image:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    img = np.concatenate([a, b], axis=1)
    return Image.fromarray(img.astype(np.uint8))


def main():
    p = argparse.ArgumentParser(description="Export side-by-side revisit frames")
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True, help="output png path")
    args = p.parse_args()

    frames = read_video_frames(args.video)
    if len(frames) < 2:
        raise SystemExit("video has <2 frames")
    ref = frames[0]
    rev = frames[-1]
    out_img = side_by_side(ref, rev)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_img.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()

