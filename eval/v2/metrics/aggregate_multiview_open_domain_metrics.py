#!/usr/bin/env python3
"""Aggregate per-view revisit closure + minimal cross-view 2D metrics (open-domain multiview)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_metrics_dir = os.path.dirname(os.path.abspath(__file__))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)
import psnr_lpips as _pl  # noqa: E402

try:
    from skimage.metrics import structural_similarity as _skimage_ssim

    _HAS_SKIMAGE = True
except Exception:
    _skimage_ssim = None  # type: ignore
    _HAS_SKIMAGE = False


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_last_frame_rgb(video_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d ** 2))


def _psnr(mse_v: float) -> float:
    if mse_v <= 0:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse_v))


def _ssim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not _HAS_SKIMAGE or _skimage_ssim is None:
        return None
    try:
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        try:
            return float(_skimage_ssim(a, b, channel_axis=2, data_range=255))
        except TypeError:
            return float(_skimage_ssim(a, b, multichannel=True, data_range=255))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate multiview closure + cross-view vs ref first frame")
    ap.add_argument(
        "--multiview_root",
        required=True,
        help="e.g. .../static_consistency/open_domain/multiview_revisit",
    )
    ap.add_argument("--ref_view", type=str, default="0", help="view_id used as reference first frame")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--device", default="cuda", help="LPIPS device")
    args = ap.parse_args()

    root = os.path.abspath(args.multiview_root)
    summary_path = os.path.join(root, "multiview_revisit_summary.json")
    if not os.path.isfile(summary_path):
        out = {
            "error": f"missing {summary_path}",
            "multiview_root": root,
            "per_view_closure": [],
            "cross_view_vs_ref": [],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return 0

    summary_data = _read_json(summary_path)
    rows: List[Dict[str, Any]] = list(summary_data.get("summary") or [])

    ref_row: Optional[Dict[str, Any]] = None
    for r in rows:
        if str(r.get("view_id")) == str(args.ref_view):
            ref_row = r
            break
    ref_first_rgb: Optional[np.ndarray] = None
    if ref_row and ref_row.get("first_frame_image"):
        p = str(ref_row["first_frame_image"])
        if os.path.isfile(p):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is not None:
                ref_first_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    per_view_closure: List[Dict[str, Any]] = []
    cross_view: List[Dict[str, Any]] = []

    lpips_model = _pl._lpips_model(device=args.device)

    for r in rows:
        vid = str(r.get("view_id", ""))
        out_dir = str(r.get("output_dir") or "")
        if not out_dir or not os.path.isdir(out_dir):
            continue
        closure_p = os.path.join(out_dir, "revisit_closure_metrics.json")
        entry: Dict[str, Any] = {"view_id": vid, "output_dir": out_dir}
        if os.path.isfile(closure_p):
            try:
                c = _read_json(closure_p)
                entry["closure_first_vs_last_mse"] = c.get("closure_first_vs_last_mse")
                entry["closure_first_vs_last_psnr"] = c.get("closure_first_vs_last_psnr")
                entry["num_chunks"] = c.get("num_chunks")
            except Exception as e:
                entry["closure_error"] = str(e)
        else:
            entry["closure_error"] = "missing revisit_closure_metrics.json"
        per_view_closure.append(entry)

        if ref_first_rgb is None or str(vid) == str(args.ref_view):
            continue
        mp4 = os.path.join(out_dir, "combo_revisit_4chunk_gen_only.mp4")
        if not os.path.isfile(mp4):
            cross_view.append({"view_id": vid, "error": "missing combo_revisit_4chunk_gen_only.mp4"})
            continue
        last_rgb = _read_last_frame_rgb(mp4)
        if last_rgb is None:
            cross_view.append({"view_id": vid, "error": "cannot read last frame"})
            continue
        if last_rgb.shape[:2] != ref_first_rgb.shape[:2]:
            ref_r = cv2.resize(ref_first_rgb, (last_rgb.shape[1], last_rgb.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            ref_r = ref_first_rgb
        mse_v = _mse(last_rgb, ref_r)
        row = {
            "view_id": vid,
            "ref_view": str(args.ref_view),
            "last_vs_ref_first_mse": mse_v,
            "last_vs_ref_first_psnr": _psnr(mse_v),
            "last_vs_ref_first_ssim": _ssim(last_rgb, ref_r),
            "last_vs_ref_first_lpips": _pl.lpips_distance(last_rgb, ref_r, lpips_model, device=args.device),
        }
        cross_view.append(row)

    metric_definitions = {
        "per_view_closure": "From run_combo_revisit_fixed_first revisit_closure_metrics.json (same-view first vs last).",
        "cross_view_vs_ref": (
            "Heuristic: last frame of view k generated video vs reference view input first frame. "
            "Not multi-view geometry; use when opendomain images depict the same object/scene."
        ),
    }
    out = {
        "multiview_root": root,
        "ref_view": str(args.ref_view),
        "metric_definitions": metric_definitions,
        "per_view_closure": per_view_closure,
        "cross_view_vs_ref": cross_view,
        "notes": [
            "Key-object ROI not applied (full frame).",
            "3D consistency requires depth+pose; open_domain uses 2D proxies only.",
        ],
    }
    outp = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate_multiview_open_domain_metrics] views={len(per_view_closure)} -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
