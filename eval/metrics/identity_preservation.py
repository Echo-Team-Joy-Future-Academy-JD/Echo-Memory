"""
Identity Preservation metrics:
- CLIP consistency: frame-to-frame and vs first-frame cosine similarity of image embeddings.
  Uses simple resize-flatten-normalize embedding when CLIP is not available; optional CLIP when available.
- Face Embedding / Character ID retention: placeholder (optional insightface/torchreid).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from .common import discover_evals_videos, load_video_frames, load_video_frames_pil

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _simple_embedding(frames: np.ndarray, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Per-frame embedding: resize, flatten, normalize. Shape (N, D)."""
    if not HAS_CV2 or frames.size == 0:
        return np.zeros((0, 0))
    h, w = size
    out = []
    for i in range(frames.shape[0]):
        f = cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_LINEAR)
        v = f.astype(np.float32).flatten()
        n = np.linalg.norm(v)
        out.append(v / n if n > 0 else v)
    return np.stack(out, axis=0)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def clip_consistency_simple(frames: np.ndarray) -> dict[str, float]:
    """
    Consistency without CLIP: use simple embedding (resize+flatten+normalize), then
    - mean consecutive cosine similarity
    - min consecutive cosine similarity
    - mean similarity to first frame
    - min similarity to first frame
    """
    emb = _simple_embedding(frames)
    if emb.shape[0] < 2:
        return {"mean_consecutive_sim": 1.0, "min_consecutive_sim": 1.0, "mean_to_first_sim": 1.0, "min_to_first_sim": 1.0}
    first = emb[0]
    consec_sims = [_cosine_sim(emb[i], emb[i + 1]) for i in range(emb.shape[0] - 1)]
    to_first_sims = [_cosine_sim(emb[i], first) for i in range(1, emb.shape[0])]
    return {
        "mean_consecutive_sim": float(np.mean(consec_sims)),
        "min_consecutive_sim": float(np.min(consec_sims)),
        "mean_to_first_sim": float(np.mean(to_first_sims)),
        "min_to_first_sim": float(np.min(to_first_sims)),
        "embedding": "simple",
    }


def _try_clip_embeddings(pil_list, device="cuda"):
    """Optional: load CLIP and return (N, D) normalized image features. Returns None if unavailable."""
    try:
        import sys
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from diffsynth.extensions.ImageQualityMetric.clip import CLIPScore
        from diffsynth.extensions.ImageQualityMetric.config import MODEL_PATHS
        import torch
        model = CLIPScore(device=torch.device(device), path=MODEL_PATHS)
        model.model.eval()
        feats = []
        for pil in pil_list:
            x = model.preprocess_val(pil).unsqueeze(0).to(device=model.device)
            with torch.no_grad():
                f = model.model.encode_image(x, normalize=True)
            feats.append(f.cpu().numpy().squeeze(0))
        return np.stack(feats, axis=0)
    except Exception:
        return None


def clip_consistency_with_clip(pil_list, device: str = "cuda") -> dict[str, float] | None:
    """CLIP-based consistency. Returns None if CLIP not available."""
    emb = _try_clip_embeddings(pil_list, device)
    if emb is None or emb.shape[0] < 2:
        return None
    first = emb[0]
    consec_sims = [float(np.dot(emb[i], emb[i + 1])) for i in range(emb.shape[0] - 1)]
    to_first_sims = [float(np.dot(emb[i], first)) for i in range(1, emb.shape[0])]
    return {
        "mean_consecutive_sim": float(np.mean(consec_sims)),
        "min_consecutive_sim": float(np.min(consec_sims)),
        "mean_to_first_sim": float(np.mean(to_first_sims)),
        "min_to_first_sim": float(np.min(to_first_sims)),
        "embedding": "clip",
    }


def run_identity_preservation(
    evals_root: str,
    use_clip: bool = False,
    device: str = "cuda",
    video_paths: list[tuple[str, str]] | None = None,
    max_frames_per_video: int | None = 100,
) -> dict[str, Any]:
    """
    Compute identity preservation (CLIP consistency) over all gen_only videos.
    When use_clip=False uses simple embedding; when use_clip=True tries diffsynth CLIP.
    """
    if video_paths is None:
        video_paths = discover_evals_videos(evals_root)

    per_video = []
    agg_consec = []
    agg_to_first = []

    for rel, absp in video_paths:
        if not os.path.isfile(absp):
            continue
        if use_clip:
            pil_list = load_video_frames_pil(absp, max_frames=max_frames_per_video)
            if not pil_list:
                per_video.append({"rel": rel, "mean_consecutive_sim": None, "mean_to_first_sim": None, "embedding": None})
                continue
            res = clip_consistency_with_clip(pil_list, device)
            if res is None:
                frames = load_video_frames(absp, max_frames=max_frames_per_video)
                res = clip_consistency_simple(frames)
        else:
            frames = load_video_frames(absp, max_frames=max_frames_per_video)
            res = clip_consistency_simple(frames)

        agg_consec.append(res["mean_consecutive_sim"])
        agg_to_first.append(res["mean_to_first_sim"])
        per_video.append({"rel": rel, **res})

    aggregate = {}
    if agg_consec:
        aggregate["mean_consecutive_sim"] = float(np.mean(agg_consec))
        aggregate["min_mean_to_first_sim"] = float(np.min(agg_to_first))
        aggregate["mean_to_first_sim"] = float(np.mean(agg_to_first))
    aggregate["face_embedding_note"] = "Optional: install insightface/torchreid for Face Embedding / character ID retention."

    return {
        "dimension": "identity_preservation",
        "params": {"use_clip": use_clip, "device": device},
        "per_video": per_video,
        "aggregate": aggregate,
        "num_videos": len(per_video),
    }


def main():
    p = argparse.ArgumentParser(description="Identity Preservation (CLIP consistency)")
    p.add_argument("--evals_root", type=str, required=True)
    p.add_argument("--use_clip", action="store_true", help="Use CLIP image encoder when available")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_frames", type=int, default=100)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    result = run_identity_preservation(
        args.evals_root,
        use_clip=args.use_clip,
        device=args.device,
        max_frames_per_video=args.max_frames,
    )
    out = json.dumps(result, indent=2)
    print(out)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)


if __name__ == "__main__":
    main()
