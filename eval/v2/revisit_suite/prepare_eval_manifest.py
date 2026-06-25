#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _step_id(path: Path) -> int:
    stem = path.stem
    if stem.startswith("Step-"):
        try:
            return int(stem.split("-", 1)[1])
        except Exception:
            return -1
    return -1


def find_ckpts(outputs_root: Path, include_dyn: bool, ckpt_policy: str = "epoch0") -> list[Path]:
    """Return checkpoints under outputs_root according to ckpt_policy.

    Evaluation candidacy is intentionally file-based: if a run directory has
    a matching checkpoint, it is considered finished enough to evaluate. The
    include_dyn flag is kept for backward compatibility and does not filter
    Echo-Memory outputs by default.
    """
    policy = (ckpt_policy or "epoch0").strip().lower()
    if policy == "epoch0":
        return sorted(outputs_root.rglob("epoch-0.safetensors"))
    if policy == "all_steps":
        return sorted(outputs_root.rglob("Step-*.safetensors"), key=lambda p: (str(p.parent), _step_id(p)))
    if policy == "latest":
        by_dir: dict[Path, Path] = {}
        for p in sorted(outputs_root.rglob("*.safetensors")):
            if p.name == "epoch-0.safetensors":
                by_dir.setdefault(p.parent, p)
            elif p.name.startswith("Step-"):
                prev = by_dir.get(p.parent)
                if prev is None or (prev.name == "epoch-0.safetensors") or _step_id(p) > _step_id(prev):
                    by_dir[p.parent] = p
        return sorted(by_dir.values())
    raise ValueError(f"unknown ckpt_policy={ckpt_policy}; expected epoch0/latest/all_steps")


def read_train_samples(metadata: Path, dataset_base: Path, limit: int) -> list[dict]:
    rows = []
    with metadata.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            video_name = (row.get("video_name") or "").strip()
            if not video_name:
                continue
            try:
                start_frame = int(row.get("start_frame", 0) or 0)
            except (TypeError, ValueError):
                start_frame = 0
            try:
                end_frame = int(row.get("end_frame", start_frame + 80) or (start_frame + 80))
            except (TypeError, ValueError):
                end_frame = start_frame + 80
            frame_path = dataset_base / "frames" / video_name / f"{start_frame:04d}.png"
            if not frame_path.is_file():
                continue
            rows.append(
                {
                    "sample_id": f"train_{len(rows):05d}_{video_name.replace('/', '__')}_{start_frame:06d}",
                    "domain": "train",
                    "video_name": video_name,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "prompt": row.get("prompt") or "A scene.",
                    "first_frame_image": str(frame_path),
                    "dataset_base": str(dataset_base),
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def read_ood_samples(ood_dir: Path, prompt: str, limit: int) -> list[dict]:
    imgs = [p for p in sorted(ood_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if limit:
        imgs = imgs[:limit]
    return [
        {
            "sample_id": f"ood_{i:05d}_{p.stem}",
            "domain": "ood",
            "video_name": None,
            "start_frame": 0,
            "end_frame": 80,
            "prompt": prompt,
            "first_frame_image": str(p),
            "dataset_base": None,
        }
        for i, p in enumerate(imgs)
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    default_dataset = repo_root / "data" / "Context-as-Memory-Dataset"
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default=str(repo_root / "outputs"))
    ap.add_argument("--dataset-base", default=str(default_dataset))
    ap.add_argument("--metadata", default=str(default_dataset / "metadata_full.csv"))
    ap.add_argument("--ood-dir", default=str(repo_root / "assets" / "opendomain_revisit"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-limit", type=int, default=8)
    ap.add_argument("--ood-limit", type=int, default=8)
    ap.add_argument("--ood-prompt", default="A toy bear in the same static scene. Preserve the bear appearance and the scene layout after camera revisit.")
    ap.add_argument("--include-dynmembench", action="store_true")
    ap.add_argument("--ckpt-policy", default="epoch0", choices=["epoch0", "latest", "all_steps"])
    args = ap.parse_args()

    ckpts = find_ckpts(Path(args.outputs_root), args.include_dynmembench, args.ckpt_policy)
    train_samples = read_train_samples(Path(args.metadata), Path(args.dataset_base), args.train_limit)
    ood_samples = read_ood_samples(Path(args.ood_dir), args.ood_prompt, args.ood_limit)
    payload = {
        "ckpts": [{"ckpt": str(p), "run_id": p.parent.name} for p in ckpts],
        "samples": train_samples + ood_samples,
        "dataset_base": args.dataset_base,
        "metadata": args.metadata,
        "ood_dir": args.ood_dir,
        "ckpt_policy": args.ckpt_policy,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare_eval_manifest] ckpts={len(ckpts)} train={len(train_samples)} ood={len(ood_samples)} -> {out}")


if __name__ == "__main__":
    main()
