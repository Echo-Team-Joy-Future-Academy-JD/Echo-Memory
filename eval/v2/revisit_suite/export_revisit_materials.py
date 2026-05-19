#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


SYSTEM = "You are a strict visual memory evaluator. Return only valid JSON."

USER_PROMPT = """Evaluate whether the revisit tail frames preserve the same object and scene as the first frame.

The most important target is the toy bear if it appears in the first frame. Judge its appearance consistency strongly:
- bear identity, color, shape, face, pose, clothing/accessories if visible
- whether the final revisit still depicts the same bear after camera motion

Scene/layout consistency is secondary. Give lower weight to background unless it contradicts the target.

Return JSON with:
{
  "bear_appearance_score": 0-5,
  "bear_presence_score": 0-5,
  "scene_consistency_score": 0-5,
  "view_revisit_score": 0-5,
  "overall_score": 0-100,
  "verdict": "pass|partial|fail",
  "reason": "short evidence"
}

Weights for overall_score: bear_appearance 45%, bear_presence 25%, view_revisit 20%, scene_consistency 10%.
"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def encode_image(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().strip("`")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in VLM response: {text[:300]}")
    return json.loads(match.group(0))


def wait_ready(api_base: str, timeout: int) -> None:
    url = f"{api_base.rstrip('/')}/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"VLM server not ready at {api_base}")


def score_one(metrics_path: Path, api_base: str, model: str, timeout: int, force: bool) -> Path:
    out_path = metrics_path.parent / "vlm_score.json"
    if out_path.is_file() and not force:
        return out_path

    metrics = load_json(metrics_path)
    images = list(metrics.get("first_frame_paths") or []) + list(metrics.get("revisit_tail_paths") or [])
    content: list[dict[str, Any]] = [{"type": "text", "text": USER_PROMPT}]
    for image_path in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(f"{api_base.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    vlm = parse_json(raw)

    out = {
        "metrics_path": str(metrics_path),
        "run_id": metrics.get("run_id"),
        "domain": (metrics.get("sample") or {}).get("domain"),
        "sample_id": (metrics.get("sample") or {}).get("sample_id"),
        "mode": metrics.get("mode"),
        "traditional": {
            "closure_psnr": metrics.get("closure_psnr"),
            "closure_ssim": metrics.get("closure_ssim"),
            "closure_mse": metrics.get("closure_mse"),
        },
        "vlm": vlm,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def maybe_score_vlm(metrics_paths: list[Path], args: argparse.Namespace) -> None:
    if not args.score_vlm:
        return
    wait_ready(args.vlm_api_base, args.startup_wait_sec)
    todo = [p for p in metrics_paths if args.force_vlm or not (p.parent / "vlm_score.json").is_file()]
    print(f"[vlm] scoring {len(todo)} / {len(metrics_paths)} cases with workers={args.vlm_workers}")
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=args.vlm_workers) as pool:
        futures = [pool.submit(score_one, p, args.vlm_api_base, args.vlm_model, args.vlm_timeout, args.force_vlm) for p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            path = fut.result()
            print(f"[vlm] {i}/{len(futures)} {path}")


def flat_row(metrics_path: Path, eval_root: Path) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    vlm_path = metrics_path.parent / "vlm_score.json"
    vlm = load_json(vlm_path).get("vlm", {}) if vlm_path.is_file() else {}
    sample = metrics.get("sample") or {}
    align = metrics.get("alignment") or {}
    init_ctx = align.get("initial_context") or {}
    key_ev = align.get("ckpt_key_evidence") or {}

    return {
        "eval_root": eval_root.name,
        "run_id": metrics.get("run_id"),
        "ckpt": metrics.get("ckpt"),
        "domain": sample.get("domain"),
        "sample_id": sample.get("sample_id"),
        "video_name": sample.get("video_name"),
        "start_frame": sample.get("start_frame"),
        "mode": metrics.get("mode"),
        "num_chunks": metrics.get("num_chunks"),
        "context_frames": metrics.get("context_frames"),
        "closure_psnr": metrics.get("closure_psnr"),
        "closure_ssim": metrics.get("closure_ssim"),
        "closure_mse": metrics.get("closure_mse"),
        "vlm_overall_score": vlm.get("overall_score"),
        "vlm_bear_appearance_score": vlm.get("bear_appearance_score"),
        "vlm_bear_presence_score": vlm.get("bear_presence_score"),
        "vlm_scene_consistency_score": vlm.get("scene_consistency_score"),
        "vlm_view_revisit_score": vlm.get("view_revisit_score"),
        "vlm_verdict": vlm.get("verdict"),
        "vlm_reason": vlm.get("reason"),
        "memory_profile": align.get("memory_profile"),
        "memory_profile_matched": align.get("memory_profile_matched"),
        "action_injection_impl": align.get("action_injection_impl"),
        "camera_inject_mode_effective": align.get("camera_inject_mode_effective"),
        "rt_encoding_effective": align.get("rt_encoding_effective"),
        "use_framepack_memory": align.get("use_framepack_memory"),
        "use_framepack_length_compress": align.get("use_framepack_length_compress"),
        "framepack_ratio": align.get("framepack_ratio"),
        "use_spatial_memory": align.get("use_spatial_memory"),
        "use_spatial_memory_legacy": align.get("use_spatial_memory_legacy"),
        "spatial_memory_tokens": align.get("spatial_memory_tokens"),
        "spatial_memory_inject_mode": align.get("spatial_memory_inject_mode"),
        "ckpt_has_camera_encoder_keys": key_ev.get("has_camera_encoder_keys"),
        "ckpt_has_spatial_memory_module_keys": key_ev.get("has_spatial_memory_module_keys"),
        "ckpt_has_ssm_keys": key_ev.get("has_ssm_keys"),
        "initial_context_source": init_ctx.get("training_memory_source"),
        "initial_context_detail": init_ctx.get("context_source_detail"),
        "initial_context_frame_count": init_ctx.get("context_frame_count"),
        "stage1_metrics_path": str(metrics_path),
        "vlm_score_path": str(vlm_path) if vlm_path.is_file() else "",
        "video_path": metrics.get("video_path"),
    }


def mean(values: list[Any]) -> float | None:
    xs = []
    for value in values:
        try:
            if value is not None and value != "":
                xs.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(xs) / len(xs) if xs else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["run_id"], row["domain"], row["mode"])].append(row)

    out = []
    for (run_id, domain, mode), group in sorted(groups.items()):
        out.append(
            {
                "run_id": run_id,
                "domain": domain,
                "mode": mode,
                "num_cases": len(group),
                "mean_closure_psnr": mean([r["closure_psnr"] for r in group]),
                "mean_closure_ssim": mean([r["closure_ssim"] for r in group]),
                "mean_closure_mse": mean([r["closure_mse"] for r in group]),
                "mean_vlm_overall_score": mean([r["vlm_overall_score"] for r in group]),
                "mean_vlm_bear_appearance_score": mean([r["vlm_bear_appearance_score"] for r in group]),
                "mean_vlm_bear_presence_score": mean([r["vlm_bear_presence_score"] for r in group]),
                "mean_vlm_scene_consistency_score": mean([r["vlm_scene_consistency_score"] for r in group]),
                "mean_vlm_view_revisit_score": mean([r["vlm_view_revisit_score"] for r in group]),
                "memory_profile": group[0]["memory_profile"],
                "context_frames": group[0]["context_frames"],
                "use_framepack_memory": group[0]["use_framepack_memory"],
                "use_framepack_length_compress": group[0]["use_framepack_length_compress"],
                "framepack_ratio": group[0]["framepack_ratio"],
                "use_spatial_memory": group[0]["use_spatial_memory"],
                "use_spatial_memory_legacy": group[0]["use_spatial_memory_legacy"],
                "spatial_memory_tokens": group[0]["spatial_memory_tokens"],
                "spatial_memory_inject_mode": group[0]["spatial_memory_inject_mode"],
                "eval_roots": "|".join(sorted({str(r["eval_root"]) for r in group})),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", action="append", required=True)
    ap.add_argument("--out-dir", default="revisit_materials")
    ap.add_argument("--prefix", default="ep0_revisit")
    ap.add_argument("--score-vlm", action="store_true")
    ap.add_argument("--force-vlm", action="store_true")
    ap.add_argument("--vlm-workers", type=int, default=8)
    ap.add_argument("--vlm-api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--vlm-timeout", type=int, default=180)
    ap.add_argument("--startup-wait-sec", type=int, default=900)
    args = ap.parse_args()

    eval_roots = [Path(p) for p in args.eval_root]
    metrics_paths: list[Path] = []
    for root in eval_roots:
        stage1 = root / "stage1"
        found = sorted(stage1.rglob("stage1_metrics.json"))
        print(f"[collect] {root}: {len(found)} stage1 metrics")
        metrics_paths.extend(found)

    maybe_score_vlm(metrics_paths, args)

    rows = []
    for root in eval_roots:
        for path in sorted((root / "stage1").rglob("stage1_metrics.json")):
            rows.append(flat_row(path, root))
    agg_rows = aggregate(rows)

    out_dir = Path(args.out_dir)
    cases_csv = out_dir / f"{args.prefix}_cases.csv"
    aggregate_csv = out_dir / f"{args.prefix}_aggregate_by_model_domain_mode.csv"
    write_csv(cases_csv, rows)
    write_csv(aggregate_csv, agg_rows)
    (out_dir / f"{args.prefix}_cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{args.prefix}_aggregate_by_model_domain_mode.json").write_text(json.dumps(agg_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] cases={len(rows)} -> {cases_csv}")
    print(f"[export] aggregates={len(agg_rows)} -> {aggregate_csv}")


if __name__ == "__main__":
    main()
