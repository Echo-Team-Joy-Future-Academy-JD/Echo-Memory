#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests


SYSTEM = """You are a strict visual memory evaluator. Return only valid JSON."""


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


def encode_image(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().strip("`")
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in VLM response: {text[:300]}")
    return json.loads(m.group(0))


def wait_ready(api_base: str, timeout: int) -> bool:
    url = f"{api_base.rstrip('/')}/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def score_one(metrics_path: Path, api_base: str, model: str, timeout: int, dry_run: bool) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    images = list(metrics.get("first_frame_paths") or []) + list(metrics.get("revisit_tail_paths") or [])
    if dry_run:
        resp = {
            "bear_appearance_score": 0,
            "bear_presence_score": 0,
            "scene_consistency_score": 0,
            "view_revisit_score": 0,
            "overall_score": 0,
            "verdict": "partial",
            "reason": "dry-run: VLM not called",
        }
    else:
        content: list[dict[str, Any]] = [{"type": "text", "text": USER_PROMPT}]
        for p in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(p)}"}})
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
        r = requests.post(f"{api_base.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        resp = parse_json(raw)

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
        "vlm": resp,
    }
    (metrics_path.parent / "vlm_score.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-root", required=True)
    ap.add_argument("--api-base", default=os.environ.get("VLM_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct"))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("VLM_TIMEOUT", "180")))
    ap.add_argument("--startup-wait-sec", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not wait_ready(args.api_base, args.startup_wait_sec):
        raise SystemExit(f"VLM server not ready at {args.api_base}")
    paths = sorted(Path(args.stage1_root).rglob("stage1_metrics.json"))
    rows = []
    for p in paths:
        if args.limit and len(rows) >= args.limit:
            break
        if (p.parent / "vlm_score.json").is_file() and not args.force:
            continue
        rows.append(score_one(p, args.api_base, args.model, args.timeout, args.dry_run))
        print(f"[vlm] scored {p}")
    summary = Path(args.stage1_root) / "vlm_scores_summary.jsonl"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[vlm] wrote {len(rows)} rows -> {summary}")


if __name__ == "__main__":
    main()
