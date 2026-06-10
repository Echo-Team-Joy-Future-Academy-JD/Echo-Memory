#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flat_row(stage1_path: Path) -> dict[str, Any]:
    m = _load_json(stage1_path)
    vlm_path = stage1_path.parent / "vlm_score.json"
    vlm = _load_json(vlm_path).get("vlm", {}) if vlm_path.is_file() else {}
    sample = m.get("sample") or {}
    align = m.get("alignment") or {}
    key_ev = align.get("ckpt_key_evidence") or {}
    init_ctx = align.get("initial_context") or {}
    row = {
        "run_id": m.get("run_id"),
        "ckpt": m.get("ckpt"),
        "domain": sample.get("domain"),
        "sample_id": sample.get("sample_id"),
        "video_name": sample.get("video_name"),
        "start_frame": sample.get("start_frame"),
        "mode": m.get("mode"),
        "num_chunks": m.get("num_chunks"),
        "context_frames": m.get("context_frames"),
        "action_injection_impl": align.get("action_injection_impl"),
        "camera_inject_mode_label_from_path": align.get("camera_inject_mode_label_from_path") or align.get("camera_inject_mode"),
        "camera_inject_mode_effective": align.get("camera_inject_mode_effective"),
        "camera_encoder_separate_t_r_effective": align.get("camera_encoder_separate_t_r_effective"),
        "camera_encoder_effective": align.get("camera_encoder_effective"),
        "rt_encoding_effective": align.get("rt_encoding_effective"),
        "ckpt_has_camera_encoder_keys": key_ev.get("has_camera_encoder_keys"),
        "ckpt_has_separate_rt_keys": key_ev.get("has_separate_rt_keys"),
        "ckpt_has_action_mlp_keys": key_ev.get("has_action_mlp_keys"),
        "ckpt_has_action_attention_keys": key_ev.get("has_action_attention_keys"),
        "ckpt_has_spatial_memory_module_keys": key_ev.get("has_spatial_memory_module_keys"),
        "ckpt_has_block_wise_ssm_keys": key_ev.get("has_block_wise_ssm_keys"),
        "ckpt_has_videossm_hybrid_keys": key_ev.get("has_videossm_hybrid_keys"),
        "ckpt_has_ssm_keys": key_ev.get("has_ssm_keys"),
        "memory_profile": align.get("memory_profile"),
        "memory_profile_matched": align.get("memory_profile_matched"),
        "use_framepack_memory": align.get("use_framepack_memory"),
        "use_framepack_length_compress": align.get("use_framepack_length_compress"),
        "framepack_ratio": align.get("framepack_ratio"),
        "use_spatial_memory": align.get("use_spatial_memory"),
        "use_spatial_memory_legacy": align.get("use_spatial_memory_legacy"),
        "spatial_memory_module_loaded": align.get("spatial_memory_module_loaded"),
        "spatial_memory_effective_impl": align.get("spatial_memory_effective_impl"),
        "spatial_memory_tokens": align.get("spatial_memory_tokens"),
        "spatial_memory_inject_mode": align.get("spatial_memory_inject_mode"),
        "initial_context_source": init_ctx.get("training_memory_source"),
        "initial_context_detail": init_ctx.get("context_source_detail"),
        "initial_context_frame_count": init_ctx.get("context_frame_count"),
        "initial_context_action_count": init_ctx.get("context_action_count"),
        "initial_context_indices": "|".join(str(x) for x in (init_ctx.get("context_indices") or [])),
        "closure_mse": m.get("closure_mse"),
        "closure_psnr": m.get("closure_psnr"),
        "closure_ssim": m.get("closure_ssim"),
        "video_path": m.get("video_path"),
        "first_frame_paths": "|".join(m.get("first_frame_paths") or []),
        "revisit_tail_paths": "|".join(m.get("revisit_tail_paths") or []),
        "first_last_chunk_change_paths": "|".join(m.get("first_last_chunk_change_paths") or []),
        "vlm_overall_score": vlm.get("overall_score"),
        "vlm_bear_appearance_score": vlm.get("bear_appearance_score"),
        "vlm_bear_presence_score": vlm.get("bear_presence_score"),
        "vlm_scene_consistency_score": vlm.get("scene_consistency_score"),
        "vlm_view_revisit_score": vlm.get("view_revisit_score"),
        "vlm_verdict": vlm.get("verdict"),
        "vlm_reason": vlm.get("reason"),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _mean(values: list[Any]) -> float | None:
    xs = []
    for v in values:
        try:
            if v is not None:
                xs.append(float(v))
        except Exception:
            pass
    return sum(xs) / len(xs) if xs else None


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("run_id"), r.get("domain"), r.get("mode"))].append(r)
    out = []
    for (run_id, domain, mode), rs in sorted(groups.items()):
        out.append(
            {
                "run_id": run_id,
                "domain": domain,
                "mode": mode,
                "num_cases": len(rs),
                "mean_closure_psnr": _mean([r.get("closure_psnr") for r in rs]),
                "mean_closure_ssim": _mean([r.get("closure_ssim") for r in rs]),
                "mean_closure_mse": _mean([r.get("closure_mse") for r in rs]),
                "mean_vlm_overall_score": _mean([r.get("vlm_overall_score") for r in rs]),
                "mean_vlm_bear_appearance_score": _mean([r.get("vlm_bear_appearance_score") for r in rs]),
                "mean_vlm_bear_presence_score": _mean([r.get("vlm_bear_presence_score") for r in rs]),
                "mean_vlm_scene_consistency_score": _mean([r.get("vlm_scene_consistency_score") for r in rs]),
                "mean_vlm_view_revisit_score": _mean([r.get("vlm_view_revisit_score") for r in rs]),
                "memory_profile": rs[0].get("memory_profile"),
                "action_injection_impl": rs[0].get("action_injection_impl"),
                "camera_inject_mode_label_from_path": rs[0].get("camera_inject_mode_label_from_path"),
                "camera_inject_mode_effective": rs[0].get("camera_inject_mode_effective"),
                "rt_encoding_effective": rs[0].get("rt_encoding_effective"),
                "ckpt_has_camera_encoder_keys": rs[0].get("ckpt_has_camera_encoder_keys"),
                "ckpt_has_separate_rt_keys": rs[0].get("ckpt_has_separate_rt_keys"),
                "ckpt_has_action_mlp_keys": rs[0].get("ckpt_has_action_mlp_keys"),
                "ckpt_has_action_attention_keys": rs[0].get("ckpt_has_action_attention_keys"),
                "ckpt_has_spatial_memory_module_keys": rs[0].get("ckpt_has_spatial_memory_module_keys"),
                "ckpt_has_block_wise_ssm_keys": rs[0].get("ckpt_has_block_wise_ssm_keys"),
                "ckpt_has_videossm_hybrid_keys": rs[0].get("ckpt_has_videossm_hybrid_keys"),
                "ckpt_has_ssm_keys": rs[0].get("ckpt_has_ssm_keys"),
                "context_frames": rs[0].get("context_frames"),
                "initial_context_source": rs[0].get("initial_context_source"),
                "initial_context_detail": rs[0].get("initial_context_detail"),
                "initial_context_frame_count": rs[0].get("initial_context_frame_count"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir-name", default="revisit_eval_summary")
    args = ap.parse_args()

    stage1_root = Path(args.stage1_root)
    manifest = _load_json(Path(args.manifest))
    ckpt_by_run = {row["run_id"]: Path(row["ckpt"]) for row in manifest.get("ckpts", [])}

    rows = [_flat_row(p) for p in sorted(stage1_root.rglob("stage1_metrics.json"))]
    agg_rows = aggregate(rows)
    _write_csv(stage1_root / "all_cases.csv", rows)
    _write_csv(stage1_root / "aggregate_by_model_domain_mode.csv", agg_rows)
    (stage1_root / "all_cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage1_root / "aggregate_by_model_domain_mode.json").write_text(json.dumps(agg_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_run[str(r.get("run_id"))].append(r)
    for run_id, rs in by_run.items():
        ckpt = ckpt_by_run.get(run_id)
        if not ckpt:
            continue
        out_dir = ckpt.parent / args.out_dir_name
        ars = aggregate(rs)
        _write_csv(out_dir / "cases.csv", rs)
        _write_csv(out_dir / "aggregate.csv", ars)
        (out_dir / "cases.json").write_text(json.dumps(rs, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "aggregate.json").write_text(json.dumps(ars, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] cases={len(rows)} aggregates={len(agg_rows)} stage1_root={stage1_root}")


if __name__ == "__main__":
    main()
