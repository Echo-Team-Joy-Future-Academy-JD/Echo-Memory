# Dynamic training set (SpatialVID) — download & CamCL-ready export

Echo-Memory’s **dynamic memory training set** follows the **DynMemBench-V2 / CAM-CL** pipeline: raw clips from [SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) are scored, split into difficulty levels, and exported into the same **Context-as-Memory–compatible layout** (`frames/`, `jsons/`, `overlap_labels/`, `metadata_*.csv`) used by Echo-Memory / CAM-CL training.

Reference implementation: **CAM-CL** `DATASET.md` (DynMemBench-V2 design) and **DynMemBench-V2** manifests/export scripts (same machine layout: `/pfs/weiyang/CAM-CL`, `/pfs/weiyang/DynMemBench-V2`).

**License note:** SpatialVID is [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial use only. Context-as-Memory and Echo-Memory static training use a different dataset; check licenses before mixing.

---

## 1. Download SpatialVID

**Hugging Face:** [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID)

- You must **accept the dataset terms** on Hugging Face (contact form) before download.
- Full corpus ≈ **7.67 TB**, split into **545 groups** (`group_0001`, …).

### Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # required after terms acceptance

export SPATIALVID_ROOT=/path/to/SpatialVID
hf download SpatialVID/SpatialVID --repo-type dataset --local-dir "${SPATIALVID_ROOT}"
```

Download **specific groups** instead of the full 7.67 TB when prototyping — see the [SpatialVID GitHub](https://github.com/NJU-3DV/SpatialVID) `download_SpatialVID.py` helper mentioned on the dataset card.

### Extract group archives

Annotation and video groups ship as `.tar.gz` files:

```bash
cd "${SPATIALVID_ROOT}"

# annotations
tar -xzvf annotations/group_0001.tar.gz
tar -xzvf annotations/group_0002.tar.gz

# videos (if downloaded separately)
tar -xzvf videos/group_0001.tar.gz
```

### Expected raw layout

```text
SPATIALVID_ROOT/
├── annotations/
│   └── group_0001/
│       └── {video_uuid}/
│           ├── caption.json
│           ├── dyn_masks.npz
│           ├── indexes.txt
│           ├── intrinsics.npy
│           ├── instructions.json
│           └── poses.npy          # (N, 7) = tx,ty,tz,qx,qy,qz,qw
├── videos/
│   └── group_0001/
│       └── {video_uuid}.mp4
├── data/train/SpatialVID_metadata.csv
└── depths/                        # optional
```

Metadata CSV fields (`id`, `group id`, `video path`, `annotation path`, `motionTags`, `sceneType`, …) are documented on the [dataset card](https://huggingface.co/datasets/SpatialVID/SpatialVID).

---

## 2. Construction pipeline (CAM-CL / DynMemBench-V2)

High-level steps (full detail in CAM-CL `DATASET.md`):

| Step | What happens |
| --- | --- |
| 1. Source | SpatialVID-HQ clips (+ optional web dynamic pool) |
| 2. Caption / tags | `caption.json`, scene & motion metadata |
| 3. Dynamic complexity score | Rank clips by motion / occlusion difficulty |
| 4. Level split | **L1** visible persistence · **L2** simple motion · **L3** complex dynamics |
| 5. Train / eval split | 30 eval clips per level (poses required) |
| 6. **CamCL-ready export** | PNG frames + pose JSON + overlap labels + metadata CSV |

Exported scale (DynMemBench-V2): **~230k train** + **90 eval** clips across L1/L2/L3.

---

## 3. CamCL-ready export format

Target root (default in CAM-CL scripts): `camcl_ready/`

```text
camcl_ready/
├── L1/
│   ├── frames/{video_id}/0000.png … 0080.png    # 81 frames (1 chunk, train)
│   ├── jsons/{video_id}.json                    # CineCameraActor poses
│   ├── overlap_labels/{video_id}/{frame}.json   # FOV overlap indices
│   ├── captions.txt
│   ├── metadata_full.csv                        # train rows
│   └── metadata_eval_2chunk.csv                 # 162-frame eval (optional)
├── L2/ …
├── L3/ …
└── mixed/                                       # optional L1+L2+L3 union
    ├── frames/L1/ → ../../L1/frames
    ├── metadata_train.csv
    └── metadata_eval.csv
```

**Per-video processing** (from SpatialVID annotations):

1. **Frame extract** — sample source MP4 to **81** PNGs @ **640×352** (train) or **162** (2-chunk eval).
2. **Pose convert** — interpolate `poses.npy` + `indexes.txt` → `jsons/{id}.json` (`CineCameraActor`, Euler degrees).
3. **Overlap labels** — FOV-based overlapping frame lists for context retrieval.
4. **Metadata CSV** — columns: `video`, `prompt`, `video_name`, `start_frame`, `end_frame`.

**Helper scripts** (DynMemBench-V2):

```bash
# Re-export eval clips with 162 frames (2 × 81 chunks)
python export_multichunk_eval.py --help   # DynMemBench-V2/

# Build mixed L1+L2+L3 training view
bash setup_mixed_dataset.sh               # DynMemBench-V2/
```

Manifests listing train/eval IDs per level: `DynMemBench-V2/manifests/L{1,2,3}_{train,eval}.jsonl`.

---

## 4. Training (CAM-CL reference)

Dynamic training recipes live under **CAM-CL** (`train_dynmembench_v2/`). They call the same `train.py` stack as Echo-Memory with FOV retrieval and context memory enabled.

```bash
cd /path/to/CAM-CL

export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DYMMEMBENCH_V2_EXPORT_ROOT=/path/to/camcl_ready

# Level-specific (example: L1, context K=1)
bash train_dynmembench_v2/train_L1_ctx1.sh

# Mixed levels
bash train_dynmembench_v2/train_mixed_ctx5.sh
```

Typical settings: **640×352**, **81 frames/chunk**, `--enable_fov_retrieval`, `--use_rt_relative`, `--enable_context_memory`.

---

## 5. Use with Echo-Memory

Point Echo-Memory at an exported level (metadata is already built — **no** `run_generate_metadata.sh` needed):

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/camcl_ready/L1   # or .../mixed
export PYTHONPATH=$PWD:${PYTHONPATH:-}

# Example: memory baseline with dyn data (override dataset in train script env)
bash train/memory_baselines_basic/run_ablation_no_memory_baseline_two_chunk.sh
```

For **dynamic evaluation** beyond static Context-as-Memory revisit, see CAM-CL `eval_dynmembench_v2/` (working-memory metrics per L1/L2/L3).

---

## 6. Quick checklist

- [ ] Hugging Face access approved for [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID)
- [ ] Required `group_****` archives downloaded and extracted
- [ ] `camcl_ready/L{1,2,3}/` contains `frames/`, `jsons/`, `overlap_labels/`, `metadata_full.csv`
- [ ] (Optional) `metadata_eval_2chunk.csv` for 2-chunk GT replay
- [ ] (Optional) `camcl_ready/mixed/` for multi-level training
- [ ] `DATASET_BASE_PATH` or `DYMMEMBENCH_V2_EXPORT_ROOT` exported before training

---

## Reference

- SpatialVID dataset: [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) · [arXiv:2509.09676](https://arxiv.org/abs/2509.09676)
- DynMemBench-V2 / export design: [CAM-CL/DATASET.md](../../CAM-CL/DATASET.md)
- Static (in-domain) dataset: [dataset_preprocessing.md](dataset_preprocessing.md)
