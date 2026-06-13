# Dynamic training pool — SpatialVID subset

Echo-Memory’s **dynamic training pool** uses a motion-filtered subset of [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID): ego-centric clips with camera poses and captions, exported into the same sample format used by the static pool.

This guide covers **download → export → training/inference settings** only. Dynamic eval is TODO; current public support is training and inference.

**License:** SpatialVID is [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial). Static and dynamic pools may have different licenses — check before mixing runs.

---

## 1. Download (subset)

**Hugging Face:** [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID)

- Accept the dataset terms on Hugging Face before download.
- Full corpus is large (~7 TB+). For Echo-Memory dynamic training, download **selected groups** only — you do not need the full 545 groups.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login

export SPATIALVID_ROOT=/path/to/SpatialVID
hf download SpatialVID/SpatialVID --repo-type dataset --local-dir "${SPATIALVID_ROOT}"
```

To fetch specific groups, use include patterns or the helper script linked from the [dataset card](https://huggingface.co/datasets/SpatialVID/SpatialVID) (`download_SpatialVID.py` on the SpatialVID GitHub).

Extract downloaded `.tar.gz` groups:

```bash
cd "${SPATIALVID_ROOT}"
tar -xzvf annotations/group_0001.tar.gz
tar -xzvf videos/group_0001.tar.gz
```

### Raw layout (per clip)

```text
SPATIALVID_ROOT/
├── annotations/group_0001/{clip_id}/
│   ├── poses.npy          # (N, 7) = tx,ty,tz,qx,qy,qz,qw
│   ├── indexes.txt        # pose index → source frame index
│   ├── caption.json       # scene / motion text
│   └── dyn_masks.npz      # optional dynamic-region masks
├── videos/group_0001/{clip_id}.mp4
└── data/train/SpatialVID_metadata.csv
```

Use `SpatialVID_metadata.csv` to filter clips (e.g. `motion score`, `dynamicRatio`, `sceneType`) when building your subset.

---

## 2. Export to Echo layout (dynamic training pool)

Use `data/dynamic-spatialvid-motion60/mixed/` as the public training root and set `DATASET_BASE_PATH` to it:

```text
data/dynamic-spatialvid-motion60/
├── L1/                                    # single-level exports are also valid roots
├── L2/
├── L3/
└── mixed/
    ├── frames/L{1,2,3}/{clip_id}/0000.png ... 0080.png
    ├── jsons/L{1,2,3}/{clip_id}.json
    ├── overlap_labels/L{1,2,3}/{clip_id}/
    ├── captions.txt
    ├── metadata_train.csv
    ├── metadata_train_sample.csv
    ├── metadata_train_sample_L1.csv
    ├── metadata_eval.csv
    └── metadata_eval_2chunk.csv
```

**Per-clip steps:**

| Step | Setting |
| --- | --- |
| Frame sample | **81** PNGs per clip, **640×352** |
| Pose | Interpolate `poses.npy` + `indexes.txt` → `jsons/{clip_id}.json` (Euler `CineCameraActor` format, same as static data) |
| Prompt | Short caption from `caption.json` (`SceneSummary` or `SceneDescription`) |
| Overlap | Build `overlap_labels/` for FOV-based context retrieval |
| Metadata row | `video`, `prompt`, `video_name`, `start_frame`, `end_frame`, optional `level` |

`metadata_train.csv` is written at export time. Use `metadata_train_sample.csv` or `metadata_train_sample_L1.csv` for local step checks. Do not re-run `run_generate_metadata.sh` unless you regenerate from raw frames only.

---

## 3. Training settings

Same env vars and on-disk layout as the static in-domain pool — only `DATASET_BASE_PATH` changes.

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/dynamic-spatialvid-motion60/mixed
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

Recommended settings for the dynamic training pool (match memory baseline scripts):

| Parameter | Typical value |
| --- | --- |
| Resolution | **640 × 352** |
| Frames / chunk | **81** |
| Context frames | **1–20** (recipe-dependent) |
| `--use_rt_relative` | on |
| `--enable_fov_retrieval` | on (when `overlap_labels/` present) |
| `--enable_context_memory` | on for context / spatial / SSM rows |
| `--timestep_shift` | **15** |
| Learning rate | **5e-5** (adjust per row) |

Example — run a dynamic row:

```bash
METADATA_NAME=metadata_train.csv bash train/dynamic_spatialvid/run_dyn_spatial_mem.sh
```

For local one-step validation:

```bash
METADATA_NAME=metadata_train_sample_L1.csv \
MAX_TRAIN_STEPS=1 \
PROGRESS_TOTAL_STEPS=30000 \
NUM_WORKERS=0 \
bash train/dynamic_spatialvid/run_dyn_block_wise_ssm.sh
```

Inference wrappers live under `inference/dynamic_spatialvid/`.

---

## 4. Demo selection

Dynamic demos are selected from training-scene replay rather than from fixed eval scripts:

1. Randomly sample candidate scenes from `metadata_train.csv` or `metadata_train_sample.csv`.
2. Use the same prompt, first frame, and GT action trajectory for all six dynamic rows.
3. Run `inference/unified_inference.py` or `inference/dynamic_spatialvid/*.sh` for each checkpoint.
4. Manually pick a representative scene where all rows are viewable.

The checked-in README previews are compressed GIFs under `assets/readme_previews/`.

---

## 5. Checklist

- [ ] Hugging Face access approved for [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID)
- [ ] Subset of `group_****` archives downloaded and extracted
- [ ] Clips filtered (poses + caption present; optional motion / dynamic filters)
- [ ] `frames/`, `jsons/`, `metadata_train.csv` under one root
- [ ] (Recommended) `overlap_labels/` for FOV retrieval
- [ ] `DATASET_BASE_PATH` exported before training/inference

---

## Reference

- SpatialVID: [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) · [arXiv:2509.09676](https://arxiv.org/abs/2509.09676)
- Static in-domain pool: [dataset_preprocessing.md](dataset_preprocessing.md)
- Dynamic training pool: [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md)
