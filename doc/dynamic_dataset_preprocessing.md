# Dynamic training data — SpatialVID subset

Echo-Memory’s **dynamic training pool** uses a **simplified subset** of [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID): ego-centric clips with camera poses and captions, exported into the **same on-disk layout** as static training (`frames/`, `jsons/`, `overlap_labels/`, `metadata_full.csv`).

This guide covers **download → export → training settings** only. It does not describe a public benchmark or level splits.

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

## 2. Export to Echo-Memory training layout

Pick a release root, e.g. `data/dynamic-memory-dataset/`:

```text
dynamic-memory-dataset/
├── frames/{clip_id}/0000.png … 0080.png   # 81 frames per clip
├── jsons/{clip_id}.json                   # CineCameraActor poses
├── overlap_labels/{clip_id}/              # FOV overlap lists (optional but recommended)
├── captions.txt                           # clip_id<TAB>prompt
└── metadata_full.csv                      # train index for VideoDataset
```

**Per-clip steps:**

| Step | Setting |
| --- | --- |
| Frame sample | **81** PNGs per clip, **640×352** |
| Pose | Interpolate `poses.npy` + `indexes.txt` → `jsons/{clip_id}.json` (Euler `CineCameraActor` format, same as static data) |
| Prompt | Short caption from `caption.json` (`SceneSummary` or `SceneDescription`) |
| Overlap | Build `overlap_labels/` for FOV-based context retrieval |
| Metadata row | `video`, `prompt`, `video_name`, `start_frame`, `end_frame` |

`metadata_full.csv` is written at export time — **do not** re-run `run_generate_metadata.sh` unless you regenerate from raw frames only.

---

## 3. Training settings

Point Echo-Memory at the exported root:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/dynamic-memory-dataset
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

Recommended settings for dynamic-memory fine-tuning (match memory baseline scripts):

| Parameter | Typical value |
| --- | --- |
| Resolution | **640 × 352** |
| Frames / chunk | **81** |
| Context frames | **1–20** (recipe-dependent) |
| `--use_rt_relative` | on |
| `--enable_fov_retrieval` | on (when `overlap_labels/` present) |
| `--enable_context_memory` | on for context / spatial / SSM rows |
| `--timestep_shift` | **15** |
| Learning rate | **1e-5** (adjust per row) |

Example — run a memory baseline on dynamic data (override paths in the shell or train script):

```bash
bash train/memory_baselines_basic/run_ablation_no_memory_baseline_two_chunk.sh
bash train/context_learning/run_pre_qkv_ctx5.sh
```

Ensure `dataset_base_path` / `dataset_metadata_path` in the script resolve to `${DATASET_BASE_PATH}` and `${DATASET_BASE_PATH}/metadata_full.csv`.

---

## 4. Checklist

- [ ] Hugging Face access approved for [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID)
- [ ] Subset of `group_****` archives downloaded and extracted
- [ ] Clips filtered (poses + caption present; optional motion / dynamic filters)
- [ ] `frames/`, `jsons/`, `metadata_full.csv` under one root
- [ ] (Recommended) `overlap_labels/` for FOV retrieval
- [ ] `DATASET_BASE_PATH` exported before training

---

## Reference

- SpatialVID: [SpatialVID/SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) · [arXiv:2509.09676](https://arxiv.org/abs/2509.09676)
- Static in-domain pool: [dataset_preprocessing.md](dataset_preprocessing.md)
