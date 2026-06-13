# Dynamic SpatialVID Training

Training recipes for the motion-filtered dynamic SpatialVID pool. They mirror the six dynamic rows used for demos and inference:

| Row | Script | Notes |
| --- | --- | --- |
| Context K=1 | `run_dyn_ctx1.sh` | FOV top-0, 1 context frame |
| Context K=5 | `run_dyn_ctx5.sh` | FOV top-4, per-frame context VAE |
| Context K=20 | `run_dyn_ctx20.sh` | FOV top-19, per-frame context VAE |
| Spatial Memory | `run_dyn_spatial_mem.sh` | 64 spatial memory tokens |
| Block-wise SSM | `run_dyn_block_wise_ssm.sh` | Paper-aligned state-space memory |
| VideoSSM hybrid | `run_dyn_videossm_hybrid.sh` | Legacy temporal-conv baseline |

Set paths through environment variables:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/dynamic-spatialvid-motion60/mixed
export OUTPUT_BASE_ROOT=$PWD/outputs/dynamic_spatialvid
```

The local validation pool used during development is:

```bash
export DATASET_BASE_PATH=/pfs/weiyang/DynMemBench-V2/camcl_spatialvid_motion60_ready/mixed
```

Public scripts should keep the relative form above. The expected dataset root contains:

```text
mixed/
├── frames/L{1,2,3}/{clip_id}/0000.png ... 0080.png
├── jsons/L{1,2,3}/{clip_id}.json
├── overlap_labels/L{1,2,3}/{clip_id}/
├── metadata_train.csv
├── metadata_train_sample.csv
├── metadata_train_sample_L1.csv
├── metadata_eval.csv
└── metadata_eval_2chunk.csv
```

For quick local validation, override the metadata and step count:

```bash
METADATA_NAME=metadata_train_sample_L1.csv \
MAX_TRAIN_STEPS=1 \
PROGRESS_TOTAL_STEPS=30000 \
NUM_WORKERS=0 \
bash train/dynamic_spatialvid/run_dyn_block_wise_ssm.sh
```

Dynamic evaluation is TODO; current public support covers training and inference.
