# Echo-Memory documentation

| Doc | Description |
| --- | --- |
| [dataset_preprocessing.md](dataset_preprocessing.md) | **Static / in-domain** — [Context-as-Memory-Dataset](https://huggingface.co/datasets/KlingTeam/Context-as-Memory-Dataset): download, extract, metadata & latent precompute |
| [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md) | **Dynamic training set** — [SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) → CamCL-ready export (DynMemBench-V2 / CAM-CL pipeline) |

**Static:** start with Context-as-Memory before `scripts/run_generate_metadata.sh` or static replay/revisit eval.

**Dynamic:** SpatialVID download + `camcl_ready/` export before dynamic-memory training (see CAM-CL `train_dynmembench_v2/`).
