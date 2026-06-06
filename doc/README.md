# Echo-Memory documentation

| Doc | Description |
| --- | --- |
| [dataset_preprocessing.md](dataset_preprocessing.md) | **Static / in-domain** training data — download, extract, metadata & optional latent precompute |
| [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md) | **Dynamic training data** — SpatialVID subset → Echo-Memory training layout & settings |

**Static:** use before `scripts/run_generate_metadata.sh` or in-domain replay/revisit eval.

**Dynamic:** SpatialVID subset export + `DATASET_BASE_PATH` before dynamic-memory training.
