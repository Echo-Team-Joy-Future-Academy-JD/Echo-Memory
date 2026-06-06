# Echo-Memory documentation

| Doc | Echo pool | Covers |
| --- | --- | --- |
| [checkpoints.md](checkpoints.md) | **Hugging Face weights** — [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory) baseline index |
| [dataset_preprocessing.md](dataset_preprocessing.md) | Static in-domain pool | download → layout → metadata → latents |
| [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md) | Dynamic training pool | subset download → export → training settings |

**Static in-domain pool:** use before `scripts/run_generate_metadata.sh` or in-domain replay/revisit eval.

**Dynamic training pool:** SpatialVID subset export + `DATASET_BASE_PATH` before training on the dynamic pool.
