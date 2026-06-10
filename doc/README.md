# Echo-Memory documentation

| Doc | Echo pool | Covers |
| --- | --- | --- |
| [DEVELOPER.md](DEVELOPER.md) | **Developer guide** — workflows, `.cursor/skills/`, Cursor Agent |
| [runtime_smoke_tests.md](runtime_smoke_tests.md) | **Runtime smoke tests** — DiffSynth model-source fix, inference smoke, 8-GPU train smoke |
| [checkpoints.md](checkpoints.md) | **Hugging Face weights** — [Echo-Team/Echo-Memory](https://huggingface.co/Echo-Team/Echo-Memory) baseline index |
| [memory_mechanisms.md](memory_mechanisms.md) | **Memory mechanisms** — paper row names, code modules, and training scripts |
| [dataset_preprocessing.md](dataset_preprocessing.md) | Static in-domain pool | Echo-Team package download → layout → metadata → latents |
| [dynamic_dataset_preprocessing.md](dynamic_dataset_preprocessing.md) | Dynamic training pool | subset download → export → training settings |

**Static in-domain pool:** download the Echo-Team package before in-domain replay/revisit eval.

**Dynamic training pool:** SpatialVID subset export + `DATASET_BASE_PATH` before training on the dynamic pool.
