# Data Construction

Scripts assume a context-based memory dataset with `frames/`, `jsons/`, and a metadata CSV. Set `DATASET_BASE_PATH` before running them.

## Metadata CSV

```bash
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
bash scripts/run_generate_metadata.sh
```

Optional variables:

- `OUTPUT_CSV`: output CSV path, defaults to `${DATASET_BASE_PATH}/metadata_full.csv`.
- `SEGMENT_LENGTH`: frames per segment, default `81`.
- `CONTEXT_FRAMES`: context frames in metadata construction, default `5`.

## Latent Precompute

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=/path/to/Context-as-Memory-Dataset
NUM_PROCESSES=8 bash scripts/run_precompute_ctx_target_latents.sh
```

Optional variables:

- `MODEL_PATHS`: JSON list of model weight paths.
- `CONTEXT_FRAMES`: number of context frames, default `20`.
- `NUM_PROCESSES`: accelerate processes, default `1`.

Open-domain revisit first-frame assets are already included under `assets/opendomain_revisit`; no dataset construction step is needed for those probes.
