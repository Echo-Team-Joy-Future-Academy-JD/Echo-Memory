# Data scripts (static in-domain pool)

Scripts for the **static in-domain pool**. Complete download and layout verification first — see **[doc/dataset_preprocessing.md](../doc/dataset_preprocessing.md)**.

Both training pools share the Echo-Memory layout (`frames/`, `jsons/`, `overlap_labels/`, `metadata_full.csv`). Set `DATASET_BASE_PATH` to the pool root before running these scripts.

## Metadata CSV

Download released metadata:

```bash
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
huggingface-cli download Echo-Team/Echo-Memory-Data metadata_full.csv \
  --repo-type dataset \
  --local-dir "${DATASET_BASE_PATH}"
```

Or regenerate it locally after changing the pool:

```bash
bash scripts/run_generate_metadata.sh
```

Optional variables:

- `OUTPUT_CSV`: output CSV path, defaults to `${DATASET_BASE_PATH}/metadata_full.csv`.
- `SEGMENT_LENGTH`: frames per segment, default `81`.
- `CONTEXT_FRAMES`: context frames in metadata construction, default `5`.

## Latent precompute

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
export DATASET_BASE_PATH=data/Context-as-Memory-Dataset
NUM_PROCESSES=8 bash scripts/run_precompute_ctx_target_latents.sh
```

Optional variables:

- `MODEL_PATHS`: JSON list of model weight paths.
- `CONTEXT_FRAMES`: number of context frames, default `20`.
- `NUM_PROCESSES`: accelerate processes, default `1`.

Open-domain revisit first-frame assets are already included under `assets/opendomain_revisit`; no dataset construction step is needed for those probes.
