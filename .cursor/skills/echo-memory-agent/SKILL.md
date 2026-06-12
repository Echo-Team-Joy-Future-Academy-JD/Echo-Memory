---
name: echo-memory-agent
description: >-
  Scope Cursor Agent prompts for Echo-Memory development — memory families,
  entry scripts, project skills, and public-repo constraints. Use when vibe
  coding, writing .cursor/rules, or planning multi-file agent tasks in this repo.
---

# Echo-Memory Agent development

## Project skills (read before large tasks)

| Skill | Path | When |
| --- | --- | --- |
| Training | `.cursor/skills/echo-memory-train/` | Memory baselines, context rows, `train/` |
| Evaluation | `.cursor/skills/echo-memory-eval/` | `eval/v2`, HF checkpoint checks |
| Release & site | `.cursor/skills/echo-memory-release/` | gh-pages, i18n, checkpoints doc |

Invoke by name in chat: e.g. *use echo-memory-eval to add a context_k1 replay check*.

## Prompt template

Include in Agent requests:

1. **Memory family** — Context, Compression, Spatial, State-Space (or row id)
2. **Entry script** — e.g. `train/.../run_spatial_memory_baseline.sh`
3. **Eval branch** — replay / in-domain / open-domain
4. **Pool** — static in-domain vs dynamic training (`DATASET_BASE_PATH`)

## Example prompts

```text
Using echo-memory-eval: download context_k1 from Echo-Team/Echo-Memory and
run eval/v2/run_basic_replay_gt.sh with the static in-domain pool.

Using echo-memory-train: add OUTPUT_BASE_ROOT override docs to
run_ablation_block_wise_ssm_two_chunk.sh without changing defaults.

Trace env/memory_baseline_runtime.py for spatial_mem → inject flags;
update doc/checkpoints.md with a short mapping table.
```

## Rules (optional)

Add `.cursor/rules/echo-memory.mdc` for standing constraints:

- Echo pool naming in public markdown
- No upload scripts, internal benchmark names, or `/pfs/` paths in GitHub
- Minimal diff; match existing bash/Python patterns in `train/` and `eval/v2/`

## Modes

- **Agent** — multi-file implementation
- **Ask** — read `diffsynth/`, trace configs, no edits
