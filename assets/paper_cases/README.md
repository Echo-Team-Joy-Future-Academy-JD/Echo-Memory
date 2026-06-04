# Paper Case Visual Assets

This directory contains paper-facing figures copied into the release for direct README rendering.

- `figure_1_abs_framework.png`: paper teaser and workflow figure for the project landing page.
- `figure_2_mem_overview.png`: overview of the memory design matrix.

To generate new open-domain revisit videos and evidence frames for a checkpoint, use:

```bash
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```
