# Paper Case Visual Assets

This directory contains qualitative assets copied into the release for direct README rendering and paper-case inspection.

- `figure_1_abs_framework.png`: paper teaser and workflow figure for the project landing page.
- `figure_2_mem_overview.png`: overview of the memory design matrix.
- `*_first.png`: source or reference first-frame evidence for one method row.
- `*_tail.png`: final revisit-tail evidence for the same method row.

These files are static paper-facing examples. To generate new videos and evidence frames for a checkpoint, use:

```bash
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```
