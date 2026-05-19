# Paper Case Visual Assets

This directory contains qualitative assets copied into the release for direct README rendering and paper-case inspection.

- `representative_sweep_panel.png`: compact panel for the representative open-domain revisit case.
- `representative_sweep_panel_highres.png`: higher-resolution version of the same panel.
- `*_first.png`: source or reference first-frame evidence for one method row.
- `*_tail.png`: final revisit-tail evidence for the same method row.
- `manifest.csv`: source manifest for the copied visual assets.

These files are static paper-facing examples. To generate new videos and evidence frames for a checkpoint, use:

```bash
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```
