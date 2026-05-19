# Open-Domain SSM Case Assets

This directory contains a paired open-domain `right45_return_2chunk` case copied from `CAM-CL/eval_outputs`.

The case is `ood_00006_1774370005`, one of the released toy-bear first-frame sources. It is useful for visually comparing the two SSM-style memory rows:

- `legacy_videossm/`: legacy VideoSSM hybrid baseline.
- `blockwise_ssm/`: paper-aligned block-wise SSM baseline.

Each subdirectory contains:

- `revisit_gen_only.mp4`: generated two-chunk return video.
- `first_00.png`: input first frame.
- `revisit_tail_*.png`: final return evidence frames.

These files are static examples for README display. To regenerate OOD SSM cases for a new checkpoint, use:

```bash
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```
