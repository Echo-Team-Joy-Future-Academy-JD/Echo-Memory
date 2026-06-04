# Open-Domain Revisit Sources

This folder contains the eight held-out first-frame sources used by the
open-domain revisit probe in the Echo-Memory paper.

Each image is treated as a first frame for a short controlled camera-return
probe. The default prompt used by `eval/v2/revisit_suite` is:

```text
A toy bear in the same static scene. Preserve the bear appearance and the scene layout after camera revisit.
```

To run the probe, point `OOD_DIR` here or use the default:

```bash
export WAN_BASE_MODEL=/path/to/Wan2.1-T2V-1.3B
PHASE=stage1 OOD_DIR=assets/opendomain_revisit \
  bash eval/v2/revisit_suite/run_one_click_revisit_eval.sh
```
