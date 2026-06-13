# Dynamic SpatialVID Evaluation TODO

Dynamic SpatialVID currently exposes training and inference recipes only:

- `train/dynamic_spatialvid/`
- `inference/dynamic_spatialvid/`

Planned evaluation work:

- Define the public dynamic replay/revisit protocol.
- Freeze the train/eval split and metric table.
- Add one-click evaluation scripts after the protocol is fixed.

For qualitative demos, sample scenes from the dynamic training metadata, replay them across the six dynamic rows, and manually pick representative results.
