## Models notes

This folder contains multiple TFocNet variants that evolved over time.

### `tfoc_vanilla_maskdraw_light_clutter_n_v1.py`

This file is an **algorithmic v1 rewrite** based on
`tfoc_vanilla_maskdraw_light_clutter_n.py`, aimed at improving training
stability and optimization behavior for the task:

- Input: \(t=9\) consecutive radar frames (e.g., RD tensors).
- Goal: use the first 8 frames to **enhance** the 9th (target) frame for better
  detection/segmentation.

#### What was problematic in the original

- **Hard mask-draw path (non-differentiable selection)**:
  `argmax → one_hot → majority vote → gather → topk` makes the "where to sample"
  decision effectively non-trainable (gradients cannot shape the selection).
- **Probability aggregation with `prod` across time**:
  multiplying probabilities across 8 frames can quickly underflow and weaken the
  signal used by the auxiliary track branch.
- **Selection direction ambiguity**:
  using `topk(... largest=False)` on a probability-like tensor is easy to get
  backwards (picking "least similar" instead of "most similar").

#### What v1 changes

- **Soft, differentiable attention** over the local \(k\times k\) candidate set:
  - Compute per-candidate 4-class probabilities (via `inner_predict`).
  - Define attention score as foreground confidence: \(1 - p(\text{bg})\).
  - Apply `softmax` across candidates, then perform a weighted sum over history
    features.
  - Perform a second soft aggregation across time (favoring higher confidence).
- **Stable auxiliary probability map**:
  use `max-over-candidates` then `mean-over-time` (instead of `prod-over-time`).
- **No `matplotlib` import / no side-effect CUDA env setting** in the model file.

#### How to use

This file defines a `TFocNet` class, similar to the original file it is based on.
To try it, update your model import/registration so that training uses
`tfoc_vanilla_maskdraw_light_clutter_n_v1.py` instead of the older variant.

If your training loop expects a different TFocNet forward signature/return
values (e.g., `rd_outputs, ra_outputs, rd_probs, ra_probs`), you should either:

- adapt the training loop for this model's outputs, or
- adapt this model's `forward()` to match the expected interface.

