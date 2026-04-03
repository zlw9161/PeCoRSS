# LMF_V6_ReXAttn Modification Report

## 1. Background and Goal

This note documents the LMF ReXAttn variant introduced on top of `Learnable_Match_FilterV6` in EchMo.

Primary objective:
- Keep the prior-guided LMF convolution path as the main backbone.
- Add a lightweight cross-attention residual branch to increase adaptive capacity.
- Preserve the original LMF tensor interface so the downstream MLDC/GDF/translator stack remains unchanged.

Design philosophy:
- Prior-guided convolution gives stable inductive bias.
- Cross-attention residual gives extra learnable flexibility with bounded impact via a small residual scale.

---

## 2. Core Idea

### 2.1 Main Path (unchanged)

The main path is exactly the complex convolution in LMF_V6:
- Real-imag decomposition convolution
- Recomposition into complex output
- Complex activation

So the baseline behavior is preserved.

### 2.2 Residual ReXAttn Path

A lightweight residual branch is added:
- Query comes from prior kernel statistics (or current kernel in residual mode).
- Key/Value come from input echo features.
- A small gating-style attention map is generated.
- Residual output is projected back to complex form and added to the main path output.

This branch does **not** replace convolution. It only augments it.

---

## 3. Mathematical View (Simplified)

Let the baseline LMF output be:

$$
Y_{\text{lmf}} = \text{LMF\_V6}(X)
$$

ReXAttn branch computes:

$$
Q = W_q \phi(K_{prior}), \quad K = W_k \psi(X), \quad V = W_v \psi(X)
$$

$$
A = \sigma\left(\frac{Q K^T}{\sqrt{d}}\right), \quad R = W_o(A \odot V)
$$

Final output:

$$
Y = \text{CReLU}(Y_{\text{lmf}}) + \alpha \cdot \text{CReLU}(R)
$$

where $\alpha = \text{rexattn\_scale}$.

Key properties:
- Output shape of $R$ is strictly aligned with $Y_{\text{lmf}}$: `[B, out_ch, P, R]`.
- No backend API change for later stages.

---

## 4. Implementation Summary

Implemented module:
- `Learnable_Match_FilterV6_ReXAttn`

Main points in implementation:
- Inherits the same kernel/bias initialization logic as V6.
- Keeps prior buffers for curriculum compatibility when `residual_mode=False`.
- Adds projections:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- Applies a lightweight sigmoid-gated interaction instead of heavy global softmax attention.
- Residual strength controlled by `rexattn_scale`.

Shape safety:
- Added shape check for channel consistency in ReXAttn branch.
- Reworked query construction to avoid mixing kernel-tap length with pulse length.
- Ensured residual branch output shape always matches the main LMF output.

---

## 5. Integration Into EchMo

`lmf_impl` switch now includes:
- `v5`
- `v6`
- `v6_rexattn`
- `v7`

Serial chain:
- `v6_rexattn` is supported.

Parallel chain:
- `v6_rexattn` is intentionally blocked for now with explicit error.
- Reason: this version is designed and validated only in serial chain path.

Curriculum compatibility:
- `Learnable_Match_FilterV6_ReXAttn` is included in curriculum module discovery in both:
  - `svrss/learners/model.py`
  - `svrss/learners/model_improved.py`

---

## 6. Runtime Notes

1. TensorBoard graph export may still fail for complex/tracing-heavy models on some runs.
2. This does not block training itself.
3. Visualization utility has been updated to print concrete exception details to aid debugging.

---

## 7. Suggested Ablation Plan

Recommended comparisons under identical training settings:

1. `lmf_impl=v6`
2. `lmf_impl=v6_rexattn` with `rexattn_scale=0.05`
3. `lmf_impl=v6_rexattn` with `rexattn_scale=0.1`
4. `lmf_impl=v7`

Track at least:
- Dice / mIoU
- Step latency
- Peak memory
- Convergence stability

Expected trend:
- `v6_rexattn` may improve flexibility and robustness in difficult interference patterns.
- Throughput may be slightly lower than pure V6 due to extra projections and gating operations.

---

## 8. Current Limitations

1. ReXAttn hyperparameters (`rexattn_dim`, `rexattn_scale`) are currently module defaults, not yet exposed in config.
2. Parallel-chain ReXAttn is not implemented.
3. The branch is intentionally lightweight; this is a practical engineering compromise, not a full transformer replacement.

---

## 9. Next Iteration Candidates

1. Expose ReXAttn hyperparameters in JSON configs.
2. Add optional per-layer schedule for `rexattn_scale`.
3. Add controlled benchmark section in `EchMo_Backbone_Compute_Report.md`.
4. Explore parallel-chain compatible ReXAttn variant.

---

## 10. Conclusion

LMF_V6_ReXAttn is a plug-in enhancement that preserves prior-guided convolution as the stable core and adds a lightweight adaptive residual capacity. It is designed for safe experimentation with minimal disruption to existing EchMo training/inference pipelines.
