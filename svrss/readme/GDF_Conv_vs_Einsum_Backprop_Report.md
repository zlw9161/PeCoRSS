# GDF Conv vs Einsum: Backpropagation Analysis and Speed Comparison

## 1. Scope

This report explains why two mathematically equivalent implementations of GDF (grouped `conv2d` vs `einsum`) can show a large backward-speed gap, and whether backward consistency is preserved when forward outputs match.

The report also consolidates measured speed tables for:

- Single GPU (`world_size=1`)
- DDP (`world_size=2`)

All measurements are from the same profiling harness:

- Script: `profile_echmo_train_stages.py`
- Dataset/config family: `echmo_itc`
- Effective profiling batch: `batch_size_per_rank=8`
- Warmup/measure: `2 / 5`

## 2. Mathematical Equivalence

Let:

- $X_r, X_i \in \mathbb{R}^{B\times G\times C\times P\times R}$ be real and imaginary inputs
- $W_c, W_s \in \mathbb{R}^{G\times D\times C\times P}$ be cosine/sine kernels
- $b_c, b_s \in \mathbb{R}^{G\times D}$ be biases

The GDF output can be written as:

$$
Y_{\mathrm{real}} = \langle X_r, W_c \rangle + \langle X_i, W_s \rangle + b_c + b_s
$$

$$
Y_{\mathrm{imag}} = \langle X_i, W_c \rangle - \langle X_r, W_s \rangle + b_c - b_s
$$

where $\langle\cdot,\cdot\rangle$ means contraction over channel/pulse axes $(C,P)$.

Both implementations compute this same mapping:

- Conv path: four grouped `conv2d` terms (`RR`, `II`, `RI`, `IR`) and recombination
- Einsum path: direct tensor contractions over $(C,P)$

So functionally:

$$
f_{\mathrm{conv}}(X,\theta) \equiv f_{\mathrm{einsum}}(X,\theta)
$$

## 3. Why Backward Speed Can Differ So Much

Even for the same function, runtime can differ because execution graphs differ.

### 3.1 Conv path (current baseline)

- Uses four grouped `conv2d` branches.
- Backward requires gradient kernels for each branch (input-grad + weight-grad paths).
- The operation shape is a full pulse projection (`k_h = P`, `stride = P`, output pulse dimension effectively 1), which is not an ideal match for general convolution backward kernels.

### 3.2 Einsum path

- Expresses the operation as direct contractions over $(C,P)$.
- Backward becomes contraction-style gradient propagation aligned with the real computation pattern.
- Avoids convolution-backward overhead specific to grouped conv kernels in this shape regime.

Result: forward difference is moderate, but backward difference is very large.

## 4. Forward and Backward Consistency

### 4.1 Forward consistency (measured)

With identical copied weights/biases and identical input:

- `max_abs_err = 0.0`
- `mean_abs_err = 0.0`
- `relative_mean_abs_err = 0.0`

### 4.2 Backward consistency (measured)

A gradient-consistency test was run on CUDA using the same model parameters and input, with loss:

$$
L = \mathbb{E}[Y_{\mathrm{real}}^2] + \mathbb{E}[Y_{\mathrm{imag}}^2]
$$

Observed differences:

| Quantity | max abs err | mean abs err | relative mean abs err |
|---|---:|---:|---:|
| output | 0.0 | 0.0 | 0.0 |
| input gradient | 0.0 | 0.0 | 0.0 |
| cos kernel gradient | 9.313e-09 | 9.354e-10 | 2.199e-07 |
| sin kernel gradient | 8.382e-09 | 9.262e-10 | 2.198e-07 |
| cos bias gradient | 0.0 | 0.0 | 0.0 |
| sin bias gradient | 0.0 | 0.0 | 0.0 |

Interpretation:

- Backward behavior is consistent.
- Tiny nonzero differences on kernel gradients are expected floating-point accumulation noise.

## 5. Single-GPU Speed Comparison (Conv vs Einsum)

Source files:

- Conv: `readme/echmo_train_stage_profile_itc_single_gpu.json`
- Einsum: `readme/echmo_train_stage_profile_itc_gdf_einsum_single_gpu.json`

| Metric (ms/iter) | Conv | Einsum | Speedup (Conv / Einsum) |
|---|---:|---:|---:|
| data | 24.178 | 21.537 | 1.12x |
| forward | 66.265 | 59.918 | 1.11x |
| loss | 0.587 | 0.550 | 1.07x |
| backward | 794.578 | 166.305 | 4.78x |
| optimizer | 2.298 | 2.598 | 0.88x |
| step_total | 888.581 | 251.472 | 3.53x |

## 6. DDP Speed Comparison (Conv vs Einsum)

Source files:

- Conv: `readme/echmo_train_stage_profile_itc.json`
- Einsum: `readme/echmo_train_stage_profile_itc_gdf_einsum_ddp.json`

| Metric (ms/iter) | Conv (DDP) | Einsum (DDP) | Speedup (Conv / Einsum) |
|---|---:|---:|---:|
| data | 21.410 | 22.773 | 0.94x |
| forward | 67.464 | 61.197 | 1.10x |
| loss | 0.585 | 0.557 | 1.05x |
| backward | 799.027 | 168.749 | 4.73x |
| optimizer | 2.166 | 2.072 | 1.05x |
| ddp_sync (est.) | 1.353 | 0.738 | 1.83x |
| step_total | 891.330 | 256.110 | 3.48x |

## 7. Frontend Backward Decomposition

### 7.1 Single GPU split-backward summary

Source files:

- Conv: `readme/echmo_train_split_backward_itc_single_gpu.json`
- Einsum: `readme/echmo_train_split_backward_itc_gdf_einsum_single_gpu.json`

| Metric (ms/iter) | Conv | Einsum | Speedup |
|---|---:|---:|---:|
| frontend_backward | 639.289 | 11.803 | 54.17x |
| head_backward | 156.647 | 155.860 | 1.01x |
| split_backward_total | 795.937 | 167.663 | 4.75x |

### 7.2 Frontend stagewise backward summary

Source files:

- Conv: `readme/echmo_frontend_stagewise_backward_itc_single_gpu.json`
- Einsum: `readme/echmo_frontend_stagewise_backward_itc_gdf_einsum_single_gpu.json`

| Stage backward (ms/iter) | Conv | Einsum | Speedup |
|---|---:|---:|---:|
| LMF | 18.248 | 18.374 | 0.99x |
| MLDC | 2.031 | 1.797 | 1.13x |
| GDF | 635.741 | 7.870 | 80.78x |
| pack | 1.035 | 1.127 | 0.92x |

## 8. Conclusion

1. Conv and einsum implementations are mathematically equivalent for GDF.
2. Forward outputs are identical in the measured check, and backward gradients are consistent up to expected floating-point noise.
3. The major speed gap comes from execution-path mismatch in convolution backward for this specific projection-style workload.
4. The speedup is robust in both single-GPU and DDP runs.
5. The dominant bottleneck removal is specifically in GDF backward; head-side backward remains nearly unchanged.

## 9. Recommended Default

For throughput-oriented EchMo ITC runs, use:

- `gdf_impl: "einsum"`

and keep the conv implementation as a fallback reference path.
