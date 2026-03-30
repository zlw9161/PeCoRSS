# EchMo Backbone Compute Report

## 1. Profiling Setup

- Model: `EchMo`
- Config: `config_files/echmo_itc_scale.json`
- Scale tag: `EchMo-LSFJamming-C8-L2-H16-LDC128-O3-D128`
- Input shape: `B=2, C=1, P=128, R=1024` (complex)
- Device: `cuda`
- Warmup: 3 iters
- Measured iters: 10
- Measurement method:
  - Stage latency: manual stage timing with CUDA synchronize
  - Params: direct parameter counting
  - Theoretical MAC: formula-based estimate for LMF/MLDC/GDF custom conv parts

### 1.1 Hyperparameter Snapshot (Model Scale Definition)

Core backbone hyperparameters used for this report:

| Group | Hyperparameter | Value |
|---|---|---|
| dataset/input | model | `echmo` |
| dataset/input | dataset | `LSFJamming` |
| dataset/input | cmplx_flag | `true` |
| dataset/input | input shape (single sample) | `1 x 128 x 1024` complex |
| LMF | lmf_chains | `8` |
| LMF | lmf_layers | `2` |
| LMF | lmf_channels | `16` |
| LMF | lmf_coefs | `tay_mf` |
| MLDC | ldc_channels | `128` |
| MLDC | ldc_orders | `3` |
| GDF | nb_doppler | `128` |
| activation | activation | `leaky_relu` |
| activation | bias | `true` |
| normalization | norm_type | `instance` |

Training/control hyperparameters (for context):

| Group | Hyperparameter | Value |
|---|---|---|
| optimization | batch_size | `12` |
| optimization | lr | `1e-4` |
| optimization | schedular | `cos` |
| optimization | Tmax | `300` |
| loss | custom_loss | `lsf_wce` |
| curriculum | lmf_v6_curriculum_enable | `true` |
| curriculum | freeze/transition ratio | `0.1 / 0.5` |
| curriculum | prior reg start/end | `0.01 / 0.0001` |
| curriculum | residual lr scale start/end | `0.3 / 1.0` |
| curriculum | residual alpha start/end | `0.1 / 1.0` |

## 2. Numerical Precision Path

- Input dtype: `torch.complex64`
- Parameter dtype:
  - LMFChain: `torch.float32`
  - MLDC: `torch.float32`
  - GDF: `torch.float32`
  - RDTranslator: `torch.float32`
- Activation dtype by stage:
  - after normalize: `torch.complex64`
  - after LMF: `torch.complex64`
  - after MLDC: `torch.complex64`
  - after GDF: `torch.complex64`
  - after pack (`|x|, angle, real, imag`): `torch.float32`
  - output logits: `torch.float32`

Conclusion: the current complex backbone computation runs in `complex64` (composed of `float32` real/imaginary parts), and the final translator head outputs in `float32`.

## 3. Stage Latency Breakdown

Measured average latency per iteration:

| Stage | Time (ms/iter) | Ratio of total |
|---|---:|---:|
| normalize | 0.449 | 0.06% |
| LMF total | 297.193 | 40.98% |
| MLDC | 402.598 | 55.51% |
| GDF | 10.353 | 1.43% |
| pack | 0.245 | 0.03% |
| RDTranslator | 14.412 | 1.99% |
| total | 725.262 | 100% |

Key observation: under the current configuration, `MLDC + LMF` dominates forward latency (about 96.5%).

## 4. Parameter Cost by Module

| Module | Params | Share |
|---|---:|---:|
| LMFChain | 1,745,664 | 11.14% |
| MLDC | 147,840 | 0.94% |
| GDF | 12,583,680 | 80.28% |
| RDTranslator | 1,198,023 | 7.64% |
| TOTAL | 15,675,207 | 100% |

Key observation: `GDF` has the largest parameter count, but `MLDC` is heavier in measured latency, indicating a strong effect from compute-memory access patterns and operator parallelism in the current implementation.

## 5. Theoretical MAC Estimate (Custom Backbone Stages)

Note: the MAC values below only cover the custom convolution backbone of LMF / MLDC / GDF, and do not include a detailed MAC breakdown for RDTranslator.

- Estimated LMF MAC total: `914,962,251,776`
- Estimated MLDC MAC total: `77,309,411,328`
- Estimated GDF MAC total: `51,539,607,552`

Corresponding ratio (within these three modules only):

| Stage | MAC | Ratio (LMF+MLDC+GDF) |
|---|---:|---:|
| LMF | 914,962,251,776 | 87.69% |
| MLDC | 77,309,411,328 | 7.41% |
| GDF | 51,539,607,552 | 4.94% |

Remark: theoretical MAC ratios are not required to match measured latency ratios, due to factors such as kernel implementation efficiency, memory bandwidth, parallelism, framework scheduling, and operator fusion.

## 6. Practical Findings

1. Precision perspective: the current backbone uses `complex64`, which is a middle-ground choice between numerical stability and throughput.  
2. Latency bottleneck: MLDC and LMF are the main hotspots, so optimizing them first yields the highest return.  
3. Parameter bottleneck: GDF contributes the largest parameter share; if model size reduction is the goal, start with GDF.  
4. If throughput is the target: prioritize checking per-order loops in MLDC and parallel efficiency on the complex-convolution path.  
5. If model size is the target: prioritize compressing GDF (grouping / low-rank / shared strategies).

## 7. Recommended Next Checks

- Keep the same input and batch, and compare consistent reports across `echmo / echmo_wide / echmo_deep / echmo_memo / echmo_4moe`.  
- Run kernel-level profiling for LMF and MLDC (`torch.profiler` + CUDA timeline) to locate exact hotspot operators.  
- Add FP16/BF16 mixed-precision comparisons to validate numerical stability and throughput gains.

## 8. Serial vs Parallel MLDC Re-Profile

Same setup (`B=2`, `P=128`, `R=1024`, `cuda`, `echmo_itc_scale`) with only `mldc_impl` changed:

### 8.1 Numerical Equivalence Check

To verify that `MLDC_Parallel` preserves the computation of serial `MLDC`, a parameter-aligned standalone module check was run before comparing end-to-end stage latency.

Setup:

- Input shape: `B=2, C_in=128, P=128, R=1024`
- Output shape: `B=2, C_out=384, P=128, R=1024`
- Device: `cuda`
- MLDC hyperparameters:
  - `ldc_orders = 3`
  - `ldc_channels = 128`
  - `activation = leaky_relu`
  - `bias = true`
- Parameter alignment strategy:
  - Copy each serial `hp_kernels[n]` into `MLDC_Parallel.hp_kernels[n]`
  - Copy each serial `hp_biases[n]` into `MLDC_Parallel.hp_biases[n]`

Reference: serial `MLDC` output.

| Comparison | max abs err | mean abs err | relative mean abs err |
|---|---:|---:|---:|
| `MLDC` vs `MLDC_Parallel` | `3.052e-5` | `2.113e-7` | `9.440e-9` |

Interpretation:

1. `MLDC_Parallel` is numerically equivalent to serial `MLDC` for practical engineering use after parameter alignment.
2. The remaining difference is at floating-point accumulation level and is negligible relative to the output magnitude.

### 8.2 End-to-End Stage Re-Profile

| Stage | Serial (ms/iter) | Parallel (ms/iter) | Delta (Parallel-Serial) |
|---|---:|---:|---:|
| normalize | 0.559 | 0.409 | -0.150 |
| LMF total | 297.205 | 299.459 | +2.254 |
| MLDC | 402.315 | 8.744 | -393.571 |
| GDF | 10.637 | 10.546 | -0.090 |
| pack | 0.257 | 0.190 | -0.067 |
| RDTranslator | 14.365 | 14.505 | +0.140 |
| total | 725.349 | 333.862 | -391.486 |

Overall throughput effect:

- `speedup_total_serial_over_parallel = 2.1726x`

Interpretation:

1. `MLDC_Parallel` almost removes MLDC as the primary runtime bottleneck in this profile.  
2. End-to-end forward latency drops from ~725ms to ~334ms per iter under this benchmark.  
3. LMF now becomes the dominant stage after MLDC optimization.

## 9. LMF Mode Equivalence and Latency (V6 / V7 / Chain-Parallel)

Test target requested for comparison:

- `LMF_Chain` with `lmf_impl=v6` (serial)
- `LMF_Chain` with `lmf_impl=v7` (serial)
- `LMF_Chain_Parallel` (parallel chain execution, `lmf_impl=v7`)

### 9.1 Equivalence Check Setup

- Same input shape: `B=2, C=1, P=128, R=1024`
- Device: `cuda`
- Same base config: `config_files/echmo_itc_scale.json`
- To avoid MLDC dominating the end-to-end runtime in this comparison, `mldc_impl` was fixed to `parallel` for all three runs.
- Parameter alignment strategy:
  - Copy `LMF_Chain(v6)` parameters to `LMF_Chain(v7)` layer-by-layer.
  - Copy `LMF_Chain(v6)` parameters to `LMF_Chain_Parallel` chain/layer slots.

### 9.2 Equivalence Metrics (LMF output)

Reference: `LMF_Chain(v6)` output.

| Comparison | max abs err | mean abs err | relative mean abs err |
|---|---:|---:|---:|
| V6 serial vs V7 serial | `2.139e-4` | `1.390e-5` | `1.026e-6` |
| V6 serial vs Chain-Parallel | `3.158e-2` | `3.955e-3` | `2.926e-4` |

Interpretation:

1. `LMF_Chain(v6)` and `LMF_Chain(v7)` are numerically almost identical after parameter alignment.
2. `LMF_Chain_Parallel` is not bitwise-identical to V6, but remains very close in relative mean error (`~2.93e-4`, i.e., ~0.029%).
3. For practical compute benchmarking, these three modes can be treated as computation-equivalent variants with small numerical drift from operator reordering/fusion.

### 9.3 Latency Comparison

Measured with warmup and averaged timing under the setup above.

| Mode | LMF-only latency (ms/iter) | End-to-end latency (ms/iter) |
|---|---:|---:|
| `LMF_Chain` + `LMF_V6` (serial) | `294.995` | `336.194` |
| `LMF_Chain` + `LMF_V7` (serial) | `78.589` | `115.732` |
| `LMF_Chain_Parallel` | `48.228` | `85.706` |

Speedup (LMF-only):

- `V7 serial / V6 serial`: `3.75x`
- `Chain-Parallel / V6 serial`: `6.12x`
- `Chain-Parallel / V7 serial`: `1.63x`

Speedup (end-to-end):

- `V7 serial / V6 serial`: `2.90x`
- `Chain-Parallel / V6 serial`: `3.92x`
- `Chain-Parallel / V7 serial`: `1.35x`
