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

## 10. DDP Training Iteration Stage Profile (`echmo_itc`)

To explain why the current `echmo_itc` training run still feels slow after the LMF / MLDC frontend refactors, a real training-path timing run was added with explicit stage timing around the main iteration phases.

Setup:

- Script: `profile_echmo_train_stages.py`
- Config: `config_files/echmo_itc.json`
- Device mode: `DDP`, `world_size=2`
- GPUs used for the profile run: `CUDA_VISIBLE_DEVICES=2,3`
- Warmup: `2` iters
- Measured iters: `5`
- Effective profiling batch: `batch_size_per_rank=8`
- `num_workers=4`
- Note: the config file uses `batch_size=32`, but the profiling run used per-rank batch `8` to fit the current GPU memory budget during timed DDP measurement.

Measurement definitions:

- `data`: includes nested `PeCoRSVDataset` / `DataLoader` creation, batch fetch, host-to-device copy, and `normalize()`.
- `forward`: model forward on the real training batch.
- `loss`: criterion evaluation and scalar loss aggregation.
- `backward`: normal backward on the real DDP training step; this already includes overlapped gradient synchronization.
- `optimizer`: `optimizer.step()` only.
- `ddp_sync`: estimated as `backward_with_sync - backward_no_sync` on the same batch. This is an approximate communication overhead indicator and is not additive with `backward`.

Measured average latency per iteration:

| Stage | Time (ms/iter) | Ratio of step total |
|---|---:|---:|
| data | `21.410` | `2.40%` |
| forward | `67.464` | `7.57%` |
| loss | `0.585` | `0.07%` |
| backward | `799.027` | `89.64%` |
| optimizer | `2.166` | `0.24%` |
| ddp_sync (est.) | `1.353` | `0.15%` |
| step_total | `891.330` | `100%` |

Interpretation:

1. Under the current real training path, the dominant cost is no longer frontend forward latency, but backward propagation.
2. The measured `forward` stage is only about `67.5 ms/iter`, which is consistent with the earlier conclusion that the LMF / MLDC optimizations did reduce the forward-side hotspot.
3. The estimated standalone DDP communication overhead is small (`~1.35 ms/iter`) relative to the total step time, so the current slowdown is not primarily caused by gradient synchronization.
4. `data` overhead is visible but secondary (`~21.4 ms/iter`); it is not the main reason the current run feels slow.
5. The current throughput bottleneck has shifted to backward compute through the full training graph, especially the translator head and activation-memory-heavy training path.

Practical takeaway:

- The earlier LMF / MLDC acceleration work was effective for forward latency, but it does not by itself guarantee proportional training throughput gains.
- For the present `echmo_itc` training setup, the next high-return optimization targets should be backward-path cost reduction and training precision/runtime strategy, rather than more frontend-only refactors.

## 11. Single-GPU Split Backward Profile (`echmo_itc`)

To identify which part of the training graph dominates backward time, the iteration backward was further decomposed into:

- `EchMo frontend`: normalize + LMF + MLDC + GDF + pack
- `RDTranslator head`: `PKCIn4ANY_cuda`

Setup:

- Script: `profile_echmo_train_stages.py`
- Config: `config_files/echmo_itc.json`
- Device mode: single GPU
- GPU used: `CUDA_VISIBLE_DEVICES=2`
- Warmup: `2` iters
- Measured iters: `5`
- Effective profiling batch: `batch_size=8`
- `num_workers=4`

Method:

1. Run `forward_frontend()` to produce the packed real-valued feature tensor passed into `RDTranslator`.
2. Detach that packed tensor, mark it trainable, and run `forward_head()`.
3. Measure `head_backward` by backpropagating loss only through `RDTranslator` to the packed feature tensor.
4. Reuse the packed feature gradient and measure `frontend_backward` by backpropagating only through the EchMo frontend graph.

This decomposed harness is intended to localize backward cost. Its sum is expected to align closely with the normal single-step backward time, but the staged measurement is not a strict runtime replacement for the original monolithic backward.

Reference single-GPU iteration timing under the same profile run:

| Stage | Time (ms/iter) |
|---|---:|
| forward | `66.265` |
| backward | `794.578` |
| step_total | `888.581` |

Decomposed forward/backward timing:

| Stage | Time (ms/iter) | Ratio of split backward total |
|---|---:|---:|
| frontend_forward | `9.555` | `-` |
| head_forward | `56.487` | `-` |
| head_backward | `156.647` | `19.68%` |
| frontend_backward | `639.289` | `80.32%` |
| split_backward_total | `795.937` | `100%` |

Interpretation:

1. The split backward total (`795.937 ms`) matches the measured single-step backward (`794.578 ms`) very closely, so the decomposition is internally consistent.
2. About `80.3%` of backward time is spent in the EchMo frontend rather than in `RDTranslator`.
3. `RDTranslator` is not free, but it is not the primary backward bottleneck for the current `echmo_itc` setting.
4. The main optimization target has shifted to the backward path of the custom frontend, especially the complex-valued LMF / MLDC / GDF pipeline and its activation footprint.

Practical takeaway:

- If the goal is to improve training throughput, the next pass should prioritize reducing frontend backward cost rather than focusing first on `RDTranslator` forward optimizations.

## 12. Frontend Stagewise Backward Profile (`LMF / MLDC / GDF / pack`)

After confirming that the EchMo frontend dominates backward time, the frontend itself was further decomposed into four stages:

- `LMF`
- `MLDC`
- `GDF`
- `pack` (`abs / angle / real / imag` packing)

Setup:

- Script: `profile_echmo_train_stages.py`
- Config: `config_files/echmo_itc.json`
- Device mode: single GPU
- GPU used: `CUDA_VISIBLE_DEVICES=2`
- Warmup: `2` iters
- Measured iters: `5`
- Effective profiling batch: `batch_size=8`
- `num_workers=4`

Method:

1. Run the real frontend stages in order: normalized input -> `LMF` -> `MLDC` -> `GDF` -> `pack`.
2. Start from the packed feature gradient produced by the head-side loss.
3. Re-run each stage on a detached boundary tensor and backpropagate only through that stage.
4. Record the isolated local backward cost of each stage.

Important note:

- The stagewise backward total in this harness is used for localization, not as a strict replacement for the monolithic frontend backward time.
- Because each stage is re-executed independently on detached boundaries, the summed local backward time can be higher than the single monolithic frontend backward measured in Section 11.
- The most reliable use of this table is relative hotspot ranking inside the frontend.

Measured stagewise timing:

| Stage | Forward (ms/iter) | Backward (ms/iter) | Ratio of stagewise frontend backward |
|---|---:|---:|---:|
| LMF | `1.732` | `18.248` | `2.78%` |
| MLDC | `0.641` | `2.031` | `0.31%` |
| GDF | `6.759` | `635.741` | `96.76%` |
| pack | `0.284` | `1.035` | `0.16%` |
| total (stagewise harness) | `9.416` | `657.055` | `100%` |

Cross-check against Section 11:

- Monolithic frontend backward from Section 11: `639.289 ms/iter`
- Stagewise frontend backward total here: `657.055 ms/iter`

Interpretation:

1. `GDF` is overwhelmingly the main backward hotspot inside the custom EchMo frontend.
2. `LMF` backward is measurable but comparatively small.
3. `MLDC` backward is already negligible under the current parallel implementation.
4. `pack` is effectively irrelevant as a training-time bottleneck.

Practical takeaway:

- If the goal is to improve training throughput further, the highest-return target is now the backward path of `GDF`, not `LMF`, not `MLDC`, and not the packing stage.

## 13. Why GDF Backward Was Heavy and a Practical Fix

### 13.1 Root Cause Analysis

The original `GDF` implementation expresses Doppler projection as four separate grouped `conv2d` calls over the pulse dimension:

- `RR`: `conv(real, cos)`
- `II`: `conv(imag, sin)`
- `RI`: `conv(real, sin)`
- `IR`: `conv(imag, cos)`

Then it reconstructs a complex output as:

$$
Y = RR + II - j\,RI + j\,IR
$$

This is mathematically valid, but it is a poor operator match for the actual computation pattern.

What `GDF` is really doing:

- For each range bin, project the pulse vector of length $P$ onto a bank of Doppler bases.
- This is a per-column temporal projection, not a spatial sliding-window convolution problem.

Why that matters:

1. The forward path is relatively cheap because cuDNN can execute the grouped conv forward efficiently enough for this shape.
2. The backward path is much worse because autograd must run gradient kernels for four grouped convolutions, including expensive input-gradient and weight-gradient reductions.
3. In the current shape regime, grouped convolution backward is a poor fit for what is effectively a batched projection / contraction over the pulse axis.
4. The stagewise profiling in Section 12 confirms that this mismatch shows up almost entirely inside `GDF` backward.

### 13.2 Practical Modification: `gdf_impl = "einsum"`

To address this, an alternative `GDF` path was added:

- New option: `gdf_impl: "conv" | "einsum"`
- New comparison config: `config_files/echmo_itc_gdf_einsum.json`

The `einsum` implementation keeps the same trainable parameters (`cos_kernel`, `sin_kernel`, `cos_bias`, `sin_bias`) and computes the same projection by reshaping the tensor into explicit group / channel / pulse axes and applying tensor contraction directly along the pulse dimension.

In other words, it replaces a convolution-based implementation of temporal projection with a contraction-based implementation that better matches the actual math.

### 13.3 Numerical Equivalence Check

An equivalence check was run after copying the same `cos/sin` kernels and biases from the original `conv` implementation into the new `einsum` implementation.

Result:

| Comparison | max abs err | mean abs err | relative mean abs err |
|---|---:|---:|---:|
| `GDF(conv)` vs `GDF(einsum)` | `0.0` | `0.0` | `0.0` |

Under the tested setup, the new path is exactly equivalent numerically.

### 13.4 Single-GPU Training Profile Comparison (`echmo_itc`, batch=8)

Measured under the same single-GPU profiling setup used in Sections 11 and 12.

Whole-step comparison:

| Metric | `gdf_impl=conv` | `gdf_impl=einsum` | Speedup |
|---|---:|---:|---:|
| forward | `66.265 ms` | `59.918 ms` | `1.11x` |
| backward | `794.578 ms` | `166.305 ms` | `4.78x` |
| step_total | `888.581 ms` | `251.472 ms` | `3.53x` |

Frontend / head split comparison:

| Metric | `gdf_impl=conv` | `gdf_impl=einsum` | Speedup |
|---|---:|---:|---:|
| frontend_backward | `639.289 ms` | `11.803 ms` | `54.17x` |
| head_backward | `156.647 ms` | `155.860 ms` | `1.01x` |
| split_backward_total | `795.937 ms` | `167.663 ms` | `4.75x` |

Frontend stagewise comparison:

| Metric | `gdf_impl=conv` | `gdf_impl=einsum` | Speedup |
|---|---:|---:|---:|
| LMF backward | `18.248 ms` | `18.374 ms` | `0.99x` |
| MLDC backward | `2.031 ms` | `1.797 ms` | `1.13x` |
| GDF backward | `635.741 ms` | `7.870 ms` | `80.78x` |
| pack backward | `1.035 ms` | `1.127 ms` | `0.92x` |

### 13.5 Interpretation

1. The bottleneck was not "Doppler filtering in general", but specifically the grouped-convolution formulation of `GDF` backward.
2. Re-expressing the same computation as tensor contraction almost completely removes the `GDF` backward hotspot.
3. The head-side backward cost remains nearly unchanged, which confirms that the speedup is localized to the intended module.
4. The result is not a marginal micro-optimization; it changes the overall training-step profile materially.

### 13.6 Recommended Adoption

For the current `echmo_itc`-style training setup, the practical recommendation is:

- adopt `gdf_impl = "einsum"` as the default experimental path for EchMo profiling and training,
- keep the original `conv` implementation as a fallback / reference path,
- then re-run DDP training profiling and validation to confirm that the single-GPU backward gains translate cleanly to the multi-GPU training workload.

## 14. DDP Validation for `gdf_impl = "einsum"`

To verify that the single-GPU gains also hold under distributed training, the same DDP profiling protocol was run with the einsum GDF path.

Setup:

- Script: `profile_echmo_train_stages.py`
- Baseline config: `config_files/echmo_itc.json` (`gdf_impl=conv`)
- Einsum config: `config_files/echmo_itc_gdf_einsum.json` (`gdf_impl=einsum`)
- Device mode: `DDP`, `world_size=2`
- GPUs: `CUDA_VISIBLE_DEVICES=2,3`
- Warmup: `2` iters
- Measured iters: `5`
- Effective profiling batch: `batch_size_per_rank=8`

Whole-step DDP comparison:

| Metric | `gdf_impl=conv` | `gdf_impl=einsum` | Speedup |
|---|---:|---:|---:|
| data | `21.410 ms` | `22.773 ms` | `0.94x` |
| forward | `67.464 ms` | `61.197 ms` | `1.10x` |
| loss | `0.585 ms` | `0.557 ms` | `1.05x` |
| backward | `799.027 ms` | `168.749 ms` | `4.73x` |
| optimizer | `2.166 ms` | `2.072 ms` | `1.05x` |
| ddp_sync (est.) | `1.353 ms` | `0.738 ms` | `1.83x` |
| step_total | `891.330 ms` | `256.110 ms` | `3.48x` |

DDP split-backward (einsum run):

| Metric | Time (ms/iter) |
|---|---:|
| frontend_backward | `11.855` |
| head_backward | `155.020` |
| split_backward_total | `166.875` |

DDP frontend stagewise backward (einsum run):

| Metric | Time (ms/iter) |
|---|---:|
| LMF backward | `18.403` |
| MLDC backward | `1.843` |
| GDF backward | `7.881` |
| pack backward | `1.146` |

Interpretation:

1. The GDF einsum path preserves the core speedup trend in DDP, not only in single-GPU profiling.
2. The main gain still comes from backward reduction (`~4.73x`), which directly translates to a large whole-step improvement (`~3.48x`).
3. Data path cost is largely unchanged, confirming that the acceleration is model-kernel local rather than data-loader side.
4. The DDP communication estimate remains small in both cases and does not dominate the result.

Practical conclusion:

- The DDP profile confirms that `gdf_impl=einsum` is a robust optimization for the current EchMo ITC training workload and should be preferred over the grouped-conv GDF path for throughput-oriented runs.
