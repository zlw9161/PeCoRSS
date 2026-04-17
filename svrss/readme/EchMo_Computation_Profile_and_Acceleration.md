# EchMo Computation Profile and Acceleration

## Baseline Profiling Experiment Setup

- Config: `config_files/echmo_itc_scale.json`
- Device: `cuda` (single GPU)
- Warmup iters: `1`
- Measured iters: `2`
- Batch size: `4`

## Basic Model Configuration

| Item | Value |
|---|---|
| model | echmo |
| dataset | LSFJamming |
| task_name | Segmentation |
| annot_type | dense |
| w_size | 128 |
| h_size | 1024 |
| nb_classes | 7 |
| nb_input_channels | 1 |
| lmf_chains | 8 |
| lmf_channels | 16 |
| lmf_layers | 2 |
| lmf_coefs | tay_mf |
| lmf_impl | v6 |
| lmf_chain_impl | serial |
| ldc_channels | 128 |
| ldc_orders | 3 |
| mldc_impl | serial |
| nb_doppler | 128 |
| gdf_impl | conv |
| pack_feat | apiq |
| activation | leaky_relu |
| bias | True |
| norm_type | instance |
| batch_size | 4 |

## Table 1. Module-wise Forward/Backward Latency

| Module | Forward Latency (ms/iter) | Backward Latency (ms/iter) |
|---|---:|---:|
| Normalize | 0.590 | 2.236 |
| LMF | 432.877 | 729.911 |
| MLDC | 770.636 | 101.250 |
| GDF | 18.842 | 45.802 |
| FeatPack | 0.205 | 1.006 |
| RDTranslator | 29.247 | 82.098 |
| Loss | 0.414 | 0.638 |
| End-to-End | 1073.637 | 761.028 |

## Table 2. Module-wise Params, Compute, DType, and GPU Memory

| Module | Params | Compute (GMAC/FLOPs) | Input DType | Compute DType | Output DType | Peak GPU Memory Forward (MB) | Peak GPU Memory Backward (MB) |
|---|---:|---|---|---|---|---:|---:|
| Normalize | 0 | N/A | torch.complex128 | torch.complex64 | torch.complex64 | 24.52 | 32.00 |
| LMF | 1745664 | 457.481 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1061.92 | 304.94 |
| MLDC | 147840 | 38.655 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 7424.00 | 3072.00 |
| GDF | 12583680 | 25.770 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1590.00 | 5316.09 |
| FeatPack | 0 | N/A | torch.complex64 | torch.complex64 | torch.float32 | 36.00 | 74.25 |
| RDTranslator | 1198023 | 326.149 GFLOPs (THOP) | torch.float32 | torch.float32 | torch.float32 | 1746.63 | 528.36 |
| Loss | 0 | N/A | torch.float32 + torch.int64 | torch.float32 | torch.float32 | 18.00 | 24.00 |
| End-to-End | 15675207 | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | torch.complex128 | torch.float32 | torch.float32 | 7940.50 | 1394.15 |

## Notes

- Latency numbers are averaged over measured iterations after warmup.
- Backward latency for Normalize/LMF/MLDC/GDF/FeatPack/RDTranslator/Loss is isolated with boundary-gradient decomposition; End-to-End backward is full-graph loss.backward().
- Compute for LMF/MLDC/GDF is analytic GMAC for the current tensor shapes; FLOPs entries are marked when THOP succeeds.
- End-to-End compute is composed as: LMF GMAC + MLDC GMAC + GDF GMAC + RDTranslator GMAC, where RDTranslator GMAC = THOP GFLOPs / 2.

## Step 2. LMF Acceleration (Appended)

- Baseline reference: `lmf_impl=v6`, `lmf_chain_impl=serial` (same as Table 1/2).
- Variant A (Packed-Conv): `lmf_impl=v7`, `lmf_chain_impl=serial`.
- Variant B (Chain-Parallel): `lmf_impl=v7`, `lmf_chain_impl=parallel`.
- Other settings unchanged from baseline (`batch_size=4`, single GPU).

### Table A1. LMF-Only Comparison (Baseline vs Acceleration Variants)

| Variant | LMF Compute | LMF Forward (ms) | LMF Backward (ms) | LMF Peak Mem FWD (MB) | LMF Peak Mem BWD (MB) | FWD Speedup vs Baseline | BWD Speedup vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (v6-serial) | 457.481 GMAC | 432.877 | 729.911 | 1061.92 | 304.94 | 1.000x | 1.000x |
| Packed-Conv (v7-serial) | 457.481 GMAC | 128.560 | 525.574 | 1167.27 | 180.27 | 3.366x | 1.389x |
| Chain-Parallel (v7-parallel) | 457.481 GMAC | 96.200 | 463.049 | 3697.06 | 1850.53 | 4.499x | 1.576x |

### Table A2. End-to-End Comparison (Baseline vs Acceleration Variants)

| Variant | End-to-End Compute | End-to-End Forward (ms) | End-to-End Backward (ms) | End-to-End Peak Mem FWD (MB) | End-to-End Peak Mem BWD (MB) | FWD Speedup vs Baseline | BWD Speedup vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (v6-serial) | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | 1073.637 | 761.028 | 7940.50 | 1394.15 | 1.000x | 1.000x |
| Packed-Conv (v7-serial) | 684.981 GMAC (composed) / 1369.961 GFLOPs (equiv.) | 752.383 | 647.108 | 7940.00 | 1394.59 | 1.427x | 1.176x |
| Chain-Parallel (v7-parallel) | 684.981 GMAC (composed) / 1369.961 GFLOPs (equiv.) | 719.276 | 588.888 | 7940.50 | 1394.58 | 1.493x | 1.292x |

### Table A3. Detailed Module Profile for Packed-Conv (v7-serial)

| Module | Params | Compute (GMAC/FLOPs) | Input DType | Compute DType | Output DType | Peak GPU Memory Forward (MB) | Peak GPU Memory Backward (MB) |
|---|---:|---|---|---|---|---:|---:|
| Normalize | 0 | N/A | torch.complex128 | torch.complex64 | torch.complex64 | 24.52 | 33.00 |
| LMF | 1745664 | 457.481 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1167.27 | 180.27 |
| MLDC | 147840 | 38.655 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 7424.00 | 3072.00 |
| GDF | 12583680 | 25.770 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1590.00 | 5316.09 |
| FeatPack | 0 | N/A | torch.complex64 | torch.complex64 | torch.float32 | 36.00 | 74.83 |
| RDTranslator | 1198023 | 326.149 GFLOPs (THOP) | torch.float32 | torch.float32 | torch.float32 | 1746.27 | 528.36 |
| Loss | 0 | N/A | torch.float32 + torch.int64 | torch.float32 | torch.float32 | 18.00 | 23.78 |
| End-to-End | 15675207 | 684.981 GMAC (composed) / 1369.961 GFLOPs (equiv.) | torch.complex128 | torch.float32 | torch.float32 | 7940.00 | 1394.59 |

### Table A4. Detailed Module Profile for Chain-Parallel (v7-parallel)

| Module | Params | Compute (GMAC/FLOPs) | Input DType | Compute DType | Output DType | Peak GPU Memory Forward (MB) | Peak GPU Memory Backward (MB) |
|---|---:|---|---|---|---|---:|---:|
| Normalize | 0 | N/A | torch.complex128 | torch.complex64 | torch.complex64 | 24.52 | 32.75 |
| LMF | 1745664 | 457.481 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 3697.06 | 1850.53 |
| MLDC | 147840 | 38.655 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 7424.00 | 3072.00 |
| GDF | 12583680 | 25.770 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1590.00 | 5316.52 |
| FeatPack | 0 | N/A | torch.complex64 | torch.complex64 | torch.float32 | 36.00 | 73.93 |
| RDTranslator | 1198023 | 326.149 GFLOPs (THOP) | torch.float32 | torch.float32 | torch.float32 | 1746.52 | 528.36 |
| Loss | 0 | N/A | torch.float32 + torch.int64 | torch.float32 | torch.float32 | 18.00 | 24.00 |
| End-to-End | 15675207 | 684.981 GMAC (composed) / 1369.961 GFLOPs (equiv.) | torch.complex128 | torch.float32 | torch.float32 | 7940.50 | 1394.58 |

### Step-2 Notes

- LMF compute remains unchanged (same algorithmic GMAC), and the measured gain comes from implementation efficiency (`v7` kernels and chain execution strategy).
- `v7-parallel` achieves the best latency speedup, but with a large LMF memory increase; this is expected from parallel chain materialization.

## Step 3. MLDC Acceleration (Appended)

- Baseline reference in this step: `mldc_impl=serial` (re-profiled under the same protocol).
- Variant: `mldc_impl=parallel`.
- Other settings unchanged from baseline (`config_files/echmo_itc_scale.json`, `batch_size=4`, single GPU).

### Table B1. MLDC-Only Comparison (Baseline vs Parallel)

| Variant | MLDC Compute | MLDC Forward (ms) | MLDC Backward (ms) | MLDC Peak Mem FWD (MB) | MLDC Peak Mem BWD (MB) | FWD Speedup vs Baseline | BWD Speedup vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (serial) | 38.655 GMAC | 736.453 | 101.112 | 7424.00 | 3072.00 | 1.000x | 1.000x |
| MLDC Parallel | 51.540 GMAC | 17.677 | 49.140 | 7436.75 | 3072.00 | 41.665x | 2.057x |

### Table B2. End-to-End Comparison (Baseline vs MLDC Parallel)

| Variant | End-to-End Compute | End-to-End Forward (ms) | End-to-End Backward (ms) | End-to-End Peak Mem FWD (MB) | End-to-End Peak Mem BWD (MB) | FWD Speedup vs Baseline | BWD Speedup vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (serial) | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | 1055.609 | 757.457 | 7940.50 | 1394.15 | 1.000x | 1.000x |
| MLDC Parallel | 697.866 GMAC (composed) / 1395.731 GFLOPs (equiv.) | 504.872 | 712.314 | 7952.75 | 1393.43 | 2.091x | 1.063x |

### Table B3. Detailed Module Profile for Baseline (mldc_impl=serial)

| Module | Params | Compute (GMAC/FLOPs) | Input DType | Compute DType | Output DType | Peak GPU Memory Forward (MB) | Peak GPU Memory Backward (MB) |
|---|---:|---|---|---|---|---:|---:|
| Normalize | 0 | N/A | torch.complex128 | torch.complex64 | torch.complex64 | 24.52 | 32.00 |
| LMF | 1745664 | 457.481 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1061.92 | 304.94 |
| MLDC | 147840 | 38.655 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 7424.00 | 3072.00 |
| GDF | 12583680 | 25.770 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1590.00 | 5316.09 |
| FeatPack | 0 | N/A | torch.complex64 | torch.complex64 | torch.float32 | 36.00 | 74.25 |
| RDTranslator | 1198023 | 326.149 GFLOPs (THOP) | torch.float32 | torch.float32 | torch.float32 | 1746.63 | 528.36 |
| Loss | 0 | N/A | torch.float32 + torch.int64 | torch.float32 | torch.float32 | 18.00 | 24.00 |
| End-to-End | 15675207 | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | torch.complex128 | torch.float32 | torch.float32 | 7940.50 | 1394.15 |

### Table B4. Detailed Module Profile for MLDC Parallel (mldc_impl=parallel)

| Module | Params | Compute (GMAC/FLOPs) | Input DType | Compute DType | Output DType | Peak GPU Memory Forward (MB) | Peak GPU Memory Backward (MB) |
|---|---:|---|---|---|---|---:|---:|
| Normalize | 0 | N/A | torch.complex128 | torch.complex64 | torch.complex64 | 24.02 | 32.22 |
| LMF | 1745664 | 457.481 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1061.92 | 305.44 |
| MLDC | 196992 | 51.540 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 7436.75 | 3072.00 |
| GDF | 12583680 | 25.770 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1590.00 | 5316.09 |
| FeatPack | 0 | N/A | torch.complex64 | torch.complex64 | torch.float32 | 36.00 | 73.50 |
| RDTranslator | 1198023 | 326.149 GFLOPs (THOP) | torch.float32 | torch.float32 | torch.float32 | 1746.27 | 528.36 |
| Loss | 0 | N/A | torch.float32 + torch.int64 | torch.float32 | torch.float32 | 18.00 | 24.00 |
| End-to-End | 15724359 | 697.866 GMAC (composed) / 1395.731 GFLOPs (equiv.) | torch.complex128 | torch.float32 | torch.float32 | 7952.75 | 1393.43 |

### Step-3 Notes

- MLDC parallel substantially reduces MLDC latency, but it increases MLDC compute/params (38.655 -> 51.540 GMAC, 147840 -> 196992 params).
- End-to-End gains are still strong in forward pass (2.091x), while backward gain is moderate (1.063x), indicating non-MLDC stages remain dominant in backward cost.

## Step 4. GDF Acceleration (Re-run with LMF/MLDC Method)

- Baseline reference in this step: `gdf_impl=conv`.
- Variant: `gdf_impl=einsum`.
- Other settings unchanged from baseline (`config_files/echmo_itc_scale.json`, `batch_size=4`, single GPU).
- Profiling method aligned with Step 2/3 (module-level forward/backward decomposition + module memory/compute summary).
- Profiling window in this re-run: warmup=`1`, measured=`2`.

### Table C1. GDF-Only Comparison (Baseline vs Einsum)

| Variant | GDF Compute | GDF Forward (ms) | GDF Backward (ms) | GDF Peak Mem FWD (MB) | GDF Peak Mem BWD (MB) | FWD Speedup vs Baseline | BWD Speedup vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (conv) | 25.770 GMAC | 18.706 | 45.378 | 1590.00 | 5316.59 | 1.000x | 1.000x |
| GDF Einsum | 25.770 GMAC | 23.103 | 43.416 | 786.00 | 1560.00 | 0.810x | 1.045x |

### Table C2. End-to-End Comparison (Baseline vs Einsum)

| Variant | End-to-End Compute | End-to-End Forward (ms) | End-to-End Backward (ms) | End-to-End Peak Mem FWD (MB) | End-to-End Peak Mem BWD (MB) | FWD Speedup vs Baseline | BWD Speedup vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (conv) | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | 1055.060 | 749.100 | 7940.50 | 1394.65 | 1.000x | 1.000x |
| GDF Einsum | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | 1063.300 | 750.602 | 7940.00 | 524.36 | 0.992x | 0.998x |

### Table C3. Detailed Module Profile for Baseline (gdf_impl=conv)

| Module | Params | Compute (GMAC/FLOPs) | Input DType | Compute DType | Output DType | Peak GPU Memory Forward (MB) | Peak GPU Memory Backward (MB) |
|---|---:|---|---|---|---|---:|---:|
| Normalize | 0 | N/A | torch.complex128 | torch.complex64 | torch.complex64 | 24.52 | 32.00 |
| LMF | 1745664 | 457.481 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1061.92 | 304.94 |
| MLDC | 147840 | 38.655 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 7424.00 | 3072.00 |
| GDF | 12583680 | 25.770 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1590.00 | 5316.59 |
| FeatPack | 0 | N/A | torch.complex64 | torch.complex64 | torch.float32 | 36.00 | 74.00 |
| RDTranslator | 1198023 | 326.149 GFLOPs (THOP) | torch.float32 | torch.float32 | torch.float32 | 1746.27 | 528.36 |
| Loss | 0 | N/A | torch.float32 + torch.int64 | torch.float32 | torch.float32 | 18.00 | 24.00 |
| End-to-End | 15675207 | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | torch.complex128 | torch.float32 | torch.float32 | 7940.50 | 1394.65 |

### Table C4. Detailed Module Profile for Einsum (gdf_impl=einsum)

| Module | Params | Compute (GMAC/FLOPs) | Input DType | Compute DType | Output DType | Peak GPU Memory Forward (MB) | Peak GPU Memory Backward (MB) |
|---|---:|---|---|---|---|---:|---:|
| Normalize | 0 | N/A | torch.complex128 | torch.complex64 | torch.complex64 | 24.52 | 32.00 |
| LMF | 1745664 | 457.481 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1061.92 | 304.94 |
| MLDC | 147840 | 38.655 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 7424.00 | 3072.00 |
| GDF | 12583680 | 25.770 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 786.00 | 1560.00 |
| FeatPack | 0 | N/A | torch.complex64 | torch.complex64 | torch.float32 | 36.00 | 73.50 |
| RDTranslator | 1198023 | 326.149 GFLOPs (THOP) | torch.float32 | torch.float32 | torch.float32 | 1746.27 | 528.36 |
| Loss | 0 | N/A | torch.float32 + torch.int64 | torch.float32 | torch.float32 | 18.00 | 24.00 |
| End-to-End | 15675207 | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | torch.complex128 | torch.float32 | torch.float32 | 7940.00 | 524.36 |

### Step-4 Notes

- Under the same profiling method as Step 2/3, `gdf_impl=einsum` still does not improve end-to-end latency in the `echmo_itc_scale` baseline.
- `gdf_backward` shows a local gain (1.045x), but `gdf_forward` is slower (0.810x), and net end-to-end improvement is not observed.
- Artifact for this re-run:
- `readme/echmo_gdf_accel_variants_profile_lmf_mldc_method.json`

### Step-4 Memory Validation (Absolute Peak Check)

To verify whether memory statistics were misleading, we re-ran conv/einsum with absolute full-step memory peaks (`max_memory_allocated` and `max_memory_reserved`) in addition to backward delta peaks.

| Variant | FWD Alloc Peak (MB) | FWD Reserved Peak (MB) | BWD Alloc Peak (MB) | BWD Reserved Peak (MB) | BWD Alloc Delta (MB) | BWD Reserved Delta (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (conv) | 11988.66 | 14038.00 | 13377.84 | 16218.00 | 1393.18 | 2180.00 |
| GDF Einsum | 13552.82 | 13874.00 | 14072.84 | 16986.00 | 524.02 | 3112.00 |

Interpretation:

- The local backward allocation delta (`BWD Alloc Delta`) is much lower for einsum, consistent with the reduced GDF-local backward memory in Table C1/C4.
- End-to-end absolute peaks (`BWD Alloc Peak`, `BWD Reserved Peak`) are not lower for einsum in this run, which indicates global peak is dominated by full-graph allocator behavior (buffer lifetime/reuse/caching), not only by the GDF-local kernel footprint.
- Therefore, this is not a simple counting bug; it is a metric-definition effect between local stage delta and full-step absolute peak.

Validation artifact:

- `readme/echmo_gdf_memory_validation_absolute.json`

## Step 5. Overall Baseline vs Improved EchMo

- Baseline config: `lmf_impl=v6`, `lmf_chain_impl=serial`, `mldc_impl=serial`, `gdf_impl=conv`.
- Improved config: `lmf_impl=v7`, `lmf_chain_impl=serial`, `mldc_impl=parallel`, `gdf_impl=einsum`.
- Profiling window: warmup=`1`, measured=`2`, batch size=`4`, single GPU.

### Table D1. End-to-End Comparison (Baseline vs Improved)

| Variant | End-to-End Compute | End-to-End Forward (ms) | End-to-End Backward (ms) | End-to-End Peak Mem FWD (MB) | End-to-End Peak Mem BWD (MB) | FWD Speedup vs Baseline | BWD Speedup vs Baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline EchMo | 684.981 GMAC (composed) / 1369.962 GFLOPs (equiv.) | 1041.676 | 746.699 | 7940.00 | 1394.54 | 1.000x | 1.000x |
| Improved EchMo | 697.865 GMAC (composed) / 1395.730 GFLOPs (equiv.) | 204.889 | 589.958 | 7952.75 | 524.61 | 5.083x | 1.266x |

### Table D2. Detailed Module Profile for Improved EchMo (`v7 + serial + parallel + einsum`)

| Module | Params | Compute (GMAC/FLOPs) | Input DType | Compute DType | Output DType | Peak GPU Memory Forward (MB) | Peak GPU Memory Backward (MB) |
|---|---:|---|---|---|---|---:|---:|
| Normalize | 0 | N/A | torch.complex128 | torch.complex64 | torch.complex64 | 24.52 | 32.72 |
| LMF | 1745664 | 457.481 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 1166.36 | 180.64 |
| MLDC | 196992 | 51.540 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 7436.75 | 3072.00 |
| GDF | 12583680 | 25.770 GMAC (analytic) | torch.complex64 | torch.float32 | torch.complex64 | 786.00 | 1559.61 |
| FeatPack | 0 | N/A | torch.complex64 | torch.complex64 | torch.float32 | 36.00 | 74.33 |
| RDTranslator | 1198023 | 326.149 GFLOPs (THOP) | torch.float32 | torch.float32 | torch.float32 | 1746.27 | 528.36 |
| Loss | 0 | N/A | torch.float32 + torch.int64 | torch.float32 | torch.float32 | 18.00 | 0.00 |
| End-to-End | 15724359 | 697.865 GMAC (composed) / 1395.730 GFLOPs (equiv.) | torch.complex128 | torch.float32 | torch.float32 | 7952.75 | 524.61 |

Artifact:

- `readme/echmo_overall_baseline_vs_improved_profile_serial_chain.json`

