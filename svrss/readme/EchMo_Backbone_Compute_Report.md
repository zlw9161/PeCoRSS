# EchMo Backbone Compute Report

This report is reorganized using the requested structure:

1. Baseline EchMo compute-cost analysis (single GPU + DDP)
2. LMF acceleration report (numerical equivalence + speedup)
3. MLDC acceleration report (numerical equivalence + speedup)
4. GDF acceleration report (numerical equivalence + speedup)
5. End-to-end acceleration summary (module-ablation style)

## 0. Data Sources and Baseline Configuration

### 0.1 Profiling Artifacts Used

- Whole-step timing:
  - svrss/readme/echmo_train_stage_profile_itc_single_gpu.json
  - svrss/readme/echmo_train_stage_profile_itc.json
  - svrss/readme/echmo_train_stage_profile_itc_gdf_einsum_single_gpu.json
  - svrss/readme/echmo_train_stage_profile_itc_gdf_einsum_ddp.json
- Split backward timing (frontend vs head):
  - svrss/readme/echmo_train_split_backward_itc_single_gpu.json
  - svrss/readme/echmo_train_split_backward_itc_gdf_einsum_single_gpu.json
  - svrss/readme/echmo_train_split_backward_itc_gdf_einsum_ddp.json
- Frontend stagewise timing:
  - svrss/readme/echmo_frontend_stagewise_backward_itc_single_gpu.json
  - svrss/readme/echmo_frontend_stagewise_backward_itc_gdf_einsum_single_gpu.json
  - svrss/readme/echmo_frontend_stagewise_backward_itc_gdf_einsum_ddp.json

### 0.2 Baseline Model Configuration (Added)

Two baseline tracks are used in this report, and they answer different questions.

#### A) Baseline for module-level forward cost and Params/MAC

- Model family: EchMo
- Scale tag: EchMo-LSFJamming-C8-L2-H16-LDC128-O3-D128
- Input: B=2, C=1, P=128, R=1024 (complex)
- Hyperparameters:
  - lmf_chains=8, lmf_layers=2, lmf_channels=16
  - ldc_channels=128, ldc_orders=3
  - nb_doppler=128
- Baseline implementation assumptions:
  - LMF: serial old path (V6 baseline)
  - MLDC: serial
  - GDF: conv

This track is used for:

- Per-module forward hotspot identification
- Parameter distribution
- Theoretical MAC distribution

#### B) Baseline for real training-step timing (single GPU and DDP)

- Profiling script: profile_echmo_train_stages.py
- Config label in archived JSON: config_files/echmo_itc.json
- Runtime profile settings used in artifacts:
  - warmup: 2 iterations
  - measured: 5 iterations
  - effective batch: batch_size_per_rank=8
  - num_workers=4
- Baseline kernel path for comparison in this report:
  - GDF conv path (before the einsum rewrite)

Note: some repository config files may evolve over time; this report follows the recorded profiling artifacts as the source of truth.

## 1. Baseline EchMo Compute-Cost Analysis (Before Acceleration)

### 1.1 Single-GPU per-module forward cost (baseline)

| Stage | Time (ms/iter) | Share |
|---|---:|---:|
| normalize | 0.449 | 0.06% |
| LMF total | 297.193 | 40.98% |
| MLDC (serial) | 402.598 | 55.51% |
| GDF (conv) | 10.353 | 1.43% |
| pack | 0.245 | 0.03% |
| RDTranslator | 14.412 | 1.99% |
| total | 725.262 | 100% |

Observation: baseline forward is dominated by LMF + MLDC (about 96.5%).

### 1.2 Single-GPU per-module backward cost (baseline)

Frontend stagewise harness (localization profile):

| Stage | Forward (ms/iter) | Backward (ms/iter) | Share in frontend backward |
|---|---:|---:|---:|
| LMF | 1.732 | 18.248 | 2.78% |
| MLDC | 0.641 | 2.031 | 0.31% |
| GDF (conv) | 6.759 | 635.741 | 96.76% |
| pack | 0.284 | 1.035 | 0.16% |
| frontend_stagewise_backward_total | 9.416 | 657.055 | 100% |

Split backward (frontend vs head):

| Metric | Time (ms/iter) |
|---|---:|
| frontend_backward | 639.289 |
| head_backward | 156.647 |
| split_backward_total | 795.937 |

Observation: baseline training bottleneck is backward, and the dominant hotspot is GDF backward.

### 1.3 Baseline parameter count and theoretical MAC

Parameter distribution:

| Module | Params | Share |
|---|---:|---:|
| LMFChain | 1,745,664 | 11.14% |
| MLDC | 147,840 | 0.94% |
| GDF | 12,583,680 | 80.28% |
| RDTranslator | 1,198,023 | 7.64% |
| TOTAL | 15,675,207 | 100% |

Theoretical MAC for custom backbone (LMF/MLDC/GDF):

| Module | MAC | Share |
|---|---:|---:|
| LMF | 914,962,251,776 | 87.69% |
| MLDC | 77,309,411,328 | 7.41% |
| GDF | 51,539,607,552 | 4.94% |

Note: MAC share and measured runtime share are not expected to match exactly due to kernel implementation and memory behavior.

### 1.4 Baseline DDP step cost

| Stage | Time (ms/iter) | Share |
|---|---:|---:|
| data | 21.410 | 2.40% |
| forward | 67.464 | 7.57% |
| loss | 0.585 | 0.07% |
| backward | 799.027 | 89.64% |
| optimizer | 2.166 | 0.24% |
| ddp_sync (est.) | 1.353 | 0.15% |
| step_total | 891.330 | 100% |

Observation: DDP baseline is still backward-dominated; communication is not the primary bottleneck.

## 2. LMF Acceleration Report

### 2.1 Numerical equivalence

Reference output: LMF V6 serial.

| Comparison | max abs err | mean abs err | relative mean abs err |
|---|---:|---:|---:|
| V6 serial vs V7 serial | 2.139e-4 | 1.390e-5 | 1.026e-6 |
| V6 serial vs Chain-Parallel | 3.158e-2 | 3.955e-3 | 2.926e-4 |

Interpretation: V7 is very close to V6; Chain-Parallel introduces small numerical drift but remains practically usable.

### 2.2 Speedup

| Mode | LMF-only (ms/iter) | End-to-end (ms/iter) |
|---|---:|---:|
| V6 serial | 294.995 | 336.194 |
| V7 serial | 78.589 | 115.732 |
| Chain-Parallel | 48.228 | 85.706 |

Speedups:

- LMF-only: V7/V6 = 3.75x, Chain-Parallel/V6 = 6.12x
- End-to-end: V7/V6 = 2.90x, Chain-Parallel/V6 = 3.92x

## 3. MLDC Acceleration Report

### 3.1 Numerical equivalence

| Comparison | max abs err | mean abs err | relative mean abs err |
|---|---:|---:|---:|
| MLDC serial vs MLDC_Parallel | 3.052e-5 | 2.113e-7 | 9.440e-9 |

Interpretation: MLDC_Parallel is numerically equivalent for practical engineering use.

### 3.2 Speedup

| Stage | Serial (ms/iter) | Parallel (ms/iter) | Speedup |
|---|---:|---:|---:|
| MLDC | 402.315 | 8.744 | 46.01x |
| total forward | 725.349 | 333.862 | 2.17x |

Interpretation: MLDC parallelization removes the largest baseline forward hotspot.

## 4. GDF Acceleration Report

### 4.1 Numerical equivalence

| Comparison | max abs err | mean abs err | relative mean abs err |
|---|---:|---:|---:|
| GDF conv vs GDF einsum | 0.0 | 0.0 | 0.0 |

Interpretation: exact numerical match was observed under the tested setup.

### 4.2 Single-GPU speedup

Whole step:

| Metric | conv (ms) | einsum (ms) | Speedup |
|---|---:|---:|---:|
| forward | 66.265 | 59.918 | 1.11x |
| backward | 794.578 | 166.305 | 4.78x |
| step_total | 888.581 | 251.472 | 3.53x |

Split backward:

| Metric | conv (ms) | einsum (ms) | Speedup |
|---|---:|---:|---:|
| frontend_backward | 639.289 | 11.803 | 54.17x |
| head_backward | 156.647 | 155.860 | 1.01x |
| split_backward_total | 795.937 | 167.663 | 4.75x |

Frontend stagewise backward:

| Stage | conv (ms) | einsum (ms) | Speedup |
|---|---:|---:|---:|
| LMF backward | 18.248 | 18.374 | 0.99x |
| MLDC backward | 2.031 | 1.797 | 1.13x |
| GDF backward | 635.741 | 7.870 | 80.78x |
| pack backward | 1.035 | 1.127 | 0.92x |

### 4.3 DDP speedup

Whole step:

| Metric | conv (ms) | einsum (ms) | Speedup |
|---|---:|---:|---:|
| data | 21.410 | 22.773 | 0.94x |
| forward | 67.464 | 61.197 | 1.10x |
| loss | 0.585 | 0.557 | 1.05x |
| backward | 799.027 | 168.749 | 4.73x |
| optimizer | 2.166 | 2.072 | 1.05x |
| ddp_sync (est.) | 1.353 | 0.738 | 1.83x |
| step_total | 891.330 | 256.110 | 3.48x |

DDP split-backward (einsum run):

| Metric | Time (ms) |
|---|---:|
| frontend_backward | 11.855 |
| head_backward | 155.020 |
| split_backward_total | 166.875 |

Interpretation: the training speedup is overwhelmingly caused by removing the GDF backward hotspot.

## 5. End-to-End Acceleration Summary (Module Ablation View)

### 5.1 Forward-side ablation summary

| Variant | Key change | End-to-end forward (ms/iter) | Relative speedup |
|---|---|---:|---:|
| Baseline | LMF V6 + MLDC serial + GDF conv | 725.349 | 1.00x |
| + MLDC parallel | MLDC serial -> parallel | 333.862 | 2.17x |
| + LMF V7 | switch LMF to V7 on top of MLDC parallel | 115.732 | 6.27x |
| + LMF Chain-Parallel | parallel chain execution on top of MLDC parallel | 85.706 | 8.46x |

Conclusion: forward acceleration gain comes first from MLDC parallelization, then from LMF implementation upgrades.

### 5.2 Training-step summary (single GPU and DDP)

| Variant | single-GPU step_total (ms) | DDP step_total (ms) | Note |
|---|---:|---:|---|
| GDF conv baseline | 888.581 | 891.330 | backward dominated |
| GDF einsum | 251.472 | 256.110 | backward strongly reduced |
| Speedup | 3.53x | 3.48x | consistent trend |

Conclusion: for training throughput, GDF conv -> einsum is the highest-impact optimization.

### 5.3 Recommended production path

1. Keep MLDC_Parallel and LMF V7/Chain-Parallel for forward efficiency.
2. Use GDF einsum path for throughput-oriented training.
3. If full symmetry is required in documentation, add one extra DDP baseline run with explicit stagewise module-split outputs before acceleration.

## 6. Reconciliation Note (Why Recent GDF Results Differ)

This note explains why the recent report (`EchMo_Computation_Profile_and_Acceleration.md`) did not show the same large GDF-einsum gain as this report.

### 6.1 Strict A/B Re-run (same script, same protocol, only `gdf_impl` changed)

Re-run date: 2026-04-16

Protocol:

- script: `profile_echmo_train_stages.py`
- warmup/measured: 2 / 5
- runtime batch per rank: 8
- num_workers: 4
- single GPU
- compared configs differ only in `gdf_impl`:
  - conv: `config_files/echmo_itc_gdf_conv_ab.json`
  - einsum: `config_files/echmo_itc.json`

Outputs:

- `readme/ab_gdf_conv_step.json`, `readme/ab_gdf_einsum_step.json`
- `readme/ab_gdf_conv_split.json`, `readme/ab_gdf_einsum_split.json`
- `readme/ab_gdf_conv_stagewise.json`, `readme/ab_gdf_einsum_stagewise.json`

Observed speedups (einsum over conv):

- whole step:
  - forward: 1.10x
  - backward: 4.71x
  - step_total: 3.52x
- split backward:
  - frontend_backward: 54.11x
  - head_backward: 1.00x
  - split_backward_total: 4.66x
- frontend stagewise backward:
  - GDF backward: 80.31x
  - frontend_stagewise_backward_total: 22.45x

Conclusion from strict A/B: the large GDF-einsum acceleration is reproducible under the old profiling protocol family.

### 6.2 Why the Recent Report Shows No End-to-End Gain

The recent Step-4 experiment used `config_files/echmo_itc_scale.json`, which is not the same regime as `config_files/echmo_itc.json` used by the old artifact family.

Key differences (`echmo_itc.json` vs `echmo_itc_scale.json`):

- batch_size: 48 vs 4 (runtime override also differed in past runs)
- LMF structure: `lmf_chains=1`, `lmf_channels=1`, `lmf_layers=1` vs `8/16/2`
- LMF kernel path: `lmf_impl=v7` vs `v6`
- MLDC path: `mldc_impl=parallel` vs `serial`
- MLDC channels: `ldc_channels=1` vs `128`
- pack mode: `pack_feat=None` vs `apiq`

These differences change where the training bottleneck lives. In the recent `itc_scale` regime, GDF is not the dominant end-to-end backward hotspot, so replacing only GDF implementation cannot yield the same step-level acceleration multiplier.

### 6.3 Practical Guidance for Follow-up Research

1. Do not compare speedup multipliers across different config families (`itc` vs `itc_scale`).
2. For ablation claims, always keep a strict twin setup (same script, same runtime overrides, only one implementation switch).
3. For paper/report wording, separate conclusions by regime:
   - old `itc` regime: GDF-einsum gives large training-step acceleration.
   - current `itc_scale` regime: GDF-einsum mainly reduces GDF-local memory and does not improve end-to-end latency materially.
