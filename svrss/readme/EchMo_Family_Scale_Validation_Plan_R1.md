# EchMo Family Scale Validation Plan R1

## 1. Objective

This R1 plan validates scaling behavior of the EchMo family under a controlled setup.

Goals:

1. Establish a reproducible backbone baseline on LSFJamming.
2. Quantify contributions of key backbone knobs (`lmf_impl`, `lmf_chain_impl`, `mldc_impl`, `gdf_impl`, `pack_feat`).
3. Compare family variants (`deep`, `wide`, `moe`, `memo`) against the same baseline.


## 2. Fixed Protocol

Keep these fixed for fair comparison unless explicitly varied in matrix:

- Dataset: `LSFJamming`
- Task: `Segmentation`
- Resolution: `w_size=128`, `h_size=1024`
- Training backend: `trainer_impl=improved`
- Scheduler: `cos`, `Tmax=300`
- Loss: `lsf_wce`
- Seeds: `torch_seed=42`, `numpy_seed=42`
- Validation cadence: `val_step=1000`


## 3. Runtime Recording Rules

For each run, record with template file:

- `svrss/readme/EchMo_Family_Scale_Experiment_Template.md`

Recommended naming:

- Record file: `svrss/readme/records/r1_<id>_<model>_<tag>.md`
- Example: `r1_b03_echmo_v7_parallel.md`


## 4. Stop / Rerun Criteria

Stop and mark as failed if any condition occurs:

1. NaN or divergence in train loss.
2. Checkpoint loading mismatch when resume is intended.
3. Repeated OOM after one conservative batch-size reduction.

Rerun once only when:

1. Infrastructure issue (node reset, I/O error, network interruption).
2. Non-deterministic crash without config/code change.


## 5. Phase 0: Sanity and Baseline Lock

| ID | Core intent | Resume path | Checkpoint handling | Config |
|---|---|---|---|---|
| R1-S0 | Verify engineering path is healthy under your current production-like setup | Yes (must exercise resume logic) | Keep a valid `checkpoint_path` and confirm resume really takes effect | `svrss/config_files/echmo_itc_lmfv6-rexattn.json` |
| R1-S1 | Lock a clean, comparable baseline for all later ablations | No (must start from random init) | Force empty/disabled `checkpoint_path` to avoid carry-over effects | `svrss/config_files/echmo_itc_scale.json` |

Quick decision rule:

1. If you want to test "can current pipeline correctly continue training from saved state", run R1-S0.
2. If you want a "fair from-scratch baseline" for ranking later variants, run R1-S1.


## 6. Phase A: Backbone Knob Ablation

Backbone only (`model=echmo`), one axis changed each run.

| ID | Axis | Variant | Config Base / Change |
|---|---|---|---|
| R1-B0 | Reference | V6 serial + MLDC serial + GDF conv | `echmo_itc_scale.json` |
| R1-B1 | GDF backend | conv -> einsum | `echmo_itc_gdf_einsum.json` |
| R1-B2 | LMF impl | v6 -> v7 (serial chain) | `echmo_itc_scale_lmf-v7.json` |
| R1-B3 | LMF chain execution | serial -> parallel (v7) | `echmo_itc_scale_lmf_chain_parallel.json` |
| R1-B4 | LMF impl | v6_rexattn serial | `echmo_itc_lmfv6-rexattn.json` (set fresh run if needed) |
| R1-B5 | pack_feat | `apiq` -> `ap` | clone `echmo_itc_scale.json`, only change `pack_feat` |
| R1-B6 | pack_feat | `apiq` -> `iq` | clone `echmo_itc_scale.json`, only change `pack_feat` |
| R1-B7 | pack_feat | `apiq` -> `amp` | clone `echmo_itc_scale.json`, only change `pack_feat` |

Notes for `pack_feat` runs:

1. Treat as architecture-change runs.
2. Do not resume from `apiq` checkpoints.


## 7. Phase B: Family Variant Comparison

Use same backbone protocol as `R1-B0` unless variant requires own keys.

| ID | Variant | Config |
|---|---|---|
| R1-V0 | Deep | `svrss/config_files/echmo_deep_itc_scale.json` |
| R1-V1 | Wide | `svrss/config_files/echmo_wide_itc_scale.json` |
| R1-V2 | MoE | `svrss/config_files/echmo_4moe_itc_scale.json` |
| R1-V3 | Memo | `svrss/config_files/echmo_memo_itc_scale.json` |


## 8. Optional Phase C: Capacity Sweep (If Time Budget Allows)

Suggested single-axis sweeps from best-performing branch in Phase A/B:

1. LMF capacity: `lmf_chains` and `lmf_channels`
2. MLDC capacity: `ldc_channels`
3. Deep stage2 scales: `deep_stage2_*_scale`
4. Wide rank/groups: `wide_rank`, `wide_groups`
5. MoE routing: `moe_num_experts`, `moe_topk`
6. Memo memory strength: `memo_beta`, `memo_gamma`


## 9. Command Template

Single process:

```bash
python svrss/run_echmo.py --cfg svrss/config_files/<cfg_name>.json
```

DDP:

```bash
bash svrss/run_ddp.sh
```


## 10. Metrics to Report (Mandatory)

For each run, report at least:

1. Val Dice, Val mIoU
2. Test Dice, Test mIoU
3. Step latency (ms/iter)
4. Peak GPU memory
5. Params / FLOPs (if available)


## 11. R1 Deliverables

At end of R1, produce:

1. Ranked table by quality (Dice/mIoU).
2. Ranked table by efficiency (latency/memory).
3. Pareto shortlist (quality-efficiency trade-off).
4. Recommended R2 direction (1-2 primary branches).


## 12. R1 Run Record Sheet (Fill Here Directly)

Use this section as your one-file run tracker.

### 12.1 Master Table (One Row per Run)

| Run ID | Status (todo/running/done/failed) | Config | Resume (Y/N) | Start Time | End Time | Val Dice | Val mIoU | Test Dice | Test mIoU | Latency (ms/iter) | Peak GPU Mem (MB) | Params | FLOPs | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R1-S0 |  | `echmo_itc_lmfv6-rexattn.json` | Y |  |  |  |  |  |  |  |  |  |  |  |
| R1-S1 |  | `echmo_itc_scale.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-B0 |  | `echmo_itc_scale.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-B1 |  | `echmo_itc_gdf_einsum.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-B2 |  | `echmo_itc_scale_lmf-v7.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-B3 |  | `echmo_itc_scale_lmf_chain_parallel.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-B4 |  | `echmo_itc_lmfv6-rexattn.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-B5 |  | `echmo_itc_scale_pack_ap.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-B6 |  | `echmo_itc_scale_pack_iq.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-B7 |  | `echmo_itc_scale_pack_amp.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-V0 |  | `echmo_deep_itc_scale.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-V1 |  | `echmo_wide_itc_scale.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-V2 |  | `echmo_4moe_itc_scale.json` | N |  |  |  |  |  |  |  |  |  |  |  |
| R1-V3 |  | `echmo_memo_itc_scale.json` | N |  |  |  |  |  |  |  |  |  |  |  |

### 12.2 Single-Run Detail Card (Copy and Fill per Run)

```markdown
#### Run: R1-XX

| Field | Value |
|---|---|
| Status | |
| Config | |
| Resume (Y/N) | |
| checkpoint_path | |
| Start Time | |
| End Time | |
| Duration | |

| Metric | Value |
|---|---:|
| Val Dice | |
| Val mIoU | |
| Test Dice | |
| Test mIoU | |
| runtime_metrics.avg_step_latency_ms | |
| runtime_metrics.peak_gpu_memory_mb | |
| model_profile.trainable_params | |
| model_profile.flops_per_sample | |
| model_profile.flops_status | |

| Decision | Value |
|---|---|
| Keep for R2 | |
| Promote as baseline | |
| Need rerun | |
| Notes | |
```

### 12.3 Final Ranking Table (Fill After All Runs)

| Rank | Run ID | Test Dice | Test mIoU | Latency (ms/iter) | Peak GPU Mem (MB) | Params | FLOPs | Pareto (Y/N) | Comment |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 |  |  |  |  |  |  |  |  |  |
| 2 |  |  |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |  |  |
| 5 |  |  |  |  |  |  |  |  |  |
