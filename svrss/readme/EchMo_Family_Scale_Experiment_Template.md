# EchMo Family Scale Experiment Template

## 1. Usage

This file is a fill-in template for EchMo family scale experiments.

Recommended workflow per run:

1. Copy this template into a new record file.
2. Fill metadata before launch.
3. Paste command/config snapshot.
4. Fill metrics after run.
5. Write conclusions and next action.


## 2. Experiment Metadata

| Field | Value |
|---|---|
| Experiment ID | |
| Date | |
| Operator | |
| Branch / Commit | |
| Model Variant | `echmo` / `echmo_deep` / `echmo_wide` / `echmo_4moe` / `echmo_memo` |
| Purpose | |
| Baseline Ref ID | |


## 3. Environment Snapshot

| Field | Value |
|---|---|
| Host / Node | |
| GPU Type | |
| GPU Count | |
| CUDA Version | |
| PyTorch Version | |
| Distributed Mode | single / DP / DDP |
| World Size | |
| Batch Size (per rank) | |
| Effective Batch Size | |
| num_workers | |
| pin_mem | |


## 4. Data and Task

| Field | Value |
|---|---|
| dataset | |
| task_name | Segmentation |
| w_size | |
| h_size | |
| nb_classes | |
| nb_input_channels | |
| transformations | |
| norm_type | |


## 5. Core Model Config (Frozen Across Scale Runs)

Keep this section fixed when comparing scale variants.

| Key | Value |
|---|---|
| lmf_impl | |
| lmf_chain_impl | |
| lmf_chains | |
| lmf_channels | |
| lmf_layers | |
| lmf_coefs | |
| mldc_impl | |
| ldc_orders | |
| ldc_channels | |
| gdf_impl | |
| nb_doppler | |
| pack_feat | `apiq` / `ap` / `iq` / `amp` |
| activation | |
| bias | |


## 6. Variant-Specific Scale Knobs

### 6.1 EchMo (backbone)

| Key | Value |
|---|---|
| model | `echmo` |
| Notes | |

### 6.2 EchMoDeep

| Key | Value |
|---|---|
| model | `echmo_deep` |
| deep_stage2_enable | |
| deep_stage2_range_scale | |
| deep_stage2_pulse_scale | |
| deep_stage2_chain_scale | |
| deep_stage2_lmf_channel_scale | |
| deep_stage2_ldc_channel_scale | |

### 6.3 EchMoWide

| Key | Value |
|---|---|
| model | `echmo_wide` |
| wide_groups | |
| wide_rank | |
| wide_share_mldc | |
| wide_fuse_residual | |

### 6.4 EchMo4MoE

| Key | Value |
|---|---|
| model | `echmo_4moe` |
| moe_num_experts | |
| moe_topk | |
| moe_gate_hidden | |
| moe_temperature | |
| moe_dropout | |

### 6.5 EchMoMemo

| Key | Value |
|---|---|
| model | `echmo_memo` |
| memo_enable | |
| memo_beta | |
| memo_gamma | |
| memo_use_state_proj | |


## 7. Optimizer / Scheduler / Trainer

| Key | Value |
|---|---|
| trainer_impl | improved / legacy |
| nb_epochs | |
| lr | |
| lr_step | |
| schedular | |
| Tmax | |
| custom_loss | |
| focal_gamma | |
| checkpoint_path | |
| use_global_dataset | |
| enable_graph_vis | |


## 8. Launch Command and Config Snapshot

### 8.1 Command

```bash
python svrss/run_echmo.py --cfg svrss/config_files/<your_cfg>.json
```

### 8.2 Config Snapshot

Paste the exact JSON used in this run.

```json
{
}
```


## 9. Runtime Observations

| Item | Observation |
|---|---|
| Start Time | |
| End Time | |
| Total Duration | |
| Peak GPU Memory | |
| OOM / NaN / Divergence | |
| Resume Status | success / fail / fresh start |
| Checkpoint Load Message | |
| Warnings | |


## 10. Quantitative Results

Copy-from-results helper (from `<logs>/<dataset>/<model>/<exp>/results/<model>_final_results.json`):

- Step Latency (ms/iter): `runtime_metrics.avg_step_latency_ms`
- Peak GPU Memory (MB): `runtime_metrics.peak_gpu_memory_mb`
- Params: `model_profile.trainable_params`
- FLOPs (optional): `model_profile.flops_per_sample` (check `model_profile.flops_status`)

### 10.1 Final Metrics

| Metric | Value |
|---|---|
| Val Dice | |
| Val mIoU | |
| Val Loss | |
| Test Dice | |
| Test mIoU | |
| Test Loss | |

### 10.2 Efficiency Metrics

| Metric | Value |
|---|---|
| Step Latency (ms/iter) | |
| Throughput (samples/s) | |
| Params (M) | |
| FLOPs (G, optional) | |


## 11. Comparison Against Baseline

| Item | Baseline | Current | Delta |
|---|---:|---:|---:|
| Val Dice | | | |
| Val mIoU | | | |
| Test Dice | | | |
| Step Latency | | | |
| Peak Memory | | | |


## 12. Qualitative Notes

1. Failure modes observed:
2. Class-wise behavior changes:
3. Stability across epochs:
4. Any mismatch with expected scaling trend:


## 13. Decision

| Decision | Value |
|---|---|
| Keep for next round | yes / no |
| Promote as new baseline | yes / no |
| Needs rerun | yes / no |
| Next experiment ID | |


## 14. Next Action Plan

1. 
2. 
3. 


## 15. Minimal Checklist (Before Marking Complete)

- Config snapshot is attached.
- Command is reproducible.
- Core frozen knobs were unchanged for fair comparison.
- Resume behavior is recorded.
- Metrics include both quality and efficiency.
- Baseline delta is filled.
