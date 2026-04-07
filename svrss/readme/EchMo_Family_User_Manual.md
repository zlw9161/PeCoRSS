# EchMo Family User Manual

## 1. Scope

This manual covers the current EchMo family in `svrss`, including:

- EchMo backbone (`echmo`)
- EchMoDeep (`echmo_deep`)
- EchMoWide (`echmo_wide`)
- EchMo4MoE (`echmo_4moe`)
- EchMoMemo (`echmo_memo`)

Primary code entry points:

- Model entry: `svrss/run_echmo.py`
- Backbone: `svrss/models/echmo.py`
- Variants:
  - `svrss/models/echmo_deep.py`
  - `svrss/models/echmo_wide.py`
  - `svrss/models/echmo_4moe.py`
  - `svrss/models/echmo_memo.py`


## 2. Quick Start

### 2.1 Training command

Single process:

python svrss/run_echmo.py --cfg svrss/config_files/echmo_itc_lmfv6-rexattn.json

DDP (example, adapt to your environment):

bash svrss/run_ddp.sh


### 2.2 Model selection from config

Set `model` in JSON config:

- `echmo` -> `EchMo`
- `echmo_deep` -> `EchMoDeep`
- `echmo_wide` -> `EchMoWide`
- `echmo_4moe` -> `EchMo4MoE`
- `echmo_memo` -> `EchMoMemo`

Trainer backend selection:

- `trainer_impl: "improved"` (recommended)
- `trainer_impl: "legacy"`


## 3. Backbone Architecture

EchMo backbone path:

1. Input normalization (`instance` or `global`)
2. LMF stage (`LMF_Chain` or `LMF_Chain_Parallel`)
3. MLDC stage (`Multi_Order_LDC` or `MLDC_Parallel`)
4. GDF stage (`Group_Purturble_DF`, `conv` or `einsum`)
5. Feature packing (`pack_feat`)
6. RD translator (`PKCIn4ANY_cuda`)

Backbone internals are in `svrss/models/echmo.py`.


## 4. Shared Core Configs

Common required fields for EchMo family:

- Data and task:
  - `dataset`, `task_name`, `w_size`, `h_size`, `nb_classes`, `nb_input_channels`
- LMF:
  - `lmf_chains`, `lmf_channels`, `lmf_layers`, `lmf_coefs`
  - `lmf_impl`: `v5` | `v6` | `v6_rexattn` | `v7`
  - `lmf_chain_impl`: `serial` | `parallel`
- MLDC:
  - `ldc_channels`, `ldc_orders`, `mldc_impl`: `serial` | `parallel`
- GDF:
  - `nb_doppler` (mapped to `df_channels`), `gdf_impl`: `conv` | `einsum`
- Global:
  - `activation`, `bias`, `norm_type`


## 5. pack_feat (Feature Packing Mode)

Supported in `EchMo` via config key `pack_feat`:

- `apiq`: 4 channels, amp + phase + real + imag
- `ap`: 2 channels, amp + phase
- `iq`: 2 channels, real + imag
- `amp`: 1 channel, amp only

Default:

- `pack_feat: "apiq"`

Important:

- `pack_feat` changes RD translator input channels.
- If resuming from old checkpoints trained with 4-channel packing, keep `pack_feat: "apiq"` for compatibility.


## 6. Variant-Specific Configs

### 6.1 EchMoDeep

File: `svrss/models/echmo_deep.py`

Additional keys:

- `deep_stage2_enable` (bool)
- `deep_stage2_range_scale` (float)
- `deep_stage2_pulse_scale` (float)
- `deep_stage2_chain_scale` (float)
- `deep_stage2_lmf_channel_scale` (float)
- `deep_stage2_ldc_channel_scale` (float)

Behavior:

- Stage-1 full resolution backbone + Stage-2 scaled backbone.
- Stage-2 logits are upsampled and fused.


### 6.2 EchMoWide

File: `svrss/models/echmo_wide.py`

Additional keys:

- `wide_groups` (int)
- `wide_rank` (int)
- `wide_share_mldc` (bool)
- `wide_fuse_residual` (bool)

Behavior:

- Adds low-rank channel mixer and grouped MLDC.
- Requires divisibility:
  - `lmf_channels * lmf_chains` divisible by `wide_groups`
  - `ldc_channels` divisible by `wide_groups`

Current note:

- Wide path currently packs fixed APIQ channels in its own forward path.


### 6.3 EchMo4MoE

File: `svrss/models/echmo_4moe.py`

Additional keys:

- `moe_num_experts` (int)
- `moe_topk` (int)
- `moe_gate_hidden` (int)
- `moe_temperature` (float)
- `moe_dropout` (float)

Behavior:

- Multiple EchMo experts.
- Gate predicts expert weights from input statistics.
- Supports top-k sparse weighting.


### 6.4 EchMoMemo

File: `svrss/models/echmo_memo.py`

Additional keys:

- `memo_enable` (bool)
- `memo_beta` (float)
- `memo_gamma` (float)
- `memo_use_state_proj` (bool)

Behavior:

- Uses a single-frame EchMo backbone and external memory accumulation.
- Supports 4D and 5D temporal input formats documented in code.


## 7. Trainer and Resume

Recommended trainer:

- `trainer_impl: "improved"`

Resume-related keys:

- `checkpoint_path`
- `use_global_dataset`

Notes:

- Improved trainer supports robust relative checkpoint path resolution.
- If checkpoint and current model config differ in structure-sensitive options (for example `pack_feat`), strict loading can fail.


## 8. Recommended Baseline Configs

Useful templates in `svrss/config_files`:

- Backbone and core ablations:
  - `echmo_itc.json`
  - `echmo_itc_gdf_einsum.json`
  - `echmo_itc_lmfv6.json`
  - `echmo_itc_lmfv6-rexattn.json`
  - `echmo_itc_scale_lmf-v7.json`
  - `echmo_itc_scale_lmf_chain_parallel.json`
- Scaling variants:
  - `echmo_deep_itc_scale.json`
  - `echmo_wide_itc_scale.json`
  - `echmo_4moe_itc_scale.json`
  - `echmo_memo_itc_scale.json`


## 9. Scale Validation Checklist

Before running scale sweeps:

1. Fix trainer and environment:
   - Use `trainer_impl: "improved"`
   - Fix random seeds and distributed setting
2. Fix baseline core configs:
   - `lmf_impl`, `lmf_chain_impl`, `mldc_impl`, `gdf_impl`, `pack_feat`
3. Change only one scaling axis each run:
   - deep or wide or moe or memo
4. Keep resume consistency:
   - Same structural config for checkpoint resume (especially `pack_feat`)
5. Track both quality and cost:
   - Dice/mIoU
   - Step latency
   - Peak memory
   - Convergence stability


## 10. Common Failure Modes

1. Checkpoint loading fails after config switch
   - Usually structural mismatch (for example changed `pack_feat` or channels).

2. Wide model raises divisibility error
   - Adjust `wide_groups`, `lmf_channels * lmf_chains`, and `ldc_channels`.

3. Parallel LMF with unsupported backend
   - `v6_rexattn` is serial-chain only.

4. DDP profiling stall risk
   - Use existing rank synchronization logic and avoid rank0-only blocking operations without barriers.


## 11. Practical Suggestions

1. Start scaling from a stable EchMo baseline and freeze all shared knobs.
2. For first comparison round, keep `pack_feat: "apiq"` to maximize compatibility with existing checkpoints.
3. If testing packed-feature ablations (`ap`, `iq`, `amp`), treat them as architecture changes and start from matching checkpoints only.


## 12. Experiment Template

For scale-study record keeping, use:

- `svrss/readme/EchMo_Family_Scale_Experiment_Template.md`
