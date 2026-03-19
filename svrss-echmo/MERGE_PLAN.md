# SVRSS -> SVRSS-ECHMO Merge Plan (2026-03-19)

## Goal
Use `svrss-echmo` as target, selectively merge useful updates from `svrss`, and keep echmo-specific research variants.

## Inventory Snapshot
- only in `svrss`: 20 files
- only in `svrss-echmo`: 36 files
- same path but different: 30 files
- same: 74 files

## Merge Strategy
- Keep target baseline: `svrss-echmo`
- Merge in batches, verify each batch before next
- Avoid one-shot overwrite for `models/*` and `learners/*`

## Batch Plan
1. Batch A (low risk): add missing entry scripts/config from `svrss` to `svrss-echmo` [DONE]
2. Batch B (low-medium): sync extension fix only; keep `svrss-echmo` loaders/utils dataset extensions [DONE]
3. Batch C (medium-high): merge `learners/*` diffs with manual review
4. Batch D (high): merge `models/*` core files manually (`echmo_debug.py`, `rechmo_debug.py`, `pkcin4any.py`, `ecoss.py`)
5. Batch E: run smoke tests and training startup verification

## Batch A Candidate Files (copy-add only)
- `config_files/rechmo.json`
- `infer_dp.py`
- `infer_scnr.py`
- `run_ddp.sh`
- `run_dp.py`
- `run_echmo.py`
- `run_ecoss.py`
- `train.sh`

## Batch A Result
- Added all 8 planned files into `svrss-echmo` successfully.

## Batch B Decision
- `extension/prf_cuda.cu`, `extension/prf_cuda_ops.cpp`: take from `svrss` (contains CUDA build compatibility fix).
- `loaders/*` and `utils/*`: keep `svrss-echmo` base for now (contains LSFJamming/KT7 support and extra reconstruction metrics paths).

## Progress Update
- Batch A completed.
- Batch B completed (`extension/*` synced).
- Batch D partial completed:
	- merged trainable-kernel + DDP-safe logic into `models/echmo_debug.py`
	- merged trainable-kernel + DDP-safe logic into `models/rechmo_debug.py`

## Remaining Common-Diff Files (27)
- config_files/config.ini
- config_files/deeplabv3plus.json
- config_files/echmo.json
- config_files/ecoss.json
- config_files/fcn8s.json
- config_files/pkcin4any.json
- config_files/pkcin4cwr.json
- config_files/unet.json
- infer.py
- learners/initializer.py
- learners/model.py
- learners/tester.py
- loaders/dataloaders.py
- loaders/dataset.py
- models/__init__.py
- models/echmo_debug.py
- models/echonet_debug.py
- models/ecoss.py
- models/model_utils/complex_layers.py
- models/model_utils/echo_tools.py
- models/pkcin4any.py
- run.py
- utils/functions.py
- utils/metrics.py
- utils/paths.py
- utils/set_paths.py
- utils/tensorboard_visualizer.py

## Recommended Next Step
- Keep `learners/*`, `loaders/*`, `utils/*`, and most `config_files/*` on `svrss-echmo` baseline.
- Continue manual merge for model-core files that affect training behavior: `models/pkcin4any.py`, `models/ecoss.py`, and `models/__init__.py`.

## Model-Core Decisions (current)
- `models/pkcin4any.py`: keep `svrss-echmo` version (only minor signature/whitespace diff; no functional merge needed).
- `models/ecoss.py`: keep `svrss-echmo` version (uses local path semantics and I/Q translator input design).
- `models/__init__.py`: keep `svrss-echmo` default export (`EchMo -> echmo_debug_ReCK3_STD`) to preserve current training behavior.
- Note: `models/echmo_debug.py` and `models/rechmo_debug.py` already received trainable-kernel/DDP-safe merges, but `EchMo` runtime entry still depends on `__init__.py` export choice.

## Config Decisions (current)
- Keep `svrss-echmo` versions for all current config diffs:
	- `config_files/config.ini`
	- `config_files/deeplabv3plus.json`
	- `config_files/echmo.json`
	- `config_files/ecoss.json`
	- `config_files/fcn8s.json`
	- `config_files/pkcin4any.json`
	- `config_files/pkcin4cwr.json`
	- `config_files/unet.json`
- Reason: these files include `LSFJamming`, `ReCK_MODE`, `pin_mem`, dataset-specific class counts and device/runtime tuning.

## Batch C Decisions (current)
- Keep `svrss-echmo` versions for runner/learner/utils files:
	- `infer.py`
	- `run.py`
	- `learners/initializer.py`
	- `learners/model.py`
	- `learners/tester.py`
	- `utils/functions.py`
	- `utils/metrics.py`
	- `utils/paths.py`
	- `utils/set_paths.py`
	- `utils/tensorboard_visualizer.py`
- Keep `svrss-echmo` versions for model utils:
	- `models/model_utils/complex_layers.py`
	- `models/model_utils/echo_tools.py`
- Reasons:
	- `svrss-echmo` contains additional dataset and experiment support (`LSFJamming`, `KT7data`, reconstruction metrics, VAE-related loss hooks).
	- `svrss-echmo/run.py` includes newer training controls (`--resume`, distributed rank handling, global-dataset trainer path).
	- `models/model_utils/complex_layers.py` in `svrss-echmo` adds `ComplexLinear`, needed by newer model variants.

## ECHONet Decision (current)
- `models/echonet_debug.py`: keep `svrss-echmo`.
- Diff is path-default/demo-only (`root_path` and visualization coef path), no core training-logic difference from `svrss`.

## Deferred Files (need later decision)
- `learners/model_F.py`, `learners/tester_F.py`, `loaders/dataloaders_F.py`
- `models/echmo_debug (copy).py`, `models/echonet_debug_cw.py`
- dataset stats under `config_files/carrada_*`
- data artifacts: `net_pd_18dB.npy`, `net_pf_18dB.npy`
