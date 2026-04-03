# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [1.0.9] - 2026-04-03

### Added

- Added `Learnable_Match_FilterV6_ReXAttn` in `svrss/models/echmo_debug.py`:
  - prior-guided LMF-V6 main path
  - lightweight cross-attention residual enhancement branch
- Added serial LMF backend option:
  - `lmf_impl: "v6_rexattn"`
- Added dedicated technical note:
  - `svrss/readme/LMF_V6_ReXAttn_Modification_Report.md`
- Added technical document index section in `README.md` for quick navigation to `svrss/readme/` notes.

### Changed

- Standardized naming in active `svrss` code and docs:
  - `Leanable_Match_Filter*` -> `Learnable_Match_Filter*`
- Updated curriculum module discovery to include the new ReXAttn class in:
  - `svrss/learners/model.py`
  - `svrss/learners/model_improved.py`

### Fixed

- Fixed ReXAttn residual branch shape alignment to ensure output matches the LMF-V6 main path shape (`[B, out_ch, P, R]`).
- Improved TensorBoard graph visualization diagnostics in:
  - `svrss/utils/tensorboard_visualizer.py`
  by printing explicit exception type/message while keeping visualization failures non-blocking.

### Version

- Project version bumped to `1.0.9` in:
  - `README.md`
  - `setup.py`
  - `CHANGELOG.md`

## [1.0.8] - 2026-04-02

### Added

- Added merged trainer implementation:
  - `svrss/learners/model_improved.py`
- Added trainer runtime switch in EchMo entrypoint:
  - `trainer_impl: "legacy"` (default)
  - `trainer_impl: "improved"`
- Added `lmf_impl: "v5"` compatibility in EchMo LMF build paths for ablation comparisons against `v6/v7`.

### Changed

- Updated `svrss/run_echmo.py` to select trainer backend by config and to pass improved-trainer options (`checkpoint_path`, `use_global_dataset`) when enabled.
- Updated `svrss/config_files/echmo_itc.json` to use improved trainer for current ITC experiments.
- Updated `svrss/learners/model_improved.py` by removing stale commented-out code blocks to improve maintainability.

### Fixed

- Improved trace/export robustness by replacing trace-sensitive Python shape checks with `torch._assert` in EchMo active paths.
- Avoided noisy startup tracing warnings by making TensorBoard graph export resilient to `TracerWarning` in `svrss/utils/tensorboard_visualizer.py`.

### Version

- Project version bumped to `1.0.8` in:
  - `README.md`
  - `setup.py`
  - `CHANGELOG.md`

## [1.0.7] - 2026-03-31

### Added

- Added configurable GDF backend switch (`gdf_impl`) in EchMo runtime path with two implementations:
  - `conv` (original grouped-conv backend)
  - `einsum` (tensor-contraction backend)
- Added standalone training-stage profiling utility:
  - `svrss/profile_echmo_train_stages.py`
- Added comprehensive profiling result artifacts in `svrss/readme/`:
  - train-stage summaries (`data/forward/loss/backward/optimizer/ddp_sync`)
  - split-backward summaries (frontend vs head)
  - frontend stagewise backward summaries (`LMF/MLDC/GDF/pack`)
- Added standalone technical report:
  - `svrss/readme/GDF_Conv_vs_Einsum_Backprop_Report.md`
- Added new config for explicit einsum-path experiments:
  - `svrss/config_files/echmo_itc_gdf_einsum.json`

### Changed

- Updated `svrss/config_files/echmo_itc.json` to use `gdf_impl: "einsum"` by default.
- Extended `svrss/readme/EchMo_Backbone_Compute_Report.md` with:
  - DDP-stage profiling results
  - single-GPU split-backward analysis
  - frontend stagewise backward analysis
  - root-cause analysis and DDP validation for the GDF einsum path
- Plumbed `gdf_impl` from config into runtime and profiling builders:
  - `svrss/run_echmo.py`
  - `svrss/profile_echmo_train_stages.py`

### Benchmarks

- DDP (`world_size=2`, batch_per_rank=8) step latency improved from `891.330 ms/iter` to `256.110 ms/iter` (`3.48x`) when switching GDF backend from `conv` to `einsum`.
- DDP backward latency improved from `799.027 ms/iter` to `168.749 ms/iter` (`4.73x`) under the same setup.
- Frontend stagewise profiling showed GDF backward reduced from `635.741 ms/iter` to `7.881 ms/iter` (`80.78x`) in the profiled ITC setting.

### Version

- Project version bumped to `1.0.7` in:
  - `README.md`
  - `setup.py`
  - `CHANGELOG.md`

## [1.0.6] - 2026-03-30

### Added

- Added `Learnable_Match_FilterV7` in `svrss/models/echmo_debug.py` as a latency-optimized independent LMF implementation.
- Added chain-parallel LMF execution path:
  - `LMF_Layer_Parallel`
  - `LMF_Chain_Parallel`
- Added LMF backend and chain-execution switches in runtime configs:
  - `lmf_impl` (`v6` / `v7`)
  - `lmf_chain_impl` (`serial` / `parallel`)
- Added new configs for V7 and chain-parallel experiments in `svrss/config_files/`.
- Added and organized EchMo scale-scheme configs in `svrss/config_files/`, including:
  - `echmo_itc_scale.json`
  - `echmo_wide_itc_scale.json`
  - `echmo_deep_itc_scale.json`
  - `echmo_memo_itc_scale.json`
  - `echmo_4moe_itc_scale.json`
  - `echmo_itc_scale_lmf-v7.json`
  - `echmo_itc_scale_lmf_chain_parallel.json`
- Added standalone LMF modification report:
  - `svrss/readme/LMF_V7_Chain_Parallel_Modification_Report.md`

### Changed

- Refactored EchMo and EchMoWide to support configurable serial/parallel LMF chain execution.
- Completed scale-scheme rollout for EchMo family variants (base / wide / deep / memo / 4moe) with aligned ITC-scale config naming.
- Updated LMF serial chain accumulation to list + single `torch.cat` to reduce unnecessary concat overhead.
- Moved root-level `svrss` markdown technical notes into `svrss/readme/` for cleaner repository structure.
- Standardized newly added technical markdown notes to English for public release consistency.

### Fixed

- Fixed temporary syntax/indentation issues introduced during LMF parallel-layer refactor.
- Fixed class-name consistency by renaming parallel LMF backend class to `LMF_Layer_Parallel`.
- Fixed curriculum class-discovery coverage for new LMF modules in `svrss/learners/model.py`.

### Benchmarks

- Added MLDC numerical equivalence results (`MLDC` vs `MLDC_Parallel`) to `svrss/readme/EchMo_Backbone_Compute_Report.md`.
- Added LMF mode equivalence and latency comparison (`LMF_V6`, `LMF_V7`, `LMF_Chain_Parallel`) to `svrss/readme/EchMo_Backbone_Compute_Report.md`.

### Version

- Project version bumped to `1.0.6` in:
  - `README.md`
  - `setup.py`
  - `CHANGELOG.md`

## [1.0.5] - 2026-03-24

### Added

- LSFJamming dataset support in `svrss`, including dataset split loader, path registration, inference/testing routing, normalization statistics, and class-weight config files.
- LSFJamming-specific custom loss option `lsf_wce` in `svrss/utils/functions.py`.
- EchMo ablation and tuning configs for NuaaSim, including LMF depth/width and LDC capacity variants.
- Recall-oriented NuaaSim EchMo configs for the LMF_V6 residual experiments.

### Changed

- EchMo LMF stack upgraded to `Learnable_Match_FilterV6`: the first layer keeps prior-guided initialization and deeper layers switch to residual correction mode.
- EchMo RD translator path now uses amp+phase+I/Q features instead of amp-only input.
- `echmo_scale.json` in `svrss` was retuned to the current moderate-scale EchMo baseline for NuaaSim.
- Test and CFAR test loops now use a single-line progress bar instead of per-sequence logging spam.
- Validation/test logs now print recall-by-class and confusion matrix to make foreground learning failures easier to diagnose.
- Focal-style losses in `svrss` now accept configurable `focal_gamma` from JSON configs.

### Fixed

- LSFJamming class handling in `svrss` was aligned to the observed 7-class mask labels (`0..6`) instead of the earlier inconsistent 9-class assumption.
- Qualitative export and class-weight selection were aligned with the corrected LSFJamming class count.
- Range-Doppler evaluation metric reporting now uses per-class TP/GT accuracy and a proper global pixel accuracy.

### Synced

- Ported the main LSFJamming data-path and training/inference support from `svrss-echmo` into `svrss`.
- Synced the customized `lsf_wce` loss behavior into `svrss`.

### Version

- Project version bumped to `1.0.5` in:
  - `README.md`
  - `setup.py`

## [1.0.4] - 2026-03-23

### Added

- Optional FLOPs profiling switch for EchMo runners via `--profile` (on-demand profiling only).
- DDP synchronization barrier after rank0-only profiling in EchMo runner.
- `prf_cuda` FP32 guard in peak convolution path for safer mixed-precision interoperability.

### Changed

- EchMo training policy finalized to stable FP32 path; AMP runtime options removed after evaluation.
- FLOPs profiling now uses batch size 1 dummy input and reports per-sample FLOPs.
- DDP behavior hardened by skipping rank0-only graph tracing in distributed mode.

### Fixed

- Reduced risk of NCCL timeout caused by rank desynchronization after rank0-only setup work.
- Reduced risk of custom op dtype mismatch in `prf_cuda` path when external autocast is active.

### Synced

- Applied the same EchMo profile/DDP/custom-op safety changes to `svrss-echmo`:
  - `run_echmo.py`
  - `learners/model.py`
  - `models/model_utils/peakconv2d.py`

### Version

- Project version bumped to `1.0.4` in:
  - `README.md`
  - `setup.py`
