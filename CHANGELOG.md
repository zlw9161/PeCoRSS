# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [1.0.5] - 2026-03-24

### Added

- LSFJamming dataset support in `svrss`, including dataset split loader, path registration, inference/testing routing, normalization statistics, and class-weight config files.
- LSFJamming-specific custom loss option `lsf_wce` in `svrss/utils/functions.py`.
- EchMo ablation and tuning configs for NuaaSim, including LMF depth/width and LDC capacity variants.
- Recall-oriented NuaaSim EchMo configs for the LMF_V6 residual experiments.

### Changed

- EchMo LMF stack upgraded to `Leanable_Match_FilterV6`: the first layer keeps prior-guided initialization and deeper layers switch to residual correction mode.
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
