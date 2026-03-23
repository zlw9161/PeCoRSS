# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

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
