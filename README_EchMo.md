# EchMo Family Guide

> This is a dedicated EchMo-family guide covering training, inference, and app visualization.
> The structure follows the style of the main project documentation and focuses on directly runnable commands.

---

## Table of Contents

- [1. EchMo Family Overview](#1-echmo-family-overview)
- [2. Environment and Path Setup](#2-environment-and-path-setup)
- [3. Training: Full EchMo Family](#3-training-full-echmo-family)
- [4. Inference: infer_echmo.py](#4-inference-inferechmopy)
- [5. Profiling: profiling_echmo_testbook.py](#5-profiling-profiling_echmo_testbookpy)
- [6. App Visualization](#6-app-visualization)
- [7. FAQ](#7-faq)

---

## 1. EchMo Family Overview

The repository currently supports the following EchMo-family models:

| Model | Example Config | Training Entry | Inference Entry |
|:--|:--|:--|:--|
| echmo | svrss/config_files/echmo_itc_scale.json | svrss/run_echmo.py | svrss/infer_echmo.py |
| echmo_deep | svrss/config_files/echmo_deep_itc_scale.json | svrss/run_echmo.py | svrss/infer_echmo.py |
| echmo_memo | svrss/config_files/echmo_memo_itc_scale.json | svrss/run_echmo.py | svrss/infer_echmo.py |
| echmo_4moe | svrss/config_files/echmo_4moe_itc_scale.json | svrss/run_echmo.py | svrss/infer_echmo.py |
| echmo_reck | svrss/config_files/echmo_reck_itc_scale.json and reck0/1/2/3 variants | svrss/run_echmo.py | svrss/infer_echmo.py |
| echmo_wide | svrss/config_files/echmo_wide_itc_scale.json | svrss/run_echmo.py | svrss/infer_echmo.py |
| rechmo | svrss/config_files/rechmo.json | svrss/run_echmo.py | svrss/infer_echmo.py |

Notes:
- [svrss/run_echmo.py](svrss/run_echmo.py) already includes full model-family branching.
- [svrss/infer_echmo.py](svrss/infer_echmo.py) already supports full-family checkpoint loading and test evaluation.

---

## 2. Environment and Path Setup

### 2.1 Recommended Environment

- Python >= 3.9
- PyTorch >= 2.1
- CUDA is optional (CPU mode is also supported)

### 2.2 Install Dependencies

Run from the repository root:

```bash
pip install -r requirements.txt
```

### 2.3 Configure Dataset and Log Paths

Please review and update:

- [svrss/config_files/config.ini](svrss/config_files/config.ini)

At minimum, make sure these paths are correct:

- logs
- nuaasim
- lsf_jamming
- any additional dataset paths you plan to use

> Please note that lsf_jamming is the data_path for RED-HiL dataset in the paper, nuaasim is the path for RED-Syn dataset.

---

## 3. Training: Full EchMo Family

The following commands assume you run from the repository root.

### 3.1 Single-GPU Training

```bash
cd svrss
CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/echmo_itc_scale.json
```

Replace the config file with the target model variant, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/echmo_deep_itc_scale.json
CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/echmo_memo_itc_scale.json
CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/echmo_4moe_itc_scale.json
CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/echmo_reck2_itc_scale.json
CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/echmo_wide_itc_scale.json
CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/rechmo.json
```

### 3.2 Multi-GPU DDP Training

```bash
cd svrss
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=12355 run_echmo.py --cfg config_files/echmo_itc_scale.json --sync-bn
```

### 3.3 Resume or Fine-Tune from a Checkpoint

```bash
cd svrss
CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/echmo_4moe_itc_scale.json --finetune /abs/path/to/your_checkpoint.pt
```

Training outputs are saved under:

- logs/<dataset>/<model>/<name_exp>_<version>/results/

Common files:

- <model>.pt
- <model>_checkpoint_epochXX.pt
- config.json

---

## 4. Inference: infer_echmo.py

### 4.1 Basic Inference

```bash
python -m svrss.infer_echmo --cfg svrss/config_files/echmo_itc_scale.json
```

### 4.2 Pick a Specific Checkpoint Epoch

```bash
python -m svrss.infer_echmo --cfg svrss/config_files/echmo_4moe_itc_scale.json --checkpoint-epoch 40
```

### 4.3 Use an Explicit Checkpoint Path

```bash
python -m svrss.infer_echmo --cfg svrss/config_files/echmo_reck2_itc_scale.json --model-path /abs/path/to/echmo_reck.pt
```

### 4.4 Disable Qualitative Export

```bash
python -m svrss.infer_echmo --cfg svrss/config_files/rechmo.json --no-quali
```

### 4.5 Checkpoint Resolution Priority

[svrss/infer_echmo.py](svrss/infer_echmo.py) resolves checkpoints in this order:

1. --model-path
2. checkpoint_path in the config JSON
3. results/<model>.pt in the resolved experiment directory
4. latest results/<model>_checkpoint_epoch*.pt

---

## 5. Profiling: profiling_echmo_testbook.py

The profiling entrypoint is:

- [svrss/profiling_echmo_testbook.py](svrss/profiling_echmo_testbook.py)

It supports:

- scratch initialization or pretrained checkpoint loading
- inference and train-style profiling
- module-wise, end2end, and front-back profiling levels
- random and config-driven synthetic input construction
- single-GPU and DDP execution
- Markdown and JSON report export

### 5.1 Parameter Source Priority

The profiling script resolves runtime settings in this order:

1. CLI explicit arguments
2. profiling-specific config fields such as `profile_batch_size` or `profile_data_mode`
3. existing training config fields such as `batch_size` or `lr`
4. script fallback defaults

Supported profiling-specific config fields:

- `profile_batch_size`
- `profile_warmup_iters`
- `profile_measure_iters`
- `profile_train_lr`
- `profile_data_mode`
- `profile_input_scale`
- `profile_input_offset`

### 5.2 Single-GPU Scratch Profiling

```bash
python svrss/profiling_echmo_testbook.py \
	--cfg svrss/config_files/echmo_itc_scale.json \
	--load-mode scratch
```

### 5.3 Single-GPU Pretrained Profiling

```bash
python svrss/profiling_echmo_testbook.py \
	--cfg svrss/config_files/echmo_itc_scale.json \
	--load-mode pretrained \
	--model-path /abs/path/to/checkpoint.pt
```

### 5.4 Random vs Config Data Mode

Use random synthetic input:

```bash
python svrss/profiling_echmo_testbook.py \
	--cfg svrss/config_files/echmo_itc_scale.json \
	--load-mode scratch \
	--data-mode random
```

Use config-driven deterministic input:

```bash
python svrss/profiling_echmo_testbook.py \
	--cfg svrss/config_files/echmo_itc_scale.json \
	--load-mode scratch \
	--data-mode config
```

### 5.5 Restrict Tasks and Profiling Levels

Inference-only end-to-end profiling:

```bash
python svrss/profiling_echmo_testbook.py \
	--cfg svrss/config_files/echmo_itc_scale.json \
	--load-mode scratch \
	--tasks infer \
	--levels end2end
```

Train and infer profiling across all levels:

```bash
python svrss/profiling_echmo_testbook.py \
	--cfg svrss/config_files/echmo_itc_scale.json \
	--load-mode scratch \
	--tasks train,infer \
	--levels module-wise,end2end,front-back
```

### 5.6 DDP Profiling

Two-GPU DDP inference profiling:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
	svrss/profiling_echmo_testbook.py \
	--cfg svrss/config_files/echmo_itc_scale.json \
	--load-mode scratch \
	--tasks infer \
	--levels end2end
```

Two-GPU DDP full profiling:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
	svrss/profiling_echmo_testbook.py \
	--cfg svrss/config_files/echmo_itc_scale.json \
	--load-mode scratch \
	--tasks train,infer \
	--levels module-wise,end2end,front-back
```

Notes:

- In DDP mode, `batch_size` means per-rank batch size.
- The report also records `global_batch_size = batch_size_per_rank * world_size`.
- Only rank 0 writes the Markdown and JSON files.

### 5.7 Output Files

By default, reports are written under:

- `svrss/readme/`

Each run generates:

- one Markdown report
- one JSON report

You can also override output locations with:

- `--output-dir`
- `--output-markdown`
- `--output-json`

## 6. App Visualization

Detailed app document see:

- [svrss/app/README.md](svrss/app/README.md)

### 6.1 Run the Official App (Built-In EchMo Baseline)

```bash
cd svrss/app
pip install -r requirements.txt
python app.py
```

Default local URL is typically:

- http://127.0.0.1:7860

You need:

- dataset files under svrss/app/datasets/
- model weights under svrss/app/models/<dataset>/<exp_folder>/results/

These weight locations must match MODEL_PATHS in [svrss/app/app.py](svrss/app/app.py).

### 6.2 Extend the App to Full EchMo Family

The current [svrss/app/app.py](svrss/app/app.py) ships with an EchMo baseline workflow. To enable switching across echmo_deep, echmo_memo, echmo_4moe, echmo_reck, echmo_wide, and rechmo in the UI, use this extension workflow:

1. Add corresponding model definition files under svrss/app/models/ aligned with implementations in svrss/models/.
2. Add imports for these model classes in [svrss/app/app.py](svrss/app/app.py).
3. Extend MODEL_PATHS for each dataset with the new model variants.
4. Extend ModelManager model-construction branching so each model name maps to the correct class instead of always using EchMo.
5. Place variant-specific checkpoints in the paths declared in MODEL_PATHS.
6. Restart app.py and select or compare the target variant in the UI.

Recommended enablement order:

- echmo_deep
- echmo_memo
- echmo_4moe
- echmo_reck
- echmo_wide

This order aligns well with existing training and inference config coverage.

---

## 7. FAQ

### Q1: Why does inference fail with checkpoint-not-found?

- Verify dataset, model, version, and exp_desc in the config match the trained experiment.
- Use --model-path to point to a known checkpoint file when in doubt.

### Q2: Why does DDP training hang or desync?

- Ensure nproc_per_node matches the number of visible GPUs in CUDA_VISIBLE_DEVICES.
- In DDP mode, avoid enabling profiling first.

### Q3: Why are no datasets shown in the app?

- Check that svrss/app/datasets/<dataset>/light_dataset_frame_oriented.json exists.

### Q4: Why does model loading fail in the app?

- Verify MODEL_PATHS entries and checkpoint files.
- Verify architecture-config compatibility.
- Check core config fields such as nb_classes, w_size, h_size, and nb_input_channels.

---

## Appendix: Quick Command Cheatsheet

```bash
# Train (single GPU)
cd svrss && CUDA_VISIBLE_DEVICES=0 python run_echmo.py --cfg config_files/echmo_itc_scale.json

# Infer (automatic checkpoint resolution)
cd .. && python -m svrss.infer_echmo --cfg svrss/config_files/echmo_itc_scale.json

# Infer (explicit checkpoint)
python -m svrss.infer_echmo --cfg svrss/config_files/echmo_reck2_itc_scale.json --model-path /abs/path/to/model.pt

# Profile (single GPU, scratch)
python svrss/profiling_echmo_testbook.py --cfg svrss/config_files/echmo_itc_scale.json --load-mode scratch

# Profile (DDP, two GPUs)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 svrss/profiling_echmo_testbook.py --cfg svrss/config_files/echmo_itc_scale.json --load-mode scratch --tasks infer --levels end2end

# Visualize
cd svrss/app && python app.py
```
