# EchMo Family Guide

> This is a dedicated EchMo-family guide covering training, inference, and app visualization.
> The structure follows the style of the main project documentation and focuses on directly runnable commands.

---

## Table of Contents

- [1. EchMo Family Overview](#1-echmo-family-overview)
- [2. Environment and Path Setup](#2-environment-and-path-setup)
- [3. Training: Full EchMo Family](#3-training-full-echmo-family)
- [4. Inference: infer_echmo.py](#4-inference-inferechmopy)
- [5. App Visualization](#5-app-visualization)
- [6. FAQ](#6-faq)

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

### 2.3 — Optional CUDA Extension for PeakConv Acceleration

> This step is optional, but it is recommended for EchMo models.

```bash
cd PeCoRSS
python setup.py develop --with-cuda
```

If you do not install cuda toolkit, you should download `cuda_12.3.0_545.23.06_linux.run` (this is the recommended cuda version, please check your GPU-CUDA support, e.g., using `nvidia-smi` to check the supported CUDA version). The following instruction is the introduction of installing `cuda_XX.run`:

```bash
$ cd your_path_to_downloaded_cuda_file
$ sudo ./cuda_12.3.0_545_23.06_linux.run
```

Please enter `space` to avoid installing driver again, then enter `install` to run the cuda installation. When the installation complete, please typing:

```bash
$ nvcc --version
```

to verify the cuda toolkit installation, if `nvcc` is not available, please add path to `~\.bashrc`;

```bash
$ vim ~\.bashrc
```

Add `export PATH="/usr/local/cuda-12.3/bin:$PATH"` under `# >>>conda initialize >>>`. Then

```bash
$ source ~\.bashrc
$ nvcc --version
```

Then go to the step of cuda implementation of PRF gathering.

#### CUDA Acceleration Effects

| Model | GPU Mem. | GMac | FPS | Input Size | #Params |
|:------|:--------:|:----:|:---:|:----------:|:-------:|
| PKCIn4CWR | 12007 MB | 162.52 | 29.5 | 1,1,5,124,2048 | 1.21 M |
| PKCIn4CWR (accelerated) | 5697 MB | 162.52 | 63.7 | 1,1,5,124,2048 | 1.21 M |
| PKCIn4PDR | 3007 MB | 32.77 | 137.5 | 1,1,5,64,800 | 1.21 M |
| PKCIn4PDR (accelerated) | 1761 MB | 32.77 | 396.6 | 1,1,5,64,800 | 1.21 M |


### 2.4 Configure Dataset and Log Paths

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

## 5. App Visualization

![EchMo vs Traditional](images/screenshot_echmo_vs_traditional.png)

Detailed app document see:

- [svrss/app/README.md](svrss/app/README.md)

### 5.1 Run the Official App (Built-In EchMo Baseline)

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

### 5.2 Extend the App to Full EchMo Family

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

## 6. FAQ

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

# Visualize
cd svrss/app && python app.py
```
