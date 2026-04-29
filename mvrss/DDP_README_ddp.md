# DDP bundle (T-Foc / mvrss)

This bundle contains the minimal files to run **DistributedDataParallel (DDP)** training for `TFocNet` (v1).

## How to run (Linux)

From the project root (the folder that contains `train_F_ddp.py`):

```bash
conda activate t-foc
torchrun --standalone --nproc_per_node 4 train_F_ddp.py --cfg ./config_files/tfoc_F_tuned_tw1_g095_progress_full_ddp.json
```

## Notes

- DDP uses **one process per GPU**, so in `train_F_ddp.py` we force `cfg["gpus"]=1`.
- Validation/test are executed **only on rank0** by default (to avoid duplicate evaluation and file writes).
- If per-GPU batch is small, consider enabling SyncBN by adding `"sync_bn": true` to your config json.

## Files included

- `train_F_ddp.py`
- `learners/initializer_ddp.py`
- `learners/model_F_ddp.py`
- `learners/tester_F.py` (progress printing support)
- `models/tfoc_vanilla_maskdraw_light_clutter_n_v1.py`
- `models/__init__.py` (ensures `from mvrss.models import TFocNet` points to v1)
- config examples under `config_files/`

- 总共用几张卡由 --nproc_per_node 决定

