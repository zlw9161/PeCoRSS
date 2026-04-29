# SVRSS (T-Foc / SV branch)

本目录整理的是 **SVRSS** 方向的一套“干净可运行”代码（以 `train_ddp_sv.py` 为训练入口，支持 DDP）。

## 算法简介

SVRSS 面向单视图（Single-View）的雷达语义分割训练/评估流水线。整体结构沿用“配置驱动 + 数据加载器 + 模型构建 + 训练器（含 DDP）”的工程化组织方式，便于替换不同的 `tfoc_ultra_sv*` 模型变体与配置文件。

## 算法构成

- **训练入口**
  - `train_ddp_sv.py`: DDP/SyncBN 训练入口（本仓库默认使用该入口）
- **配置**
  - `config_files/`: 训练/测试配置（JSON）
- **训练逻辑**
  - `learners/initializer_ddp_sv.py`: DDP 训练的实验目录/数据加载初始化
  - `learners/model_ddp_sv.py`: DDP 训练主循环（日志、验证、保存等）
- **数据**
  - `loaders/dataloaders_sv.py`: SV 数据集与 dataloader
- **模型**
  - `models/`、`models_3D/`: `tfoc_ultra_sv*` 系列网络实现
- **工具**
  - `utils/`: 常用函数（归一化、loss、可视化等）

## 训练命令

README 中训练命令按照你指定的 tmux 格式给出（注意：入口脚本为 `train_ddp_sv.py`）：

```bash
tmux new -d -s tfoc_sv_rd 'bash -lc "
cd \"/media/ycc/4TB/PaperWork/TFocNet/T-Foc/svrss\" &&
mkdir -p logs &&
TS=\$(date +%Y%m%d_%H%M%S) &&
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.. \
conda run --no-capture-output -n t-foc \
torchrun --standalone --nproc_per_node 1 train_ddp_sv.py --cfg config_files/tfoc_sv_ddp_rd.json --no-sync-bn \
> >(tee \"logs/ddp_sv_v1_revise_rd_\${TS}.out.log\") \
2> >(tee \"logs/ddp_sv_v1_revise_rd_\${TS}.err.log\" >&2)
"'
```

## 备注

- 若你的数据路径依赖 `utils/paths.py` 或配置里的路径字段，请先按你本机路径完成一次配置（否则 dataloader 会找不到数据）。
- DDP 模式下建议使用 `torchrun` 启动，并确保 `CUDA_VISIBLE_DEVICES` 与 `--nproc_per_node` 对应。
