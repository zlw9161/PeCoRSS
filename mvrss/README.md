# MVRSS (T-Foc / _ddp bundle)

本目录整理的是 **MVRSS** 方向的 DDP 训练代码（以 `train_F_ddp.py` 为入口），从原始工程中抽出并清理无关日志/缓存文件，保留可运行所需的核心模块与配置。

## 算法简介

MVRSS 面向多视图（Multi-View）的雷达语义分割训练/评估框架。该版本在原有 MVRSS 训练框架上提供了 **DDP（DistributedDataParallel）** 入口与训练循环适配，用于多卡训练并在 rank0 进行集中式日志/保存。

## 算法构成

- **训练入口**
  - `train_F_ddp.py`: DDP 启动入口（构建模型、初始化分布式、DDP wrap）
- **配置**
  - `config_files/`: 训练配置（JSON），例如 `tfoc_F_tuned_tw1_g095_progress_full_ddp.json`
- **训练逻辑**
  - `learners/initializer_ddp.py`: DDP 数据加载与实验目录结构初始化
  - `learners/model_F_ddp.py`: DDP 训练主循环（loss/val/test/保存等，默认 rank0 写盘）
- **数据**
  - `loaders/`: Carrada 等数据集封装与 dataloader
- **模型**
  - `models/`: MVRSS/TFocNet 等网络实现（`train_F_ddp.py` 内按 `cfg["model"]` 构建）
  - `timesformer/`: 相关时序模块（若配置/模型引用到）
- **工具**
  - `utils/`: loss、归一化、可视化、路径管理等通用函数
  - `tools/`: 常用脚本（如 DDP 启动脚本）

## 训练命令

从项目根目录进入本目录后运行（示例为 4 卡；卡数由 `--nproc_per_node` 决定）：

```bash
cd "/media/ycc/4TB/PaperWork/TFocNet/T-Foc/mvrss"
torchrun --standalone --nproc_per_node 4 train_F_ddp.py --cfg ./config_files/tfoc_F_tuned_tw1_g095_progress_full_ddp.json
```

## 备注

- 若你使用的是单机多卡，推荐 `torchrun --standalone`；集群环境可按需要改 `--nnodes/--node_rank/--rdzv_*`。
- 如果每卡 batch 很小，可在配置中加入 `"sync_bn": true` 来启用 SyncBN（`train_F_ddp.py` 会自动转换）。
