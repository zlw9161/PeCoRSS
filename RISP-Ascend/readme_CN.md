# RISP_Ascend

本项目用于记录并沉淀在华为昇腾 Ascend 平台上部署算法模型的完整流程和常见问题处理方法。文档面向需要将 PyTorch 模型迁移到 Ascend NPU、导出 ONNX、通过 ATC 工具转换为 OM 模型，并最终使用 ACL 或 msame 在 Ascend 310 / 310P / 310B 等推理设备上执行离线推理的开发者。

与普通安装说明不同，本 README 更强调可复现的工程实践：既给出基础环境检查和安装命令，也集中整理部署过程中容易出现的版本、权限、输入维度、数据类型、batch size、`soc_version` 和性能分析问题，便于后续成员快速接手项目。

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 适用环境与软硬件说明](#2-适用环境与软硬件说明)
- [3. 快速开始：NPU 状态检查与 CANN 版本确认](#3-快速开始npu-状态检查与-cann-版本确认)
- [4. Python / PyTorch NPU 环境配置](#4-python--pytorch-npu-环境配置)
- [5. Ascend 910 在线训练与推理注意事项](#5-ascend-910-在线训练与推理注意事项)
- [6. 本项目算法部署流程](#6-本项目算法部署流程)
- [7. 模型转换：`.pth -> .onnx -> .om`](#7-模型转换pth---onnx---om)
- [8. OM 离线推理与 ACL 调用](#8-om-离线推理与-acl-调用)
- [9. 性能分析、精度对比与混合精度](#9-性能分析精度对比与混合精度)
- [10. 常见报错与解决办法](#10-常见报错与解决办法)
- [11. 参考资料索引](#11-参考资料索引)

## 1. 项目简介

本项目主要覆盖 PyTorch 模型在华为昇腾 Ascend 平台上的部署链路，包括在线训练 / 推理环境配置、ONNX 导出、ATC 转换、OM 离线推理、ACL 调用、性能分析、精度对比和常见问题排查。

几个核心概念如下：

- **Ascend 910**：通常用于模型训练、PyTorch 在线推理，或将 `.pth` 权重导出为 ONNX 模型。
- **Ascend 310 / 310P / 310B**：通常用于 OM 模型离线推理部署。
- **CANN**：昇腾异构计算架构软件栈，包含 toolkit、runtime、driver、toolbox 等组件。
- **ATC**：Ascend Tensor Compiler，用于将 ONNX 等模型转换为昇腾可执行的 OM 离线模型。
- **ACL**：Ascend Computing Language，用于加载 OM 模型、管理内存、执行推理和获取输出。

典型部署链路如下：

```text
PyTorch 权重文件 (.pth)
        |
        v
ONNX 模型文件 (.onnx)
        |
        v
Ascend OM 离线模型 (.om)
        |
        v
ACL / msame 推理
```

## 2. 适用环境与软硬件说明

原始验证环境的设备内核信息如下。该信息主要用于说明当时的验证环境，实际部署时不要求完全一致，但需要重点确认 CPU 架构、NPU 型号、驱动版本、CANN 版本、Python 版本和 `torch-npu` 版本是否匹配。

```bash
Linux user 5.4.0-26-generic #30-Ubuntu SMP Mon Apr 20 16:59:40 UTC 2020 aarch64 aarch64 aarch64 GNU/Linux
```

| 项目 | 说明 | 建议检查方式 |
| --- | --- | --- |
| NPU 型号 | 确认当前设备是 Ascend 910、310、310P、310B 或其他型号。 | `npu-smi info` |
| CPU 架构 | 确认系统为 `aarch64` / `arm64` 或 `x86_64`，不同架构的软件包路径不同。 | `uname -a` |
| CANN 版本 | 确认 toolkit、runtime、toolbox 等组件版本。 | `cat ascend_toolkit_install.info` |
| Python 版本 | 建议优先使用已验证的 Python 3.9 环境；部分环境中 Python 3.10 可能触发兼容性问题。 | `python --version` |
| PyTorch / torch-npu | `torch` 与 `torch-npu` 需要按版本配套安装，避免接口或 ABI 不兼容。 | `pip show torch torch-npu` |

## 3. 快速开始：NPU 状态检查与 CANN 版本确认

### 3.1 NPU 状态检查

在使用 Ascend 相关设备前，应先检查 NPU 是否被系统识别，并确认具体型号、健康状态和可用设备编号。

```bash
sudo su
npu-smi info
```

如果命令无法执行，或看不到设备，建议优先检查以下内容：

- Ascend 驱动是否正确安装；
- NPU 相关服务是否正常启动；
- 当前用户权限是否足够；
- 当前系统架构与驱动包是否匹配；
- 当前设备型号是否与安装的软件包版本匹配。

### 3.2 查询 CANN 软件包版本

在环境适配或排查版本问题时，需要确认 CANN 软件包版本。以下以 `Ascend-cann-toolkit` 为例，其他组件请按照实际安装路径替换。

```bash
cd /usr/local/Ascend/ascend-toolkit/latest/{arch}-linux
cat ascend_toolkit_install.info
```

说明：

- `/usr/local/Ascend` 是 root 用户默认安装路径，若实际安装路径不同，应替换为本机路径。
- `{arch}` 表示 CPU 架构，常见取值包括 `arm64` 或 `x86_64`。
- toolkit、toolbox、driver、runtime 的版本需要与目标设备和开发框架保持兼容。

## 4. Python / PyTorch NPU 环境配置

### 4.1 Conda 环境创建

原始笔记中记录 Python 3.9 环境测试可用。如遇到 `SystemError: PY_SSIZE_T_CLEAN macro must be defined for # formats` 等错误，通常与 Python 版本或依赖编译兼容性有关，可优先回退到 Python 3.9。

```bash
conda create --name py39test python=3.9
conda activate py39test
```

### 4.2 安装基础依赖、PyTorch 与 torch-npu

以下命令保留原始笔记中的可用安装方式。部署时应结合 CANN 版本和 `torch-npu` 官方适配关系调整版本号。

```bash
conda install pyyaml setuptools wheel typing_extensions numpy==1.26.4 protobuf attrs pathlib2 scipy requests psutil absl-py decorator
pip3 install torch==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install torch-npu==2.1.0rc1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **版本匹配提示**  
> `torch` 与 `torch-npu` 不建议随意交叉安装。若导入 `torch_npu`、模型加载或算子执行阶段出现异常，应优先回到“设备型号—驱动—CANN—torch—torch-npu—Python”这一条版本链路进行排查。

### 4.3 处理 static TLS block 报错

在导入 PyTorch、`torch_npu` 或相关依赖时，若出现以下报错：

```text
ImportError: /usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block
# 或
Import Bug: cannot allocate memory in static TLS block
```

可通过预加载 `libgomp` 解决。注意路径应以报错中提示的实际路径为准。

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD

# 重新打开 terminal，或执行：
source ~/.bashrc
```

### 4.4 PyTorch NPU 基础导入方式

在 Ascend 910 上进行 PyTorch 训练或在线推理时，通常需要导入 `torch_npu`，并根据项目需要启用自动迁移和 AMP。

```python
import time
import torch
import torch_npu
from torch_npu.npu import amp                  # AMP 混合精度模块
from torch_npu.contrib import transfer_to_npu  # 使能自动迁移
```

## 5. Ascend 910 在线训练与推理注意事项

### 5.1 权限与设备设置

原始验证环境中，部分 Ascend 910 训练 / 推理脚本需要在 `sudo su` 权限下执行。若普通用户运行时报权限错误，可先用 root 验证，再根据实际情况配置设备节点权限、用户组或运行环境。

进入 `eval` 或在线推理流程前，建议显式设置 NPU 设备号，避免默认设备未初始化导致报错。

```python
torch.npu.set_device(0)
```

### 5.2 模型权重加载

在 NPU 环境中加载模型权重时，建议先将权重映射到 CPU，再按需迁移到 NPU，可降低跨设备加载导致的兼容性问题。

```python
model.load_state_dict(torch.load(model_dir, map_location="cpu"))
```

### 5.3 推理耗时统计

NPU 执行具有异步特性，直接用 `time.time()` 包裹 `model(input)` 可能得到偏小的耗时。计时前后应加入设备同步。推荐优先使用 `torch.npu.synchronize()`；如果项目中通过自动迁移保留了 CUDA 接口，应确认 `synchronize` 调用实际同步的是 NPU 执行流。

```python
torch.npu.synchronize()
start = time.time()
result = model(input)
torch.npu.synchronize()
end = time.time()
print("inference time:", end - start)
```

## 6. 本项目算法部署流程

本章节用于说明本项目具体算法模型的部署流程。整体流程为：

```text
.pth 权重文件 -> ONNX 模型文件 -> OM 推理文件 -> ACL 推理得到结果
```

### 6.1 模型说明

该部署项目使用 `pkcin4any` 模型进行部署。整体思路是：

1. 将训练后得到的 `pkcin4any.pth` 通过 `./pth2onnx.py` 导出为 `pkcin4any.onnx`；
2. 使用 ATC 转换工具将 ONNX 文件转换为 `./pkcin4any.om`；
3. 使用 `./ACL_inference.py` 调用 OM 模型进行推理，得到最终结果。

其中：

- `./onnx_sim.py` 用于对 `.onnx` 文件进行化简，并可将 FP32 转换为 FP16；如有需要，可独立使用。
- 项目中给出了 `./PKC_model.om` 权重文件，可直接用于推理测试。

### 6.2 `.pth -> .onnx` 导出流程

使用 `./pth2onnx.py` 即可完成导出。需要注意以下几点：

- 代码中的 `input_shape` 需要根据输入数据的实际情况进行修改，可以直观理解为输入数据的维度；
- `batch_size` 不用修改，保持为 `1` 即可；
- 当前代码中使用 `./models/deeplabv3plus.py` 的模型代码；
- 当前代码中使用 `./weight/deeplabv3plus.pt` 的权重文件（后续上传至huggingface）；
- 运行后得到 `./deeplab_model.onnx` 文件（后续上传至huggingface）；
- 如果需要使用自己的模型代码和权重文件，修改相关路径以及 `input_shape` 即可。

### 6.3 `.onnx -> .om` 转换流程

使用 ATC 工具进行转换。以下是实际使用过程中的转换命令示例：

```bash
atc --model=deeplab_model.onnx \
    --framework=5 \
    --output=deeplab_model \
    --input_shape="input:1,1,1,124,2048" \
    --input_fp16_nodes="input" \
    --soc_version=Ascend310P3
```

实际使用时，重点修改以下参数：

- `--model`：ONNX 模型路径；
- `--output`：输出 OM 文件前缀；
- `--input_shape`：输入节点名称和输入维度；
- `--soc_version`：目标推理设备型号。

### 6.4 OM 推理与结果验证

使用 `./ACL_inference.py` 即可完成 OM 推理。实际使用时，主要修改以下内容：

- OM 模型文件路径；
- 输入数据路径；
- 输入数据 shape 和 dtype；
- 输出解析和后处理逻辑。


## 7. 模型转换：`.pth -> .onnx -> .om`

### 7.1 `.pth` 导出 ONNX

通常建议在训练或在线推理环境中完成 `.pth -> .onnx` 导出。导出前需要确认模型处于 `eval` 模式，并保证 dummy input 的 shape、dtype、device 与后续部署一致。

```bash
# 示例：具体参数以本项目 pth2onnx.py 为准
python pth2onnx.py --weight ./weights/model.pth --output ./models/model.onnx
```

注意事项：

- `input_names` 必须与后续 ATC 命令中的 `--input_shape` 前缀保持一致；
- 导出后建议用 `onnx.checker` 或 ONNX Runtime 先验证模型结构和输出是否正常；
- 若模型存在自定义算子或不支持算子，需要在导出阶段或 ATC 转换阶段重点处理。

### 7.2 onnxsim 安装失败处理

如果安装 `onnxsim` 失败，处理方式是先升级 pip 侧的 `cmake`，再重新安装。

```bash
pip3 install --upgrade cmake
pip3 install onnxsim
```

### 7.3 使用 ATC 将 ONNX 转为 OM

ATC 转换时需要明确模型路径、框架类型、输出名、输入维度和目标设备型号。以下命令为DnCNN示例。

```bash
atc --model=DnCNN.onnx \
    --framework=5 \
    --output=DnCNN \
    --input_shape="input:1,1,64,64" \
    --soc_version=Ascend310P3
```

| 参数 | 含义 | 注意事项 |
| --- | --- | --- |
| `--model` | 输入 ONNX 模型路径 | 确认模型已通过 ONNX 基础检查。 |
| `--framework=5` | 表示输入模型框架为 ONNX | 不同框架取值不同，ONNX 常用 `5`。 |
| `--output` | 输出 OM 模型前缀 | 最终生成 `.om` 文件。 |
| `--input_shape` | 模型输入名称和静态输入维度 | `input` 必须与 `pth2onnx.py` 中 `input_names` 一致；维度必须与模型实际输入匹配。 |
| `--soc_version` | 目标推理芯片型号 | 必须与部署设备匹配，如 `Ascend310P3`、`Ascend310B` 等。 |

### 7.4 Batch size 与输入维度

OM 模型通常会固化 ATC 转换时指定的输入 shape。若 batch size 大于 1，需要在 `pth2onnx.py` 导出脚本、ATC 命令和推理脚本中同步修改 batch 维度。

- `pth2onnx.py` 中的 dummy input batch size 要与导出目标一致；
- ATC 命令中的 `--input_shape` 第一个维度要同步修改；
- ACL 推理脚本的输入数据预处理和内存申请也要与 OM 模型输入维度一致。

### 7.5 输入 FP16 优化

在部分模型中，OM 推理默认会通过 Cast 操作将 FP32 输入转换为 FP16。若输入节点可以直接使用 FP16，可在 ATC 转换时指定 `--input_fp16_nodes` 以减少 Cast 开销，从而提升推理效率。

```bash
atc --model=model.onnx \
    --framework=5 \
    --output=model \
    --input_shape="input:1,1,64,64" \
    --input_fp16_nodes="input" \
    --soc_version=Ascend310P3
```

使用该参数后，推理脚本中的 `process_input` 需要同步将输入转换为 `float16`；否则可能出现输入类型不匹配或输出异常。

## 8. OM 离线推理与 ACL 调用

### 8.1 ACL 推理基本流程

使用 `.om` 文件推理时，可参考 Ascend samples 中的 `sampleResnetQuickStart` Python 示例。实际项目一般需要重点修改 `process_input(self, input_path)` 和 `get_result(self)` 两个函数，以适配模型输入格式和输出解析逻辑。

ACL 推理基本流程如下：

1. `acl.init`：初始化 ACL 运行环境，必要时传入 profile 配置文件路径；
2. `acl.rt.set_device` / `create_context`：设置推理设备并创建上下文；
3. `acl.mdl.load_from_file`：加载 OM 模型；
4. `prepare input`：读取输入文件，完成 resize、归一化、维度调整、dtype 转换等预处理；
5. `acl.mdl.execute`：执行模型推理；
6. `get result`：读取输出内存并转换为 numpy 数组，完成后处理；
7. `release`：释放模型、数据集、内存、上下文和设备资源。

### 8.2 输入 dtype 注意事项

原始笔记中记录，DnCNN 示例中 `np.save` 默认保存为 `float64`，而 `sampleResnetQuickStart.py` 中通常按 `float32` 处理。如果 `process_input` 不显式转换 dtype，可能导致输出异常。

```python
input_data = np.load(input_path)
input_data = input_data.astype(np.float32)  # 或根据 ATC 参数改为 np.float16
```

### 8.3 msame 命令行推理

如果只是快速验证 OM 模型能否执行，可使用 `msame` 工具进行命令行推理。该方式适合做模型连通性验证、基础性能评估和与 ACL 自写脚本对照。

```bash
# 示例：具体参数以 msame 工具说明和模型输入为准
./msame --model ./model.om --input ./input.bin --output ./outputs
```


## 9. 性能分析、精度对比与混合精度

### 9.1 PyTorch Profiler 数据采集

在线训练或 PyTorch 推理阶段，可使用 Ascend PyTorch Profiler 采集算子耗时、数据搬移和执行流信息，用于定位性能瓶颈。

建议在关键推理函数前后加入同步，并对 warm-up 和正式统计阶段进行区分，避免首次编译、缓存和数据加载影响结果。

### 9.2 ACL Profile 数据采集与分析

OM 离线推理阶段，可在 `acl.init(config_path)` 中传入 profile 配置文件，生成 profile 数据目录，然后使用官方工具进行性能解析和导出。

```python
config_path = "./acl_profile.json"
ret = acl.init(config_path)
```

注意事项：

- 采集前确认 profile 配置文件路径正确；
- 采集后先确认 profile 文件夹是否生成，再进行性能解析；
- 导出的性能数据可用于定位算子耗时、数据搬移开销和模型瓶颈。

### 9.3 不同 Device 精度对比

从 PyTorch 到 ONNX，再到 OM 推理，建议至少进行一次同输入数据下的输出对比。若结果偏差较大，可通过 dump 文件和精度比对工具定位差异节点。

建议流程如下：

1. 准备固定输入样例，确保预处理完全一致；
2. 分别保存 PyTorch、ONNX Runtime、OM 推理输出；
3. 比较输出均值、最大误差、相对误差和关键指标；
4. 必要时生成 dump 文件，逐节点定位差异来源。

### 9.4 混合精度训练 AMP

混合精度训练或推理可能提升性能，但并非对所有模型都有效。若开启 AMP 后性能无明显提升，或精度下降较大，需要结合模型算子结构、输入输出 dtype、转换配置和 profiler 结果进一步调优。

```python
from torch_npu.npu import amp

# 具体 autocast / GradScaler 用法以项目训练脚本为准
```

## 10. 常见报错与解决办法

本节将原始笔记中分散记录的报错集中整理，便于后续排查。建议实际排错时按“环境版本—模型转换—输入输出—运行权限—性能精度”的顺序逐层定位。

| 问题 / 报错 | 可能原因 | 处理建议 |
| --- | --- | --- |
| `npu-smi info` 无法识别设备 | 驱动未安装、设备权限不足、NPU 服务异常或设备型号不匹配。 | 先确认驱动和 CANN 安装，再检查 root 权限、设备节点、系统架构和硬件连接。 |
| `cannot allocate memory in static TLS block` | `libgomp` 等动态库加载顺序导致 TLS 空间不足。 | 执行 `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`，路径以报错提示为准。 |
| `SystemError: PY_SSIZE_T_CLEAN macro must be defined for # formats` | Python 版本与部分依赖编译接口不兼容。 | 优先使用已验证的 Python 3.9 环境，重新安装依赖。 |
| `eval` 阶段报设备相关错误 | 未显式设置 NPU 设备或模型未正确迁移。 | 在推理前加入 `torch.npu.set_device(0)`，并确认模型和输入在同一设备。 |
| 加载权重时报错 | 权重保存设备与当前设备不一致。 | 使用 `torch.load(model_dir, map_location="cpu")` 后再加载到模型。 |
| `onnxsim` 安装失败 | `cmake` 或编译依赖版本较旧。 | 先执行 `pip3 install --upgrade cmake`，再重新安装 `onnxsim`。 |
| ATC 转换失败：`input` 不匹配 | ATC 中 `--input_shape` 的输入名与 ONNX `input_names` 不一致。 | 检查 `pth2onnx.py` 中 `input_names`，并与 ATC 命令保持一致。 |
| ATC 转换后推理维度异常 | 导出 ONNX、ATC 转换和 ACL 输入预处理的 batch / shape 不一致。 | 同步修改 dummy input、`--input_shape`、输入 bin / npy 生成逻辑和 ACL 内存申请。 |
| OM 推理输出异常 | 输入 dtype 与 OM 期望不一致，如 `float64` 未转 `float32` / `float16`。 | 在 `process_input` 中显式 `astype(np.float32)` 或 `astype(np.float16)`。 |
| `soc_version` 报错或模型无法运行 | ATC 目标芯片型号与部署设备不一致。 | 通过 `npu-smi info` 确认设备型号，并修改 `--soc_version`。 |
| 耗时统计不稳定或偏小 | NPU 异步执行，计时未同步。 | 推理前后加入 `torch.npu.synchronize()`，并设置 warm-up。 |

## 11. 参考资料索引

以下链接整理自原始笔记，按用途进行归类。

| 类别 | 资料名称 | 链接 |
| --- | --- | --- |
| 环境适配 | PyTorch 模型迁移与 Ascend 环境适配 | [打开链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/ptmigr/AImpug_0001.html) |
| 性能分析 | Ascend PyTorch Profiler 数据采集与分析 | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000068.html) |
| 性能分析 | `acl.init()` 配置 profile 文件生成 profile 数据 | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/devtools/auxiliarydevtool/atlasprofiling_16_0170.html) |
| 性能分析 | Profile 文件夹生成后的性能解析 | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/devtools/auxiliarydevtool/atlasprofiling_16_0038.html) |
| 性能分析 | 导出性能数据 | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/devtools/auxiliarydevtool/atlasprofiling_16_0040.html) |
| 精度对比 | 同模型不同 device 精度对比 | [打开链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0034.html) |
| 精度对比 | 准备 / 生成 dump 文件 | [打开链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0018.html) |
| 精度对比 | 执行精度对比操作 | [打开链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0019.html) |
| 混合精度 | 混合精度训练 AMP | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000019.html) |
| 模型转换 | `.pth` 导出 ONNX 参考 | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000095.html) |
| 模型转换 | ATC 工具介绍 | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/atctool/atlasatc_16_0007.html) |
| 模型转换 | ATC 快速入门案例 | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/quickstart/quickstart/quickstart_18_0007.html) |
| OM 推理 | Ascend samples：`sampleResnetQuickStart` | [打开链接](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetQuickStart/python) |
| ACL 推理 | ACL Python 开发指南 | [打开链接](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/aclpythondevg/aclpythondevg_0000.html) |
| 命令行推理 | `msame` 工具 | [打开链接](https://gitee.com/ascend/tools/tree/master/msame) |
| 社区参考 | OM 推理流程基本介绍 | [打开链接](https://blog.csdn.net/fradeet/article/details/131994362) |
| 社区参考 | Ascend 开发全流程参考 | [打开链接](https://blog.csdn.net/ygf666/article/details/124296461) |
