# RISP_Ascend

This project records the complete workflow and common troubleshooting methods for deploying algorithm models on the Huawei Ascend platform. It is intended for developers who need to migrate PyTorch models to Ascend NPU, export ONNX models, convert them into OM models using ATC, and finally run offline inference on Ascend 310 / 310P / 310B devices using ACL or msame.

Unlike a simple installation note, this README focuses on reproducible engineering practice. It includes basic environment checks, installation commands, model conversion steps, deployment scripts, and practical troubleshooting notes related to versions, permissions, input dimensions, data types, batch size, `soc_version`, performance profiling, and accuracy comparison.

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Applicable Hardware and Software Environment](#2-applicable-hardware-and-software-environment)
- [3. Quick Start: NPU Status Check and CANN Version Check](#3-quick-start-npu-status-check-and-cann-version-check)
- [4. Python / PyTorch NPU Environment Setup](#4-python--pytorch-npu-environment-setup)
- [5. Notes for Ascend 910 Online Training and Inference](#5-notes-for-ascend-910-online-training-and-inference)
- [6. Deployment Workflow for This Project](#6-deployment-workflow-for-this-project)
- [7. Model Conversion: `.pth -> .onnx -> .om`](#7-model-conversion-pth---onnx---om)
- [8. OM Offline Inference and ACL Usage](#8-om-offline-inference-and-acl-usage)
- [9. Performance Profiling, Accuracy Comparison, and Mixed Precision](#9-performance-profiling-accuracy-comparison-and-mixed-precision)
- [10. Common Errors and Solutions](#10-common-errors-and-solutions)
- [11. Reference Links](#11-reference-links)

## 1. Project Overview

This project covers the deployment pipeline of PyTorch models on Huawei Ascend, including online training / inference environment setup, ONNX export, ATC conversion, OM offline inference, ACL usage, performance profiling, accuracy comparison, and troubleshooting.

Key concepts are listed below:

- **Ascend 910**: Commonly used for model training, PyTorch online inference, or exporting `.pth` weights to ONNX.
- **Ascend 310 / 310P / 310B**: Commonly used for OM offline inference deployment.
- **CANN**: The Ascend heterogeneous computing software stack, including toolkit, runtime, driver, toolbox, and related components.
- **ATC**: Ascend Tensor Compiler, used to convert ONNX and other models into Ascend executable OM offline models.
- **ACL**: Ascend Computing Language, used to load OM models, manage memory, execute inference, and obtain outputs.

A typical deployment pipeline is shown below:

```text
PyTorch weight file (.pth)
        |
        v
ONNX model file (.onnx)
        |
        v
Ascend offline model (.om)
        |
        v
ACL / msame inference
```

## 2. Applicable Hardware and Software Environment

The kernel information of the original verification environment is shown below. This is mainly used to document the tested environment. The actual deployment environment does not need to be exactly the same, but the CPU architecture, NPU model, driver version, CANN version, Python version, and `torch-npu` version must be checked carefully.

```bash
Linux user 5.4.0-26-generic #30-Ubuntu SMP Mon Apr 20 16:59:40 UTC 2020 aarch64 aarch64 aarch64 GNU/Linux
```

| Item | Description | Suggested Check Method |
| --- | --- | --- |
| NPU model | Check whether the device is Ascend 910, 310, 310P, 310B, or another model. | `npu-smi info` |
| CPU architecture | Check whether the system is `aarch64` / `arm64` or `x86_64`. Software package paths differ across architectures. | `uname -a` |
| CANN version | Check versions of toolkit, runtime, toolbox, and other components. | `cat ascend_toolkit_install.info` |
| Python version | Python 3.9 is recommended because it has been verified. Python 3.10 may cause compatibility issues in some environments. | `python --version` |
| PyTorch / torch-npu | `torch` and `torch-npu` must be installed as a compatible pair to avoid API or ABI incompatibility. | `pip show torch torch-npu` |

## 3. Quick Start: NPU Status Check and CANN Version Check

### 3.1 NPU Status Check

Before using Ascend devices, check whether the NPU is recognized by the system, and confirm its model, health status, and available device IDs.

```bash
sudo su
npu-smi info
```

If the command cannot be executed or no device is detected, check the following items first:

- Whether the Ascend driver is installed correctly;
- Whether NPU-related services are running properly;
- Whether the current user has sufficient permissions;
- Whether the system architecture matches the installed driver package;
- Whether the installed software package version matches the target device model.

### 3.2 Check CANN Package Version

When adapting the environment or troubleshooting version issues, confirm the CANN package version first. The following example uses `Ascend-cann-toolkit`. Replace the path according to the actual installation directory for other components.

```bash
cd /usr/local/Ascend/ascend-toolkit/latest/{arch}-linux
cat ascend_toolkit_install.info
```

Notes:

- `/usr/local/Ascend` is the default installation path for the root user. Replace it with the actual local path if needed.
- `{arch}` indicates the CPU architecture. Common values include `arm64` and `x86_64`.
- The versions of toolkit, toolbox, driver, and runtime must be compatible with the target device and development framework.

## 4. Python / PyTorch NPU Environment Setup

### 4.1 Create a Conda Environment

According to the original notes, Python 3.9 has been verified. If errors such as `SystemError: PY_SSIZE_T_CLEAN macro must be defined for # formats` occur, they are usually related to Python version or dependency compilation compatibility. In that case, first consider switching back to Python 3.9.

```bash
conda create --name py39test python=3.9
conda activate py39test
```

### 4.2 Install Basic Dependencies, PyTorch, and torch-npu

The commands below are retained from the verified setup in the original notes. In actual deployment, adjust the version numbers according to the compatibility relationship between CANN and `torch-npu`.

```bash
conda install pyyaml setuptools wheel typing_extensions numpy==1.26.4 protobuf attrs pathlib2 scipy requests psutil absl-py decorator
pip3 install torch==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install torch-npu==2.1.0rc1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **Version matching note**  
> Do not install arbitrary combinations of `torch` and `torch-npu`. If errors occur during `torch_npu` import, model loading, or operator execution, troubleshoot along the following version chain first: device model -> driver -> CANN -> torch -> torch-npu -> Python.

### 4.3 Fix the static TLS block Error

When importing PyTorch, `torch_npu`, or related dependencies, the following error may occur:

```text
ImportError: /usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block
# or
Import Bug: cannot allocate memory in static TLS block
```

This can usually be fixed by preloading `libgomp`. Use the actual path reported in the error message.

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD

# Reopen the terminal, or run:
source ~/.bashrc
```

### 4.4 Basic PyTorch NPU Imports

When running PyTorch training or online inference on Ascend 910, import `torch_npu` and enable automatic migration or AMP if required by the project.

```python
import time
import torch
import torch_npu
from torch_npu.npu import amp                  # AMP mixed precision module
from torch_npu.contrib import transfer_to_npu  # Enable automatic migration
```

## 5. Notes for Ascend 910 Online Training and Inference

### 5.1 Permissions and Device Setting

In the original verification environment, some Ascend 910 training / inference scripts had to be executed under `sudo su`. If permission errors occur when running as a normal user, first verify the script as root, and then configure device node permissions, user groups, or the runtime environment as needed.

Before entering the `eval` or online inference process, explicitly set the NPU device ID to avoid errors caused by uninitialized default devices.

```python
torch.npu.set_device(0)
```

### 5.2 Loading Model Weights

When loading model weights in an NPU environment, it is recommended to map the weights to CPU first and then move them to NPU as needed. This reduces compatibility issues caused by cross-device loading.

```python
model.load_state_dict(torch.load(model_dir, map_location="cpu"))
```

### 5.3 Inference Time Measurement

NPU execution is asynchronous. Wrapping `model(input)` directly with `time.time()` may produce an underestimated runtime. Add device synchronization before and after inference. Prefer `torch.npu.synchronize()`. If CUDA APIs are retained through automatic migration, confirm that the synchronization call actually synchronizes the NPU execution stream.

```python
torch.npu.synchronize()
start = time.time()
result = model(input)
torch.npu.synchronize()
end = time.time()
print("inference time:", end - start)
```

## 6. Deployment Workflow for This Project

This section describes the deployment workflow for the project-specific algorithm model. The overall process is:

```text
.pth weight file -> ONNX model file -> OM inference model -> ACL inference result
```

### 6.1 Model Description

This deployment project uses the `pkcin4any` model. The overall workflow is as follows:

1. Export the trained `pkcin4any.pth` model to `pkcin4any.onnx` using `./pth2onnx.py`;
2. Convert the ONNX model to `./pkcin4any.om` using the ATC conversion tool（later upload to huggingface）;
3. Run inference with the OM model using `./ACL_inference.py` and obtain the final result.

Additional notes:

- `./onnx_sim.py` is used to simplify the `.onnx` model and can also convert FP32 to FP16. It can be used independently if needed.
- The project provides `./PKC_model.om`, which can be used directly for inference testing.

### 6.2 `.pth -> .onnx` Export Workflow

Use `./pth2onnx.py` to export the ONNX model. Pay attention to the following items:

- `input_shape` should be modified according to the actual input data. It can be regarded as the input tensor dimension;
- `batch_size` does not need to be modified and should remain `1`;
- The current script uses the model definition file `./models/deeplabv3plus.py`;
- The current script uses the weight file `./weight/deeplabv3plus.pt`;
- The output ONNX file is `./deeplab_model.onnx`;
- To use your own model code and weight file, modify the corresponding paths and `input_shape`.

### 6.3 `.onnx -> .om` Conversion Workflow

Use ATC to convert the ONNX model into an OM model. The following is an actual command example:

```bash
atc --model=deeplab_model.onnx \
    --framework=5 \
    --output=deeplab_model \
    --input_shape="input:1,1,1,124,2048" \
    --input_fp16_nodes="input" \
    --soc_version=Ascend310P3
```

In actual use, modify the following parameters as needed:

- `--model`: path to the ONNX model;
- `--output`: output prefix of the OM file;
- `--input_shape`: input node name and input dimensions;
- `--soc_version`: target inference device model.

### 6.4 OM Inference and Result Verification

Use `./ACL_inference.py` to run OM inference. In actual use, modify the following items:

- OM model file path;
- Input data path;
- Input data shape and dtype;
- Output parsing and post-processing logic.


## 7. Model Conversion: `.pth -> .onnx -> .om`

### 7.1 Export `.pth` to ONNX

It is usually recommended to export `.pth -> .onnx` in the training or online inference environment. Before export, make sure the model is in `eval` mode and that the shape, dtype, and device of the dummy input are consistent with the later deployment setup.

```bash
# Example: use the actual arguments defined in pth2onnx.py
python pth2onnx.py --weight ./weights/model.pth --output ./models/model.onnx
```

Notes:

- `input_names` must be consistent with the prefix used in `--input_shape` in the later ATC command;
- After export, use `onnx.checker` or ONNX Runtime to verify the model structure and output;
- If the model contains custom operators or unsupported operators, handle them carefully during ONNX export or ATC conversion.

### 7.2 Fix onnxsim Installation Failure

If `onnxsim` installation fails, the original notes suggest upgrading `cmake` through pip first and then reinstalling `onnxsim`.

```bash
pip3 install --upgrade cmake
pip3 install onnxsim
```

### 7.3 Convert ONNX to OM Using ATC

When using ATC, specify the model path, framework type, output name, input shape, and target device model. The following command keeps the DnCNN example.

```bash
atc --model=DnCNN.onnx \
    --framework=5 \
    --output=DnCNN \
    --input_shape="input:1,1,64,64" \
    --soc_version=Ascend310P3
```

| Parameter | Description | Notes |
| --- | --- | --- |
| `--model` | Input ONNX model path | Make sure the model has passed basic ONNX checks. |
| `--framework=5` | Specifies that the input model framework is ONNX | Different frameworks use different values. ONNX commonly uses `5`. |
| `--output` | Output OM model prefix | The final output is a `.om` file. |
| `--input_shape` | Model input name and static input dimensions | `input` must be consistent with `input_names` in `pth2onnx.py`; dimensions must match the actual model input. |
| `--soc_version` | Target inference chip model | Must match the deployment device, such as `Ascend310P3` or `Ascend310B`. |

### 7.4 Batch Size and Input Dimensions

The OM model usually fixes the input shape specified during ATC conversion. If batch size is greater than 1, update the batch dimension consistently in `pth2onnx.py`, the ATC command, and the inference script.

- The dummy input batch size in `pth2onnx.py` must match the export target;
- The first dimension in `--input_shape` must be updated accordingly;
- Input preprocessing and memory allocation in the ACL inference script must also match the OM model input dimensions.

### 7.5 Input FP16 Optimization

In some models, OM inference may insert a Cast operation to convert FP32 input to FP16 by default. If the input node can directly accept FP16, specify `--input_fp16_nodes` during ATC conversion to reduce Cast overhead and improve inference efficiency.

```bash
atc --model=model.onnx \
    --framework=5 \
    --output=model \
    --input_shape="input:1,1,64,64" \
    --input_fp16_nodes="input" \
    --soc_version=Ascend310P3
```

After using this parameter, the `process_input` function in the inference script must also convert the input to `float16`; otherwise, input type mismatch or abnormal output may occur.

## 8. OM Offline Inference and ACL Usage

### 8.1 Basic ACL Inference Workflow

For `.om` inference, refer to the `sampleResnetQuickStart` Python example in Ascend samples. In real projects, the functions `process_input(self, input_path)` and `get_result(self)` usually need to be modified to match the model input format and output parsing logic.

A typical ACL inference workflow is:

1. `acl.init`: Initialize the ACL runtime environment. Pass a profile configuration path if needed;
2. `acl.rt.set_device` / `create_context`: Set the inference device and create the runtime context;
3. `acl.mdl.load_from_file`: Load the OM model;
4. `prepare input`: Read the input file and perform preprocessing such as resize, normalization, dimension adjustment, and dtype conversion;
5. `acl.mdl.execute`: Execute model inference;
6. `get result`: Read output memory, convert it to a numpy array, and run post-processing;
7. `release`: Release the model, dataset, memory, context, and device resources.

### 8.2 Input dtype Notes

According to the original notes, `np.save` saves data as `float64` by default in the DnCNN example, while `sampleResnetQuickStart.py` usually processes data as `float32`. If `process_input` does not explicitly convert dtype, abnormal output may occur.

```python
input_data = np.load(input_path)
input_data = input_data.astype(np.float32)  # Or np.float16 according to ATC configuration
```

### 8.3 Command-line Inference with msame

If you only need to quickly verify whether the OM model can run, use `msame` for command-line inference. This is suitable for model connectivity verification, basic performance evaluation, and comparison with custom ACL scripts.

```bash
# Example: adjust parameters according to msame documentation and model input
./msame --model ./model.om --input ./input.bin --output ./outputs
```


## 9. Performance Profiling, Accuracy Comparison, and Mixed Precision

### 9.1 PyTorch Profiler Data Collection

During online training or PyTorch inference, Ascend PyTorch Profiler can be used to collect operator latency, data transfer, and execution stream information for locating performance bottlenecks.

It is recommended to add synchronization around key inference functions and distinguish between warm-up iterations and formal measurement iterations, so that initial compilation, cache effects, and data loading do not affect the results.

### 9.2 ACL Profile Data Collection and Analysis

During OM offline inference, pass a profile configuration file to `acl.init(config_path)` to generate a profile data directory, and then use the official tools to analyze and export performance data.

```python
config_path = "./acl_profile.json"
ret = acl.init(config_path)
```

Notes:

- Make sure the profile configuration file path is correct before data collection;
- After collection, confirm that the profile directory has been generated before running performance analysis;
- Exported performance data can be used to locate operator latency, data transfer overhead, and model bottlenecks.

### 9.3 Accuracy Comparison Across Different Devices

From PyTorch to ONNX and then to OM inference, it is recommended to compare outputs under the same input data at least once. If the deviation is large, use dump files and accuracy comparison tools to locate the node that introduces the difference.

Recommended workflow:

1. Prepare a fixed input sample and ensure preprocessing is exactly the same;
2. Save outputs from PyTorch, ONNX Runtime, and OM inference separately;
3. Compare mean value, maximum error, relative error, and task-specific metrics;
4. Generate dump files if necessary and locate the difference node by node.

### 9.4 AMP Mixed Precision Training

Mixed precision training or inference may improve performance, but it does not work for all models. If AMP does not improve performance significantly, or if accuracy drops noticeably, further tuning is needed based on model operators, input/output dtype, conversion configuration, and profiler results.

```python
from torch_npu.npu import amp

# Use autocast / GradScaler according to the project training script
```

## 10. Common Errors and Solutions

This section summarizes the scattered troubleshooting notes from the original document. In practice, it is recommended to troubleshoot layer by layer in the following order: environment version -> model conversion -> input/output -> runtime permission -> performance and accuracy.

| Error / Issue | Possible Cause | Suggested Solution |
| --- | --- | --- |
| `npu-smi info` cannot detect the device | Driver not installed, insufficient permissions, abnormal NPU service, or device model mismatch. | Check driver and CANN installation first, then check root permissions, device nodes, system architecture, and hardware connection. |
| `cannot allocate memory in static TLS block` | Dynamic library loading order, such as `libgomp`, causes insufficient TLS space. | Run `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD`. Use the actual path shown in the error message. |
| `SystemError: PY_SSIZE_T_CLEAN macro must be defined for # formats` | Python version is incompatible with some compiled dependencies. | Use the verified Python 3.9 environment and reinstall dependencies. |
| Device-related error during `eval` | NPU device is not explicitly set or the model is not migrated correctly. | Add `torch.npu.set_device(0)` before inference and ensure the model and input are on the same device. |
| Error when loading weights | The device used to save the weights is inconsistent with the current device. | Load the weights with `torch.load(model_dir, map_location="cpu")` and then load them into the model. |
| `onnxsim` installation fails | `cmake` or build dependencies are outdated. | Run `pip3 install --upgrade cmake`, then reinstall `onnxsim`. |
| ATC conversion fails due to input mismatch | The input name in `--input_shape` does not match ONNX `input_names`. | Check `input_names` in `pth2onnx.py` and keep it consistent with the ATC command. |
| Abnormal inference dimensions after ATC conversion | Batch / shape settings are inconsistent among ONNX export, ATC conversion, and ACL input preprocessing. | Update dummy input, `--input_shape`, input bin / npy generation logic, and ACL memory allocation consistently. |
| Abnormal OM inference output | Input dtype does not match OM expectations, such as `float64` not converted to `float32` / `float16`. | Explicitly run `astype(np.float32)` or `astype(np.float16)` in `process_input`. |
| `soc_version` error or model cannot run | ATC target chip model does not match the deployment device. | Use `npu-smi info` to confirm the device model and update `--soc_version`. |
| Runtime measurement is unstable or too small | NPU execution is asynchronous and timing is not synchronized. | Add `torch.npu.synchronize()` before and after inference, and use warm-up iterations. |

## 11. Reference Links

The following links are organized from the original notes by usage.

| Category | Reference | Link |
| --- | --- | --- |
| Environment adaptation | PyTorch model migration and Ascend environment adaptation | [Open link](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/devguide/moddevg/ptmigr/AImpug_0001.html) |
| Performance profiling | Ascend PyTorch Profiler data collection and analysis | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000068.html) |
| Performance profiling | Configure profile generation using `acl.init()` | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/devtools/auxiliarydevtool/atlasprofiling_16_0170.html) |
| Performance profiling | Analyze the generated profile directory | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/devtools/auxiliarydevtool/atlasprofiling_16_0038.html) |
| Performance profiling | Export performance data | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/devtools/auxiliarydevtool/atlasprofiling_16_0040.html) |
| Accuracy comparison | Accuracy comparison for the same model on different devices | [Open link](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0034.html) |
| Accuracy comparison | Prepare / generate dump files | [Open link](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0018.html) |
| Accuracy comparison | Run accuracy comparison | [Open link](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/devaids/auxiliarydevtool/atlasaccuracy_16_0019.html) |
| Mixed precision | AMP mixed precision training | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000019.html) |
| Model conversion | Export `.pth` to ONNX | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_000095.html) |
| Model conversion | ATC tool introduction | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/atctool/atlasatc_16_0007.html) |
| Model conversion | ATC quick start example | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/700/quickstart/quickstart/quickstart_18_0007.html) |
| OM inference | Ascend samples: `sampleResnetQuickStart` | [Open link](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetQuickStart/python) |
| ACL inference | ACL Python development guide | [Open link](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/aclpythondevg/aclpythondevg_0000.html) |
| Command-line inference | `msame` tool | [Open link](https://gitee.com/ascend/tools/tree/master/msame) |
| Community reference | Basic OM inference workflow | [Open link](https://blog.csdn.net/fradeet/article/details/131994362) |
| Community reference | Ascend full development workflow reference | [Open link](https://blog.csdn.net/ygf666/article/details/124296461) |
