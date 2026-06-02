#!/bin/bash
set -e
ENV_NAME="PeCoLab-py312"
PYTHON_VERSION="3.12"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# ------------------------------------------------------------------
# Function: Install PyTorch (automatically try multiple domestic mirrors)
# ------------------------------------------------------------------
install_torch() {
    # ----------------------------------------------------------------
    # Mirror notes:
    #   NJU mirror    - fully mirrors official Simple API directory, supports --index-url
    #                   need --extra-index-url for common dependencies (filelock, etc.)
    #   Aliyun mirror - static directory, does NOT support --index-url, must use --find-links (-f)
    #                   pip resolves versions from PYPI_MIRROR, then finds wheel files from Aliyun directory
    #   Official      - final fallback, slow but complete
    # ----------------------------------------------------------------

    echo ">>> [1/3] Attempting NJU mirror (--index-url mode)..."
    if pip install torch==2.9.0 torchvision==0.24.0 \
        --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu128 \
        --extra-index-url "$PYPI_MIRROR" \
        --timeout 60 --retries 2; then
        echo "✓ NJU mirror installation succeeded"
        return 0
    fi
    echo "✗ NJU mirror failed, trying next..."

    echo ">>> [2/3] Attempting Aliyun mirror (--find-links mode)..."
    if pip install torch==2.9.0 torchvision==0.24.0 \
        --index-url "$PYPI_MIRROR" \
        --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128 \
        --timeout 60 --retries 2; then
        echo "✓ Aliyun mirror installation succeeded"
        return 0
    fi
    echo "✗ Aliyun mirror failed, trying official source..."

    echo ">>> [3/3] Using official source (final fallback)..."
    if pip install torch==2.9.0 torchvision==0.24.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url "$PYPI_MIRROR" \
        --timeout 120 --retries 3; then
        echo "✓ Official source installation succeeded"
        return 0
    fi

    echo "Error: All mirrors failed to install PyTorch. Please check your network or install manually."
    exit 1
}

install_torch

# ------------------------------------------------------------------
# Install other dependencies
# ------------------------------------------------------------------
echo "Installing other Python dependencies..."
# svrss need
pip install -i "$PYPI_MIRROR" \
    numpy pillow scikit-image tensorboard 
# svrss/app need
pip install -i "$PYPI_MIRROR" \
    pandas gradio matplotlib plotly einops h5py
# prf_cuda operator need
pip install -i "$PYPI_MIRROR" setuptools==69.5.1

python setup.py develop --with-cuda
echo "✓ Development environment setup completed. Activation command: conda activate $ENV_NAME"
