#!/bin/bash
set -e

# =============================================
# Step 1: Environment Setup for HyCoRe
# =============================================
# System: Ubuntu 20.04, RTX 4090D, CUDA 11.8
# =============================================

# Initialize conda
source /A432/jhr/anaconda3/etc/profile.d/conda.sh

# Remove existing hycore env if present
conda deactivate 2>/dev/null || true
conda env remove -n hycore -y 2>/dev/null || true

# Create conda environment with Python 3.8 (newer than 3.7 for better CUDA 11.8 compat)
conda create -n hycore python=3.8 -y

# Activate environment
conda activate hycore

# Set CUDA_HOME for compilation
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# Install PyTorch 2.0.1 with CUDA 11.8 support (RTX 4090 requires CUDA >= 11.8, sm_89)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2

# Install geoopt for hyperbolic space operations
pip install geoopt

# Install pointnet2_ops_lib (CUDA extension)
# Need to set TORCH_CUDA_ARCH_LIST for RTX 4090 (sm_89)
export TORCH_CUDA_ARCH_LIST="8.9"
cd /A432/jhr/HyCoRe/pointnet2_ops_lib
pip install .
cd /A432/jhr/HyCoRe

echo ""
echo "============================================"
echo "Environment 'hycore' setup complete!"
echo "============================================"
echo "Verifying installation..."
conda run -n hycore python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"

conda run -n hycore python -c "
import pointnet2_ops
print('pointnet2_ops imported OK')
"

conda run -n hycore python -c "
import geoopt
print('geoopt imported OK')
"

conda run -n hycore python -c "
import h5py, einops, sklearn, tqdm
print('All basic deps OK')
"

echo "All checks passed!"