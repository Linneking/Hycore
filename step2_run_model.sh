#!/bin/bash
set -e

# =============================================
# Step 2: Run PointMLP + HyCoRe on ModelNet40
# 官方默认配置: epoch=300, batch_size=32, workers=8
# 进度条已启用（在 main_pointmlp_hycore.py 中已取消注释）
# =============================================

source /A432/jhr/anaconda3/etc/profile.d/conda.sh
conda activate hycore

# Set CUDA
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
# 使用全部 4 张 RTX 4090D（不限制 CUDA_VISIBLE_DEVICES）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Go to classification_ModelNet40 dir (required for Python imports)
cd /A432/jhr/HyCoRe/classification_ModelNet40

echo "============================================"
echo "Starting PointMLP + HyCoRe Training"
echo "============================================"
echo "配置:"
echo "  GPU         = 4x RTX 4090D (DataParallel)"
echo "  epoch       = 300 (官方默认)"
echo "  batch_size  = 128 (total, 32/GPU x4)"
echo "  workers     = 16"
echo "  seed        = 4780"
echo "  msg         = Offv_pointmlp_hycore_var (官方默认)"
echo "============================================"
echo "Date: $(date)"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU names:"
python -c 'import torch; [print(f"  GPU {i}: {torch.cuda.get_device_name(i)}") for i in range(torch.cuda.device_count())]'
echo "============================================"

# 4 卡完整训练
python main_pointmlp_hycore.py \
    --seed=4780 \
    --workers=16 \
    --batch_size=128 \
    --msg=Offv_pointmlp_hycore_var

echo ""
echo "============================================"
echo "完整训练完成!"
echo "============================================"
echo "Date: $(date)"