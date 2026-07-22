#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3       # 使用 GPU 3（物理 index 3）
cd /mnt/BBB5090/jhr/HyCoRe/inter_hierarchy_MN40
rm -rf __pycache__ models/__pycache__ models/manifolds/__pycache__ models/ops/__pycache__
/mnt/BBB5090/jhr/anaconda3/envs/hycore/bin/python3 main_inter_hierarchy.py \
    --msg B1_v1 --alpha 0.01 --beta_ic 0.05 --beta_inter 0.0 --warmup_epochs 0 \
    --epoch 200 --K 5 --seed 42 --batch_size 20