#!/bin/bash
# 阶段 A 实验：三组平行对照，利用 4 张 RTX 5090 并行
# A1: Baseline warmup (GPU 0, alpha=0.01)
# A2: Relaxed intra  (GPU 1, alpha=0.001)
# A3: No R_hier      (GPU 2, alpha=0)
set -e
PY="/mnt/BBB5090/jhr/anaconda3/envs/hycore/bin/python3"
cd "$(dirname "$0")"

echo "Launching Stage A experiments on GPUs 0,1,2..."

CUDA_VISIBLE_DEVICES=0 nohup $PY main_inter_hierarchy.py \
    --msg A1_baseline_warmup --alpha 0.01 --warmup_epochs 30 --epoch 30 --K 5 --seed 42 \
    > /tmp/A1_baseline.log 2>&1 &
echo "A1 (GPU 0, alpha=0.01) PID: $!"

CUDA_VISIBLE_DEVICES=1 nohup $PY main_inter_hierarchy.py \
    --msg A2_relaxed_intra --alpha 0.001 --warmup_epochs 30 --epoch 30 --K 5 --seed 42 \
    > /tmp/A2_relaxed.log 2>&1 &
echo "A2 (GPU 1, alpha=0.001) PID: $!"

CUDA_VISIBLE_DEVICES=2 nohup $PY main_inter_hierarchy.py \
    --msg A3_no_rhier --alpha 0.0 --warmup_epochs 30 --epoch 30 --K 5 --seed 42 \
    > /tmp/A3_no_rhier.log 2>&1 &
echo "A3 (GPU 2, alpha=0) PID: $!"

echo "Done. Monitor: tail -f /tmp/A1_baseline.log /tmp/A2_relaxed.log /tmp/A3_no_rhier.log"