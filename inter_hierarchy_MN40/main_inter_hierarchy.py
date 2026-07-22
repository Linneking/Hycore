"""
Inter-Sample Hierarchy 训练主脚本
基于 V2.1 方案，在 HyCoRe 预训练权重基础上新增类内/跨类 inter-sample 层级

三阶段训练流程：
  阶段零：加载 HyCoRe 预训练权重 + 首次计算 w_ij
  阶段一（warmup）：仅 L_task + L_intra（v1.1 修正版），β_ic=0
  阶段二：激活 L_intra-class + L_inter，周期性重算 w_ij

用法:
  python main_inter_hierarchy.py --msg test_run --warmup_epochs 15 --K 5
"""

import argparse
import os
import sys
import csv

# 将 pointnet2_ops_lib 加入搜索路径（在根目录）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pointnet2_ops_lib'))

import logging
import datetime

# 必须在 import torch 前设置 CUDA_VISIBLE_DEVICES，否则 DataParallel 会占用所有可见 GPU
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    pass  # 由外部环境变量控制
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 默认使用 GPU 2

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from models.pointmlp import Hype_pointMLP
from utils import Logger, mkdir_p, progress_bar, save_model, save_args
from data import ModelNet40
from hutil import (hype_triplet_losses, get_children_np, cal_loss,
                   hypHC_softmax_loss, inter_margin_loss, gromov_product)
from embedding_cache import (recompute_embeddings, build_index,
                             compute_w_batch, monitor_w_stability)
from models.manifolds import PoincareBall
import geoopt
import numpy as np
import random
import sklearn.metrics as metrics
from torch.optim.lr_scheduler import CosineAnnealingLR


def parse_args():
    parser = argparse.ArgumentParser('Inter-Hierarchy Training')
    # 基础参数
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, default='inter_hierarchy', help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=40, help='batch size')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, default=22, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='workers')

    # HyCoRe 预训练权重路径
    parser.add_argument('--pretrained', type=str,
                        default='checkpoints/InterHierarchy-A3_no_rhier-20260626192840/best_checkpoint.pth',
                        help='path to pretrained checkpoint')

    # Inter-Sample Hierarchy 参数
    parser.add_argument('--alpha', type=float, default=0.01, help='L_intra 权重')
    parser.add_argument('--beta_ic', type=float, default=0.05, help='L_intra-class 目标权重')
    parser.add_argument('--beta_inter', type=float, default=0.005, help='L_inter 权重（安全网）')
    parser.add_argument('--tau', type=float, default=0.1, help='HypHC softmax 温度')
    parser.add_argument('--margin_inter', type=float, default=0.15, help='跨类 margin')
    parser.add_argument('--r_part', type=float, default=0.5, help='part 嵌入最大范数（v1.1）')
    parser.add_argument('--warmup_epochs', type=int, default=15, help='阶段一 warmup 轮数')
    parser.add_argument('--K', type=int, default=5, help='w_ij 重算周期（epoch）')
    parser.add_argument('--classes_per_batch', type=int, default=5, help='每 batch 选几类')
    parser.add_argument('--samples_per_class', type=int, default=8, help='每类采样几个')

    return parser.parse_args()


def load_pretrained(net, path, printf):
    """加载 HyCoRe 预训练权重（兼容 DataParallel 的 module. 前缀）"""
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['net']
    # 剥离 module. 前缀（checkpoint 是 DataParallel 保存的，我们加载到裸模型）
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=True)
    printf(f"[Pretrained] Loaded from epoch {checkpoint['epoch']}, "
           f"best_test_acc={checkpoint.get('best_test_acc', 'N/A')}%")


def sample_class_balanced_batch(train_set, classes_per_batch, samples_per_class, class_counts):
    """
    类平衡采样：生成 batch 索引列表
    Returns:
        sampled_indices: list of global indices
    """
    all_classes = sorted(class_counts.keys())
    selected_classes = random.sample(all_classes, min(classes_per_batch, len(all_classes)))

    sampled_indices = []
    for c in selected_classes:
        c_count = class_counts[c]
        n_sample = min(samples_per_class, c_count)
        # 从该类中随机采样 n_sample 个样本
        # 注意：这里需要知道每个类对应的全局索引范围
        # 简化实现：回到 data.py 的 label 数组来获取
        pass  # 将在 __init__ 时建立映射

    return sampled_indices, selected_classes


def train_one_epoch(net, trainloader, optimizer, criterion, ball, args, epoch,
                    Z_norm, class_to_indices, beta_ic_current, device):
    """
    单轮训练（含 inter-sample hierarchy 损失）
    """
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()

    # 统计各损失分量
    sum_std_loss = 0
    sum_h_loss = 0
    sum_ic_loss = 0
    sum_inter_loss = 0
    n_batches = 0

    for batch_idx, (data, label, global_idx) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        global_idx = global_idx.tolist() if isinstance(global_idx, torch.Tensor) else global_idx
        data = data.permute(0, 2, 1)

        optimizer.zero_grad()

        # ── Intra-sample: part 采样 ──
        mar_par, data_wp, n_points = get_children_np(data.clone(), kmin=800, kmax=1024)
        mar, pos_data, _ = get_children_np(data.clone(), starting=n_points, kmin=200, kmax=600)

        pos_mu, _ = net(pos_data, emb=True)
        parent_mu, logits = net(data)

        # ── L_intra（v1.1 修正版）──
        pn, posn, pd, nd, t_loss, h_loss = hype_triplet_losses(
            parent_mu, pos_mu, hier_margin=mar, contr_margin=4, ball_dim=256)
        std_loss = criterion(logits, label)

        # ── L_intra-class + L_inter（仅阶段二）──
        ic_loss = torch.tensor(0.0, device=device)
        inter_loss = torch.tensor(0.0, device=device)

        if beta_ic_current > 0 and Z_norm is not None and epoch >= args.warmup_epochs:
            unique_classes = torch.unique(label).tolist()

            for c in unique_classes:
                # 从缓存取 w_c
                w_c, global_ids_c = compute_w_batch(
                    Z_norm, class_to_indices, int(c), global_idx)

                if w_c is not None and w_c.shape[0] >= 3:
                    # 找到类别 c 在 batch 中的局部索引
                    mask_c = (label == c)
                    parent_mu_c = parent_mu[mask_c]  # [n_c, dim]

                    if parent_mu_c.size(0) >= 3:
                        w_c_dev = w_c.to(device)
                        ic_loss += hypHC_softmax_loss(parent_mu_c, w_c_dev, args.tau, ball)

            # 跨类安全网
            if args.beta_inter > 0:
                inter_loss = inter_margin_loss(parent_mu, label, ball, margin=args.margin_inter)

        # ── 总损失 ──
        loss = std_loss + args.alpha * (t_loss + h_loss) \
               + beta_ic_current * ic_loss \
               + args.beta_inter * inter_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # 统计
        train_loss += loss.item()
        sum_std_loss += std_loss.item()
        sum_h_loss += (t_loss + h_loss).item()
        sum_ic_loss += ic_loss.item()
        sum_inter_loss += inter_loss.item()
        n_batches += 1

        preds = logits.max(dim=1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        total += label.size(0)
        correct += preds.eq(label).sum().item()

        if batch_idx % 20 == 0:
            progress_bar(batch_idx, len(trainloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)

    return {
        "loss": float("%.3f" % (train_loss / max(1, n_batches))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost,
        "std_loss": float("%.4f" % (sum_std_loss / max(1, n_batches))),
        "h_loss": float("%.4f" % (sum_h_loss / max(1, n_batches))),
        "ic_loss": float("%.4f" % (sum_ic_loss / max(1, n_batches))),
        "inter_loss": float("%.4f" % (sum_inter_loss / max(1, n_batches))),
    }


def validate(net, testloader, criterion, device):
    """验证（与原始 HyCoRe 一致）"""
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label, _) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            _, logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = 'cuda'
    torch.cuda.set_device(0)  # 使用 CUDA_VISIBLE_DEVICES 映射后的唯一设备
    if args.seed is not None:
        print('Setting seed --> ', args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    message = f"-{args.msg}{time_str}"
    args.checkpoint = 'experiments/B1_v1/checkpoints/B1' + message
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    printf(f"args: {args}")

    # ── 模型 ──
    printf('==> Building model..')
    net = Hype_pointMLP()
    net = net.to(device)
    cudnn.benchmark = True

    # ── 加载预训练权重（裸模型，不需要 DataParallel）──
    printf(f'==> Loading pretrained weights from {args.pretrained}')
    load_pretrained(net, args.pretrained, printf)
    # 不包装 DataParallel——单 GPU 直接裸模型运行，避免 GPU 分配问题

    # ── 数据加载 ──
    printf('==> Preparing data..')
    train_set = ModelNet40(partition='train', num_points=args.num_points)
    test_set = ModelNet40(partition='test', num_points=args.num_points)
    class_counts = train_set.get_class_counts()

    train_loader = DataLoader(train_set, num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    # 用于全量重算的 DataLoader（不 shuffle，需要 idx）
    full_train_loader = DataLoader(train_set, num_workers=args.workers,
                                   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── 优化器 ──
    optimizer = geoopt.optim.RiemannianSGD(
        net.parameters(), lr=args.learning_rate, momentum=0.9,
        weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr)

    # ── 损失函数 ──
    criterion = cal_loss
    ball = PoincareBall(c=1.0, dim=256)

    # ── 阶段零：首次计算 w_ij（使用预训练权重）──
    printf('\n' + '=' * 60)
    printf('Phase 0: Initial w_ij computation with pretrained weights')
    printf('=' * 60)
    Z_all, all_labels = recompute_embeddings(net, full_train_loader, device, verbose=True)
    Z_norm, class_to_indices = build_index(Z_all, all_labels, verbose=True)
    Z_norm_prev = Z_norm.clone()

    printf(f'w_ij cache ready. Memory: {Z_norm.element_size() * Z_norm.nelement() / 1024 / 1024:.1f} MB')

    # ── 训练状态初始化 ──
    best_test_acc = 0.
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")

    save_args(args)
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="InterHierarchy")
    logger.set_names(["Epoch-Num", 'Learning-Rate',
                      'Train-Loss', 'Train-acc-B', 'Train-acc',
                      'Valid-Loss', 'Valid-acc-B', 'Valid-acc',
                      'StdLoss', 'HLoss', 'ICLoss', 'InterLoss'])

    printf(f"\n{'='*30} Training Start {'='*30}")
    printf(f"Phase 1: warmup {args.warmup_epochs} epochs (intra only, β_ic=0)")
    printf(f"Phase 2: inter-sample hierarchy activated (β_ic→{args.beta_ic}, K={args.K})")

    # ── CSV 保存每 epoch 指标 ──
    metrics_csv = os.path.join('experiments/B1_v1', 'metrics.csv')
    csv_header = ['epoch', 'lr', 'train_loss', 'train_acc', 'train_acc_avg',
                  'test_loss', 'test_acc', 'test_acc_avg',
                  'std_loss', 'h_loss', 'ic_loss', 'beta_ic']
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    for epoch in range(args.epoch):
        # ── 计算当前 β_ic（warmup 线性调度）──
        if epoch < args.warmup_epochs:
            beta_ic_current = 0.0
        else:
            # 从 warmup_epochs 到 warmup_epochs+10 线性增长
            warmup_steps = min(10, args.epoch - args.warmup_epochs)
            progress = min(1.0, (epoch - args.warmup_epochs) / warmup_steps)
            beta_ic_current = args.beta_ic * progress

        # ── 周期性重算 w_ij ──
        if epoch >= args.warmup_epochs and epoch % args.K == 0 and epoch > args.warmup_epochs:
            printf(f'\n[Epoch {epoch}] Recomputing w_ij...')
            Z_all_new, all_labels_new = recompute_embeddings(
                net, full_train_loader, device, verbose=False)
            Z_norm_new, class_to_indices_new = build_index(
                Z_all_new, all_labels_new, verbose=False)

            # 监控稳定性
            tau = monitor_w_stability(Z_norm, Z_norm_new)
            printf(f'  w_ij Kendall τ = {tau:.4f} (vs previous)')

            Z_norm = Z_norm_new
            class_to_indices = class_to_indices_new

        # ── 训练 ──
        train_out = train_one_epoch(
            net, train_loader, optimizer, criterion, ball, args, epoch,
            Z_norm, class_to_indices, beta_ic_current, device)

        # ── 验证 ──
        test_out = validate(net, test_loader, criterion, device)
        scheduler.step()

        # ── 记录最佳结果 ──
        best_test_acc = max(best_test_acc, test_out["acc"])
        best_train_acc = max(best_train_acc, train_out["acc"])
        best_test_acc_avg = max(best_test_acc_avg, test_out["acc_avg"])
        best_train_acc_avg = max(best_train_acc_avg, train_out["acc_avg"])
        best_test_loss = min(best_test_loss, test_out["loss"])
        best_train_loss = min(best_train_loss, train_out["loss"])

        is_best = test_out["acc"] >= best_test_acc and epoch > 0

        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )

        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"],
                       train_out.get("std_loss", 0), train_out.get("h_loss", 0),
                       train_out.get("ic_loss", 0), train_out.get("inter_loss", 0)])

        # ── 每 epoch 打印 + 写 CSV ──
        lr = optimizer.param_groups[0]['lr']
        printf(f'\nEpoch {epoch:3d} LR={lr:.4f} β_ic={beta_ic_current:.4f}')
        printf(f'  Train: loss={train_out["loss"]} acc={train_out["acc"]}% acc_avg={train_out["acc_avg"]}%')
        printf(f'  Test:  loss={test_out["loss"]} acc={test_out["acc"]}% acc_avg={test_out["acc_avg"]}%')
        printf(f'  Losses: task={train_out.get("std_loss",0):.4f} '
               f'h_ier={train_out.get("h_loss",0):.4f} '
               f'L_ic={train_out.get("ic_loss",0):.4f} '
               f'Best OA={best_test_acc}%')

        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, lr, train_out["loss"], train_out["acc"], train_out["acc_avg"],
                            test_out["loss"], test_out["acc"], test_out["acc_avg"],
                            train_out.get("std_loss",0), train_out.get("h_loss",0),
                            train_out.get("ic_loss",0), beta_ic_current])

    logger.close()

    printf(f"\n{'='*30} Final Results {'='*30}")
    printf(f"Best Train acc: {best_train_acc}% | Best Test acc: {best_test_acc}%")
    printf(f"Best Train acc_avg: {best_train_acc_avg}% | Best Test acc_avg: {best_test_acc_avg}%")
    printf(f"Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}")


if __name__ == '__main__':
    main()