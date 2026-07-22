"""
全局嵌入缓存管理器
负责：全量前向收集 z_whole → L2 归一化 → 按类建索引 → 按需计算 w_ij
内存占用：~20 MB（Z_norm + 索引），远小于全量 388 MB w_ij 矩阵
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.stats import kendalltau


def recompute_embeddings(net, dataloader, device, verbose=True):
    """
    全量遍历训练集（no_grad），收集所有样本的 z_whole（parent_mu）

    Args:
        net:        模型（eval 模式）
        dataloader: 训练集 DataLoader（需返回 (data, label, idx)）
        device:     cuda 设备
        verbose:    是否打印进度条

    Returns:
        Z_all:   [N_train, dim] 所有 whole 嵌入（庞加莱球坐标，detach 后）
        labels:  [N_train] 对应标签
    """
    net.eval()
    all_mu = []
    all_labels = []

    iterator = tqdm(dataloader, desc="Recomputing embeddings") if verbose else dataloader
    with torch.no_grad():
        for data, label, _ in iterator:
            data = data.to(device)
            data = data.permute(0, 2, 1)  # [B, 3, 1024]
            parent_mu, _ = net(data)       # parent_mu: [B, dim]
            all_mu.append(parent_mu.detach().cpu())
            all_labels.append(label.cpu())

    Z_all = torch.cat(all_mu, dim=0)          # [N, dim]
    labels = torch.cat(all_labels, dim=0).squeeze()  # [N]

    if verbose:
        print(f"[EmbeddingCache] Collected {Z_all.shape[0]} embeddings, dim={Z_all.shape[1]}")
    return Z_all, labels


def build_index(Z_all, labels, verbose=True):
    """
    对 Z_all 做 L2 归一化，并按类建索引

    Args:
        Z_all:  [N, dim] whole 嵌入
        labels: [N] 标签

    Returns:
        Z_norm:          [N, dim] L2 归一化后的向量
        class_to_indices: dict[int, list[int]] 类别 → 全局样本索引列表
    """
    Z_norm = F.normalize(Z_all, p=2, dim=1)

    class_to_indices = {}
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    for c in np.unique(labels_np):
        c = int(c)
        indices = np.where(labels_np == c)[0].tolist()
        class_to_indices[c] = indices

    if verbose:
        print(f"[EmbeddingCache] Built index for {len(class_to_indices)} classes")
        for c in sorted(class_to_indices.keys()):
            print(f"  Class {c:2d}: {len(class_to_indices[c])} samples")

    return Z_norm, class_to_indices


def compute_w_batch(Z_norm, class_to_indices, class_c, global_ids_in_batch):
    """
    从缓存中取出类别 c 在 batch 中的样本归一化向量，计算余弦相似度矩阵 w_c

    Args:
        Z_norm:              [N, dim] 全局 L2 归一化向量
        class_to_indices:    dict[int, list[int]]
        class_c:             当前类别
        global_ids_in_batch: 当前 batch 中所有样本的全局索引集合（set 或 list）

    Returns:
        w_c:           [n_c, n_c] 余弦相似度矩阵（stop-gradient，来自缓存快照）
        global_ids_c:  类别 c 在当前 batch 中的全局索引列表
    """
    all_ids_c = set(class_to_indices.get(class_c, []))
    batch_ids_set = set(global_ids_in_batch)
    global_ids_c = sorted(all_ids_c & batch_ids_set)

    if len(global_ids_c) < 2:
        return None, global_ids_c

    z_c_norm = Z_norm[global_ids_c]  # [n_c, dim]
    w_c = z_c_norm @ z_c_norm.T      # [n_c, n_c] 余弦相似度矩阵
    return w_c, global_ids_c


def monitor_w_stability(Z_norm_prev, Z_norm_curr, sample_size=5000):
    """
    监控两次重算间 w_ij 排名的稳定性（Kendall τ）

    Args:
        Z_norm_prev: 上一次重算的归一化向量 [N, dim]
        Z_norm_curr: 当前重算的归一化向量 [N, dim]
        sample_size: 随机采样多少对来计算 τ（全量 N² 太大）

    Returns:
        tau: Kendall τ 相关系数（-1~1，1 表示完全一致）
    """
    N = Z_norm_prev.shape[0]
    # 随机采样样本对
    idx_i = np.random.randint(0, N, sample_size)
    idx_j = np.random.randint(0, N, sample_size)

    z_i_prev = Z_norm_prev[idx_i]  # [sample, dim]
    z_j_prev = Z_norm_prev[idx_j]
    z_i_curr = Z_norm_curr[idx_i]
    z_j_curr = Z_norm_curr[idx_j]

    w_prev = (z_i_prev * z_j_prev).sum(dim=1).numpy()  # [sample]
    w_curr = (z_i_curr * z_j_curr).sum(dim=1).numpy()

    tau, _ = kendalltau(w_prev, w_curr)
    return tau


def compute_full_w_matrix(Z_norm):
    """
    计算全量 w_ij 矩阵（用于可视化和评估，非训练）

    Args:
        Z_norm: [N, dim] 归一化向量

    Returns:
        W: [N, N] 余弦相似度矩阵
    """
    return (Z_norm @ Z_norm.T).numpy()