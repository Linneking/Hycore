#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于原 classification_ModelNet40/hutil.py
新增：HypHC softmax 类内三元组损失、跨类 margin 安全网、Gromov 积计算
"""

import numpy as np
import torch
import torch.nn.functional as F
from geoopt.manifolds.stereographic.math import mobius_add, dist
from models.manifolds import PoincareBall
import random

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_children_np(data, starting=1023, kmin =100, kmax=500):
    k = random.randint(kmin, kmax)
    idknn = knn(data[:,:3,:],k)
    pos_child = data[:,:,:k]
    for id in range(data.shape[0]):
        starting_point = random.randint(0, starting)
        pos_child[id,:,:] = data[id,:,idknn[id, starting_point ,:]]
    mar = 1000./k
    return mar, pos_child, k-1

def hype_triplet_losses(parent_mu, pos_child_mu, hier_margin=0.2, contr_margin=4, ball_dim=256,
                         one_child=False, opposite_hier=False):
    if one_child:
        neg_child_mu = torch.flip(parent_mu, [0])
    else:
        neg_child_mu = torch.flip(pos_child_mu, [0])

    ball = PoincareBall(c=1.0, dim=ball_dim)
    parent_norm = torch.norm(parent_mu, dim=1)
    pos_norm = torch.norm(pos_child_mu, dim=1)
    par_norm = ball.dist0(parent_mu)
    p_norm = ball.dist0(pos_child_mu)
    distance_positive = ball.dist(parent_mu, pos_child_mu)
    distance_negative = ball.dist(parent_mu, neg_child_mu)

    distance = distance_positive - distance_negative + contr_margin
    if opposite_hier:
        norm = par_norm - p_norm + hier_margin
    else:
        norm = -par_norm + p_norm + hier_margin

    triplet = torch.mean(torch.max(distance, torch.zeros_like(distance)))
    hierarch = torch.mean(torch.max(norm, torch.zeros_like(norm)))
    return parent_norm.mean(), pos_norm.mean(), distance_positive.mean(), distance_negative.mean(), triplet, hierarch

def euc_triplet_losses(parent_mu, pos_child_mu, hier_margin=2, contr_margin=4,
                        one_child=False, opposite_hier=False):
    if one_child:
        neg_child_mu = torch.flip(parent_mu, [0])
    else:
        neg_child_mu = torch.flip(pos_child_mu, [0])
    parent_norm = torch.norm(parent_mu, dim=1)
    pos_norm = torch.norm(pos_child_mu, dim=1)
    distance_positive = F.pairwise_distance(parent_mu, pos_child_mu)
    distance_negative = F.pairwise_distance(parent_mu, neg_child_mu)
    distance = distance_positive - distance_negative + contr_margin
    if opposite_hier:
        norm = parent_norm - pos_norm + hier_margin
    else:
        norm = -parent_norm + pos_norm + hier_margin
    triplet = torch.mean(torch.max(distance, torch.zeros_like(distance)))
    hierarch = torch.mean(torch.max(norm, torch.zeros_like(norm)))
    return parent_norm.mean(), pos_norm.mean(), distance_positive.mean(), distance_negative.mean(), triplet, hierarch


def cal_loss(pred, gold, smoothing=True):
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss


# ═══════════════════════════════════════════════════════════════
#  ★ 新增：Inter-Sample Hierarchy 相关函数
# ═══════════════════════════════════════════════════════════════

def gromov_product(z_i, z_j, ball):
    """
    计算两点在庞加莱球中的 Gromov 积（即双曲 LCA 到原点的距离）
    (z_i | z_j)_0 = (d_o(z_i) + d_o(z_j) - d(z_i, z_j)) / 2

    Args:
        z_i, z_j: [n_i, dim], [n_j, dim] 庞加莱球坐标
        ball: PoincareBall 实例
    Returns:
        [n_i, n_j] Gromov 积矩阵
    """
    d_i = ball.dist0(z_i)          # [n_i]
    d_j = ball.dist0(z_j)          # [n_j]
    d_ij = ball.dist(z_i, z_j)     # [n_i, n_j]
    return (d_i.unsqueeze(1) + d_j.unsqueeze(0) - d_ij) / 2.0


def hypHC_softmax_loss(parent_mu_c, w_c, tau, ball):
    """
    ★ 类内 HypHC softmax 三元组损失
    对类别 c 的 n 个 whole 嵌入，枚举所有 C(n,3) 三元组，
    用 w_c 作为 softmax 权重，强制相似度排序转化为 Gromov 积排序

    Args:
        parent_mu_c: [n, dim] 类别 c 在 batch 中的 whole 嵌入（有梯度）
        w_c:        [n, n] stop-gradient 的余弦相似度矩阵
        tau:        softmax 温度
        ball:       PoincareBall 实例
    Returns:
        scalar loss
    """
    n = parent_mu_c.size(0)
    if n < 3:
        return torch.tensor(0.0, device=parent_mu_c.device)

    # 计算所有对的 Gromov 积 [n, n]
    g_prod = gromov_product(parent_mu_c, parent_mu_c, ball)  # [n, n]

    losses = []
    # 枚举所有三元组 (i, j, k)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # 三个 Gromov 积（LCA 深度）
                g_ij = g_prod[i, j]
                g_ik = g_prod[i, k]
                g_jk = g_prod[j, k]
                g_vec = torch.stack([g_ij, g_ik, g_jk])  # [3]

                # 三个相似度权重（stop-gradient）
                w_ij = w_c[i, j]
                w_ik = w_c[i, k]
                w_jk = w_c[j, k]
                w_vec = torch.stack([w_ij, w_ik, w_jk])  # [3]

                # HypHC softmax: reward LCA depth ordering consistent with w
                soft = torch.softmax(g_vec / tau, dim=0)  # [3]
                weighted = (w_vec * soft).sum()
                w_sum = w_ij + w_ik + w_jk

                triplet_loss = w_sum - weighted
                losses.append(triplet_loss)

    return torch.stack(losses).mean()


def inter_margin_loss(parent_mu, labels, ball, margin=0.15):
    """
    跨类分离安全网（加权 Gromov 积 margin 三元组）
    对 batch 中所有可用的跨类三元组，要求同类对的 Gromov 积 > 异类对 + margin

    Args:
        parent_mu: [B, dim] batch 内所有 whole 嵌入
        labels:    [B] 对应标签
        ball:      PoincareBall 实例
        margin:    Gromov 积间隔
    Returns:
        scalar loss with weight β_inter (由调用方加权)
    """
    B = parent_mu.size(0)
    if B < 3:
        return torch.tensor(0.0, device=parent_mu.device)

    g_prod = gromov_product(parent_mu, parent_mu, ball)  # [B, B]

    losses = []
    for i in range(B):
        for j in range(i + 1, B):
            if labels[i] != labels[j]:
                continue  # 只对同类对找跨类负样本
            # (i, j) 是同类对
            for k in range(B):
                if labels[k] == labels[i]:
                    continue  # k 必须是异类
                # 同类对的 Gromov 积应大于跨类对 + margin
                loss = torch.clamp(g_prod[i, k] - g_prod[i, j] + margin, min=0)
                losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=parent_mu.device)
    return torch.stack(losses).mean()


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()