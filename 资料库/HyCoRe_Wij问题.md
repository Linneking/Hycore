好问题。`w_ij` 是整个框架的"信号源"，必须精确定义。让我拆解全链路。

---

## w_ij 计算：完整技术路径

### 1. 数据来源：z_whole 的确切定义

看 `Hype_Model.forward()` 的代码（`pointmlp.py` L420-430）：

```python
x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)  # [B, 1024]  欧式池化特征
x = self.proj(x)       # Linear(1024→512) + BN + ReLU  → [B, 512]
x = self.manifold.expmap0(x)  # expmap_0: R^512 → B^512
mu = self.emb(x)       # MobiusLayer(512→256)  → [B, 256]   ← 这就是 z_whole
```

所以 `z_whole` 是对 `mu`（Möbius 层的输出）**取 `.detach()`** 后的 256 维庞加莱球向量。

关键问题：**`parent_mu` 已经在训练循环里了**。看 `main_pointmlp_hycore.py` L225：

```python
parent_mu, logits = net(data)   # parent_mu 就是 mu，shape [B, 256]
```

所以每个 batch 的 `parent_mu` 天然可用，无需额外前向。

---

### 2. w_ij 的数学定义

根据 v2.1 §3.4：

$$w_{ij} = \frac{\langle z_i^{\text{whole}}, z_j^{\text{whole}} \rangle}{\|z_i^{\text{whole}}\| \cdot \|z_j^{\text{whole}}\|}$$

即 **庞加莱球坐标向量上的欧氏余弦相似度**。这不是双曲距离——它度量的是两个 whole 嵌入在球内的**方向一致性**（是否指向同一"语义锥"），而非它们在球上的远近。

两个样本即使范数不同（一个靠内一个靠外），只要指向同一方向，w_ij 就趋近 1。这正是我们需要的——同类样本应当在方向上聚集。

---

### 3. 全局 w_ij 的存储策略

ModelNet40 训练集有 9843 个样本。全量 9843² 的 float32 矩阵约 388 MB，可以但不优雅。

**推荐方案：存储归一化向量 Z_norm，按需计算**

```
阶段：周期性重算（每 K epoch 触发一次）

Step 1: 全量前向
  遍历整个训练集（no_grad），对每个样本：
    data → net(data) → parent_mu → .detach().cpu()
  结果：Z_all = [N_train, 256]  全部 z_whole 向量

Step 2: L2 归一化
  Z_norm = F.normalize(Z_all, p=2, dim=1)   → [9843, 256]
  约 10 MB，完全可接受

Step 3: 按类分组索引
  class_to_indices = {0: [0,5,12,...], 1: [3,7,9,...], ...}
  为每个类维护其样本在 Z_norm 中的行索引列表
```

**训练时按需计算**：

```python
# batch 内 class c 的样本在 Z_norm 中的全局索引
global_ids_c = class_to_indices[c] ∩ batch_global_indices
# 取出对应的归一化向量
z_c = Z_norm[global_ids_c]          # [n_c, 256]
# 计算余弦相似度矩阵
w_c = z_c @ z_c.T                   # [n_c, n_c], 这就是 w_ij
```

对于每类 5-8 个样本，这个矩阵乘法是 `8×256 @ 256×8 = 8×8`，完全可以忽略不计。

核心 Trick：**全局索引映射**。`DataLoader` 默认不返回样本的全局索引，需要修改 `ModelNet40.__getitem__` 使其额外返回 `item`（即该样本在数据集中的序号），然后在 batch 中携带这个索引，用于从 `Z_norm` 中按行取向量。

---

### 4. 为什么 w_ij 需要 detach（以及这天然成立）

`Z_norm` 在周期性重算时通过 `.detach()` 切断梯度。训练时 `w_c = z_c @ z_c.T` 中的 `z_c` 来自 `Z_norm`（一个已 detach 的张量），因此 w_ij 天然不在计算图中。`L_intra-class` 的梯度仅通过 Gromov 积流向当前的 `z_whole`（正在被优化的嵌入），不回流到 w_ij。

这避免了"损失用自己的输出监督自己"的自举不稳定。

---

### 5. 周期性重算的触发机制

```python
if epoch % K == 0:
    Z_norm, class_to_indices = recompute_embeddings(net, train_loader, device)
    # 可选：计算 Kendall τ 监控稳定性
    if prev_Z_norm is not None:
        tau = kendall_tau(prev_Z_norm, Z_norm, sample=5000)
        print(f"[Epoch {epoch}] w_ij Kendall τ = {tau:.4f}")
    prev_Z_norm = Z_norm.clone()
```

`recompute_embeddings()` 就是一个全量 eval forward，收集所有 `parent_mu`，归一化，建索引。

---

### 6. 端到端数据流总结

```
═════════════════════════════════════════════════════════
 每 K epoch 触发一次
═════════════════════════════════════════════════════════
 for each sample in trainset (no_grad):
     data → net(data) → parent_mu.detach()  → 存入列表
 → Z_all = stack(all parent_mu)          [9843, 256]
 → Z_norm = F.normalize(Z_all)           [9843, 256]  存入内存
 → class_to_indices = build_index(labels)              存入内存

═════════════════════════════════════════════════════════
 每个 training batch
═════════════════════════════════════════════════════════
 # 原有逻辑
 parent_mu, logits = net(data)           ← 这个 parent_mu 有梯度
 pos_mu, _ = net(pos_data, emb=True)
 L_intra = hype_hier_v11(parent_mu, pos_mu) + R_contr

 # 新增 inter 逻辑
 for class_c in unique_classes_in_batch:
     global_ids = class_to_indices[c] ∩ batch_global_ids
     if len(global_ids) >= 3:
         z_c_norm = Z_norm[global_ids]           # detach 后的归一化向量
         w_c = z_c_norm @ z_c_norm.T             # [n_c, n_c] 余弦相似度
         L_ic += hypHC_softmax(parent_mu[mask_c], w_c, tau)
                                                        ↑
                                            这里有梯度，流向当前的 parent_mu
         L_inter += cross_class_margin(...)              
```

关键点：**w_ij 来自历史快照（detach），Gromov 积来自当前嵌入（有梯度）**——HypHC softmax 用"旧的相似度排序信号"来规约"当前的几何布局"。

---

### 7. 首轮 w_ij 的初始化时机

v2.1 的训练流程中，w_ij 的首次计算发生在**阶段一（v1.1 适配）结束后、阶段二开始时**：

```
阶段零（可选）：HyCoRe 预训练
阶段一（10-15 epoch）：L_task + L_intra_v1.1，β_ic=0, β_inter=0
    ↓ 
    此时 whole 嵌入已经在"自由范数"设定下初步成型
    ↓
阶段二开始前：首次全量前向 → 计算 Z_norm → 建索引
    ↓
阶段二：激活 L_intra-class + L_inter，β_ic 从 0 warm-up
```

如果跳过预训练（从随机初始化开始），阶段一需要更长（20-30 epoch），给编码器足够时间建立有意义的特征分布。

---

你对这个方案满意吗？特别是以下几点我想确认：

1. **w_ij 用欧氏余弦相似度（庞加莱球坐标上的）**，而非双曲距离——是否符合你的预期？
2. **存储 Z_norm（~10MB）+ 按类建索引**，而非全量 w_ij 矩阵（~388MB）——OK？
3. `DataLoader` 需要改造以返回全局样本索引——是否接受这个改动？