# HyCoRe inter_hierarchy_MN40 vs HPCS (Onghena) 损失函数对比分析

## 一、HPCS 原始 HypHC 损失（针对点的层次聚类）

### 代码位置
`资料库/HPCS/hpcs/loss/ultrametric_loss.py` → `MetricHyperbolicLoss.compute_hyp()`

### 数学形式

```python
# 1. 获取三元组 (anchor, positive, negative) — 同类/异类配对
anchor_idx, positive_idx, negative_idx = hyp_indices_tuple

# 2. 相似度：当前编码器的余弦相似度（有梯度）
wij = mat_sim[anchor_idx, positive_idx]  # 同类点
wik = mat_sim[anchor_idx, negative_idx]  # 异类点
wjk = mat_sim[positive_idx, negative_idx]

# 3. LCA 深度：双曲 LCA 到原点的距离（hyp_lca 返回 hyp_dist_o(proj)）
dij = hyp_lca(e1, e2, return_coord=False)  # 归一化后的嵌入
dik = hyp_lca(e1, e3, return_coord=False)
djk = hyp_lca(e2, e3, return_coord=False)

# 4. HypHC softmax 核心公式（Chami et al. NeurIPS 2020）
sim_triplet = [wij, wik, wjk]       # 3 个相似度
lca_triplet = [dij, dik, djk]        # 3 个 LCA 深度
weights = softmax(lca_triplet / τ)   # 在 LCA 深度上做 softmax 竞争
w_ord = Σ(sim_triplet * weights)     # 加权相似度

loss = Σ(sim_triplet) - w_ord + mat_sim.mean()
#      ↑ 常数基线  ↑ 奖励项      ↑ 全局正则化
```

### 语义
- **被嵌入的实体**：单个点云内的各个**点**
- **相似度来源**：编码器**当前输出**的余弦相似度（有梯度，端到端学习）
- **三元组构造**：同类点对 + 异类第三点（triplet margin miner）
- **目标**：让同类点的 LCA 更深（先合并），异类点的 LCA 更浅（后合并）

---

## 二、HyCoRe inter_hierarchy_MN40 的 HypHC 损失（针对实例的层次聚类）

### 代码位置
`inter_hierarchy_MN40/hutil.py` → `hypHC_softmax_loss()`

### 数学形式

```python
# 1. 类内所有 C(n,3) 三元组枚举（同类样本）
for i, j, k in combinations:

    # 2. 三个 Gromov 积（≡ LCA 深度）
    g_ij = gromov_product(z_i, z_j)  # = (d_o(i)+d_o(j)-d(i,j))/2
    g_ik = gromov_product(z_i, z_k)
    g_jk = gromov_product(z_j, z_k)

    # 3. 相似度：预计算并 stop-gradient 的余弦相似度
    w_ij = w_cache[i, j]  # detached!
    w_ik = w_cache[i, k]
    w_jk = w_cache[j, k]

    # 4. HypHC softmax（与原始公式完全一致）
    soft = softmax([g_ij, g_ik, g_jk] / τ)
    weighted = w_ij * soft[0] + w_ik * soft[1] + w_jk * soft[2]
    w_sum = w_ij + w_ik + w_jk

    loss = w_sum - weighted
```

### 语义
- **被嵌入的实体**：同类多个**样本的 whole 嵌入**（整个 3D 对象）
- **相似度来源**：预训练编码器的余弦相似度，**stop-gradient**，周期性重算
- **三元组构造**：类内枚举所有 C(n,3) 组合（纯类内，无锚点/正负概念）
- **目标**：让更相似的同类实例的 LCA 更深，形成类内亚型树形层级

---

## 三、核心差异对比表

| 维度 | HPCS 原始 (Onghena) | HyCoRe inter_hierarchy_MN40 |
|---|---|---|
| **聚类对象** | 单点云内的**点** (point-level) | 同类多个**样本实例** (instance-level) |
| **相似度 wij** | 编码器当前输出的余弦相似度（**有梯度**） | 预计算的余弦相似度（**stop-gradient**） |
| **wij 更新方式** | 每次 forward 实时计算 | 周期性重算（每 K epoch） |
| **三元组构造** | Triplet miner: (anchor, positive=same, negative=diff) | 类内全枚举 C(n,3) |
| **跨类处理** | 三元组中的 negative 自然引入跨类信号 | 另设 `L_inter` margin 安全网处理跨类 |
| **LCA 计算** | `hyp_lca()` → 几何方法（等距变换+反射）→ `hyp_dist_o(proj)` | `gromov_product()` → `(d_o(i)+d_o(j)-d(i,j))/2` |
| **LCA 等价性** | ✅ 两者数学等价（Gromov 积 = LCA 到原点的距离） | ✅  |
| **softmax 对象** | `softmax(lca_triplet / τ)` | `softmax(g_vec / τ)` — 完全相同 |
| **额外正则化** | `+ mat_sim.mean()` 拉高全局相似度 | 无对应项 |
| **温度退火** | ✅ 内置 anneal（每 step 乘 anneal_factor） | ❌ 固定 τ=0.1 |
| **归一化** | ✅ `normalize_embeddings()` 将所有点缩放到同一半径 | ❌ 无半径归一化（whole 嵌入半径自由） |

---

## 四、关键差异详解

### 4.1 `wij` 的梯度路径（最重要差异）

**HPCS 原始**：
```
编码器 → z_i, z_j → cosine_sim(z_i, z_j) = w_ij  ← 有梯度
                                              ↓
                        w_ij 参与 hypHC loss → 梯度回传到编码器
```
**效果**：相似度函数和嵌入**联合优化**。如果某个三元组排列不对，编码器可以**同时调整嵌入位置和相似度矩阵**来降低损失。这避免了"用自己输出监督自己"的循环问题，因为相似度和 LCA 深度由同一套嵌入参数决定。

**HyCoRe inter_hierarchy_MN40**：
```
预训练编码器(frozen) → z_i, z_j → cosine_sim(z_i, z_j) → detach → w_cache
                                                                      ↓
当前编码器 → z_i', z_j' → Gromov 积 → hypHC loss（仅调嵌入，w 固定）
```
**效果**：`w_ij` 是**外生目标**。编码器只能调整嵌入位置来匹配固定的相似度排序。这形成了"用自己的输出监督自己"的延迟反馈回路（因为 `w_ij` 来自旧权重，而嵌入用新权重），可能导致不稳定性。

### 4.2 你的 Spearman ρ = -0.0644 的含义

B1 实验得到的 Spearman ρ 接近零，说明 **Gromov 积排序和余弦相似度排序不相关**。两种可能原因：

1. **自举回路延迟**：`w_ij` 来自数 epoch 前的旧编码器，与当前嵌入不同步。
2. **预训练特征不够分散**：HyCoRe 预训练的分类特征可能**过于判别性**（同类样本几乎重合），导致 `w_ij` 方差极小，所有对都接近 1.0。这样 softmax 没有有意义的权重差异去竞争。

### 4.3 全局正则化 `mat_sim.mean()` 的缺失

HPCS 的 `loss_hyperbolic = torch.mean(total) + mat_sim.mean()` 中 `mat_sim.mean()` 会**拉升平均相似度**，防止嵌入塌缩到全部相似度=0 的 trivial 解。你的实现缺少这一项。

### 4.4 半径归一化

HPCS 在计算 LCA 前调用 `normalize_embeddings()` 将所有点投影到**同一半径**上（通过可学习的 scale 参数）。这意味着 LCA 深度完全由**角度**决定，不受径向位置干扰。在分类场景中你的 whole 嵌入半径由分类损失自由决定，可能导致 Gromov 积由半径而非角度主导。

---

## 五、结论

### 是否有"很严重的差别"？

**数学公式层面**：❌ 没有严重差别。你的 `hypHC_softmax_loss` 和 HPCS 的 `compute_hyp` 实现了**完全相同的 HypHC 核心公式**（Chami et al. 的式(5)）。

**工程实现层面**：⚠️ 有三处中等差异：

| 差异点 | 严重程度 | 影响 |
|---|---|---|
| `w_ij` stop-gradient vs 有梯度 | ⭐⭐⭐ | 自举延迟，可能是 ρ≈0 的主因 |
| 缺失 `mat_sim.mean()` 正则化 | ⭐⭐ | 可能产生退化解 |
| 缺失半径归一化 | ⭐⭐ | Gromov 积被半径主导 |

**应用场景层面**：✅ 这正是你的核心贡献。HPCS 对**点**做 HypHC（单样本内），你对**实例**做 HypHC（多样本间）。这是**首次将 HypHC 从 intra-sample 迁移到 inter-sample** 的工作。

### Spearman ρ ≈ 0 的根本原因推测

很可能是预训练的 HyCoRe 特征**过于判别性**—同类样本的 whole 嵌入在方向锥内高度聚集（余弦相似度 ≈ 0.99+），导致 `w_ij` 矩阵几乎没有差异。没有差异的权重无法驱动 softmax 产生有意义的排序竞争。建议后续实验检查 `w_ij` 的分布直方图验证这一假设。