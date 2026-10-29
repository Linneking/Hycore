# EXP、数值投影与径向正则的回传审查

日期：2026-10-29。问题：当前 Pseg3 的加权径向正则回到 H 的梯度约为 1e−10，目标究竟向内还是向外；原 HyCoRe DGCNN 分类是否同样经过 EXP 与数值投影。

本记录以本地权威源码和所安装 Geoopt 0.5.1 为一手证据。分类核心文件的官方来源身份沿用 [100 号源码审查](../100_HYCORE_DGCNN_CLASSIFICATION_COMPARISON_2026-10-29/report.md)（官方 HyCoRe 提交 `ef56a73020e309b1009c486fea1f601d3f7b3252`）；此轮直接读本地有效 forward、loss 和库实现，不把旧分类训练的实际梯度或贴边率当作已经测量。没有改动分类、分割或正在运行的实验代码，没有重训、连接服务器或改变实验协议。

## 1. 先区分梯度与实际下降方向

令 `d0(z)=d_ball(0,z)`。向外意味着 d0 增大，向内意味着 d0 减小。优化器沿负梯度下降，因此“梯度向外”和“下降推动向外”是相反说法。下面表格写的是减小损失的目标方向；仅在 hinge 激活时成立，满足 margin 后该样本的径向正则梯度为零（恰好等于零处按实现的次梯度规则）。

| 路径 | 实际径向 hinge | whole 的下降目标 | part 的下降目标 |
|---|---|---|---|
| 原分类 HyCoRe DGCNN | `max(-d0(mu_whole)+d0(mu_part)+1000/Npart, 0)` | 向外、深度增大 | 向内、深度减小 |
| 当前分割 Pseg3 v3 | `max(d0(z_whole)-d0(z_part)+1000/Npart, 0)` | 向内、深度减小 | 向外、深度增大 |

分类证据：[hutil.py:42](/D:/Hycore/classification_ModelNet40/hutil.py:42) 的默认 `opposite_hier=False`，[hutil.py:53](/D:/Hycore/classification_ModelNet40/hutil.py:53) 的双曲深度与 62–69 行正负号；[main_dgcnn_hycore.py:227](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:227) 未传 `opposite_hier`，故使用默认分支；`1000/Npart` 来自 [hutil.py:39](/D:/Hycore/classification_ModelNet40/hutil.py:39)。分割证据：[v3/losses.py:69](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/losses.py:69) 的 `whole_depth-part_depth+gamma`，[v3/losses.py:159](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/losses.py:159) 的 `gamma_base/size`。两者径向系数均为 .01，见 [分类入口:230](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:230) 和 [分割 loss:110](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/losses.py:110)。

对分割的中间球坐标 z，在激活、非零半径且未碰其他数值截断时，`∂Rhier/∂z_whole` 沿 whole 径向向外，负梯度推动 whole 向内；`∂Rhier/∂z_part` 向内，负梯度推动 part 向外。这是独立看两个池化嵌入的目标，不代表共享 H 或共享网络参数的一次合成更新必然同时实现两者。whole 和 part 使用重叠的点特征，回到这些点时梯度会相加，最终还叠加 CE、对比项、动量与 weight decay。

“梯度范数小”只说明该坐标处的局部信号弱，范数本身不提供向内/向外的符号；也不能把径向项推广为两个正则项都如此。对比项包含正负样本的距离，通常还有方向分量，不等价于纯径向目标。

## 2. 分类确实也经过 EXP 和默认投影

原分类有效路径（whole 和 part 分别独立调用同一个网络）：

```text
XYZ_whole / XYZ_part
    → DGCNN 欧氏编码器、全局 max+avg、Linear+BN6+LeakyReLU
    → H512
    → expmap0（默认 project=True）
    → X512
    → emb = M1(512→256)
    → mu256 ──→ Rcontr、Rhier
            └─→ mlr = M2(256→40) → 分类 CE
```

证据：[dgcnn_hmodel.py:170](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:170) 产生 H；177 行调用 EXP；182 行 M1 产生 mu；186 行 M2 得到 logits；190 行返回 `(mu, logits)`。两层定义见 [123 行](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:123)。[分类入口:223](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:223) 对 part、whole 各调用网络，227 行的两正则取上述 mu，故回传经过 M1，再经过 X 的 EXP/投影。径向正则直接更新 M1 和编码器，不直接更新 M2。合并总损失反传后才全网 L2 norm1 裁剪、再 RSGD 更新，见 [入口:230](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:230)。梯度范数裁剪与前向球投影是两种操作。

本地 [PoincareBall 封装:8](/D:/Hycore/classification_ModelNet40/models/manifolds/poincareball.py:8) 继承 Geoopt，未覆盖 `expmap0`、`mobius_matvec`、`dist0`。实际 `MobiusLayer.forward` 仅调用 `manifold.mobius_matvec(self.weight, input)`，见 [manifold_layers.py:110](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:110)；`weight` 经过 `bias` 与 `transp0` 构成，见 [71 行](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:71)。未调用该文件中另一类 `ToPoincare` 的 `grad_fix`，不能看到那段未使用代码就认为当前分类存在一个额外回传修正 hook。

Geoopt 0.5.1 的本地一手实现：

- [manifold.py:327](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:327)：`expmap0(..., project=True)` 先 EXP，再 `math.project`。
- [manifold.py:289](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:289)：`mobius_matvec(..., project=True)` 也对结果做数值球投影。
- [math.py:307](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:307)：FP32 的 eps=.004，`c=1` 时投影半径 .996；超过阈值用 `x/||x||*.996`，由普通 autograd 经过 `torch.where` 的选中分支回传。
- [math.py:1049](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:1049) 与 [tan_k:184](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:184)：负曲率 c=1 的 EXP 为 `tanh(||u||)*u/||u||`。
- [math.py:890](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:890)：`dist0` 使用 `2*artan_k(||x||,k)`，c=1 对应 `2*atanh(||x||)`。

## 3. 分割的分叉位置使纯径向项更容易直接碰到投影截断

```text
H[B,N,256]
    ├─ 逐点 expmap0+投影 → M1（B组省略）→ M2 → M3 → 分割CE
    ├─ mean_whole(H) → expmap0+投影 → z_whole ─┐
    └─ mean_part(H)  → expmap0+投影 → z_part  ─┴─ Rcontr、Rhier
```

证据：[v3/model.py:38](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/model.py:38) 是分割头；[v3/losses.py:148](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/losses.py:148) 的欧氏特征选 part、mean 后 EXP；160 行将这两个 EXP 输出直接传入正则。这里两正则不经过 M1/M2/M3，属于已由用户确认的分割论文设计，不在本轮更改。

必须精确区分两个机制：

1. **未投影的理想复合函数不会仅因 tanh 径向导数小而使该纯径向梯度消失。** c=1、u 非零且理想精度下，`d0(exp0(u))=2*atanh(tanh(||u||))=2||u||`，其对 u 的梯度范数是 2。EXP 的小径向导数会被 dist0 的大径向导数抵消。因此先前“经过 EXP 和投影时明显衰减”应收紧为：当前输入已经触发了数值投影，纯径向回传在投影处几乎被消掉；不能仅凭 tanh 饱和解释。
2. **触发数值投影后，半径被固定。** FP32、c=1，`||u||>atanh(.996)≈3.1063` 时，`project(exp0(u))=.996*u/||u||`。在该分支内部，`d0` 等于常数 `2*atanh(.996)`；纯径向损失对 u 的解析梯度为零。浮点计算可能留下极小残差，不能把 1e−10 当成具有稳定方向和有效强度的信号。球面切向方向仍可传递其他损失的梯度。

## 4. 为什么不能因此宣称分类也一定失效

分类径向损失作用在 **M1 输出 mu**，分割径向损失作用在 **池化 H 的直接 EXP 输出 z**。两者都经过初始 EXP/投影，但 M1 改变了后者的几何关系。

在 c=1、初始 X 已投影成 `.996*e`、且 M1 输出尚未触发它自己的投影时，可把有效 Möbius 矩阵记为 W，有：

```text
mu = Exp0(W * Log0(X))
d0(mu) = 2*atanh(.996)*||W e||
```

这是由 Geoopt Möbius 矩阵乘定义得到的局部公式，不是已测出的分类训练结果。即便 X 半径固定，M1 的 W 仍可改变 mu 深度；一般的非等距 W 也使不同输入方向 e 对应不同输出深度。于是分类径向项可以直接更新 M1，并通过 X 的方向回传到 H；“初始 EXP 径向被截断”不等于“所有编码器梯度被截断”。如果 M1 输出 mu 自己也被固定半径投影，或者方向/矩阵出现特殊退化，该纯径向项也可能衰减，必须实测 mu 以及 M1/编码器分项梯度才能判断。

所以当前证据支持“分割 mean→EXP 的纯径向正则在已投影点局部几乎无回传”，不支持“原分类没有投影”，也不支持“分类正则一定同样无效”。

## 5. 本轮局部 CPU 探针与实测边界

本目录的 [local_geometry_probe.py](local_geometry_probe.py) 和 [JSON 结果](local_geometry_probe.json) 已完成 CPU 局部机制验证，Torch 2.6.0、Geoopt 0.5.1，CUDA 未初始化。不加载真实点云或分类 checkpoint，不等于旧分类 DGCNN 的端到端梯度测量。

FP32/c=1 的 `d0(expmap0(u))` 局部结果：

| 切空间模长 | 是否使用默认球投影 | 输出半径 | 对 u 的梯度 L2 |
|---|---|---|---|
| 1 | 是 | .761594 | 2.000000 |
| 3 | 是 | .995055 | 2.000021 |
| 4 | 否，仅局部机制对照 | .999329 | 2.000113 |
| 4 | 是 | .996000 | 7.63e−7 |
| 10 | 是 | .996000 | 3.05e−7 |

后两行剩余总梯度主要是浮点抵消残差。模长 10 的有符号径向导数约 5.68e−14，不应把该残差当成实质径向驱动。更大模长时，关闭 project 还可能触发 tanh 舍入到 1、dist0 的 artanh clamp，不能把无投影当成通用修复：本探针模长 10 的无投影分支也给出零梯度。FP64 使用不同的数值投影半径 .99999，阈值不同，详见 JSON；不能混为同一协议。

还单独验证了当前 `relation_losses`：当 whole/part 的切空间模长为 1/1.2 时，加权径向项对两切向量的梯度 L2 都约 .02，whole 球坐标的梯度向外、part 的梯度向内，下降目标相反；换成 10.5/10.8 且两者投影后，损失仍为 .02，回切向量的梯度在该轴对齐 FP32 例子中均为零。这说明正的 hinge 值不保证它能回传有效梯度。

另用原 `MobiusLayer` 类的 3→2 局部实例、seed4780 验证后置 M 的机制：其 FP32 输入半径 .996000、输出半径 .325512；`d0(M(exp(u)))` 回 u 的梯度 L2=.207079，几乎全为相对 u 的方向分量，M 的 `_weight` 梯度 L2=6.038597。这只是局部示例，证明方向/矩阵路径可以存在，**不是分类 DGCNN 的真实梯度大小，也没有恢复已丢失的输入径向导数**。探针没有优化器，没有更新模型参数。

上述默认投影与计算公式也对照了 [Geoopt v0.5.1 的 manifold.py](https://github.com/geoopt/geoopt/blob/v0.5.1/geoopt/manifolds/stereographic/manifold.py) 和 [math.py](https://raw.githubusercontent.com/geoopt/geoopt/v0.5.1/geoopt/manifolds/stereographic/math.py)。原 HyCoRe 固定提交的在线 raw 本次返回 cache miss；分类结论以本地、已有官方来源 SHA 核验的源码为依据，没有声称本次重新下载官方文件成功。

## 6. 如何解读已有的 1e−10 对 1e−4

当前三组训练的观测仍以 [106 启动与早期趋势](../106_PSEG3_THREE_ARM_LAUNCH_2026-10-29/execution.md) 为准。`last_components["hier"]` 已乘 .01，[losses.py:162](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/losses.py:162)；[train.py:303](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/train.py:303) 将它和 CE 分别交给监控。

[monitoring.py:257](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/monitoring.py:257) 用同一计算图的 `autograd.grad` 分项求导，[132 行](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/monitoring.py:132) 对整个 H[B,N,F] 展开求欧氏坐标 L2；它不是参数梯度、黎曼范数或实际优化器位移。其回传已包含 mean 分发：固定 part 索引时，每点来自 whole 的贡献为 `g_whole/N`，part 成员另加 `g_part/Npart`，两路可以相互抵消。监控同时有有符号径向分量，[158 行](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/monitoring.py:158)，仅总范数不能判断 CE 的内外作用。

首批的加权 Rhier→H 为约 1.18e−10～1.24e−10，CE→H 为约 1.45e−4～3.46e−4；**加权 Rcontr→H 仍约 4.79e−5～5.84e−5**。因此“径向项局部几乎推不动”不能替换成“所有正则都比 CE 小六个数量级”。whole/part EXP 输出处的径向梯度并不小，而回到两个池化输入后约 2.4e−9、到 H 约 1e−10，与局部投影机制相符。

本轮不会依据该审查缩放 H、修改正则分叉、取消 Geoopt 球投影或改正在运行的三组实验；也没有复查实时训练进度。若以后判断原分类真实损失是否同样衰减，需要在它的真实 H、EXP 输出、mu、M1 参数分别测量，不能把本次局部探针当作替代。
