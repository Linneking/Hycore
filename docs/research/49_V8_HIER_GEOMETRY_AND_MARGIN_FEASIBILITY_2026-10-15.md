# V8：恢复 HIER 几何设置与 HyCoRe margin 的数值可行性

日期：2026-10-15。状态：源码核查与 CPU 解析数值计算完成；以下训练配置为候选，不是已经运行的 V8。

## 1. 判断

V8 可以以“让 HyCoRe 适配官方 HIER 的几何设置”为主线：统一曲率幅值 c=0.1，恢复最终共享表示与代理的切空间 cap=2.3、官方逆度量梯度放缩、官方代理初始化和 HIER 距离/选择设置。

但只按一个估计的“空间距离范围”缩放 HyCoRe 的 R_hier 不足以保证兼容。需要同时看 R_contr 的 margin=4，以及 cap 放置的位置。当前 V5–V7 的活跃模型、intra 和 HIER 都使用 c=1；这里的“不一致”主要是相对官方 HIER 原配置的差异，不能直接断言同一旧训练内用了两个不同曲率，更不能据此认定热点的主因已经查明。[当前基础损失](../../inter_hierarchy_MN40/hier_proxy_scratch_v5/base_protocol.py:93)、[V7 HIER 入口](../../inter_hierarchy_MN40/hier_proxy_scratch_v7/hier_loss.py:9)。

建议的数值起点是：HIER 的几何与 margin 保持官方设置；HyCoRe 两个 margin 使用共同候选倍数 0.5，即 gamma=500、delta=2。这个倍数用于分配有限深度预算，不是曲率单位换算，也不是已经验证的最优超参数。

## 2. 原论文与源码能提供什么

HyCoRe 原论文确有曲率幅值 1、0.5、0.1、0.01 的消融，但公开论文/补充材料及当前官方代码未提供 c=0.1 专属的 gamma、delta 或 cap 配置。不能把论文的曲率消融视为“加入 HIER cap 后无需适配”。原设置 gamma=1000、delta=4；实际 part 点数 N 在 200–600 间统一按 batch 抽取，因此 R_hier 的 margin 范围准确为 1.667–5。[完整来源核查](V8_HYCORE_CURVATURE_SOURCE_AUDIT_2026-10-15.md)，[原论文 §4.1、表5](https://proceedings.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)，[官方损失](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/hutil.py)。

HIER 原版几何来源固定为公开缓存 commit `3986a744a1a54fd357e307d1cb3f2e81910b9ffc`：whole 和代理都使用相同的 ToPoincare，先限制切空间模长，再 expmap、数值投影及梯度放缩。[官方映射](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/hyptorch/nn.py#L204)、[模型](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/models/model.py#L111)、[代理与两个分项](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/losses.py#L27)。

## 3. 应该比较的单位

本文 c 表示负曲率的正幅值，实际曲率为 −c。记 t 是原点切空间向量的欧氏模长，r 是球内坐标模长，h 是原点双曲深度：

$$
r=\frac{\tanh(\sqrt c\,t)}{\sqrt c},
\qquad
h=\frac{2}{\sqrt c}\operatorname{artanh}(\sqrt c\,r)=2t.
$$

所以，切空间 cap=2.3 对应深度上限 D=4.6，而不是深度2.3。这个深度上限不随 c 改变。数学双曲空间本身的原点深度没有有限上限；球坐标边界有限不等于双曲距离有限。表中按标准数学公式计算，忽略官方截断使用的微小 epsilon 与浮点舍入。

| 设置 | 球坐标边界半径 | cap后的坐标半径 | 归一化坐标半径 | 原点深度上限 |
|---|---:|---:|---:|---:|
| c=1，切空间cap2.3 | 1.0000 | 0.9801 | 0.9801 | 4.6000 |
| c=0.1，切空间cap2.3 | 3.1623 | 1.9651 | 0.6214 | 4.6000 |

在当前 Geoopt 0.5.1 FP32 投影策略下，归一化半径最大为0.996，因此旧 c=1 的数值深度上限为：

$$
D_{old}=2\operatorname{artanh}(0.996)=6.21260609575.
$$

这个值是特定 dtype/数值投影的上限，不是旧 whole 的实际平均深度，也不是双曲空间的固有范围。仅把 c 改为0.1且不加额外cap，同一投影策略的深度上限反而增至约19.646。[Geoopt 0.5.1 投影](https://github.com/geoopt/geoopt/blob/v0.5.1/geoopt/manifolds/stereographic/math.py#L295)。

如果按新旧深度预算比缩放，有：

$$
s=\frac{4.6}{6.21260609575}=0.74043001103.
$$

它可以保持“margin / 最大深度”的比例，不能保证学习到的位置或全部点对距离仍等价。用户的2.3/4只是示意；实际换算时应在分子分母使用同一种单位。

## 4. 两个 margin 为什么必须一起审查

记 whole 为 w_i，本实例 part 为 p_i，负 part 为 p_j。当前每个 batch 使用共同 N，正负 part 都由同一个网络产生，并接受对应的层次项。两个 HyCoRe 目标为：

$$
R_{hier}=[h(p_i)-h(w_i)+m_N]_+,
\qquad m_N=\frac{\gamma}{N},
$$

$$
R_{contr}=[d_c(w_i,p_i)-d_c(w_i,p_j)+\delta]_+.
$$

若 whole/part 都被限制到深度D，要让所有 R_hier 为零，所有 part 的深度就不能超过 D−m_N。再由三角不等式：

$$
d_c(w_i,p_j)-d_c(w_i,p_i)
\le d_c(p_i,p_j)
\le h(p_i)+h(p_j)
\le2(D-m_N).
$$

因此，两个 hinge 同时为零的必要条件是：

$$
\delta\le2(D-m_N).
$$

这里的负例必须是 part；若负例改成whole，界会不同。这也是全部 radial hinge 同时为零的条件性分析，不能把它解释成违反时训练会失败。原版软正则本来就允许折中。

只按0.74043缩放 gamma 时，N=200 要求深度差3.70215，part即使在最有利情况下也只能到深度0.89785。此时 contrastive 的距离差最多为1.79570，小于原delta=4。即便把delta同时缩到2.96172，仍不能使这两个目标同时归零。

| 方案 | gamma | delta | N=200的层次margin | N=600的层次margin | 由必要界判定无法联合归零的N |
|---|---:|---:|---:|---:|---|
| cap后保留原margin | 1000 | 4 | 5.0000 | 1.6667 | 200–384，其中200–217连层次项也无法归零 |
| 仅层次项按深度预算缩放 | 740.4300 | 4 | 3.7022 | 1.2341 | 200–284 |
| 两项都按深度预算缩放 | 740.4300 | 2.9617 | 3.7022 | 1.2341 | 200–237 |
| 仅层次项减半 | 500 | 4 | 2.5000 | 0.8333 | 该必要界不排除；方向仍可能不满足 |
| 两项margin减半，候选 | 500 | 2 | 2.5000 | 0.8333 | 该必要界不排除；方向仍可能不满足 |

旧 c=1、Dold≈6.2126、gamma=1000、delta=4 本身也有N=200–237不能全部联合归零。因此不能把这种软目标张力都归因于V8迁移。

如果坚持delta=4，只调整gamma，要对全部N排除上述必要界冲突，需要：

$$
\gamma\le200\left(4.6-\frac42\right)=520.
$$

520是理论边界；达到边界需要极端相反方向，实际应留余量。如果两个margin统一乘alpha，则同一条件给出：

$$
4\alpha\le2(4.6-5\alpha),
\qquad \alpha\le\frac{9.2}{14}=0.657142857.
$$

这说明0.74043能保持旧空间占比，却不是排除该冲突的倍率。

## 5. 为什么把两项减半作为候选起点

候选采用gamma=500、delta=2，保留较小part要求更大深度差的顺序。在whole位于D=4.6、层次项刚好满足时，N=200/400/600的part深度分别为2.1/3.35/3.7667；不需要把小part压得几乎只剩原点附近。

进一步构造一组理想表示：同实例whole与part同方向，不同实例方向相差90°；whole深度4.6，part深度4.6−500/N。在c=0.1下，N=200的正距离为2.5，负距离为5.31113，差值2.81113，超过候选delta=2；若保持delta=4，这个构型仍不满足。对全部401个整数N=200–600逐一计算，候选都可以使这两个hinge为零。这个构型也可以在足够高维空间给多类别安排正交方向，因此不只依赖两个实例恰好反向。

**这是存在性示例，不是对真实网络、增强后part方向或训练收敛的保证。** margin减半可能通过改变hinge激活集合与频率改变有效梯度，并不会直接把活跃hinge的距离梯度减半；不预先声称CE与intra的权重无需再审查，更不根据纯公式直接宣布V8一定改善热点。

更改曲率也不是统一距离换算。例如固定whole深度4.6、part深度2.1：同向点对在c=1与c=0.1下距离都为2.5，90°点对距离却从6.02183变为5.31113，反向点对都为6.7。一个margin倍率只能做预算适配，不能保留所有邻居关系。

## 6. 恢复官方几何时不能漏掉的作用路径

1. **全链曲率统一。** 两个Möbius流形、whole/part映射、intra距离、HIER距离、代理映射及审查计算都使用c=0.1。保留官方HIER margin/tau=0.1，不把它们与HyCoRe的两个margin一起缩放。
2. **cap放在最终共享表示。** HyCoRe原路径是512维proj→expmap→Möbius512→256→mu→分类层。只在第一个expmap前截断不能限制后续mu。候选是在最终mu的logmap0上限制模长再映回球；分类、intra、HIER都读取同一处理后的whole，part走同样处理。也可重构最终映射，但那是额外架构变动，需单独标记。[实际forward](../../classification_ModelNet40/models/pointmlp.py:424)。
3. **共享输出处启用官方梯度hook。**

$$
g\leftarrow\frac{(1-c\|x\|_2^2)^2}{4}g.
$$

在c=0.1、cap2.3处因子约0.094197；c=1、同cap处仅约0.0003883。该差异约243倍，所以搬回曲率与cap会实质改变优化几何。若hook作用于共享mu，它同时改变CE/intra/HIER向主干回传的梯度；只给HIER分支加hook不等价。共享输出只施加一次该hook，避免在初始expmap与最终mu重复放缩。[官方梯度代码](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/hyptorch/pmath.py#L30)。
4. **恢复官方代理初始化与forward cap。** 随机切空间参数乘2.3×0.9/sqrt(D)，不使用V7的校准深度初始化。官方没有step后约束保存参数；V7的step后深度6约束属于自行新增方案，是否保留必须单独标记。forward截断仍可能让参数径向更新变弱，不能称它解决了所有代理学习问题。
5. **避免官方代码的全局曲率变量污染。** 原RiemannianGradient.c是类级变量；适配应把曲率绑定到具体算子的ctx或实例，避免混曲率诊断时相互覆盖。
6. **区分几何设置与训练协议。** 官方代理学习率为基础学习率50倍，使用AdamW；其梯度裁剪为逐坐标±10。HyCoRe原主干优化器为RSGD并用norm裁剪。这些不能静默当作同一种操作。V8主线应先明确恢复官方几何，优化器/采样/数据/选模变更另列，避免多个因素混称曲率修复。[官方参数组](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/utils.py#L757)、[梯度裁剪](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/utils.py#L212)。
7. **保留已确认的正确性修复。** 官方self-k候选问题不应为了“完全搬运”重新引入。官方pairwise距离的epsilon近似与标准精确距离应各自记录；本文必要界使用标准度量及HyCoRe的原距离口径，不把近似算子当成全范围严格精确的度量。

## 7. 数值验收和使用边界

实现：[独立数值脚本](diagnostics/v8_curvature_margin_feasibility.py)；最终结果：[解析数值汇总](diagnostics/results/20261015_v8_curvature_margin_accepted.json)。

完成500组双曲余弦定理与独立Poincaré坐标公式核对，最大误差6.72e−14；原点映射反算、同向/反向三角界、所有401种part点数的区间判断通过；候选正交方向构型的401个联合零hinge存在性检查通过。计算只使用CPU标量float64，无训练数据、权重、模型前向或优化器更新。

本次可以确认单位关系、截断预算和局部目标的必要可行性。实际whole/part径向分布、梯度hook后的真实参数步幅、代理激活和热点是否改善，需要V8后续训练验证；不能把已有SOP热点和V7方向冗余仅凭更改曲率就视为已解决。
