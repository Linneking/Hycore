# V8：原版 HyCoRe 曲率、距离单位与 margin 来源核查

日期：2026-10-15。范围：原版 HyCoRe 公开论文、补充材料、官方代码及本地对应源码；只读核查和解析计算，未修改训练代码、未启动训练。本文件中的新方案均为 **proposal**。

## 结论

让 HyCoRe 在统一曲率并明确各距离算子的条件下适配 HIER，是可实现的研究方向。但“只缩放 `R_hier` 的 margin”不足以保证两者兼容：原 HyCoRe 还有以双曲距离为单位的 `R_contr` 固定 margin=4；cap 必须作用于实际共享的最终 whole/part embedding；分类器、映射及损失也必须传入同一曲率。此前热点的全部原因不能仅凭此次源码核查归于曲率。

## 1. 原版有曲率 0.1 的结果，但未找到可直接搬用的专用参数配置

HyCoRe 论文表 5 的曲率幅值为 1、0.5、0.1、0.01，DGCNN+HyCoRe 的 AA 分别为 80.2%、79.4%、78.7%、78.5%。该消融属于 DGCNN / ScanObjectNN，而非我们当前的 PointMLP / ModelNet40；它支持低曲率可以训练，但不能替代当前组合方案的验证。论文的统一实验设置写有 γ=1000、δ=4、两正则权重各 0.01，没有说明曲率消融时另设的 γ、δ 或半径 cap。补充材料主要介绍测地线与可视化，没有补给曲率 0.1 的专用参数。[论文 §4.1、表 5、§4.3](https://proceedings.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)，[补充材料](https://proceedings.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Supplemental-Conference.pdf)。

本地原版入口 `parse_args` 没有曲率参数；`-c` 实际是 checkpoint 路径。PointMLP 的两个 manifold 均硬编码 `c=1`，intra 损失内部另外创建一个 `c=1` 的球。公开代码的对应位置一致。因此，**未找到官方可直接复用的 c=0.1 训练命令或配置**；不能推断论文没有做该实验，也不能擅自声称作者使用了某个缩放系数。定位：[入口](/D:/Hycore/classification_ModelNet40/main_pointmlp_hycore.py:29)、[模型](/D:/Hycore/classification_ModelNet40/models/pointmlp.py:356)、[损失](/D:/Hycore/classification_ModelNet40/hutil.py:49)；[官方模型](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/models/pointmlp.py#L356)、[官方损失](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/hutil.py#L49)。

这里 `c>0` 表示负曲率的幅值，几何曲率实际为 −c；不要与论文前文使用的带符号曲率混淆。

## 2. `1000/N` 是原点双曲深度差，不是点对距离或坐标模长

原训练每个 batch 先统一抽取 whole 点数 800–1024，再统一抽取 part 点数 N=200–600；各实例的局部中心分别随机选取。`random.randint` 两端均包括。helper 的默认范围 100–500 被主入口覆盖，不能把默认值当实际协议。[采样 helper](/D:/Hycore/classification_ModelNet40/hutil.py:27)、[实际调用](/D:/Hycore/classification_ModelNet40/main_pointmlp_hycore.py:210)；[官方训练入口](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/main_pointmlp_hycore.py#L210)。

记 whole 为 w，本实例 part 为 p，原点深度为 h(x)=d_c(0,x)。源码实际计算：

$$
R_{\mathrm{hier}}=[h(p)-h(w)+m_N]_+,
\qquad m_N=\frac{1000}{N}.
$$

所以 N=200 时 m=5，N=600 时 m≈1.667；用户估计的“2–5”接近，但下界应为 1.667。零损失要求 whole 比 part 至少深 m，并不要求 whole–part 的实际双曲距离恰好等于 m。[公式实现](/D:/Hycore/classification_ModelNet40/hutil.py:53)。

同一源码还计算：

$$
R_{\mathrm{contr}}=[d_c(w,p)-d_c(w,n)+\delta]_+,
\qquad \delta=4.
$$

负 part n 由 batch 翻转获得，代码没有检查其是否属于不同类别；论文描述的跨类负例与实际代码的这一差别应保留可复现记录。两个 margin 都是未平方的双曲距离单位；返回日志中的 `parent_norm`、`pos_norm` 却是 Euclidean 坐标模长，不能拿它们直接判断层次约束。[负例与距离](/D:/Hycore/classification_ModelNet40/hutil.py:44)、[hinge 与返回值](/D:/Hycore/classification_ModelNet40/hutil.py:60)。

## 3. “可用深度约 6”来自数值投影，不是双曲空间的有限边界距离

对曲率幅值 c，Poincaré 球坐标半径范围和原点双曲深度分别为：

$$
0\leq\|x\|_2<\frac1{\sqrt c},
\qquad
h(x)=\frac2{\sqrt c}\operatorname{artanh}(\sqrt c\|x\|_2).
$$

数学深度没有有限上限。采用原点指数映射时，切空间 Euclidean 模长 t 与深度的关系为：

$$
\|\exp_0^c(v)\|_2=\frac{\tanh(\sqrt c\,t)}{\sqrt c},
\qquad h(\exp_0^c(v))=2t.
$$

因此，**切空间 cap=2.3 对应原点深度上限 4.6，在 c=0.1 或 c=1 下都是如此**，只改变球内坐标半径。用户提出的 `2.3/4` 如果分母 4 是“双曲深度”，就混用了单位；应先把两端都转为深度，或都转为切空间模长。

HyCoRe 的活跃映射使用继承自 Geoopt 的 `expmap0` 和 `mobius_matvec`。Geoopt 当前公开实现对 FP32 默认将归一化坐标半径限制到 0.996，故该数值策略下：

$$
h_{\max}^{\mathrm{FP32}}(c)
=\frac{2\operatorname{artanh}(0.996)}{\sqrt c}.
$$

c=1 得 6.212606；c=0.1 得约 19.645985。若另行采用最终切空间 cap=2.3，两者都改为 4.6。**这些值取决于 dtype、依赖版本及 cap 位置，应在 V8 实际环境再确认。** 本地另一套 `pmath.project` 写有 0.999 投影，但它并不是当前 PointMLP 所调用的 Geoopt manifold 投影。[继承关系](/D:/Hycore/classification_ModelNet40/models/manifolds/poincareball.py:2)、[未直接使用的另一套投影](/D:/Hycore/classification_ModelNet40/models/ops/pmath.py:100)；[Geoopt 投影来源](https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/stereographic/math.py)、[Geoopt 映射默认 project=True](https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/stereographic/manifold.py)。

## 4. cap 与梯度处理的作用位置不能靠名称判断

实际模型路径为：512 维投影 → 第一个球的 expmap → Möbius 512→256 → 得到共享 embedding μ → 第二个球的 Möbius 256→40 分类。whole、part 都经同一个模型。intra 作用于 256 维 μ，分类器也读取 μ。[定义](/D:/Hycore/classification_ModelNet40/models/pointmlp.py:394)、[forward](/D:/Hycore/classification_ModelNet40/models/pointmlp.py:424)。

**Proposal：**若 V8 要让 HIER、intra、CE 共用深度≤4.6 的表示，应明确限制最终 μ 对应的 `logmap0(μ)`。只给第一个 512 维映射加 cap，不能保证后面的 Möbius 层不重新放大半径。需一次性向两个 manifold、intra 距离及 HIER 距离传入同一个 c；只改模型或只改 HIER 会继续出现尺度错配。

原版文件虽定义了带 RiemannianGradient 的 `ToPoincare` 类，但实际 PointMLP forward 不调用该类；它直接调用 manifold.expmap0。主训练使用 RiemannianSGD，并对模型参数做 norm=1 的梯度裁剪。这不能等同于已经在共享特征端启用了 HIER 的额外梯度 hook。[实际 forward](/D:/Hycore/classification_ModelNet40/models/pointmlp.py:425)、[未调用的 ToPoincare](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:35)、[优化器](/D:/Hycore/classification_ModelNet40/main_pointmlp_hycore.py:136)、[裁剪](/D:/Hycore/classification_ModelNet40/main_pointmlp_hycore.py:226)。

## 5. 可以按深度预算缩放，但不能仅凭一个比例保证原相对位置

**Proposal：**若以原 c=1、FP32 数值深度上限为参照，新的深度预算比例为：

$$
s=\frac{4.6}{6.212606}\approx0.740430,
\qquad m_N'=s\frac{1000}{N}.
$$

它保持的是“要求的深度差 / 深度上限”的比例，不是训练后的坐标位置、邻居关系或全部距离。计算结果如下；`0.575` 一栏只复算用户假设的比值，并不表示单位推导成立。

| part 点数 N | 原 margin | 用户假设 s=0.575 | 按深度上限 s≈0.740430 |
|---:|---:|---:|---:|
| 200 | 5.000000 | 2.875000 | 3.702150 |
| 300 | 3.333333 | 1.916667 | 2.468100 |
| 400 | 2.500000 | 1.437500 | 1.851075 |
| 500 | 2.000000 | 1.150000 | 1.480860 |
| 600 | 1.666667 | 0.958333 | 1.234050 |

有一个同时约束的必要条件需要纳入判断。设每个 whole 深度≤D，同一 batch 的 part 使用相同 m；若所有 `R_hier` 均为零，则所有 part 深度≤D−m。利用度量的三角不等式：

$$
d(w,n)-d(w,p)
\leq d(p,n)
\leq h(p)+h(n)
\leq2(D-m).
$$

要让 `R_contr` 同时为零，必须满足：

$$
\delta\leq2(D-m).
$$

D=4.6、N=200、s=0.575 时 m=2.875，右侧仅 3.45，小于原 δ=4；只改层次 margin，部分小 part 的两个 hinge 无法同时为零。按 s≈0.740430 时小 part 的条件更紧。**这不代表训练无效或源码错误**：它说明软正则间仍有取舍，零损失可行性与训练收益是不同问题；原版在有限数值预算下也可能有此取舍。

**Proposal：**全局尺度适配至少同时审查 `R_hier` 与 `R_contr` 两种 margin，并检查最终共享 embedding、梯度 hook 和分类器的作用路径。保持两者相对强度时，还须考虑缩放距离和 margin 会改变正则数值与其相对 CE 的梯度尺度。是否缩放 δ、是否保留原损失权重属于新设计，不能归为官方 c=0.1 的已知配置。

最后，曲率改变与半径压缩是两件事：若保持归一化坐标 √c‖x‖ 和角度不变，所有双曲距离随 1/√c 缩放；若保持切空间 cap=2.3，原点深度上限却固定为 4.6，角向距离变化仍依赖角度与深度。单个全局 margin 倍数无法使整套几何严格等价。
