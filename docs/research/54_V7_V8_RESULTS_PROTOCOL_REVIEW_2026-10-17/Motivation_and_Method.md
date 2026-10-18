# Motivation与Method：HyCoRe中的跨实例软层次学习

本稿与[实验结果与协议文档](report.md)配套，用于在汇报实验演变前介绍研究问题和当前方法。Method以已执行的V8为准；V7属于此前实现，尚未实施的优化器及BN修改不纳入方法。

## Motivation：为什么引入跨实例层次

点云分类通常以离散类别作为监督。同一类别中的物体仍可能具有不同的局部组成与整体形态，不同类别之间也可能共享某些形态因素。类别标签可以提供判别边界，却没有显式规定这些差异与共性应该如何在多个抽象层次上组织。我们希望在保留分类能力的同时，使表示进一步表达实例之间的细粒度亲缘关系。

HyCoRe为这一目标提供了基础：它在双曲空间中联合学习物体的整体表示whole与随机局部表示part，通过径向次序和对比约束，表达单个实例的局部—整体关系。原方法也利用其他实例的part构造负例，但这类跨实例交互主要用于判别分离，尚未显式引入可以被多个whole共享的、多粒度祖先节点。因此，我们希望在已有part–whole约束之外，进一步组织不同whole之间的关系。

具体而言，期望形态接近的实例共享较具体的祖先，差异较大的实例通过较抽象的祖先关联，从而同时探索类内形态变化与类间共性。为此，我们借用HIER的可学习祖先代理，把跨实例关系作为附加正则，与HyCoRe的分类和part–whole目标联合优化。代理不固定绑定某个类别，也不要求等于某个真实part；关系随当前嵌入在线更新，无需事先给定类别树。双曲空间提供表达层次的几何条件，具体学习到的结构是否具有形态意义，仍需实验验证。

## Method：共享表示上的intra与inter联合学习

### 1. 保留HyCoRe的分类与局部—整体约束

给定带标签的点云，对同一物体采样whole与part视图，经共享PointMLP及双曲表示头得到256维表示。记整体表示、局部表示分别为：

$$
\mu_i=F_\theta(X_i^{\mathrm{whole}}),
\qquad
\nu_i=F_\theta(X_i^{\mathrm{part}}).
$$

分类器读取同一个whole表示，保留HyCoRe带标签平滑的交叉熵。设双曲距离为下式，其中原点距离称为嵌入深度：

$$
d_c(x,y),
\qquad
h_c(x)=d_c(0,x).
$$

保留HyCoRe的两项intra约束：

$$
R_{\mathrm{hier}}
=\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}
\left[h_c(\nu_i)-h_c(\mu_i)+\frac{\gamma}{N}\right]_+,
$$

$$
R_{\mathrm{contr}}
=\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}
\left[d_c(\mu_i,\nu_i)-d_c(\mu_i,\nu_{j(i)})+\delta\right]_+.
$$

这里B集合表示当前batch，N为当前part点数，正部函数为ReLU；负part沿用HyCoRe的batch翻转配对。径向项鼓励part比对应whole浅，对比项鼓励whole更接近自己的part。两个目标保留原方法的局部—整体先验，新增inter目标直接作用于whole与代理，不额外加入proxy–part损失。

### 2. 在线构造whole之间的关系

引入512个可学习代理，作为不同三元组可以共享的候选祖先。它们由随机切空间参数初始化，不设置每类代理配额，也不预先区分pair祖先与triple祖先。

样本三元组由当前whole嵌入在线构造。当前实现结合双曲距离与类别标签定义相似度：

$$
S_{ij}=\exp\bigl(-d_c(\mu_i,\mu_j)\bigr)
+\mathbb{1}[y_i=y_j].
$$

在此基础上构造互惠近邻：j从i的互惠邻居中抽取，k从非互惠邻居中抽取，排除自身作为负候选。V8按源码使用含self的top20，再移除自环；至少具有两个非自身互惠邻居且存在负候选的锚点，带放回抽取50组三元组。标签为同类关系提供偏置，但三元组并非简单地固定为“同类正例、异类负例”；这里也不是完全无监督的结构发现。近邻及三元组采样不参与反向传播，whole与代理通过后续可微关系损失更新。

主实验每类只有两例，因此合格锚点的互惠候选还需要包含跨类连接。当前方法组织的是跨whole关系；类内形态层次能否从中获益，是需要单独验证的问题。

### 3. 用两个祖先角色表达相对亲缘关系

对每个样本三元组，遍历全部512个代理，分别计算覆盖二元组和三元组的最大距离代价：

$$
a_p=\max\{d_c(\mu_i,p),d_c(\mu_j,p)\},
\qquad
b_p=\max\{a_p,d_c(\mu_k,p)\}.
$$

将代价取负后交给两次独立的hard Gumbel–softmax，选择pair祖先A与triple祖先B：

$$
q^{\mathrm{pair}}
=\operatorname{STGumbel}\left(-a/\tau\right),
\qquad
q^{\mathrm{triple}}
=\operatorname{STGumbel}\left(-b/\tau\right).
$$

负号使最大距离代价较小的代理更容易被选中。前向使用一个离散代理，反向保留软选择的straight-through梯度。这里A、B只是当前三元组中的角色，同一代理可在不同三元组中承担不同角色。

HIER的三项距离约束写为：

$$
\ell(i,j,k)=\mathbb{1}[A\ne B]
\left(
\sum_{u\in\{i,j\}}
\left[d_c(\mu_u,A)-d_c(\mu_u,B)+m\right]_+
+\left[d_c(\mu_k,B)-d_c(\mu_k,A)+m\right]_+
\right).
$$

它鼓励i、j更接近自己的pair祖先，而k更接近三元组共同祖先。当两次选择落到同一代理时，该组三元组贡献为零；平均值仍以全部采样三元组为分母。由此得到样本—代理分项R_sample。

同时，将代理自身视为被组织的实体，按代理距离构造互惠近邻三元组，再复用上述祖先选择与三项约束，得到代理—代理分项R_proxy。此时相似度只使用距离项，不加类别偏置；代理三元组的祖先仍从同一512个代理池中选择。这一设计来自HIER，用于进一步组织代理之间的关系。

$$
R_{\mathrm{inter}}=R_{\mathrm{sample}}+R_{\mathrm{proxy}}.
$$

当前损失中的“祖先”是相对距离角色：它没有硬性要求每个祖先都比样本更靠近原点，也没有保证全部局部关系能够组成一棵全局一致树。因此，方法目标是学习软层次关系，而不是预先宣称已经恢复真实类别树。

### 4. V8的几何适配与联合训练

V8让HyCoRe采用与HIER一致的曲率幅值c=0.1，并在最终共享whole、part表示和代理映射中使用切空间cap与逆度量梯度放缩。记待映射的切空间向量为u：

$$
\bar u=\min\left(1,\frac{2.3}{\|u\|}\right)u,
\qquad
z=\operatorname{Exp}_0^c(\bar u).
$$

whole与part在最终Möbius表示后，先取原点logmap，再执行上述映射；分类、intra和inter读取同一处理后的whole。对应梯度放缩为：

$$
g\leftarrow\frac{(1-c\|z\|^2)^2}{4}\,g.
$$

cap2.3对应原点深度上限约4.6。它提供有限的有效深度预算，但只截断超出上限的向量，不会把内部whole自动拉到同一球壳，也不会将保存的代理参数投影回边界。

为适配这一预算，V8将HyCoRe的两项margin设为gamma=500、delta=2；HIER保持margin与tau均为0.1。margin减半是当前实验的预算适配选择，不是曲率变化下的严格等价换算。总目标为：

$$
\mathcal L
=\mathcal L_{\mathrm{CE}}
+0.01R_{\mathrm{hier}}
+0.01R_{\mathrm{contr}}
+0.5\left(R_{\mathrm{sample}}+R_{\mathrm{proxy}}\right).
$$

四项目标从第一次更新共同训练，不使用HIER warmup。主模型使用RSGD，代理使用AdamW；两者学习率分别由0.1降至0.005、由0.01降至0.0005，采用300轮余弦调度。主实验每卡16类×2例，两卡类别互斥，global batch64。代理通过联合损失直接更新，part通过共享网络及intra约束随之适应；这条梯度路径本身不保证形成统一组合树。

## 与实验文档的衔接

汇报可以按“HyCoRe已有part–whole先验 → 希望增加跨whole亲缘结构 → 用HIER代理实现 → V7/V8检查迁移条件”的顺序展开，再进入结果文档第4、5节。

V7到V8的改动属于几何与优化配置的适配，研究目标没有变。现有图表支持代理持续参与及热点集中程度有所改善；它们尚不能单独证明学到了有意义的形态层次，也没有建立同协议HIER分类收益。后续需要将分类成绩、独立形态信号和关系稳定性共同作为判断依据。

近期[机制诊断](../57_V8_CAP_AND_WHOLE_CAUSES_2026-10-18/report.md)还表明，代理Adam历史状态与whole的BN前向条件会影响几何分布。代理eps调整、whole/part统计分离及固定深度正则目前属于后续候选，不应写成已经完成的V8方法。

## 方法配置与来源

| 项目 | 当前V8口径 |
|---|---|
| 主任务 | PointMLP + ModelNet40分类 |
| 表示维度、曲率幅值 | D=256，c=0.1 |
| 代理与关系 | P=512；K=20；每合格锚点50组三元组 |
| HIER参数 | margin=0.1，tau=0.1，sample/proxy分项各乘0.5 |
| 距离实现 | HIER沿用官方数值距离，intra使用Geoopt；曲率均为0.1，并非逐位相同的算子 |
| HyCoRe参数 | gamma=500，delta=2，两intra系数各0.01 |
| 代理优化器 | AdamW，eps=1e−8，weight decay=0.01；没有step后raw投影 |
| 原V8数据协议 | 8856训练/984验证，验证选best；官方2468 test |
| 后续FULL9840协议 | 全9840训练、每轮test选best，单独标记；不与原V8验证协议混写 |

研究构想见[原idea](../48_CURRENT_RESEARCH_IDEA_2026-10-10.md)与[技术交接](../25_HYCORE_HIER_HPCS_TECHNICAL_HANDOFF_2026-10-05.md)；当前实现见[联合配置](../../../inter_hierarchy_MN40/hier_proxy_scratch_v8/train.py)、[intra目标](../../../inter_hierarchy_MN40/hier_proxy_scratch_v8/base_protocol.py)、[代理与三元组](../../../inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py)及[三项距离损失](../../../inter_hierarchy_MN40/hier_proxy_scratch_v6/hier_loss.py)。HIER祖先机制为借鉴部分，本研究探索其与HyCoRe分类及局部—整体先验的结合。
