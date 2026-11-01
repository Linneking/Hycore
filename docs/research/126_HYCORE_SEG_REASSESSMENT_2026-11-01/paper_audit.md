# HyCoRe / HyCoRe-seg 论文与代码复核

日期：2026-11-01。范围：论文、官方分类源码、本地已实现的 H/I 路径及 Geoopt 算子；无训练、GPU操作、模型修改或推送。本文把论文明确项、源码明确项和本次数学推导分开。

## 1. 论文证据及复现边界

分类论文 Table 3 的 ScanObjectNN、256 维 AA：DGCNN 77.8%，Hype-DGCNN（双曲头、无正则）76.5%，EuCoRe 79.0%，完整 HyCoRe 80.2%。因此用户所说的“只换双曲层不一定有益”有直接依据；但不能扩大成所有维数必然更差。Table 4 的仅层次、仅对比、两项组合 AA/OA 分别为77.9/80.5、79.2/81.6、80.2/82.1。作者也将改进双曲基线列为开放问题。[分类论文§4.3、Tables3–5](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7)

HyCoRe-seg Fig.1/§3 的结构是：一次编码得到逐点 H；在 H 上取特征近邻；whole/part 分别欧氏平均→EXP→正则。主支则逐点 EXP→一个或多个共享 Möbius 映射→CE。层次方向是 whole 靠内、part 靠外。Table1 的 DGCNN/完整方法 Class、Instance mIoU 为80.9/85.1与82.8/85.7；Table2仅对比85.6、仅层次85.5；没有“相同双曲头、两项都关闭”的独立分割行。§3写两margin均1024/Npart，§4写gamma1000、delta4，存在未解释差异。论文未给出精确层宽、head切分、BN/dropout、LR等完整可执行配方。[分割论文Fig.1、§3–4及Tables1–3](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf#page=1)

由以上定义可推得：分割正则不经过主支M是论文自身的设计，不是此前实现遗漏了分类代码中的步骤。没有证据要求分割正则必须先经过M才能有效。当前1000/Npart与常数4是借助分类源码形成的重建解释，不能写成唯一无歧义的原论文设置。

## 2. 分类官方代码实际上做了什么

`DGCNN.forward` 在全局池化和Linear/BN/Leaky后，执行 `EXP→emb(512,256)→mlr(256,40)`，将最后球坐标直接作为logits。名称mlr不意味着实现了双曲多项逻辑回归：活动类实际是 `MobiusLayer`，没有输出logmap或温度。[官方DGCNN源码](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/models/dgcnn_hmodel.py#L121)

官方 `MobiusLayer.forward` 只有 `mobius_matvec(self.weight,input)`。`_weight` 用 `kaiming_normal_(a=sqrt(5))` 初始化，`_bias[out,1]` 在 `±4/sqrt(fan_in)` 均匀初始化；有效权重是 `transp0(expmap0(_weight*_bias),_weight)`。这里的_bias通过权重重参数化生效，不是在输出上做 Möbius 平移，也没有集成BN/激活/dropout。当前本地分割继承了这套原类与初始化，没有加载分类训练权重。[官方层源码](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/models/ops/manifold_layers.py#L51)、[本地H共享主支](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/model.py:57)

PointMLP训练入口在 `net.train()` 下先part再whole；两次共享编码都更新BN，part通过 `emb=True` 跳过最终分类层。正则作用于首M后的mu。它没有在part前传临时设置eval。[官方PointMLP训练](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/main_pointmlp_hycore.py#L196)、[官方PointMLP模型](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/models/hpointmlp.py#L411)

I将原始part重新编码，再经平均→EXP→共享首M，因特征选点依赖whole而按whole→part更新BN。这是分类启发的新变体，同时改变正则表示、part上下文和BN统计；不能据此说它更接近分割论文原流程。[I模型](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v8/model.py:25)、[已确认方案](/D:/Hycore/docs/research/125_PSEG3_I_REENCODE_2026-11-01/plan.md)

## 3. 从实际算子得到的两个重要推论

### 3.1 多层M不等于多层非线性判别

以下是本次根据源码做的数学推导，不是论文声称的结论。

对固定有效权重W，Möbius矩阵乘的输出方向是Wx，长度另乘一个非负径向系数。EXP及球投影也只改变长度。因此在当前无平移、无逐坐标激活的头中，除零向量和数值误差情况外：

```text
z = M3(M2(M1(EXP(H)))) = a(H) · W3 W2 W1 H,   a(H)>0
argmax(z) = argmax(W3 W2 W1 H)
```

无裁切时还有精确复合关系 `M2⊗(M1⊗x)=(M2 M1)⊗x`。径向裁切会改变长度与优化，但仍保留方向关系。_bias依赖参数而不依赖输入，不破坏这一推论。

所以A的三M与B的两M主要改变参数分解、数值轨迹与优化条件，并没有恢复被删除的Conv/BN/Leaky所提供的逐坐标非线性判别能力。不能把“再加一层M”默认视为增加一个普通MLP非线性层。[Möbius算子](https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#mobius_matvec)、[实际层定义](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/models/ops/manifold_layers.py#L97)

### 3.2 50维球坐标直接做logits，CE有约2.94的下界

Geoopt的 `mobius_matvec` 默认投影；当前FP32、曲率绝对值1时半径上限是0.996。分类和本地分割都直接将这些坐标交给CE。[Geoopt包装算子](https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/manifold.html#Stereographic.mobius_matvec)、[Geoopt投影源码](https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/stereographic/math.html#project)

本次独立推导：设类别数K、logits为z且||z||₂≤r。由指数函数Jensen不等式和Cauchy–Schwarz，

`CE ≥ log[1+(K−1)exp(−r sqrt(K/(K−1)))]`。

等号可由正确类坐标 `r sqrt((K−1)/K)`、其余各类坐标 `−r/sqrt(K(K−1))` 取得。K=50、r=.996时：

- 正确类最大softmax概率：0.05286458845；
- 最小普通CE：2.94002156973；
- 均匀猜测CE：log(50)≈3.9120。

因此CE约2.97可以非常接近该输出空间的最优值，不能解释为模型几乎未学会，也不应与无界欧氏logits的CE直接比较。该限制不能单独证明mIoU下降：argmax仍可很准确。

一个正比例logit scale在推理时不会改变argmax，因此单独给已有checkpoint乘温度不能提高mIoU。若要检验它，必须作为新的训练读出方案，让CE梯度改变，同时保持原嵌入及正则路径。`CE(s*z)`只改变读出，R仍用原球点；应做同结构R-off/on配对。它属于有意改进，不能冒称补上官方遗漏。更一般的双曲超平面分类器也属于新模型，不能仅凭本地未启用的GeodesicLayer名称就认为可无风险替换。

## 4. 边界现象应怎样解释

FP32半径0.996对应最大原点双曲距离 `2atanh(.996)=6.212606`，而 `EXP` 的欧氏输入范数超过 `atanh(.996)=3.106303` 就会触发球投影。对part200，当前层次margin=5；若要满足层次hinge，whole深度必须≤1.212606，即whole半径≤0.541519。whole与part若都靠近数值边界，这个约束必然违背。

但part靠外正是分割正则的目标。不能将“part贴边”与“训练已经失败”画等号。需要分别观察whole向内移动、part向外移动、活跃hinge比例，以及这部分梯度经EXP前后是否仍存在。

后置M可以通过权重及输入方向产生新的输出半径，给其自身参数提供有效正则梯度；它不能把已经由EXP+投影删除的输入径向信息恢复出来。I正则梯度非零说明该计算图仍有可优化方向，不证明所有原始H的径向分量都恢复了，也不证明分割成绩获益。

## 5. 对下一步方案的约束

1. 冻结H或另一个已选主支切分，先停止同时调整多处层次。报告每个新变化是论文重建、数值/读出方案，还是额外正则设计。
2. 优先做已有checkpoint的诊断：logit范数/正确类概率、CE与2.940理论下界的距离、whole/part输入范数和投影率、CE/Rcontr/Rhier在共享参数上的范数与夹角。损失值小不能代替有效梯度判断，梯度非零不能代替性能收益。
3. 若诊断确认入球前饱和严重，研究统一的入球前尺度，保持同一个尺度在主支及R支路、曲率不变，和原头做R-off/on比较。此时不同时更改margin、层数、BN和初始化；尺度的设定只用训练/验证信息。
4. 若优先保留原球嵌入几何，先评估仅在CE前加正的固定logit scale，配对关闭/开启R；不要把降低CE本身当作成功，应看mIoU及每类表现。
5. 若要判断I是否因正则受益，新增I0必须保留whole→part的两次训练态前传及BN更新，仅让正则不参与loss。跳过part前传会同时改变BN，已有H0不能作为I的严格零正则对照。
6. 原论文未公开精确分割实现的部分必须保持“不确定”标记；论文消融没有证明M是R有效的必要条件，也没有给出双曲分割零R成绩。单seed和跨硬件/迁移训练的比较只能提供当前配置下证据。

这些是待用户审阅的研究建议，不是新实验启动安排。
