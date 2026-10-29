# 后置 Möbius 层能否避免径向死区

日期：2026-10-29。范围：原 HyCoRe DGCNN 分类与当前 Pseg3 的径向正则路径。本轮只做源码与局部机制审查及CPU探针，不运行真实网络/checkpoint诊断、不训练、不连接服务器、不改变实验配方。新 CPU 共享 M whole–part hinge 探针已完成。

**结论：不能断言原分类没有径向死区，这尚未通过真实网络梯度测量验证。** 后置 M 不能恢复初始 EXP 投影已经消掉的 H 纯径向导数；但当 M 输出没有触发自己的投影时，其输出深度可以依赖输入方向和 M 参数，因而完整径向正则可能仍有学习通道。可能存在不等于必然有效，单分支、单深度探针也不足以证明共享 M 的完整 pair 损失有效。

## 1. 两条真实路径

```text
原分类：XYZ_whole / XYZ_part → 分别编码 → H512 → EXP+投影 → X512
                                                → 共享M1(512→256) → mu256 → Rhier
当前分割：同一H[B,N,256] → whole/part分别mean → EXP+投影 → z256 → Rhier
```

分类 H/EXP/M1 位于 [dgcnn_hmodel.py:170](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:170)、177、182行；M1定义见 [123行](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:123)。[分类入口:223](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:223)–227对两个输入各编码，再在两个mu上算正则。分割见 [v3/losses.py:148](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/losses.py:148)–161，其 mean→EXP 分叉是用户已确认的分割论文设计，本审查不将其视为bug。

分类径向项为`β*mean(relu(-d0(mu_whole)+d0(mu_part)+γ))`，β=.01、γ=1000/Npart。证据：[hutil.py:42](/D:/Hycore/classification_ModelNet40/hutil.py:42)、53–69，入口227行未覆盖默认`opposite_hier=False`；[入口230行](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:230)乘.01。它的下降目标是whole向外、part向内。

## 2. M不能恢复被初始投影消掉的H径向导数

令c=1，安全半径R，H=u，e=u/||u||。当初始EXP超出数值安全半径，有：

```text
X=P(Exp0(u))=R*e
∂X/∂u=(R/||u||)*(I−eeᵀ)
(∂X/∂u)*e=0
```

所以任何仅改变u模长、保持方向e的局部扰动，都不会改变X，也就不能经后置M改变mu。对任意后置可微映射和loss，经过这个分支回到u的梯度，其相对u的纯径向分量在解析意义下仍为零。有限精度可能留下极小残差，不能将残差当作恢复机制。

这个结论不意味着全部梯度为零：`I−eeᵀ`保留切向方向。数值阈值R=.996只限定于Geoopt0.5.1、FP32、c1的默认投影。投影与继承链证据见 [109号算子审查](../109_GEOOPT_OPERATOR_AUDIT_2026-10-29/audit.md)。

## 3. M输出未投影时：方向与参数可控制输出深度

用M表示该层**实际有效矩阵**，包含原类weight属性里的bias/平行输运重参数化，不把它等同于裸`_weight`。原层forward只执行Möbius矩阵乘，见 [manifold_layers.py:71](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:71)–80、114–116。在初始X=R*e且M输出未触发自己的投影时：

```text
mu=Exp0(M*Log0(X))
d0(mu)=2*atanh(R)*||M e||
```

即使X的半径固定，不同方向e通常有不同`||M e||`；改变M参数也可改变该值。因此径向项可以在M输出深度上学习，并经输入方向回到H。这是方向→输出深度的通道，不是恢复H模长的通道。若M具有特殊对称性、两个方向相似、共享梯度相消，方向或参数梯度仍可能很弱。

上述公式的“未触发输出投影”条件不能省略：Geoopt的`mobius_matvec`默认还会project，见 [manifold.py:289](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:289)。矩阵可将X映回内部，也可能把输出推到安全边界，不能认为有M就自动没有死区。

## 4. 必须看whole–part共享M的完整hinge

设`a=atanh(R)`；两路初始X均投影、两路M输出均未投影时，单个pair的激活hinge为：

```text
L_pair=β*(-2*a*||M e_whole||+2*a*||M e_part||+γ)
```

这里同一个M同时出现在两项中。求共享M参数梯度时需要把两项相加，不能用某一路`d0(MX)`梯度非零，替代完整`L_pair`梯度非零。对batch求mean后，不同pair之间也可能进一步抵消。若hinge已满足，径向项本来就应不产生梯度；恰好为零处按源码所用`torch.max`的次梯度规则处理。

| M输出状态（同一个pair） | 可作的条件性判断 |
|---|---|
| whole、part均未投影 | 方向/矩阵通道可能存在，仍需看hinge是否激活及共享梯度是否相消。 |
| 只有一路投影 | 该路自身纯深度导数被截断；另一路可能仍可学习，不能推断整个pair、batch或网络均无效。 |
| 两路均处于活动投影分支 | 两路深度均固定于数值上限，径向hinge局部成为常数；该pair的纯Rhier解析梯度为零，浮点残差除外。 |
| batch内部分pair两路投影 | 只说明这些pair局部无径向驱动，不能推出其他pair或整个batch的Rhier梯度为零。 |

即使某个pair的Rhier局部无效，也不能推出训练永久停滞：CE、对比项、其他样本、共享参数、BN状态、动量和WD仍可能改变后续特征。反之，也不能因为模型总梯度或CE梯度非零，就认定Rhier正在有效建立深度差。

## 5. 除M外，保留三个影响归因的主要差异

1. **分别编码与同H池化：** 分类在XYZ上裁剪并复写前缀，whole/part各跑完整网络（[hutil.py:27](/D:/Hycore/classification_ModelNet40/hutil.py:27)–40、分类入口220–224）；分割同一次H上选特征邻域再mean，索引离散而gather值可导，且不修改H（[v2/losses.py:14](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/losses.py:14)–37）。
2. **维度与统计：** 分类是独立H512、BN6按B统计且两次编码；分割H256、conv8/BN8按B与N统计，只编码一次（[v2/model.py:97](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/model.py:97)、106–121；[v3/train.py:291](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/train.py:291)–292）。BN不是样本模长上限；训练whole也不直接使用part刚更新的running均值做归一化。
3. **径向方向：** 分类whole外/part内；分割whole内/part外（[v3/losses.py:69](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/losses.py:69)–72）。两者输入分布、池化/BN和目标均不同，不能将实际差异全归因于有无M。

## 6. 已有证据的不足与本轮局部补充

[107号探针](../107_EXP_RADIAL_GRADIENT_AUDIT_2026-10-29/audit.md)的3→2单实例`d0(M(exp(u)))`证明方向/M参数通道可以存在，但它没有真实whole–part pair、完整hinge、共享M梯度相消或分类特征分布，不足以证明原分类完整Rhier无死区。109号探针核查实际算子归属与投影梯度，也没有测分类checkpoint。

主审已运行 [pair_probe.py](pair_probe.py)，结果为 [pair_probe.json](pair_probe.json)。Torch2.6.0/Geoopt0.5.1、CPU/FP32，CUDA未初始化；没有完整网络、真实点云、checkpoint、优化器或参数更新。两路均使用同一个M，损失为原分类方向的`.01*relu(d_part-d_whole+2.5)`，四例hinge均严格激活；在这些点与原`torch.max`实现的梯度一致。

二维例子均取whole/part的EXP输入模长10.5/10.8、方向(.8,.6)/(.6,.8)，初始EXP输出半径均约.996。结果：

| 情形 | 正则位置的whole/part半径 | 加权Rhier→H总L2 | H径向L2 | H方向L2 | M参数梯度L2 |
|---|---|---|---|---|---|
| 直接EXP，无M | .996/.996 | 1.58e−8 | 6.28e−15 | 1.58e−8 | 无M |
| M=diag(.2,.4)，输出内部 | .714344/.786281 | .00152839 | 6.51e−11 | .00152839 | .0224193 |
| M=2I，两路输出再次投影 | .996/.996 | 3.17e−9 | 1.11e−16 | 3.17e−9 | 2.34e−8 |
| 官方同款独立512→256层，输出内部 | .858002/.844388 | .00473701 | 3.11e−11 | .00473701 | _weight .0860021；_bias .000218581 |

第一、三行的残余梯度处于浮点抵消量级，不当作有效学习信号。第二行的共享pair梯度已经同时包含whole和part两项，证明本受控例子没有把单路梯度误写成完整Rhier；它不保证所有pair、所有batch都不相消。第二行输出深度分别1.791998/2.123231，损失.0283123；即使正则局部方向是球坐标径向，回到H也可以几乎完全是方向梯度。

此处H指两个受控的EXP输入向量，不是生产分割的完整H[B,N,256]；没有生产batch平均及点平均梯度分发。因此不将这里的总L2与106的生产1e−10直接作定量增益比较，也不跨二维/512维行评价模型优劣。

第四行复用官方MobiusLayer类和初始化规则（seed4780），输入方向由独立固定CPU随机数生成后归一化，模长仍10.5/10.8。它是**独立层的官方初始化规则实例**，不是完整DGCNN构建骨干后消耗相同RNG的M1权重，也不是任何已训练分类checkpoint。它补充说明相同512→256接口/官方参数化下通道确实可以存在；不声称原分类真实训练全程无死区。

本轮没有实测单路投影或专门构造共享梯度完全相消的例子；这些条件仍是第4节的数学判断。Möbius算子公式来自已核验的[Geoopt v0.5.1原始实现](https://raw.githubusercontent.com/geoopt/geoopt/v0.5.1/geoopt/manifolds/stereographic/math.py)，有效层参数化来源为第3节引用的官方HyCoRe源码。

对于当前三组分割，M头仍存在于CE分支，而Rhier从H的mean→EXP分出，绕过M。删不删CE头的M1，不会直接为这条Rhier新增M参数/方向转深度通道；它仍可能通过CE训练改变以后生成的H，因此不能说两组整个训练过程径向表现必然相同。本文不改变该论文分叉或任何运行配置。
