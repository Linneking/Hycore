# 实验参数命名与HIER梯度缩放范围

日期：2026-10-20。按用户最新决定重整V8和原版修改线；早期V1–V7仍由33旧规范负责。本规范更新其“几何参数和seed只放元数据”的旧约定。旧运行身份保持，通过[机器映射](experiment_registry_v2.json)使用新的展示名。

## 命名与未来默认协议

```
{谱系}-{采样族}[-K20-W0]-C{曲率}-CAP{切空间上限或none}-GS{0/1}-M{gamma}_{delta}-{选模}-S{seed}
```

| 字段 | 含义 |
| --- | --- |
| ORIG / V8 | 原版训练流程的修改线 / V8实现及其对照。关闭HIER不会将V8变成ORIG |
| B0-32 / B64 / HIER64 | shuffle32 / 类别均衡global64且关闭HIER / 同类别均衡global64开启HIER |
| K20-W0 | HIER近邻参数20、无warmup；只有HIER族需要 |
| C0.1 / C1 | 全链双曲运算曲率 |
| CAP2.3 / CAPnone | whole/part最终切空间模长上限2.3 / 不加此上限，仍有数值安全投影；HIER版代理用相同前向cap |
| GS1 / GS0 | 启用 / 关闭逆度量梯度缩放；GS不是norm裁剪、AMP GradScaler或学习率 |
| M500_2 / M1000_4 | Rhier阈值500/N、Rcontr阈值2 / 原阈值1000/N、4；两损失系数仍各0.01 |
| TS / VS984 | 全9840训练、每轮2468 test选best / 历史8856训练、984独立验证选best |
| S4780 / S22 | 实际随机seed；以后ORIG修改线4780，V系列22 |

今后新实验使用TS、官方9840训练，不再另划984验证集；test只评估，不反向、不更新BN。按300轮完整预算保存best，不因test早停。历史VS984不重标TS、历史seed不改写。官方launcher本来显式使用4780；22是Python参数未传seed时的fallback，不将二者混淆。

本次新组：`ORIG-B0-32-C0.1-CAP2.3-GS1-M500_2-TS-S4780`。对照61为完全相同参数名中的`GS0`。

名称用来定位主要条件；完整可复现性仍需manifest中的源码、数据行顺序、增强/随机流、BN、损失、优化器、worker、评估batch和环境。ORIG与V8即使C/CAP/GS/M相同，仍有seed、随机流等差异，不能据此声称完全相同实现。

## 既有实验的展示映射

以下是身份映射，不改变已有目录、日志、checkpoint和结果。VS984与TS是不同训练，不能合并。

| 旧称或范围 | 新展示名 |
| --- | --- |
| ORIG-B0-32-S4780 | `ORIG-B0-32-C1-CAPnone-GS0-M1000_4-TS-S4780` |
| V8-B0-32-C01 | `V8-B0-32-C0.1-CAP2.3-GS1-M500_2-VS984-S22` |
| V8-B64 | `V8-B64-C0.1-CAP2.3-GS1-M500_2-VS984-S22` |
| V8-HIER64-K20-W0 | `V8-HIER64-K20-W0-C0.1-CAP2.3-GS1-M500_2-VS984-S22` |
| V8-B0-32-C01-FULL9840-TESTSEL | `V8-B0-32-C0.1-CAP2.3-GS1-M500_2-TS-S22` |
| V8-B64-FULL9840-TESTSEL | `V8-B64-C0.1-CAP2.3-GS1-M500_2-TS-S22` |
| V8-HIER64-K20-W0-FULL9840-TESTSEL | `V8-HIER64-K20-W0-C0.1-CAP2.3-GS1-M500_2-TS-S22` |
| V8-B0-32-C01-NOHOOK-FULL9840-TESTSEL | `V8-B0-32-C0.1-CAP2.3-GS0-M500_2-TS-S22` |
| ORIG-B0-32-C01-S4780 | `ORIG-B0-32-C0.1-CAPnone-GS0-M1000_4-TS-S4780` |
| ORIG-B0-32-C01-CAP23-S4780 | `ORIG-B0-32-C0.1-CAP2.3-GS0-M1000_4-TS-S4780` |
| ORIG-B0-32-C01-CAP23-MHALF-S4780 | `ORIG-B0-32-C0.1-CAP2.3-GS0-M500_2-TS-S4780` |
| 本次新增GS组（已迁8000） | `ORIG-B0-32-C0.1-CAP2.3-GS1-M500_2-TS-S4780` |

新运行从创建时写新名；已运行实验通过registry展示新名，保留原manifest的canonical_name作不可变溯源，不追溯编辑。别名必须同时匹配谱系和选模协议，不能用裸`V8-B0-32`区分两批训练。

## HIER的梯度缩放究竟放在哪里

核对官方HIER仓库固定commit `3986a744a1a54fd357e307d1cb3f2e81910b9ffc` 的本地源码副本：

1. [hyptorch/nn.py](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/hyptorch/nn.py#L176)：`ToPoincare`先截切向量模长，再expmap/project，最后调用grad_fix。前向位置不因grad_fix改变。
2. [hyptorch/pmath.py](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/hyptorch/pmath.py#L30)：`RiemannianGradient.backward`在球内输出x处，将上游梯度乘逆度量因子。
3. [models/model.py](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/models/model.py#L111)：图像网络最终head使用`ToPoincare(riemannian=True)`；样本嵌入经过这里。
4. [losses.py](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/losses.py#L30)：HIERLoss的512个层次代理lcas也单独经过`ToPoincare(riemannian=True)`。样本—代理与代理—代理项都使用这些输出。
5. [train.py](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/train.py#L153)：监督度量损失和层次损失先相加，统一backward。官方AdamW参数组覆盖网络、监督损失模块和层次损失模块，不是只优化层次代理。

$$
s(x)=\frac{(1-c\lVert x\rVert_2^2)^2}{4},\qquad
\widetilde{\frac{\partial L}{\partial x}}=s(x)\frac{\partial L}{\partial x}.
$$

这里的梯度由损失和整条计算图决定，不是某一个样本自行提供的固定数。样本侧会汇合监督度量损失与HIER样本项的梯度；层次代理侧会汇合HIER样本项和代理项的梯度。代理—代理项本身没有到图像网络的直接路径；它改变代理后会在未来间接影响样本项。

缩放不是只管理“HIER正则的梯度”。同一映射输出后，只要损失到参数的反向路径穿过该输出，该路径贡献就会被缩放。因此样本侧缩放会沿head继续传给可训练backbone。HIER中的PA监督分类代理是另一个概念：它们在PALoss_Angle里做余弦归一化，没有经过HIERLoss层次代理的ToPoincare，不能说所有名为proxy的参数都直接接受该缩放。

## V8如何移植，哪些参数并非直接缩放

V8在最终共享whole mu、part nu映射末端各放一次InverseMetric，之后才分支到分类器、HyCoRe正则和HIER；512层次代理各经过自身的一次相同映射。没有在每项loss再加一次，也没有在每层网络重复乘。

```mermaid
flowchart LR
    E[PointMLP及嵌入层] --> S[whole/part最终嵌入：cap＋GS]
    S --> C[分类器与CE]
    S --> B[Rhier与Rcontr]
    S --> H[HIER样本项]
    P[层次代理参数] --> Q[代理映射：cap＋GS]
    Q --> H
    Q --> R[HIER代理项]
```

上图前向向右；反向沿箭头返回。CE、Rhier、Rcontr和HIER样本项传回PointMLP/嵌入层的贡献都经过GS。即使HIER完全关闭，前三项也仍经过GS，因此无HIER的B0/B64开启GS是有效实验条件。

分类器参数位于GS之后：CE对分类器参数的直接autograd梯度不穿过共享mu的GS，故不直接乘该因子。HIER样本项不经过分类器，也没有对分类器参数的直接梯度。注意全模型norm裁剪1、动量、weight decay及后续网络演化可产生间接影响；“直接梯度没过GS”不等于最终分类器更新在两条完整训练中完全相同。

$$
\widetilde{g}_{\theta}=\sum_i J_i(\theta)^{\mathsf T}s(x_i)g_{x_i}.
$$

每个whole、part、proxy有自己的s(x)。因此它会重加权样本/分支贡献，汇合后的网络梯度方向也可能改变；不能理解为“给整个网络统一乘一个学习率”。源代码的RSGD对ManifoldParameter的处理、裁剪及优化器自身变换，也与这个局部GS是不同步骤。cap仅影响前向模长及映射导数，不保证所有点到球壳；GS前向本身是恒等映射。

c0.1且切空间cap2.3处，s约为0.094197；球心处为0.25。这是局部反传梯度的系数，不能直接当成AdamW实际位移比例。新CPU合同以同输入、同权重比较GS0/GS1：前向逐元素一致，嵌入之前的梯度按s变化，位于其后的分类器直接梯度一致；正式GPU仍须独立短测通过。

实现依据：`inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py`的InverseMetric/to_ball/FinalGeometry，`hier_loss.py`的proxies，`train.py`的统一loss.backward；原版GS补充使用相同cap/map函数且保留原RSGD。


## 2026-10-20补充：共享GS、HIER专用GS与cap不能混为一项

当前V8的GS1同时改变CE、Rhier、Rcontr及HIER样本项到共享网络的梯度。这是计算图位置导致的已确认影响，不等于已经证明它损害分类。两组目标共享一个编码器；并非两套独立的网络分别训练。

原版PointMLP+MN40的实际forward不经过HIER式输出GS。代码库虽保留ToPoincare/RiemannianGradient类定义，当前Hype_Model调用的是manifold.expmap0及MobiusLayer，没有调用该grad_fix。原版的RiemannianSGD、流形参数转换、普通映射导数和norm裁剪另算，不能将“没有新增GS”说成“没有双曲梯度处理”。官方HIER的监督PA和样本—层次代理项都通过样本出口GS传给图像网络；HIER层次代理也有自身GS。PA分类代理与512个层次代理不是同一套参数。

可以构造HIER专用GS，基本计算图如下。这是候选设计，尚未修改或启动任何正式训练：

```mermaid
flowchart LR
    E[共享编码器及嵌入层] --> C[cap：不含共享GS]
    C --> B[whole/part：CE及HyCoRe正则]
    C --> S[仅whole的HIER分支GS]
    S --> H[HIER样本项]
    P[独立代理参数及自身cap和GS] --> H
    P --> R[HIER代理项]
```

同一个whole数值分两路：基础损失直接使用，HIER样本项使用GS后的恒等前向版本；part的基础分支不加GS。代理侧保留现有GS。必须先移除共享出口GS，再只在HIER分支添加一次；保留共享GS后追加一次，会让HIER样本侧经历两次缩放，基础分支仍被缩放，达不到隔离目的。不能detach whole，否则会关闭HIER到主网络的梯度，成为另一种实验。

以一个共享嵌入x为例，忽略其他分支并在梯度汇合、全局裁剪及优化器更新之前：

$$
g_{\mathrm{current}}=J^{\mathsf T}s(x)\bigl(g_{\mathrm{base}}+\lambda g_{\mathrm{HIER}}\bigr),
\qquad
g_{\mathrm{HIER\ only}}=J^{\mathsf T}\bigl(g_{\mathrm{base}}+\lambda s(x)g_{\mathrm{HIER}}\bigr).
$$

这使同一状态下的基础分支直接梯度不再经过GS，但HIER样本项仍会更新共享编码器，后续whole和part都会随网络权重变化。HIER代理—代理项对编码器没有直接梯度，先改变代理，再影响未来样本项。全模型norm裁剪作用于合并梯度、共享参数及BN演化仍使两组目标耦合；因此可以说“减少额外改变基础梯度的操作”，尚不能承诺整体影响更小或准确率更高。

在CPU float64小型计算图中用现有InverseMetric/to_ball/official_distance核对了路由：HIER专用GS前向值不变、基础分支梯度不变、总梯度等于基础梯度加HIER分支梯度、同状态代理梯度与共享GS方案相同。该检查是代数路由验证，未使用真实网络/数据或完整gHHC训练，不是分类收益实验；未分配GPU、未加载checkpoint、未更新BN/优化器。

whole和part是同一编码器对不同点云输入的临时输出；512层次代理则是独立可训练参数表，经映射得到球内位置，不是PointMLP输出的第3种输入表示。原HIER的图像网络产生图像/增强视图表示，层次代理也独立；选择PA时另有类别代理。原HIER没有HyCoRe的显式whole-part径向约束。

cap不只是数值安全开关：超过阈值时改变表示和距离，也改变反向导数；被截部分的径向梯度受限，角向仍可变化。V8共享cap同时作用于CE输入、Rhier和Rcontr；c0.1/cap2.3令原点深度上限约4.6，原1000/N在N200要求径向差5，超出该预算。只改为HIER专用GS，基础分支仍受现有c/cap/margin改动影响。若只研究GS作用范围，应保持这些条件一致；进一步把cap也局部化属于另一个方法变化。

当前GS1的名字仍指历史共享出口GS；本段HIER专用GS只是待讨论的候选，不能把现有GS1运行重新解释为HIER专用。


## 用户进一步提出：原生HyCoRe＋等深度HIER分支

本节比上节更进一步：whole/part在基础目标中不使用新增cap或GS；HIER不直接使用原生whole半径，而只使用其方向在深度4.6处的分支表示。代理保留HIER自身初始化、前向cap与GS。该方案尚未实现或启动训练。

令原生whole为mu，HIER曲率为cH，目标深度hH=4.6。在mu非零时：

$$
R_H=\frac{\tanh(\sqrt{c_H}\,h_H/2)}{\sqrt{c_H}},\qquad
\mu_H=R_H\frac{\mu}{\|\mu\|}.
$$

cH=0.1时RH约1.9651。HyCoRe的CE、Rhier、Rcontr继续使用原生mu/nu；只有HIER样本项使用GS(muH)。同一次PointMLP输出分支即可，不需要额外跑一遍编码器，也不增加BN更新。这里“无新增cap”仍保留HyCoRe原有双曲映射和数值边界保护。

如果希望连曲率与margin都保留原版，基础目标可保持c1与1000/N、4，HIER独立使用c0.1；上述映射明确为保留原点方向的跨曲率映射，不是等距映射，不直接混算两个球的坐标距离。如果基础目标保留c0.1，则仍属于原版c0.1修改线。原生mu与muH是同一对象的两种目标视图，不应把muH成壳表述为原生HyCoRe whole成壳。

这较共享出口cap/GS更能保留原任务计算，但不是两篇源码完全复现。官方HIER的ToPoincare只设上界，低于阈值不抬高；实测原版CUB/Cars/SOP whole都接近4.6壳，是已检查权重的结果，不是官方代码对任意样本的硬要求。SOP的反例是“样本同壳仍无法保证浅代理”，不是“SOP whole不成壳”。

该候选确有可证明的机制：当所有HIER whole严格同壳时，对于任意固定非零代理，精确双曲距离的whole近邻排序仅由方向夹角决定，whole原生半径不再影响排序。方向相同的不同深度代理仍会得到相同NN4，故不能保证热点消失；代理位于原点时则全部whole并列。官方距离epsilon和近并列仍需检查数值容差。

固定壳并非只改前向展示。令r=||mu||、u=mu/r，则理想投影的导数为：

$$
\frac{\partial\mu_H}{\partial\mu}=\frac{R_H}{r}(I-uu^{\mathsf T}).
$$

因此HIER到原生mu的直接梯度去掉径向分量，保留角向信号；不应detach whole，否则主网络完全收不到该项梯度，也不能用detach半径或straight-through替代后仍宣称同一导数。r很小时角向导数可能放大，r=0方向未定义，后续实现需有明确数值保护及例外统计。共享网络参数更新仍能改变原生whole/part半径，合并梯度norm裁剪也继续耦合各项目标。

在固定HIER球壳处，样本侧GS系数固定约0.094197；但投影导数仍含RH/r，因此消除的是HIER距离/检索的样本径向偏置，不是消除所有与原生半径有关的梯度效应。代理仍各按自身位置缩放，其前向cap可保证深度不超过4.6，却不能保证严格浅于whole、形成分层或被有效选择。已有SOP代理大量贴cap，构成“whole同壳必然得到浅代理”的反例。

PA与CE只能按监督作用类比：官方PA对X和类别代理归一化后做余弦，固定状态下样本纯径向变换不改变PA前向；HyCoRe的双曲分类器、Rhier及Rcontr则确实依赖半径。这解释了为什么共享cap在HyCoRe基础任务中的影响不能照搬PA来判断。当前候选的可实现性与上述梯度性质已核实；分类收益、代理层级和热点改善尚待后续独立对照，不以这些理论性质代替训练结果。
