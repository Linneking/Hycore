# 原 HyCoRe DGCNN 的 EXP 边界防护审查

日期：2026-10-29。范围：ModelNet40 分类实际 DGCNN 调用链；只写本研究文档，不改训练源码、不连接服务器。源码身份沿用 [100 号审查](../100_HYCORE_DGCNN_CLASSIFICATION_COMPARISON_2026-10-29/report.md)中官方 HyCoRe `ef56a73020e309b1009c486fea1f601d3f7b3252` 的核验记录；本轮再次核对官方来源清单中的 dgcnn_hmodel、manifold_layers、poincareball、hutil 四个文件 LF SHA，全部一致。

**结论：实际 DGCNN 没有专门限制初始 EXP 输入 H 的整体模长，也未启用绕过投影径向死区的回传修正。** 有 BN、M层初始化、M1后正则、数值投影和参数梯度裁剪，但不能将这些写成“保证初始EXP不触边”。纯径向在活动投影分支的局部梯度为零，不等于整个投影Jacobian为零，也不证明训练永远回不来。

## 1. 实际调用链与版本条件

```text
ModelNet40数据/XYZ增强 → part、whole各调用同一个DGCNN
DGCNN欧氏骨干 → global max+avg → Linear → BN6 → LeakyReLU → H512
    → expmap0（默认投影）→ X512 → M1(512→256) → mu256 ─→ 两正则
                                                    └→ M2(256→40) → CE
总损失反传 → 全网参数梯度L2 norm clip1 → RSGD更新
```

H/EXP/M1/M2位于 [dgcnn_hmodel.py:170](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:170)、177、182、186行；190行返回`mu,logits`。part/whole两次调用、在mu上的正则及总loss见 [main_dgcnn_hycore.py:220](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:220)–233行。模型构造明确固定c=1、dim512/256，见 [97行](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:97)。

数值阈值“.996”限定于**Geoopt 0.5.1、FP32、c=1及默认eps**。本地 [README.md:16](/D:/Hycore/README.md:16)指向Geoopt，[30行](/D:/Hycore/README.md:30)安装命令为`pip install geoopt`，未pin版本，不能将当前阈值自动追认成作者所有历史环境的常量。本地 [PoincareBall封装:8](/D:/Hycore/classification_ModelNet40/models/manifolds/poincareball.py:8)继承Geoopt，未覆盖expmap0/mobius_matvec/dist0，没有隐藏H归一化。

已联网对照[官方HyCoRe安装说明](https://github.com/diegovalsesia/HyCoRe#installation)及[Geoopt v0.5.1 RSGD源码](https://raw.githubusercontent.com/geoopt/geoopt/v0.5.1/geoopt/optim/rsgd.py)。未成功取得历史Geoopt v0.4.0在线原文件，不声称核实了作者历史运行版本的默认投影行为。

## 2. 实际启用

| 因素 | 有效实现与精确位置 | 作用与边界 |
|---|---|---|
| XYZ缩放、平移、shuffle | [data.py:49](/D:/Hycore/classification_ModelNet40/data.py:49)、[68行](/D:/Hycore/classification_ModelNet40/data.py:68) | 原坐标增强，不是高维H范数控制。 |
| BN6、LeakyReLU | [dgcnn_hmodel.py:122](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:122)、[170行](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:170) | 标准逐通道处理，无逐样本向量模长上限。 |
| 固定c1、dim512/256 | [dgcnn_hmodel.py:97](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:97) | dim是元信息，没有自动除以sqrt(dim)。 |
| EXP及M输出球投影 | [Geoopt manifold.py:327](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:327)、[289行](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:289) | 数值安全球；超阈值时固定半径。 |
| M初始化/重参数化 | [manifold_layers.py:71](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:71)–88行 | Kaiming normal(a=√5)、bias均匀±4/√fan_in；有效weight经bias/transp0构成。均在初始EXP之后。 |
| M1后计算正则 | [分类入口:223](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:223)–227行 | 正则直接更新M1，并可通过输入方向回到编码器。 |
| 全网参数梯度clip1 | [分类入口:230](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:230)–233行 | 总loss backward之后执行，与前向球投影不同。 |
| RSGD、momentum、WD | [分类入口:144](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:144)；默认WD见[44行](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:44) | 优化设置，不是H模长硬约束。普通Parameter不因此自动成为流形激活。 |

## 3. 仅存在但未使用

| 因素 | 源码位置与实际状态 |
|---|---|
| ToPoincare及train_c/train_x/grad_fix | [dgcnn_hmodel.py:137](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:137)构造已注释；类见[ops/nn.py:190](/D:/Hycore/classification_ModelNet40/models/ops/nn.py:190)–229、[manifold_layers.py:10](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:10)–48。实际forward直接调用manifold.expmap0。 |
| RiemannianGradient | [ops/pmath.py:32](/D:/Hycore/classification_ModelNet40/models/ops/pmath.py:32)–47定义；不进当前DGCNN。其反向仅乘(1−c||x||²)²/4，不是投影直通估计器；即使启用ToPoincare，仍会经过project。 |
| over_param=True | [manifold_layers.py:65](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:65)、[111行](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:111)；实际M构造未覆盖默认False，不走ManifoldParameter分支。 |
| weight_norm=True | [manifold_layers.py:111](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:111)默认False；Mobius forward114–116也不读取它，读取者是另一类GeodesicLayer的95–98行。 |
| HypLinear/MobiusMLR/GeodesicLayer/logmap0候选头 | [dgcnn_hmodel.py:133](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:133)–142、179、183为注释或候选，实际头均为MobiusLayer。 |
| 自动维度选c | [ops/pmath.py:494](/D:/Hycore/classification_ModelNet40/models/ops/pmath.py:494)定义auto_select_c，实际DGCNN未调用，c明确为1。 |
| H整体缩放、L2 normalize、norm cap、温度、register_hook | 有效forward、封装、M层和主循环已逐项核查，未发现；170→177之间仅有注释。 |

## 4. 不能保证初始EXP不触投影

| 因素 | 不能提供保证的原因 |
|---|---|
| BN、原始XYZ归一化尺度 | BN控制逐通道统计，原坐标尺度也不直接约束512维H的样本模长。 |
| M1初始化 | EXP之后可以把X映回内部mu，但不能改变本次EXP输入H是否已经超阈值。 |
| Weight decay、动量、RSGD | 参数更新设置，不是前向逐样本H范数约束。 |
| 全网梯度clip1 | 发生在反传后，既不限制本次H，也不创造已为零的径向分量。 |
| 球投影 | 保证数值安全半径内，却让超阈值径向变化不影响局部输出；不是避免触发投影的措施。 |

[launcher_dgcnn.sh:4](/D:/Hycore/classification_ModelNet40/launcher_dgcnn.sh:4)仅指定LR=.1、300轮、message、workers，没有另开H缩放或几何保护选项。

RSGD也没有隐藏的激活梯度恢复功能：当前两层M默认`over_param=False`，其_weight/_bias均为普通Parameter，其余Conv/Linear/BN亦如此，曲率不学习。Geoopt的普通参数走默认Euclidean分支，见本地[rsgd.py:91](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/optim/rsgd.py:91)及[mixin.py:6](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/optim/mixin.py:6)。入口未传stabilize，库默认None；即使启用也只处理流形参数，不能为中间H或EXP输出恢复投影处已为零的导数。这里不声称RSGD的所有动量细节与torch SGD完全相同。

## 5. a/a化简的条件与一般Jacobian

Geoopt 0.5.1 [math.py:307](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:307)–318使用严格`norm>maxnorm`条件，且范数分母未detach。令r=||x||、安全半径R，则：

```text
P(x) = x                         r ≤ R
       R*x/||x||                 r > R
二维x=(a,0)、a>R>0、只改变a： P_x=R*a/sqrt(a²)=R*a/a=R，∂P_x/∂a=0。
一般x=(a,b)、保持b不变：       ∂P_x/∂a=R*b²/(a²+b²)^(3/2)。
一般活动分支r>R，e=x/r：       J_P=(R/r)*(I−eeᵀ)。
```

因此`a/a`只是在二维轴向例子中恰好沿径向改变时的化简。一般b≠0时上面的偏导非零；即使在(a,0)，切向偏导`∂P_y/∂b=R/a`仍非零。**任何方向的纯径向变化都被消除，因为J_P*e=0；切向变化仍能通过。** 反传时纯径向上游梯度被J_Pᵀ消掉，一般梯度的切向分量保留。这不是整个J为零；分段接缝本身也不能套用分支内部的可微性结论。

用户标出的`a−a`来自商的求导法则`(f/g)'=(f'g−fg')/g²`：此处f=a、g=a，两者导数都是1，所以分子为`1*a−a*1`。这当然必为零，因为该分支中`a/a`本来就是常数1；它不是把一般向量投影都写成常数的理由。

另用标量计算核对一般二维例子x=(.8,.6)、R=.996：输出(.7968,.5976)，J的两行分别为(.35856,−.47808)、(−.47808,.63744)。J乘径向(.8,.6)为零（双精度残差约5.55e−17），乘切向(−.6,.8)为(−.5976,.7968)，明确不是整个Jacobian为零。

## 6. EXP复合：局部径向死区不等于训练死局

理想c1、未投影时`d0(Exp0(u))=2*atanh(tanh(||u||))=2||u||`；不能仅因tanh导数小断言纯径向回传弱。FP32/Geoopt0.5.1/c1默认R=.996，当`||u||>atanh(.996)≈3.1063`，`P(Exp0(u))=.996*u/||u||`，其原点距离固定为`2*atanh(.996)`；直接纯径向loss对u的解析梯度为零，有限精度可能留下极小残差。机制与局部CPU证据见[107号审查](../107_EXP_RADIAL_GRADIENT_AUDIT_2026-10-29/audit.md)。

这证明的是特定分支、坐标及损失方向的局部事实。实际还有其他损失、共享参数、其他样本、BN状态、动量和WD，可能改变以后生成的H。分类又在M1输出mu上算正则：M1矩阵及X方向可以改变mu深度，若mu未被其自身投影固定，径向项仍可更新M1并沿方向路径回到编码器。若把u当孤立叶子、只用已投影后的纯径向loss且没有其他更新，它则确实没有局部径向恢复驱动力；两种情形不能混同。

一个简化M变换的计算例子：令X半径已经是.996，有效矩阵M=sI且s=.5，则Möbius输出半径`tanh(.5*atanh(.996))≈.914304`，回到球内部；其深度为`2*s*atanh(.996)≈3.106303`，对s的导数为`2*atanh(.996)≈6.212606`，并非零。这是数学局部示例，不是实际512→256矩阵的训练测量，也不是作者明确声明的anti-saturation设计；它解释后置可学习矩阵为何使分类不必随初始EXP贴边而停止学习。

本轮未测旧分类checkpoint的H/mu贴边率或端到端分项梯度，不能将静态机制写成旧分类已观测失效，也不据此改动当前三组分割实验。
