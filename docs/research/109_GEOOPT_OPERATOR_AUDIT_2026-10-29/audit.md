# Geoopt 算子与 HyCoRe wrapper 的实际调用审查

日期：2026-10-29。问题：Geoopt 或原 HyCoRe 是否通过自定义算子，避免 EXP 投影后的径向梯度死区。范围是原 ModelNet40 DGCNN 的实际调用链；只写研究记录和CPU局部探针，不改训练源码、不连接服务器。依赖算子核查与 wrapper 核查独立完成，运行时局部探针也已通过。

**结论：Geoopt 有多处避免越界、除零和非有限值的数值安全设计；HyCoRe 有 bias 的 EXP/平行输运重参数化。但未发现已接入该 DGCNN 的径向梯度恢复设计。** “有数值保护”与“对投影截断使用直通回传”不能等同。项目自带 `pmath.Artanh` 确有自定义反向，但这条 DGCNN 路径没有使用它。

## 1. 依赖和源码身份

当前具体库结论限定于 **Geoopt 0.5.1**。`.996` 还要求 **FP32、c=1、默认投影 eps**。原 [README.md:30](/D:/Hycore/README.md:30) 使用未 pin 版本的 `pip install geoopt`，所以当前阈值不能自动追认成作者所有历史环境的常量。分类核心来源身份沿用 [100 号审查](../100_HYCORE_DGCNN_CLASSIFICATION_COMPARISON_2026-10-29/report.md) 中的官方 HyCoRe `ef56a73020e309b1009c486fea1f601d3f7b3252` 核验记录。

以下 Geoopt 行号对应本机 `.codex-local/seg_dev_env` 安装的 0.5.1 源码；HyCoRe 行号对应权威工作副本。不能仅根据文件中定义了某个算子，就推断当前网络实际调用了它。

已联网逐函数对照 [Geoopt v0.5.1 manifold.py](https://raw.githubusercontent.com/geoopt/geoopt/v0.5.1/geoopt/manifolds/stereographic/manifold.py) 和 [math.py](https://raw.githubusercontent.com/geoopt/geoopt/v0.5.1/geoopt/manifolds/stereographic/math.py)：EXP默认投影、project的普通归一化/where、Möbius矩阵乘的范数与反双曲函数链均与本轮关键结论一致。在线文件与本地排版行号不同，表格行号明确以本地文件为准。

## 2. 实际调用链

```text
dgcnn_hmodel.py:20 → models.manifolds.PoincareBall
    → models/manifolds/__init__.py:2 → poincareball.py:8 继承Geoopt
H512 → Geoopt.expmap0 → math.expmap0 → math.project
     → MobiusLayer.forward → Geoopt.mobius_matvec → math.mobius_matvec → math.project
     → mu256 → Geoopt.dist0/dist → 两正则
             → 第二个MobiusLayer → logits → CE
```

H/EXP/M1/M2 分别在 [dgcnn_hmodel.py:170](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:170)、177、182、186行；190行返回mu、logits。实际两头是 [123–124行](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:123) 构造的 MobiusLayer，非 GeodesicLayer 或 MobiusMLR。正则取mu，见 [main_dgcnn_hycore.py:223](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:223)–230和 [hutil.py:53](/D:/Hycore/classification_ModelNet40/hutil.py:53)–57。

## 3. Geoopt 实际算子：安全操作存在，未替换投影径向导数

| 算子 | 一手源码位置 | 行为与反向含义 |
|---|---|---|
| `expmap0` | [manifold.py:327](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:327) | 默认`project=True`；先调用math.expmap0，再调用math.project，未绕过投影回传。 |
| EXP公式 | [math.py:1049](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:1049) | 对输入模长clamp_min防除零，再使用tan_k；c1负曲率分支为tanh模长乘单位方向。 |
| `project` | [math.py:307](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:307)–318 | FP32 eps=.004；`where(norm>R, x/norm*R, x)`。无detach、无STE或自定义径向反向。 |
| `mobius_matvec`包装 | [manifold.py:289](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:289) | 默认对Möbius乘结果再次project；不是普通线性层或投影直通层。 |
| Möbius矩阵乘公式 | [math.py:1224](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:1224) | 使用输入/矩阵乘后模长、artan_k/tan_k与零值分支；存在模长下界等数值保护。 |
| `dist0` | [math.py:890](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:890) | `2*artan_k(norm(x),k)`，c1时为`2*atanh(norm(x))`，不会自行恢复前面project丢失的径向信息。 |
| `dist` | [math.py:854](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:854) | 由Möbius差的模长及artan_k求距离；不是本项目pmath的自定义Artanh。 |
| `tanh` | [math.py:30](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:30) | 输入clamp到±15后tanh，属于普通autograd安全计算，不是直通回传。 |
| `artanh` | [math.py:35](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:35) | 输入clamp到±(1−1e−7)后对数表达式；普通autograd，不采用HyCoRe pmath.Artanh的自定义backward。 |
| `transp0` | [manifold.py:338](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:338)、[math.py:1692](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:1692) | 原点平行输运为向量乘几何缩放因子；它改变有效M权重，不为先前EXP的投影提供逆导数。 |

这些设计使运算更安全，但“是否保证不会NaN”和“某个局部径向导数是否被保留”是不同问题；安全操作也不构成所有输入下绝对不出现非有限值的保证。

`@torch.jit.script`不提供额外反向替代。`@__scaling__(ScalingInfo(...))`也不是当前模型的隐藏输入缩放：本地[base.py:84](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/base.py:84)–91只把规则注册到字典并返回原函数；实际应用这些缩放需要`Scaled`包装，而原DGCNN未使用它。有效算子链也没有把project的norm分母detach，没有投影直通估计器。

Geoopt一手文档推荐双精度，但HyCoRe数据显式float32，主路径没有切到double。仅改双精度也不自动消除该机制：默认数值半径变成.99999，c1切空间触投影阈值约6.103，模长10仍会越过阈值。这里不据此更改任何运行精度。

## 4. HyCoRe 自定义 wrapper：有哪些修改、哪些真正被调用

| 自定义项 | 源码与调用结论 |
|---|---|
| PoincareBall基础封装 | [poincareball.py:8](/D:/Hycore/classification_ModelNet40/models/manifolds/poincareball.py:8)–27仅继承父类、记录dim和便捷属性；未覆盖expmap0、projx、mobius_matvec、dist0、dist、transp0或egrad2rgrad。 |
| `inner` override | [同文件:33](/D:/Hycore/classification_ModelNet40/models/manifolds/poincareball.py:33)–37。确实有自定义：给_lambda_x传self.c；父类inner使用self.k。不能称整个wrapper与Geoopt完全同义，但有效EXP、M乘、dist0/dist、transp0直接调用各自math函数，不调self.inner，故此差异不构成DGCNN的反向保护。 |
| `normdist2plane` | [同文件:50](/D:/Hycore/classification_ModelNet40/models/manifolds/poincareball.py:50)–64有自定义和数值clamp；调用者是[GeodesicLayer.forward:95](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:95)–98。实际DGCNN未构造该层。 |
| `expmap_polar` / `logdetexp` | [同文件:39](/D:/Hycore/classification_ModelNet40/models/manifolds/poincareball.py:39)–48 / 29–31；未进入实际DGCNN；不能把polar公式中的r/2当作初始EXP隐藏缩放。 |
| PoincareBallExact | [同文件:68](/D:/Hycore/classification_ModelNet40/models/manifolds/poincareball.py:68)–83替换retraction等优化别名；实际构造的是PoincareBall。 |
| Euclidean恒等EXP | [euclidean.py:29](/D:/Hycore/classification_ModelNet40/models/manifolds/euclidean.py:29)–33；实际DGCNN用c1 PoincareBall，不用该恒等映射。 |

ModelNet40与ScanObject的`models/manifolds`及`models/ops/manifolds`共四份PoincareBall源码SHA相同；四份Euclidean源码也相同。另 [ops/manifolds/__init__.py:2](/D:/Hycore/classification_ModelNet40/models/ops/manifolds/__init__.py:2)直接重导入`models.manifolds.poincareball.PoincareBall`，没有切换到隐藏版本。有效链未发现算子monkeypatch或gradient hook。

## 5. Möbius层的真正自定义：参数化，不是径向梯度恢复

[manifold_layers.py:62](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:62)–80将默认`_weight/_bias`作为普通Parameter，bias由`expmap0(_weight*_bias)`产生，weight由`transp0(bias,_weight)`构成；[114行](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:114)的forward仅调用`manifold.mobius_matvec(self.weight,input)`，无额外输入缩放、BN、激活、norm cap或自定义backward，也不是普通`W⊗x⊕b`加偏置。

这能改变M输出深度，初始化也可能把贴边X映回内部mu；因此分类的M1后径向正则不能直接等同于分割的`EXP(mean(H))`径向项。但它没有恢复初始EXP在活动投影分支丢失的输入径向分量。默认`over_param=False/weight_norm=False`见[111行](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:111)，且Mobius forward不读取weight_norm。

## 6. PMath 自定义反向确实存在，但这条模型路径没用

| 项目 | 准确结论 |
|---|---|
| `ops.pmath.Artanh` | [pmath.py:18](/D:/Hycore/classification_ModelNet40/models/ops/pmath.py:18)–29：forward先clamp到±(1−1e−5)，保存clamp后的值；backward返回g/(1−saved²)，未乘clamp导数。它对原输入在clamp区外给出非标准外延梯度，不能把该事实漏掉。 |
| 是否被原DGCNN调用 | 实际模型、有效M层和hutil距离经models.manifolds/Geoopt，不调用`ops.pmath.artanh`。DGCNN虽然import hypnn模块，实例化/forward不用其候选头；导入存在不等于计算图中使用。 |
| `ToPoincare.grad_fix` | [nn.py:190](/D:/Hycore/classification_ModelNet40/models/ops/nn.py:190)–229存在；[dgcnn_hmodel.py:137](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:137)构造已注释。其forward依然先EXP/project，再乘RiemannianGradient，不是project的STE。 |
| `RiemannianGradient` | [pmath.py:32](/D:/Hycore/classification_ModelNet40/models/ops/pmath.py:32)–47只在反向乘(1−c||x||²)²/4，既不在该DGCNN链中，也不为project零径向提供逆导数。 |

## 7. 局部梯度结论与运行时探针边界

活动投影分支`P(x)=R*x/||x||`的`J=(R/||x||)(I−eeᵀ)`：径向为零，切向可过；不是整个J为零。理想未投影`d0(Exp0(u))=2||u||`，不能单凭tanh导数小断言纯径向回传消失；触发固定半径投影后直接径向目标才成为局部常数。详细推导见[108号审查](../108_HYCORE_BOUNDARY_PROTECTION_AUDIT_2026-10-29/audit.md)。

这只证明局部分支/坐标/损失的性质，不证明训练永远无法离开该区域。其他损失、方向梯度、共享参数、其他样本、BN状态、动量与WD可能改变后续H；分类M1参数也可改变mu深度。本文不把静态机制写成旧分类checkpoint已测出的贴边率或端到端正则失效。

主审已运行 [operator_probe.py](operator_probe.py)，结果见 [operator_probe.json](operator_probe.json)，Torch2.6.0/Geoopt0.5.1、CPU，无完整网络/点云/checkpoint/optimizer，CUDA未初始化。断言全部通过：

- 运行时比较方法对象：官方源文件加载的PoincareBall之`expmap0/projx/mobius_matvec/dist0/dist/transp0/egrad2rgrad`七项与Geoopt父类方法是同一函数对象，`inner`确为不同函数。不是依据相似函数名猜测继承关系。
- FP32的`projx((1.2,0))`给出Jacobian约`diag(-5.96e-8,.83000004)`；径向为浮点残差，切向非零。输入(.8,.6)也验证了非轴对齐情况下径向近零、切向保留。
- 对`d0(expmap0((4,0)))`，默认投影后球坐标处梯度为`(250.50424,0)`，回到EXP输入为`(0,0)`；关闭project的局部机制对照中，输入梯度为`(2.00015,0)`。该对照不代表提议在实际训练关闭投影。
- 单独输入1.0001、FP64，实际Geoopt的`artanh`因普通clamp导数为0；未用的HyCoRe `pmath.Artanh`因自定义backward给出约50000.25。两者clamp阈值也不同。这证实自定义反向存在于另一条路径，不能把它归入DGCNN，或与project径向导数混为一谈。此超域值仅为机制检查，不声称正常投影输出会超过数学球边界。

探针只能验证所安装版本的算子行为，不等同于旧分类训练的端到端测量。

wrapper核查覆盖两个分类任务中的两套几何目录、包导出、有效M层参数属性和损失输入；“没有接入”由调用链判定，不仅由关键词搜索判定。
本轮不根据这些机制追加实验、修改分叉或调整正在运行的三组配方。
