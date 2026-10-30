# E 组尾部修改边界与训练协议复查

2026-10-30。本项新做本地源码静态核对，以及一次 CPU 构造器检查（未前向、未反向、未优化、CUDA 未初始化）。既有前向/梯度合同及生产验收仅作为既有证据引用，没有重跑训练。本项未修改任何训练源码、配置、模型或服务器结果。

## 可以保证的范围

**就网络结构而言，E 的修改边界确实在原 DGCNN partseg 的 Conv9/BN9/LeakyReLU 之后。Conv9 及之前的 STN、动态图、三个局部阶段、全局最大池化、类别 one-hot 支路、1280→256 融合、BN8、LeakyReLU、dp1(.5) 全部保留。**

严格的逐层对应为：

```text
两者共同且保留：
  STN → 三阶段 EdgeConv → local192 + broadcast(global1024, category64)
  → concat1280 → Conv8(1280→256) → BN8 → LeakyReLU → dp1(.5)
  → Conv9(256→256) → BN9 → LeakyReLU → H[B,N,256]

我们跑过的普通 DGCNN：
  H → dp2(.5) → Conv10(256→128) → BN10 → LeakyReLU
    → Conv11(128→50) → plain CE

E：
  H → EXP → 官方完整 M(256→128) → 官方完整 M(128→50) → plain CE
```

因此，不能把这项改动简写成“只把两层卷积换成同维双曲线性”：实际同时删除尾部 **dp2、BN10、LeakyReLU**，并新增入球 EXP 和双曲层的参数化/投影。也没有把原 Conv9 变成同宽 M1：E 中没有该 M1。

一手源码依据：原普通基线封装逐层前向见 [v1/model.py:46](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/model.py:46)；原层定义及尾部见 [vendor/dgcnn_model.py:254](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/vendor/dgcnn_model.py:254) 和 [同文件:305](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/vendor/dgcnn_model.py:305)。E 的直接父类实际继承并实例化原 partseg，删除旧尾部，保留原 Conv8/dp1 等模块，见 [v2/model.py:90](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/model.py:90)；完整共同前向见 [同文件:106](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/model.py:106)。v5 从同一构造起点恢复原 Conv9/BN9，见 [v5/model.py:27](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/model.py:27)，实际 H 与 EXP/M/M 前向见 [同文件:51](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/model.py:51)。

## 初始化、参数和 BN 别名

v5 在 `fork_rng` 中按原构造起点建立参考原 DGCNN，取回其 Conv9/BN9；这一恢复不额外推进外部全局 RNG，也不覆盖已保留的 M2/M3 初值。`bn9` 与 `conv9[1]` 是同一模块别名，不是新加第二个 BN。旧 Conv10/Conv11/BN10/dp2 和被删除的 M1 不残留在可训练参数中。依据 [v5/model.py:27](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/model.py:27)、[结构合同:59](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/test_contracts.py:59)。

本次仅 CPU 构造器实测（相同 seed4780、k40、dropout.5）：

| 检查 | 结果 |
|---|---|
| 原 DGCNN 与 E 同名共享 state 条目 | 141 项逐项完全相等，包含保留层权重和 BN 状态 |
| 原 DGCNN 可训练参数 | 1,460,361 |
| E 可训练参数 | 1,460,283，净少78 |
| E 头四个参数张量类型 | `emb2._weight/_bias`、`mlr._weight/_bias` 都是普通 `Parameter` |
| 构造结束后的全局 Torch RNG 与原 DGCNN | **不相同** |
| CUDA 初始化 | false |

净少78与删改相符：旧 Conv10+BN10+Conv11 为39,424个参数；两层 M 的权重及其 `_bias` 共39,346个参数。BN10原有256个仿射参数，M 新增178个参数，差−78。这里的 `_bias` 是官方重参数化中的参数，不能解释为简单输出加性偏置；定义见 [native.py:74](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/native.py:74)、[native.py:83](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/native.py:83)。

现有合同已核对：保留层共享初值及原 Conv9 初值完全相等，[test_contracts.py:87](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/test_contracts.py:87)；在显式给定相同 dropout RNG 后，train/eval 两种模式下到 Conv9 的输出与原 DGCNN 精确相等，[同文件:117](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/test_contracts.py:117)；14个唯一 BN 每批更新一次，[同文件:162](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/test_contracts.py:162)。先前本机及服务器各26项合同、完整 CPU 配对入口通过的既有记录见 [113/cpu_acceptance.json](../113_PSEG3_DE_REG_CONTROL_2026-10-29/cpu_acceptance.json)、[113/queue_acceptance.json](../113_PSEG3_DE_REG_CONTROL_2026-10-29/queue_acceptance.json)。

**必须区分：保留层结构/初值一致，不代表与原 DGCNN 的整个训练随机流一致。** v5 新建三层 M 后再删同宽 M1，其初始化消耗的随机数仍推进了全局 Torch RNG；上述新实测确认原 DGCNN 与 E 的构造结束 RNG 不同。现有“构造不额外移 RNG”合同实际上与 v3 B 配对，而非与原普通 DGCNN 配对，见 [test_contracts.py:106](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/test_contracts.py:106)。train/eval 共同前端精确回归合同是在两次前向前手动重设相同 seed113，不能据此声称实际 baseline/E 全程 dropout mask 相同。

D/E 之间则是完整初始 state、构造后 RNG、首前向/dropout 和独立采样 RNG 都配对，见 [test_contracts.py:219](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/test_contracts.py:219) 及既有 CPU 入口证据。也不能将“骨干未改结构”说成“骨干权重被冻结”：E 仍将 `model.parameters()` 全部交给优化器，骨干会接受新头传回的梯度，全网联合训练，见 [v5/train.py:191](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:191)。

## E 无正则是否真的成立

成立。E 的显式系数只能为0/0，混合系数会拒绝；实际训练总损失直接取 `ce` 对象，而非把一个可能影响图的零正则相加，见 [v5/train.py:81](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:81)、[v5/train.py:304](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:304)。

E 仍用 detached H、`no_grad` 计算 whole/part 几何，供观察使用；两项有效正则返回无梯度的严格零，仅推进其独立 part/negative RNG，不重复编码点云、不更新额外 BN，见 [v5/losses.py:39](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/losses.py:39)。既有合同逐个核对 E 总损失反传等于纯 CE 的 H/所有可训练参数梯度，并核对不改变全局 RNG，见 [test_contracts.py:279](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/test_contracts.py:279)。因此该观察支路不构成隐藏训练正则。

## 不能保证“整个实验只改最后几层”

“只改最后几层”可用于描述网络结构，不能用于描述 E 与普通 DGCNN 的全部训练配方。明确存在优化器差异：普通基线使用 PyTorch SGD，E使用 Geoopt RiemannianSGD；相同 lr/momentum/weight_decay 名字不构成逐步更新等价证明。代码见 [v1/train.py:180](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/train.py:180)、[v5/train.py:191](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:191)。当前普通 M 层默认 `over_param=False`，`_weight/_bias` 都是普通 Parameter，所以不能只用“头是双曲空间，因此所有参数都必须用流形优化”解释全部全网优化器差异；本项不因此擅改现有协议。

以下要逐项区分：

| 项目 | 我们普通 DGCNN 基线 | E | 审查结论 |
|---|---|---|---|
| 编码、融合、Conv8/9、dp1 | 社区原结构 | 同结构、共享初值 | 已核对 |
| Conv9之后 | dp2/Conv10/BN10/Leaky/Conv11 | EXP/M/M | 有意尾部组合替换 |
| 损失 | `F.cross_entropy` | 官方 `cal_loss(...,smoothing=False)` | 都是 plain CE，未引入 smoothing 差异 |
| 优化器 | PyTorch SGD | Geoopt RSGD | 真正额外的协议差异 |
| LR调度 | .1→.001、200轮余弦 | 同 | 同量纲配方；不代表优化器等价 |
| 训练/评估BN切换 | 每轮train，评估eval/no_grad | 复用同一评估函数 | 当前无额外BN模式分叉 |
| 数据类、loader | v1 ShapeNetPart/loader | 直接导入v1同一实现 | 代码共享，不是重新实现 |
| 同seed后的全局Torch流 | 原模型构造消耗 | 多出M层初始化消耗 | baseline/E未严格配对；D/E已配对 |

plain CE 一手依据 [v1/train.py:272](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/train.py:272)、[v5/train.py:102](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:102)、[native.py:141](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/native.py:141)。社区原始入口 `cal_loss` 是否默认 smoothing 是另一项来源对照，不能误说成我们此次 E 相对我们已跑基线改了 CE。

数据/loader/评估的直接导入见 [v5/train.py:21](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:21)。训练样本仅做同顺序的点/标签 permutation，见 [data.py:42](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/data.py:42)。loader的shuffle/drop_last/worker种子策略见 [v1/train.py:92](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/train.py:92)。普通基线和E的train/eval切换分别见 [v1/train.py:250](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/train.py:250)、[v5/train.py:285](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:285)、[共用evaluate:109](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/train.py:109)。共用evaluate以 `model.eval()` 评估，训练下一轮重新 `model.train()`；E没有覆盖 `train()` 的特殊方法。

k40、dp1.5、batch32、seed4780等可见 [E配置](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/configs/tail2_e.json)。不要只看baseline JSON默认workers4就认定实际训练workers改变：此前111对生产manifest审计记录普通基线与A/B/C均workers8、同batch与同数据；见 [111/audit.md:9](../111_BATCH_AND_HEAD_CAUSES_2026-10-29/audit.md:9)。D/E使用workers8的已审阅配方见 [113/plan.md:33](../113_PSEG3_DE_REG_CONTROL_2026-10-29/plan.md:33)；如需对“实际历史运行”做更强逐步一致性保证，应以相应生产manifest及逐步记录核验，而不是从parser/config默认值推断。

## 对后续结论的约束

1. D−E 是当前架构、优化器及其他配方固定条件下，两项正则联合开关的有效对照。
2. E−普通 DGCNN 同时涉及尾部替换、SGD/RSGD及构造后随机流；不能将全部差值归因于单一 Möbius 算子或正则。
3. “按 HyCoRe 的方法迁移了尾部”是有源码支持的描述；“完整精确复原作者未公开的 HyCoRe-seg 全训练实现”超出了现有证据。
