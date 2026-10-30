# DGCNN 官方/社区分割、E 的改动边界与 HyCoRe 训练设置

日期：2026-10-30。当前生产 D/E 源为 `9033a80135d1b45b0ce11bc7e2135f79ddc5a8a0`。本次只研究与审查；不改变模型、训练配方、数据或服务器结果，不追加实验。构造器级 CPU 核查仅实例化模型并比较 state/RNG，没有前向、反向、优化器或 CUDA。

## 结论

1. 我们使用的 AnTao 社区 PyTorch 分割模型与作者 TensorFlow 分割的大拓扑一致；并不是把分类模型冒充原版分割。但 k、激活、Dropout、BN、bias、初始化、训练与评估实现有具体差异。
2. **E 的架构修改确实只发生在原 Conv9/BN9/LeakyReLU 之后**，完整保留 dp1 和前端；不过 E 相对我们已跑的普通 DGCNN 还使用了 RSGD，不是训练协议完全相同的“只换层”消融。骨干仍端到端训练，没有冻结。
3. 官方 HyCoRe DGCNN 没有查到 part-eval/whole-train 的特殊 BN 策略。分类 part/parent 都在 train 前向，测试才 eval/no_grad；当前分割的一次编码/一次 BN 更新与论文的特征聚合路线相符。
4. 值得继续讨论的非结构设置有：训练/推理 BN 特征尺度差异、分类实际启用的标签平滑、分类全网 clip1、完整初始化/参数化及输出尺度。哪些已经继承、哪些是用户明确批准的偏离，需要分别记录，不能混为正则的效果。

## 来源及版本

- 作者 DGCNN：`WangYueFt/dgcnn@f765b469a67730658ba554e97dc11723a7bab628`。作者 PyTorch 路径是分类；本次对应 ShapeNetPart 的作者入口是 `tensorflow/part_seg`。[README](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/README.md)、[分割模型](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/part_seg_model.py)。
- 社区分割：`antao97/dgcnn.pytorch@07d534c2702905010ec9991619f552d8cacae45b`，由作者仓库列为 Other Implementations。我们 vendored 文件仅把 graph 的硬编码 cuda 设备改成输入设备及清理尾部空白。[来源记录](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/vendor/UPSTREAM.md)、[社区模型](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/model.py)。
- HyCoRe 分类复用源：`ef56a73020e309b1009c486fea1f601d3f7b3252`。[固定来源清单](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/vendor_manifest.json)。
- HyCoRe-seg 一手论文：[作者接收稿，§3–4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。目前能核验的公开 HyCoRe 仓库提供分类；本次没有取得可确认属于论文的完整分割源码，不能声称恢复了作者未公开的所有细节。

## 1. 官方和社区的共同拓扑与差异

共同主线：输入3D变换 → 三段动态 EdgeConv（段内64→64、64→64、64）→ 三个64维局部特征拼成192 → 映射1024并全局max → 拼类别one-hot经16→64编码 → 全局/类别广播与192局部拼成1280 → 逐点256→256→128→50。保留点轴，没有点下采样或把每点特征池化掉；全局池化是旁路上下文。[作者分割模型](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/part_seg_model.py)、[社区模型](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/model.py)。

| 项目 | 作者 TensorFlow 分割 | 当前社区底座 / D/E 保留部分 |
|---|---|---|
| kNN 的 k | 20 | 40 |
| 激活 | ReLU | LeakyReLU(.2) |
| 原尾部两次 Dropout | keep .6，即 drop .4 | 原社区 drop .5；D/E只保留dp1 |
| 原卷积 bias / 初始化 | 有bias；Xavier | bias=False；PyTorch默认初始化 |
| BN eps | 1e−3 | 1e−5 |
| BN统计更新 | TF训练传入衰减调度，decay .5→最高.99 | PyTorch固定momentum .1，即旧统计权重.9 |
| edge拼接通道 | [中心,邻居−中心] | [邻居−中心,中心] |

edge通道反序不改变可表达函数族，但参数不能直接逐位置照搬。TF decay 和 PyTorch momentum 的定义相反，不能把两个数直接比较。上述是具体层/算子实现和设置差异，而非骨干宽度被重新设计。[作者 tf_util](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/utils/tf_util.py)、[作者 train](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/train.py)。

训练/评估还需另列：官方默认 Adam .003/阶梯降LR，与社区 SGD/cosine 不同；官方测试会按对象类别屏蔽非法part，社区全16类主路径直接50-way argmax。我们主指标沿用社区unmasked，另记录masked secondary。当前数据也不是原Stanford HDF5逐位副本，而是既有raw/normal-bearing来源的XYZ适配；这影响与原论文成绩的严格可比性，但 D/E 与我们配对基线共享同一准备数据与口径。[官方测试](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/test.py)、[社区训练](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/main_partseg.py)、[本地数据协议](/D:/Hycore/docs/research/98_HYCORE_SEG_REPRODUCTION_2026-10-27/selection_protocol.md)。

## 2. E 的实际结构边界

```text
社区DGCNN：
  [原前端/变换网/动态图/全局类别广播融合]
  → Conv8(1280→256) → BN8 → LeakyReLU → dp1(.5)
  → Conv9(256→256) → BN9 → LeakyReLU
  → dp2(.5) → Conv10(256→128) → BN10 → LeakyReLU
  → Conv11(128→50) → CE

E：
  [相同前端]
  → Conv8(1280→256) → BN8 → LeakyReLU → dp1(.5)
  → Conv9(256→256) → BN9 → LeakyReLU
  → H[B,N,256] → EXP → M(256→128) → M(128→50) → CE
```

明确变更只有：删dp2；把Conv10/Conv11换为两个完整官方M；删BN10及其激活；在两M前加EXP。M内部没有补回BN、Dropout或LeakyReLU。原来的等维M1已不存在，且E没有启用正则梯度。[v5模型](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/model.py:27)、[继承前端](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/model.py:90)、[原分割模型](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/vendor/dgcnn_model.py:305)。

本次CPU构造核对：141个共同state_dict项逐项一致，包含保留权重及BN缓存；基线1,460,361参数，E1,460,283，净少78，差异可由被删除的尾部BN参数和新M参数化对账。既有合同还核验保留前端至Conv9在train/eval下的输出一致，D/E初值与dropout随机流一致；本次不重新进行前向训练。[已有合同](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/test_contracts.py:87)、[两端CPU验收](/D:/Hycore/docs/research/113_PSEG3_DE_REG_CONTROL_2026-10-29/cpu_acceptance.json)。

但必须限定保证范围：

- **相对我们已跑DGCNN：**架构改动限于尾部；优化器由SGD换为RSGD，不能把整体成绩差值全部解释为换头效果。两边都是plain CE、相同数据适配、batch32/shuffle/drop_last、点与标签同步置换及相同train/eval评估入口。[基线优化器及CE](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/train.py:180)、[E优化器](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:191)、[E CE](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:102)。
- **基线/E完整随机训练轨迹不保证相同：**新M构造额外消耗Torch随机数；保留前端参数完全一致，不代表模型构造后的全局RNG仍等于原基线。原基线与E的dropout随机流并非被严格配对；D/E彼此已严格配对。
- **D/E彼此：**架构、初始化、数据和随机流配对；D两正则各.01，E训练图直接CE。E对detached H做no_grad观察只用于保留whole/part模长统计，既不重新跑模型，也不额外更新BN。[E损失包装](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/losses.py:38)、[实际训练图](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:310)。

## 3. BN 和其他非结构设置

| 设置 | 官方 HyCoRe 分类的实际行为 | 当前 D/E / 含义 |
|---|---|---|
| BN模式 | part前向、whole前向均train；测试eval+no_grad | 分割一次train编码，再特征聚合；测试同样eval+no_grad |
| 每批BN更新次数 | 两次编码，更新两次 | 一次编码，更新一次；不是遗漏分支模式 |
| 分类CE | utils.cal_loss默认标签平滑ε=.2；parser False没有传入 | plain CE；与我们分割基线和固定社区分割默认一致 |
| 梯度裁剪 | 总loss.backward后全网clip1 | 只观察梯度，不裁剪；用户明确批准的偏离 |
| 优化器/学习率 | RSGD、momentum.9、WD1e−4；launcher .1、300轮，cosine下限LR/100 | RSGD同规则，.1→.001；200轮是分割配方 |
| M内部 | 完整weight/bias/transport与原初始化，没有BN/激活/Dropout | 已复用；无隐藏train/eval模式 |

证据：[官方分类训练](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/main_dgcnn_hycore.py)、[实际分类CE](/D:/Hycore/classification_ModelNet40/utils/misc.py:218)、[官方层实现](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/models/ops/manifold_layers.py)、[v5模式及BN断言](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:285)。普通BN的train/eval差异是框架机制，不是HyCoRe专属贡献；eval不等于no_grad，no_grad也不会自动切换BN或Dropout。

分类两次编码改变BN的统计更新权重。以momentum=.1的running mean为例，两次后为`.81旧值 + .09 part均值 + .10 whole均值`；分割单次为`.9旧值 + .1完整输入均值`。这解释为何统计行为不逐位相同，不构成要求分割补一个虚假前向的理由。分割论文的whole/part在同一H上聚合，支路自身没有BN。[论文§3](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。

RSGD也应按实际参数类型理解：当前默认M的`_weight/_bias`是普通Parameter，曲率固定，并非把所有骨干权重直接放进Poincaré球优化。Geoopt0.5.1即便对欧氏参数，首次momentum初始化/更新也与PyTorch SGD不同，所以两者不宜简单写成等价。[Geoopt RSGD源码](https://github.com/geoopt/geoopt/blob/v0.5.1/geoopt/optim/rsgd.py#L79-L103)。

## 4. 借鉴优先级及复现边界

1. 先用已有权重诊断train/eval特征尺度与BN统计差异；如果做train模式观测，需要隔离Dropout随机性并保存恢复BN缓存，不能污染正式模型，也不能把测试改成train来报告主指标。本次仅提出，不实施。
2. 标签平滑是一个明确的非结构单因素候选，改变CE目标和梯度；分类代码实际启用，不等于分割论文已指定ε=.2，也不保证能修复弱类别问题。D/E都用plain CE，因此它不解释D−E的差异。
3. 官方分类clip1可列为独立候选，但此前去clip是用户选择，并非漏实现。RSGD、LR比例、完整M初始化/参数化与数值球投影已经存在，不应重复当作待补模块。
4. 不把仓库里存在但官方DGCNN没有调用的ToPoincare/grad_fix当作正在生效的防饱和设计；也不自动增加M层间BN或激活来声称更忠实。[实际DGCNN前向](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:177)。

论文明确支持逐点EXP、一个或多个M、H的欧氏平均后EXP、特征邻域part、反向层次及两正则；但没有提供可核验的全部分割头宽度、截断位置、BN/Dropout、完整训练日程与初始化细节。因此D是**遵循论文主线并借用分类官方组件的可审计实现**，E是它的无正则对照；二者不能被保证为论文作者的逐项精确实现。论文“参数量不变”与我们净少78参数也不是字面完全相同。[论文§3.1/§4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。

完整专项证据：[官方/社区](official_community.md)、[E边界](e_tail_audit.md)、[HyCoRe设置](hycore_settings.md)。本次未启动新实验。
