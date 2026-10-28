# 当前 HyCoRe-seg v1：来源偏离与可保留工程审查

日期：2026-10-28。范围：本地源码静态审查；没有修改模型、运行测试、连接服务器或干预已有训练。本文为主审查报告的专项证据。用户最新要求是从官方 HyCoRe 的 DGCNN 分类模型出发，复用已有组件并按分割论文进行必要改造。

## 1. 审查结论

当前 v1 的实际代码来源是 AnTao 的 `DGCNN_partseg`，仅把末层替换为直接使用 Geoopt 运算的单层双曲头。这能作为已有工程对照，但不满足用户这次重申的“从官方 HyCoRe 分类模型适配”的代码来源要求。偏离涉及编码器拓扑、双曲头层数、有效权重参数化和训练协议，不能仅把类名改为 HyCoRe-seg 来修正。

同时，当前分割正则的单次编码、特征邻域、欧氏平均后指数映射，以及 whole 浅 / part 深的方向，有明确分割任务理由，不应为追求复用而恢复成分类版的原始点裁剪、两次编码和相反径向方向。直接复用的目标是组件和可核验语义，需适配的任务行为仍要按分割论文调整。

## 2. 逐项偏离

| 项目 | 当前 v1 的实际行为 | 官方分类代码参考 | 判定与下一步 |
|---|---|---|---|
| 模型入口 | `model.py:8,32–44` 导入并继承 vendor `DGCNN_partseg` | `classification_ModelNet40/models/dgcnn_hmodel.py:91–124` 定义官方 `DGCNN` | 用户要求的来源不符。v2 应从官方类提取组件和前向阶段，明确每处结构变更。 |
| 编码器 | STN；三阶段局部输出均为 64 维；拼接 192→1024；加入 16→64 类别分支；逐点 256→256→128 | 官方为四阶段 EdgeConv，64/64/128/256，拼接 512→1024，再全局 max+avg、2048→512 | 两者共享 DGCNN 思想，但不是同一网络实例。分类全局输出必须适配为逐点输出；不能默认为 vendor 三阶段与官方四阶段等价。 |
| 双曲头 | `MobiusPointHead` 仅一层 128→50，`expmap0` 后直接 `mobius_matvec(W,x)`；W 沿用原分割末层初始化 | 官方 `expmap0`→`emb` 512→256→`mlr` 256→40，两个 `MobiusLayer` | 当前没有移植官方 `emb/mlr`。v2 先复现官方层，再做逐点共享适配；具体宽度应列为工程选择与参数预算。 |
| 有效权重 | `model.py:23–28` 直接保存普通卷积权重并使用 | `models/ops/manifold_layers.py:62–86,110–116` 包含 `_weight`、`_bias`；有效 W 由 `transp0(expmap0(_weight*_bias),_weight)` 得到 | 这不是省略普通加法偏置那么简单。缺少 `_bias` 会改变有效权重与梯度；参数量相同不等于官方等价。 |
| logits | 双曲输出球坐标直接交给 CE | 官方 `dgcnn_hmodel.py:177–190` 也直接返回 `self.mlr(mu)`，主训练入口直接计算分类损失 | 该输出方式已有官方参考，不应笼统标为“没有源码依据”。球坐标有界也不是 v1 独有差异。 |
| 正则特征位置 | `forward_features` 返回 `conv10` 后的 128 维欧氏逐点特征；whole/part 在这里平均并 exp | 分类版正则用经过官方 `emb` 的 256 维 `mu`；整物体和裁剪输入分别编码 | 位置变化属于分割论文适配。v2 需明确 `H=E(X)` 截止位置，不应直接平均球坐标，也不应默认沿用分类 `mu`。 |
| 几何实现 | model/loss 各自新建 `geoopt.PoincareBall(c=1)` | 官方自定义 `models.manifolds.PoincareBall` 继承 Geoopt，并供官方层和 `hutil` 使用 | 核心 exp/dist 可能共享父类实现，但 v1 没有执行原类的前向/梯度等价回归。v2 应优先复用官方类并补维度/设备适配。 |
| 关系损失 | `losses.py:41–60` 重新写径向/对比 hinge；异类别负样本；有效负样本单独归约 | `hutil.py:42–71` 为相同几何 hinge 原型，已有 `opposite_hier=True` 分支；负样本为 batch flip | v1 的异类筛选和方向是必要适配；可从官方损失核构建带显式负样本输入的版本。不能原封不动复用 flip，以免自身或同类负样本。 |
| CE | `train.py:122,272` 始终普通 `F.cross_entropy` | `main_dgcnn_hycore.py:23,101,228` 使用 `utils.cal_loss`；`utils/misc.py:218–235` 默认 smoothing=True、eps=.2 | 协议差异需显式列出。原入口的 CLI `--smoothing` 默认 False 并未传给 cal_loss，不能凭 CLI 判断实际不平滑。v2 建议复用函数并明确传参，选定分割策略后 H0/HC 保持一致。 |
| 梯度裁剪 | v1 记录梯度范数和有限性，随后直接 optimizer.step | 官方 `main_dgcnn_hycore.py:230–233` 每步 `clip_grad_norm_(...,1)` | v1 没有复用。v2 应把原裁剪行为显式配置、记录，并在最初采用官方策略时核验裁剪前后范数。 |
| 优化器/调度 | 双曲组 RiemannianSGD，baseline SGD；LR .1→.001，cosine200 | 官方 RSGD/.9/WD1e-4；入口默认 LR .01，cosine 至 LR/100 | 优化器类型可复用；LR 和普通 SGD 对照属于分割协议选择。不能把 .1 描述成已核实的官方 HyCoRe 分类默认。 |

来源：[当前模型](../../../segmentation_ShapeNetPart/hycore_seg_v1/model.py)、[当前损失](../../../segmentation_ShapeNetPart/hycore_seg_v1/losses.py)、[当前训练](../../../segmentation_ShapeNetPart/hycore_seg_v1/train.py)、[vendor 分割模型](../../../segmentation_ShapeNetPart/hycore_seg_v1/vendor/dgcnn_model.py)、[官方来源分类模型的本地副本](../../../classification_ModelNet40/models/dgcnn_hmodel.py)、[官方层本地副本](../../../classification_ModelNet40/models/ops/manifold_layers.py)、[分类损失本地副本](../../../classification_ModelNet40/hutil.py)、[分类训练本地副本](../../../classification_ModelNet40/main_dgcnn_hycore.py)。官方上游对应文件见 [HyCoRe](https://github.com/diegovalsesia/HyCoRe/tree/main/classification_ModelNet40)。本专项不单独认定本地副本与上游固定提交逐字一致，来源锁定由主审查完成。

## 3. 可保留的 v1 工程

| 可保留部分 | 证据 | 保留条件 |
|---|---|---|
| ShapeNetPart HDF5 读入、类别和点标签同步置换、对象 ID | `data.py:10–55` | 固定现有数据 SHA；每种协议分别标明成员和预处理，避免重下载引入新变量。 |
| Instance/Class mIoU 和逐类别计数 | `metrics.py:5–51`；`train.py:110–143` | 主指标维持用户选定 unmasked Instance mIoU，masked 另列；空并集=1；最终16类必须齐全。 |
| 用户确定的选模 | `train.py:160–173,313–346` | train+val / 每轮 test Instance mIoU 取 `>=` 保存、并列取后轮；保留该用户决策，不把分类 OA 直接搬入分割。 |
| 实验来源、配置、输入 SHA、初始化 SHA、完整状态与随机数保存 | `train.py:194–245`；`utils.py:15–85` | v2 更新模型来源哈希和公共编码器参数筛选，不能继续写死 `conv11` 排除规则。 |
| GPU 空闲检查、独立输出目录、原子保存、失败状态 | `train.py:76–106,370–380` | 新版本新目录；旧结果不重写；现有任务不更改。 |
| 分割正则的离散选择/连续特征梯度、独立part RNG | `losses.py:8–38,63–86` | 可作为分割 adapter 保留；数值核改为官方几何并做等价/梯度回归；采样分布仍须标为工程约定。 |
| 正则到共享层的梯度诊断、BN 每批一次更新检查 | `train.py:273–279,303–304` | v2 的共享层不能写死 `model.conv10[0].weight`；用明确 `H` 和编码器接口定位。 |

数据来源仍有既知局限：公开 normal-bearing raw release 转为 XYZ、全点居中/单位最大半径/first2048，不保证与原 Stanford HDF5 相同。`prepare_antao_protocol.py:28–56` 还原原 trainval/test 成员并记录重复与跨 split 共享 ID；不得将它重新称为已去重协议。保留该缓存能减少新旧比较的数据变量，但不能消除上游打包差异。[准备代码](../../../segmentation_ShapeNetPart/hycore_seg_v1/prepare_data.py)、[协议准备代码](../../../segmentation_ShapeNetPart/hycore_seg_v1/prepare_antao_protocol.py)。

README 的“仅val选模/未自动联训”描述已落后于新增 `antao_testsel` 入口；v2 文档应以解析后的配置及 manifest 为准，历史 README 不应被用来推断现有运行行为。

## 4. 16 项合同证明了什么

`test_contracts.py` 共16项。前两项把 v1 欧氏模型与 vendor AnTao 分割模型逐张量、前向、梯度和 BN 对齐；这证明 wrapper 保留 vendor 行为。第三项证明当前单层双曲改造保持同名初始化与参数数量。后续覆盖形状/有限梯度、Geoopt单位球解析几何、采样、负样本、margin、disabled 零采样、指标、点标签置换、优化器差异和简化网络的完整状态恢复。

这些测试有工程价值，但没有把官方 `MobiusLayer` 作为 reference；没有验证官方 `_bias` 和平行移动的梯度；没有验证官方 DGCNN 编码器逐层输出；没有验证官方 CE 平滑和梯度裁剪。`test_hyperbolic_head_shape_and_finite_gradients` 用 `logits.square().mean()` 反传，仅能证明该损失下有限，不能单独验证真实分割 CE 或作者实现等价。完整状态恢复合同使用的是小型欧氏网络，并不代替新双曲层及其优化器状态的实际恢复测试。

因此不能把“16项通过”写成“官方 HyCoRe 头已复现”。应保留这些合同，新增对官方组件的独立数值回归。[测试来源](../../../segmentation_ShapeNetPart/hycore_seg_v1/test_contracts.py#L29)

## 5. 建议的 v2 模块映射与验收

建议路径 `segmentation_ShapeNetPart/hycore_seg_v2/`，不覆盖 v1。下列为设计接口，尚未编码。

| 新模块 | 来源及改动边界 | 必要验收 |
|---|---|---|
| `native_source/` 或有固定来源的官方组件导入 | 锁定官方 `dgcnn_hmodel`、`manifold_layers`、`PoincareBall`、loss/CE 相关函数；保留许可及修改清单 | 按固定上游 SHA 和本地 SHA 建清单；新增改动仅设备/命名空间等兼容补丁时，证明不改变数值。 |
| `encoder.py` | 从官方 DGCNN 拆出 EdgeConv 和中间特征；将原全局分类出口改为可供逐点融合的特征接口 | 先在原分类输入/模式下与锁定官方实现逐层对齐，再检查逐点输出形状、点置换等变和 BN。 |
| `seg_adapter.py` | 保留局部特征并融合广播全局特征；类别条件和宽度列为明确新增项 | 每一层说明必要性和参数预算；对照模型使用同一 adapter，避免把编码器变化算为正则收益。 |
| `head.py` | 官方 `MobiusLayer` 包成逐点共享形式，保留 `_weight`、`_bias`、初始化及有效 W；输出50类 | 把 `[B,N,F]` 展平为 `[B*N,F]` 的结果/梯度，与逐点调用官方 reference 比较；检查 `_bias` 梯度非零、所有中间点在球内。 |
| `losses.py` | 复用官方几何 hinge 核，显式 `opposite_hier`/分割方向；新增特征part和异类负样本适配 | 固定几何点复算、正负距离、1000/N′和delta4；独立叶子方向；有效负样本分母；所选特征回传；单次编码计数。 |
| `train.py`、`data.py`、`metrics.py`、`utils.py` | 保留 v1 数据/指标/完整状态框架，替换模型工厂和共享层诊断；CE/clip/optimizer显式来源 | H0/HC同初始化/输入/RNG流；CE与正则共享梯度；裁剪前后范数；真实双曲模型下一步恢复；轮次、样本数、评估次数和best身份。 |

先通过组件回归和真实尺寸短测，再请求审阅修订后的正式实验计划。已有 v1 H0/HC 成绩保留为“AnTao分割骨干＋简化双曲末层”的历史工程结果，不冒充此次要求的原生HyCoRe迁移结果；现有任务无需因本次静态审查而停止或覆盖。
