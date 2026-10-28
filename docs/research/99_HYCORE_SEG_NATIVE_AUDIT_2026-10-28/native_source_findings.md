# 官方 HyCoRe DGCNN 分类源码：迁移到 HyCoRe-seg 的复用审查

> 修订提示：下文保留最初源码审查事实及历史迁移候选。用户三批注后，现行主方案已改为省最后global池化、linear1改1024→512逐点投影，无one-hot/新增融合桥，保留官方flip并记录配对类别；旧融合桥和替换负例建议不再是主方案。以[当前审查报告](audit_report.md)和[复现计划](reproduction_plan.md)为准。

审查日期：2026-10-28。范围是源码静态审查与迁移设计；未修改模型、未连接训练服务器、未使用 GPU、未启动训练。

## 1. 来源身份与结论

用户要求以官方 HyCoRe 的 DGCNN 分类模型为起点改造分割模型。因此模型谱系应为“官方 HyCoRe 分类模型 → 逐点化与分割任务适配 → 论文分割正则”，而不是“AnTao DGCNN 分割成品 → 简化双曲末层”。复用指实现、参数化、计算路径和可检验语义；不等同于加载 ModelNet40 分类预训练权重。

本地核验：官方提交 `ef56a73020e309b1009c486fea1f601d3f7b3252` 在本地对象库存在；以该提交与当前工作区比对，下列七个文件的 `git diff` 为空：`models/dgcnn_hmodel.py`、`models/ops/manifold_layers.py`、`models/manifolds/poincareball.py`、`hutil.py`、`main_dgcnn_hycore.py`、`data.py`、`utils/misc.py`。因而下面分析针对已核验的官方实现，不是本地实验分支改写。根审查同时核验这些核心实现与首发提交 `4bceb571e978bb676a706432e8cd498c328e1b59` 相同。下文固定官方首发提交链接，避免 `main` 随时间变化。

可直接保留的核心包括：四层动态图 EdgeConv、两种全局池化、欧氏到双曲的指数映射、官方 `MobiusLayer` 的完整权重参数化、双曲距离/原点距离、双曲嵌入到 logits 的两层路径，以及 RSGD/余弦调度/梯度裁剪的训练机制。必须改造的部分包括：全局预测到逐点预测、40 类到 50 部件、特征空间 part 选择、单次编码后欧氏池化、分割径向方向、跨类别负样本、分割 CE 与 mIoU、数据与保存协议。分类源码中的未使用候选层不应为了“全部利用”而一并接入。

## 2. 原分类模型的真实计算路径

[官方模型](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/models/dgcnn_hmodel.py)，本地 `classification_ModelNet40/models/dgcnn_hmodel.py:91–190`：

```text
XYZ [B,3,N]
  EdgeConv1: graph 6 → 64，邻域 max → x1 [B,64,N]
  EdgeConv2: graph 128 → 64，邻域 max → x2 [B,64,N]
  EdgeConv3: graph 128 → 128，邻域 max → x3 [B,128,N]
  EdgeConv4: graph 256 → 256，邻域 max → x4 [B,256,N]
  concat(x1,x2,x3,x4) [B,512,N]
  conv5: 512 → emb_dims(默认1024)，BN + LeakyReLU(.2)
  全局 max 1024 + 全局 avg 1024 → [B,2048]
  linear1: 2048 → 512，BN6 + LeakyReLU(.2)
  expmap0 → 双曲512维
  emb: 官方 MobiusLayer 512 → 256 → mu
  mlr: 官方 MobiusLayer 256 → 40 → logits
```

其 `dp1/dp2` 与其他头部候选都处于注释中；运行路径没有 dropout，也没有在两个 Möbius 层之间额外插入激活或 logmap。`logits = self.mlr(mu)` 被直接送入损失；“双曲坐标作为 logits”有官方代码依据，不能描述为没有前作参考的猜测。

局部特征 `x1..x4` 在全局池化前本来就保留点轴，因此可以从官方 encoder 暴露它们，新增 dense 解码接口；不需要抛弃 encoder 并另取一个成品分割网络。

## 3. 部件复用清单

| 部件 | 官方实际行为与证据 | 迁移处理 |
|---|---|---|
| `knn` / `get_graph_feature` | 模型本地24–55行；先计算欧氏特征距离，再构造 `(邻点−中心, 中心)` | 复用数值逻辑；把第39行硬编码 `cuda` 改成输入设备，保留索引、含中心与并列规则 |
| 四层 EdgeConv、BN1–5、LeakyReLU | 模型100–120、147–165行 | 原样保留通道数及运算；`k` 的分类默认20与既有分割40须在配置中明示；不要把分割版三组64通道误称同一 encoder |
| 全局 max+avg | 模型166–168行 | 保留两种上下文，再广播回每点；全局向量不能单独承担逐点预测，否则每个点会收到相同预测 |
| `linear1`、BN6 | 模型121–122、170行 | 推荐保留原global路径与输入维数2048→512，后接新增dense桥；备选A才将linear1改为逐点并改变输入维数，二者不可混写 |
| `PoincareBall` | [官方几何类](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/models/manifolds/poincareball.py)，本地8–64行 | 直接复用 `c=1` 约定与 Geoopt 运算；按每条特征的最后轴使用。`dim` 是附加元信息，`zero/proju0` 等辅助函数会使用它，512/256/50维不能混用 |
| `expmap0`、`dist`、`dist0` | 几何类继承 Geoopt；模型177行、hutil53–57行实际调用 | 复用原运算，区分双曲原点距离 `dist0` 与欧氏模长。不得用日志中的欧氏 `parent_norm` 冒充径向损失 |
| `emb` / `mlr` | 模型123–124、182–186行 | 保留两个官方 `MobiusLayer` 实例及初始化；逐点输入可展平为 `[B*N,F]`；末层输出宽改成50 |
| `hype_triplet_losses` 两个 hinge、mean reduction | [官方 hutil](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/hutil.py)，本地42–71行 | 抽取/适配其距离与 hinge 核心；用论文的 whole/part 嵌入替换分类网络两次前向的 `mu`；径向开启分割方向；负样本索引必须新接口 |
| `1000/k` margin | hutil27–40行 | 可复用公式与整数采样概念；用户已确认分割用1000/N′，不直接复用有副作用的采样函数 |
| 分类 `cal_loss` | [官方损失/保存工具](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/utils/misc.py)，本地218–235行 | 复用实现并把 `[B,N,50]`、`[B,N]` 同序展平；显式传入 smoothing。不要把未生效CLI默认当实际训练行为 |
| RSGD、余弦调度、梯度裁剪 | [官方训练入口](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/main_dgcnn_hycore.py)，本地144–148、230–233行 | 可复用机制与实现；native路线建议来源优先 LR=.01、eta=.0001、momentum=.9、WD=1e−4、clip=1；分割论文未完整披露这些值，不能说已证实作者分割配方 |
| seed/日志/best-last | 训练58–67、117–136、165–190行；misc167–178行 | 复用日志与状态保存思想；seed按用户改为4780，mIoU替换OA，记录完整RNG/scheduler/step/数据身份；不沿用有缺项的自动resume行为 |
| 数据增强的缩放和平移 | [官方数据](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/data.py)，本地49–54、68–75行 | 可复用坐标增强函数；分割 shuffle/采样必须同时重排逐点部件标签，分类原只shuffle点坐标的接口不能直接搬入 |
| `GeodesicLayer`、`MobiusMLR`、`ToPoincare`、`GyroplaneConvLayer` | 前两类替代头在模型注释中；`ToPoincare` 未在主路径实例化；Gyroplane用于其他网格形式 | 不进入首轮主模型；它们的存在不等于官方DGCNN使用过。特别 `ToPoincare` 依赖 `pmath`，当前文件没有对应导入，不能为了增加复用率启用未走过路径 |

## 4. 必须保留的官方 Möbius 权重参数化

[官方 `RiemannianLayer` / `MobiusLayer`](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/models/ops/manifold_layers.py)，本地55–116行：

```text
_weight: [out,in] 普通 Parameter
_bias:   [out,1] 普通 Parameter（默认 over_param=False）
bias = expmap0(_weight * _bias)
weight = transp0(bias, _weight)
output = mobius_matvec(weight, input)
```

因此 `_bias` 不作为传统加性偏置加到输出，但参与有效权重计算、训练与初始化，不能删除后还称为完整复用。默认初始化是 `_weight` 用 Kaiming normal、`_bias` 用 `[-4/sqrt(in),4/sqrt(in)]` 均匀分布。官方没有最终 `logmap0` 或 logit 温度。当前v1直接使用上游普通卷积权重的 `mobius_matvec(W,x)`，少了这一参数化，且只有一层128→50；这属于实质性头部差异。

以512→256→50为候选时，双曲两层可训练参数共 `(512*256+256)+(256*50+50)=144178`。对照欧氏无偏置两层只有143872个参数，相差306个 `_bias` 参数；使用普通有偏置的同宽两层Linear则恰好同为144178，可作为后续参数数量匹配控制，但两种偏置的运算语义不同。论文“参数数量不变”的表述不能自动解决整体结构差异；报告应列出精确计数，不为对齐总数而删除官方参数。首轮H0/HC本身可以做到同结构、同参数数量。

建议验收：同一输入与同一 state_dict，逐点包装输出/反向应与官方展平调用一致；检查 `_weight`、`_bias` 都收到有限非零梯度，保存重载后输出一致。球坐标有界在官方也存在，不能单凭CE数值偏高判定移植失败；须联合检查分割准确率、logit范围、梯度及投影。

## 5. 分类正则哪些保留、哪些必须调整

官方 `hype_triplet_losses` 的距离项为 `relu(d(parent,pos)-d(parent,neg)+delta)`，默认径向为 `relu(-dist0(parent)+dist0(pos)+gamma)`；该默认要求分类 whole 更深。源码已有 `opposite_hier=True` 分支，会使用 `relu(dist0(parent)-dist0(pos)+gamma)`，可以作为分割方向的现成参考。不能只因函数名相同直接默认调用。[本地 hutil42–71行](../../../classification_ModelNet40/hutil.py)

但以下数据流不能照搬：

1. **负样本**：当前函数只有 `flip(batch)`，没有对象ID/类别输入；会产生同类负样本，奇数batch还可能选到自身。分割论文要求另一对象且异类，需显式的合法负样本映射、无候选策略和有效分母。
2. **part采样**：`get_children_np` 用XYZ（`data[:,:3,:]`）计算邻域，并令 `pos_child=data[:,:,:k]` 后原地覆盖。它会修改调用方data；分割还存在逐点标签错位风险。必须改成在已编码的逐点特征中选邻域，用gather生成新张量，不原地重写输入。
3. **编码次数与池化位置**：分类训练先裁whole800–1024、再裁part200–600，各做一次网络前向，并在post-emb256维 `mu` 上施损失。分割应编码一次；从同一逐点欧氏H选part，分别欧氏平均后exp，再计算损失。不能多跑part encoder或先exp再平均。
4. **梯度**：离散kNN索引可不求导，但选中特征值必须回传；不能在正则支路整体detach。CE/contrastive/hierarchy到共享encoder的独立梯度与梯度夹角应在短测读出。

前两项是正确性/任务语义修正，第三项是论文明确的任务适配，径向翻转也是论文要求；不得与“正则有无收益”混为一个因果解释。现有分类实现保留原样，新路径中完成适配。

## 6. 训练与保存的隐含行为

- **平滑CE实际开启**：训练入口导入的是`utils.cal_loss`，只调用`criterion(logits,label)`；函数默认`smoothing=True`、eps=.2。CLI的`--smoothing`默认False并没有传入该调用。因此native原分类实际行为不能根据CLI帮助判断。分割候选如沿用当前plain CE，应明确调用`smoothing=False`，记录为任务/协议选择；“classic CE”本身不足以证明作者没有label smoothing。
- **LR与clip**：官方分类默认LR=.01、min=.0001，固定clip1；当前v1的LR=.1并不等于官方HyCoRe训练默认。来源优先的native方案可保留原值，但这是重建选择，不是分割论文事实。
- **RSGD**：默认官方头的`_weight/_bias`是普通Parameter；“用了RSGD”并不意味着所有参数都是ManifoldParameter。不要擅自把它们替换为球上参数；后者改变优化语义。Geoopt版本及普通参数的动量更新行为仍须锁定。
- **选模**：分类每轮读test，按三位小数OA用严格`>`选best。用户已明确分割用trainval训练、逐轮test Instance mIoU、并列取较晚轮、固定200轮不early-stop，应改成该协议，不能继续沿用OA或含糊称val。
- **resume**：分类入口自动从best载入`epoch`与optimizer，scheduler没有单独保存完整state，RNG也未保存；`start_epoch=checkpoint['epoch']`还需核查轮次定义。新分割入口应显式resume last，保存恢复所有训练状态，并用连续运行/断点恢复下一步一致性验收；不为“复用”保留缺陷。
- **设备/依赖**：训练入口第11行固定可见GPU，graph第39行固定cuda；绝对导入`models`、`utils`会与其他实验路径冲突。只在版本化适配路径修正为显式设备与包内导入，原分类目录不改；未用库不连带引入。

证据：[官方训练入口](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/main_dgcnn_hycore.py)、[损失与保存工具](https://github.com/diegovalsesia/HyCoRe/blob/4bceb571e978bb676a706432e8cd498c328e1b59/classification_ModelNet40/utils/misc.py)。

## 7. 对 native dense 两个候选的独立评估

根审查最初候选A：原四层EdgeConv和conv5保持；得到局部512维、global max+avg2048维；将global广播，与局部和物体类别16维one-hot拼接成2576维；逐点共享`linear1(2576,512)+BN6+LeakyReLU`；再保持`exp0→官方emb512→256→官方mlr256→50`。其维度自洽，但修改原linear1输入形状、把原BN6的统计对象从物体变成点，复用边界不够清楚。

更新候选B，也是本审查推荐方案：完整保留原`global max+avg2048→linear1(2048,512)+BN6+LeakyReLU`，得到原结构的global context512；将它广播，与局部cat512和物体类别one-hot16拼成1040维；只新增逐点融合`1040→512+新BN+LeakyReLU`生成H，再保持`exp0→官方emb512→256→官方mlr256→50`。

候选B没有维度硬伤，且原linear1/BN6仍在全局路径，其结构和训练统计含义都更接近原分类实现。新增dense桥本身就提供点间差异，避免广播同一物体全局向量后所有点同预测。代价是明确新增了一个融合模块：无偏置1040→512映射有532480个权重，BN512有1024个可训练标量，共533504；再加40→50读出多出的2570个标量，全模型相对原40类分类模型增加536074个可训练标量。实现后仍需实数核验，不以推导替代模型计数。两候选都含任务所需的类别条件、广播和逐点预测适配；都不能宣称恢复了作者未公开的分割架构。主方案从头初始化，不因某些层尺寸保持就自动加载旧分类权重。

正则取512维pre-exp逐点H、先欧氏池化再exp，可与论文定义一致；它不经过`emb`，因此正则会影响共享encoder和dense融合，但不会直接对`emb/mlr`参数施加梯度。这个分支与分类post-emb256维正则不同，应作为必要位置适配明确记录并验收，而不是悄悄用分类mu替换。

需在正式实施前写明的风险：

1. **论文结构与native保留程度的取舍**：作者的DGCNN分割细节没有公开；保留分类四层并新增dense融合是本项目的来源优先候选，不是严格原作者结构事实。普通AnTao baseline只是外部benchmark；因骨干也不同，不能把它与native HC的差异全归因于正则。
2. **显存和耗时**：N2048、k40、batch32下，原四层中128/256通道的邻域张量可能比当前v1大得多；不能从v1显存或97秒/轮外推native一定可跑。完整尺寸单步及2×2短测须先验证。若需改batch/k/AMP/重计算，单列协议变更，不默默降规格。
3. **特征维数**：512维pre-exp可能更容易达到数值投影区；先记录欧氏模长、映射半径、投影率与可用径向差，不预先增加cap/hook/温度或缩放来迎合成绩。
4. **对照匹配**：至少native H0与native HC严格同encoder、同双曲头、同初始化、同CE/优化器/数据流，只切换两项正则。若再加native欧氏控制，也需同dense encoder及训练协议；当前v1 H0和AnTao DGCNN均不能直接充当这个匹配控制。

## 8. 实施前验收建议

顺序应为：固定官方源码与逐文件映射 → 原分类计算回归 → 逐点包装与头部参数/梯度回归 → 分割shape/标签同步/负样本/径向方向 → 精确的优化器与checkpoint恢复 → 完整尺寸短测 → 用户审阅后正式训练。

必要检查包括：官方层与适配层展平调用的FP64前向/反向一致；参数数目与key逐项列出；logits/标签展平顺序；只计算一次encoder；无原地输入改写；正part来自同对象、负part来自另一对象异类；1000/N′逐样本或逐batch约定明确；HC正则对共享层非零梯度、H0零调用；CE与两正则分别记录；保存恢复下一步一致；GPU full-size测量记录真实峰值与BN次数。验收成功只说明实现可执行且符合已声明候选，不直接证明论文数值已复现。
