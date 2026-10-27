# HyCoRe-seg 源码核查与论文复现计划

日期：2026-10-27（Asia/Shanghai）。状态：**调研与计划已完成，等待审阅；尚未实现、连接服务器或启动训练。**

后续用户决定：先独立复现原生DGCNN分割，验收后再进行DGCNN + HyCoRe-seg；层次margin明确采用 **1000/N′**，对比margin沿用4。下文执行顺序与主候选已据此更新，1024/N′仅保留为原文证据，不再作为当前主候选。本轮仍为计划讨论，没有启动训练。

## 1. 结论与范围

截至本次检索，未找到可以核验为论文 **Towards Hyperbolic Regularizers For Point Cloud Part Segmentation** 的公开实现。作者官方 [diegovalsesia/HyCoRe](https://github.com/diegovalsesia/HyCoRe) 对应 NeurIPS 2022 的分类工作；公开主分支、作者 fork、其他公开 forks、标签和 issues 未发现分割入口。这是本次检索的结论，不是“作者从未发布过代码”的证明。范围、提交身份和访问限制见 [源码检索记录](code_search.md)。

目标是复现 ICASSP 2023 的 **DGCNN + HyCoRe-seg / ShapeNetPart**，先获得可审计的实现与匹配对照，再判断能否接近论文结果。分类 HyCoRe、现有 HIER/PartHier 和 HPCS 的结果不充当这个分割基线。论文 DOI：[10.1109/ICASSP49357.2023.10095253](https://doi.org/10.1109/ICASSP49357.2023.10095253)；[作者接收稿](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。

## 2. 已确认的论文规格

以下来自论文 §3–4、图1、表1–3；完整证据及未报告细节见 [论文规格](paper_spec.md)。页码以论文自身第1–5页为准，大学存档可能另有封面。

| 项目 | 论文信息 | 复现含义 |
|---|---|---|
| 任务与骨干 | ShapeNetPart；16种对象类别；实测采用 DGCNN | 第一阶段只做该组合，不扩展 ModelNet40、场景分割或其他骨干 |
| 特征 | 编码器输出每点欧氏特征，随后接共享的双曲分割头 | 保留点维，不能将分类头的40改成50就当分割 |
| part 定义 | 随机中心点在**特征空间**的近邻，点数随机200–600 | XYZ 邻域仅作为论文表3的后续消融 |
| whole / part 表示 | 同一次编码的欧氏特征分别平均，再作指数映射 | 不裁子云后二次编码，不在球内直接取算术平均 |
| 层次方向 | whole靠近球心、part靠近球边 | **与分类版相反** |
| 负 part | 来自另一对象，且对象类别不同 | 不能直接 batch flip 后默认负样本有效 |
| 损失权重 | α=β=0.01 | 两项分别乘0.01 |
| 优化 | Riemannian SGD；PyTorch、Geoopt | LR、epoch、batch等具体配方未完整披露 |

设逐点特征为 \(F_i=E(X_i)\in\mathbb R^{N\times d}\)，特征邻域为 \(S_i\)：

\[
z_i^w=\exp_0\left(\frac1N\sum_n F_{in}\right),\qquad
z_i^p=\exp_0\left(\frac1{|S_i|}\sum_{n\in S_i}F_{in}\right).
\]

\[
L=L_{CE}+.01R_{contr}+.01R_{hier},
\]
\[
R_{hier}=\left[d(0,z_i^w)-d(0,z_i^p)+\gamma(N')\right]_+,
\]
\[
R_{contr}=\left[d(z_i^w,z_i^p)-d(z_i^w,z_j^p)+\delta(N')\right]_+,
\qquad y_j\ne y_i.
\]

这里的 \(y\) 是对象类别，不是点级 part 标签；\(\|z\|_{\mathbb D}=d(0,z)\) 是双曲原点距离，不能替换成欧氏模长。论文式(1)及其后两式为上述依据。[来源：论文 §3.1](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)

实现中用正值 \(\kappa=1\) 表示实际曲率 \(K=-1\)，统一距离、指数映射和头部的符号约定。FP64性质检查使用标准 Poincaré 距离及 \(d(0,z)=2\operatorname{atanh}\|z\|\)；实际FP32保护与投影阈值单独记录。

## 3. 必须保留的歧义清单

### 3.1 margin 存在原文冲突

- **方法节 §3.1**：\(\gamma(N')=\delta(N')=1024/N'\)。
- **实验节 §4**：另写 \(\gamma=1000,\delta=4\)，没有说明如何与前面的函数关系对应。
- 作者博士论文的相关章节仍出现这组冲突，未找到可核验的更正或分割配置。[作者博士论文 §5.2](https://tesidottorato.depositolegale.it/bitstream/20.500.14242/157866/7/conv_phdPolito_AntonioMontanaro.pdf)

**用户确认后的处理：** 用户在本轮明确选择“1000/N′，按part点数缩放”。因此将配置系数 `gamma_base=1000` 与实际margin函数 `gamma(N′)=1000/N′` 分开命名，对比margin取4；不再将常数1000视为当前待选项。这是用户选定的重建配方，原文冲突仍保留用于说明复现边界。

| 配置名 | γ / δ | 身份与用途 |
|---|---|---|
| `user_1000_over_n_4` | 1000/N′、4 | **当前主配方，用户已确认**；不是已恢复的作者分割配置 |
| `method_1024_over_n` | 二者均1024/N′ | 原文方法节证据；不自动增加对照或搜索 |
| §4字面解释 | 1000、4 | 仅保留歧义记录，不实施 |

当前实际损失为：

\[
R_{hier}=\left[d(0,z_i^w)-d(0,z_i^p)+1000/N'\right]_+,\qquad
R_{contr}=\left[d(z_i^w,z_i^p)-d(z_i^w,z_j^p)+4\right]_+.
\]

N′为200/400/600时，层次margin分别为5/2.5/约1.667。后续仅对该主配方做解析与训练数据短测，不按test成绩在解释间选优；成果标为“按用户确认配方重建”，不声称精确恢复作者实现。

### 3.2 分割头与训练配方不完整

论文描述逐点指数映射、一个或多个 Möbius 层，实验节说以 Möbius 与 softmax 替换分割头、参数量不变，但没有公开准确截断层、宽度、bias、logit处理、初始化、训练轮数及完整优化配置。[来源：论文 §3.1、§4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)

建议首个可核查候选：保留 DGCNN 到末层前的128维逐点特征，将最后的无bias `128→50` 点卷积替换为同尺寸、无bias的 Möbius 映射；CE接收明确记录的逐点logits。先核对输出范围、梯度及参数量。如果该解释发生严重数值退化，停止主训练并提交新的头部规格，不悄悄增加温度、投影头或额外线性层。

本地 `MobiusLayer` 继承类还包含 `_bias` 参数，不能仅凭类名就声称与无bias卷积参数量相同；复用前须审查。分类模型最终输出的流形选择也不能直接当作分割论文的证据。

## 4. 源码底座与隔离方式

推荐工程骨架：[AnTao97/dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch)，它由 [DGCNN作者仓库](https://github.com/WangYueFt/dgcnn)列为其他实现，包含 PyTorch part-seg 训练、模型与数据读取；**它本身不是 HyCoRe-seg 源码**。本次公开引用的 `master` 提交为 `07d534c2702905010ec9991619f552d8cacae45b`；实施时再核验该提交并固定源码和许可证。

拟新增独立路径：`segmentation_ShapeNetPart/hycore_seg_v1/`，包含数据适配、DGCNN骨干、分割头、feature-part selector、分割版loss、metrics、训练/评估入口、配置与必要合同检查。保持分类、inter-hierarchy、HPCS原目录不变。

| 现有本地资源 | 可复用范围 | 需重新实现/核对的部分 |
|---|---|---|
| `classification_ModelNet40/models/manifolds/poincareball.py`、同目录`__init__.py`及`models/ops/manifold_layers.py` | 距离、exp/log、Möbius运算的参考 | 曲率、投影、bias、初始化与逐点维度 |
| `classification_ModelNet40/hutil.py` | 分类方法差异参照 | 不复用其原地part覆盖、XYZ裁剪、flip负样本、径向方向 |
| `资料库/HPCS/hpcs/nn/dgcnn/dgcnn_partseg.py` | 点级DGCNN结构参考 | 不搬HPCS损失、旋转设置、真实part辅助解码或评估 |

不引入 teacher、512 hierarchy proxies、等深4.6视图、B05 cap/GS、PartHier队列或跨对象HIER目标；这些没有分割论文依据。如果以后研究这些扩展，另行版本和对照。

## 5. 数据与评估协议

### 5.1 数据冻结

使用 ShapeNetPart 官方对象ID与 train/val/test 划分，首先验收50个part标签到16个对象类别的映射、重复ID和数据身份。输入候选为2048个XYZ点、16维对象类别条件、无normals；2048点来自 DGCNN 底座，是本计划选择，分割论文没有完整说明该设置。[DGCNN论文 §4.4](https://arxiv.org/pdf/1801.07829)

AnTao 的HDF5读取默认使用 trainval，且不保证保留官方val的对象ID映射；若数据包不能还原该映射，先取得带对象ID和官方split的版本，再完成适配。不能把随机拆分改称官方val，也不能把“16881对象”的论文总数直接当成实际数据包验收结果。

数据清单记录文件SHA、对象ID、split、对象类别、原始/输入点数、标签范围、采样与归一化规则。训练按普通shuffle32起步，不移植分类项目的类别均衡采样；part损失不改变数据加载顺序。逐点采样/打乱始终同步标签。原始数据若已归一化，不再未经核对重复归一化。训练增强以固定底座的真实代码为准，不自动添加旋转、抖动或缩放。

### 5.2 主评估口径

本计划首先固定 **AnTao全16类的unmasked50预测**，以便工程配对一致：网络接收对象类别one-hot，但主预测仍取50个part分数的argmax。源码并未默认将预测限制为本对象合法part。[训练与评估源码](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/main_partseg.py)

- 每个对象：对所属对象类别的合法part计算IoU，预测与真值均为空时该part记1，然后在合法part间平均。
- **Instance mIoU**：平均全部对象的上述IoU；这是论文表1的 Inst. mIoU 对应概念。
- **Class mIoU**：先按对象类别平均对象IoU，再对16类宏平均；另存每类结果。
- 如附加使用真实对象类别屏蔽非法part的 masked 指标，单独命名、单独列表，所有对照使用同一规则；不能混用两口径选最优结果。
- 验证/测试输入的采样、点数、归一化与随机流固定，测试前冻结；首阶段无投票或测试增强。

作者 DGCNN 的 TensorFlow 评估确有合法part限制，且原点数/补点等流程与 AnTao不同，因此**论文85.7不能被当作已确认与本工程口径严格同协议的门槛**。[官方测试源码](https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/part_seg/test.py)

### 5.3 选模与最终报告

开发阶段只用train训练、val选设置和checkpoint，主选模指标为val Instance mIoU；并列时用val CE，再取较早epoch。固定总轮数，无基于test的早停。官方test仅在配置和checkpoint冻结后作最终报告。

如需要 train+val 的论文基准比较，先在开发协议固定完整配方和轮数，再从头在train+val训练，按预先固定的终点checkpoint评估一次test。该结果与val-selected结果分表，不把“train+val＋每轮test选best”包装为验证选模。

## 6. 首个工程配方（均为待审计划）

| 参数 | 建议初值 | 来源/身份 |
|---|---|---|
| 骨干 | DGCNN part-seg；k40、emb1024、dropout0.5 | 固定PyTorch底座 |
| 输入/输出 | XYZ2048、对象条件16；逐点50类 | DGCNN选择，需数据验收 |
| batch | 训练32、评估16；普通shuffle | 底座起点；GPU容量以短测实测 |
| epoch | 200，完整遍历训练split | 底座默认；不是已知HyCoRe-seg轮数 |
| seed | 首轮4780 | 项目默认；所有匹配对照相同 |
| optimizer | SGD / RSGD，LR0.1、momentum0.9、WD1e-4 | LR/WD/调度来自底座；RSGD来自分割论文 |
| scheduler | cosine，最低LR0.001 | 底座起点；所有对照同预算 |
| CE | 普通CE，不默认ε=.2平滑 | 底座`cal_loss`默认不平滑 |
| HyCoRe-seg margin | Rhier用1000/N′，Rcontr用4 | 用户本轮明确选择；DGCNN基线不构建这两项 |
| 精度/曲率 | FP32、K=-1；几何检查FP64 | 数值与可复现性选择；记录投影保护 |
| part | 每对象每步1个随机中心；feature欧氏近邻；N′∈[200,600] | “每对象1个/整数均匀/批内共用N′”为明确的实现选择 |
| 负样本 | 在batch内均匀选另一对象类别的part | 无合格负样本时仅跳过该anchor的Rcontr并计数，不误用同类 |
| reduction | CE按点均值；Rhier按对象均值；Rcontr按有效anchor均值 | 论文未交代reduction，须在所有对照固定 |
| 额外缩放 | 无B05 cap、GS或固定壳 | 不将既有分类改动带入复现 |

特征kNN只计算“随机中心到本对象全部点”的距离，无须构造额外的N×N关系矩阵；离散选择不求导，gather后的特征仍反传到共享编码器。默认不作L2归一化，包含中心点，正负part同一步产生；这些细节都写入resolved config。

欧氏臂保持底座SGD；双曲臂使用论文RSGD。先验收RSGD在普通欧氏参数上的更新与匹配SGD一致，否则增加/修正优化器控制，不能把优化器差异算成正则收益。RSGD不是此前HIER代理AdamW。

## 7. 分阶段执行与验收

### P0：先固定DGCNN底座与数据

固定公开PyTorch分割实现的提交、许可证、数据ID/split、预处理、CE、优化器、评估和checkpoint协议。先只准备原欧氏DGCNN，不要求先解决双曲头或完成双曲代码。作者原仓库的TensorFlow part-seg代码与作者仓库列出的社区PyTorch实现分开标注。

### P1：只运行原生DGCNN，完整验收后再进入分割正则复现

先对公开DGCNN作必要的环境/路径适配，保持模型和底座数学行为；核对点与标签同步、官方split/对象ID、人工Instance/Class mIoU、空union和masked/unmasked差异，以及保存/恢复下一步一致。先训练小子集检查能否学习，再作独立2epoch×2step短测，最后从头做一条完整SEG-E0；不同时排队SEG-H0或SEG-HC。

基线验收包含：完整训练/评估与有限状态、配置/预算一致、checkpoint可重载、逐对象与逐类指标可独立复算、重复评估一致、分割图无明显错位。若同协议成绩明显偏低，先定位实现/数据/指标原因；不以未匹配协议的85.1/85.2数值作机械门槛，也不在基线异常时靠加正则补救。

DGCNN已有完整分割实现，因此属于已有代码的环境、数据、训练与评估复现，不必从论文重新编写EdgeConv。精确复现某个论文分数仍依赖对应数据协议和随机性。

### P2：冻结DGCNN基线，再逐步重建HyCoRe-seg

在同一底座上固定feature截断层、逐点双曲头、参数量、用户确认margin及未公开的实现选择，并写入 `implementation_choices.md`。必要检查包括：

1. 距离、exp/log与独立embedding叶子的Rhier径向梯度符号；共享编码器的实际深度移动另作观测，不要求一次更新同时让whole向内、part向外。
2. feature近邻选点正确且不跨对象；先欧氏平均后exp；输入/标签不被覆盖，梯度流经共享特征。
3. 负part确为异类，纯单类别batch明确处理，两个正则的分母及1000/N′可独立复算。
4. 关闭正则时零构建part selector、零额外编码；输出范围、参数量、梯度、BN、optimizer与恢复状态均可核查。

先验收双曲头无正则SEG-H0，再验收并运行完整正则SEG-HC。各自先作train-only有界短测；短测权重不初始化正式训练。监测N′分层深度gap、完整margin满足率、loss、共享编码器的加权梯度、投影比例、显存与耗时。

| ID | 分割头 | 正则 | 要回答的问题 |
|---|---|---|---|
| SEG-E0 | 原欧氏头 | 无 | 本工程的数据、评估和DGCNN基线是否可靠 |
| SEG-H0 | 同一候选双曲头 | 无 | 头部/几何改动本身的影响 |
| SEG-HC | 与SEG-H0相同 | .01Rcontr+.01Rhier | 两个正则相对双曲头控制的增量 |

三臂按E0→H0→HC依次推进，采用同seed、同split、同输入顺序/增强、同预算与选模；额外part选择用独立RNG，避免改变dropout或后续数据随机流。保存初始权重与输入指纹，公共骨干采用匹配初始化。**“先复现DGCNN”是工程与验收顺序，不意味着必须从已训练DGCNN权重微调**：主方法仍从头训练，以便比较完整训练增量；如另做微调，单独标记并配微调控制。本表为待审计划，尚未排队。

### P3：主结果稳定后补论文消融

增加“仅Rcontr”“仅Rhier”和“完整正则但XYZ邻域”三臂，分别对应论文表2、表3。先做seed4780；达到预登记的技术与预算继续条件后，最终主方法/控制结论均需3个预登记seed的配对重复，报告均值、SD及逐seed差值，不以单seed差值的大小或正负决定是否需要重复。未确认前只报告单seed观察；第二、第三seed在确认阶段登记，当前不自动扩展任务。

margin敏感性、数据协议变化、数值纠错、头部变化分别标记，不合并成一个方法增益。当前目标没有跨对象HIER增量；以后如加入，另配同协议开关对照。

### P4：最终产物

交付版本化源码和配置、数据身份/指标定义、完整选择记录、best/last检查点身份、分割图、失败案例、逐类mIoU、深度/完整margin统计、参数/显存/耗时以及未解决的论文歧义。小part不一定是语义部件，径向排序也不能直接证明发现了真实形态树。

## 8. 参考结果与完成标准

| 论文设置 | Class mIoU | Instance mIoU |
|---|---:|---:|
| DGCNN（论文表1列出的基线） | 80.9 | 85.1 |
| DGCNN + HyCoRe-seg | 82.8 | 85.7 |
| 仅Rcontr | — | 85.6 |
| 仅Rhier | — | 85.5 |
| XYZ定义part，完整正则 | — | 85.6 |

来源：[HyCoRe-seg表1–3](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。这些是论文报告值，本次没有产生任何复现成绩。DGCNN原论文另报实例85.2，不将不同来源的85.1/85.2混接为同次训练。

完成标准分三层：①算子、数据和指标正确且可恢复；②同协议E0/H0/HC可比较；③在协议差异明示、多seed结果支持的情况下评价对85.7的接近程度。不会以“某次best超过85.7”代替前两项，也不预先捏造允许误差或保证收益。

## 9. 预算与项目操作边界

预计实现、规格核对和整理需约3–5工作日；这是工作量估计。GPU用时与显存尚未实测：P1记录每步/每轮与验证耗时后，再估算3臂及后续消融，不能用现有分类任务的用时外推分割。

本轮只完成调研文档。审阅入口为 `04_NEXT_EXPERIMENT_PLAN.md`。后续实现须在本机专用分支编辑与检查，明确文件提交推送，再在服务器干净工作区fetch/pull --ff-only；训练前检查nvidia-smi，仅用完全空闲GPU。每次短测/正式run使用新目录，记录commit、config、seed、GPU、时间和checkpoint/teacher身份（从头训练、teacher无）。数据、权重、完整日志、缓存及私有服务器信息不进Git。现有结果和正在运行的其他实验不改动。

## 10. 调研来源

- [作者机构论文记录与公开接收稿入口](https://iris.polito.it/handle/11583/2982650)
- [HyCoRe官方仓库](https://github.com/diegovalsesia/HyCoRe)
- [DGCNN作者仓库](https://github.com/WangYueFt/dgcnn)及[原论文](https://arxiv.org/pdf/1801.07829)
- 固定PyTorch底座：[模型](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/model.py)、[训练/评估](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/main_partseg.py)、[数据](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/data.py)、[CE实现](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/util.py)
- 本目录另附源码检索记录与论文规格；本机双语转录仅作检索辅助，原文冲突以作者公开稿交叉核对。
