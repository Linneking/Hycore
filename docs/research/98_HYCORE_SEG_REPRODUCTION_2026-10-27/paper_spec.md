# HyCoRe-seg：论文证据与待定实现项

核对日期：2026-10-27。本文为复现准备，不包含新训练结果。

## 一手来源及版本

- [论文 DOI / ICASSP 2023](https://doi.org/10.1109/ICASSP49357.2023.10095253)：Antonio Montanaro、Diego Valsesia、Enrico Magli，*Towards Hyperbolic Regularizers For Point Cloud Part Segmentation*，pp. 1–5。
- [都灵理工官方文献记录](https://iris.polito.it/handle/11583/2982650)及[开放作者接受稿 PDF](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。本次工具解析为 6 页：第 1 页为学校版权封面，后面 5 页为论文。因此“正文 p3”对应 PDF 第 4 页（零起页码 3），正文 p4 对应 PDF 第 5 页（零起页码 4）。PDF 文本已完整读取；网页截图工具返回 cache miss，未完成独立图像核验。
- [第一作者博士论文](https://tesidottorato.depositolegale.it/bitstream/20.500.14242/157866/7/conv_phdPolito_AntonioMontanaro.pdf)，§5.2、印刷页 93–94：检索索引可读取该节，全文直接打开返回 403；该节重复了下述 margin 矛盾，未提供消歧依据。
- [DGCNN 作者仓库](https://github.com/WangYueFt/dgcnn)及[原论文 arXiv:1801.07829v2](https://arxiv.org/pdf/1801.07829)，§4.4 / PDF p8、Fig.3、Table6：仅用于骨干与指标背景，不能充当 HyCoRe-seg 未公开配方的证据。
- 本地已有中英转录：`资料库/5.Towards_Hyperbolic_Regularizers_For_Point_Cloud_Part_Segmentation - 副本.md`；与上面可获取的一手 PDF 文本核对。未发现另附的官方补充材料；这是本轮检索范围内的结论。

## 可直接恢复的模型定义

正文 §3、§3.1、Fig.1：编码器输出逐点欧氏特征 `H = E(X)`，形状 `[B,N,F]`；分割头经指数映射及逐点共享的 Möbius 层给出部件类别分数。正则分支使用同一组逐点特征，在欧氏域先平均，再映到双曲域。

\[
z_{w,i}=\exp_0\left(\frac1N\sum_n H_{i,n}\right),\qquad
z_{p,i}=\exp_0\left(\frac1{N'_i}\sum_{n\in I_i}H_{i,n}\right).
\]

`I_i` 是随机中心点在**编码器特征空间**中的 `N'` 个最近邻。它可以包含空间上分离但特征相似的点；不要求遵循真实语义部件标签。论文 §4 给出 `N'` 在 200–600 随机选择。

这里的 part 是从已有逐点特征中选择并池化。论文未要求像分类 HyCoRe 那样裁剪原始点云后再次运行编码器；也未引入教师、层次代理、LCA 或 HIER 目标。[依据：接受稿 §3.1 / Fig.1](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)

## 正则项和方向

正文 §3.1、式 (1)：

\[
L=L_{CE}+\alpha R_{contr}+\beta R_{hier}.
\]

\[
R_{hier}=[\|z_w\|_{\mathbb D}-\|z_p\|_{\mathbb D}+\gamma(N')]_+,
\]

\[
R_{contr}=[d_{\mathbb D}(z_w^+,z_p^+)-d_{\mathbb D}(z_w^+,z_p^-)+\delta(N')]_+.
\]

正 part 来自同一对象；负 part 来自**另一对象且对象类别不同**。分割方向要求 part 比 whole 更深；与分类 HyCoRe 的 whole 更深方向相反（§3.2）。因此不能直接复用分类损失的径向符号。

正文 §2.2 的单位球距离：

\[
d_{\mathbb D}(x,y)=\operatorname{acosh}\!\left(1+\frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right).
\]

把 `||z||_D` 实现成原点双曲距离 `d_D(0,z)=2 atanh(||z||_2)` 是标准几何解释；应以解析合同验证，不能误用欧氏模长。论文背景同时混用负曲率和 `sqrt(c)` 记号，且若干显示公式存在疑似排印问题；实现应固定 Geoopt 正参数 `c=1` 对应几何曲率 `−1`，逐式验算并记录符号约定，不能逐字照搬有歧义的公式。[依据：接受稿 §2.2–§3.2](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)

## 关键未解决矛盾：margin

| 位置 | 写法 | 可确认程度 |
|---|---|---|
| 正文 §3.1 末段（正文 p3） | `gamma(N') = delta(N') = 1024/N'` | 已从官方 PDF 文本确认 |
| 正文 §4（正文 p4） | `alpha=beta=.01, gamma=1000, delta=4` | 已从官方 PDF 文本确认 |
| 博士论文 §5.2（印刷 p93–94） | 同时保留上述两套表述 | 官方索引文本交叉确认；没有消歧 |

§4 没有明确写 `gamma(N')=1000/N'`；它也没有解释如何与 §3.1 的两个 `1024/N'` 同时成立。`1000/N' + 常数4` 是一种受分类前作启发的解释，不能标成已核实的分割作者配方。字面常数 1000 亦不宜未经检查直接训练。复现计划需保存 competing interpretations，先做量纲/可达性核验，必要时在训练数据/验证集上做有界比较；不可通过测试集成绩决定“原公式”。本轮没有联系作者。

## 论文明确给出的实验事实

正文 §4：ShapeNetPart、16,881 对象、16 类；实验骨干是 DGCNN。分割头改成共享 Möbius + softmax，参数总数宣称不变；`alpha=beta=.01`；Riemannian SGD；PyTorch / Geoopt；Titan RTX。[依据：接受稿 §4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)

| 表格 | 论文报告 |
|---|---|
| Table1 | DGCNN Cls/Inst mIoU：80.9 / 85.1；HyCoRe-seg：82.8 / 85.7 |
| Table2 | 两项：85.7；仅 contrastive：85.6；仅 hierarchy：85.5（Inst mIoU） |
| Table3 | 特征空间选 part：85.7；输入 XYZ 空间：85.6（Inst mIoU） |

这些是论文数值，既不是当前项目已复现结果，也没有给出多 seed 方差；0.1pp 的消融差距不能先验视为稳定增益。[依据：接受稿 Table1–3](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)

## 必须显式补足的实现信息

论文未披露：完整 DGCNN 代码版本；`E` 截止层/特征宽度；Möbius 层数量与宽度；是否沿用类别向量、各层 BN/激活/dropout；分割输出的具体 softmax 实现；输入点数；法线；增强；训练/验证/测试使用细节；LR、动量、WD、epoch、batch、调度、seed；CE 平滑；原点 expmap 的尺度及数值投影；正则 reduction。

part 选择还缺：中心与 `N'` 的分布及每对象/每批共享规则、每对象 part 数、近邻度量及特征归一化、是否含中心、并列距离规则。负样本缺精确挖掘规则；纯单类别 batch 必须有明确 skip 或重采样策略，不能把同类别 `flip` 误当异类。kNN 的索引可按离散选择处理，但是否显式 detach 未写明；对选中特征值的梯度应保留，这是建议实现约定。

以上缺项应写成版本化工程决策，不能移植 ModelNet40/PointMLP 的 300 轮、batch32、1000/N、margin4、cap2.3 或 HIER 梯度 hook 后声称忠实复现。

## DGCNN 一手资料可提供的独立起点

原 DGCNN 论文 §4.4 使用 ShapeNetPart 的 16 类、50 部件标签、每训练形状 2048 点、官方 train/val/test；分割网络包括 STN、三层 EdgeConv、逐点 256/256/128 层。其训练沿用分类设置（SGD .1、cosine 至 .001、momentum .9、batch32），这些只是**骨干参考设置**。HyCoRe-seg 没有明确声称沿用全部设置。

原论文 Table6 的 DGCNN Inst mIoU 为 85.2，而 HyCoRe-seg 表内基线为 85.1，不能认为是同一 checkpoint。原 DGCNN §4.4 描述 shape IoU、类别均值与全 shape 均值；评估实现仍需固定合法 part 集合、空并集规则、对象类别 mask、点采样与 voting，并分别报 class / instance mIoU。[依据：DGCNN 原论文 §4.4、Table6](https://arxiv.org/pdf/1801.07829)

建议先实现欧氏 DGCNN 和双曲头无正则对照，再逐项加入两个正则，独立检查“头替换”与“正则增量”；此为复现实验设计建议，并非论文已提供的完整消融。
