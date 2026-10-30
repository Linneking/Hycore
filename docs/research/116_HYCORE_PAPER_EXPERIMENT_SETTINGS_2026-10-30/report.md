# HyCoRe / HyCoRe-seg 论文设置与 Möbius 必要性的证据边界

日期：2026-10-30。本次核对两篇一手论文、NeurIPS官方补充材料，以及本地固定版本分类源码；仅研究，不修改模型或训练，不启动实验。

## 核心结论

**两篇论文都提出并验证“用Möbius层的预测头＋正则”的组合，但没有给出Möbius层是正则有效必要条件的证明，也没有完成“保留欧氏预测头，只在支路使用原双曲正则”的单独消融。** 不应把作者选择的完整方法直接扩大为“没有M，正则必然无效”。

这里要分开三个问题：

1. 论文的原方法是否含M？是。
2. 能否不经M、只用EXP和双曲距离定义正则？数学上可以，分割论文的正则支路本身就是此路径。
3. 保留欧氏预测头时这些双曲正则是否改善任务成绩？两篇论文所审阅的消融没有回答。

来源：[HyCoRe正文§3.3/4.3](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)、[HyCoRe-seg§3–4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。

## 论文明确的设置

| 项目 | HyCoRe分类（NeurIPS2022） | HyCoRe-seg（ICASSP2023） |
|---|---|---|
| 主数据集 | ModelNet40、ScanObjectNN，正文名义输入1024点 | ShapeNetPart，16881对象、16对象类别 |
| 基础模型 | DGCNN、PointNet++、PointMLP等；另报告PointNeXt | DGCNN |
| 正则嵌入 | E(P)→EXP→共享M→z；whole/part分别编码 | 同一逐点H→分别欧氏平均whole/part→EXP→z |
| 主支预测 | 超曲率头，Möbius＋softmax | 逐点共享的一个或多个M＋softmax |
| 正则嵌入维数 | 默认256；消融16/64/256/512/1024 | 未明确给出具体F及完整层宽 |
| part定义 | 原始3D坐标中随机中心的近邻 | 编码特征空间中随机中心的近邻 |
| part点数 | 随机200–600 | 随机200–600 |
| 分类whole点数 | 每次随机800–1024 | 未给出分类式whole再裁剪要求 |
| 层次方向 | whole远、part近原点 | whole近、part远原点 |
| 正则系数 | α=β=.01 | α=β=.01 |
| margin | 层次1000/Npart；对比4 | §3和§4有未解释冲突，见下文 |
| 优化器 | RSGD | RSGD |
| 几何框架 | PyTorch、Geoopt；标准Poincaré球，另做曲率消融 | PyTorch、Geoopt；标准Poincaré球 |
| 计算资源 | Nvidia A6000 | Nvidia Titan RTX |

分类还比较曲率绝对值1/.5/.1/.01；这不是调学习率或投影半径。分割论文没有报告对应曲率消融。分类正文明确CE只施加在whole对象上，part通过正则学习，而非额外把每个part都用其对象标签做CE。[分类正文§3.3/§4](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)。

### 分割margin的冲突不能忽略

- §3.1 写 `gamma(N′)=delta(N′)=1024/N′`。
- §4 写 `gamma=1000, delta=4`，没有解释如何与前者统一。
- 本项目目前使用 `1000/N′ + 对比常数4`，是此前用户批准的重建解释；不是分割论文给出的无歧义唯一配方。

本次不改这些已选设置，不依据哪组test更高倒推原文。[分割作者稿§3.1/§4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。

## 消融究竟证明了什么

### 分类：EuCoRe不能等同于“欧氏头＋双曲正则”

分类§4.3明确EuCoRe同时把层和距离定义为欧氏版本。Table3在ScanObjectNN、256维时报告AA：

| 方法 | AA(%) | 消融含义 |
|---|---:|---|
| DGCNN | 77.8 | 普通欧氏模型 |
| DGCNN＋EuCoRe | 79.0 | 欧氏层＋欧氏对应正则 |
| Hype-DGCNN | 76.5 | 双曲头，无正则 |
| DGCNN＋HyCoRe | 80.2 | 双曲头＋双曲正则 |

作者明确承认双曲基线可能弱于欧氏基线，正则后得到明显收益。这支持完整组合有效，也支持欧氏对应正则有一定收益；不能从中抽出M的必要性。最关键的混合组“欧氏主支＋原双曲正则”不在此表中。[分类Table3/§4.3](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)。

Table4在同一分类消融数据集报告 AA/OA：普通DGCNN77.8/80.3，仅Rhier77.9/80.5，仅Rcontr79.2/81.6，完整80.2/82.1。文字描述是在HyCoRe框架内删除一个正则，并没有将这些行定义成去掉M的消融。

Table5在曲率绝对值1/.5/.1/.01时，Hype-DGCNN AA为76.5/76.9/76.6/76.9，完整HyCoRe为80.2/79.4/78.7/78.5。它比较曲率，不是M层数量或初始化。

### 分割：没有单独报告“去M”和“仅M、无正则”

Table1普通DGCNN Class/Instance为80.9/85.1，完整方法82.8/85.7。Table2只消融正则：两项85.7，仅Rcontr85.6，仅Rhier85.5（Instance mIoU）。这些行处于双曲预测头框架语境，不能当成欧氏预测头实验。Table3特征近邻part85.7，输入空间近邻85.6。

因此，分割论文没有给出与本项目E相同的“M头＋零正则”独立结果，也没有“原欧氏预测头＋双曲正则”的结果。更不能从0.1pp差异推断任何未做设置必然无效。[分割Tables1–3](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。

## 为什么分割公式不要求正则先经过M

分割论文定义：

```text
主支：H → EXP → M层预测头 → 点级CE
支路：H → whole/part欧氏平均 → EXP → 双曲距离/原点距离 → 正则
```

正则表达式不含主支M参数，只要求送入双曲距离的z是有效球内点。即使主支是欧氏映射，这个支路也可以被定义并对共享H求导。这是由公式得到的计算可行性判断，**不是性能保证**；主支与支路在训练中仍通过共享编码器耦合。

相对地，分类论文所用z是经过共享M的embedding。因此，两篇论文的R输入位置不同，不能将分类实现中的M出现位置移植为分割支路的硬性要求。

## 论文没有补齐的细节

分割正文没有给出可核验的完整层宽/截断层、各层BN/激活/Dropout、初始化、LR、调度、batch、epoch、momentum、WD、seed、CE平滑、裁剪、投票/TTA及精确模型选择策略。未找到可核验的额外官方分割补充材料。

分类正文/补充也没有完整列出这些训练配方。官方3页Supplemental仅补测地线、最近邻part展示和UMAP，没有额外训练超参数表。[官方Supplemental](https://papers.nips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Supplemental-Conference.pdf)。分类末尾Checklist明确不报告误差条、采用best value；未交代可还原为均值±方差的重复实验数，不能将表中小差异视为已证实稳定收益。[Checklist3(c)](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=13)。

## 源码可补充的分类设置——不能标为分割论文设置

以下来自固定HyCoRe源码ef56a73的ModelNet40路径及launcher，不是正文完整披露的训练表：

| 设置 | DGCNN＋HyCoRe | PointMLP＋HyCoRe |
|---|---|---|
| 实际launcher起点LR | .1 | .1 |
| 余弦下限LR | .001（LR/100） | .005 |
| epoch | 300 | 300 |
| batch训练/测试 | 32/32 | 32/16 |
| momentum | .9 | .9 |
| weight decay | 1e−4 | 2e−4 |
| launcher实际seed | 未传入，parser默认12 | 显式4780 |
| workers | launcher显式8 | launcher显式8 |
| 裁剪 | 总loss.backward后全网L2 clip1 | 相同 |
| CE | 实际utils.cal_loss默认平滑ε=.2 | 相同 |
| BN模式 | part/whole都train，测试eval/no_grad | 同类训练/评估行为 |

证据：[DGCNN入口](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py)、[DGCNN launcher](/D:/Hycore/classification_ModelNet40/launcher_dgcnn.sh)、[PointMLP入口](/D:/Hycore/classification_ModelNet40/main_pointmlp_hycore.py)、[PointMLP launcher](/D:/Hycore/classification_ModelNet40/launcher_pointmlp.sh)、[实际CE](/D:/Hycore/classification_ModelNet40/utils/misc.py:218)。

初始化Kaiming与_bias系数4、删除DP、Geoopt投影等属于可从代码核验的实现细节；没有在上述论文中发现对应的独立初始化消融或“必须如此才能有效”的结论。DGCNN与PointMLP的minLR/WD/seed各不相同，不能笼统用某一条分类配方替代全部模型。

## 本轮D/E附问

D/E保留Conv9本体、BN9和LeakyReLU(.2)，保留其前dp1=.5；其后dp2删除，BN10及其激活也删除。两者相同主支为 `Conv8/BN8/Leaky→dp1→Conv9/BN9/Leaky→EXP→M256128→M12850`。[实际模型](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/model.py:27)。

专项笔记：[分类论文及消融](classification.md)、[分割论文及消融](segmentation.md)。以上结论不改变当前优先推进A/B双曲改造的方向，也不授权新增实验。
