# HyCoRe 分类论文：实验设置与 Möbius 必要性的证据边界

2026-10-30，仅一手来源研究与本地源码只读核对。未训练、未修改模型、未使用GPU。

## 一手来源

- [正式 NeurIPS 2022 主文，13页含参考文献与checklist](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)
- [正式补充材料，3页](https://papers.nips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Supplemental-Conference.pdf)
- [论文入口](https://papers.nips.cc/paper_files/paper/2022/hash/da8f9fc2b555d122369f36a9684415c1-Abstract-Conference.html)
- [官方代码](https://github.com/diegovalsesia/HyCoRe)，本地固定来源记录为ef56a73020e309b1009c486fea1f601d3f7b3252；下面另列的代码设置不冒充论文逐项写明的设置。

## 1. 论文没有证明“正则只有配 Möbius 层才有效”

正文§3.3/p6提出的完整方法采用双曲嵌入与双曲分类头，正则定义于`H(EXP(E(input)))`；这是作者选用的结构。§4.3/p9没有给出Möbius必要性定理，也没有报告“保留欧氏分类头、但仍在双曲空间计算正则”这一混合方案的直接消融。[主文§3.3、§4.3](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)

Table3的EuCoRe需要特别区分：其距离和网络层都改成欧氏版本，不能把它当作“欧氏主支＋原双曲正则”的实验。它说明欧氏版本的正则也有增益，但不直接回答上述混合方案能否有效。

### Table3：ScanObjectNN，Average Accuracy（AA%，不是OA）

| 模型 / 嵌入维数 | 16 | 64 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| DGCNN | 76.6 | 77.5 | 77.8 | 76.6 | 76.3 |
| DGCNN+EuCoRe | 78.2 | 78.9 | 79.0 | 78.8 | 79.0 |
| Hype-DGCNN，无正则 | 76.8 | 75.9 | 76.5 | 76.0 | 77.5 |
| 完整HyCoRe | 79.1 | 80.0 | 80.2 | 80.2 | 79.7 |

因此在f256时：欧氏正则版本比原DGCNN的AA高1.2个百分点；双曲头无正则反而低1.3；完整方法比双曲无正则高3.7。该结果支持作者组合的有效性，不能推出M层是正则产生任何收益的必要条件。[主文Table3/p7及解释§4.3/p9](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)

正文§3.3/p6及§4.3/p9还主动讨论了双曲分类基线弱于欧氏对应物，提出改善双曲基线本身仍是开放问题。这里不能将“头换为双曲”与“已有保证的性能改善”画等号。

## 2. 正文明确的设置

| 项目 | 主文明确内容 |
|---|---|
| 数据与对象类别 | ModelNet40：40类；ScanObjectNN：15类；输入1024点 |
| 架构 | DGCNN、PointNet++、PointMLP；ScanObjectNN结果还列PointNeXt |
| 嵌入维数 | 默认f=256；Table3消融16/64/256/512/1024 |
| 正则系数 | α=β=.01 |
| 层次margin | γ/Npart，γ=1000 |
| 对比margin | δ=4 |
| part | 随机200–600点，在XYZ空间取随机中心的最近邻 |
| whole | 训练时随机800–1024点 |
| 分类损失 | 对whole计算CE；part由正则约束，不额外做分类CE |
| 几何/优化 | Poincaré球、Riemannian SGD、PyTorch和Geoopt |
| GPU | Nvidia A6000 |

依据为主文§3.3/p6与§4.1/p7。方法文字含“嵌入投影M层＋分类M层”；精确的原骨干截断位置、每层宽度、Dropout/BN等没有在实验节逐层列出，需代码补充。

正文和三页补充中未给出完整的batch size、学习率、轮数、BN设置、初始化或标签平滑配方；补充主要是测地线与UMAP可视化，不是完整训练配置表。[补充材料全文](https://papers.nips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Supplemental-Conference.pdf)

Checklist/p13 §3(c)明确没有报告随机种子重复实验的误差条，采用最佳值比较；因此不能将表中单值称为多seed均值或均值±标准差，也未据此获得每项实验的确切seed数。§3(b)说明超参数选择考虑公平比较或粗略交叉验证。[主文checklist/p13](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)

## 3. 另外两项消融

### Table4：ScanObjectNN，删除某个正则

| 模型 | AA% | OA% |
|---|---:|---:|
| 原DGCNN参考 | 77.8 | 80.3 |
| 只保留Rhier | 77.9 | 80.5 |
| 只保留Rcontr | 79.2 | 81.6 |
| 两项均保留 | 80.2 | 82.1 |

单正则行是完整HyCoRe中的删项，不是“保持原欧氏头，只加该双曲正则”。在作者这组设置中，对比项的增益大于单独层次项，联合最好；不能外推到分割或我们的seed配置。[主文Table4/p8、§4.3/p9](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)

### Table5：ScanObjectNN，曲率幅值

| c幅值 | 1 | .5 | .1 | .01 |
|---|---:|---:|---:|---:|
| Hype-DGCNN，无正则AA% | 76.5 | 76.9 | 76.6 | 76.9 |
| 完整HyCoRe AA% | 80.2 | 79.4 | 78.7 | 78.5 |

表5用正数报告曲率幅值，与负截面曲率的符号约定区分。主文也比较了稀疏输入、局部part输入与随机crop增强；这些不是去除M层的实验。[主文Table5/p8、Figure5/p10](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)

## 4. 公开分类代码另提供的配方信息

以下是发布代码的ModelNet40路径，非论文表格逐次运行manifest。

| 设置 | DGCNN+HyCoRe | PointMLP+HyCoRe |
|---|---|---|
| 实际launcher轮数 | 300（覆盖parser200） | 300 |
| 起始LR / 最小LR | .1 / .001（覆盖parser .01） | .1 / .005 |
| momentum / WD | .9 / 1e−4 | .9 / 2e−4 |
| train/test batch | 32 / 32 | 32 / 16 |
| launcher workers | 8 | 8 |
| launcher seed | 未传，parser12 | 显式4780，覆盖parser22 |
| 总梯度裁剪 | 全网L2 clip1 | 全网L2 clip1 |
| 优化器 | RSGD | RSGD |

代码来源：[DGCNN launcher](/D:/Hycore/classification_ModelNet40/launcher_dgcnn.sh)、[DGCNN入口](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py)、[PointMLP launcher](/D:/Hycore/classification_ModelNet40/launcher_pointmlp.sh)、[PointMLP入口](/D:/Hycore/classification_ModelNet40/main_pointmlp_hycore.py)。官方在线对应入口：[DGCNN](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/main_dgcnn_hycore.py)、[PointMLP](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/main_pointmlp_hycore.py)。

共享ModelNet40数据代码的训练增强为各坐标轴scale∈[2/3,1.5]、平移∈[−.2,.2]及点顺序shuffle；不是ShapeNetPart社区数据代码的仅点序置换。分类CE实际走utils.cal_loss默认ε=.2；BN为普通train/eval、训练part和whole均train、测试eval/no_grad。这些代码事实不能据此声称HyCoRe-seg论文也规定同一增强、BN或标签平滑设置。[数据代码](/D:/Hycore/classification_ModelNet40/data.py:49)、[实际CE](/D:/Hycore/classification_ModelNet40/utils/misc.py:218)、[模式切换](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:206)。

本次未发现主文/官方三页补充直接测试“欧氏主分类支路＋双曲辅助正则、完全不使用M”的结果。不能把“本文采用M”写成“已证明不用M必无效”。
