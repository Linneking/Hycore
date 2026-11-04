# HyCoRe 输入点数与 part 比例核查（2026-11-04）

## 分类与分割的输入点数

| 场景 | 输入与采样 | 一手依据 |
|---|---|---|
| 原版 DGCNN，ModelNet40 分类 | 原论文主设置为每物体 1024 点，另有 2048 点、`k=40` 的单列结果。PyTorch 默认 `num_points=1024`，从 `modelnet40_ply_hdf5_2048` 加载数据，再按 `num_points` 截取。因此 HDF5 的采样容量与分类网络实际收到的点数是两件事。 | [DGCNN §4.1 与表2](https://arxiv.org/pdf/1801.07829#page=7)，[训练入口](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/main.py#L185-L193)，[数据切片](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/data.py#L17-L71) |
| HyCoRe 作者复训的 DGCNN 分类 baseline | 发布代码在 ModelNet40 下同样默认 1024 点，launcher 未覆盖，训练与测试将 loader 点集直接送入模型。论文表1的 DGCNN 行为作者复训的 92.9 OA 设置。 | [HyCoRe §4.1–4.2](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7)，[作者 baseline 入口](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/main_dgcnn.py)，[launcher](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/launcher_dgcnn.sh) |
| DGCNN＋HyCoRe 分类 | ModelNet40 loader 仍给 1024 点。训练先从中用 XYZ KNN 取 800–1024 点作为 whole，再在该 whole 中取 200–600 点作为 part，各自编码一次，测试直接使用 1024 点。分类论文对 ScanObjectNN 也给出名义 1024 点。作者发布的 ModelNet40 PointMLP 与 DGCNN 的 nominal 1024 与两级采样设置一致。 | [HyCoRe §4.1](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7)，[DGCNN 训练代码](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/main_dgcnn_hycore.py#L214-L232)，[PointMLP 训练代码](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/main_pointmlp_hycore.py#L202-L227) |
| 原版 DGCNN，ShapeNetPart 部件分割 | 原论文明确每个训练形状采样 2048 点，对输入点逐点预测部件标签。2048 是 DGCNN 在这一分割基准上的采样协议。 | [DGCNN §4.4](https://arxiv.org/pdf/1801.07829#page=8) |
| HyCoRe-seg 论文 | 以 DGCNN 做 ShapeNetPart，定义 whole 为全部 $N$ 个逐点特征的池化，part 为 $N'$ 个特征空间近邻的池化。实验节给出 $N'=200$–$600$，接受稿未明列固定的输入 $N$。方法节的 $1024/N'$ 出现在 margin 公式中。 | [HyCoRe-seg 接受稿 §3.1、§4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)，[本地接受稿核对](98_HYCORE_SEG_REPRODUCTION_2026-10-27/paper_spec.md) |
| 本项目的 ShapeNetPart DGCNN baseline 与分割复现 | 两侧都配置 2048 点，dataset 同步截取点与部件标签。分割只对 2048 点编码一次，whole 对 2048 个逐点特征做均值，part 从同一组 2048 个特征里取 200–600 个近邻后做均值。 | [baseline 配置](../../segmentation_ShapeNetPart/hycore_seg_v1/configs/dgcnn_baseline.json)，[输入切片](../../segmentation_ShapeNetPart/hycore_seg_v1/data.py#L45)，[分割正则配置](../../segmentation_ShapeNetPart/hycore_seg_v11/configs/s5_tan_r.json)，[特征采样](../../segmentation_ShapeNetPart/hycore_seg_v2/losses.py#L14)，[whole 和 part 池化](../../segmentation_ShapeNetPart/hycore_seg_v3/losses.py#L148) |

ShapeNetPart 的原始点表可以有不同长度。本项目按每物体 2048 个 XYZ 槽位及同索引标签打包，长度不足时按取模重复。DGCNN 分割网络里的 `emb_dims=1024` 表示特征通道宽度。数据预处理和 `num_points` 决定输入点数。 [本项目数据打包](../../segmentation_ShapeNetPart/hycore_seg_v1/prepare_data.py#L92)，[DGCNN 分割结构及实验协议](https://arxiv.org/pdf/1801.07829#page=8)

分类作者代码的裁剪函数将 `data[:,:,:k]` 当作写入目标。这里的训练 whole 和 part 点数指张量的点槽位数。比较覆盖范围时，还要区分分类对 XYZ 选邻居后重新编码、分割对同一次编码得到的特征选邻居后池化。分类的默认层次方向要求 whole 更深，分割要求 part 更深。两者的比例一致只约束采样规模。 [分类采样及损失实现](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/hutil.py#L27-L68)，[分割损失实现](../../segmentation_ShapeNetPart/hycore_seg_v3/losses.py#L61)

## 翻倍后的比例与 margin

| 尺度参照 | part / whole 点数比例 |
|---|---:|
| 分类的名义 1024 点与 part 200–600 | 19.53%–58.59% |
| 分类实际训练的 whole 800–1024 与 part 200–600，按极端端点计算 | 19.53%–75.00% |
| 本项目现用 whole 2048 与 part 200–600 | 9.77%–29.30% |
| 本项目候选 whole 2048 与 part 400–1200 | 19.53%–58.59% |

400–1200 精确匹配相对名义 1024 点的比例，也保留了分类实际训练比例的下界，但没有覆盖其 58.59%–75% 的上段。若目标是逐样本映射分类的实际训练比例，公式应为 $N'_{seg}=\operatorname{round}(2048N'_{cls}/N_{whole,cls})$，并保留原分母的随机性。

现用分割代码中，层次 hinge 的间隔是 $m=\texttt{gamma\_base}/N'$，要求 $d(0,z_p)-d(0,z_w)\ge m$。分割稿方法节给两个 margin 写了 $1024/N'$，实验节另列 $\gamma=1000$ 与 $\delta=4$。本项目具体采用 $1000/N'$ 和常数 4。 [分割论文 §3.1、§4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)，[本地实际计算](../../segmentation_ShapeNetPart/hycore_seg_v3/losses.py#L159)

| 对同一原始抽样 $k$，将 $N'=k$ 改为 $N'=2k$ | `gamma_base` | 实际 intra margin 相对旧值 |
|---|---:|---:|
| 只扩大 part | 1000 | 0.5 倍 |
| 扩大 part 并保持原实际 margin | 2000 | 1 倍 |
| 扩大 part 且实际 margin 加倍 | 4000 | 2 倍 |

对比项的常数 margin $\delta=4$ 可固定。增大 part 会改变正负配对距离，故即便 $\delta$ 不变，实际对比 loss 的活跃率也可能改变。保持 intra margin 的配对候选是 `part_min=400, part_max=1200, gamma_base=2000, delta=4`。若希望两个离散采样分布严格配对，先采 $k\in[200,600]$ 再用 $2k$，而直接均匀采 400–1200 还会包含奇数点数。

本地用实际 v2 采样与 v3 loss 对合成 `[4,2048,16]` 特征做 CPU 核验。相同随机中心下，原 $k$ 个邻居是 $2k$ 个邻居的前缀，whole 深度一致而 part 均值改变。对原始 $k=200,400,600$，依次实测 `旧1000/k、扩大后1000/(2k)、扩大后2000/(2k)、扩大后4000/(2k)` 为 `[5,2.5,5,10]`、`[2.5,1.25,2.5,5]`、`[1.666667,.833333,1.666667,3.333333]`。对比 margin 均为 4，CUDA 未初始化。

若真将实际 intra margin 加倍，400 点对应 $m=10$。当前单位 Poincaré 球 FP32 投影半径约 .996，原点最大测地深度约 $2\operatorname{atanh}(.996)=6.213$，因此部分小 part 的 hinge 无法归零。这个设置需要独立的强化约束假设。 [投影半径与诊断](../../segmentation_ShapeNetPart/hycore_seg_v3/losses.py#L74)
