# 双曲点云分割公开代码核查

核查日期：2026-11-01。范围：作者仓库、作者主页、论文正式页面和实际源文件。这里只做源码与文献研究，未安装或训练这些外部项目。下文的“未找到”表示本轮检索未能确认，不表示可以证明代码从未公开。

## 结论

**目前最贴近 DGCNN + ShapeNetPart 的实际公开实现是 HPCS；本轮未确认 HyCoRe-seg 官方分割代码。** HPCS 可以提供点级双曲嵌入与层次损失的实现参照，但其任务目标、输出与评估方法不等同于我们的 50 类部件语义分割。另一类值得借鉴的来源是双曲图像分割：逐像素和逐点都需要将双曲特征变成每类实数分数，其分类器比整套骨干更容易迁移。

## 按本项目的参考价值排序

| 优先级 | 仓库与可用状态 | 数据/任务 | 已核实的核心实现 | 对本项目的用途 |
|---|---|---|---|---|
| 1 | [TheCrossProduct/HPCS](https://github.com/TheCrossProduct/HPCS)，有训练入口、模型、损失与推理代码 | ShapeNet 部件、PartNet 多层部件 | DGCNN/VN-DGCNN、PointNet/VN-PointNet；EXP 或 MLP+EXP；CosFace/Triplet 与 HypHC 联合训练 | 最接近的点云实现参照；不宜直接拿其层次聚类分数对比我们的 Instance mIoU |
| 2 | [mikhail-vlasenko/hyp-segmentation](https://github.com/mikhail-vlasenko/hyp-segmentation)，有 PyTorch 模型/训练/分类头源文件 | SPIN 对象、部件、子部件；图像 | SegFormer 融合特征后 EXP；原型距离分数或双曲 MLR；有分类分数尺度和边界数值保护 | 优先借鉴双曲特征到实数 logits 的读出，不是点云开箱即用模型 |
| 3 | [MinaGhadimiAtigh/HyperbolicImageSegmentation](https://github.com/MinaGhadimiAtigh/HyperbolicImageSegmentation)，官方仓库明确标注代码尚不完整 | COCO-Stuff 等图像分割 | 双曲分割与类别层次；TensorFlow 1.14，样例采用 c=0.1、冻结 BN | 双曲 MLR/分割语义的原始参考；环境较老，迁移成本高 |
| 4 | [Pangoraw/horospherical](https://github.com/Pangoraw/horospherical)，有原型与分类器组件，点云训练未发布 | 论文含 NuScenes；当前代码主要为分类 | 边界原型、层次指导的原型分配、horospherical 分类器 | 可读组件；不能列为已提供 NuScenes 完整复现的代码 |
| 5 | [diegovalsesia/HyCoRe](https://github.com/diegovalsesia/HyCoRe)，官方分类代码可用 | ModelNet40、ScanObjectNN 分类 | PointMLP/DGCNN、Möbius 层与 whole/part 正则 | 核对原始分类方法与初始化，不能代替未公开的分割实现 |

以上仓库是否有训练代码是静态核查结论，**并不意味着它们已经在我们的依赖版本下运行通过**。

## HPCS：最接近任务，但需看清其学习和评估对象

[论文](https://openaccess.thecvf.com/content/ICCV2023W/SHARP/papers/Onghena_Rotation-Invariant_Hierarchical_Segmentation_on_Poincare_Ball_for_3D_Point_Cloud_ICCVW_2023_paper.pdf)明确给出项目仓库。实际 [train.py](https://github.com/TheCrossProduct/HPCS/blob/main/train.py) 支持 ShapeNet、PartNet，前者设 16 个物体类别、50 个部件类别；模型选项包含普通和 VN 版本的 DGCNN/PointNet。训练入口按欧氏/双曲嵌入维度是否相同选择 EXP 或 MLP+EXP，并使用 validation loss 做调度/早停。这是有实际源文件支撑的点云分割实现，不是只写了论文标题的占位仓库。

其结构可压缩为：

```text
原始点 + 类别输入
  → 点云特征网络（可选 DGCNN/VN-DGCNN）
  → 逐点欧氏嵌入
  → EXP 或 MLP+EXP
  → 逐点双曲嵌入
      ├─ CosFace / Triplet 部件监督
      └─ 三元组相似度 + 双曲 LCA 的 HypHC 目标
```

[ShapeNet 模型](https://github.com/TheCrossProduct/HPCS/blob/main/hpcs/models/shapenet_hyp_hc.py)中，两种嵌入来自每个点，模型默认使用类别 one-hot；可选 `class_vector` 会从部件真值构建存在部件向量，不能不加区分地启用为普通推理输入。

[损失源码](https://github.com/TheCrossProduct/HPCS/blob/main/hpcs/loss/ultrametric_loss.py)采用 CosFace margin=0.35、scale=2 或三元组损失。HypHC 分支把三元组点归一到共同半径，再利用 LCA 深度计算 softmax 权重；共同半径由可学习 `scale` 控制并夹在 `[1e-4,1]`。该参数在 [Base 类](https://github.com/TheCrossProduct/HPCS/blob/main/hpcs/models/base_hyp_hc.py)初始化为 `1e-3`。**这是 LCA 目标中的共同叶半径，不是入球前缩放；也不能直接用来支持 HyCoRe-seg 的 whole 浅、part 深径向关系。**

评估有两项必须单独辨认：

- `BaseSimilarityHypHC.test_step` 把每个物体的真值传给 `get_optimal_k(targets, linkage, 'iou')`，获得其最优聚类切分得分。这种 `score` 与我们的固定类别 argmax Instance mIoU 不同，不能直接比较数字。
- 同一仓库另报 CosFace 的 `test_iou`；但 `get_logits(embeddings, labels)` 使用真值掩码给目标类别施加 CosFace margin，再用于 `compute_iou`。移植时应重新明确“训练加 margin，推理不依赖真值”的分类协议。

这两点均直接来自上述 Base 类与损失源文件。建议只借鉴其点级监督、尺度监控和模块实现，并用我们自己的标准数据/推理/指标代码评估。

## 双曲图像分割：最值得借鉴的是最后的分类分数

[hyp-segmentation 的分割头](https://github.com/mikhail-vlasenko/hyp-segmentation/blob/master/segformer_head.py)保留 SegFormer 融合模块的 BN、activation、dropout，再降维并进行 EXP。随后可走两种读出：

```text
逐像素/逐点欧氏特征 → EXP → 双曲嵌入 z
  ├─ 距离原型 p_k：logit_k = -τ · d(z,p_k)
  └─ 双曲 MLR：以类别 offset、normal 得到实数 logit_k
       → CE
```

源码中的 `tau` 是乘子，不应根据常见温度记法误写成除以 tau。原型例程将均匀原型缩到 0.95；MLR 路径把 offset 初始化为零、normal 初始化为小量后入球。[hyperbolic_layers.py](https://github.com/mikhail-vlasenko/hyp-segmentation/blob/master/hyperbolic_layers.py)的 MLR 分数最终为带法向量模长因子的 `asinh` 表达式，并有投影与分母保护。**双曲嵌入仍在球内，分类分数则是标量；两者不应默认视为同一张量。**

这为本项目提供一个局部候选：保留已经确定的主干、首层 Möbius 和正则位置，仅改变最后 `128→50` 的读出方式，比较当前 Möbius 坐标 logits 与真正的 hyperbolic MLR/原型距离 logits。它是需要命名的新方法变体，并非已经证明符合 HyCoRe-seg 未公开代码的修正。

该库包含直接 `.cuda()` 调用和图像专用张量布局，不能原封不动复制到 CPU/多卡点云代码。其 [losses.py](https://github.com/mikhail-vlasenko/hyp-segmentation/blob/master/losses.py)另含原型距离比值损失及可选范数惩罚；它们的存在不等于应该同时添加。先解决单一读出问题，再独立验证额外目标，才可归因。

## 尚不能作为现成点云分割代码的候选

- **HyCoRe-seg**：核查 [作者官方 HyCoRe 仓库](https://github.com/diegovalsesia/HyCoRe)时，顶层模型目录为 `classification_ModelNet40` 和 `classification_ScanObject`，README 也只声明这两个分类数据集。未确认分割训练入口、分割配置或作者分割 checkpoint。
- **Horospherical Learning with Smart Prototypes**：论文有点云分割实验，但 [仓库 README](https://github.com/Pangoraw/horospherical)明确把 Semantic Segmentation 与 Point-Cloud Segmentation 标为 `TO BE RELEASED`。可引用其思想，不能把它当完整分割复现基座。
- **HiPO，CVPR 2025**：确实是 S3DIS/ScanNetv2 的少样本增量点云语义分割，[CVF 正式论文](https://openaccess.thecvf.com/content/CVPR2025/html/Sur_Hyperbolic_Uncertainty-Aware_Few-Shot_Incremental_Point_Cloud_Segmentation_CVPR_2025_paper.html)和[第一作者主页](https://surtantheta.github.io/)可核实；本轮未确认其公开实现链接。其 CLIP 语义原型、增量协议与本项目普通 ShapeNetPart 分割有明显差异。
- **EDGCNet / TIH**：出版社页面能确认双曲点云分割方法，但本轮未核实对应公开完整代码：[EDGCNet](https://www.sciencedirect.com/science/article/pii/S0957417423020535)、[TIH](https://www.sciencedirect.com/science/article/pii/S1568494626003637)。不以二手搜索片段中的其他项目链接冒充其代码。
- **Kalitzke 的 Hyperbolic Segmentation on Point Clouds**：[TUM 课程报告](https://cvg.cit.tum.de/_media/members/webersi/kalitzke2024hyperbolic.pdf)讨论 ShapeNet、双曲 MLR 和初始化敏感性；本轮未确认作者训练仓库。它是课程项目证据，不是正式同行评审的优越性证明。
- **HypLiLoc**：[作者仓库](https://github.com/sijieaaa/HypLiLoc)是 LiDAR 位姿回归，有真实代码但不是分割。
- **Hypformer**：[官方仓库](https://github.com/Graph-and-Geometric-Learning/hyperbolic-transformer)提供图节点学习与通用双曲层，没有在本次核查中发现 ShapeNetPart 分割入口；整套换骨干会引入过多变量。
- **HypMix**：检索到的是超谱解混包、文本插值增强或图表示研究等同名项目；未确认贴合本任务的双曲点云分割实现。

## 对后续计划的直接建议

1. 继续以现有分割主干为主线，先区分“球内特征”和“CE 输入分数”的数值范围。若我们的 CE 长期停在约 3，应先核查最终坐标 logits 的可达置信度；公开分割代码支持把这项列为可验证的重点，而非立即归因于正则无效。
2. HPCS 可用作另一路方法参考，但其共同半径 LCA、点级标签监督、旋转设计和目标依赖的聚类评估均需单列，不能声称它验证了 HyCoRe-seg 的径向正则。
3. 若尝试新读出，使用同一主干、同一初值策略及同一数据协议，在同一读出下比较正则开/关；这样才能回答“正则是否提高分割成绩”。不把更换读出、调整曲率、改变正则系数和采样规则合成一次实验。
