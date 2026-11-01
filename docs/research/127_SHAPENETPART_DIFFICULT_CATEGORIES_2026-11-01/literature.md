# ShapeNetPart 困难类别：文献证据与诊断含义

日期：2026-11-01。范围：文献调研；未修改训练代码、未连接服务器、未启动实验。本页记录三篇原始论文，区分作者的直接观察与本项目待验证假设。

## 三种问题不能合并成“小类”

| 层面 | 定义 | 应观察的量 |
|---|---|---|
| 物体类别样本少 | bag、rocket 等整类物体在训练集中少 | 每类训练物体数、出现批次、每类 IoU |
| 物体内部部件少 | 一个物体中某部件仅占少量点 | 各部件真实/预测点数、recall、precision、IoU |
| 几何辨识困难 | 细长结构、弱几何边界、几何相似但语义不同 | 丢失整块部件还是仅边界错分；局部邻域和全局上下文 |

当前项目统计由本次主审查提供：trainval 共 14007 个物体，rocket 54、earphone 55、bag 62、skateboard 121、motorbike 151，确属低样本类别。但 cap 仅 44 个仍达到 H 类别 IoU 88.895%，mug 146 个达到 94.685%。因此“样本少”不是充分解释。mug 的两部件点数约 6.04%/93.96%，也说明点数不均衡本身不能推出失败。

## 直接研究 ShapeNetPart 不均衡：3D U-net，2019

Roman Janovský、David Sedláček、Jiří Žára，*On Improving 3D U-net Architecture*，ICSOFT 2019，649–656，DOI 10.5220/0007830306490656。[出版方论文页](https://www.scitepress.org/PublishedPapers/2019/78303/)，[论文全文](https://www.scitepress.org/PublishedPapers/2019/78303/78303.pdf)。

§4.4、Figure 3 直接展示 bag 细带、earphone 带、rocket 头部，以及 motorbike 油箱/座椅等被相邻部件吞并。§4.5–4.7研究类别不均衡、损失加权及过采样；将类别补到相同数量、每批每类一个样本，改善部分困难类，但存在整体 IoU 下降的代价。简单频率加权也未显著改善训练。

**适用范围：**这是体素 3D U-net 在 ShapeNetPart 上的结果，证明相似困难已有研究；不能据此认定我们的 DGCNN 双曲头失败原因相同，或保证均衡采样有效。对本项目最有用的是它提出的“整类曝光量”和“少数部件被相邻部件吞并”两条独立诊断线索。

## 小部件与语义歧义：PartNet，CVPR 2019

Kaichun Mo 等，*PartNet: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level 3D Object Understanding*。[CVF 正式论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mo_PartNet_A_Large-Scale_Benchmark_for_Fine-Grained_and_Hierarchical_Part-Level_3D_CVPR_2019_paper.pdf)。

§5.1 指出细粒度分割要识别小部件与外形相似的语义部件：门把手平均只占 10000 点中的 77 点；床的立柱、侧杆、腿等局部几何相似。作者强调同时需要局部判别特征和整物体上下文。基准结果显示更细粒度分割明显更难。

**适用范围：**PartNet 与本项目 ShapeNetPart 是不同数据集，不能混用成绩或类频率。它支持区分“低样本类别”和“低点数部件”，以及检查几何相似部件的混淆，不能证明双曲层必然损害局部几何。

## 这些部件存在可改善空间：PointNeXt，NeurIPS 2022

Guocheng Qian 等，*PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies*。[官方补充材料](https://proceedings.neurips.cc/paper_files/paper/2022/file/9318763d049edf9a1f2779b2a59911d3-Supplemental-Conference.pdf)。

补充材料 §C、Figure III 比较 PointNet++ 与 PointNeXt 的 ShapeNetPart 预测，作者明确指出改进了 earphone 麦克风、motorbike 车身、rocket 尾翼、skateboard 轴承等部位的分割。论文同时研究训练策略和网络扩展，因此不能把收益单独归因于某一个模块。

**适用范围：**说明这些部件并非只能接受低分，细节表征和训练设计值得检查；这只是文献线索，不构成更换当前 backbone 或增加某模块必定有效的证据。

## 对当前 DGCNN / H0 / H 的直接建议

1. 固定已训练 checkpoint、相同点云和评估过程，先统计困难类所有物体的逐部件预测点数、recall、precision 和 IoU；不进行参数更新。
2. 优先验证 bag、rocket 是否把整物体预测成最大部件。主审查发现 H 分数恰与“恒预测最大部件”的理论分数一致，这是强线索，但尚须实际预测确认，不能写成已经证实。
3. 保留 cap、mug 作为成功的小样本/少数部件对照，避免把所有尾部类别都归为困难类别。
4. 若整个少数部件消失，重点检查该部件 logits、正确类排名和监督梯度；若只在边界错分，才更偏向局部几何/邻域尺度问题。两者应分开报告。
5. 重采样增加整类物体的曝光，未必改变类内每个部件的点数比例；部件加权改变监督贡献，也不等于补足整类形状多样性。只有诊断后才选择对应对照，不能用一个“均衡”方案代替两种问题。

本文不主张立即改变 batch、损失或 backbone。当前第一优先级是解释：为什么相同数据下原 DGCNN 能分开的部件，在 H/H0 中是否发生整体消失。
