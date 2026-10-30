# 作者 DGCNN 与固定社区版：ShapeNetPart 分割核查

核查日期：2026-10-30。仅源码/文档阅读；没有模型运行、GPU、训练或结果修改。

## 来源与“官方版”的范围

作者仓库固定到 `WangYueFt/dgcnn@f765b469a67730658ba554e97dc11723a7bab628`；社区仓库固定到 `antao97/dgcnn.pytorch@07d534c2702905010ec9991619f552d8cacae45b`。两者SHA通过公开GitHub commits/master API只读核实。社区当前master恰好就是本地固定提交。

作者README把AnTao放在 Other Implementations 中，不能称为作者原始实现；但它确实是作者列出的社区实现。作者PyTorch目录的模型是分类网络，不能拿其四级分类EdgeConv当作“官方分割网络”。本次对应的作者ShapeNetPart实现是 `tensorflow/part_seg/part_seg_model.py`。[作者README](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/README.md)、[作者PyTorch模型](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py)、[作者分割入口](https://github.com/WangYueFt/dgcnn/tree/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg)。

本地来源记录为 [UPSTREAM.md](D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/vendor/UPSTREAM.md:3)：`dgcnn_model.py`来自该社区提交的`model.py`，记录的兼容修改是把硬编码cuda改为输入的device，并清理尾部空白；不是重写分割骨干。[固定社区模型](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/model.py#L214-L312)。

## 相同的大拓扑

两者均为以下分割框架，而不是分类网络的四级64/64/128/256。

```text
XYZ → 基于邻域边特征的Transform-Net → 3×3变换 → XYZ'
    → 动态建图 → 64 → 64 → 邻域max → x1:64
    → 对x1动态建图 → 64 → 64 → 邻域max → x2:64
    → 对x2动态建图 → 64 → 邻域max → x3:64
concat(x1,x2,x3):192 → 1024 → 对全点max → global1024
类别one-hot16 → 64 → category64
broadcast(global1024+category64) + x1+x2+x3 → 1280
    → 256 → dropout → 256 → dropout → 128 → 50
```

空间变换、三个动态邻域阶段、每阶段通道数、多尺度局部拼接、全局max和广播、类别输入、1280融合以及尾部宽度均对应。中间共享映射含BN与激活，最终50通道映射不含BN/激活。[作者分割模型](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/part_seg_model.py#L21-L119)、[社区分割模型](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/model.py#L214-L312)。

## 不能称为逐算子一致的细节

| 项目 | 作者TensorFlow分割 | 固定社区PyTorch分割 / 我们沿用的基线 |
|---|---|---|
| kNN大小 | 模型内固定20 | 入口默认40，本项目也40 |
| 边特征通道顺序 | `[中心, 邻居−中心]` | `[邻居−中心, 中心]` |
| 隐层激活 | ReLU | LeakyReLU，负斜率.2 |
| 两个尾部dropout | keep_prob=.6，即drop=.4 | drop=.5 |
| Conv bias | 封装始终创建bias，包括BN前Conv | 分割Conv均bias=False |
| Conv/普通FC初始化 | 封装默认Xavier，bias初始化0 | 使用PyTorch模块默认初始化；分割入口没有调用util中额外init函数 |
| Transform-Net最终3×3 | 权重0，bias含单位阵 | 同样权重0、bias单位阵 |
| BN训练/测试模式 | train用当前批统计并更新总体统计，test用总体统计 | train/eval标准BatchNorm切换，同样区分批统计和运行统计 |
| BN数值和统计更新 | epsilon1e−3；moving-average decay由训练入口调度 | 默认epsilon1e−5、momentum=.1，无BN调度 |

模型差异依据：[作者模型的k/dropout](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/part_seg_model.py#L21-L119)、[社区模型](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/model.py#L27-L312)。封装的初始化/bias/激活见[tf_util.py](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/utils/tf_util.py#L23-L170)，BN实现见[同文件L502–535](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/utils/tf_util.py#L502-L535)，边拼接见[同文件L675–706](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/utils/tf_util.py#L675-L706)。[Transform-Net](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/models/transform_nets.py)与[社区Transform-Net](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/model.py#L166-L211)。

边特征反序本身是可通过重排第一层输入通道补偿的排列，不代表多一个/少一个信息分支；但这不允许不做转换直接搬权重。k改变实际邻域，激活、dropout、bias与初始化、BN设置改变具体训练行为，不能压缩为纯粹Tensor维度记法差别。作者BN的`decay`是旧统计权重，而PyTorch`momentum`是新统计权重；不能把两个同名系数直接相等比较。作者从decay=.5开始，逐步增至最高.99（当前批权重从.5下降到不低于.01）。这来自[官方BN调度L145–151](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/train_multi_gpu.py#L145-L151)。

## 训练/评价协议另列，不能算作多改了骨干层

| 项目 | 作者TensorFlow分割 | 固定社区PyTorch分割 |
|---|---|---|
| 输入点数 | 训练2048；独立test.py为3000，补点后恢复原点数输出 | 默认2048 |
| 默认优化器/LR | Adam .003，按样本计数阶梯衰减，最低1e−5 | SGD实际.1（入口.001×100）、momentum.9/WD1e−4，cosine到.001 |
| batch默认 | 2 GPU×每GPU16 | DataParallel，总batch32 |
| CE | 普通稀疏softmax CE | `cal_loss(...,smoothing=False)`默认普通CE |
| BN/Dropout的训练与评价模式 | 显式is_training True/False | 显式model.train()/eval() |
| 独立测试的类别约束 | test.py屏蔽不属于真类别的part后argmax | 全数据模式直接50-way argmax |
| 保存/选模 | 每5轮保存；独立test.py指定checkpoint | 逐轮test Instance mIoU，`>=`时覆盖best |

依据：[作者训练入口L19–24/L54–65/L136–157](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/train_multi_gpu.py#L19-L65)、[作者测试L194–200](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/tensorflow/part_seg/test.py#L194-L200)、[社区训练入口L140–287](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/main_partseg.py#L140-L287)、[社区默认参数L366–394](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/main_partseg.py#L366-L394)、[社区CE L448–465](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/util.py#L448-L465)。

尤其不能把分类的label smoothing默认值套到社区分割上：固定分割提交默认确实是False；我们plain CE与此一致。社区ShapeNetPart训练数据只重排点和对应标签，scale/shift调用被注释；本项目v1data也仅重排，与社区保持。[社区数据L320–329](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/data.py#L320-L329)。作者代码的`is_dist`不应未经检查就描述为现代SyncBatchNorm；其源码仍在单个传入张量上求moments。

## 对本次问题的结论

社区版保留了作者分割网络的主干/融合/尾部宽度设计，不是另一个完全不同的分割架构。但官方TensorFlow与社区PyTorch在邻域、激活、dropout、BN、bias、初始化和训练/评价协议上存在实际差别。我们E相对“本项目固定社区基线”的修改范围，应由E继承链独立核对；不能因为社区与官方存在上述差别，就称E额外改动了这些项目。本次官方对照本身不要求改回TensorFlow或启动任何新实验。

取证说明：已获取固定提交官方model/util/train/test及社区train/util/data到研究tmp目录，只读分析。一个额外的“重新下载社区完整model并逐行归一化比较”命令被自动权限审批超时阻断，未执行；本笔记不把该未执行检查声称为逐字节验证，来源兼容范围以上游记录和已读本地源码为依据。
