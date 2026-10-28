# HyCoRe-seg 论文复核：从官方分类模型改造分割的约束

核查日期：2026-10-28。范围为论文和源码的静态证据审查，没有改模型、连接服务器或新增训练。本文供总审查报告采用。

## 证据来源和级别

- **P：论文明确规定**：[作者接受稿 PDF](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)，DOI 10.1109/ICASSP49357.2023.10095253。PDF共6页，首页为学校封面；正文第2/3/4页分别是PDF第3/4/5页。
- **S：官方源码事实**：[HyCoRe分类DGCNN](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/models/dgcnn_hmodel.py)、[官方双曲层](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/models/ops/manifold_layers.py)。这些是分类实现，不能独自证明分割作者的逐层结构。
- **U：用户已确定**：以官方HyCoRe分类DGCNN为改造起点、尽量复用原部件；seed4780；层次间隔1000/N′、对比间隔4；此前选定的分割trainval/逐轮test Instance mIoU选best。均属于本项目决定。
- **D：工程推导**：为适配逐点任务、维度和数据所需的改动；须记录为复现选择。

已重新读取官方PDF全文文本，并与本地[中英转录](../../../资料库/5.Towards_Hyperbolic_Regularizers_For_Point_Cloud_Part_Segmentation%20-%20副本.md)对应段落核对。Fig.1说明文字可读，但网页PDF截图仍返回cache miss；本次不声称已完成图像独立核验。

## 直接约束及其对源码复用的影响

| 事项 | 结论和出处 | 对原分类模型的改造要求 |
|---|---|---|
| 编码器与双曲头边界 | **P**：§3.1将E定义为R^(N×3)→R^(N×F)，输出欧氏逐点特征；其后是exp和共享Möbius层。 | 分割必须暴露`[B,N,F]`，原分类最终单个`mu`不足以作为该接口。 |
| 双曲层数量 | **P**：§3.1允许一个或多个共享Möbius层；§4未列逐层宽度。 | 沿用官方`emb→mlr`两层有方法依据；两层及其512/256宽度仍是重建选择。 |
| 分类分数 | **P**：§3.1、§4给出双曲分割头和CE/softmax。**S**：官方分类forward直接返回`mlr(mu)`为logits。 | 可以优先沿用原分类读出约定；没有论文依据要求另加温度、logmap或超平面分类器。 |
| 正则入口 | **P**：§3.1、Fig.1说明与公式使用E的欧氏特征，先平均，再exp。 | whole/part应来自同一H；不能直接把官方分类`mu`当成论文定义的z。 |
| part选择 | **P**：§3.1及§4使用E特征中随机中心的近邻，N′在200–600随机选择。 | 原分类XYZ裁剪函数的调度接口可参考；其坐标近邻和重新编码逻辑须改。 |
| 正负关系 | **P**：§3.1正part与whole同物体，负part来自异物体且异物体类别。 | 负样本必须检查类别；直接翻转batch不保证合法。 |
| 径向次序 | **P**：§3.2分割需要whole浅、part深；分类方向相反。 | 几何距离运算可复用，分类层次项的符号必须适配。 |
| 损失与优化 | **P**：§3.1总损失为CE与两正则之和；§4两系数均.01，采用Riemannian SGD和Geoopt。 | 优先复用分类损失框架、优化器和几何库；不自动继承未披露的学习率等配方。 |

以上论文结论主要来自§3–§4；正文中关于part嵌入写成`zwhole`存在笔误，应结合上下文与公式按part理解。具体随机分布、每批/每对象共享N′、选几个part、负样本挖掘、reduction及选模频率未完整规定。

## 原码最大复用时需要显式处理的三个问题

### 1. 保留双曲层不等于保留全局分类接口

官方分类网络的全局max/avg和`linear1`产生每物体512维向量，随后`exp→emb(512,256)→mlr(256,40)`。逐点任务可保留这一全局分支作为上下文，并补上局部特征融合/广播，让E产生每点512维向量，再逐点执行原`emb/mlr`，只把输出类别调整到50。这个方向充分复用官方部件，符合论文通用接口，但新增融合层、四个EdgeConv的保留方式及512维宽度是**D**，并非论文公布的DGCNN分割精确结构。依据：[官方分类forward](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/models/dgcnn_hmodel.py)、[论文§3.1](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。

若采取上述F=512方案，正则几何也工作在512维。分类版`mu`为256维，原损失里固定的`ball_dim=256`必须参数化；不应为保留该常数而改用错误的正则入口。需要用张量形状和梯度测试锁定H、zwhole、zpart及logits之间的关系。

### 2. 参数不变与官方权重参数化要分别核验

论文§4声称相对原DGCNN分割模型总参数量保持不变，但没有给逐层参数表。官方`MobiusLayer`默认可训练参数包括`_weight[out,in]`及`_bias[out,1]`；后者经`expmap0`、`transp0`参与有效权重，forward最终执行`mobius_matvec`，没有普通输出加法偏置。本地行号：[manifold_layers.py:55](../../../classification_ModelNet40/models/ops/manifold_layers.py#L55)；对应[官方实现](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/models/ops/manifold_layers.py)。

审查建议（**D**）：保留原层完整语义，先列明逐层参数数目。512→256→50的两个官方默认层为`512×256+256+256×50+50=144178`个可训练标量，与具有普通偏置的两层欧氏Linear数量一致；相对无偏置层多306。能否满足论文的总量不变，取决于所选分割结构、偏置/BN和新增融合模块。不能删去`_bias`后仍声称逐项复用官方层，也不能把原分类模型与原分割基线的总量相等作为前提。

### 3. 复用源码路线与论文表格可比性需分开陈述

用户要求的是官方HyCoRe原生模型的分割适配，应优先使组件来源、参数化和训练继承关系可追踪。论文的表1以DGCNN分割为参照，并没有明确声称实验网络由分类版本完整四块encoder移植而来。若改造后编码器/融合/头宽不同，必须为该原生适配结构设置自己的匹配对照；既有AnTao分割基线仍可作为外部参照，其差值不能全部归因于HyCoRe-seg正则。

## 已确定配方与仍需声明的选择

margin原文存在两套表述：§3.1写两者1024/N′，§4写gamma1000、delta4。用户已确定执行**1000/N′与4**；此次审查不重开选项，不把该决定改写为论文唯一确定公式。

建议在修订计划中一次列明：E的截止层与F、全局上下文融合、Möbius层数及完整权重参数化、类别输入、CE是否平滑、梯度裁剪、采样/负例规则、exp尺度与投影、数据预处理、训练周期与学习率。H0和HC采用完全相同结构及训练协议，仅切换正则系数；不增加教师、代理、LCA或历史HIER目标。现有运行保留为旧路线结果，不借静态审查宣布性能归因。
