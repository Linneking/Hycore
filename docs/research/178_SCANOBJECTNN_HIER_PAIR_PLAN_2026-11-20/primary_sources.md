# ScanObjectNN候选：论文、官方数据说明与源码的一手核对

2026-11-20。只读研究；没有连接服务器、下载数据、初始化GPU或运行训练。本文区分正式论文成绩、发布代码实际行为和待验收的数据规模，供两组对照计划使用。

## 1. 建议的数据变体与骨干

建议用 **ScanObjectNN / PB_T50_RS / main_split / 含背景**，优先沿用 **PointMLP＋HyCoRe** 做H0，再仅加入HIER。理由是官方HyCoRe已有对应发布入口、与当前MN40主线骨干一致，而且有较强的论文成绩参照；这不预设HIER在新数据上会提高。官方仓库明确提供PointMLP与DGCNN在两种分类数据集上的实现，README也给出ScanObjectNN的DGCNN运行例子。[HyCoRe官方仓库](https://github.com/diegovalsesia/HyCoRe)

HyCoRe正式主文没有直接写PB_T50_RS名称。变体身份由**发布代码和数据作者的文件命名表交叉确定**：本地固定官方源码`ef56a73020e309b1009c486fea1f601d3f7b3252`的`classification_ScanObject/ScanObjectNN.py:37`读取`data/h5_files/main_split/{training,test}_objectdataset_augmentedrot_scale75.h5`；ScanObjectNN官方README将该后缀明确映射到PB_T50_RS。官方数据作者规定main_split对应主文实验，split1–4为额外划分；`_nobg`是移除背景的另一身份，不混用。[HyCoRe数据入口](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ScanObject/ScanObjectNN.py#L31)、[ScanObjectNN官方命名说明](https://github.com/hkust-vgd/scanobjectnn#h5-files)

## 2. HyCoRe正式论文的准确成绩

| ScanObjectNN，Table2 | AA（%） | OA（%） |
| --- | ---: | ---: |
| DGCNN | 77.8 | 80.3 |
| DGCNN＋HyCoRe | 80.2 | 82.1 |
| PointMLP | 84.4 | 86.1 |
| PointMLP＋HyCoRe | 85.9 | 87.2 |
| PointNeXt＋HyCoRe | 87.0 | 88.3 |

以上来自NeurIPS2022正式主文Table2；f默认256、输入1024点，PointNeXt虽在论文有结果，官方仓库README列出的实现是PointMLP和DGCNN。论文没有给每个表格运行的完整manifest或随机种子误差条，不能把87.2视作我们新环境单seed的保证；它也不是“加入HIER”的结果。[HyCoRe主文Table2与§4.1](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)

## 3. 规模与类别平衡：能证实到哪一层

官方站点说明15类、2,902个独立原始实例，并将包含扰动版本的整体规模概括为约15,000。[ScanObjectNN作者站点](https://hkust-vgd.github.io/scanobjectnn/)

数据论文Table2给出全体原始实例数：

| 类别 | 数量 | 类别 | 数量 | 类别 | 数量 |
| --- | ---: | --- | ---: | --- | ---: |
| bag | 78 | bed | 135 | bin | 201 |
| box | 127 | cabinet | 347 | chair | 395 |
| desk | 149 | display | 181 | door | 221 |
| pillow | 105 | shelf | 267 | sink | 118 |
| sofa | 254 | table | 242 | toilet | 82 |

论文§3.2描述每个原始对象生成5个扰动样本、最多14,510，随后过滤无效样本；§4.2按不同场景划分约80%训练／20%测试。Table2的最大最小比395/78≈5.06，说明原始集合并非严格类均衡。**这不是PB_T50_RS的训练类计数**。[ScanObjectNN原论文§3.2、Table2、§4.2](https://arxiv.org/pdf/1908.04616)

本轮没有从上述一手文档找到 **11,416训练／2,882测试** 的逐项明示，也没有实际HDF5可读验收。可以把它们列为常用文件的预期规模，不能写成已核验事实；不能由2,902×80%×5推导精确行数。训练前应从已获授权且身份明确的两份HDF5核对`data.shape`、`label.shape`、15类计数、浮点类型、背景变体和SHA256。若实测Ntrain=11,416，源码SH32/drop_last下为356步/轮、300轮106,800更新；最终预算应以实测值为准。

## 4. 发布代码与MN40配方的区别

本轮直接用`git show ef56a730...:<path>`核对官方冻结源码，非依据后来的V9/V10入口猜测。

- ScanObject PointMLP launcher为300轮、LR .1、workers8；未传seed，parser默认为12。如果本次成对改为4780，应明确是共同覆盖seed，而非声称逐字运行默认launcher。[launcher](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ScanObject/launcher_pointmlp.sh)
- 训练DataLoader为shuffle=True/drop_last=True；测试DataLoader也是shuffle=True、drop_last=False，并使用相同`args.batch_size`。两臂须共同保留实际选择，额外HIER随机数不能消费主流。[训练入口](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ScanObject/main_pointmlp_hycore.py#L134)
- 实际ScanObject whole采样请求600–1024，part请求200至`n_points//2`；辅助函数有`k-1`返回细节。基础总损失是`CE + .01×Rcontr + .001×Rhier`，层次margin乘`.5`，对比margin4。这些代码与论文§4.1的一般描述、MN40入口均有区别；仅加HIER时不应默默换成MN40的crop/系数。[源码训练段](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ScanObject/main_pointmlp_hycore.py#L214)
- 数据读取后按前三维XYZ训练：每对象原HDF5为2048点、源码先取前1024；训练逐轴缩放、平移和点序shuffle。PB_T50_RS的离线扰动与这些在线增强不是同一步。[数据入口](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ScanObject/ScanObjectNN.py#L49)、[官方HDF5说明](https://github.com/hkust-vgd/scanobjectnn#h5-files)

## 5. 对计划有影响的判断点

- **骨干**：两组资源优先PointMLP，DGCNN可作为后续另一骨干；两者的论文OA不同，不按低起点预设谁更适合HIER。
- **batch**：保持源码SH32可以满足“唯一新增HIER”。15类不能原样采用MN40的16个不同类×2例CB32；若改CB，类别数／每类例数／总batch须另定，两臂都改，且不再是字面默认源码基线。
- **几何**：源码HyCoRe的c1和基础损失应保留；HIER接入的曲率、cap/GS与是否固定壳是新方法的选择，不能把B05已有cap/GS说成源码自带。由主计划明确。
- **评估**：由用户确认新任务是沿源码逐轮test选模，还是另设验证。若另设验证，需按原始对象／场景分组，不能把同一原始对象的扰动版本随机拆到train/val；官方场景划分理由支持这一要求，但实际文件需先验收元数据。
- **结果解释**：真实扫描含背景、遮挡与不完整形状，这是新域的测试；只换数据集不是“单独验证类不平衡原因”。OA与15类AA均应报告，单seed只回答这一配方的初步增量。

官方数据站点目前要求经Terms of Use取得下载链接；本轮未提交表单或取得数据。源码中的镜像地址不等于已验收的数据身份。[官方获取说明](https://hkust-vgd.github.io/scanobjectnn/)
