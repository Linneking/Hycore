# HyCoRe-seg 公开源码检索

核查日期：2026-10-27（Asia/Shanghai）。范围：公开论文、作者机构页面、作者 GitHub、官方仓库及其公开分支、标签、release、issue 和列出的 forks。

## 结论

本次没有找到可核验的 **HyCoRe-seg 完整开源实现**。用户给出的 [diegovalsesia/HyCoRe](https://github.com/diegovalsesia/HyCoRe) 是 NeurIPS 2022 **点云分类**论文的官方实现；README 明确限定 PointMLP/DGCNN、ModelNet40/ScanObjectNN，公开主树没有 ShapeNetPart 分割训练入口。它可提供双曲运算参考，但不能当作 ICASSP 2023 分割方法的源码。

该结论表示“在下列已查公开范围内未找到”，不能证明所有未索引、私人或改名仓库均不存在。没有向作者发消息或新建 issue。

## 论文身份

目标论文为 Antonio Montanaro、Diego Valsesia、Enrico Magli 的 *Towards Hyperbolic Regularizers For Point Cloud Part Segmentation*，ICASSP 2023，DOI [10.1109/ICASSP49357.2023.10095253](https://doi.org/10.1109/ICASSP49357.2023.10095253)。作者机构公开记录提供 [Author's Accepted Manuscript](https://iris.polito.it/retrieve/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。来源：[Politecnico di Torino 论文记录](https://iris.polito.it/handle/11583/2982650)。

IEEE Resource Center 另有 [同题报告视频](https://resourcecenter.ieee.org/conferences/icassp-2023/spsicassp23vid1482)，其 `10.17023/ccjc-0v65` 是视频资源标识，不应混作论文 DOI。

## 可复查证据

| 已核查对象 | 结果 | 一手链接 |
|---|---|---|
| 官方 README 与主树 | 标注 NeurIPS 2022 分类论文；顶层为 `classification_ModelNet40`、`classification_ScanObject`、`pointnet2_ops_lib` 等 | [固定提交树](https://github.com/diegovalsesia/HyCoRe/tree/ef56a73020e309b1009c486fea1f601d3f7b3252)、[README](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/README.md) |
| 官方公开 refs | 只读 Git `ls-remote --heads --tags` 实查只有 `main`，无公开 tag；主提交 `ef56a73020e309b1009c486fea1f601d3f7b3252` | [仓库](https://github.com/diegovalsesia/HyCoRe)、[branches](https://github.com/diegovalsesia/HyCoRe/branches)、[tags](https://github.com/diegovalsesia/HyCoRe/tags) |
| 官方 releases | 页面显示没有发布 release | [releases](https://github.com/diegovalsesia/HyCoRe/releases) |
| 第一作者账号 | 公开列表中 HyCoRe 是上述分类仓库的 fork；没有发现独立 HyCoRe-seg 项目 | [AntonioMontanaro96](https://github.com/AntonioMontanaro96)、[repositories](https://github.com/AntonioMontanaro96?tab=repositories) |
| 公开 fork 网络 | 列出 `AntonioMontanaro96/HyCoRe`、`miguelgondu/HyCoRe`、`whuhxb/HyCoRe`；三个均用只读 Git 实查 refs，只有同一 `main` SHA，与官方一致 | [fork 列表（含不活跃项）](https://github.com/diegovalsesia/HyCoRe/forks?include=active%2Cinactive%2Carchived&page=1&period=5y&sort_by=last_updated)、[作者 fork](https://github.com/AntonioMontanaro96/HyCoRe)、[miguelgondu fork](https://github.com/miguelgondu/HyCoRe)、[whuhxb fork](https://github.com/whuhxb/HyCoRe) |
| 官方所有可见 issue #1–#6 | 内容涉及分类代码公开、导入/Geoopt、可视化和损失；未见分割源码发布链接 | [#1](https://github.com/diegovalsesia/HyCoRe/issues/1)、[#2](https://github.com/diegovalsesia/HyCoRe/issues/2)、[#3](https://github.com/diegovalsesia/HyCoRe/issues/3)、[#4](https://github.com/diegovalsesia/HyCoRe/issues/4)、[#5](https://github.com/diegovalsesia/HyCoRe/issues/5)、[#6](https://github.com/diegovalsesia/HyCoRe/issues/6) |

特别注意 [issue #6](https://github.com/diegovalsesia/HyCoRe/issues/6)：第一作者 2024-06-25 指向两个分类目录的 `hutil.py` 中 `hype_triplet_losses`。这是分类正则的位置说明，不是 HyCoRe-seg 已开源的证据。[issue #5](https://github.com/diegovalsesia/HyCoRe/issues/5) 的作者回复也在解释分类函数中 whole/part 变量命名。

## 检索词与限制

已用论文完整题名、`HyCoRe-seg github`、`HyCoRe-seg code`、题名加 GitHub、官方仓库加 segmentation，以及作者账号进行检索。搜索命中主要为原论文、机构记录、分类仓库和引用该论文的其他工作，没有定位到可以核验为本方法实现的第三方仓库。HPCS/HIER 等其他双曲点云方法不能仅因使用相近术语就当作此论文实现。

检索时公开 GitHub REST API 出现未认证限流，部分浏览抓取不可用；上述 SHA、公开 branches/tags 使用成功的只读 Git 远程枚举复核，issue/fork/release 也通过公开页面复查。搜索引擎无法穷尽全站代码，因此保持有限范围的否定表述。

## 可复用的工程参考

可以复用分类 HyCoRe 的几何层作为实现参考；分割数据流、特征空间 part 构建、方向相反的 whole/part 径向目标与评估仍须按目标论文重建。另一个 DGCNN 分割工程参考为 [antao97/dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch/tree/07d534c2702905010ec9991619f552d8cacae45b)，2026-10-27 只读远程核查其 `master` 为 `07d534c2702905010ec9991619f552d8cacae45b`；它本身不是 HyCoRe-seg 实现。
