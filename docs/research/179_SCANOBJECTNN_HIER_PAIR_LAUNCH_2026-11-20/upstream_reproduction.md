# HyCoRe ScanObjectNN 87.2% 的公开复现材料

2026-11-21 核对。NeurIPS 2022 论文 Table 2 的 ScanObjectNN 成绩是 PointMLP 86.1% OA／84.4% AA，PointMLP＋HyCoRe 87.2% OA／85.9% AA。本地 SO-H0 的 best OA 为 85.357%，与后者差 1.843 个百分点。论文成绩和本次运行的针对性比较见 [配对记录](run.md)。[论文 Table 2 与 §4.1](https://proceedings.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7)

## 作者是否公开了权重

在作者官方仓库已查到代码，尚未找到 Table 2 对应的 PointMLP＋HyCoRe 权重。README 的预训练章节只写 “Coming soon!”。GitHub API 返回 releases 和 tags 均为空，默认分支完整文件树共 265 项，未见 `weight`、`checkpoint` 或 `.pth`／`.pt`／`.ckpt` 文件。作者在 issue #1 表示后续会发链接，该帖没有发出权重链接。[官方 README](https://github.com/diegovalsesia/HyCoRe#pretrained-models) · [Releases API](https://api.github.com/repos/diegovalsesia/HyCoRe/releases) · [Tags API](https://api.github.com/repos/diegovalsesia/HyCoRe/tags) · [GitHub 文件树](https://api.github.com/repos/diegovalsesia/HyCoRe/git/trees/main?recursive=1) · [作者回复](https://github.com/diegovalsesia/HyCoRe/issues/1#issuecomment-1326490070)

PointMLP 作者的 Model Zoo 列有 **纯 PointMLP** 的 ScanObjectNN 非投票 86.1% OA／84.4% mAcc 和日志、预训练权重链接。该历史学校镜像在本次浏览请求中超时，下载可用性未证实。即使取得其权重，适合复评普通 PointMLP 的输入和评估流程。HyCoRe 论文的 87.2% 来自带双曲头与正则的另一模型。[PointMLP 官方 Model Zoo](https://github.com/ma-xu/pointMLP-pytorch#model-zoo) · [官方链接的 86.1% 权重目录](https://web.northeastern.edu/smilelab/xuma/pointMLP/checkpoints/fixstd/scanobjectnn/pointMLP-20220204021453/)

## 现有 issue 各自证实什么

| Issue | 内容和证据范围 |
| --- | --- |
| [HyCoRe #1](https://github.com/diegovalsesia/HyCoRe/issues/1) | 代码发布，权重链接预计稍后发布。 |
| [HyCoRe #2](https://github.com/diegovalsesia/HyCoRe/issues/2) | 运行 ScanObjectNN DGCNN 命令时，导入 PointMLP 模块遇到 `modelsnet` 不存在。作者确认是目录改名时的遗留。支持将命名空间问题单独看作运行入口兼容性。 |
| [HyCoRe #3](https://github.com/diegovalsesia/HyCoRe/issues/3#issuecomment-1364649053) | 报告者运行 PointMLP 命令遇到 Geoopt 的 `torch.tensordot` 参数错误，并自报使用 Geoopt 0.3 可运行。它是兼容性实测，作者在帖内没有确认论文运行版本或汇报 OA。 |
| [HyCoRe #4](https://github.com/diegovalsesia/HyCoRe/issues/4) | Fig. 4 的双曲 UMAP 可视化。 |
| [HyCoRe #5](https://github.com/diegovalsesia/HyCoRe/issues/5#issuecomment-1415509658) | 作者澄清公式(5)中的 `par_norm` 对应 whole，`p_norm` 对应 positive part。 |
| [HyCoRe #6](https://github.com/diegovalsesia/HyCoRe/issues/6) | 正则实现位于各任务目录的 `hutil.py`，函数 `hype_triplet_losses`。 |

官方 API 的全部状态列表返回以上六条，三条开启、三条关闭。未见针对 ScanObjectNN PointMLP＋HyCoRe 87.2% 的重跑得分、必要配置或修复。纯 PointMLP 仓库 #1 的重跑波动讨论主要针对 **ModelNet40**，不能用它解释 HyCoRe 在 ScanObjectNN 的这 1.843 个百分点。[全部 HyCoRe issues](https://api.github.com/repos/diegovalsesia/HyCoRe/issues?state=all&per_page=100) · [PointMLP #1](https://github.com/ma-xu/pointMLP-pytorch/issues/1)

README 推荐 Python 3.7、PyTorch 1.10.1／CUDA 10.2，另给 Ampere 用的 PyTorch 1.8.1／CUDA 11.1 示例。Geoopt 在 README 中未锁定。论文 §4.1 给出 A6000 与通用 HyCoRe 配方，但 Table 2 没有逐行训练命令、种子或重复运行统计。因此当前公开记录可作为复测依据，无法直接确定本地与 87.2% 差距归因。[官方安装说明](https://github.com/diegovalsesia/HyCoRe#installation) · [论文 §4.1](https://proceedings.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7)

已读 NeurIPS 补充材料全文，共三页，内容为测地线分析和嵌入可视化，未提供 ScanObjectNN Table 2 的训练命令、种子或权重。PointMLP 历史权重目录的响应头经重定向后为 HTML，尚未核验实际权重文件。[NeurIPS 补充材料](https://papers.nips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Supplemental-Conference.pdf)
