# 论文基准核验表

核验日期：2026-11-27。只列本次分类汇报需要的参考，OA／AA单位为%。已核对HyCoRe正式论文第7页原表、DeLA论文第9页原表及其作者README；point2vec按当前作者PDF原表核验。结构化读数见 [paper_baselines.json](paper_baselines.json)。

| 数据／骨干 | 论文原骨干 OA／AA | 论文骨干＋HyCoRe OA／AA | 评估口径 | 原始出处 |
|---|---:|---:|---|---|
| 完整MN40／PointMLP | **93.4／91.2** | **94.3／91.7** | 1024点；单次；原骨干行带双星号，为HyCoRe作者重实现且未精确复现PointMLP原参考 | [HyCoRe Table 1，第7页](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7) |
| 完整MN40／PointMLP，投票 | 93.7／91.4 | 94.5／91.9 | 1024点；voting另列，不与本地单次成绩混用；投票次数未在本轮核实 | [HyCoRe Table 1，第7页](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7) |
| ScanObjectNN PB_T50_RS／PointMLP | **86.1／84.4** | **87.2／85.9** | 1024点；表中无voting标记；同任务单次参照 | [HyCoRe Table 2，第7页](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7) |
| ScanObjectNN PB_T50_RS／DGCNN | **80.3／77.8** | **82.1／80.2** | 1024点；表中无voting标记；勿用另一论文改配方的86.1替换80.3 | [HyCoRe Table 2，第7页](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7) |
| ScanObjectNN PB_T50_RS／DeLA v1 | **90.4／89.3** | — | 无投票；主结果best run；论文表未单列输入点数，本地/冻结作者入口1024点 | [DeLA Table 6，第9页及§4.2.3](https://arxiv.org/pdf/2308.16532#page=9) |
| 完整MN40／DeLA v1 | **94.0／92.2** | — | 无投票；主结果best run；点数说明同上 | [DeLA Table 7，第9页及§4.2.3](https://arxiv.org/pdf/2308.16532#page=9) |
| 完整MN40／point2vec | **94.7／92.0** | — | ShapeNet预训练后MN40微调；1024点；无投票；投票OA94.8另列 | [当前作者PDF Table 1](https://karimknaebel.com/point2vec.pdf) |
| 自定义MN20／PointMLP | **没有同划分论文基准** | **没有同划分论文基准** | 依训练类频130–400筛出20类；仅按TS51/54等同子集H0比较 | [实际子集清单](../175_MN40_FILTER20_DUAL_SAMPLER_PLAN_2026-11-19/subset_manifest.json) |

以上HyCoRe原骨干与＋HyCoRe是论文参考，不能当作本项目重新训练的HIER开关组。MN40主表93.4不能用其他论文表中的PointMLP 94.1替换；SO两个骨干同样按HyCoRe原表各自读数对应。训练seed与论文对应checkpoint仍缺。

## DeLA：四类读数分开讲

| 来源与统计量 | SO OA／AA | MN40 OA／AA | 用途 |
|---|---:|---:|---|
| 论文best run，无投票 | 90.4／89.3 | 94.0／92.2 | 原始文献参考 |
| 官方README三次随机训练mean±SD | 90.15±.25／89.01±.39 | 93.75±.18／91.15±.61 | 作者重复训练统计，不是论文主表单次读数 |
| 作者已发布权重，本项目单次复评 | 90.42／89.24 | 94.00／92.25 | 检查发布权重与当前测试链路是否相容 |
| 本项目原版DeLA从头训练，seed4780 | best89.59／88.29；e25089.38／88.44；末10均值89.243／88.240 | 本阶段未从头重训 | 按真实训练口径比较，不能把权重复评算成从头训练 |

三seed数值出自 [冻结DeLA作者README](https://raw.githubusercontent.com/Matrix-ASC/DeLA/0f427c36c5a991a28397ffdaaf8c205b80d6b30d/readme.md)，而非论文主表；README还提示公开代码相对训练发布模型时已有小幅修改。本地复评及训练读数出自 [197非TS核验表](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md)。原版DeLA分类＋坐标辅助项，与本项目新DeLA双曲头＋Part–Whole HLoss不同；新HIER净增量必须对同一新H0计算，不能只与原版89.38交叉归因。

## point2vec：同一权重的推理采样参照

本地同一作者e125微调权重、1024XYZ、单次无投票：FPS seed0/1/12/4780 的FP16 OA分别为 **94.287／94.125／94.368／94.327**；AA分别 **91.903／92.012／91.861／91.845**。seed0另测FP32得到相同OA/AA。这是推理FPS起点敏感性，不是四颗训练seed或四次从头训练。[本地原日志核验摘要与协议](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md)

旧184记“作者94.65”，当前作者PDF Table1列94.7，不能把94.65继续标成当前论文表原数。它可能来自更细精度的发布资产，但本轮未重新核明其确切出处与版本，暂保留来源精度未统一的缺口。[旧184记录](../184_V9_V10_BASELINE_AND_HIER_ROADMAP_2026-11-24/report.md)

## 会改变汇报含义的口径提醒

- **H0不是纯骨干。** 本项目HyCoRe H0已保留同对象Part–Whole；额外HIER开关回答跨对象目标的增量。原骨干论文行只作旁边的背景参照。
- 完整MN40的**TS06 94.327%**是改版B05的本地历史目标。自定义MN20、SO与其他骨干使用各自同协议H0；不能把94.327套到所有实验。
- HyCoRe现有资料未提供已核实的论文对应发布权重实评。TS01是本地历史训练，A3是派生续训，两者均不补上这个缺口。
- 论文best run、发布权重单次复评、历史逐轮test-best、末10平均与新DeLA固定e250要分别标注。论文值未披露对应训练seed，单次差距不能拆给某一个未隔离因素。
