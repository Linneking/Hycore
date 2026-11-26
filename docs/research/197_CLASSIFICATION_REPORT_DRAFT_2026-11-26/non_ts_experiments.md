# 非 TS 正式训练与作者权重复评核验表

核验截止：**2026-11-26 14:03:39（北京时间）**；17 条历史非 TS 正式训练均已从服务器正式逐轮日志直接复算，均完整覆盖 300 或 250 轮。精简原始证据见 [日志核验摘要](non_ts_log_audit.json)，包含原日志相对路径、连续轮次、成绩、日志 SHA-256 和能取得的清单时间。未重新推理模型、未启动训练。194、196 的最新运行由 [最新运行核验](latest_experiments.md) 承接（若主稿改名，以主稿链接为准）。

## 口径和配置字典

- **MN40**：完整官方 train 9,840／test 2,468，40 类；**SO**：ScanObjectNN PB_T50_RS / main_split / 含背景，train 11,416／test 2,882，15 类。表内训练全部单卡 RTX 5090，从随机初值开始，无教师与续训检查点。
- **PMLP原生**：PointMLP＋HyCoRe，c=1，保留同对象 part先／whole后共享编码、原别名覆写和双前向BN。MN40 原生 margin 1000/Npart、4，两原生正则各.01；SO PMLP 原生 margin 500/Npart、4，权重.001、.01，whole600–1024，part200–floor((whole−1)/2)。SO DGCNN whole800–1024／part200–600，两正则各.01。
- **B05**：PointMLP＋HyCoRe 的改版主支 c=.1、切空间cap2.3、原生GS、margin500/Npart和2。固定壳HIER从whole方向生成深度4.6的副本，和“实际深度cap”不同。
- **实际深度cap**：从whole μ执行 `logmap0(c=1) → tangent cap2.3 → expmap0(c=.1) → 独立GS`。保留上限内深度变化；样本也可能自然全部触cap，不能把代码称为固定壳。已直接核对 [support.py](../diagnostics/scanobjectnn_hier_pair_20261120/support.py) 的 `native_whole_view` 和 [作者接入](../diagnostics/scanobjectnn_author_s12/entry.py)。
- **A1/K20/.5**：512×256代理、初始缩放.9，AdamW LR .005→.0005／eps1e−4，样本Ks=20／代理Kp=20、T50，两项λ各.5，W0。分类batch和HIER候选池分别列出，FIFO第一步池32，此后64；池增大也改变候选年龄。
- 以下17条均**逐轮完整官方test、OA严格上升取best、平局取早轮、AA取best同轮、固定预算无早停**。本地/改版日志按三位小数OA；作者入口保留原评估顺序和实际OA规则。DeLA原日志的“val”实际是test，另把e250作固定终点；它仍有逐轮test可读best，不能与新版DeLA只测e250混为一种协议。末10均值来自最后10个完整轮，不是10次独立训练。

## 正式训练（每组一行）

启动时间均为2026年北京时间。以下时间优先用worker运行清单，较计划/控制器记录可相差数秒；DeLA原版精确启动秒未找到。成绩为OA／AA（%）。

| 实验／原名 | 实际启动 | 数据／骨干底座 | seed；卡数 | 分类组批；HIER池 | 步×轮＝更新预算 | HIER开关／本组实际改动 | 状态 | best轮·OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 直接对照／可比性 | 证据／原目录 |
|---|---|---|---|---|---|---|---|---:|---:|---:|---|---|
| `SO-H0` | 11-20 01:20:40 | SO / PMLP原生，本地适配 | 4780；1 | SH32；池— | 356×300=106,800 | H0；原生Part–Whole | 完成e300 | e211·85.357／83.903 | 83.657／82.606 | 83.796／82.501 | SO-HIER；同入口配对 | [复算](non_ts_log_audit.json)；原记录179；`scanobject_hier_pair_20261120_A/SO-H0/main` |
| `SO-HIER` | 11-20 01:20:40 | SO / PMLP原生，本地适配 | 4780；1 | SH32；池32 | 356×300=106,800 | 实际深度cap分支；A1/K20/.5 | 完成e300 | e260·86.051／84.691 | 84.629／83.485 | 84.608／83.299 | SO-H0；同入口配对 | [复算](non_ts_log_audit.json)；原记录179；`scanobject_hier_pair_20261120_A/SO-HIER/main` |
| `SO-S12-H0` | 11-21 14:11:32 | SO / PMLP原生，作者main | 12；1 | SH32；池— | 356×300=106,800 | H0；原生Part–Whole | 完成e300 | e209·86.294／84.575 | 84.074／82.271 | 84.042／82.515 | SO-S12-HIER；同入口配对 | [复算](non_ts_log_audit.json)；原记录180/182；`scanobject_author_s12_20261121_B/SO-S12-H0/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore` |
| `SO-S12-HIER` | 11-21 14:11:17 | SO / PMLP原生，作者main | 12；1 | SH32；池32 | 356×300=106,800 | 实际深度cap；A1/K20/.5 | 完成e300 | e210·86.294／84.278 | 84.837／83.663 | 84.639／83.317 | SO-S12-H0；同入口配对 | [复算](non_ts_log_audit.json)；原记录180/182；`scanobject_author_s12_20261121_B/SO-S12-HIER/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore` |
| `MN40作者H0-S4780` | 11-21 19:31:41 | MN40 / PMLP原生，作者launcher | 4780；1 | SH32；池— | 307×300=92,100 | H0；训练分片3/0/2/4/1 | 完成e300 | e200·93.598／90.411 | 93.395／90.770 | 93.002／90.573 | 作者分片组；TS02还混入口 | [复算](non_ts_log_audit.json)；原记录181/182；`mn40_author_s4780_20261121_a/source/classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780` |
| `SO-S4780-H0` | 11-23 14:49:59 | SO / PMLP原生，作者main | 4780；1 | SH32；池— | 356×300=106,800 | H0；原生Part–Whole | 完成e300 | e253·85.600／84.575 | 84.629／83.411 | 84.733／83.476 | SO-S4780-HIER；同入口配对 | [复算](non_ts_log_audit.json)；原记录183；`scanobject_author_s4780_20261123/SO-S4780-H0/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore` |
| `SO-S4780-HIER` | 11-23 14:49:58 | SO / PMLP原生，作者main | 4780；1 | SH32；池32 | 356×300=106,800 | 实际深度cap；A1/K20/.5 | 完成e300 | e208·85.600／83.875 | 84.074／83.109 | 84.289／83.159 | SO-S4780-H0；同入口配对 | [复算](non_ts_log_audit.json)；原记录183；`scanobject_author_s4780_20261123/SO-S4780-HIER/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore` |
| `MN40-B05-S4781-H0` | 11-23 14:51:48 | MN40 / PMLP B05 | 4781；1 | SH32；池— | 307×300=92,100 | H0；沿TS06数据顺序 | 完成e300 | e282·93.882／90.853 | 93.801／90.816 | 93.254／90.507 | FIFO1/CURRENT32；后者跨主机 | [复算](non_ts_log_audit.json)；原记录183；`mn40_b05_fifo64_s4781_20261123/h0/main` |
| `MN40-B05-S4781-HIER-FIFO1` | 11-23 14:51:47 | MN40 / PMLP B05 | 4781；1 | SH32；池32→64 | 307×300=92,100 | 固定壳；前一步detach，轮首清空；A1/K20/.5 | 完成e300 | e243·93.801／90.783 | 93.355／90.545 | 93.039／90.561 | 同seed H0；复验TS44整套方案 | [复算](non_ts_log_audit.json)；原记录183；`mn40_b05_fifo64_s4781_20261123/hier/main` |
| `DG-S12-H0` | 11-24 01:55:30（启动记录） | SO / DGCNN原生，作者launcher | 12；1 | SH32；池— | 356×250=89,000 | H0；原生Part–Whole | 完成e250 | e248·81.992／79.429 | 80.916／77.900 | 81.607／78.876 | DG-S12-HIER；同卡不同时间 | [复算](non_ts_log_audit.json)；原记录185；`hycore_author_dgcnn_scanobjectnn_s12_20261124_fresh/classification_ScanObject/checkpoints/PointNet-toyproof-Offv_dgcnn_hycore` |
| `DeLA v1原版从头训练` | 11-24 01:50附近；精确秒缺 | SO / DeLA作者train.py | 4780；1 | SH32；池— | 356×250=89,000尝试 | 分类+坐标MSE；无显式HyCoRe Part–Whole | 完成e250 | e222·89.590／88.290 | 89.380／88.440 | 89.243／88.240 | 新DeLA-H0同时换头/加HLoss/评估流，不隔离单项 | [复算](non_ts_log_audit.json)；原记录185；`de_la_v1_author_train_scanobjectnn_s4780_20261124_0150/Matrix-ASC-DeLA-0f427c3/ScanObjectNN/output/log/01` |
| `DG-S12-HIER` | 11-24 14:14:22 | SO / DGCNN原生 | 12；1 | SH32；池32 | 356×250=89,000 | 实际深度cap；A1/K20/.5 | 完成e250 | e242·81.888／79.475 | 81.055／78.464 | 81.263／78.720 | DG-S12-H0；首步验收匹配 | [复算](non_ts_log_audit.json)；原记录188/195；`scanobject_dgcnn_hier_20261124_1415/DG-S12-HIER/main/checkpoints/PointNet-toyproof-Offv_dgcnn_hycore` |
| `DG-S4780-H0` | 11-24 14:14:24 | SO / DGCNN原生 | 4780；1 | SH32；池— | 356×250=89,000 | H0；原生Part–Whole | 完成e250 | e223·82.616／79.749 | 81.263／78.602 | 80.940／78.131 | DG-S4780-HIER；同seed配对 | [复算](non_ts_log_audit.json)；原记录188/195；`scanobject_dgcnn_hier_20261124_1415/DG-S4780-H0/main/checkpoints/PointNet-toyproof-Offv_dgcnn_hycore` |
| `DG-S4780-HIER` | 11-24 14:19:51 | SO / DGCNN原生 | 4780；1 | SH32；池32 | 356×250=89,000 | 实际深度cap；A1/K20/.5 | 完成e250 | e219·81.298／78.997 | 80.222／77.598 | 80.326／77.458 | DG-S4780-H0；同seed配对 | [复算](non_ts_log_audit.json)；原记录188/195；`scanobject_dgcnn_hier_20261124_1415/DG-S4780-HIER/main/checkpoints/PointNet-toyproof-Offv_dgcnn_hycore` |
| `MN40-B05-S4781-HIER-CURRENT32` | 11-24 21:38:30 | MN40 / PMLP B05 | 4781；1 | SH32；池32 | 307×300=92,100 | 固定壳；关FIFO；A1/K20/.5 | 完成e300 | e282·94.084／91.070 | 93.841／90.565 | 93.371／90.569 | 同seed H0/FIFO1；主机不同 | [复算](non_ts_log_audit.json)；原记录189；`mn40_b05_s4781_current32_20261124/hier-current/main` |
| `SO-S12-H0-PAPER-INTRA` | 11-24 21:50:08 | SO / PMLP作者main | 12；1 | SH32；池— | 356×300=106,800 | H0；原生层次权重.001→.01和margin500/N→1000/N一起改 | 完成e300 | e243·85.010／83.664 | 83.727／82.843 | 83.664／82.761 | SO-S12-H0；跨主机；只回答两项合改 | [复算](non_ts_log_audit.json)；原记录190/195；`scanobject_author_paper_intra_s12_20261124/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore` |
| `MN40作者H0训练分片顺序组` | 11-24 21:50:10 | MN40 / PMLP原生，作者launcher | 4780；1 | SH32；池— | 307×300=92,100 | H0；训练分片改1/4/3/2/0，测试仍0/1 | 完成e300 | e182·93.679／89.369 | 93.112／90.553 | 92.970／90.836 | 原作者H0；另有主机/Python补丁差异 | [复算](non_ts_log_audit.json)；原记录190；`mn40_author_train_order_s4780_20261124/source/classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780` |

## 主要匹配差值

下列全部为“开启或改动组 − 指定对照”的百分点。OA差值按固定测试数从日志OA还原最近整数正确数后计算，末10还原十轮正确数总和；因此个别差值与表中三位读数相减差.001。AA按原日志计算；best AA取各组best OA所在轮，它不等于独立best AA。

| 配对 | Δbest OA／AA | Δ末轮 OA／AA | Δ末10 OA／AA | 可回答的问题 |
|---|---:|---:|---:|---|
| SO-HIER − SO-H0 | +0.694／+0.788 | +0.972／+0.879 | +0.812／+0.798 | 本地入口seed4780，当前批实际深度HIER净增量 |
| SO-S12-HIER − SO-S12-H0 | +0.000／-0.297 | +0.763／+1.392 | +0.597／+0.802 | 作者入口seed12的同配方增量 |
| SO-S4780-HIER − SO-S4780-H0 | +0.000／-0.700 | -0.555／-0.302 | -0.444／-0.318 | 同作者入口第二seed；末段方向反转 |
| MN40-S4781-FIFO1 − H0 | -0.081／-0.070 | -0.446／-0.271 | -0.215／+0.054 | HIER＋一步缓存整体；复验历史TS06→44 |
| MN40-S4781-CURRENT32 − H0 | +0.203／+0.217 | +0.041／-0.251 | +0.118／+0.062 | 当前批HIER；主机不同仍有残余混杂 |
| MN40-S4781-FIFO1 − CURRENT32 | -0.284／-0.287 | -0.486／-0.020 | -0.332／-0.008 | 固定HIER下加一步缓存；主机不同 |
| DG-S12-HIER − DG-S12-H0 | -0.104／+0.046 | +0.139／+0.564 | -0.344／-0.156 | DGCNN第一个seed；末10下降，e250略升 |
| DG-S4780-HIER − DG-S4780-H0 | -1.318／-0.752 | -1.041／-1.004 | -0.614／-0.673 | DGCNN第二seed；末10再次下降 |
| SO-S12-PAPER-INTRA − SO-S12-H0 | -1.284／-0.911 | -0.347／+0.572 | -0.378／+0.246 | 原生层次权重与margin合改；跨主机 |
| MN40分片顺序组 − 原作者H0 | +0.081／-1.042 | -0.284／-0.217 | -0.032／+0.262 | 训练顺序改动；跨主机，未恢复TS02成绩 |

末10精确差值：S4781 CURRENT32−H0 为 +.1175 OA／+.0620 AA；FIFO1−CURRENT32 为 −.3323 OA／−.0083 AA。作者分片顺序组−原作者为 −.0324 OA／+.2623 AA。末10用原日志三位小数逐轮复算，不以已四舍五入的表格均值相减。

ScanObjectNN作者PointMLP两颗seed的末段OA方向相反（+.5969、−.4442）；DGCNN两颗seed末段均负（−.3435、−.6143）。完整MN40 FIFO1的seed4780历史末10OA增量+.389，seed4781为−.2148；而“FIFO1相对当前批”的方向在seed4780为正（TS35→44），seed4781为负。因此不能将“加HIER”“加缓存”和“跨seed复验成功”合并表述。

## 作者发布权重：实际复评与缺口

**HyCoRe作者发布权重：现有记录未查到可用、可核实的论文对应发布checkpoint，本项目没有该权重的实际复评。** 179／182当时的公开资产审查记载预训练小节占位、release/tag与问题记录未提供对应资产；本次直接核验的是现有运行，不将该历史网页检查冒称11/26重新联网核验。TS01是我们历史复现权重，A3是其alpha=0续训派生权重，二者均不能代替“作者发布权重”。本次作者入口结果是随机初值从头训练。[公开证据审查](../179_SCANOBJECTNN_HIER_PAIR_LAUNCH_2026-11-20/upstream_reproduction.md) · [复现审查](../182_MN40_V10_REPRO_AND_HIER_AUDIT_2026-11-23/report.md)

以下是**其他骨干**的作者已训练权重单次无投票复评。11/26 14:05直接读取原eval日志，与184/195记载一致。不存在本地训练预算、best／末轮／末10，不能将单次复评成绩填作训练best。每个推理条件一行：

| 实验／实际复评时间 | 数据／骨干及权重身份 | 推理seed；卡数；batch／池 | 本地训练预算／HIER | 实测OA／AA（%） | best／末轮／末10 | 对照／可比性／原目录与证据 |
|---|---|---|---|---|---|---|
| DeLA作者权重-MN40，11-24 01:06–01:15 | MN40 test2468；DeLA v1 commit0f427c3；权重SHA36019c4e… | 作者FPS；1；B32／— | 不训练；HIER无 | 94.00／92.25 | 不适用 | 作者日志94.00／92.25一致；低于TS06历史94.327，但不是同模型匹配；`de_la_v1_author_checkpoint_20261124_0106/modelnet40_eval.log` |
| DeLA作者权重-SO，11-24 01:06–01:15 | SO test2882；同源码；权重SHA0db19dcf… | 作者FPS；1；B32／— | 不训练；HIER无 | 90.42／89.24 | 不适用 | 作者日志90.46／89.30，OA约少1个正确样本；`de_la_v1_author_checkpoint_20261124_0106/scanobjectnn_eval.log` |
| point2vec-S0-FP16，11-24 01:31–01:33 | MN40 test2468；ShapeNet预训练后MN40微调e125；checkpoint SHA6c1d4884… | FPS0；1；B32／— | 不训练；HIER无 | 94.286872／91.903488（2327正确） | 不适用 | 预先主条件；`point2vec_author_checkpoint_20261124_0127/eval.log` |
| point2vec-S0-FP32，11-24 01:31–01:33 | 同一微调权重；精度对照 | FPS0；1；B32／— | 不训练；HIER无 | 94.286872／91.903488（2327） | 不适用 | 与FP16一致；`fps_sensitivity.log`同目录 |
| point2vec-S1-FP16，11-24 01:31–01:33 | 同一微调权重 | FPS1；1；B32／— | 不训练；HIER无 | 94.124797／92.011628（2323） | 不适用 | 只换推理FPS；`fps_sensitivity.log` |
| point2vec-S12-FP16，11-24 01:31–01:33 | 同一微调权重 | FPS12；1；B32／— | 不训练；HIER无 | 94.367909／91.861047（2329） | 不适用 | 只换推理FPS；`fps_sensitivity.log` |
| point2vec-S4780-FP16，11-24 01:31–01:33 | 同一微调权重 | FPS4780；1；B32／— | 不训练；HIER无 | 94.327391／91.844767（2328） | 不适用 | 与TS06正确数相同，不能称训练seed复验；`fps_sensitivity.log` |

DeLA复评原作者`test.py`、原FPS、FP16、单pass；point2vec用锁定作者tokenizer、Transformer、原FPS/kNN算子，先FPS1024 XYZ后居中/单位球归一化，64组×32点，封装绕过Lightning以适配运行时。完整SHA、源码、日志身份见 [184权重复评](../184_V9_V10_BASELINE_AND_HIER_ROADMAP_2026-11-24/report.md)。四个FP16条件跨度.2431个百分点（净6样本），只说明同一权重推理取点敏感性；不估计训练方差。

## 本轮缺口与记录冲突

1. 190截止11/24、195截止11/25的“运行中”均为历史快照；本表17条已全部终局，包括189 CURRENT32和MN40分片组。194／196以最新表为准。
2. 180、183的控制器“正式开始”与worker清单有2–4秒差异。表采用worker `manifest.started_at`；DG-S12-H0只有185的01:55:30记录，DeLA原版缺精确启动秒，目录0150和01:57已完成64轮只能支持启动窗口，不能把目录名当精确时间。
3. 17条的连续轮次和日志成绩本轮直接复算；权重SHA一致性沿用原终局验收记录。本轮未再次读取全部大权重，未宣称重新完成参数级或逐样本预测核验。
4. DeLA原版89,000是计划batch/尝试步数，原FP16可发生跳步，原版实际有效更新总数仍缺；新版194则应报实测跳步。这影响原版与新版的预算可比性。
5. 跨主机组（MN40 CURRENT32、MN40顺序、SO论文intra）已核对同型号GPU、库和数据身份，仍不能消除主机及低层数值轨迹差异；顺序组另有Python补丁3.9.23→3.9.25。
6. 公开候选PointNeXt、SinPoint、KPConv、PointRWKV、PointGPT、GPr-Net等未见本项目完整分类实测，不纳入正式训练总表。GPr-Net CPU载入和作者episode日志、DeLA两批接入短测均只属可运行性证据；ShapeNetPart分割也不混入分类结果。
