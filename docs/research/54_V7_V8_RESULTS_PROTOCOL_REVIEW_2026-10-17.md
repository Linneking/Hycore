# V7—V8实验结果、划分与原始HyCoRe复测

日期：2026-10-17。本次完成五份预先固定best权重的服务器复测；OA/AA均与保存记录一致，零backward、零optimizer更新、模型/BN和源权重保持不变。

[完整图文报告](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/report.html) · [Markdown](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/report.md) · [CSV成绩表](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/tables/selected_results.csv)

## 结果先看这张表

| 模型 | best轮次 | val OA | val AA | test OA | test AA |
|---|---:|---:|---:|---:|---:|
| 原始HyCoRe：ORIG-B0-32-S4780 | 原元数据229 | 无独立val | 无独立val | **94.0438%** | 91.5448% |
| V7-HIER64-K20-W0 | 187 | 93.8008% | 90.7654% | 92.2204% | 89.7204% |
| V8-HIER64-K20-W0 | 221 | 94.2073% | 91.5457% | 92.7877% | 90.0372% |
| V8-B64 | 132 | 94.3089% | 91.5306% | **92.9092%** | **90.3157%** |
| V8-B0-32-C01 | 249 | **94.5122%** | **91.8165%** | 92.6256% | 89.8128% |

原版轮次使用原日志0起编号；现代四组为1起completed epoch。“221轮原版”经实际文件核对需要更正：e221属于V8-HIER主模型，原始seed4780为epoch元数据229。表中所有现代test均属于所列best，不是e300。

论文参照另行区分：PointMLP+HyCoRe无voting为test OA94.3%/AA91.7%，有voting为94.5%/91.9%，见[原论文第7页表1](https://proceedings.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf#page=7)。它们不是本地seed4780权重的成绩，本轮不做voting。

## shuffle还原程度与数据划分

**V8-B0-32-C01还原的是采样类型及关键输入算子，不是完整原版训练/选模协议。** 原版用9840个官方HDF5 train对象训练，没有独立验证集，每轮对2468个test前向并按test选best。V7及三组V8从相同9840内分层留出984作验证，训练8856；它们共享同一划分SHA和原官方2468 test，验证选模后才在训练结束评估test。

| 项目 | 原始HyCoRe | V8-B0-32-C01 |
|---|---|---|
| batch与顺序 | 32、shuffle、无放回、drop_last | 同类采样；显式独立epoch随机种子 |
| 每轮更新 | 307，丢16对象 | 276，丢24对象 |
| 300轮名义模型更新 | 92100 | 82800 |
| 曲率/cap/hook | c=1，无V8额外cap/hook | c≈.1、最终cap2.3、一次hook |
| intra margin | 1000/N，4 | 500/N，2 |
| 每轮前向评估的数据 | test | 独立val |
| test是否前向 | 每轮进行 | 结束后对val-best进行 |
| test是否反向更新 | 否 | 否 |

原版在现代984子集上的诊断OA98.9837%/AA98.7055%，但这些对象已参加原版训练，**不能记作独立val**。本轮又按用户要求对固定best各复测一次，不能把“训练运行结束时一次test”误写成权重整个历史只接触过一次test。

V7只跑过一个正式完整主训练，没有另训V7-B64或V7-B0。历史参考实际为V6-B64：e231 val OA94.3089%/AA91.8307%，对应test OA91.8558%/AA89.9994%，继承V6的20轮共同前缀；不是V7严格同协议消融。

## 从V7的问题到V8的改进

V7已通过更新后参数投影避免代理进入数值映射饱和区，但末期仍有245个代理当轮不被sample选作祖先，NN4仅覆盖27个对象。V8因此转向“让HyCoRe适配官方HIER几何”：c=.1、共享表示/代理forward cap2.3、官方hook与距离算子、官方高斯初始化，取消V7的post-step参数投影；同时将HyCoRe margin改为500/N和2，HIER权重.1→.5。相同split、seed22、K20/T50、D256/P512、global64、300×200步等保留。

需澄清：V7内部模型、intra、HIER均c=1，不能说已经查明同一次训练混用了两种曲率。相对官方HIER几何的差异是V8动机；多项改变的总收益尚未被单因素分解。

固定e300，V7→V8的实际sample祖先使用267→512，NN4覆盖27→231，最大重复四例287→34，最热四对象槽份额80.47%→24.85%；whole等深后覆盖221→1051。详细径向、方向、角色、cap梯度和独立形态图均在完整报告。

同时保留未解决的问题：V8有510/512代理触forward cap、输出深度几乎4.6；最浅1%whole仍占72.17%近邻槽。使用组代理有类别方向亲和，但不位于细类90%方向锥内。固定e300全池Chamfer富集有局部正面证据，best与其他指标却无一致改善。

分类方面，V8主模型比V7多对14/2468个对象；但同协议B64仍比V8-HIER多对3例。因此当前可以报告训练参与与检索几何改善，不能宣称明确HIER分类收益或普遍真实形态层级收益。

## 核验与来源

- [复测聚合结果](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/summary.json)、[原始权重复测](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/tables/original_retest.csv)、[逐类划分](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/tables/split_class_counts.csv)、[e300训练/验证](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/tables/e300_train_validation.csv)。
- [只读固定权重评估脚本](diagnostics/fixed_hycore_checkpoint_eval_v1.py)，首次代码提交cb83192；所有best在本轮评估前固定，输入/划分/源SHA与不变性核查通过，未用本轮test重新选模或调参。
- [原版数据与测试循环](../../classification_ModelNet40/main_pointmlp_hycore.py)、[原版输入](../../classification_ModelNet40/data.py)、[统一划分函数](../../inter_hierarchy_MN40/hier_proxy_scratch_v4/train.py)、[V8对照训练与最终test](../../inter_hierarchy_MN40/hier_proxy_scratch_v8/controls.py)。
- 动机与既有证据：[36：V7设计](36_V7_PROXY_SAFETY_DECISION_AND_LAUNCH_2026-10-07.md)、[37：V7完成](37_V7_COMPLETED_2026-10-07.md)、[49：V8几何/margin分析](49_V8_HIER_GEOMETRY_AND_MARGIN_FEASIBILITY_2026-10-15.md)、[51：V8对照口径](51_V8_CONTROLS_PLAN_2026-10-15.md)、[53：全系统审查](53_V8_SYSTEM_AUDIT_2026-10-16.md)。

原始权重、缓存、数据、完整日志、服务器路径和私有运行spec均不在此公开材料中。本轮没有启动新训练。
