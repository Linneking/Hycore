# 新对话交接：HyCoRe实验与V9嵌入审查

交接日期：2026-10-21，北京时间。此文件用于继续讨论当前研究，当前重点是阅读和解释69报告，以及在用户提出具体要求后继续诊断。

## 1. 接手入口与当前状态

权威代码工作区为 `D:\Hycore`。按项目AGENTS要求，先依次阅读：

1. [00：研究入口](D:/Hycore/docs/research/00_RESEARCH_HANDOFF.md)。
2. [01：实验历史](D:/Hycore/docs/research/01_EXPERIMENT_HISTORY.md)。
3. [02：工作流程](D:/Hycore/docs/research/02_WORKFLOW.md)。
4. [04：计划与已授权配置](D:/Hycore/docs/research/04_NEXT_EXPERIMENT_PLAN.md)。
5. [本地服务器映射](D:/Hycore/.codex-local/SERVER_PATHS.md)，文件存在时读取。

然后阅读本次重点：[69审查报告Markdown](D:/Hycore/docs/research/69_V9_EMBEDDING_REVIEW_2026-10-21/report.md)，或[图文网页](D:/Hycore/docs/research/69_V9_EMBEDDING_REVIEW_2026-10-21/report.html)。用户上一轮正在浏览这个网页。69已完成，不需要从头重做采集。

截至本次交接核验，68所列17组实验中，16组已完成，只有最新c0.1循环HIER组仍在训练。状态来自保存记录的只读查询，没有重新评估或操作进程。

| 最新循环实验 | 核验时间（北京时间） | 状态 | best test OA／同轮AA | 第300轮test OA |
| --- | --- | --- | --- | --- |
| c0.1循环B64，无HIER | 2026-10-21 23:27:17 | 已完成300轮 | 93.395462%／90.570343%，e223 | 92.787682% |
| c0.1循环HIER64 | 2026-10-21 23:26:44 | 已完成140轮；当时e141 step88 | 93.314425%／90.745342%，e91；中途结果 | 尚无 |

[匿名状态快照](D:/Hycore/docs/research/artifacts/new_chat_handoff_2026-10-21/status_snapshot.json)保留精度和时间。它不是实时监控；用户询问状态时应重新读取manifest与heartbeat。**68总表的16:44快照较旧，不要继续把循环无HIER组说成正在训练。** 69对循环HIER的几何测量固定在e120，并非最新轮数。

本次只完成交接，没有启动新训练、停止任务或改变实验参数。审查使用的GPU已释放；下次用卡前重新核对占用。

## 2. 必须区分的实验身份

所有下表组均使用完整9840训练对象、2468 test逐轮选best。旧VS984组保留8856/984验证协议，见68，不与下表混名。

| 组别 | 原生曲率、cap、GS、margin | batch／更新预算 | HIER输入与代理 | seed |
| --- | --- | --- | --- | --- |
| 历史原版HyCoRe | c1，无额外cap/GS；1000/N与4 | 单卡shuffle32；历史运行含早期resume | 无层次代理 | 4780 |
| 原版c0.1＋cap＋半margin＋GS | c0.1，cap2.3、共享GS；500/N与2 | 单卡shuffle32，307步×300轮 | 无层次代理 | 4780 |
| 已完成V9-B64，无HIER | c1，无额外cap/GS；1000/N与4 | 双卡类别均衡64，200步×300轮 | 完全关闭HIER | 22 |
| 已完成V9-HIER64 | c1，无额外cap/GS；1000/N与4 | 同上 | c0.1固定深度4.6视图；代理cap2.3与自身GS | 22 |
| V8-HIER64／TS | c0.1，cap2.3、共享GS；500/N与2 | 双卡类别均衡64，200步×300轮 | 原生whole实际半径；代理cap2.3与自身GS | 22 |
| 最新c0.1循环B64，无HIER | c0.1，cap2.3、GS；500/N与2 | 双卡循环队列64，300步×300轮 | 关闭HIER | 4780 |
| 最新c0.1循环HIER64 | 同上 | 同上 | 固定4.6视图；基础路径与HIER路径各一次GS | 4780 |

旧c1 V9与最新c0.1循环组是两套实验。规范全名及全部历史成绩查[68：17组总表](D:/Hycore/docs/research/68_ALL_V8_ORIGINAL_V9_EXPERIMENTS_2026-10-21.md)；循环两组的最终状态以本交接快照或新的服务器查询为准。

原版c0.1基线→旧V9-B64同时改变几何、margin、seed和训练协议，不能单独归因于batch。最新循环组与该基线的几何和seed一致，但还改变类别均衡、循环抽样、全局负例与307→300步，应称batch构建协议效应。各自配对的无HIER↔有HIER才用于检验HIER分支整体增量。

## 3. 69报告已经确认了什么

审查覆盖上述五个已完成核心组各best与终点，加循环无HIER的best/e300/e120、循环HIER的e120，共14份固定权重。所有模型使用同9840对象、相同前1024点及输入字节、batch32、clean/eval历史BN。新增11组PNG/SVG和完整分位统计，没有test前向、反向传播、优化更新或正式BN校准。

### 代理径向分布没有改善

- 共同e300：V8有510/512代理贴cap，V9为512/512。V9代理输出深度约4.599992–4.599994，差别只是浮点微差，没有可辨认的径向层次。
- V9保存的切空间参数模长中位数为6.885，V8为6.915，均远超前向cap2.3。参数模长与映射后代理深度必须分开。
- 两组初始代理完整张量哈希一致。V9在e5只有378个贴cap，低于V8的510个，早期饱和延缓；但e99存档已全部贴cap。没有记录每个代理首次越界的准确step。
- 参数模长中期约9.8、末期约6.9，晚期确有回缩，但未回到cap以内。本次观察不能单独判定即时梯度、Adam历史状态、eps或weight decay谁是唯一原因。

### 原始热点缓解，但不能据此宣布方向结构更好

| e300指标 | V8-HIER64／TS | V9-HIER64 |
| --- | --- | --- |
| 实际半径／实际HIER视图的NN4覆盖 | 216 | 725 |
| 最热四对象的近邻槽位占比 | 32.67% | 6.49% |
| 最大重复四例／512代理 | 36 | 7 |
| 冻结whole后统一深度的覆盖 | 1025 | 725 |
| 代理方向集中度R | 0.412 | 0.625 |

NN4是每个代理在全9840候选中检索4个不同对象，共2048个槽位；不是训练实际使用样本总数。V9的实际、等深度、纯方向三种检索逐代理四例集合完全一致。V8冻结等深度覆盖仍高于V9，且V9的R更高，不能声称V9方向分布更分散。

### 原生whole仍有明显分散，HIER视图成壳是构造

| 共同e300原生whole | 均值 | 中位数 | 标准差 | 曲率 |
| --- | --- | --- | --- | --- |
| 历史原版 | 4.565 | 4.490 | 0.508 | c1 |
| 原版c0.1＋cap＋半margin＋GS | 3.836 | 3.837 | 0.514 | c0.1 |
| V9-B64，无HIER | 4.039 | 4.326 | 0.970 | c1 |
| V9-HIER64 | 3.888 | 4.254 | 1.093 | c1 |
| V8-HIER64／TS | 3.841 | 3.802 | 0.323 | c0.1 |

跨曲率按图中分面解读。V9-HIER64的原生whole在best e204的均值/中位数为4.831/5.082，末轮为3.888/4.254；两者不是同一份权重。原版c0.1基线best e184贴cap96.96%，末轮12.47%。这些跨checkpoint变化涉及权重与BN共同变化，本次没有对V9做新的BN因果干预。

V8与V9的best/e300均有全部512代理被sample与proxy分项在整轮中选中过；正hinge且非碰撞的选择集合也覆盖512。**V9当前不是V7式大量代理休眠，而是代理在同一径向壳上继续参与关系学习。**

## 4. 下一轮解释数据时的边界

1. **双域**：旧V9原生whole/part为c1；HIER whole视图与代理为c0.1。代理内外关系与后者比，不能直接用原生c1深度作参照。固定4.6视图不证明原生whole学成壳。
2. **三种量**：保存参数模长、球坐标半径、原点双曲深度分别报告。69的q明确为sqrt(c)乘球坐标半径；旧文件字段命名应核对定义。
3. **选择与梯度**：硬祖先被选中、正hinge且非碰撞、非零梯度、真正径向位移是不同概念。选择频率熵的指数称“有效数量”，不是非零梯度代理数。
4. **两种前向与时点**：全量clean/eval用历史BN；图7–8是首批global64训练位置、该轮第一次更新前；图10–11是轮末固定权重。图11右列是参数模长热图，不是祖先选择次数。
5. **最浅1%**：69热点表按HIER所见whole排序。V9该视图同壳，指标无径向意义；原生c1 whole仍有深浅差异。
6. **性能与结构**：V9主实验best test OA93.314%，匹配B64为93.436%；末轮分别88.452%与91.167%。热点改善不能替代分类收益证据。
7. **原始HIER反例**：CUB/Cars/SOP的whole都约4.6薄壳，但SOP代理497/512越cap。whole成壳不足以保证代理浅层，见45/46。原版图像资料不需要为本次讨论重新下载。

## 5. 按问题查找材料与代码

| 需要做的事 | 入口 |
| --- | --- |
| 阅读当前结论、看图、核对分位数 | [69报告](D:/Hycore/docs/research/69_V9_EMBEDDING_REVIEW_2026-10-21/report.md)、同目录summary.json、stats.csv、class_stats.csv、proxy_stats.csv、hotspots.csv及acceptance.json |
| 查全部实验配置与成绩 | [68总表](D:/Hycore/docs/research/68_ALL_V8_ORIGINAL_V9_EXPERIMENTS_2026-10-21.md)及其中匿名JSON |
| 理解旧V9设计与匹配对照 | [65方案](D:/Hycore/docs/research/65_V9_DUAL_CURVATURE_PLAN_AND_LAUNCH_2026-10-20.md)、[66对照](D:/Hycore/docs/research/66_V9_B64_CONTROL_AND_OFFICIAL_REPEAT_2026-10-21.md) |
| 理解最新循环采样配对 | [67方案](D:/Hycore/docs/research/67_ORIGINAL_C01_CYCLE300_HIER_PAIR_2026-10-21.md) |
| 核对GS作用范围与官方HIER dataflow | [63说明](D:/Hycore/docs/research/63_EXPERIMENT_NAMING_AND_GRADIENT_SCOPE_2026-10-20.md) |
| 查代理Adam/eps、BN干预与whole/part间隔 | [57机制](D:/Hycore/docs/research/57_V8_CAP_AND_WHOLE_CAUSES_2026-10-18/report.md)、[58密度与BN](D:/Hycore/docs/research/58_V8_WHOLE_DENSITY_AND_BN_REVIEW_2026-10-19/report.md)、[59配对margin](D:/Hycore/docs/research/59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/report.md) |
| 查原版HIER三图像数据集 | [45几何](D:/Hycore/docs/research/45_ORIGINAL_HIER_BEST_GEOMETRY_AUDIT_2026-10-08.md)、[46机制](D:/Hycore/docs/research/46_ORIGINAL_HIER_SOP_MECHANISM_AND_ACTIVATION_2026-10-08.md) |
| 复用69采集/分析/图形 | [工具README](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/README.md)：collect.py、history.py、analyze.py、plot.py、build_report.py、test_contract.py |
| 查现成完整缓存、权重SHA、环境、服务器路径 | [私有69工程交接](D:/Hycore/.codex-local/handoff/69_V9_EMBEDDING_REVIEW_PRIVATE_2026-10-21.md)；查循环生产进程再读[私有67交接](D:/Hycore/.codex-local/handoff/67_ORIGINAL_C01_CYCLE300_HIER_PRIVATE_2026-10-21.md) |

生产代码分开保留：旧V9位于 [hier_proxy_scratch_v9](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v9)，最新循环配对位于 [hier_proxy_cycle_v9c](D:/Hycore/inter_hierarchy_MN40/hier_proxy_cycle_v9c)。原版c0.1基线适配在 [hycore_c01_gs_halfmargin_20261020](D:/Hycore/inter_hierarchy_MN40/hycore_c01_gs_halfmargin_20261020)。

69工具显式适配V9双域；旧tools/hier_postrun_audit单曲率入口不能直接套用旧V9。已有完整缓存、私有诊断副本与匿名表优先复用。只在需要新观察时再创建独立输出目录，避免覆盖旧结果。

## 6. 工作区、权限和未完成事项

交接前的公开成果分支为 `codex/v9-embedding-audit-20261021`，69报告完成提交为 `ed96b4f7dc9ad8bef6bbf94a2e1edc6384cd953c`，已推送并同步两台服务器。本交接文档的提交在其后。运行中的训练使用独立冻结源码，仓库文档同步不会更改其参数。

保留用户的未提交修改：47汇报稿及删除的30分钟稿、54的report/Motivation_and_Method、57/report、59/report。另有用户未跟踪资料和原版HIER交接目录；新对话先查看git status，提交只选本次明确修改的文件。

未来训练默认已经由用户明确改为全9840训练／2468 test选best、不提前停止；ORIG修改线seed4780，V系列seed22。最新循环配对明确例外使用4780。历史VS984和seed保持原身份，不追溯改名。这个授权覆盖旧通用validation优先条款，但不授权任意新实验矩阵。

服务器凭据、主机详情、原始日志、权重及完整特征缓存只留私有位置。用户已明确允许本次约266MB网络诊断副本暂存本机并转到已有项目私有目录，现有副本已经完成转存；无需重复索取同一用途授权。连接、运行环境与本地副本位置查私有69交接，不在公开文档复制密码。

接下来根据用户问题继续：

- 用户学习报告：先解释已有图表与定义，区分观测事实和机制推断。
- 用户询问运行情况：只读刷新循环HIER状态，区分best、末轮和中途成绩。
- 用户要求审查循环配对最终结果：等其自然完成后固定best/e300，使用同一对象与69工具补充比较，保留本次e120记录。
- 用户要求继续定位撞cap：先提出基于固定特征/关系的有界机制对照。当前尚未对V9逐项分离即时梯度、Adam历史、eps、weight decay和参数约束，候选不写成已证结论。
- 任何新完整训练、参数修改或提前停止，按用户后续具体指令确定；本交接不自动触发这些动作。

## 7. 可直接复制的新对话提示词

请接手 `D:\Hycore` 项目。先按AGENTS规定的顺序阅读必读文件，再读 `D:\Hycore\docs\research\70_NEW_CHAT_HANDOFF_2026-10-21.md`，重点理解69的V9 whole/代理审查。先简述当前研究状态，然后等我提出具体问题。请区分已完成的c1 V9与最新c0.1循环配对，已有报告和缓存优先复用；只有涉及实时训练状态时再查服务器。本条仅用于接续上下文，不启动新训练或改变已有任务。
