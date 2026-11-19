# 分类HIER：共享出口GS训练与独立分支的历史溯源

2026-11-19。本次只读本机已同步报告、结构化成绩采集和Git冻结源码；不连接服务器、不新增梯度审查或训练。范围是HyCoRe点云分类，原图像HIER仅用于核对GS原意，不纳入下方实验成绩。

## 1. 结论与定义

**确认有两次完整的“共享主支GS＋HIER”训练：旧V8的VS984版本，以及全训练集/测试选模版本TS10。** 二者不是同一运行的重复记账；它们分别有同协议CB64-H0参照。已核查的V5–V7关闭额外HIER输出GS；V4讨论的是HIER项专用的度量预条件，不能因为用了“梯度缩放”一词就归入共享GS。

这里的共享GS指：

```text
PointMLP及嵌入层 → 共享cap → 共享GS ┬→ 分类器/CE
                                      ├→ HyCoRe whole–part两项
                                      └→ HIER样本项
独立层次代理 → 自身cap/GS → HIER样本项、代理内部项
```

whole和part各经过自己那次出口GS；“共享”指同一个whole出口的GS同时处理基础目标和HIER样本项的回传，不是全batch共用一个标量。代理仍有自己的GS。

后来拆分**不只是为了消融GS，也没有以GS位置单因素训练证明共享GS有害**。10月20日先讨论保护原生HyCoRe的基础梯度，随后进一步加入固定壳以排除whole深浅对HIER近邻竞争的影响；正式V9同时恢复c1与原margin等。10月21日以后又保留B05的基础GS，在它之前分出独立固定壳HIER。此后仍用独立分支，并不表示GS的作用范围已经被完整消融证实。

## 2. 两次完整共享GS实验及匹配H0

共同条件：PointMLP、D256、c=.1、最终whole/part切空间cap2.3、共享出口GS、HyCoRe margin500/N和2；CB64=两卡local32、全局32类×有放回2例；300轮×200步=60,000更新，seed22。开启组512代理、初始化.9、Ks20/Kp20、T50、margin/tau=.1、两项λ=.5、W0；代理A0 AdamW LR.01→.0005/eps1e−8，代理自身cap/GS。HIER看到共享cap/GS后的实际whole深度，没有强制固定壳。

| 协议 | 实验身份 | HIER | 选中轮 | 选模OA | 选中权重test OA | 同轮test AA |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| train8856/val984；验证OA并以较低CE打破平局；最后一次test2468 | 旧V8-B64，VS984 | H0 | 132 | val94.3089 | 92.9092 | 90.3157 |
| 同上 | **旧V8-HIER64-K20-W0，VS984** | **共享GS** | **221** | **val94.2073** | **92.7877** | **90.0372** |
| train9840；每轮test2468严格更高OA选best，同分取首次 | TS09 / B08 | H0 | 276 | test93.8817 | 93.8817 | 91.9204 |
| 同上 | **TS10 / B08 / Hshared-A0** | **共享GS** | **297** | **test93.3549** | **93.3549** | **91.4326** |

旧VS984开启组相对匹配H0：验证OA−.1016pp、选中权重test OA−.1216pp（3个对象）、test AA−.2785pp。没有显示分类增益；只是一个seed，不能据3个对象认定GS有害。验证选中权重的test不称作“最高test OA”。

| 全训练集配对 | best OA | e300 OA | e291–300 OA均值±总体SD |
| --- | ---: | ---: | ---: |
| TS09 H0 | 93.8817 | 93.4360 | 93.1848 ± .2008 |
| TS10 共享GS＋HIER | 93.3549 | 92.9498 | 92.9660 ± .1832 |
| 开启−关闭（pp） | **−.5267** | **−.4862** | **−.2188** |

TS10的e300 AA90.3826，TS09为91.5576。该配方下三个OA指标均未提高，说明这套共享GS、实际深度HIER、A0与λ=.5的组合没有给出收益；不能从开关组直接分离GS位置、样本径向作用或代理优化因素。

身份保留：

- 旧VS984运行目录标识：`v8_hier64_20261015_022043/main`，生产提交`14f8a7201cd108a77bde3b2ec9c3053e3cc07579`，不另造TS编号。
- TS10原运行目录标识：`v8_full9840_testsel_20261017_232056/main`，生产提交`ff98bf9ab98601efec10271b398557262fb82717`；2026-10-17 23:22至10-18 11:38北京完成。
- 本轮对上述两个Git提交直接核对`hier_proxy_scratch_v8/geometry.py`：`FinalGeometry`在最终嵌入hook内调用带GS的`to_ball`；两个冻结版本均在`backbone.emb`注册该hook。TS10训练以同一个`values["mu"]`计算HIER，与base相加后统一反传。

成绩来源：[54协议复核](../54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/report.md)、[54已核验标量](../54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/summary.json)、[75图谱](../75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/report.md)、[75逐轮采集A](../75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/tables/source_capture_a.json)、[75逐轮采集B](../75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/tables/source_capture_b.json)、[现行TS登记表](../experiment_registry_ts_v4.json)。旧VS984与TS版本训练池、选模不同，不跨行直接归因。

## 3. 为什么后来分支：按当时记录区分动机与证据

### 10月3日–7日：V5–V7尚未使用共享出口GS

V5/V6固定c1、关闭额外cap和inverse-metric hook；V7继承这一点，改的是代理初始化、权重与更新后参数约束。V7配置显式记录`HIER_backward_hook=False`。因此不能把它们的成绩作为“统一GS＋HIER”的结果。[25实现交接的几何说明](../25_HYCORE_HIER_HPCS_TECHNICAL_HANDOFF_2026-10-05.md)、[36 V7执行](../36_V7_PROXY_SAFETY_DECISION_AND_LAUNCH_2026-10-07.md)、[V7配置实现](../../../inter_hierarchy_MN40/hier_proxy_scratch_v7/train.py)。

更早V4提案明确写CE/intra不接收HIER hook，比较ordinary与HIER-only metric-preconditioned backward；这不是共享主支GS。[10 V4提案](../10_HIER_PROXY_V4_PLAN_2026-10-02.md)。

### 10月15日–19日：V8主动迁移官方HIER几何

V7存在浅whole近邻热点、代理分成两群等问题。V8选择统一c=.1，移植cap2.3和GS，期望whole接近等深叶节点，并调整intra margin和代理初始化。它改变的是整套迁移配方，不是只开GS。[50实际实现](../50_V8_IMPLEMENTATION_AND_LAUNCH_2026-10-15.md)、[54动机与结果](../54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17/report.md)。

后续审查发现旧V8末轮clean/eval whole仍深浅不一（深度中位3.7408、SD.3503），代理却接近4.6薄壳（SD.0285）；把whole临时等深后NN4覆盖可显著增加。BN来源也影响whole深度读出，不能将现象只归因GS。旧V8的样本实际祖先并集已512/512，说明“没有分类收益”不等于HIER没有工作。[53系统审查](../53_V8_SYSTEM_AUDIT_2026-10-16/report.md)、[58深度与BN复核](../58_V8_WHOLE_DENSITY_AND_BN_REVIEW_2026-10-19/report.md)。

### 10月20日：从GS作用范围讨论，推进到原生主支＋固定壳

[63作用范围记录](../63_EXPERIMENT_NAMING_AND_GRADIENT_SCOPE_2026-10-20.md)的原文明确：共享GS影响所有目标“**不等于已经证明它损害分类**”。先提出只把GS移到HIER路径，保护基础分支在同一状态下的直接梯度；这一步只有CPU路由验证，没有单独完整分类训练。

随后记录“用户进一步提出：原生HyCoRe＋等深度HIER分支”：CE/whole–part保留原生径向表示，HIER仅取方向构造深度4.6视图，并用分支GS。主要机制目标是排除浅whole在HIER距离/近邻竞争中的径向偏置，同时尽量保留原生分类与whole–part计算；不保证代理浅层化，也不隔离共享网络的全部影响。[64初步设想](../64_V9_INITIAL_IDEA_2026-10-20.md)。

正式TS12/V9将主支恢复c1、无新增cap/GS和margin1000/N、4，HIER为c.1固定壳＋专用GS。因此与TS10相比同时改了曲率、cap、margin、表示视图和梯度路由，不能称GS位置消融。[65正式配置](../65_V9_DUAL_CURVATURE_PLAN_AND_LAUNCH_2026-10-20.md)。

### 10月21日–23日：B05基础GS被明确保留，HIER继续独立

TS13/14以B05几何为底座：cap后、基础GS前分叉，一路基础GS，另一路固定壳＋HIER GS。10月23日严格B05的TS17继续按用户选择保留基础GS，后续B13和SH32开启组延续此路由。[67配对设计](../67_ORIGINAL_C01_CYCLE300_HIER_PAIR_2026-10-21.md)、[79明确保留B05](../79_S4780_NATIVE_B05_LAUNCH_2026-10-23.md)。

因而“我们后来因为GS把HIER独立出去”只描述了一部分历史：最初确实讨论避免共享GS额外改变原生基础梯度；最终持续使用的却是**主支仍有GS＋HIER固定壳自己的GS**。这一路由服务于两种样本视图，而不是已验证共享GS应被淘汰。

## 4. 当前证据能回答与不能回答的事

- 能回答：共享GS并开启HIER确实跑过完整训练，两套数据/选模协议下均没有分类增益；不是从未试过。
- 能回答：拆分有明确的径向表示与近邻机制动机，超出单纯GS消融；B05的主支GS后来又被明确保留。
- 不能回答：在同一强底座、同采样、同优化器和同样本视图下，共享GS是否比独立GS差。现有历史没有这类GS位置单因素完整配对。
- 不能回答：当前固定壳比允许样本实际深度的共享表示更适合分类。历史TS10→TS12以及TS10与TS14/17的比较均有其他变化。
- 现有“主支GS开/关”完整组如TS07/TS08均为H0，不填补HIER开启时的GS范围缺口；不能把共享GS的HIER开关负结果反推成GS有害证据。

后续若讨论回到共享表示，应围绕固定底座核对“是否固定壳”和“GS在哪里”两个因素；本报告只列历史证据与缺口，不新增实验矩阵或启动任务。
