# V8系统审查：结果索引

日期：2026-10-16。已完成V8-HIER64与V8-B64的全量审查，并与V7及原版HIER的CUB、Cars、SOP best分别比较。

2026-10-17补记：单卡V8-B0-32-C01现已完成，best e249、val OA94.5122%、test OA92.6256%；五份固定权重复测与协议对照见[54](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17.md)。下文和本目录原报告保留2026-10-16采集状态，未覆盖旧审查结果。

优先阅读：[可视化报告](53_V8_SYSTEM_AUDIT_2026-10-16/report.html)；[完整Markdown](53_V8_SYSTEM_AUDIT_2026-10-16/report.md)。报告包含18组PNG/SVG科学图、14张匿名汇总CSV和[机器可读汇总](53_V8_SYSTEM_AUDIT_2026-10-16/summary.json)。

## 当前判断

1. **休眠改善，热点仍在。** 固定e300，实际训练sample非碰撞祖先并集由V7的267/512升至V8的512/512；V8正hinge祖先并集也为512。反向NN4覆盖由27升至231，最大重复四例由287降至34，但最浅1%对象仍占72.17%近邻槽。
2. **径向状态发生了变化。** V8末期510个代理原参数超过forward cap，球内输出几乎同深4.6；clean whole中位数3.741、std .350，明显区别于原版三个数据集whole全部在4.6薄壳。参数范数中位数6.923不等于代理球内深度。
3. **两种热点机制仍有证据。** 仅统一whole深度，V8覆盖231→1051；再使用随机代理方向，等深覆盖1476–1545。随机方向的raw覆盖却只有53–62，说明已学习方向部分抵消径向偏置；不能直接把代理随机重置作为修复。
4. **代理有类别方向亲和，但不是细类簇内节点。** V8全部512个均sample-used，最近类别方向夹角中位数74.39°，随机参考约82.7°；没有落入最近细类的90%方向锥。V7角度对比已只算当轮使用的267个代理，中位数76.08°。原版CUB/Cars的代理也不在该方向锥内。
5. **尚无分类增益；有局部形态正面证据。** V8-HIER test OA92.7877%，同协议B64为92.9092%，仅差3个测试对象。固定e300全池Chamfer富集差值+.0436，按类别配对bootstrap区间[+.0166,+.0695]；best比较和其他指标未显示一致优势，不能扩展成普遍结构收益。

![V7/V8的真实训练代理使用轨迹](53_V8_SYSTEM_AUDIT_2026-10-16/figures/18_proxy_lifetimes.png)

![原版三个数据集、V7和V8的热点控制](53_V8_SYSTEM_AUDIT_2026-10-16/figures/05_coverage_counterfactuals.png)

## 已完成的审查与验证

复用全系统审查工具，并显式适配V8的c≈.1、最终共享表示cap2.3/hook、官方距离与gamma500/delta2。完成主/对照各12个8856对象全量快照，结合V7为35个系统快照；8个冻结机制快照、12个独立形态快照、3个真实编码器梯度探针。154项实现与合同测试通过。

18组图覆盖性能、半径、参数cap、实际激活、角色、生命周期、全池NN4、热点类别与点云、类别方向、热点剔除、绑定稳定性、独立形态、Gumbel噪声及编码器/分项梯度。输入身份与只读状态验收；零optimizer更新，没有新增val/test评估。完整ID、缓存、点云、权重及日志留私有，不进Git。

V8-B0-32-C01单卡shuffle对照在本次采集时仍训练中：北京时间2026-10-16 01:01，e283第157/276步，当前best e249、val OA94.5122%，没有final test。因此主报告没有伪造该组最终结论或将其当作已完成几何对照。该单卡组也采用V8 cap/hook/margin，并非原始c=1 HyCoRe。

## 复用入口与归因边界

源训练commit：主V8为14f8a72，B64为85fc812。审查适配提交ad53d36、7ebc081，当前报告另行提交于专用分支。见[复现说明](53_V8_SYSTEM_AUDIT_2026-10-16/复现说明.md)、[报告生成器](diagnostics/v8_system_review_v1.py)、[方向演化工具](diagnostics/hier_directional_evolution_v2.py)和[实际V8算子近邻验收](diagnostics/hier_v8_production_retrieval_v1.py)。

V7→V8同时改变曲率、cap、hook、intra margin与HIER系数，收益不能作单因素归因。原版CUB/Cars/SOP的数据域、维度、类别数与训练协议也不同；三组best提供参考几何，不能还原训练时间因果。原版CUB的严格距离覆盖1646与官方FP32的1647差1，已在报告区分。

已有依据：[V7系统审查](40_HIER_SYSTEM_AUDIT_RESULTS_2026-10-07.md)、[V7方向与热点](44_V7_DIRECTION_EVOLUTION_AND_HOTSPOTS_2026-10-08.md)、[原版best几何](45_ORIGINAL_HIER_BEST_GEOMETRY_AUDIT_2026-10-08.md)、[原版SOP机制与真实图像激活](46_ORIGINAL_HIER_SOP_MECHANISM_AND_ACTIVATION_2026-10-08.md)、[V8几何方案](49_V8_HIER_GEOMETRY_AND_MARGIN_FEASIBILITY_2026-10-15.md)、[V8对照计划](51_V8_CONTROLS_PLAN_2026-10-15.md)。

后续优先检验径向偏置、方向共同分量与cap后径向可优化性；本次没有启动新训练矩阵，也没有基于test挑选下一版。
