# HIER 训练后全系统检测方案 v1 — 2026-10-07

状态：**用户已审阅并批准本方案，2026-10-07开始实现扩展模块和V5/V6/V7跨版本只读审查。** 此次包括实际保存检查点的固定输入GPU回放、CPU机制诊断及独立形状检验；不启动新主训练矩阵、不更新模型、不新增test前向。执行预算与对照见[39](39_HIER_SYSTEM_AUDIT_EXECUTION_2026-10-07.md)。适用于V5/V6/V7，以及不带代理的HyCoRe基线。实验展示名称遵循[33](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md)。

## 1. 要回答的六个问题

1. whole 是否随着训练集中到数值边界，还是形成了有差别的径向分布？这种变化发生在训练采样状态，还是固定样本的推理状态也发生？
2. 代理的真实保存参数、映射后的球内位置、被约束后的参数分别怎样变化？被实际使用的代理与暂未使用的代理是否分化？
3. 模型使用了多少代理，使用是否集中，代理是否长时间休眠后重新启用？仅计算“非零计数数量”会漏掉使用极不均衡。
4. 同一个代理是否持续关联同一批对象？选择变化来自嵌入变化、采样/增强/BN变化，还是Gumbel随机性？
5. 反复出现的四个最近邻是否主要是径向偏置？代理是否与可辨认的对象结构对应，而不是只形成共同热点？
6. 几何与代理使用的变化，是否伴随验证表现、泛化差距、损失有效率或优化异常的变化？

检测报告提供描述和证据等级，不自动把“更稳定”“更靠内”“使用更多”判为更好的层次。

## 2. 数据盘点与分析层级

首先生成inventory，列出实际记录轮次、缺轮、检查点名称、字段来源与不可用项。不能从最后一轮倒推出未记录历史。

| 版本/运行 | 实际逐轮汇总 | 实际代理ID计数 | 可以直接做什么 |
| --- | --- | --- | --- |
| V7-HIER64-K20-W0 | e1–300，共300轮 | sample/proxy及三种选择域均有 | 逐轮几何、损失、使用热图与集合变化 |
| V6-HIER64-K20-W20 | e1–300，共300轮 | 同上 | 同定义的历史时间轨迹 |
| V6-B64（共同e20前缀后续训） | e21–300，共280轮 | 无代理目标 | 记录到的56000次更新及whole轨迹；前20轮不能补造 |
| V6-B0-32、V6-B0-64 | 各300轮 | 无代理目标 | 已记录的分类/协议指标；缺几何时明确缺失 |
| V5-HIER64及旧B32 | 各200轮 | 无完整structure计数 | 保存的标量和几何；不能还原逐代理选择史 |

V7目录实际有**21个检查点文件**，包含初始化、e1/e5、每20轮存档、e99和best/last别名。别名可以与存档指向同一轮，文件数不等于独立观察轮数。其steps日志约436MB，默认不扫描；常规报告读取300条轮汇总即可。

分析分为三个层级：

- **A：逐轮日志，默认CPU。** 覆盖已保存轮次，观察真实训练过程。whole统计包含增强、重复采样、BN及训练前向条件，因此不是同一对象的配对轨迹。
- **B：固定输入的检查点回放。** 只使用实际保存权重；同训练split、同对象ID、同点云字节、同clean输入和eval BN。default每类最多32个对象的固定面板；full模式覆盖全部训练对象。代理/whole轨迹和邻居稳定性由此计算。
- **C：聚焦机制与独立结构验证。** 固定query/mining/Gumbel的噪声对照、梯度分解及独立点云形状关系，用于检验解释。首版不把这些未做的实验填成已完成结果。

选择模型仍依据已保存的validation结果。官方test只读取训练结束时已保存的一次选中模型结果，检测工具不新增test前向。

## 3. 单位及数据身份必须统一

设曲率为-c，c>0，球坐标x：

- 原始球坐标半径 r=||x||，合法范围0≤r<1/√c。
- 归一化半径 q=√c·r，范围0≤q<1，适合说明离数值边界多近。
- 原点双曲深度 d0=2·atanh(q)/√c，单位与HIER距离margin相同。
- 保存的切空间参数模长 s=||u||。只有未被forward cap或数值投影改变时，expmap0映射才满足d0=2s。
- 对球内位置做logmap0得到的模长等于d0/2；它**不能替代保存参数的模长**。V5/V6旧字段proxy.tangent_radius_max属于这种映射后反算值。
- proxy输出数值投影、forward tangent cap和V7的post-AdamW参数约束分别统计；shadow诊断cap不能当成实际被执行的约束。

每条曲线注明群体和时点：全代理/实际sample祖先/非碰撞/有效hinge、更新前输出/更新后参数、训练采样/eval固定面板。代理ID仅在同一训练的稳定参数行中可比较，跨seed不把行号当同一语义对象。

## 4. 必备指标与图表

| 检测模块 | 必备数值 | 必备视觉内容 | 判断边界 |
| --- | --- | --- | --- |
| whole几何 | r/q/d0的mean、median、p10/p90、max，接近数值边界比例；按类切片 | 逐轮分位带；快照直方图/分布；固定对象径向变化 | 平均值会掩盖两群分化，球半径与双曲深度不能混用 |
| 代理几何 | 保存s与映射r/q/d0；约束/饱和事件；全体、used、inactive分布；跨快照径向及角向位移 | whole/代理深度分布对照；代理ID×epoch轨迹/热图；depth与邻居关系散点 | 平均whole与平均代理的比较不能证明祖先关系 |
| 实际代理使用 | pair/triple/combined分别统计all draws、noncollision、active noncollision；used P、exp(entropy)、集中度、零计数 | 代理ID×epoch使用热图；有效代理数与used数曲线 | active hinge不等于非零梯度，top4近邻不等于训练激活 |
| 使用稳定性 | 相邻记录的used集合Jaccard、新启用/失活数、使用频率JSD、tie-aware Spearman | 集合/频率变化曲线及使用热图 | 不同batch查询的变化含采样噪声；稳定的塌缩也可有高Jaccard |
| 对象关系稳定性 | 同代理topk对象ID的Jaccard/保留率，按实际快照间隔报告；whole/proxy配对位移 | proxyID×快照转移的邻居保留热图；真实点云四例gallery | 只有样本池、标签、输入SHA、eval模式和proxy IDs均一致才配对 |
| 热点与径向偏置 | topk union覆盖、槽位集中度、最常重复四例、低半径对象占比；raw双曲/direction/等半径对照 | 覆盖与热点配对图；半径分位与命中频率；热点真实点云 | 等半径是反事实检索诊断，不是自然训练的因果证据 |
| 损失和优化 | CE/intra/sample-HIER/proxy-HIER、加权HIER、collision和有效hinge比例、LR、梯度范数、模型裁剪、实际cap、跨rank代理差异 | 分项损失/有效率/优化事件曲线 | 权重与梯度降低不保证AdamW实际步幅同比缩小 |
| 分类与协议 | validation OA/AA/CE，clean train和validation差距；实际/名义更新轴、配置及split身份 | 分类轨迹、泛化差距、同协议对照 | 单seed多因素历史对照不能归因给HIER |

topk覆盖的分母和理论上限同时报告：512个代理、k=4只有2048个近邻槽位，union覆盖不可能超过2048个对象。不能要求它覆盖全部8856个训练对象，也不能用面板覆盖率冒充全训练集覆盖率。

missing显示为空值和原因；没有代理的基线不判为512个“死代理”。class purity仅辅助描述类内关系；同类四例不证明代理位于簇内，也不证明捕获了形态层次。

## 5. 必须保留的扩展检查

这些模块已规定接口/判读方法，但不宣称首版已完成全部计算：

1. **休眠与重新激活。** 对每个proxy ID统计连续零使用轮数、首次/最后使用、重新启用及历史累计使用。起始缺轮时只报告观察窗口内的休眠，不说从未激活。
2. **固定query选择重复性。** 在相同checkpoint、相同whole、相同triplet/mining下，多次固定或变化Gumbel seed；再跨checkpoint复用query。区分同权重随机波动与模型变化。不要用epoch使用集合代替这种检查。
3. **梯度与实际更新。** selected、active hinge、非零梯度、真正发生参数位移四个概念分别测量；选少量真实checkpoint做sample/proxy分项径向与角向作用、base/HIER梯度夹角和Adam步幅。不是重新训练主任务。
4. **独立形状信号。** 固定点数/尺度/旋转处理下的Chamfer或独立形状特征，检查代理近邻一致性和结构排序关联；对象/类别为重采样单位，不能把大量相关pair当独立证据。HyCoRe teacher属于自蒸馏，不能当独立形态真值。
5. **按类及失败样本定位。** 查看类内whole深度离散、代理使用集中、误分类混淆、热点是否来自重复/异常点云。可比较matched B64固定输入缓存；跨版本池不一致时拒绝稳定性比较。
6. **边界邻域数值与代理冗余。** 比较实际训练距离实现与audit精确general-c距离，输出算子身份；邻近重复代理的角向/距离分布需分块计算。不能静默把新距离当旧目标。

判定分级：数值非有限、非法坐标、身份不符或源文件变化属于错误；字段缺失属于能力缺失；使用集中、半径饱和、晚期泛化下降属于需要结合其他模块解释的观察，不预设通用“80%正常”等阈值。

## 6. 固定检查点与输入规范

标准policy只从实际现有文件中选e0、1、5、warmup末/后、20、40、99、100、160、200、best、last。默认最多12个检查点，不生成缺失权重；all模式必须显式提高预算。

fixed panel默认每类最多32、seed22；记录完整sample IDs、labels、输入SHA、checkpoint SHA、curvature、input mode、batch size和源commit。whole导出使用原模型的clean first1024/eval BN，没有增强，没有optimizer，检查BN/state和源checkpoint SHA未变。旧checkpoint缺train IDs或映射规则未知时拒绝猜split或代理位置。

邻居比较采用同一候选池：raw精确双曲距离、direction cosine和同半径双曲检索。排名保持对象ID并采用确定的tie规则。图像显示原始点云与对象ID；不把高维嵌入的二维降维图说成原始Poincare位置。

## 7. 工具目录与调用

代码目录：`tools/hier_postrun_audit/`，入口`python -m tools.hier_postrun_audit`。模块分为inventory/adapters、geometry/snapshots、extract、plots/report/gallery。无改动V5/V6/V7生产入口。

默认调用（在代码仓库根目录，RUN_DIR是已有训练arm，OUT_DIR是源目录外的新目录）：

```bash
python -m tools.hier_postrun_audit inventory --run-dir "$RUN_DIR"
python -m tools.hier_postrun_audit run --run-dir "$RUN_DIR" --out-dir "$OUT_DIR"
```

已有clean缓存进行CPU补算：

```bash
python -m tools.hier_postrun_audit run --run-dir "$RUN_DIR" \
  --snapshot-spec "$SNAPSHOT_SPEC_JSON" --out-dir "$OUT_DIR"
```

需要从已完成权重导出时显式使用`--extract --gpu N --data-dir ...`，先检查该卡完全空闲；默认固定面板、最多12检查点、3600秒，预算到达保存已完成部分，不杀进程、不使用忙卡、不读test。该GPU模式本轮未运行，属于下次审阅后可选执行。

跨版本使用`--compare-run`并列展示，默认不输出谁“更优”的因果结论。未结束训练必须显式`--allow-incomplete`且报告标识partial；输出目录存在即拒绝，不写进源运行目录。

固定输出：
- `audit_manifest.json`：检测工具commit、源身份、状态、能力与预算。
- `normalized_runs.json`、`epochs.csv`：统一字段、原字段来源、缺失值。
- `report.html`及`report_data.json`：离线报告与可筛选轮表，PNG/SVG科学图。
- `snapshots/snapshot_summary.json`及NPZ：配对几何、真实ID近邻关系与转换结果。
- 可选`feature_exports/`：checkpoint只读导出的私有缓存/输入身份/点云池；不提交Git。

完整日志、缓存、权重与私有服务器信息不入Git。对外只保留经过审查的汇总、图片及文档。

## 8. 交付与后续顺序

本次先完成：统一日志适配、general-c数学与身份校验、快照关系分析、离线报告、可选GPU只读导出入口，以及V7真实CPU报告验收。它已经能作为每次主训练结束后的独立调用，不需要重训。

接着优先补V7的固定面板快照，并与V6-HIER64/V6-B64在实际共同检查点比较；第一次先选e20、100、200、best、last或其中可用子集，核查输入和映射身份后再扩大。第三层的Gumbel/query控制和独立形状验证单独记录计划及预算，不能因为批量检测工具存在就自动开新实验。

这份文件规范“检测什么、怎么比较、何时不能下结论”。后续版本增加字段时走新adapter或schema升级，保留旧报告和旧生产路径。

## 9. 本轮实现验收

V7真实300轮日志已通过入口生成离线报告，包含44张PNG及44张SVG。人工检查whole深度、sample非碰撞代理ID使用热图和激活曲线；激活图按sample/proxy和三种选择域分别绘制，图例放到图外。可筛选运行和轮表，缺数据明确显示。

本地43项单元校验中42通过，1项因本地缺Matplotlib跳过；服务器此前版本42项全部通过，包含使用合成3D点云输入的PNG/SVG像素测试。随后新增零方向反例，避免等半径控制为原点代理伪造角向邻居。GPU导出入口只做合同校验，本轮未实际调用。

V7本轮没有提供固定输入嵌入缓存，因此报告没有V7的top4对象保留图、真实数据四例gallery或独立形状结论。其e300训练日志中sample非碰撞祖先使用267个代理，熵有效数量197.5884；这是实际训练选择计数，不是代理最近邻样本稳定性。完整报告保存在服务器新审计目录及本地非Git交付目录；仅保留[简洁验收记录](diagnostics/results/20261007_hier_postrun_audit_v1_acceptance.json)入Git。
