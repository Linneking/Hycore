# Next experiment plan — current review and historical drafts

实验展示名统一遵循[33：实验命名规范](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md)。旧文件名、运行ID及代码字段保留用于溯源；本次只规范称呼。

Status: **唯一V7已于2026-10-07 14:43:33完成300轮；16:00只读核查通过，GPU1/2释放。验证选中e187（OA93.8008%），最终test OA92.2204%、AA89.7204%。全程代理数值投影计数0，参数cap保持有效。完成记录见[37](37_V7_COMPLETED_2026-10-07.md)，配置与启动身份见[36](36_V7_PROXY_SAFETY_DECISION_AND_LAUNCH_2026-10-07.md)。未启动新实验；下方为历史计划。**

## 2026-10-07：训练后系统检测计划

用户已审阅并批准[38：HIER训练后全系统检测](38_HIER_POSTRUN_AUDIT_PLAN_2026-10-07.md)，授权实现、调试及串联V5/V6/V7。执行记录见[39](39_HIER_SYSTEM_AUDIT_EXECUTION_2026-10-07.md)：全8856训练对象固定输入回放、matched B64、代理休眠与固定query/Gumbel/梯度、独立形状检验及收益账本。各源冻结、零优化更新、不新增test前向；新工具独立位于`tools/hier_postrun_audit/`，不改生产训练路径。21个检查点文件不等于每轮都保存了嵌入。

## 2026-10-07：V7唯一完整训练的历史方案

完整配置与判读边界见[34：V7单次300轮安排](34_V7_SINGLE_RUN_PLAN_2026-10-07.md)。

2026-10-07追加核查见[35：margin与边界审计](35_HIER_MARGIN_BOUNDARY_AND_PROXY_GRADIENT_AUDIT_2026-10-07.md)：
未发现margin单位错误，但官方数值距离与现用稳定距离在c1边界有显著差异；297组末期sample梯度可忽略，
proxy参数梯度几乎全角向，大量切空间参数已进入投影饱和区。下面参数仍为候选，正式启动前的有界诊断
优先补固定mining/三元组/Gumbel下的距离算子影响，以及真实AdamW更新与投影前模长增长；不直接调整margin当作纠错。

- 两卡只跑 **V7-HIER64-K20-W0**：300×200步，global64/local32及V6基础协议保留；
  HIER从第一次更新启用，固定lambda=0.1，保留sample+proxy分项及原代理LR。
- 建议代理随机方向、统一精确原点深度 **d_init=min(5,0.9×m0)**；m0为零更新新模型的16批实际训练whole中位深度。
  固定5并不保证起始在whole内侧；校准用副本恢复BN/RNG，不进行优化器更新，不是warmup。
- 第三卡优先补已完成V6-B64 e99/e200/e300几何与独立形状对照，再做最多400次冻结whole代理短更新；
  之后读取V7早期快照诊断。它不能替代第二条完整训练，也不用于中途改变主配置。
- 初始化内部不保证后续内部；代理与whole数值上限不同，降低lambda也不意味着AdamW代理步幅同比降低。
  跟踪实际sample祖先集合与未使用代理的分化，分别报告训练状态和clean/eval几何。
- 最新授权“决策权给你……敲定一个方案进行V7实验，挂起后退出”已允许代理完成实现、验收和启动，无需再次等待审阅。
  针对代理隐藏参数越过数值投影阈值，追加受限对照并采用更新后参数约束候选，见36；34中的“无永久代理约束”暂由此更新覆盖。

历史授权：**2026-10-06用户已审阅并批准V6-B64及持续机制诊断至使用限额；该主任务已完成，诊断当时已触发限额。**

## 2026-10-06：已批准执行与持续诊断

完成补记18:40：同协议V6-B64已完成300轮及验证选中模型的单次官方test，见
[32](32_V6_BALANCED_B0_COMPLETED_2026-10-06.md)。此前机制诊断已触发使用限额；
本次只核查与归档，不追加实验。固定e99/e200/e300结构对照仍未完成，保留为后续工作。

用户原话中的旧称按33映射；其中“B0补同协议”现称V6-B64。

用户原话：“H20之后的训练不要20轮warmup了。直接启用HIER权重。”
“我同意补做B0补同协议，纳入V6范畴。挂起该主任务后，你可以尽情进行小规模机制诊断，发现新问题就提出新计划，进行新的实验，但不要阻碍主任务进行。实验直到触发限额”。

- 首要主任务：V6-B64，继承历史V6第20轮完整checkpoint，直接进行e21–300。
  这是已完成的共同CE/intra前缀，不重复训练20轮；V6-B64全程关闭HIER及proxy更新。
- 数据8856/984，seed22，两rank各local32，32类×2、有放回，200步/轮；
  CE/intra、alias/两次BN/FPS、workers2、eval32、300轮cosine和原优化器均保留。
  恢复模型、动量、scheduler last_epoch20、两rank BN/RNG/sampler，不重置LR。
  源e20最佳验证模型作为已完成前缀的候选保留；e21–300继续validation选模，最终官方test一次。
- 使用新版本化V6-B64入口与新目录，严格核对源完整format/config/split/ID/rank状态；
  先做状态校验和受限恢复短测，随后独立后台启动主任务。
  不把e20→e40完整V6-HIER64-K20-W20重放作为启动阻塞条件；旧V6-HIER64-K20-W20报告为同协议历史参照，
  未核验前不承诺逐位复现或精确配对。结构固定观察e99/e200/e300。
- 后续从随机初始化开始的新HIER训练：warmup=0，第一轮lambda_H=.5；
  从现有checkpoint做机制分叉则在分叉首步启用，清楚记录继承历史，不能称从头零warmup。
  原V5/V6入口及旧结果不修改。
- 主任务两张完全空闲GPU；额外GPU和CPU开展下方A/B/D诊断。
  主任务运行期间不占用其GPU、不改其运行代码文件或配置，不改现有结果。
  先无更新分解，再有身份记录的少量副本更新/冻结whole短实验；每个试验新目录。
- 据新证据更新计划并自主执行聚焦的新实验，拆分目标/协议/正确性变化。
  不用官方test反复选配置；对使用限额按系统实际状态停止，保存可续接交接。

具体启动和诊断身份记录于[29](29_V6_BALANCED_B0_AND_MECHANISM_LAUNCH_2026-10-06.md)。

### 首批结果触发的聚焦扩展（本次持续诊断授权内）

首批4个冻结eval-cache batch显示proxy pair约44%选择其三元组端点；
2个条件热点batch中，sample的完整ST共享参数梯度明显大于固定祖先hinge梯度，
且eval与trainBN的比例差异很大。它们是有限面板，不作为长训练因果结论。

- e99/e200/e300各16个相同balanced64计划，CPU重放sample/proxy径向分工与端点重合；
  e300另做100步冻结whole代理更新，sample/proxy/合用三项与原/等半径mining交叉。
  继承原AdamW状态、同有效LR；加零梯度动量+decay与fresh同LR decay控制，
  末端用相同原mining计划无更新评估。固定LR诊断不冒充完整cosine训练。
- GPU3以自然抽样16个batch扩大e300的eval/trainBN/local32重分组VJP；
  再对既有2个热点面板补同输入重复Gumbel基线。
- 对e300热点做base、base+.5sampleST、base+.5sample_fixed配对一步：
  完整继承模型与RSGD动量、有效LR、norm1裁剪；旧BN固定clean1024读出，
  判断参数更新的即时径向和角向响应。每批独立还原起点，不能称持续训练。
- 如上述信号经扩展稳定，再在e20/e99/e200定位时间差异，并补独立点云几何关系信号。
  source checkpoint/HDF/cache不变；各试验新目录，GPU1/2主训练资源不动。

结果与执行身份另记[30](30_HIER_MECHANISM_DIAGNOSTICS_2026-10-06.md)。

## 2026-10-06：用户已审阅的诊断设计

本节原为待审阅候选，现按上方最新用户授权执行；具体时长、恢复验收和资源分配以上方为准。
依据[28](28_HIER_EVIDENCE_AND_NEXT_DECISIONS_2026-10-06.md)：e300新增的297个静态合格边界代理
当轮没有被sample选作祖先，放大了总体近四集中。真实sample使用集合仍为188个，最大同近四集合
92/188，best99为94/188；真实使用集合的近四仍有90.16%低半径1%槽位。
因此应分别解释冻结检索偏置、实际祖先组织、训练导致的whole变化，不能用一个指标替代三者。

### A. 优先：固定条件、无正式更新的机制检查

- 选V6完整e20、best99、e200、e300，固定16个global64计划、数据/增强/whole-part/negative配对；
  热点dresser、普通dresser、bathtub及其它类别分开，既保留源式balanced抽样，也记录每类样本/角色覆盖。
- 分解CE、`.01Rcontr`、`.01Rhier`、`.5Rsample`的有符号径向和角向作用；
  同时看共享backbone与Möbius层参数梯度，不能只用mu偏导或总体范数解释模型更新。
  Rproxy不直接依赖whole，作为仅代理项报告；它可能经后续proxy位置间接改变Rsample。
- 同一输入分别测evalBN、源式trainBN和local32成员重分组；重分组时固定negative身份。
  额外诊断始终恢复BN/RNG；whole/part仍保留原复写与两次BN流程。
- 固定输入和triplet，先固定祖先身份查看hinge直接作用，再单独重放hard-Gumbel选择查看完整代理选择路径；
  原mining与半径控制mining另作比较，不混入固定关系梯度的结论。
- 关系稳定性同时统计：相同输入重复Gumbel的基线、跨增强、BN模式和重分组的改变。
  确定性minimax可作辅助，不冒充实际随机祖先使用。
- 如需测实际单步深度改变，在模型/optimizer副本上从同一状态做少量配对一步控制：base vs base+Rsample。
  原始权重不改；保留动量、投影、裁剪与有效LR，报告固定eval输入的更新后深度。
  这是即时响应检验，不能单独证明e20到e300的长期原因。

决策：若差异主要随BN模式/组别改变，先定位关系稳定性；若相同条件下HIER在早中期反复增加热点
向内的实际响应，优先定位sample目标与mining；若只在一个末期batch出现，不据此推广整个训练过程。

### B. 与A配套：冻结whole，分解代理项

固定e200/e300缓存、同一proxy起点、batch和随机流，比较Rsample、Rproxy及合用的梯度。
记录实际sample pair/triple分工、proxy祖先与端点身份重合、各层使用熵、原点深度排序及投影前后移动。
先做不更新检查；如需100–200步仅代理更新，另列为有新优化器更新的短诊断，审阅后启动。

若代理项确证破坏sample所用代理的组织，再讨论仅代理的有界参数化；若实际分工合理而全局最近四检索
仍集中，应修正检索解释并检查独立形态信号，不能仅凭近四热点改模型。

### C. 必要的训练因果对照：V6-B64

- 从V6第20轮完整状态分叉。固定8856/984、两rank各local32、32类×2、有放回抽样、200步/轮、
  原300轮cosine、CE/intra、原part/BN/FPS、seed/workers和验证选模；只关闭HIER权重及proxy更新。
- 恢复两rank BN/RNG/sampler、模型optimizer动量、scheduler已完成轮次与DDP行为；
  V5 e20不可替代V6 e20，因为cosine周期不同。
- 当前V6 resume要求training_config全等且HIER权重固定；需单独版本化分叉入口，显式审计唯一目标差异。
  不绕过校验，也不只加载net后新建optimizer或重启cosine。
- 先核对恢复结果与可用的逐步日志/已有存档轮次。若没有e21权重或绝对状态摘要，不能宣称已验收旧e21参数；
  可用已有e40存档验收重放，或从共同e20状态新跑两个短分支。
  若原V6-HIER64-K20-W20不能在事先约定的数值容差下重现，主对照两臂共同恢复重跑，不把旧曲线当精确配对结果。
- 先短窗口检查分叉正确性，再决定同预算训练。事先固定e99/e200/e300的结构观察点；
  模型选择使用validation，官方test仅最终选定模型一次，不按test挑结构或轮次。
- 主终点包括逐类depth、类内角度、whole-part间隔、validation与独立形态邻居一致性。
  V6-B64没有训练祖先代理；不直接用V6-HIER64-K20-W20 proxy去检索未对齐的V6-B64坐标作主结构对照。

决策：V6-B64同样出现类相关径向变化，说明其不是HIER独有，应追共同基础协议；只有V6-HIER64-K20-W20出现稳定增量差异，
才支持HIER的训练作用。分类改善与形态改善分别报告。

### D. 修改目标前，先验证关系质量

对固定同类样本对比较原mining与半径控制mining的增强稳定性及独立形态信号，
例如统一居中/等比例归一化的点云形状距离和预先选定的盲评样本；同类标签本身不作形态真值。
只有关系更可靠时才短测“只改mining、loss仍用原whole”；只有代理项问题被定位时才做代理限定修正。
不同时改CE、BN、lambda、cap和采样。不把whole等半径检索的改善直接变成等半径训练，
也不直接照搬whole的2.3 cap。当前不优先增加epoch/K/P或做超参矩阵。

## 最新只读诊断依据 — 2026-10-05

补充2026-10-06：[27](27_HIER_WHOLE_RADIAL_RETRIEVAL_DIAGNOSIS_2026-10-06.md)
已完成用户本次要求的现有缓存CPU诊断。十份V5/V6-HIER64-K20-W20、ORIG-B0-32-S4780、V6-B0-32及V6-B0-64缓存按文件名＋行号对齐8856ID；
径向检索控制表明 whole 半径异质性是集中现象的主要因素，删除低半径样本只让新热点接替，
仅统一 proxy 半径效果很小。Dresser在V6末期更浅且角向更宽，而全体中位深度接近V6-B0-64。
这些均为冻结控制，没有改变训练mining/loss/BN，也没有启动梯度更新或主矩阵。
下一步应解释CE/intra/HIER sample的实际径向/角向梯度及proxy项的移动作用；
同协议balanced64 V6-B64仍需另行审阅，不因本次诊断自动获准。

用户要求比较原始HyCoRe与V6-B0-64的batch和逐类whole/part。六权重全量train/test和无参数更新probe已完成，见[26](26_ORIGINAL_HYCORE_B64_CLASS_GEOMETRY_2026-10-05.md)。V6-B0-64没有一致外移或分类崩塌；原版也有弱类和后期拟合差距。三组都呈现明显train/eval BN径向间隔差异，local32组成员还会移动双曲几何而基本保持分类。

在追加主训练前，建议先固定同ID、负配对、part中心，检查train/eval及不同local32组的互惠关系与祖先身份稳定性。该建议仍待审阅，不是自动冻结BN、改SyncBN或增加训练队列的授权。应将BN组别敏感性与完整训练batch的因果效应分别报告。

## 已批准 V6-B0-64 基线 — 2026-10-05

- 从头随机初始化seed22，c1/D256，300epoch，双卡各32；关闭HIER，不建立或优化proxy。
- 全部9840训练实例，每轮全局shuffle一次、无放回，只保留完整global64；153step、9792个唯一实例、丢弃48个尾部实例。连续64分给rank0前32和rank1后32，不使用DistributedSampler补齐，不强制类别块，不截断到200step或重复遍历补步数。
- CE、intra均作用于global64，原eps=.2、`.01Rcontr+.01Rhier`、margin4和`1000/Npart`；保留part视图复写、child/whole两次普通local32 BN更新、原FPS512。global child.flip(0)允许自然出现同类negative并记录，不能沿用balanced入口的异类断言。
- 原RiemannianSGD，LR.1、momentum.9、WD2e−4，300轮cosine到.005，不随V6-B0-64放大学习率；模型全局L2梯度范数阈值1裁剪。
- 用户明确选择原源码评估口径：每轮同一官方2468test、按三位小数test OA严格更高保存best，平局不按CE更换。评估batch16以保持与既有原V6-B0-32一致，同时记录未四舍五入OA/AA、逐类结果、源式batch均值CE和样本加权CE。无验证划分。
- 保留每步CE/intra/总损失、训练OA/AA、globalflip同类负比例、batch类别数、每轮实际唯一ID/逐类覆盖、丢尾ID、whole/part半径深度及shadowcap、BN/alias/FPS断言、有限性、梯度裁剪、DDP梯度/参数一致性、LR/耗时/显存。每10轮无增强全训练集评估；每轮完整last/best与metrics，每20轮存档，保存两rank RNG、sampler和BN。
- 新路径`inter_hierarchy_MN40/hycore_b64_v6/`，保留旧HyCoRe和V5/V6入口。全局排列与双卡增强/RNG是显式适配，不宣称与单卡RandomSampler逐位重现。workers4/rank，普通BN不是global64 SyncBN。
- 本地实现/检查、独立分支提交推送、服务器干净工作区ff-only更新、空闲双卡2epoch×2step短测（部分test且明确smoke），生产重新随机初始化并保存第1轮完整checkpoint后确认挂起。

这组检验原采样方式下的双卡V6-B0-64稳定性。相比V6-HIER64-K20-W20，它还改变了抽样、训练数据量和每轮步数（153 vs200），**不是只去掉HIER的严格同协议对照**；相比原V6-B0-32，它改变globalbatch和优化器更新次数，每epoch覆盖基本相同。详细运行身份、完成结果和监测已在[23](23_V6_B64_SHUFFLE_START_2026-10-05.md)记录。本次只授权这一组，不自动追加其它训练。

## 结果后的建议顺序 — 2026-10-04，待审阅

1. 同协议global64 V6-B64：V6数据划分、初始化、batch/增强/BN/part、200step及300轮cosine，仅关闭HIER；双卡顺序运行，原源码V6-B0-32不替代该对照。
2. 冻结现有whole缓存和相同proxy起点，短测sample/proxy分项梯度及各自更新；监测投影后的径向梯度、角向冲突、祖先端点身份、深度和使用分工。明确问题后再讨论仅代理的cap/hook或平滑有界参数化，不直接给whole加2.3 cap。
3. 固定pair/triple及独立点云形态信号，验证原关系与半径控制关系；若证据支持，再短测只改变mining、loss仍在原whole上的适配。

保留c1、原CE/intra/part及self-k排除。V6最优验证在99轮、最终proxy检索更集中，当前不优先加epoch/K/T/P或直接调整权重/LR。上述为[22](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md)解释的候选，**没有授权排队新的主训练，也没有因结果审计自动启动训练。**

## 已批准 V6 启动 — 2026-10-04

- V6-HIER64-K20-W20：新随机初始化seed22，300epoch×200step；保持V5的固定8856/984划分、c1/D256、global64、part复写、两次普通BN更新、K20/T50/P512、原CE/intra与优化器数值。仅在sample和proxy两图的负候选中排除索引k=i；相同实例ID在不同位置的重复抽样仍保留。两个cosine调度周期相应延长到300轮，下限仍为模型.005、代理.0005；前20轮base-only，随后lambda_H=.5。代理LR仍.01，不采用之前降LR候选。
- V6-B0-32：按原classification_ModelNet40/main_pointmlp_hycore.py默认运行，完整9840训练样本、batch32、shuffle/drop_last、每轮307step、300epoch、seed22、workers8；保留源码优化器、300轮cosine、增强/RNG/part复写/BN/FPS。用户本次要求源码默认协议，因此保留逐轮官方test与best-test选择，明确作为源码复现工程基线，不作为验证选中或同协议global64的HIER增益证据。
- 三张完全空闲GPU：两张V6-HIER64-K20-W20，一张V6-B0-32。每个训练新目录，记录commit/config/seed/GPU/时间；每轮checkpoint与关键监测。V6-HIER64-K20-W20保留V5所有监测并加入已同意的祖先使用、深度、跨增强稳定性和梯度分项检查。额外probe恢复BN/RNG，不增加optimizer更新。
- 可视化：只读V5-HIER64-K20-W20 checkpoint_epoch_200，全部8856训练实例无增强whole1024评估。使用proxy图K20的至少两个非self互惠正候选资格，seed固定随机选10个合格代理；按c1双曲距离展示每个代理最近4个独立训练ID的点云。top4为展示数，与训练K20分开。输出离线可旋转HTML、静态图与选择/权重身份元数据。
- 代码位于hier_proxy_scratch_v6和独立可视化脚本，保留V5及原HyCoRe入口不变。流程：本地实现与检查、提交推送、服务器干净工作区ff-only更新、短GPU检查与可视化，随后独立进程启动两项300轮训练；不等待训练完成、不自动追加其它实验。

V6改变了self-k规则和训练预算/调度周期，两者必须分别披露。单卡V6-B0-32与V6-HIER64-K20-W20的数据划分、batch、抽样和选模协议不同，不能单独隔离HIER目标的因果影响。具体启动身份与验收记录将在[21](21_V6_TRAINING_START_2026-10-04.md)补充。

## V6 讨论记录 — 2026-10-04

以下记录启动授权之前的讨论；已由上方“已批准V6启动”部分更新。此前范围为新增结构监测及候选解释，没有自动授权训练。

已同意监测：

- 实际选中的 pair/triple 代理深度、深度差及相对端点的位置。
- 各代理作为 pair/triple 祖先的使用次数，与 proxy 三元组 i/j/k 角色统计分开。
- 相同实例跨增强的邻居及祖先选择稳定性。
- self-k 与不同实例三元组的损失及实际参数梯度分开统计。
- proxy 的径向/角向更新，区分 sample 与 proxy 项的推动作用。

资源约束：用户目前只有三张可用卡，其中两张用于 V6-HIER64-K20-W20，剩余单卡不能预设能容纳直接 batch64。global64 V6-B64 的建议采用双卡依次运行，不要求与 V6-HIER64-K20-W20 同时占用四张卡；具体排期尚未确定。剩余单卡的 V6-B0-32 可做辅助基线，但不能作为同协议 global64 的 HIER 增量对照。普通两次 batch32 梯度累积会改变跨批 intra 负配对与 BN buffer 更新，不能直接宣称等价。

解释口径：V5 的 0.5 是 HIER 损失权重，不是学习率；模型 LR .1→.005，proxy LR .01→.0005。mask 只检查 pair/triple 是否选中同一个祖先代理，不自动排除 i=k。约32%的 proxy 合格率是可作为三元组 anchor 的代理位置比例，不是启用代理数量或活跃三元组比例。排除 self-k、调整 proxy LR 仍是待审阅候选，没有因同意监测而自动获准。

## 已批准主测试：V5-HIER64-K20-W20，2026-10-03

追加检测见 [16：topK覆盖、稳定性与实际参数梯度](16_V5_TOPK_FOLLOWUP_2026-10-03.md)。
选择K20主要依据覆盖改善和已检查的数值/显存预算，不是分类最优或真实形态关系的证据。

### 固定配置

| 项目 | 主测试设置 |
|---|---|
| 数据 | 既有seed22固定8856/984训练/验证划分；官方test只在最终选定checkpoint上评估一次 |
| 初始化 | 新V5随机PointMLP；不加载V4、历史原HyCoRe或teacher权重 |
| 预算 | 总200epoch，每轮200step；前20轮base预热计入200 |
| batch | 2GPU，各16个不重复类×2，全局32个类×2＝64；类别均匀、类内有放回，保留相邻类块 |
| 损失范围 | CE、intra、inter均global64；intra negative是全局child.flip(0) |
| HyCoRe输入 | 原whole800–1024、part200–600，视图复写；每步两卡共享点数；原FPS固定512首层中心 |
| BN | 普通训练BN，每卡32，child与whole均更新，DDP广播rank0 buffer；不使用SyncBN或HIER的预训练BN冻结 |
| 空间/精度 | c=1固定，D=256，FP32；保持HyCoRe原数值球投影 |
| CE/intra | 原eps=.2平滑CE；`.01×Rcontr + .01×Rhier`；margin4及`1000/Nchild`不变 |
| HIER关系 | sample/proxy共用K20，K含self；仅sample图同类相似度加1，proxy图只用距离；至少2个非self互惠j，否则跳过i |
| HIER抽取 | 每个合格i有放回抽j/k各50次；负候选为互惠集合补集，保留源码k=i行为 |
| inter | 一个`Rsample + Rproxy`，原三个hinge、hard Gumbel、同proxy碰撞mask后包含零值的均值 |
| HIER数值 | margin=.1，tau=.1；额外tangent cap和Riemannian反向hook关闭，只做shadow监测 |
| hierarchy proxy | P=512、D=256，seed22随机切空间初始化；不绑定类别、不约束part |
| 模型优化器 | RiemannianSGD，LR.1，momentum.9，WD2e−4；原模型global norm1裁剪 |
| proxy优化器 | AdamW，LR预算.01，betas(.9,.999)，eps1e−8，WD.01；单独同步梯度，不混入模型norm1裁剪 |
| HIER启用 | 第1–20轮λ_H=0且proxy不step，第21–200轮固定λ_H=.5；不再加ramp或动态梯度标定 |
| 模型选择 | 每轮固定无增强验证，按最高val OA选best，OA相同取较低val CE；记录AA及逐类准确率 |

联合损失：

\[
L=L_{CE}^{\varepsilon=.2}+.01R_{contr}+.01R_{hier}
  +\lambda_H(e)(R_{sample}+R_{proxy}).
\]

两优化器共用200轮相对cosine因子（e为已完成epoch数）：

\[
s(e)=.05+.95\frac{1+\cos(\pi e/200)}2,\qquad
lr_{model}=.1s(e),\quad lr_{proxy}=.01s(e).
\]

proxy前20轮不更新；启用时实际LR约.00977，调度终点.0005。骨干调度不在第21轮重启。共用cosine是结合HyCoRe训练流的适配；CUB发布脚本实际每5轮减半，官方warmup只暂停预训练body更新，不等同于本方案的20轮base-only。

### 归因与最小对照

主配置V5-HIER64-K20-W20用于检查恢复算子、global64协议下的联合训练。要判断HIER是否有利，至少再有同V5配置的V5-B64：只令λ_H=0，其余训练预算、初始化、数据顺序/增强、BN、输入和验证方式一致。LEGACY-V4-B32-Q4x8-BN1不能替代这个对照。

本次授权运行V5-HIER64-K20-W20，剩余单卡运行V5-B0-32-CAP200稳定性诊断；没有授权再自动排队一个完整双卡V5-B64。V5-HIER64-K20-W20保存新的第20轮prefix，以便后续获准时接续同协议V5-B64。prefix包含模型、optimizer/scheduler、split与sampler身份及各rank RNG状态；不能使用旧V4 prefix。

V5-B0-32-CAP200使用同一8856/984划分、随机初始化seed22、恢复原part复写与两次BN更新、c1/D256、CE/intra与模型优化器。每轮将训练集无放回打乱，训练前200个完整batch32，即6400次抽取；保持用户200step预算，因此并非原版完整遍历（完整drop-last为276步）。负part仍为本batch的flip，允许原随机batch中同类负配对并记录其比例。它与V5-HIER64-K20-W20的global64/类别均匀有放回协议不同，不能作为严格HIER增益对照。详见[17：启动与监测配置](17_V5_TRAINING_START_2026-10-03.md)。

拟议的V5-B64同协议对照恢复HyCoRe核心算子，但采用新batch/预算/验证划分，不应称作原训练协议逐参数复现；这组尚未运行。先区分baseline恢复、协议适配和inter增量三个效果。

### 运行记录与监测

- 始终记录实际唯一训练ID、逐类抽取次数/作为i的次数、任意角色覆盖、有效/碰撞/self-k三元组及两张图的合格anchor。
- 记录μ/ν深度与半径、shadow cap和贴近原数值球边界比例、proxy径向移动、安全project、非有限值及裁剪前norm。
- 第21、40、100、160、200轮首batch测CE/intra/HIER共享参数梯度，分别报告欧氏特征层和Mobius层；不能只用μ偏导或全模型平均范数判断协同。
- 每10轮额外做固定无增强train-eval，和训练模式accuracy、validation一起观察BN/train-eval差距；不借test选模型。
- 保存best/last及每20轮checkpoint、两个optimizer/scheduler、各rank RNG、sampler/config/commit、GPU和时间；新目录运行，保留旧结果。

当前A/200step理论单轮总体唯一覆盖66.08%，最大类chair约32.98%；继续遵守200step，不把它误写成80–90%覆盖。K20的参考合格率约60.39%，全部40类曾合格，仍有条件覆盖偏差及增强关系不稳定风险。

8步K20参考检测均有限，峰值28.73GiB/卡，净step中位数.6366秒；单臂200×200净step外推约7.07小时，实际总时间更长。短测不是完整训练性能或时间保证。

主训练、V5-B0-32-CAP200入口及恢复/验证/保存循环已完成启动检查和200轮生产运行。**启动记录见17，最终结果见19，原始身份以服务器manifest为准。**

## Authorized diagnostic scope — 2026-10-03

The user reviewed the V5 proposal and authorized diagnostics only:
schemeA,200steps nominal epoch,global64 CE/intra/inter,c1,D256,random
proxies,unchanged numeric margin/tau=.1,disabled extra HIER clipping and
backward hook with shadow monitoring. See
[bounded diagnostic plan](14_V5_DIAGNOSTIC_PLAN_2026-10-03.md).
This authorization does not launch main training or a full experiment matrix.
The bounded checks have completed; see
[diagnostic results](15_V5_DIAGNOSTIC_RESULTS_2026-10-03.md).
The earlier review section below records the proposal before these decisions.

## Historical pre-diagnostic review — 2026-10-03: dual-GPU V5 successor

The user requested V4 lessons and a concrete training-change catalogue for
review before the next dual-GPU tests and full training. The current proposal
is [V4 lessons](12_V4_LESSONS_2026-10-03.md) and
[dual-GPU V5 change index](13_DUAL_GPU_V5_CHANGE_INDEX_2026-10-03.md).
This section records the proposal before diagnostics. The current completed
checks and pending main-test recommendation are recorded above.

- Restore original HyCoRe part overwriting and child/whole BN updates;
  retain its intra expressions, smoothed CE and base optimizer.
- Proposed dual-GPU batch: 16 distinct classes x 2 instances per rank;
  32 disjoint classes x 2 globally. This is an explicit protocol adaptation.
- Screen K6/8/10 with HIER's released reciprocal rule (at least two nonself
  positives; skip other anchors), source K including self, and replacement
  draws. State the released self-negative behavior and all adaptations.
- Increasing an epoch from138 to200 steps gives expected overall unique
  coverage66.08%, but largest-class coverage32.98% under class-uniform
  replacement sampling on the current training split. It cannot meet an
  80–90% largest-class single-epoch target; sampler/budget choice is pending.
- Validate differentiable gathering, synchronized hierarchy proxies and
  DDP gradient scaling before a full run. Keep the validation-selection
  protocol; do not use the official test set for per-epoch selection.
- Separate baseline restoration, protocol changes and HIER's incremental
  effect. Specific curvature/backward, weights and full matrix remain to
  be resolved by the reviewed diagnostics.

The earlier teacher-based proposal below is retained as historical context.
Its temporary test-every-epoch debugging exception does not apply to V5.
V4's separately authorized, completed scope is recorded in
[its plan](10_HIER_PROXY_V4_PLAN_2026-10-02.md).

## Historical draft — teacher and restricted-scope experiments

## Revision after user review

- Temporarily preserve the original HyCoRe train/test evaluation protocol while debugging optimization stability. These runs are engineering diagnostics, not final paper evidence. A validation split is deferred until the method is stable.
- Audit checkpoint identity before any new run. The original HyCoRe reproduction and the derived A3 checkpoint must never be mixed or labelled interchangeably.
- Build matched feature caches from both checkpoints at two locations: Euclidean/tangent features before `expmap0`, and hyperbolic `mu` after the Mobius embedding layer. The same deterministic cache is reused across matched experiments.
- Freeze the point backbone first. Diagnose `head_only`, then `hyperbolic_and_head`, and only then consider full unfreezing.
- Do not use a fixed 0.3 percentage-point OA gate before baseline variance is known. Early runs use catastrophic-failure and persistent-drift criteria; a statistical non-inferiority margin is set only after stable repeated runs exist.
- Random seeds are deferred until one protocol is stable.
- The intended hierarchy has two distinct targets: a directed radial specificity signal and a symmetric angular kinship signal. Equal-radius LCA ranking addresses only the latter and cannot by itself realize the proposed ordinary-to-special ordering.

## Decision to make

Determine first which checkpoint/representation supplies meaningful and stable instance relations, then whether an angular inter-ranking loss can be added without destructive fine-tuning drift.

The immediate goal is not to obtain a higher headline OA. It is to establish a stable protocol where the zero-increment control preserves the pretrained checkpoint, then isolate the incremental effect of the inter objective.

## Phase 0 — checkpoint and representation audit

1. Verify and hash every candidate checkpoint, beginning with the original HyCoRe reproduction and derived A3.
2. Export four primary deterministic caches: original/pre-`expmap0`, original/post-Mobius `mu`, A3/pre-`expmap0`, and A3/post-Mobius `mu`.
3. For each cache, store source commit/checkpoint hash, sample IDs, labels, feature location, geometry, number of views, augmentation configuration, and checksum.
4. Measure dynamic range, augmentation-stable mutual-kNN, and correlations against candidate independent geometry. Spearman here means rank correlation between a declared teacher relation and a declared student/geometry relation; it must never be reported without naming both variables.
5. Add `--teacher_cache` so matched runs consume the identical artifact.
6. Add `--trainable_scope` with at least:
   - `all`;
   - `hyperbolic_and_head` after inspecting exact parameter names;
   - `head_only` as the first drift diagnostic.
7. Emit a run manifest containing commit, command, teacher, seed, environment, GPU, timing, and diagnostic/non-final status.

Acceptance gate: tests pass, `beta_inter=0` does not construct/load an unnecessary teacher, and one-batch forward/backward is finite for every trainable scope.

## Phase 1 — learning-rate and drift screen

Single seed (`22`), maximum 8 epochs, original HyCoRe evaluation protocol retained for comparability, and an identical teacher cache within each matched pair:

| ID | LR | Scope | beta_inter | Purpose |
|---|---:|---|---:|---|
| P0 | 5e-4 | head_only | 0 | zero-increment retention |
| P1 | 5e-4 | head_only | 0.05 | incremental inter effect |
| P2 | 1e-4 | head_only | 0 | lower-LR retention |
| P3 | 1e-4 | head_only | 0.05 | lower-LR inter effect |

Run P0/P1 in parallel on two confirmed-idle GPUs, then P2/P3. Reuse the same teacher cache.

Diagnostic gate relative to the initialization and matched zero-increment control:

- no NaN/Inf and stable triplet coverage;
- stop obvious failures such as a persistent OA loss above roughly 1 percentage point after the initial transient, but do not treat 0.3 as a justified universal threshold;
- declared structural metrics do not collapse as in the first run;
- inter run improves at least two of three matched structural metrics (Spearman, satisfaction, ranking loss) across multiple epochs rather than one isolated point.

If neither learning rate preserves the original representation, stop method comparison and move to Phase 2. Do not increase `beta_inter` to compensate for an unstable optimizer.

## Phase 2 — trainable-scope diagnosis

Using the better learning rate from Phase 1, expand unfreezing in stages:

| ID | Scope | beta_inter |
|---|---|---:|
| S0 | hyperbolic_and_head | 0 |
| S1 | hyperbolic_and_head | 0.05 |
| S2 | all | 0 |
| S3 | all | 0.05 |

This phase identifies whether drift originates mainly in the Euclidean point backbone or the hyperbolic/classification layers. Add a frozen-teacher embedding-anchor loss only if restricted unfreezing still fails to retain structure; treat that anchor as a protocol control, not as the proposed hierarchy contribution.

## Phase 3 — confirmation

Only after one matched pair is stable, first introduce a deterministic validation split and repeat the selected configuration. Then:

- run seeds 22, 42, and 2026;
- report mean, standard deviation, paired differences, and confidence intervals;
- evaluate official ModelNet40 test data once for the selected checkpoint per seed;
- retain runtime, memory, triplet coverage, and failure cases.

## Phase 4 — two-axis hierarchy target

Before making a morphology-hierarchy claim, separate the target into:

- radial specificity `s_i`: a directed ordinary/prototypical-to-special ordering;
- angular kinship `k_ij`: a symmetric measure of shared attributes or branch membership.

The raw hyperbolic embedding may receive the radial loss. A same-direction equal-radius copy may be used only for the angular/LCA loss so radius cannot trivially solve kinship. Initial independent-signal candidates include:

- normalized Chamfer distance on consistently normalized point clouds, treated only as symmetric geometric closeness and not as a specificity direction;
- global spectral or shape descriptors;
- part-count/proportion descriptors where available;
- a fused graph whose edges must be stable under augmentation.

Teacher acceptance criteria should include within-class dynamic range, cross-augmentation mutual-kNN Jaccard, neighbourhood stability across seeds, and correlation with independently defined geometry. Frozen HyCoRe distance remains a self-distillation baseline.

## Questions for user review

1. Is the immediate paper claim intended to be “structure preservation/regularization” or “discovery of a new morphology hierarchy”? The latter requires the independent teacher in Phase 4.
2. Which observable signal should define radial specificity: attribute/part inclusion, prototype-to-outlier ordering, or another explicit rule? This choice materially changes the scientific claim.
3. Are ShapeNetPart/PartNet semantic parts acceptable as a later source of specificity supervision, even though the initial classifier is validated on ModelNet40?
