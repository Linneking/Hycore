# Next experiment plan — current review and historical drafts

## 2026-10-27：TS17 best代理角色审查已在8000完成

用户批准执行96计划。见[96结果](96_TS17_BEST_PROXY_ROLE_AUDIT_2026-10-27/report.md)：仅best e273、9840正常嵌入缓存后重建4.6视图，512×256代理保原深度，5seed×200批。两段谷值3.678；外段163代理仅1个样本角色入选，其余162未入选，不能对应整体pair层。活跃291；S25⊆C142⊆T291，同关系非碰撞pair更深73.89%/94.31%，差中位.034/.200，属于重叠使用偏好。深度3.028–4.142，无cap命中；纯类top4全512为370/512，活跃189/291；活跃top4覆盖668/9840，最大曝光14，类组成随机p95为4。8+4+4点云卡和8组图/9张CSV齐全，12合同、20逐关系复算及62文件取回SHA核验通过。读出3f25a31、CPU整理f08d729；旧结果和源权重未变，无网络前向/反传/优化器/BN/test读出，GPU已释放。当前无后续待执行训练。


## 2026-10-27：TS17 best代理角色分层审查计划

用户只要求设计TS17/B12 best e273的pair/triple分层检测，见[96计划](96_TS17_BEST_PROXY_ROLE_AUDIT_2026-10-27/plan.md)。8000现有53.8MB冻结权重及9840缓存SHA与85归档一致，CPU加载含net/512×256 proxy，CUDA未初始化。计划正常whole→4.6视图后选代理，保留proxy实际深度；同类pair/异类pair/triple交叠、深度箱形图/最近类角、全局top4覆盖/热点/双分母纯类比例及8+4+4形态卡；五seed各200批确定性主读出＋原Gumbel概率复核，重复对象/碰撞单列。已有best深度约3.03–4.14不预设触cap。尚无本轮角色结果、网络前向或GPU分析，不自动执行审查或新增训练。


## PartHier五问审查已完成，无待执行计算

用户指定8000的五问审查已完成并验收：[结果](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/followup_results/report.md)。源e4c757c，12CPU合同与原94固定读出精确回归通过；四权重旧缓存/队列只读，全部新增计算/绘图在8000，GPU释放。供给/径向/角度、曝光热点、intra输出梯度、ord/chain三状态、HIER失败组合及6例候选可行性已交付。下一步由用户阅读结果后决定，不自动重训margin、改变候选尺度或新增矩阵。

## 2026-10-26：五问冻结审查已批准在8000执行

用户明确同意[94后续方案](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/followup_plan.md)并指定8000，覆盖原8001部署地点。只执行已列五问，复用4份9840缓存与配套队列，不重训或重提；13输入文件复制SHA已核对。源实现和合成CPU检查就绪，正式固定200批读出及5图报告待实际完成。原稿旧“待审/只在8001”属于历史记录，不再阻止本次已批准执行；不得扩大训练矩阵。

## 2026-10-26：PartHier浅层选择与失败机制，五问方案待审

[94后续计划](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/followup_plan.md)按用户每问列实验与数据：候选供给/径向角度/热点；同对象三尺度夹角箱图与百分比分布；intra两项局部梯度；ord与chain分开三状态反例；HIER三项失败组合与原ijk几何。继续8001，复用4缓存及配套Q，正常冻结读出，未启动新增分析/训练。只有用户明确要求的旧图数量/百分比重复展示作重绘，科学统计不变。先审本计划，不追加训练矩阵或恢复用户删除的复杂对照。

## 2026-10-26：93总账TS23/TS24两组300轮结果已补全

用户报告8000两组结束并要求补93。20:03:56只读采集确认两条均300轮／60000更新／BN120000、exit0与保存验收通过，best/last存在、GPU0/1空闲。TS23（Ks10/Kp10/lambda.1）best e262 OA94.003241／AA91.827905、e300 OA93.719611、末10均值93.350891±.235880；TS24（Ks10/Kp20/lambda.05）best e277 OA93.598055／AA91.086626、e30092.463533、末10均值92.739060±.312178。93报告、CSV、注册表v4、来源与缺口同步，24正式记录现23完成/1中断/0运行；TS15未到300轮的空值不造补。其他22条配置/分数保持；保留用户在93中删去训练OA章节的当前结构，训练OA完整保存在CSV/JSON。本次零训练/前向/服务器写入；采集和11表/轮次/同轮AA/末10独立复算检查通过。

## 2026-10-26：94已完成，无待执行审查任务

用户“继续”后确认8001原后台已完成，取回并核验[94结果](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/report.md)：4权重全9840、5组图、8CSV和summary，不重复提取，不新增控制或训练，不转本地计算。chain0.2三边联合仅7.77%→4.02%，A/B选中关系部分达标但均偏小part。下一步由用户阅读结果后决定，不自动扩大消融矩阵；旧暂停交接已完成续接。

## 2026-10-26：按用户要求保存94审查，等待唤起

用户5小时额度告罄，要求保存且坚持8001。[交接](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/HANDOFF_2026-10-26.md)：4权重全9840冻结嵌入已验收；8001现有readout后台保留，最后running/metrics，尚无完成报告。本地没有启动统计。停止主动工作/监看，不新增任务；唤起后先检查现有任务，继续8001统计和报告，不重提已有合格缓存。

## 2026-10-26：PartHier v1简化嵌入审查已批准执行

用户在简化[94方案](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/plan.md)后明确“开始检测吧”，现已授权8001冻结采集与分析。仅4主权重、全9840对象建立类别中心、正常ijk选择A/B；五项主图及原intra附表。只读配套Q2048，不新增训练或复杂控制；独立运行目录、空闲GPU2/3，原权重与BN保持不变。先完成本次简化审查，再讨论后续实验；准备状态不冒充结果。

## 2026-10-26：8001恢复，93总账缺口已补齐

用户明确8001连接恢复并要求补全统计。本次12:56只读原始manifest/best轮/末轮/末10轮及进程列表：TS15实际保存54连续轮、best e52 OA91.855754/AA90.107554，当前无对应进程，manifest旧running属残留，归为已中断，不恢复训练；TS18 PartHier已300轮，best e201 OA92.990276/AA90.861619、e300 OA88.006483、末10均值89.444895±1.609896，e300在线train99.515625。TS19/20末10均值/SD及TS16/17 best同轮在线train精确值均补齐。93报告/全部CSV/注册表v4/恢复采集来源/缺口表同步更新；24正式记录现21完成、2运行、1中断，无状态未知。8000顺带更新完成前缀TS23 e101/TS24 e98；仍不填e300。剩余19空字段均为未完成终点或协议未记录（含PartHier clean/eval），没有连接失败造成的缺口；10个Markdown表及原14条已完成分数/原15条身份保留检查通过。未新增GPU计算、前向、训练、重启、停止或文件改写到服务器。

## 2026-10-26：V9至今实验总账93已统一补录

用户要求沿75风格统计全部V9及后续实验、统一新增命名并补齐缺失数据。见[93总账](93_V9_EXPERIMENT_ATLAS_2026-10-26/report.md)：保留TS01–TS15/B01–B10，新增TS16–TS24/B11–B14；24正式记录（20完成、2进行中、2后续未知），另列5条机制实验。B05严格仍SH32×307；B05几何的CB64改称B12、CB32改称B13。CSV/命名映射/注册表v4/缺口表及脱敏来源齐全；旧75报告的用户未提交修改未动。采集截止10:39:59，最新两组各完成17轮，仅前缀best，不填e300。8001拒绝连接：TS15尾段、TS18首轮后、TS19/20末10轮、TS17 best同轮在线OA仍显式缺失；其余可取得的完整成绩和训练OA已补录。只读现有文件、无GPU/训练/BN干预；数据单位、轮次、原15条身份/分数保持及9个Markdown表检查通过。没有扩大实验或自动等待服务器恢复。

## 2026-10-26：用户批准继续B05 CB32 A1两组单变量消融

8000 GPU0样本K10／代理K10／lambda0.1；GPU1样本K10／代理K20／lambda0.05。相对已完成91第二组K10/K20/lambda0.1，分别只改代理K和lambda。共同严格B05固定壳新HIER、CB32有放回16类×2、AdamW.005/1e-4、seed4780、fresh300×200、原TS协议；直接授权启动，不续旧权重或扩大矩阵。见[92配置](92_B05_CB32_A1_K10_WEIGHT_FOLLOWUP_2026-10-26.md)。源码061d86f已推送GitHub并以同一发布提交Git增量包fetch/pull --ff-only同步至8000。两组各4CPU、训练数据SHA／相同初始化及真实2×2短测通过，实际K、lambda、AdamW、BN8与完整保存状态验收通过。北京时间10/26 10:10:30两组均fresh进入正式300×200并已有真实更新；GPU0/1绑定正确。私有可视化目录b05_cb32_a1_followup_20261026保存dispatch/状态/验收/manifest。不重复提交，不逐轮监看。

## 2026-10-26：用户批准B05 CB32 A1两项完整消融

最新修订：8000 GPU0样本K5／代理K10、lambda0.5；GPU1样本K10／代理K20、lambda0.1。两组均10/25严格B05有放回CB32新HIER A1、LR.005/eps1e-4、seed4780、fresh300×200，其余保持86协议。用户直接启动授权覆盖旧待审矩阵限制，沿用历史TS选模；无新增H0/CB64。见[91配置](91_B05_CB32_A1_K_WEIGHT_ABLATIONS_2026-10-26.md)。第一组联合调整两种K，不能称两种K各自单变量；第二组仅改系数。源afb01ee已推送并在8000以Git增量包fetch/pull --ff-only同步。两组各4CPU及真实2×2短测／实际K、系数、AdamW、BN8与保存状态均通过；北京时间10/26 00:06两组分别在GPU0/1 fresh进入正式300×200，已见真实更新。私有可视化目录b05_cb32_a1_ablations_20261026保存dispatch、状态、验收与manifest；10/26 09:27核实两组均300轮／60000步、BN120000、exit0和完整保存验收通过；原PID已退出，GPU0/1空闲。K5/10 best93.598055、AA91.486043、e217、末轮92.990276；lambda.1 best94.043760、AA91.632557、e280、末轮93.517018。best/last各约162MB存在，完成manifest/状态/末轮metrics私有备份。不重启、不新增训练。

## 2026-10-25：90单条早期复跑与几何审查完成，不重复提交

用户授权的B05 CB32旧HIER A0前10轮已于21:14完成实际10×200／300cosine、逐轮保存及85全量审查，GPU释放，见[90结果](90_B05_CB32_A0_EARLY_AUDIT_2026-10-25/report.md)。14缓存本机SHA核验、33CPU和实际GPU／80例回归通过，原e1/e5与复跑分开。当前无待执行训练矩阵或PartHier审查授权，不因历史running快照重启任务。

## 2026-10-25：用户已批准单条B05 CB32 A0十轮复跑及审查

仅10月25日CB32旧HIER A0 .01/eps1e-8；用户撤回PartHier审查。原只存e0/e1/e5，明确同配置fresh10轮、每轮完整权重；seed4780／K10-K20／300调度及原评估随机流保留。按[90](90_B05_CB32_A0_EARLY_AUDIT_2026-10-25/plan.md)用8000空闲单卡短测后fresh，结束按85冻结全9840训练对象分析e0..10及原e1/e5参考；12新CPU检查和18文件备份核对通过，尚不宣称GPU或审查完成，不重复排队其他实验。

## 2026-10-25：88两组十轮检测完成，不重复提交

用户要求检查，已核实8000两组各2000步／BN4000，16:17:26正常完成、GPU释放、本机81文件SHA一致，见[88结果](88_B05_EPS_ONLY_10E_2026-10-25/report.md)。同LR.005，仅eps改变：E8末轮500/512越cap，E4为0/512；E4第4轮末起无cap，第6轮起逐步也无新增。数据支持eps改变实际更新及方向，HIER反传仍存在；没有test或分类收益判断。初始化为高斯缩放而非严格全体4.14、起始4个越cap已澄清。当前两组检测授权已执行完，不自动扩大训练矩阵或检查8001。

## 2026-10-25：用户批准8000双卡B05 eps单变量十轮审查

用户已明确两条均代理LR0.005，仅eps1e-8／1e-4；覆盖此前单卡建议，采用双卡CB64、K20/K20、seed4780、10×200和300轮调度。新增代理深度移动百分比分布；记录每步梯度、更新、越cap／返回和每轮首batch主干梯度，零评估／选模。按[88计划](88_B05_EPS_ONLY_10E_2026-10-25/plan.md)实施：源码c8c05bc、本机／服务器各13CPU及两条真实双卡2×2验收通过，初始化／首批／首步梯度相同；15:44:05已在8000 GPU0/1 fresh启动E8，E4及CPU汇总自动顺序执行。完整十轮尚未完成；不得重复提交。当前授权不含新增完整训练或其他实验矩阵。

## 2026-10-25：85四组审查已完成

用户指定8000后，已完成16份同9840 clean/eval端点统计和主／辅各22幅PNG/SVG，见[85结果](85_EMBEDDING_AUDIT_2026-10-24/report.md)。固定80例历史batch回归坐标／深度误差0、21CPU通过、16本机raw备份哈希一致，GPU释放，无新增训练／test。A1代理回内和样本深度次序改善成立，代理内部严格次序仍低；原生末轮flower_pot–plant局部纠缠须与分类因果分开。此前“85仅计划／等待连接”现为历史状态。未追加训练、BN干预或检查CB32队列；后续用户要求另行执行。

## 2026-10-25：已批准B05单卡A1／A0分别排8001 GPU0／1

用户已确认两组为新HIER A1与旧HIER A0，覆盖旧默认双卡batch配置；共同单卡16类×2、sampleK10/proxyK20、seed4780、300×200和严格B05固定壳，其余保留。仅代理初始LR/eps不同；不新增H0。用户最新明确挂好完整后台任务即结束，不等待GPU释放/验收或逐轮监看。按[86](86_B05_SINGLECARD_A0_A1_QUEUE_2026-10-25.md)实施，只在各指定卡完全空闲后自动独立短测并fresh正式训练；8项本机CPU合同通过，尚未执行的CUDA验收不得宣称完成。提交状态另存私有交接，切勿重复提交。

## 2026-10-24：先完成四组嵌入审查计划，等待用户通知恢复连接

用户确认TS13/TS14/新B05-A1/新原生A1为主四组，允许有用历史对照；两服务器GPU满，8000当前SSH拒绝。最新指令“你先完善检测计划，我去搞定服务器，能链通后再告知你”控制当前阶段。见[85](85_EMBEDDING_AUDIT_2026-10-24/plan.md)。目前仅计划与工具/冻结副本准备，无新训练、BN校准、四组全量提取或自动重连监控。完整模型回归和TS13缓存补齐是继续采集的前置项；等用户通知后在原授权审查范围内继续，不追加训练矩阵。

## 2026-10-24：79两条seed4780固定壳HIER已完整结束

原生／严格B05两组均300轮、60000步正常完成。只读复核与讨论见[83](83_S4780_NATIVE_B05_COMPLETION_2026-10-24/report.md)：best OA92.747/93.476，e30089.506/93.112，末10轮90.312±1.257/92.986±0.204；best和终点代理均0/512越cap。仅证明本组合在完整训练中避免贴壳终态，未确立HIER净分类增益；现有历史SH32或不同seed/随机流的CB64不等于匹配H0。冻结BN/方向检查与同协议H0为后续建议，本次不执行新前向或新增训练；不影响单独已授权运行的PartHier v1。

## 2026-10-24 09:45：PartHier v1已按后续授权改8001启动

用户明确“先用8001”，并要求提高显存利用率。source82478cd采用部分欧氏主干重算，仅200/400重算、600/whole保留，数学配置与global64/FP32/BN4不变；33CPU/Gloo、真实CUDA与独立2×2、满Q/whole1024均通过。最终每卡27.79GiB allocated/28.21GiB reserved、空闲至少2.27GiB，有界步时由1.527降至1.343s。09:45:48在8001两空闲卡fresh启动单条300×200；09:50首轮200更新/2468 test/BN800/有限RSGD与保存状态/300轮调度/参数及Q一致性、初始化和数据身份验收通过，已进e2后台继续。见[80](80_PARTHIER_V1_PLAN_AND_LAUNCH_2026-10-24.md)。不继续增加checkpoint候选、Q0或完整矩阵，不重复恢复8000或提交另一完整任务。只保存交接，不逐轮监看。

## 2026-10-24：PartHier v1已获直接授权，待恢复8000完成验收

用户本轮明确授权自主决定后续实现/短测，并将尽可能还原PartHier的单条完整实验挂8000，已答复混合尺度祖先选择、完整Q2048、三级intra各0.01相加。该直接指令覆盖本次旧“等待计划审阅”；不再索要配置批准。原生HyCoRe c1、无新增cap/GS、同中心200/400/600、seed4780、非循环CB64、300×200、全9840/2468 TS。每次短测后读五小时额度，剩余低于30%不追加可选对照；必要可运行验收不能以未通过替代。

当前实现/CPU与Gloo已完成；普通四pass双卡短测显存不足，CPU暂存重测状态随后因8000 SSH无响应而无法核实，正式训练未启动。后备采用BN/RNG保持checkpoint，恢复连接后先核实本次旧进程/日志和空闲GPU，再做同源码真实CUDA等价检查、双卡2×2与满Q/whole1024容量验收，最后fresh正式训练。备用四卡忙，不中断。详见[80](80_PARTHIER_V1_PLAN_AND_LAUNCH_2026-10-24.md)。不自动扩大完整消融矩阵。

## 2026-10-23：两条seed4780固定壳HIER已获启动授权

用户选择“严格保留B05”（c0.1、cap2.3、既有基础GS、500/N与2），要求两条均挂8001，随后明确“启动”。此直接指令覆盖下方断电期间仅设计/不连接的旧约束，授权实现、验收、提交推送、服务器快进冻结及fresh启动。只执行原生c1＋Hshell-A1、严格B05＋Hshell-A1两条；不自动增加H0或其他矩阵。统一seed4780、非循环有放回CB64、300×200、全9840/2468逐轮TS、无早停、A1代理LR0.005/eps1e-4。使用共用连续随机流训练骨架与各自已验证的base/geometry/audit/telemetry，B05从cap后基础GS前分叉。详见[79：执行与验收](79_S4780_NATIVE_B05_LAUNCH_2026-10-23.md)；最终启动状态以该报告实际验收为准。

已上线：源码5517661，两组各18项CPU、各自Gloo与独立2×2训练限定短测通过；18:17:36在8001 GPU0/1（原生）与GPU2/3（严格B05）fresh生产。真实短测AdamW/调度器/BN8/有限状态已验收，首轮记录已可读。用户随后要求“短测通过就结束吧”，不再追加首轮专项采集或逐轮监看；任务后台完整300轮继续，不创建自动监控或新增训练。

## 2026-10-23：仅设计；统一seed4780与非循环类别均衡64

最新约束覆盖下方旧未来默认：所有新实验seed4780，不再使用类别循环64；采用有放回类别均衡global64。服务器断电，用户只授权设计讨论，不连接/实现/排队/启动/自动恢复。历史运行seed、结果和不可变身份保持。

拟议8001原生HyCoRe c1＋新AdamW固定壳HIER；8000 c0.1适配配方＋同组件，详见[76](76_S4780_FIXED_SHELL_BASELINE_DESIGN_2026-10-23.md)。B05基础GS是HIER移植hook；若主pipe绝不含该hook，8000应改称B04式GS0，不能宣称严格B05。此核心分歧待用户明确。每个底座增加匹配H0、统一300×200及共同随机流均为设计建议，尚未批准执行。仅两条HIER-on不足以量化HIER增量，也不能单独隔离曲率效应。

## 2026-10-22：V9-HIER AdamW联合候选单条完整实验

用户本轮请求将十轮中修改的AdamW参数进行一次完整V9-HIER实验。按[74：完整配置与验收](74_V9_ADAMW_FULL_RUN_2026-10-22/report.md)准备唯一fresh300×200：代理初始LR0.005、eps1e-4、最低LR0.0005，其余原生c1 V9/seed22/global64/300轮余弦/TS保持，不续训十轮权重。先CPU和独立train-only短测，核验真实保存的optimizer和初始化，再从头主训练；优先原指定服务器的完全空闲双卡，不中断他人。不新增margin、分类头、循环采样或其他矩阵。状态须以74后续实际启动记录为准。

已实际启动：38项CPU+Gloo及独立2×2 train-only短测通过，源码8408273，8001 GPU1/2于21:28:18 fresh主训练。21:31:26首轮完整200更新、2468test、有限best/last/e1、BN400/代理同步、实际optimizer与300轮scheduler等16项验收通过；完整300轮在后台继续，不因早期test表现停止。后续无需重复提交。

## 2026-10-22：V9代理LR/eps十轮联合候选已授权并完成

用户明确要求将V9-HIER代理初始AdamW学习率改为0.005、eps改为1e-4，从头运行10轮观察代理；优先指定主服务器的两张完全空闲卡，不足则用备用服务器。该指令授权本次版本化实现、独立短测与唯一10×200步诊断，不再等待确认。详见[73：配置与观察口径](73_V9_LR_EPS_10E_DIAGNOSTIC_2026-10-22/report.md)。

对照为71的新鲜10轮机制轨迹（e1/e5/e10越cap229/444/502），不是69历史训练。只改变代理初始LR与eps，最低代理LR仍0.0005，两调度器保持300轮余弦时间尺度；主干、初始化seed22、global64类别采样、几何、GS、cap、lambda0.5、margin/tau0.1及分类头/标签平滑保持。无test/validation前向、选模或早停；单独fresh 2×2短测后重新初始化正式诊断。联合候选不能分离LR/eps各自因果效果，也不能仅凭cap减少声称层次或分类改善。

完成结果：18:03:59正常结束2000步、80关键快照，所用GPU释放；11项完整与7项补充验收通过。新配置e1/e5/e10越cap62/3/0，曾越界122个均回内；末态深度中位3.576。关系损失继续下降但同轮较高，实际更新明显减小；sample与whole径向分离恢复，代理内部次序和祖先使用存在得失。该组合是具备前十轮稳定证据的后续候选，不是已验证的完整训练方案；本次不自动启动长程、单因素或margin实验。

## 2026-10-22：Cars/V9分布与更新尺度只读审查完成

用户授权使用已有权重检查类内/类间角度、参考中心和数值迁移力度，结果见[72](72_HIER_CARS_V9_ANGLE_AND_STEP_AUDIT_2026-10-22/report.md)。末态类内聚集确实更强，但首步真实训练尚未形成紧类簇，多数HIER pair跨类；因此不把分类聚集写成全部代理贴cap的充分原因。首步LR/eps固定状态控制支持实际更新相对cap余量过大，lambda缩小不同比减小Adam步长。后续数值候选优先eps单因素，再独立核对代理LR，并持续记录首次越界/回归及关系优化进度；显式径向次序属于方法设计的另一轴。这里仅记录候选与证据，没有启动新训练、完整矩阵或更改已有配方。

## 2026-10-22：V9代理快速贴cap的有界诊断已授权并完成

用户直接授权8000从头5–10轮观察代理模长。独立10轮×200步、300轮原调度、完整训练池与原V9 c1/HIER c0.1/固定4.6视图/seed22保持；不读test或选模，不改已有正式任务。运行源码1ba0a8e，2000步与80关键步快照完成。后续固定状态CPU一步干预、同GPU保存Gumbel重放均只用于机制定位，未新建完整训练矩阵。结果见[71](71_V9_PROXY_CAP_CAUSE_2026-10-22/report.md)：早期ST选择路径能转外；越界后前向cap抑制当前径向反馈，历史m带来持续外移。只改eps是后续单因素候选，显式径向目标/参数约束属于另一路方法设计，均没有由本次诊断自动授权完整训练。原版SOP反例及历史复现差异必须保留。

## 2026-10-21：原版c0.1底座循环300步双卡配对已授权

用户确认在原版c0.1/cap2.3/GS1/500_N与2/seed4780底座上做两组双卡实验：每轮300步、300轮、类内打乱后顺序循环且游标跨轮保留；HIER组使用固定深度4.6视图，基础与HIER两路径各一次GS。指定更耗时的HIER组部署8001，无HIER对照部署8000。保持全9840训练/2468 test逐轮选best，无提前停止。本次直接授权覆盖旧V系列默认seed22，且不将新组与既有c1 V9混名。具体配置与检查见[67](67_ORIGINAL_C01_CYCLE300_HIER_PAIR_2026-10-21.md)。上线完成：源码c0d2c89，两服务器各39项CPU及Gloo比较、train-only2×2短测通过；HIER组13:02:54在8001 GPU1/2、无HIER组13:03:09在8000 GPU0/1从头启动。两组首轮300更新/完整test/有限best-last-e1/BN600及配对输入验收通过，后台继续300轮。

## 2026-10-21：V9-B64完整同协议对照与单卡官方复现已授权

用户明确表示理解双卡设计并授权设计、验收和挂起V9-B64。保持V9原生c1/无额外cap或GS/1000/N与4、seed22、global64类别均衡、300×200及完整9840/test2468逐轮选best，HIER视图/损失/代理零构造零调用。只新增这条双卡控制，见[66](66_V9_B64_CONTROL_AND_OFFICIAL_REPEAT_2026-10-21.md)。原V8-B64 TS已完成但几何不同，不重复。用户随后明确确认单卡官方c1/seed4780/原版配置从头300轮复现，授权CPU与train-only GPU短验收后fresh生产；两条使用8000空闲GPU，独立新目录，不追加先前seed22高表现组复核。已上线：B64源码89710b5于00:55双卡从头启动，官方源码21f6c69于00:59单卡从头启动；各自CPU/短测及首轮完整评估/检查点验收通过，后台继续完整300轮。

## 2026-10-20：V9各自曲率方案已获直接授权

用户要求设计并部署8001，随后明确继续双卡类别均衡64、300轮。固定原生HyCoRe c1/1000/N/4、不额外cap或GS；HIER c0.1仅看深度4.6的保方向whole视图、分支GS，代理保留官方cap/GS/初始化/AdamW。seed22、完整9840训练和2468 test逐轮选best按用户未来协议；本次只启动此一条300×200主实验。授权包括版本化实现、CPU合同、训练限定2×2短测与从头正式上线，不需再次等待旧计划审阅。方案及验收见[65](65_V9_DUAL_CURVATURE_PLAN_AND_LAUNCH_2026-10-20.md)。下方“曲率待定”为此前历史状态，已由本条覆盖。上线完成：30项CPU、双进程梯度及双卡2×2短测通过；源码b5d2d8b，19:44:55从头正式启动，首轮200更新/完整test/有限检查点验收通过，后台继续。

## V9初步idea已记录，曲率组合待定 — 2026-10-20

用户认可“原生HyCoRe＋等深度HIER分支”为V9初步设想：基础CE/intra不加共享cap/GS，HIER仅看保留方向、深度4.6的whole视图，代理保留自身HIER映射。统一c0.1与基础c1/HIERc0.1两种曲率组合尚未定，不擅自选定或启动矩阵。见[64：idea](64_V9_INITIAL_IDEA_2026-10-20.md)；[63](63_EXPERIMENT_NAMING_AND_GRADIENT_SCOPE_2026-10-20.md)补充官方HIER三路dataflow，区分PA类别代理与512层次代理。仅文档记录，无代码/训练修改。

## 2026-10-20：统一未来选模/seed，并授权原版GS补充

用户明确指定：今后正式比较使用官方9840训练/2468 test逐轮选best；原版HyCoRe修改线seed4780，版本化V系列seed22。历史验证选模实验保持真实标签，不追溯改成test选模；无早停授权。该直接指令覆盖本文件及既有通用说明中的旧validation偏好。

已直接授权[62：原版c0.1+cap+GS+半margin](62_ORIGINAL_C01_CAP_GS_HALF_MARGIN_2026-10-20.md)，原先在8001排队，现按用户15:59更正取消该后台并改到8000单卡从头300轮。只在完全空闲卡上先train-only短测再正式运行；资源占用时后台等待，不启动其他矩阵。命名改为显式C/CAP/GS/M/选模/seed，见63规范。 更正部署已完成：8001无训练并取消；8000通过37项CPU与2×2短测，于16:02在GPU2从头正式启动，源码e20cf8c，完整300轮。

## 2026-10-20：原版c0.1＋cap的margin减半组已获直接授权

用户要求新服务器补充原版HyCoRe c0.1＋cap，margin改为500/N与2。按[61：补充配置](61_ORIGINAL_C01_CAP_HALF_MARGIN_2026-10-20.md)保留官方launcher seed4780、cap2.3、无InverseMetric hook、原版shuffle32/307步/300轮、完整9840训练与每轮2468 test选best。该直接指令授权版本化实现、独立CPU/GPU短验收及从头正式启动；不修改正在运行的V8或已完成原版组，不追加其他条件。两个正则系数仍各0.01。 上线：32项CPU合同及2×2训练限定GPU短测通过，11:50在空闲GPU1从头启动300轮；初始化和训练数据身份匹配已完成cap对照。源码e40a071，正式首轮验收记录见61。

## 2026-10-20：三组c0.1/cap/nohook补充训练已获明确确认

用户确认[60：三组单卡方案](60_C01_CAP_NOHOOK_ABLATIONS_2026-10-20.md)：V8测试集版B0-32仅关闭InverseMetric hook、其余保持seed22/cap2.3/gamma500/delta2；原版HyCoRe仅全链c=0.1，及其仅增加cap2.3的配对组，后两组均保留官方launcher seed4780/gamma1000/delta4且不加hook。三组各一张新服务器空闲卡，完整9840训练/2468 test、shuffle32/drop-last/307步、300轮、每轮test选best。该用户直接确认覆盖本次旧有validation默认约束，授权版本化实现、CPU/GPU短验收和从头正式启动；不追加其他实验矩阵。短验收只用训练数据，正式结果与旧8856/984协议分开。

上线完成：28项CPU合同与三组2×2 train-only GPU短验收通过，三条均在独立目录从头运行。01:46首轮完整test/307步/权重保存全部验收，V8/原版c0.1/原版c0.1+cap已各完成6/2/2轮，之后继续完整300轮，不以当前test提前停止。实际启动与源码身份见60；不追加新矩阵。

## 2026-10-19：whole/part逐对象审查完成，尚未改变训练方案

用户确认原版与旧8856/984、新FULL9840 V8-HIER64均检查best/终点。结果见[59](59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/report.md)：500/N、2与1000/N、4源码实现明确，但clean/eval径向完整达标比例普遍低；原版也不保证逐对象兑现。恢复whole壳不能单独解决N600 part贴cap和径向间隔；未来候选必须同时验收N分层的Rhier/Rcontr及联合满足率。当前批BN面板不同于完整训练，不直接改正式统计；本次无test选条件或新增完整训练。8001两条FULL9840对照均已完成300轮，排队阶段结束。

## 2026-10-19：四模型密度与BN兼容性冻结审查完成

用户确认使用旧同8856/984已完成版本比较，补齐B0 best/e300，并完成原版HyCoRe参照与三V8四条件BN试验，见[58](58_V8_WHOLE_DENSITY_AND_BN_REVIEW_2026-10-19/report.md)。HIER相对B64 e300更深且窄，不能将whole分散单独归因HIER；同壳也不保证浅代理。借原版最终BN在三模型均错配；各自whole-only校准恢复壳但把part推深，故不直接覆盖正式统计。后续候选应同时验收whole/part逐对象径向间隔、方向保持和代理结构，whole/part角色统计分离仍为冻结检验候选；本次未实施该方案，也未启动新的完整训练矩阵。12个前向条件已完成、参数及状态恢复，GPU释放，无test选条件。

## 2026-10-18：代理cap与whole球壳的受控机制诊断已获授权

用户要求使用新服务器空闲GPU，自主查明raw代理模长增长、梯度/Adam作用及whole不维持球壳的原因，并比较原版权重，直到有一定结论。该指令授权副本静态干预和有界短轨迹，见[57：诊断索引](57_V8_CAP_AND_WHOLE_CAUSES_2026-10-18.md)。只使用空闲卡，不改主训练，不以test选参数，不自动启动V9完整实验矩阵；下一轮配置仍需根据机制结果讨论。

诊断已完成：早期Adam历史m直接促进越cap后增模；匹配1024步代理eps1e−4显著减少越界但优化较慢。固定晚期模型的BN库干预支持共用part/whole统计参与eval whole内缩，匹配40前向的whole→whole/part→whole库贴cap99.22%/1.56%。优先候选为代理eps单因素和whole/part统计分离的冻结验证；不能把恢复壳或减少cap直接等同于结构/分类收益。完整报告57列明原版AMP/FP32差别、SOP反例、part/方向未测等边界。未授权或启动额外完整训练。

## 2026-10-18：完整9840协议的两组V8对照获直接授权

用户要求在旧服务器按空闲情况安排单卡V8-B0-32 shuffle及双卡V8-B64无HIER，维持主实验官方9840/2468与每轮test选模；完整300轮。V8几何、margin、seed22、优化器保持；B0为307步/轮，B64为200步/轮。允许后台独立等空闲后短验收并自动启动，不要求三张卡同时空闲。见[56：两组对照及排队协议](56_V8_FULLTRAIN_CONTROLS_2026-10-18.md)。用户直接指令覆盖本次旧版validation/test默认约束，不启动额外矩阵。

## 2026-10-17：完整9840训练池与test选模的单条V8已获直接授权

用户要求两卡重跑V8-HIER64，使用原版HyCoRe全部9840训练对象，每轮官方2468 test选best，batch保留类别抽样双卡global64；随后指定新服务器执行。该明确授权覆盖本次旧有默认validation/test约束，无需再次审阅等待。300×200步、seed22与既有V8几何/损失/优化设置保持，独立版本和新目录。见[55：本次协议及验收](55_V8_FULLTRAIN_TEST_SELECTION_2026-10-17.md)。测试不反向，但按test选模的结果单独标记；未启动其他矩阵。


## 2026-10-17：现有结果复核完成，无新训练

用户明确授权汇总V7/V8及原始HyCoRe，并补做固定权重test。本轮五份best复测完成，见[54](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17.md)：原始为e229而非e221；B0-32-C01已经完成，test OA92.6256%。它继承V8几何与独立val选模，不能当成完全原版复现；V7/V8划分一致，和原版9840全训练/best-test不同。若后续要归因V8几何适配，仍缺同8856/984、seed22、shuffle32/276步、验证选模的c1原始几何参照；原seed4780模型不能替代。本条只记录证据缺口，不授权或启动新训练。原有冻结机制检查建议保持。

## 2026-10-16：V8主/双卡对照审查完成，尚无新训练矩阵

结果见[53](53_V8_SYSTEM_AUDIT_2026-10-16.md)。V8休眠改善（sample祖先512），热点仍在（NN4覆盖231、最浅1%占72.17%槽）；相同协议HIER开关没有OA收益，固定e300形态富集有局部正差但其他指标不一致。接下来优先冻结检验whole径向离散、代理共同方向与forward cap后的径向梯度；扩大覆盖不能单独视为结构收益。单卡V8-B0-32-C01仍运行，本次未伪造其最终审查。用户本轮授权的是审查，未启动任何新训练矩阵；新训练方案仍需审阅。

## 2026-10-15：新服务器三卡用于V8两个对照，旧主训练继续

执行：两组11:41后台启动，首轮完整验证/checkpoint通过，代码85fc812，任务`v8_controls_20261015_113942`。真实GPU smoke均通过；两卡B64与单卡B0-32共享V8几何，HIER严格关闭。见[52](52_V8_CONTROLS_ACCOUNT_AND_LAUNCH_2026-10-15.md)。

用户已审阅并修订：旧服务器已经启动的V8继续；新服务器两卡运行V8-B64（相同V8协议，仅关闭HIER目标），另一卡运行V8-B0-32-C01（batch32 shuffle，同样使用V8的c=.1、cap2.3/hook、gamma500/delta2，关闭HIER）。公共8856/984验证选模、e300、结束后一次test。具体配置与归因边界见[51：V8对照计划](51_V8_CONTROLS_PLAN_2026-10-15.md)。已授权实现、验收和后台启动；不修改旧主训练。

## 2026-10-15：V8配置已批准，授权实现、验收及后台启动

完成更新：有效任务`v8_hier64_20261015_022043`已在独立后台等待两张完全空闲GPU；代码14f8a72，9项合同与CPU双进程梯度验收通过，真实训练/验证划分核对。02:23:30尚未GPU smoke或生产更新。空闲后自动2×2 smoke，再从头300×200步，无需用户手动启动。详细状态与私有工程入口见50。

用户已同意上一轮明确的优化配置，并要求“开始编码挂到服务器上进行训练，问题自行解决，挂起后退出”。单卡16类×2例，两卡类别不重复；c=0.1、最终共享mu的cap2.3及一次hook，gamma500/delta2，两intra系数各0.01，HIER系数0.5、W0；主模型RSGD0.1→0.005、代理AdamW0.01→0.0005，300轮余弦。该授权覆盖下方仅数值候选的等待状态，允许独立smoke和唯一完整e300主训练。实现和真实启动状态见[50](50_V8_IMPLEMENTATION_AND_LAUNCH_2026-10-15.md)。只使用两张完全空闲GPU；不足两张时允许任务后台等待再自动验收和启动，不能中断其他进程。验证选模与最终一次test保持。

## 2026-10-15：V8官方HIER几何适配，数值可行性已核查

用户提出让HyCoRe适配官方HIER的c=0.1、切空间cap2.3及梯度放缩，并先要求源码与数值核查。结果与候选见[49：V8几何和margin可行性](49_V8_HIER_GEOMETRY_AND_MARGIN_FEASIBILITY_2026-10-15.md)。cap2.3对应最终共享表示深度4.6；原part–whole margin=1000/N需要同时审查Rcontr的margin4。按深度预算倍率0.74043缩放不能排除两项目标在小part时的联合冲突，候选是只对HyCoRe的两项margin共同乘0.5（gamma500、delta2），保留官方HIER margin/tau0.1。全链曲率、最终mu的cap/hook位置、代理初始化及官方数值距离均需明确；采样/优化协议与几何变更分开记录，self-k修复保留。CPU解析核查完成，候选尚未实现成训练入口；本条不授权启动新实验矩阵。

实验展示名统一遵循[33：实验命名规范](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md)。旧文件名、运行ID及代码字段保留用于溯源；本次只规范称呼。

Status: **唯一V7已于2026-10-07 14:43:33完成300轮；16:00只读核查通过，GPU1/2释放。验证选中e187（OA93.8008%），最终test OA92.2204%、AA89.7204%。全程代理数值投影计数0，参数cap保持有效。完成记录见[37](37_V7_COMPLETED_2026-10-07.md)，配置与启动身份见[36](36_V7_PROXY_SAFETY_DECISION_AND_LAUNCH_2026-10-07.md)。未启动新实验；下方为历史计划。**

## 2026-10-07：训练后系统检测计划

用户已审阅并批准[38：HIER训练后全系统检测](38_HIER_POSTRUN_AUDIT_PLAN_2026-10-07.md)，实现及V5/V6/V7、matched B64串联审查已执行。结果见[40](40_HIER_SYSTEM_AUDIT_RESULTS_2026-10-07.md)，执行记录见[39](39_HIER_SYSTEM_AUDIT_EXECUTION_2026-10-07.md)：31个真实检查点、全8856训练对象固定输入回放、代理休眠与固定query/Gumbel/梯度、独立形状检验及收益账本。V7数值约束有效，但热点覆盖与独立形态收益未全面改善。各源冻结、零优化更新、不新增test前向；工具独立位于`tools/hier_postrun_audit/`，不改生产训练路径。检查点文件不等于每轮都保存了嵌入；不自动启动下一轮主训练。

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
