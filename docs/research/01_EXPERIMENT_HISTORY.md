# Experiment history

## 2026-10-27：用户授权TS25，TS23初始化系数0.9→0.7完整消融

用户选TS23/B13而非TS20，明确在8000跑完整300轮。新别名TS25/B13，CB32有放回16类×2／200步、Ks10/Kp10、lambda.1、A1 AdamW.005/1e-4、seed4780；唯一配方变化为高斯初始化系数.9→.7，保留原方向及模长差异，不额外等深归一。见[97配置](97_TS25_TS23_INIT07_FULL_2026-10-27/run.md)。GPU1先核查空闲，独立train-only2×2短测和完整保存状态通过后fresh300×200，保留原TS评估及300余弦；不复用短测权重。6本机合同已通过；截至本条仅准备，实际启动以后续验收为准，避免重复提交。注册表已预留TS25，成绩不填造。


## 2026-10-27：用户澄清只看TS20原0.9短测，8000现有前5轮已核对

用户明确“短测使用0.9系数，不是新实验”，服务器改8000。本次直接CPU只读90号既有同配置0.9短测e0–e5，初始代理与seed4780高斯0.9公式逐位一致；A0 LR.01/eps1e-8、CB32/K10-K20、300调度/每轮200步核对通过。轮末越cap为e0–e5：4/246/406/503/510/511，共512；原TS20真实e1/e5为241/508，不能与复跑混接。见[前5轮复核](90_B05_CB32_A0_EARLY_AUDIT_2026-10-25/FIRST5_RECHECK_2026-10-27.md)。无新训练、无GPU分配；0.7准备草稿未提交或执行，已私有归档。当前不排队0.7新实验，也不重复运行已有0.9短测。


## 2026-10-27：TS17外层代理三步审查已在8000完成

用户批准后完成[外层代理审查](96_TS17_BEST_PROXY_ROLE_AUDIT_2026-10-27/outer_report.md)。同类去重40类×100对，实际内/外候选较大夹角中位60.67°/81.78°、评分差1.404；只换深度/方向后差1.092/.225，均未翻盘。初始方向角与前5轮样本pair使用rho−.130，初始深度rho−.697；最浅/最深128初始代理中，最终外层冷落分别1/101个。冷落162个初始深度中位4.295，活跃内段290个4.027，起始方向角89.45°/89.39°相近；前5轮样本pair份额.121%/98.047%，e273内部pair仍使用冷落组17.428%。更支持关注初始深度不均及早期机会差异，不宣称初始化方向唯一因果。e0/e5原件从8001复制SHA相同，正式1632对象冻结前向、best32缓存逐坐标一致；首次设置不匹配回归已停并保留。6合同、273轮计数及独立评分复算通过，18导出SHA一致、4图已检。读出de2644d/复核480df74/整理a48d08b；GPU释放，零反传/优化器/BN/test更新，原结果未变。无后续待执行训练。


## 2026-10-27：TS17外层代理落选原因，三步简化计划待审

用户要求只设计简单解释性审查，见[外层代理计划](96_TS17_BEST_PROXY_ROLE_AUDIT_2026-10-27/outer_proxy_plan.md)。同类ij去重、每类最多100对，对比最佳内/外候选实际角度与minimax分数；固定I/O分别仅换深度、仅换方向；追同ID的e0/e5/best与e1–273真实Gumbel使用。只需必要e0/e5各800对象冻结方向探针，best复用96缓存，不重训。初始化是高斯尺度设计而非逐代理等深，控制初始深度；e5/终态类别方向的回顾关联不称初始捕获因果。原源码会保存e0/e5和512项逐轮计数，但现存性待实施核验。本轮仅本机读文档/源码、解析几何演示和设计，未连接服务器、启动网络前向或新增训练，待用户审阅。


## 2026-10-27：TS17 best代理角色审查已在8000完成

用户批准执行96计划。见[96结果](96_TS17_BEST_PROXY_ROLE_AUDIT_2026-10-27/report.md)：仅best e273、9840正常嵌入缓存后重建4.6视图，512×256代理保原深度，5seed×200批。两段谷值3.678；外段163代理仅1个样本角色入选，其余162未入选，不能对应整体pair层。活跃291；S25⊆C142⊆T291，同关系非碰撞pair更深73.89%/94.31%，差中位.034/.200，属于重叠使用偏好。深度3.028–4.142，无cap命中；纯类top4全512为370/512，活跃189/291；活跃top4覆盖668/9840，最大曝光14，类组成随机p95为4。8+4+4点云卡和8组图/9张CSV齐全，12合同、20逐关系复算及62文件取回SHA核验通过。读出3f25a31、CPU整理f08d729；旧结果和源权重未变，无网络前向/反传/优化器/BN/test读出，GPU已释放。当前无后续待执行训练。


## 2026-10-27：TS17 best代理角色分层审查计划

用户只要求设计TS17/B12 best e273的pair/triple分层检测，见[96计划](96_TS17_BEST_PROXY_ROLE_AUDIT_2026-10-27/plan.md)。8000现有53.8MB冻结权重及9840缓存SHA与85归档一致，CPU加载含net/512×256 proxy，CUDA未初始化。计划正常whole→4.6视图后选代理，保留proxy实际深度；同类pair/异类pair/triple交叠、深度箱形图/最近类角、全局top4覆盖/热点/双分母纯类比例及8+4+4形态卡；五seed各200批确定性主读出＋原Gumbel概率复核，重复对象/碰撞单列。已有best深度约3.03–4.14不预设触cap。尚无本轮角色结果、网络前向或GPU分析，不自动执行审查或新增训练。


## PartHier五问冻结审查：8000完成

按用户批准方案在8000 GPU0执行，源e4c757c；12CPU合同、缓存13文件SHA不变、TS18 best/e300的抽签/有效/碰撞及四项联合率均与94精确相同。四权重角度各3×9840、原200批输出叶子intra梯度及候选/热点/反例完成；零编码器/网络反向/优化器。5组PNG/SVG、14CSV、summary及代表对象/关系反例保留，28派生文件取回SHA一致。等深候选p200概率约94→46%、91→54%；同类通过组A并不浅，失败组ij角度大。径向项无切向驱动；HIER六个展示反例同池有可行替代，不代表总体比例。GPU释放，见[报告](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/followup_results/report.md)。无新训练，不再追加消融。

## 2026-10-26：93总账TS23/TS24两组300轮结果已补全

用户报告8000两组结束并要求补93。20:03:56只读采集确认两条均300轮／60000更新／BN120000、exit0与保存验收通过，best/last存在、GPU0/1空闲。TS23（Ks10/Kp10/lambda.1）best e262 OA94.003241／AA91.827905、e300 OA93.719611、末10均值93.350891±.235880；TS24（Ks10/Kp20/lambda.05）best e277 OA93.598055／AA91.086626、e30092.463533、末10均值92.739060±.312178。93报告、CSV、注册表v4、来源与缺口同步，24正式记录现23完成/1中断/0运行；TS15未到300轮的空值不造补。其他22条配置/分数保持；保留用户在93中删去训练OA章节的当前结构，训练OA完整保存在CSV/JSON。本次零训练/前向/服务器写入；采集和11表/轮次/同轮AA/末10独立复算检查通过。

## 2026-10-26：94 PartHier四权重嵌入审查完成

8001原后台19:19正常完成，用户唤起后取回验收，未重复计算或改用本地。四权重全9840同裁剪、冻结BN及输入/状态验收通过，TS18各200个正常CB64关系batch、K20/T50、各自静态Q2048；提取807a5d6，统计6cc302b。5组PNG/SVG、8CSV及summary齐全，21文件SHA一致并视觉检查。TS18 chain0.2联合best7.774%/末轮4.024%低于原版24.949%/14.268%，主缺口p600与whole；A/B排序部分兑现但均偏p200、不能称三尺度祖先分工。无训练或额外消融，本地只整理文字，GPU释放。完整结果与局限见[94报告](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/report.md)。

## 2026-10-26：PartHier v1简化嵌入审查启动并保存进度

用户批准94简化方案。8001完成TS18 best/e300及TS02 R2 best/e300共4权重、每份全9840对象5种输入的严格FP32冻结提取；同输入/裁剪、原权重身份、BN不变及前向一致性均验收。提取807a5d6，统计6cc302b；18CPU检查通过，无训练/参数更新。统计后台最后running/metrics，已出现MODEL TS18_best start，曾长时间磁盘等待；尚无最终结果。用户额度告罄要求保存并坚持8001，停止主动工作，保留现有后台，本地仅备份无CPU统计。见[94交接](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/HANDOFF_2026-10-26.md)。

## 2026-10-26：8001恢复，93总账缺口已补齐

用户明确8001连接恢复并要求补全统计。本次12:56只读原始manifest/best轮/末轮/末10轮及进程列表：TS15实际保存54连续轮、best e52 OA91.855754/AA90.107554，当前无对应进程，manifest旧running属残留，归为已中断，不恢复训练；TS18 PartHier已300轮，best e201 OA92.990276/AA90.861619、e300 OA88.006483、末10均值89.444895±1.609896，e300在线train99.515625。TS19/20末10均值/SD及TS16/17 best同轮在线train精确值均补齐。93报告/全部CSV/注册表v4/恢复采集来源/缺口表同步更新；24正式记录现21完成、2运行、1中断，无状态未知。8000顺带更新完成前缀TS23 e101/TS24 e98；仍不填e300。剩余19空字段均为未完成终点或协议未记录（含PartHier clean/eval），没有连接失败造成的缺口；10个Markdown表及原14条已完成分数/原15条身份保留检查通过。未新增GPU计算、前向、训练、重启、停止或文件改写到服务器。

## 2026-10-26：V9至今实验总账93已统一补录

用户要求沿75风格统计全部V9及后续实验、统一新增命名并补齐缺失数据。见[93总账](93_V9_EXPERIMENT_ATLAS_2026-10-26/report.md)：保留TS01–TS15/B01–B10，新增TS16–TS24/B11–B14；24正式记录（20完成、2进行中、2后续未知），另列5条机制实验。B05严格仍SH32×307；B05几何的CB64改称B12、CB32改称B13。CSV/命名映射/注册表v4/缺口表及脱敏来源齐全；旧75报告的用户未提交修改未动。采集截止10:39:59，最新两组各完成17轮，仅前缀best，不填e300。8001拒绝连接：TS15尾段、TS18首轮后、TS19/20末10轮、TS17 best同轮在线OA仍显式缺失；其余可取得的完整成绩和训练OA已补录。只读现有文件、无GPU/训练/BN干预；数据单位、轮次、原15条身份/分数保持及9个Markdown表检查通过。没有扩大实验或自动等待服务器恢复。

## 2026-10-26：用户批准继续B05 CB32 A1两组单变量消融

8000 GPU0样本K10／代理K10／lambda0.1；GPU1样本K10／代理K20／lambda0.05。相对已完成91第二组K10/K20/lambda0.1，分别只改代理K和lambda。共同严格B05固定壳新HIER、CB32有放回16类×2、AdamW.005/1e-4、seed4780、fresh300×200、原TS协议；直接授权启动，不续旧权重或扩大矩阵。见[92配置](92_B05_CB32_A1_K10_WEIGHT_FOLLOWUP_2026-10-26.md)。源码061d86f已推送GitHub并以同一发布提交Git增量包fetch/pull --ff-only同步至8000。两组各4CPU、训练数据SHA／相同初始化及真实2×2短测通过，实际K、lambda、AdamW、BN8与完整保存状态验收通过。北京时间10/26 10:10:30两组均fresh进入正式300×200并已有真实更新；GPU0/1绑定正确。私有可视化目录b05_cb32_a1_followup_20261026保存dispatch/状态/验收/manifest。不重复提交，不逐轮监看。

## 2026-10-26：用户批准B05 CB32 A1两项完整消融

最新修订：8000 GPU0样本K5／代理K10、lambda0.5；GPU1样本K10／代理K20、lambda0.1。两组均10/25严格B05有放回CB32新HIER A1、LR.005/eps1e-4、seed4780、fresh300×200，其余保持86协议。用户直接启动授权覆盖旧待审矩阵限制，沿用历史TS选模；无新增H0/CB64。见[91配置](91_B05_CB32_A1_K_WEIGHT_ABLATIONS_2026-10-26.md)。第一组联合调整两种K，不能称两种K各自单变量；第二组仅改系数。源afb01ee已推送并在8000以Git增量包fetch/pull --ff-only同步。两组各4CPU及真实2×2短测／实际K、系数、AdamW、BN8与保存状态均通过；北京时间10/26 00:06两组分别在GPU0/1 fresh进入正式300×200，已见真实更新。私有可视化目录b05_cb32_a1_ablations_20261026保存dispatch、状态、验收与manifest；10/26 09:27核实两组均300轮／60000步、BN120000、exit0和完整保存验收通过；原PID已退出，GPU0/1空闲。K5/10 best93.598055、AA91.486043、e217、末轮92.990276；lambda.1 best94.043760、AA91.632557、e280、末轮93.517018。best/last各约162MB存在，完成manifest/状态/末轮metrics私有备份。不重启、不新增训练。

## B05 CB32 A0十轮复跑及85逐轮审查完成 — 2026-10-25

源c5a9423，同配方10×200、调度300、每轮完整权重；原e1/e5辅助，新e0..10明确fresh。21:14完成33CPU／CUDA短测／BN4000／80对象回归及14份9840缓存，11PNG/SVG和完整85表见[90](90_B05_CB32_A0_EARLY_AUDIT_2026-10-25/report.md)。原e5 cap508/512、类内40.10°；新e3 cap503/512、41.26°，e10 cap511/512、31.34°，所以整类收紧至2°不是早期cap前提。内部严格次序0%有有效分母；proxy热点须区分all-draw／active非碰撞，sample使用没有同向塌缩。新旧首步主干范数微差导致后续分叉，据实保留原e1/e5读出。不涉及PartHier或新300轮任务。

## B05 CB32 A0早期复跑及逐轮85审查授权 — 2026-10-25

用户只审10月25日B05均衡32＋旧HIER（.01/1e-8），明确不审PartHier；因原e2—4／e6—10权重缺失，授权fresh10×200、每轮保存，保持原300余弦和完整训练协议。18原始文件备份SHA一致、12新CPU合同通过，真实GPU与全量几何待执行；[90计划](90_B05_CB32_A0_EARLY_AUDIT_2026-10-25/plan.md)区分新复跑和历史e1/e5，并沿用85固定9840 clean/eval／BN和全部统计定义。不得称恢复原历史未保存轮次，不扩大其他组。

## B05同LR、eps1e-8／1e-4：十轮结果 — 2026-10-25

8000两组CB64/K20-K20/seed4780/代理LR.005、fresh各10×200，16:17:26全部完成验收、零评估，源c8c05bc。E8末轮500/512 raw越cap，E4为0/512；深度中位4.600／3.439，raw模长中位2.618／1.720。E4第4轮末起cap0，第6轮起轮内也零新增；曾在cap外70个均曾返回。首步梯度同值，更新1.799→.446；E4末轮梯度均值更大、更新更小，区别反传与AdamW更新。初始化名义4.14为高斯尺度、非严格等深，起始已有4个raw越cap。81文件备份SHA一致，9幅分布／梯度图和逐轮CSV见[88](88_B05_EPS_ONLY_10E_2026-10-25/report.md)，不据此推断完整分类准确率。

## B05双卡eps单变量十轮检测 — 2026-10-25

用户授权8000双卡B05、CB64、sample/proxyK20，seed4780。固定代理LR0.005，比较eps1e-8与1e-4，fresh各10×200但调度保留300轮，零评估／选模；区别于8001单卡A0/A1全程任务。新增每步proxy实际梯度／AdamW分母与更新／越cap返回、每轮首batch主干梯度和512代理深度及位移分布。源码c8c05bc、本机／服务器各13项CPU及两组真实双卡2×2验收通过，初始化／首批／首步梯度配对一致。15:44:05在8000 GPU0/1 fresh启动E8，E4随后自动执行及汇总；完整十轮结果尚未取得，见[88](88_B05_EPS_ONLY_10E_2026-10-25/plan.md)。

## 四组whole／代理完整审查完成 — 2026-10-25

按用户指定8000执行85：8主端点＋8辅助，同9840 sorted-shard clean first1024/eval savedBN。TS13等10份复用、TS14／新两组六份补齐；固定80例历史batch上下文回归坐标／深度误差0，21CPU通过。16缓存身份与本机备份SHA一致，GPU已释放、无新训练或test。交付[85报告](85_EMBEDDING_AUDIT_2026-10-24/report.md)、32组深度百分比分布、40类／780类对、主辅各22幅PNG/SVG。

A1回内成立但代理内部次序不普遍：末轮排除碰撞／端点自选后三项严格B05约17.264%、原生21.854%。B05类别广度与深度无稳定关联；冻结热点改善不能替代真实使用熵。原生flower_pot–plant中心52.055°→8.522°是明确局部方向退化候选，未验证test分类因果。采集源8f8f4fc、最终统计／图bd3f1e6，早期门槛／资源释放失败记录保留，不放宽容差或改历史结果。

## B05单卡：新旧HIER A1／A0 — 2026-10-25

用户确认两条A1/A0对比，指定8001 GPU0/1，要求完整后台排队后结束会话。共同严格B05 c.1/cap2.3/baseGS/半margin、pre-base-GS固定深度4.6、CB32（16类×2）、Ks10/Kp20、seed4780、fresh300×200；仅代理初始LR/eps为.005/1e-4与.01/1e-8。新版本保持原入口，修正单rank集合及global32目标/审查，8项本机CPU检查通过。GPU短测须在获得对应空闲卡之后由队列执行，不把尚未执行的测试或排队算正式成绩。配置和边界见[86](86_B05_SINGLECARD_A0_A1_QUEUE_2026-10-25.md)。

## 四组嵌入审查计划与本机工具准备 — 2026-10-24

用户要求TS13/TS14/新B05-A1/新原生A1的whole与代理审查，允许辅助消融；最新要求先完善计划，待服务器连通后通知继续。[85计划](85_EMBEDDING_AUDIT_2026-10-24/plan.md)包含best/e300同9840 clean/eval输入、深度百分比hist/ECDF、类内/类间分位与固定类别直接对比、代理稳健分层/语义边界。仅完成缓存/身份盘点、6份冻结net-only本机SHA核验、原FPS本机算子11项验证和统计程序9项合成CPU测试；未执行四组全量模型提取或发布实测图。原始数据/权重保持私有，服务器GPU未分配。TS13资料须待8000恢复后取得，不以旧200例短测或TS14 e120替代。

## TS13/TS14正增量与新A1配置比较 — 2026-10-24

只读比较与候选机制见[84](84_TS14_VS_NEW_HIER_2026-10-24/report.md)。TS13/14主要隔离HIER分支，均CY64×300/seed4780，best和e300均+12个test对象，末10稳定性改善；不能由新组低于SH32历史高分反推HIER普遍负作用。补查TS14实际best/last均511/512贴cap，末轮raw模长中位6.198；它仍有分类正增益，支持区分径向目标与方向关系正则作用。新B05换CB200和A1后末10均值92.986，与TS14的92.978接近，保留best低0.405pp的真实差距，不归因为eps单因素。

CPU sampler机制模拟使用真实40类数量、合成类内ID，不含点云/模型：CY200/300首轮覆盖7748/8959，CB200/300为6218/7321；CB200同ID二元对25/6400≈0.39%，不能将结果归咎大量同物体正对。未启动新训练、冻结前向或BN校准，不自动恢复循环64。

## 两条seed4780固定壳HIER完成与只读汇报 — 2026-10-24

79源5517661两组均正常300×200完成，原生02:49:14、严格B05 02:47:34结束。原生best e153 OA92.747/AA90.462、e300 OA89.506/AA89.329；B05 best e273 OA93.476/AA91.508、e300 OA93.112/AA90.758。末10轮OA分别90.312±1.257与92.986±0.204；e300 clean/eval训练OA97.266与99.817。两组best/last代理均0/512越cap，e300深度中位3.221/3.440，最大3.827/4.118。实际更新/调度器/BN/有限性与逐轮完整test核对通过；final_test.json是best汇总，不是e300。见[83](83_S4780_NATIVE_B05_COMPLETION_2026-10-24/report.md)及完整精度JSON。

代理回内已得到完整300轮证据，但分类未超历史SH32高成绩，且尚无同协议S4780 CB64 H0，不能把差值全归HIER。B05整体组合更稳；原生后期退化值得冻结检查BN/方向结构，本次只作建议、未新前向或校准。两条完成任务与后续PartHier v1分开，不重启、不增加矩阵。

## PartHier v1在8001正式启动 — 2026-10-24

用户后续改用8001并要求提高显存利用率。33CPU/Gloo及实际CUDA/2×2/满Q容量验收后，采用source82478cd的部分欧氏主干checkpoint：200/400反传重算，600/whole保存，原FP32/global64/BN4和PartHier全部数学目标保持。满Q/whole1024每卡allocated27.79GiB、reserved28.21GiB、至少2.27GiB设备空闲；有界后3步1.343s，对全重算18.07GiB/1.527s约少12%耗时。

09:45:48在8001 GPU0/1从seed4780初始化、空Q/新optimizer独立fresh300×200，官方9840/2468逐轮严格OA选best无早停。09:50首轮200更新/2468 test、两rank38层BN800、有限模型/真实RSGD/保存状态、300轮调度、Q2048与参数/队列一致性及初始化/数据身份通过，已进e2后台继续。启动与验收详见[80](80_PARTHIER_V1_PLAN_AND_LAUNCH_2026-10-24.md)。初版OOM与两次重算兼容错误保留为工程修复，不计作独立方法实验或成绩；8000旧offload状态未知。不将短测或首轮解释为层次/分类效果，不追加候选或逐轮监看。

## PartHier v1实现，真实GPU验收未完成 — 2026-10-24

用户确定原生c1、200/400/600同中心嵌套、三尺度混合候选、Q2048、逐级0.01 intra求和，授权有界短测与8000单条fresh完整训练。e6a7168/ca1402c版本已推送/服务器快进冻结，18项CPU及global64双rank Gloo梯度对照通过。双卡local32普通四pass短测在第四whole前向OOM，未完成更新；CPU暂存重测随后提交，但8000 SSH无登录响应，结果/进程状态未知，不将连接异常归因于已证明的内存耗尽。8001四卡忙，未干扰。

新增保持BN/RNG的checkpoint后备，本机CPU wrapper输出/梯度/状态对照已通过；实际CUDA/DDP及满Q检查未执行完成，无完整300轮任务、无分类成绩或角色/尺度结论。故本条记为工程实现与连接阻断，不纳入正式TS成绩表。详见[80](80_PARTHIER_V1_PLAN_AND_LAUNCH_2026-10-24.md)。

## seed4780原生/B05固定壳HIER双组已启动 — 2026-10-23

用户选择严格保留B05基础GS，随后授权两条均在8001启动。源5517661，统一非循环有放回CB64、seed4780、300×200、TS与Hshell-A1代理AdamW LR0.005/eps1e-4；原生组保留c1/无新增cap或GS/1000_N与4，B05组c.1/cap2.3/基础GS/500_N与2，HIER从cap后基础GS前分叉。各18项CPU、各自Gloo和独立train-only2×2短测通过，保存的真实优化器/调度器、BN8与有限状态均验收。18:17:36从头进入生产，原生GPU0/1、B05 GPU2/3；首轮200步/完整2468 test记录已产出。用户要求短测通过即结束，因此后台继续训练，本会话不再逐轮监看或自动跟进，不增加H0或重启旧断电任务。详见[79](79_S4780_NATIVE_B05_LAUNCH_2026-10-23.md)。

## 导师汇报的TS实验与idea时间线整理 — 2026-10-23

新增[78阶段汇报](78_SUPERVISOR_TS_EXPERIMENT_SUMMARY_2026-10-23.md)，以75注册表/脱敏采集和55–77历史审查为依据，整合全部正式TS设置、分类和训练成绩、9组消融，以及协议对齐→V8→基础适配→V9固定壳→循环配对→AdamW候选的时间线。14条切换后任务加1历史原版参照；71/73及57–59冻结/短轨迹另列。TS15宕机后状态未知，无e300，仅e34定量快照；旧V6 batch及VS→TS协议参照附后，未纳入新任务计数。没有服务器连接、新前向或训练。

## 原版HyCoRe/HIER/B05源码dataflow与GS命名澄清 — 2026-10-23

新增[77：三张dataflow与源码定位](77_HYCORE_HIER_B05_DATAFLOW_2026-10-23.md)。原版HyCoRe的实际梯度操作是backward后参数梯度总范数1裁剪→RSGD，没有B05式输出GS；B05在共享emb出口新增cap2.3/GS并保留原裁剪/优化器。官方HIER则在样本及层次代理出口各有GS，更新前另逐分量裁剪±10→AdamW；区分PA类别代理和层次代理，避免把B05画成已含HIER或固定壳。仅读本地源码和写文档，无服务器连接、训练或配置修改。

## seed4780与非循环均衡64的设计讨论 — 2026-10-23

用户统一未来全部实验seed4780并弃用循环64；历史身份不改。用户报告服务器断电，要求只讨论设计，未连接服务器、实现/启动/重跑任务。见[76](76_S4780_FIXED_SHELL_BASELINE_DESIGN_2026-10-23.md)：拟比较原生c1与c0.1适配底座各加新AdamW固定壳HIER；B05基础GS来自HIER移植，并非原版自带，保留完整B05与主pipe绝不含该hook是两种配置，尚待明确。建议每种底座配H0以量化增量，不用历史SH32 OA直接代替新CB64对照。旧TS15进度只视为断电前快照，断电后状态未核查。

## TS实验统一别名及全部成绩核对 — 2026-10-22

新增[75：baseline图谱](75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/report.md)和[TS注册表v3](experiment_registry_ts_v3.json)：V8/HC改版/V9共15条TS记录，归入10种基础配方；seed仅作后缀，组件分别为无HIER、共享实际半径HIER、固定4.6壳HIER与新AdamW候选。排除VS984与短测，不改历史运行名称或结果。仅读取已保存指标/历史标量日志，检查13条现代完成任务的连续300份轮记录与1条进行中前缀；best AA取同轮，补齐在线/clean-eval训练OA、e300与末10轮统计。历史原版e300 OA92.828%，best复测94.043760%。23:06:55快照A1完整任务已34/300，不作最终排名，不启动新矩阵。新旧名映射及完整CSV见75。

## V9-HIER新AdamW单条300轮启动与昨日活动复核 — 2026-10-22

用户请求完整验证73组合，新增versioned wrapper保持原V9生产loop，只改proxyLR0.005/eps1e-4及版本/名称声明；fresh S22、300×200、nativec1/HIERc.1/H4.6、TS、其余原配方保持。执行8408273先推送/服务器快进冻结，38项CPU+Gloo与独立2×2短测通过。21:28:18在8001两张完全空闲GPU1/2从头启动；21:31:26首轮200步/2468test/有限状态/BN400/代理同步、实际AdamW与300轮scheduler、初始化/数据/首批身份等16项验收通过；不逐轮监看或提前停止，不追加矩阵。见[74](74_V9_ADAMW_FULL_RUN_2026-10-22/report.md)。

同次只读核验10月21日四条新训练已全部完成，见[昨日汇总](74_V9_ADAMW_FULL_RUN_2026-10-22/yesterday_summary.md)。原版c0.1循环HIER终态现为300轮、best e212 OA93.881686%/AA91.752911%、e300 OA93.273906%；10月22日10:55:59自然结束，更新70的历史中途状态，不修改旧记录。循环无HIERbest93.395462%、e30092.787682%，单seed两种OA差均12个对象；不称形态层次已验证。

## V9代理LR0.005/eps1e-4联合十轮完成 — 2026-10-22

用户直接授权保持V9/71设置仅改代理初始LR与eps；优先服务器空闲GPU1/2完成10×200步，17:35:33–18:03:59，原300轮调度、零test/选模。源码03e6fa7先推送/快进/冻结，41项CPU与fresh2×2短测通过；数据分片/初始化/首批身份匹配71。末轮越cap0/512，对照502；122个曾越界代理全部回内，深度中位3.576、范围3.147–4.572。新e10 WD压过减弱后的历史/当前外移，实际切向位移约对照4.84%；关系损失下降但同轮更高。sample祖先与whole间隔恢复，代理内部次序/覆盖不全面改善，不能推出分类/语义层次收益或300轮稳定。11项完成与7项补充验收、80快照SHA/同图梯度和、独立结果复核通过；CPU分析6b67040，GPU已释放。见[73图文报告](73_V9_LR_EPS_10E_DIAGNOSTIC_2026-10-22/report.md)。用户暂停监看后按再次询问读保存数据，未启动其他实验。

## Cars/V9只读角度与Adam有效步幅审查 — 2026-10-22

用户要求先回答lambda、eps/AMP和历史m，再检查类内外角度及迁移力度。[72报告](72_HIER_CARS_V9_ANGLE_AND_STEP_AUDIT_2026-10-22/report.md)复用Cars8054/98类和V9同9840/40类clean缓存，加71六个真实训练前向；共同c.1/H4.6参考下，末态V9类内极紧但异类更分散，三点中心仍显著浅于whole，实际pair多数跨类。7份真实Adam状态CPU重放误差≤5.96e-8；第一步余量中位.06549的19越界代理实际增模中位.10390，LR/eps一步控制减少越界，lambda缩放大多被Adam归一化抵消。实际Cars/V9代理状态均FP32，eps差异是显式配方，不将AMP等同纯半精度Adam。代码2ed430f先推送、双服务器快进后CPU执行，14检查和4组PNG/SVG通过；零GPU、新前向和生产更新，不替换历史V9，不追加训练。

## V9前10轮代理外移机制诊断 — 2026-10-22

用户授权8000从头5–10轮，本次独立诊断10×200步、原300轮余弦，不读test、不选模。e1/5/10有229/444/502代理越cap，2000步502首次越界且0返回；这是新机制复现，非历史V9逐位重放。真实Adam核算显示越界后历史m径向分量为主要外移项，固定状态去旧m径向及eps1e−4一步对照能转内。7个关键步14个分项恢复CUDA Gumbel状态，同GPU距离/硬选择/hinge/梯度全部逐元素一致；e1 step5两项直接距离梯度平均向内，实际选择路径使full-ST转外。无显式祖先深度顺序，固定whole壳和0.9初值不足以保证内层。40项合同/分析/重放检查、11项完成验收、3组PNG/SVG通过，源及旧任务不改，GPU释放。见[71](71_V9_PROXY_CAP_CAUSE_2026-10-22/report.md)。

## V9 whole与代理深度审查完成 — 2026-10-21

[69：固定权重几何审查](69_V9_EMBEDDING_REVIEW_2026-10-21/report.md)：14份固定权重×同9840对象clean/eval，比较五组best/终点及循环组中途配对。V9 e300代理512贴cap、参数模长中位6.885，V8为510与6.915；V9 e5饱和较慢但e99全cap，晚期回缩未回到cap以内。V9原始NN4覆盖216→725、最热四对象32.67%→6.49%，但V8冻结等深度覆盖1025，不能视作已学出更好的代理方向或径向层级。V9原生whole仍深浅分散，HIER视图固定4.6是构造；代理整轮选择与有效关系覆盖均512，与V7休眠现象区别。11组图、全分位/方差/分类别/代理使用统计与验收SHA归档。没有新训练、BN校准或test前向，不改正在训练的循环HIER组。

## V8、原版修改、V9完整配置与成绩表 — 2026-10-21

[68：17组总表](68_ALL_V8_ORIGINAL_V9_EXPERIMENTS_2026-10-21.md)已按两台服务器正式记录核对：V8七组、原版线六组、V9两组、循环配对两组。15组已结束，循环两组在16:44快照分别完成128/44轮、当前best OA92.828%/92.382%，不能作最终结果。总表保留VS984与TS、best及同轮AA、e300和当前轮分离，原版0起轮次单独注明。已完成的原版cap+半margin+GS组test OA94.327%；V8-B64 TS为93.882%，V8-HIER64 TS为93.355%；V9-B64/HIER64为93.436%/93.314%。匿名完整精度数据在68链接的JSON；本次零训练/重测/停训。

## 原版c0.1循环B64与固定壳HIER配对已正式运行 — 2026-10-21

用户确认原版c0.1/cap2.3/GS1/500_N与2/seed4780底座，双卡global64类别均衡循环抽样、跨轮保留游标、300轮×300步；HIER只看深度4.6的pre-base-GS whole视图，两条路径各一次GS。更耗时HIER组在8001 GPU1/2于13:02:54启动，无HIER对照在8000 GPU0/1于13:03:09启动，均从头且独立后台。源码c0d2c89已推送/快进/冻结；39项CPU与Gloo参考梯度、train-only2×2短测及正式首轮验收通过：300更新/完整2468 test/有限best-last-e1/BN600/每类480例次。初始网络与原底座一致，两组数据及首批增强输入完全匹配，首轮8959不同对象，恢复队列后首两轮累计9840。保持TS、不earlystop。见[67：方案与上线](67_ORIGINAL_C01_CYCLE300_HIER_PAIR_2026-10-21.md)及匿名验收JSON；私有路径/PID见忽略的67工程交接。旧c1 V9与这些ORIG c0.1派生组不混称。

## V9-B64匹配对照与官方c1全新复现已启动 — 2026-10-21

用户授权双卡V9-B64，并明确确认余下一卡从头重跑官方c1 S4780。8000 GPU0/1于00:55:07启动V9-B64，c1/无额外cap或GS/1000/N与4/seed22、global64类别均衡、300×200、全9840/2468 test选best；HIER视图、损失和代理零构造零调用。GPU2于00:59:06启动官方c1 R2，seed4780、shuffle32/307×300/workers8/eval16、官方构造/train/validate/持续随机流，不加cap/GS，不resume。两组均不earlystop，不追加此前seed22高表现组复核。

B64源码89710b5、33项CPU及Gloo；官方21f6c69、10项CPU；各自train-only2×2后fresh生产。首轮200/307更新、完整2468 test、有限best/last/e1与恢复状态通过，后台继续。B64初始化/数据/首批ID和crop与V9主训练匹配；官方训练数据身份与现有ORIG c0.1组完全匹配。新[66：方案及验收](66_V9_B64_CONTROL_AND_OFFICIAL_REPEAT_2026-10-21.md)，私有路径/PID见忽略的66工程交接；旧结果不改。此前ORIG c0.1+cap+GS+半margin已自然结束300轮，best OA94.327%，不将单次高成绩称为准确率理论上限。

## V9双曲率/分支GS正式训练已启动 — 2026-10-20

用户确认双卡类别均衡64、300轮，原生HyCoRe c1/1000/N/4、不加额外cap/GS；HIER仅看c0.1固定深度4.6的保方向whole视图并使用分支GS，代理保留官方cap2.3/GS/随机初始化/AdamW。完整9840训练/2468 test逐轮选best、seed22、W0/lambda0.5、300×200步，不提前停止。仅启动这一条V9，不追加矩阵，旧训练与结果保持。

30项CPU检查、global64双进程梯度参考和双卡2×2 train-only短测通过；19:44:55在8001空闲GPU1/2从头正式启动。源码`b5d2d8b`已推送、快进同步并冻结，短测与生产e0网络/代理逐元素一致。19:47首轮200更新/完整2468 test/有限best-last-e1权重及Adam状态通过，BN每步+2、代理双rank差异0，后台继续。HIER视图4.6与原生whole/part c1分域记录，并保留实际训练前向快照；旧单曲率审查不能直接套用。方案、实现边界与验收见[65](65_V9_DUAL_CURVATURE_PLAN_AND_LAUNCH_2026-10-20.md)，命名见63/registry v2，路径/PID在忽略的私有65交接。

## V9初步idea已记录，曲率组合待定 — 2026-10-20

用户认可“原生HyCoRe＋等深度HIER分支”为V9初步设想：基础CE/intra不加共享cap/GS，HIER仅看保留方向、深度4.6的whole视图，代理保留自身HIER映射。统一c0.1与基础c1/HIERc0.1两种曲率组合尚未定，不擅自选定或启动矩阵。见[64：idea](64_V9_INITIAL_IDEA_2026-10-20.md)；[63](63_EXPERIMENT_NAMING_AND_GRADIENT_SCOPE_2026-10-20.md)补充官方HIER三路dataflow，区分PA类别代理与512层次代理。仅文档记录，无代码/训练修改。

## 用户更正GS实验到8000，旧8001后台已取消 — 2026-10-20

用户更正同一`ORIG-B0-32-C0.1-CAP2.3-GS1-M500_2-TS-S4780`执行位置为8000。旧8001队列13:23已因nvidia-smi查询超时退出，没有GPU短测/生产；15:59确认无存活进程、无训练子目录，追加取消迁移记录，不重启。8000新队列37项CPU合同与2×2 train-only GPU短测通过；16:02:05在空闲GPU2从头正式启动300×307，配置/初始化/数据与既有原版对照保持。GPU0/1现有训练不改。源码e20cf8c已推送/快进并冻结；仅增强队列超时重试，不改训练。首轮完整test、307步与best/last/e1有限权重保存通过。当前结果与路径见[62](62_ORIGINAL_C01_CAP_GS_HALF_MARGIN_2026-10-20.md)及本地私有62，旧8001等待快照已失效。

## 原版c0.1＋cap＋GS＋半margin后台待空闲；命名规范更新 — 2026-10-20

用户授权8001单卡补充`ORIG-B0-32-C0.1-CAP2.3-GS1-M500_2-TS-S4780`：与61原版cap半margin组仅切GS1，原版训练/seed4780保持，完整300×307与9840/2468逐轮test选模。36项CPU合同、初始权重SHA和训练文件SHA通过；跨机器数据枚举差异用对照原行顺序显式对齐。12:13独立后台提交，12:15四卡均被其他用户占用，状态waiting_for_idle_gpu；GPU短测和正式训练尚未开始，空闲后自动短测再从头生产。源码859e0e3已推送/快进并冻结，勿重复提交。见[62](62_ORIGINAL_C01_CAP_GS_HALF_MARGIN_2026-10-20.md)。

用户确定未来默认：TS全9840训练/2468 test选best、不提前停止；ORIG修改线seed4780，V系列seed22。历史协议和seed不重标。新[63显式参数命名与GS解释](63_EXPERIMENT_NAMING_AND_GRADIENT_SCOPE_2026-10-20.md)及[registry v2](experiment_registry_v2.json)映射12组原版/V8；旧目录/日志/权重身份不改。GS作用于通过映射输出的所有反向分支，V8的CE/intra也会传过GS；下游分类器自身梯度不直接乘因子。HIER层次代理与PA分类代理分开解释。

## 原版c0.1＋cap的margin减半组已启动 — 2026-10-20

用户直接要求[61：原版cap组margin减半](61_ORIGINAL_C01_CAP_HALF_MARGIN_2026-10-20.md)。以已完成ORIG-B0-32-C01-CAP23-S4780为基准，仅改gamma1000→500、delta4→2；保留c0.1/cap2.3/无hook/seed4780、原版shuffle32、完整9840训练/2468 test每轮选best、300×307。32项CPU合同及训练限定2×2 GPU短测通过，初始权重和训练数据身份与完成对照完全一致。11:50在空闲GPU1从头正式运行，源码e40a071已推送/快进同步并冻结，后台独立；GPU0既有V8保持。首轮307步、完整2468 test及best/last/e1有限权重验收通过，后台继续。两项正则系数各0.01，不是将损失减半。私有路径与进程见本地61交接。

## c0.1/cap/nohook补充三组上线 — 2026-10-20

用户确认每卡一条完整300轮：V8 FULL9840 shuffle32只关hook、保持seed22/cap2.3/gamma500/delta2；两个原版组seed4780/gamma1000/delta4，仅统一c0.1及配对增加cap。新服务器GPU0于01:24:49启动V8，GPU1/2于01:41:54启动原版配对。首轮均307步、2468 test、best/last与有限权重检查通过；01:46各已完成6/2/2轮。28项CPU与三个2×2 train-only smoke通过，原版初始参数/数据身份相同。旧原版源码直接复用train/validate、随机流和BN，未引入V8 margin/RNG；新增V8旧runner仅换反向hook。见[60](60_C01_CAP_NOHOOK_ABLATIONS_2026-10-20.md)及匿名启动验收；私有60保留实际root/PID/runtime与失败重试记录。正式源码c78a168/eebaabf，后续提交仅归档。

## N600同深口径澄清 — 2026-10-20

对10月19日已保存的18观察作CPU复核，未改用户主报告或源缓存。当前批BN旧V8/FULL各927/935对whole与part同时贴cap、占1280的72.42/73.05%；1e−4同深容差下真实反转仅7.81/8.52%，原始g>0约51%不足以解释径向顺序。原版同面板N600顺序正确64.45→80.08%，与完整margin达标不同。独立图文和计数见[补充](59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/clarifications/N600_BN_ORDER_2026-10-19/report.md)。本次无GPU、模型前向、优化或test。

## whole/part配对margin满足率与输入强验证 — 2026-10-19

原版best229/last299、旧V8 best221/e300、FULL V8 best297/e300，统一8856对象、N200/400/600固定同中心嵌套part。原版旧缓存复用；V8全量冻结导出，六模型另补同1280对象BN-only诊断。完整Rhier终点原版.01/3.12/1.94%，旧V8 1.78/10.16/4.55%，FULL 2.01/9.49/4.42%；90%宽松目标、深浅顺序、违例幅度和40类表另存。共同20异类负part检验delta4/2及联合达标，不作为历史训练flip比例。V8 N400当前批BN显著改善，但N600 part约75–76%贴cap、径向仍差；不据此改正式BN。全9840历史part索引SHA→canonical8856 GPU索引强证明、源SHA和状态恢复通过；旧V8 native whole误差0。见[59](59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/report.md)。无test前向、无优化更新或新训练。

## FULL9840对照自然完成 — 2026-10-19

只读核对8001已存结果：B0-32完成300轮于01:54，best203 test OA93.719611%/AA90.156972%；B64完成300轮于22:49:49，best276 test OA93.881686%/AA91.920358%。此前按约34分钟估计建议继续，最终没有执行kill或改变协议。完整test选模结果与旧8856/984验证协议分开。

## V8四模型whole密度与冻结BN统计对照 — 2026-10-19

完成原HyCoRe、旧同8856/984的V8-B0-32/B64/HIER64 best与last共8缓存对齐，确认同8856对象、标签及点云输入字节。训练终点均深度分别4.5659/3.8851/3.5407/3.7628；原版c1无V8 cap，仅为参照。HIER较同batch B64更深且窄，不支持whole分散全归HIER。原HIER三集类内方向余弦.727/.734/.776，跨类.335/.333/.345，同壳仍有类别分离；SOP代理反例保留。新三模型×四BN冻结探针，320测量/640独立训练whole校准；直接借原HyCoRe末轮running buffer明显错配，97.5%预测改变；自身whole校准可恢复壳，却把part推深，主HIER约50.94% part贴cap。所有参数/BN/RNG与源文件保持或恢复，native输出与既有完整缓存最大差0。11组PNG/SVG、匿名统计及可复现工具见[58](58_V8_WHOLE_DENSITY_AND_BN_REVIEW_2026-10-19/report.md)。没有训练或test前向，不改正式BN；完整9840重跑不混入本次比较。

## V8代理cap、Adam与whole BN机制诊断 — 2026-10-18

按用户授权，在空闲单卡和本地CPU完成代理梯度/实际Adam位移/历史m、eps/LR/hook/等深whole/分项损失干预，及whole前向条件与BN库控制。匹配1024步仅代理更新，eps1e−8与1e−4分别387/6越cap，固定评分loss .07685/.10911（同初值.38869）；不提供完整训练分类结论。原V8 e5越cap组m径向消除使平均外移转回缩，但128步越cap178→169，说明不能当作完整修复。原三集真实Adam及36图像head探针提供配置/尺度对照，原SOP仍497越cap。晚期B64 e300同128对象BN模式3.605→4.584；固定参数、640校准训练对象排除128测量对象，匹配构造与40前向的whole→whole/part→whole统计库使eval贴cap99.22%/1.56%，支持共用运行统计机制。8组图、12张匿名CSV、9项Torch测试与限制见[57](57_V8_CAP_AND_WHOLE_CAUSES_2026-10-18/report.md)。源权重/参数/BN/RNG均保持或恢复，无test前向；私有副本有实际Adam更新，不混写为零计算更新。完整主实验和对照配置保持，未启动V9。

## V8完整9840协议两组对照已排队 — 2026-10-18

用户授权旧服务器单卡V8-B0-32-C01 shuffle和双卡V8-B64无HIER，保持全9840训练/2468 test每轮选模。300轮，B0为307步/轮，B64为200步/轮；V8 cap/hook/margin与seed22不变。00:04独立机会调度后台存活，14项CPU合同通过，五份训练HDF5与主实验SHA一致；四卡均被其他用户占用，两组均待空闲，未GPU短验收或生产更新。有效代码a25cc36已推送/快进部署；初次浮点精度断言失败记录保留，重试仅改测试精度。见[56：对照及后台排队](56_V8_FULLTRAIN_CONTROLS_2026-10-18.md)。满足一张/两张空闲卡条件后自动验收并启动，不需重复提交。

## V8完整9840训练池、test选模已启动 — 2026-10-17

用户明确授权一条V8-HIER64改用原版官方9840训练/2468测试、无独立验证集、每轮test选best，随后指定新服务器执行。展示名V8-HIER64-FULL9840-TESTSEL；双rank各16类×2、global64，300×200步与V8几何/损失/seed22保持。23:22启动，首轮完整2468 test与best/last通过，14项CPU单元合同及global64双rank梯度、独立2×2 GPU短验收通过；23:28已e3step65。新入口生产提交ff98bf9，已推送并快进部署；没有旧服务器重复任务。见[55：方案及启动验收](55_V8_FULLTRAIN_TEST_SELECTION_2026-10-17.md)。该结果将按测试集选模单独报告，不混作984独立验证协议。

## V7/V8固定权重成绩复测与划分核查 — 2026-10-17

原始HyCoRe e229（原0起编号）本次test94.0438%/AA91.5448%；无独立val。现代984子集原模型已训练过，诊断OA98.9837%不作val。V7 e187、V8主e221、B64 e132、B0-32-C01 e249四组984验证/2468 test均复现保存成绩。新补B0最终成绩：val OA94.5122%/AA91.8165%，test OA92.6256%/AA89.8128%。四组split SHA相同，两个服务器输入SHA一致，5份源权重与模型/BN保持，零backward/optimizer更新；只测试预先固定best，未重新选模。完整拉表、原版与现代训练/验证/test差异及V7→V8改进逻辑见[54](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17.md)。V7只有一条完整主训练，无新V7-B64。没有启动新训练矩阵。

## V8 postrun：主模型与无HIER双卡对照完成 — 2026-10-16

V8-HIER64 best e221，val OA94.2073%、test OA92.7877%/AA90.0372%；V8-B64 best e132，val OA94.3089%、test OA92.9092%/AA90.3157%。仅差3测试对象，当前单seed不支持HIER分类收益。复用并适配V7全系统审查，对照原版CUB/Cars/SOP：V8末期sample祖先均512、热点NN4覆盖231、等whole深度1051、510代理forward cap；whole径向分散与方向冗余仍在。固定e300全池Chamfer富集较B64有条件配对正差，best与其他形态指标不一致。18组图及14张汇总表见[53](53_V8_SYSTEM_AUDIT_2026-10-16.md)。零训练更新、无新增val/test，154项合同通过。V8-B0-32-C01尚未完成，采集时e283、best249 val94.5122%；不得与最终结果混用。

## V8新账户及两组无HIER对照 — 2026-10-15

用户确认新服务器单卡也采用V8 c=.1/cap2.3/hook、gamma500/delta2。完成普通账户和Conda配置，两卡V8-B64与单卡V8-B0-32-C01于11:41启动，代码85fc812。12项CPU合同、global64双进程梯度合同与各2×2 GPU smoke通过；正式首轮完整984验证及checkpoint通过，进入e2。无HIER调用、代理参数或代理优化器；旧主V8继续，初始化/划分SHA一致。没有最终效果结论。见[52](52_V8_CONTROLS_ACCOUNT_AND_LAUNCH_2026-10-15.md)。

## V8实现与后台排队 — 2026-10-15

按用户明确授权完成V8-HIER64-K20-W0：统一官方HIER几何c=.1/cap2.3/一次hook及Gaussian代理初始化，HyCoRe gamma500/delta2、两项系数各.01、HIER系数.5，首步联合训练；主干RSGD与代理AdamW各自300轮余弦。9项合同和CPU双进程global64参数梯度检查通过，真实8856/984划分核对。源码快照14f8a72及有效后台任务`v8_hier64_20261015_022043`已固定，02:23:30仍无两张空闲卡，未使用GPU、未读test、未开始主训练；自动等待后依次GPU smoke与e300。无OA或结构收益结论。原Conda标准库I/O异常通过同版本私有快速运行时绕开，原环境未改。实际身份和交接见[50](50_V8_IMPLEMENTATION_AND_LAUNCH_2026-10-15.md)。

## 导师汇报：既有V3—V7材料编纂 — 2026-10-09

完成[75分钟汇报稿](47_SUPERVISOR_REPORT_2026-10-10/汇报稿.md)与[素材索引](47_SUPERVISOR_REPORT_2026-10-10/素材索引.md)，另给60/90分钟使用方案。覆盖Onghena→HIER选择、V3历史快照、V4—V7全部主实验、V6→V7四项变化、仅sample-used代理方向、径向/方向热点以及原版CUB/Cars/SOP对照。27张PNG与14张已有SVG原样复制，既有来源原文随包保存；不新增训练、科学图或分析数据。三路资料复核已修正参数cap、激活口径和因果边界；本地链接、公式与素材哈希验收记录随包保存。完整权重、数据、缓存、日志、身份表和服务器信息不纳入材料包。

实验展示名统一遵循[33：实验命名规范](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md)。旧文件名、运行ID及代码字段保留用于溯源；本次只规范称呼。

## 原版 HIER 三数据集热点机制与激活来源 — 2026-10-08

[46](46_ORIGINAL_HIER_SOP_MECHANISM_AND_ACTIVATION_2026-10-08.md) 补充完整高维方向与官方分项激活：三组whole都在d0≈4.6薄壳，SOP代理R=.6482、497/512超过forward cap；CUB/Cars代理R约.044/.047，越cap数9/1。SOP整体旋转覆盖432→880–1065，方向减均值→1778但同细类纯度下降，因此覆盖改善不等于结构收益。缓存一轮长度窗口和真实图像32/44/330批全部完成，三组sample有效祖先及前2batch两个分项full-ST梯度均512个；图像SOP径向梯度占比2.59e−6/1.82e−6，仍有角梯度。36张clean anchor前向与原缓存最低余弦>.999999，网络/BN/lcas与源SHA保持，零参数更新。公开数据三集已下载验收；权重和缓存副本经用户明确授权放项目服务器私有新目录。三个best的LR/WD、轮次和每轮步数不同，不能把SOP几何差异完全归因于数据域。20实现测试通过；完整ID/数据/权重/日志均不进Git。

## 原版HIER best完整几何对照 — 2026-10-08

用户提供三份best与12份全缓存，并确认此次只看best。完成SHA/完整身份、官方
映射及全池官方FP32距离复核，whole train/eval/aug全部在d0≈4.6的cap薄壳，
std3–6e−7；官方NN4覆盖CUB1647/Cars1664/SOP432。原版SOP本身存在方向热点。
V7e300统一whole深度覆盖27→221；保留代理半径、随机方向五seed则等深度覆盖
1210–1295、raw仅5，支持径向偏置与方向冗余共同作用，不支持覆盖即结构证据。
恢复40批冻结sample角色，CUB/Cars pair与triple集合几乎完全重叠，SOP差98.6%
在±1e−4内；不是固定两层代理。增强全量对照及7数学/4身份/4角色核验通过。
见[45](45_ORIGINAL_HIER_BEST_GEOMETRY_AUDIT_2026-10-08.md)。CPU虚拟环境，零
训练更新/图像前向/GPU/下载；完整ID保留私有。没有启动新实验。

## V7 frozen direction evolution and hotspot identities — 2026-10-08

用户要求数据和合理解释；完成11快照/6基线同8856对象审查，完整缓存身份及
标签均核对，基线保留9840全池排名。新增全类方向、LOO、热点排除亲和、
代理方向冗余、同ID径向轨迹和基线映射脚本，输出六组科学图与离线报告。
whole早期按类别分离；末期raw花盆占88.13%，direction只1/2048槽。
最热四例e200仍深、无raw槽，e300才最浅；原HyCoRe last同ID接近最深端。
去径向后laptop方向集中仍在，193个inactive代理形成方向冗余群；全类亲和
去热点后稳健，但同代理e5到e300最近类别仅1.95%保持，不支持稳定类别绑定。
无前向/训练/GPU，源SHA不变，7项数学性质测试及全512行既有检索复核通过。
见[44](44_V7_DIRECTION_EVOLUTION_AND_HOTSPOTS_2026-10-08.md)。完整ID交付保持私有。

## Frozen radial selection diagnosis — 2026-10-08

用户要求原版径向统计与V7比较，并追问e5深层代理退出。读取已有完整8856
训练对象缓存、512代理及实际使用计数，新增CPU固定64对象/512查询、8噪声
生产ST梯度诊断；全部条件固定，仅径向移位对照，不更新参数。未选247代理
sample祖先概率近零，移到浅层中位深度后非碰撞pair/triple概率约7.5%/17.0%；
多数梯度极弱。62代理后期恢复、185未恢复，反驳永久硬退出解释。浅深两群
不是pair/triple两群；原版proxy完整统计已补，whole完整缓存缺失。见
[42](42_HIER_RADIAL_SELECTION_DIAGNOSIS_2026-10-08.md)与其公共标量证据。无新训练/GPU。

## Reusable post-training audit accepted — 2026-10-07

用户批准后完成v2扩展与V5/V6/V7、matched B64真实串联：31个冻结检查点、
每个全8856训练对象，相同输入SHA；11组固定噪声机制、15组独立形状检验。
V6采样纠错已验证，V7 e300数值饱和323→0、样本祖先188→267；raw top4
覆盖28→27，形态收益未持续全面超过B64。源数据不变、零优化更新、不新增
validation/test前向。结果和判读见[40](40_HIER_SYSTEM_AUDIT_RESULTS_2026-10-07.md)，
公开精简证据见[JSON](diagnostics/hier_system_v2_20261007_interpretation.json)。
最终151项服务器测试全部通过、无skip；三版e200真实编码器探针的8次重放
误差均0。V7加权HIER/base参数梯度比约0.242，V6约1.738，属于固定eval
条件观察。257组PNG/SVG及786个离线链接经本地/服务器校验；验收记录见
[v2 acceptance](diagnostics/results/20261007_hier_system_audit_v2_acceptance.json)。
下述v1接收记录保留为历史阶段，当前能力与结论以v2为准。

Implemented `tools/hier_postrun_audit/` and[38](38_HIER_POSTRUN_AUDIT_PLAN_2026-10-07.md).
CPU-only V7 audit reads all300 epoch summaries and outputs44 PNG/SVG scientific
figures plus offline HTML. No new training, GPU replay or official-test forward.
Fixed-object caches are still required for nearest-object retention/shape evidence.
The reviewed scalar/figure delivery stays outside Git; only the concise
[acceptance record](diagnostics/results/20261007_hier_postrun_audit_v1_acceptance.json)
is committed. V5/V6/V7 adapters preserve missing epochs and distinct activation domains.

## V7-HIER64-K20-W0 completed — 2026-10-07 14:43

Read-only audit16:00 confirms300x200 updates,984 validation objects per epoch,
finite telemetry, W0/lambda0.1 and exact proxy replicas. Process exited andGPU1/2
released. Best validation187 OA93.8008%; selected official test OA92.2204%,
AA89.7204%,2468 objects. Actual best/last SHA256 match. Post-step parameter cap6
holds in FP32 tolerance and proxy numerical projection count0 throughout.
Last validationOA91.0569%, clean train98.4982%; whole depth remains near boundary.
Single-seed historical testOA gains: +0.9724points vsV6-HIER64, +0.3647 vsV6-B64;
testAA is lower thanV6-B64. See[37](37_V7_COMPLETED_2026-10-07.md). No new experiment.

## V7-HIER64-K20-W0 detached — 2026-10-07 02:52

Explicit final decision authority received; unique300x200 run started02:48:51
Shanghai on idle GPU1/2, code165981f, new run v7_hier64_20261007_0250.
Five CPU contracts and2x2-update dual-GPU smoke passed, no official test read.
W0/lambda0.1, actual calibrated proxy depth2.240048, post-step parameter depthcap6;
V6 base protocol/global64 preserved. e1step59 verified finite and proxy replicas exact;
no full epoch or accuracy result claimed. Separate128-update CPU mechanism comparison
completed, extra experiments stopped, main remains detached. See[36](36_V7_PROXY_SAFETY_DECISION_AND_LAUNCH_2026-10-07.md).

## V6-B64 completed and final artifacts checked — 2026-10-06 18:40

Production finished12:13:15 Asia/Shanghai, continuing the complete shared
historical e20 through300. All280 new epochs are present with200 updates,
984 validation objects, finite telemetry and HIER weight0 (56000 new updates).
Validation-selected best is231, OA94.3089%; its single official test is
OA91.8558%, AA89.9994%,2468 objects. Best and e300 archive actual hashes
match recorded identities; manifest/heartbeat/console agree, process exited.
Historical V6-HIER64-K20-W20 best99 has the same validation OA and test OA91.2480%,
AA88.9169%; V6-B64 is+0.6078 OA points/15 objects in this single-seed historical
comparison. V6-B64 last validation OA91.2602%, clean train97.9788%, so late
validation decline also occurs without HIER. Geometry comparisons pending.
See [32](32_V6_BALANCED_B0_COMPLETED_2026-10-06.md). No new experiment launched.

## Completed mechanism evidence and V6-B64 progress — 2026-10-06 08:41

Read-only consolidation of completed natural16, shape4x64 and matched e40
results, plus10/10 completed augmentation/Gumbel/one-step jobs. The conditional
Gumbel8 panels add48 replica updates: all16 panel-repeat hotspot ST increments
relative to base are inward, but all16 total ST hotspot movements remain outward.
Natural16 direction and clipping qualification prevent a population collapse claim.
Matched e40 training OA is V6-HIER64-K20-W20 94.9526%, V6-B64 95.0090%; four inspected classes
have deeper V6-HIER64-K20-W20 embeddings. Independent shape signals weaken from99 to300,
without a matched V6-B64 shape comparison. Frozen-proxy six arms performed600
updates, then failed the strict proxy-only paired endpoint gate; controls pending.
V6-B64 completed198 epochs, training199, best validation94.1057% at191; official
test unread. See [31](31_HIER_MECHANISM_FINDINGS_AND_B0_STATUS_2026-10-06.md).

## V6-B64 authorized, smoke passed and detached — 2026-10-06

User approved the matched V6-B64 first, sustained mechanism experiments afterward
until usage limits, and no warmup for future newly initialized HIER runs.
Versioned v6_balanced_b0 preserves historical e20 complete model/momentum/
300-period scheduler/two-rank BN/RNG/sampler and continues e21–300; only
HIER/proxy updates are disabled. Six local/server contract tests and a separate
two-update dual-GPU restore smoke pass, with exact state checks and zero
gradient/parameter replica difference. Smoke test data were partial and
nonreportable, no official test was read, and its checkpoint cannot initialize
production. Code f649fcc on codex/v6-balanced-b0-mechanisms.

Production detached at 02:26:20 Asia/Shanghai after fresh idle-GPU checks,
using the independently hashed original e20. Full e21 saved at02:30:56 with
200 updates and984 validation objects; OA90.44715447%, CE3.01096090.
Its complete checkpoint was reloaded and passed the strict V6-B64 contract,
scheduler last_epoch21; training proceeded into e22.
See [29](29_V6_BALANCED_B0_AND_MECHANISM_LAUNCH_2026-10-06.md).

## Actual-use subset audit and pending next decisions — 2026-10-06

Read-only NumPy analysis joined matching-epoch sample pair/triple counts to
V6 best99/e200/e300 clean-eval caches by proxy ID. At each checkpoint, the
all-draw and active-draw masks coincide with its eligible nonboundary subset
(188 IDs); this does not assert unchanged identity across checkpoints.
Their maximum repeated raw top4 counts are 94/56/92; raw ID coverage 30/39/27,
direction ID coverage 328/173/163. The e300 used subset retains a 90.16%
bottom-radius 1% slot share. Its 297 extra eligible boundary proxies were
unused by sample at epoch300; 281 share the same dresser top4, explaining
373 = 92 + 281. This corrects the denominator interpretation, not the raw
retrieval result or proof of training descendants.

No inference, GPU, backpropagation or optimizer update. Sanitized subset
statistics and explanations are in [28](28_HIER_EVIDENCE_AND_NEXT_DECISIONS_2026-10-06.md).
[04](04_NEXT_EXPERIMENT_PLAN.md) now records pending fixed-condition mechanism
checks and a matched V6-B64 fork with resume-fidelity conditions; no launch.

## V6-HIER64-K20-W20 whole / radial proxy retrieval diagnosis — 2026-10-06

Existing V5-HIER64-K20-W20 e200, V6-HIER64-K20-W20 best99/e200/e300 and six
ORIG-B0-32-S4780/V6-B0-32/V6-B0-64 best/last caches were analyzed on CPU,
ten models aligned to the same8856 HIER training objects.
Sorted HDF5 versus glob-order ID mismatch was remapped by shard name/row;
all labels and final input-cache SHA256 checks pass. Code6e77c59,
branchcodex/hier-whole-cache-diagnostic. No GPU, inference, optimizer update
or new main training. Five numerical/identity/rotation checks and interactive
desktop/320px layout/data-update checks pass.

V6e300 raw top4 covers28IDs, whole equal-radius190, proxy equal-radius30.
373/485proxies share the largest raw set; whole equal-radius reduces this
to53, proxy equal-radius retains367. Deleting89lowest-radius objects yields
51IDs and281repetitions of a new dresser set;87.68%of slots come from the
remaining bottom1%. Borrowing V6-B0-32 radii retains concentration but shifts
identities;99.95%of slots now come from the borrowed bottom1%. Equal whole
radius and direction rankings are exactly equivalent, not independent evidence.

V6e300 dresser median depth/angle3.558/18.76degrees versus V6-B0-64 last4.568/2.97;
overall depth medians4.617/4.684. Current controls diagnose frozen retrieval
radial bias, not which loss caused the class-specific change. Existing baseline
protocol differences still preclude isolating the HIER objective. See[27](27_HIER_WHOLE_RADIAL_RETRIEVAL_DIAGNOSIS_2026-10-06.md).

## Original HyCoRe/V6-B0-64 matched class geometry and no-update batch probes — 2026-10-05

Six best/last checkpoints of ORIG-B0-32-S4780, V6-B0-32 and
V6-B0-64 were evaluated on full9840train and2468test, identical clean1024whole
and ID-stable nested200/400/600point patches. Alltest OA reproduce original
logs;12splits/1920class rows and three4batch no-update probes passed matched
input/checkpoint hashes, finite, unchanged/restored BN/RNG/parameter checks.
See[26](26_ORIGINAL_HYCORE_B64_CLASS_GEOMETRY_2026-10-05.md).

V6-B0-64 best clean-test whole depth4.139 is shallower than historical4.364 and
source4.390; final is deeper than both. Original flower_pot was1/20 correct,
source2/20, V6-B0-64 3/20. Historical last clean train/test gap6.725pp and V6-B0-64
6.583pp show this ordinary gap is not unique to V6-B0-64. Historical logs also
contain an early resume/repeatedepoch5 and independent-bestAA metadata trap.

Fixed-input trainBN increases whole–part400 gap to about3.0 from eval below.9
in allthree models. Regrouping local32 with fixed negative IDs yields mean
whole hyperbolic movement5.7–6.0 while class prediction changes at most.391%
in these256probe anchors. This confirms conditional geometry sensitivity,
not a full training cause or unique V6-B0-64 failure. No optimizer updates or
new main training occurred; source paths/checkpoint files remain untouched.

## V6-B0-64 completed and two-document handoff — 2026-10-05

V6-B0-64 finished all300epochs at09:24:55Asia/Shanghai in8h35m08s. Best-test
OA94.1653%/AA91.7116%@267; final300 OA93.1524%/AA90.7872%. Every epoch
used153updates/9792distinct IDs/drop48 and all2468test objects. Actual
best/last SHA256 matched their save records;600first/last-step replica
checks were zero, all finite checks passed,15periodic archives exist.
Full9840 cumulative training coverage was reached by epoch2.

V6-B0-32 best-test is93.841%@216 and final93.233%; V6-B0-64's single-seed
best advantage is about.324pp, not a robust batch-size gain. Final V6-B0-64
clean-train OA99.7358% exceeds same-epoch test by6.5834pp; flower_pot
is3/20 correct at best and final. The base model also has high whole
boundary proximity, so this observation is not specific to HIER.
Different V6-HIER64-K20-W20 data/sampling/update/selection protocols preclude treating
this V6-B0-64 as a matched zero-HIER control.

See[23: completed V6-B0-64](23_V6_B64_SHUFFLE_START_2026-10-05.md),
[25: technical handoff](25_HYCORE_HIER_HPCS_TECHNICAL_HANDOFF_2026-10-05.md),
and the ignored local
[24: private engineering handoff](/D:/Hycore/.codex-local/handoff/24_PROJECT_ENGINEERING_HANDOFF_2026-10-05.md).
Only reviewed derived curves and sanitized documentation enter Git; no
GPU analysis, new training, checkpoint alteration or result deletion occurred.

## V6-B0-64 baseline authorized and detached — 2026-10-05

The user selected full9840training and original every-epoch official test/
best-test selection. One new300epoch baseline uses153shuffled no-replacement
global64batches per epoch (9792distinct IDs,48dropped), two local32 ranks,
original c1/D256 CE+intra, alias/BN/FPS and SGD/cosine/norm1; no HIER/proxy.
Production code commit `56b40a2b11d3274ff7a9832b8e4cced3b366ec5c`, branch
`codex/v6-b64-shuffle-baseline`; detached at00:48:45Asia/Shanghai on two idle GPUs.
Eight sampler, two real source-criterion and six distributed/operator CPU
checks passed. A fresh2epoch x2step GPU smoke passed with zero gradient and
parameter replica differences, finite/clipped updates and reloadable full
checkpoints; partial smoke test results are not experiment results.
See[23](23_V6_B64_SHUFFLE_START_2026-10-05.md) for production acceptance.
This changes several V6-HIER64-K20-W20 protocol factors and is not the pending matched
balancedglobal64 V6-B64. No additional matrix was launched.

## V6 completed and V5/V6 structure audit — 2026-10-04

V6-HIER64-K20-W20 completed300x200 updates at12:01:52Asia/Shanghai; original-source
V6-B0-32 completed300x307 at09:53:24. V6-HIER64-K20-W20 bestval94.3089%@99 yields final
testOA91.2480%/AA88.9169%, versusV5 test92.1394%/AA89.1581%.
Original V6-B0-32 best-test93.841%@216 is an engineering source reproduction,
not a matchedglobal64 or validation-selected control.

Read-only V6 best99/e200/e300 visualizations each evaluated all8856clean
training instances. At e300,485proxy anchors are eligible, but373 share
one dresser top-four set; sample-to-nearest-proxy entropy-effective count
is15.59, versusV5 25.43. Late actual sample ancestors are shallower than
endpoints and pair is deeper than triple98.17%; proxy ancestor ordering
and triple usage concentration remain concerns. Single-epoch coverage stays
about66%, both runs finite and proxy replicas identical. A fixed64batch
cache control confirms radial sensitivity of late sample reciprocal graphs.

Self-k exclusion, extended cosine/budget and diagnostic input-clone fix are
reported separately. See[22](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md)
for metrics, units, limits and proposed diagnostics. No new main training.

## V6 authorized — 2026-10-04

The user approved dual-GPU V6-HIER64-K20-W20 with V5 operators/hyperparameters, sample
and proxy index self-k excluded, and a300epoch cosine/budget. The spare
single GPU is assigned to V6-B0-32, full9840train
and307steps/epoch for300epochs. Original test-every-epoch selection is
retained for this requested engineering reproduction; V6-HIER64-K20-W20 keeps validation
selection and final-test-only evaluation. A V5epoch200 read-only proxy
top-four point-cloud script is requested first. See[21](21_V6_TRAINING_START_2026-10-04.md)
for actual implementation, checks and launch status. Both jobs launched at
01:33Asia/Shanghai with commitb3542cb, completed their first full epoch
checkpoint and enteredepoch2. V6-HIER64-K20-W20 usesGPUs1/2 and V6-B0-32 usesGPU3.
The startup gates and all8856-instance V5 visualization have passed; no final
V6 training result is reported at launch.

## V5-B0-32-CAP200 source-fidelity audit — 2026-10-03

Read-only training/source audit confirms that V5-B0-32-CAP200 changed more than the
200-step cap: it also uses a held-out split, 200-epoch cosine period, rewritten
RNG flow, workers2, and validation selection. The model directories match,
and original part overwriting, two BN updates, CE/intra margins and weights,
RiemannianSGD and norm1 clipping are retained. Source default seed22 matches
V5-B0-32-CAP200, whereas the saved94.044% historical run used seed4780/300epochs/workers8.
Strict hinge-zero and floating-point grouping differences are also disclosed.
See [20: complete audit](20_B32_HYCORE_FIDELITY_AUDIT_2026-10-03.md).
No new training or training-code change was performed for this confirmation.

## V5 completed — 2026-10-03

Both jobs completed200epochs ×200steps with finite telemetry and one final
official-test evaluation on the validation-selected best checkpoint.
V5-HIER64-K20-W20 bestval94.0041%@179, testOA92.1394%/AA89.1581%; V5-B0-32-CAP200 bestval94.5122%@172,
testOA92.5851%/AA89.3285%. Both epoch200 validationOA is93.0894%; cleantrain
OA is99.6161%/99.4354%. V4's severe late inference decline did not recur.
V5-HIER64-K20-W20 has substantial whole/proxy radial saturation and only about32% proxy
anchor eligibility late in training. V5-B0-32-CAP200 differs in batch and sampling,
so its advantage is not an isolated estimate of the HIER effect.
See [19: complete result and monitoring summary](19_V5_FINAL_RESULTS_2026-10-03.md).

## V5 production launch authorized — 2026-10-03

The user approved200×200 V5-HIER64-K20-W20 (two local32 ranks, global64) and a spare
single-card V5-B0-32-CAP200 operator-stability diagnostic, including all recommended
monitoring. V5-B0-32-CAP200 uses an epoch permutation capped at6400 distinct instances;
it is not a matchedglobal64 V5-B64 or a full original-protocol reproduction.
Implementation and bounded startup checks are recorded in
[17](17_V5_TRAINING_START_2026-10-03.md); actual running status is recorded
after detached jobs and completed updates/checkpoints have been verified.

## Environment and resource audit

- Server environment: Python 3.9, PyTorch 2.8.0 + CUDA 12.8, Geoopt 0.5.1.
- Hardware: four RTX 5090 GPUs. The first run used only GPUs 2 and 3 after confirming they were idle.
- ModelNet40 data and historical checkpoints remain server-local.
- Original HyCoRe reproduction (`...hycore_var-4780/best_checkpoint.pth`): epoch 229, OA 94.044%, AA 91.545% as recorded in the checkpoint.
- Derived A3 checkpoint independently evaluated at OA 93.9222% and AA 91.5907%. A3 starts from the original HyCoRe reproduction and continues with `alpha=0`, so it is not the original HyCoRe weight.

## Correctness findings in the legacy path

1. Legacy `gromov_product()` treated equal-shaped tensors elementwise rather than producing an `[N,N]` matrix.
2. Cached teacher similarities were sorted by global ID while student embeddings retained random batch order.
3. Class-balanced arguments existed but were not connected to the DataLoader.
4. A single random augmented view was used as a teacher cache.
5. No same-protocol `beta_inter=0` control existed.
6. Student-derived target distances risked circular supervision.
7. Large-learning-rate fine-tuning confounded inter-objective effects with protocol drift.

## Teacher gate

### Cosine teacher — stopped after one epoch

- Reliable triplets: 229 per epoch.
- Initial training-batch satisfaction: 88.65%.
- Interpretation: class-discriminative cosine features have too little within-class dynamic range and too little optimization headroom.

### Frozen hyperbolic-distance teacher — passed signal gate

- Average reliable triplets: approximately 17,050 per epoch.
- First-epoch satisfaction: 59.11%.
- Finite, non-zero inter gradient.
- Interpretation: adequate coverage and headroom; selected as the first main teacher.

## Twenty-epoch seed-22 comparison

Shared setup: learning rate `5e-3`, class-balanced batches, original HyCoRe losses, identical seed and schedule.

| Setting | Best OA | Best AA | Final OA | Final AA |
|---|---:|---:|---:|---:|
| Derived A3 initialization | 93.9222 | 91.5907 | — | — |
| Same protocol, `beta_inter=0` | 93.4360 | 92.3192 | 92.0583 | 91.3448 |
| Hyperbolic teacher + equal radius + exact LCA | 93.4765 | 91.8157 | 92.1394 | 91.3029 |

The main method exceeds the control by only 0.0405 OA at each run's best epoch and by 0.0810 OA at the final epoch. These are not meaningful single-seed classification gains.

## Fixed no-augmentation structural evaluation

Same 50 class-balanced batches, 7,000 within-class pairs, and 3,440 reliable triplets:

| Checkpoint | Spearman | Satisfaction | Ranking loss |
|---|---:|---:|---:|
| Derived A3, evaluated against an A3-derived teacher | 0.8050 | 72.238% | 0.03974 |
| `beta_inter=0`, final | 0.4576 | 59.535% | 0.05951 |
| Main method, final | 0.4708 | 60.320% | 0.05257 |

Relative to the zero-increment control, the main method improves Spearman by 0.0133, satisfaction by 0.785 percentage points, and ranking loss by about 11.65%. Both fine-tuned models remain far below their A3 initialization. Because teacher and reference are both derived from A3, the high A3 Spearman is a self-consistency measure, not evidence that original HyCoRe already contains a valid morphology hierarchy. The defensible conclusion is only that inter ranking weakly mitigates drift relative to A3 self-distillation.

## Known protocol flaw to fix

The first v2 run evaluated the ModelNet40 test set after every epoch. The next protocol must create a deterministic stratified train/validation split, select checkpoints using validation metrics, and use the official test set only for final reporting.

## V4 completed and V5 review — 2026-10-03

The shared-whole HIER V4 matrix completed200epochs at seed22 on the fixed
8856/984 train/validation split. Validation-selected final test OA was
88.3712% for LEGACY-V4-B32-Q4x8-BN1,89.5462% for LEGACY-V4-HIER32-K3-W20-R20-Q4x8-BN1 and88.1686% for LEGACY-V4-HIER32-K5-W20-R20-Q4x8-BN1. The shared protocol
also showed substantial late inference decline. LEGACY-V4-B32-Q4x8-BN1 changed part storage,
BN updates and batch construction, so it is not an original-protocol
HyCoRe reproduction. See [V4 lessons](12_V4_LESSONS_2026-10-03.md) for
evidence and limits of causal conclusions.

The user requested lessons and a training-change catalogue before the next
dual-GPU tests. [V5 review](13_DUAL_GPU_V5_CHANGE_INDEX_2026-10-03.md)
records coverage calculations, source-fidelity choices, script interfaces
and acceptance gates. No V5 GPU run has been launched; full training awaits
review of the current plan.

## V5 bounded diagnostics completed — 2026-10-03

The user authorized schemeA/200step,c1,D256,random proxies,numeric
margin/tau=.1,no additional HIER cap/backward hook,and global64 for all
three losses. Source sampling/relations, restored HyCoRe operators and
distributed gradient checks passed18 CPU checks. A16-step dual-rank joint
diagnostic completed with finite values, BN updated twice and identical
proxy replicas. Fixed-feature K/LR checks followed; no main training or
test/validation model evaluation was launched.

See [V5 diagnostic results](15_V5_DIAGNOSTIC_RESULTS_2026-10-03.md).
K10 and proxy LR.01 are coverage/numerical candidates only. Mature-reference
anchor coverage remains low and shadow cap rates increase, so short-run
success cannot be extrapolated to200epochs.

## V5 expanded K diagnostics and main-test recommendation — 2026-10-03

Additional detection authorized by the user completed64global64 plans with
two augmented training views each. K10/12/16/20 anchor coverage was
37.30/42.38/51.94/60.39%; all40classes had eligible anchors in the observed
window. Paired cross-class reciprocal graph stability remained low and
radius-dependent; this is not independent morphology evidence.

An8-step K20 joint diagnostic from a read-only V4epoch20 model copy passed.
Actual shared-encoder HIER/base parameter-gradient norm ratio at lambda.5
was.327 then.503, correcting the much larger partial-gradient ratio at mu.
Peak memory was28.73GiB/card; whole output frequently approached the native
numerical ball boundary, so finite short-run output is not a long-run guarantee.

See [16: expanded results](16_V5_TOPK_FOLLOWUP_2026-10-03.md) and the
current [main-test settings](04_NEXT_EXPERIMENT_PLAN.md). K20/lambda.5,
randomP512/D256,c1,proxyLR.01,and20base-only epochs within200total epochs
are recommended for review. No main training was launched; GPU jobs exited.
