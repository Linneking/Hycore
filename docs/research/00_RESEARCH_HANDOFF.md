# Research handoff: HyCoRe inter-sample hierarchy

## 2026-10-30：F0/G0/H0零正则三组已在原服务器启动

用户要求F/G/H两正则置0、在原服务器三空闲卡运行，已执行[121](121_PSEG3_FGH_ZERO_REG_2026-10-30/execution.md)：独立v7源ff91a07已推送隔离部署，本机/服务器各三组CPU完整2×2通过，真实N2048/B32短测各49项通过，17:16:40北京fresh200。F0/G0/H0为GPU0/1/2；17:19独立验收均e2、首轮437更新/BN437/test2874，actual loss=CE、R训练梯度None、有限状态/几何日志完整、后台父进程1。F0/H0首轮输入/CE/测试指标完全一致。原8002正则开启FGH不动；所有旧队列禁止重启，本次只追加新零正则目录。私有根/PID见.codex-local/PSEG3_V7_FGH_ZERO_HANDOFF_2026-10-30.md，不重复提交。

## 2026-10-30最新迁移：新服务器优先F，旧三队列已停止

迁移已完成；**最新F/G/H分别GPU3/0/2**。旧有效三等待队列全部核验停止，原记录保留。新源c8b93dc冻结、数据/环境/初始hash一致，42CPU合同和三组CPU入口通过，三组真实短测各47项通过，F优先14:06生产、G/H14:08后生产。14:14独立断开重连验收各首轮437更新/BN437/test2874/有限状态及完整几何梯度日志。用户随后要求F别用GPU1；已停旧F自有组，从第5轮/update2185状态迁GPU3，14:31第6轮/update2535验收父进程1、GPU1释放，模型/优化器/调度/RNG精确恢复，总预算仍200。F需查新续训目录，G/H原迁移目录不变；禁止重启旧服务器队列或新机GPU1旧F。[120](120_PSEG3_NEW_SERVER_MIGRATION_2026-10-30/execution.md)为准，私有路径/PID见.codex-local/PSEG3_NEW_SERVER_MIGRATION_2026-10-30.md。密码不落盘，下方原服务器安排均历史。

## 2026-10-30最新调度：H改GPU2，取消等待F

用户明确“H挂在gpu2”，覆盖以下旧GPU2禁用/H等F安排。新H源ef9603b独立部署，本机/服务器26调度合同通过，model/train/losses/configs与b59c9a6相同。13:15北京旧H仅等待程序核验后停止，旧结果保留；新H13:16登记GPU2且无F依赖，13:18独立确认父进程1、waiting_for_idle_gpu，未GPU短测/生产。GPU2仍有他人任务，等完全空闲，不打断。F/G原进程/GPU0/1及冻结源b59c9a6不动。详见[119](119_PSEG3_H_GPU2_REQUEUE_2026-10-30/execution.md)，私有新H根/PID见.codex-local/PSEG3_V6_H_GPU2_HANDOFF_2026-10-30.md；勿重复启动旧或新H。

## 2026-10-30：用户已授权F/G/H实施和排队

用户确认117后明确“挂起实验”。源b59c9a6已推送/同提交隔离部署，独立v6本机/服务器各39合同、三组完整CPU两轮×两次更新通过。北京时间12:50挂起F/G/H，12:51重连确认三进程父进程1、命令/源码/配置匹配。F/G分别等GPU0/1完全空闲，H等F成功后用GPU0；GPU2禁止使用，0/1均为其他用户占用，尚未GPU短测或200轮生产。三组均原Conv9BN9Leaky+EXP+两M、两正则各.01；F删dp1/Rpre9，G留dp1/Rpre9，H删dp1/Rpost9。seed4780及初始化/既有200轮配方不改。勿重复提交或向冻结源拉后续文档；私有路径/PID见.codex-local/PSEG3_V6_FGH_HANDOFF_2026-10-30.md。[118执行记录](118_PSEG3_FGH_LAUNCH_2026-10-30/execution.md)为准；下方仅设计/不启动是此前历史状态。

## 2026-10-28：用户授权三组原生HyCoRe-seg，8000三卡正式训练已启动

用户明确要求挂P0/Pglobal/Pseg3，seed4780/batch32/shuffle；不增加H0或其他矩阵。独立v2实现`ac6c712`已推送，原分类和v1未改；服务器HTTPS TLS失败后同提交Git bundle+fetch/pull --ff-only隔离部署。18本机/18服务器CPU合同、三组各17项合成全入口和真实batch32/N2048各2×2短测均过。三卡分别GPU0/1/2于22:15:25 fresh200，22:18核对e1完整437更新/2874test/BN437，e0-best-last和状态均有限，现e2；初值匹配短测、三组首输入SHA相同。运行ID及证据见[99执行记录](99_HYCORE_SEG_NATIVE_AUDIT_2026-10-28/execution.md)，勿重复提交或向活动源码拉文档后续提交。原trainval/test选模及26共有ID协议保留并已披露；原flip、1000/N′/4、RSGD.01/cosine200/clip1、两正则各.01。首批whole/part均触球投影边界、深度未拉开，正则共享梯度非零不能证明层次项有效；诊断留存，不静默改尺度。训练未完成，无自动监控或下一矩阵。

## 2026-10-28：第三组恢复one-hot与1280融合，官方新尺寸Möbius原层CPU初始化检查通过

用户要求Pseg3保留原DGCNN类别输入，恢复one-hot16→64、1024global+64category+192local=1280融合；P0/Pglobal仍无one-hot。三层继续复用官方MobiusLayer类及reset_parameters，仅按256→256→128→50新建，不复制旧分类权重。原层独立CPU/FP32检查在seed4780下通过：同seed初始state一致，两组模拟尺度的前向/输入及_weight/_bias梯度均有限且非零，无CUDA初始化、无optimizer.step。证据99/native_layer_init_probe.json，源码LF哈希与本机对应原文件核对；完整v2、真实特征分布/GPU及训练尚未验证。此轮仅原层CPU探针及文档/图修订，未改服务器源码/训练或结果。E3静态参数恢复类别后相对原AnTao净少334，不再扣去原类别支路17536。

## 2026-10-28：用户新增第三组Pseg3，保留DGCNN广播融合与三层同维双曲头

用户认可P0/Pglobal之外，再补原DGCNN分割MLP同维双曲替换，并指定保留global池化/广播/局部融合。99计划§7和图已定义Pseg3：去one-hot但保留1024global+192local=1216融合→256（BN8/激活/dp1保留），H[B,N,256]→exp→原Mobius256→256→128→50三层；该256维欧氏H用于正则。三层不含前融合层，尾部BN9/10、激活/dp2拟按HyCoRe头风格不保留，配套变化与仅同宽替换区别说明。此组是分割骨干结构参照，P0/Pglobal仍保留分类骨干主线；不把骨干/头宽差异归因于单独层数。仅补设计/图，无模型代码修改或新训练。

## 2026-10-28：P0主结构获用户认可，全局拼接列为优先消融

用户认可简单逐点原生适配：原conv5输出[B,N,1024]→共享linear1(1024,512)→H[B,N,512]→原exp/emb512→256/mlr256→50。拟比较Pglobal=concat(T1024,广播全局max1024)→共享linear1(2048,512)，其余结构/协议相同，无one-hot；比P0多524288权重，不与旧额外低层绕接混合。99报告和计划补齐分类普通Linear与HyCoRe两Mobius的维度对应及BN/dropout差异。仅认可设计方向，尚未实施/排队新训练；此前旧桥候选继续仅留历史。

## 2026-10-28：用户三批注后简化seg主候选，去one-hot/新增桥，保留原flip

当前主候选见[99修订审查](99_HYCORE_SEG_NATIVE_AUDIT_2026-10-28/audit_report.md)与[计划](99_HYCORE_SEG_NATIVE_AUDIT_2026-10-28/reproduction_plan.md)：保留HyCoRe四EdgeConv/原内部cat/conv5，主路径省去分类末尾global max+avg，原linear1入维2048改1024、输出512并逐点调用，沿用BN6/激活和完整exp/emb/mlr，输出[B,N,50]。不输入one-hot，不新增1040融合桥；原encoder内部多层cat保留，额外绕接仅列后续消融。两篇论文负part均定义为异物体异类，非分割新增；按用户源码优先意见保留官方flip，仅统计同类/自身比例，不擅自换采样器，明确存在异类候选不等于flip配到异类。

新主候选静态参数较原分类减少521718，较旧桥方案减少1057792；BN需[B,512,N]或[B*N,512]。以下保留global/1040桥的条目为历史初稿，已由本条覆盖。本轮仅文档/流程图修订，无v2实现、服务器连接、GPU或新训练。

## 2026-10-28：HyCoRe-seg按用户重申改走官方分类源码原生迁移，审查与图示完成待审

用户明确要求从官方HyCoRe DGCNN分类模型改造分割，尽量完整复用组件；此前AnTao分割末层简化替换的v1不符合该源码继承路线。见[99审查报告与组件对照图](99_HYCORE_SEG_NATIVE_AUDIT_2026-10-28/audit_report.md)、[复现计划](99_HYCORE_SEG_NATIVE_AUDIT_2026-10-28/reproduction_plan.md)。建议保留四EdgeConv/conv5/global max+avg/原linear1-BN6/完整exp-emb-mlr，新增local512+global512+category16→512逐点融合桥，mlr40→50；正则从512维欧氏H池化后exp，复用原损失核/opposite_hier=True并适配异类负例。官方_weight/_bias/transp0完整保留，不再用裸W替代。

99包含模型及正则两组PNG/SVG与Mermaid对照图、三专项源码/论文审查和验收方案。原生LR.01/clip1与显式plain CE属待审主建议；seed4780、1000/N′/4、既定trainval/test选模继续。新增融合与四层编码器不同于论文未公开分割结构，参数量和显存须独立核验。此次仅本地静态审查/文档/绘图，未实现v2、连接服务器、用GPU或新增/停止训练；旧v1和结果保留。禁止把该计划当作已授权的新矩阵排队。

## 2026-10-28：分割选模已由用户明确改为AnTao默认，三匹配任务已提交等待数据

用户要求HyCoRe-seg实验，核查选模后选择train+val训练/逐轮test Instance mIoU选best，全部seed4780；覆盖旧分割val偏好，不中途改已有val基线。见[98选模与任务状态](98_HYCORE_SEG_REPRODUCTION_2026-10-27/selection_protocol.md)。源1ffd9f5，16本机/服务器CPU及合成test-selection完整入口通过。新数据恢复原raw列表trainval14007/test2874（保留原列表重复身份，非原HDF5逐位复现）。DGCNN/H0/HC分别已提交GPU0/1/2，运行ID见98；10:06状态均waiting_for_prepared_data，尚无新协议生产结果，数据完成且空闲GPU短测通过后自动fresh200。禁止重复提交。旧val基线继续后台，源c153715不动。

## 2026-10-28：DGCNN分割基线已在8000启动，HyCoRe-seg完成代码与CPU验收

用户授权实现、下载数据/建环境并在8000训练。见[98执行记录](98_HYCORE_SEG_REPRODUCTION_2026-10-27/implementation.md)：独立分割目录、固定MIT DGCNN、16本机/服务器CPU合同及真实batch32/2048点2×2 GPU短测通过；普通SGD.1/.9/1e-4、200轮cosine、seed4780、验证选模/最终一次test。2026-10-28 09:41:32在GPU0 fresh启动200轮，runID `dgcnn_seg_v1_s4780_20261028_A/main`，源码c153715；首轮378更新/1867验证/BN378/best-last及与短测初始化/首输入一致验收通过，09:44已e2step350。禁止重复提交；不改旧训练或结果。

公开normal-bearing数据去normals后XYZ归一化/first2048适配；原官方清单9/23/3跨split重复ID按test>val>train去重，实际12105/1867/2874，来源和SHA保留，不能冒称原HDF5逐位复现。HyCoRe-seg按1000/N′与4、单次编码/whole浅part深实现并通过合成完整入口CPU检查，尚未做真实数据GPU验收或完整训练。仅基线后台进行，无新增矩阵/自动监控；RSGD首次动量不同于SGD，已明确归为协议差异。

## 2026-10-27：用户授权TS25，TS23初始化系数0.9→0.7完整消融

用户选TS23/B13而非TS20，明确在8000跑完整300轮。新别名TS25/B13，CB32有放回16类×2／200步、Ks10/Kp10、lambda.1、A1 AdamW.005/1e-4、seed4780；唯一配方变化为高斯初始化系数.9→.7，保留原方向及模长差异，不额外等深归一。见[97配置](97_TS25_TS23_INIT07_FULL_2026-10-27/run.md)。GPU1先核查空闲，独立train-only2×2短测和完整保存状态通过后fresh300×200，保留原TS评估及300余弦；不复用短测权重。6本机合同、服务器CPU检查及独立2×2短测均通过；2026-10-27 17:01在8000 GPU1启动，正式训练已完成e1/200步并保存e0/e1/best/last，进入e2。运行目录标识`ts25_ts23_init07_full_20261027_170102/main`，源码`c9f0c3a`，脱离会话运行；禁止重复提交。完整证据私有保存，97目录含脱敏验收JSON；300轮尚未完成。


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


## PartHier五问后续审查已在8000完成

用户指定8000实施，源e4c757c；12项服务器CPU合同及两PartHier固定200批读出精确回归通过。4×9840输入不变，13文件SHA复核、无编码前向/优化器/网络反传，仅输出叶子梯度；GPU0已释放。见[五问结果](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/followup_results/report.md)：5组PNG/SVG、14汇总CSV、反例卡和summary，28文件取回SHA一致并逐图检查。Q-p200供给中位1.70/2.06，入选A约1.41；等深副本使p200概率94.20→45.51%、91.09→53.59%。同类通过组A深3.67/4.07而失败组实际ij角59.31/77.77°；热点实测贴近评分概率预期。正part夹角总体缩小，径向无直接切向驱动；ord反例A约79%/77%为p200，HIER六例均有同池可行替代但不外推总体。原始逐关系留8000，未扩大训练或修改旧结果。本地只核验与整理文字；本次无待执行任务。

## 2026-10-26：PartHier五问后续审查已批准，改用8000实施

用户明确“同意此计划…使用8000服务器”，覆盖此前8001限制，科学范围仍[94五问方案](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/followup_plan.md)。4份已验收9840缓存和各自Q/94参考已复制到8000独立目录，13文件SHA匹配；GPU0/1检查空闲。新增版本化parthier_followup95_v1，局部输出叶子梯度、固定ijk选择重放、供给/热点/等深副本、ord/chain/HIER反例，不构建编码器或优化器。11项本地合成合同已通过，服务器原生梯度检查及正式200批读出以后续状态为准。只用已有冻结特征，不重训/重提，不写旧结果。

## 2026-10-26：94读者追问，新增五问审查计划待审

用户要求解释浅p200祖先/热点、同对象part–whole方向、统一对比margin4与径向1000/N、ord及HIER反例。见[94后续计划](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/followup_plan.md)，逐问列疑问/实验/输出；拟仅在8001复用缓存，正常200批ijk冻结读出，不训练或重提。已纠正A为双whole minimax候选、ord≠chain、HIER无额外代理–代理项。用户要求旧图删重复数量行，仅保留百分比及ECDF；该版式重绘复用原汇总，另新目录，不影响科学数据。新增分析仍是计划，等待用户审阅，不自动执行；保留report.md内用户新增疑问。

## 2026-10-26：93总账TS23/TS24两组300轮结果已补全

用户报告8000两组结束并要求补93。20:03:56只读采集确认两条均300轮／60000更新／BN120000、exit0与保存验收通过，best/last存在、GPU0/1空闲。TS23（Ks10/Kp10/lambda.1）best e262 OA94.003241／AA91.827905、e300 OA93.719611、末10均值93.350891±.235880；TS24（Ks10/Kp20/lambda.05）best e277 OA93.598055／AA91.086626、e30092.463533、末10均值92.739060±.312178。93报告、CSV、注册表v4、来源与缺口同步，24正式记录现23完成/1中断/0运行；TS15未到300轮的空值不造补。其他22条配置/分数保持；保留用户在93中删去训练OA章节的当前结构，训练OA完整保存在CSV/JSON。本次零训练/前向/服务器写入；采集和11表/轮次/同轮AA/末10独立复算检查通过。

## 2026-10-26：94 PartHier嵌入审查已在8001完成

用户唤起后只读确认原后台已19:19完成，未重复启动，全部计算/绘图仍在8001。[94结果](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/report.md)及5组PNG/SVG、8张汇总CSV、summary/核验已取回，21原始派生文件SHA一致并逐图查看。TS18 best/e300的chain三边0.2联合仅7.774%/4.024%，原版R2为24.949%/14.268%；末轮p600→whole逐对象深度差中位0.000127、基本次序50.081%。ord/HIER设计间隔联合80.061%/70.993%→68.518%/67.681%，A/B均主要选p200（A94.17→91.00%、B99.92→99.80%），不能把祖先关系成功写成三尺度分层成功。i/j同类仅8.37→9.73%，其A到共同类中心中位4.88°→2.56°。配套历史Q只读保存状态，不是训练重放或匹配H0。零训练/BN更新/test前向，GPU2/3已释放，本地仅报告整理。来源提取807a5d6、统计6cc302b；后续无待追加计算。

## 2026-10-26：94审查已提取四权重，用户额度告罄暂停主动工作

用户要求保存进度，后续坚持8001，等其唤起。见[94当前交接](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/HANDOFF_2026-10-26.md)。四份全9840缓存均已在8001严格FP32提取并通过冻结状态/相同输入/前向一致性验收；现有readout后台保留，最后为running/metrics、`MODEL TS18_best start`，经历较长磁盘等待，尚无结果报告。源码807a5d6（提取）和6cc302b（统计）已推送并同步。停止本轮主动监看或新增工作；本地仅备份，未启动CPU统计。下次先读现有状态，避免重复启动；仍在8001完成五组图与附表。

## 2026-10-26：PartHier v1简化嵌入审查已获执行授权

用户明确“开始检测吧”，批准执行[94简化审查](94_PARTHIER_V1_EMBEDDING_AUDIT_2026-10-26/plan.md)：TS18 best e201/e300与原版TS02 R2 best e189/e300共4权重、全9840对象和全量类别中心；正常ijk→A/B→检查原关系。五项主结果为深度数量/百分比分布、chain、ord/HIER、A/B尺度来源及类别中心角度，原intra仅附表。PartHier用各自保存队列，只读不续写；不恢复用户删去的复杂控制。8001 GPU2/3经检查空闲；独立目录、权重/数据身份校验、冻结BN和无梯度提取。原版仅参照，不是匹配H0；保存队列含历史train-mode特征，不称精确训练重放。当前处于采集准备阶段，尚无审查结果，不新增训练。

## 2026-10-26：8001恢复，93总账缺口已补齐

用户明确8001连接恢复并要求补全统计。本次12:56只读原始manifest/best轮/末轮/末10轮及进程列表：TS15实际保存54连续轮、best e52 OA91.855754/AA90.107554，当前无对应进程，manifest旧running属残留，归为已中断，不恢复训练；TS18 PartHier已300轮，best e201 OA92.990276/AA90.861619、e300 OA88.006483、末10均值89.444895±1.609896，e300在线train99.515625。TS19/20末10均值/SD及TS16/17 best同轮在线train精确值均补齐。93报告/全部CSV/注册表v4/恢复采集来源/缺口表同步更新；24正式记录现21完成、2运行、1中断，无状态未知。8000顺带更新完成前缀TS23 e101/TS24 e98；仍不填e300。剩余19空字段均为未完成终点或协议未记录（含PartHier clean/eval），没有连接失败造成的缺口；10个Markdown表及原14条已完成分数/原15条身份保留检查通过。未新增GPU计算、前向、训练、重启、停止或文件改写到服务器。

## 2026-10-26：V9至今实验总账93已统一补录

用户要求沿75风格统计全部V9及后续实验、统一新增命名并补齐缺失数据。见[93总账](93_V9_EXPERIMENT_ATLAS_2026-10-26/report.md)：保留TS01–TS15/B01–B10，新增TS16–TS24/B11–B14；24正式记录（20完成、2进行中、2后续未知），另列5条机制实验。B05严格仍SH32×307；B05几何的CB64改称B12、CB32改称B13。CSV/命名映射/注册表v4/缺口表及脱敏来源齐全；旧75报告的用户未提交修改未动。采集截止10:39:59，最新两组各完成17轮，仅前缀best，不填e300。8001拒绝连接：TS15尾段、TS18首轮后、TS19/20末10轮、TS17 best同轮在线OA仍显式缺失；其余可取得的完整成绩和训练OA已补录。只读现有文件、无GPU/训练/BN干预；数据单位、轮次、原15条身份/分数保持及9个Markdown表检查通过。没有扩大实验或自动等待服务器恢复。

## 2026-10-26：用户批准继续B05 CB32 A1两组单变量消融

8000 GPU0样本K10／代理K10／lambda0.1；GPU1样本K10／代理K20／lambda0.05。相对已完成91第二组K10/K20/lambda0.1，分别只改代理K和lambda。共同严格B05固定壳新HIER、CB32有放回16类×2、AdamW.005/1e-4、seed4780、fresh300×200、原TS协议；直接授权启动，不续旧权重或扩大矩阵。见[92配置](92_B05_CB32_A1_K10_WEIGHT_FOLLOWUP_2026-10-26.md)。源码061d86f已推送GitHub并以同一发布提交Git增量包fetch/pull --ff-only同步至8000。两组各4CPU、训练数据SHA／相同初始化及真实2×2短测通过，实际K、lambda、AdamW、BN8与完整保存状态验收通过。北京时间10/26 10:10:30两组均fresh进入正式300×200并已有真实更新；GPU0/1绑定正确。私有可视化目录b05_cb32_a1_followup_20261026保存dispatch/状态/验收/manifest。不重复提交，不逐轮监看。

## 2026-10-26：用户批准B05 CB32 A1两项完整消融

最新修订：8000 GPU0样本K5／代理K10、lambda0.5；GPU1样本K10／代理K20、lambda0.1。两组均10/25严格B05有放回CB32新HIER A1、LR.005/eps1e-4、seed4780、fresh300×200，其余保持86协议。用户直接启动授权覆盖旧待审矩阵限制，沿用历史TS选模；无新增H0/CB64。见[91配置](91_B05_CB32_A1_K_WEIGHT_ABLATIONS_2026-10-26.md)。第一组联合调整两种K，不能称两种K各自单变量；第二组仅改系数。源afb01ee已推送并在8000以Git增量包fetch/pull --ff-only同步。两组各4CPU及真实2×2短测／实际K、系数、AdamW、BN8与保存状态均通过；北京时间10/26 00:06两组分别在GPU0/1 fresh进入正式300×200，已见真实更新。私有可视化目录b05_cb32_a1_ablations_20261026保存dispatch、状态、验收与manifest；10/26 09:27核实两组均300轮／60000步、BN120000、exit0和完整保存验收通过；原PID已退出，GPU0/1空闲。K5/10 best93.598055、AA91.486043、e217、末轮92.990276；lambda.1 best94.043760、AA91.632557、e280、末轮93.517018。best/last各约162MB存在，完成manifest/状态/末轮metrics私有备份。不重启、不新增训练。

## B05 CB32 A0前十轮85审查已完成 — 2026-10-25

用户仅审旧HIER CB32，后授权同配方fresh10轮逐轮保存；21:14全部结束，14同9840缓存（新e0..10＋原e1/e5＋TS13参考）、33CPU、单卡短测、实际2000步／BN4000／300cosine和85固定80对象回归通过，GPU释放；14raw及47结果文件本机SHA一致。见[90结果](90_B05_CB32_A0_EARLY_AUDIT_2026-10-25/report.md)。原e5代理508/512越cap而类内中位40.10°；新e3为503/512、41.26°，e10为511/512、31.34°。不需先出现整类约2°紧致才贴cap，但未排除训练局部窄邻域。

新e10代理内部三项严格次序为0，有有效分母；只剩一个内层例外。proxy全抽样有效数75.83／top1 41.14%，有效非碰撞仍351.63／top1 9.49%，sample有效数反而增长。首步前向／损失／proxy更新完全相同、主干范数微差后轨迹分叉，新轮次不当作历史权重恢复；net差异含BN。仅此任务、不涉及PartHier，未追加完整训练。

## B05 CB32旧HIER前十轮：已批准补跑与85口径审查 — 2026-10-25

用户最终明确仅10月25日B05有放回CB32旧HIER A0（LR.01/eps1e-8），取消PartHier审查。原任务早期完整权重仅e0/e1/e5；用户选择同配置fresh十轮、每轮保存，再按85方法逐轮分析。见[90计划](90_B05_CB32_A0_EARLY_AUDIT_2026-10-25/plan.md)。保持seed4780、Ks10/Kp20、10×200实际更新但300轮调度、原test/clean-eval/审查及随机流；e0与原初始化／首批输入严格核对，原e1/e5并列，不冒充历史缺失轮恢复。18份原始文件本机／8000备份SHA一致，12项新CPU检查通过；GPU短测／正式复跑／全量缓存尚未完成。冻结几何仍用85共同CB64/K20读出，与本训练真实CB32/K10-K20 usage分列，不启动其他实验。

## B05 eps单变量十轮审查已完成 — 2026-10-25

用户要求检查结果；8000两组各10×200、BN4000、300轮调度及有限状态／副本验收通过，16:17:26结束并释放GPU。见[88完整结果](88_B05_EPS_ONLY_10E_2026-10-25/report.md)，9幅PNG/SVG、6份CSV、summary及独立inspection已保存，81文件本机备份SHA一致。仅eps不同：E8末轮500/512越cap、映射深度中位4.600／raw中位2.618；E4为0/512、深度中位3.439，第4轮末起0cap、轮内短暂越界到第5轮仍有，第6轮起零新增。E4曾cap70个全部曾返回；最终69.73%比自身初始化浅，不能说全体回到初始内侧。

首步梯度相同0.11456，更新1.799→0.446；E4第10轮代理梯度是E8约3.03倍、更新仅15.0%，首batch主干HIER/base9.60%／9.88%，不支持HIER反传消失。eps也改变坐标相对权重／方向，不等价统一降LR。初始化是randn/sqrt256×2.3×.9，初始深度中位4.13975但非全体等深，已4个越cap；累计人数／重复事件／轮末状态要分开。无test／validation／准确率结论，不外推300轮分类收益；不追加实验或自动监看8001。

## B05双卡eps单变量十轮审查已获授权 — 2026-10-25

用户指定8000、此前双卡B05 CB64／K20-K20，并追加代理深度移动分布。两组fresh seed4780、10×200，均LR0.005，仅eps1e-8／1e-4不同；保持300轮调度，零test／validation／选模。逐步记录实际代理梯度、AdamW更新／eps主导度、越cap及返回；逐轮记录512代理当前深度、相对上一轮／初始化位移与raw模长分布。主干HIER/base梯度仅为每轮首batch探针，不称整轮均值。见[88检测计划与启动](88_B05_EPS_ONLY_10E_2026-10-25/plan.md)。源码c8c05bc；本机／服务器各13项CPU、两组真实双卡2×2短测验收均通过，初始化／首批输入／首步梯度哈希相同。15:44:05在8000空闲GPU0/1 fresh启动E8，随后自动fresh E4及生成报告；完整十轮结果待取得。私人后台入口另存，不需逐轮监看，不混8001单卡A0/A1队列或历史LR.01的A0。

## 四组嵌入审查已在8000完成 — 2026-10-25

用户明确恢复85审查并指定8000。主TS13／TS14／10月23日双卡B05-A1／原生A1各best/e300，加TS06／TS01 HIST／TS11／TS12辅助端点，共16份相同9840 clean/eval缓存。完整[85结果报告](85_EMBEDDING_AUDIT_2026-10-24/report.md)、每类／780类对统计与主辅各22幅PNG/SVG齐全；raw备份SHA16/16一致。8000固定80例保留历史batch32后坐标／深度误差0、预测一致；21项CPU检查通过，源权重／BN不变，零训练／test前向，GPU已释放。代码采集8f8f4fc、最终统计图bd3f1e6，私有交接见87。

A1两组代理均0/512贴cap，末轮深度中位3.440／3.221；TS14仍511/512贴cap。冻结非碰撞样本三项次序92.827%／98.760%，但代理内部仅5.396%／6.743%；排除端点自选后17.264%／21.854%，不能称完整代理树。原生A1 flower_pot–plant中心由best52.055°到e3008.522°，跨类中位7.948°，总体异类约92°会掩盖此局部几何退化；尚非test错误因果证明。图表已处理构造壳ECDF抖动及热图色条布局。未检查／改变另行排队的CB32任务；不得混入本报告。

## B05单卡A1／A0新旧HIER队列 — 2026-10-25

用户确认GPU0新A1（代理LR.005/eps1e-4）、GPU1旧A0（.01/1e-8），两条均严格B05固定壳HIER、seed4780、单卡非循环16类×2＝32、样本K10／代理K20、300×200，等待指定卡空闲后短测再fresh训练。用户明确挂好即退出，不等待GPU验收或逐轮监看。详见[86执行说明](86_B05_SINGLECARD_A0_A1_QUEUE_2026-10-25.md)。本机8项CPU合同通过，单rank收集、global32和审查K范围已适配；服务器提交与实际短测／训练状态以后续私有交接为准，不将CPU通过写成CUDA已通过。没有新增H0、续旧权重或修改历史结果。

## 四组whole／代理几何审查：计划就绪，等待连接恢复 — 2026-10-24

用户确认主四组为TS13、TS14、新B05-A1、新原生A1，各best/e300，共8份；允许加入有用历史消融。完整[85检测计划](85_EMBEDDING_AUDIT_2026-10-24/plan.md)已列c-aware深度百分比分布/ECDF、40类类内与780类对分位数/中心热图/固定代表类别对，以及代理稳健祖先次序、使用率/热点、深度与类别广度及监督/语义边界。TS06和原版H0作辅助，现成旧V9配对按身份可靠性纳入；不混历史HIST与R2。

两服务器GPU满，未新增服务器GPU计算。8000 SSH拒绝连接，TS13全量缓存/权重本机也无副本；用户明确先完善计划、待其通知连通后继续。六份TS14/新两组net-only已只读复制到本机私有目录且SHA核对一致；训练输入包尚未验收。桌面“独立测试”实际是连接诊断、无几何缓存，未执行其中脚本/读取凭据。原FPS本机NVRTC适配11项算子验证、统计工具9项合成CPU测试通过；完整模型回归/全量主缓存/实测图均未执行。不得将预备测试、e120或200对象plot smoke充作最终几何审查；不自动重连/提取/训练。

## TS13/TS14正收益与新HIER差异解释 — 2026-10-24

新增[84对比报告](84_TS14_VS_NEW_HIER_2026-10-24/report.md)。TS13→TS14同循环64/90000更新的配对：best和e300均+0.486pp，末10均值91.831±1.164→92.978±0.229；不能拿多步数解释这对内部差值。TS14与新B05几何／fixedshell／pre-base-GS分叉／continuous基础流相同，主要共同改变采样、90000→60000更新节奏和A0→A1。新CPU核实TS14 best e212与e300均511/512 raw代理越cap，末态模长中位6.198，故旧分类正收益不依赖整体代理回内；方向／关系正则为候选解释，非径向语义树证据。

旧TS14与新B05末10均值92.978/92.986、SD0.229/0.204近乎相同，新组best低0.405pp但不作全面退化判断。纯CPU真实类数／合成ID采样模拟：首轮CY300覆盖8959、CB200覆盖6218；同200步CY覆盖7748，说明覆盖差不全来自步数，不能据此认定循环分类必优。没有新模型前向或GPU/训练，未恢复用户弃用的循环协议。

## seed4780原生／B05＋新固定壳HIER均已完成300轮 — 2026-10-24

只读完成审查见[83：结果与分析](83_S4780_NATIVE_B05_COMPLETION_2026-10-24/report.md)。79两任务于02:49:14／02:47:34正常结束、各60000更新、300连续轮指标、完整test2468、有限模型/代理与副本同步通过；实际AdamW eps1e-4、最终LR0.0005、300轮scheduler与BN120000核实。原生best e153 OA92.747/AA90.462、e30089.506，末10轮90.312±1.257；B05 best e273 OA93.476/AA91.508、e30093.112，末10轮92.986±0.204。best/e300均代理0/512越cap；末轮深度中位3.221/3.440、最大3.827/4.118。完整精度与口径存83/results.json。

数值回内维持到完整训练末期，但原生分类仍退化；B05整体兼容性更好，不作HIER净增益或纯曲率因果结论。e300单批加权HIER/base梯度比约0.907%/13.009%，不能认为相同λ给主干同等力度。祖先深度次序有局部倾向，但代理内部pair祖先严格浅于端点并不普遍；不可仅凭壳层分离称语义树。未新前向/BN干预、未增加H0或自动监控，也未触碰另行运行的PartHier v1。

PartHier v1研究设计阅读入口：[82：实验设计说明](82_PARTHIER_V1_EXPERIMENT_DESIGN_2026-10-24.md)，包含混合祖先池、实际损失与训练协议、短测依据和归因边界；执行实证仍以80为准。该文档整理未新增训练或消融。

## PartHier v1已改8001启动，部分重算提高利用率 — 2026-10-24

用户明确“先用8001”，随后要求提高显存利用率；覆盖下方8000连接阻断的当前部署地点。最终源82478cd，33CPU/Gloo、真实CUDA一致性、独立2×2与14步满Q/whole1024通过。只重算part200/400欧氏主干，600/whole保留激活、双曲头普通autograd；原global64/FP32/BN4、三intra各.01、原生c1/200-400-600/混合A-B/Q2048、seed4780/300×200/TS保持。满Q活跃峰值27.79GiB/卡、预留28.21GiB、设备空闲至少2.27GiB，短测1.343s/step对全重算1.527s。

2026-10-24 09:45:48在8001 GPU0/1独立fresh正式启动；09:50首轮200更新/2468 test、两rank各38层BN800、有限模型/实际RSGD/保存状态、300轮调度、Q2048/队列SHA、参数同步误差0及初始化/数据身份全部通过，已进入e2后台继续，不再追加测试或逐轮监看。不是probe续训。见[80](80_PARTHIER_V1_PLAN_AND_LAUNCH_2026-10-24.md)。8000旧offload短测状态仍未知，不擅自kill/restart；没有其他正式PartHier或自动监控。私有运行指针另见忽略的81交接。

## PartHier v1实现与8000连接阻断 — 2026-10-24

用户授权自主实现、短测、每次短测后查询五小时额度，剩余低于30%即停止追加短测、优先完整挂8000。已确认原生HyCoRe c1/无新增cap、同中心200/400/600、混合祖先选择、完整Q2048、三级intra各0.01求和；统一seed4780/非循环CB64/300×200/TS。新增独立parthier_v1，源e6a7168、ca1402c已推送并在8000冻结，18项CPU和双rank Gloo通过。

真实双卡普通四pass在whole前向OOM，零更新；随后CPU激活暂存短测提交后8000 SSH不再返回登录响应，该次完成/退出状态未知，未确认故障原因，也未确认进程已停止。8001可连但四卡均忙，未使用或打断。后备为保持BN/RNG的非重入checkpoint，本机wrapper测试通过；真实PointNet2 CUDA、双卡四pass、满Q2048容量与角色/尺度轨迹仍待验收，**完整训练尚未启动**。不得把已提交短测或CPU证明写成正式上线。详见[80：方案与实证状态](80_PARTHIER_V1_PLAN_AND_LAUNCH_2026-10-24.md)，私有恢复指针另在本地忽略的80交接。

## seed4780原生/B05双HIER执行 — 2026-10-23

用户明确选择严格保留B05既有基础GS，要求两条均在8001，随后授权“启动”，覆盖下方仅设计/不连接旧状态。仅两条fresh HIER-on：原生c1/无新增cap或出口GS/1000_N与4，以及B05 c.1/cap2.3/基础GS/500_N与2；共同seed4780、非循环有放回CB64、300×200、TS、固定壳HIER与AdamW LR0.005/eps1e-4。独立短测后重新初始化生产，不续旧断电任务。见[79：执行与验收](79_S4780_NATIVE_B05_LAUNCH_2026-10-23.md)。两条整体方案比较不等于HIER开关消融或纯曲率消融。

执行完成：源5517661，8001两组各18项完整CPU合同、相应双rank Gloo及train-only2×2短测通过，真实AdamW/调度器及BN8有限状态验收。18:17:36两组fresh生产：原生GPU0/1，严格B05 GPU2/3，后台继续300×200。首轮200步/2468 test记录已可读。用户明确“短测通过就结束吧”，故不再补专项首轮采集、逐轮监看或设置自动监控；本次仅保存启动交接，等待用户下次询问。不要重复提交或因文档更新重启任务。

## 导师汇报：TS全实验、idea时间线与消融汇总 — 2026-10-23

按用户要求新增[78：导师汇报Markdown](78_SUPERVISOR_TS_EXPERIMENT_SUMMARY_2026-10-23.md)：覆盖10/17后14条正式任务（13条完成300轮、1条无完整结果），另列TS01原版历史锚点及早期V6/旧V8参照；全称、曲率、batch/预算、margin、cap/GS、HIER视图、代理AdamW、seed、best OA同轮AA、e300及末10轮统计齐全，并按idea→实验→结果→消融梳理。TS15仅使用10/22 23:06:55已存e34定量快照，e47–48只有文字进度，不推测成绩或宕机后状态。三路独立复核配置/数值、机制边界与时间线；仅本地资料整理，不连接服务器、重测或启动任务。

## 三套dataflow与梯度裁剪/GS澄清 — 2026-10-23

用户要求核实原版HyCoRe的梯度操作并画原版HyCoRe、原版HIER和B05三图。见[77：源码与dataflow](77_HYCORE_HIER_B05_DATAFLOW_2026-10-23.md)：原版HyCoRe实际PointMLP没有B05出口GS，但backward后有全网络参数梯度总范数1裁剪，随后RSGD。B05保留这套更新流程，另在共享whole/part emb出口加入cap2.3和GS；“基础GS”指基础分支位置，不代表原版自带。官方HIER样本和层次代理分别cap/map/GS，另在更新前逐分量裁剪±10后AdamW；PA类别代理与512层次代理独立。三者均不强制whole固定4.6，历史B05也无HIER目标。仅本地源码核实与文档，未连接服务器或改训练/设计选择。

## 未来统一seed4780、弃用循环64；仅讨论设计 — 2026-10-23

用户明确今后所有谱系实验统一seed4780，放弃类别循环64，采用有放回类别均衡64。历史seed/运行身份不改。用户报告服务器断电，当前只做设计；不得连接服务器、排队/启动/自动恢复。旧TS15的running及e47–48等为断电前快照，断电后状态/文件完整性未知，不当作仍在运行或已完成。

新[76：两条底座的固定壳HIER设计](76_S4780_FIXED_SHELL_BASELINE_DESIGN_2026-10-23.md)评估8001原生c1＋Hshell-A1及8000 c0.1适配＋Hshell-A1。关键澄清：B05的基础GS本身就是HIER移植到共享出口的InverseMetric hook，原版HyCoRe实际forward无此hook；RSGD/流形运算则保留。若8000保留完整B05应保留已有基础GS；若主pipe绝不含HIER来源GS，则是B04式GS0配方，不能继续称OA94.3的B05。此分歧待用户明确。

两条HIER-on可检验组合行为；要量化HIER增量，建议各自补同CB64/seed4780/预算的H0，形成四臂。旧94.0/94.3为SH32历史成绩，不能直接当新CB64控制；两底座还变cap/GS/margin，跨组不单独隔离曲率。76仅为设计建议，未授权新增执行。

## TS实验统一命名与baseline总表 — 2026-10-22

用户要求梳理V8、HyCoRe改版与V9，仅保留全9840训练/2468 test逐轮选模，seed只在名称后注明。新[75：统一命名与baseline图谱](75_TS_EXPERIMENT_BASELINE_ATLAS_2026-10-22/report.md)采用“基础配方B01–B10 / HIER组件H0、Hshared-A0、Hshell-A0/A1”的组合名，15条正式记录TS01–TS15；原目录/权重/日志/canonical身份不改。独立[TS注册表v3](experiment_registry_ts_v3.json)及CSV保存新旧映射、batch/更新预算、基础cap/GS、HIER视图/GS、margin、同轮训练与test准确率。VS984不纳入，不重审。

北京时间23:06:55只读采集：14条已有300轮终点（含历史原版续训），新AdamW TS15完成34/300、best e28 OA91.653160%，只是快照不排名。补齐末10轮均值/总体SD，训练在线与clean/eval分列；原版历史e300日志OA92.828%，best沿用54固定权重复测94.043760%。最高best仍是HC/c0.1/cap/GS/半margin/shuffle32的94.327%，无GS对应组best=e300=93.963%、末10轮均值93.6265%；循环HIER93.881686%，匹配无HIER93.395462%。75用于用户选择新HIER底座，不启动矩阵；现有A1完整任务保持后台运行，不逐轮监看。

## V9-HIER AdamW候选完整300轮已启动 — 2026-10-22

用户请求把73的AdamW参数进行一次完整V9-HIER实验。唯一fresh300×200、seed22、TS，实际只改代理初始LR0.005和eps1e-4，最低LR0.0005及其余原V9/300轮余弦保持；不resume十轮权重，不新增矩阵。版本化wrapper位于diagnostics/v9_adamw_full_20261022，保留原生产审计/评估/检查点，不加载73密集observer。源码8408273先本地提交推送、服务器fetch/pull --ff-only冻结；38项CPU+Gloo、独立fresh2×2 train-only短测通过。8001空闲GPU1/2于21:28:18正式启动；21:31:26首轮200步、2468 test、有限best/last/e1、BN400/双rank同步与实际optimizer/scheduler等16项验收通过。旧模型/代理初始化、数据字节/标签/划分、首批ID/crop一致。保持后台完整300轮，不逐轮监看；后续按用户询问读取保存数据。见[74：方案、上线与验收](74_V9_ADAMW_FULL_RUN_2026-10-22/report.md)，私有路径见忽略的74工程交接。

用户同时询问昨天的实验：[10月21日汇总](74_V9_ADAMW_FULL_RUN_2026-10-22/yesterday_summary.md)确认当天新启动4条完整训练，现均已完成。最新原版c0.1循环HIER于10月22日10:55:59完成300轮，best e212 OA93.881686%，e300 OA93.273906%；对应循环无HIER为93.395462%/92.787682%。与旧c1 V9分开命名，不沿用70的140轮旧快照。

## V9代理LR/eps十轮完成并验收 — 2026-10-22

[73：联合候选结果与CE澄清](73_V9_LR_EPS_10E_DIAGNOSTIC_2026-10-22/report.md)：代理初始AdamW LR0.005、eps1e-4，其余V9/71配置与300轮调度保持；唯一10×200步任务18:03:59自然完成并释放GPU。用户要求停止逐轮监看后，按其再次“继续”仅读取保存结果做CPU分析，无新增训练。e1…e10越cap62/65/28/9/3/2/1/1/1/0，对照71末轮502。122个曾越界代理全部返回，2737次越界/返回是重复事件；末态深度中位3.576、范围3.147–4.572。

新e10历史m和当前径向分量仍可向外，向内WD已能压过，净模长变化−6.40e-5；实际切向位移仅对照约4.84%，关系损失持续下降但同轮较高。sample与whole径向间隔恢复，代理内部顺序与覆盖表现混合；不称层次/分类改善，不外推300轮或分离LR/eps。41项启动CPU检查、11项完整验收和7项补充核验通过，80快照SHA与同图梯度和核验通过。执行源码03e6fa7、CPU工具6b67040；初始模型/代理、数据与首批身份匹配71，跨硬件不作逐位重放承诺。私有路径见忽略的73交接，旧结果与用户修改保留；没有启动更长训练或margin修改。

## Cars/V9角度与实际更新尺度审查完成 — 2026-10-22

[72：答疑与审查](72_HIER_CARS_V9_ANGLE_AND_STEP_AUDIT_2026-10-22/report.md)按用户建议只读Cars/V9已有特征和71真实快照，零GPU/编码器前向/生产更新。Cars与V9 e300类内夹角中位42.503°/1.993°，同壳pair中点3.815/4.598；异类夹角70.652°/91.908°，同类pair加异类k的绝对三点中心2.807/2.308。V9早期实际前向尚未紧聚集，实际挖掘pair多跨类，不能用末态类簇解释首次越界或全部代理贴壳。首步CPU重放原19越cap，lambda.5→.1仍18，LR减半6，eps1e-4为0；这是一步对照，不是新训练结果。Cars虽启用AMP，代理/m/v实际FP32，eps在保存状态中主导分母。14项独立检查与4组图完成，原始任务和用户修改保持；没有新实验矩阵。

## V9代理贴cap的10轮机制诊断完成 — 2026-10-22

[71：图文报告](71_V9_PROXY_CAP_CAUSE_2026-10-22/report.md)记录用户授权的8000从头10×200步诊断，保留300轮调度、无test/选模。e1/e5/e10越cap为229/444/502，502个首次越界均未返回；不替换历史69的196/378等记录。7步×两分项同GPU恢复Gumbel状态，距离/选择/hinge/full-ST梯度逐元素重放一致；早期选择路径可使原本向内的直接距离合力转向外侧。越界后87.63%代理—步仍增模，历史m径向贡献约+.001654、当前径向贡献近零，WD约−.000389；固定状态仅去旧m径向或改eps可使多处平均位移转内。参数和Adam状态只有前向cap，没有内层约束；原SOP也497/512越cap。40项检查、3组图、2000步核算通过；GPU释放，旧任务不改，没有新完整矩阵。私有路径见忽略的71交接。

新对话接续：完成AGENTS规定的必读顺序后，阅读[70：当前交接](70_NEW_CHAT_HANDOFF_2026-10-21.md)，其中区分旧V9与最新循环配对，并保留23:27运行快照和继续讨论的入口。

## V9 whole与代理深度审查完成 — 2026-10-21

[69：图文报告](69_V9_EMBEDDING_REVIEW_2026-10-21/report.md)比较历史原版、原版c0.1/cap/半margin/GS、已完成V9-B64/HIER64及V8-HIER64 TS的best/终点，另附循环配对同e120，共14份固定权重、同9840训练对象clean/eval、11组PNG/SVG。V9代理e300仍512/512贴cap，参数模长中位6.885；V8为510/512、6.915。V9早期e5贴cap378少于V8的510，但e99已全部贴cap，未形成径向层次。V9原始NN4覆盖725、最热四对象槽占6.49%，优于V8的216/32.67%；V8等深度覆盖1025仍高于V9，方向结构不能据此称普遍改善。V9全部512代理在sample和proxy分项都有整轮选择及正hinge非碰撞覆盖，不沿用V7休眠解释。原生whole c1均值/中位/标准差e300为3.888/4.254/1.093，独立HIER视图固定4.6需分域。六项统计测试、同输入/源SHA/模型BN保持及独立数图审查通过；零训练更新、零test前向，审查GPU已释放，循环HIER主训练未改。工具位于diagnostics/v9_embedding_review_20261021，私有路径见忽略的69交接。

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

## c0.1/cap/nohook三组补充训练已启动 — 2026-10-20

用户明确确认[60：三组配置与验收](60_C01_CAP_NOHOOK_ABLATIONS_2026-10-20.md)。新服务器GPU0运行V8-B0-32-C01-NOHOOK-FULL9840-TESTSEL（seed22，cap2.3、500/N、2，仅去InverseMetric）；GPU1/2运行原版c0.1与其仅加cap2.3的配对组（seed4780，1000/N、4，均无hook）。三组shuffle32、全9840/2468 test选模、300×307。最终28项CPU合同及三组训练限定2×2 GPU短验收通过，原版配对初始化/数据完全一致，所有首轮test及best/last/检查点已保存；01:46快照V8完成6轮、原版两组各完成2轮。源码分别c78a168/eebaabf，后台持续运行；实际两根目录和进程见私有60交接。首次导入失败与CPU缺依赖均留存，不重复正式任务；旧结果保持。cap+1000/N在N200–217存在理论不可达间隔，已记录，未私改参数。

## N600顺序的同深补充分解 — 2026-10-20

[补充复核](59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/clarifications/N600_BN_ORDER_2026-10-19/report.md)保持用户主报告不变，CPU读取既有配对缓存。以1e−4区分数值同深，同1280对象N600当前批BN下：原版part浅/同深/反转为80.08/0/19.92%；旧V8为19.77/72.42/7.81%；FULL为18.44/73.05/8.52%。V8同深集合恰等于两者同时贴cap集合，原约51%直接g>0率混入数值微差，不代表更多真实倒置。原版eval→当前批BN的顺序改善64.45→80.08%，完整margin却2.19→1.80%，需分开解释。margin改变目标和hinge激活范围，不直接缩小已违例样本对gap的梯度；N600残余hinge反高于N200，不能据此认定小margin是训练根因。无新增前向、GPU或训练。

## whole/part逐对象margin审查完成 — 2026-10-19

[59：配对图文报告](59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/report.md)覆盖原HyCoRe及两版V8-HIER64的best/终点，同8856对象及N200/400/600完全相同part，6模型全量/1280 BN诊断/同面板native共18观察、54逐N统计、16组图。完整Rhier终点原版.01/3.12/1.94%，旧V8 1.78/10.16/4.55%，FULL 2.01/9.49/4.42%；仅whole更深的比例远高，不能混同margin达标。20异类负part控制显示Rcontr与径向约束不同，不冒称历史训练flip。V8同1280对象切当前批BN后N400达标约95–97%，N600仍约.55–1.17%，part贴cap约75–76%。完整历史9840 part索引SHA与本次canonical8856逐元素强验证通过；旧V8全量whole逐元素重放误差0。源SHA、参数/BN/RNG恢复通过，GPU释放，无test前向、新训练或正式BN改动。

## FULL9840两组对照已完成 — 2026-10-19实时核对

8001两条原排队任务均自然完成300轮：B0-32于10月19日01:54结束，best e203 test OA93.7196%/AA90.1570%；B64于22:49:49结束，best e276 OA93.8817%/AA91.9204%。不再等待GPU或运行；既有每轮test选模协议保留。本次仅读取保存记录，未重测或停止任务。原排队快照见56，不能当作当前状态。

## 四模型whole密度与BN兼容性审查完成 — 2026-10-19

见[58：Markdown图文报告](58_V8_WHOLE_DENSITY_AND_BN_REVIEW_2026-10-19/report.md)。按用户确认使用已完成8856/984协议，四模型best/last共8份完整缓存按同8856对象及实际输入字节对齐；e300的B0/B64/HIER whole均深度3.8851/3.5407/3.7628，HIER较同batch无HIER更深且更窄，原版last也非等深壳。原HIER类内/跨类方向余弦显著分离，但同壳不保证浅代理，SOP仍497越cap。新增三模型×四BN条件冻结前向，320测量/640校准训练对象无交集；原HyCoRe最终运行统计直接移植后whole均深度0.88–1.32，97.5%面板预测改变，不能当通用BN。自身whole-only校准使whole贴cap90.94%/95.63%/98.75%，但part也变深，HIER part贴cap50.94%、whole-part均深度差.508→.331。三模型native与既有缓存逐元素一致，参数/BN/RNG恢复、源SHA保持、零backward/optimizer/test前向；11组图与匿名表、独立脚本已归档。GPU已释放，未改正式BN或启动完整训练。

## V8代理cap与whole球壳机制诊断完成 — 2026-10-18

见[57：图文报告](57_V8_CAP_AND_WHOLE_CAUSES_2026-10-18/report.md)，8组图与独立诊断目录。原V8代理e1已有196个越cap、e5为507；历史m径向分量的一步配对干预可使早期已越cap组平均增模转负，不能只看当前梯度。固定特征1024步eps对照越cap387→6，两组损失均下降但eps1e−4较慢；原版三份实际FP16/eps1e−4，官方FP32分支本就1e−8，且SOP仍497越cap，不称普遍代码bug。whole早期形成壳，晚期clean/eval内缩与BN条件有直接关系：同128对象仅换BN模式3.605→4.584；640独立训练对象重估临时统计，匹配whole构造与40次前向，whole→whole库贴cap99.22%，part→whole库1.56%。固定参数与源BN/RNG恢复，支持共用统计机制，未测分类修复/part收益。原版36张head探针raw约22–54；只搬曲率/cap/hook不等于搬head尺度。9项Torch测试通过，空闲单卡诊断完成、GPU释放，无test前向，不改主训练、不启动V9。

## V8完整9840协议两组对照已排队 — 2026-10-18

用户授权旧服务器单卡V8-B0-32-C01 shuffle和双卡V8-B64无HIER，保持全9840训练/2468 test每轮选模。300轮，B0为307步/轮，B64为200步/轮；V8 cap/hook/margin与seed22不变。00:04独立机会调度后台存活，14项CPU合同通过，五份训练HDF5与主实验SHA一致；四卡均被其他用户占用，两组均待空闲，未GPU短验收或生产更新。有效代码a25cc36已推送/快进部署；初次浮点精度断言失败记录保留，重试仅改测试精度。见[56：对照及后台排队](56_V8_FULLTRAIN_CONTROLS_2026-10-18.md)。满足一张/两张空闲卡条件后自动验收并启动，不需重复提交。

## V8完整9840训练池、test选模已启动 — 2026-10-17

用户明确授权一条V8-HIER64改用原版官方9840训练/2468测试、无独立验证集、每轮test选best，随后指定新服务器执行。展示名V8-HIER64-FULL9840-TESTSEL；双rank各16类×2、global64，300×200步与V8几何/损失/seed22保持。23:22启动，首轮完整2468 test与best/last通过，14项CPU单元合同及global64双rank梯度、独立2×2 GPU短验收通过；23:28已e3step65。新入口生产提交ff98bf9，已推送并快进部署；没有旧服务器重复任务。见[55：方案及启动验收](55_V8_FULLTRAIN_TEST_SELECTION_2026-10-17.md)。该结果将按测试集选模单独报告，不混作984独立验证协议。

## V7/V8结果与原始HyCoRe协议复核 — 2026-10-17

[54：结果及协议](54_V7_V8_RESULTS_PROTOCOL_REVIEW_2026-10-17.md)已完成五份固定best权重val/test复测，OA/AA与历史记录一致，零更新且源/BN保持。原始seed4780实际epoch元数据229（0起），test OA94.0438%/AA91.5448%，没有独立val；现代984子集的98.9837%只能视为训练内诊断。e221实际属于V8主模型。V8-B0-32-C01已完成300轮，best249 val OA94.5122%/AA91.8165%，test OA92.6256%/AA89.8128%。V7与三组V8共享8856/984及官方2468 test；shuffle B0保留采样与输入语义，但不是原版9840训练/每轮test选模的完整复现。V7无独立完整B0/B64，历史参考为V6-B64。新增8组图及6张汇总表，串起V7问题→V8几何方案→代理激活/热点改善及剩余问题。复测代码cb83192，材料另行提交；没有新训练。

## V8完整主/对照系统审查 — 2026-10-16

[53：V8审查](53_V8_SYSTEM_AUDIT_2026-10-16.md)已完成：V8-HIER64与同协议V8-B64均e300，主模型test OA92.7877%，B64为92.9092%（差3对象、单seed）。新增24个8856对象快照，与V7及原版CUB/Cars/SOP比较；18组科学图、14张汇总CSV、8个冻结机制快照、12个独立形态快照、3个真实编码器梯度探针。V8 e300 sample非碰撞/正hinge祖先均512；NN4覆盖27(V7)→231(V8)，等whole深度后1051；510代理forward cap，clean whole中位数3.741，最浅1%占72.17%近邻槽。休眠改善、热点仍在，形态仅局部正面证据，不能宣称普遍收益。154项测试、只读/输入身份/图文链接通过，无新test或optimizer更新。单卡V8-B0-32-C01采集时仍e283、best249，未填最终结果。源适配ad53d36/7ebc081，报告另在专用分支提交。下方为历史启动快照。

## V8新服务器两组对照已启动 — 2026-10-15

用户确认单卡也保留V8的cap/hook/margin。V8-B64（两卡global64、关闭HIER）与V8-B0-32-C01（单卡shuffle32、关闭HIER）11:41启动，首轮完整984验证及checkpoint通过，进入e2。三组初始化/划分一致；旧主V8继续，不迁移。代码85fc812，12项CPU合同、双进程梯度与两组GPU smoke通过。参见[52](52_V8_CONTROLS_ACCOUNT_AND_LAUNCH_2026-10-15.md)。下方排队记录为更早快照。

## V8已部署，后台等待两张空闲GPU — 2026-10-15

用户批准统一c=.1、最终共享mu的cap2.3/hook、gamma500/delta2、HIER权重.5、W0；两rank各16类×2例，保留双优化器300轮cosine与验证选模。独立V8入口已实现，9项CPU合同与global64 Gloo/参数审计通过，真实数据9840→8856/984及split身份核对。有效任务`v8_hier64_20261015_022043`固定代码14f8a72，已独立后台运行；02:23:30 Asia/Shanghai仍等待两张完全空闲卡，尚无GPU短验收或主训练更新。空闲后自动先2×2 smoke再全新初始化e300。见[50：实现与后台交接](50_V8_IMPLEMENTATION_AND_LAUNCH_2026-10-15.md)及私有`.codex-local/handoff/50_V8_ENGINEERING_HANDOFF_2026-10-15.md`。本条为最新状态；下方V7及c1方法描述保留为历史。

实验展示名统一遵循[33：实验命名规范](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md)。旧文件名、运行ID及代码字段保留用于溯源；本次只规范称呼。

## 原版 HIER 三数据集热点机制与激活来源 — 2026-10-08

[46](46_ORIGINAL_HIER_SOP_MECHANISM_AND_ACTIVATION_2026-10-08.md) 补充完整高维方向与官方分项激活：三组whole都在d0≈4.6薄壳，SOP代理R=.6482、497/512超过forward cap；CUB/Cars代理R约.044/.047，越cap数9/1。SOP整体旋转覆盖432→880–1065，方向减均值→1778但同细类纯度下降，因此覆盖改善不等于结构收益。缓存一轮长度窗口和真实图像32/44/330批全部完成，三组sample有效祖先及前2batch两个分项full-ST梯度均512个；图像SOP径向梯度占比2.59e−6/1.82e−6，仍有角梯度。36张clean anchor前向与原缓存最低余弦>.999999，网络/BN/lcas与源SHA保持，零参数更新。公开数据三集已下载验收；权重和缓存副本经用户明确授权放项目服务器私有新目录。三个best的LR/WD、轮次和每轮步数不同，不能把SOP几何差异完全归因于数据域。20实现测试通过；完整ID/数据/权重/日志均不进Git。

## Original HIER best complete geometry — 2026-10-08

The user supplied3 original best checkpoints and12 complete CUB/Cars/SOP caches,
and confirmed best-only scope. [45](45_ORIGINAL_HIER_BEST_GEOMETRY_AUDIT_2026-10-08.md)
verifies hashes, sample identities, official proxy maps and full official FP32 NN4.
All original whole are on the cap shell d0≈4.6 (std3–6e−7), including augmentations;
official train coverage1647/1664/432. Original SOP has angular hotspots. V7e300
whole-only equal-depth coverage27→221; frozen random proxy directions yield
1210–1295, while raw random coverage is only5. Both radial bias and current angular
redundancy matter; coverage alone is not learned hierarchy evidence. Original
sample pair/triple role sets mostly overlap, not two fixed proxy layers. Complete
whole caches are now available, superseding older missing-cache statements below.
CPU-only frozen audit; no training, image forward, GPU or dataset download. Full
IDs remain private; reusable scripts and anonymous aggregate evidence are recorded.

## Frozen direction and hotspot evolution — 2026-10-08

[44](44_V7_DIRECTION_EVOLUTION_AND_HOTSPOTS_2026-10-08.md) joins11 V7 snapshots
and6 historical baseline caches, canonical8856 shared object identities verified.
Whole directional LOO macro accuracy is80.50% at e5,92.98% at e20,98.96% at e300;
whole common direction disperses while proxy common direction concentrates.
Final raw flower_pot slots1805/2048 versus direction1/2048; the hottest four are
the shallowest four only at e300, still deep with zero raw slots at e200 and deep
in ORIG last. Equal-depth direction retrieval retains laptop concentration and
inactive proxy direction redundancy. All-class affinity is robust to removing
hotspots, but fixed proxy class preference is not temporally stable. Raw reverse
retrieval does not establish a causal training feedback loop. No new training,
encoder/optimizer/test forward or GPU use. Seven mathematical tests passed;
complete IDs stay private, anonymous aggregate evidence accompanies the scripts.

## Radial selection diagnosis — 2026-10-08

[43](43_V7_WHOLE_ONLY_EQUAL_DEPTH_HOTSPOTS_2026-10-08.md) adds full512proxy
raw versus WHOLE-only equal-depth retrieval on the same8856 e300 inputs.
Coverage27→221, maximum repeatedquartet287→45. Raw mainquartet contains the
four shallowestwhole objects;two proxies atdepth2.160 and6.000,98.5degrees
apart,share it. Equalwholedepths2/4.808/6 all exactly match direction ranks.
This is retrieval evidence, distinct from training ancestor activation.

[42](42_HIER_RADIAL_SELECTION_DIAGNOSIS_2026-10-08.md) interprets V7 e5 against
actual per-role selections and same-ID lifetime records. Of247 sample-inactive
proxies,62 return at e148–185 and185 do not return;31 returners were already at
the parameter depthcap6 in e5. A frozen64-object/512-query CPU radial control
raises this group's noncollision pair/triple probability from almost0 to
7.5%/17.0% without changing directions. Most sample gradients are tiny.
This supports radial competition and weak feedback, not a hard depth exclusion
or a unique cause of early outward movement. Original proxy/argmin statistics
are recoverable, but original whole caches are absent from the local review.
No new training, optimizer update, encoder/test forward or GPU use.

## Post-training audit method — 2026-10-07

The user approved [38](38_HIER_POSTRUN_AUDIT_PLAN_2026-10-07.md) and authorized
implementation/debugging and a real V5/V6/V7 audit with matched B64.
See [40: system audit results](40_HIER_SYSTEM_AUDIT_RESULTS_2026-10-07.md),
[39: execution](39_HIER_SYSTEM_AUDIT_EXECUTION_2026-10-07.md), and
`tools/hier_postrun_audit/`. The audit joins31 frozen checkpoints, identical8856
training inputs,11 controlled mechanism panels and15 independent shape panels.
V6 fixes invalid mining; V7 eliminates proxy numerical saturation and broadens
sample ancestor usage. Raw top4 coverage remains concentrated, and independent
shape gains over B64 are mixed. Training/eval geometry, retrieval/actual ancestors,
cached partials/shared encoder gradients/Adam state have separate meanings.
No source results were changed, no optimizer update or new test forward was made.
Final151 server tests passed without skips. Three real e200 shared-encoder probes
passed exact feature replay on the same64 objects/crops. V7's weighted HIER/base
gradient norm ratio is0.242 versus V6's1.738 under this fixed eval condition;
this does not establish actual Adam displacement or isolated training causality.
Acceptance is recorded in
[v2 acceptance](diagnostics/results/20261007_hier_system_audit_v2_acceptance.json).
The following completion/status entries are historical; geometry findings in40
supersede their pending top4/shape statements.

## V7 completed — 2026-10-07 16:00 check

V7-HIER64-K20-W0 completed300 at14:43:33 Shanghai,60000 updates; process exited,
GPU1/2 released. Validation selected187 (OA93.8008%); its one official test is
OA92.2204%, AA89.7204%,2468 objects. Every epoch has200 updates and984 validation
objects, finite scalar summaries, lambda0.1 and proxy replica difference0.
Proxy numerical projection count remains0; post-step parameter cap holds within
FP32 tolerance. Actual best/last hashes match saved identities. e300 validation
OA91.0569% shows late decline. Classification changes are a single-seed multi-factor
historical comparison; whole depth still near its boundary, top4/shape checks pending.
See[37](37_V7_COMPLETED_2026-10-07.md). No new experiment launched; below are historical snapshots.

## V7 launched under delegated decision authority — 2026-10-07 02:52

The latest user explicitly delegated final V7 decisions and requested launch then exit.
The unique V7-HIER64-K20-W0 e300 run detached on GPU1/2 at02:48:51 Shanghai,
code165981f, run v7_hier64_20261007_0250. Five CPU tests and separate four-update
dual-GPU smoke passed. W0/lambda0.1, calibrated initial depth2.240048 from16
zero-update batches (whole median2.488942), post-AdamW proxy parameter cap depth6,
original balanced64/base protocol and validation selection. At02:52:07 e1step59
is finite with replica difference0, official test unread; no completed epoch yet.
The bounded128-update CPU comparison completed. Background parent is PID1;
interactive work exits while main continues. Final decision and handoff are
[36](36_V7_PROXY_SAFETY_DECISION_AND_LAUNCH_2026-10-07.md); this supersedes pending-review statements below.

## V7 single-run design requested — 2026-10-07

The user chose V7: HIER from the first update with lower weight, and proxy
initialization inside the whole median; budget is one complete e300 run,
with other cards reserved for bounded diagnostics. The concrete proposal is
[34](34_V7_SINGLE_RUN_PLAN_2026-10-07.md), pending review through
[04](04_NEXT_EXPERIMENT_PLAN.md): dual-GPU global64, W0, lambda0.1,
one-time exact proxy depth min(5,0.9*m0), retaining the V6 base protocol.
Depth5 is not guaranteed inside an untrained whole distribution. Initial
geometry calibration has zero optimizer updates and restores production
BN/RNG. AdamW proxy movement need not scale proportionally with lambda.
No new production entry point or training has been started. Historical
first-epoch lambda0.5 recommendations below do not override this new design.

[35](35_HIER_MARGIN_BOUNDARY_AND_PROXY_GRADIENT_AUDIT_2026-10-07.md) adds a CPU audit:
no missing margin unit/factor, but released numerical distances differ
substantially from the current exact c1 formula near the boundary. A same-ID
join of existing gradients finds negligible late sample gradients for297
boundary proxies and almost purely angular proxy parameter gradients;
many tangent parameters already exceed the projection saturation threshold.
These results do not isolate the cause of initial outward movement. V7
preflight should check fixed-relation numerics and actual AdamW displacement.

## V6-B64 completed — 2026-10-06 18:40

[32: V6-B64 completed result](32_V6_BALANCED_B0_COMPLETED_2026-10-06.md) supersedes
earlier running snapshots. V6-B64 finished300 at12:13:15 Asia/Shanghai and evaluated
the validation-selected e231 model once on2468 official test objects.
Best validation OA94.3089%, test OA91.8558%, AA89.9994%; historical V6-HIER64-K20-W20 has
the same best validation OA at99 and test OA91.2480%, AA88.9169%.
V6-B64 is+0.6078 test OA points (15 more correct objects), a single-seed historical
comparison without a bitwise replay claim. All280 continuation epochs have
200 updates,984 validation objects, finite telemetry and HIER0; actual best
and final archive hashes match saved identities. Process exited, GPUs released.
e300 validation OA91.2602% also shows late decline without HIER. Same-epoch
geometry and independent shape comparisons remain pending; no new run launched.

## Latest mechanism findings and V6-B64 status — 2026-10-06 08:41

[31: mechanism findings and V6-B64 status](31_HIER_MECHANISM_FINDINGS_AND_B0_STATUS_2026-10-06.md)
records the completed natural16, shape4x64, matched e40 and conditional Gumbel8
one-step evidence. HIER reduces outward movement on the four fixed e300 hotspots
under all eight noise repetitions, while their total update remains outward;
the natural population direction is not uniform. Matched e40 V6-HIER64-K20-W20 is deeper than
V6-B64 on the four inspected classes, without established classification benefit.
V6-B64 has completed198 epochs and is training199; best validation OA94.1057%
at191, no official test read. The bounded diagnostic queue completed10/10.
The proxy100-step strict endpoint gate failed after600 actual updates; its
recovery and independent matched V6-B64 morphology comparison remain pending.

## Authorized V6-B64 continuation and sustained diagnostics — 2026-10-06

The user approved V6-B64 and continuing small mechanism
experiments until usage limits, with the main training taking priority.
Future newly initialized HIER runs must enable weight .5 from epoch1 without
20-epoch warmup. Historical prefixes remain accurately labelled.

[29: launch and diagnostic record](29_V6_BALANCED_B0_AND_MECHANISM_LAUNCH_2026-10-06.md)
records a strict complete-state historical V6-HIER64-K20-W20 e20 fork for V6-B64. Code f649fcc,
branch codex/v6-balanced-b0-mechanisms. Local/server six contract checks and
two-update dual-GPU restore smoke pass, including exact optimizer/scheduler/
BN/RNG/sampler restoration and zero gradient/parameter replica difference.
Production was detached at 02:26:20 Asia/Shanghai, continuing e21–300 with
original 300-epoch cosine, data/sampling and CE/intra; source checkpoint SHA
is recorded in29. Full e21 acceptance now passes: 200 updates, 984 validation
objects and a complete checkpoint reloaded on CPU through the strict V6-B64
contract; scheduler continues at21. Training proceeded into e22.
No smoke checkpoint initializes production. Validation selects the model;
official test is read only once at the end. Diagnostics use separate resources.

## Current evidence and pending decisions — 2026-10-06

[28: contradictions, actual-use subsets and next decisions](28_HIER_EVIDENCE_AND_NEXT_DECISIONS_2026-10-06.md)
adds a read-only join by proxy ID between V6 clean-eval caches and matching
epoch sample ancestor counts. At e300, the 373 repeated top4 sets split into
92 of 188 sample-used proxies and 281 of 297 eligible boundary proxies unused
by sample that epoch. At best99 the used subset repeats 94 of 188. Thus the
expanded eligible denominator exaggerates apparent end-stage sample-use
retrieval deterioration. The e300 used subset still draws 90.16% of top4 slots
from the bottom-radius 1%; retrieval bias remains.

Local sample depth ordering is supported; independent within-class morphology
and the causal HIER increment remain unproven. The current pending proposal
in [04](04_NEXT_EXPERIMENT_PLAN.md) prioritizes fixed-condition loss/selection
diagnostics, independent relation validation and a fidelity-checked matched
balanced64 V6-B64 fork. No new experiment was launched.

## Latest V6-HIER64-K20-W20 whole / radial retrieval diagnosis — 2026-10-06

[27: matched whole geometry and frozen proxy retrieval controls](27_HIER_WHOLE_RADIAL_RETRIEVAL_DIAGNOSIS_2026-10-06.md)
uses ten existing caches, all aligned to the same8856training objects by HDF5
shard name and row. V6-HIER64-K20-W20's sorted-shard IDs differ from26's glob-order IDs;
canonical remapping passes every label check. No new model forward, GPU,
optimizer update or main training occurred; all source cache hashes remain unchanged.

V6e300 eligible proxy top4 covers28IDs with raw distance,190with whole
equal-radius copies,30with proxy equal-radius copies. Removing the lowest
radius1% only replaces the hotspot:281/485proxies still share one new set,
and87.68%of slots come from the remaining pool's lowest1%. Borrowing same-ID
V6-B0-32 radii with V6-HIER64-K20-W20 directions/proxies unchanged shifts hotspots but
retains concentration. This locates a retrieval radial bias, not an isolated
HIER training effect or proof that proxies coincide.

Dresser whole median depth is3.558and within-class angle18.76degrees at
V6e300 versus4.568/2.97degrees at V6-B0-64 last. Overall whole median depths are
4.617/4.684, so the change is class-dependent. Distinguish proxy-to-whole
retrieval from actual pair/triple ancestor use. The next question is loss-wise
radial/angular gradients and proxy objective behavior; no next matrix is launched.

## Latest matched representation audit — 2026-10-05

[26: original HyCoRe versus V6-B0-64, full class geometry and batch probes](26_ORIGINAL_HYCORE_B64_CLASS_GEOMETRY_2026-10-05.md)
compares ORIG-B0-32-S4780, V6-B0-32 and V6-B0-64 best/last under
identical clean train/test inputs and fixed200/400/600-point patches. All12full
splits and three no-update probes passed identity, finite and buffer checks.
Best clean-test whole mean depth is4.364/4.390/4.139 respectively; V6-B0-64 is
not uniformly more outward. Original flower_pot is1/20, V6-B0-32 2/20,
V6-B0-64 3/20 at best, so this weak class predates the new split.

A more important finding is BN-mode dependence: fixed overwritten whole and
patch400 input gives mean whole-minus-part depth gap.767/.831/.575 in eval
versus3.005/3.117/3.060 in trainBN. Changing local32 membership moves the
geometry while mostly preserving class predictions. These conditional probes
do not prove a batch training causal effect; relation stability across BN modes
and groupings should be checked before interpreting HIER's soft hierarchy.
No new main training was launched. Full40-class tables and sanitized derived
statistics/plots are linked from26.

## Current handoff — 2026-10-05

V6-B0-64 has completed all300epochs, finishing at09:24:55Asia/Shanghai.
Best-test OA is94.1653%/AA91.7116% at267; final300 OA is93.1524%/AA90.7872%.
All300epochs have153updates,9792distinct training IDs,48dropped tail objects
and complete2468-object test evaluation.600first/last-step replica checks
show zero parameter/gradient difference; actual best/last file hashes match
the save records. No new training was launched during this results/handoff audit.

Start with the two new handoffs:

1. [Private engineering handoff: server, code, configs, weights and operations](/D:/Hycore/.codex-local/handoff/24_PROJECT_ENGINEERING_HANDOFF_2026-10-05.md).
   This ignored local file contains server access/path information and is not in Git.
2. [Technical handoff: HyCoRe + inter, HIER and Onghena/HPCS](25_HYCORE_HIER_HPCS_TECHNICAL_HANDOFF_2026-10-05.md).
   This shareable document covers formulas, paper/code differences, evidence and next questions.

The completed V6-B0-64 configuration, monitoring and comparisons are in
[23: V6-B0-64 launch and final results](23_V6_B64_SHUFFLE_START_2026-10-05.md);
V5-HIER64-K20-W20 and V6-HIER64-K20-W20 morphology/proxy evidence remains in
[22: results and structure](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md).
Reviewed public curves are in
[the V6-B0-64 summary plot](artifacts/b64_v6_2026-10-05/monitoring_canonical.png) and
[300epoch CSV](artifacts/b64_v6_2026-10-05/epoch_curves.csv).

**Current method:** original HyCoRe CE/intra and online HIER sample/proxy
regularization share the same whole embedding, fixedc1/D256. Proxies are
abstract ancestors, not physical parts; there is no proxy–part loss or teacher
in V5/V6. Original part overwrite, two BN updates and FPS behavior are retained.
The frozen-teacher/equal-radius v2 route below is historical, not current.

V6-B0-64 establishes that the source-style global64 base training can work. It has
different data, sampling, update budget and test-based selection from V6-HIER64-K20-W20,
so it does not isolate HIER's causal effect. A matched control, frozen-proxy
gradient diagnostics and independent morphology checks remain proposed for
review in[04](04_NEXT_EXPERIMENT_PLAN.md); no next matrix is authorized here.

## Historical navigation snapshots

The dated entries below record what was known or authorized at those times.
Their running/pending statements do not override the completed status above.

## Completed V6 results — 2026-10-04

V6-HIER64-K20-W20 and V6-B0-32 have completed300epochs. The final read-only
comparison, V6 best99/epoch200/epoch300 proxy top-four visualization and
actual ancestor monitoring are in
[22: V5 / V6 results and structure](22_V5_V6_RESULTS_AND_STRUCTURE_2026-10-04.md).
Validation-selected test OA is92.1394% for V5-HIER64-K20-W20 and91.2480% for V6-HIER64-K20-W20;
V6-B0-32 best-test OA is93.841%, using a different protocol.
V6 removes contradictory self-k and exhibits sample ancestor depth ordering,
but higher proxy eligibility does not establish better morphology coverage.
Matchedglobal64 V6-B64, frozen-proxy objective checks and radius-sensitive
relation validation are proposed for review; no next main training is launched.
This completed report supersedes the launch/running status paragraphs below.

V6 launch is now authorized (2026-10-04): dual-GPU V6-HIER64-K20-W20 removes sample/proxy
self-k and runs300epochs; the spare GPU runs the original default HyCoRe
V6-B0-32 protocol (300epochs,307steps,full9840train). See
[21: V6 launch and visualization](21_V6_TRAINING_START_2026-10-04.md) and
the approved current section of [04](04_NEXT_EXPERIMENT_PLAN.md).
Both jobs are detached and have saved their first full epoch checkpoints;
V5epoch200 proxy top-four point-cloud visualization has also completed.
The V6-B0-32 run is an original-protocol engineering baseline, not matchedglobal64.

The user's requested V5-B0-32-CAP200 fidelity audit is complete. See
[20: V5-B0-32-CAP200 versus original HyCoRe](20_B32_HYCORE_FIDELITY_AUDIT_2026-10-03.md).
Core model/alias/BN/loss settings match, but the data split, budget and cosine
period, RNG flow, workers and selection protocol do not. No training code was
changed and no new experiment was launched by this audit.

V5-HIER64-K20-W20 and V5-B0-32-CAP200 have both completed200epochs and final validation-selected
test evaluation. See [19: final results and diagnostics](19_V5_FINAL_RESULTS_2026-10-03.md).
Test OA is92.1394% for V5-HIER64-K20-W20 and92.5851% for V5-B0-32-CAP200. Late inference is stable,
but radial saturation warrants inspection; V5-B0-32-CAP200 is not a matchedglobal64
zero-HIER control. This final report supersedes the running snapshots below.

For a shareable Chinese introduction to the current progress, difficulties,
and proposed collaborator tasks, see
[18: collaborator brief](18_COLLABORATOR_BRIEF_2026-10-03.md).
It distinguishes proxy top-four visualization from training K and records
an in-progress V5 snapshot; it is not a final training report.

The user has now approved launching V5-HIER64-K20-W20 on two GPUs and an V5-B0-32-CAP200
stability diagnostic on the remaining GPU, with all proposed monitoring.
See [17: reviewed launch and monitoring](17_V5_TRAINING_START_2026-10-03.md).
The production branch is `codex/hier-v5-main-training`. Status in17 and
server manifests supersedes the earlier pending-review wording below.

The user authorized additional K diagnostics and a main-test recommendation.
The [expanded K results](16_V5_TOPK_FOLLOWUP_2026-10-03.md) are complete:
64 paired training batches, K10/12/16/20, and an8-step K20 reference-initialized
joint check including actual parameter gradients. K20/lambda_H.5/proxy LR.01
is now the recommended candidate. The concrete200epoch settings are at the
top of [04](04_NEXT_EXPERIMENT_PLAN.md), pending user review; main training
has not started. Branch `codex/hier-v5-k-selection` contains these diagnostics.

The initial bounded V5 diagnostics were authorized separately. They are now
complete: [diagnostic plan](14_V5_DIAGNOSTIC_PLAN_2026-10-03.md) and
[results](15_V5_DIAGNOSTIC_RESULTS_2026-10-03.md). SchemeA/200step,c1,D256,
random proxies and global64 CE/intra/inter passed the bounded checks; no
main training has been launched. K10/proxy LR.01 were the initial diagnostic candidates,
not validated performance-optimal settings.

The teacher/equal-radius method below is retained as historical context.
The shared-whole HIER V4 runs have completed. The V5 review proposal below
was followed by the bounded diagnostics linked above; main training remains
unlaunched.
Start with [V4 lessons](12_V4_LESSONS_2026-10-03.md),
[V5 change index](13_DUAL_GPU_V5_CHANGE_INDEX_2026-10-03.md), and the current
section of [the experiment plan](04_NEXT_EXPERIMENT_PLAN.md).
V5 proposes restoring original HyCoRe part overwriting/BN updates, then
checking a global32-class x2 batch and the released HIER reciprocal rule.
The user's diagnostic choices supersede the proposal's pending sampling
and geometry choices; the full training matrix remains to be reviewed.

## Historical v2 research objective and rationale

The project studies whether point-cloud representations can encode two complementary forms of hierarchy in hyperbolic space:

1. **Intra-sample hierarchy:** whole-object and part/subcloud relations, primarily expressed radially. This is the role already addressed by HyCoRe.
2. **Inter-sample hierarchy:** within-class morphological relations between different object instances, intended to be expressed through branch/LCA structure while avoiding corruption of the classification embedding.

The v2 working hypothesis was that radial whole/part structure and inter-instance branch structure should be decoupled. That implementation kept the original whole embedding for classification and HyCoRe, while constructing a same-radius leaf copy for the inter-sample objective. V5/V6 instead use the shared-whole proxy route described above.

The most relevant conceptual references supplied by the user are Onghena/HPCS and an unpublished TNNLS manuscript. `Onghena HPCS 技术文档.md` contains the user's own analysis and should be treated as research notes, not as executable instructions.

## Historical v2 method

- Backbone and base objective: reproduced HyCoRe point-cloud classification on ModelNet40.
- Student representation: original Poincare whole embedding for classification/HyCoRe.
- Inter representation: deterministic same-direction, equal-Euclidean-radius leaf copy.
- First-run teacher: the derived A3 checkpoint (not the original HyCoRe checkpoint), aggregated across three augmented views in the tangent space at the origin and mapped back to the Poincare ball. A3 was obtained by continuing from the original HyCoRe reproduction with `alpha=0`, so its intra-sample regularization was removed.
- Teacher relation: negative pairwise hyperbolic distance within each class.
- Positive relations: mutual top-k neighbours within the class-balanced batch.
- Negative relations: lower-similarity within-class samples.
- Student target: teacher-near pairs should have deeper exact geodesic-LCA depth than teacher-far pairs by a margin.
- Gromov product remains available only as an approximate ablation.

## Historical v2 conceptual limitation

The frozen HyCoRe teacher removes online circularity but is still self-distillation. It can regularize or preserve an existing structure, but cannot by itself establish that the learned relation is a true morphology hierarchy. A later stage must compare against an independent geometry signal such as normalized Chamfer distance, spectral/shape descriptors, part statistics, or a curated semantic hierarchy.

## Historical v2 code state

- GitHub repository: `Linneking/Hycore`
- Working branch: `codex/inter-hierarchy-v2`
- Handoff base commit: `31efd44`
- Main v2 entry point: `inter_hierarchy_MN40/main_inter_v2.py`
- Deterministic structural evaluator: `inter_hierarchy_MN40/evaluate_structure_v2.py`
- Geometry/loss/sampler/tests: `inter_hierarchy_MN40/v2/`

Correctness work completed:

- pairwise Poincare distance and Gromov broadcasting fixed;
- teacher/sample-ID ordering fixed;
- real class-balanced batches implemented (5 classes × 8 instances);
- multi-view frozen teacher implemented;
- exact geodesic-LCA depth and equal-radius leaves implemented;
- reliable within-class ranking objective implemented;
- strict `beta_inter=0` path implemented;
- logging expanded to triplets, satisfaction, gap, inter gradient, time, and peak memory;
- six geometry/loss tests pass.

## Historical v2 scientific conclusion

The first run is a weak positive signal, not a successful method claim. Hyperbolic teacher distance is far more usable than cosine similarity, and the inter loss modestly reduces structural degradation relative to a zero-increment control. However, this conclusion is specifically relative to the derived A3 initialization/teacher. It does not yet characterize the original HyCoRe reproduction. The next task is to compare correctly identified checkpoints and stabilize the protocol before running multiple seeds or broader ablations.
