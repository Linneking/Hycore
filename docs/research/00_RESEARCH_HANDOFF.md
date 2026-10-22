# Research handoff: HyCoRe inter-sample hierarchy

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
