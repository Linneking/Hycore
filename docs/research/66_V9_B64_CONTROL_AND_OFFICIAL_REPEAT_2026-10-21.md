# V9同协议B64对照与官方HyCoRe复现候选

日期：2026-10-21。双卡V9-B64已由用户明确授权设计、验收并后台启动；用户随后明确确认第三张单卡启动官方c1重跑，两条均已授权实现、验收和从头正式启动。所有旧训练、权重和结果保留。

## 双卡：V9-B64无HIER

规范名：`V9-B64-C1-CAPnone-GS0-M1000_4-TS-S22`。

| 条件 | 正在运行的V9-HIER64 | 本次V9-B64 |
| --- | --- | --- |
| 原生whole/part | c1、无额外cap/GS、1000/N与4 | 完全保持 |
| 采样与BN | 双卡local32/global64，每卡16类×2、共享local32 BN、part先whole后 | 完全保持 |
| 训练 | seed22、从头、300轮×200步 | 完全保持 |
| 数据与选模 | 全9840训练、2468 test逐轮strict OA best、无earlystop | 完全保持 |
| 模型优化 | RSGD .1→.005余弦、momentum.9、wd.0002、norm clip1 | 完全保持 |
| CE与HyCoRe | CE+.01 Rcontr+.01 Rhier | 完全保持 |
| HIER视图/损失/代理 | c0.1深度4.6视图，lambda.5，512代理与AdamW | 零构造、零调用、无代理优化器 |

原V8-B64-TS确实已完成，OA93.881686%；其基础分支为c0.1/cap2.3/GS1/500/N与2，无法替代这里c1无cap/GS的对照。V6-B64则为8856/984协议，也不替代完整9840对照。

本次只检验V9的HIER分支整体增量，包括其固定壳样本视图和层次代理目标；不将与V8或ORIG的差异全归因HIER。控制采样/增强种子、global flip负例、数据行顺序、模型初始权重和学习率预算。跨机器不承诺浮点逐位训练重放。

## 实现与验收

新增`hier_proxy_scratch_v9/controls.py`及`controls_queue.py`，保留V9主入口不动。strict no-HIER合同、全训练集选模合同、原HyCoRe数值/梯度合同及CPU双进程global64梯度对照通过后，才等待两张完全空闲GPU。每张卡使用UUID，连续两次资源检查和本用户锁，禁止中断别人。

先独立2轮×2步train-only smoke，不读取test；检查c1、无cap/GS、无代理、loss=base、BN每步+2、4次更新、有限checkpoint。然后重新初始化正式300轮，不能从smoke续训。native初始SHA须匹配V9-HIER：`fd7251e0eb6b397ecaf3862e9cc00a6bc2be9b684329fdc31dc05915b388d29d`。

正式每轮保存指标和best/last，周期checkpoint，以及与V9主训练相同audit epochs的真实训练原生whole/part快照。首轮核对200次更新、2468 test、双rank状态、BN计数400、有限参数、学习率及数据身份。源码本地独立分支提交/推送后服务器fetch/pull --ff-only并冻结。

## 单卡：全新官方c1复现（已确认）

规范名：`ORIG-B0-32-C1-CAPnone-GS0-M1000_4-TS-S4780-R2`，R2明确是新一次从头复现，不覆盖历史同名实验。

- 全链c1，不新增HIER式cap或GS，不加入HIER及代理。
- 保留官方PointMLP+HyCoRe训练/评估函数、part先whole后、共享BN、原数据增强与持续随机流、1000/N及4、两正则系数各.01。
- seed4780，shuffle32/drop-last，307步/轮，完整300轮，workers8、eval batch16，FP32；原RSGD .1→.005余弦/momentum.9/wd.0002/clip1。
- 从随机权重开始，不resume、不加载旧best。全9840训练，每轮2468 test选best，无voting，无提前终止。
- 新环境、数据内容/行顺序、配置、源码和初始权重身份全部记录；短测仅训练集，正式从头重置。

已有V6-B0-32是seed22、从头官方训练300×307，test best93.841%，但并非当前8000环境的seed4780配对组；不声称此前从未完整复现。历史seed4780原版固定best复测为94.043760%，论文无voting为94.3%；差约0.256个百分点，不能单凭这一差值判定代码错误。新从头复现可验证当前环境的完整官方训练结果，并与c0.1原版修改线建立c1参照；一条run不能保证达到论文数值，也不能估计跨seed方差。用户已在本轮明确确认启动此单卡组。先前建议的ORIG高表现组改seed22未被选用，不额外启动第三条实验。

## 上线状态

两条均已从头正式启动、完成首轮验收并在后台持续运行；不追加其他实验，不以test提前停止。

| 实验 | GPU | 正式启动（北京时间） | 生产源码 | 首轮验收 |
| --- | --- | --- | --- | --- |
| V9-B64原生c1无HIER | 0、1 | 2026-10-21 00:55:07 | `89710b5` | 200更新、完整2468 test、BN计数400、双rank状态、有限best/last/e1 |
| 官方HyCoRe c1 S4780 R2 | 2 | 2026-10-21 00:59:06 | `21f6c69` | 307更新、完整2468 test/155批、BN计数614、有限网络/优化器、best/last/e1 |

双卡33项CPU与global64双进程梯度对照、单卡10项CPU均通过；各自2轮×2步train-only短测后，重新随机初始化正式训练。B64初始网络SHA与V9主训练一致；两组各自smoke/main的e0网络张量逐元素相同。正式训练不继承短测结果。

B64与V9主训练的训练标签、test标签、划分SHA相同；首个真实global64批次的ID、标签哈希、whole835/part324也完全一致，补足采样和crop身份验证。HIER关闭是该匹配对照的主要目标差异。官方c1复现的五个训练HDF5、标签SHA和行顺序与当前ORIG c0.1组完全匹配；保留原版连续随机流。

首轮B64约126.82秒，官方c1约118.53秒；这是启动观察，不是完整训练耗时承诺。两组均300轮，暂不据早期OA判定效果。

验收证据：[V9-B64](artifacts/v9_b64_and_official_c1_2026-10-21/v9_b64_startup_acceptance.json)、[官方c1](artifacts/v9_b64_and_official_c1_2026-10-21/official_c1_startup_acceptance.json)。源码均先本地提交并推送，再服务器fetch/pull --ff-only及冻结；运行目录/进程/环境在本地忽略的66私有交接，不提交权重或原始日志。

官方组第一次CPU导入缺少matplotlib，未分配GPU或正式训练；随后使用此前原版组已安装的私有依赖扩展目录重新提交新目录，10项CPU及GPU短测通过。没有修改共享环境、损失或训练参数。此前高表现ORIG c0.1/cap/GS/半margin组已自然完成300轮，best test OA94.327%；本次纯c1重复可为该修改线提供当前环境的原版锚点。
