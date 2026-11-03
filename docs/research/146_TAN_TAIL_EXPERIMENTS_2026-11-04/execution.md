# 三组TAN尾部实验执行记录

用户授权“先诊断取得初步结论，再按8000空闲GPU数量开展相关实验”。145完整只读诊断指出table差距97.29%来自GT不存在部件时的额外误报，lamp还有真实部件/内部误差；冻结COS反事实降低成绩。依据这些结果选择输出bias、切空间BN/Leaky及其联合组，具体假设和层次见[plan.md](plan.md)。

## 版本与检查

最终生产源为de24c6b，已在本机提交推送、同提交Git包fetch/pull --ff-only更新到新独立对象库的隔离源。旧源码、旧结果保持；0747171为此前两组准备版本，未启动过生产。新v13与原v10共享全部原模型张量和构造RNG，只增加所述参数及观察指标。

本机/服务器各8项CPU合同全部通过：三组各完整2轮×2步，BN4、50部件、保存恢复、调度200、无R训练图、无CUDA初始化/无真实test访问；所有共享模型初始hash在服务器为17c51802…，与R0精确一致。新增参数分别BIAS50、BNLEAKY256、联合306；独立零bias前向精确同R0，联合零bias前向精确同单BN组；新增观察与原mIoU计算完全一致。见[cpu_checks_local.json](cpu_checks_local.json)、[cpu_checks_server.json](cpu_checks_server.json)。

## 正式启动

原GPU0任务自然完成后，连续三次核验GPU0/1/2均无计算进程、0%且仅空载显存。北京时间2026-11-04 01:02:13直接fresh200，无GPU短测：

|实验|GPU|新增可训练参数|正则|
|---|---:|---:|---|
|TAN-BIAS-R0|2|50|关闭|
|TAN-BNLEAKY-R0|1|256|关闭|
|TAN-BNLEAKY-BIAS-R0|0|306|关闭|

seed4780、CE(5q)/5、自然shuffle32、N2048/k40、RSGD .1→.001/200、momentum.9/WD1e-4、无总裁剪、既有trainval/test协议保持。每批一次编码，新BN只在训练主支更新一次。新的absence误报与GT存在条件IoU仅观察，不用于改变预测或选择checkpoint。

01:02:52断开重连三组均e1/step125/update125，后台父进程1，无failure，实际共享初始hash均匹配R0，训练loss=CE且reg=0。完整首轮保存后另补验收结果，不能用此启动快照代替首轮验收或终局收益。

冻结源不得拉入后续文档或重复启动。完整私有路径/PID/命令在本机忽略的146交接中，公开调度摘要见[dispatch_summary.json](dispatch_summary.json)。所有旧结果保持。

## 完整首轮验收通过

北京时间01:06核验，BIAS/BNLEAKY各39项、联合组47项全部通过，见[launch_acceptance.json](launch_acceptance.json)：共享e0张量/hash/CPU RNG、数据/首输入/part配对与R0精确相同；完整437更新/BN437/test2874/50parts、Roff无训练图、CE到原双M及新增模块的梯度、新参数实际更新、模型优化器有限、源提交一致，无failure。新增参数数目和零初始化按方案正确，q始终按实数分数观察。

三组已进入e3，快照update约974–1024。首轮分别BIAS110.49秒、BNLEAKY107.31秒、联合107.45秒，按当前负载预计北京时间11月4日07:00–07:30左右完成，实际速度可能变化。初轮分数仅用于启动验收，不作性能结论。

本机/服务器最终均为8项CPU合同、三组各2×2完整合成入口；诊断GPU进程已正常结束后才使用空闲卡。本轮没有恢复旧任务、改原结果或新增第四条训练。以后按同轮best、固定e200和末20Class/Instance，以及GT-absence误报/GT-present召回代价报告结论。
