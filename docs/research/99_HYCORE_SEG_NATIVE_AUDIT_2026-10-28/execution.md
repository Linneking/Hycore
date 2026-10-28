# Three native HyCoRe-seg runs, 2026-10-28

用户在审阅三组结构后明确要求实现并在8000三个空闲GPU启动；统一seed4780、batch32、shuffle。此授权覆盖此前仅设计状态，只启动以下三组完整HC，不追加H0或其他消融。训练开始和验收事实在完成后补录，不把排队或短测写成正式训练。

| 实验 | 配置名 | 主支 | 正则输入F | one-hot | 可训练参数 |
|---|---|---|---:|---|---:|
| 一 | p0 | HyCoRe分类编码器→逐点1024→512→exp→原Möbius512→256→50 | 512 | 无 | 1287346 |
| 二 | pglobal | 同编码器→点1024+广播全局max1024→2048→512→同双曲头 | 512 | 无 | 1811634 |
| 三 | pseg3 | AnTao分割STN/局部192/global1024/category64→1280→256→exp→原Möbius256→256→128→50 | 256 | 保留 | 1460027 |

代码位于`segmentation_ShapeNetPart/hycore_seg_v2`。原分类和v1未修改。官方geometry直接读原文件；完整RiemannianLayer/MobiusLayer、triplet与CE核忠实迁移，import时验证五份原文件LF SHA。分类graph仅将硬编码CUDA设备改为输入设备。Pseg3直接复用已固定的AnTao前端，移除被替换尾部BN/激活/dp2，融合端dp1保留。新层按原规则fresh初始化，不加载分类或v1权重、无教师。

统一配置：XYZ2048、k40、batch32/eval16、训练shuffle和drop_last、workers8/CPU线程4、FP32单卡；RSGD .01→.0001 cosine200、momentum.9、WD1e-4、总梯度clip1；原CE显式smoothing=False；两正则各.01、gamma1000/N′、delta4、feature part200–600、原batch flip且不筛同类。所有训练只单次编码H；正则仅在逐对象whole/part均值exp上计算，CE在训练入口合并一次。独立part RNG完整保存。

生产数据按已选协议trainval14007/test2874，逐轮unmasked test Instance mIoU取>=保存best，固定200轮、不early-stop；每轮437更新，共87400。该协议有test选模偏差；当前raw适配保留26个跨集合共有ID与9个trainval重复条目，不能称独立测试或原HDF5逐位复现。与旧v1相比，架构/完整双曲部件与LR/clip均变化，不能把结果差异全部归因于正则。

验收顺序：本机CPU合同及原组件回归→提交推送→独立服务器工作区快进部署→服务器CPU合同/合成全入口→每卡真实batch32/N2048无test短测2轮×2步→重新初始化fresh200。短测检查BN4、模型/优化器有限、保存状态、200调度、两支共享梯度非零且独立头无正则梯度。正式入口在第一步前验证初始state与短测相同。

## 当前验收

- 本机独立源回归：编码器、Pseg3前端train/eval与五种尺寸Möbius输出/输入及参数梯度最大绝对误差均0；原类/损失核/CE语义AST一致。
- 本机/服务器各18项CPU合同通过；三架构各2轮×1步合成完整入口的17项验收均通过。
- 三卡真实batch32/N2048短测各2轮×2步通过，BN4、模型/优化器有限、共享CE/正则梯度非零且独立头无正则梯度；峰值分配显存P0/Pglobal/Pseg3为22.463/22.715/19.496GiB。
- 源码提交`ac6c712699b1428d4b243703387eb611aabd495a`已推GitHub；服务器HTTPS拉取TLS失败后，以同一提交的增量Git版本包传输，隔离工作区git fetch并pull --ff-only成功，源码状态干净。生产工作区冻结于此提交，后续文档提交不拉入活动运行。
- 2026-10-28 22:15:25（Asia/Shanghai）三组均fresh200启动，初始state与各自短测逐位SHA相同。22:18:01核对均已完成e1的437更新/2874测试对象，模型及优化器有限，BN437，e0/best/last保存，已进入e2；没有failure。完整启动证据见[startup_acceptance.json](startup_acceptance.json)，不是最终训练成绩。

| GPU | 实验运行ID | 首轮用时 | 22:18:01累计更新 |
|---|---|---:|---:|
| 0 | seg_native_P0_HC_s4780_20261028_A | 122.785s | 562 |
| 1 | seg_native_Pglobal_HC_s4780_20261028_A | 124.749s | 537 |
| 2 | seg_native_Pseg3_HC_s4780_20261028_A | 110.435s | 637 |

三组正式首批输入SHA均相同；正则中心/part RNG独立。首批whole/part投影比例均1.0、深度约6.21261且差接近0，说明初始层次深度受标准球投影限制。合并正则对编码器梯度非零不能单独证明层次项有效；此数值现象已在会话说明并保留诊断，不擅自添加缩放/归一化或把数值有限等同于收敛。

运行目录、提交、物理GPU、开始时间及环境保存于各独立manifest；服务器连接信息仅留本地私有配置。完整日志/权重/数据不入Git。
