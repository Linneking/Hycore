# V6 B64 shuffle 基线：配置与启动记录

## 授权和问题

2026-10-05，用户批准补一个双卡 V6 B64 基础训练，要求使用 HyCoRe 的打乱、无放回、丢弃尾部方式，并监测基础训练指标。用户随后明确选择全部 9840 训练样本、原版逐轮官方 test 和 best-test 选模，训练 300 轮。

本实验检验原采样方式下双卡 B64 的基础训练是否稳定，帮助理解 V6 H20 与原 B32 之间的差异。它同时改变了 H20 的抽样、数据量和每轮步数，因此不是严格的 HIER 单变量消融。

## 固定设置

| 项目 | 配置 |
|---|---|
| 入口 | `inter_hierarchy_MN40.hycore_b64_v6.train` |
| 初始化 | seed22，从头随机，不读取历史权重/teacher |
| 空间、主干 | c=1，D256，原 Hype_pointMLP |
| 数据 | 全 9840 train、官方 2468 test，没有 validation |
| 批量 | 两卡 local32，global64 |
| 采样 | 每轮一个完整全局排列，153 个完整 batch；9792 个唯一实例，丢尾 48，无补齐/有放回 |
| 预算 | 300 轮×153 步＝45900 次更新 |
| 输入 | whole800–1024、part200–600，part 视图复写 whole；首层 FPS 请求512 |
| 基础损失 | 原标签平滑 CE eps=.2，`.01Rcontr+.01Rhier`，margin4及`1000/Npart` |
| 负配对 | global child.flip(0)，自然同类负配对不筛掉 |
| HIER | 无 HIER loss、无 proxy、无 proxy optimizer |
| BN | 每卡32的普通训练BN，child/whole各更新一次；DDP广播rank0 buffer |
| 优化 | RiemannianSGD，LR.1，momentum.9，WD2e−4；300轮cosine至.005 |
| 梯度裁剪 | 模型全局 L2 范数阈值1，不因B64改变LR |
| test/选模 | 每轮test；按原三位小数OA严格更高保存best，平局不换 |
| eval 分块 | 16，以保持与V6原源码B32一致 |
| workers | 4/rank，总8 |
| 保存 | 每轮last/best/metrics/heartbeat，每20轮存档；optimizer/scheduler与两rank RNG/sampler/BN |

## 算子兼容与限制

全局64的连续位置分给 rank0 前32和 rank1 后32，汇总两卡的嵌入和logits后计算全局 CE/intra。使用已有的可微 all-gather：反向 SUM 与 DDP 参数梯度平均抵消，不额外乘除 world size。全局 flip 会把负 part 配到另一张卡；不能改成各卡 local flip。

旧 balanced 入口强制 flip 异类，不能直接用于普通 shuffle。新路径允许源式随机 batch 的同类负配对并监测其比例。intra 的零点使用原 `torch.max` 算子，与 ReLU 在精确零点的梯度区别明确保留。

普通 local32 BN 和单卡 B64 BN 不等价；rank0 buffer 广播也不是 SyncBN。各 rank 的 running mean/var 可不同，梯度与模型参数应同步，BN计数每步均增加2。采样排列和双卡随机流可重现，但不声称与原单卡 DataLoader 的 PyTorch 随机流逐位一致。

原脚本直接使用train batch64时，test分块默认会是32。本次显式使用16，与既有原B32保持相同评估分块；完整test数量和源式选模规则不变。

## 监测与验收

- CE、Rcontr、Rhier、加权总损失，训练模式OA/AA和逐类结果。
- 每步实际global64的类别数、flip同类negative比例；实际采样ID与全局计划顺序一致。
- 每轮唯一实例与逐类覆盖、重复次数、丢尾；生产断言153步、9792不同ID。
- whole/part半径与原点深度、球边界附近比例及关闭额外截断的shadow统计。
- 原part复写、两次BN更新、FPS请求、固定c1；非有限值失败记录。
- 裁剪前后范数/触发比例、两rank梯度和参数一致性、实际LR/耗时/显存。
- 每轮官方test OA/AA、逐类准确率及CE；保存source-rounded与raw值。源码CE按batch等权，另存样本加权CE，避免混合口径。
- 每10轮额外全量clean train评估，观察train/test差距；eval及诊断恢复RNG和训练状态。

启动前：采样CPU测试、源损失与分布式基础梯度检查、双卡2轮×2步短测；短测只测有限test batch并明确标smoke，不能使用其权重初始化生产。正式训练须确认第1轮全153步、完整2468test、checkpoint和第2轮推进后才报告挂起成功。

## 实际状态

分支：`codex/v6-b64-shuffle-baseline`。生产提交：`56b40a2b11d3274ff7a9832b8e4cced3b366ec5c`。已按本地提交/推送、服务器干净工作区fetch及pull --ff-only更新。

验收记录：

- 8项采样CPU检查、2项新入口与原源码损失/梯度检查、6项已有分布式/输入算子CPU检查通过。损失值逐值一致，独立FP32梯度图最大约3e−11累加差异，梯度测试容差rtol1e−6/atol1e−10；精确零点hinge半梯度另作严格检查。
- 独立短测目录完成2轮×2步、共4次更新。每轮实际128个唯一训练ID；每轮只看32例test，均明确标记smoke/partial，不用于报告分类结果或初始化生产。
- 短测两rank梯度差、更新后参数差均为0；裁剪后范数最大0.99999988，单卡峰值allocated显存28444.64MiB。四步同类flip负数分别0/0/6/2，全部合法且正常通过。
- BN每步+2、输入alias复写和固定c1断言通过，test/cleantrain评估buffer不变；last/best保存和checksum通过。重新CPU加载checkpoint，确认248模型状态项、126优化器状态项、调度器及两rank RNG/sampler/BN均存在；未宣称断点恢复已测试。
- 正式任务使用全新结果目录，2026-10-05 00:48:45Asia/Shanghai在两张完全空闲GPU启动，已脱离当前连接运行。生产从头随机seed22，不读取短测权重。服务器路径、PID、GPU编号/UUID仅保存在忽略的本地私有记录和服务器manifest/launcher_state，不提交仓库。

正式挂起验收已通过：

- 第1轮完成153次更新，实际draws/唯一ID均9792，覆盖99.5122%，重复0、补齐0、丢尾48。实际完整test2468例、155个eval batch，buffer保持不变。
- 第1轮last/best和metrics均保存，last约160.03MB；首末步两rank梯度差/参数差均0。153步损失/梯度保持有限，裁剪前范数最高6.3041、24步触发原norm1裁剪（15.69%），裁剪后最大0.99999982；同类flip负配对比例3.6560%。峰值allocated显存28445.89MiB。
- 第1轮训练及test/保存统计耗时104.65秒（不含前置初始化）。首次test OA19.7326%只是启动轮监测，不是训练最终结果。
- 最后验收时heartbeat已到第3轮第4步，manifest保持running，官方全量test已执行，训练进程已脱离当前连接独立运行。后续每轮监测和保存由训练进程自行继续；当前不声明300轮已完成。
