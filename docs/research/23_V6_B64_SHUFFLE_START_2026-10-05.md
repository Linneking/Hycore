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

全局64的连续位置分给 rank0 前32和 rank1 后32，梯度汇总后再计算全局 CE/intra。使用已有的可微 all-gather：反向 SUM 与 DDP 参数梯度平均抵消，不额外乘除 world size。全局 flip 会把负 part 配到另一张卡；不能改成各卡 local flip。

旧 balanced 入口强制 flip 异类，不能直接用于普通 shuffle。新路径允许源式随机 batch 的同类负配对并监测其比例。intra 的零点使用原 `torch.max` 算子，与 ReLU 在精确零点的梯度区别明确保留。

普通 local32 BN 和单卡 B64 BN 不等价；rank0 buffer 广播也不是 SyncBN。各 rank 的 running mean/var 可不同，梯度与模型参数应同步，BN计数每步均增加2。采样排列和双卡随机流可重现，但不声称与原单卡 DataLoader 的 PyTorch 随机流逐位一致。

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

当前为实现与验收阶段；尚未声明正式训练已启动。分支：`codex/v6-b64-shuffle-baseline`。生产提交、GPU、开始时间与启动结果将在确认后补充，完整私有身份以新结果目录manifest为准。
