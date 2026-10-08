# 实验命名规范与旧名称映射

日期：2026-10-06。本次只统一实验身份和展示名称，不启动训练，不改写已有结果。
机器可读映射见 [experiment_registry.json](experiment_registry.json)。

## 1. 六个主实验族

名称先回答“怎样组成 batch”，再回答“是否加入 HIER”。32/64 都指 **global batch size**；双卡 local32 对应 global64。

| 主族 | 采样方式 | 训练目标 | 用途 |
|---|---|---|---|
| B0-32 | HyCoRe 式普通 shuffle、无放回、drop_last；global32 | CE + HyCoRe intra，关闭 HIER | 原采样协议的32版基础训练 |
| B0-64 | 同上，global64 | 同上 | 原采样协议的64版基础训练 |
| B32 | HIER 式 class-balanced；global32 | CE + HyCoRe intra，关闭 HIER | 分离 balanced batch 的影响 |
| B64 | HIER 式 class-balanced；global64 | 同上 | HIER64 的同协议关闭对照 |
| HIER32 | 与相应 B32 相同的 balanced batch | CE + HyCoRe intra + HIER | global32 的 HIER 实验 |
| HIER64 | 与相应 B64 相同的 balanced batch | 同上 | global64 的 HIER 实验 |

当前 V5/V6 的 balanced64 为 **32 个不同类别 × 每类2个实例**，类内有放回，双卡各16类×2。
global32 的现代同规则版本应为16类×2；目前没有已完成的该版本 B32 或 HIER32 主训练。

B0 的命名体现还原 HyCoRe 采样和基础训练的初衷。它不自动保证整套协议完全复刻：
数据划分、训练步数、BN、随机数、评估和选模仍须看版本及 registry。
例如 V5-B0-32-CAP200 恢复了核心算子，但不是完整原训练协议；
V6-B0-64 保留原采样规则，却采用双卡 local32 BN、global64 损失和显式分布式随机流程。

## 2. 当前已运行实验的唯一展示名

| 统一展示名 | 旧名称/源别名 | 核心身份 |
|---|---|---|
| **ORIG-B0-32-S4780** | 历史原版 HyCoRe、original best/last、var-4780 | 全9840训练；shuffle32；300轮名义预算；历史 seed4780 |
| **V5-B0-32-CAP200** | V5 B32、B32 stability diagnostic | 8856/984；shuffle32，每轮只取前200批；200轮；seed22 |
| **V6-B0-32** | B0_original_B32、original_source_B32、source_b32_best/last | 全9840；原源码 shuffle32、307步/轮；300轮；seed22 |
| **V6-B0-64** | B64_shuffle、旧 B64、b64_best/last | 全9840；shuffle64、153步/轮；300轮；seed22 |
| **V6-B64** | B0_balanced、V6 matched balanced64 B0、刚完成的同协议 B0 | 8856/984；balanced64、200步/轮；从完整共同e20状态分叉至300；seed22 |
| **V5-HIER64-K20-W20** | V5 H20、H20 | balanced64；K20；前20轮仅base，从e21启用λ=.5；200轮；seed22 |
| **V6-HIER64-K20-W20** | V6 H20、H20 | balanced64；K20；前20轮仅base，从e21启用λ=.5；排除索引self-k；300轮；seed22 |

**最需要纠正的两个名称：旧 V6 B64 shuffle 改称 V6-B0-64；刚完成的 V6 balanced B0 改称 V6-B64。**
两者数据、采样、预算和选模不同，不能按同一名称合并曲线或指标。
V6-B64 与 V6-HIER64-K20-W20 才是当前的同协议历史对照；不承诺逐位重放。

32/64 不是模型维度，均使用当前的 D256、c1。
K20 是训练关系使用的近邻参数；top4 是检索展示数量。
源别名 **H20 的20指K20，不代表warmup20**。历史 H20 恰好同时有20轮warmup，因此正式展示名分开写 K20 和 W20。
未来新初始化 HIER 按已批准规则写 W0，从首轮启用 HIER，实际 λ、学习率及启用轮次由 manifest 记录；不把旧 W20 结果重标为 W0。

## 3. 协议信息放在元数据中

短名保留版本、实验族以及必要的 K/W。seed、总轮数、每轮更新数、ForkE20、
class composition、数据划分、BN 和模型选择放入 registry、manifest 和图注。
比较多seed时追加 S22 等后缀；ORIG 保留 S4780，避免与源码默认 seed22 混淆。

| 实验 | 实际训练/验证 | 每轮模型更新数 × 总轮数 | 选模与官方 test |
|---|---|---|---|
| ORIG-B0-32-S4780 | 9840 / 无 | 名义307×300；历史早期resume，实际完整计数未重新验收 | 历史逐轮test；test选模 |
| V5-B0-32-CAP200 | 8856 / 984 | 200×200＝40000 | 验证选模，最终test一次 |
| V6-B0-32 | 9840 / 无 | 307×300＝92100 | 原逐轮test，test选模 |
| V6-B0-64 | 9840 / 无 | 153×300＝45900 | 原逐轮test，test选模 |
| V6-B64 | 8856 / 984 | 200×300＝60000；本次续训56000 | 验证选模，最终test一次 |
| V5-HIER64-K20-W20 | 8856 / 984 | 200×200＝40000 | 验证选模，最终test一次 |
| V6-HIER64-K20-W20 | 8856 / 984 | 200×300＝60000 | 验证选模，最终test一次 |

更新数指模型 optimizer step；HIER的proxy optimizer另计，不与模型更新相加。
“300轮”包括共同前缀。V6-B64只新运行e21–300，完整恢复了e20的模型、动量、调度、BN、RNG和sampler，没有重训prefix或重新warmup。

V5-B64 只出现于历史待批准的同协议计划，未形成完成主运行。
现代 B32/HIER32 也没有完成结果；旧shuffle B32不得作为它们的运行证据。

## 4. V1–V4 保留为历史方法，避免冒充新主族

| 历史源别名 | 历史展示映射 | 为什么须单列 |
|---|---|---|
| proxy V1 B0 | LEGACY-V1-BASE40 | 原e229初始化，冻结backbone/BN，5类×8的global40微调控制 |
| proxy V1 B1/B2 | LEGACY-V1-HIER40-IN / LEGACY-V1-HIER40-INOUT | 固定原特征关系；B1类内样本项+proxy，B2再加入类间项；不是现代K20协议 |
| proxy V2 online B1/B2，scale1/2/3 | LEGACY-PROXY-V2-HIER40-IN / LEGACY-PROXY-V2-HIER40-INOUT，保留scale字段 | 原e229与冻结backbone/BN，学生在线mining；8轮短跑和30轮warmstart是不同运行 |
| scratch V3 B1_scale3/B2_scale3 | LEGACY-V3-HIER32-IN-SCALE3 / LEGACY-V3-HIER32-INOUT-SCALE3 | 4类×8、跨batch重复，307步/轮；20prefix+20ramp；两臂没有同协议B0 |
| scratch V4 B0 | LEGACY-V4-B32-Q4x8-BN1 | 4类×8全覆盖队列；独立child存储、仅whole更新BN；不能作为原版B0或现代源式B32 |
| scratch V4 H3 | LEGACY-V4-HIER32-K3-W20-R20-Q4x8-BN1 | sample K3；4×8队列，独立child存储、BN1；20prefix后另有20轮ramp |
| scratch V4 H5 | LEGACY-V4-HIER32-K5-W20-R20-Q4x8-BN1 | sample K5；除K外与H3同协议 |

V4三臂已完成200轮；其281步/轮与现代200步/轮不同。
V1/V2历史精确完成状态及V3最终300轮验收没有在本次重新读取完整源manifest，registry相应标 historical/unknown，不以默认CLI预算代替运行证据。
教师/equal-radius的 inter-v2 与 proxy-V2-online 是两条不同方法线；称呼必须包含方法线。
一步诊断中的 base、ST、fixed 是从同一 H20状态恢复的临时实验臂，继承其历史动量，**不是 B0/B32/B64 的完整训练基线**。

## 5. 保留归档身份，通过映射读取

不改源run ID、服务器运行目录、checkpoint文件名、旧代码参数、原日志/metrics字段和既有缓存键。
旧论文笔记、历史计划中的旧称保留出处；当前展示与比较以本表为准。
同一个 B0/H20/B32 字符串在不同版本中可能指不同实验，必须结合 version、method line、run context 查 registry，禁止全局字符串替换。
机器canonical ID与source alias跨映射只改变展示名，不改变样本数、数值或结果所属实验。

主要证据：[V4计划](10_HIER_PROXY_V4_PLAN_2026-10-02.md)、
[V4教训](12_V4_LESSONS_2026-10-03.md)、
[V5启动](17_V5_TRAINING_START_2026-10-03.md)、
[V5源码一致性审计](20_B32_HYCORE_FIDELITY_AUDIT_2026-10-03.md)、
[V6启动](21_V6_TRAINING_START_2026-10-04.md)、
[旧B64 shuffle完成记录](23_V6_B64_SHUFFLE_START_2026-10-05.md)、
[同协议balanced完成记录](32_V6_BALANCED_B0_COMPLETED_2026-10-06.md)。

七个主运行的生产提交和源组别已按上述启动/完成记录复核。历史 ORIG 的生产源码快照未知，registry保留null；
其c1/D256由[原版模型与实际checkpoint审计](26_ORIGINAL_HYCORE_B64_CLASS_GEOMETRY_2026-10-05.md)确认：
原模型为Hype_pointMLP，四处曲率state经softplus对应c1，双曲embedding为D256。
