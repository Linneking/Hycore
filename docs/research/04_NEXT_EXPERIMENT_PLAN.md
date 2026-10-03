# Next experiment plan — current review and historical drafts

Status: **V5-H20双卡与B32单卡均完成200轮和验证选中模型的最终test。结果与诊断见[19](19_V5_FINAL_RESULTS_2026-10-03.md)；以下保留已批准配置，不自动授权下一轮训练。**

## 已批准主测试：V5-H20，2026-10-03

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

主配置V5-H20用于检查恢复算子、global64协议下的联合训练。要判断HIER是否有利，至少再有同V5配置的B0：只令λ_H=0，其余训练预算、初始化、数据顺序/增强、BN、输入和验证方式一致。V4 B0不能替代这个对照。

本次授权运行H20，剩余单卡运行B32稳定性诊断；没有授权再自动排队一个完整双卡B0。H20保存新的第20轮prefix，以便后续获准时接续同协议B0。prefix包含模型、optimizer/scheduler、split与sampler身份及各rank RNG状态；不能使用旧V4 prefix。

B32使用同一8856/984划分、随机初始化seed22、恢复原part复写与两次BN更新、c1/D256、CE/intra与模型优化器。每轮将训练集无放回打乱，训练前200个完整batch32，即6400次抽取；保持用户200step预算，因此并非原版完整遍历（完整drop-last为276步）。负part仍为本batch的flip，允许原随机batch中同类负配对并记录其比例。它与H20的global64/类别均匀有放回协议不同，不能作为严格HIER增益对照。详见[17：启动与监测配置](17_V5_TRAINING_START_2026-10-03.md)。

这组B0恢复HyCoRe核心算子但采用新batch/预算/验证划分，不应称作原训练协议逐参数复现。先区分baseline恢复、协议适配和inter增量三个效果。

### 运行记录与监测

- 始终记录实际唯一训练ID、逐类抽取次数/作为i的次数、任意角色覆盖、有效/碰撞/self-k三元组及两张图的合格anchor。
- 记录μ/ν深度与半径、shadow cap和贴近原数值球边界比例、proxy径向移动、安全project、非有限值及裁剪前norm。
- 第21、40、100、160、200轮首batch测CE/intra/HIER共享参数梯度，分别报告欧氏特征层和Mobius层；不能只用μ偏导或全模型平均范数判断协同。
- 每10轮额外做固定无增强train-eval，和训练模式accuracy、validation一起观察BN/train-eval差距；不借test选模型。
- 保存best/last及每20轮checkpoint、两个optimizer/scheduler、各rank RNG、sampler/config/commit、GPU和时间；新目录运行，保留旧结果。

当前A/200step理论单轮总体唯一覆盖66.08%，最大类chair约32.98%；继续遵守200step，不把它误写成80–90%覆盖。K20的参考合格率约60.39%，全部40类曾合格，仍有条件覆盖偏差及增强关系不稳定风险。

8步K20参考检测均有限，峰值28.73GiB/卡，净step中位数.6366秒；单臂200×200净step外推约7.07小时，实际总时间更长。短测不是完整训练性能或时间保证。

主训练、B32入口及恢复/验证/保存循环已完成启动检查和200轮生产运行。**启动记录见17，最终结果见19，原始身份以服务器manifest为准。**

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
