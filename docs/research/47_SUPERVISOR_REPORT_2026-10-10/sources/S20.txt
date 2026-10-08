# V5-B0-32-CAP200 与原版 HyCoRe 的一致性核对

命名提示：本文展示名称按[统一实验命名规范](33_EXPERIMENT_NAMING_CONVENTION_2026-10-06.md)更新；原名称可查映射。文件名、运行目录、代码字段与既有结果身份保留。

日期：2026-10-03。核对已完成的 V5-B0-32-CAP200；本次没有修改训练代码，没有启动新训练，没有改写已有结果。

## 1. 结论与核对范围

**V5-B0-32-CAP200 不是只改变了每轮 200 step 的原版 HyCoRe。** 它恢复了原版网络、part 复写、BN 和主要损失/优化算子，但还改变了训练集划分、总轮数及 cosine 周期、随机数流程、workers 和模型选择规则。“恢复核心算子”不能等同于“完整复刻原训练设置”。

核对三种身份，避免混淆：

1. 发布源码：本地官方 HyCoRe 审计快照及 `classification_ModelNet40/`。两个入口的训练逻辑相同，本地入口仅恢复了进度条打印；`hutil.py`、`data.py` 相同。
2. 用户保存的历史原版实验 ORIG-B0-32-S4780：`Hype_PointNet-Offv_pointmlp_hycore_var-4780`，保存的 `args.txt` 为 seed=4780、300 epoch、workers=8、batch=32；best 为 epoch229、test OA=94.044%。历史目录没有完整的软件环境与源码快照，因此不能承诺逐位重放历史训练。
3. V5-B0-32-CAP200：生产代码 `6e13ff9f59ad3bec8f7c3db857e6558e37473c03`，运行配置与源码均已核对。当前 HEAD 的相关训练/模型文件与该生产代码没有差异。

## 2. 影响训练协议的差异

| 项目 | 原版源码 / 历史原版实验 | V5-B0-32-CAP200 实际设置 | 影响及恢复要求 |
|---|---|---|---|
| 训练数据 | 全部 9840 个官方训练实例，无独立验证划分 | 8856 train / 984 val；验证实例从不参与训练 | 不只是每轮少用实例，还永久留出了 10%。恢复原版训练数据需要使用完整 9840 |
| 单轮抽样 | 普通 `shuffle=True`、`drop_last=True`；batch32 时 307 step、9824 个不同实例，尾部丢 16 个 | 打乱 8856，取前 200 step、6400 个不同实例，余下 2456 本轮未使用 | V5-B0-32-CAP200 每轮为无放回，不是类别均匀抽样，也不是 4类×8。完整原版应为 307 step |
| 总预算与调度 | 默认/历史均为 300 epoch；`CosineAnnealingLR(T_max=300)` | 200 epoch；`T_max=200` | 即使初始 LR=.1、最低 LR=.005 相同，衰减轨迹仍不同；总更新数 92100 对 40000 |
| seed | 源码默认 22；保存的历史实验实际 4780 | 22 | 对照源码默认时相同；对照保存的 94.044% 实验时不同 |
| 随机数流程 | 启动时设 seed 一次；DataLoader、裁剪点数与中心继续消耗随机数流 | 每轮和每步重新设 seed；sampler/loader 使用私有 generator；点数另用私有 Python RNG | 同 seed 不等价于同随机过程；严格复刻应直接复用原版 loader 与 `get_children_np` |
| workers | 默认/历史均为 8 | 2 | 除吞吐外，worker 的 NumPy 增强随机数与样本分配也不同，不能只视为速度参数 |
| cuDNN benchmark | 原入口最终设为 True，deterministic=True | `seed_all` 持续设置 benchmark=False，deterministic=True | 算法选择/速度设置不同；不能据此断言精度退化原因 |
| 选模与评估 | 每轮评估官方 test，按严格更高 test OA 保存 best | 每轮固定 val，最高 val OA、同 OA 取较低 CE；结束后只测一次 test | 两个 best 成绩的选择依据不同，不能直接作为同协议性能差异 |

原版加载器见 `classification_ModelNet40/main_pointmlp_hycore.py:130`；优化器与调度器见 `:136`、`:140`；V5 数据划分与模型配置见 `inter_hierarchy_MN40/hier_proxy_scratch_v5/train_b32.py:311`，随机数/加载器见 `:368`、`:379`。

调度差异可以用已完成 epoch 数 e 表示：

\[
lr(e)=0.005+0.095\frac{1+\cos(\pi e/T)}{2},
\qquad T_{original}=300,\quad T_{B32}=200.
\]

在第 200 轮结束这个边界，原版 300 轮调度约为 .02875，V5 为 .005。这不是只少训练 100 轮，而是在前 200 轮内就已采用不同学习率。历史原版曾有早期续训，其具体日志也应结合该续训身份读取。

### 裁剪随机数流程为何不完全相同

原版每步依次消耗同一 Python RNG：

```text
抽 whole 点数 → 抽 32 个 whole 中心 → 抽 part 点数 → 抽 32 个 part 中心
```

V5-B0-32-CAP200 则是：

```text
seed s 重设全局 RNG
以同一个 s 初始化私有 RNG，连续抽 whole/part 两个点数
全局 RNG 抽 32 个 whole 中心，再抽 32 个 part 中心
```

其中 `s=22+1000003*epoch+1009*step`。两条 RNG 用同一个 seed 从开头分别取值，点数和中心的随机序列有耦合的可能；不能称为原版随机过程。whole 点数仍在 800–1024、part 仍在 200–600，裁剪的 kNN 和复写操作相同。当前证据没有证明这种随机数改写导致了精度下降。

V5-B0-32-CAP200 200 轮累计在第 7 轮已经用过全部 8856 个训练实例。因此“每轮未覆盖”与“整个训练永久漏用”是不同问题；永久不参与训练的是独立留出的 984 个验证实例。

## 3. 已确认一致的核心训练设置

| 项目 | 核对结论 |
|---|---|
| 网络 | 两个 `models/` 目录逐文件一致；同一个 `Hype_pointMLP()`、40 类分类器、双曲 embedding D=256；此模型没有启用 Dropout |
| 曲率 | 原版本来就是 c=1；当前 Geoopt 构造默认 `learnable=False`。V5 的显式 freeze 没有把原版可学习曲率改成固定曲率 |
| 初始权重 | 都由网络随机初始化；V5-B0-32-CAP200 未加载旧权重、teacher 或预训练网络。具体随机身份受 seed 与环境影响 |
| 点云输入与增强 | 前 1024 点、逐轴缩放 [2/3,3/2]、平移 [-.2,.2]、打乱点序；均未启用 jitter、旋转或 random dropout。公式一致，随机流不同 |
| whole/part | whole 800–1024 点，part 200–600 点；每实例每步一个新 part；part 仍通过视图复写 whole，whole 的输入确实受影响 |
| 前向与 BN | 先 part embedding，再 whole 分类；普通训练 BN 每步更新两次。V5-B0-32-CAP200 没有冻结 BN、没有改成 whole-only BN |
| FPS | 原首层仍请求 512 个 FPS 中心，part 不足 512 时保留原重复索引行为 |
| CE | eps=.2，非目标类别各分配 `.2/(40-1)`，对样本取均值；与原版 `cal_loss` 一致 |
| intra | 正对为 whole/自己的 part；负 part 为本 batch 的 `child.flip(0)`，不额外排除同类碰撞；margin 与系数相同 |
| 优化器 | RiemannianSGD，初始 LR=.1、momentum=.9、WD=2e-4；当前 Geoopt 下均无 Nesterov，均无额外 stabilize |
| 梯度裁剪 | 全模型 global norm 上限 1；单卡 V5-B0-32-CAP200 没有 hierarchy proxy 混入裁剪 |
| HIER | V5-B0-32-CAP200 没有 HIER 损失、hierarchy proxy、HIER 特有的截断或自定义反向 hook。原网络本身的 Geoopt 数值投影保持一致 |

共同的数学目标为：

\[
L=L_{CE}^{\varepsilon=.2}+0.01R_{contr}+0.01R_{hier},
\]

\[
R_{contr}=\frac1{32}\sum_i
\left[d_1(\mu_i,\nu_i)-d_1(\mu_i,\nu_{31-i})+4\right]_+,
\]

\[
R_{hier}=\frac1{32}\sum_i
\left[-d_1(0,\mu_i)+d_1(0,\nu_i)+1000/N_{part}\right]_+.
\]

这里索引 i 从 0 开始，d1 是 c=1 的 Poincare 距离。原版随机 batch 的翻转负对也可能同类；V5-B0-32-CAP200 保留了这个行为，并增加碰撞比例监测。

## 4. 严格逐算子复刻仍需说明的差异

这些差异应披露，但不能直接解释为训练结果差的原因。

1. **hinge 的零点反向**：原版 `hutil.py:68–69` 使用 `torch.max(x, zeros_like(x))`；V5 `base_protocol.py:97–98` 使用 `F.relu(x)`。函数值一致，但 x 恰为 0 时，前者对 x 的导数是 .5，后者是 0。若要求逐算子一致，应直接复用原函数。
2. **损失浮点加法顺序**：原版 `CE + .01*Rcontr + .01*Rhier`，V5-B0-32-CAP200 `CE + .01*(Rcontr+Rhier)`。实数数学相同，FP32 的舍入顺序不同。
3. **距离类与设备处理**：原版创建带 dim 的项目 `PoincareBall`，V5 创建 Geoopt `PoincareBall` 并迁移到 embedding 的设备/dtype。项目类未覆盖 `dist/dist0`，两者继承同一实现；严格代码复用时仍应恢复原函数。
4. **数据分片顺序**：原版直接使用 `glob.glob` 返回顺序，V5 显式 sorted。样本内容不变，但索引与随机遍历的对应关系可能改变。
5. **封装与内存布局**：原版套单卡 DataParallel，V5 直接单卡 forward wrapper；V5 显式 contiguous、pin_memory 与 non_blocking。当前原版只暴露一张卡，不存在把原版 32 再拆成多卡小 batch 的差异。
6. **评估统计**：原版 CE 是 batch mean 再平均，V5 按样本加权；最后非完整 batch 的 CE 权重不同。OA/AA 定义一致，原版打印/返回有三位小数舍入，V5 保留更多精度；best 还存在 OA 同分时是否按 CE 更新的差别。
7. **监测与恢复**：V5 每 10 轮额外 clean-train eval，保存/恢复 RNG，检查 BN 不更新；梯度分组审计复用现有图，不增加训练前向或累计 `.grad`。这些监测增加时间，不新增损失。V5 保存完整 scheduler/RNG 等状态，原入口续训保存/恢复内容较少。本轮 V5-B0-32-CAP200 没有 resume。
8. **运行环境**：当前 V5-B0-32-CAP200 为 Python3.9.25、PyTorch2.8.0+cu128、Geoopt0.5.1、NumPy1.26.4；历史实验未保存完整环境身份，无法确认所有软件版本一致。

## 5. 要做到原版一致，需要恢复什么

训练端最低要求是直接复用原版 Dataset/DataLoader、模型构造、`get_children_np`、`cal_loss`、`hype_triplet_losses`、RiemannianSGD 和训练函数，使用完整 9840 个训练实例、batch32、307 step/epoch，以及 300 轮对应的 cosine 周期和 workers=8；避免额外每轮/每步 reseed 与私有裁剪点数 RNG。

- 若目标是 **源码默认配置**，seed=22。
- 若目标是 **对齐用户保存的 94.044% 实验配置**，seed=4780；还要承认历史续训与缺失环境记录对逐位重放的限制。

原版每轮 test-best 与当前 val-best 是另一项必须明示的差别。项目的新实验规则要求验证集选模，不能在审计中偷偷把 test-best 恢复成新的选择规则。若保留验证流程，就应准确命名为“原版训练算子 + 验证选模协议”，不能称为整个实验流程完全相同。

当前 V5-B0-32-CAP200 的非 smoke 入口还硬性限制 `200 epoch × 200 step`（`train_b32.py:126`），所以恢复原版预算不能只传入新 CLI 参数。需要单独的版本化原版复刻入口与清楚的评估身份，待用户审阅下一轮计划后使用新目录运行。此次仅完成差异确认。
