# 双卡 V5 训练修改目录与审查稿

日期：2026-10-03。状态：**待用户审查，未启动新 GPU 任务。**

本文是实施目录和验收标准，训练代码与测试脚本尚未实现。本轮先交 V4 总结和这份目录；审查后按目录实现、进行双卡和 K 测试，再确定完整训练配置。原 HyCoRe、legacy inter 和 V4 路径保留。

## 1. 200 step 能改善覆盖，但达不到大类单轮八九成

以下“官方式采样”专指 **HIER 发布代码的 UniqueClassSampler**：每步均匀无放回选类别，类内有放回抽实例。它与原 HyCoRe 的训练实例打乱遍历不同。

拟定 batch：两卡各 16 个不同类、每类 2 例；两卡类别不重叠，全局每步 32 / 40 类、64 例。以 V4 固定训练划分为例，N=8856，最大类 chair=800。

对有 n_c 个实例的类，每个指定实例单步未出现的概率是：

\[
q_c=(1-32/40)+(32/40)(1-1/n_c)^2.
\]

若不同 step 独立抽取，则 T step 的期望唯一覆盖为：

\[
E[U_c(T)]=n_c(1-q_c^T),\qquad E[D_c(T)]=2(32/40)T=1.6T.
\]

全集覆盖是按各类 n_c 加权的唯一实例覆盖，不能用“总 draw / N”代替。

| 每轮 step | 总 draw | 全集期望唯一数 | 全集覆盖 | chair800 覆盖 |
|---:|---:|---:|---:|---:|
| 138 | 8832 | 4890.3 | 55.22% | 24.13% |
| 200 | 12800 | 5852.4 | 66.08% | 32.98% |
| 400 | 25600 | 7423.6 | 83.83% | 55.08% |
| 805 | 51520 | 8431.0 | 95.20% | 80.02% |
| 1151 | 73664 | 8686.9 | 98.09% | 90.00% |

还有一个不依赖有放回概率的硬限制：**200 step × 每类每步最多 2 例 = 400 个位置，chair 有 800 例，即使每步都选中且完全不重复也至多覆盖 50%。** 均匀选类时，期望只有 320 个位置；无放回队列也只能期望覆盖约 40%。

因此，200 step 可以作为计算预算，但不能标成“大类单轮覆盖八九成”。增至 400 step 的有放回方案也仍只有 chair55.08%，不能与全覆盖队列混为一谈。

### 待审的采样选择

| 方案 | 类内 / 类间安排 | 200 step 的大类覆盖 | 与 HIER 源码差异 | 用途 |
|---|---|---:|---|---|
| A：严格发布采样 | 均匀选32类，类内有放回2例 | chair 期望32.98% | 采样规则一致；每轮200步是新预算 | 优先核对原关系机制；本稿建议用它做首轮诊断 |
| B：均匀选类 + 每轮新队列 | 均匀选32类，每轮从新打乱的类内队列开始，跨步无放回直到耗尽 | chair 期望40% | 改变类内有放回规则 | fresh队列下400 / 450步，chair期望约80% / 90%，需独立协议对照 |
| C：整轮全覆盖计划 | 为每类安排足够二例块，填充到32类×2，并打乱完整batch计划 | 200步不可能保证chair覆盖 | 改变选类概率与实例采样 | 至少400步；合理排程可覆盖全部8856例，但25600draw中有16744次重复 |

方案 C 的步数下界为 `max(max_c ceil(n_c/2), ceil(sum_c ceil(n_c/2)/32))=400`。实现必须检查这个下界能否被具体排程达到，不能只用公式宣称 sampler 正确。完整 batch 计划随机化，避免复现 V4 的类大小时序偏置。

另一个变体是将200步队列连续保留到下一轮，以提高跨轮累计覆盖。但某一轮跨过队列耗尽重洗时，旧周期尾部和新周期头部可能重复，不能再承诺该轮的unique数等于draw数。上表B的40% / 80% / 90%估计限定每轮从fresh队列开始；连续队列变体需按覆盖周期和实际每轮unique分别报告。

**建议**：先用 A、200 step 检查双卡和 K；完整训练前由用户决定是否接受其大类覆盖，或改用 B / C。若“大类单轮80–90%”优先于类内有放回忠实性，B 的400–450步更符合目标，但工作量和训练协议都会变化。

## 2. 双卡数据和梯度路径

| 阶段 | rank0 | rank1 | 全局行为 |
|---|---|---|---|
| 类别选择 | 全局32类中的前16类 | 后16类 | 同一全局类别列表，卡间无重复类 |
| 实例输入 | 16类×2＝32例，类内两行相邻 | 同左 | 全局32类×2＝64；不再全行shuffle破坏flip对应 |
| HyCoRe | 本卡采whole/child，child→whole前向，CE+intra | 同左 | 本卡child.flip(0)为负；每卡检查负标签异类 |
| HIER sample | 接收可微gather后的64行whole | 同左 | 同一全局候选图、triplet与Gumbel随机流；不另取图像/点云batch |
| HIER proxy | 同一组层次代理 | 同一组层次代理 | 初始化广播，梯度同步，每步参数一致 |
| 参数更新 | DDP同步backbone / head | 同左 | 正确处理全局HIER损失和本卡base均值的缩放 |

两卡仅分担网络前向；HIER 对同一个全局64例建立关系。CE 和 intra 的对象仍是本卡32个实例。新类别块使 intra flip 负样本严格异类，这比原 HyCoRe 的随机batch更强，是明确的协议变化。

普通 BN 保持本卡32例统计；part和whole均更新。将两个encoder调用放入同一个外层 DDP forward，避免多次外层前向带来的 reducer / buffer 行为差异。明确 checkpoint 保存哪一 rank 的 BN buffers，并诊断两卡统计；默认 buffer 广播不能称为 SyncBN。

### 不能照搬的分布式细节

官方 HIER 使用 `no_grad all_gather` 再恢复本卡有梯度切片；与 DDP 梯度平均结合，会使 backbone 的全局 HIER 梯度比完整全局目标缩小 world_size 倍。官方 HIERLoss 代理也没有像 backbone 一样被 DDP 包装。

V5 使用可微 gather 的跨卡求和反向，并对代理显式同步。两卡各计算同一个全局 HIER 目标时，gather 的反向求和与 DDP 平均应得到正确尺度；不能再随手乘 / 除 world_size。验收用固定64行输入、固定三元组/Gumbel、关闭随机层或固定eval统计的单卡/双卡梯度对照。普通训练BN本来按本卡32统计，不要求它等价于单卡64统计。

## 3. 保留 HyCoRe 的哪些实现

记 whole 为 w_i，child 为 p_i，flip得到的负child为 n_i。保留：

\[
L_{base}=L_{CE,\epsilon=.2}+0.01R_{contr}+0.01R_{hier},
\]

\[
R_{contr}=\frac1B\sum_i[d_1(w_i,p_i)-d_1(w_i,n_i)+4]_+,
\]

\[
R_{hier}=\frac1B\sum_i[d_1(0,p_i)-d_1(0,w_i)+1000/N_p]_+.
\]

| 保留项 | 实施核对 |
|---|---|
| whole800–1024点，child200–600点 | 使用原kNN随机中心采样，核对随机数和点索引 |
| part覆写whole | 原切片视图赋值语义保留；分类和HIER都读覆写后的whole嵌入 |
| child→whole顺序 | 两次共享encoder前向；BN均处于train并更新统计 |
| 每次每个实例一个随机child | 不改成额外多part目标，不引入proxy–part约束 |
| 原CE、intra、分类头和c=1 | 标签平滑.2；两个intra各.01；保留双曲分类头 |
| 原模型优化 | RSGD：LR.1、minLR.005、momentum.9、WD2e-4；模型norm裁剪1 |
| 原FPS行为 | 暂保留固定512 anchor；动态FPS另列单因素诊断 |

200 epoch、固定验证划分、双卡类别块及global64是新协议。源码默认300 epoch和原随机batch不能在这些改变后仍称为完整复现。

## 4. 按 HIER 发布代码建立 ijk

参考固定提交 `3986a744a1a54fd357e307d1cb3f2e81910b9ffc` 的 [HIERLoss](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/losses.py)。以下是发布代码语义，论文的标签无关近邻描述与其中同类加1有差别。[论文§3.2](https://arxiv.org/html/2212.14258v3#S3.SS2)

\[
s_{ij}=\exp(-d_1(w_i,w_j))+\mathbf1[y_i=y_j].
\]

1. 在包含 self 的64列中取topK，得到有向邻接 A；选图使用detach后的距离。
2. M=(A+A^T)/2，再令 M_ii=−1。
3. J_i={j:M_ij=1}；若 |J_i|<2，跳过该anchor。
4. 源码负集合 N_i={k:M_ik<1}，包括单向近邻和完全非近邻。
5. 每个合格anchor独立、有放回抽 j∈J_i 和 k∈N_i，共T次；发布forward用T=50。

**源码细节**：因为 M_ii=−1，源码允许 k=i。这是发布实现与“严格三个不同实例”之间的差异。拟保留 `source_exact` 模式作为参照，同时提供单独声明的 `exclude_self_negative` 修正模式；首轮源码核对不静默排除self。无合格anchor时返回保持梯度连接的零损失，避免源码空拼接崩溃，记录skip原因。

在二例类别块中，另一同类位置通常会被同类加1优先纳入并互惠，因此合格anchor还需要至少一个跨类互惠j。除self外，k应为异类。浮点极端情况可能出现分数并列：同类的exp项很小而使加1后舍入为1，同时近零异类距离使exp项舍入为1。脚本必须检查同类被选中率和tie，避免把数值假设当成保证。

K6 / 8 / 10 使用官方计数口径，分别最多包含5 / 7 / 9个非self有向邻居；其中一个通常为同类，其余为跨类候选。互惠集合是双方topK交集，其大小可以更小。

## 5. 一个 inter 与原 proxy 正则的形式

对样本或proxy的三元组(i,j,k)，令代理集合为P：

\[
C_{ij}(p)=\max\{d(i,p),d(j,p)\},\quad
C_{ijk}(p)=\max\{d(i,p),d(j,p),d(k,p)\}.
\]

分别以 `hard=True` 的两个独立 straight-through Gumbel-softmax，在−C_ij/τ和−C_ijk/τ上选择p_ij、p_ijk。保留原三个hinge：

\[
\ell_{ijk}=[d(i,p_{ij})-d(i,p_{ijk})+m]_+
+[d(j,p_{ij})-d(j,p_{ijk})+m]_+
+[d(k,p_{ijk})-d(k,p_{ij})+m]_+.
\]

当p_ij=p_ijk，整条draw归零；**包括这些零值一起求平均**。sample和proxy–proxy各自求均值，再相加：

\[
L_{total}=L_{base}+\lambda_H(R_{sample}+R_{proxy}).
\]

不再拆in / out，不另对part加HIER。proxy图不使用类别加1，其余关系选择和hinge核对原代码；首版T_sample=T_proxy=50作为来源明确的起点。

## 6. 忠实还原和必须声明的适配

“忠实还原”分为 HyCoRe 算子、HIER 核心关系/损失，以及融合所需适配。两份源码的曲率、优化器和BN前提不同，无法把各项设置同时原样复制到共享whole。

| 项目 | HIER发布CUB脚本 / 源码 | V5建议或待审 |
|---|---|---|
| 主目标 | 归一化embedding上的Proxy Anchor | 保留HyCoRe标签平滑CE；不新增类别绑定PA代理 |
| 维度 / 层次代理数 | D512 / P512 | D256维持HyCoRe；建议首轮P512作源码容量参照，P256为之后独立容量对照 |
| 曲率 | c=.1 | 共享空间c=1，用户已指定 |
| margin / tau | .1 / .1 | 暂列单位换算候选.0316227766 / .0316227766；原生c1的.1 / .1为对照，不与K同时扫 |
| sample / proxy K | 同一个K | 源码参照共用K；sample K筛查可固定proxy K20，但要明示拆分适配，最后核对共用K参照 |
| proxy初始化 | 随机高斯切空间初始化 | 使用随机初始化，广播两卡；不默认继承V4 k-means |
| proxy优化 | AdamW组LR=50×base；CUB global180时proxy=.005，源码还乘global batch/180 | global64的源码预算参照为proxy LR=.00177778×schedule、WD=.01；单独AdamW，不把HyCoRe .1机械乘50变成5 |
| 模型优化 | AdamW及脚本特定schedule | 保留HyCoRe RSGD；完整学习率/schedule和总step作为适配登记 |
| HIER总权重 | parser默认1，CUB脚本.5 | 首轮需选择单一λ_H；.5为发布脚本参照，1为论文参照；V4 .03不自动沿用 |
| warmup | CUB1epoch冻结预训练body LR，HIER仍参与 | 不把V4关闭HIER20epoch+ramp20说成官方warmup；是否保留base预热由短诊断后审查 |
| 梯度裁剪 | 逐元素截到[−10,10] | HyCoRe模型norm1保持；代理按源码元素裁剪；分开记录触发率 |
| BN | 冻结预训练BN运行统计，affine仍训练 | 保留HyCoRe训练BN；普通BN每卡32，非SyncBN |
| 数值精度 | CUB使用AMP | 首次正确性/几何诊断float32；AMP需要单独核对而非静默更换 |

官方proxy初始切向量尺度为`clip_r*.9/sqrt(D)`，并经同一ToPoincare截断/映射；c1单位换算与原生参数两种方案的proxy尺度也应分别登记。官方AdamW在AMP下eps=1e-4，非AMP下eps=1e-8；CUB每5epoch乘.5的schedule不可悄悄替换成V4 proxy cosine。学习率、schedule、WD与反向一起通过兼容性诊断后，再固定K实验配置。

P512与“90类”没有固定比例关系：90是CUB一次global batch选择的类别数，CUB训练类别总数为100。按40/90缩放代理数缺少论文依据；P数量在K筛查中必须固定。

上述参数是审查候选，不是已选定的完整训练配置。短诊断可以淘汰数值不兼容组合，但不能把源码出处当作已验证的点云最优参数。

### 曲率、输出约束与反向必须先过门槛

对c0=.1的对应缩放坐标，`d1(sqrt(c0)x,sqrt(c0)y)=sqrt(c0)d_c0(x,y)`；这解释.1→.0316227766的距离单位换算，**不证明优化轨迹等价**。

官方映射前切空间截断半径2.3，若按几何等比例搬到c1，半径为.7273，径向深度最大1.45465；HyCoRe要求whole比part深至少1000/N_p，最小也为1.6667。共享whole套用这个约束会使radial hinge无法为零。直接在c1用原数字2.3也不能容纳所有part大小对应的margin（最大5）。

建议保持HyCoRe原有数值投影，**不增加共享whole的HIER输出cap**；proxy映射的初始化/截断单独记录，不影响intra。

官方自定义反向在公共whole输出上乘 `(1−c||z||²)²/4`，会同时改变主目标和HIER梯度；proxy也使用。若只在HIER支路乘，或采用V4普通反向，均是适配。实现前用固定前向和固定Gumbel分别测普通、HIER支路预条件和共享预条件，报告CE/intra/inter的梯度范数、余弦和半径；不能用“系数小于1”代替整体兼容性结论。

## 7. 待实现的文件目录

新路径 `inter_hierarchy_MN40/hier_proxy_scratch_v5/`，不覆盖旧文件：

```text
hier_proxy_scratch_v5/
  train.py                  # 双卡入口、阶段控制、验证、manifest
  joint_model.py            # 一次外层forward内完成part、whole、代理路径
  base_protocol.py          # 原HyCoRe采样视图/前向/intra/CE的明确封装
  sampler.py                # 全局选类、rank划分；source/queue模式明确分开
  distributed.py            # 可微gather、proxy同步、归约尺度
  relations.py              # 源码口径topK、互惠池、跳过、j/k重抽
  hier_loss.py              # Gumbel祖先、3 hinge、碰撞mask、两个均值
  diagnose_batch.py         # 无GPU采样/覆盖/类别时序审计
  sweep_k.py                # 固定64例特征缓存上的K和T诊断
  diagnose_geometry.py      # 固定batch的c1/cap/反向兼容性诊断
  check_distributed.py      # 固定输入单卡/双卡loss及梯度核对
  configs/                  # 显式完整配置；不靠隐藏默认值
  tests/                    # 源码数值核对、empty关系池、rank/梯度同步
```

训练manifest记录提交、解析后的完整配置、seed、数据划分和实例ID、GPU、起始时间、初始化/检查点SHA和诊断状态。权重、完整日志、特征缓存及主机详情不提交。

checkpoint必须包含模型、层次代理、两类优化器、scheduler、sampler状态和各rank随机状态，才能验证断点恢复；不能只保存backbone而遗漏层次代理。若采用base预热，需核对DDP未使用代理参数的处理。

## 8. K 测试脚本的输入、统计和选择规则

### 固定输入，避免把batch变化误当成K效果

1. 先运行`diagnose_batch.py`：验证每卡16类×2、global32类无重复、flip异类，输出多个seed和多个epoch的逐类唯一覆盖及类别时序。
2. 从训练数据提取多个实际batch的whole特征；记录采样/增强/part复写、BN模式、特征位置、检查点SHA和batch实例ID。随机初始化与base短训检查点都取样，避免只凭一个阶段定K。
3. `sweep_k.py`重用完全相同的缓存、三元组/Gumbel随机种子。主候选K=6,8,10；K=3,20只作旧设置与源码设置参照。选图按官方包含self口径。
4. 固定P、proxy K、反向、权重等其它设置。先按T50；T32/64可在关系筛查阶段作为预算参照，不同时用于分类性能矩阵。

计划接口示例（审查后实现，当前不是可执行命令）：

```text
diagnose_batch.py --mode source --world-size 2 --classes-per-rank 16
                  --instances-per-class 2 --steps 200 --seeds 22 42 2026
sweep_k.py --features <training-feature-cache> --ks 3 6 8 10 20
           --sample-t 50 --proxy-k 20 --negative-mode source_exact
           --rng-seeds 22 42 2026 --output <new-diagnostic-directory>
```

### 每个batch必须输出的统计

| 指标 | 分母 / 定义 |
|---|---|
| 合格anchor、skip | `eligible/64`；skip按正候选不足/负池空分别列出 |
| 互惠度数 | 非self的J_i大小分布；同类位置入选率与分数tie |
| 实例触达 | draw中作为i/j/k出现过的唯一batch位置/64；另按唯一数据ID去重 |
| j类别组成 | 跨类j draw/全部draw；按anchor统计，不能只看总量 |
| k类别与self | 同类k、异类k、k=i分开报告 |
| 正/负pair覆盖 | 每个anchor的distinct j/|J_i|、distinct k/|N_i|；macro与合并计数均报告 |
| triplet覆盖 | distinct `(i,j,k)` / `sum_eligible |J_i||N_i|` |
| 重抽重复 | 1−distinct triplets/全部draw；与实例采样重复分开 |
| 祖先碰撞 | `p_ij=p_ijk` draw/全部draw；sample/proxy分别列出 |
| 约束激活/满足 | 非碰撞draw中任一hinge>0的比例及各hinge；碰撞不能当满足 |
| 计算成本 | draws、耗时、峰值内存；空anchor图时零损失/零梯度路径 |

先按候选图可靠性排除极差K，再用完全匹配的短训验证推理稳定性和验证OA。不能仅按anchor覆盖最高选K；K越大越可能放入弱近邻。离线统计不能单独宣布最终K最优，当前也不预先指定6/8/10中哪一个。

## 9. 从审查到完整训练的顺序与门槛

| 阶段 | 工作 | 通过条件 |
|---|---|---|
| R：本轮 | V4教训、覆盖率核算、修改目录 | 用户审查本文及采样/适配选择 |
| C：CPU和源码核对 | 采样、互惠池、hinge、碰撞均值、原CE/intra | 固定输入与来源实现一致；声明例外有单独测试 |
| D：双卡smoke | 小步数前向/反向、proxy一致性、DDP尺度 | loss/梯度有限；两rank参数同步；单/双卡梯度核对通过 |
| B：基础协议诊断 | HIER关闭、恢复覆写/BN、训练集推理与val | 没有未解释的严重train/eval差异、类别失败或数值退化；单项问题继续隔离 |
| K：筛查 | K6/8/10离线图与匹配短训 | 覆盖、梯度、推理稳定性和val共同支持候选 |
| F：完整训练 | 同协议H0与选中K的HIER，固定完整预算 | 启动前更新具体配置与矩阵供审查；不把smoke成功当稳定证明 |

短诊断总step上限和是否需要共享base前缀应写入批准后的配置。短训无法保证200epoch后期稳定，因此完整训练持续记录训练集clean eval、逐类准确率、BN和关系诊断；200epoch固定预算不以test做早停。

初期所有组固定seed22和同一划分，后续稳定后再加seed；正式test只在完整训练结束后评估验证选中的模型一次。200step/epoch×200epoch是40000次更新、2560000次全局实例draw；V4是56200次更新、1798400次draw，二者既不是同更新数，也不是同实例计算量。

每个双卡任务需要两块完全空闲GPU，启动前检查`nvidia-smi`。若只有三块空闲GPU，不能同时运行三组双卡训练；先做CPU/缓存筛查，双卡任务顺序安排。运行与结果写入新目录，不修改旧权重或日志。

## 10. 请重点审查的决定

1. **覆盖与预算**：先按A、200step作诊断可行；完整训练接受大类约33%单轮期望覆盖，还是优先B的400–450step队列？
2. **源码细节**：首轮以`source_exact`保留k=i并统计；排除self列为独立修正，而非混入K比较。
3. **融合适配**：接受HyCoRe覆写/BN/intra/CE恢复；HIER关系/hinge按发布代码核对；c1的截断、反向、margin/tau和λ先做兼容性诊断再确定，不能直接宣称完整原样还原。
4. **参数可比性**：K筛查期间代理数量、初始化、优化和其它系数固定；完整矩阵需包含同协议HIER关闭基线。

相关入口：[V4经验教训](12_V4_LESSONS_2026-10-03.md)、[下一轮实验计划与执行门槛](04_NEXT_EXPERIMENT_PLAN.md)。
