# V6 启动记录 — 2026-10-04

## 授权与目的

用户已批准两项300轮任务，要求启动测试通过后退出对话，次日自行检查。代码分支为`codex/hier-v6-selfk300`。V5与原HyCoRe源码保持可复现，新增入口在`inter_hierarchy_MN40/hier_proxy_scratch_v6/`。

| 项目 | V6 H20 | B0_original_B32 |
|---|---|---|
| GPU | 双卡，各local32 | 单卡batch32 |
| 数据 | 固定8856train/984val | 原全部9840train |
| 抽样 | 32类×2，全局64，类别均匀、类内有放回 | 原DataLoader随机打乱、drop_last |
| 预算 | 300epoch×200step | 300epoch×307step |
| 随机种子 | 22 | 原默认22 |
| HIER | K20/T50/P512/D256/c1，排除两图索引self-k | 无 |
| 基础损失 | 原CE/intra、part复写、两次普通BN更新、FPS512 | 直接调用原源码 |
| 模型LR | .1→.005，cosine300 | 原.1→.005，cosine300 |
| 代理LR | .01→.0005，cosine300 | 无 |
| HIER权重 | 前20轮0，后280轮.5 | 0 |
| 模型选择 | validation OA，平局较低CE；最终test一次 | 原逐轮test与best-test |
| workers | 2/rank | 原默认8 |

H20从头随机初始化，不接续V5第200轮或V4权重。第21轮不重启学习率；300轮周期下首次代理更新约.00989620。额外HIER切空间截断与反向hook仍关闭，保留shadow监测与原数值球投影。样本索引i=k被排除；类内有放回得到的相同实例ID处于不同位置时，仍按原采样协议处理并记录。

V6同时改变self-k规则和总训练预算/调度周期。B0_original_B32是用户要求的源码复现工程基线，其数据、batch和选模协议不同，不能单独隔离HIER目标的影响。保留原test选择是本次源码默认复现要求，不改变H20验证选模协议。

## 监测与保存

H20沿用V5的唯一实例/逐类覆盖、anchor与三元组覆盖、重复与碰撞、self-k、CE/intra/inter、梯度裁剪前后、代理双卡一致性、whole/part/proxy球半径、shadow截断/反向因素、非有限值、每轮耗时、显存和实际LR。每轮验证与last/best完整checkpoint，每20轮存档；每10轮无增强train-eval。参数及跨增强结构probe轮次为21/40/100/160/200/240/300。

新增结构监测将祖先选择次数与代理三元组i/j/k角色分开，记录所选pair/triple祖先的深度及相对端点位置，代理径向/角向更新与sample/proxy分项实际梯度。额外检查不更新optimizer，并恢复生产BN/RNG。

V6参数审计对输入副本做额外前向，防止诊断中的part复写修改随后真实训练的输入；真实训练保留原复写。这属于监测正确性修复。跨增强probe使用固定轮次首batch的相同ID/位置，eval状态的两份独立scale/shift/shuffle whole1024视图，记录互惠图与确定性minimax祖先的保持率；后者明确不是训练hard Gumbel的逐次复现。

原版B32包装器保留训练实现及随机流，增加运行身份、失败状态和逐轮保存/被动监测。生产步数必须核实为307，不再采用V5的200step截断。

## V5 第200轮代理近邻点云脚本

入口：`inter_hierarchy_MN40/visualize_v5_proxy_neighbors.py`。读取完整`checkpoint_epoch_200.pth`，不使用validation最优权重。模型在eval状态提取8856个训练实例的无增强whole1024特征；代理使用V5保存的c1映射。

“合格代理”定义为：在全部512代理的K20互惠图中至少有两个非self正候选，并有负候选。以固定seed从合格集合随机选择十个，不按类别纯度挑选；每个代理展示c1双曲距离最近的四个独立实例ID。top4仅为展示数量。资格不等于被训练选作祖先或真实形态共同祖先；V5未保存实际祖先ID，不能伪造使用统计。

输出无需网络的旋转点云HTML、静态PNG、样本ID/类别/距离及权重hash/数据划分/选择规则元数据。原始点云、权重与完整输出留在结果目录，不提交Git。

## 启动验收

启动前验收代码提交为`08edb3e`。服务器CPU关系/监测6项与可视化7项均通过。双卡H20完成2轮×2step，覆盖base-only与joint阶段，并实际保存参数梯度审计、双增强probe及完整checkpoint。原版B32完成2轮×2train/test batch，验证原9840/307step loader与默认配置；smoke未用于生产初始化。

V5epoch200脚本已完成全部8856训练实例推理，当前proxy图165/512合格，固定随机抽出10个代理并生成40个点云的离线HTML与PNG，静态布局已目视检查。该结果不用于V6选模或调参。

当前状态：验收通过，准备正式独立进程启动。正式启动后补充GPU/PID、代码提交和首轮checkpoint确认；原始运行身份以新目录manifest与launcher_state为准。所有既有结果目录保持不变。
