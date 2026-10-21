# V9 代理贴 cap：损失与径向作用审查

日期：2026-10-21。范围：读取生产源码与本地固定官方参考；执行少量 CPU 双精度几何计算。未使用 GPU，未启动训练，未改生产实现。本文区分数学上存在的机制和已经测得的 V9 历史因果；后者仍需早期真实更新诊断。

## 1. 实际运行的损失没有显式径向祖先顺序

V9 的 [train.py:29](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v9/train.py:29) 导入 V8 HIERLoss；V8 的 [hier_loss.py:8](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py:8) 再使用 V6 `ghhc_loss`。V8 的 sample 与 proxy 分项分别使用 whole×proxy 和 proxy×proxy 距离矩阵，最后相加，见 [hier_loss.py:57](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py:57)。

对一个有效三元组，令 a 为本次 pair 代理、b 为 triple 代理，损失是：

\[
\ell=[d(i,a)-d(i,b)+m]_+ +[d(j,a)-d(j,b)+m]_+
      +[d(k,b)-d(k,a)+m]_+ .
\]

这与本地官方固定参考的 [losses.py:43](D:/Hycore/.codex-local/references/HIER-CVPR23/hier/losses.py:43) 至 60 行一致；现用计算见 [V6 hier_loss.py:90](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v6/hier_loss.py:90)。其中没有 `d0(a)<d0(i)`、`d0(b)<d0(a)`、深度惩罚、固定目标深度或原点锚点。minimax 是**选择代理的代价**，不是另加一个需要最小化的 minmax 距离项。因此它鼓励的是这些相对距离关系；“祖先”的名字不能自动赋予原点深度顺序。

固定壳映射见 [V9 geometry.py:60](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v9/geometry.py:60)。它移除了 whole 原生半径对 HIER 排序/梯度的直接贡献，但没有改变上述三项损失，也没有约束代理半径。

## 2. 统一 whole 深度并不意味着理想代理要保持 0.9 倍深度

即使暂时把代理理解为真实几何中心，理想深度仍由角度决定。两个深度同为 H、夹角为 Δ 的点，其测地中点/二点 minimax 中心的原点深度为

\[
h_*={1\over\sqrt c}\operatorname{atanh}
 [\tanh(\sqrt c H)\cos(\Delta/2)] .
\]

在 c=0.1、H=4.6 下：

| whole 两点夹角 Δ | 几何中点深度 h* |
|---:|---:|
| 5° | 4.58629 |
| 10° | 4.54582 |
| 20° | 4.39292 |
| 30° | 4.16498 |
| 60° | 3.27757 |
| 90° | 2.36561 |

所以“whole 成壳”只固定了 H，仍未固定 Δ。若角度很接近，正确的中心本来就可以很接近外壳；初始典型深度 4.14 甚至比该中心更浅。不能以初始层差存在来推导梯度应一直维持该层差。V9 的高斯初始化只是典型模长约为 0.9×2.3，并非逐代理精确固定模长，见 [V8 hier_loss.py:30](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py:30)。

更直接地，对于 whole 深度 H、代理深度 h、夹角 θ，理想双曲距离满足

\[
{\partial d\over\partial h}=
{\cosh(\sqrt cH)\sinh(\sqrt ch)-\sinh(\sqrt cH)\cosh(\sqrt ch)\cos\theta
 \over\sinh(\sqrt c d)}.
\]

在 H=4.6、h=4.14、c=.1，临界角为 15.4833°。夹角小于该值时，**减小吸引距离会把代理向外推**；夹角更大时，同一吸引项会把它向内拉。排斥项符号相反。因此“正则中有吸引”也不等于“代理必然内移”。这些公式为无数值稳定项的理想几何；实际 V9 采用带 1e−5 稳定项的官方距离，故真实梯度应直接测算。

## 3. 固定选择下，也有明确的向外动力

令三个 hinge 的激活指示为 α、β、γ，`g_xa=∂d(x,a)/∂h_a`。固定本次 a/b 选择后：

\[
\partial_{h_a}\ell=\alpha g_{ia}+\beta g_{ja}-\gamma g_{ka},\qquad
\partial_{h_b}\ell=-\alpha g_{ib}-\beta g_{jb}+\gamma g_{kb}.
\]

以通常的角度较大区域、各距离径向导数约为 +1 举例：若三项都激活，a 的直接导数约 +1（梯度下降向内），b 约 −1（梯度下降向外）。a 需要靠近 i/j 并远离 k，b 需要远离 i/j 并靠近 k；这种相反角色并非一个统一的“祖先内缩”目标。激活组合、角度及每个代理担任两类角色的频率决定合力。

CPU 调用实际 `official_distance`，whole H=4.6，代理 h=4.14、夹角90°时，对单个排斥项 `−d(whole,proxy)` 测得无 GS 的 `∂L/∂h = −0.88633964`；带实际 `to_ball` GS 后，切空间径向导数仍为负（二维代理在 y 轴时 y 导数 −0.19873661）。GS 是正因子，不能将向外符号改为向内。实现见 [V8 geometry.py:28](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py:28)。这说明向外作用确实可由 HIER 本身产生，但不是它在 V9 长期占主导的证据。

## 4. Hard Gumbel 的路由梯度是第二种独立机制

当前 hard Gumbel 使用 straight-through：前向 one-hot，反向 softmax，见 [V6 hier_loss.py:49](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v6/hier_loss.py:49)。不能只按被选代理的直接拉/推解释总梯度。

在固定本次 hinge 激活状态时，定义候选 l 的

\[
v_l=\alpha d(i,l)+\beta d(j,l)-\gamma d(k,l).
\]

局部损失为 `(a−b)·v+常数`。若 pair/triple 各自的 soft Gumbel 概率为 q（两者使用独立噪声），从选择路径得到：

\[
{\partial\ell\over\partial C^{pair}_l}=-{q_l\over\tau}(v_l-E_qv),\quad
{\partial\ell\over\partial C^{triple}_l}=+{q_l\over\tau}(v_l-E_qv),
\]

其中 `C_pair=max(di,dj)`，`C_triple=max(di,dj,dk)`。一个不适合担任某角色的候选，可以通过**增加自己的选择代价来降低被选概率**；增大距离是可用路径，向外移动可能实现它。这与直接改善被选代理几何的梯度不同。tau=.1 还在该路径导数中引入 1/tau=10 的因子，但 softmax 权重、距离结构和梯度抵消共同决定实际量级，不能只凭该因子断定它占主导。

因此应测 `g_fullST`、同一次硬选择的 `g_fixed`、两者差 `g_routing`，而不是把总梯度称为“几何祖先吸引”。

## 5. 一个“全部在壳上且损失为零”的明确反例

CPU FP64、实际官方距离函数，c=.1，三个 whole 与两个候选代理深度全为4.6，二维方向分别为：

- whole i/j/k：−5°、5°、160°；
- pair 候选 a：0°；triple 候选 b：100°。

距离矩阵为：

| | a | b |
|---|---:|---:|
| i | .557835 | 7.920340 |
| j | .557835 | 7.530903 |
| k | 9.113170 | 5.629339 |

确定性 minimax pair 选 a，triple 选 b；margin=.1 的三个 hinge 均为0。二者都没有比 whole 浅，却满足相对距离目标。随机 Gumbel 仍可能偶然换选；该反例的意义是**存在符合正常 minimax 选择的壳上零 hinge 几何**，不是宣称随机目标所有抽样均为零，也不是当前模型的实测三元组。

## 6. 对早期趋势诊断的具体建议

最低限度在 e0、更新1/5/20/100和逐轮记录，避免只在径向映射已饱和后测量。

1. 使用同一次真实训练的关系、Gumbel 噪声、a/b 和 hinge 激活，分别计算 sample/proxy 的 full-ST、fixed-choice 及 routing residual；已有 [fixed_ancestor_loss](D:/Hycore/docs/research/diagnostics/hier_whole_mechanism_v1.py:150) 可作参考。
2. 每个分项记录切空间 `dot(g,u)` 的均值/分位数/正负比例以及角向范数，按未越cap/已越cap分组。负导数代表普通梯度下降有向外作用，但不能直接代替实际 AdamW 位移。
3. proxy×proxy 矩阵中同一参数兼任端点和祖先候选。固定相同 triplets/选择/噪声，将距离改为 `dist(P.detach(),P)` 与 `dist(P,P.detach())`，可分解候选角色与端点角色，两部分导数应相加恢复完整导数。改写仅限诊断副图，不能改变真实训练。
4. 记录实际 `Δ||t||`、径向位移和正交位移，并与上面的当前梯度配对；历史动量和逐坐标自适应需要另行拆解。
5. 记录选中 pair/triple 的角度覆盖范围、whole→代理夹角，以及激活/角色频率，检验“局部方向很窄因而吸引向外”和“路由逃避选择”的条件是否真的发生。

官方/现用 cap 都在前向映射中执行，不强制存储参数或 Adam 状态回到内部，见 [V8 hier_loss.py:44](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py:44)、[V8 geometry.py:41](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py:41)。超cap后径向雅可比接近零，晚期的当前径向梯度小不能反证早期没有外移作用。由于实现中的 `+1e−5`，精确数学上有极小径向残差，表述宜为近似饱和。

## 判断边界

已确认：V9 没有显式祖先深度顺序；固定壳和0.9初始化不能保证内层；实际三项和 ST 路由都允许向外作用；壳上零 hinge 配置确实存在。

尚未确认：在真实 V9 前5–10轮中，sample/proxy、直接/routing、局部窄方向、AdamW历史状态/逐坐标预条件各占多少。需要同一步分解证据，不能把数学可行机制直接写成长期因果结论。
