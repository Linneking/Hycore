# 四组whole／代理嵌入审查：统计口径与证据边界

日期：2026-10-24。主组已由用户确认：TS13（循环64，HIER关闭）、TS14（循环64，旧HIER A0）、本轮严格B05＋HIER A1、本轮原生HyCoRe＋HIER A1。每组分别分析best与e300；已有可靠缓存可作为辅助消融。本文是审查方法，不是未运行项目的结果。两服务器GPU满载时不占用其GPU、不更改模型／BN／训练状态；优先复用缓存，缺失缓存由主任务另行确认可用提取路径。

## 1. 先锁定测量对象、数据与几何空间

| 对象 | 原生whole空间 | HIER whole视图 | 代理 |
|---|---|---|---|
| TS13 | c0.1，cap2.3，基础GS | 不构造 | **不存在，标N/A** |
| TS14 | c0.1，cap2.3，基础GS | c0.1，保方向固定深度4.6 | c0.1，512×256，A0 |
| 新B05＋A1 | c0.1，cap2.3，基础GS | 与TS14同样的固定壳分叉 | c0.1，512×256，A1 |
| 新原生＋A1 | c1，无新增出口cap／GS | c0.1，保方向固定深度4.6 | c0.1，512×256，A1 |

TS14与新B05都在cap之后、基础GS之前构造视图；新原生保留c1主路径，另构造c0.1视图。不能将原生c1 whole与c0.1代理直接按深度大小宣称祖先内外次序。对原生组，“代理在whole内层”的直接参照是其**同c0.1、固定4.6的HIER视图**。对于B05，也应区分实际native whole与被重新归一化的HIER视图：代理浅于4.6并不保证浅于每一个native whole。[V9几何域与映射](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v9/geometry.py:17)、[B05分叉](D:/Hycore/inter_hierarchy_MN40/hier_proxy_cycle_v9c/geometry.py:40)、[B05固定壳](D:/Hycore/inter_hierarchy_MN40/hier_proxy_cycle_v9c/geometry.py:84)。

主图固定使用同一批9840个官方训练对象，clean first1024、eval、各自保存的历史BN、无数据增强和crop覆盖。以canonical ID对齐，不以不同训练分片次序下的行号碰巧相同代替对象身份。best由既有test选模确定，本审查不重新选best、不用几何图挑权重。69号导出代码已实现固定输入SHA、标签SHA、checkpoint SHA、模型/BN不变检查，可复用这些约束。[输入与导出域](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/collect.py:129)、[缓存与身份](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/collect.py:199)、[跨缓存ID校验](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:150)。

每份缓存需记录：运行名、实际epoch、best/e300身份、checkpoint SHA、数据/标签/ID SHA、native c与HIER c、D/N/P、输入点数、eval/BN状态、提取脚本与算子身份。历史原版的raw e229/e299对应完成230/300，不能沿用行名直接当现代一起轮次。[旧身份验收](D:/Hycore/docs/research/69_V9_EMBEDDING_REVIEW_2026-10-21/acceptance.json)。

## 2. 现有缓存schema能做什么

69号完整导出NPZ包含：

- 必需的`native_mu[N,D]`、`labels[N]`、`canonical_ids[N]`；它们足够计算原生深度、类内/类间角度、类中心及代表类别对。
- HIER组额外包含`hier_mu[N,D]`、`proxy_ball[P,D]`、`proxy_tangent[P,D]`；足够比较实际映射代理深度、raw参数越cap、方向冗余及样本绑定。
- 可选`nn4`、`nn4_equal`、`nn4_angle`，以及sample/proxy和active_noncollision的每代理usage数组。可用于复核距离热点、径向敏感性和真实训练使用，缺失usage不影响静态角度/深度审查，但不能补造“训练使用率”。[导出字段](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/collect.py:199)、[HIER/NN4字段](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/collect.py:202)、[usage提取](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/collect.py:52)。

公开的`stats.csv`、`class_stats.csv`和摘要JSON是聚合；仅有均值、分位数或方向集中度，不能重建40×40类中心角度和具体类对的完整分布。需要原始NPZ，不能从聚合均值反推原始分布。69号分析中的`plot`数组保存了部分逐对象深度，可用于已有径向图，但不能替代包含D维方向的完整cache。[聚合写出与方向读取](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:172)。

69号验收清单已明确存在过以下全量clean/eval导出身份：TS13/cycle_off的best e223、e300与e120；TS14/cycle_hier**只有e120**；辅助有TS06/orig_c01_gs的best e184/e300、历史原版、S22 V9-B64和V9-HIER A0，以及V8-HIER。不能将TS14 e120改标为best e212或e300；新两条A1的83/84结果是检查点/指标审查，本身不是9840对象的D维whole缓存。实际NPZ是否仍本机可读，必须由缓存盘点确认，验收清单不等于文件当前存在。[69原始验收清单](D:/Hycore/docs/research/69_V9_EMBEDDING_REVIEW_2026-10-21/acceptance.json)。

保存的`actual_training_epoch_XXX.pth`是首个global64、训练模式、更新前的增强与crop前向，含不同对象ID和当批whole_count。它可补充真实训练场景，但不等价于全9840 clean/eval，不能混在同一条总体CDF中；不得用一批两例/类的夹角代表整个类别。[真实训练快照](D:/Hycore/inter_hierarchy_MN40/hier_proxy_cycle_v9c/train.py:401)。

## 3. 深度百分比分布图

所有计算使用声明并验证的正曲率幅值c：

\[
q=\sqrt c\|x\|,\quad h_c(x)=\frac{2}{\sqrt c}\operatorname{artanh}(q),\qquad 0\le q<1.
\]

保留h（原点双曲距离）与q（相对球边界半径）两种读出。h属于各自声明曲率的度量，q是无量纲相对边界位置；两者不能互称。[实际c-aware径向计算](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:43)、[官方映射/深度](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/geometry.py:41)。

建议主图如下：

1. **native whole百分比直方图**：best/e300分列，c0.1与c1分面；横轴h，纵轴`100×bin_count/N`。所有曲线的同一图面使用相同bin边界，建议间隔0.2；显示完整范围，必须记录超出范围的数量，不能悄悄裁掉深样本。whole每份各自归一化为100%，不是按模型总样本数混合。
2. **HIER whole与proxy百分比直方图**：统一c0.1，HIER view用虚线标记4.6，代理单独归一化为100%（P=512）。将固定壳标为“构造视图”，不能称whole学出了薄壳。TS13代理面板显示N/A，不能画成深度0的512个代理。
3. **ECDF**：`100×count(h≤t)/population_size`，用阶梯函数，作为bin无关复核。给出P1/P5/P25/P50/P75/P95/P99、最小/最大/均值/SD；直方图与CDF都来自同一保存深度数组。[既有分位口径](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:23)。

bin边界必须注明左闭右开、最终bin右闭；4.6附近的浮点误差单独按容差统计，不能因4.599992落到另一侧bin就制造层差。分别报告：`raw_norm>2.3`的参数越cap数、`|h−4.6|≤1e−3`的输出贴壳比例；这两者不是完全相同事件。实际HIER实现只在前向cap，保存参数不做更新后投影。[raw监测定义](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py:42)、[已有贴壳容差](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:19)。

除全量对象微平均外，补一份40类等权的whole深度分布，防止大类主导；每类自身的百分比图按本类N_c归一化。proxy不是40类均分对象，不能无证据将512代理强行等分成每类。

## 4. 类内、类间方向角：不只报均值

使用原点方向`u=x/||x||`、`θ_ij=acos(clip(u_i·u_j,−1,1))×180/π`。零或极小模长方向标未定义并报告数量，不填0°。原点出发的径向映射保留方向，故此项不受样本半径影响；不要改成直接用未归一化mu点积。[已有方向函数](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:35)、[固定视图的方向保持](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v9/geometry.py:60)。

- **类内样本对角度**：每类所有`i<j`不重复对，给N_c、pair_count、P5/P25/P50/P75/P95、均值与SD；另给样本到本类方向中心的角度分布，清楚标为另一个量。
- **类间样本对角度**：每个不同类别对a<b计算N_a×N_b全部对的分布，输出780行类对CSV；既给按所有样本对等权的micro汇总，也给按780类对等权的macro汇总。类内macro按40类等权。类数不均衡时两者不能混名。
- **40×40类中心角度**：先求`m_c=mean(u_i)`，再归一化`v_c=m_c/||m_c||`，矩阵为`acos(v_a·v_b)`。同时输出`R_c=||m_c||`，中心接近零时标未定义；不要先均值原始mu，否则深样本会重新获得更大方向权重。[实际类方向中心定义](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:66)。
- **角度分离诊断**：可以补样本的最近异类角度与典型同类角度差，但明确这是方向几何读出，不是当前分类器的决策边界或准确率。

所有同类/类间分布按固定对象ID、同一cache口径计算，尽量全对精确统计。若内存需分块，分块只改变实现，不改变pair集合；若必须抽样，应记录固定seed、样本对ID/数量和加权方式，不能只抽每类一个例子。比较模型内部的类对关系；不同模型允许整体旋转，跨模型直接计算“同一类中心向量之间的角度”不等于类语义漂移。

### 代表类别直接比较的预先规则

主面板固定展示语义上容易解释的类别对，不按最后显著性挑选：chair–stool、chair–sofa、desk–table、dresser–night_stand、bottle–vase、bowl–cup；另以airplane–bowl等明显不同形态对作参照。名称必须来自已核验的ModelNet40类名表，缺类别就标记而非替换成有利案例。每对同时展示：两个类的类内CDF、跨类样本对CDF、中心夹角、P5/P50/P95，以及分离是否改善。

再加探索性面板：以**TS13 best**为固定参考，选其最接近的5个类中心对，并在全部模型复用同一名单。最大变化类对只能另列为“探索性”，不得替代固定面板。若使用分类混淆挑案例，应以共同的既有基准混淆或预先规则确定，不对每个模型各选最有利类对。

## 5. “代理有意义分层”至少需要四类证据

### 5.1 径向分离只是第一层证据

报告proxy深度分位数、方差、range、贴cap比例，以及相对HIER view深度4.6的间隔。它证明代理是否具有深度自由度，不能单独证明ancestor语义。对于同c0.1的native whole，还可报告真实被关联端点的深度差，但不把相对全体whole中位数的间隔当成每条边都成立。[已有proxy分组定义](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:236)。

### 5.2 祖先深度次序要带容差、碰撞和端点身份

实际训练监测中的`noncollision`只表示pair祖先与triple祖先索引不同；它不保证祖先与端点不同。应同时报告全部抽样、noncollision，以及存在正hinge的active_noncollision三个分母。[真实定义](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v6/structure_monitor.py:47)、[c0.1单位变换](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py:13)。

需要统计：pair/triple祖先碰撞率；pair祖先等于i/j的比例、triple祖先等于i/j/k的比例；`h_pair−h_triple`；`min(h_i,h_j)−h_pair`；`min(h_i,h_j,h_k)−h_triple`。每种gap都分“正序／近似等深／逆序”，主容差预先设为`δ_h=1e−3`，补`1e−4`与`1e−2`敏感性。历史83的`positive_fraction`来自`gap>0`，新结果另列legacy零容差以便对照，不能把同壳约1e−7的浮点差当稳定层级。[历史严格零符号计数](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v6/structure_monitor.py:25)。

若从缓存重新做确定性pair/triple minimax argmin，必须称“冻结几何读出”，不是训练Gumbel重放。建议使用同一组预先固定对象／triplet，另报各模型自身关系挖掘；跨模型共用triplet不意味着它在每个模型的KNN图都合法。[确定性固定读出](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v6/structure_monitor.py:187)。

只有静态深度数组而无祖先索引时，不能从深度直方图反推祖先次序。若建立代理关系图，再报告self-loop、循环强连通分量、根/连通分量、深度严格递减边比例；可视化只画部分高频边时也保留全图指标，避免筛边后看起来自动成树。

### 5.3 代理是否被使用、是否覆盖多个不同对象

分开统计：训练真实sample/proxy祖先usage（来自该checkpoint自己轮次）与冻结几何最近代理分配。每类输出活跃P、零使用P、usage熵、`exp(H)`有效代理数、top1/top10%选择份额。active_noncollision使用与仅被抽到也分开，不把“选中过”直接等于“学出有效层次”。[usage计数合并](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v6/structure_monitor.py:107)、[熵与有效数](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:55)。

同时看每个proxy最近4／最近k个不同whole的类别组成、不同对象覆盖、重复quartet、最热对象占比。分别用实际双曲距离、whole等深距离、纯角度三种读出，区分深浅导致的热点与方向覆盖。K与分母固定，深度几乎全同壳时不解释“最浅1%”排名。[已有热点/去径向读出](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/analyze.py:86)、[导出的三种NN4](D:/Hycore/docs/research/diagnostics/v9_embedding_review_20261021/collect.py:219)。

跨模型同编号proxy不是同语义节点。需要跨模型配对时，应另做明确标注的方向对齐与匹配探索，不能凭同ID将不同模型的proxy轨迹连线后称为祖先保持。

### 5.4 语义／形态意义不能只由训练标签循环证明

输出proxy×40类绑定矩阵、每proxy类别熵、最近类中心夹角与同类/跨类子集构成。标签置换（保持类数）和保持半径的随机方向可作为CPU零模型；这些是可选冻结几何对照，不改变训练。单类纯度高也可能只是类内方向极紧，不能单独证明跨类别树。HIER样本关系本身加入同标签相似度偏置，因此用同标签聚集验证该目标不属于独立语义信号。[标签偏置与真实关系构造](D:/Hycore/inter_hierarchy_MN40/hier_proxy_scratch_v8/hier_loss.py:60)。

“家具／容器”等预先声明的类别组，可作人可读案例，但应标探索性语义组织。若没有独立part属性、形态标注或外部语义树，仅凭分类标签、半径和NN近邻，最终结论应停留在“几何层级/绑定现象”，不能写成已经验证了真实形态祖先。

## 6. 建议交付与最低验收

主四组best/e300：深度百分比直方图＋ECDF；40类类内角度分位表；780个类对分位表；40×40中心角热图及相对TS13的变化；固定代表类别对直接CDF面板；proxy深度/使用/绑定/稳健祖先次序表。辅助旧消融保持实际身份、epoch、seed和提取口径，不能补成匹配对照。

发布前检查：所有深度有限且在声明球内；hist每人口合计100%；CDF单调且终点100%；每类计数和pair_count正确；矩阵对称且中心角对角为0；单位方向与角余弦有限；HIER view方向与native一致、深度4.6在容差内；无HIER的代理列为N/A；所有跨模型对象ID/标签/输入哈希一致；本机若用了替代PointNet算子，先通过主任务的算子/特征一致性验收再合并到旧cache比较。暂未通过提取一致性的数据只能独立标记，不能冒充已有同算子全量结果。

本文只规定统计口径和读取证据，未连接服务器或执行模型/GPU操作。
