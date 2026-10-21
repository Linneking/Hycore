# V9嵌入审查的纯NumPy分析器

输入为私有导出目录中的`export_manifest.json`及各观察点NPZ。运行：

```text
python analyze.py --export-root PRIVATE_EXPORT_DIRECTORY --out ANONYMOUS_OUTPUT_DIRECTORY
```

每条`observations`记录至少包含`key,label,stage,epoch,c_native,kind,cache`；`cache`必须相对导出目录。可附`checkpoint_sha256,c_hier,native_cap_tangent,hier_cap_tangent,hier_whole_cap_tangent,usage_source_epoch`。没有额外cap的原版/V9原生分支必须把`native_cap_tangent`设为null。HIER代理默认cap为2.3。

同一模型各阶段保留相同`key`，以`key__stage`构成唯一`observation_id`。所有CSV同时保存模型key、阶段及观察点ID；绘图数组以前缀`key__stage__`区分，避免best/e300覆盖。

NPZ必需键为`native_mu,labels,canonical_ids`。所有观察点必须包含相同规范对象身份和标签，行顺序可以不同。HIER记录另含`hier_mu,proxy_ball,proxy_tangent`；可附`nn4,nn4_equal,nn4_angle`（代理×4，值是当前NPZ样本行索引）、`sample_usage,proxy_usage`（每代理all_draws的pair+triple选择次数，只能来自该权重对应轮次）、`sample_effective_usage,proxy_effective_usage`（同轮active_noncollision的pair+triple计数）。缺失的激活统计保持null。

球坐标半径、归一化半径和原点双曲深度分别为：

$$
r=\|x\|,\qquad q=\sqrt c\,r,\qquad d_0(x)=\frac{2}{\sqrt c}\operatorname{atanh}(q).
$$

切空间cap为2.3对应深度上限4.6；贴cap容差为双曲深度0.001。代理保存参数模长超过cap的统计使用2.3+0.000001。深浅代理以**相同HIER域whole中位深度**为参照，容差0.0001。原生c1 whole和HIER c0.1视图不混合计算深浅关系。统计标准差采用总体标准差ddof=0。

方向集中度为单位方向均值的范数：

$$
R=\left\|\frac1n\sum_i\frac{x_i}{\|x_i\|}\right\|.
$$

类别方向中心先对该类别所有样本的单位方向取均值，再归一化；代理的最近类别取方向余弦最大者。这里只表示相对方向亲近，不能单凭最近类别解释为语义绑定。使用组由同轮sample/proxy选择计数定义；all_draws使用不等于hinge为正，active_noncollision使用代表非碰撞且hinge为正，但仍不能保证代理径向梯度非零。这些计数跨越整轮更新，并非在冻结的轮末缓存上重新选择。

NN4覆盖是512×4检索槽中的不同样本数；最大重复四例忽略四例顺序。最浅1%使用向上取整对象数；边界并列按输入行稳定排序，并额外报告并列数。同壳情况下这项不能解释径向偏置，`bottom_1pct_meaningful_radial_order`会标为false。

输出`summary.json,stats.csv,class_stats.csv,proxy_stats.csv,hotspots.csv,plot_data.npz`。热点表仅含排序、类别、次数、深度，不含样本ID或原始行号；绘图NPZ仅包含一维深度、归一化半径、代理参数模长和使用计数，禁止导出完整嵌入和样本身份。
