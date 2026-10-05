# HIER机制诊断：分离选择、BN和实际更新

日期：2026-10-06。用户授权主B0独立运行后持续小规模机制诊断和据证据执行新计划。
本文件只记录可核验聚合与方法；完整checkpoint、特征、逐draw记录及服务器路径不入库。

主任务V6同协议balanced64 B0已完成首个续训轮e21的200步、984例validation及可恢复checkpoint验收。
独立后台继续e22–300；诊断仅CPU及额外GPU3。生产身份与限制见[29](29_V6_BALANCED_B0_AND_MECHANISM_LAUNCH_2026-10-06.md)。

## 首批有限面板

- 冻结e300 clean-eval缓存：4个源式balanced64计划、K20/T50、FP32 CPU重放，0次更新。
  proxy pair端点自身重合占非collision draw的44.32175%；proxy triple为0.73134%。
  sample pair比triple深87.3936%，两者均比whole浅；这与历史train约98%的比例
  使用了不同BN/输入/随机核，不能当成历史重放的矛盾。
- e300两个条件热点面板，同输入/negative/增强，顺序两个local32。
  eval的sampleST共享参数范数为固定祖先hinge的12.49/17.94倍；
  trainBN为2.64/18.53倍。trainBN的sampleST/base范数为0.2804/0.3134。
  eval与trainBN不能互替。ST包含选择路径，固定祖先仅留下同一scalar hinge的直接导数。
- eval→trainBN同triplet同Gumbel的pair agreement约83–85%，triple约49–51%；
  还需相同输入重复Gumbel基线，才能判断几何变动超过多少固有随机性。

以上不证明长期塌缩、形态层次或分类收益。局部mu偏导、共享参数梯度、裁剪后带动量
的实际位移分别报告；proxy项没有直接whole梯度，仍可能间接改变后续sample祖先。

## 扩展计划与验收

1. e99/e200/e300各16个相同自然balanced64计划，冻结CPU重放，无更新；
   e300另做100步冻结whole代理更新，6个目标/mining臂及2个零梯度控制。
   checkpoint AdamW m/v会携带旧目标历史；终端公共评估与零梯度控制一起解释。
2. GPU3的e300自然16batch：eval、trainBN和local32重分组，无参数更新。
   拆开自然覆盖与首批强制热点覆盖。
3. 固定热点输入重复Gumbel8次，比较同输入噪声与跨BN同seed差异。
4. e300热点配对真实一步：base、base+.5sampleST、base+.5sample_fixed。
   恢复完整RSGD、有效LR、norm1；主读出固定旧BN与clean1024，隔离参数作用。
   每批起点重置，proxy固定，HIER分叉首步权重.5，不新增warmup。

每个新入口先CPU gate与独立审查，再提交、同步并在新目录运行。记录源码commit/script hash、
源完整checkpoint身份、输入hash、seed、时间、GPU及所有状态还原检查；原结果不改。
如来源/finite/state gate失败，保存失败身份、修复入口，再使用新目录。

## 扩展结果：冻结CPU与共享参数

e99/e200/e300各16个相同自然balanced64计划已完成；计划hash一致，全部来源/finite/不变检查通过。
1024个样本位置含26个dresser位置，底半径1%位置仅13/6/9，不能推广成全体底1%。

| 同协议冻结CPU重放 | e99 | e200 | e300 |
|---|---:|---:|---:|
| sample非collision pair比triple深 | 94.01% | 83.91% | 87.53% |
| active sample子集pair比triple深 | 58.90% | 50.61% | 51.68% |
| proxy pair端点自身重合 | 17.26% | 16.52% | 44.30% |
| dresser加权sample径向偏导均值 | +0.02915 | +0.10841 | −0.05951 |

e300 dresser的25/26局部sample偏导为负（欧氏mu梯度下降指向外）；不能直接解释长期或共享参数更新。
各项约束的是双曲两点距离hinge，并未直接约束原点深序；active draw是困难子集。
proxy端点作祖先候选是源规则允许的行为。eligible/draw增加不等于loss权重增加，损失仍取draw均值。

e300自然16batch的ST/base共享参数范数比中位数为eval2.045、trainBN0.486、重分组0.169；
ST/fixed中位数为15.88、4.80、3.54。trainBN的ST/base范围0.094–2.696，作用强度依batch变化。
eval径向margin违反均值99.02%，trainBN39.26%；同输入的BN模式影响几何/损失。
同triplet同Gumbel的eval→trainBN pair/triple agreement均值83.17%/52.69%；
还需结合重复Gumbel噪声、trainBN→重分组直接对照解释。所有VJP重放误差为0。

聚合数据：[CPU16](diagnostics/results/20261006_proxy_cpu16_aggregate.json)、
[whole自然16](diagnostics/results/20261006_whole_natural16_aggregate.json)。

## e300真实配对一步（条件热点，4个独立ID）

两个固定热点batch完成6次副本RSGD更新；源模型/proxy/HDF/checkpoint不变。
完整继承H20历史动量和有效LR，三臂norm1均未触发裁剪；各臂无更新读出的mu/logits误差均0。
主读出固定旧BN和clean1024。base是当前一步移除sample项，不能称从头B0。

| 热点平均原点深度变化 | batch0 | batch1 |
|---|---:|---:|
| base实际一步 | +0.003945 | +0.035233 |
| base+.5sampleST实际一步 | +0.001218 | +0.031929 |
| ST相对base的增量 | −0.002727 | −0.003304 |
| fixed祖先相对base的增量 | −0.001177 | −0.000446 |

三臂总体仍向外；HIER的增量在本面板减少向外幅度，完整选择路径影响大于固定祖先直接项。
局部eval-cache偏导、source_train共享参数作用、带动量实际位移不是同一个量。
4个热点不代表所有dresser或长训练；下一步同面板e20/e99/e200与e300自然16batch，检验时间和覆盖。

聚合数据：[真实一步](diagnostics/results/20261006_one_step_e300_hotspot2_aggregate.json)。
科学静态图（PNG/PDF均已逐张视觉检查）：
[CPU分工与偏导](artifacts/hier_mechanism_figures_preview_20261006_v1/proxy_cpu_mechanisms.png)、
[BN与共享参数](artifacts/hier_mechanism_figures_preview_20261006_v1/whole_bn_parameter_mechanisms.png)、
[实际总更新与增量](artifacts/hier_mechanism_figures_preview_20261006_v1/one_step_total_and_increment.png)。

重复Gumbel8次的热点面板：相同输入但独立噪声的pair identity agreement约11%，
triple约2.6–4.8%；跨BN同seed的pair约83–84%、triple约50–54%。
两种比较不相减作因果分量，28个两两噪声比较也不是28个独立重复。
trainBN热点sample μ-gradient的noise-vector RMS约0.132/0.094，mean-gradient norm约0.096/0.089。
因此进一步在source_train自然mining固定三元组下，只改Gumbel seed做8次真实一步，
同时扩展A/B增强：固定clean crop物理成员与anchor的一组、源式自然重crop的一组。
各组分别保留crop成员身份，避免相同中心整数掩盖不同物理点。

GPU噪声探针已完成；顺序管理器在前一进程退出后瞬时util未归零时停止了后续启动，
未打断任何进程。重新确认GPU3完全空闲后在新目录单独执行一步；失败管理器及成功噪声结果保留。

最终test不参与诊断或选配置。
