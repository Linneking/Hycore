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

执行结果待追加；最终test不参与诊断或选配置。
