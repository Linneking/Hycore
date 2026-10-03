# V5 已授权诊断计划（不启动主训练）

状态：2026-10-03检测已完成。结果见 [15：双卡诊断结果](15_V5_DIAGNOSTIC_RESULTS_2026-10-03.md)。主训练未启动。

用户于2026-10-03审查后授权本轮检测，选择方案A、200step/epoch；只执行CPU和有上限的双卡测试，不执行完整epoch训练或主训练。

## 固定选择

- 两卡各16类×2，全局32类×2＝64；类内有放回，块内相邻。
- whole、part、logits可微gather，CE/intra/inter都在global64计算；intra negative改为全局child.flip(0)。每卡32仅是encoder和普通BN的本地规模。
- 恢复原whole/part视图覆写、child→whole两次BN更新；共享每步whole/part采样点数，保留原FPS512行为并统计child<512。
- c=1、D256、随机切空间代理，暂保留P512。margin=.1、tau=.1保留源码数字，不再使用V4的.0316228。
- HIER额外切空间截断与Riemannian反向hook关闭；原HyCoRe数值球投影和模型norm1梯度裁剪保留。只计数“若启用截断将命中多少”，不实际施加。
- HIER source reciprocal规则：K含self、同类加1，至少2个非self互惠j，T50有放回；首轮K8、sample/proxy同K。K6/8/10及3/20参照用同一特征缓存筛查。
- 主模型RSGD LR.1、momentum.9、WD2e-4；代理AdamW LR.01、WD.01、eps1e-8。LR.01是诊断候选，约为global64源码预算参照.0017778的5.625倍。
- λ_H=.5作为CUB发布脚本的诊断参照，非完整训练的已选权重；不叠加V4 warmup/ramp。

## 检测与预算

1. CPU采样模拟：3seed×3epoch，比较138/200步；逐类unique、重复、类别时序、卡间不重类和globalflip异类。
2. CPU源逻辑/梯度检查：原CE/intra/alias/BN；HIER三个hinge和碰撞均值；fixed64单进程与Gloo双进程损失、参数及嵌入梯度尺度。
3. 完全空闲双GPU：默认16个joint optimizer step，硬上限24；无epoch循环，无test或validation选择。
4. reference无梯度probe默认4batch、硬上限6：V4 epoch20仅作为已有表示参考，读取原文件但不更新它；不当作V5初始化或忠实基线。
5. 同feature K筛查；i=k在相同Gumbel draw上的非碰撞、loss和梯度贡献。若pair/triple祖先不同，i=k的第一、第三hinge之和至少2margin，不把源码行为自动当成无影响。
6. 固定detached feature的代理LR短诊断：.0017778/.005/.01/.02，各12个proxy-only update；只检查数值、碰撞、更新幅度，不用来证明分类收益。
7. 时间统计排除前三个joint step，报告最大rank耗时；用短测median估算138/200步。200相对138步位置预算增加44.93%，不是增加200个epoch。

whole的shadow cap统计用最终mu的logmap0半径作反事实检查，不等于直接观测Mobius层前特征截断。将这一区别及原数值投影与新增HIER cap分开记录。

新脚本在`inter_hierarchy_MN40/hier_proxy_scratch_v5/`。每个服务器诊断写新目录、记录commit/config/seed/GPU/时间/split/checkpoint SHA；只提交代码与必要汇总，缓存和完整日志保持Git忽略。诊断结果出来后更新本文件的结论链接，主训练仍需另行确定配置。
