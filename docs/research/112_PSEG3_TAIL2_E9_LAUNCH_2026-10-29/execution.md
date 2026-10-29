# D组实现与排队记录

日期：2026-10-29。用户已明确授权本组；结构、B组对照和共同协议见[plan.md](plan.md)。本记录不会将排队或CPU检查写成已通过真实GPU短测。

## 已完成的本机验证

- 独立新路径`segmentation_ShapeNetPart/hycore_seg_v4`，旧v1/v2/v3未修改。
- 10项模型CPU合同通过：保留dp1与原Conv9/BN9/Leaky、两M、无残留废弃参数；共享前端与B存活M初值完全一致，Conv9/BN9与原DGCNN一致；构建后的CPU RNG也与B一致。train/eval到Conv9的前传与原结构逐位一致。
- 6项启动器合同通过：旧任务三份完成证据齐全才放行；失败或半写JSON不误放行；只写新队列；先等待旧B，再idle检查、fresh真实2×2 smoke、CPU验收、fresh200，绝不resume短测。
- 完整合成CPU入口已运行两轮各两步、4次参数更新；N32/k4/batch4、train8/val8、无test文件或评估、CUDA未初始化；BN含新BN9均为4，模型/优化器状态及全局/独立RNG可恢复，cosine仍保持200轮。
- pre_conv9/newH及point/whole/part/M输出、CE/两正则分项梯度及euclidean9组均记录；正则可回Conv9与共享前端，M头未使用梯度正确记为None。最大总梯度范数1.5902979，未裁剪。
- CPU入口使用显式synthetic staging身份，不作为生产提交或真实数据验收；完整本机临时权重、合成数据不提交。

## GPU与接替安排

20:56:44北京时间只读检查：A/B/C均训练中，GPU0/1/2均占用。按近10轮耗时估计B约22:38正常结束，A约22:45，C约22:57；预计时间不是抢卡依据。

唯一新组D目标GPU1，启动器须同时看到旧B的queue complete、main status complete及acceptance.complete，随后再通过nvidia-smi空闲检查。届时先真实batch32/N2048、clean train/val的独立2×2短测，41项验收通过后同初始化fresh200正式trainval/test。旧B异常则停止新队列，不自行重启旧任务；其他用户占卡则继续等空闲。

## 已部署并排队，尚未占用GPU

生产源码`a82f5c15edff9585f9db5e109d5ec23bb7aee9bf`已推送独立分支；同提交bundle SHA256为`420c326e7c9543e73f6a614accd016da0c4c8afaedfee90ccf88b4f727dec561`，服务器verify/fetch/独立worktree/pull --ff-only通过。活动旧v3工作区仍为9c332b0，未更新或修改。

服务器16项CPU合同及完整合成CPU入口也通过：4更新、无CUDA、保存恢复与新监控正确；该环境最大总梯度范数2.46840，未裁剪。它与本机不同环境的数值不作逐位相等要求。

2026-10-29 21:09:51北京时间，新run `pseg3_v4_tail2_e9_dp1_s4780_20261029_A` 已提交至GPU1等待队列。21:10:34只读复核：启动器存活且PPid=1，queue=`waiting_for_predecessor`；smoke/main均尚未创建，没有提前使用忙卡，没有正式训练或真实GPU短测结果。

同次检查旧A/B/C仍正常训练：A e146step325、B e149step325、C e142step50。按最近10轮中位耗时外推剩余约94.7/87.7/106.1分钟，预计北京时间22:45/22:38/22:57左右；新D须等待B完整成功验收、GPU1空闲及真实短测通过才fresh200。具体源提交/配置和脱敏状态见[queue_acceptance.json](queue_acceptance.json)。

后续不重复提交，不向此活动工作区拉入文档后续提交。路径与PID仅保留在本地私有交接文件。真实短测41项和首轮训练验收等待旧B正常结束后由已部署启动器执行；不把排队写成GPU已开始训练。
