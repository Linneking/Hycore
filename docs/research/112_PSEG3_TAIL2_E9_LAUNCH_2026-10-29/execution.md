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

实际部署提交与排队验收待后续补入。本文件当前不证明队列已经启动，也不证明真实短测已通过。
