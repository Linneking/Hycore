# TS26 / B15：原版 HyCoRe 改 CB32

用户于2026-10-27直接授权在8000挂一条实验，并明确每轮200步，与TS23更新预算一致。此授权只涉及这一条，不新增矩阵。

名称：**HC-C1-Cap0-GS0-M1000_4-CB32x200 / H0 (S4780)**。

- fresh300轮×200步，共60000次更新；seed4780；官方完整train9840。
- CB32：均匀抽16个不同类别，每类有放回抽2个对象；单卡，自动选择完全空闲GPU，锁定后再次检查。
- 原版PointMLP、c=1、D256；无新增切空间cap、无输出GS、无HIER及代理。
- 原版CE平滑.2，基础两项权重各.01，contrastive margin4、radial margin1000/Nchild。
- 原版whole800–1024、part200–600、part复写whole、child/whole各更新一次普通BN；intra负例为CB32内child.flip(0)。
- 原版RiemannianSGD：LR.1、momentum.9、WD2e-4、范数裁剪1；300轮cosine到.005。
- workers8、eval batch16。沿用当前已明确的历史TS协议：每轮完整test2468，严格更高OA选best，平局保留首次，不早停；不是独立验证集选模。
- 相比原版SH32×307，改变了采样与每轮预算；CB类别块也改变flip负配对分布。相比TS23，基础几何与margin也不同，不能作为只去掉HIER的单因素对照。

复用SHA固定的V9C连续随机流/保存循环，以版本化runner选择已核对原版数学与前向实现，原入口保持不变。每次执行保存commit、配置、种子、物理GPU/UUID、时间、数据SHA和初始化身份。短测独立2轮×2步、train-only、零test；验收通过后重新fresh正式训练，不沿用短测权重。

入口：[runner.py](/D:/Hycore/docs/research/diagnostics/native_cb32_h0_20261027/runner.py)；部署：[launch_queue.py](/D:/Hycore/docs/research/diagnostics/native_cb32_h0_20261027/launch_queue.py)。CPU检查包括CB32真实原版损失/梯度对照、采样均衡/恢复、原版初始化及无额外hook；另运行已有原版算子回归。

当前状态：准备与检查中。启动身份与结果在验收后补录；私有路径、原始日志和权重不进仓库。
