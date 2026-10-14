# V8实现、验收与后台训练交接

日期：2026-10-15。状态：**已完成编码与CPU/分布式验收，后台任务已提交；截至02:23:30（Asia/Shanghai）等待两张完全空闲GPU，尚未运行GPU短验收或生产训练。** 用户已明确同意V8修改与优化配置，并授权“开始编码挂到服务器上进行训练，问题自行解决，挂起后退出”。

## 已确认配置

- 名称V8-HIER64-K20-W0，seed22，从头300轮×200步；单卡16类×2例，两卡类别不重复，全局64例。
- 保留V7的8856训练/984验证划分、part视图复写whole、两次local32 BN更新、原FPS和增强流程。
- 全链曲率幅值c=0.1。最终共享mu统一logmap0→切空间cap2.3→expmap→一次逆度量梯度放缩；CE、intra、HIER读取同一whole，part执行同样映射。
- 代理按官方Gaussian×2.3×0.9/sqrt256初始化，执行同样forward cap/hook，取消V7校准初始化与step后参数投影。
- HyCoRe：CE+0.01 Rcontr(delta2)+0.01 Rhier(gamma500/N)。HIER：0.5(Rsample+Rproxy)，K20/T50，margin/tau=0.1，从第一次更新启用。
- 主模型RSGD学习率0.1→0.005；代理AdamW 0.01→0.0005；两者300轮余弦。保留主模型norm1梯度裁剪与原代理优化协议，不把官方图像优化器50倍倍率乘到SGD的0.1上。
- 保留self-k排除。HIER距离使用官方Mobius分母epsilon和Artanh导数，分块重计算节省内存；intra使用Geoopt c=0.1距离。
- validation选模，完整训练结束后仅对选中模型读取一次官方test。短验收不读test，不用于初始化生产训练。

## 实现组织

独立入口位于`inter_hierarchy_MN40/hier_proxy_scratch_v8/`；旧版本文件不改。

- `geometry.py`：曲率、最终共享表示hook、官方映射/距离、深度与更新统计。
- `base_protocol.py`：V8两项intra margin和global64联合目标，保留源式输入处理。
- `hier_loss.py`：官方代理映射与两分项、修正的采样、实际祖先选择统计。
- `telemetry.py`、`parameter_audit.py`：以c=0.1真实双曲距离记录whole/part和祖先深度，分解参数梯度。
- `train.py`：双卡训练、验证选模、完整状态保存/新目录恢复、最后一次test。
- `queue.py`：固定源码快照，CPU核查，等待两张空闲卡，独立双卡smoke，然后从头生产训练。
- `tests/`：官方前向/梯度一致性、margin、共享表示、采样、监测单位、资源排队与Gloo梯度归一化。

## 资源与启动规则

初次资源检查四张GPU均有计算进程。后台任务只选择无计算进程、利用率0且空闲显存至少30000MiB的两张卡，连续检查后才能使用；不会把“利用率暂时0但仍有进程”当作空闲。

若未能立即取得两张空闲卡，任务保持后台排队，CPU核查和训练状态分别记录。GPU可用后自动运行独立2轮×2步smoke；只有smoke完整通过且设备仍空闲，才启动新初始化的300轮主训练。任何失败写入状态，不能把排队或smoke说成主训练已开始。

源码快照、编译扩展身份、日志、权重和完整样本ID仅保存在私有新目录；公共记录只保留提交、配置、脱敏验收和运行状态。

V8同时改变几何、margin、初始化与HIER权重，结果应解释为整体协议迁移，不能作为纯曲率的单变量因果对照。

## 启动前已通过的核查

测试实现提交5556dc5：8项CPU合同检查全部通过、无跳过；官方ToPoincare与pairwise距离分别在FP32/FP64核对前向和反向；包含margin、最终共享表示、代理参数不做step后投影、实际采样和监测单位检查。Gloo双进程global64梯度与单进程参照一致，且参数分项审计不会改变正式梯度。该分布式检查使用D8/P32的CPU小模型，不冒充完整PointMLP的GPU短训练。训练入口与已有PointNet2扩展导入通过。

原Conda标准库多次卡在旧文件读取。现使用相同Python3.9.25安装包建立私有临时磁盘副本，并复用Torch2.8、Geoopt0.5.1、NumPy1.26.4的既有快速缓存；缺失的辅助模块按原版本补齐。PyTorch辅助代码从官方2.8.0 wheel局部提取。旧环境和其他用户进程均未修改。代码及训练输入也在私有临时磁盘缓存，checkpoint和运行记录保存在新结果目录。

CPU验收摘要见[记录](diagnostics/results/20261015_v8_prelaunch_cpu_acceptance.json)。完整GPU短训练由后台任务在两张卡空闲后执行；其通过前不得报告生产训练已启动。

## 最终后台任务与交接状态

- 唯一有效任务ID：`v8_hier64_20261015_022043`，展示名V8-HIER64-K20-W0。训练代码固定提交`14f8a7201cd108a77bde3b2ec9c3053e3cc07579`，分支`codex/v8-hier-geometry-training-20261015`。
- 提交时间2026-10-15 02:20:51 Asia/Shanghai；后台队列已由PID1接管，具有独立session，脱离本次SSH/对话继续执行。
- 最终源码快照再次通过9项合同检查、无跳过；新增检查固定物理GPU UUID，避免5090/5090D的CUDA编号排序差异。CPU双进程global64梯度与分项审计检查通过。
- 真实HDF5训练集核对9840对象、40类、每对象2048点；训练/验证8856/984，split SHA256为`3bbf0c79b3cb9581caeafe3f7f12caa6861f236f0159512a5998d5e9bcf25105`。训练输入只读缓存完成，未读官方test内容。
- 已核实运行时：Python3.9.25、PyTorch2.8.0+cu128、Geoopt0.5.1、NumPy1.26.4、SciPy1.13.1、h5py3.14.0。CPU核查没有初始化CUDA。
- 02:23:30四卡均仍有计算进程，队列状态`waiting_for_two_idle_gpus`，`production_started=false`。目前没有V8训练轮次或分类结果可报告。
- 两卡空闲后自动依次执行：单独smoke目录2轮×2步（含实际参数梯度审计与checkpoint）→确认通过且设备仍空闲→重新seed22初始化主训练300轮×200步。训练失败会写明错误状态；不把排队、smoke或进程存活当作完整训练成功。
- 早先021244等待任务仅用于队列预检，已在任何GPU使用前结束并标记superseded；其只读数据缓存被后续任务复用，不是另一个主实验。

脱敏启动记录：[最终验收](diagnostics/results/20261015_v8_queued_startup.json)。私有工程交接：`.codex-local/handoff/50_V8_ENGINEERING_HANDOFF_2026-10-15.md`，包含实际队列、日志、快照、运行时及后续checkpoint位置；不提交Git。后续查进展应读取服务器实时`queue_status.json`及main下manifest/heartbeat，本节是提交时快照。
