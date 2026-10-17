# V8 cap / whole 半径机制诊断

本目录是独立诊断工具，不是训练入口；不改变 V7/V8 的生产配置。图文结论见 [报告57](../../57_V8_CAP_AND_WHOLE_CAUSES_2026-10-18/report.md)。

所有输入参数均指向本地或服务器私有文件；不得把权重、特征、完整ID、Adam状态、图像、主机信息或凭证提交到Git。每次使用新输出目录，先核实输入身份及源文件SHA。GPU运行前检查空闲并持有项目锁，不能中断其他用户进程。

| 文件 | 用途 | 计算范围 |
|---|---|---|
| `reproduce.py` | 复现已保存模型的cap及whole径向分散症状 | CPU统计；症状成立时退出1 |
| `extract_v8.py` | 从旧V8权重和审查缓存提取最小副本 | CPU；私有输出 |
| `extract_available.py` | 从当前服务器已有权重生成固定训练面板 | 只读编码器前向；无test |
| `original_optimizer_extract.py` | 提取原版完整checkpoint的代理Adam状态 | CPU；源SHA校验 |
| `original_head_probe.py` | 观测原版head在cap之前的模长 | 每集12个训练anchor；无更新 |
| `optimizer_math.py` | 分解径向梯度、实际一步更新和有限步增模 | 不将梯度等同于Adam位移 |
| `test_optimizer_math.py` | 检查分解恒等式、Adam及weight decay行为 | 9项实际Torch测试 |
| `moment_controls.py` | 固定当前梯度，对历史一、二阶矩做一步干预 | 原V8副本；源状态不变 |
| `proxy_probe.py` | sample/proxy分项、eps/LR/hook/whole深度的条件干预 | 固定特征，仅更新私有代理副本 |
| `v8_whole_probe.py` | whole/part cap前模长与坐标叶节点梯度 | 不回传共享编码器、不更新参数 |
| `whole_condition_probe.py` | BN、增强、构造方式的固定对象因子试验 | 128训练对象；每条件恢复BN/RNG |
| `bn_calibration_probe.py` | 固定参数，更换临时BN统计库 | 640校准训练对象与128测量对象不重叠 |
| `plot_proxy_evidence.py` | 生成报告代理图与匿名统计CSV | 本地聚合资料 |
| `plot_whole_evidence.py` | 生成whole前向条件对照图与CSV | 本地聚合资料 |
| `render_report.py` | 把保存的Markdown渲染为HTML | 保留原始$$公式；图片本地引用 |

常用验收：

```powershell
python -m unittest discover -s docs/research/diagnostics/v8_cap_mechanism -p test_optimizer_math.py -v
```

`proxy_probe.py --mode trajectory --core-only --steps 1024` 比较eps；`--transport-only`只检验历史一阶矩径向分量消除，**不是严格双曲平行移动**。两个开关各自使用新目录。GPU UUID必须与刚确认空闲的设备相同。

`whole_condition_probe.py --include-full-train` 增加完整train模式补充；主比较只改变BN模式而关闭dropout。`--fixed-ids`可读取第一次输出的私有NPZ，使不同模型使用同128训练对象。不同checkpoint的训练池必须包含这些对象。旧B64 e300与FULL9840 HIER e40分别解释，不能视为同轮HIER开关实验。

统计口径：raw代理模长在cap之前；球内深度是原点双曲距离；cap为raw切空间长度2.3，c=.1时深度约4.6。汇总表分位数使用NumPy，个别GPU运行日志使用Torch下中位数，不能把二者微小差异当作状态变化。whole贴cap率使用depth≥4.5999，代理越cap使用raw norm>2.3。

`loss_parts`返回的sample/proxy均已乘外系数0.5，固定面板的total为0.5×sample_raw+0.5×proxy_raw；不包含分类及intra。表中评分和实际代理更新采用相同尺度，不能再把该total称为未加权原始loss。

受控副本优化仅解释给定特征和优化状态的条件响应，不是新的完整训练结果，不提供分类性能结论；任何新完整训练仍需另列方案。
