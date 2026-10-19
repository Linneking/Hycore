# whole密度与BN统计兼容性审查

本目录是2026-10-19冻结审查的独立工具，不修改生产入口。图文报告位于[报告58](../../58_V8_WHOLE_DENSITY_AND_BN_REVIEW_2026-10-19/report.md)。

## 工具与数据约定

- density_figures.py：读取私有配置中的四模型best/last缓存，按HDF5名字和文件内行号转换原版ID，核验8856对象、标签、输入摘要，重算各自曲率下的原点双曲深度。输出统一坐标直方图、ECDF、单图及匿名分位数。
- original_hier_direction_figure.py：从既有匿名原版HIER几何聚合中提取类内/跨类单位方向余弦，输出科学图及CSV。
- bn_reference_probe.py：明确指定三份e300网络、同8856训练池和参考BN运行buffer，在空闲GPU执行固定320测量对象/640校准对象的四条件前向；每条件恢复完整状态与RNG。不更新参数、不使用test、不直接移植γ/β。
- bn_reference_figures.py：读取探针匿名汇总，绘制whole、part和方向/预测变化，不把训练面板预测当作test成绩。
- render_review.py：将本目录报告的Markdown渲染为带图片和公式的HTML，并验证本地图文链接。

所有权重、缓存、完整对象身份及服务器路径通过私有配置提供；不要写入Git。

## 运行

本地科学绘图需要NumPy与Matplotlib，HTML渲染需要Python-Markdown。GPU前向另需原项目可运行的Torch/Geoopt/pointnet2环境。

    python density_figures.py --config PRIVATE_DENSITY_CONFIG.json --output REPORT_DIR
    python original_hier_direction_figure.py --aggregate ORIGINAL_PUBLIC_AGGREGATE.json --out-dir REPORT_DIR
    python bn_reference_probe.py --spec PRIVATE_PROBE_SPEC.json --out-dir NEW_PRIVATE_DIR --gpu ID --max-seconds 1200
    python bn_reference_figures.py --summary PRIVATE_PROBE_SUMMARY.json --out-dir REPORT_DIR
    python render_review.py --report-dir REPORT_DIR

探针输出目录必须为全新目录。它在绑定物理GPU UUID之前检查空闲，在条件间检查其他计算进程；正式结果目录不可复用。参考BN NPZ只包含去module前缀后的running_mean、running_var、num_batches_tracked，共38层114个buffer，不含affine。

BN输出中的private_*.npz与manifest.json是私有数据；公开材料只采用匿名条件统计及验证结论。

## 本次验收

四模型best/last共8份缓存，均同8856对象与点云字节；原版历史epoch沿用0起身份。三份V8 native面板重放与既有全池缓存逐元素一致，最大mu差为0，320个预测完全相同。三模型×四BN条件均完成，完整state/BN/RNG和源文件SHA校验通过。恢复whole壳不作为分类或层次收益的充分证据。
