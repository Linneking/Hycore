# whole与part逐对象margin审查

对应[报告59](../../59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/report.md)。本目录独立于生产训练入口；输入权重、向量、对象身份和运行配置保留在私有目录。

## 工具

- setup_local_inputs.py：读取报告58的私有密度配置，映射原版旧ID到canonical顺序，提取共同8856对象、中心及原版best/last的完整part缓存。
- forward_probe.py：只在完全空闲GPU上读取固定权重，导出N200/400/600的whole/part向量。native覆盖8856对象；仅BN当前批统计使用40类均衡1280面板。每N恢复完整参数、BN和RNG，无backward、optimizer或test。
- validate_pair_inputs.py：核对三组GPU结果的对象、标签、中心、嵌套索引及旧V8全量whole的逐元素重放；从native向量提取与BN相同1280对象的子缓存。
- verify_part_index_bridge.py：CPU从原始训练HDF5重新生成历史9840索引，桥接历史SHA、canonical转换及本次8856实际索引。
- analyze_pairs.py：CPU统计逐对象径向完整/90%margin达标、顺序、违例幅度及40类结果；使用共同20个异类负part计算对比与联合指标，生成16组PNG/SVG。
- build_report.py：从匿名CSV生成Markdown正文，避免手抄数值；HTML可复用相邻whole_density_bn_20261019/render_review.py。

## 主要调用

    python setup_local_inputs.py --config PRIVATE_DENSITY.json --out-dir NEW_PRIVATE_INPUTS
    python forward_probe.py --spec PRIVATE_SPEC.json --out-dir NEW_PRIVATE_RUN --gpu ID --max-seconds 2400
    python verify_part_index_bridge.py --spec PRIVATE_BRIDGE.json --out-dir NEW_PRIVATE_BRIDGE
    python validate_pair_inputs.py --inputs-dir PRIVATE_INPUTS --gpu-run-dir PRIVATE_RUN --output PUBLIC_VALIDATION.json
    python analyze_pairs.py --config PRIVATE_ANALYSIS.json --out-dir REPORT_DIR --private-out-dir NEW_PRIVATE_OBJECT_STATS
    python build_report.py --report-dir REPORT_DIR

前向spec通过source_root、data_dir、source_train_ids_npz、centers_npz及models传入路径；每模型明确kind、c、gamma、delta、checkpoint、label。原版可eval_enabled:false复用已有全量缓存，只补当前批BN诊断。分析config的模型字段另含stage、epoch、condition及私有cache路径；native_eval_panel和BN_current_batch_only必须同ID/标签。

validate_pair_inputs.py可用--historical-index-bridge合并独立CPU索引强证明；--existing-panel-dir只读复核已经生成的native1280面板，避免覆盖旧结果。

## 判读边界

Rhier检验d0(whole)−d0(part)≥gamma/N；完整达标保留1e−4数值容差，并另存无容差计数。90%目标是显式宽松诊断。Rcontr检验负part距离减自身part距离≥delta，20个异类负例是共同几何面板，不冒称历史训练flip。BN-only前向没有增强、别名覆盖或生产batch组成，不能作为真实训练通过率。

完整对象ID、权重、缓存、私有路径和完整日志不可提交Git。公开输出仅保留匿名统计、SHA、图和验证结论。
