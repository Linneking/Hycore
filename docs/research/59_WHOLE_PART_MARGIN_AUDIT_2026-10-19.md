# whole与part逐对象margin审查

2026-10-20补充：[N600的BN与同深口径复核](59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/clarifications/N600_BN_ORDER_2026-10-19/report.md)。V8两者贴cap时，原始深度差正负包含浮点微差，应分开报告part更浅、数值同深、part更深；不能将约51%原始正差率直接解释为近半真实反转。主报告用户编辑保持原样。

2026-10-19。见[互动Markdown报告](59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/report.md)及[浏览版](59_WHOLE_PART_MARGIN_AUDIT_2026-10-19/report.html)。

覆盖原版HyCoRe、旧8856/984 V8-HIER64和FULL9840 V8-HIER64的best/终点，同8856对象及完全相同N200/400/600嵌套part；统计完整径向margin、90%目标、深浅顺序，以及共同异类负part下的对比/联合达标。另有相同1280对象的保存BN与当前批BN对照。16组PNG/SVG、逐N/逐类匿名统计、历史索引强验证与源状态恢复验收随正文保存。

500/N与delta2实现正确，但clean/eval仅少数对象完整达到径向margin；原版1000/N也不是普遍兑现的硬约束。当前批BN改善V8 N400，N600却因part也靠近cap而仍差，不能用恢复whole壳替代整个intra验收。诊断无参数优化、无test前向、不改变正式模型。
