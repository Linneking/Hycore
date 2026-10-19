# 四模型whole分布与BN兼容性审查

日期：2026-10-19。正文见[可互动Markdown报告](58_V8_WHOLE_DENSITY_AND_BN_REVIEW_2026-10-19/report.md)，另附[浏览版](58_V8_WHOLE_DENSITY_AND_BN_REVIEW_2026-10-19/report.html)。

比较原版HyCoRe与已完成、同8856/984协议的V8-B0-32、V8-B64无HIER和V8-HIER64，覆盖best及训练终点的whole密度、原HIER类别方向分离、同壳与浅代理的关系，以及三模型四种BN统计来源的冻结干预。11组科学图、匿名数值表、输入字节与状态恢复验收随报告保存；可复现工具位于[独立诊断目录](diagnostics/whole_density_bn_20261019/README.md)。

自己的whole-only BN校准能够恢复whole壳，同时会把part推深；原版最终BN直接移植明显错配。本次没有修改正式模型或BN，没有test前向，不启动新的完整训练。
