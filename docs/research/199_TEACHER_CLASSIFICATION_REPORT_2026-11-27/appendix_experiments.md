# 实验附录：11月12日起52组完整正式训练

本附录供40分钟汇报时追问配置与原始读数。**主范围为TS27–TS55共29组、17组后续非TS、194/196最新6组，共52组完整正式训练**，均已完成既定250或300轮。另列两条已启动2轮后停止的B17验证协议训练；它们不并入52条完整训练。7个发布权重推理条件和诊断也分别登记，不能计作训练。TS06、TS23仅作历史参照，其他TS01–26不展开。

本次只使用本机197/198已核验材料，无服务器连接、模型推理或训练。数字核验截止分别保留源记录时间；λ=1.0采用198在11-26 17:42的完整300轮终局，取代197早期e298快照。DeLA逐对象诊断证据截止11-26 18:03:51。完整身份、精确标量与可供画图的数据见 [experiments.json](experiments.json)，唯一ID/覆盖检查见 [coverage.json](coverage.json)。

## 读表约定与共用配方

- 所有日期为2026年北京时间，按**实际worker启动**分日；入队时间不当启动时间。DeLA v1原版只核到11-24约01:50，精确秒保留缺口。表中长原目录保留实际run名，canonical身份另存结构化副本。
- 数字单位为%。best后AA为该组best OA同轮AA，非独立最高AA；末10为预定e291–300或e241–250算术均值，非多seed平均。展示三位采用十进制四舍五入，配对差值应使用JSON中的源值。
- MN40=官方train9840/val0/test2468；MN20=自定义MN40-F20-N130_400派生子集4298/0/1398，不是官方ModelNet20，不与完整MN40的TS06 94.327%排同榜。SO=ScanObjectNN PB_T50_RS/main_split含背景11416/0/2882，15类。
- **B05/B13/B16/B17/B18/B19**主支均为PointMLP+HyCoRe的c=.1、tangent cap2.3、基础GS、margin500/Npart与2、两项intra各.01。B05=SH32×307；B13=CB32×200；B16=CB32×300；B17每两个SH32累积一次更新、153次/轮；B18/19=MN20 SH/CB32×134。
- SH32无放回shuffle/drop_last；完整MN40每轮9824对象，B17因成对取批只用9792。CB32=16类各有放回2对象；SO只有15类，196的CB32为15类各2，加轮换2类各1。分类batch与HIER候选池独立列出。
- **Hshell**：whole保方向副本固定深度4.6/c=.1，独立GS。**实际深度cap**：原生whole μ经logmap0(c1)→tangent cap2.3→expmap0(c.1)→独立GS，保留上限内变化；最后批样本可能自然全触cap，不改称固定壳。
- HIER默认P512/D256、初始化.9、T50/W0。A1代理AdamW LR .005→.0005/eps1e−4；A0=.01→.0005/eps1e−8，切换A0同时改LR和eps。Ks/Kp及两项权重以逐组配方为准。SO作者PMLP原生c1，hier系数.001、margin500/Npart、对比.01；作者DGCNN原生两项各.01。
- 历史逐轮test-best组固定预算无早停，末10按各组预定窗口。新版DeLA H0/HIER仅固定e250测一次官方test；**best和末10写未测**，唯一终点不复制到这些栏。原版DeLA仍有逐轮test日志（名为val），保留自身peak和末10，89,000为尝试预算，实际有效更新总数缺。新版两组88,961有效/39 AMP跳步总数相同，但逐轮跳步位置不同。

## 52组完整正式训练：按实际启动日期分组

每组在本节仅出现一行。数据栏省略val=0但沿用上述数据字典；所有新训练均随机初始化、无冻结教师。配置／读数细节沿每行原目录与证据可查。

### 2026-11-12（3组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `TS27`；01:17:12 | MN40 9840/2468／PointMLP／B13；S4780 | CB32／1卡；池32 | 200×300=60,000 | Hshell-A1-Ks10-Kp10-L0.1-W0-P256；代理P512→256（前256共享初值） | e277：93.801 / 90.824 | e300：93.395 / 91.116 | 93.262 / 91.028 | TS23 P512→256；逐轮test-best；固定300轮；best AA取同轮 | `ts23_proxy_ablations_20261112_B/ts23ablate_20261112_B_p256/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../160_MN40_TS23_PROXY_ABLATIONS_2026-11-12.md) |
| `TS28`；01:17:12 | MN40 9840/2468／PointMLP／B13；S4780 | CB32／1卡；池32 | 200×300=60,000 | Hshell-A1-Ks10-Kp10-L0.1-W0-I05；代理初始系数.9→.5 | e280：93.841 / 92.241 | e300：93.314 / 91.153 | 93.201 / 90.995 | TS23初始化.9→.5；逐轮test-best；固定300轮；best AA取同轮 | `ts23_proxy_ablations_20261112_B/ts23ablate_20261112_B_init05/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../160_MN40_TS23_PROXY_ABLATIONS_2026-11-12.md) |
| `TS29`；01:17:12 | MN40 9840/2468／PointMLP／B13；S4780 | CB32／1卡；池32 | 200×300=60,000 | Hshell-A1-Ks10-Kp10-L0.1-W0-PCap29；只放宽代理cap2.3→2.9；样本壳4.6不变 | e276：93.760 / 91.483 | e300：93.679 / 91.420 | 93.181 / 91.093 | TS23代理cap2.3→2.9；逐轮test-best；固定300轮；best AA取同轮 | `ts23_proxy_ablations_20261112_B/ts23ablate_20261112_B_cap29/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../160_MN40_TS23_PROXY_ABLATIONS_2026-11-12.md) |

### 2026-11-13（6组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `TS30`；00:41:24 | MN40 9840/2468／PointMLP／B13；S4780 | CB32／1卡；池— | 200×300=60,000 | H0 | e262：93.841 / 91.365 | e300：93.355 / 90.799 | 93.088 / 90.724 | TS23/19–29/32–34共同H0；TS31预算；逐轮test-best；固定300轮；best AA取同轮 | `b13_cb32_controls_20261113_B/b13control_20261113_B_h0_200/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../161_MN40_B13_CB32_CONTROLS_2026-11-13.md) |
| `TS31`；00:41:25 | MN40 9840/2468／PointMLP／B16；S4780 | CB32／1卡；池— | 300×300=90,000 | H0 | e292：93.801 / 91.483 | e300：93.112 / 90.453 | 93.039 / 90.819 | TS30只改200→300步；TS06跨协议；逐轮test-best；固定300轮；best AA取同轮 | `b13_cb32_controls_20261113_B/b13control_20261113_B_h0_300/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../161_MN40_B13_CB32_CONTROLS_2026-11-13.md) |
| `TS32`；00:41:27 | MN40 9840/2468／PointMLP／B13；S4780 | CB32／1卡；池32 | 200×300=60,000 | Hshell-A1-Ks20-Kp20-L0.5-W0 | e277：94.125 / 91.228 | e300：93.193 / 90.695 | 92.796 / 90.552 | TS30匹配H0；TS33优化器；TS34权重；逐轮test-best；固定300轮；best AA取同轮 | `b13_cb32_controls_20261113_B/b13control_20261113_B_a1_200/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../161_MN40_B13_CB32_CONTROLS_2026-11-13.md) |
| `TS33`；23:31:41 | MN40 9840/2468／PointMLP／B13；S4780 | CB32／1卡；池32 | 200×300=60,000 | Hshell-A0-Ks20-Kp20-L0.5-W0 | e262：93.760 / 91.962 | e300：93.436 / 91.478 | 93.051 / 91.101 | TS32 A1→A0，LR/eps联动；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_ts33_ts35_20261113/mn40v10_20261113_ts33/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../162_MN40_V10_TS_CONTINUATION_AND_SH32_2026-11-13.md) |
| `TS34`；23:31:41 | MN40 9840/2468／PointMLP／B13；S4780 | CB32／1卡；池32 | 200×300=60,000 | Hshell-A1-Ks20-Kp20-L1.0-W0 | e198：93.841 / 92.261 | e300：93.233 / 91.437 | 93.027 / 91.004 | TS32 λ.5→1；TS30 H0；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_ts33_ts35_20261113/mn40v10_20261113_ts34/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../162_MN40_V10_TS_CONTINUATION_AND_SH32_2026-11-13.md) |
| `TS35`；23:38:39 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池32 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L0.5-W0 | e184：94.206 / 91.006 | e300：93.031 / 90.008 | 92.926 / 90.628 | TS06匹配H0；入口等价短测通过；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_ts33_ts35_20261113/mn40v10_20261113_ts35_r2/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../162_MN40_V10_TS_CONTINUATION_AND_SH32_2026-11-13.md) |

### 2026-11-14（2组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `TS36`；12:45:20 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池32 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L0.1-W0 | e184：93.801 / 90.728 | e300：93.193 / 90.324 | 93.043 / 90.664 | TS35两项λ.5→.1；TS06 H0；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_ts35_lambda_20261114/TS36/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../162_MN40_V10_TS_CONTINUATION_AND_SH32_2026-11-13.md) |
| `TS37`；12:45:20 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池32 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L1.0-W0 | e248：94.003 / 90.924 | e300：93.274 / 90.574 | 93.306 / 90.915 | TS35两项λ.5→1；TS06 H0；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_ts35_lambda_20261114/TS37/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../162_MN40_V10_TS_CONTINUATION_AND_SH32_2026-11-13.md) |

### 2026-11-15（3组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `TS38`；00:00:16 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池32 | 307×300=92,100 | Hshell-A1-Ks10-Kp10-L0.5-W0 | e193：93.882 / 90.853 | e300：93.355 / 90.716 | 93.055 / 90.577 | TS35同时K20/20→10/10；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_ts38_k10_20261114/TS38/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../162_MN40_V10_TS_CONTINUATION_AND_SH32_2026-11-13.md) |
| `TS41`；13:49:13 | MN40 9840/2468／PointMLP／B17；S4780 | SH2x32／1卡；池实时64 | 153×300=45,900 | Hshell-A1-Ks20-Kp20-L0.5-W0-SP64；每2个SH32累积，合并实时64池挖关系 | e179：93.760 / 90.883 | e300：93.152 / 90.203 | 93.075 / 90.434 | TS39匹配H0；TS40池32→64；GPU型号有别；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_b17_sh_pool_testbest_20261115/H64/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../164_MN40_V10_SH_POOL32_64_2026-11-15.md) |
| `TS39`；13:49:29 | MN40 9840/2468／PointMLP／B17；S4780 | SH2x32／1卡；池— | 153×300=45,900 | H0；每2个SH32累积更新一次；H0 | e265：93.801 / 90.720 | e300：92.828 / 90.479 | 92.739 / 90.175 | TS40/41共同H0；B17 test-best重跑；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_b17_sh_pool_testbest_20261115/H0/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../164_MN40_V10_SH_POOL32_64_2026-11-15.md) |

### 2026-11-16（5组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `TS40`；02:55:25 | MN40 9840/2468／PointMLP／B17；S4780 | SH2x32／1卡；池32+32分别挖掘 | 153×300=45,900 | Hshell-A1-Ks20-Kp20-L0.5-W0-SP32；每2个SH32累积，各自32池挖关系、抽样数加权 | e231：93.963 / 91.091 | e300：92.869 / 89.628 | 92.970 / 90.063 | TS39匹配H0；TS41池大小；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_b17_sh_pool_testbest_20261115/H32/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_tail_audit.json)；[配置](../164_MN40_V10_SH_POOL32_64_2026-11-15.md) |
| `TS43`；13:03:04 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池32 | 307×300=92,100 | Hshell-A1-Ks20-Kp10-L0.5-W0 | e248：93.841 / 90.704 | e300：93.558 / 90.749 | 93.148 / 90.600 | TS35仅Kp20→10；TS38/42为2×2；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_sh32_ts42_ts44_20261116/TS43/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../165_MN40_V10_SH32_K_AND_CACHE_POOL_2026-11-16.md) |
| `TS42`；13:05:20 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池32 | 307×300=92,100 | Hshell-A1-Ks10-Kp20-L0.5-W0 | e235：94.044 / 91.382 | e300：93.841 / 90.820 | 93.310 / 90.511 | TS35仅Ks20→10；TS38/43为2×2；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_sh32_ts42_ts44_20261116_r2/TS42/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../165_MN40_V10_SH32_K_AND_CACHE_POOL_2026-11-16.md) |
| `TS44`；13:05:20 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池当前32锚+历史32；最多64 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L0.5-W0-Hist1-SP64；当前32锚+前一步detach32候选，轮首清空 | e300：94.084 / 91.299 | e300：94.084 / 91.299 | 93.416 / 90.642 | TS35加一步缓存；TS06整体HIER+缓存；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_sh32_ts42_ts44_20261116_r2/TS44/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../165_MN40_V10_SH32_K_AND_CACHE_POOL_2026-11-16.md) |
| `TS45`；23:52:56 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池32 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-Ls0to0.5e51-Lp0.5；λsample从e1=0线性升至e51=.5，λproxy=.5 | e235：94.003 / 91.628 | e300：93.679 / 91.124 | 93.079 / 90.551 | TS35仅样本项日程；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_sh32_ts45_ts46_20261116_r3/TS45/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../166_MN40_V10_SH32_SPLIT_HIER_WEIGHTS_2026-11-16.md) |

### 2026-11-17（1组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `TS46`；00:42:34 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池32 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-Ls0.5-Lp0.0-W0；λsample=.5、λproxy=0，代理受样本项更新 | e241：93.882 / 91.249 | e300：93.193 / 90.428 | 93.185 / 90.576 | TS35仅代理内部项关闭；代理仍受样本项更新；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_sh32_ts45_ts46_20261116_r3/TS46/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../166_MN40_V10_SH32_SPLIT_HIER_WEIGHTS_2026-11-16.md) |

### 2026-11-18（4组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `TS47`；01:10:50 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池当前32锚+历史64；最多96 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L0.5-W0-Hist2-SP96；当前32+前两批detach64；FIFO96 | e182：93.760 / 91.011 | e300：93.274 / 90.770 | 93.006 / 90.682 | TS44 FIFO1→2，池大小和年龄共同变；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_sh32_fifo96_20261118/TS47/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../167_MN40_V10_SH32_FIFO96_2026-11-18.md) |
| `TS50`；12:50:07 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池当前32锚+历史32；最多64 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L0.1-W0-Hist1-SP64；FIFO64、样本/代理权重各.1 | e189：93.801 / 89.548 | e300：93.193 / 90.428 | 93.023 / 90.398 | TS44两项λ.5→.1；TS36池大小；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_history_selection_20261118/TS50/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../168_MN40_V10_HISTORY_SELECTION_2026-11-18.md) |
| `TS48`；12:50:14 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池当前32锚+历史32；最多64 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L0.5-W0-Hist3-SP64-CP16x2；前三批最多96中优先选16类×2；加当前32 | e248：93.882 / 91.341 | e300：93.193 / 90.728 | 93.152 / 90.706 | TS49同窗口同池按类选；TS44候选年龄不同；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_history_selection_20261118/TS48/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../168_MN40_V10_HISTORY_SELECTION_2026-11-18.md) |
| `TS49`；12:50:14 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池当前32锚+历史32；最多64 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L0.5-W0-Hist3-SP64-Random32；同前三批96窗口无放回随机选32；加当前32 | e182：93.963 / 90.208 | e300：93.314 / 90.899 | 93.047 / 90.475 | TS48随机选择控制；TS44宽窗口；逐轮test-best；固定300轮；best AA取同轮 | `mn40_v10_history_selection_20261118/TS49/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../168_MN40_V10_HISTORY_SELECTION_2026-11-18.md) |

### 2026-11-19（5组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `TS54`；01:54:49 | MN20 4298/1398／PointMLP／B19；S4780 | CB32／1卡；池— | 134×300=40,200 | H0；MN20-CB32 H0 | e220：92.918 / 90.973 | e300：91.702 / 90.899 | 91.803 / 90.766 | TS55匹配开关；TS51采样；逐轮test-best；固定300轮；best AA取同轮 | `mn40_filter20_dual_20261119/TS54/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../176_MN40_FILTER20_DUAL_LAUNCH_2026-11-19/run.md) |
| `TS55`；05:40:35 | MN20 4298/1398／PointMLP／B19；S4780 | CB32／1卡；池32 | 134×300=40,200 | Hshell-A1-Ks20-Kp20-L0.5-W0；MN20-CB32、固定壳A1/K20/20/.5 | e221：93.062 / 91.057 | e300：92.561 / 91.699 | 92.532 / 91.527 | TS54匹配H0；TS52采样；逐轮test-best；固定300轮；best AA取同轮 | `mn40_filter20_dual_20261119/TS55/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../176_MN40_FILTER20_DUAL_LAUNCH_2026-11-19/run.md) |
| `TS51`；09:17:19 | MN20 4298/1398／PointMLP／B18；S4780 | SH32／1卡；池— | 134×300=40,200 | H0；MN20-SH32 H0 | e95：93.634 / 90.857 | e300：92.775 / 91.049 | 92.711 / 91.062 | TS52匹配开关；TS54采样；逐轮test-best；固定300轮；best AA取同轮 | `mn40_filter20_dual_20261119/TS51/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../176_MN40_FILTER20_DUAL_LAUNCH_2026-11-19/run.md) |
| `TS53`；09:24:19 | MN40 9840/2468／PointMLP／B05；S4780 | SH32／1卡；池独立新鲜CB32 | 307×300=92,100 | Hshell-A1-Ks20-Kp20-L0.5-W0-AuxFreshCB32-BNrestore；分类SH32，额外新鲜CB32只算HIER；辅助BN长期统计恢复 | e214：93.841 / 90.682（原日志213=完成214轮） | e300：93.395 / 90.466 | 92.942 / 90.331 | TS06 H0；TS35同批HIER，额外计算未匹配；逐轮test-best；固定300轮；best AA取同轮 | `mn40_filter20_dual_20261119/TS53/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../176_MN40_FILTER20_DUAL_LAUNCH_2026-11-19/run.md) |
| `TS52`；09:36:35 | MN20 4298/1398／PointMLP／B18；S4780 | SH32／1卡；池32 | 134×300=40,200 | Hshell-A1-Ks20-Kp20-L0.5-W0；MN20-SH32、固定壳A1/K20/20/.5 | e196：93.348 / 91.665 | e300：92.060 / 91.300 | 92.153 / 91.505 | TS51匹配H0；TS55采样；逐轮test-best；固定300轮；best AA取同轮 | `mn40_filter20_dual_20261119/TS52/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json)；[配置](../176_MN40_FILTER20_DUAL_LAUNCH_2026-11-19/run.md) |

### 2026-11-20（2组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `SO-H0`；01:20:40 | SO 11416/2882／PointMLP／SO / PMLP原生，本地适配；S4780 | SH32／1卡；池— | 356×300=106,800 | H0；原生Part–Whole | e211：85.357 / 83.903 | e300：83.657 / 82.606 | 83.796 / 82.501 | SO-HIER；同入口配对；逐轮test-best；固定预算；best AA取同轮 | `scanobject_hier_pair_20261120_A/SO-H0/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `SO-HIER`；01:20:40 | SO 11416/2882／PointMLP／SO / PMLP原生，本地适配；S4780 | SH32／1卡；池32 | 356×300=106,800 | 实际深度cap分支；A1/K20/.5 | e260：86.051 / 84.691 | e300：84.629 / 83.485 | 84.608 / 83.299 | SO-H0；同入口配对；逐轮test-best；固定预算；best AA取同轮 | `scanobject_hier_pair_20261120_A/SO-HIER/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |

### 2026-11-21（3组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `SO-S12-HIER`；14:11:17 | SO 11416/2882／PointMLP／SO / PMLP原生，作者main；S12 | SH32／1卡；池32 | 356×300=106,800 | 实际深度cap；A1/K20/.5 | e210：86.294 / 84.278 | e300：84.837 / 83.663 | 84.639 / 83.317 | SO-S12-H0；同入口配对；逐轮test-best；固定预算；best AA取同轮 | `scanobject_author_s12_20261121_B/SO-S12-HIER/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `SO-S12-H0`；14:11:32 | SO 11416/2882／PointMLP／SO / PMLP原生，作者main；S12 | SH32／1卡；池— | 356×300=106,800 | H0；原生Part–Whole | e209：86.294 / 84.575 | e300：84.074 / 82.271 | 84.042 / 82.515 | SO-S12-HIER；同入口配对；逐轮test-best；固定预算；best AA取同轮 | `scanobject_author_s12_20261121_B/SO-S12-H0/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `MN40-AUTHOR-H0-S4780（MN40作者H0-S4780）`；19:31:41 | MN40 9840/2468／PointMLP／MN40 / PMLP原生，作者launcher；S4780 | SH32／1卡；池— | 307×300=92,100 | H0；训练分片3/0/2/4/1 | e200：93.598 / 90.411 | e300：93.395 / 90.770 | 93.002 / 90.574 | 作者分片组；TS02还混入口；逐轮test-best；固定预算；best AA取同轮 | `mn40_author_s4780_20261121_a/source/classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |

### 2026-11-23（4组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `SO-S4780-HIER`；14:49:58 | SO 11416/2882／PointMLP／SO / PMLP原生，作者main；S4780 | SH32／1卡；池32 | 356×300=106,800 | 实际深度cap；A1/K20/.5 | e208：85.600 / 83.875 | e300：84.074 / 83.109 | 84.289 / 83.159 | SO-S4780-H0；同入口配对；逐轮test-best；固定预算；best AA取同轮 | `scanobject_author_s4780_20261123/SO-S4780-HIER/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `SO-S4780-H0`；14:49:59 | SO 11416/2882／PointMLP／SO / PMLP原生，作者main；S4780 | SH32／1卡；池— | 356×300=106,800 | H0；原生Part–Whole | e253：85.600 / 84.575 | e300：84.629 / 83.411 | 84.733 / 83.476 | SO-S4780-HIER；同入口配对；逐轮test-best；固定预算；best AA取同轮 | `scanobject_author_s4780_20261123/SO-S4780-H0/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `MN40-B05-S4781-HIER-FIFO1`；14:51:47 | MN40 9840/2468／PointMLP／MN40 / PMLP B05；S4781 | SH32／1卡；池32→64 | 307×300=92,100 | 固定壳；前一步detach，轮首清空；A1/K20/.5 | e243：93.801 / 90.783 | e300：93.355 / 90.545 | 93.039 / 90.561 | 同seed H0；复验TS44整套方案；逐轮test-best；固定预算；best AA取同轮 | `mn40_b05_fifo64_s4781_20261123/hier/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `MN40-B05-S4781-H0`；14:51:48 | MN40 9840/2468／PointMLP／MN40 / PMLP B05；S4781 | SH32／1卡；池— | 307×300=92,100 | H0；沿TS06数据顺序 | e282：93.882 / 90.853 | e300：93.801 / 90.816 | 93.254 / 90.507 | FIFO1/CURRENT32；后者跨主机；逐轮test-best；固定预算；best AA取同轮 | `mn40_b05_fifo64_s4781_20261123/h0/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |

### 2026-11-24（8组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `DeLA-v1-AUTHOR-S4780（DeLA v1原版从头训练）`；约01:50；精确秒缺 | SO 11416/2882／DeLA v1／SO / DeLA作者train.py；S4780 | SH32／1卡；池— | 356×250=89,000尝试；原FP16有效更新总数缺 | 分类+坐标MSE；无显式HyCoRe Part–Whole | e222：89.590 / 88.290 | e250：89.380 / 88.440 | 89.243 / 88.240 | 新DeLA-H0同时换头/加HLoss/评估流，不隔离单项；作者逐轮test峰值；附固定e250 | `de_la_v1_author_train_scanobjectnn_s4780_20261124_0150/Matrix-ASC-DeLA-0f427c3/ScanObjectNN/output/log/01`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `DG-S12-H0`；01:55:30 | SO 11416/2882／DGCNN／SO / DGCNN原生，作者launcher；S12 | SH32／1卡；池— | 356×250=89,000 | H0；原生Part–Whole | e248：81.992 / 79.429 | e250：80.916 / 77.900 | 81.607 / 78.876 | DG-S12-HIER；同卡不同时间；逐轮test-best；固定预算；best AA取同轮 | `hycore_author_dgcnn_scanobjectnn_s12_20261124_fresh/classification_ScanObject/checkpoints/PointNet-toyproof-Offv_dgcnn_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `DG-S12-HIER`；14:14:22 | SO 11416/2882／DGCNN／SO / DGCNN原生；S12 | SH32／1卡；池32 | 356×250=89,000 | 实际深度cap；A1/K20/.5 | e242：81.888 / 79.475 | e250：81.055 / 78.464 | 81.263 / 78.720 | DG-S12-H0；首步验收匹配；逐轮test-best；固定预算；best AA取同轮 | `scanobject_dgcnn_hier_20261124_1415/DG-S12-HIER/main/checkpoints/PointNet-toyproof-Offv_dgcnn_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `DG-S4780-H0`；14:14:24 | SO 11416/2882／DGCNN／SO / DGCNN原生；S4780 | SH32／1卡；池— | 356×250=89,000 | H0；原生Part–Whole | e223：82.616 / 79.749 | e250：81.263 / 78.602 | 80.940 / 78.131 | DG-S4780-HIER；同seed配对；逐轮test-best；固定预算；best AA取同轮 | `scanobject_dgcnn_hier_20261124_1415/DG-S4780-H0/main/checkpoints/PointNet-toyproof-Offv_dgcnn_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `DG-S4780-HIER`；14:19:51 | SO 11416/2882／DGCNN／SO / DGCNN原生；S4780 | SH32／1卡；池32 | 356×250=89,000 | 实际深度cap；A1/K20/.5 | e219：81.298 / 78.997 | e250：80.222 / 77.598 | 80.326 / 77.458 | DG-S4780-H0；同seed配对；逐轮test-best；固定预算；best AA取同轮 | `scanobject_dgcnn_hier_20261124_1415/DG-S4780-HIER/main/checkpoints/PointNet-toyproof-Offv_dgcnn_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `MN40-B05-S4781-HIER-CURRENT32`；21:38:30 | MN40 9840/2468／PointMLP／MN40 / PMLP B05；S4781 | SH32／1卡；池32 | 307×300=92,100 | 固定壳；关FIFO；A1/K20/.5 | e282：94.084 / 91.070 | e300：93.841 / 90.565 | 93.371 / 90.569 | 同seed H0/FIFO1；主机不同；逐轮test-best；固定预算；best AA取同轮 | `mn40_b05_s4781_current32_20261124/hier-current/main`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `SO-S12-H0-PAPER-INTRA`；21:50:08 | SO 11416/2882／PointMLP／SO / PMLP作者main；S12 | SH32／1卡；池— | 356×300=106,800 | H0；原生层次权重.001→.01和margin500/N→1000/N一起改 | e243：85.010 / 83.664 | e300：83.727 / 82.843 | 83.664 / 82.761 | SO-S12-H0；跨主机；只回答两项合改；逐轮test-best；固定预算；best AA取同轮 | `scanobject_author_paper_intra_s12_20261124/main/checkpoints/PointNet-toyproof-Offv_pointmlp_hycore`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |
| `MN40-AUTHOR-TRAIN-ORDER-S4780（MN40作者H0训练分片顺序组）`；21:50:10 | MN40 9840/2468／PointMLP／MN40 / PMLP原生，作者launcher；S4780 | SH32／1卡；池— | 307×300=92,100 | H0；训练分片改1/4/3/2/0，测试仍0/1 | e182：93.679 / 89.369 | e300：93.112 / 90.553 | 92.970 / 90.836 | 原作者H0；另有主机/Python补丁差异；逐轮test-best；固定预算；best AA取同轮 | `mn40_author_train_order_s4780_20261124/source/classification_ModelNet40/checkpoints/Hype_PointNet-Offv_pointmlp_hycore_var-4780`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_log_audit.json)；[配置](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md) |

### 2026-11-25（2组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `DeLA-H0`；18:57:22 | SO 11416/2882／DeLA v1／DeLA+共享双曲头+同对象Part–Whole+CE/坐标MSE/.02HLoss；S4780 | SH32／1卡；池— | 356×250=89,000尝试；88,961有效/39 AMP跳步 | H0，保留同对象Part–Whole；建立对应底座H0 | —未测 | e250：87.474 / 86.092 | —未测（协议无该读数） | DeLA-HIER直接H0；与原版同时换头/HLoss/评估流；固定e250，只测一次官方test；无best/末10测量 | `de_la_pw_dmr_hier_20261125/formal_seed4780/h0/work`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/latest_8000_evidence.json)；[配置](../194_DELA_H0_HIER_PAIR_2026-11-25/run.md) |
| `DeLA-HIER`；18:57:22 | SO 11416/2882／DeLA v1／DeLA+共享双曲头+同对象Part–Whole+CE/坐标MSE/.02HLoss；S4780 | SH32／1卡；池当批32 | 356×250=89,000尝试；88,961有效/39 AMP跳步 | 实际深度cap/A1/P512/K20/20/T50/两项λ=.5；只开启跨对象HIER实际深度分支 | —未测 | e250：88.237 / 86.899 | —未测（协议无该读数） | DeLA-H0，固定终点开关；同AMP策略但逐轮跳步轨迹不同；固定e250，只测一次官方test；无best/末10测量 | `de_la_pw_dmr_hier_20261125/formal_seed4780/hier/work`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/latest_8000_evidence.json)；[配置](../194_DELA_H0_HIER_PAIR_2026-11-25/run.md) |

### 2026-11-26（4组）

| 编号／原运行名；启动 | 数据／骨干／底座；seed | 分类batch／卡数；HIER候选池 | 步×轮＝总预算 | HIER配方与实际改动 | best轮：OA／同轮AA | 末轮OA／AA | 末10 OA／AA | 对照与评估边界 | 原运行目录／出处 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `SO-S12-HIER-L0.1`；00:25:00 | SO 11416/2882／PointMLP／作者PointMLP+HyCoRe原生c1 Part–Whole；S12 | SH32／1卡；池当批32 | 356×300=106,800 | 实际深度cap/A1/P512/K20/20/T50/两项λ=0.1；样本项和代理内部项同时.5→0.1 | e284：85.566 / 84.249 | e300：84.559 / 83.132 | 84.445 / 83.092 | SO-S12-H0及SO-S12-HIER(.5)；跨同型主机残差，胜出权重未跨seed复验；逐轮test-best；预定e291–300 OA主指标 | `scanobject_weights_cb_20261126/runs/SO-S12-HIER-L0.1`；[成绩](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/weights_final.json)；[配置](../196_SCANOBJECTNN_WEIGHT_CB_PAIR_2026-11-26/run.md) |
| `SO-S12-HIER-L1.0`；00:25:00 | SO 11416/2882／PointMLP／作者PointMLP+HyCoRe原生c1 Part–Whole；S12 | SH32／1卡；池当批32 | 356×300=106,800 | 实际深度cap/A1/P512/K20/20/T50/两项λ=1.0；样本项和代理内部项同时.5→1.0 | e188：85.253 / 83.742 | e300：84.525 / 82.922 | 84.087 / 82.567 | SO-S12-H0及SO-S12-HIER(.5)；跨同型主机残差，胜出权重未跨seed复验；逐轮test-best；预定e291–300 OA主指标 | `scanobject_weights_cb_20261126/runs/SO-S12-HIER-L1.0`；[成绩](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/weights_final.json)；[配置](../196_SCANOBJECTNN_WEIGHT_CB_PAIR_2026-11-26/run.md) |
| `DG-S4780-CB32-HIER`；00:32:39 | SO 11416/2882／DGCNN／作者DGCNN+HyCoRe原生Part–Whole；S4780 | CB32=15类×2+轮换2类各加1／1卡；池当批32 | 356×250=89,000 | 实际深度cap/A1/P512/K20/20/T50/两项λ=.5；只开启跨对象HIER实际深度分支 | e218：81.437 / 79.180 | e250：80.465 / 78.354 | 80.569 / 78.406 | DG-S4780-CB32-H0直接开关；SH32-HIER用于采样比较；逐轮test-best；预定e241–250 OA主指标 | `scanobject_weights_cb_20261126/runs_verified/DG-S4780-CB32-HIER`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/latest_8000_evidence.json)；[配置](../196_SCANOBJECTNN_WEIGHT_CB_PAIR_2026-11-26/run.md) |
| `DG-S4780-CB32-H0`；07:51:08 | SO 11416/2882／DGCNN／作者DGCNN+HyCoRe原生Part–Whole；S4780 | CB32=15类×2+轮换2类各加1／1卡；池— | 356×250=89,000 | H0，保留同对象Part–Whole；建立对应底座H0 | e247：81.783 / 79.690 | e250：81.124 / 79.134 | 81.065 / 79.015 | DG-S4780-CB32-HIER直接H0；SH32-H0用于采样比较；逐轮test-best；预定e241–250 OA主指标 | `scanobject_weights_cb_20261126/runs_verified/DG-S4780-CB32-H0`；[成绩](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/latest_8000_evidence.json)；[配置](../196_SCANOBJECTNN_WEIGHT_CB_PAIR_2026-11-26/run.md) |

## 两条正式启动后停止的B17验证协议运行

它们分别由TS39/41全量test-best重跑替代，旧目录与文件保留。H32旧队列只通过短测、没有正式启动，不再计一条正式训练。初次调度在GPU前失败的目录也不算训练。

| 原运行名／替代关系 | 实际启动；状态 | 数据／底座／seed | 分类batch／卡；实际与原计划预算 | HIER池／配方 | 评估选模与实测分数 | 原目录／出处 |
| --- | --- | --- | --- | --- | --- | --- |
| B17-H64-VS984 → TS41 | 11-15 13:35；完成e2后停止 | MN40 train8856/val984/test2468；PointMLP B05几何；训练4780/划分22 | 2×SH32累积；1卡；138×2=276已完成，原计划138×300=41400 | 实时64；Hshell A1/K20/20/λ.5/P512 | 验证OA选模；e1–2验证读数未在所读摘要提供；无最终test/末10 | `mn40_v10_b17_sh_pool_20261115_r2/H64/main`；`c1ed4cc`；[164原记录](../164_MN40_V10_SH_POOL32_64_2026-11-15.md) |
| B17-H0-VS984 → TS39 | 11-15 13:36；完成e2后停止 | MN40 train8856/val984/test2468；PointMLP B05几何；训练4780/划分22 | 2×SH32累积；1卡；138×2=276已完成，原计划41400 | HIER关；池无 | 验证OA选模；e1–2验证读数未在所读摘要提供；无最终test/末10 | `mn40_v10_b17_sh_pool_20261115_r2/H0/main`；`c1ed4cc`；[164原记录](../164_MN40_V10_SH_POOL32_64_2026-11-15.md) |

## 仅作参照的TS06与TS23

两条早于本次主线，不计52组，也不展开其前期设计史。TS06的94.327%只用于完整MN40历史目标；MN20和SO各用同协议H0。

| 参照 | 配方／seed／数据；预算 | best轮：OA／同轮AA | e300 OA／AA | 末10 OA／AA | 用途／原目录与出处 |
| --- | --- | ---: | ---: | ---: | --- |
| TS06 | B05／H0／S4780／9840-2468；307×300=92,100 | e184：94.327 / 91.415 | 93.517 / 91.141 | 93.027 / 90.619 | 完整MN40 B05 H0与94.327%历史目标；`orig_c01_cap_gs_halfmargin_20261020_160111/gs_half_margin`；[197原表](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json) |
| TS23 | B13／Hshell-A1-Ks10-Kp10-L0.1-W0／S4780／9840-2468；200×300=60,000 | e262：94.003 / 91.828 | 93.720 / 91.262 | 93.351 / 91.237 | B13 CB32低权重HIER；TS27–29代理消融的直接对照；`b05_cb32_a1_followup_20261026_020947/k10p10w01/main`；[197原表](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json) |

## 7个作者发布权重复评条件（均非本地训练）

这些是其他骨干作者已训练checkpoint的推理条件，不是7次训练，也无可填的本地训练best/末轮/末10。DeLA使用各自数据集微调权重；上述七项中point2vec的5项共享同一ShapeNet预训练后MN40微调权重，只换FPS或推理精度。HyCoRe论文对应的作者发布checkpoint在所读记录中未取得可核实资产，不能用我们历史TS01或A3冒充。[复评原记录与完整身份](../184_V9_V10_BASELINE_AND_HIER_ROADMAP_2026-11-24/report.md)；[197核验条件表](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/non_ts_experiments.md)。

| 推理条件／日期 | 数据与权重身份 | 单卡推理batch／seed／精度 | 实测OA／AA | 对照意义／原日志 |
| --- | --- | --- | ---: | --- |
| DeLA作者权重-MN40；11-24 01:06–01:15 | MN40 test2468；DeLA v1 0f427c3；checkpoint SHA36019c4e… | B32；作者FPS；FP16单pass | 94.000 / 92.250 | 作者日志一致；与TS06非同模型匹配；`de_la_v1_author_checkpoint_20261124_0106/modelnet40_eval.log` |
| DeLA作者权重-SO；11-24 01:06–01:15 | SO test2882；DeLA v1；checkpoint SHA0db19dcf… | B32；作者FPS；FP16单pass | 90.420 / 89.240 | 原作者90.46/89.30；约少1正确样本；`de_la_v1_author_checkpoint_20261124_0106/scanobjectnn_eval.log` |
| point2vec-FPS0-FP16；11-24 01:31–01:33 | MN40 test2468；ShapeNet预训练→MN40微调e125；checkpoint SHA6c1d4884… | B32；FPS seed0；FP16 | 94.286872 / 91.903488（2327正确） | 预先主条件；`point2vec_author_checkpoint_20261124_0127/eval.log` |
| point2vec-FPS0-FP32；同上 | 同一微调checkpoint | B32；FPS0；FP32 | 94.286872 / 91.903488（2327） | 精度控制，与FP16一致；同目录`fps_sensitivity.log` |
| point2vec-FPS1-FP16；同上 | 同一微调checkpoint | B32；FPS1；FP16 | 94.124797 / 92.011628（2323） | 仅换推理采样；`fps_sensitivity.log` |
| point2vec-FPS12-FP16；同上 | 同一微调checkpoint | B32；FPS12；FP16 | 94.367909 / 91.861047（2329） | 仅换推理采样；`fps_sensitivity.log` |
| point2vec-FPS4780-FP16；同上 | 同一微调checkpoint | B32；FPS4780；FP16 | 94.327391 / 91.844767（2328） | 正确数与TS06相同，不是训练seed复验；`fps_sensitivity.log` |

四个FP16采样条件跨度.2431个百分点、净6个测试样本，描述同一权重的推理取点敏感性；不能估计训练seed方差，也不能与从头训练平均成绩混算。

## 关键诊断与短测：不计正式完整训练

下表只列会影响52组解释的关键证据，不把CPU重放、首批短测或固定权重复评扩成新的训练组。机制数值只在声明的模型、epoch和输入范围内成立。

| 诊断／日期 | 已有读数与支持的假设 | 对结论的限制／所缺直接对照 | 原始出处 |
| --- | --- | --- | --- |
| TS23冻结代理/早期优化诊断；早于主线，按需追溯 | A1与较低K/λ给出后续代理消融起点；A0/A1同时改LR及eps。固定LR eps短测e10越cap500/512→0/512 | train-only数值/优化线索，不等于300轮OA改进，也不证明真实形态；只改变代理cap的TS29才检验该局部改动 | [V9机制D03/D04](../93_V9_EXPERIMENT_ATLAS_2026-10-26/report.md)；[88 eps审查](../88_B05_EPS_ONLY_10E_2026-10-25/report.md) |
| TS27–29及后续生产前合同；11-12起 | 共享初值、P256取原前256、初始化缩放、cap以及代理更新路径已有短测验收 | 短测只确认实际改动与可运行性；分类结果用正式300轮，不把短测另计一组 | [160代理消融](../160_MN40_TS23_PROXY_ABLATIONS_2026-11-12.md)；[197身份表](../197_CLASSIFICATION_REPORT_DRAFT_2026-11-26/ts_inventory.json) |
| SH/CB SAMPLER与错误审查；11-18 | TS06每对象300轮曝光295–300，CB31–820（跨类极值）；SH独对55、CB独对43、共同错97，净12；近邻未见稳定少采形态联系 | 4000次条件重放及多重校正未检出类内异常分散；不能归为坏seed或证明不平衡无关；要隔离采样需同预算/随机流/负例规则匹配 | [171逐对象审查](../171_MN40_TS06_TS30_SAMPLING_ERROR_AUDIT_2026-11-18/report.md)；[173频次/seed补查](../173_MN40_CB4780_EXPOSURE_AND_SEED_2026-11-18/report.md) |
| 两个seed的FIFO关系供给；11-26现成日志盘点 | TS35→44 active/步127.01→151.02；4781 CURRENT32→FIFO1为122.36→130.20，两FIFO缓存参与都约84.1% | 关系更多而分类增量跨seed反向；active是有重复的非碰撞未满足hinge抽样数，不是独特/真实形态关系；缺等年龄/重编码候选对照 | [198机制字段与13组汇总](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/mechanism_inventory.md)；[紧凑JSON](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/mechanism_evidence.json) |
| SO实际深度、代理与hinge；11-26现成日志盘点 | 作者PMLP两seed末段三约束同时达标率94.885%/95.200%，最后批代理近cap均0；分类末段一正一负。DG SH→CB active142/141→112每步，增量仍负 | SO深度只记每轮最后批；映射后近cap不等于原代理参数越cap，不能说全训练分布全贴壳或代理回内足以带来OA增益 | [198机制盘点](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/mechanism_inventory.md)；[字段与来源JSON](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/mechanism_evidence.json) |
| DeLA实际代码/数据/AMP轨迹；11-26审查 | 两组均89000尝试/88961有效/39跳步；逐轮跳步轨迹不同；冻结代码和数据身份可核 | 同策略配对仍有效；不能称每步有效更新完全一致，不能把+.763全归因几何。原版→新H0多改动，缺同双曲头关HLoss等匹配训练 | [DeLA生产证据](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/provenance_dela.json)；[代码身份](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/provenance_code.json) |
| DeLA早中末首批几何；11-26现成日志 | e1两组whole/part深度约.845/.840；e250 whole约6.206/6.213、part1.531/1.514；whole尾部embedding范数p95=.996 | 每轮首批更新前读数，非完整训练分布。共同边界现象不能归因HIER，也不能单独解释原版→新H0下降；入球初始尺度.0761不保证训练后远离边界 | [198几何表](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/mechanism_inventory.md)；[首批证据JSON](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/mechanism_evidence.json) |
| DeLA固定e250逐对象复评；11-26 18:02 | 两原混淆矩阵逐元素复现；同输入指纹、参数/BN前后不变、0优化。共同对2451、H0独对70、HIER独对92、共同错269；净22 | 诊断不参与选模或改成绩；HDF扰动行不是已核实独立原始物体，不作独立对象显著性检验。首次构建失败留档，成功复核用既有二进制 | [逐对象说明](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/dela_object_diagnostic.md)；[精简证据](../198_CLASSIFICATION_PREPOWER_AUDIT_2026-11-26/dela_object_diagnostic_summary.json) |

## 附录覆盖与尚存资料边界

- 52条完整训练：TS29组、历史非TS17组、194/196最新6组；每条唯一ID、启动日和证据都写入coverage.json。两条B17验证协议中止训练、TS06/23参照及7个权重推理条件分别计数。
- 历史逐轮test-best、预先末10主指标、固定e250和checkpoint单次复评保留各自口径。新DeLA没有best/末10测试；DeLA原版缺精确启动秒和有效更新总数；B17两条e2停止组在所读摘要中缺逐轮验证分数。
- 新训练都保留同对象Part–Whole的对照原则；DeLA原版是高OA参照，本身无新增HyCoRe Part–Whole。没有独立形态/语义验证，不把代理回内、loss满足或新增判对样本当作真实形态层次证明。
