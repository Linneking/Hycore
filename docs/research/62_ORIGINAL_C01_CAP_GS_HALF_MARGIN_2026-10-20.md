# 原版HyCoRe：c0.1、cap、梯度缩放、margin减半

用户2026-10-20直接授权在8001补充一条完整300轮训练。规范名：`ORIG-B0-32-C0.1-CAP2.3-GS1-M500_2-TS-S4780`。

| 项目 | 配置 |
| --- | --- |
| 代码谱系 | 原版HyCoRe train/validate、持续随机流、原数据增强及alias/part先whole/共享BN |
| seed | 4780 |
| 数据和选模 | 官方9840训练、无独立val、2468 test逐轮OA选best、完整300轮、不提前停止 |
| batch | 单卡shuffle32、drop-last、307次更新/轮；workers8、eval16 |
| 几何 | 四处网络流形及损失c0.1，最终shared whole/part切空间cap2.3 |
| GS | 启用HIER逆度量反向缩放，whole/part最终共享嵌入各一次；没有HIER/代理 |
| margin | Rhier 500/N、Rcontr 2；各自系数仍0.01 |
| 优化 | 原RSGD .1→.005余弦、momentum.9、wd.0002、全模型grad norm裁剪1 |

相对[61无GS组](61_ORIGINAL_C01_CAP_HALF_MARGIN_2026-10-20.md)，本组仅开启GS；不同机器的算子与运行环境可能留下残余差异。保持初始模型SHA及训练文件、标签和逐行顺序一致：跨机器目录枚举顺序不同，新入口按对照已记录的shard顺序显式拼接，避免隐含改变训练对象顺序；不改数据内容。

独立入口`inter_hierarchy_MN40/hycore_c01_gs_halfmargin_20261020/`。CPU检查后冻结源码，后台仅等待一张完全空闲GPU。空闲卡连续两次检查通过后，先做2轮×2步仅train短测；初始化/训练身份、曲率、hook与有限权重通过后，从同seed重新初始化正式300轮。保留所有旧结果与其他用户进程。

当前资源检查：8001四张GPU均被其他用户占用，按用户授权提交后台等待，不抢占。实际源码、进程和后续状态另行记录。
