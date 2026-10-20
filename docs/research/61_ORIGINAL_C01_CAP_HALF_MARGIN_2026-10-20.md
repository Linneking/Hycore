# 原版HyCoRe c=0.1＋cap：margin减半补充实验

用户于2026-10-20直接要求在新服务器补做一条完整实验。以已完成的`ORIG-B0-32-C01-CAP23-S4780`为对照，仅把Rhier的gamma1000改为500、Rcontr的delta4改为2；不改变损失系数。

| 项目 | 新实验：ORIG-B0-32-C01-CAP23-MHALF-S4780 |
| --- | --- |
| 曲率 | 全链c=0.1，包括四处模型流形与距离损失 |
| cap | 最终whole/part共享表示的切空间模长2.3，原点深度上限约4.6 |
| 梯度缩放 | 无InverseMetric hook；保留原RSGD及norm1梯度裁剪 |
| margin | Rhier为500/N；Rcontr为2 |
| 损失 | CE＋0.01Rcontr＋0.01Rhier，无HIER/代理 |
| seed / 初始化 | 4780；从头初始化，匹配已完成原版cap组 |
| 训练 / 测试 | 官方9840训练、无独立验证、2468 test每轮选best；不提前停止 |
| batch / 预算 | 单卡batch32 shuffle/drop-last，每轮307步，共300轮 |
| 其余协议 | 原版数据增强、持续随机流、part先whole、alias及共享BN；workers8/eval16 |
| 优化器 | RiemannianSGD，momentum0.9、weight decay0.0002、LR0.1至0.005余弦 |

新版本位于`inter_hierarchy_MN40/hycore_c01_halfmargin_20261020/`。原版源码仍产生1000/N及4，由新版本的criterion包装仅将两项阈值乘0.5后调用原损失；不缩放整个损失或正则系数，不额外消耗随机数。

先完成CPU数学/梯度及初始化合同，再进行独立目录的2轮×2步训练数据限定GPU短测。短测不读取test，检查初始权重SHA与已完成cap组一致、训练数据顺序/文件/标签一致、全链曲率与权重有限后，从头启动正式300轮。运行目录、源码快照、物理GPU UUID、seed、时间和依赖身份分别记录，保留旧结果和正在训练的V8。

本组与此前原版cap组比较两个margin的联合修改效果；与V8去hook组仍有seed和随机流等协议差别。当前为单seed补充，不把一次成绩差异视为统计显著性证据。

状态：已获直接授权，正在验收和启动。
