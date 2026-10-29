# D/E：恢复欧氏 Conv9 后，两项正则的匹配对照

用户已审阅并同意：D、E 不再对照 dp1；两组都完整保留 dp1、Conv9、BN9 和 LeakyReLU，仅关闭 E 的两项正则。D 排 GPU0、E 排 GPU1；GPU2 不增加后继任务。既有 A/B/C 按原配置继续。

## 结构与唯一配方差异

```text
D/E 相同主干（原 DGCNN partseg，含类别 one-hot）：
  多尺度逐点特征 + 广播全局特征 + 广播类别特征
  → concat1280
  → Conv8(1280→256) → BN8 → LeakyReLU → dp1(p=.5)
  → Conv9(256→256) → BN9 → LeakyReLU
  → H[B,N,256]
  → EXP → M(256→128) → M(128→50) → 逐点 CE

共同观察支路：H → whole/part 欧氏平均 → EXP → 几何与负例统计
D 训练损失：CE + .01 Rcontr + .01 Rhier
E 训练损失：CE；alpha=beta=0
```

| 对比项 | D / GPU0 | E / GPU1 |
|---|---|---|
| dp1、欧氏 Conv9/BN9/LeakyReLU | 保留 | 相同 |
| 最后两层完整官方 Möbius 映射 | 256→128→50 | 相同 |
| 尾部 dp2、BN10、LeakyReLU | 删除 | 相同 |
| 训练用对比/层次正则系数 | .01 / .01 | 0 / 0 |
| 初始模型、种子、数据和优化协议 | 配对 | 相同 |

每组可训练参数 1,460,283。相同种子下，两组完整初始 state（含 BN 状态）、构造后的全局随机数状态和保留的 M 参数必须逐项一致。E 是无正则双曲头对照，不是原版 DGCNN；D−E 检验两项正则的联合效果，不能分别归因于其中一项。

## 保持的训练协议

seed4780；batch32、shuffle/drop_last、workers8；2048点、k40；原类别 one-hot；RSGD、momentum .9、weight decay1e−4；LR .1→.001、200轮余弦；FP32，无总梯度范数裁剪，只记录范数及大于1的比例，保留 Geoopt 数值球投影。fresh200，不继承短测或旧实验权重。无 teacher、无输入 checkpoint。

维持现有可比较的 trainval14007/test2874 和逐轮 test Instance mIoU 选模协议；不更换数据、batch 组织、CE 或选模规则。现有跨集合26个共有身份及 test 选模限制沿用原记录。

D 的正则仍按论文从新 H 做欧氏平均再 EXP，绕过两层 M。part200–600，层次 margin1000/Npart、对比 margin4；每个 anchor 在当前 batch 的异类样本中均匀抽取负 part、允许复用。part 与负例各自使用独立且可保存的 RNG，不消耗 dropout/数据 RNG。

E 也观察同一支路的 whole/part 模长、贴边率和负例合法性，以便解释训练趋势；这些量必须明确标为仅诊断，不参与其训练反传。E 的训练正则贡献必须严格为0；不得把仅诊断量的潜在梯度写成 E 的有效正则梯度。每25更新及每轮首批记录几何，每200更新及每轮首批记录有效分项梯度；每批仅一次模型前向，BN 每批更新一次。

## 执行与验收

使用独立 v5 训练入口，复用冻结 v4 模型数学路径；旧 v3/v4 源码及全部旧结果保留。先 CPU 验证完整初值/随机数配对、E 的 CE-only 梯度、保存恢复和两组完整合成2×2入口，再提交推送并部署同提交独立服务端工作树。

已核对原 D/GPU1 仅 waiting_for_predecessor，无 smoke/main 或训练子进程；2026-10-29 21:22:46 北京时间停止该等待进程，原目录保留，追加 cancellation.json。此次使用全新 D、E 目录，不复用旧队列目录。

D 等旧 A/GPU0 成功完成，E 等旧 B/GPU1 成功完成；必须同时满足旧 queue complete、main status complete、acceptance complete，再检查 nvidia-smi 完全空闲。之后各自进行真实 batch32/N2048 无test 2轮×2更新短测；通过结构、有限值、BN、保存/RNG/调度及对应正则状态验收后，重新初始化 fresh200。任何失败均停止该后继，不自动抢卡或替换原任务。

本方案不停止 GPU2 上的旧 C；用户仅指定 GPU2 不增加新任务。
