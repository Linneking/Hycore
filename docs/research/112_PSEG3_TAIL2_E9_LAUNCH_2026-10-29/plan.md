# 新组D：保留欧氏Conv9，仅替换最后两层

日期：2026-10-29。用户已审阅上一条结构候选并明确要求“记下配置，与标记B比较，检查GPU剩余时间并挂上”。本次仅授权一个新组D，等待GPU1原B正常完成；不停止、不改动A/B/C，不新增其他矩阵。

## 结构与标记B的比较

公共前端仍为原分割STN、三阶段局部192、global1024、类别one-hot16→64、融合1280→256及Conv8/BN8/Leaky。每物体2048点，维持逐点输出。M层使用原HyCoRe的_weight/_bias/EXP/transp0完整参数化，无输出scale等额外改动。

```text
B（已在运行，原v3）：
融合 → Conv8/BN8/Leaky → H256
     → EXP → M(256→128) → M(128→50) → CE
                    （dp1关闭，无Conv9/BN9/Leaky）
H256 → whole/part平均 → EXP → Rcontr + Rhier

D（本次新组，按“前面全保留”）：
融合 → Conv8/BN8/Leaky → dp1(.5)
     → Conv9(256→256)/BN9/Leaky → 新H256
     → EXP → M(256→128) → M(128→50) → CE
新H256 → whole/part平均 → EXP → Rcontr + Rhier
```

相对原DGCNN分割，D保留Conv9/BN9/Leaky和dp1；只将Conv10/Conv11换为两层M，删除dp2、BN10及Conv10后的Leaky。相对标记B，D同时恢复dp1和欧氏Conv9块，并把正则输入移到该块之后，属于组合结构对照，不将成绩差归因为单独BN/激活或单独dp1。

CPU构建实测：B可训练参数1,394,235；D为1,460,283，比B增加66,048（Conv9权重65,536及BN9的512个仿射参数），比原DGCNN分割少78。BN模块共14个，每次训练前传各更新一次。

新组独立`hycore_seg_v4`路径，旧v1/v2/v3不改。实际两M沿用`emb2`/`mlr`名字，没有等维M1；监控`has_m1=False`仅表示无等维M，与是否恢复欧氏Conv9区分。

## 共同协议

- fresh随机初始化，seed4780，不加载teacher、B权重或短测权重。
- 相同trainval14007/test2874成员、预处理及每轮test Instance mIoU选best，固定200轮；同既有协议保留重复ID规则。
- shuffle32/drop_last、2048XYZ/k40、eval16、workers8，每完整轮437更新。
- RSGD .1→.001，cosine200、momentum.9、WD1e−4；plain 50类CE，无label smoothing。
- `.01*Rcontr+.01*Rhier`，gamma=1000/Npart、delta4、part200–600；严格batch异类样本均匀负例，可复用，独立RNG；无异类仅跳contrast。
- 不做全参数梯度范数裁剪，记录G>1及有限性；保留Geoopt数值球投影。
- 每25更新及每轮首批记录几何，每200及每轮首批拆CE/两正则梯度；保留point/whole/part EXP、模长、深度差、满足率、M输出与参数梯度/位移。新增pre_conv9与新H，区分恢复欧氏块前后的尺度。

## 初始化与验收

保留共同前端与B的emb2/mlr初值配对；恢复原DGCNN的Conv9/BN9初始化，不消耗改变共享M的随机数。通过CPU结构/真实梯度合同、完整合成训练入口及保存恢复检查。

GPU仍忙时，启动器仅等待旧B完整完成验收和GPU空闲；CPU检查不使用GPU。获得空闲卡后，先在独立clean train/val上做真实batch32/N2048两轮各两步无test短测，验收通过后重新初始化正式fresh200。新run目录不复用，记录源提交、配置、seed、GPU与开始时间；旧结果不改。

## 本次只读资源检查

2026-10-29 20:56:44北京时间：A e138/step350，B e141/step300，C e134/step225；三卡均有训练进程、显存约21GiB。最近10轮中位耗时A104.81s、B102.64s、C108.11s。估算剩余A108.6min/B101.5min/C119.8min，预计22:45/22:38/22:57左右，属于近期开销外推而非保证完成时间。新组目标GPU1，必须等待旧B正常完成及nvidia-smi空闲检查，不能据预计时间直接抢卡。

实施和排队状态见同目录execution.md；此计划本身不是已启动或已通过GPU短测的证据。
