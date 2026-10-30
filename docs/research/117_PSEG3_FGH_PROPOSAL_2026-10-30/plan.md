# 用户确认的双曲 F/G/H：逐层配置

状态：2026-10-30用户确认G正则分叉在Conv9前后，明确要求“挂起实验”，已授权按下列配置实施和后台排队。实际执行状态见[118](../118_PSEG3_FGH_LAUNCH_2026-10-30/execution.md)。本文F/G均指本轮用户提出的双曲实验，覆盖此前未实施的纯欧氏候选同名建议。

## G分叉已明确

F明确在Conv9前分正则；H明确在Conv9后分正则；用户后续明确“G分支放在conv9前”，因此G在dp1后、Conv9前分叉。

- 原先若沿D在Conv9后分叉会重复D；该歧义已由用户澄清。
- 新F/G/H与已有D组成dp1开关×分叉前后的2×2。确认分叉不自动表示已执行或已分配GPU。

| dp1 | 正则Conv9前 | 正则Conv9后 |
|---|---|---|
| 删除 | F | H |
| 保留p=.5 | G | 已有D |

## 共同模块定义

```text
原前端：Transform-Net → 动态EdgeConv → 多尺度局部特征
        全局1024 + 类别64 广播，与局部192拼接为1280
      → Conv8(1280→256,bias=False) → BN8 → LeakyReLU(.2)
      → T[B,N,256]

L9：Conv1d(256→256,kernel=1,bias=False)
    → BN9(256) → LeakyReLU(.2)
    整块不带Dropout

共同双曲输出头：逐点EXP(c=1)
    → M1(256→128) → M2(128→50) → 逐点plain CE
```

Conv9原实现是1×1 Conv1d，与逐点Linear256→256数学等价。为保持与已有D一致，建议沿用Conv1d实现；若写成Linear，BN仍需按通道256处理[B,256,N]，不能悄悄改变归一化轴。本轮M1/M2为两层输出头的新编号；没有恢复旧A中的同宽M256→256。

偏置澄清：原社区Conv9已经bias=False，D也如此；当前写bias=False是在保持既有规格，不是新增消融。BN9仍为affine=True，保留可训练gamma/beta；两层M仍沿用官方_weight/_bias参数化，不因Conv9的bias=False而删除M参数。

三组均没有dp2、BN10或其LeakyReLU；M层内部也无额外BN、激活、Dropout。若无额外参数，三组可训练参数均为1,460,283，dp1开关不改变参数量。

## F：去dp1，正则在L9之前

```text
T256 ──┬── 正则：选part / whole-part欧氏平均 → EXP → Rcontr + Rhier
       └── L9 → EXP → M1(256→128) → M2(128→50) → CE
```

相对B，仅在主支、EXP之前增加原欧氏L9；正则仍取B的T。正则不直接经过L9，也不经过两层M；它通过T更新更早的共享编码/融合层。CE经过全部主支。

## G：保留dp1，正则在L9之前（已确认）

```text
T256 → dp1(.5) → U256
U256 ──┬── 正则：选part / whole-part欧氏平均 → EXP → Rcontr + Rhier
       └── L9 → EXP → M1(256→128) → M2(128→50) → CE
```

G与F仅差dp1开关；与D仅差正则分叉位置。

## H：去dp1，正则在L9之后

```text
T256 → L9 → V256
V256 ──┬── 正则：选part / whole-part欧氏平均 → EXP → Rcontr + Rhier
       └── EXP → M1(256→128) → M2(128→50) → CE
```

H与F主支完全一致，仅正则取点前后不同。H与已有D仅差dp1删除。H正则直接回传L9及更早共享层，但依然绕过两层M。

## 共同训练设置（已确认，沿用既有配方）

全部fresh、seed4780，保留共享参数与后两M的配对初始化；不新改初始化算法，不加载分类/旧实验训练权重。两正则均开启：CE+.01Rcontr+.01Rhier；part在各自指定的T/U/V上按特征近邻选择，200–600点，gamma1000/Npart、delta4、批内异对象类别随机负part、允许复用。独立part/negative RNG与数据/Dropout随机流分开。

同数据trainval14007/test2874、原one-hot、N2048/k40、batch32、workers8、shuffle/drop_last，RSGD momentum.9/WD1e−4、LR.1→.001、200轮余弦，FP32、plain CE、无总梯度范数裁剪但保留数值球投影。保持现有test选模协议，以便与A–E比较。逐点/whole/part模长和贴边率、分项梯度继续记录；额外分别标记Conv9前后特征，不混记两种H。

每批只编码一次，BN9主支每批只更新一次；F/G不能为了计算正则再次经过L9。改变分叉不仅改变哪些层接收正则梯度，也改变part的特征近邻集合与正则约束的特征空间；F−H不能称为只开关梯度路径。

## 对照关系

- B→F：固定正则取点，给主支加入L9。
- F↔H：无dp1条件下，正则取点L9前/后。
- F↔G：正则取点L9前时，dp1关/开。
- H↔D：正则取点L9后时，dp1关/开。
- G↔D：有dp1条件下，正则取点L9前/后。

以上为本次已确认配置；没有额外改变正则公式、层次方向、初始化、label smoothing、logit scale或训练预算。F分配GPU0、G分配GPU1，均等待完全空闲；H排在F成功完成后使用GPU0。不使用已交给他人的GPU2。
