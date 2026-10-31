# Experiment I: H 主支与 PointMLP 式分类头前正则

## 已核对的来源

官方 `classification_ModelNet40/main_pointmlp_hycore.py` 先对 part、再对 whole 各做一次训练态前传；`models/hpointmlp.py` 中两者都经过完整编码器、`proj`、`EXP` 与首层 `MobiusLayer(512,256)`，正则在最终 `classifier=MobiusLayer(256,class_num)` 前计算。`emb=True` 仅跳过最终分类层。训练入口没有为 part 切换 `eval()` 或冻结 BN，故 BN 每批更新两次，先 part 后 whole。

I 受这一分叉与梯度范围启发，但仍保留 H 的分割网络、特征近邻 part 选择、欧氏均值后 EXP、whole 浅/part 深方向、异类负例、margin 与训练设置。I 不是 PointMLP 的 max 池化，也不是把分类权重加载到分割模型。

## H 与 I 的逐层定义

共同主支及初值：原分割编码器与类别输入 → `T256=Conv8/BN8/Leaky` → 原 `L9=Conv9(256→256,bias=False)/BN9/Leaky` → `Vwhole[B,N,256]` → 逐点 EXP → 官方 `M(256→128)` → 官方 `M(128→50)` → 分割 CE。无 dp1、dp2、总梯度裁剪；普通 CE。I 的模型可训练参数名称、数值初始状态、第一次 whole 前传 logits 与 H 精确相同。

H 的正则：`Vwhole` 中按特征近邻取 part，然后 whole/part 分别对 256 维点特征求均值 → EXP → 在 256 维球点上计算两正则；该旁支绕过两层 M。

I 的正则：

1. whole 训练态前传得到 `Vwhole` 和主支 logits；在 `Vwhole` 上沿用 H 的随机中心、稳定特征近邻排序和 200–600 点选择，取得原始 XYZ 的索引。
2. 用同一模型将选中的原始 XYZ 独立训练态前传到 `L9`，得到 `Vpart[B,Npart,256]`。两次前传都带梯度，类别 one-hot 相同；part 不经过最后分类 M。
3. `Vwhole.mean(points) → EXP → 共享 M(256→128) → μwhole128`；`Vpart.mean(points) → EXP → 同一 M(256→128) → μpart128`。
4. 对 `μwhole128/μpart128` 计算 H 同方向、同 gamma/delta 的径向与对比正则；仅 whole 的逐点主支走最终 `M(128→50)` 与 CE。

这样两正则可回传至第一层 M、L9、Conv8、STN 与 EdgeConv；最终 128→50 层仅接 CE。I 相对 H **同时**改变正则取点、part 原始子云再编码与训练态 BN 次数，不作为单因素消融解释。

## BN 与比较边界

H 每个训练 batch 各 BN 更新一次。I 因要先用 whole 的 `Vwhole` 保留 H 的特征近邻选点，实际调用顺序为 **whole → part**，各 BN 更新两次；推理仍只做 whole 一次 eval 前传。PointMLP 的顺序为 part → whole，不能声称 BN 轨迹完全一致。默认 BN momentum 0.1 时，I 每步的 running mean 为 `0.81×旧值 + 0.09×whole 统计 + 0.10×part 统计`。训练态前传各用自己的当前 batch 统计，测试态才用累计 running 统计。

## 固定协议和验收

- 与 H 同数据/类别输入、seed 4780、N2048/k40、batch32/eval16、200轮、RSGD 0.1→0.001、momentum0.9、weight decay1e−4、两正则各0.01、part200–600、gamma1000/Npart、delta4、无总梯度裁剪；test Instance mIoU 每轮选 best。使用全新独立目录，不续 H 权重。
- CPU 合同须验证 H 初值/whole logits 相同、选点一致、14个 BN 每 batch 各+2、Rcontr 与 Rhier到共享层及首 M而不到最终 M、CE 到最终 M、类别负例合法、检查点完整及有限。
- 真实尺寸 GPU smoke 使用 batch32、N2048、强制 part600，2轮×2步、仅 validation 不读 test；检查显存、有限状态、BN计数与首 M/末 M 的梯度路由。失败即不启动正式训练。
- 正式启动前重新检查 GPU 完全空闲；仅选空闲卡，不干扰其他进程。smoke通过后重新从 seed4780 初始化 fresh200，并核对初始模型 SHA。

## 当前结果

本机模型合同 6/6 通过；合成数据 CPU 完整入口 2轮×2步通过，14个 BN 计数均为8；记录中的 CE 梯度到最终 M，两项正则梯度到共享层、L9和首 M而最终 M为 `None`。真实 GPU smoke 和正式训练状态以独立执行记录为准。
