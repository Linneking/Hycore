# Batch 与分割头对类别退化的可能影响

日期：2026-10-29。范围：原 DGCNN partseg 的 test-selected 基线与 Pseg3 v3 的 A/B/C 三组。仅只读审查、局部 CPU 头验证和本研究记录；不更改训练源码、配置、服务器或现有结果，不启动训练。

**结论：目前证据不支持把新增类别退化归因于 batch 组织变了。分割头有明确且重要的数学变化：50维球坐标直接作为 CE logits，且原尾部的额外 BN/LeakyReLU 被移除。两者是优先待验证因素，但尚不能凭现有成绩确定其对小类退化的因果份额。**

## 1. Batch 与比较范围

主审本轮只读核对原基线和 A/B/C 的生产 manifest/逐轮记录：相同 data manifest、seed4780、workers8、batch32/eval16、N2048/k40、trainval/test 协议，每完整训练轮437次更新。A已记录1–104轮、B1–107轮、C1–101轮的每轮首批输入 hash/IDs 均与原基线对应轮相同；这里不保存或提交原始IDs。

此核验支持相同数据与 batch 组织，且已记录的每轮首批实际输入匹配；它不等于证明每轮全部437个 batch 的增强随机流均逐位相同。训练过程中的模型 dropout、初始化、独立part/negative RNG与普通数据loader RNG需分别看待。

即使两者都用 shuffle32，原数据的类别不均衡仍在；但它对原 DGCNN 同样存在，因此不能单独解释为什么换头之后小类明显更差。后者可反映模型/目标对相同不均衡的敏感性不同。

## 2. 实际头部变化

原 DGCNN 参考 [vendor/dgcnn_model.py:254](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/vendor/dgcnn_model.py:254)–265、305–310：

```text
融合1280 → Conv8(256)+BN8+Leaky → dp1
          → Conv9(256)+BN9+Leaky → dp2
          → Conv10(128)+BN10+Leaky → Conv11(50) → CE
```

当前 A/C 参考 [v2/model.py:94](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/model.py:94)–104、120–125 和 [v3/model.py:38](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/model.py:38)–49：

```text
同融合前端 → H256 → EXP/投影 → M1(256→256) → M2(256→128) → M3(128→50) → CE
```

A去dp1，C保留dp1；B另删M1。原Conv9/10/11、BN9/10、dp2被显式删除。M内部没有额外BN/LeakyReLU/dropout。当前普通CE显式`smoothing=False`，[v3/train.py:97](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/train.py:97)–99；原基线使用普通`F.cross_entropy`，[v1/train.py:272](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/train.py:272)。

官方同款 `MobiusLayer` 并不把一个独立偏置加到输出上：其 `bias=EXP(_weight*_bias)`，有效 `weight=transp0(bias,_weight)`，forward仅`mobius_matvec(weight,input)`，见 [native.py:83](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/native.py:83)–107。原DGCNN这三层Conv的bias也为False，因此不能简单说本次退化是“删除原输出bias”；原尾部BN的仿射参数与激活则确实消失。

## 3. 球坐标 logits 的概率上界与 CE 下界

当前M3返回球坐标，直接作为50类logits。Geoopt默认输出投影，FP32/c1时每点logit向量近似满足`||z||₂≤R=.996`，不是仅每一维限制在±.996；见 [Geoopt manifold.py:289](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/manifold.py:289)–296。

令正确类为y、总类数K、其他logit均值为m。Jensen给出`Σ(j≠y) exp(z_j−z_y) ≥ (K−1)exp(m−z_y)`；Cauchy给出`z_y−m ≤ R*sqrt(K/(K−1))`。两界可同时取等号，故：

```text
p_y,max = 1 / [1+(K−1)*exp(−R*sqrt(K/(K−1)))]
CE_min  = log[1+(K−1)*exp(−R*sqrt(K/(K−1)))]
```

K=50、R=.996时，最优正确类logit约.985990，其余49维各−.0201222，正确类概率至多 **5.28646%**，单点plain CE理论下界 **2.94002**。这是整球内任意logits的几何界，不依赖训练是否收敛。B最佳权重CE约3.001与此相容；原DGCNN无同样logit范数约束，CE约.172，不能把两者CE绝对值差距全部解释成后者没学会。

**该概率界绝不是准确率上界。** 正确类概率即使只有5%，也可以每个点都比其余49类更大，argmax准确率仍可100%。它揭示的是置信度/损失形状与优化条件改变，不能单凭该界推断小类必然失败。

主审独立计算得到p_max=.0528645884509、CE_min=2.94002156973，和上述推导一致。实际算子行为也对照了[Geoopt v0.5.1 manifold.py](https://raw.githubusercontent.com/geoopt/geoopt/v0.5.1/geoopt/manifolds/stereographic/manifold.py)与[math.py](https://raw.githubusercontent.com/geoopt/geoopt/v0.5.1/geoopt/manifolds/stereographic/math.py)。

## 4. 当前多层 M 的 argmax 决策关于 H 实际是线性的

对固定c>0，非退化输入的Möbius矩阵乘为`M⊗x = a(M,x)*Mx`，系数a为正；原点EXP与数值投影也只乘正标量。Geoopt具体式子见 [math.py:1224](/D:/Hycore/.codex-local/seg_dev_env/Lib/site-packages/geoopt/manifolds/stereographic/math.py:1224)–1239；安全投影见307–318。

各层有效矩阵由参数决定、与当前输入无关，且层间没有额外激活/平移，所以即使存在中间数值投影，方向仍满足：

```text
z(H) = positive_scalar(H) * [M3*M2*M1] * H      A/C
z(H) = positive_scalar(H) * [M3*M2] * H         B
argmax z(H) = argmax [effective_matrix * H]
```

这是**对给定H的预测决策性质**。不能把它扩大成“整个网络是线性的”，因为H由深层非线性编码器产生；也不能说该头的输出概率、CE或训练梯度等同于一个普通Linear，因为输入依赖的模长缩放和投影会改变它们。零向量、浮点异常及极近并列分数还需按具体实现处理。

局部CPU验证已完成：沿用真实NativePseg3头和官方M参数化，每组256个合成H点，输入模长覆盖.1/1/5/20；在A/B/C三组，实际头输出与有效矩阵乘积作用H的argmax **全部一致**，方向余弦误差最大分别1.79e−7/2.38e−7/1.79e−7。未运行编码器、未更新参数、CUDA未初始化。这是机制检验，不是训练checkpoint或真实样本成绩。

因此，原Conv9/10后的BN/LeakyReLU能够构建额外的非线性判别变换；将它们换成连续无激活M层，改变的不只有空间几何。多一层M主要还会改变因子分解、初始化、缩放、投影与优化路径，不能自动视为增加同等非线性表达能力。

## 5. 哪些影响已确认，哪些仍是假设

- **已确认：** batch组织/数据成员与原基线匹配；当前头的logits有整向量范数约束；尾部BN/Leaky/dp2被移除；无激活M链的argmax具有上述等价性；多个小类IoU长期停滞。
- **优先假设：** 有界logits/投影改变了CE优化，非线性尾部减少使部分部件更难区分；两者可能放大相同类别/部件不均衡对小类的影响。这些仍不是已证实的退化因果。
- **仍需隔离：** 原DGCNN只有CE，三组另加对比/径向正则；正则梯度非零不保证对语义分割有益。SGD与RSGD的实际更新协议也不同，不能把全部差异归于单个头部性质。

建议后续优先使用同配置、同初始化的受控对照：同当前头仅去正则，检验正则是否损伤小类；保留M结构、只在训练CE前增加logit尺度，检验受限logit优化；以及保留/恢复尾部非线性的结构对照。它们只列为待审候选，不在本轮实施。固定权重后单纯乘正温度或logmap只改置信度、不会改argmax，因此不能用一次后处理后的mIoU不变否定训练期优化假设。
