# HPCS主支与其VN-DGCNN分割基线的源码审查

2026-11-02。只读官方源码，未运行HPCS模型、修改现有实验或服务器训练。将旋转不变前端和HypHC独特层次目标作为本轮不展开的部分，追踪分割特征到双曲表示、再到类别分数的主路径。实际保留的CosFace属于主语义监督，不能连同辅助层次目标一起忽略。固定来源及逐文件哈希见[sources.json](sources.json)。

## 1. 来源与实际入口

- HPCS官方仓库固定提交：`c464387adb540b3e79896b0728685ca370d22e28`。[仓库](https://github.com/TheCrossProduct/HPCS/tree/c464387adb540b3e79896b0728685ca370d22e28)
- 直接结构参照为Vector Neurons官方仓库VNN的`models/vn_dgcnn_partseg.py`，固定提交`f3197e210022b5f0015e0da6456adf66bd0cd73e`。[基线源码](https://github.com/FlyingGiraffe/vnn/blob/f3197e210022b5f0015e0da6456adf66bd0cd73e/models/vn_dgcnn_partseg.py)
- HPCS的README ShapeNet示例选择`vn_dgcnn_partseg`、欧氏嵌入32、双曲嵌入32；`train.py`实际构造VN_DGCNN_partseg输出欧氏嵌入维度，然后单独配置双曲embedder。它不是把普通DGCNN后三层换成三层Möbius的实现。[README示例](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/README.md#train)、[入口与工厂](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/train.py#L64-L106)

这里“基线”指可直接逐层核对的官方VN-DGCNN分割网络；HPCS确有普通DGCNN类，但默认/README运行主线是VN版本。旋转不变的VN运算及标准化坐标框架作为共享前缀压缩，不把它们当成双曲适配改动。

## 2. 共享部分压缩后，逐层比较

设T256为conv8/BN8/Leaky输出的逐点融合特征。官方VN-DGCNN与HPCS对应网络在该位置都从2299维融合输入映射到256；普通AnTao DGCNN是1280→256。这一输入维度差异来自前端VN表示，不是把特征放进双曲空间所需的修改。[VNN定义](https://github.com/FlyingGiraffe/vnn/blob/f3197e210022b5f0015e0da6456adf66bd0cd73e/models/vn_dgcnn_partseg.py#L14-L53)、[HPCS定义](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/nn/dgcnn/vn_dgcnn_partseg.py#L17-L57)

```text
共享前缀：VN局部/全局/类别特征融合 → Conv8 → BN8 → LeakyReLU → T256

官方VN-DGCNN分割基线：
T256
 → dp1(.5)
 → Conv9(256→256,bias=False) → BN9 → LeakyReLU(.2)
 → dp2(.5)
 → Conv10(256→128,bias=False) → BN10 → LeakyReLU(.2)
 → Conv11(128→50,bias=False)
 → 50类监督

HPCS，README等维32→32配置：
T256
 → dp1(.5)
 → Conv9(256→256,bias=False) → BN9 → LeakyReLU(.2)
 → dp2(.5)
 → Conv10(256→128,bias=False) → BN10 → LeakyReLU(.2)
 → Conv11(128→32,bias=False) → 新增BN11
 → 欧氏逐点嵌入e32
 → EXP₀
 → 双曲逐点嵌入z32
 → 学习到的类别权重计算余弦分数：32维嵌入→50类分数
 → CosFace主分类监督
```

|层/操作|官方VN-DGCNN|HPCS实际主线|判断|
|---|---|---|---|
|Conv8+BN8+Leaky|保留|保留|融合路径共享|
|dp1|p=.5|默认p=.5|未因双曲适配而删除|
|Conv9+BN9+Leaky|256→256|同宽保留|保留一组判别非线性|
|dp2|p=.5|默认p=.5|未因双曲适配而删除|
|Conv10+BN10+Leaky|256→128|同宽保留|保留第二组判别非线性|
|Conv11|128→50|128→d，示例d=32|从类别输出变成嵌入输出|
|BN11|没有|新增，通道数d|位于EXP之前，没有新增末端Leaky|
|EXP|没有|e_d→z_d|独立于欧氏特征提取器|
|Möbius映射层|没有|README实际主线也没有|不能把仓库内未使用的M类当成主支|
|类别读出|Conv11的50个通道|CosFace自己的可训练类别权重|嵌入维度与类别数量分离|

逐层执行顺序由两个官方forward明确给出；HPCS保留原dp1、dp2及Conv9/10配套块。[基线forward](https://github.com/FlyingGiraffe/vnn/blob/f3197e210022b5f0015e0da6456adf66bd0cd73e/models/vn_dgcnn_partseg.py#L96-L108)、[HPCS forward](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/nn/dgcnn/vn_dgcnn_partseg.py#L97-L104)

## 3. 入球模块到底做什么

`configure_hyperbolic_embedder`按欧氏/双曲维度是否相同选择分支：相同则纯ExpMap，不同则MLPExpMap。ExpMap调用`expmap_1(e,0)`；这里没有Möbius线性层。[工厂](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/train.py#L99-L106)、[实际embedder](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/nn/hyperbolic/hyp_embed.py#L6-L23)

维度不同时：

```text
e_d → 普通Linear(d→h,bias=False) → Dropout(p=0，默认) → EXP₀ → z_h
```

虽然类名叫MLP且接收negative_slope参数，`hpcs.nn.MLP`实际只构造Linear+Dropout，没有BN或激活；不能按函数名补出不存在的层。[MLP实际定义](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/nn/__init__.py#L1-L9)

`vn_dgcnn_expo.py`及`utils/manifold_layers.py`包含其他EXP/Möbius代码，但上述入口没有实例化VN_DGCNN_expo或MobiusLayer。文件存在不等于论文推荐运行链路在用它。[实际选择器](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/train.py#L64-L78)

## 4. 主分类分数与我们的球坐标CE不同

ShapeNet `_forward`先取得欧氏e，再调用nn_emb得到球嵌入z；默认输入类别one-hot，非默认class_vector分支不作为此处主线。[ShapeNet主路径](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/models/shapenet_hyp_hc.py#L75-L91)

HPCS调用`CosFaceLoss(num_classes=C, embedding_size=h, margin=.35, scale=2)`，且主分类loss接收球嵌入z，不是直接将z的坐标作为C类logits。因而“去掉独特层次正则”后仍应保留这条主监督；CosFace不是附加HypHC正则。[分类器构造](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/loss/ultrametric_loss.py#L34-L38)、[分类loss调用](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/loss/ultrametric_loss.py#L117-L138)

其类别权重由metric-learning库作为可训练参数持有，RAdam优化`self.parameters()`时一并更新；它们不是从baseline的50通道Conv11直接沿用的输出权重。[优化器](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/models/base_hyp_hc.py#L141-L155)

按环境文件指定的pytorch-metric-learning 1.6.3核对，类别矩阵W的形状为[h,C]，嵌入与每个类别向量均L2归一化。CosFace覆盖父类的scale_logits，只乘显式scale，不乘原始向量模长。HPCS训练的类别分数是：

\[
\ell_j=2\left(\frac{z^T w_j}{\|z\|\|w_j\|}-.35\,\mathbf1[j=y]\right).
\]

这里2是HPCS显式设置的分类尺度，.35是真值类别margin。它们与层次目标的半径scale是不同参数。源码依据：[依赖版本](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs-env.yaml#L283)、[类别W与余弦计算](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v1.6.3/src/pytorch_metric_learning/losses/large_margin_softmax_loss.py#L20-L60)、[CosFace实际缩放](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v1.6.3/src/pytorch_metric_learning/losses/cosface_loss.py#L9-L30)、[余弦归一化](https://github.com/KevinMusgrave/pytorch-metric-learning/blob/v1.6.3/src/pytorch_metric_learning/distances/cosine_similarity.py#L4-L8)。

**由公式作出的推论：** 对非零e，在理想数值条件下，原点EXP满足z=a(e)e且a(e)>0，因此normalize(z)=normalize(e)，cos(z,w)=cos(e,w)。若单独去掉HypHC，主CosFace分类目标不会因添加这个纯EXP获得新的角向表达能力；EXP仍用于构建层次几何。这里不包含零向量、下溢或有限精度误差，也不等于实际训练全程经过实验验证。

这与当前球坐标CE有实质区别：HPCS不要求50个分类分数共同组成一个单位球内向量，分类尺度独立于球半径。但其余弦分数本身仍有界，不能称为无界实数MLR或双曲超平面分类器。

重要的区别是：球表示回答“点嵌入在哪里”，类别分数由额外读出来回答“属于哪个部件”。当前H0/S5直接把末M输出的50维球坐标交给CE；正在运行的S5-TAN-R0则用末层切空间实数分数q交给CE。三者都涉及双曲中间表示，但分类头和主监督并不相同。

## 5. 没有发现可直接称为防贴边的隐藏结构

主用ExpMap的`expmap_1`计算tanh径向映射后调用mobius_add，没有调用同文件的project。`tanh`实现夹紧输入到[-15,15]，这是数值运算选择，不能称为阻止嵌入接近边界的机制；BN11本身也不约束每个点向量的模长。[EXP与独立project](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/utils/poincare.py#L50-L77)、[tanh](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/utils/math.py#L81-L82)

模型中初始1e-3的可学习scale不是给主支e乘的小系数。它用于层次目标的`normalize_embeddings`：先L2归一，再乘共同半径；直接将其搬到我们的whole/part径向正则会改变目标含义。[scale创建](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/models/base_hyp_hc.py#L62-L75)、[共同半径归一](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/loss/ultrametric_loss.py#L139-L143)

## 6. 由源码得到的启发及范围

1. **保留完整欧氏判别栈，再在末端适配双曲，是一条已有实际实现的路线。** HPCS没有照HyCoRe分类的方式删掉Conv9/10的BN/Leaky和所有Dropout。因此，不能把“所有双曲改造都应删除尾部配套层”当成必要规则。它也没有证明恢复这些层一定改善我们的motorbike。
2. **优先区分双曲表示与类别分数。** HPCS用独立余弦类别读出，不让32维球坐标兼任50类分数。这加强了当前先查主读出的动机，但不是S5-TAN-R0有效的实验证据。
3. **类别权重归一化值得单独检验。** 131观察到若干漏检类别的有效权重行模长较小；归一化余弦头不会让某个类别仅因权重模长小就在分数上直接处于劣势。它是否改善motorbike仍是待验证的假设，不能把相关性说成已证原因。
4. **若进一步借鉴，将BN11/完整Conv10块与类别读出分开对照。** BN11是逐通道BN，不是每个点向量的单位范数归一化。当前S5-TAN-R0先完成原预算；若还需后继，可保持现有128维双曲特征和Roff，仅考察归一化余弦读出+独立尺度，先不引入CosFace特有margin。完整恢复Conv10配套模块属于另一结构变量。共同改变dropout、BN/激活、嵌入维度、分类目标和层次目标，会失去与H0/S5的可解释对照。
5. **不宜立即照搬全部HPCS。** 它的旋转前端、CosFace、维度、优化器及层次解码同时不同；当前复现目标仍是HyCoRe-seg，不能用HPCS结果代替我们的无R主支验证，也不能因源码有BN11就宣称原问题是BN缺失。HPCS保留dropout是基线继承，没有在此提供“我们的模型恢复dropout更好”的证据。

## 7. 复用前需要注意的代码分支边界

- 普通DGCNN选择器传入`out_features=num_class`，不是`eucl_embedding`；README32维嵌入配置直接把model字符串改成dgcnn_partseg，可能导致实际特征维与embedder/CosFace期望不一致，不能当成已验证的普通DGCNN双曲分支。[选择器](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/train.py#L64-L70)
- `--pretrained`为显式选项，README示例没有启用。代码支持预训练不等于该示例依赖预训练。[参数及加载](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/train.py#L55-L96)
- `get_logits(embeddings,labels)`会对真值类别施加CosFace margin，评估函数也调用它；如借鉴成真实无标签预测头，应单独规定推理读出，不能原样复制标签相关分数。[get_logits](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/loss/ultrametric_loss.py#L95-L115)、[评估调用](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/models/base_hyp_hc.py#L89-L102)
- 另一条test `score`经get_optimal_k使用真值选择切树k及簇匹配，语义iou又使用MulticlassJaccardIndex；这些均不能直接当成我们的ShapeNetPart物体内先平均部件、再对16类宏平均的Class mIoU。源码设计可借鉴，分数不直接横比。[test路径](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/models/base_hyp_hc.py#L192-L198)、[k选择与匹配](https://github.com/TheCrossProduct/HPCS/blob/c464387adb540b3e79896b0728685ca370d22e28/hpcs/utils/scores.py#L141-L175)。

本次是静态源码审查。没有新增训练、读出探针或HPCS复现实验，以上实验方向均为建议，待与当前S5-TAN-R0结果结合后决定。
