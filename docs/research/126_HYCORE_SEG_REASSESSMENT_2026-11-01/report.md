# HyCoRe-seg 复现路线重新审视

2026-11-01。依据：两篇原论文、官方分类源码、实际公开分割仓库、本项目已完成实验及只读服务器日志。当前仅研究与方案，不修改模型、不启动或停止实验。I 的最新快照为北京时间12:47、第121轮，尚未完成。

## 1. 结论与优先方向

目前应固定 H 的主支，优先检查预测读出和正则目标是否与逐点分割相容，暂缓继续围绕 dp1、M 层数量做细小调整。已有多组正则开关对照，未出现稳定的两指标收益；不是缺少任何零正则对照。分类论文成功的组合不能直接证明当前分割移植也会成功。

最值得借鉴的公开点云代码是 HPCS；其学习目标和评估协议与本项目不同。对于最后的分类头，公开双曲图像分割代码提供了更直接的组件参照：球内嵌入与送入 CE 的实数 logits 分开。最小实验可先检查训练时的 logit 尺度，无需立刻更换主干。

I 应当作为分类启发的额外变体分析。它同时改变正则位置、part 的输入上下文以及 BN 统计；当前结果无法单独回答“正则经过 M 是否更有效”。可以用单次 whole 编码的共享 M 支路，把这个问题单独拆出来。

## 2. 可用的外部实现

| 来源 | 已核实可用内容 | 与本任务的关系 |
|---|---|---|
| [HPCS](https://github.com/TheCrossProduct/HPCS) | ShapeNet/PartNet；DGCNN/VN-DGCNN；逐点嵌入、CosFace/Triplet 与 HypHC/LCA | 最贴近数据与骨干，但不是 HyCoRe 的 whole/part 正则 |
| [hyp-segmentation](https://github.com/mikhail-vlasenko/hyp-segmentation) | 图像分割；EXP 后接距离原型分数或真正双曲 MLR | 最值得迁移的是分类读出，不能声称是点云开箱即用代码 |
| [HyperbolicImageSegmentation](https://github.com/MinaGhadimiAtigh/HyperbolicImageSegmentation) | 原始双曲图像分割参考，TF1.14，仓库标注尚不完整 | 思路参考，环境和迁移成本较高 |
| [Horospherical](https://github.com/Pangoraw/horospherical) | 原型/分类器组件；README 将点云分割标为待发布 | 不能当完整点云分割复现代码 |
| [官方HyCoRe](https://github.com/diegovalsesia/HyCoRe) | ModelNet40/ScanObjectNN 分类代码 | 本轮仍未确认作者 HyCoRe-seg 分割入口与配置 |

HPCS 的层次聚类 score 使用真值选择最优树切分；其另一个 CosFace IoU 路径也需审查目标 margin 在推理时的使用。它们不能不加区分地与我们的固定50类 argmax mIoU比较。其可学习 scale 控制 LCA 共同叶半径，不是 EXP 前缩放。[HPCS评估](https://github.com/TheCrossProduct/HPCS/blob/main/hpcs/models/base_hyp_hc.py)、[损失](https://github.com/TheCrossProduct/HPCS/blob/main/hpcs/loss/ultrametric_loss.py)。

外部库没有在本环境安装或训练通过；已确认的是源文件及实现路径。详细核查与未找到完整代码的候选见 [public_code.md](public_code.md)。

## 3. 两篇论文不应混成同一条复现路线

分类 Table3（ScanObjectNN、256维 AA）为 DGCNN77.8、仅双曲头76.5、完整HyCoRe80.2。用户关于“只有M可能低于基线”的判断有直接依据，但不是所有任务的保证。[分类论文](https://papers.neurips.cc/paper_files/paper/2022/file/da8f9fc2b555d122369f36a9684415c1-Paper-Conference.pdf)。

分割 Table1 的 Class/Instance 是普通DGCNN80.9/85.1、完整方法82.8/85.7。Table2为两正则85.7、仅对比85.6、仅层次85.5，没有单列同头零正则结果。论文还没有给出精确尾部层宽、BN/dropout及完整训练配置；§3两margin均1024/Npart与§4 gamma1000/delta4存在未解释差异。[分割作者稿](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)。

```text
分类：whole/part各自编码 → EXP → 共享M → μ → 最终M → whole CE
                                         └→ 正则（whole深、part浅）

分割论文：whole一次编码 → 逐点H → EXP → M头 → 点CE
                           └→ whole/特征近邻part分别mean → EXP → 正则
                                                               whole浅、part深
```

因此 H 的正则绕过 M 符合分割论文的计算图；I 的重新编码与M后正则来自分类启发。不能因 I 更像分类代码，就说它必然更接近分割原方法。初始化已复用官方类规则，不是加载分类训练权重；目前没有证据把失败归因于一个确定的初始化实现错误。见 [paper_audit.md](paper_audit.md)。

## 4. 本项目的有效对照

均为最佳Test Instance及同轮Class，单位%。单seed4780；F/G/H前段在RTX4090、后迁RTX5090，零正则组全程RTX5090，仍有硬件混杂。F0/H0的活跃模型和训练路径相同，不是两次独立seed验证。

| 对照 | 开正则 Instance / Class | 关正则 Instance / Class | 开−关（百分点） |
|---|---:|---:|---:|
| v1 HC / H0 | 83.179 / 67.569 | 83.431 / 68.073 | −0.252 / −0.504 |
| D / E | 82.119 / 64.162 | 82.897 / 64.797 | −0.778 / −0.635 |
| F / F0 | 84.297 / 71.278 | 84.521 / 76.093 | −0.224 / −4.815 |
| G / G0 | 82.385 / 64.491 | 82.897 / 64.797 | −0.512 / −0.306 |
| H / H0 | 84.618 / 74.821 | 84.521 / 76.093 | +0.097 / −1.272 |

普通DGCNN为85.383/81.777。H0已经低于基线，说明不能将全部差距都归因于正则。DGCNN对H0还包含尾部BN/非线性/Dropout与优化器变化，不能将差距全算成曲率效果。

I在12:47快照已完成120轮，best仍为e47的77.519/63.988，e120为70.329/59.708，最近10轮平均Instance64.667。公平地限制到前64轮，H最佳83.145/72.904，I最佳77.519/63.988；这说明差距并非只因I尚未满200轮，但也不将中途结果冒充最终结果。

原始汇总见 [server_evidence.json](server_evidence.json)、[同预算比较](matched_epoch_evidence.json)、[全部结构及结果](local_audit.md)。旧结果未改写。

## 5. 比继续调M层数更值得验证的因素

### 5.1 球内坐标直接做logits，约束了CE的动态范围

官方实际 MobiusLayer.forward 仅调用 mobius_matvec；_bias通过transport改变有效weight，没有输出平移或逐坐标激活。当前主支最终M输出球坐标直接进入普通CE，官方分类也这么做。[官方层](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ModelNet40/models/ops/manifold_layers.py#L51)。

本次从实际约束独立推导：若K类logits满足||z||₂≤r，则

`CE_min = log(1+(K−1)exp(−r sqrt(K/(K−1))))`。

FP32、c=1、r=.996、K=50给出 CE_min≈2.94002，正确类softmax概率上限≈5.286%。实际H最后CE≈2.96776，H0≈2.96524；欧氏DGCNN最佳轮CE≈.08671。它们的数值不能直接比较，也不能据此说双曲模型几乎没学会。概率低不等于argmax准确率低。

这不是已确认导致掉点的唯一原因，分类官方也有相同约束。但它是可以最小修改验证的假设：保持球嵌入/R不动，只在训练CE前给logits乘固定正尺度s。推理时给旧checkpoint乘s不会改变argmax，也就不会改善mIoU。

公开双曲分割的另一做法是把球点转成实数分类分数，例如−τd(z,p_k)或双曲超平面MLR。后者可改变判别器，比温度实验改动更大，应作为后续独立变体。[分割头](https://github.com/mikhail-vlasenko/hyp-segmentation/blob/master/segformer_head.py)、[MLR实现](https://github.com/mikhail-vlasenko/hyp-segmentation/blob/master/hyperbolic_layers.py)。

### 5.2 连续的当前M层没有恢复被删掉的Leaky非线性判别

对固定权重，Möbius matvec、EXP和数值球投影都保留向量方向的线性映射关系。因此无平移/逐坐标激活的当前头满足

`argmax(M3(M2(M1(EXP(H))))) = argmax(W3 W2 W1 H)`，

零向量和浮点退化情况除外。这是判别方向关系，不是训练梯度等价：参数分解、模长、投影与优化仍不同。它解释了为什么三M相对两M不等价于多一个普通MLP非线性块，继续调层数未必解决逐点判别问题。

### 5.3 需要记录margin可达率，不能只看贴边率

当前最大球深度是2atanh(.996)=6.2126。要满足分割径向约束，必须有 `dwhole≤6.2126−1000/Npart`：

| Npart | margin | whole半径必须不超过 |
|---:|---:|---:|
| 200 | 5.000 | .5415 |
| 400 | 2.500 | .9523 |
| 600 | 1.667 | .9790 |

whole半径若为.98，即使part放到最大数值半径，当前任一种part大小的margin也无法满足，whole必须先向内移动。这不等于全网梯度都为零。part向外是设计目标，所以part贴边不能单独当失败证据。

实际末20条记录，H的层次满足率约2.81%、whole/part贴边67.5%/71.25%；I在早先约65轮窗口满足率42.81%、贴边17.34%/83.28%。几何关系改善与任务改善已经分离。不同轮次窗口仅用于机制观察，不作为公平性能消融。

应补逐对象、按Npart分桶的可达率、径向/切向梯度和CE–R梯度夹角。I一条已有记录中，共享层的加权Rcontr/Rhier梯度范数约.028/.034，CE约.060；虽然R标量约.011而CE约2.986，R梯度并不因此可以忽略。

文献中的入球前特征幅度控制可供参考，但新硬截断也会在超阈值区消掉径向导数，不能把它当自动修复；优先研究训练统计设定的温和尺度，再检查与margin的相容性。[CVPR2022特征裁切研究](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Clipped_Hyperbolic_Classifiers_Are_Super-Hyperbolic_Classifiers_CVPR_2022_paper.html)。

### 5.4 I要先检查BN分布，再解释正则效果

I每batch whole→part各训练态前传一次；momentum=.1时running mean更新为`.81旧+.09whole+.10part`。推理只有whole。PointMLP的顺序相反，两者都混入part统计。两次train前传本身合法，但分割whole与局部子云的分布差异可能更大。

建议复制I的best与late checkpoint，冻结所有学习参数，只用训练whole数据重新统计BN，再在固定验证数据上比较重校准前后成绩；同时保留原BN版本。它是待执行的诊断，不声称BN已被证明是主因。若要做I0，必须保留两次前传和BN更新，仅关R；H0不是I的严格零正则组。

若想只验证“R经过M”，可用下列单次编码变体：

```text
H主支保持：T256 → L9 → U256 → EXP → M256→128 → M128→50 → CE
候选支路：同一次whole得到U
          ├─ mean(Uwhole) → EXP → 共享M256→128 → μwhole
          └─ mean(U中选中的part) → EXP → 同一个M → μpart
                                                     └→ 两正则
```

这已经让两正则回传到主干、L9和首M，无需为扩大这个梯度范围而必须重新编码原始part。它保留whole上下文，不能替代I关于“独立子云编码”的问题，也属于偏离分割论文的有意变体。

### 5.5 优先检查少数物类中的部件塌缩

多个模型bag、rocket等分数反复落在同一个值，值得核查逐对象预测直方图、部件recall、每类混淆和单标签预测比例。仅凭同一个IoU数值还不能证明常量预测。H/I在best时，物类mask后的辅助Instance与原50类输出相同；F0仅提升约.0184个百分点，因而跨物类非法部件预测解释不了主要差距。

另一个待验证假设是R更多利用已输入的物类one-hot来分离物类，而没有学到物体内部细部。可先将梯度按类别支路Conv7、局部EdgeConv、全局融合分组，测同一物体不同部件的可分性；不能仅看异物类距离拉开就宣布分割层次有效。此时不先改变batch采样或给少数类加权，以免新增混杂。

## 6. 建议的下一步顺序（待审，不启动）

**第一步：利用已有checkpoint做上述诊断。** 最优先是I的BN重校准对照、H/H0/I的逐部件预测统计，以及logit/梯度/margin可达率。I仍运行，当前未终止或修改。诊断若显示严重BN失配，就应先处理该因素，而不是继续扩大I结构。

**第二步：若只选一组最小训练对照，固定H主支做读出尺度开/关R配对。** 示例候选J0/J1如下，s为训练前固定的同一正值；可先提议s=5做小规模验证，它不是已验证最优值，不根据测试集挑s。当前s=1的H0/H作为历史参照，新两组必须在同一硬件、初值、数据和BN协议下运行。

```text
共享主支：T256 → 原L9(Conv9 bias=False + BN9 + Leaky)
                → EXP → 官方M256→128 → 官方M128→50 → z50
J0：CE(s·z50)，两正则关闭
J1：CE(s·z50) + .01Rcontr + .01Rhier
    正则仍从U256 mean→EXP取值；不把s·z当球点送入R
```

J1−J0回答新读出下R是否有益；J0相对旧H0回答读出本身是否有益。不能把J1胜过旧H0全部归因于R。主指标仍看Instance和Class、各部件recall；CE下降只是尺度变化的预期结果。两者均属于明确命名的读出改进，不冒称作者未公开代码。

**第三步：只针对诊断支持的方向扩展。** 若I有BN问题但M后正则值得保留，先做上述单次编码M后支路，或保留两次前传做真正I0；不要同时改BN、尺度和loss。若当前两项R仍无增益，在固定读出/几何上拆contrast-only与hier-only，以确定哪项有用。若logit尺度有效，再考虑真正双曲MLR/原型读出，比较是否仍需要原两项R。

暂不建议把换骨干、曲率、margin、学习率、采样器、分类器和BN混成一轮“大修”。旧单seed、逐轮test选best及历史重复ID协议适合保留作工程对照；最后的方法结论应在固定验证选模、独立测试和多seed下确认，不能用一次best的0.1pp作结论。

## 7. 本次产物

- [公开代码核查](public_code.md)
- [论文与源码复核、数学推导](paper_audit.md)
- [全部历史结构、开关对照及本地源码证据](local_audit.md)
- [服务器汇总证据](server_evidence.json)
- [同64轮预算、DGCNN损失与I最新快照](matched_epoch_evidence.json)

本研究没有取得或宣称取得HyCoRe-seg未公开的原始训练配置。下一实验矩阵须用户审阅后再写入执行安排。
