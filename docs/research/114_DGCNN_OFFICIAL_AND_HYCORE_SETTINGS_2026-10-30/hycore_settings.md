# HyCoRe 的训练设置：BN、损失、优化器与当前 D/E 的核对

审查日期：2026-10-30。范围为官方 HyCoRe 的 ModelNet40 DGCNN 分类路径，以及当前冻结 v5 分割 D/E；仅源码/文档研究，没有模型前反向、GPU、参数或训练设置变更。

本地复用来源记录为 `diegovalsesia/HyCoRe` 的 `ef56a73020e309b1009c486fea1f601d3f7b3252`；模型、层、几何、损失源文件由 LF SHA256 验证。见[来源记录](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/vendor_manifest.json:1)及[导入时验证](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/native.py:32)。在线交叉核对作者 [HyCoRe 仓库](https://github.com/diegovalsesia/HyCoRe)及其分类训练和模型文件；部分固定 SHA raw 页面无法抓取，固定版本判据以本地来源记录为准。

## 1. BN 的事实：分类 DGCNN 没有发现特殊的 part-eval/whole-train 切换

官方分类 `train()` 在循环前调用 `net.train()`；每批先对 `pos_data` 前向，再对 `data` 前向，两次之间没有 `eval()`、BN 专用模式切换、detach 或 no_grad。`validate()` 才调用 `net.eval()` 并在 no_grad 下计算。模型 `DGCNN.forward()` 也没有依据 part/whole 改模式。证据：[本地训练入口](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:206)、[两次前向](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:220)、[验证](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:257)、[官方在线源码](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/main_dgcnn_hycore.py#L206-L267)。同时检查了本地 PointMLP 的训练入口和 hpointmlp 模型，也没有发现这种 part/whole 专门切换；本报告的确定性结论仍以本次所比较的 DGCNN 路径为主。

BN 本身是普通 `nn.BatchNorm1d/2d`，未显式修改 eps、momentum、affine、track_running_stats，因而采用 PyTorch 默认值。训练用当前 batch 统计量并更新 running statistics，测试用 running statistics。这是 PyTorch 的标准行为，不能作为 HyCoRe 独创的设置。`eval()` 改模块行为，不等于关闭 autograd；no_grad 也不会自动把 BN/Dropout 改成 eval。BN 的 affine 参数是否求导与 running statistics 是否更新应分别判断。证据：[HyCoRe DGCNN BN 构造](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:100)、[PyTorch 2.8 BN 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm1d.html)、[Module.eval](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html#torch.nn.Module.eval)、[no_grad](https://docs.pytorch.org/docs/2.8/generated/torch.no_grad.html)。

分类确有一个值得注意的行为差异：同一 BN 每批被执行两次，第一次是较小的 part，第二次是较大的 parent。以默认 momentum=.1 的 running mean 为例，按源码执行次序推导：

```text
分类：r_after_part = .9 r_old + .1 mean_part
      r_after_parent = .81 r_old + .09 mean_part + .10 mean_parent

分割 D/E：r_new = .9 r_old + .1 mean_full
```

这是由两次/一次编码推导出的统计更新差异，不是另一个训练超参数。不能因为分类更新两次，就在分割中增加一次虚假前向来“补齐”；分割论文的特征层 whole/part 聚合路线原本就避免重复编码。当前 v5 每批仅一次模型前向，whole/part 是已编码 H 的平均，正则中没有 BN；训练逐轮 `model.train()`，测试复用 `model.eval()+no_grad()`，且每轮断言所有 BN 的计数只增加真实 steps 次。证据：[v5 模式与前向](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:285)、[BN 计数断言](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:352)、[测试入口](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v1/train.py:109)。

因此值得借鉴的是**检查训练/推理的特征尺度是否相差过大，并核对统计量更新是否符合真实前向次数**。若发现 train/eval 差异，才能讨论 BN 的统计轴、momentum 或只用训练数据重估 running statistics；不应直接让测试使用 train 模式，那还会重新开启 dp1 并改变评估。

## 2. 一个明确的非结构差异：官方分类实际用了 label smoothing

官方入口导入的是 `utils.cal_loss`，不是通过命令行 `--smoothing` 决定损失。它设置 `criterion=cal_loss` 后，以两个参数调用 `criterion(logits, label)`。`utils.misc.cal_loss` 的默认 `smoothing=True`，epsilon=.2。虽然 parser 的 `--smoothing` 默认是 False，该参数没有传入实际损失调用，不能据 parser 默认值称官方分类使用 plain CE。证据：[导入和 criterion 绑定](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:23)、[损失调用](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:228)、[实际 CE 定义](/D:/Hycore/classification_ModelNet40/utils/misc.py:218)。

当前 v5 明确调用 `cal_loss(..., smoothing=False)`，是 plain CE。它与我们已经跑过的普通分割基线一致，但与上述官方 HyCoRe 分类损失设置不同。应称为已确认的训练协议差异，不应混入“只改最后几层”的结构声明。证据：[v5 plain_ce](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:102)。

若后续试验要更接近分类训练配方，label smoothing 可作为单独开关；它改变 CE 的目标分布及梯度，可能影响置信度、少数类别和入球前尺度，但仅凭源码不能保证提升。D/E 当前都使用同一个 plain CE，因此它不是本次 D−E 结果差异的解释变量。

## 3. 已沿用和明确未沿用的设置

| 设置 | 官方 HyCoRe 分类 DGCNN 的实际路径 | 当前 v5 D/E | 判断 |
|---|---|---|---|
| 全网络优化器 | Geoopt RiemannianSGD，momentum=.9、WD=1e−4 | 相同 | 已借鉴；不应据此把与普通 SGD 基线的差值全算作层替换效果 |
| LR | launcher 显式 .1，300轮，cosine 下限 LR/100 | .1→.001，200轮 | 起点/比例相同；训练轮数为任务协议差异 |
| 参数梯度裁剪 | 总 loss.backward 后，全网络 L2 norm clip1，再 optimizer.step | 取消，只记录范数和>1比例 | 用户此前明确要求的偏离，不是漏实现；若重试应单独控制 |
| 分类 CE | smoothing epsilon=.2 | plain CE | 明确协议差异，可单独试验 |
| BN | 普通 BN，无 part 特殊 eval；每批两次编码 | 普通 BN，每批一次编码 | 对应不同 whole/part 构造，不需要凭空补前向 |
| 负例 | batch 反转 pos_mu；不检查类别 | batch 异类样本随机抽取，允许复用 | 用户确认的正确性/采样协议变化，不能说严格逐位复现分类 |
| 正则系数 | 两项各 .01 | D 两项各 .01，E 两项均0 | D 借鉴系数；E 为无正则对照 |
| margin | hier=1000/Npart，contr=4 | 相同数值规则 | 已借鉴；分割层次方向按分割设计，与分类方向不同属有意适配 |
| 训练数据组织 | shuffle=True、drop_last=True，batch32 | 相同类别混合 batch 组织 | 已沿用；MN40/ShapeNetPart 的类别频率不相同 |
| part/negative RNG | 原 Python random 和固定 flip | 独立、可保存 part/negative RNG | 保证 D/E 配对与 CE/dropout 随机流隔离，属工程/协议变化 |

优化器/调度证据：[官方训练入口](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:144)、[原 launcher](/D:/Hycore/classification_ModelNet40/launcher_dgcnn.sh:4)、[v5 优化器](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:191)。裁剪及系数证据：[官方 backward/clip/step](/D:/Hycore/classification_ModelNet40/main_dgcnn_hycore.py:230)。负例和 margin 证据：[原始 sampler/损失](/D:/Hycore/classification_ModelNet40/hutil.py:27)、[当前采样核](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v3/losses.py:24)。

## 4. M 层没有隐藏的 train/eval 技巧，也没有被漏掉的层间 BN

官方 `MobiusLayer` 继承 `RiemannianLayer`，forward 只走 manifold Möbius matrix-vector 运算。其 weight/bias 属性含 expmap 与 parallel transport 参数化；初始化为 Kaiming normal(a=√5)，内部 bias 的均匀区间为 ±4/√fan_in。该层不含 BN、Dropout、额外 LeakyReLU，不读取 `self.training`，所以不能把层外删除的 BN/激活理解为“已经内置进 M”。这些完整参数化和初始化已被当前 native 组件复用。证据：[官方层实现](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:55)、[在线主源](https://github.com/diegovalsesia/HyCoRe/blob/main/classification_ModelNet40/models/ops/manifold_layers.py#L55-L116)、[本地复用](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v2/native.py:67)。

层文件里虽还定义 `ToPoincare`/grad_fix，官方 DGCNN 当前前向实际直接调用 `self.manifold.expmap0`，没有实例化并调用该 ToPoincare；不能把仓库里存在的辅助类当作已生效的径向防饱和机制。官方最终名为 `mlr` 的对象实际也被构造为 `MobiusLayer`；不是另一种带隐藏归一化、learnable logit scale 或 BN 的分类器。当前双 M 复用了这一实际实现。证据：[DGCNN 构造](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:123)、[实际入球和输出](/D:/Hycore/classification_ModelNet40/models/dgcnn_hmodel.py:177)、[ToPoincare 定义及未激活引用](/D:/Hycore/classification_ModelNet40/models/ops/manifold_layers.py:10)。

E 的 whole/part 观察核仅用 detached H、no_grad，不重新调用模型、不更新 BN，不制造正则梯度；训练总图直接为 CE。其有效分项梯度 none 与“贴边导致梯度消失”不是同一概念。证据：[E 观察包装](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/losses.py:38)、[E 总损失选择](/D:/Hycore/segmentation_ShapeNetPart/hycore_seg_v5/train.py:310)。

## 5. 后续讨论优先级（不是启动授权）

1. **先把已有事实分开**：结构上仅改尾部，与优化器/CE/训练轮数等协议是否一致分别陈述；BN 普通 train/eval 现已具备，不存在已证实“缺一个 HyCoRe BN 开关”的问题。
2. **优先用现有权重做有针对性的只读诊断**：同一批训练样本，在不污染原 BN 状态的条件下比较 train/eval 的 H 模长、EXP 贴边和预测，检查 running statistics/BN affine 尺度；训练数据重估 BN 是另一个待审选项。应保存和恢复 BN/Dropout 模式与 buffers，不能让诊断改掉正式权重。
3. **若追加训练，最清晰的非结构候选是 label smoothing 单开关**；全网 clip1 也是已知分类配方，可在有梯度证据时单独比较。二者都不能与头层修改、正则目标变化一并归因，不能承诺改善。
4. **保持已经正确复用的部分**：RSGD、LR 比例、M 完整参数化和初始化、测试 eval/no_grad、一次编码的分割支路。不要盲目增加第二次编码、把测试设为 train，或激活官方分类根本未使用的 ToPoincare/grad_fix 来声称更忠实。

本次没有实施以上诊断或新增实验。若只根据当前一个种子的 D/E 完成结果，最多说明该设置下正则未带来分割收益，不能从此判定 BN 或 label smoothing 是原因。
