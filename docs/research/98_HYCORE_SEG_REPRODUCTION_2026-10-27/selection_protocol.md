# 2026-10-28：选模核查、用户决定和首轮HyCoRe-seg实验

用户明确选择沿用已选AnTao PyTorch：train+val训练、逐轮test Instance mIoU选best；全部seed4780。该直接决定覆盖此前新分割计划的val选模偏好。既有val选模DGCNN继续原任务，不在中途改变数据或选模。

## 一手证据

- [HyCoRe-seg论文§4](https://iris.polito.it/retrieve/handle/11583/2982650/73ff1be4-1ad0-44ac-b539-34f81d452df4/Hyperbolic_regularizer-1.pdf)没有明确val/test选best、每轮评估频率或early-stopping规则。分类HyCoRe代码的test选模不能证明分割作者采用同一规则。
- [AnTao固定源码](https://github.com/antao97/dgcnn.pytorch/blob/07d534c2702905010ec9991619f552d8cacae45b/main_partseg.py)：train_dataset用trainval，逐轮test；mean(test_ious)>=best_test_iou则保存，包含并列取较晚轮。默认seed由用户覆盖为4780。
- [DGCNN作者TF分割源码](https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/part_seg/train_multi_gpu.py)：TESTING_FILE_LIST实际指向val_hdf5_file_list.txt；每轮eval且每5轮保存，无按best自动筛选。作者代码与社区默认必须准确区分。

此前val-selected工程组不是AnTao默认复现，保留为独立协议。新组明确标test-selected，不把test叫val，不采用test early-stop，固定200轮。

## 数据与比较

新数据目录恢复公开raw release的原train+val列表：12137+1870=14007条，test2874；保留原列表中的9个train-val重复条目和26个trainval-test共享ID。这是用户选定的benchmark成员规则；原无重叠数据和结果不改。XYZ预处理复用同一已校验缓存，原Stanford HDF5不可用，因此仍不能称HDF5字节/随机流完全复现。

新实验全部200轮、seed4780、batch32/eval16、k40、2048XYZ、dropout.5、LR.1→.001余弦、WD1e-4、momentum.9、8workers、普通CE。DGCNN用SGD；双曲头及完整组都用RSGD。完整组两项各.01，margin1000/N′与4，随机part200–600。主指标为unmasked Instance mIoU；masked与Class mIoU分列。

最小比较：新test-selected DGCNN、双曲头无正则H0、完整HyCoRe-seg HC。H0/HC可隔离正则；SGD/RSGD差异使DGCNN/H0不能单独归因为头部。H0/HC匹配初始化、输入流、采样协议；part独立RNG。两者fresh训练，不续基线或短测权重。

真实HC初始2×2短测已完成，参数/损失有限；whole/part有较高投影率、径向margin均未满足。正式版本增加首步正则与CE到共享层的独立梯度读出，必须确认正则能回传再进入生产；不能把短测有限性、投影或分类结果当作真实层次已建立。

新发布版本先CPU合同及合成test-selection端到端检查，再独立full-size2×2 smoke（仅clean train/val，无test），通过后重新初始化正式test-selected训练。旧val基线运行中源码冻结，所有新部署/结果独立，不覆盖旧目录。

## 本轮提交状态

发布源`1ffd9f5`，本机/8000各16CPU合同通过；合成完整HC入口两轮8更新，明确每轮test、累计2次评估，最终指标标注test且没有混写val。初始真实HC短测使用早期冻结源c153715，2×2/4更新有限，未读test；不把它算当前正式版本完整验收。

2026-10-28 10:06（Asia/Shanghai）只读核查：恢复原成员的数据缓存仍在生成，三个独立后台启动器均为`waiting_for_prepared_data`，**尚未开始新协议正式训练**。

| 运行ID | 目标GPU | 内容 |
|---|---:|---|
| `seg_DGCNN_testsel_s4780_20261028_A` | 0 | 匹配test-selected DGCNN，旧val基线占用时等待 |
| `seg_H0_testsel_s4780_20261028_A` | 1 | 仅双曲头、RSGD、无正则 |
| `seg_HC_testsel_s4780_20261028_A` | 2 | 完整HyCoRe-seg；共享梯度非零、margin1000/N′与有效负样本必须通过 |

共同门槛：数据完整manifest出现后检查空闲GPU，独立2×2 smoke、BN4、有限模型/状态、200轮cosine、零test，通过后fresh200。启动器会自动继续，不需重复提交；任何门槛失败会保留日志并标记failed，不跳过验收。现有val基线、原数据和旧服务器结果未变；本轮没有声称新协议最终成绩。
