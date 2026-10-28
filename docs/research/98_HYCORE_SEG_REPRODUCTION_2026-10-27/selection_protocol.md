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
