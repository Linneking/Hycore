# ModelNet40 作者入口 H0 复现

用户批准一条完整训练。原作者提交 `ef56a73020e309b1009c486fea1f601d3f7b3252` 的 PointMLP+HyCoRe launcher、模型、损失、采样和选模代码随同冻结。从空检查点目录执行作者 launcher，使用 seed 4780、全量 9,840 个训练对象、2,468 个测试对象、SH32、每轮 307 步、300 轮。保留原生 `c=1` 和 margin `1000/N、4`，无新增 HIER。原入口每轮测试并按 OA 选 best，配套 AA 取同轮。

外层脚本 [launch.sh](launch.sh) 放置官方数据，在 CUDA 初始化前仅限定可见 GPU，并装入已验收的兼容 PointNet2 扩展。源码文件来自发布提交的 Git 归档。预检比对 TS02 的全部 HDF5 文件指纹和对象数，并记录作者加载器实际枚举的文件顺序。训练过程使用作者加载器的自然枚举。

历史参照为 TS02 的 94.003% test OA。正式结果用于补齐直接执行作者 launcher 的记录。对比时需查实际文件枚举和运行时版本。 [TS02 验收](../66_V9_B64_CONTROL_AND_OFFICIAL_REPEAT_2026-10-21.md)
