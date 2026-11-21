# ScanObjectNN 作者入口 seed 12 配对

用户批准用作者原入口重跑，并在同一入口上增加一组 HIER。第一组用于比较作者发布方式与此前 seed 4780 的本地入口。第二组用于测量 HIER 在新入口和 seed 下的增量。此前两组的入口和 seed 都与本批不同。

| 实验 | 共同底座 | HIER | 分配 |
| --- | --- | --- | --- |
| SO-S12-H0 | 作者 PointMLP＋HyCoRe 的 `main()` 与 launcher | 关闭 | 8000 GPU0 |
| SO-S12-HIER | 同一个 `main()`、launcher、运行时与数据 | 上一轮 HIER 配方 | 8000 GPU1 |

两组从空目录和随机初值启动。数据为 ScanObjectNN PB_T50_RS、main_split、含背景，train 11,416／test 2,882。作者数据加载器直接打开预先放到其预期目录的 HDF 文件，启动前检查文件哈希。作者 launcher 的四个参数为 LR .1、300 轮、原实验名、8 workers。其余使用作者默认值，包括 seed 12、SH32、1024 点、15 类、WD `1e-4`。每轮 356 次更新，300 轮 106,800 次更新。作者训练与测试函数、完整 test OA 严格上升才更新 best 的规则都保留。AA 在结果汇总时从 best OA 同轮日志读取。

[数据身份](../178_SCANOBJECTNN_HIER_PAIR_PLAN_2026-11-20/dataset_identity_verified.json) · [作者 launcher](https://github.com/diegovalsesia/HyCoRe/blob/ef56a73020e309b1009c486fea1f601d3f7b3252/classification_ScanObject/launcher_pointmlp.sh) · [上轮与发布源码的差异](../179_SCANOBJECTNN_HIER_PAIR_LAUNCH_2026-11-20/run.md#so-h0-的-872-复现差距审查)

两个入口都采用同一最小兼容处理。运行时登记 `modelsnet` 命名空间别名，外层绑定到验收的空闲卡，并把已核验数据放到作者的原目录。H0 运行源码 `main()`，保留其中所有函数。HIER 组在原 `train()` 的基础总损失后加 HIER，在原主优化器之后更新代理，并在每轮保存代理及其统计。实际深度映射使用 c=.1、切空间 cap2.3 与独立 GS。沿用当前 32 候选、512 代理、初值系数 .9、A1、Ks20/Kp20、T50、两项权重各 .5，seed 由 12 派生。

运行环境为两组共同的 Python 3.9.23、PyTorch 2.8.0+cu128、Geoopt 0.5.1、NumPy 1.26.4、h5py 3.14.0、SciPy 1.13.1、scikit-learn 1.3.2、Matplotlib 3.9.4。PointNet2 使用上一轮验收的同一扩展，SHA-256 `ae139590cc43d313a0901a1b4cca5bbb73114ad92b693c762d18b0bc76a2f03d`。作者 README 给的是旧依赖的安装建议和 Ampere 示例。当前 Blackwell 卡采用官方支持的运行时。这是发布入口和配置复跑，环境版本如上。[作者安装说明](https://github.com/diegovalsesia/HyCoRe#installation) · [PyTorch Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)

生产提交 `b876d85238a8e8e254df2adab9a4dd26ee721476`，分支 `codex/scanobjectnn-author-launcher-s12`。正式目录标识为 `scanobject_author_s12_20261121_B`。首个短测在初始化前读不到 Conda 的 Torch 包元数据，已改为读取实际模块版本，旧短测目录保留。

两组同卡短测各执行 2 轮×2 步，初始模型 SHA-256 `b7fca0bf94678dabb180b033f38da8e5679c7678dd4e4949d08293b5e176f278` 一致。四批增强输入、标签、whole／part 裁剪完全相同，首步 CE／对比／局部层次损失同为 2.708136／3.982100／1.466736。结束时 Python、NumPy、PyTorch CPU 和 CUDA 主随机状态一致。HIER 的代理完成四次更新，样本及代理关系均实际产生。正式训练另用空目录，从作者 seed 12 从头初始化。

## 正式启动

北京 2026-11-21，SO-S12-H0 于 14:11:29、SO-S12-HIER 于 14:11:14 分别在 GPU0、GPU1 启动，显卡通过两次空闲检查。后台控制器已独立于登录会话，生产清单核对了作者 `main()`、seed 12、356 步、300 轮、软件环境与 HDF 指纹。首轮结果以原 `log.txt` 记录为准，日志从 epoch 0 开始计数。

14:16 起两组均完成正式首轮，保存 best／last。SO-S12-H0 的 e1 OA／AA 是 35.288%／25.184%，SO-S12-HIER 是 28.799%／22.001%。只读检查时 H0 已完成 e2 并更新 best，权重为有限值，两次前向对应 BN 计数 1,424，e2 best SHA-256 为 `45dc33961850a67743afdad40bb2a62c2ff4e7dabdd124b1ec2f7c5c3dcea016`。HIER 的 e1 best 权重有限，BN 计数 712，代理 AdamW 实际步数 356，e1 best SHA-256 为 `97e603c4173a9ecb33c32321f4123429c8c51b5235831f7635d98ce3301c15ac`。首轮用于核对运行，两组继续原预算。
