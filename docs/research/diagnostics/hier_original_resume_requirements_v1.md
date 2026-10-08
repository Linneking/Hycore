# 原版 HIER best：图像 epoch 代理使用审查与续训条件

日期：2026-10-08。范围：已提供的 CUB、Cars、SOP 三个 best；只解释恢复条件和身份验收，不启动训练或 GPU 作业。

## 1. 当前权重足够检查什么

已验收的 loader manifest 记录三个检查点分别为已完成第 17、45、96 轮，均有 632 个网络状态张量、105 组 BN running mean、512×512 的 HIER 切空间 `lcas`。三个模型的有效配置都是 `bn_freeze=True`、`hyp_c=.1`、`clip_r=2.3`、维度 512、`global_crops_number=1`。SOP 命令里的 `--bn_freeze False` 被官方 `type=bool` 解析为 True，不能据命令字符串把它当成未冻结 BN。

这些状态足以恢复网络前向和 HIER 样本项／代理项的距离、选祖先及梯度。在固定 best 上跑一个源式采样、随机增强的完整训练 epoch，但不做任何 optimizer 更新，可以观察“当前 best 条件下的一轮使用”。这不是当时训练最后一轮的历史记录，也不能推断代理在继续优化后的命运。

原作者源码将 PA 与 HIER 设置为两个独立 loss：`PALoss_Angle.proxies` 是按训练类别数量分配的主任务分类代理，`HIERLoss.lcas` 是 512 个共享层次代理。两种代理不能混用。

## 2. 为什么不能声称严格续训

| 恢复项目 | 当前状态 | 对一轮审查的影响 |
|---|---|---|
| 网络／BN／HIER lcas | 完整，身份已验收 | 可恢复前向和 HIER 目标 |
| PA 分类代理参数值 | **未保存**；交接和实读 manifest 均明确 | PA 主任务原损失和原梯度无法恢复 |
| AdamW optimizer | 已保存状态与参数组 | 动量不是参数值，不能补回缺失的 PA proxies |
| AMP scaler | 已保存 | 可恢复缩放状态；不消除版本／设备差异 |
| 学习率 | 源式 schedule 可由有效 args、已完成 epoch、真实 loader 长度重建 | 应从下一 epoch 的 global step 索引继续，不能重启 schedule |
| Python／NumPy／Torch／CUDA RNG、worker RNG | 未列入已保存顶层状态 | 下一轮随机增强、采样及 Gumbel 不能逐位重放 |
| sampler epoch | 可由保存 epoch 推算；随机流缺失 | 可复现协议形式，不能恢复原随机序列 |
| 原 `load_from` 入口 | 官方 train.py 仅声明参数，未使用它恢复状态 | 不能直接加该参数就视为完成恢复 |

PyTorch optimizer `state_dict()` 保存每个参数的优化器状态和参数组中的参数 ID，不保存该参数当前值；AdamW 的 `exp_avg`、`exp_avg_sq` 与 `step` 无法唯一确定当前参数。重新随机初始化 PA proxies 再装载旧动量会改变目标，并不是原协议续训。修改为 HIER-only 更新也属于新的机制分叉，不能冠以原训练的延续。

只读激活审查不需要 PA 值，也不需要 teacher。若将来确实要原协议更新，需要向原训练者补要 `sup_metric_loss.state_dict()`（尤其 `PALoss_Angle.proxies`）；若要求逐位恢复，还需两 rank 的 RNG、采样及 worker 状态和原环境。

## 3. 正确的一轮采样和激活定义

源式设置 local batch 90、world size 2、global batch 180、IPC 2：每 batch 选 90 个类别，每类有放回抽 2 个图像。官方 `UniqueClassSampler` 的 batch 数由训练池大小决定：

| 数据集 | 训练图像／类别 | 完整源式 epoch batch 数 | global 图像槽数 |
|---|---:|---:|---:|
| CUB | 5864／100 | 32 | 5760 |
| Cars | 8054／98 | 44 | 7920 |
| SOP | 59551／11318 | 330 | 59400 |

这些是采样槽数，不是不同图像覆盖数；有放回抽样并且 drop-last。不能把“每个对象遍历一次”称为源式训练 epoch。类别由 `batch_idx*10000+epoch` 的 NumPy seed 决定，具体图像选择另使用 Python random 派生种子；必须记录审查 seed 和模拟下一 epoch 索引。

增强为 `RandomResizedCrop(224, scale=(.08,1), BICUBIC)`、随机水平翻转、ImageNet 归一化。此源式 ResNet50 用 avg pool＋max pool，BN2d 被冻结；普通 torchvision ResNet50 单 avg pool 并不等价。已有完整网络状态时，用 `pretrained=False` 建立相同架构再 strict load 即可，避免无意义地下载或加载额外 ImageNet 初始权重。

每一代理应区分以下口径，并分别对 sample 项和 proxy 项记录：

1. 被 pair／triple 选择为祖先的次数；包括碰撞。
2. pair 与 triple 不同的有效选择次数；官方 loss 将碰撞整项置零。
3. 非碰撞且正 hinge 的祖先次数。
4. 每个分项产生非零／超过数值阈值的代理参数梯度次数、梯度范数、径向／角向分量。
5. proxy 项中作为查询节点 i、j、k 的次数和产生的梯度；这与“被选为祖先”是不同角色。

代理项查询池包含全部 512 代理，即使某代理从未被 whole 选为祖先，也可能通过自身查询角色或代理项祖先角色收到梯度。全项梯度相消也不能用来证明两分项都无作用。应对各分项单独求梯度并记录 combined，保持官方 straight-through Gumbel、距离数值及 ToPoincare 梯度放缩，不以 argmin 的冻结统计替代随机训练条件统计。

优先做无更新 epoch。它可回答激活来源，但不会产出“再训练后改善了”的结论。源检查点、BN、lcas 与缓存均应保持 SHA 不变。

## 4. 图像下载体量与来源

用户后续明确授权自行下载，覆盖此前累计大于 3 GB 时只提供命令的限制。这里仅提供已查证资料，下载和存放由执行任务统一安排，避免并发重复下载。

| 数据集 | 图像总量 | 压缩包体量 | 查证来源 |
|---|---:|---:|---|
| CUB-200-2011 | 11788 | 官方约 1.2 GB，TFDS 标示 1.11 GiB | CaltechDATA、TensorFlow Datasets |
| Cars | 16185 | 原 train＋test 包合计 1956619750 bytes；完整 car_ims 镜像 1956628579 bytes | TFDS checksum、镜像 HEAD／LFS |
| SOP | 120053 | 3083860082 bytes（2.872 GiB） | 作者仓库、TFDS checksum |

合计压缩包约 **6.23 GB（5.80 GiB）**，建议图像目录预留 **15–20 GB** 给压缩包、解压文件、身份映射和运行输出。这是容量预算，不是尚未实测的解压目录精确大小。本次 source/metadata 核查没有下载大图像包。

- CUB 官方文件：[CaltechDATA](https://data.caltech.edu/records/65de6-vp158)，文件 URL `https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1`，原包 MD5 `97eceeb196236b17998738112f37df78`。本机 HEAD 返回 403，需服务器实测；fastai 官方代码也提供 `https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz` 镜像。若镜像重新打包导致包 hash 不同，必须验收图像路径、计数和同图像前向，不能直接声称与原包字节同一。
- SOP：[作者仓库](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16) 提供 `ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip` 与 Google Drive ID `1TclrpQOF_ullUP99wk_gjGN8pKvtErG8`。TFDS 对原 ZIP 的 SHA256 为 `b04fa78210e69fab050ce73939550e1cd0f96890bdd98ec9fee732c3b756ef48`。以原 `Ebay_train.txt`／`Ebay_test.txt` 与缓存中的 image_paths 验收，不另做 ImageFolder 重排序。
- Cars：torchvision 官方明确昔日 Stanford 下载地址已失效。fastai 官方代码提供 `https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz`，本机 HEAD 200、1957803273 bytes；该包布局需解压后核查。原 `car_ims.tgz` 的第三方镜像 `https://huggingface.co/datasets/XiN0919/FGVC/resolve/921e8dce38536cd5982739d85d5e9235d2cb2a76/car_ims.tgz` HEAD 200、1956628579 bytes，上传 LFS SHA256 为 `4d96e3e3e892c6b7e4a71959325c7de4804037d27ba5aaf1c756100fd94a5342`；其原标注为该 repository 的 `cars_annos.mat`。第三方镜像必须以 annotation 和缓存身份匹配验证，不能仅凭文件名认为它对应同学的版本。

图像 archive 含全数据集，审查只 forward 各自训练类；CUB／Cars 的官方 image-wise train/test 包不能当作 HIER 类别互斥划分直接使用。

## 5. Cars 自定义命名的必要验收

实读用户提供的 `cars_annos.mat`，其路径是 **car_ims/000001.jpg 至 car_ims/016185.jpg，六位、从 1 起始**；交接 prose 中“00000 至 16184”的描述与实际文件不一致。以 MAT、缓存 image_paths 和完整 sample_id map 为准。

只读加载镜像的小 MAT（394471 bytes）并按 `(class,bbox_x1,bbox_y1,bbox_x2,bbox_y2)` 与用户 MAT 连接，16185 行全部匹配：16181 行唯一，4 行形成两个歧义对。第一行目标路径对应不同的镜像原文件名，说明同名路径绝不能直接套用。

两组具体歧义身份保持在私有执行记录。下载后先检查候选文件内容是否完全相同；若非完全相同，做固定 clean-transform 和已验收 best 网络前向，与提供缓存的相应对象嵌入匹配来解歧义。保留原镜像不改名，在新目录建立映射／副本或由 loader 显式解析映射；全部 identity 验收后才将该数据目录用于 epoch 审查。

## 6. 可核查源码证据

源版本 `3986a744a1a54fd357e307d1cb3f2e81910b9ffc`：

- [train.py：BN argparse 和 load_from 声明](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/train.py#L75)
- [train.py：train_one_epoch、两 loss 和 LR 索引](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/train.py#L124)
- [train.py：模型、独立 PA/HIER criterion、optimizer、保存内容](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/train.py#L256)
- [losses.py：HIER mining 和样本／代理两分项](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/losses.py#L17)
- [losses.py：PA proxies 参数](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/losses.py#L248)
- [sampler.py：UniqueClassSampler](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/sampler.py)
- [models/resnet.py：avg＋max 和 BN 冻结](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/models/resnet.py)
- [utils.py：MultiTransforms 与四组 optimizer 参数](https://github.com/sung-yeon-kim/HIER-CVPR23/blob/3986a744a1a54fd357e307d1cb3f2e81910b9ffc/hier/utils.py#L97)
- [PyTorch optimizer state_dict 内容定义](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html)
- [TFDS SOP archive 大小与 SHA256](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/stanford_online_products/checksums.tsv)
- [TFDS Cars 原 archive 大小与 SHA256](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/url_checksums/cars196.txt)
- [fastai 官方 URL 表](https://github.com/fastai/fastai/blob/master/fastai/data/external.py)
- [torchvision 官方 Cars 下载失效说明](https://github.com/pytorch/vision/blob/main/torchvision/datasets/stanford_cars.py)

未读取或记录任何服务器 host、凭据或私钥；没有修改输入权重、缓存、原标注或原代码。
