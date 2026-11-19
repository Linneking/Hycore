# ScanObjectNN PB_T50_RS：公开数据来源与小请求核验

2026-11-20。为已授权两组训练定位数据；本轮只浏览来源并发公开元数据／HEAD请求，没有连接服务器、下载大数据、提交表单或接受条款。

## 1. 首选：PointMLP作者认可的GitHub release

PointMLP官方当前`classification_ScanObjectNN/ScanObjectNN.py`的下载函数明确指定：

```text
https://github.com/ma-xu/pointMLP-pytorch/releases/download/dataset/h5_files.zip
```

来源链：[作者官方loader](https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ScanObjectNN/ScanObjectNN.py) → [作者release页，scanobjectnn_dataset](https://github.com/ma-xu/pointMLP-pytorch/releases/tag/dataset) → 上述公开asset。该loader随后读取`main_split/{training,test}_objectdataset_augmentedrot_scale75.h5`，与HyCoRe固定源码和数据作者PB_T50_RS文件命名一致。

本机HEAD实测：HTTP **200**，最终域`release-assets.githubusercontent.com`，`Content-Length=315941783`字节（约316MB），支持`Accept-Ranges: bytes`；无登录。GitHub release API请求当时返回403 rate limit，故本轮没有取得官方asset digest，也没有计算完整压缩包哈希。HEAD的ETag不是SHA256，不能充当内容完整性证明。

首选获取方式：从该作者release取得包，只取计划需要的含背景`main_split`两文件；解压后先核对文件形状、标签及下方独立镜像公布的SHA256，保存实际下载来源和实际哈希，不因文件名相同便宣称身份通过。

## 2. 备用及校验参照：公开Hugging Face镜像

镜像：[cminst/ScanObjectNN](https://huggingface.co/datasets/cminst/ScanObjectNN/tree/main/main_split)，维护者页面显示Computer & Math Institute。它包含`main_split`与`main_split_nobg`，须取前者。此镜像不是ScanObjectNN数据作者官方仓库；许可栏为`other`，来源身份应在验收记录中明确，不冒称作者亲自发布。

| 文件 | HEAD实测字节数 | 文件页公开SHA256 |
| --- | ---: | --- |
| `training_objectdataset_augmentedrot_scale75.h5` | 253395628 | `bd1ebda43155b112fac0ec9335723e66c4885449022002a2a8d370b56376eed5` |
| `test_objectdataset_augmentedrot_scale75.h5` | 63029147 | `f23822ee2f1e1d18331e91334d74adb10c8718d72d503bfa76ebe0a9348a7ea1` |

两个直接URL均HEAD **200**、无需登录、支持bytes Range：

```text
https://huggingface.co/datasets/cminst/ScanObjectNN/resolve/main/main_split/training_objectdataset_augmentedrot_scale75.h5
https://huggingface.co/datasets/cminst/ScanObjectNN/resolve/main/main_split/test_objectdataset_augmentedrot_scale75.h5
```

哈希来源：[训练文件页](https://huggingface.co/datasets/cminst/ScanObjectNN/blob/main/main_split/training_objectdataset_augmentedrot_scale75.h5)、[测试文件页](https://huggingface.co/datasets/cminst/ScanObjectNN/blob/main/main_split/test_objectdataset_augmentedrot_scale75.h5)。页面还显示Xet hash；HEAD ETag对应Xet对象，不同于表中普通文件SHA256，不能混淆。

另定位到[adfishing/scanobjectnn-h5](https://huggingface.co/datasets/adfishing/scanobjectnn-h5/tree/main)，目录列出同名两HDF；本轮未核实其内容哈希或直接可达性，不作为已验收来源。

## 3. 官方作者站点与后续身份验收

[ScanObjectNN作者站点](https://hkust-vgd.github.io/scanobjectnn/)目前通过Terms of Use表单提供链接；本轮没有提交表单。[官方数据README](https://github.com/hkust-vgd/scanobjectnn#h5-files)确认PB_T50_RS后缀、main_split和nobg的区别，可作为文件身份语义依据。旧HyCoRe/PointMLP镜像地址的403不表示当前作者release不可用。

上述请求只确认来源链、公开可达性和发布的文件指纹。实际数据仍需下载后确认：两HDF确为含背景PB_T50_RS/main_split、data与label形状、15类计数、有限性及SHA256；不能将本轮HEAD检查写成已完成数据验收。只需约316MB的作者简版包，不必为了两份HDF下载官方所有变体约13GB的完整处理包。
