# 从公开下载复现 HyCoRe Seg

代码入口：[GitHub固定复现版本](https://github.com/Linneking/Hycore/tree/hycore-seg-reproduce-20261108-v2/handoff/segmentation)。

仅从头训练时，这份说明加仓库访问权限即可；复评历史权重时，还需已有154交接包中的三份权重。收到旧交接包的使用者，本次只需更新这份说明和Git代码。

本流程供已收到154交接包的同学使用。数据由你在自己的机器下载和处理，无需从原服务器拷贝。主线对应32条历史开发记录；SEG33独立验证基线另用第一阶段数据，不混入主线比较。

## 1 获取后继代码

在Linux上使用完整Git工作副本。若已有仓库，执行fetch和checkout两条即可。

```bash
git clone git@github.com:Linneking/Hycore.git
cd Hycore
git fetch origin --tags
git checkout hycore-seg-reproduce-20261108-v2
```

原交接标签hycore-seg-handoff-20261107保留。新标签新增数据准备和内容校验入口，模型及历史训练代码不变。离线源码ZIP没有Git元数据，正式运行请使用上述完整仓库。

原55MB交接包继续提供Word、指标日志和三份权重；执行程序请使用新标签的代码，不再使用旧包中的code.zip或entrypoints副本。

## 2 创建匹配环境

```bash
conda create -n hycore-seg python=3.9.23 -y
conda activate hycore-seg
python -m pip install -r handoff/segmentation/requirements.txt
```

历史环境是Python3.9.23、Torch2.8.0+cu128、NumPy1.26.4、h5py3.14.0、Geoopt0.5.1。不要使用Python3.13安装此旧环境。4090可以读取同一数据格式；驱动需支持所装PyTorch。固定seed不保证不同CPU、驱动或GPU下200轮逐位相同。

## 3 下载同一份公开原始数据

在仓库根目录执行。data/已被Git忽略。原始压缩包约707MB，解压、两阶段中间文件与最终数据需要额外磁盘空间，建议至少预留6GB。

```bash
mkdir -p data
curl --location --fail --retry 3 \
  'https://huggingface.co/datasets/wangps/shapenet_segmentation/resolve/dbde146b974e1fc8628b47b1b1c4e50d8bc1a2ef/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip' \
  --output data/shapenetpart_normal.zip
```

准备入口会核对压缩包SHA256：
`0e26411700bae2da38ee8ecc719ba4db2e6e0133486e258665952ad5dfced0fe`。
不要换用名称相近的其他HDF5版本。

## 4 一条命令完成两步处理和内容核验

```bash
python handoff/segmentation/prepare.py \
  --archive data/shapenetpart_normal.zip \
  --output data/shapenetpart_rebuilt
```

此步骤在CPU上执行，不需要空闲GPU。新入口仅将HDF5无损块缓存提高到每数据集64MiB以改善逐对象读写，原预处理算法不变；它执行两个历史预处理脚本：

1. prepare_data：全点XYZ中心化、单位最大半径归一化；舍弃法线，取前2048点；点不足时按索引循环补齐。先生成去跨集合重复的train/val/test。
2. prepare_antao_protocol：按原发布列表顺序恢复主线trainval14007、test2874，保留历史成员及重复关系。
3. data_identity：对坐标、逐点标签、物类标签、对象ID和各自顺序，核对从原历史数据计算的参考指纹。

输出目录包含trainval.h5、test.h5、manifest.json、rebuild_verification.json。中间目录为同级shapenetpart_rebuilt_disjoint_source，先保留。为了保护已有数据，两个输出目录都必须是新路径；失败后查明原因，重试时换一个输出名，不覆盖旧目录。

成功时最后打印“Verified historical array contents”。你也可单独复查：

```bash
python handoff/segmentation/data_identity.py --data data/shapenetpart_rebuilt
```

内容校验不会要求生成时间、Git版本或HDF5压缩布局与原文件一致，但要求实际训练/测试数组一致。若失败，应核对下载版本、预处理和环境，不能只关闭校验。对全部CPU/NumPy实现的位级一致性不作保证。

## 5 用交接包权重复评

把下面HANDOFF替换成154交接包解压后的根目录。三份权重已经在旧包内，无需重新传数据或权重。

先运行nvidia-smi，选择完全空闲的GPU。以下以GPU0为例；若是其他卡，修改CUDA_VISIBLE_DEVICES。输出JSON必须是新文件。

```bash
export HANDOFF=/path/to/154_HYCORE_SEG_HANDOFF_2026-11-07
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python handoff/segmentation/evaluate.py \
  --experiment TAN-R0 \
  --checkpoint "$HANDOFF/private_assets/TAN-R0_best.pth" \
  --data data/shapenetpart_rebuilt \
  --output results/TAN-R0_best_rebuilt.json \
  --device cuda --data-check content
```

DGCNN和TAN-R分别把experiment、权重名与输出名改成DGCNN、DGCNN_best.pth及对应结果名，或TAN-R、TAN-R_best.pth及对应结果名。预期历史结果如下，均是按Instance选中的best，同一检查点取Class：

| 模型 | 轮次 | Instance mIoU | Class mIoU |
|---|---:|---:|---:|
| DGCNN | 186 | 85.3825% | 81.7770% |
| TAN-R0 | 180 | 85.3972% | 83.0043% |
| TAN-R | 181 | 85.0662% | 82.8230% |

--data-check content用于“历史权重 + 重新生成的数据清单”：先由检查点的原manifest SHA找到可信参考，再检查本地文件完整性及实际测试数组。未知数据参考会拒绝。结果会记录采用哪种校验。默认exact模式仍适用于原始缓存，或新训练权重与其自身缓存。

## 6 从头重训

只有要重新训练时才执行。数据准备和复评不要求先重训。以下仍须使用完全空闲GPU，输出目录必须不存在。

```bash
python handoff/segmentation/launch.py \
  --experiment DGCNN --data data/shapenetpart_rebuilt \
  --out runs/reproduce_DGCNN_s4780_A --gpu 0 --run

python handoff/segmentation/launch.py \
  --experiment TAN-R0 --data data/shapenetpart_rebuilt \
  --out runs/reproduce_TAN-R0_s4780_A --gpu 0 --run

python handoff/segmentation/launch.py \
  --experiment TAN-R --data data/shapenetpart_rebuilt \
  --out runs/reproduce_TAN-R_s4780_A --gpu 0 --run
```

三条逐个运行，勿把同一卡同时启动三份。仅查看实际命令时去掉--run。其他实验用experiments.json中的唯一名称或SEG编号。配置保留历史seed4780、200轮、batch32、2048点和各组模型/损失设置；不为4090另改batch或结构。部分入口还核对共享初值，失败时先核对固定环境，不直接删除约束。

这是从头训练流程。使用重建清单不能直接恢复旧检查点的训练状态，原resume入口仍要求旧数据身份。新训练生成的检查点会绑定它自己的新manifest，后续对同一缓存评估用默认exact即可。

## 成绩解读

主线历史协议合并trainval、逐轮test选best；原raw列表的交集保留，trainval内部9条重复、与test共享26个ID。当前流程复现这套历史设置，不把它称作无交集独立测试。SEG33采用第一阶段独立train/val/test和验证选模，不能和主线混排。

先确认旧权重复评，再判断从头训练差异。4090与5090的数值误差、软件和随机流可能造成重新训练分数波动，不能承诺逐位复现。训练成绩请同时记录best、同轮Class、末轮和末20轮均值。
