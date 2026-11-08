# HyCoRe Seg 接手入口

2026年11月8日更新：无需传输prepared数据的完整步骤见 [从公开下载复现](REPRODUCE_FROM_DOWNLOAD.md)。新prepare入口执行历史两阶段处理，并用可信内容指纹验证；旧权重复评显式使用--data-check content。原exact默认与训练模型保持不变。

这份交接保留已训练实验的实际实现。先读研究交接主文，再按实验注册表定位配置和原始提交。

## 交接范围

主开发线32条记录，其中C停止于180轮，其他31条到达200轮。另有1条独立验证集选模的DGCNN。正文详述17组，15组配套列出。experiments.json保留全部记录和历史别名。SEG02 V1-H0与SEG17 H0是不同实验。

当前参考为SEG22 TAN-R0。其末20轮Instance/Class为85.1054/83.0379%，同协议DGCNN为85.2892/82.0569%。已有正则变体尚未取得末段双指标优势。

## 取得代码与环境

使用完整Git工作副本，切换到交接tag。原有v1到v15实现和分类几何来源文件都需要保留，native.py会验证上游源码指纹。

    git clone git@github.com:Linneking/Hycore.git
    cd Hycore
    git checkout hycore-seg-reproduce-20261108-v2
    python -m pip install -r handoff/segmentation/requirements.txt

历史生产环境为Linux/Python3.9.23、Torch2.8.0+cu128、CUDA12.8、Geoopt0.5.1、NumPy1.26.4、h5py3.14.0。代码设置了固定seed和部分确定性选项，跨硬件或版本仍无法据此保证逐位一致。

## 数据协议

接手复核旧成绩时优先取得现有prepared数据。当前trainval14007/test2874，2048XYZ、k40，包含原raw列表的9条train-val重复和26个trainval-test共享ID。原train-test23、val-test3，合并前已经存在。test-selected开发结果和独立验证协议分开。

prepared目录含trainval.h5、test.h5、manifest.json。默认exact评估要求原清单和文件身份。无需传输数据时，按REPRODUCE_FROM_DOWNLOAD.md下载原始包并运行prepare.py；该入口验证实际数组与历史参考一致。历史权重复评加--data-check content，新训练沿用自身清单。仅凭对象数相同不足以认定数据一致。

若另建数据，先用hycore_seg_v1.prepare_data处理normal archive，再用prepare_antao_protocol恢复原列表成员。archive SHA为0e26411700bae2da38ee8ecc719ba4db2e6e0133486e258665952ad5dfced0fe。另建无交集划分时，需重新运行匹配DGCNN，不能覆盖历史数据。

## 实验和源码定位

experiments.csv便于查看。experiments.json含父实验、结构、正则位置、配置、历史run ID、源码SHA及完整训练参数。configs/SEGxx.json是从生产manifest提取的配置。原始commit始终保留，交接分支是统一索引与兼容入口。

统一launch默认打印命令，显式加--run才训练。新的out必须不存在，训练入口会核对GPU完全空闲。用--experiment指定SEG编号或唯一历史名。

    python handoff/segmentation/launch.py --experiment TAN-R0 --data DATA_DIR --out NEW_RUN_DIR --gpu 0
    python handoff/segmentation/launch.py --experiment TAN-R0 --data DATA_DIR --out NEW_RUN_DIR --gpu 0 --run

新训练均fresh200，seed4780。C若从头执行配置也会跑200轮，历史结果仍只代表180轮。F/G/H的迁移恢复分段归并到各自身份，统一入口给出fresh运行，旧断点续训须回到原始提交和原保存配置，按对应版本恢复模型、优化器、调度器及所有RNG状态。

## 关键权重与评估

private_assets中交接DGCNN、TAN-R0、TAN-R三个best检查点，附SHA核验及服务器best/last位置清单。best用于选中模型的性能，last用于固定末轮分析或恢复。完整检查点含BN、优化器和随机状态。仅加载可信的项目检查点。

    python handoff/segmentation/evaluate.py --experiment TAN-R0 --checkpoint TAN-R0_best.pth --data DATA_DIR --output NEW_RESULT.json --device cpu

若评估使用GPU，先检查nvidia-smi，仅选完全空闲卡，并设置CUDA_VISIBLE_DEVICES。eval入口严格加载模型，核对数据manifest和test文件指纹，保存对象数、checkpoint SHA和两项指标，并检查评估没有改变模型/BN。

DGCNN best为e186，预期Instance/Class85.3825/81.7770%。TAN-R0 best为e180，85.3972/83.0043%。TAN-R best为e181，85.0662/82.8230%。第200轮和末20均值另见注册表。不同机器上的浮点数差异应结合环境核对。

## 本次交接验收

已逐一解析33份配置，三份关键best文件已核对SHA，所选轮次与final_test一致。沿用既有评估函数，新增入口不改变IoU定义。145记录已有固定e200 DGCNN/TAN-R0全2874对象复评。交接整理期间三张GPU均占用，本轮未新增三模型全量推理；接手时按以上命令复核。

GitHub包含代码、配置和脱敏记录。private_assets保留在本机交接包，通过双方认可的私有渠道传递。完整日志、数据和其余检查点从私有资产索引定位。

代码离线归档与Word、注册表、入口及三份关键权重一并放在本机交接包。离线归档用于阅读；训练时请使用固定tag的完整Git工作副本。验证字段no_logged_nonfinite_state只表示日志没有记录非有限状态，旧日志缺少字段时不能推断做过完整张量检查。

SEG33必须使用独立train/val/test文件的早期去交集缓存，不能用主开发的trainval/test缓存代替。其数据身份见注册表与私有运行记录。

另已在本机项目环境Torch2.6.0+cu126、Geoopt0.5.1完成三模型CPU接口检查：严格加载保存权重，输入2个64点的合成对象，输出形状均为2×50×64、数值有限、模型和BN状态保持不变。该检查验证权重与接口兼容，不是完整分割性能复评。生产环境仍以此前列出的Torch2.8.0+cu128为准。
