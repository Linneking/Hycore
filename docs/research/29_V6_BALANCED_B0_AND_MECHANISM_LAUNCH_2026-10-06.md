# V6同协议balanced64 B0与持续机制诊断

2026-10-06。用户已批准先独立启动主B0，再持续小规模机制诊断与据新证据形成的新实验，直到使用限额；诊断不阻碍主任务。实现分支`codex/v6-balanced-b0-mechanisms`。当前条目为执行记录，将在实际启动验收后补充身份与状态。

## 主任务定义

- 起点为原V6完整`checkpoint_epoch_020.pth`，format `hycore-hier-v6-h20-selfk300-1`；已完成4000次CE/intra更新，不含HIER更新。
- 继续e21–300，每轮200步、global64，两rank local32。8856/984训练/验证划分，seed22、32类×2有放回、workers2、eval32。
- 保留CE、`.01Rcontr+.01Rhier`、whole/part alias复写、child/whole两次普通BN、原FPS512、c1/D256、范数1裁剪。
- 模型RiemannianSGD LR.1到.005、momentum.9、WD2e-4、T_max300；恢复时scheduler last_epoch20，LR `.09896201103485577`，不重启cosine或动量。
- 仅关闭HIER及proxy更新。HIER的mining私有随机流、专用Gumbel generator与augmentation/模型随机流独立，B0不需要dummy HIER消费。
- 两rank BN/RNG/sampler完整恢复；源sampler均epoch19，下步set_epoch20。原源码不变，新入口严格核对完整源身份。
- 源e20最佳验证OA90.95528455%，恰为e20模型；作为共同前缀的候选保留。e21–300继续验证选模，最终选定模型官方test一次。
- e99/e200/e300固定结构存档；旧H20为同协议历史参照，未重放到可核对存档前不称逐位配对复现。

本次不重新跑20轮基础训练。用户要求的“后续HIER无需warmup”应用于新HIER：从头训练第一轮lambda_H=.5；从旧checkpoint分叉则分叉首步启用，并如实记录继承的前缀。B0自身始终没有HIER。

## 执行与资源

先检查完全空闲GPU，再进行受限恢复验收与独立后台主训练。主训练GPU与诊断GPU分开，运行文件不被后续诊断编辑。原结果目录、权重和其他用户进程不改。

本次source CPU检查确认完整checkpoint有模型/代理optimizer、两scheduler、两rank BN/RNG/sampler、best_net、split和ID；原e20文件160917100字节，独立SHA256为`cacb8ecd394e66c57c1cc0e95f9ab81c62c2dd47c74eba062f781409b8667e50`。生产commit、运行时间/GPU及新目录仅在实际验收后记录，不提前写成已启动。

### 已完成短测与生产派发

- 生产实现commit `f649fcc9a61b9ace22f7b793c3837f379ba6faa2`。六项恢复合同/采样测试在本地及服务器通过。
- 新目录的双卡恢复短测完成e21的两步，官方test未读取；每rank模型参数、全部BN buffer、optimizer整树、scheduler、sampler、完整RNG和初始模型身份exact核对均通过。
  每步BN+2、原alias复写、finite、clip与c1检查通过；首/末梯度及更新参数副本差为0。
  完整短测checkpoint已重载验证且带`continuation_smoke_not_resumable=true`，不用于生产。
- 正式B0于**2026-10-06 02:26:20 Asia/Shanghai**派发，独立session、父进程脱离SSH。
  派发前重新确认指定两卡完全空闲；采用原e20 checkpoint，再次核对独立SHA256。
  进程启动存活检查通过；首个完整e21（200步、984验证、保存）验收尚待后续补充，未把“存活”当成首轮完成。
- 私有路径/PID/GPU UUID仅在server manifest/launcher_state及忽略的本地记录，不提交仓库。

## 第一批机制诊断

1. CPU冻结whole/proxy：实际hard-Gumbel选择、端点代理身份重合、sample/proxy项的径向/角向梯度和使用熵；原mining与仅mining半径控制，loss保持原whole。
2. 单独诊断GPU：固定global64关系和输入，按local32分别forward，分解CE/intra/HIER sample的whole偏导和共享参数梯度，观察热点/普通dresser与其它类；恢复BN/RNG，副本一步控制另作标记。
3. 在相同输入重复Gumbel随机基线之上，检查增强、BN模式/组别引起的邻居与祖先变化，并增加独立形态关系检查。

每项新目录记录commit/config/seed、输入checkpoint/cache SHA256、开始/结束与设备；不提交向量、权重、全日志或私有连接。新发现将写入后续诊断条目，并更新[04](04_NEXT_EXPERIMENT_PLAN.md)再执行聚焦实验。用户已给予持续授权，不重复要求确认。
