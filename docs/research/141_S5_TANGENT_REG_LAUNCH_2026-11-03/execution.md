# S5-TAN-R恢复正则实验：配置与执行记录

用户于2026-11-03明确批准“把正则项加回来再挂上一组”。本次仅新增S5-TAN-R；结构与S5-TAN-R0完全相同，恢复U256处分叉的原Rcontr/Rhier，系数均.01。

理由：TAN-R0已恢复原漏检部件并在单seed Class超过DGCNN，需要在该成功主支上独立检验双曲正则是否仍有增益。旧球坐标头的R开/关结果不足以回答。

设置：fresh200、seed4780、原初始参数、自然shuffle32、N2048/k40、CE(5q)/5、RSGD .1→.001余弦、momentum.9、WD1e-4、无总梯度裁剪；same trainval14007/test2874与原选模协议。正则分支为Conv9/BN9/Leaky后U256→whole/特征近邻part分别均值→EXP；part200–600，gamma1000/Npart、delta4、同批异类负例。每批编码一次，BN一次。

版本化v11复用v10模型/监控/指标及原loss核；不改旧v10或旧结果。本机及服务器CPU合同通过后，同提交隔离部署；正式运行情况见下文。

判断终点：best-Instance同轮、固定e200、末20轮Class与Instance，motorbike31/33/34及table/lamp和其余类代价。梯度非零或loss下降不作为正则有益的证据；结束后在对话说明理由、控制、结果、结论及下一步。层次对照与现有成绩见[comparison.md](comparison.md)。

本机4项CPU合同和独立静态审查通过。完整合成2轮×2步核验BN4、50部件统计、保存恢复及R活跃路由；没有CUDA初始化或test访问。服务器同提交复核及正式首轮验收均已通过。

## 正式启动及验收

源7e0cb8d已推送，同提交增量Git包fetch/switch/pull --ff-only部署到新隔离目录，旧v10源与结果未动。服务器4项CPU合同通过，原服务器初始化hash17c51802…匹配。

北京时间2026-11-03 10:42:31在GPU2直接fresh200。连续三次确认GPU2无计算进程、0%利用率，未干预其他用户任务。断开重连确认进程父进程1、实际GPU训练进行；没有GPU短测。

正式首轮26项验收通过，见[launch_acceptance.json](launch_acceptance.json)：

- 与TAN-R0的e0所有模型张量/CPU RNG、数据manifest、首输入45b5910e…和首次part/negative配对精确一致。
- 配置仅允许路径/GPU与R开关、alpha/beta三项差异；均为s5、.01/.01正则。
- 完整437更新、BN437、test2874及50部件，实际模型和优化器有限，无failure。
- CE/contr/hier梯度分项齐全，两R活跃并连至U；两头部映射不被R直接经过。q有正确类概率记录、无球边界字段。

10:44快照e2/step75/update512，首轮110.33秒；预计北京时间今天16:50左右完成，以实际负载为准。首轮Instance74.6846%、Class61.4995%仅作验收，不用于判断最终正则收益。

生产工作区冻结于7e0cb8d，后续文档不拉入其中；勿重复启动或改已有运行。私有PID/完整路径在忽略的141运行交接中。
