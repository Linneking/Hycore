# S5-TAN-R恢复正则实验：配置与执行记录

用户于2026-11-03明确批准“把正则项加回来再挂上一组”。本次仅新增S5-TAN-R；结构与S5-TAN-R0完全相同，恢复U256处分叉的原Rcontr/Rhier，系数均.01。

理由：TAN-R0已恢复原漏检部件并在单seed Class超过DGCNN，需要在该成功主支上独立检验双曲正则是否仍有增益。旧球坐标头的R开/关结果不足以回答。

设置：fresh200、seed4780、原初始参数、自然shuffle32、N2048/k40、CE(5q)/5、RSGD .1→.001余弦、momentum.9、WD1e-4、无总梯度裁剪；same trainval14007/test2874与原选模协议。正则分支为Conv9/BN9/Leaky后U256→whole/特征近邻part分别均值→EXP；part200–600，gamma1000/Npart、delta4、同批异类负例。每批编码一次，BN一次。

版本化v11复用v10模型/监控/指标及原loss核；不改旧v10或旧结果。准备完成CPU合同后以同提交隔离部署，在完全空闲GPU直接训练，无GPU短测。当前此初始记录尚未启动；实际启动及首轮验收将追加。

判断终点：best-Instance同轮、固定e200、末20轮Class与Instance，motorbike31/33/34及table/lamp和其余类代价。梯度非零或loss下降不作为正则有益的证据；结束后在对话说明理由、控制、结果、结论及下一步。层次对照与现有成绩见[comparison.md](comparison.md)。

本机4项CPU合同和独立静态审查通过。完整合成2轮×2步核验BN4、50部件统计、保存恢复及R活跃路由；没有CUDA初始化或test访问。服务器同提交复核与正式启动记录待追加。
