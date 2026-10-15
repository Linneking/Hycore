# V8 两组对照：账户、环境与后台启动

日期：2026-10-15。状态：用户确认单卡也采用V8的cap/hook及margin；两组GPU短验收通过，11:41开始300轮训练。11:44核查均已完成第1轮、保存完整checkpoint并进入第2轮，标量有限，未读取正式test。旧服务器V8继续运行，11:45核查已完成26轮、进入27轮。

## 实际安排

| 实验 | 资源 | 每轮训练 | 目标 |
|---|---|---|---|
| V8-HIER64-K20-W0 | 原服务器两卡，保持运行 | 200步；global64，单卡16类×2例 | CE + .01 Rcontr + .01 Rhier + .5 HIER |
| V8-B64 | 新服务器GPU0、1 | 200步；同主训练采样 | CE + .01 Rcontr + .01 Rhier |
| V8-B0-32-C01 | 新服务器GPU2 | 276步；batch32 shuffle/drop_last | CE + .01 Rcontr + .01 Rhier |

三者均c=.1、最终共享whole/part切空间cap2.3、一次逆度量梯度hook、gamma500/delta2；seed22，300轮cosine。两条新对照没有代理参数和代理优化器，完全不调用HIER。单卡保留原HyCoRe的flip负例语义，允许同类并记录碰撞率。

三组初始化模型、8856/984划分一致；验证选模，结束后仅一次官方test。B64提供HIER增量对照；B0-32还同时改变batch、采样和每轮步数，不能拆分为单一因素。单卡属于V8几何下的HyCoRe shuffle参照，不是完全原版复刻。

## 源码与环境

生产代码`85fc812d06fd355e1153b8d15bad7b4083ce82bd`，分支`codex/v8-controls-c01-20261015`；已推送并经Git fetch/pull --ff-only部署。入口为`inter_hierarchy_MN40/hier_proxy_scratch_v8/controls.py`，安全后台队列为`control_queue.py`，新合同为`tests/test_controls.py`。原主V8入口不变。

新普通账户jhr按现有用户的存储目录规则创建，主目录700、授权公钥文件600，通过用户本机原公钥登录。未修改驱动、其他用户或原环境。Conda安装包核对官方SHA256；临时内网迁移授权已撤销、临时密钥已清理。私有服务器入口、目录、PID、GPU UUID与状态命令只保存在忽略Git的工程交接中。

Conda环境`hycore_v8`：Python3.9.23、Torch2.8.0+cu128、Geoopt0.5.1、NumPy1.26.4、SciPy1.13.1、h5py3.14.0。核心数值库沿用已验证缓存；旧Python3.9.25与新3.9.23有补丁版本差异，不能称环境逐包完全一致。新旧冻结快照的V8 geometry.py、base_protocol.py和PointNet编译扩展SHA256一致，初始模型也一致。

## 验收

- 本地源码语法及本次提交差异检查通过。新Linux账户完成9项原V8合同、双进程global64梯度合同，以及3项新合同：几何配置一致性、shuffle整轮无重复和回放、独立HyCoRe公式的前向/梯度对照。无skip。
- 两个GPU smoke各2轮×2更新、各验证一批，未读取test，均保存checkpoint并标记completed。生产从头初始化，不继承smoke。
- 三组初始模型SHA256：`f94e52202df5aaae21f21b8b9c6beaa3925424c4e04a6dcc7d2b84b89af04b73`；划分SHA256：`3bbf0c79b3cb9581caeafe3f7f12caa6861f236f0159512a5998d5e9bcf25105`。
- 两组正式首轮均验证完整984对象，保存last、best和e1归档；每卡每步BN更新2次。lambda_H=0、HIER调用0、代理参数0，loss等于base，已进入e2。当前只确认启动与初期数值稳定，没有效果收益结论。

运行ID：`v8_controls_20261015_113942`。队列已脱离SSH；两组各有manifest、heartbeat、steps/metrics、best/last与归档。源码独立冻结，后续文档提交不改变训练代码。

私有交接：`.codex-local/handoff/51_V8_NEW_SERVER_CONTROLS_2026-10-15.md`；[具体计划](51_V8_CONTROLS_PLAN_2026-10-15.md)；[前序主V8](50_V8_IMPLEMENTATION_AND_LAUNCH_2026-10-15.md)。
