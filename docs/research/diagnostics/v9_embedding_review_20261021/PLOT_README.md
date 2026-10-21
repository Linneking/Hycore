# V9嵌入审查绘图器

`plot.py`仅读取已经导出的数组，不加载网络、修改BN或运行训练。

```powershell
python plot.py --export <private_export_directory> --out <public_figure_directory>
```

输入`export_manifest.json`可以是观察列表，也可以是含`observations`列表的对象。每条观察包含`key`、`label`、`stage`、`epoch`、`c_native`、`c_hier`、`kind`、`cache`。`stage`用`best`或`e300`；`last`也映射为`e300`，因此只有确实完成300轮的终点才能标为last。历史无终点时不要补造该行。

`plot_label`为可选的简短英文展示名，优先于`label`；中文label回退到key以避免字体依赖。`cache`支持相对导出目录或绝对路径。

NPZ必需`native_mu`；应保存`labels`和`canonical_ids`供上游核验同一对象。HIER缓存另外保存`hier_mu`、`proxy_ball`、`proxy_tangent`。`sample_usage`和`proxy_usage`各为与代理数相同的一维硬祖先计数，缺失时明确显示不可用，不替换成零。可选`nn4`、`nn4_equal`、`nn4_angle`应为512×4的完整池邻居索引或规范ID。

可选`longitudinal.json`是列表，或含`observations`/`snapshots`列表的对象；每项含`key`、`epoch`，以及数组`native_whole_depth`、`hier_whole_depth`、`proxy_depth`、`proxy_tangent_norm`、`sample_usage`、`proxy_usage`。这些数组应来自保存的真实训练前向，不能混入clean/eval导出；曲线和热图会明确标为当前批BN训练快照。

输出01–06为clean/eval统计，07–08为可选训练快照。每图输出PNG/SVG，并写`figure_manifest.json`。所有图形均为聚合统计，不输出点云、样本ID或原始向量。

解释边界：跨曲率的native深度分面；归一化坐标半径q可同图展示，但不表示相同双曲距离。HIER对照只接受c0.1。V9固定深度4.6是人为视图，不是原生whole学出球壳的证据。硬选择次数与full-ST/查询端点梯度分别解释，反向NN4覆盖不称为训练激活。

功能验证：使用独立临时目录中的6份合成缓存及两组4时刻训练数组，完整生成8PNG/8SVG，04和08已视觉检查；合成结果不进入科学报告。
