"""Render the reviewed, proposed architecture. Does not import training code."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})
COLORS = {
    "keep": ("#e4f3e8", "#438657"),
    "add": ("#e3effc", "#3975af"),
    "change": ("#fff0dc", "#bf7c24"),
    "io": ("#f0f2f5", "#89929e"),
}
LEFT = (.035, .445)
RIGHT = (.565, .975)


def setup(title, subtitle, height):
    fig = plt.figure(figsize=(16, height), facecolor="white")
    ax = fig.add_axes([.025, .065, .95, .79])
    ax.set(xlim=(0, 1.02), ylim=(-.015, 1.02))
    ax.axis("off")
    fig.suptitle(title, fontsize=23, y=.985, fontweight="bold", color="#172a3a")
    fig.text(.5, .949, subtitle, ha="center", fontsize=12.5, color="#465566")
    fig.text(.25, .895, "官方 HyCoRe · DGCNN 分类", ha="center", fontsize=17, fontweight="bold")
    fig.text(.75, .895, "拟改 HyCoRe-seg v2 · 原生适配", ha="center", fontsize=17, fontweight="bold")
    legend = [Patch(facecolor=COLORS[k][0], edgecolor=COLORS[k][1], label=label)
              for k, label in [("keep", "保留官方组件"), ("add", "新增分割接口"),
                               ("change", "调整接口 / 语义"), ("io", "输入 / 输出")]]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .012),
               ncol=4, frameon=False, fontsize=12)
    return fig, ax


def box(ax, span, y, text, kind="keep", height=.066, fontsize=12.2):
    x0, x1 = span
    face, edge = COLORS[kind]
    patch = FancyBboxPatch((x0, y-height/2), x1-x0, height,
                          boxstyle="round,pad=0.006,rounding_size=0.009",
                          facecolor=face, edgecolor=edge, linewidth=1.45, zorder=3)
    ax.add_patch(patch)
    ax.text((x0+x1)/2, y, text, ha="center", va="center", fontsize=fontsize,
            linespacing=1.35, color="#182b3a", zorder=4)
    return ((x0+x1)/2, y, height)


def arrow(ax, upper, lower):
    x, y, h = upper
    xx, yy, hh = lower
    ax.add_patch(FancyArrowPatch((x, y-h/2-.006), (xx, yy+hh/2+.006),
                                arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.35, color="#647080", zorder=2))


def chain(ax, span, specs):
    nodes = [box(ax, span, y, text, kind, h) for y, text, kind, h in specs]
    for first, second in zip(nodes, nodes[1:]):
        arrow(ax, first, second)
    return nodes


def save(fig, stem):
    fig.savefig(OUT / (stem+".png"), dpi=180, facecolor="white")
    svg = OUT / (stem+".svg")
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",
                   encoding="utf-8")
    plt.close(fig)


def model():
    fig, ax = setup("模型结构对照：保留原分类组件，新增逐点分割接口",
                    "右侧为待审设计，尚未实现；绿色模块保留原实现与完整参数化，物体分类头改为逐点共享头。", 16)
    common = [
        (.94, "输入：XYZ 点云  [B, N, 3]", "io", .052),
        (.85, "四层 EdgeConv  conv1…4\n逐点局部特征：64 / 64 / 128 / 256", "keep", .066),
        (.76, "局部特征拼接 512 → conv5 → 1024\n保留 BN 与 LeakyReLU", "keep", .066),
        (.67, "全局 max pooling + avg pooling\n拼接得到每物体 2048 维", "keep", .066),
        (.58, "linear1：2048 → 512\nBN6 + LeakyReLU", "keep", .066),
    ]
    chain(ax, LEFT, common + [
        (.40, "expmap0：512 维欧氏向量 → 双曲球", "keep", .055),
        (.31, "原 MobiusLayer · emb：512 → 256\n包含 _weight / _bias / transp0", "keep", .066),
        (.22, "原 MobiusLayer · mlr：256 → 40", "keep", .055),
        (.13, "物体类别分数  [B, 40]\n同时返回物体嵌入 mu [B, 256]", "io", .066),
        (.04, "物体分类 cal_loss\n原入口实际默认 label smoothing = .2", "io", .066),
    ])
    chain(ax, RIGHT, common + [
        (.485, "新增逐点融合桥：1040 → 512\n局部512 + 广播全局512 + 类别one-hot16\n得到欧氏逐点 H [B, N, 512]", "add", .079),
        (.40, "复用 expmap0：每点 512 维 → 双曲球", "keep", .055),
        (.31, "复用原 emb：512 → 256，逐点共享\n完整保留 _weight / _bias / transp0", "keep", .066),
        (.22, "复用原 mlr：256 → 50\n调整：物体类别40 → 部件标签50", "change", .066),
        (.13, "逐点部件分数  [B, N, 50]\nH 另供论文正则分支使用", "change", .066),
        (.04, "复用 cal_loss，监督改为逐点标签\n显式 smoothing=False（分割适配选择）", "change", .066),
    ])
    ax.text(.24, .49, "保留每物体分类路径", ha="center", va="center", fontsize=11, color="#697482")
    # Expose local features without passing through the global pooling bottleneck.
    ax.plot([RIGHT[0], .505, .505, RIGHT[0]], [.85, .85, .485, .485],
            color=COLORS["add"][1], linestyle="--", linewidth=1.4, zorder=1)
    ax.annotate("", xy=(RIGHT[0], .485), xytext=(.525, .485),
                arrowprops={"arrowstyle": "-|>", "color": COLORS["add"][1], "lw": 1.4}, zorder=2)
    ax.text(.492, .676, "局部多尺度特征 512", rotation=90, va="center", ha="center",
            fontsize=10.5, color=COLORS["add"][1])
    ax.text(.77, .985, "另输入已知物体类别 one-hot16 → 融合桥", ha="center", fontsize=10.5,
            color=COLORS["add"][1])
    save(fig, "model_comparison")


def regularizer():
    fig, ax = setup("训练关系分支对照：复用几何与损失核，按论文改变任务语义",
                    "分类路径以 post-emb mu 建关系；分割路径以同一次编码的欧氏逐点 H 建关系。", 13)
    ys = [.93, .79, .65, .51, .37, .23, .09]
    left = [
        ("输入 XYZ\n分类训练脚本先构造 whole 视图", "io"),
        ("get_children_np：XYZ 近邻裁剪\n得到 part；原函数会写入输入切片", "io"),
        ("whole 与 part 分别前向\n同一编码器运行两次", "io"),
        ("取 emb 后的物体 / part 嵌入\nmu_whole, mu_part：256 维", "keep"),
        ("负样本 = 翻转 batch\n未保证异类 / 非自身", "io"),
        ("hype_triplet_losses\n复用基础：dist、dist0、hinge、mean", "keep"),
        ("分类默认方向：whole 更深\nCE + .01 对比项 + .01 层次项", "io"),
    ]
    right = [
        ("整物体只编码一次\nH = E(X)：[B, N, 512]，位于 exp 之前", "change"),
        ("H 特征空间内选 part\n随机中心近邻；N′ = 200…600", "add"),
        ("欧氏平均：whole = mean(所有 H)\npart = mean(选中 H)", "add"),
        ("复用原 expmap0 / dist / dist0\nwhole 与 part 均为 512 维球嵌入", "keep"),
        ("显式负例索引：异物体且异类别\n无合法负例只跳过对应对比项", "change"),
        ("复用损失核并传入合法负例\nopposite_hier=True：要求 part 更深", "change"),
        ("L = CE + .01 Rcontr + .01 Rhier\n层次间隔 1000/N′；对比间隔 4", "change"),
    ]
    chain(ax, LEFT, [(y, text, kind, .088) for y, (text, kind) in zip(ys, left)])
    chain(ax, RIGHT, [(y, text, kind, .088) for y, (text, kind) in zip(ys, right)])
    save(fig, "regularizer_comparison")


if __name__ == "__main__":
    model()
    regularizer()
    print("Rendered two diagram pairs (PNG and SVG).")
