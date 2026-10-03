"""Read-only V5 epoch200 proxy neighbourhoods: clean features, PNG and offline HTML.

This entry point uses the full checkpoint's contemporaneous net/proxy states,
never its validation-selected best_net. The displayed nearest four samples
are a visualisation choice; proxy anchor eligibility uses the training K20
reciprocal rule over all 512 proxies. Eligibility does not establish that a
proxy was actually selected as a pair/triple ancestor during training.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CLASS_NAMES = (
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car",
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot",
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", "night_stand",
    "person", "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs",
    "stool", "table", "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox",
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A NEW directory; existing results are never overwritten")
    parser.add_argument("--device", default="cuda:0", help="CUDA device for PointMLP inference")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=4,
                        help="Displayed nearest distinct samples, not training K20")
    parser.add_argument("--num-proxies", type=int, default=10,
                        help="Number sampled uniformly from all eligible proxies")
    parser.add_argument("--split", choices=("train_ids", "validation_ids"), default="train_ids",
                        help="Checkpoint-defined IDs in the official training shards; no test reading")
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.topk, args.num_proxies) < 1:
        parser.error("batch-size, topk and num-proxies must be positive")
    if args.num_proxies > 512:
        parser.error("num-proxies must be at most 512")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")


def poincare_distances(x, y):
    """FP64 raw high-dimensional d_1; no labels, boosts or 2-D projection.

    d_1(x,y) = 2 asinh(||x-y|| / sqrt((1-||x||²)(1-||y||²))).
    Dot-product evaluation avoids an [N,M,D] temporary. Clip only squared
    Euclidean roundoff below zero, without projecting or altering features.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x/y must be [N,D]/[M,D]")
    xn, yn = np.einsum("ij,ij->i", x, x), np.einsum("ij,ij->i", y, y)
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or (xn >= 1).any() or (yn >= 1).any():
        raise ValueError("Features must be finite and strictly inside the c1 ball")
    squared = np.maximum(xn[:, None] + yn[None, :] - 2 * x @ y.T, 0)
    # Make exactly identical rows have exactly zero distance, despite dot
    # accumulation order, when this function computes a self-distance matrix.
    if x.shape == y.shape and np.array_equal(x, y):
        np.fill_diagonal(squared, 0)
    denominator = np.sqrt((1 - xn)[:, None] * (1 - yn)[None, :])
    return 2 * np.arcsinh(np.sqrt(squared) / denominator)


def nearest_distinct_ids(distances, sample_ids, topk):
    """Return position indices, sorted by distance then ID then position.

    Repeated dataset IDs can never fill multiple slots in the same row.
    The production split itself contains unique IDs, but this explicit
    invariant also protects future callers that supply repeated positions.
    """
    distances = np.asarray(distances)
    sample_ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
    if distances.ndim != 2 or distances.shape[1] != len(sample_ids):
        raise ValueError("Distance columns must align with sample IDs")
    if topk < 1 or len(np.unique(sample_ids)) < topk or not np.isfinite(distances).all():
        raise ValueError("Need finite distances and at least topk distinct IDs")
    positions = np.arange(len(sample_ids))
    result = []
    for row in distances:
        order = np.lexsort((positions, sample_ids, row))
        picked, seen = [], set()
        for position in order:
            sample_id = int(sample_ids[position])
            if sample_id in seen:
                continue
            seen.add(sample_id)
            picked.append(int(position))
            if len(picked) == topk:
                break
        result.append(picked)
    return np.asarray(result, dtype=np.int64)


def choose_proxies(eligible, count, seed):
    eligible_ids = np.flatnonzero(np.asarray(eligible, dtype=bool))
    if len(eligible_ids) < count:
        raise ValueError(f"Only {len(eligible_ids)} eligible proxies; requested {count}")
    # Selection uses only eligibility and the fixed seed, never sample labels,
    # nearest-neighbour purity, radius, degree magnitude or visual appearance.
    return np.random.default_rng(seed).choice(eligible_ids, size=count, replace=False)


def proxy_graph(proxies, topk=20, seed=22):
    """Use V5's actual FP32 torch topk/mutual rule, including its tie behavior."""
    from hier_proxy_scratch_v5.relations import mine_reciprocal_triplets, poincare_distance
    import torch

    with torch.no_grad():
        distances = poincare_distance(proxies)
        mined = mine_reciprocal_triplets(torch.exp(-distances), topk=topk,
                                         t_per_anchor=1, seed=seed,
                                         exclude_self_negative=False)
    return {"eligible": mined["eligible"].cpu().numpy(),
            "degree": mined["mutual"].sum(-1).cpu().numpy(),
            "mutual": mined["mutual"].cpu().numpy(),
            "negative": mined["negative"].cpu().numpy()}


def validate_checkpoint(saved):
    if saved.get("format") != "hycore-hier-v5-h20-1" or saved.get("epoch") != 200:
        raise ValueError("Require the complete V5-H20 epoch200 checkpoint")
    if saved.get("model_selection_only") or not all(key in saved for key in
            ("net", "proxy", "training_config", "train_ids", "validation_ids")):
        raise ValueError("Require complete contemporaneous net/proxy states and split IDs")
    config = saved["training_config"]
    for key, expected in (("c", 1), ("D", 256), ("P", 512), ("proxy_K", 20)):
        if config.get(key) != expected:
            raise ValueError(f"Checkpoint {key} differs from V5 setting {expected}")
    train = np.asarray(saved["train_ids"], dtype=np.int64).reshape(-1)
    val = np.asarray(saved["validation_ids"], dtype=np.int64).reshape(-1)
    if len(train) != 8856 or len(val) != 984 or len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val):
        raise ValueError("Require unique V5 split IDs: 8856 training and 984 validation")
    if np.intersect1d(train, val).size:
        raise ValueError("Training/validation split overlaps")
    digest = hashlib.sha256(train.tobytes() + val.tobytes()).hexdigest()
    if saved.get("split_sha256") != digest:
        raise ValueError("Checkpoint split hash does not match its saved IDs")
    return train, val


def load_training_shards(data_dir):
    """Same sorted shard order as V5; never downloads or opens test shards."""
    import h5py

    shards = sorted(Path(data_dir).resolve(strict=True).glob("ply_data_train*.h5"))
    if not shards:
        raise FileNotFoundError("No ModelNet40 training shards found")
    points, labels = [], []
    for path in shards:
        with h5py.File(path, "r") as stream:
            points.append(stream["data"][:].astype(np.float32))
            labels.append(stream["label"][:].astype(np.int64).reshape(-1))
    points, labels = np.concatenate(points), np.concatenate(labels)
    if points.ndim != 3 or points.shape[1] < 1024 or points.shape[2] != 3 or len(points) != len(labels):
        raise ValueError("Training shards must provide aligned [N,>=1024,3] points and labels")
    if not np.isfinite(points).all() or (labels < 0).any() or (labels >= 40).any():
        raise ValueError("Nonfinite points or invalid ModelNet40 labels")
    return points, labels, [path.name for path in shards]


def state_without_ddp_prefix(state):
    return {name.removeprefix("module."): value for name, value in state.items()}


def extract_features(saved, points, ids, batch_size, device_name, seed):
    import torch
    from models.pointmlp import Hype_pointMLP
    from hier_proxy_scratch_v5.hier_loss import HIERLoss

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("PointMLP's native FPS requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = Hype_pointMLP().to(device)
    model.load_state_dict(state_without_ddp_prefix(saved["net"]), strict=True)
    for name in ("manifold", "manifold2"):
        manifold = getattr(model, name)
        manifold.requires_grad_(False)
        if not math.isclose(float(manifold.c.detach()), 1., abs_tol=1e-5):
            raise ValueError(f"Loaded {name} curvature is not c1")
    model.eval()
    proxy = HIERLoss(num_proxies=512, dim=256, c=1, margin=.1, tau=.1, seed=seed).to(device)
    proxy.load_state_dict(state_without_ddp_prefix(saved["proxy"]), strict=True)
    proxy.eval()
    features = np.empty((len(ids), 256), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            # Exactly the clean V5 evaluation input: raw first1024 points,
            # no part crop, translation, jitter, point shuffle or augmentation.
            cloud = torch.from_numpy(np.ascontiguousarray(points[ids[start:stop], :1024])).to(device)
            mu, _logits = model(cloud.transpose(1, 2).contiguous())
            if mu.shape != (stop - start, 256) or not torch.isfinite(mu).all():
                raise RuntimeError("Invalid clean whole embedding")
            features[start:stop] = mu.cpu().numpy()
            if start == 0 or stop == len(ids) or (start // batch_size + 1) % 25 == 0:
                print(f"Clean features {stop}/{len(ids)}", flush=True)
        ball_proxies = proxy.proxies().detach()
        graph = proxy_graph(ball_proxies, topk=20, seed=seed)
        proxies = ball_proxies.cpu().numpy()
    runtime = {"torch": torch.__version__, "numpy": np.__version__,
               "cuda": torch.version.cuda, "device": str(device),
               "gpu_name": torch.cuda.get_device_name(device),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    return features, proxies, graph, runtime


def ancestor_usage_report(saved):
    """Preserve observed aggregate support without inventing per-proxy counts.

    V5 training forward requests return_mining, not return_details, and its
    telemetry stores aggregate collision/activity counts. Pair/triple chosen
    proxy IDs are therefore absent from the saved training metrics. Rerunning
    Gumbel choices on clean features would not recover historical usage.
    """
    hierarchy = saved.get("metrics", {}).get("telemetry", {}).get("hierarchy") or {}
    aggregates = {}
    for name in ("sample_graph", "proxy_graph"):
        graph = hierarchy.get(name) or {}
        counts = graph.get("counts", {})
        aggregates[name] = {key: counts.get(key) for key in
                            ("eligible_anchors", "batch_size", "triplets", "collisions",
                             "noncollision_triplets", "active_triplets")}
    return {
        "status": "unavailable",
        "per_proxy_pair_selected_counts": None,
        "per_proxy_triple_selected_counts": None,
        "per_proxy_active_selected_counts": None,
        "reason": "V5 epoch200 checkpoint/telemetry stores aggregate activity, not selected pair/triple proxy IDs. Eligibility and nearest samples are not historical ancestor-usage evidence.",
        "source_epoch": saved.get("epoch"),
        "epoch_aggregate_activity_only": aggregates,
        "source_code": ["hier_proxy_scratch_v5/hier_loss.py:174-175",
                        "hier_proxy_scratch_v5/telemetry.py:_GraphAggregate"],
    }


def cloud_projection(points, yaw=-.6, pitch=.3):
    """Display-only box centering and uniform scale; preserve aspect ratio."""
    points = np.asarray(points, dtype=np.float64)
    center = (points.min(0) + points.max(0)) / 2
    scale = max(float(np.max(points.max(0) - points.min(0))), 1e-12)
    xyz = (points - center) / scale * 1.65
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    x = cy * xyz[:, 0] + sy * xyz[:, 2]
    z = -sy * xyz[:, 0] + cy * xyz[:, 2]
    y = cp * xyz[:, 1] - sp * z
    depth = sp * xyz[:, 1] + cp * z
    return np.column_stack((x, y, depth))


def render_png(path, rows, clouds, epoch=200):
    """Pillow orthographic point rendering, with one fixed view for all cells."""
    from PIL import Image, ImageDraw, ImageFont

    def font_at(size):
        for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Older Pillow on some server environments.
            return ImageFont.load_default()
    font, small, title_font = font_at(18), font_at(15), font_at(24)
    cols = len(rows[0]["neighbors"])
    width, cell_h, gutter, left, header = cols * 360 + 100, 305, 14, 80, 135
    height = header + len(rows) * (cell_h + 45 + gutter) + 55
    picture = Image.new("RGB", (width, height), "#edf1f6")
    draw = ImageDraw.Draw(picture)
    draw.text((left, 22), f"V5-H20 epoch {epoch} | {len(rows)} eligible proxies x {cols} nearest samples",
              fill="#13243a", font=title_font)
    draw.text((left, 62), "Clean checkpoint training split | raw high-dimensional hyperbolic distance | distinct IDs",
              fill="#43536a", font=small)
    draw.text((left, 90), "Random eligible proxies (seed fixed); graph eligibility is not recorded ancestor usage.",
              fill="#43536a", font=small)
    for row_index, row in enumerate(rows):
        y0 = header + row_index * (cell_h + 45 + gutter)
        heading = f"Proxy {row['proxy_id']:03d}  |  radius {row['proxy_radius']:.5f}  |  reciprocal degree {row['reciprocal_degree']}"
        draw.text((left, y0), heading, font=font, fill="#13243a")
        for col, neighbor in enumerate(row["neighbors"]):
            x0, y1 = left + col * 360, y0 + 35
            draw.rounded_rectangle((x0, y1, x0 + 345, y1 + cell_h), radius=12,
                                   fill="white", outline="#d2dce8", width=1)
            draw.text((x0 + 12, y1 + 12), f"#{col + 1}  {neighbor['class_name']}  |  ID {neighbor['sample_id']}",
                      fill="#13243a", font=font)
            draw.text((x0 + 12, y1 + 40), f"d = {neighbor['distance']:.5f}", fill="#43536a", font=small)
            projected = cloud_projection(clouds[row_index, col])
            for index in np.argsort(projected[:, 2], kind="stable"):
                px, py, depth = projected[index]
                sx, sy = x0 + 172 + px * 128, y1 + 183 - py * 128
                tone = int(np.clip(100 + depth * 45, 55, 160))
                draw.ellipse((sx - 1.5, sy - 1.5, sx + 1.5, sy + 1.5),
                             fill=(30, tone, 200))
    draw.text((left, height - 31), "Same display camera in every cell; open proxy_neighbors.html to rotate individual point clouds.",
              fill="#43536a", font=small)
    picture.save(path)


def render_html(path, rows, clouds, metadata):
    """A single offline file. Each canvas supports its own rotation and zoom."""
    payload = {"rows": rows, "clouds": np.round(clouds, 6).tolist(), "metadata": metadata}
    packed = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    title = f"V5-H20 第200轮 · {len(rows)}个代理的最近{len(rows[0]['neighbors'])}例"
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;font:15px system-ui,-apple-system,"Segoe UI",sans-serif;color:#14243b;background:#edf1f6}
*{box-sizing:border-box}body{margin:0}main{max-width:1580px;padding:28px;margin:auto}h1{font-size:28px;margin:0 0 12px}
.intro{line-height:1.7;color:#43536a;max-width:1100px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:18px 0}
button{font:inherit;border:1px solid #c7d3e3;border-radius:7px;padding:7px 15px;background:white;color:#14243b;cursor:pointer}
.proxy{margin-top:28px}.rowhead{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;margin-bottom:12px}.rowhead h2{font-size:20px;margin:0}
.muted{color:#52637b;font-size:14px}.grid{display:grid;grid-template-columns:repeat(__COLS__,minmax(0,1fr));gap:14px}
.card{background:white;border:1px solid #d2dce8;border-radius:12px;overflow:hidden;min-width:0}.caption{padding:12px 14px 3px;line-height:1.55}
.caption strong{display:block;font-size:16px}canvas{display:block;width:100%;height:285px;touch-action:none;cursor:grab}
canvas:active{cursor:grabbing}.footer{margin:28px 0;color:#52637b;font-size:13px}a{color:#2351a3}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:18px}}
@media(max-width:540px){.grid{grid-template-columns:1fr}h1{font-size:23px}}
</style></head><body><main><h1>__TITLE__</h1>
<div class="intro">从全部512个代理中，按训练K20互惠规则找到可作为anchor的代理，再用固定随机种子抽取展示对象。
每格为同一代理最近的不同训练实例，排序采用原始高维双曲距离，未按类别筛选。这里的最近4例是展示设置。
“有效”仅表示当前代理图具备关系抽取条件；训练时每个代理被选作共同祖先的次数未记录。</div>
<div class="controls"><button id="reset">重置视角</button><button id="sync">统一当前视角</button>
<span class="muted">拖动单个点云可旋转；滚轮可缩放；双击重置该格。</span></div><div id="gallery"></div>
<div class="footer">__SPLIT__ · __COUNT__例 · 无增强前1024点 · epoch200完整模型及代理。原始坐标见selected_clouds.npz，检索记录见neighbors.csv。</div>
</main><script id="data" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), viewers=[];
const initial={yaw:-.6,pitch:.3,zoom:1};let lastView=null;
function view(canvas,raw){
 const minima=[0,1,2].map(a=>Math.min(...raw.map(p=>p[a]))),maxima=[0,1,2].map(a=>Math.max(...raw.map(p=>p[a])));
 const center=minima.map((v,a)=>(v+maxima[a])/2),scale=Math.max(...maxima.map((v,a)=>v-minima[a]),1e-12);
 const points=raw.map(p=>p.map((v,a)=>(v-center[a])/scale*1.65));
 const state={...initial},ctx=canvas.getContext('2d');let drag=null;
 function draw(){
  const rect=canvas.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio||1,2),w=rect.width,h=rect.height;
  canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.fillStyle='#ffffff';ctx.fillRect(0,0,w,h);
  const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);
  const transformed=points.map(p=>{const x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2];return[x,cp*p[1]-sp*z,sp*p[1]+cp*z]}).sort((a,b)=>a[2]-b[2]);
  const unit=Math.min(w,h)*.40*state.zoom;
  for(const p of transformed){const tone=Math.max(55,Math.min(160,100+p[2]*45));ctx.fillStyle=`rgb(30,${tone},200)`;
   ctx.beginPath();ctx.arc(w/2+p[0]*unit,h/2-p[1]*unit,1.6,0,Math.PI*2);ctx.fill()}
  ctx.font='11px system-ui';ctx.fillStyle='#7b8ba0';ctx.fillText('X / Y / Z 原始坐标 · 拖动旋转',12,h-12);
 }
 canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);lastView={state,draw}});
 canvas.addEventListener('pointermove',e=>{if(!drag)return;state.yaw+=(e.clientX-drag[0])*.009;state.pitch+=(e.clientY-drag[1])*.009;drag=[e.clientX,e.clientY];draw()});
 canvas.addEventListener('pointerup',()=>{drag=null});canvas.addEventListener('pointercancel',()=>{drag=null});
 canvas.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.max(.35,Math.min(3,state.zoom*Math.exp(-e.deltaY*.001)));lastView={state,draw};draw()},{passive:false});
 canvas.addEventListener('dblclick',()=>{Object.assign(state,initial);draw()});
 const viewer={state,draw};viewers.push(viewer);new ResizeObserver(draw).observe(canvas);return viewer;
}
data.rows.forEach((row,r)=>{
 const section=document.createElement('section');section.className='proxy';
 const head=document.createElement('div');head.className='rowhead';
 const name=document.createElement('h2');name.textContent=`代理 ${String(row.proxy_id).padStart(3,'0')}`;
 const detail=document.createElement('span');detail.className='muted';detail.textContent=`半径 ${row.proxy_radius.toFixed(5)} · K20非self互惠邻居 ${row.reciprocal_degree} · 当前图可用`;
 head.append(name,detail);section.append(head);const grid=document.createElement('div');grid.className='grid';section.append(grid);
 row.neighbors.forEach((n,c)=>{const card=document.createElement('article');card.className='card';const caption=document.createElement('div');caption.className='caption';
 const title=document.createElement('strong');title.textContent=`${c+1}. ${n.class_name} · ID ${n.sample_id}`;
 const detail=document.createElement('span');detail.className='muted';detail.textContent=`双曲距离 ${n.distance.toFixed(5)}`;
 const canvas=document.createElement('canvas');canvas.setAttribute('aria-label',`${n.class_name} ID ${n.sample_id} 可旋转点云`);
 caption.append(title,detail);card.append(caption,canvas);grid.append(card);});document.getElementById('gallery').append(section);
 grid.querySelectorAll('canvas').forEach((canvas,c)=>view(canvas,data.clouds[r][c]));
});
document.getElementById('reset').onclick=()=>viewers.forEach(v=>{Object.assign(v.state,initial);v.draw()});
document.getElementById('sync').onclick=()=>{const camera={...(lastView?lastView.state:viewers[0].state)};viewers.forEach(v=>{Object.assign(v.state,camera);v.draw()})};
</script></body></html>"""
    page = page.replace("__TITLE__", html.escape(title)).replace("__COLS__", str(len(rows[0]["neighbors"])))
    page = page.replace("__SPLIT__", html.escape(metadata["split"])).replace("__COUNT__", str(metadata["sample_count"]))
    Path(path).write_text(page.replace("__DATA__", packed), encoding="utf-8")


def main(argv=None):
    args = arguments(argv)
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    sys.path.insert(0, str(REPO / "pointnet2_ops_lib"))
    sys.path.insert(0, str(HERE))
    checkpoint_path = args.checkpoint.resolve(strict=True)
    before = checkpoint_path.stat()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "command_argv": [sys.executable, *sys.argv],
              "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)},
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}}
    write_json(output / "summary.json", report)
    try:
        import torch

        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_ids, validation_ids = validate_checkpoint(saved)
        ids = train_ids if args.split == "train_ids" else validation_ids
        points, labels, shards = load_training_shards(args.data_dir)
        all_ids = np.concatenate((train_ids, validation_ids))
        if len(points) != 9840 or all_ids.min() < 0 or all_ids.max() >= len(points):
            raise ValueError("Checkpoint IDs do not match the 9840-row official training shards")
        if args.topk > len(ids):
            raise ValueError("topk exceeds the split's distinct sample count")
        features, proxies, graph, runtime = extract_features(saved, points, ids, args.batch_size, args.device, args.seed)
        selected = choose_proxies(graph["eligible"], args.num_proxies, args.seed)
        # All512 are queried against all clean samples. Only the independently
        # sampled eligible rows are visualised. Retrieval has no class boost.
        distance_blocks = [poincare_distances(proxies, features[start:start + 512])
                           for start in range(0, len(features), 512)]
        distances = np.concatenate(distance_blocks, axis=1)
        nearest = nearest_distinct_ids(distances, ids, args.topk)
        radius = np.linalg.norm(proxies.astype(np.float64), axis=1)
        rows = []
        for proxy_id in selected:
            neighbors = []
            for position in nearest[proxy_id]:
                sample_id, label = int(ids[position]), int(labels[ids[position]])
                neighbors.append({"sample_id": sample_id, "label": label,
                                  "class_name": CLASS_NAMES[label],
                                  "distance": float(distances[proxy_id, position])})
            rows.append({"proxy_id": int(proxy_id), "proxy_radius": float(radius[proxy_id]),
                         "reciprocal_degree": int(graph["degree"][proxy_id]),
                         "eligible": True, "historical_ancestor_usage": None,
                         "neighbors": neighbors})
        selected_positions = nearest[selected]
        clouds = points[ids[selected_positions], :1024].copy()
        # Exact FP32 coordinates used by the clean model, before display-only
        # centering/scaling. JSON/HTML coordinates alone round to six decimals.
        np.savez_compressed(output / "selected_clouds.npz", points=clouds,
                            sample_ids=ids[selected_positions], labels=labels[ids[selected_positions]],
                            proxy_ids=selected, distances=distances[selected[:, None], selected_positions])
        np.savez_compressed(output / "feature_cache.npz", mu=features, proxy_ball=proxies,
                            sample_ids=ids, labels=labels[ids], proxy_eligible=graph["eligible"],
                            proxy_degree=graph["degree"], proxy_mutual=graph["mutual"],
                            all_proxy_nearest_ids=ids[nearest],
                            all_proxy_nearest_distances=distances[np.arange(512)[:, None], nearest])
        with (output / "neighbors.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["proxy_id", "proxy_radius", "reciprocal_degree", "neighbor_rank",
                      "sample_id", "label", "class_name", "distance"]
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for row in rows:
                for rank, neighbor in enumerate(row["neighbors"], 1):
                    writer.writerow({key: row[key] for key in fields[:3]} |
                                    {"neighbor_rank": rank} | neighbor)
        metadata = {"split": args.split, "sample_count": len(ids)}
        render_png(output / "proxy_neighbors.png", rows, clouds)
        render_html(output / "proxy_neighbors.html", rows, clouds, metadata)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        after = checkpoint_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Source checkpoint changed during visualisation")
        report.update(status="complete", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            wall_seconds=time.perf_counter() - started, code_commit=commit, runtime=runtime,
            checkpoint={**report["checkpoint"], "epoch": saved["epoch"],
                        "training_commit": saved.get("commit"), "state_used": "net + proxy; not best_net",
                        "size_bytes": before.st_size},
            dataset={"split": args.split, "sample_count": len(ids), "ids_sha256": ids_sha256(ids),
                     "checkpoint_split_sha256": saved["split_sha256"], "training_shards_sorted": shards,
                     "labels_sha256": ids_sha256(labels[ids]), "test_read": False,
                     "feature_input": "raw first1024 points; no augmentation; full eval mode; whole mu FP32"},
            geometry={"c": 1, "D": 256, "P": 512, "proxy_mapping": "V5 expmap0_c1 with native .999 numerical projection",
                      "retrieval": "FP64 raw high-dimensional Poincare distance; no label boost; ties by ID then position",
                      "display_normalization_only": "box center and one uniform scale; raw coordinates retained in npz",
                      "distance_formula": "2*asinh(norm(x-y)/sqrt((1-norm(x)^2)*(1-norm(y)^2)))"},
            eligibility={"training_K_including_self": 20, "graph_domain": "all512 checkpoint proxies",
                         "rule": "source FP32 reciprocal topK; diagonal removed; degree>=2 and negative pool nonempty; self-k allowed",
                         "eligible_count": int(graph["eligible"].sum()),
                         "eligible_proxy_ids": np.flatnonzero(graph["eligible"]).tolist(),
                         "all_proxy_degrees": graph["degree"].tolist(),
                         "interpretation": "relation-mining eligible, not validated ancestor effectiveness"},
            selection={"method": "uniform random without replacement over all eligible proxies",
                       "seed": args.seed, "depends_on_class_purity": False,
                       "selected_proxy_ids": selected.tolist(), "display_topk": args.topk,
                       "display_topk_is_training_K": False, "distinct_sample_ids_per_row": True},
            actual_ancestor_usage=ancestor_usage_report(saved), rows=rows,
            artifacts=["proxy_neighbors.png", "proxy_neighbors.html", "neighbors.csv",
                       "selected_clouds.npz", "feature_cache.npz", "summary.json"])
        write_json(output / "summary.json", report)
        print(json.dumps({"status": "complete", "eligible_count": int(graph["eligible"].sum()),
                          "selected_proxy_ids": selected.tolist(), "output_dir": str(output)}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status="failed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      wall_seconds=time.perf_counter() - started, error=str(exc))
        write_json(output / "summary.json", report)
        raise


if __name__ == "__main__":
    main()
