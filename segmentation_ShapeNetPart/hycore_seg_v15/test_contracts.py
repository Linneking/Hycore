import contextlib,io,json,sys,tempfile
from pathlib import Path
import torch
from segmentation_ShapeNetPart.hycore_seg_v15.losses import HyCoReSegLoss,overwrite_whole_prefix
from segmentation_ShapeNetPart.hycore_seg_v11 import train as reference
from segmentation_ShapeNetPart.hycore_seg_v11.losses import HyCoReSegLoss as BaselineLoss
from segmentation_ShapeNetPart.hycore_seg_v6.cpu_entrypoint_check import write_fixture
from segmentation_ShapeNetPart.hycore_seg_v3.monitoring import GeometryMonitor
torch.set_num_threads(2)
checks={}
for values,indices,expected,gradient in (
 ([0.,1.,2.,3.,4.,5.],[4,3,5],[4.,3.,5.,3.,4.,5.],[0.,0.,0.,2/6,2/6,2/6]),
 ([0.,1.,2.,3.],[1,3],[1.,3.,2.,3.],[0.,.25,.25,.5]),
):
 h=torch.tensor(values).reshape(1,-1,1).requires_grad_()
 saved=h.detach().clone()
 x=overwrite_whole_prefix(h,torch.tensor([indices]))
 torch.testing.assert_close(x.detach().flatten(),torch.tensor(expected))
 x.mean().backward()
 torch.testing.assert_close(h.grad.flatten(),torch.tensor(gradient))
 assert torch.equal(h.detach(),saved)
checks["prefix_order_original_unchanged_and_multiplicity_gradient"]=True
h=(torch.randn(4,32,256,generator=torch.Generator().manual_seed(22))*.01).requires_grad_()
cats=torch.arange(4);original=h.detach().clone()
baseline=BaselineLoss(part_min=4,part_max=8,seed=4780)
candidate=HyCoReSegLoss(part_min=4,part_max=8,seed=4780)
monitor=GeometryMonitor().begin(True)
candidate.monitor=monitor
value,diag=candidate(h,cats)
_,base_diag=baseline(h,cats)
for k in ("negative_indices","part_centers","part_points","part_indices_sha256"):
 assert diag[k]==base_diag[k],k
assert torch.equal(candidate.generator.get_state(),baseline.generator.get_state())
assert torch.equal(candidate.negative_generator.get_state(),baseline.negative_generator.get_state())
assert torch.equal(h.detach(),original)
assert torch.isfinite(value) and value.requires_grad
value.backward()
assert h.grad is not None and torch.isfinite(h.grad).all() and h.grad.norm()>0
assert all(x.requires_grad for x in candidate.last_components.values())
torch.testing.assert_close(monitor.nodes["whole_original_tangent"],h.mean(1))
checks["one_original_selection_rng_negative_and_gradients"]=True
with tempfile.TemporaryDirectory() as temp:
 p=Path(temp);data=p/"data";out=p/"out";write_fixture(data)
 args=reference.arguments(["--config","segmentation_ShapeNetPart/hycore_seg_v15/configs/tan_r_prefix.json",
  "--data",str(data),"--out",str(out),"--arm","h","--device","cpu","--allow-dirty-cpu",
  "--epochs","1","--num-points","32","--k","4","--batch-size","2","--eval-batch-size","2",
  "--part-min","4","--part-max","8","--workers","0","--max-steps","1","--no-final-test","--protocol","validation"])
 if not sys.platform.startswith("linux"):args.expected_initial_sha256=None
 with contextlib.redirect_stdout(io.StringIO()):
  reference.main(args,loss_type=HyCoReSegLoss,run_format="hycore_seg_v15")
 m=json.loads((out/"manifest.json").read_text())
 row=json.loads((out/"metrics.jsonl").read_text().splitlines()[0])
 g=json.loads((out/"geometry.jsonl").read_text().splitlines()[0])
 assert m["display_name"]=="TAN-R-PREFIX" and m["regularizers_active"]
 assert row["updates"]==1 and row["model_finite"] and row["optimizer_finite"]
 assert all(v==1 for v in row["bn_counts"].values())
 assert g["gradients"]["hier"]["loss_requires_grad"] and g["gradients"]["contr"]["loss_requires_grad"]
 assert g["gradients"]["ce"]["nodes"]["h"]["status"]=="nonzero"
 assert g["gradients"]["hier"]["nodes"]["h"]["status"]=="nonzero"
 assert g["gradients"]["contr"]["nodes"]["h"]["status"]=="nonzero"
 assert m["precision"]=="fp32" and not torch.cuda.is_initialized()
checks["full_cpu_entrypoint_initial_guard_BN_once_Ron_and_CE_R_gradients"]=True
print(json.dumps(checks))
