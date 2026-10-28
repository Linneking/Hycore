"""Isolated reuse of original HyCoRe hyperbolic components.

Source: diegovalsesia/HyCoRe at ef56a73020e309b1009c486fea1f601d3f7b3252.
The geometry file is loaded directly, bypassing the unrelated pointMLP package
initializer and legacy absolute imports. The two layer classes, triplet kernel,
and CE function below are faithful source copies. Graph extraction changes only
its hard-coded CUDA device to the input device. See vendor_manifest.json.
"""

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from geoopt import ManifoldParameter


def _repository_root():
    root = Path(__file__).resolve().parents[2]
    if (root / "classification_ModelNet40").is_dir():
        return root
    # Permit the writable development staging location before installation.
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "classification_ModelNet40").is_dir():
            return candidate
    raise ImportError("Cannot locate the original HyCoRe repository")


def verify_provenance():
    """Reject drift in original sources used for this fixed adaptation."""
    manifest_path = Path(__file__).with_name("vendor_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = _repository_root()
    for item in manifest["files"]:
        # Normalized LF hashes are invariant to Windows checkout line endings.
        content = (root / item["path"]).read_text(encoding="utf-8")
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual != item["lf_sha256"]:
            raise RuntimeError("Original source provenance mismatch: " + item["path"])
    return manifest


SOURCE_PROVENANCE = verify_provenance()


def _official_geometry():
    module_name = "hycore_seg_v2_official_poincareball"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _repository_root() / "classification_ModelNet40/models/manifolds/poincareball.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PoincareBall = _official_geometry().PoincareBall

class RiemannianLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold

        self._weight = Parameter(torch.Tensor(out_features, in_features))
        self.over_param = over_param
        self.weight_norm = weight_norm
        if self.over_param:
            self._bias = ManifoldParameter(torch.Tensor(out_features, in_features), manifold=manifold)
        else:
            self._bias = Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()

    @property
    def weight(self):
        return self.manifold.transp0(self.bias, self._weight) # weight \in T_0 => weight \in T_bias

    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(self._weight * self._bias) # reparameterisation of a point on the manifold

    def reset_parameters(self):
        init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad(): self._bias.set_(self.manifold.expmap0(self._bias))

class MobiusLayer(RiemannianLayer):
    def __init__(self, in_features, out_features, manifold, over_param=False, weight_norm=False):
        super(MobiusLayer, self).__init__(in_features, out_features, manifold, over_param, weight_norm)

    def forward(self, input):
        res = self.manifold.mobius_matvec(self.weight, input)
        return res

def hype_triplet_losses(parent_mu , pos_child_mu, hier_margin=0.2, contr_margin=4, ball_dim = 256, one_child = False, opposite_hier=False):

        if one_child:
               neg_child_mu = torch.flip(parent_mu,[0])
        else:
               neg_child_mu = torch.flip(pos_child_mu,[0])

        ball = PoincareBall(c=1.0,dim=ball_dim)
        parent_norm = torch.norm(parent_mu,dim=1)       #dist(parent_mu, parent_mu*0, keepdim=True).pow(2).sum(1)
        pos_norm = torch.norm(pos_child_mu ,dim=1)

        par_norm = ball.dist0(parent_mu)
        p_norm = ball.dist0(pos_child_mu)       #.pow(2).sum(1)

        distance_positive = ball.dist(parent_mu, pos_child_mu)  #, keepdim=True).pow(2).sum(1)
        distance_negative = ball.dist(parent_mu, neg_child_mu)  #, keepdim=True).pow(2).sum(1)


        distance = distance_positive - distance_negative + contr_margin
        #distance = -distance_positive + distance_negative + mar_dist
        if opposite_hier:
               norm = par_norm-p_norm + hier_margin
        else:
               norm = -par_norm+p_norm + hier_margin            #Standard version wich improves the baseline
        #norm = par_norm-p_norm + margin                        #Possible change: parent becomes the mean of its children --EXperiment 07--

        triplet = torch.mean(torch.max(distance, torch.zeros_like(distance)))                   #F.relu(distance_positive - distance_negative + 0.5 )
        hierarch =torch.mean(torch.max(norm, torch.zeros_like(norm)))                   #F.relu(-par_norm+p_norm + margin)

        return parent_norm.mean(), pos_norm.mean()  ,distance_positive.mean(),distance_negative.mean(),triplet, hierarch

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
