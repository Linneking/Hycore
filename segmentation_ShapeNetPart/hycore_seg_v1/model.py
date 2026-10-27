"""Pointwise DGCNN, plus the explicitly reconstructed hyperbolic head."""

from types import SimpleNamespace

import torch
from torch import nn

from .vendor.dgcnn_model import DGCNN_partseg, get_graph_feature


class MobiusPointHead(nn.Module):
    """One shared, bias-free Möbius 128->50 map, preserving parameter count.

    Output ball coordinates are the logits supplied to cross entropy, following
    the paper's Möbius+softmax description. This unresolved author-code detail is
    an explicit reconstruction choice, not a recovered implementation.
    """

    def __init__(self, euclidean_weight):
        super().__init__()
        import geoopt

        self.weight = euclidean_weight
        self.ball = geoopt.PoincareBall(c=1.0)

    def forward(self, features):
        points = self.ball.expmap0(features.transpose(1, 2))
        logits = self.ball.mobius_matvec(self.weight[:, :, 0], points)
        return logits.transpose(1, 2).contiguous()


class DGCNNPartSeg(DGCNN_partseg):
    """Upstream architecture with the pre-final 128-d features exposed."""

    def __init__(self, head="euclidean", k=40, emb_dims=1024, dropout=0.5):
        if emb_dims != 1024:
            raise ValueError("Pinned upstream part-seg architecture requires emb_dims=1024")
        super().__init__(SimpleNamespace(k=k, emb_dims=emb_dims, dropout=dropout), 50)
        if head not in {"euclidean", "mobius"}:
            raise ValueError(head)
        self.head_kind = head
        if head == "mobius":
            # Retain exactly the originally initialized weight and RNG stream.
            self.conv11 = MobiusPointHead(self.conv11.weight)

    def forward_features(self, x, category):
        b, _, n = x.shape
        transform = self.transform_net(get_graph_feature(x, k=self.k))
        x = torch.bmm(x.transpose(2, 1), transform).transpose(2, 1)
        x = self.conv2(self.conv1(get_graph_feature(x, k=self.k)))
        x1 = x.max(dim=-1)[0]
        x = self.conv4(self.conv3(get_graph_feature(x1, k=self.k)))
        x2 = x.max(dim=-1)[0]
        x3 = self.conv5(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]
        pooled = self.conv6(torch.cat((x1, x2, x3), dim=1)).max(dim=-1, keepdim=True)[0]
        label = self.conv7(category.view(b, 16, 1))
        global_features = torch.cat((pooled, label), dim=1).repeat(1, 1, n)
        x = self.dp1(self.conv8(torch.cat((global_features, x1, x2, x3), dim=1)))
        x = self.dp2(self.conv9(x))
        return self.conv10(x)

    def forward(self, x, category, return_features=False):
        features = self.forward_features(x, category)
        logits = self.conv11(features)
        if return_features:
            return logits, features.transpose(1, 2).contiguous()
        return logits
