"""Three explicit pointwise adaptations using the complete original HyCoRe head.

P0/Pglobal migrate the activity of official dgcnn_hmodel.DGCNN before global
pooling. Pseg3 retains the pinned community DGCNN_partseg frontend. Source
identities and the exact scope of reuse are recorded in vendor_manifest.json.
"""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ..hycore_seg_v1.vendor.dgcnn_model import (
    DGCNN_partseg,
    get_graph_feature as partseg_graph_feature,
)
from .native import MobiusLayer, PoincareBall, get_graph_feature


class _ClassNativePartSeg(nn.Module):
    """Official four-stage classification encoder, adapted before its pooling."""

    def __init__(self, architecture, k=40):
        super().__init__()
        self.architecture = architecture
        self.k = k
        self.feature_dim = 512
        self.uses_category = False
        self.manifold = PoincareBall(c=1, dim=512)
        self.manifold2 = PoincareBall(c=1, dim=256)

        # Preserve the original module types, widths, initialization order,
        # BN aliases, and activations from dgcnn_hmodel.DGCNN.
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1, nn.LeakyReLU(negative_slope=.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2, nn.LeakyReLU(negative_slope=.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3, nn.LeakyReLU(negative_slope=.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4, nn.LeakyReLU(negative_slope=.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5, nn.LeakyReLU(negative_slope=.2))
        projection_input = 1024 if architecture == "p0" else 2048
        self.linear1 = nn.Linear(projection_input, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.emb = MobiusLayer(512, 256, self.manifold)
        self.mlr = MobiusLayer(256, 50, self.manifold2)

    def forward_encoder(self, points):
        x1 = self.conv1(get_graph_feature(points, k=self.k)).max(dim=-1)[0]
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1)[0]
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1)[0]
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1)[0]
        return self.conv5(torch.cat((x1, x2, x3, x4), dim=1))

    def forward_features(self, points, category=None):
        del category  # P0/Pglobal never use object-category metadata as input.
        features = self.forward_encoder(points).transpose(1, 2).contiguous()
        if self.architecture == "pglobal":
            global_max = features.max(dim=1, keepdim=True)[0]
            features = torch.cat((features, global_max.expand_as(features)), dim=-1)
        features = self.linear1(features)
        # BN1d channels are F; the statistics include B and N, once per batch.
        features = F.leaky_relu(self.bn6(features.transpose(1, 2)), negative_slope=.2)
        return features.transpose(1, 2).contiguous()

    def forward_head(self, features):
        return self.mlr(self.emb(self.manifold.expmap0(features)))

    def head_parameters(self):
        for layer in (self.emb, self.mlr):
            yield from (parameter for parameter in layer.parameters() if parameter.requires_grad)

    def forward(self, points, category=None, return_features=False):
        features = self.forward_features(points, category)
        logits = self.forward_head(features).transpose(1, 2).contiguous()
        return (logits, features) if return_features else logits


class _CommunityNativePartSeg(DGCNN_partseg):
    """Original partseg frontend with its last three maps replaced at equal width."""

    def __init__(self, k=40, dropout=.5):
        super().__init__(SimpleNamespace(k=k, emb_dims=1024, dropout=dropout), 50)
        # The original frontend is instantiated directly. Unused affine/BN
        # parameters must not remain in the optimizer or the state dictionary.
        for name in ("conv9", "conv10", "conv11", "bn9", "bn10", "dp2"):
            delattr(self, name)
        self.architecture = "pseg3"
        self.feature_dim = 256
        self.uses_category = True
        self.manifold = PoincareBall(c=1, dim=256)
        self.manifold2 = PoincareBall(c=1, dim=256)
        self.manifold3 = PoincareBall(c=1, dim=128)
        self.emb = MobiusLayer(256, 256, self.manifold)
        self.emb2 = MobiusLayer(256, 128, self.manifold2)
        self.mlr = MobiusLayer(128, 50, self.manifold3)

    def forward_features(self, points, category):
        batch_size, _, num_points = points.shape
        if category is None or category.shape != (batch_size, 16):
            raise ValueError("Pseg3 requires the original [B,16] category one-hot")
        transform = self.transform_net(partseg_graph_feature(points, k=self.k))
        x = torch.bmm(points.transpose(2, 1), transform).transpose(2, 1)
        x = self.conv2(self.conv1(partseg_graph_feature(x, k=self.k)))
        x1 = x.max(dim=-1)[0]
        x = self.conv4(self.conv3(partseg_graph_feature(x1, k=self.k)))
        x2 = x.max(dim=-1)[0]
        x3 = self.conv5(partseg_graph_feature(x2, k=self.k)).max(dim=-1)[0]
        pooled = self.conv6(torch.cat((x1, x2, x3), dim=1)).max(dim=-1, keepdim=True)[0]
        label = self.conv7(category.reshape(batch_size, 16, 1))
        global_features = torch.cat((pooled, label), dim=1).repeat(1, 1, num_points)
        features = self.dp1(self.conv8(torch.cat((global_features, x1, x2, x3), dim=1)))
        return features.transpose(1, 2).contiguous()

    def forward_head(self, features):
        ball_features = self.manifold.expmap0(features)
        return self.mlr(self.emb2(self.emb(ball_features)))

    def head_parameters(self):
        for layer in (self.emb, self.emb2, self.mlr):
            yield from (parameter for parameter in layer.parameters() if parameter.requires_grad)

    def forward(self, points, category, return_features=False):
        features = self.forward_features(points, category)
        logits = self.forward_head(features).transpose(1, 2).contiguous()
        return (logits, features) if return_features else logits


def NativePartSeg(architecture, k=40, dropout=.5):
    """Construct a fresh arm; input [B,3,N], logits [B,50,N], features [B,N,F]."""
    architecture = architecture.lower()
    if architecture in ("p0", "pglobal"):
        return _ClassNativePartSeg(architecture, k=k)
    if architecture == "pseg3":
        return _CommunityNativePartSeg(k=k, dropout=dropout)
    raise ValueError("Unknown native architecture: " + architecture)
