"""ScanObjectNN DeLA v1 with a shared HyCoRe part–whole embedding head."""

from contextlib import contextmanager
from math import isfinite

import torch
from delacls import DelaCls, Stage, calc_pwd, pointnet2_utils
from modelsnet.manifolds import PoincareBall
from modelsnet.ops.manifold_layers import MobiusLayer
from torch import nn


class DelaPartWhole(DelaCls):
    WHOLE_POINTS = 1024
    TRAIN_FPS_POINTS = 1200
    MIN_PART_POINTS = 384
    MAX_PART_POINTS = 511

    def __init__(self, args, part_seed, *, exp_input_scale=1.0):
        if args.ns[0] != self.WHOLE_POINTS or max(args.ns[1:]) > self.MIN_PART_POINTS:
            raise ValueError("Part–whole views require the ScanObjectNN stage point counts")
        if not isfinite(exp_input_scale) or exp_input_scale <= 0:
            raise ValueError("exp_input_scale must be finite and positive")
        super().__init__(args)
        self.head = self.head[:3]
        self.manifold = PoincareBall(c=1, dim=512)
        self.emb = MobiusLayer(512, 256, self.manifold)
        self.classifier = MobiusLayer(
            256, args.num_classes, PoincareBall(c=1, dim=256)
        )
        self.exp_input_scale = float(exp_input_scale)
        self.last_geometry = {}
        self._part_seed = int(part_seed)
        self._part_cpu_rng = torch.Generator().manual_seed(self._part_seed).get_state()
        self._part_cuda_rng = None

    def get_extra_state(self):
        return {
            "exp_input_scale": self.exp_input_scale,
            "seed": self._part_seed,
            "cpu_rng": self._part_cpu_rng,
            "cuda_rng": self._part_cuda_rng,
        }

    def set_extra_state(self, state):
        self.exp_input_scale = state["exp_input_scale"]
        self._part_seed = state["seed"]
        self._part_cpu_rng = state["cpu_rng"].cpu()
        cuda_rng = state["cuda_rng"]
        self._part_cuda_rng = cuda_rng.cpu() if cuda_rng is not None else None

    @contextmanager
    def _part_random_stream(self, device):
        cuda = device.type == "cuda"
        devices = [device.index] if cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._part_cpu_rng)
            if cuda:
                if self._part_cuda_rng is None:
                    generator = torch.Generator(device=device).manual_seed(self._part_seed)
                    self._part_cuda_rng = generator.get_state()
                torch.cuda.set_rng_state(self._part_cuda_rng, device)
            try:
                yield
            finally:
                self._part_cpu_rng = torch.get_rng_state()
                if cuda:
                    self._part_cuda_rng = torch.cuda.get_rng_state(device)

    @contextmanager
    def _part_encoder(self, point_count):
        stages = [stage for stage in self.stage.modules() if isinstance(stage, Stage)]
        tracked_bn = [
            module for module in self.modules()
            if isinstance(module, nn.BatchNorm1d) and module.track_running_stats
        ]
        stage_training = [stage.training for stage in stages]
        whole_points = self.stage.n
        try:
            self.stage.n = point_count
            # Only Stage uses its training flag for the coordinate head.
            # Do not propagate the flag to train-mode DropPath or LFP children.
            for stage in stages:
                stage.training = False
            for bn in tracked_bn:
                bn.track_running_stats = False
            yield
        finally:
            for bn in tracked_bn:
                bn.track_running_stats = True
            for stage, training in zip(stages, stage_training):
                stage.training = training
            self.stage.n = whole_points

    def _sample_whole(self, xyz):
        fps_points = self.TRAIN_FPS_POINTS if self.training else self.WHOLE_POINTS
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[1] < fps_points:
            raise ValueError(f"Expected [B, N, 3] xyz with at least {fps_points} points")
        idx = pointnet2_utils.furthest_point_sample(xyz, fps_points).long()
        if self.training:
            subset = torch.randperm(fps_points, device=xyz.device)[:self.WHOLE_POINTS]
            idx = idx[:, subset]
        return xyz.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))

    def _part_indices(self, whole_xyz, part_count):
        batch_size, point_count, _ = whole_xyz.shape
        center_idx = torch.randint(point_count, (batch_size, 1, 1), device=whole_xyz.device)
        center = whole_xyz.gather(1, center_idx.expand(-1, -1, 3))
        distances = (whole_xyz - center).square().sum(dim=-1)
        near = distances.topk(part_count, dim=1, largest=False, sorted=False).indices
        return near.sort(dim=1).values

    def _encode(self, xyz, *, view):
        features, closs = self.stage(None, xyz, None, calc_pwd(xyz))
        batch, points, channels = features.shape
        features = self.proj(features.view(batch * points, channels))
        features = features.view(batch, points, -1).max(dim=1).values
        features = self.head(features)
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            tangent = features.float() * self.exp_input_scale
            mapped = self.manifold.expmap0(tangent)
            z = self.emb(mapped)
            self.last_geometry[view] = {
                "tangent_norm": tangent.detach().norm(dim=-1),
                "expmap_norm": mapped.detach().norm(dim=-1),
                "embedding_norm": z.detach().norm(dim=-1),
            }
        return z, closs

    def _whole_output(self, whole_xyz):
        z, closs = self._encode(whole_xyz, view="whole")
        with torch.autocast(device_type=whole_xyz.device.type, enabled=False):
            scores = self.classifier(z)
        return scores, closs, z

    def forward_whole(self, xyz):
        """Whole-only control with the same sampler, encoder and double Möbius head."""
        scores, closs, z = self._whole_output(self._sample_whole(xyz))
        return (scores, closs, z) if self.training else scores

    def forward(self, xyz, *, part_count=None):
        if not self.training:
            return self.forward_whole(xyz)

        whole_xyz = self._sample_whole(xyz)
        with self._part_random_stream(xyz.device):
            if part_count is None:
                part_count = int(torch.randint(self.MIN_PART_POINTS, self.MAX_PART_POINTS + 1, ()))
            if not self.MIN_PART_POINTS <= part_count <= self.MAX_PART_POINTS:
                raise ValueError("Part point count must be between 384 and 511")
            idx = self._part_indices(whole_xyz, part_count)
            part_xyz = whole_xyz.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
            with self._part_encoder(part_count):
                z_part, _ = self._encode(part_xyz, view="part")

        scores, closs, z_whole = self._whole_output(whole_xyz)
        return scores, closs, z_whole, z_part, part_count
