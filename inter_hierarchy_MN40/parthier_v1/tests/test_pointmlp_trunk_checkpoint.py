"""CPU layout stub checked against the released PointMLP forward source."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import unittest

import geoopt
import torch
from torch import nn
import torch.nn.functional as F

import inter_hierarchy_MN40
from inter_hierarchy_MN40.parthier_v1.base_protocol import HyCoReTrainingForward


def released_forward():
    source = next(Path(root) / "models" / "pointmlp.py" for root in inter_hierarchy_MN40.__path__
                  if (Path(root) / "models" / "pointmlp.py").is_file())
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Hype_Model")
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = {"F": F}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["forward"]


class Group(nn.Module):
    def forward(self, xyz, features):
        indices = torch.arange(0, xyz.shape[1], 2, device=xyz.device)
        return xyz[:, indices], features[:, indices, None, :].expand(-1, -1, 2, -1)


class Pre(nn.Module):
    def __init__(self, source_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(source_dim, target_dim, 1, bias=False), nn.BatchNorm1d(target_dim), nn.ReLU())

    def forward(self, grouped):
        return self.net(grouped.mean(2).permute(0, 2, 1))


class Embedding(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv1d(3, 6, 1, bias=False), nn.BatchNorm1d(6), nn.ReLU(), nn.Dropout(.2))
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return super().forward(value)


class Ball(geoopt.PoincareBall):
    def __init__(self):
        super().__init__(c=1.)
        self.exp_calls = 0

    def expmap0(self, value, **kwargs):
        self.exp_calls += 1
        return super().expmap0(value, **kwargs)


class Mobius(nn.Module):
    def __init__(self, source_dim, target_dim, ball):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(target_dim, source_dim) * .1)
        self.ball = ball
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return self.ball.mobius_matvec(self.weight, value)


class Hype_Model(nn.Module):
    """CPU layout stub; its forward is loaded verbatim from the real source."""
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.stages = 2
        self.local_grouper_list = nn.ModuleList([Group(), Group()])
        self.pre_blocks_list = nn.ModuleList([Pre(6, 8), Pre(8, 10)])
        self.pos_blocks_list = nn.ModuleList([
            nn.Sequential(nn.Conv1d(dim, dim, 1, bias=False), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(.1))
            for dim in (8, 10)])
        self.proj = nn.Sequential(nn.Linear(10, 12, bias=False), nn.BatchNorm1d(12), nn.ReLU(), nn.Dropout(.25))
        self.manifold = Ball()
        self.manifold2 = Ball()
        self.emb = Mobius(12, 9, self.manifold)
        self.classifier = Mobius(9, 5, self.manifold2)


class PointMLPTrunkCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        Hype_Model.forward = released_forward()

    def test_original_source_outputs_gradients_bn_rng_and_heads_match_with_trunk_only_recompute(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                torch.manual_seed(4780)
                normal = Hype_Model().to(dtype=dtype).train()
                checked = copy.deepcopy(normal)
                plain = HyCoReTrainingForward(normal)
                wrapped = HyCoReTrainingForward(checked, activation_checkpoint=True)
                self.assertEqual(set(plain.state_dict()), set(wrapped.state_dict()))
                points = torch.randn(4, 820, 3, dtype=dtype)
                crops = {"centers": [799, 403, 17, 0], "whole_centers": [0, 17, 300, 819]}
                self.assertFalse(points.requires_grad)
                initial_rng = torch.get_rng_state().clone()
                expected = plain(points, 800, **crops)
                sum(expected[key].square().sum() for key in ("mu", "parts", "logits")).backward()
                expected_rng = torch.get_rng_state().clone()
                torch.set_rng_state(initial_rng)
                actual = wrapped(points, 800, **crops)
                sum(actual[key].square().sum() for key in ("mu", "parts", "logits")).backward()
                for key in ("mu", "parts", "logits"):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                gradient_error_squared, gradient_reference_squared = 0., 0.
                for (name, reference), (other_name, value) in zip(normal.named_parameters(), checked.named_parameters()):
                    self.assertEqual(name, other_name)
                    if reference.requires_grad:
                        self.assertIsNotNone(reference.grad)
                        self.assertIsNotNone(value.grad)
                        # Four shared-parameter contributions can accumulate in
                        # a different FP32 order across checkpoint boundaries.
                        # This absolute floor remains 10x below the real CUDA
                        # gate; the additional global L2 gate below is tighter.
                        rtol, atol = ((1e-5, 1e-6) if dtype == torch.float32 else (1e-12, 1e-14))
                        torch.testing.assert_close(value.grad, reference.grad, rtol=rtol, atol=atol)
                        gradient_error_squared += float((value.grad.double() - reference.grad.double()).square().sum())
                        gradient_reference_squared += float(reference.grad.double().square().sum())
                global_relative_l2 = (gradient_error_squared / max(gradient_reference_squared, 1e-60)) ** .5
                self.assertLessEqual(global_relative_l2, 1e-5 if dtype == torch.float32 else 1e-12)
                for (name, reference), (other_name, value) in zip(normal.named_buffers(), checked.named_buffers()):
                    self.assertEqual(name, other_name)
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                    if name.endswith("num_batches_tracked"):
                        self.assertEqual(int(value), 4)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                self.assertEqual((normal.embedding.calls, checked.embedding.calls), (4, 8))
                for model in (normal, checked):
                    self.assertEqual(model.manifold.exp_calls, 4)
                    self.assertEqual(model.emb.calls, 4)
                    self.assertEqual(model.classifier.calls, 1)
                self.assertEqual(actual["input_stats"]["checkpoint_scope"], "pointmlp_euclidean_trunk")


if __name__ == "__main__":
    unittest.main(verbosity=2)
