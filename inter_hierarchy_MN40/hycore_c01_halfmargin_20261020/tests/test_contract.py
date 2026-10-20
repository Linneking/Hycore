"""Contracts for changing margin thresholds without changing loss weights or RNG."""
import ast
from pathlib import Path
import unittest
import torch
import geoopt
from inter_hierarchy_MN40.hycore_c01_halfmargin_20261020 import train as adapter

REPO = Path(__file__).resolve().parents[3]

class Ball(geoopt.PoincareBall):
    def __init__(self, dim, c=.1):
        super().__init__(c=c)
        self.register_buffer('dim', torch.tensor(dim))

class HalfMarginTests(unittest.TestCase):
    def test_real_loss_and_gradients_match_half_thresholds(self):
        source = REPO / 'classification_ModelNet40/hutil.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'hype_triplet_losses')
        scope = {'torch': torch, 'PoincareBall': lambda dim, c=1.: Ball(dim, c=.1)}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
        original = scope['hype_triplet_losses']
        wrapped = adapter.half_margin_criterion(original)
        ball = Ball(2)
        for count in (200, 400, 600):
            u = torch.tensor([[.3, .0], [.1, .1], [.0, .2]], dtype=torch.float64, requires_grad=True)
            whole, part = ball.expmap0(u), ball.expmap0(u * .6)
            actual = wrapped(whole, part, hier_margin=1000/count, contr_margin=4, ball_dim=2)
            expected = original(whole, part, hier_margin=500/count, contr_margin=2, ball_dim=2)
            for x, y in zip(actual, expected):
                self.assertTrue(torch.allclose(x, y))
            a = torch.autograd.grad(actual[4]+actual[5], u, retain_graph=True)[0]
            b = torch.autograd.grad(expected[4]+expected[5], u)[0]
            self.assertTrue(torch.allclose(a, b))
            old = original(whole.detach(), part.detach(), hier_margin=1000/count, contr_margin=4, ball_dim=2)
            self.assertFalse(torch.allclose(actual[4], old[4]*.5))

    def test_wrapper_preserves_options_and_does_not_draw_rng(self):
        received = {}
        def spy(*args, **kwargs):
            received.update(kwargs)
            return args
        before = torch.get_rng_state().clone()
        adapter.half_margin_criterion(spy)(1, 2, 5, 4, 64, True, True)
        self.assertEqual(received, dict(hier_margin=2.5, contr_margin=2., ball_dim=64, one_child=True, opposite_hier=True))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_same_initialization_as_completed_cap_baseline(self):
        original = adapter.import_original(REPO)
        torch.manual_seed(4780)
        model = adapter.configure_model(original.Hype_pointMLP(), capped=True)
        self.assertEqual(adapter.state_digest(model), '8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d')
        adapter.verify_curvature(model)

    def test_only_new_margin_wrapper_changes_original_training_calls(self):
        from inter_hierarchy_MN40.hycore_c01_ablation_20261020 import original_adapter as baseline
        for name in ('configure_model', 'install_data_root', 'install_loss_curvature', 'cap_final_mu', 'rng_state'):
            import inspect
            self.assertEqual(inspect.getsource(getattr(adapter, name)), inspect.getsource(getattr(baseline, name)))
        tree = ast.parse(Path(adapter.__file__).read_text(encoding='utf-8-sig'))
        calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        self.assertIn('original.train', calls)
        self.assertIn('original.validate', calls)
        self.assertIn('half_margin_criterion', calls)

if __name__ == '__main__':
    unittest.main()