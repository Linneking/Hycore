"""Independent base-only loss/gradient and sampling contracts."""
from argparse import Namespace
import unittest
import geoopt
import torch
from ..controls import ShuffleBatchSampler, control_base_losses, fixed_configuration
from ..train import fixed_configuration as main_configuration


def args(control):
    return Namespace(control=control, seed=22, epochs=300, steps_per_epoch=200 if control=='B64' else 276,
        workers=2, eval_batch_size=32, train_eval_every=10, audit_epochs=(), smoke=False,
        smoke_eval_batches=1, warmup_epochs=0, lambda_hier=0.)


class ControlContract(unittest.TestCase):
    def test_shared_geometry_and_model_protocol(self):
        keys = ('c','D','clip_r','final_depth_cap','HIER_backward_hook','hook_scope',
                'extra_HIER_tangent_cap','radial_gamma','contrastive_margin',
                'alpha_radial','alpha_contrastive','model_optimizer','BN',
                'part_overwrites_whole','FPS_first_stage','validation')
        main = main_configuration(args('B64'))
        for control in ('B64','B0-32'):
            config = fixed_configuration(args(control))
            self.assertEqual({k: config[k] for k in keys}, {k: main[k] for k in keys})
            self.assertEqual(config['P'],0)
            self.assertEqual(config['lambda_hier_after_warmup'],0)
            self.assertIsNone(config['proxy_optimizer'])

    def test_shuffle_is_complete_without_replacement_and_replays(self):
        sampler = ShuffleBatchSampler(8856,22)
        batches = list(sampler)
        flat = sum(batches, [])
        self.assertEqual(len(batches),276)
        self.assertEqual(len(flat),8832)
        self.assertEqual(len(set(flat)),8832)
        self.assertTrue(all(len(batch)==32 for batch in batches))
        state = sampler.state_dict()
        other = ShuffleBatchSampler(8856,22)
        other.load_state_dict(state)
        self.assertEqual(batches,list(other))
        sampler.set_epoch(1)
        self.assertNotEqual(batches,list(sampler))

    def test_shuffle_loss_and_gradients_match_independent_original_formula(self):
        torch.manual_seed(83)
        # All labels equal deliberately: shuffle must retain same-class flip negatives.
        mu = (torch.randn(32,8,dtype=torch.float64)*.25).requires_grad_()
        nu = (torch.randn(32,8,dtype=torch.float64)*.25).requires_grad_()
        logits = torch.randn(32,40,dtype=torch.float64,requires_grad=True)
        labels = torch.zeros(32,dtype=torch.long)
        actual = control_base_losses(dict(mu=mu,nu=nu,logits=logits,child_count=200),labels,'B0-32')['base']
        ball = geoopt.PoincareBall(c=.1).double()
        target = torch.full_like(logits,.2/39)
        target[:,0] = .8
        reference = -(target*logits.log_softmax(-1)).sum(-1).mean()
        reference += .01*torch.relu(ball.dist(mu,nu)-ball.dist(mu,nu.flip(0))+2).mean()
        reference += .01*torch.relu(ball.dist0(nu)-ball.dist0(mu)+2.5).mean()
        torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-12)
        for left,right in zip(torch.autograd.grad(actual,(mu,nu,logits),retain_graph=True),
                              torch.autograd.grad(reference,(mu,nu,logits))):
            torch.testing.assert_close(left,right,atol=1e-12,rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
