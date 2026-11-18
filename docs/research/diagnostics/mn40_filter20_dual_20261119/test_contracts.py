"""Small CPU contracts for task identity, sampling and pinned training routing."""
import ast
import importlib.util
import random
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('filtered_runner',HERE/'runner.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class Contracts(unittest.TestCase):
    def test_script_directory_does_not_shadow_standard_queue(self):
        self.assertFalse((HERE/'queue.py').exists())

    def test_subset_counts_and_mapping(self):
        import csv
        p=HERE.parents[1]/'171_MN40_TS06_TS30_SAMPLING_ERROR_AUDIT_2026-11-18/best_classes.csv'
        with p.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        keep=[r for r in rows if 130<=int(r['train_size'])<=400]
        self.assertEqual(tuple(int(r['class_id']) for r in keep),runner.KEEP)
        self.assertEqual(sum(int(r['train_size']) for r in keep),4298)
        self.assertEqual(sum(int(r['test_size']) for r in keep),1398)
        self.assertEqual(4298//32,134)

    def test_cb32_retains_sixteen_pairs_and_independent_rng(self):
        from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler
        from collections import Counter
        labels=[c for c in range(20) for _ in range(200)]
        sampler=SourceClassBatchSampler(labels,world_size=1,classes_per_rank=16,
            instances_per_class=2,steps=134,seed=4780)
        before=random.getstate()
        for e in (0,1,299):
            for step in (0,1,133):
                plan=sampler.global_plan(step,epoch=e)
                self.assertEqual(len(plan.indices),32)
                self.assertEqual(sorted(Counter(plan.labels).values()),[2]*16)
                self.assertEqual(plan,sampler.global_plan(step,epoch=e))
        self.assertEqual(before,random.getstate())

    def test_pinned_adapter_has_expected_sizes(self):
        adapter=runner.previous.pinned_text(runner.base.__file__,runner.previous.ADAPTER_SHA)
        modified=runner.main_source(adapter);ast.parse(modified)
        self.assertIn('exp.train_size',modified)
        self.assertIn('exp.make_train_loader(train_set)',modified)
        self.assertNotIn('if len(train_set) != 9840',modified)

    def test_dual_branch_backwards_before_one_update(self):
        source=runner.previous.pinned_text(runner.previous.ORIGINAL,runner.previous.TRAIN_SHA)
        fake=type('Fake',(),dict(enabled=True,arm='TS53'))()
        modified=runner.train_source(source,fake);ast.parse(modified)
        self.assertEqual(modified.count('optimizer.step()'),1)
        self.assertLess(modified.index('loss.backward()'),modified.index('exp.aux_backward(net)'))
        self.assertLess(modified.index('exp.aux_backward(net)'),modified.index('clip_grad_norm_'))
        self.assertNotIn('exp.add_loss(loss',modified)

    def test_off_has_no_auxiliary_or_hierarchy_loss(self):
        source=runner.previous.pinned_text(runner.previous.ORIGINAL,runner.previous.TRAIN_SHA)
        for arm in runner.ARMS:
            fake=type('Fake',(),dict(enabled=False,arm=arm))()
            modified=runner.train_source(source,fake);ast.parse(modified)
            self.assertNotIn('exp.aux_backward(net)',modified)
            self.assertNotIn('exp.add_loss(loss',modified)

    def test_distinct_names_all_seed4780(self):
        values=[runner.canonical(a) for a in runner.ARMS]
        self.assertEqual(len(set(values)),5)
        self.assertTrue(all(v.endswith('(S4780)') for v in values))


if __name__=='__main__':unittest.main()
