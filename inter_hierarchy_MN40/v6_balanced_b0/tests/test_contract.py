"""Behavioral regression gates for complete-state source/fork acceptance."""
import copy
import math
import unittest
from inter_hierarchy_MN40.v6_balanced_b0.contract import (
    SOURCE_VERSION,CHECKPOINT_VERSION,derive_b0_config,validate_checkpoint_contract)
from inter_hierarchy_MN40.hier_proxy_scratch_v5.sampler import SourceClassBatchSampler

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.reference={"epochs":300,"steps_per_epoch":200,"warmup_epochs":20,
                        "lambda_hier_after_warmup":.5,"proxy_optimizer":{"name":"AdamW"},
                        "schedule":"old","audit_epochs":[21,40],"workers":2,"BN":"local32"}
        self.config=derive_b0_config(self.reference)
        labels=[c for c in range(40) for _ in range(3)]
        sampler=SourceClassBatchSampler(labels,seed=22,steps=200)
        sampler.set_epoch(19)
        self.saved={
            "format":SOURCE_VERSION,"completed_epochs":20,"model_selection_only":False,
            "training_config":copy.deepcopy(self.reference),"split_sha256":"split",
            "train_ids":[1,3],"validation_ids":[2],"net":{"p":1},
            "optimizer":{"state":{0:{"momentum_buffer":"retained"}},
                "param_groups":[{"lr":.005+.095*(1+math.cos(math.pi*20/300))/2,"momentum":.9,"weight_decay":2e-4}]},
            "scheduler":{"last_epoch":20,"T_max":300,"eta_min":.005},
            "rank_states":[{"rank":rank,"sampler":sampler.state_dict(),
                "BN_buffers":{"mean":rank},"rng":{"python":1,"numpy":2,"torch":3,"cuda":4}} for rank in (0,1)],
            "best":{"epoch":12},"best_net":{"p":"selected12"},
            "proxy":{},"proxy_optimizer":{},"proxy_scheduler":{},
            "initialization":{},"initialized_model_sha256":"initial","commit":"source"}
        self.labels=labels
    def check(self,saved=None,config=None,reference=None):
        return validate_checkpoint_contract(saved or self.saved,config or self.config,
            reference or self.reference,"split",[1,3],[2])
    def test_source_and_b0_resume_with_history(self):
        self.check()
        b0=copy.deepcopy(self.saved)
        b0.update(format=CHECKPOINT_VERSION,training_config=self.config)
        self.check(b0)
    def test_reject_selection_partial_or_incomplete_rank_states(self):
        for key in ("model_selection_only","diagnostic_only","continuation_smoke_not_resumable"):
            bad=copy.deepcopy(self.saved);bad[key]=True
            with self.assertRaises(RuntimeError):self.check(bad)
        bad=copy.deepcopy(self.saved);bad["rank_states"]=bad["rank_states"][:1]
        with self.assertRaises(RuntimeError):self.check(bad)
        bad=copy.deepcopy(self.saved);del bad["rank_states"][1]["rng"]["cuda"]
        with self.assertRaises(RuntimeError):self.check(bad)
    def test_reject_wrong_source_epoch_or_restart(self):
        for key,value in (("last_epoch",0),("T_max",280),("eta_min",.001)):
            bad=copy.deepcopy(self.saved);bad["scheduler"][key]=value
            with self.assertRaises(RuntimeError):self.check(bad)
        bad=copy.deepcopy(self.saved);bad["completed_epochs"]=21
        with self.assertRaises(RuntimeError):self.check(bad)
    def test_reject_training_protocol_difference(self):
        for key,value in (("workers",4),("BN","SyncBN"),("steps_per_epoch",199)):
            bad=copy.deepcopy(self.saved);bad["training_config"][key]=value
            with self.assertRaises(RuntimeError):self.check(bad)
        bad=copy.deepcopy(self.config);bad["workers"]=4
        with self.assertRaises(RuntimeError):self.check(config=bad)
    def test_reject_identity_lr_and_missing_momentum(self):
        bad=copy.deepcopy(self.saved);bad["train_ids"]=[3,1]
        with self.assertRaises(RuntimeError):self.check(bad)
        bad=copy.deepcopy(self.saved);bad["optimizer"]["param_groups"][0]["lr"]=.1
        with self.assertRaises(RuntimeError):self.check(bad)
        bad=copy.deepcopy(self.saved);bad["optimizer"]["state"]={}
        with self.assertRaises(RuntimeError):self.check(bad)
    def test_sampler_restores_same_first_continuation_plan_for_both_ranks(self):
        for rank in (0,1):
            restored=SourceClassBatchSampler(self.labels,seed=22,steps=200,rank=rank)
            restored.load_state_dict(self.saved["rank_states"][rank]["sampler"])
            restored.set_epoch(20)
            original=SourceClassBatchSampler(self.labels,seed=22,steps=200,rank=rank)
            original.set_epoch(20)
            self.assertEqual(restored.global_plan(0),original.global_plan(0))
            with self.assertRaises(ValueError):
                wrong=copy.deepcopy(restored.state_dict());wrong["steps"]=199
                restored.load_state_dict(wrong)

if __name__=="__main__":
    unittest.main()
