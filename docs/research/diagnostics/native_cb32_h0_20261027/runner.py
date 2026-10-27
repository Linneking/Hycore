"""TS26: original c1 HyCoRe CE/intra, CB32 replacement batches, no HIER.

The reviewed continuous-stream loop is reused through SHA-checked edits.
Original entry points and all historical implementations remain unchanged.
"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ORIGINAL = REPO / 'inter_hierarchy_MN40/hier_proxy_cycle_v9c/train.py'
PARENT_SHA = '4754913a64bb355080e55853e9a502d211d6153fbb6a23cd0c182dad437c7e56'
VERSION = 'hycore-native-cb32-h0-20261027-v1'
NAME = 'HC-C1-Cap0-GS0-M1000_4-CB32x200 / H0 (S4780)'
INITIAL_DIGEST = 'a207a450feb3c70c34d7ae567b923130b9ff6d88a1a10f4d40755b18d9fb2baf'
INITIAL_HASH = '459a93c9589ee369e1473cd74236c066d11278aeaea5c5309d67b461e480f297'

REPLACEMENTS = (
    ('CHECKPOINT_VERSION = "hycore-cycle-v9c-c01-cap23-gs1-m5002-testselect-1"', f'CHECKPOINT_VERSION = "{VERSION}"'),
    ('parser.add_argument("--hier", choices=("off", "on"), required=True)', 'parser.add_argument("--hier", choices=("off",), required=True)'),
    ('parser.add_argument("--steps-per-epoch", type=int, default=300)', 'parser.add_argument("--steps-per-epoch", type=int, default=200)'),
    ('!= (300, 300, 8, 16)', '!= (300, 200, 8, 16)'),
    ('Production requires300x300, workers8, eval16', 'Production requires300x200, workers8, eval16'),
    ('def fixed_configuration(args):', 'def parent_configuration(args):'),
    ('from .sampler import ClassCycleBatchSampler', 'from ..hier_proxy_scratch_v5.sampler import SourceClassBatchSampler'),
    ('ClassCycleBatchSampler(labels[train_ids], rank=rank,', 'SourceClassBatchSampler(labels[train_ids], world_size=1, classes_per_rank=16, instances_per_class=2, rank=rank,'),
    ('if int(os.environ["WORLD_SIZE"]) != 2:', 'if int(os.environ["WORLD_SIZE"]) != 1:'),
    ('Cycle-B64 requires exactly two ranks', 'Native CB32 requires exactly one rank'),
    ('len(saved["rank_states"]) != 2', 'len(saved["rank_states"]) != 1'),
    ('devices = [None, None]', 'devices = [None]'),
    ('rank_peaks = [None, None]', 'rank_peaks = [None]'),
    ('rank_states = [None, None]', 'rank_states = [None]'),
    ('"rank_order": [0, 1]', '"rank_order": [0]'),
    ('"child_count": child_count, "native_c": .1,', '"child_count": child_count, "native_c": 1.,'),
    ('"whole_part": "c.1 cap2.3 baseGS once"', '"whole_part": "original c1; no added cap/GS"'),
    ('"V9C_CYCLE_B64_" + args.hier.upper() + "_COMPLETED"', '"NATIVE_CB32_H0_COMPLETED"'),
)


def sha_text(source):
    return hashlib.sha256(source.replace('\r\n', '\n').encode()).hexdigest()


def transformed_source(source):
    source = source.replace('\r\n', '\n')
    if sha_text(source) != PARENT_SHA:
        raise RuntimeError('Reviewed parent source SHA differs')
    for before, after in REPLACEMENTS:
        if source.count(before) != 1:
            raise RuntimeError('Edit boundary is not unique: ' + before)
        source = source.replace(before, after, 1)
    ast.parse(source)
    return source


def configuration(args, original):
    if args.hier != 'off':
        raise ValueError('This experiment has no HIER')
    result = dict(original)
    result.update(
        canonical_name=NAME, display_name=NAME, experiment_alias='TS26', baseline_alias='B15',
        implementation_variant=VERSION, world_size=1, global_batch=32,
        classes_per_rank=16, instances_per_class=2,
        sampler='uniform 16 distinct classes; 2 instances/class with replacement; SourceClassBatchSampler v5',
        sampler_epoch_reset=True, sampler_state_semantics='deterministic seed/epoch/step; no cyclic cursors',
        c=1., base_c=1., native_c=1., native_extra_cap=None, native_GS=False,
        contrastive_margin=4., radial_gamma=1000., radial_margin='1000 / child_count',
        geometry_domains={'native_whole_part': 1.}, hook_scope='no added base or HIER hook',
        BN='ordinary local32; part then whole; single-rank DDP; no SyncBN',
        source_checkpoint=None, teacher=None,
        schedule='300-epoch cosine; no restart; fresh initialization',
        comparison_scope='original HyCoRe geometry/objective; sampling and epoch budget change; no HIER',
        training_source={'parent_relative_path': str(ORIGINAL.relative_to(REPO)),
                         'parent_normalized_sha256': PARENT_SHA,
                         'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'transformed_normalized_sha256': sha_text(transformed_source(ORIGINAL.read_text(encoding='utf-8')))})
    return result


def build_namespace():
    for path in (REPO, REPO / 'inter_hierarchy_MN40'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    source = transformed_source(ORIGINAL.read_text(encoding='utf-8'))
    namespace = {'__name__': 'native_cb32_h0_successor', '__file__': str(ORIGINAL),
                 '__package__': 'inter_hierarchy_MN40.hier_proxy_cycle_v9c'}
    exec(compile(source, str(ORIGINAL) + '[native-cb32-h0]', 'exec'), namespace)
    from inter_hierarchy_MN40.hier_proxy_scratch_v9 import base_protocol, geometry, telemetry

    class TrainingForward(base_protocol.HyCoReTrainingForward):
        def __init__(self, backbone, hier_enabled=False):
            if hier_enabled:
                raise ValueError('HIER is prohibited in this baseline')
            super().__init__(backbone)

    class Telemetry(telemetry.EpochTelemetry):
        def __init__(self, train_ids, train_labels, num_classes=40, hier_enabled=False):
            if hier_enabled:
                raise ValueError('HIER is prohibited in baseline telemetry')
            super().__init__(train_ids, train_labels, num_classes=num_classes)

        def summary(self):
            result = super().summary()
            result['geometry'].pop('hier_view_domain', None)
            return result

    def base_losses(local_outputs, local_labels):
        return base_protocol.global_base_losses(local_outputs, local_labels, expected_global_batch=32)

    original_arguments = namespace['arguments']

    def fresh_arguments(argv=None):
        args = original_arguments(argv)
        if args.resume is not None:
            raise ValueError('This launch starts fresh; no checkpoint input')
        return args

    namespace.update(arguments=fresh_arguments,
        fixed_configuration=lambda args: configuration(args, namespace['parent_configuration'](args)),
        HyCoReTrainingForward=TrainingForward, global_base_losses=base_losses,
        configure_backbone=geometry.configure_backbone, freeze_curvature=geometry.freeze_curvature,
        EpochTelemetry=Telemetry, EXPECTED_ORIGINAL_INITIAL_DIGEST=INITIAL_DIGEST)
    return namespace


if __name__ == '__main__':
    build_namespace()['main']()
