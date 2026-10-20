"""One authorized half-margin run using the established idle-GPU/smoke manager."""
import json
import os
from pathlib import Path
import subprocess
import sys
from ..hycore_c01_ablation_20261020 import queue as manager

VARIANT = 'half_margin'
NAME = 'ORIG-B0-32-C01-CAP23-MHALF-S4780'


def command(variant, data, directory, smoke=False):
    if variant != VARIANT:
        raise ValueError('Only the authorized half-margin arm is allowed')
    cmd = [sys.executable, '-m', 'inter_hierarchy_MN40.hycore_c01_halfmargin_20261020.train',
           '--variant', 'c01-cap', '--data-dir', str(data), '--run-dir', str(directory)]
    if smoke:
        cmd += ['--smoke', '--epochs', '2', '--steps-per-epoch', '2']
    return cmd


def main():
    # Private reference paths are provided at launch, never embedded in source.
    reference = Path(os.environ['HYCORE_HALF_REFERENCE'])
    baseline = json.loads((reference / 'manifest.json').read_text())
    config = baseline['training_config']
    if (baseline['status'] != 'completed' or baseline['completed_epochs'] != 300
            or config['radial_gamma'] != 1000 or config['contrastive_margin'] != 4
            or config['c'] != .1 or config['clip_r'] != 2.3
            or config['seed'] != 4780 or config['gradient_hook']):
        raise RuntimeError('Reference is not the completed original cap control')
    root = Path(sys.argv[sys.argv.index('--root')+1])
    cpu_log = root.with_name(root.name + '_half_contract.log')
    with cpu_log.open('x') as stream:
        subprocess.run([sys.executable, '-m', 'unittest',
                        'inter_hierarchy_MN40.hycore_c01_halfmargin_20261020.tests.test_contract', '-v'],
                       check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600)

    def use_identical_data(data_dir, root_name, output):
        data = Path(data_dir)
        if not data.is_dir():
            raise FileNotFoundError(data)
        manager.write(output / 'data_identity.json', {
            'mode': 'reuse_readonly_exact_completed_baseline_cache', 'path': str(data),
            'reference': str(reference), 'expected_train': baseline['dataset_identity']['train']})
        return data

    def verify_smoke(directory, variant, commit):
        m = json.loads((directory / 'manifest.json').read_text())
        c = m['training_config']
        if (m['status'] != 'completed' or m['completed_epochs'] != 2 or m['optimizer_steps'] != 4
                or m['official_test_read'] or m['commit'] != commit or m['canonical_name'] != NAME
                or m['train_size'] != 9840 or m['validation_size'] != 0
                or c['radial_gamma'] != 500 or c['contrastive_margin'] != 2
                or c['c'] != .1 or c['clip_r'] != 2.3 or c['gradient_hook'] or c['seed'] != 4780
                or m['initialized_model_sha256'] != baseline['initialized_model_sha256']
                or m['dataset_identity']['train'] != baseline['dataset_identity']['train']):
            raise RuntimeError('Half-margin smoke/config/initialization/data identity failed')
        import torch
        state = torch.load(directory / 'last_checkpoint.pth', map_location='cpu', weights_only=False)
        if not all(bool(torch.isfinite(t).all()) for t in state['net'].values()):
            raise RuntimeError('Smoke checkpoint is not finite')
        return {'status': 'passed', 'commit': commit, 'canonical_name': NAME,
                'completed_epochs': 2, 'updates': 4, 'official_test_read': False,
                'same_initialization_and_train_identity_as_reference': True,
                'initialized_model_sha256': m['initialized_model_sha256'],
                'baseline_reference': str(reference), 'half_margin_cpu_log': str(cpu_log)}

    manager.VARIANTS = (VARIANT,)
    manager.NAMES = {VARIANT: NAME}
    manager.command = command
    manager.cache_data = use_identical_data
    manager.verify_smoke = verify_smoke
    manager.main()

if __name__ == '__main__':
    main()