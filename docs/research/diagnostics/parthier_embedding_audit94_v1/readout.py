"""Analyze four validated frozen exports without rebuilding embeddings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback

from .collect import idle_gpu, now, sha, write
from .pipeline import MODULE, Pipeline, checked_export_identity


class ReadoutPipeline(Pipeline):
    def run(self):
        cache = Path(self.settings['reuse_cache']).resolve()
        self.update('export_gates')
        identity, gates = checked_export_identity(cache)
        for label in gates:
            manifest = json.loads((cache / (label + '_manifest.json')).read_text())
            if manifest.get('tf32') is not False:
                raise ValueError('Only strict FP32 validated exports can be reused')
        if json.loads((cache / 'input_identity.json').read_text()) != identity:
            raise ValueError('Common input identity changed')
        # Cache and previous result directory remain strictly read-only.
        write(self.root / 'export_acceptance.json', gates)
        import fcntl
        gpu = self.gpus[0]
        with open('/tmp/hycore_%d_gpu_%d.lock' % (os.getuid(), gpu), 'a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            snapshot = subprocess.check_output(['nvidia-smi'], text=True)
            (self.root / 'metrics_nvidia_smi.txt').write_text(snapshot)
            uuid = idle_gpu(gpu)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
            analysis = self.root / 'analysis'
            if analysis.exists():
                raise FileExistsError('Analysis directory already exists')
            self.update('metrics', metrics_gpu=gpu, metrics_gpu_uuid=uuid)
            child, record = self.spawn('metrics', [MODULE + '.metrics', '--cache-dir', str(cache),
                '--out-dir', str(analysis), '--device', 'cuda:0'], env=env, gpu=gpu)
            self.wait_one(child, record)
        self.update('report')
        public = self.root / 'public'
        if public.exists():
            raise FileExistsError('Publication directory already exists')
        child, record = self.spawn('report', [MODULE + '.report', '--analysis-dir', str(analysis),
                                            '--out-dir', str(public)])
        self.wait_one(child, record)
        publication = json.loads((public / 'report_acceptance.json').read_text())
        if publication['figure_groups'] != 5 or len(publication['figure_files']) != 10:
            raise ValueError('Incomplete figure delivery')
        write(self.root / 'pipeline_acceptance.json', {
            'status': 'completed', 'ended_utc': now(), 'labels': gates,
            'shared_input': identity, 'reused_validated_exports': True,
            'metrics_summary_sha256': sha(analysis / 'summary.json'),
            'report_acceptance': publication, 'training_updates': 0})
        self.update('completed', status='completed', ended_utc=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    args = parser.parse_args()
    pipeline = None
    try:
        pipeline = ReadoutPipeline(args.settings)
        pipeline.run()
    except Exception as error:
        if pipeline is not None:
            pipeline.update(status='failed', error_type=type(error).__name__,
                            error=str(error), ended_utc=now())
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
