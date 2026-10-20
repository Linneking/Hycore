"""Single original GS arm; waits for an idle GPU without touching existing jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from ..hycore_c01_ablation_20261020 import queue as manager

VARIANT='gs_half_margin'
NAME='ORIG-B0-32-C0.1-CAP2.3-GS1-M500_2-TS-S4780'


def command(variant,data,directory,smoke=False):
    if variant!=VARIANT:raise ValueError('Only the authorized original GS arm is allowed')
    result=[sys.executable,'-m','inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020.train',
            '--variant','c01-cap','--data-dir',str(data),'--run-dir',str(directory)]
    if smoke:result+=['--smoke','--epochs','2','--steps-per-epoch','2']
    return result


def main():
    reference=json.loads(Path(os.environ['HYCORE_ORIGINAL_IDENTITY']).read_text())
    expected_train=reference['dataset_identity']['train']
    if reference['initialized_model_sha256']!='8926765a4842652b751873fb89aa37d24d9b0f4f2d831d853b9cc0c1cd46472d':
        raise RuntimeError('Unexpected original seed4780 reference')
    os.environ['HYCORE_TRAIN_SHARD_ORDER']=json.dumps(expected_train['file_order'])
    root=Path(sys.argv[sys.argv.index('--root')+1])
    with root.with_name(root.name+'_gs_contract.log').open('x') as log:
        subprocess.run([sys.executable,'-m','unittest',
            'inter_hierarchy_MN40.hycore_c01_gs_halfmargin_20261020.tests.test_contract',
            'inter_hierarchy_MN40.hycore_c01_halfmargin_20261020.tests.test_contract','-v'],
            check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)

    def readonly_data(data_dir,root_name,output):
        from ..hycore_c01_gs_halfmargin_20261020.train import digest
        data=Path(data_dir)
        # Hash train only before reserving a GPU; test is untouched until production.
        for name,sha in expected_train['file_sha256'].items():
            if digest(data/name)!=sha:raise RuntimeError('Training HDF5 differs from original reference')
        manager.write(output/'data_identity.json',{'mode':'readonly_cache_with_explicit_original_row_order',
            'path':str(data),'expected_train':expected_train,'test_read':False})
        return data

    def verify_smoke(directory,variant,commit):
        m=json.loads((directory/'manifest.json').read_text());c=m['training_config']
        if (m['status']!='completed' or m['completed_epochs']!=2 or m['optimizer_steps']!=4
            or m['official_test_read'] or m['commit']!=commit or m['canonical_name']!=NAME
            or m['train_size']!=9840 or m['validation_size']!=0 or c['radial_gamma']!=500
            or c['contrastive_margin']!=2 or c['c']!=.1 or c['clip_r']!=2.3
            or not c['gradient_hook'] or c['seed']!=4780
            or m['initialized_model_sha256']!=reference['initialized_model_sha256']
            or m['dataset_identity']['train']!=expected_train):
            raise RuntimeError('Original GS smoke identity/config mismatch')
        import torch
        state=torch.load(directory/'last_checkpoint.pth',map_location='cpu',weights_only=False)
        if not all(bool(torch.isfinite(t).all()) for t in state['net'].values()):
            raise RuntimeError('Nonfinite smoke checkpoint')
        return {'status':'passed','canonical_name':NAME,'source_commit':commit,'train_only_updates':4,
                'official_test_read':False,'same_initialization_and_training_identity_as_original_cap_baseline':True}

    manager.VARIANTS=(VARIANT,);manager.NAMES={VARIANT:NAME}
    manager.command=command;manager.cache_data=readonly_data;manager.verify_smoke=verify_smoke
    manager.main()

if __name__=='__main__':main()