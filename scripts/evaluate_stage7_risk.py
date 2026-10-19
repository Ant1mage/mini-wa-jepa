"""Frozen risk evaluation, class coverage, calibration and paired world ablation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from configs.stage7 import Stage7Config
from dataset.risk_features import RiskFeatures
from evaluation.risk import binary_metrics, paired_seed_interval
from mini_wa_jepa.provenance import provenance, write_json
from models.risk_head import ConsequenceRisk
from scripts.train_stage7_risk import predict
from training.flow_matching import file_sha256


def evaluate(data_root,runs,output):
    cfg=Stage7Config();torch.set_num_threads(4);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    data=RiskFeatures(data_root,'test');y=np.asarray(data.arrays['risk']);valid=np.asarray(data.arrays['risk_mask'])
    seeds=np.asarray(data.arrays['seed']);report=dict(**provenance(cfg),stage=7,completed=False,models={},paired={},
        cache_manifest_sha256=file_sha256(Path(data_root)/'manifest.json'),scientific_gates={})
    probabilities={};initials={}
    for seed in cfg.training_seeds:
        for kind in ('world','zero'):
            name=f'{kind}_{seed}';root=Path(runs)/name;run=json.loads((root/'report.json').read_text())
            if not run['completed'] or run['smoke'] or len(run['epochs'])!=cfg.risk_epochs:
                raise ValueError(f'Incomplete training {name}')
            saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
            model=ConsequenceRisk(**saved['arch']).to('cuda');model.load_state_dict(saved['model'])
            _,p=predict(model,data);probabilities[name]=p;initials[name]=saved['initial_hash']
            calibration=json.loads((root/'validation.json').read_text());metrics={}
            for k,outcome in enumerate(('crash','offroad')):
                mask=valid[:,k];metrics[outcome]=binary_metrics(y[mask,k],p[mask,k],
                    calibration['thresholds'][k],calibration['train_prevalence'][k])
                metrics[outcome]['seed_clusters']=len(np.unique(seeds[mask]))
                metrics[outcome]['positive_seed_clusters']=len(np.unique(seeds[mask & (y[:,k]>0)]))
                metrics[outcome]['negative_seed_clusters']=len(np.unique(seeds[mask & (y[:,k]==0)]))
                metrics[outcome]['censored']=int((~mask).sum())
            report['models'][name]=dict(metrics=metrics,epoch=saved['epoch'],
                checkpoint_sha256=file_sha256(root/'best.pt'),validation=calibration)
            np.savez_compressed(out/f'{name}_raw.npz',probability=p,target=y,valid=valid,seeds=seeds,
                groups=data.arrays['group'],commands=data.arrays['command'])
            del model
        if initials[f'world_{seed}']!=initials[f'zero_{seed}']:raise ValueError('Initialization mismatch')
        gates={};pairs={}
        for k,outcome in enumerate(('crash','offroad')):
            mask=valid[:,k];a=probabilities[f'world_{seed}'][mask,k];b=probabilities[f'zero_{seed}'][mask,k]
            target=y[mask,k];pair=paired_seed_interval((a-target)**2,(b-target)**2,seeds[mask],cfg.bootstrap_samples,seed)
            m=report['models'][f'world_{seed}']['metrics'][outcome]
            reference=report['models'][f'zero_{seed}']['metrics'][outcome]
            gain=1-m['brier']/max(reference['brier'],1e-12);pair['relative_brier_gain']=gain;pairs[outcome]=pair
            gates[outcome]=dict(coverage=m['positives']>=20 and m['negatives']>=20 and
                m['positive_seed_clusters']>=4 and m['negative_seed_clusters']>=4,
                discrimination=m['auroc'] is not None and m['auroc']>=cfg.min_auroc and
                    m['average_precision']>m['prevalence'],
                calibrated_baseline=m['brier']<m['constant_train_prevalence_brier'],
                world_gain=gain>=.05 and pair['ci95'] is not None and pair['ci95'][1]<0)
        report['paired'][str(seed)]=pairs;report['scientific_gates'][str(seed)]=gates
    report.update(completed=True,scientific_passed=all(v for s in report['scientific_gates'].values()
        for outcome in s.values() for v in outcome.values()))
    write_json(out/'report.json',report)
    print(json.dumps(dict(scientific_passed=report['scientific_passed'],paired=report['paired'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True)
    p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();evaluate(a.data,a.runs,a.output)
