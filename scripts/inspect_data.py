"""Validate the v2 contract, rescue masks and train-only normalization."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from dataset.storage import ShardReader
from dataset.baseline import validate_manifest
from mini_wa_jepa.geometry import ego_to_world,wrap_angle
from mini_wa_jepa.normalization import Normalizer
from mini_wa_jepa.provenance import provenance,write_json


def inspect(root,output):
    manifest=validate_manifest(root);summary=dict(**provenance(manifest['config']),schema_version=2,
        total_frames=0,splits={},normalization_fit_split='train',dataset_source=manifest['source_tree_sha256'])
    maps={};seeds={};fit=[];train_speed=[]
    for split in ('train','val','test'):
        reader=ShardReader(Path(root)/split,cache_shards=1)
        names=['state','next_state','episode_id','map_id','seed','step_id','action_mask','terminal_failure',
               'transition_valid','future_trajectory','future_valid','noise_active','recovery_active','lateral_m']
        fields=reader.metadata_many(names)
        trajectory=fields['future_trajectory'];valid=fields['future_valid'];action=fields['action_mask']
        assert trajectory.shape==(len(reader),10,4) and valid.shape==(len(reader),10)
        assert np.isfinite(trajectory).all()
        np.testing.assert_array_equal(action,~fields['terminal_failure'] & fields['transition_valid'])
        # Observe at t+0.2, not t+0.1, and compare directly with recorded world states.
        indices=np.arange(len(reader)-2)
        selected=indices[(fields['episode_id'][:-2]==fields['episode_id'][2:]) & valid[:-2,0]]
        local=trajectory[selected,0,:2];heading=fields['state'][selected,2]
        c,s=np.cos(heading),np.sin(heading)
        xy=np.stack([local[:,0]*c-local[:,1]*s,local[:,0]*s+local[:,1]*c],axis=-1)+fields['state'][selected,:2]
        np.testing.assert_allclose(xy,fields['state'][selected+2,:2],atol=3e-4)
        np.testing.assert_allclose(trajectory[selected,0,2],fields['state'][selected+2,3],atol=1e-5)
        np.testing.assert_allclose(trajectory[selected,0,3],wrap_angle(fields['state'][selected+2,2]-fields['state'][selected,2]),atol=1e-5)
        channel_counts=np.zeros(4,np.int64)
        for shard_id in range(len(reader.ends)):
            bev=reader.shard(shard_id)['bev']
            assert bev.dtype==np.uint8 and bev.shape[1:]==(4,128,128)
            assert np.all((bev==0)|(bev==255)) and np.all(bev[:,3]==255)
            channel_counts+=(bev>0).sum((0,2,3))
        maps[split]=set(fields['map_id']);seeds[split]=set(fields['seed'].tolist())
        stats=dict(frames=len(reader),episodes=len(set(fields['episode_id'].tolist())),
            map_hashes=sorted(maps[split]),seeds=sorted(seeds[split]),action_frames=int(action.sum()),
            terminal_failure_frames=int(fields['terminal_failure'].sum()),valid_waypoints=int(valid.sum()),
            noise_frames=int(fields['noise_active'].sum()),recovery_frames=int(fields['recovery_active'].sum()),
            noisy_action_frames=int((fields['noise_active']&action).sum()),
            recovery_action_frames=int((fields['recovery_active']&action).sum()),
            channel_positive_pixels=channel_counts.tolist(),
            lateral_m_quantiles=np.percentile(fields['lateral_m'],[0,5,50,95,100]).tolist(),
            trajectory_quantiles={name:np.percentile(trajectory[:,:,i][valid],[0,5,50,95,100]).tolist()
                                  for i,name in enumerate(['x_m','y_m','speed_m_s','heading_rad'])})
        summary['splits'][split]=stats;summary['total_frames']+=len(reader)
        if split=='train':
            fit=trajectory[valid & action[:,None]]
            train_speed=fields['state'][action,3]
    for left,right in [('train','val'),('train','test'),('val','test')]:
        assert not maps[left]&maps[right],f'Geometry leakage: {left}/{right}'
        assert not seeds[left]&seeds[right],f'Seed leakage: {left}/{right}'
    assert summary['total_frames']==manifest['stored_frames']
    normalizer=Normalizer(1.,fit.mean(0),np.maximum(fit.std(0),[.5,.1,.1,.02]),
                          float(train_speed.mean()),max(float(train_speed.std()),.1))
    write_json(Path(root)/'normalization.json',normalizer.to_dict())
    summary['normalization']=normalizer.to_dict()
    summary['storage_bytes']=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
    write_json(output,summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(14,3))
    for column,name in enumerate(['forward (m)','left (m)','speed (m/s)','heading (rad)']):
        axes[column].hist(fit[:,column],bins=50);axes[column].set_xlabel(name)
    fig.tight_layout();fig.savefig(Path(output).with_suffix('.png'),dpi=130);plt.close(fig)
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('outputs/data/stage2_v2_100k'))
    p.add_argument('--output',type=Path,default=Path('artifacts/restart_stage2/data_stats.json'))
    args=p.parse_args();inspect(args.data,args.output)


if __name__=='__main__':main()
