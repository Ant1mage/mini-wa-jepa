"""Read-only validation diagnostics, not a new training run or test selection."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from dataset.flow_features import FlowFeaturesDataset
from models.flow_matcher import ConditionalFlowMatcher
from mini_wa_jepa.provenance import provenance,write_json
from training.flow_matching import file_sha256
root=Path('artifacts/restart_stage7');out=root/'diagnosis_001';torch.set_num_threads(4)
data=FlowFeaturesDataset(root/'flow_features_001','val')
indices=np.random.default_rng(777).choice(len(data),64,replace=False)
items=[data[int(i)] for i in indices]
history=torch.stack([x['history'] for x in items]).to('cuda');ego=torch.stack([x['ego'] for x in items]).to('cuda')
targets=np.stack([x['target'].numpy() for x in items]);target_si=data.normalizer.inverse_trajectory(targets)
report=dict(**provenance(),split='val',samples=64,validation_indices=indices.tolist(),noise_seed=777,
 note='Inference-only diagnostic; no weights, selection rules, or formal results changed',seeds={})
arrays=dict(indices=indices,targets=target_si,ego=ego.cpu().numpy())
for seed in (20261021,20261022):
 saved=torch.load(root/f'flow_{seed}/best.pt',map_location='cpu',weights_only=False)
 arch={k:v for k,v in saved['model_arch'].items() if k!='time_emb_dim'}
 model=ConditionalFlowMatcher(**arch).to('cuda').eval();model.load_state_dict(saved['model'])
 row=dict(checkpoint_sha256=file_sha256(root/f'flow_{seed}/best.pt'),runs={})
 with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
  condition=model.extract_condition(history,ego)
  generator=torch.Generator(device='cuda').manual_seed(777)
  noise=torch.randn(64*8,10,4,generator=generator,device='cuda').reshape(64,8,10,4).cpu().numpy()
  for steps,kind in [(10,'correct'),(50,'correct'),(100,'correct'),(10,'permuted')]:
   c=condition if kind=='correct' else condition.roll(1,0)
   torch.cuda.synchronize();started=time.perf_counter()
   norm=model.sample(c,num_steps=steps,num_candidates=8,seed=777).cpu().numpy()
   elapsed=time.perf_counter()-started;actual=data.normalizer.inverse_trajectory(norm)
   error=np.linalg.norm(actual[...,:2]-target_si[:,None,:,:2],axis=-1)
   delta=np.diff(actual[...,:2],axis=-2);length=np.linalg.norm(delta,axis=-1)
   heading=np.arctan2(delta[...,1],delta[...,0]);herror=np.abs((heading-actual[...,1:,3]+np.pi)%(2*np.pi)-np.pi)
   moving=length>.1
   v=actual[...,2];good=np.isfinite(actual).all(-1)&(v>=0)&(v<=15)
   good[...,1:]&=np.abs(np.diff(v,axis=-1))/.2<=8
   good[...,1:]&=(~moving)|(herror<=.35)
   ratios=[];correlations=[]
   for channel in range(4):
    a=norm[...,channel]-norm[...,channel].mean(1,keepdims=True)
    b=noise[...,channel]-noise[...,channel].mean(1,keepdims=True)
    ratios.append(float((a*a).mean()/max((b*b).mean(),1e-12)))
    correlations.append(float((a*b).sum()/max(np.sqrt((a*a).sum()*(b*b).sum()),1e-12)))
   result=dict(sample1_ade_m=float(error[:,0].mean()),sample1_fde_m=float(error[:,0,-1].mean()),
    pointwise_valid=float(good.mean()),whole_candidate_valid=float(good.all(-1).mean()),
    backwards_segment_fraction=float((delta[...,0]<0).mean()),
    moving_heading_mismatch=float((herror[moving]>.35).mean()),
    conditional_variance_ratio_to_noise_xyvh=ratios,output_input_noise_correlation_xyvh=correlations,
    seconds=elapsed)
   name=f'{kind}_{steps}';row['runs'][name]=result;arrays[f'{seed}_{name}']=actual
 row['validation_baseline_ade_m']=float(np.linalg.norm(data.normalizer.inverse_trajectory(
  np.stack([x['mse_prediction'].numpy() for x in items]))[...,:2]-target_si[...,:2],axis=-1).mean())
 report['seeds'][str(seed)]=row;write_json(out/'report.json',report)
np.savez_compressed(out/'raw.npz',**arrays)
print(json.dumps(report['seeds'],indent=2))
