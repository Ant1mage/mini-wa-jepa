"""Constant-position features must have zero content covariance/rank."""
import json
import torch
from evaluation.latent import LatentMoments,spectrum_metrics
torch.set_num_threads(2);torch.manual_seed(3)
z=torch.randn(1,16,8).repeat(32,1,1).double()
position_sum=z.sum(0);flat=z.flatten(0,1)
naive=(flat.T@flat-position_sum.T@position_sum/len(z))/(16*(len(z)-1))
naive_health,_=spectrum_metrics(naive.numpy())
stable=LatentMoments(16,8,'cpu');stable.update(z);stable_health,_=stable.finish()
print(json.dumps(dict(seed=3,shape=list(z.shape),expected_content_rank=0,
 naive_effective_rank=naive_health['effective_rank'],naive_covariance_trace=naive_health['covariance_trace'],
 stable_effective_rank=stable_health['effective_rank'],stable_covariance_trace=stable_health['covariance_trace']),indent=2))
assert stable_health['effective_rank']==0
