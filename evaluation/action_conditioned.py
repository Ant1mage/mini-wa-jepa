"""Same-context causal consequence metrics; no learned selector is evaluated."""
import numpy as np


def cluster_interval(numerator_a,numerator_b,denominator,seeds,samples=2000,seed=20261019):
    a,b,n,seeds=map(np.asarray,(numerator_a,numerator_b,denominator,seeds))
    unique=np.unique(seeds)
    stats=np.array([[a[seeds==s].sum(),b[seeds==s].sum(),n[seeds==s].sum()] for s in unique])
    rng=np.random.default_rng(seed)
    totals=stats[rng.integers(len(unique),size=(samples,len(unique)))].sum(1)
    delta=(totals[:,0]-totals[:,1])/np.maximum(totals[:,2],1)
    ratio=totals[:,0]/np.maximum(totals[:,1],1e-15)
    return dict(seed_clusters=len(unique),bootstrap_samples=samples,
                delta_ci95=np.percentile(delta,[2.5,97.5]).tolist(),ratio_ci95=np.percentile(ratio,[2.5,97.5]).tolist())


def consequence_metrics(prediction,target,valid,groups,seeds):
    errors=np.square(prediction-target).mean((-1,-2))
    if not np.isfinite(errors).all():raise ValueError('Nonfinite predictions')
    confusion=np.zeros((4,4),np.int64);correct=total=ties=0
    pair_errors=[];target_separation=[];prediction_separation=[];pair_seeds=[];pair_groups=[]
    for group in np.unique(groups):
        idx=np.flatnonzero(groups==group)
        if len(idx)!=4:raise ValueError('Every context must retain all four candidates')
        common=valid[idx].all(0)
        if common.any():
            p=prediction[idx][:,common];t=target[idx][:,common]
            costs=np.square(p[:,None]-t[None]).mean((2,3,4))
            for i in range(4):
                order=np.argsort(costs[i]);total+=1
                if np.isclose(costs[i,order[0]],costs[i,order[1]],atol=1e-8,rtol=0):ties+=1
                else:
                    confusion[i,order[0]]+=1;correct+=int(order[0]==i)
        for i in range(4):
            for j in range(i):
                mask=valid[idx[i]]&valid[idx[j]]
                if not mask.any():continue
                pd=prediction[idx[i],mask]-prediction[idx[j],mask]
                td=target[idx[i],mask]-target[idx[j],mask]
                pair_errors.append(float(np.square(pd-td).mean()));target_separation.append(float(np.square(td).mean()))
                prediction_separation.append(float(np.square(pd).mean()));pair_seeds.append(seeds[idx[i]]);pair_groups.append(group)
    metrics=dict(latent_mse=float(errors[valid].mean()),valid_future_count=int(valid.sum()),
        per_horizon_mse=[float(errors[:,h][valid[:,h]].mean()) if valid[:,h].any() else None for h in range(4)],
        delta_mse=float(np.mean(pair_errors)),target_delta_mse=float(np.mean(target_separation)),
        predicted_delta_mse=float(np.mean(prediction_separation)),retrieval_accuracy=correct/max(total,1),
        retrieval_chance=.25,retrieval_eligible=total,retrieval_ties=ties,retrieval_confusion=confusion.tolist())
    return metrics,dict(error=errors,valid=valid,pair_error=np.asarray(pair_errors),
        target_pair_energy=np.asarray(target_separation),pair_seeds=np.asarray(pair_seeds),pair_groups=np.asarray(pair_groups))
