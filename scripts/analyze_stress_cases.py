"""仅按输入/原始标签选择诊断样本，不根据哪个模型更好来挑例子。"""
from pathlib import Path
import numpy as np
from configs.stage5 import condition, CHECKPOINTS, Stage5Config
from evaluation.degradation import DegradedSequences
from evaluation.stress import cohort_data
from mini_wa_jepa.provenance import write_json


def analyze(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mini_wa_jepa.visualization import RerunLogger, colorize_bev_4ch
    from mini_wa_jepa.geometry import ego_to_pixels
    data=cohort_data(Stage5Config.data);folder=Path(root)/'edge_cases';folder.mkdir(exist_ok=True)
    anchors=data.anchors;areas=np.count_nonzero(data.bev[:,2],axis=(1,2))
    previous=np.maximum(0,anchors-1)
    same=(data.episodes[previous]==data.episodes[anchors])&(data.steps[anchors]-data.steps[previous]==1)
    new_actor=same&(areas[previous]==0)&(areas[anchors]>0)
    delta=np.where(same,np.abs(areas[anchors]-areas[previous]),-1)
    selectors={'participant_first_appears':int(np.flatnonzero(new_actor)[0]) if new_actor.any() else None,
               'largest_actor_pixel_change':int(np.argmax(delta)) if same.any() else None,
               'first_recovery_label':int(np.flatnonzero(data.recovery[anchors])[0]) if data.recovery[anchors].any() else None,
               'first_noise_label':int(np.flatnonzero(data.noisy[anchors])[0]) if data.noisy[anchors].any() else None}
    reports={};plots=[];cases=('clean','occlusion_l3','boundary_l3','dropout_l3')
    for label,index in selectors.items():
        if index is None:
            reports[label]=dict(found=False);continue
        anchor=int(anchors[index]);fig,axes=plt.subplots(2,2,figsize=(12.8,6.4),layout='constrained')
        rows=[]
        for ax,case in zip(axes.flat,cases):
            sample=DegradedSequences(data,condition(case))[index]
            ax.imshow(colorize_bev_4ch(sample['bev'][-1]))
            gt=np.asarray(data.trajectories[anchor]);pixels=ego_to_pixels(gt[:,:2],128,64)
            ax.plot(pixels[:,0],pixels[:,1],color='orange',label='actual future')
            for model,color in zip(CHECKPOINTS,('cyan','magenta','lime')):
                raw=np.load(Path(root)/'open_loop'/model/case/'raw.npz')
                if int(raw['anchors'][index])!=anchor:raise ValueError('Diagnostic anchor misaligned')
                plan=raw['prediction'][index];pix=ego_to_pixels(plan[:,:2],128,64)
                ax.plot(pix[:,0],pix[:,1],color=color,label=model)
                rows.append(dict(condition=case,model=model,ade_m=float(raw['ade_m'][index]),fde_m=float(raw['fde_m'][index])))
                with RerunLogger(app_id='mini-wa-jepa-stage5-edge',spawn=False,
                                 save_path=folder/f'{label}_{case}_{model}.rrd') as logger:
                    logger.set_step(int(data.steps[anchor]),float(data.steps[anchor])*.1)
                    logger.log_context(stage=5,selection=label,model=model,condition=case,anchor=anchor,
                                       episode_id=int(data.episodes[anchor]),selection_uses_model_errors=False)
                    logger.log_bev(sample['bev'][-1]);logger.log_camera(None);logger.log_state(data.states[anchor])
                    logger.log_trajectories(planned=plan,ground_truth=gt)
            ax.set_title(case);ax.axis('off');ax.legend(fontsize=6)
        fig.suptitle(f'{label} | actual input-selected test anchor {anchor} | fixed cohort, not a new success-rate estimate')
        path=folder/f'{label}.png';fig.savefig(path,dpi=100);plt.close(fig);plots.append(path)
        reports[label]=dict(found=True,anchor=anchor,episode=int(data.episodes[anchor]),step=int(data.steps[anchor]),
                            noisy=bool(data.noisy[anchor]),recovery=bool(data.recovery[anchor]),
                            actor_pixels=int(areas[anchor]),previous_actor_pixels=int(areas[previous[index]]),metrics=rows,figure=str(path))
    write_json(folder/'cases.json',dict(selection='data-only rules, no model-error-based cherry picking',cases=reports,
               limitation='Participant entry can be field-of-view entry; recovery/noise are original dataset labels, not new closed-loop control interventions.'))
    return reports,plots
