"""实际轨迹/latent证据；所有多模态流均经公共 RerunLogger。"""
from pathlib import Path
import numpy as np
import torch

from mini_wa_jepa.provenance import write_json


def open_samples(policy, dataset, output, sample_ids, condition, writer, condition_step):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mini_wa_jepa.visualization import RerunLogger, colorize_bev_4ch
    from mini_wa_jepa.geometry import ego_to_pixels
    folder = Path(output) / 'visualizations'; folder.mkdir(parents=True, exist_ok=True)
    selected = [i for i in sample_ids if i < len(dataset)]
    if not selected:
        selected = [0]
    samples = [dataset[i] for i in selected]
    raw = {key: np.stack([s[key] for s in samples]) for key in ('bev', 'ego', 'future')}
    result = policy.batch(raw, include_future=True)
    plans = policy.physical(result['trajectory'])
    arrays = {key: value.float().cpu().numpy() for key, value in result.items()}
    targets = dataset.normalizer.inverse_trajectory(np.stack([s['trajectory'] for s in samples]))
    np.savez_compressed(folder/'predictions.npz', **arrays, prediction_physical=plans,
                        target_physical=targets, samples=np.asarray(selected),
                        anchors=np.asarray([s['anchor'] for s in samples]),
                        observation_sources=np.stack([s['observation_sources'] for s in samples]),
                        world_valid=np.stack([s['world_valid'] for s in samples]))
    evidence = []
    for slot, (sample_id, sample) in enumerate(zip(selected, samples)):
        stem = f'sample_{sample_id:05d}'
        rrd = folder/f'{stem}.rrd'
        world_valid = sample['world_valid']
        with RerunLogger(app_id=f'mini-wa-jepa-stage5-{policy.name}', spawn=False, save_path=rrd) as logger:
            logger.set_step(condition_step, sample['step'] * dataset.dt_s)
            logger.log_context(stage=5, model=policy.name, condition=condition.name, sample_id=sample_id,
                               episode_id=sample['episode'], episode_step=sample['step'],
                               observation_sources=sample['observation_sources'].tolist(),
                               latent_target='same frozen encoder on actual clean future BEV')
            logger.log_bev(sample['bev'][-1]); logger.log_camera(None)
            logger.log_state(dataset.states[sample['anchor']])
            logger.log_trajectories(planned=plans[slot], ground_truth=targets[slot][sample['valid']])
            if policy.name != 'resnet' and world_valid.any():
                logger.log_latents(predicted=arrays['latent'][slot][world_valid],
                                   ground_truth=arrays['target'][slot][world_valid])
        fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.4), dpi=100, layout='constrained')
        axes[0, 0].imshow(colorize_bev_4ch(dataset.data.bev[sample['anchor']]))
        axes[0, 0].set_title('Actual clean BEV'); axes[0, 0].axis('off')
        axes[0, 1].imshow(colorize_bev_4ch(sample['bev'][-1]))
        for trajectory, color, label in [(plans[slot], 'cyan', 'prediction'), (targets[slot], 'orange', 'actual future')]:
            pix = ego_to_pixels(trajectory[:, :2], 128, 64)
            axes[0, 1].plot(pix[:, 0], pix[:, 1], color=color, label=label)
            axes[0, 2].plot(trajectory[:, 0], trajectory[:, 1], color=color, label=label)
        axes[0, 1].set_title('Received BEV + predicted/true trajectory'); axes[0, 1].legend(fontsize=6)
        axes[0, 2].set(xlabel='ego forward m', ylabel='ego left m'); axes[0, 2].legend(fontsize=7)
        if policy.name != 'resnet' and world_valid.any():
            pred = arrays['latent'][slot][world_valid].reshape(-1, policy.config.encoder_dim).T
            target = arrays['target'][slot][world_valid].reshape(-1, policy.config.encoder_dim).T
            bound = max(float(np.quantile(np.abs(np.concatenate([pred, target], 1)), .99)), .01)
            for ax, value, title in zip(axes[1], [pred, target, np.abs(pred-target)],
                                       ['Predicted latent features', 'Actual clean future latent features', 'Absolute latent error']):
                ax.imshow(value, aspect='auto', cmap='coolwarm', vmin=-bound if 'Absolute' not in title else 0, vmax=bound)
                ax.set_title(title, fontsize=9)
        else:
            axes[1, 0].plot(np.arange(1, 11)*.2, plans[slot, :, 2], label='prediction')
            axes[1, 0].plot(np.arange(1, 11)*.2, targets[slot, :, 2], label='actual')
            axes[1, 0].legend(); axes[1, 0].set(xlabel='future seconds', ylabel='speed m/s')
            for ax in axes[1, 1:]:
                ax.text(.1, .5, 'ResNet has no world-prediction head'); ax.axis('off')
        fig.suptitle(f'Stage 5 | {policy.name} | {condition.name} | test sample {sample_id} | actual clean future targets')
        png = folder/f'{stem}.png'; fig.savefig(png); plt.close(fig)
        if writer:
            from PIL import Image
            writer.add_image(f'{policy.name}/{condition.name}/sample_{sample_id}',
                             np.asarray(Image.open(png).convert('RGB')).transpose(2, 0, 1), condition_step)
        evidence.append(dict(sample_id=sample_id, anchor=sample['anchor'], png=str(png), rerun=str(rrd)))
    write_json(folder/'manifest.json', dict(stage=5, condition=condition.name, model=policy.name, samples=evidence))
    return evidence


def closed_recording(output, model_name, condition, map_spec, seed, records, states, final_state, dt):
    from dataset.recording import future_labels_v2
    from evaluation.visuals import VideoWriter, compose
    from mini_wa_jepa.geometry import ego_to_pixels
    from mini_wa_jepa.visualization import RerunLogger
    from scripts.evaluate_joint_closed_loop import delayed_latent_targets
    from PIL import Image, ImageDraw
    output = Path(output); steps = np.asarray([r['step'] for r in records])
    actual = np.asarray(states+[final_state])
    gt, valid, _ = future_labels_v2(actual, np.zeros(len(actual), bool), 10, round(.2/dt))
    latent_pred = latent_target = world_valid = None
    if model_name != 'resnet':
        latent_pred, latent_target, world_valid = delayed_latent_targets(records, dt)
    stem = f'{model_name}_{condition.name}_{map_spec}_{seed}'
    video = VideoWriter(map_spec, seed, stage=f'stage5_{model_name}_{condition.name}', fps=10)
    rrd = output/f'{stem}.rrd'; snapshots=[]
    try:
        with RerunLogger(app_id=f'mini-wa-jepa-stage5-{model_name}', spawn=False, save_path=rrd) as logger:
            for index, row in enumerate(records):
                step = row['step']; logger.set_step(step, step*dt)
                logger.log_context(stage=5, model=model_name, condition=condition.name, map_spec=map_spec,
                                   seed=seed, control_step=step, observation_source_step=row['source_step'],
                                   history_indices=row.get('history_indices'),
                                   latent_target='clean actually observed future of this same policy rollout')
                logger.log_bev(row['bev']); logger.log_camera(row['rgb']); logger.log_state(states[step])
                logger.log_trajectories(planned=row['plan'], ground_truth=gt[step][valid[step]])
                logger.log_telemetry(speed_m_s=states[step][3], steering=row['action'][0],
                                     throttle_brake=row['action'][1], route_completion=row['route'],
                                     lateral_error_m=row['lateral'])
                if world_valid is not None and world_valid[index].any():
                    mask=world_valid[index]
                    logger.log_latents(predicted=latent_pred[index][mask], ground_truth=latent_target[index][mask])
                canvas=Image.fromarray(compose(row['bev'], row['plan'], states[step], row['action'],
                    f'{model_name} {condition.name} {map_spec} seed {seed} step {step} route {row["route"]:.2f}',
                    row['rgb'], caption='cyan model / orange delayed actual trajectory'))
                draw=ImageDraw.Draw(canvas)
                points=(ego_to_pixels(gt[step][valid[step],:2],128,64)+.5)*4+np.array([748,40])
                if len(points)>1:
                    draw.line([tuple(p) for p in points], fill='orange', width=3)
                draw.rectangle((0,580,1280,640),fill=(17,21,29))
                draw.text((20,602),f'Stage 5 {condition.name} | frozen {model_name} | actual delayed truth | offline playback',fill=(230,230,240))
                video.write(np.asarray(canvas))
                if index in (0,len(records)//2,len(records)-1):
                    path=output/f'{stem}_step_{step:05d}.png';canvas.save(path);snapshots.append(str(path))
    finally:
        video.close()
    raw=dict(states=actual, steps=steps, prediction=np.stack([r['plan'] for r in records]),
             target=gt[steps], valid=valid[steps], observation_source_step=np.asarray([r['source_step'] for r in records]))
    if latent_pred is not None:
        raw.update(latent_prediction=latent_pred,latent_target=latent_target,world_valid=world_valid)
    path=output/f'{stem}_actual.npz';np.savez_compressed(path,**raw)
    return dict(video=str(video.path),rerun=str(rrd),raw=str(path),snapshots=snapshots,
                recorded_frames=len(records),valid_latent_targets=int(world_valid.sum()) if world_valid is not None else None)
