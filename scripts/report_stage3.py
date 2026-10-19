"""Generate a measured stage 3 report, health plots and acceptance video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from mini_wa_jepa.provenance import provenance, write_json


def create_video(images, output):
    import imageio_ffmpeg
    from PIL import Image
    if output.exists():
        raise FileExistsError(output)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x640', '-r', '2', '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for path in images:
            rgb = Image.open(path).convert('RGB').resize((1280, 640))
            process.stdin.write(np.asarray(rgb).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('Stage 3 video encoder failed')
    return dict(path=str(output), frames=len(images), fps=2, duration_s=len(images)/2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/restart_stage3'))
    p.add_argument('--run', default='vicreg_001')
    args = p.parse_args(); root = args.root; run = root / args.run
    for name in ('REPORT.md', 'metrics.json', 'stage3_acceptance.mp4'):
        if (root/name).exists():
            raise FileExistsError(f'Preserve the existing report: {root/name}')
    report = json.loads((run/'report.json').read_text()); cfg = report['config']
    if report.get('max_batches') is not None:
        raise ValueError('Smoke runs cannot be promoted to formal acceptance')
    if not report['epochs']:
        raise ValueError('No measured epochs to report')
    completed = bool(report.get('completed'))
    if not completed and not report.get('error'):
        raise ValueError('The formal run is still active; do not freeze a partial acceptance report')
    test = report.get('test')
    gates = report.get('gates', dict(full_run=False))
    primary_fields = ['effective_rank_ratio', 'pooled_effective_rank_ratio', 'std_mean', 'std_p10',
                      'latent_mse', 'copy_last_mse', 'prediction_copy_ratio', 'normalized_latent_mse']
    manifest = dict(visualization_schema_version=1, spec_version=2, stage=3,
        **{key:report[key] for key in ('config', 'git_commit', 'goal_sha256', 'source_tree_sha256')},
        seeds=[cfg['seed']], runs=[], rank_definition=report['rank_definition'])
    for epoch, metrics in [(0, report['untrained_validation'])] + [(r['epoch'], r['validation']) for r in report['epochs']]:
        manifest['runs'].append(dict(run_id=f'vicreg-val-epoch-{epoch:03d}', model='ConvStem+Transformer/VICReg',
            ablation='untrained reference' if epoch == 0 else 'single-step deterministic latent prediction',
            split='val', seed=cfg['seed'], metrics={key:metrics[key] for key in primary_fields}))
    if test:
        manifest['runs'].extend([
            dict(run_id='vicreg-test-selected', model='ConvStem+Transformer/VICReg',
                ablation='validation-selected checkpoint; single step +0.1s', split='test', seed=cfg['seed'],
                metrics={key:test[key] for key in primary_fields}),
            dict(run_id='copy-current-test', model='Copy-current encoded tokens',
                ablation='same frozen encoder, target, and test pairs', split='test', seed=cfg['seed'],
                metrics=dict(latent_mse=test['copy_last_mse'], copy_last_mse=test['copy_last_mse']))])
    write_json(root/'metrics.json', manifest)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['epochs']; epoch = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, label in [('effective_rank_ratio', 'content covariance'), ('pooled_effective_rank_ratio', 'frame pooled')]:
        axes[0].plot(epoch, [r['validation'][key] for r in rows], marker='.', label=label)
    axes[0].axhline(cfg['rank_threshold'], color='red', linestyle='--', label='rank gate .75')
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel('effective rank / 256'); axes[0].legend(fontsize=8)
    for key in ['std_mean', 'std_p10']:
        axes[1].plot(epoch, [r['validation'][key] for r in rows], marker='.', label=key)
    axes[1].axhline(cfg['variance_gamma'], color='gray', linestyle='--', label='training gamma')
    axes[1].set_ylabel('content channel standard deviation'); axes[1].legend(fontsize=8)
    axes[2].plot(epoch, [r['validation']['prediction_copy_ratio'] for r in rows], marker='.', label='prediction / copy')
    axes[2].axhline(1, color='gray', linestyle='--', label='copy-current')
    axes[2].axhline(1-cfg['minimum_prediction_gain'], color='red', linestyle='--', label='5% gain gate')
    axes[2].set_ylabel('validation MSE ratio'); axes[2].legend(fontsize=8)
    for ax in axes: ax.set_xlabel('epoch')
    fig.tight_layout(); fig.savefig(root/'health_curves.png', dpi=140); plt.close(fig)
    selected = report.get('selected_epoch', rows[-1]['epoch'])
    selected_row = next(r for r in rows if r['epoch'] == selected)
    val = selected_row['validation']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metrics, label in [(report['untrained_validation'], 'untrained'), (val, f'selected epoch {selected}')]:
        axes[0].semilogy(np.arange(1, 257), np.maximum(metrics['eigenvalues'], 1e-12), label=label)
    axes[0].set_xlabel('covariance eigenvalue index'); axes[0].set_ylabel('content eigenvalue (log view)'); axes[0].legend()
    axes[1].hist(val['channel_std'], bins=24); axes[1].axvline(1, color='red', linestyle='--')
    axes[1].set_xlabel('content channel standard deviation'); axes[1].set_ylabel('channels')
    fig.tight_layout(); fig.savefig(root/'covariance_spectrum.png', dpi=140); plt.close(fig)
    selected_step = selected_row['global_step']
    samples_path = run/'samples'/f'step_{selected_step:06d}'/'latents.npz'
    if samples_path.exists():
        samples = np.load(samples_path); statistics = np.load(run/f'validation_epoch_{selected:03d}.npz')
        mean, basis = statistics['position_mean'], statistics['pca_basis'][:, :2]
        fig, axes = plt.subplots(1, len(samples['sample_ids']), figsize=(14, 3.5), squeeze=False)
        for i, index in enumerate(samples['sample_ids']):
            for key, marker, label in [('target', '.', 'target'), ('prediction', 'x', 'prediction')]:
                projected = (samples[key][i] - mean) @ basis
                axes[0, i].scatter(projected[:, 0], projected[:, 1], s=12, alpha=.6, marker=marker, label=label)
            axes[0, i].set_title(f'val sample {index}'); axes[0, i].set_xlabel('content PC1'); axes[0, i].set_ylabel('content PC2')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle('Actual encoder targets and predictions in the same validation-fitted PCA basis; no image decoder')
        fig.tight_layout(); fig.savefig(root/'latent_pca.png', dpi=140); plt.close(fig)
    video_images = [path for visual in report['visualizations']
        if visual['epoch'] == 0 or visual['step'] in {r['global_step'] for r in rows}
        for path in visual['paths']]
    diagnostics = run/'selected_test_diagnostics/manifest.json'
    if diagnostics.exists():
        video_images.extend(json.loads(diagnostics.read_text())['paths'])
    video = create_video(video_images, root/'stage3_acceptance.mp4')
    acceptance = dict(completed=completed, gates=gates, selected_epoch=selected,
        selected_validation=val, test=test, encoder_parameters=report['encoder_parameters'],
        predictor_parameters=report['predictor_parameters'], parameter_count=report['parameter_count'],
        peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows), video=video,
        source_report=str(run/'report.json'),
        source_report_sha256=hashlib.sha256((run/'report.json').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(Path(report['selected_checkpoint']).read_bytes()).hexdigest()
            if report.get('selected_checkpoint') else None,
        report_provenance=provenance())
    write_json(root/'acceptance.json', acceptance)
    lines = ['# 阶段 3：确定性单步潜变量预测与 VICReg','',
        f"正式运行：`{args.run}`；已执行 {len(rows)}/{cfg['epochs']} epochs；运行{'完成' if completed else '异常停止'}。",
        '**阶段 4 未启动。验收指标来自原始潜变量，不在白化/PCA 后计算。**','',
        f"编码器 {report['encoder_parameters']:,} 参数，预测器 {report['predictor_parameters']:,}，总计 {report['parameter_count']:,}。",
        f"训练相邻对 {report['train_pairs']:,}；固定验证对 {report['validation_pairs']:,}，预测时间间隔 {report['horizon_s']:.1f}s。",
        f"按预定验证规则选择 epoch {selected}；训练峰值分配显存 {acceptance['peak_allocated_gib']:.3f} GiB。",'',
        '## 验收结论','', *[f"- {'PASS' if value else 'FAIL'}: {key}" for key,value in gates.items()], '',
        '只有门槛全部满足且用户确认后才能进入下一阶段；报告完成不等同于门槛通过。','',
        '| 指标 | 选中验证 checkpoint | 完整测试 |','|---|---:|---:|']
    for label,key in [('内容协方差有效秩 / 256','effective_rank_ratio'),('整帧池化有效秩 / 256','pooled_effective_rank_ratio'),
                      ('通道标准差均值','std_mean'),('通道标准差 P10','std_p10'),('预测 MSE','latent_mse'),
                      ('复制当前帧 MSE','copy_last_mse'),('预测/复制 MSE','prediction_copy_ratio')]:
        test_value = f"{test[key]:.6f}" if test else '未产出'
        lines.append(f"| {label} | {val[key]:.6f} | {test_value} |")
    if test:
        ci = test['bootstrap']
        lines.extend(['', f"测试相邻对 {test['frames']:,}；episode 数 {ci['episode_count']}。",
            f"按 episode 成簇 bootstrap 的配对 MSE 差 95% 区间：{ci['delta_ci95']}；比值区间：{ci['ratio_ci95']}。"])
        raw = np.load(run/'test_raw.npz')
        lines.extend(['','| 测试子集 | 帧对数 | 预测 MSE | 复制 MSE |','|---|---:|---:|---:|'])
        for label,mask in [('全部',np.ones(len(raw['anchors']),bool)),('扰动',raw['noisy']),('恢复',raw['recovery'])]:
            if mask.any():
                lines.append(f"| {label} | {int(mask.sum())} | {raw['prediction_mse'][mask].mean():.6f} | {raw['copy_mse'][mask].mean():.6f} |")
        data_manifest = json.loads((Path(report['dataset'])/'manifest.json').read_text())
        for map_spec in sorted({row['map_spec'] for row in data_manifest['episodes'] if row['split']=='test'}):
            ids = [row['episode_id'] for row in data_manifest['episodes'] if row['split']=='test' and row['map_spec']==map_spec]
            mask = np.isin(raw['episodes'],ids)
            lines.append(f"| 地图 {map_spec} | {int(mask.sum())} | {raw['prediction_mse'][mask].mean():.6f} | {raw['copy_mse'][mask].mean():.6f} |")
    if report.get('error'):
        lines.extend(['','## 异常现场','', '```text', report['error'], '```',
            '失败 batch、模型与配置见运行目录 failure_mre.pt；不自动重训或进入下一阶段。'])
    lines.extend(['','## 口径、限制与可复现性','',
        '- 六层 d=256/four-head 编码器；两层 stride-2 Conv 后为 32×32，再以 4×4 patch 投影成 8×8 token；没有缩小输入 BEV 或使用预训练权重。',
        '- 直接对共享编码器的真实 token 做 VICReg，两分支都有梯度；不使用 EMA/额外投影头。预测器只看到当前 BEV、当前速度与从过去计算的角速度。',
        '- 主有效秩来自先按固定空间位置扣除全验证/测试集均值的协方差谱，理论通道上限 256；另报未校正 token rank 和帧池化 rank。浮点稳定性使用 float64 Welford 累积，未通过加噪声或白化抬高秩。',
        '- 256 是协方差矩阵的代数通道维度上限，不是独立 episode 数。token/相邻帧存在相关性，不将池化后的低秩隐藏为“全局表示健康”。',
        '- 严格使用 v2 的同 episode、连续 step 对；扰动/恢复/终止目标均不按动作掩码过滤。未跨 reset、未将未来 ego state 或专家导航作为输入。',
        '- 固定 4096 对验证用于选择 checkpoint；测试不用于调参。管线冒烟的 3-batch 评测单独保留，不计入正式得分。',
        '- 预测目标在训练中可共同变化，因此必须同时审查原始 copy MSE、标准差与有效秩；低 MSE 本身不能证明世界模型有效。',
        '- 95% 区间按 episode 重采样，不把相邻帧当成独立观测；单次 seed 实验仍有局限。主秩门槛须在末三次验证及选中测试 checkpoint 上均满足。',
        '- 周期图/Rerun 中的全验证秩和 PCA 参考来自最近一次完成的验证；图中的预测/目标是该 optimizer step 的真实输出。逐 epoch 曲线和最终 PCA 使用对应 checkpoint 的验证统计。',
        '- 潜变量热图/PCA 是特征可视化，不是预测未来像素。阶段 3 无轨迹头，ADE/FDE、路线完成率、碰撞率和闭环推理延迟均未产出。',
        '- 没有运行旧同进程 Panda3D/Pygame 窗口；阶段 2 的键盘验证与 Rerun 截图退出 panic 原样保留。本阶段离线训练不生成虚构的第三人称 RGB。',
        f"- 方法、预先固定的阈值及权重见 [协议]({os.path.relpath('docs/STAGE3_PROTOCOL.md', root)})；正式配置与源码快照保存在 artifacts/restart_stage3/formal_protocol.json。",'',
        '## 证据','',
        f'- [完整训练/验证/测试记录]({args.run}/report.json) / [逐步日志]({args.run}.log) / [TensorBoard]({args.run}/tensorboard)',
        f'- [完整测试原始误差与协方差]({args.run}/test_raw.npz) / [实际潜变量样本与 Rerun]({args.run}/samples)',
        f'- [固定测试地图样本与实际预测/目标]({args.run}/selected_test_diagnostics/manifest.json) / [帧对/episode/seed 清单](dataset_pair_inventory.json) / [Rerun 内容验证](rrd_contents_check.txt)',
        '- [有效秩/标准差/预测比曲线](health_curves.png) / [协方差谱](covariance_spectrum.png) / [共享 PCA](latent_pca.png)',
        '- [验收视频](stage3_acceptance.mp4) / [Marimo 指标](metrics.json) / [Marimo 导出](dashboard.html)',
        '- [训练源码快照](formal_training_source.tar.gz) / [ROCm 验证](rocm.json) / [新增测试](unit_tests.log)',
        '- [稳定统计修正前的失败测试](unit_tests_initial_cancellation_failure.log) / [舍入问题最小复现](covariance_cancellation_mre.py) / [复现结果](covariance_cancellation_mre.json)',
        '- [全部 33 项测试](tests_all.log) / [前阶段源码](pre_stage3_source.tar.gz)',''])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed=completed,gates=gates,selected_epoch=selected,test=test),indent=2))


if __name__ == '__main__':
    main()
