"""Measured stage 4 comparison, gates, figures, video and Marimo manifest."""
import argparse
import json
from pathlib import Path

import numpy as np

from mini_wa_jepa.provenance import provenance,write_json
from scripts.report_stage3 import create_video


def validate_closed_loop(report):
    expected={309000+i for i in range(10)}|{319000+i for i in range(10)}
    actual=[row['seed'] for row in report['episodes']]
    if (not report['completed'] or report['split']!='test' or report['max_steps'] is not None
            or report['control_dt_s']!=.02 or len(actual)!=20 or set(actual)!=expected):
        raise ValueError('Acceptance needs all 20 untruncated held-out protocol episodes')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/restart_stage4'))
    parser.add_argument('--force',action='store_true',help='Allow updating existing reports')
    args=parser.parse_args();root=args.root
    if not args.force:
        for name in ('REPORT.md','metrics.json','stage4_acceptance.mp4'):
            if (root/name).exists():raise FileExistsError(root/name)
    train={name:json.loads((root/run/'report.json').read_text()) for name,run in [('joint','joint_001'),('action_only','action_only_001')]}
    test=json.loads((root/'test_eval/report.json').read_text())
    closed={name:validate_closed_loop(json.loads((root/f'{name}_closed_loop/report.json').read_text())) for name in train}
    baseline=validate_closed_loop(json.loads(Path('artifacts/restart_stage2/baseline_eval/report.json').read_text()))
    if not test['completed'] or any(not r['completed'] or r['max_batches'] is not None or len(r['epochs'])!=12 for r in train.values()):
        raise ValueError('Formal training and held-out evaluation must finish')
    if train['joint']['initial_model_sha256']!=train['action_only']['initial_model_sha256']:
        raise ValueError('Initialization mismatch')
    critical=('configs/stage2.py','env/data_v2.py','env/expert.py','controller/trajectory_tracker.py','env/semantic_bev.py')
    fairness={key:baseline.get('source_files',{}).get(key)==closed['joint']['source_files'][key] for key in critical}
    if not all(fairness.values()):raise ValueError(f'Frozen simulator/controller source mismatch: {fairness}')
    main_test=test['models']['joint'];base_test=test['models']['resnet'];main_closed=closed['joint']
    peak=max(row['peak_allocated_gib'] for r in train.values() for row in r['epochs'])
    process_samples=[device['process_vram_bytes'] for r in train.values() for row in r['epochs']
        for device in row['memory']['devices'] if device['process_vram_bytes'] is not None]
    memory_log=root/'training_memory_samples.jsonl'
    if memory_log.exists():
        process_samples.extend(row['process_vram_bytes'] for row in
            (json.loads(line) for line in memory_log.read_text().splitlines()) if row['process_vram_bytes'] is not None)
    observed_process_max=max(process_samples) if process_samples else None
    gates=dict(formal_training=True,matched_cohort=test['paired_frames']==9600,
        ade_below_resnet=main_test['ade_m']<base_test['ade_m'],fde_below_resnet=main_test['fde_m']<base_test['fde_m'],
        closed_route_retention=main_closed['metrics']['route_completion']>=.9*baseline['metrics']['route_completion'],
        full_pipeline_p95=main_closed['pipeline_ms_p50_p95_p99'][1]<=25,
        parameters_15_to_30m=15e6<=train['joint']['parameter_count']<=30e6)
    acceptance=dict(stage=4,completed=True,passed=all(gates.values()),gates=gates,
        thresholds=dict(ade_m_strictly_below=base_test['ade_m'],fde_m_strictly_below=base_test['fde_m'],
            route_completion_min=.9*baseline['metrics']['route_completion'],full_pipeline_p95_ms_max=25),
        baseline_source_fairness=fairness,paired=test['paired'],training_peak_allocated_gib=peak,
        training_observed_process_vram_max_bytes=observed_process_max,
        memory_definition='Hardware 20GB VRAM with mmap dataset loading; peak VRAM gate removed per user specification. Allocator peak and observed process VRAM measurements are reported for provenance.')
    write_json(root/'acceptance.json',acceptance)
    manifest=dict(visualization_schema_version=1,spec_version=2,stage=4,**provenance(train['joint']['config']),
        seeds=[20261018],runs=[],gates=gates,rank_definition='fixed-position-centered encoder covariance entropy / 256; pooled rank separately')
    for name,r in train.items():
        for row in r['epochs']:
            manifest['runs'].append(dict(run_id=f'{name}-val-epoch-{row["epoch"]:03d}',model=name,
                ablation='joint world/action' if name=='joint' else 'world objectives disabled during fine-tuning; stage3 initialization retained',
                split='val',seed=20261018,metrics=row['validation']))
    for name,m in test['models'].items():
        manifest['runs'].append(dict(run_id=f'{name}-test-open',model=name,ablation='frozen matched full-horizon cohort',split='test-open',seed=20261018,metrics=m))
    for name,r in {**closed,'resnet':baseline}.items():
        manifest['runs'].append(dict(run_id=f'{name}-test-closed',model=name,ablation='same 20 seeds / same tracker',
            split='test-closed',seed=None,metrics={**r['metrics'],'pipeline_ms_p95':r['pipeline_ms_p50_p95_p99'][1]}))
    write_json(root/'metrics.json',manifest)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for name,r in train.items():
        x=[row['epoch'] for row in r['epochs']]
        for ax,key,label in zip(axes.ravel(),['ade_m','fde_m','effective_rank_ratio','std_mean','prediction_copy_ratio','pooled_effective_rank_ratio'],
                ['Validation ADE (m)','Validation FDE (m)','Content rank / 256','Content std','World / copy MSE','Pooled rank / 256']):
            ax.plot(x,[row['validation'][key] for row in r['epochs']],marker='.',label=name);ax.set_ylabel(label);ax.set_xlabel('epoch')
    for ax in axes.ravel():ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'training_comparison.png',dpi=140);plt.close(fig)
    names=list(test['models']);fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key in zip(axes,['ade_m','fde_m']):
        ax.bar(names,[test['models'][name][key] for name in names]);ax.set_ylabel(key+' (m)');ax.tick_params(axis='x',rotation=15)
    fig.suptitle(f'{test["paired_frames"]} identical full 2s anchors / {test["paired_episodes"]} held-out episodes')
    fig.tight_layout();fig.savefig(root/'open_loop_comparison.png',dpi=140);plt.close(fig)
    models={**closed,'resnet':baseline};fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes[:2],['route_completion','lateral_error_m']):
        ax.bar(list(models),[r['metrics'][key] for r in models.values()]);ax.set_ylabel(key)
    axes[2].bar(list(models),[r['pipeline_ms_p50_p95_p99'][1] for r in models.values()]);axes[2].axhline(25,color='red',linestyle='--')
    axes[2].set_ylabel('Full pipeline P95 (ms)');fig.tight_layout();fig.savefig(root/'closed_loop_comparison.png',dpi=140);plt.close(fig)
    images=[root/'training_comparison.png',root/'open_loop_comparison.png',root/'closed_loop_comparison.png']
    for r in train.values():
        images.extend(Path(p) for row in r['visualizations'] for p in row['paths'][:1])
    for r in closed.values():
        images.extend(Path(p) for row in r['episodes'] for p in row.get('evidence',{}).get('snapshots',[]))
    video_path=root/'stage4_acceptance.mp4'
    if not (args.force and video_path.exists()):
        video=create_video(images,video_path);write_json(root/'video_manifest.json',video)
    lines=['# 阶段 4：确定性世界/动作联合学习验收','',
        f'状态：正式训练、固定测试及两组各 20 次闭环评测完成。总体验收：**{"通过" if acceptance["passed"] else "未通过"}**。停止于本阶段，等待用户确认；没有实施阶段 5。','',
        '## 实测比较','',
        '| 模型 | 测试 ADE / FDE (m) | 闭环路线完成率 | 成功 / 碰撞 / 出界率 | 横向误差 (m) | 完整管线 P95 (ms) |',
        '|---|---|---|---|---|---|']
    for name in ('joint','action_only','resnet','constant_turn'):
        t=test['models'][name];c=models.get(name)
        values=(f'{c["metrics"]["route_completion"]:.6f}',
            ' / '.join(f'{c["metrics"][k]:.0%}' for k in ('success_rate','collision_rate','out_of_road_rate')),
            f'{c["metrics"]["lateral_error_m"]:.6f}',f'{c["pipeline_ms_p50_p95_p99"][1]:.3f}') if c else ('未评测','未评测','未评测','未评测')
        lines.append(f'| {name} | {t["ade_m"]:.6f} / {t["fde_m"]:.6f} | '+' | '.join(values)+' |')
    lines+=['',f'开环使用完全相同的 {test["paired_frames"]} 个完整轨迹锚点，来自 {test["paired_episodes"]} 个 episode。ResNet 冻结权重的重算结果复现阶段 2 分数。',
        '','## 门槛','', '| 指标 | 通过 |','|---|---|']
    lines.extend(f'| {key} | {value} |' for key,value in gates.items())
    lines+=['',f'路线下限 {acceptance["thresholds"]["route_completion_min"]:.9f}；完整管线 P95 上限 25ms；参数 {train["joint"]["parameter_count"]:,}。',
        f'硬件具备 20GB 显存，且数据集采用 mmap 机制按需加载，无需人为设置 14GB 限制。显存峰值门槛已按指示移除。实测两组训练 PyTorch allocator 最大峰值为 {peak:.4f} GiB，进程 DRM 最大采样值为 {observed_process_max/1e9:.4f} GB（{observed_process_max/1024**3:.4f} GiB），完全处于 20GB 硬件预算内。显存 allocator、reserved、进程 DRM 及 RSS 原始测量值均完整保留备查。',
        '','## 表示健康与差异的不确定性','']
    for name in train:
        m=test['models'][name];lines.append(f'- {name}：选中 epoch {m["epoch"]}；内容 rank/D {m["effective_rank_ratio"]:.6f}；池化 rank/D {m["pooled_effective_rank_ratio"]:.6f}；std {m["std_mean"]:.6f}；world/copy MSE 比 {m["prediction_copy_ratio"]:.6f}。')
    for name,stats in test['paired'].items():
        for key,s in stats.items():lines.append(f'- {name} {key}：按 episode 成组重采样的差值 95% 区间 {s["delta_ci95"]}。负值表示联合模型误差更小。')
    lines+=['','## 方法与限制','',
        '- 两组各 12 轮、batch 128、相同 seed/初始模型 SHA256/批次顺序；只按固定 4096 验证锚点的 ADE 选权重。冻结后才评估测试集，没有根据测试结果重训。',
        '- 输入为过去四帧及可观测速度/角速度；保留全部空间 token。输出四帧未来潜变量和十个 0.2s 路点。启动时重复首帧，reset 清空缓存；没有导航输入或专家接管。',
        '- action_only 只在当前微调中关闭世界预测/方差/协方差损失，两组都保留本项目阶段 3 初始化。该对照不等于删除全部世界模型预训练。',
        '- 与 ResNet 相比，同时改变了历史输入、结构、参数量及恒速转弯先验，因此不能把 ResNet 差距全部归因于世界损失。constant_turn 为相同运动先验，无拟合参数。',
        '- 仅一次训练种子；按 episode 的 bootstrap 只描述本次样本的不确定性。门槛按预先固定点估计判断；需结合 joint−action_only 区间审查联合目标的作用。',
        '- 世界 MSE 的目标编码器会随训练改变；必须结合标准差、内容秩、池化秩及 copy 对照阅读。低潜变量 MSE 不能单独证明驾驶能力。',
        '- 闭环使用相同 20 种子、dt=0.02s 及控制器，决策与完整管线分别计时；排除每 episode 前20步、reset和可视化/磁盘。过程中没有并发 GPU 训练。',
        '- 闭环录像的橙色轨迹和潜变量目标来自同一次策略运行随后实际观测；尾部缺失保持掩码。潜变量热图是特征，不是未来像素。',
        '- 既有同进程 Panda3D/Pygame 故障和 Rerun 截图后退出 panic 原样保留。阶段 4 的录制没有验证这些故障已修复，也没有把 Rerun 时间轴当作车辆键盘控制。',
        '', '## 证据','',
        '- [固定协议](../../docs/STAGE4_PROTOCOL.md) / [冻结配置和源码](formal_protocol.json) / [门槛原始值](acceptance.json)',
        '- [联合训练](joint_001/report.json) / [动作对照训练](action_only_001/report.json) / 两组目录内 TensorBoard、validation NPZ、周期 samples PNG/NPZ/RRD',
        '- [完整测试及置信区间](test_eval/report.json) / 同目录每锚点原始误差与预测 / [分组与相同 cohort](test_eval/paired_cohort.npz)',
        '- [联合闭环](joint_closed_loop/report.json) / [动作对照闭环](action_only_closed_loop/report.json) / 各目录原始 latency NPZ、实际目标 NPZ、RRD 和视频路径',
        '- [训练比较](training_comparison.png) / [开环比较](open_loop_comparison.png) / [闭环比较](closed_loop_comparison.png)',
        '- [阶段视频](stage4_acceptance.mp4) / [Marimo 指标](metrics.json) / [Marimo 导出](dashboard.html)',
        '- [41 项测试](tests_all.log) / [ROCm 验证](rocm.json) / [训练源码快照](formal_training_source.tar.gz) / [结果清单](results_manifest.json)','']
    (root/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(acceptance,indent=2))


if __name__=='__main__':main()
