"""阶段5完整实测报告、配对统计、图表、视频和交付审计。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from configs.stage5 import Stage5Config, CONDITIONS, CHECKPOINTS
from evaluation.latent import paired_episode_bootstrap
from mini_wa_jepa.provenance import provenance, write_json
from scripts.report_stage3 import create_video


def episode_ci(joint_rows, other_rows, metric, cfg):
    get=lambda rows: {r['seed']:float(r[metric]) for r in rows}
    a,b=get(joint_rows),get(other_rows)
    if set(a)!=set(b):raise ValueError('Closed-loop seeds do not match')
    keys=sorted(a)
    result=paired_episode_bootstrap([a[k] for k in keys],[b[k] for k in keys],keys,cfg.bootstrap_samples,cfg.seed)
    # 碰撞/成功率的bootstrap分母可能为零；只报告有定义的配对差值区间。
    result['ratio_ci95']=None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('artifacts/restart_stage5'))
    args=parser.parse_args();root=args.root;cfg=Stage5Config()
    if (root/'REPORT.md').exists():raise FileExistsError('Report already exists; preserve previous delivery')
    opened=json.loads((root/'open_loop/report.json').read_text())
    suite=json.loads((root/'suite_progress.json').read_text())
    if not opened['completed'] or opened['smoke_samples'] or not suite['completed']:
        raise ValueError('Formal evaluations incomplete')
    jobs=[j for j in suite['jobs'] if not j['smoke']]
    closed={};seedset={309000,309001,309002,319000,319001,319002};evidence=[]
    gates=dict(open_cohort=True,clean_open_reproduced=True,closed_cohort=True,
               frozen_checkpoints=True,finite_metrics=True,multimodal_evidence=True)
    reference=np.load('artifacts/restart_stage4/test_eval/paired_cohort.npz')
    for model in CHECKPOINTS:
        if len(opened['models'][model]['conditions'])!=10:gates['open_cohort']=False
        expected=opened['models'][model]['checkpoint_sha256']
        gates['frozen_checkpoints'] &= hashlib.sha256(Path(CHECKPOINTS[model]).read_bytes()).hexdigest()==expected
        for spec in CONDITIONS:
            folder=root/'open_loop'/model/spec.name;raw=np.load(folder/'raw.npz')
            gates['open_cohort'] &= np.array_equal(raw['anchors'],reference['anchors']) and raw['full_mask'].all()
            gates['finite_metrics'] &= all(np.isfinite(raw[k]).all() for k in ('prediction','target','ade_m','fde_m'))
            if spec.name=='clean':
                gates['clean_open_reproduced'] &= all(v['absolute_delta']<=2e-6 for v in opened['models'][model]['conditions'][spec.name]['clean_parity'].values())
            evidence+=opened['models'][model]['conditions'][spec.name]['evidence']
    if len(jobs)!=21:gates['closed_cohort']=False
    for job in jobs:
        row=json.loads((Path(job['output'])/'report.json').read_text())
        name=job['model'];case=job['condition'];closed.setdefault(case,{})[name]=row
        gates['closed_cohort'] &= bool(row['completed'] and row['max_steps'] is None and len(row['episodes'])==6
            and {e['seed'] for e in row['episodes']}==seedset and row['control_dt_s']==.02)
        gates['frozen_checkpoints'] &= row['checkpoint_sha256']==opened['models'][name]['checkpoint_sha256']
        gates['finite_metrics'] &= all(np.isfinite(v) for v in row['metrics'].values() if v is not None)
        recordings=[e['evidence'] for e in row['episodes'] if e['evidence']]
        gates['multimodal_evidence'] &= len(recordings)==1
        for recording in recordings:
            evidence.append(recording)
            actual=np.load(recording['raw'])
            gates['finite_metrics'] &= np.isfinite(actual['prediction']).all() and np.isfinite(actual['target']).all()
            if name!='resnet':
                gates['multimodal_evidence'] &= actual['world_valid'].any() and np.isfinite(actual['latent_target']).all()
    paired_closed={case:{'joint_minus_'+other:{metric:episode_ci(models['joint']['episodes'],models[other]['episodes'],metric,cfg)
                                              for metric in ('route_completion','collision','out_of_road','arrive_dest')}
                        for other in ('resnet','action_only')} for case,models in closed.items()}
    clean_reference={}
    for name in CHECKPOINTS:
        path=Path('artifacts/restart_stage2/baseline_eval/report.json') if name=='resnet' else Path(f'artifacts/restart_stage4/{name}_closed_loop/report.json')
        old={r['seed']:r for r in json.loads(path.read_text())['episodes']}
        rows=closed['clean'][name]['episodes']
        clean_reference[name]=[dict(seed=r['seed'],route_completion_delta=r['route_completion']-old[r['seed']]['route_completion'],
                                     collision_agrees=r['collision']==old[r['seed']]['collision'],
                                     success_agrees=r['arrive_dest']==old[r['seed']]['arrive_dest']) for r in rows]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=[]
    for key in ('ade_m','fde_m'):
        fig,axes=plt.subplots(1,3,figsize=(12.8,4.8),layout='constrained')
        for ax,kind in zip(axes,('occlusion','boundary','dropout')):
            cases=[CONDITIONS[0]]+[c for c in CONDITIONS if c.kind==kind]
            for model in CHECKPOINTS:
                values=[opened['models'][model]['conditions'][c.name]['metrics'][key] for c in cases]
                ax.plot(range(4),values,'o-',label=model)
            ax.set(title=kind,xlabel='severity level (0 = clean)',ylabel=key);ax.set_xticks(range(4));ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Stage 5 | same 9,600 held-out anchors | actual measurements')
        path=root/f'open_loop_{key}.png';fig.savefig(path,dpi=100);plt.close(fig);plots.append(path)
    fig,axes=plt.subplots(1,2,figsize=(12.8,5),layout='constrained');cases=list(cfg.closed_conditions)
    for index,name in enumerate(CHECKPOINTS):
        x=np.arange(len(cases))+(index-1)*.24
        axes[0].bar(x,[closed[c][name]['metrics']['route_completion'] for c in cases],width=.24,label=name)
        axes[1].bar(x,[closed[c][name]['metrics']['collision_rate'] for c in cases],width=.24,label=name)
    for ax,title in zip(axes,('Closed-loop route completion','Closed-loop collision rate')):
        ax.set(title=title,ylim=(0,1.05));ax.set_xticks(np.arange(len(cases)),cases,rotation=30,ha='right',fontsize=7);ax.legend(fontsize=8)
    path=root/'closed_loop_comparison.png';fig.savefig(path,dpi=100);plt.close(fig);plots.append(path)
    manifest=dict(visualization_schema_version=1,spec_version=2,stage=5,**provenance(cfg),seeds=[cfg.seed],runs=[])
    from mini_wa_jepa.visualization.reports import METRICS
    for name,model in opened['models'].items():
        for case,row in model['conditions'].items():
            manifest['runs'].append(dict(run_id=f'{name}-{case}-open',model=name,ablation=case,split='test-open-9600',seed=cfg.seed,
                degradation_type=row['condition']['kind'],degradation_level=row['condition']['level'],
                metrics={k:v for k,v in row['metrics'].items() if k in METRICS},raw=str(root/'open_loop'/name/case/'raw.npz')))
    for case,models in closed.items():
        for name,model in models.items():
            for row in model['episodes']:
                metrics={k:row.get(k) for k in ('route_completion','lateral_error_m','inference_ms','pipeline_ms_p95')}
                metrics.update(collision_rate=float(row['collision']),out_of_road_rate=float(row['out_of_road']),success_rate=float(row['arrive_dest']))
                manifest['runs'].append(dict(run_id=f'{name}-{case}-closed-{row["seed"]}',model=name,ablation=case,
                                            split='test-closed',seed=row['seed'],metrics=metrics))
    write_json(root/'metrics.json',manifest)
    # 同时检查绝对误差优势与相对clean的退化增量，不把干净基线差异冒充鲁棒性收益。
    assessment={}
    for spec in CONDITIONS[1:]:
        joint=np.load(root/'open_loop/joint'/spec.name/'raw.npz')
        resnet=np.load(root/'open_loop/resnet'/spec.name/'raw.npz')
        joint_clean=np.load(root/'open_loop/joint/clean/raw.npz')
        resnet_clean=np.load(root/'open_loop/resnet/clean/raw.npz')
        delta=paired_episode_bootstrap(joint['ade_m']-joint_clean['ade_m'],resnet['ade_m']-resnet_clean['ade_m'],
                                      joint['episodes'],cfg.bootstrap_samples,cfg.seed)
        absolute_ci=opened['paired'][spec.name]['joint_minus_resnet']['ade_m']['delta_ci95']
        assessment[spec.name]=dict(joint_minus_resnet_absolute_ade_ci95=absolute_ci,
             joint_minus_resnet_degradation_increment_ci95=delta['delta_ci95'],
             absolute_ade_result='joint_lower' if absolute_ci[1]<0 else ('joint_higher' if absolute_ci[0]>0 else 'inconclusive'),
             degradation_increment_result='joint_smaller' if delta['delta_ci95'][1]<0 else ('joint_larger' if delta['delta_ci95'][0]>0 else 'inconclusive'))
    from scripts.analyze_stress_cases import analyze
    edge_cases,edge_plots=analyze(root)
    summary=dict(**provenance(cfg),stage=5,completed=True,open_loop=opened,closed_loop=closed,paired_closed=paired_closed,
                 robustness_assessment=assessment,edge_cases=edge_cases,
                 clean_closed_reference=clean_reference,plots=[str(p) for p in plots],closed_episode_count=sum(len(r['episodes']) for m in closed.values() for r in m.values()))
    write_json(root/'summary.json',summary)
    video_path=root/'stage5_acceptance.mp4'
    image_paths=plots+edge_plots+[Path(r['png']) for r in evidence if 'png' in r]+[Path(p) for r in evidence for p in r.get('snapshots',[])]
    create_video(image_paths,video_path)
    gates['multimodal_evidence'] &= video_path.is_file()
    delivery=[]
    import imageio_ffmpeg
    for row in evidence:
        rrd=row.get('rerun')
        if rrd:
            ok=Path(rrd).is_file() and Path(rrd).stat().st_size>0
            gates['multimodal_evidence'] &= ok
        if row.get('video'):
            path=Path(row['video']);test=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-f','null','-'],capture_output=True)
            delivery.append(dict(video=str(path),decoded=test.returncode==0,bytes=path.stat().st_size))
            gates['multimodal_evidence'] &= test.returncode==0
    gates={key:bool(value) for key,value in gates.items()}
    acceptance=dict(stage=5,completed=True,passed=all(gates.values()),gates=gates,
                    interpretation='These gates audit protocol completion and evidence; they do not assert a positive robustness result.')
    write_json(root/'acceptance.json',acceptance)
    write_json(root/'delivery_checks.json',dict(videos=delivery,rrd_count=sum(bool(r.get('rerun')) for r in evidence),gates=gates))
    lines=['# 阶段5：冻结模型感知退化压力测试','',
           f'正式开环：3模型 × 10条件 × 9,600锚点；闭环：3模型 × 7条件 × 6种子 = {summary["closed_episode_count"]} episodes。',
           '闭环样本比阶段4的20种子小，CI可能较宽；不代表完整20种子复验。所有轨迹均为实际模型预测，没有重新训练。','',
           '## 开环匹配测试结果','', '| 条件 | Joint ADE/FDE m | Action-only ADE/FDE m | ResNet ADE/FDE m |','|---|---|---|---|']
    for spec in CONDITIONS:
        metrics=[opened['models'][m]['conditions'][spec.name]['metrics'] for m in CHECKPOINTS]
        lines.append('| '+spec.name+' | '+' | '.join(f'{r["ade_m"]:.6f} / {r["fde_m"]:.6f}' for r in metrics)+' |')
    lines+=['','## 闭环六种子平均','', '| 条件 | Joint 完成/碰撞 | Action-only 完成/碰撞 | ResNet 完成/碰撞 |','|---|---|---|---|']
    for case in cfg.closed_conditions:
        metrics=[closed[case][m]['metrics'] for m in CHECKPOINTS]
        lines.append('| '+case+' | '+' | '.join(f'{r["route_completion"]:.4f} / {r["collision_rate"]:.1%}' for r in metrics)+' |')
    lines+=['','## 鲁棒性假说的实际检验','',
            '| 条件 | Joint 对 ResNet 绝对 ADE | 相对 clean 的 ADE 增量 |','|---|---|---|']
    for case,row in assessment.items():
        lines.append(f'| {case} | {row["absolute_ade_result"]}，CI={row["joint_minus_resnet_absolute_ade_ci95"]} | '
                     f'{row["degradation_increment_result"]}，CI={row["joint_minus_resnet_degradation_increment_ci95"]} |')
    lines+=['','joint_lower/smaller表示95% episode配对bootstrap区间完全小于0；higher/larger表示完全大于0；跨0为inconclusive。',
            '这是每条件未做多重比较校正的探索性区间，不能把任一单项结果外推为普遍鲁棒性保证。','',
            '## 关键样本诊断','',
            '按照输入/标签选择参与者由无到有、参与者像素最大变化、首个恢复和首个噪声样本；见edge_cases/cases.json和图。',
            '不按模型误差挑选案例。参与者出现可能只是进入视野；恢复标签来自原数据，本阶段闭环没有额外注入动作扰动。','',
            '## 如何解释这些结果','',
            '相同条件下模型绝对误差与相对clean增量都要看；某模型干净表现较差，不意味着退化增量小就更适合驾驶。',
            '逐episode配对bootstrap原始结果在 open_loop/report.json 和 summary.json；CI跨零时不能声称显著优越。',
            '固定空间掩码、周期性传输丢失、ego状态仍可用，均限制结论外推。闭环执行后各模型轨迹分叉，无法逐帧配对，只按相同seed配对episode。',
            'Clean闭环与历史逐种子偏差单独存于 summary.json，不能把开环数值重现当成闭环一定完全相同。','',
            '## 交付与来源','',f'配置/随机种子：configs/stage5.py；协议SHA-256：{opened["protocol_sha256"]}。',
            '源码commit、GOAL hash、checkpoint hash、原始数组与条件配置均随每次报告保存；失败/smoke另目录保留。',
            'TensorBoard在open_loop/tensorboard和各closed_loop job目录；.rrd/实际预测与干净目标latent、视频/真值图见各job evidence。',
            '验收视频：stage5_acceptance.mp4；主图：open_loop_ade_m.png、open_loop_fde_m.png、closed_loop_comparison.png。',
            f'交付门禁：{gates}；overall passed={acceptance["passed"]}。',
            '阶段5报告完成后停止；阶段6未启动。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # 导出统一的实测看板；不以合成教学页替代正式结果。
    with (root/'dashboard_export.log').open('w') as log:
        export=subprocess.run(['.venv/bin/marimo','export','html','scripts/dashboard_marimo.py','-o',str(root/'dashboard.html')],stdout=log,stderr=subprocess.STDOUT)
    write_json(root/'dashboard_export.json',dict(returncode=export.returncode,exists=(root/'dashboard.html').is_file()))
    if export.returncode:raise RuntimeError('Measured dashboard export failed; preserve log')
    tasks=Path('TASKS.md');text=tasks.read_text()
    finished=['Implement parameterized BEV degradation suite:', 'Run matched open-loop degradation evaluation',
              'Run closed-loop stress evaluation', 'Analyze critical edge cases:', 'Deliver Stage 5 report,']
    if acceptance['passed']:
        for prefix in finished:
            text=text.replace('- [ ] '+prefix,'- [x] '+prefix)
    text += ('\n## Stage 5 measured report completion\n\n'
             f'Formal evaluation completed: 9,600 matched anchors × 3 models × 10 conditions; 126 closed-loop episodes. '
             f'Protocol/delivery gates passed={acceptance["passed"]}. Results, per-condition paired CIs, raw predictions, '
             'TensorBoard, RRD, actual rollout videos, acceptance video and measured dashboard are under '
             '`artifacts/restart_stage5/`. Lower absolute error and smaller degradation increments are evaluated separately; '
             'no universal robustness gain is presumed. See REPORT.md and summary.json. Stop here awaiting explicit Stage 6 confirmation.\n')
    tasks.write_text(text)
    print(json.dumps(acceptance),flush=True)


if __name__=='__main__':main()
