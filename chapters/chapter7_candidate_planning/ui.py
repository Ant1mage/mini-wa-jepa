"""生成、评分、执行：所有画面均来自冻结模型或真实仿真记录。"""
from functools import lru_cache
import inspect
import json
from pathlib import Path
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from chapters.learning import ROOT,slider,sections,lesson_page,map_axes,figure,source_files
from evaluation.planning_view import decision_figure,score_rows,COLORS


ARTIFACTS=ROOT/'artifacts/restart_stage7'


def recording_options():
    options={}
    names={'mse':'原 MSE 轨迹头','geometry':'Flow＋几何评分','full':'Flow＋世界风险评分'}
    for path in sorted(ARTIFACTS.glob('closed_*/*/actual.npz')):
        report=path.parents[1]/'report.json'
        if not report.exists():continue
        m=json.loads(report.read_text());seed_label='对照配对组' if m['arm']=='mse' else '训练种子'
        label=f'{"冒烟，仅验证链路" if m["smoke"] else "正式"} · {names[m["arm"]]} · {path.parent.name} · {seed_label}{m["training_seed"]}'
        options[label]=str(path)
    return options or {'尚未产生闭环记录':'missing'}


def make_controls():
    options=recording_options()
    return mo.ui.dictionary({
        'section':sections('1 从噪声生成路线','2 看评分、选择与执行','3 看实测与限制'),
        'seed':mo.ui.radio({'训练种子 20261021':20261021,'训练种子 20261022':20261022},value='训练种子 20261021',label='查看哪次训练'),
        'case':mo.ui.dropdown({'固定测试样本 0':0,'固定测试样本 16':16,'固定测试样本 99':99},value='固定测试样本 0',label='真实测试起点'),
        'flow_step':slider(0,10,0,'积分进度（0 是噪声，10 是最终输出）'),
        'recording':mo.ui.dropdown(options,value=next(iter(options)),label='真实驾驶记录',full_width=True),
        'decision':slider(0,999,0,'决策步（每步 0.1 秒；左右键可单步）'),
    })


@lru_cache(maxsize=3)
def load_record(path):
    # 明确读取同一个 actual.npz；没有另外拼接演示用轨迹。
    with np.load(path) as z:return {key:z[key] for key in z.files if key not in ('latent_prediction','observed_future_under_replanning')}


def compute(values):
    if values['section'].startswith('1'):
        path=ARTIFACTS/'flow_paths'/f'{values["seed"]}_{values["case"]}.npz'
        if not path.exists():return dict(missing='积分过程记录尚未生成，保持缺失。')
        with np.load(path) as z:data={k:z[k] for k in z.files}
        return dict(kind='flow',data=data,step=int(values['flow_step']))
    if values['recording']=='missing':return dict(missing='闭环记录尚未生成，保持缺失。')
    data=load_record(values['recording']);index=min(int(values['decision']),len(data['plans'])-1)
    return dict(kind='planning',data=data,index=index,rows=score_rows(data,index))


def render(values,controls=None):
    result=compute(values)
    if 'missing' in result:
        widgets=[controls[k] for k in ('section','seed','case','flow_step','recording','decision')] if controls is not None else []
        return mo.vstack([mo.md('# Chapter 7 · 多条候选怎样变成一次驾驶决定？'),*widgets,mo.md(result['missing'])])
    section=values['section']
    if section.startswith('1'):
        data=result['data'];step=result['step'];fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
        map_axes(ax,data['bev'],'真实起点地图；叠加真实模型的积分中间结果')
        from mini_wa_jepa.geometry import ego_to_pixels
        for k,path in enumerate(data['path'][step]):
            xy=ego_to_pixels(path[:,:2],128,64);ax.plot(xy[:,0],xy[:,1],'-o',color=COLORS[k],markersize=2,label=chr(65+k))
        xy=ego_to_pixels(data['ground_truth'][:,:2],128,64);ax.plot(xy[:,0],xy[:,1],color='orange',lw=3,label='专家实际轨迹，仅作核对')
        ax.legend(fontsize=8);keys=('seed','case','flow_step')
        question='同一个场景，八份噪声怎样逐步变成八条路线？'
        explanation=f'当前是 **第 {step}/10 次积分**。这是冻结模型实际计算并保存的中间数组；初始噪声不是可执行路线。Flow Matching 学习每一步应怎样移动轨迹数组，历史 BEV 和车辆运动作为条件。橙线来自专家已实际执行的记录，只用于离线核对，不参与部署时的候选选择。'
        task='从 0 单步拖到 10，观察候选如何变化；再换训练种子，比较最终路线是否合理。'
        details='时间 t 是生成过程的积分进度，不是车辆已经行驶的秒数。每条最终路线包含 10 个间隔 0.2 秒的 x/y/速度/航向路点。只有通过可行性和风险检查的候选才可能被执行。'
        extra=None
    else:
        data=result['data'];index=result['index'];fig=decision_figure(data,index,chinese=True)
        keys=('recording','decision');question='这一轮为什么选它，车后来实际怎么走？'
        selected=str(data['selected_ids'][index]);source=str(data['selection_sources'][index])
        source_label={'candidate_score':'固定评分规则选择','rule_emergency_stop_all_rejected':'候选全部拒绝后的规则刹停',
            'frozen_stage4_action_head':'原 MSE 轨迹头直接输出'}.get(source,source)
        explanation=f'当前实际决策步 **{index}**，选中 **{selected}**，来源为 **{source_label}**。虚线是这一轮完整候选，粗青线是交给控制器的路线；白线是已走过的轨迹，橙线是随后在持续重新规划下的实际行驶。向右一步即可看到下一轮预测与选择。'
        if selected=='STOP':explanation+=' **所有候选被拒绝，执行的是规则生成的刹停路线；这不是 Flow 成功生成并选中了一条路线。**'
        task='在同一主画面查看全部候选，再对照下方分项成本与拒绝原因；拖动一步比较下一轮。'
        details='本页是同一条真实记录的回放，不会重新启动仿真或训练。无风险预测的几何对照中，碰撞/越界概率保持缺失，不能解释为零风险。候选潜变量没有被伪装成未来道路或车辆图片；动态几何使用历史观测的匀速近似。'
        extra=mo.ui.table(result['rows'],selection=None,page_size=8)
        if section.startswith('3'):
            question='生成了多条路线，就一定开得更好吗？'
            report=ARTIFACTS/'REPORT.md'
            explanation='正式门槛与实际结论见下方报告。未通过就是未通过；不能用最接近真值的 oracle 候选误当成可部署选择器的成绩。'
            extra=mo.vstack([mo.md(report.read_text()) if report.exists() else mo.md('正式阶段报告尚未完成。'),extra])
    if section.startswith('2'):
        labels={'progress':'进度项','speed':'速度项','lateral_accel':'横向加速度项','curvature':'转向平滑项',
            'static':'静态几何项','dynamic':'动态几何项','crash':'世界碰撞风险项','offroad':'世界越界风险项'}
        reasons={'nonfinite':'含非有限值','speed':'速度越界','acceleration':'加速度过大','heading_position':'航向与位置不一致',
            'crash_threshold':'碰撞风险过高','offroad_threshold':'越界风险过高'}
        rows=[]
        for row in result['rows']:
            translated={labels.get(k,k):'未产出' if v is None else v for k,v in row.items()}
            translated['拒绝原因']='；'.join(reasons.get(x,x) for x in str(row['拒绝原因']).split(';'))
            rows.append(translated)
        return mo.vstack([
            mo.md('# Chapter 7 · 多条候选怎样变成一次驾驶决定？'),
            controls['section'] if controls is not None else mo.md(''),mo.md('## '+question),
            mo.hstack([mo.vstack([controls[k] for k in keys]) if controls is not None else mo.md(''),
                figure(fig),mo.vstack([mo.md('**本轮实际评分**\n\n总成本越低越优；各分项是加权前原值。'),
                    mo.ui.table(rows,selection=None,page_size=8)])],widths=[1,2,2],align='start',wrap=True),
            mo.callout(mo.md(explanation)),mo.md('**试一试：** '+task),
            mo.accordion({'记录与近似方法说明':mo.md(details),
                '查看实际回放计算代码':mo.md('```python\n'+inspect.getsource(compute)+'\n```')}),
            source_files('planning/candidates.py','docs/STAGE7_PROTOCOL.md')])
    return lesson_page(7,'多条候选怎样变成一次驾驶决定？',
        '先观察生成过程，再看真实评分、选择、执行和下一轮规划。所有曲线都有实际模型或仿真记录来源。',
        controls,question,[controls[k] for k in keys] if controls is not None else [],fig,explanation,task,compute,
        details=details,source_paths=('models/flow_matcher.py','planning/candidates.py','docs/STAGE7_PROTOCOL.md'),extra=extra)
