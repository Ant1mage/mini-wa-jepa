"""真实记录回放与特征匹配对照；不伪造未来图像或候选选择器。"""
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from chapters.learning import slider,sections,lesson_page,controls_at,map_axes,measured_note
from chapters.chapter6_action_conditioned.src.lesson_data import (
    NAMES,COLORS,available,scene_options,context,local_positions,closest_outcome,compare,
)


def make_controls():
    scenes=scene_options()
    return mo.ui.dictionary({
        'section':sections('1 同场景执行四种方案','2 把方案告诉模型','3 故意给错方案'),
        'scene':mo.ui.dropdown(scenes,value=next(iter(scenes)),label='真实场景',full_width=True),
        'candidate':mo.ui.radio(dict(zip(NAMES,range(4))),value=NAMES[0],label='查看哪个方案'),
        'time':slider(0.,2.,0.,'执行到哪一秒（拖动或按左右键）',.1),
        'horizon':slider(.5,2.,2.,'查看多少秒后的结果',.5),
        'model':mo.ui.radio({'告诉模型方案':'conditioned','只给历史':'unconditional'},value='告诉模型方案',label='给模型哪些信息'),
        'wrong':mo.ui.switch(False,label='故意给模型一条错的方案'),
    })


def compute(values):
    c=context(int(values['scene']))
    chosen=int(values['candidate']);step=round(float(values['time'])/.1)
    positions=[local_positions(branch['states']) for branch in c['branches']]
    return c,chosen,step,positions


def future_card(ax, branch, slot, title, border=None):
    if branch['valid'][slot]:map_axes(ax,branch['future'][slot],title)
    else:ax.text(.5,.5,'该时刻没有真实记录',ha='center');ax.axis('off')
    if border:
        for spine in ax.spines.values():spine.set_color(border);spine.set_linewidth(4)


def render(values, controls=None):
    if not available():
        return mo.md('# Chapter 6 · 不同驾驶方案会带来什么后果？\n\n正式阶段6记录当前不可用；本章不会用随机输出或手工结果替代模型预测。')
    section=values['section'];c,chosen,step,positions=compute(values)
    if section.startswith('1'):
        fig,ax=plt.subplots(figsize=(7.4,4.7),layout='constrained');map_axes(ax,c['history'][-1],'共同起点地图（背景固定）')
        for i,(command,xy,color) in enumerate(zip(c['commands'],positions,COLORS)):
            u,v=63.5-2*command[:,1],63.5-2*command[:,0]
            ax.plot(u,v,'--',color=color,alpha=.65,lw=1.5)
            end=min(step,len(xy)-1);past=xy[:end+1]
            ax.plot(63.5-2*past[:,1],63.5-2*past[:,0],color=color,lw=3 if i==chosen else 1.5,label=NAMES[i])
            ax.scatter(63.5-2*xy[end,1],63.5-2*xy[end,0],color=color,s=110 if i==chosen else 40,edgecolor='white',zorder=5)
        ax.set_xlim(32,95);ax.set_ylim(82,20);ax.legend(loc='upper left',fontsize=8)
        xy=positions[chosen][min(step,len(positions[chosen])-1)]
        keys=('scene','candidate','time');code=local_positions
        question='同一个起点，换一种方案，车真的会走到不同位置吗？'
        text=f'查看到 **{values["time"]:.1f}秒**：**{NAMES[chosen]}** 的实际位置是前方 **{xy[0]:.2f}米**、左侧 **{xy[1]:.2f}米**。虚线是执行前指定的路线，实线与圆点是仿真实际记录。你在切换对照方案，这里没有模型选路器。'
        task='保持同一场景，把进度从0拖到2秒，比较四个圆点怎样分开。'
    else:
        wrong=bool(values['wrong']) if section.startswith('3') else False
        kind='conditioned' if section.startswith('3') else values['model']
        c,slot,matched,distance=compare(values['scene'],chosen,kind,values['horizon'],wrong)
        actual_name='暂时无法唯一对应' if matched is None else NAMES[matched]
        if section.startswith('2'):
            fig,axes=plt.subplots(2,2,figsize=(7.4,5.3),layout='constrained')
            for i,ax in enumerate(axes.flat):
                label=NAMES[i]+(' · 请求方案' if i==chosen else '')+(' · 最近匹配' if i==matched else '')
                future_card(ax,c['branches'][i],slot,label,'#16a34a' if i==matched else None)
            keys=('candidate','model','horizon');code=closest_outcome
            question='模型看到了方案以后，能区分对应的后果吗？'
            model_text='同时接收历史和方案' if kind=='conditioned' else '只接收历史，不接收方案'
            text=f'请求 **{NAMES[chosen]}**，查看 **{values["horizon"]:.1f}秒** 后。当前模型 **{model_text}**；它的预测特征最接近 **{actual_name}** 的记录。四张图都是真实执行后的地图，绿色框只表示特征匹配，不是选路，也不是模型生成了这张图。'
            task='先选“只给历史”，依次切换A/B/C/D，观察最近匹配是否保持不变；再告诉模型方案，看看结果怎样变化。'
        else:
            fig,axes=plt.subplots(1,2,figsize=(8,4.3),layout='constrained')
            future_card(axes[0],c['branches'][chosen],slot,'要比较的真实后果：'+NAMES[chosen])
            if matched is None:axes[1].text(.5,.5,'当前特征无法唯一匹配',ha='center');axes[1].axis('off')
            else:future_card(axes[1],c['branches'][matched],slot,'预测特征最接近的记录：'+NAMES[matched],'#16a34a' if matched==chosen else '#ef4444')
            supplied=(chosen-1)%4 if wrong else chosen
            value=distance[chosen];score='缺失' if not np.isfinite(value) else f'{value:.5f}'
            keys=('candidate','wrong','horizon');code=compare
            question='起点没变，只把给模型的方案换错，会发生什么？'
            text=f'我们要比较 **{NAMES[chosen]}** 的实际后果，却给模型 **{NAMES[supplied]}**。当前预测与目标的特征距离为 **{score}**，最近对应 **{actual_name}**。右图仍是用于核对的已记录后果，不是生成图像；特征距离没有米或碰撞概率的含义。'
            task='开关“故意给模型一条错的方案”，比较同一个目标下的距离和匹配结果；也可以找到模型仍会弄错的案例。'
    return lesson_page(6,'不同驾驶方案会带来什么后果？','本章使用阶段6真实仿真记录和已冻结的模型输出。先看实际执行，再理解方案条件；切换控件不重新训练，也不启动模拟器。',controls,question,controls_at(controls,*keys),fig,text,task,code,
        details='32个测试起点，每个起点保留4条方案，共128条测试分支，来自8个独立种子。本页使用训练种子20261019的已冻结输出。模型预测的是未来特征，图片来自实际仿真；最近后果检索只用于离线核对，不能当作自主驾驶成功率或部署的候选选择器。各未来地图以该时刻自车为中心，所以不应拿车在图上的中心位置衡量行驶距离。',
        source_paths=('models/action_conditioned.py','docs/STAGE6_PROTOCOL.md'),extra=measured_note(6))
