"""面向初学者的分节交互实验；合成演示与正式记录分别标注。"""
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from chapters.learning import (
    ROOT, slider, sections, lesson_page, controls_at, map_axes, bev_scene, measured_note,
)


def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 先看四帧历史", "2 看懂两个任务", "3 路线与执行"),
        "speed": slider(0.,30.,10.,"当前速度（米/秒）",.5),
        "yaw": slider(-1.,1.,.15,"转向变化率（弧度/秒）",.05),
        "residual": slider(-2.,2.,0.,"教学中的横向修正（米）",.1),
        "time": slider(1,10,1,"沿路线查看第几步"),
        "frame": slider(0,3,3,"历史帧：0最早，3当前"),
        "phase": slider(0,4,0,"计算过程：逐步向右查看"),
    })

def compute(values):
    """调用正式先验函数；红线残差由滑块手工设置，不是网络推理。"""
    import torch
    from models.joint import constant_turn_trajectory
    # 正式输入是四帧观测，每帧包含 speed/10 与 yaw_rate。
    ego = torch.tensor([[[values["speed"] / 10, values["yaw"]]] * 4], dtype=torch.float32)
    with torch.no_grad():
        prior = constant_turn_trajectory(ego)[0].numpy()
    t = np.arange(1, 11) * .2
    modified = prior.copy()
    modified[:, 1] += values["residual"] * (t / 2)**2
    index = int(values["time"]) - 1
    return dict(prior=prior, modified=modified, index=index, t=t)

def history_frame(index):
    # 四帧相隔0.1秒，前车每帧接近0.5米。
    return bev_scene(actor_forward=13.5 - .5 * index)


def process_boxes(ax, step):
    # 框代表计算环节，不是单个神经元；当前步骤用蓝色强调。
    labels=['四帧历史','共享特征','估计未来变化','给出驾驶路线','控制器执行']
    xy=[(.03,.45),(.24,.45),(.47,.7),(.47,.2),(.76,.2)]
    for i,(label,(x,y)) in enumerate(zip(labels,xy)):
        ax.add_patch(Rectangle((x,y),.19,.15,facecolor='#2563eb' if i==step else '#e2e8f0',edgecolor='white'))
        ax.text(x+.095,y+.075,label,ha='center',va='center',color='white' if i==step else '#334155',fontsize=10)
    for a,b in [(0,1),(1,2),(1,3),(2,3),(3,4)]:
        # 世界预测特征也连接到轨迹任务，控制器最终执行路点。
        x,y=xy[a];u,v=xy[b]
        ax.annotate('',xy=(u,v+.075),xytext=(x+.19,y+.075),arrowprops=dict(arrowstyle='->',color='#64748b'))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')

def render(values, controls=None):
    section=values['section'];data=compute(values)
    fig,ax=plt.subplots(figsize=(7.5,4.3),layout='constrained')
    if section.startswith('1'):
        i=int(values['frame']);map_axes(ax,history_frame(i),f'正在看：{(i-3)*.1:+.1f}秒的观测')
        question='为什么让模型连续看四帧？';keys=('frame',);code=history_frame
        text=f'四个时刻分别是 **−0.3、−0.2、−0.1、0秒**。现在看第 **{i+1}帧**，红色前车在移动；连续观察才有助于分清“在哪里”和“怎样移动”。这些是合成历史，不是未来输入。'
        task='从第0帧逐步拖到第3帧，观察前车位置变化。'
    elif section.startswith('2'):
        step=int(values['phase']);process_boxes(ax,step)
        explanations=['先收集已经发生的四帧观测，不能把未来标签放进推理输入。','编码器把四帧变成共享特征；两个任务使用这份表示。','世界分支估计未来特征变化；这些特征不是可直接观看的未来图像。','轨迹分支结合当前和预测特征，输出十个驾驶路点。','控制器把路点转换为转向和速度控制，车辆执行后产生新的观测。']
        question='“世界预测”和“驾驶路线”是怎样接起来的？';keys=('phase',);code=process_boxes
        text=f'**第{step+1}步：** {explanations[step]} 蓝色框是当前步骤。这里只是流程示意，没有执行一次真实模型推理。'
        task='聚焦滑块后按左右键，逐步读每个模块在做什么。'
    else:
        prior=data['prior'];points=data['modified'];i=data['index']
        ax.plot(prior[:,1],prior[:,0],'--',color='#94a3b8',label='保持当前运动的先验')
        ax.plot(points[:,1],points[:,0],'-',color='#2563eb',label='教学修正后的路线')
        ax.scatter(points[i,1],points[i,0],s=130,marker='^',color='#ea580c',label='沿路线查看的位置')
        ax.set(xlabel='左向位置 y（米）',ylabel='前向位置 x（米）',title=f'查看到未来{data["t"][i]:.1f}秒的路点');ax.legend(fontsize=9);ax.grid(alpha=.2)
        question='有了一条路线，下一步怎样走？';keys=('speed','yaw','residual','time');code=compute
        text=f'当前高亮的是 **{data["t"][i]:.1f}秒** 路点，位置为 **前方{points[i,0]:.2f}米、左侧{points[i,1]:.2f}米**。灰线来自正式运动学先验函数，蓝线的修正由你手工设置；移动标记是在查看路线，不是新的仿真驾驶结果。'
        task='先把转向变化率和修正都设为0，再逐步增加转向变化率，看看直线怎样变成弯线。'
    return lesson_page(4,'把历史、世界变化和驾驶路线接起来','先看历史，再理解两个学习任务，最后沿一条示意路线逐步查看。',controls,question,controls_at(controls,*keys),fig,text,task,code,
        details='联合模型保留64个空间Token，世界分支预测特征，轨迹分支给出10个(x,y,v,heading)路点。Action-only对照使用相同结构但关闭世界损失，不能把流程图中的框当成真实神经元或已测安全保证。',source_paths=('models/joint.py','training/joint_loss.py'),extra=measured_note(4))
