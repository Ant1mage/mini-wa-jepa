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
        "section": sections("1 示范与误差", "2 修正一次偏差", "3 缺失标签"),
        "speed": slider(2.,15.,6.,"示范速度（米/秒）",.5),
        "bias": slider(-2.,2.,1.,"横向偏差（米）",.1),
        "noise": slider(0.,1.,.15,"点的位置扰动（米）",.05),
        "valid": slider(1,10,10,"有真实记录的未来点数"),
        "seed": slider(0,20,7,"扰动种子"),
        "updates": slider(0,12,0,"学习步数（拖动或按左右键）"),
        "rate": slider(.05,.5,.2,"每一步修正的幅度",.05),
    })

def compute(values):
    """手工构造参考/预测轨迹，用于讲解指标；不冒充训练模型输出。"""
    t = np.arange(1, 11) * .2  # 0.2, 0.4, ... , 2.0 秒。
    target = np.stack([t * values["speed"], .4 * t**2], axis=-1)
    rng = np.random.default_rng(int(values["seed"]))
    prediction = target + rng.normal(size=(10, 2)) * values["noise"]
    prediction[:, 1] += values["bias"]  # 单独添加横向系统偏差。
    mask = np.arange(10) < int(values["valid"])
    error = np.linalg.norm(prediction - target, axis=1)  # 每个点的欧氏距离。
    ade = error[mask].mean()  # 只统计有效标签，不把补零标签当真值。
    fde = float(error[-1]) if mask[-1] else None  # 第 10 点无真值时保持缺失。
    # 演示 x/y 两维的加权 L1；实际训练还含速度、航向和归一化。
    position_l1 = (np.abs(prediction - target) * 2)[mask].sum() / mask.sum()
    return dict(t=t, target=target, prediction=prediction, mask=mask,
                error=error, ade=float(ade), fde=fde, position_l1=float(position_l1))

def bias_updates(initial_bias, rate, steps):
    # 只学习一个横向偏差b，教学损失为b²。
    history = [float(initial_bias)]
    for _ in range(steps):
        gradient = 2 * history[-1]  # b²对b的导数。
        history.append(history[-1] - rate * gradient)
    return np.asarray(history)


def draw_paths(ax, target, prediction, mask):
    # x向前画在纵轴，y向左画在横轴，沿用前一章的空间直觉。
    ax.plot(target[:,1],target[:,0],'o-',color='#16a34a',label='示范路线')
    ax.plot(prediction[:,1],prediction[:,0],'o-',color='#2563eb',label='当前示例输出')
    for a,b,valid in zip(target,prediction,mask):
        ax.plot([a[1],b[1]],[a[0],b[0]],color='#e9aa3c' if valid else '#cbd5e1',lw=2)
    ax.set(xlabel='左向位置 y（米）',ylabel='前向位置 x（米）')
    ax.legend(loc='upper left',fontsize=9);ax.grid(alpha=.15)

def render(values, controls=None):
    section=values['section'];data=compute(values);target=data['target'];prediction=data['prediction'];mask=data['mask']
    fig,ax=plt.subplots(figsize=(7.2,4.3),layout='constrained')
    if section.startswith('2'):
        learned=bias_updates(values['bias'],values['rate'],int(values['updates']))
        prediction=target+np.array([0.,learned[-1]])
        draw_paths(ax,target,prediction,np.ones(10,bool));ax.set_title(f'第{values["updates"]}步：修正一个偏差参数')
        question='“学习一次”到底改变了什么？';keys=('bias','rate','updates');code=bias_updates
        text=f'从偏差 **{values["bias"]:.2f}米** 开始，每一步按 **{values["rate"]:.2f}倍梯度** 修正；经过 **{values["updates"]}步**，当前偏差为 **{learned[-1]:.4f}米**，教学损失为 **{learned[-1]**2:.6f}**。步数为0时尚未执行更新。这里真的更新了一个数值参数，但没有训练正式ResNet。'
        task='先将学习步数归零，再慢慢向右推进，看蓝线怎样接近绿线。'
    elif section.startswith('3'):
        draw_paths(ax,target,prediction,mask)
        ax.scatter(target[~mask,1],target[~mask,0],s=90,facecolors='white',edgecolors='#94a3b8',label='缺失的标签')
        ax.set_title('橙色误差参与统计，灰色不参与')
        question='最后几步没有记录，应该算零误差吗？';keys=('valid','bias');code=compute
        fde='缺失（没有2秒时刻的标签）' if data['fde'] is None else f'{data["fde"]:.3f}米'
        text=f'目前有 **{int(mask.sum())}/10个** 有效标签。平均位置误差ADE为 **{data["ade"]:.3f}米**；固定2秒终点误差FDE为 **{fde}**。缺失标签不能当成零误差。'
        task='把有效点数从10降到7，注意终点误差变为“缺失”，而不是变为零。'
    else:
        draw_paths(ax,target,prediction,mask);ax.set_title('橙色小线段：每个输出点与示范点的距离')
        question='模型的输出离示范有多远？';keys=('bias','noise');code=compute
        text=f'绿线是手工构造的参考示范，蓝线是带偏差的教学输出。橙色线段越长，位置误差越大；当前平均距离为 **{data["ade"]:.3f}米**。这张图不是正式模型的测试成绩。'
        task='把扰动设为0、横向偏差设为1米，验证每一条橙线都是1米。'
    return lesson_page(2,'从示范中学习，先学会看误差','模型学习模仿记录下来的驾驶。先看输出差在哪里，再用一个小参数体验“修正一次”。',controls,question,controls_at(controls,*keys),fig,text,task,code,
        details='正式输出有10个(x,y,v,heading)路点，间隔0.2秒。ADE是有效点距离的平均值；FDE是固定2秒终点的距离。正式训练还包括归一化、速度和航向损失。开环误差小不能直接推出闭环安全。',source_paths=('models/baseline.py','training/baseline.py','dataset/baseline.py'),extra=measured_note(2))
