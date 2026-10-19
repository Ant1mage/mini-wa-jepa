"""面向初学者的分节交互实验；合成演示与正式记录分别标注。"""
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from chapters.learning import (
    ROOT, slider, sections, lesson_page, controls_at, map_axes, bev_scene, measured_note,
)
from chapters.chapter5_stress_testing.src.degradation import (apply_front_blindspot, apply_boundary_loss, apply_temporal_frame_dropout)


def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 挡住一块画面", "2 丢掉边界信息", "3 画面卡住了"),
        "depth": slider(0.,24.,16.,"遮挡向前延伸多远（米）",1.),
        "width": slider(2.,24.,10.,"遮挡有多宽（米）",1.),
        "boundary": slider(0.,1.,.3,"边界信息丢失比例",.05),
        "dropped": slider(0,3,2,"最近连续丢失几帧"),
        "frame": slider(0,3,3,"时间进度：第几帧（拖动或按左右键）"),
        "seed": slider(0,20,7,"边界缺失随机种子"),
    })

def compute(values):
    """先模拟传感器连续丢帧，再向收到的历史注入空间退化。"""
    # 合成前车每0.1秒接近1米，便于看清变化；不能重复四张一样的图。
    clean = np.stack([bev_scene(actor_forward=13.5 - 1. * i) for i in range(4)])
    count = int(values["dropped"])
    dropped_indices = tuple(range(4 - count, 4)) if count else ()
    frozen = apply_temporal_frame_dropout(clean, dropped_indices, "hold_previous")
    spatial = apply_front_blindspot(frozen, width_m=values["width"], depth_m=values["depth"])
    degraded = apply_boundary_loss(spatial, drop_channels=(1,),
                                   drop_rate=values["boundary"], seed=int(values["seed"]))
    index = int(values["frame"])
    # 合成数组真值已知，因此可以对照观测差异；这不是驾驶轨迹误差 ADE。
    changed = np.any(clean[index] != degraded[index], axis=0)
    invisible = degraded[index, 3] == 0
    return dict(clean=clean, frozen=frozen, degraded=degraded, changed=changed,
                invisible=invisible, index=index, dropped_indices=dropped_indices)

def received_frame_index(frame, dropped):
    # 最近dropped帧丢失时，接收端保持最近一次成功收到的帧。
    last_received = 3 - dropped
    return min(frame, last_received)

def render(values, controls=None):
    section=values['section'];local=dict(values)
    if section.startswith('1'):local.update(boundary=0.,dropped=0,frame=3)
    elif section.startswith('2'):local.update(depth=0.,dropped=0,frame=3)
    else:local.update(depth=0.,boundary=0.)
    data=compute(local);i=data['index'];fig,axes=plt.subplots(1,2,figsize=(8,4.3),layout='constrained')
    map_axes(axes[0],data['clean'][i],'实际场景（教学真值）')
    map_axes(axes[1],data['degraded'][i],'模型能收到的输入')
    if section.startswith('1'):
        question='挡住前方以后，“看不见”和“没有车”一样吗？';keys=('depth','width');code=compute
        text=f'右图有 **{data["invisible"].mean():.1%}** 区域不可见。遮挡同时清空了对应的可见性信息，左图里的车仍然存在；不能把右图看不到红车理解为道路已经安全。'
        task='把遮挡深度从0慢慢调到16米，观察红车怎样从可见变为被遮住。'
    elif section.startswith('2'):
        question='边界信息缺了一些，车辆也会一起消失吗？';keys=('boundary','seed');code=compute
        text=f'当前删除 **{values["boundary"]:.0%}** 的静态边界像素。这个实验只改边界通道；道路、其他车辆和可见性不应被一起删除。边界通道包括人行道、围栏、标志等，车道线属于道路通道。'
        task='把比例调到100%，确认右图边界消失，红色车辆仍在。'
    else:
        source=received_frame_index(i,int(values['dropped']));age=(i-source)*.1
        question='真实世界还在移动，模型为什么看到一张旧画面？';keys=('dropped','frame');code=received_frame_index
        axes[0].set_title(f'真实时间：{i*.1:.1f}秒');axes[1].set_title(f'收到的画面来自：{source*.1:.1f}秒')
        text=f'本段设置为 **最近{values["dropped"]}帧没有收到**。真实时间来到 **{i*.1:.1f}秒**，输入仍来自 **{source*.1:.1f}秒**，已经过期 **{age:.1f}秒**。'+('红车已经移动，但右图保持旧位置。' if age else '当前这一帧正常收到；继续向后查看才会遇到丢帧。')
        task='设置连续丢失2帧，再把时间进度从0拖到3，观察右图何时停止变化。'
    return lesson_page(5,'理解看不见、边界缺失和画面卡住','每次只改变一种输入问题，左右对照观察。这里是可重复的合成实验，不会因拖动而重跑驾驶评测。',controls,question,controls_at(controls,*keys),fig,text,task,code,
        details='遮挡是空间信息缺失，丢帧是时间信息陈旧，两者并不一样。正式压力测试另行比较冻结模型的轨迹误差与闭环表现；不能从本页两张输入图直接推出某个模型更鲁棒。',source_paths=('evaluation/degradation.py','configs/stage5.py'),extra=measured_note(5))
