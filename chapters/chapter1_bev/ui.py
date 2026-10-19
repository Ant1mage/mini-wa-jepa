"""面向初学者的分节交互实验；合成演示与正式记录分别标注。"""
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from chapters.learning import (
    ROOT, slider, sections, lesson_page, controls_at, map_axes, bev_scene, measured_note,
)
from chapters.chapter1_bev.src.semantic_bev import ego_to_pixels, pixels_to_ego


def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 移动车辆", "2 拆开四层", "3 像素里的数字"),
        "x": slider(-25.,25.,12.,"目标在车前多远（米）",.5),
        "y": slider(-15.,15.,-2.,"目标在车左多远（米）",.5),
        "heading": slider(-180,180,15,"目标车头朝向（度）",5),
        "road_width": slider(6.,20.,12.,"道路宽度（米）",.5),
        "layer": mo.ui.radio({"道路与标线":0,"静态边界":1,"其他车":2,"可见性":3},value="其他车",label="查看哪一层",inline=True),
        "pixel_u": slider(0,127,68,"像素列号 u"),
        "pixel_v": slider(0,127,40,"像素行号 v"),
    })

def compute(values):
    """同一组物理坐标同时驱动 BEV 中的车辆位置、箭头和坐标读数。"""
    point = np.array([[values["x"], values["y"]]], dtype=float)
    # 物理坐标 x 向前、y 向左；像素坐标 u 向右、v 向下。
    pixel = ego_to_pixels(point)[0]
    theta = np.deg2rad(values["heading"])
    arrow_end = point + 3 * np.array([[np.cos(theta), np.sin(theta)]])
    # 使用同一个几何转换函数画箭头，避免显示方向和坐标读数矛盾。
    arrow_pixel = ego_to_pixels(arrow_end)[0]
    bev = bev_scene(values["x"], values["y"], values["road_width"])
    restored = pixels_to_ego(pixel)
    return dict(bev=bev, pixel=pixel, arrow_pixel=arrow_pixel, restored=restored)

def inspect_pixel(bev, u, v):
    # 图像先按行v、再按列u索引；取出这个格子的四层数值。
    channels = bev[:, v, u]
    # 像素间隔0.5米；中心在63.5，前方对应更小的行号。
    metres = pixels_to_ego(np.array([u, v]))
    return channels, metres


def one_layer(bev, channel):
    # 第一个维度选择通道；这层仍是一张128×128的数字地图。
    return bev[channel]

def render(values, controls=None):
    data=compute(values);section=values['section'];bev=data['bev']
    if section.startswith('1'):
        fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained');map_axes(ax,bev,'白色：自车 · 红色：目标车')
        ax.annotate('',xy=data['arrow_pixel'],xytext=data['pixel'],arrowprops=dict(arrowstyle='->',color='#facc15',lw=3))
        ax.text(4,12,'↑ 车前方',color='white');ax.text(4,119,'← 车左侧',color='white')
        u,v=data['pixel'];keys=('x','y','heading');question='车在前方或左侧，地图里会怎样移动？';code=compute
        text=f'目标现在位于 **前方{values["x"]:.1f}米、左侧{values["y"]:.1f}米**，对应像素 **(列{u:.1f}，行{v:.1f})**。x增加会向图上方移动，y增加会向图左侧移动；黄色箭头只表示目标朝向。'
        task='把x和y都设为0，让目标来到自车中心，再只改变朝向。'
    elif section.startswith('2'):
        fig,axes=plt.subplots(1,2,figsize=(8,4.3),layout='constrained');map_axes(axes[0],bev,'合成地图：把四层叠在一起')
        channel=int(values['layer']);names=['道路与标线','静态边界要素','其他交通参与者','可见性']
        axes[1].imshow(one_layer(bev,channel),cmap='gray',vmin=0,vmax=255);axes[1].set_title(names[channel]);axes[1].axis('off')
        keys=('layer','road_width');question='模型收到的是一张彩色图片吗？';code=one_layer
        meanings=['道路、斑马线、车道线等；车道线没有独立通道。','人行道、围栏、交通标志等；不是其他车辆。','其他车或行人；白色自车标记只是画面叠加，不写入这一层。','255表示可见，0表示不可见；不可见不等于确认没有障碍物。']
        text=f'你正在单独查看 **第{channel}层：{names[channel]}**。{meanings[channel]} 模型真正收到的是四层数组；左图的颜色只是为了方便人阅读。'
        task='先选“其他车”，再选“道路与标线”，看看同一个位置在不同层里是否相同。'
    else:
        u,v=int(values['pixel_u']),int(values['pixel_v']);numbers,xy=inspect_pixel(bev,u,v)
        fig,axes=plt.subplots(1,2,figsize=(8,4.3),layout='constrained');map_axes(axes[0],bev,'选中的格子')
        axes[0].plot(u,v,'s',markerfacecolor='none',markeredgecolor='#facc15',markersize=12,markeredgewidth=2)
        axes[1].barh(['道路','边界','其他车','可见性'],numbers,color=['#479d75','#e9aa3c','#ed6f68','#547ba8']);axes[1].set_xlim(0,280);axes[1].set_xlabel('这个格子存的数值')
        keys=('pixel_u','pixel_v');question='点到地图里的一个格子，实际读出了什么？';code=inspect_pixel
        text=f'当前索引是 **bev[:, {v}, {u}]**，四层数值依次为 **{numbers.tolist()}**。这个像素中心对应自车坐标 **前方{xy[0]:.2f}米、左侧{xy[1]:.2f}米**。'
        task='将u、v移动到红色车辆处，观察“其他车”这一层怎样从0变成255。'
    return lesson_page(1,'把驾驶场景变成 BEV 数组','BEV就是从车顶往下看的地图。本项目用128×128个格子覆盖64×64米，每格间隔0.5米。',controls,question,controls_at(controls,*keys),fig,text,task,code,
        details='坐标约定：x向前、y向左；像素u向右、v向下。中心为(63.5,63.5)。u=63.5−2y，v=63.5−2x。此处为合成语义地图，不是相机识别精度的演示。',source_paths=('chapters/chapter1_bev/src/semantic_bev.py','env/semantic_bev.py'))
