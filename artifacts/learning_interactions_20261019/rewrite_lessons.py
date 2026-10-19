"""一次性保留旧 compute 数学实验，重组0–5章的简单原生交互。"""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HEADER='''"""面向初学者的分节交互实验；合成演示与正式记录分别标注。"""
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from chapters.learning import (
    ROOT, slider, sections, lesson_page, controls_at, map_axes, bev_scene, measured_note,
)
'''


def write(folder,controls,helpers,render,imports=''):
    path=ROOT/'chapters'/folder/'ui.py'
    old=path.read_text();tree=ast.parse(old)
    compute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute')
    source=HEADER+imports+'\n\n'+controls+'\n\n'+ast.get_source_segment(old,compute)+'\n\n'+helpers+'\n\n'+render+'\n'
    ast.parse(source);path.write_text(source)


write('chapter0_hardware', '''def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 一张图有多大", "2 一次放几张", "3 分几次再更新"),
        "batch": slider(1, 128, 8, "一次处理的样本数"),
        "accum": slider(1, 8, 1, "累计几个小批次再更新"),
        "tokens": mo.ui.dropdown({"64 块":64,"256 块":256,"1024 块":1024},value="64 块",label="图像分成多少块"),
        "bytes": mo.ui.dropdown({"FP32：每个数4字节":4,"BF16：每个数2字节":2},value="FP32：每个数4字节",label="计算时的数字格式"),
    })''', '''def image_storage(element_bytes):
    # 四层地图，每层128×128个数；原始uint8每个数占1字节。
    raw_bytes = 4 * 128 * 128
    converted_bytes = raw_bytes * element_bytes
    return raw_bytes / 1024, converted_bytes / 1024


def update_batches(batch, accum):
    # 每次只放batch个样本，累计accum次才更新一次参数。
    return np.arange(1, accum + 1) * batch''', '''def render(values, controls=None):
    section=values['section'];data=compute(values)
    fig,ax=plt.subplots(figsize=(7.2,4.1),layout='constrained')
    if section.startswith('1'):
        raw,converted=image_storage(values['bytes'])
        for i,c in enumerate(['#479d75','#e9aa3c','#ed6f68','#547ba8']):
            ax.add_patch(Rectangle((.1+i*.06,.22+i*.06),.32,.46,facecolor=c,edgecolor='white'))
        ax.text(.38,.08,'4层 × 128行 × 128列',ha='center',fontsize=12)
        ax.text(.73,.67,f'原始地图\\n{raw:.0f} KiB',ha='center',fontsize=15)
        ax.text(.73,.33,f'转换后\\n{converted:.0f} KiB',ha='center',fontsize=15,color='#2563eb')
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
        question='同一张地图，为什么换个数字格式就变大了？'
        keys=('bytes',);code=image_storage
        text=f'地图的格子数量没变。每个数从 **1字节** 变为 **{values["bytes"]}字节**，所以从 **{raw:.0f} KiB** 变为 **{converted:.0f} KiB**。这是输入数组大小，不是训练峰值。'
        task='在FP32和BF16之间切换，观察格子数量不变、存储量怎样变化。'
    elif section.startswith('2'):
        count=int(values['batch']);shown=min(count,16)
        for i in range(shown):
            x=(i%8)*.115;y=.58-(i//8)*.27
            ax.add_patch(Rectangle((x,y),.085,.18,facecolor='#5b9bd5',edgecolor='white'))
            ax.text(x+.0425,y+.09,str(i+1),ha='center',va='center',color='white')
        ax.text(.43,.16,f'一次处理 {count} 张'+('（仅画前16张）' if count>16 else ''),ha='center',fontsize=13)
        ax.set(xlim=(-.03,1),ylim=(0,1));ax.axis('off')
        question='一次多放几张图，会发生什么？';keys=('batch','tokens');code=compute
        text=f'这次同时处理 **{count}个样本**。每个样本拆成 **{values["tokens"]}个Token（图块特征）**。单层显式注意力分数矩阵的理论存储量为 **{data["score_mib"]:.3f} MiB**，随Batch线性增长、随Token数量平方增长。'
        task='先只把样本数翻倍；再将图块数从64改到256，比较两种增长速度。'
    else:
        cumulative=update_batches(values['batch'],values['accum'])
        for i,total in enumerate(cumulative):
            ax.barh(i,values['batch'],left=i*values['batch'],color='#5b9bd5')
            ax.text(total+.1,i,f'累计 {total} 张',va='center')
        ax.set(yticks=np.arange(len(cumulative)),yticklabels=[f'小批次 {i+1}' for i in range(len(cumulative))],xlabel='累计处理的样本数')
        ax.set_title('多次小批次 → 一次参数更新')
        question='放不下一大批时，能不能分几次处理？';keys=('batch','accum');code=update_batches
        text=f'每次放 **{values["batch"]}张**，做 **{values["accum"]}次**后更新参数，共累计 **{data["effective_batch"]}张**。单次前向仍只看{values["batch"]}张；VICReg的批内统计也仍按这个小批次计算。'
        task='只增加累计次数，观察更新前累计的样本增多，但单次Batch没有变。'
    return lesson_page(0,'先认识硬件与张量内存','先从一张地图开始，再理解一批样本和一次更新。此处只做可计算的示意，不启动训练。',controls,question,controls_at(controls,*keys),fig,text,task,code,
        details='张量就是带形状和数字格式的数组。训练还要保存参数、中间结果、梯度和优化器状态，因此输入大小不能代表全部显存。单层显式分数矩阵公式：B×4×N×N×每数的字节数；实际SDPA实现与峰值需另行实测。',source_paths=('mini_wa_jepa/hardware.py','mini_wa_jepa/calibration.py'))''')


write('chapter1_bev', '''def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 移动车辆", "2 拆开四层", "3 像素里的数字"),
        "x": slider(-25.,25.,12.,"目标在车前多远（米）",.5),
        "y": slider(-15.,15.,-2.,"目标在车左多远（米）",.5),
        "heading": slider(-180,180,15,"目标车头朝向（度）",5),
        "road_width": slider(6.,20.,12.,"道路宽度（米）",.5),
        "layer": mo.ui.radio({"道路与标线":0,"静态边界":1,"其他车":2,"可见性":3},value="其他车",label="查看哪一层",inline=True),
        "pixel_u": slider(0,127,68,"像素列号 u"),
        "pixel_v": slider(0,127,40,"像素行号 v"),
    })''', '''def inspect_pixel(bev, u, v):
    # 图像先按行v、再按列u索引；取出这个格子的四层数值。
    channels = bev[:, v, u]
    # 像素间隔0.5米；中心在63.5，前方对应更小的行号。
    metres = pixels_to_ego(np.array([u, v]))
    return channels, metres''', '''def render(values, controls=None):
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
        axes[1].imshow(bev[channel],cmap='gray',vmin=0,vmax=255);axes[1].set_title(names[channel]);axes[1].axis('off')
        keys=('layer','road_width');question='模型收到的是一张彩色图片吗？';code=inspect_pixel
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
        details='坐标约定：x向前、y向左；像素u向右、v向下。中心为(63.5,63.5)。u=63.5−2y，v=63.5−2x。此处为合成语义地图，不是相机识别精度的演示。',source_paths=('chapters/chapter1_bev/src/semantic_bev.py','env/semantic_bev.py'))''',
      'from chapters.chapter1_bev.src.semantic_bev import ego_to_pixels, pixels_to_ego\n')


write('chapter2_baseline', '''def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 示范与误差", "2 修正一次偏差", "3 缺失标签"),
        "speed": slider(2.,15.,6.,"示范速度（米/秒）",.5),
        "bias": slider(-2.,2.,1.,"横向偏差（米）",.1),
        "noise": slider(0.,1.,.15,"点的位置扰动（米）",.05),
        "valid": slider(1,10,10,"有真实记录的未来点数"),
        "seed": slider(0,20,7,"扰动种子"),
        "updates": slider(0,12,0,"学习步数（拖动或按左右键）"),
        "rate": slider(.05,.5,.2,"每一步修正的幅度",.05),
    })''', '''def bias_updates(initial_bias, rate, steps):
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
    ax.legend(loc='upper left',fontsize=9);ax.grid(alpha=.15)''', '''def render(values, controls=None):
    section=values['section'];data=compute(values);target=data['target'];prediction=data['prediction'];mask=data['mask']
    fig,ax=plt.subplots(figsize=(7.2,4.3),layout='constrained')
    if section.startswith('2'):
        learned=bias_updates(values['bias'],values['rate'],int(values['updates']))
        prediction=target+np.array([0.,learned[-1]])
        draw_paths(ax,target,prediction,np.ones(10,bool));ax.set_title(f'第{values["updates"]}步：修正一个偏差参数')
        question='“学习一次”到底改变了什么？';keys=('bias','rate','updates');code=bias_updates
        text=f'从偏差 **{values["bias"]:.2f}米** 开始，经过 **{values["updates"]}步**，当前偏差为 **{learned[-1]:.4f}米**，教学损失为 **{learned[-1]**2:.6f}**。这里真的更新了一个数值参数，但没有训练正式ResNet。'
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
        details='正式输出有10个(x,y,v,heading)路点，间隔0.2秒。ADE是有效点距离的平均值；FDE是固定2秒终点的距离。正式训练还包括归一化、速度和航向损失。开环误差小不能直接推出闭环安全。',source_paths=('models/baseline.py','training/baseline.py','dataset/baseline.py'),extra=measured_note(2))''')


write('chapter3_latent_vicreg', '''def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 用数字表示场景", "2 不能全部一样", "3 也不要全部重复"),
        "scale": slider(0.,2.,1.,"样本之间的差异大小",.05),
        "shared": slider(0.,1.,.2,"两个特征的重复程度",.05),
        "error": slider(0.,1.,.2,"预测的扰动",.05),
        "variance_weight": slider(0,50,25,"方差约束的权重"),
        "batch": slider(4,128,32,"示意样本数量",4),
        "collapse": mo.ui.switch(False,label="让所有样本输出完全一样"),
        "check": mo.ui.switch(True,label="检查样本之间是否还有差异"),
    })''', '''def cloud_points(values):
    # 只取合成特征的两个通道来画点，不把它们说成物理坐标。
    data = compute(values)
    points = data['z'][:, 0, :2]
    return data, points''', '''def render(values, controls=None):
    section=values['section'];local=dict(values)
    if section.startswith('2'):
        local['scale']=0. if values['collapse'] else 1.
        local['variance_weight']=25 if values['check'] else 0
    if section.startswith('3'):local['scale']=1.
    data,points=cloud_points(local)
    fig,ax=plt.subplots(figsize=(7.2,4.3),layout='constrained')
    ax.scatter(points[:,0],points[:,1],c=np.arange(len(points)),cmap='viridis',s=55,alpha=.8)
    ax.axhline(0,color='#cbd5e1',lw=1);ax.axvline(0,color='#cbd5e1',lw=1)
    ax.set(xlabel='示意特征1（没有米的含义）',ylabel='示意特征2',xlim=(-4,4),ylim=(-4,4))
    if section.startswith('1'):
        ax.set_title(f'{len(points)}个示意样本 → 每个样本的一组数字')
        question='“特征”是什么？为什么不直接理解成一张图片？';keys=('scale','batch');code=cloud_points
        text='每个圆点代表一个示意样本，这里只画两个特征数值。调整样本差异，点的位置会变化。正式编码器每个空间位置输出256个可学习数值，不能预先把某一维叫作“车的位置”。'
        task='将差异大小逐渐减小，观察不同颜色的点是否越来越难区分。'
    elif section.startswith('2'):
        ax.set_title('所有点重叠时：不同场景无法再区分' if values['collapse'] else '保留不同样本之间的区别')
        question='所有场景都回答“0”，是不是也能拿到很小的预测误差？';keys=('collapse','check');code=compute
        text=f'当前预测误差为 **{data["prediction"]:.5f}**，样本差异惩罚为 **{data["variance"]:.3f}**，总损失为 **{data["total"]:.3f}**。'+('点都重叠了，预测与目标都为零，单看预测误差会被骗；开启差异检查才能识别这个坏解。' if values['collapse'] else '先开启“让所有样本输出完全一样”，再开关差异检查，比较两种评分。')
        task='让所有输出一样，先关掉差异检查，再打开，看看“零预测误差”为什么不够。'
    else:
        ax.plot([-4,4],[-4,4],'--',color='#e9aa3c');ax.set_title('挤到对角线上：两个特征在重复记同一件事')
        question='有很多数字，就一定记录了很多不同信息吗？';keys=('shared',);code=compute
        text=f'当前特征共享系数为 **{values["shared"]:.2f}**，协方差惩罚为 **{data["decorrelation"]:.3f}**。越靠近1，特征越容易重复，点云会挤向对角线。这里只改变合成数据，不是在实时训练网络。'
        task='把重复程度调到1，再调回0，比较一条线和一团点的差别。'
    return lesson_page(3,'理解特征为什么不能塌缩','用一团可操作的点理解：既要预测得接近，也要让表示保留区别、减少重复。所有点均为合成示意。',controls,question,controls_at(controls,*keys),fig,text,task,code,
        details='VICReg包括预测对齐、方差约束和协方差约束。此页调用正式的内容协方差函数，但只使用小型合成数组；二维图不是潜变量的物理场景重建。完整训练还需检查有效秩和下游效果。',source_paths=('training/vicreg.py','models/latent.py'),extra=measured_note(3))''')


write('chapter4_joint_world_action', '''def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 先看四帧历史", "2 看懂两个任务", "3 路线与执行"),
        "speed": slider(0.,30.,10.,"当前速度（米/秒）",.5),
        "yaw": slider(-1.,1.,.15,"转向变化率（弧度/秒）",.05),
        "residual": slider(-2.,2.,0.,"教学中的横向修正（米）",.1),
        "time": slider(1,10,1,"沿路线查看第几步"),
        "frame": slider(0,3,3,"历史帧：0最早，3当前"),
        "phase": slider(0,4,0,"计算过程：逐步向右查看"),
    })''', '''def history_frame(index):
    # 四帧相隔0.1秒，前车每帧接近0.5米。
    return bev_scene(actor_forward=13.5 - .5 * index)


def process_boxes(ax, step):
    labels=['四帧历史','共享特征','估计未来变化','给出驾驶路线','控制器执行']
    xy=[(.03,.45),(.24,.45),(.47,.7),(.47,.2),(.76,.2)]
    for i,(label,(x,y)) in enumerate(zip(labels,xy)):
        ax.add_patch(Rectangle((x,y),.19,.15,facecolor='#2563eb' if i==step else '#e2e8f0',edgecolor='white'))
        ax.text(x+.095,y+.075,label,ha='center',va='center',color='white' if i==step else '#334155',fontsize=10)
    for a,b in [(0,1),(1,2),(1,3),(2,3),(3,4)]:
        x,y=xy[a];u,v=xy[b]
        ax.annotate('',xy=(u,v+.075),xytext=(x+.19,y+.075),arrowprops=dict(arrowstyle='->',color='#64748b'))
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')''', '''def render(values, controls=None):
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
        details='联合模型保留64个空间Token，世界分支预测特征，轨迹分支给出10个(x,y,v,heading)路点。Action-only对照使用相同结构但关闭世界损失，不能把流程图中的框当成真实神经元或已测安全保证。',source_paths=('models/joint.py','training/joint_loss.py'),extra=measured_note(4))''')


write('chapter5_stress_testing', '''def make_controls():
    return mo.ui.dictionary({
        "section": sections("1 挡住一块画面", "2 丢掉边界信息", "3 画面卡住了"),
        "depth": slider(0.,24.,16.,"遮挡向前延伸多远（米）",1.),
        "width": slider(2.,24.,10.,"遮挡有多宽（米）",1.),
        "boundary": slider(0.,1.,.3,"边界信息丢失比例",.05),
        "dropped": slider(0,3,2,"最近连续丢失几帧"),
        "frame": slider(0,3,0,"时间进度：第几帧（拖动或按左右键）"),
        "seed": slider(0,20,7,"边界缺失随机种子"),
    })''', '''def received_frame_index(frame, dropped):
    # 最近dropped帧丢失时，接收端保持最近一次成功收到的帧。
    last_received = 3 - dropped
    return min(frame, last_received)''', '''def render(values, controls=None):
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
        text=f'真实时间来到 **{i*.1:.1f}秒**，输入仍来自 **{source*.1:.1f}秒**，已经过期 **{age:.1f}秒**。'+('红车已经移动，但右图保持旧位置。' if age else '当前这一帧正常收到。')
        task='设置连续丢失2帧，再把时间进度从0拖到3，观察右图何时停止变化。'
    return lesson_page(5,'理解看不见、边界缺失和画面卡住','每次只改变一种输入问题，左右对照观察。这里是可重复的合成实验，不会因拖动而重跑驾驶评测。',controls,question,controls_at(controls,*keys),fig,text,task,code,
        details='遮挡是空间信息缺失，丢帧是时间信息陈旧，两者并不一样。正式压力测试另行比较冻结模型的轨迹误差与闭环表现；不能从本页两张输入图直接推出某个模型更鲁棒。',source_paths=('evaluation/degradation.py','configs/stage5.py'),extra=measured_note(5))''',
      'from chapters.chapter5_stress_testing.src.degradation import (apply_front_blindspot, apply_boundary_loss, apply_temporal_frame_dropout)\n')

print('Rebuilt Chapter0–5, preserving original compute functions and all src modules.')
