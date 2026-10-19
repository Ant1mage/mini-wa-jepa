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
        "section": sections("1 一张图有多大", "2 一次放几张", "3 分几次再更新"),
        "batch": slider(1, 128, 8, "一次处理的样本数"),
        "accum": slider(1, 8, 1, "累计几个小批次再更新"),
        "tokens": mo.ui.dropdown({"64 块":64,"256 块":256,"1024 块":1024},value="64 块",label="图像分成多少块"),
        "bytes": mo.ui.dropdown({"FP32：每个数4字节":4,"BF16：每个数2字节":2},value="FP32：每个数4字节",label="计算时的数字格式"),
    })

def compute(values):
    """只估算一层显式注意力分数的存储量，不能代替峰值显存实测。"""
    batch, tokens = int(values["batch"]), int(values["tokens"])
    heads, element_bytes = 4, int(values["bytes"])
    # 一个 head 中，每个 token 都要与 N 个 token 计算关联：N × N。
    elements = batch * heads * tokens * tokens
    score_mib = elements * element_bytes / 2**20
    # 梯度累积只改变何时更新参数，不会让单次前向看到更多样本。
    effective_batch = batch * int(values["accum"])
    batches = np.arange(1, 129)
    curve = batches * heads * tokens**2 * element_bytes / 2**20
    return dict(score_mib=score_mib, effective_batch=effective_batch,
                batches=batches, curve=curve)

def image_storage(element_bytes):
    # 四层地图，每层128×128个数；原始uint8每个数占1字节。
    raw_bytes = 4 * 128 * 128
    converted_bytes = raw_bytes * element_bytes
    return raw_bytes / 1024, converted_bytes / 1024


def update_batches(batch, accum):
    # 每次只放batch个样本，累计accum次才更新一次参数。
    return np.arange(1, accum + 1) * batch

def render(values, controls=None):
    section=values['section'];data=compute(values)
    fig,ax=plt.subplots(figsize=(7.2,4.1),layout='constrained')
    if section.startswith('1'):
        raw,converted=image_storage(values['bytes'])
        for i,c in enumerate(['#479d75','#e9aa3c','#ed6f68','#547ba8']):
            ax.add_patch(Rectangle((.1+i*.06,.22+i*.06),.32,.46,facecolor=c,edgecolor='white'))
        ax.text(.38,.08,'4层 × 128行 × 128列',ha='center',fontsize=12)
        ax.text(.73,.67,f'原始地图\n{raw:.0f} KiB',ha='center',fontsize=15)
        ax.text(.73,.33,f'转换后\n{converted:.0f} KiB',ha='center',fontsize=15,color='#2563eb')
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
        details='张量就是带形状和数字格式的数组。训练还要保存参数、中间结果、梯度和优化器状态，因此输入大小不能代表全部显存。单层显式分数矩阵公式：B×4×N×N×每数的字节数；实际SDPA实现与峰值需另行实测。',source_paths=('mini_wa_jepa/hardware.py','mini_wa_jepa/calibration.py'))
