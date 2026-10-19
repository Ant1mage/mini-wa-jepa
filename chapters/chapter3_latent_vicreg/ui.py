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
        "section": sections("1 用数字表示场景", "2 不能全部一样", "3 也不要全部重复"),
        "scale": slider(0.,2.,1.,"样本之间的差异大小",.05),
        "shared": slider(0.,1.,.2,"两个特征的重复程度",.05),
        "error": slider(0.,1.,.2,"预测的扰动",.05),
        "variance_weight": slider(0,50,25,"方差约束的权重"),
        "batch": slider(4,128,32,"示意样本数量",4),
        "collapse": mo.ui.switch(False,label="让所有样本输出完全一样"),
        "check": mo.ui.switch(True,label="检查样本之间是否还有差异"),
    })

def compute(values):
    """使用 [B,8,4] 小矩阵演示正式 VICReg 的内容中心化统计。"""
    rng = np.random.default_rng(17)  # 固定种子，比较参数时不额外改变噪声。
    batch = int(values["batch"])
    independent = rng.normal(size=(batch, 8, 4))
    shared = rng.normal(size=(batch, 8, 1))
    mixing = values["shared"]
    z = values["scale"] * ((1 - mixing) * independent + mixing * shared)
    target = z + values["scale"] * .1 * rng.normal(size=z.shape)
    predicted = target + values["scale"] * values["error"] * rng.normal(size=z.shape)
    # 正式函数在同一个 token 位置跨样本去均值，排除固定位置编码。
    from training.vicreg import regularization, content_covariance
    import torch
    current_tensor = torch.tensor(z, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.float32)
    var_a, cov_a = regularization(current_tensor)
    var_b, cov_b = regularization(target_tensor)
    variance = float((var_a + var_b) / 2)
    covariance_loss = float((cov_a + cov_b) / 2)
    covariance = content_covariance(current_tensor).numpy()
    mse = float(np.mean((predicted - target)**2))
    total = 25 * mse + values["variance_weight"] * variance + covariance_loss
    return dict(z=z, covariance=covariance, std=np.sqrt(np.diag(covariance) + 1e-4),
                prediction=mse, variance=variance, decorrelation=covariance_loss, total=total)

def cloud_points(values):
    # 只取合成特征的两个通道来画点，不把它们说成物理坐标。
    data = compute(values)
    points = data['z'][:, 0, :2]
    return data, points

def render(values, controls=None):
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
        details='VICReg包括预测对齐、方差约束和协方差约束。此页调用正式的内容协方差函数，但只使用小型合成数组；二维图不是潜变量的物理场景重建。完整训练还需检查有效秩和下游效果。',source_paths=('training/vicreg.py','models/latent.py'),extra=measured_note(3))
