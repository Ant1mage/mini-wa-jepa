"""教学页面的展示工具；不参与正式模型训练，也不写实验结果。"""
import inspect
from pathlib import Path

import marimo as mo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = {
    "Chapter 0 · 先认识硬件与张量内存": "ch0",
    "Chapter 1 · 把驾驶场景变成 BEV 数组": "ch1",
    "Chapter 2 · 从专家轨迹学会计算误差": "ch2",
    "Chapter 3 · 理解潜变量预测与 VICReg": "ch3",
    "Chapter 4 · 把世界预测与轨迹预测接起来": "ch4",
    "Chapter 5 · 设计可复现的感知退化实验": "ch5",
    "Chapter 6 · 不同驾驶方案会带来什么后果？": "ch6",
}

# 使用系统已有中文字体，避免图中的中文变成方框。
_available = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [
    name for name in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    if name in _available
]
plt.rcParams["axes.unicode_minus"] = False


def slider(start, stop, value, label, step=1):
    """轻量教学实验随拖动更新；左右方向键可逐步查看。"""
    return mo.ui.slider(start=start, stop=stop, step=step, value=value,
                        label=label, show_value=True, include_input=True, debounce=False, full_width=True)


def figure(fig):
    """先转成页面内容，再释放 Matplotlib figure，防止反复拖动累积内存。"""
    result = mo.as_html(fig)
    plt.close(fig)
    return result


def annotated_code(function):
    """展示正在生成当前图表的实际代码，而非另写一份无法运行的伪代码。"""
    module = inspect.getmodule(function)
    return mo.vstack([
        mo.md("### 这张图是怎样算出来的？\n\n下面是当前实验实际调用的函数，"
              "可以对照中文注释逐行阅读；`np` 等名称来自本章模块顶部的导入。\n\n```python\n"
              + inspect.getsource(function) + "\n```\n\n"
              "在项目根目录的 Python 环境中，可以这样单独运行这个计算：\n\n```python\n"
              f"from {function.__module__} import make_controls, compute\n\n"
              "values = make_controls().value  # 取得默认实验参数\n"
              "result = compute(values)       # 只计算，不启动训练或模拟器\n"
              "print(result.keys())           # 查看返回了哪些数组和指标\n```"),
        mo.accordion({"本章完整 Python 源码（含导入、计算与中文注释）":
                      mo.md("```python\n" + inspect.getsource(module) + "\n```")}),
    ])


def source_files(*paths):
    """正式源码只读展示，路径与教学计算分开，避免混淆示例和训练实现。"""
    return mo.accordion({f"阅读源码：{path}": mo.md(
        f"仓库路径：`{path}`\n\n```python\n"
        + (ROOT / path).read_text() + "\n```") for path in paths})


def glossary(rows):
    return mo.md("### 先认识这些词\n\n| 术语 | 用初学者能理解的话解释 |\n"
                 "|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows))


def bev_scene(actor_forward=12.0, actor_left=-2.0, road_width=12.0):
    """合成场景采用正式 schema；自车只画标记，不放入参与者通道。"""
    rows, cols = np.indices((128, 128))
    left = (63.5 - cols) * 0.5
    forward = (63.5 - rows) * 0.5
    bev = np.zeros((4, 128, 128), dtype=np.uint8)
    bev[0, np.abs(left) < road_width / 2] = 255
    bev[1, np.abs(np.abs(left) - road_width / 2) < 0.6] = 255
    bev[2, (np.abs(forward - actor_forward) < 2.0)
        & (np.abs(left - actor_left) < 1.0)] = 255
    bev[3] = 255  # 所有区域可见；后续遮挡时将受损区域置零。
    return bev


def sections(*names):
    """仅用原生标签页导航；内容由独立消费 cell 按当前页生成。"""
    return mo.ui.tabs({name: mo.md('') for name in names}, value=names[0])


def controls_at(controls, *names):
    return [controls[name] for name in names] if controls is not None else []


def lesson_page(number, title, intro, controls, question, widgets, fig, explanation,
                task, code, details='', source_paths=(), extra=None):
    """每次只展示一个小实验，实际控件保持在 notebook 的拥有者 cell 中。"""
    sidebar=mo.vstack([mo.md('### 动手试试'),*widgets,
        mo.md('拖动即更新；滑块聚焦后按 **← / →** 可逐步查看。')],gap=.8)
    panels={'继续了解原理':mo.md(details or intro)}
    if source_paths:
        panels['阅读正式实现（只读）']=source_files(*source_paths)
    if extra is not None:panels['查看本项目的实测记录']=extra
    return mo.vstack([
        mo.md(f'# Chapter {number} · {title}\n\n{intro}'),
        controls['section'] if controls is not None else mo.md(''),
        mo.md(f'## {question}'),
        mo.hstack([sidebar,figure(fig)],widths=[1,2],align='start',gap=1.5,wrap=True),
        mo.callout(mo.md(explanation),kind='neutral'),
        mo.md('**试一试：** '+task),
        mo.accordion({'看看这一步的代码 · 中文注释':mo.md('```python\n'+inspect.getsource(code)+'\n```')}),
        mo.accordion(panels),
    ],gap=.8)


def map_axes(ax, bev, title=''):
    from chapters.chapter1_bev.src.semantic_bev import colorize_bev_4ch
    ax.imshow(colorize_bev_4ch(bev))
    ax.plot(63.5,63.5,marker='^',color='white',markersize=10)
    ax.set_title(title,fontsize=12);ax.set_xlim(0,127);ax.set_ylim(127,0)
    ax.set_xticks([]);ax.set_yticks([])


def measured_note(stage):
    """实测表只在折叠区域读取；不会把教学滑块当成正式模型配置。"""
    from mini_wa_jepa.visualization.reports import load_stage_runs
    rows,_,errors=load_stage_runs(ROOT,stage)
    keys=('model','ablation','split','seed','ade_m','fde_m','route_completion',
          'latent_mse','retrieval_accuracy','source')
    return mo.vstack([mo.md('这里读取磁盘中的真实报告；上方教学操作不会改写这些结果。'),
        mo.ui.table([{k:r.get(k) for k in keys} for r in rows],selection=None,page_size=5)
        if rows else mo.md('尚无可读取的实测结果，保持缺失。'),
        mo.md('\n'.join(errors)) if errors else mo.md('')])
