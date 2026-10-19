import marimo._code_mode as cm
async with cm.get_context() as ctx:
    changes = {
        'Hbol': '''from pathlib import Path
import sys
import importlib
_start = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
repo_root = next(p for p in (_start, *_start.parents) if (p / "GOAL.md").exists())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
import marimo as mo
import chapters.learning as learning
importlib.reload(learning)
CHAPTERS = learning.CHAPTERS
from chapters.chapter0_hardware import ui as lesson0
from chapters.chapter1_bev import ui as lesson1
from chapters.chapter2_baseline import ui as lesson2
from chapters.chapter3_latent_vicreg import ui as lesson3
from chapters.chapter4_joint_world_action import ui as lesson4
from chapters.chapter5_stress_testing import ui as lesson5
for _lesson in (lesson0, lesson1, lesson2, lesson3, lesson4, lesson5):
    importlib.reload(_lesson)''',
        'MJUe': '''chapter_select = mo.ui.radio(options=CHAPTERS, value=next(iter(CHAPTERS)), label="选择学习章节")
mo.sidebar([
    mo.md("# Mini-WA-JEPA\\n\\n从一张地图，学到世界模型。"),
    chapter_select,
    mo.md("**每次学一件事**\\n\\n选一个小节 → 动手操作 → 看画面与解释 → 展开中文代码。\\n\\n滑块可拖动，也可用左右方向键逐步查看。切换章节会保留已有参数。"),
])''',
        'BYtC': '''# 控件现在与对应图示在同一小节内展示；这里不再堆放整章参数。
pass''',
        'RGSE': '''selected = chapter_select.value
_lesson, _controls = {
    "ch0": (lesson0, ch0_controls), "ch1": (lesson1, ch1_controls),
    "ch2": (lesson2, ch2_controls), "ch3": (lesson3, ch3_controls),
    "ch4": (lesson4, ch4_controls), "ch5": (lesson5, ch5_controls),
}[selected]
# 控件由上游cell拥有；此处读取value并显示相同控件，避免操作后重置。
content = _lesson.render(_controls.value, _controls)
content''',
    }
    for cid, code in changes.items():
        old = ctx.cells[cid].code
        print('Updating', cid, 'previous chars', len(old))
        ctx.edit_cell(cid, code)
    ctx.run_cell('Hbol')
    ctx.run_cell('MJUe')
    ctx.run_cell('RGSE')
