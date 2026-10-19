"""独立学习入口：控件由上游cell拥有，图示与控件就近展示。"""
import marimo
__generated_with = "0.25.1"
app = marimo.App(width="full")

@app.cell
def _():
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2] if "__file__" in locals() else Path.cwd()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import marimo as mo
    from chapters.chapter2_baseline import ui as lesson
    return lesson, mo

@app.cell
def _(lesson):
    controls = lesson.make_controls()
    return (controls,)

@app.cell
def _(controls, lesson):
    # 先读取上游控件的value，保持清楚的响应式依赖。
    content = lesson.render(controls.value, controls)
    content
    return (content,)

if __name__ == "__main__":
    app.run()
