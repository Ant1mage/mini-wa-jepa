import marimo
__generated_with='0.25.1'
app=marimo.App(width='full')

@app.cell
def _():
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[2] if '__file__' in locals() else Path.cwd()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from chapters.chapter7_candidate_planning import ui as lesson
    return (lesson,)

@app.cell
def _(lesson):
    controls=lesson.make_controls()
    return (controls,)

@app.cell
def _(controls,lesson):
    content=lesson.render(controls.value,controls)
    content
    return (content,)

if __name__=='__main__':app.run()
