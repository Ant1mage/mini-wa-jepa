import marimo._code_mode as cm
async with cm.get_context() as ctx:
    old=ctx.cells['Hbol'].code
    new=old.replace('for _lesson in (lesson0, lesson1, lesson2, lesson3, lesson4, lesson5):', 'from chapters.chapter6_action_conditioned import ui as lesson6\nfor _lesson in (lesson0, lesson1, lesson2, lesson3, lesson4, lesson5, lesson6):')
    assert new!=old
    ctx.edit_cell('Hbol',new)
    old=ctx.cells['RGSE'].code
    new=old.replace('"ch5": (lesson5, ch5_controls),','"ch5": (lesson5, ch5_controls),\n    "ch6": (lesson6, ch6_controls),')
    assert new!=old
    ctx.edit_cell('RGSE',new)
    assert ctx.cells['BYtC'].code.strip().endswith('pass')
    ctx.delete_cell('BYtC')
    cid=ctx.create_cell('ch6_controls = lesson6.make_controls()')
    print('Chapter6 controls cell:',cid)
    ctx.run_cell('Hbol')
