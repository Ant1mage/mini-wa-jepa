import importlib
folders=['chapter0_hardware','chapter1_bev','chapter2_baseline','chapter3_latent_vicreg','chapter4_joint_world_action','chapter5_stress_testing']
for folder in folders:
    app=importlib.import_module('chapters.'+folder+'.app').app
    outputs,defs=app.run()
    assert 'controls' in defs and 'content' in defs
    print(folder, 'app.run passed', flush=True)
