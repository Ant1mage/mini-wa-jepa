"""Codex acceptance diagnostic: does an observed roadside context expose 2s off-road events?"""
from pathlib import Path
import hashlib
import json
import sys
import traceback
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from configs.stage2 import Stage2Config
from configs.stage7 import Stage7Config
from dataset.recording import vehicle_state
from dataset.action_branches import remaining_command
from controller.trajectory_tracker import TrajectoryTracker
from env.data_v2 import configure_egl, make_env, NativeBEVProvider
from env.expert import NavigationExpert
from mini_wa_jepa.provenance import provenance, write_json
from scripts.preflight_risk_collection import make_risk_candidates, scene_state, compute_risk_targets

out=Path('artifacts/restart_stage7/offroad_context_probe_001');out.mkdir(exist_ok=False)
cfg=Stage2Config();stage=Stage7Config();_,restore=configure_egl(1)
report={**provenance(stage),'purpose':'train-only acceptance diagnostic; not formal dataset','completed':False,'rows':[],
        'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
write_json(out/'report.json',report)
try:
 sim=make_env('S',710000,cfg,num_scenarios=1000,control_dt_s=.1,environment_overrides={'force_destroy':True})
 reference=None
 try:
  for name in ['keep','right_12m_fast']:
   obs,_=sim.reset(seed=710000);provider=NativeBEVProvider(sim,cfg.bev);provider.reset(obs)
   expert=NavigationExpert(sim,cfg.expert,.1);prefix_actions=[];prefix_flags=[]
   bevs=[];states=[]
   for step in range(50):
    action=expert.act()[0];obs,_,tm,tc,info=sim.step(action);provider.update(obs)
    prefix_actions.append(action);prefix_flags.append([info['crash'],info['out_of_road'],tm,tc])
    if tm or tc:raise RuntimeError('Unexpected expert prefix termination')
   # This intervention is entirely BEFORE the model observation/branch anchor.
   prefix_command=make_risk_candidates(expert,[('roadside',-3.,7.,1.)])['roadside']
   origin=vehicle_state(sim.agent);tracker=TrajectoryTracker(sim.agent,.1)
   for step in range(20):
    action,_=tracker.act(remaining_command(prefix_command,origin,vehicle_state(sim.agent),step*.1),sim.agent.speed)
    obs,_,tm,tc,info=sim.step(action);provider.update(obs)
    prefix_actions.append(action);prefix_flags.append([info['crash'],info['out_of_road'],tm,tc])
    bevs.append(provider.capture().raster.copy());states.append(vehicle_state(sim.agent))
    if tm or tc:raise RuntimeError('Roadside prefix terminated; preserve evidence, reject this initial context')
   anchor=vehicle_state(sim.agent);history=np.stack(bevs[-4:]);scene=scene_state(sim)
   if reference is None:reference=(history.copy(),np.stack(states[-4:]),scene.copy())
   else:
    np.testing.assert_array_equal(history,reference[0]);np.testing.assert_allclose(states[-4:],reference[1],atol=1e-5,rtol=0)
    np.testing.assert_allclose(scene,reference[2],atol=1e-5,rtol=0)
   command=make_risk_candidates(expert)[name];tracker=TrajectoryTracker(sim.agent,.1)
   realized=[anchor];flags=[];typed=[];future=[];actions=[]
   for step in range(20):
    action,_=tracker.act(remaining_command(command,anchor,vehicle_state(sim.agent),step*.1),sim.agent.speed)
    obs,_,tm,tc,info=sim.step(action);provider.update(obs)
    flags.append([info['crash'],info['out_of_road'],tm,tc]);realized.append(vehicle_state(sim.agent));actions.append(action)
    typed.append([info[k] for k in ['crash_vehicle','crash_object','crash_building','crash_human','crash_sidewalk']])
    future.append(provider.capture().raster.copy())
    if tm or tc:break
   target,mask,*_=compute_risk_targets(flags)
   np.savez_compressed(out/f'{name}.npz',history=history,prefix_states=states,prefix_actions=prefix_actions,prefix_flags=prefix_flags,
                       scene=scene,command=command,states=realized,flags=flags,typed=typed,actions=actions,future=future,risk_target=target,risk_mask=mask)
   row={'candidate':name,'prefix_crash_seen':bool(np.asarray(prefix_flags)[:,0].any()),'anchor_state':anchor.tolist(),
        'steps':len(flags),'risk_target':target.tolist(),'risk_mask':mask.tolist(),
        'typed_event_seen':np.asarray(typed,dtype=bool).any(0).tolist()}
   report['rows'].append(row);print(json.dumps(row),flush=True);write_json(out/'report.json',report)
 finally:sim.close()
 report['completed']=True
except BaseException:
 report['error']=traceback.format_exc();raise
finally:
 restore();write_json(out/'report.json',report)
