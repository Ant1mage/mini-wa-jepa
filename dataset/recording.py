"""Teacher data recorder with explicit transition and horizon validity semantics."""
import numpy as np

from configs.stage2 import ActionNoiseConfig
from env.expert import ActionNoise, NavigationExpert
from evaluation.visuals import compose, main_view
from mini_wa_jepa.geometry import world_to_ego, wrap_angle


def vehicle_state(car):
    return np.array([*car.position,car.heading_theta,car.speed,*car.velocity],dtype=np.float32)


def future_labels(states, collision, out_of_road, lateral, manual, steps=25):
    n=len(states)-1
    trajectory=np.zeros((n,steps,3),np.float32)
    valid=np.zeros((n,steps),np.bool_)
    future_collision=np.zeros_like(valid);future_out=np.zeros_like(valid)
    good=np.zeros(n,np.bool_)
    for t in range(n):
        length=min(steps,n-t)
        future=states[t+1:t+length+1]
        trajectory[t,:length,:2]=world_to_ego(future[:,:2],states[t,:2],states[t,2])
        trajectory[t,:length,2]=wrap_angle(future[:,2]-states[t,2])
        valid[t,:length]=True
        future_collision[t,:length]=collision[t:t+length]
        future_out[t,:length]=out_of_road[t:t+length]
        good[t]=length==steps and not np.any(collision[t:t+length]) and not np.any(out_of_road[t:t+length]) \
            and not np.any(manual[t:t+length]) and np.max(np.abs(lateral[t:t+length]))<1.0
    return trajectory,valid,future_collision,future_out,good


def record_episode(sim,provider,cfg,seed,episode_id,frame_limit,viewer=None,noise_enabled=True,on_step=None):
    expert=NavigationExpert(sim,cfg.expert,cfg.dt_s)
    from dataclasses import replace
    noise=ActionNoise(cfg.seed+seed,replace(cfg.action_noise,enabled=noise_enabled))
    states=[vehicle_state(sim.agent)]
    buffers={k:[] for k in ['bev','expert_action','executed_action','noise_active','manual',
                             'collision','out_of_road','terminated','truncated','lateral_m']}
    reason='collection_budget';info={};reset_requested=False
    # Continue at most K steps past a storage quota to label its last waypoint.
    for step in range(min(cfg.horizon,frame_limit+cfg.future_steps)):
        if viewer:
            from scripts.expert_drive import planned_trajectory
            while not viewer.ready() and not viewer.quit and not viewer.reset_requested:
                viewer.draw(compose(provider.capture().raster,planned_trajectory(expert),states[-1],
                    np.zeros(2),'PAUSED',main_view(sim)))
            if viewer.quit:reason='user_quit';break
            if viewer.reset_requested:
                viewer.reset_requested=False;reset_requested=True;reason='user_reset';break
        action,detail=expert.act();executed,noisy=noise.apply(action)
        manual=bool(viewer and viewer.manual)
        if manual:executed=viewer.manual_action();noisy=False
        bev=provider.capture().raster
        if viewer:
            status=f"{'MANUAL' if manual else 'AUTO'} | {'NOISE ACTIVE' if noisy else 'tracking / recovery'} | lateral {detail['lateral_m']:+.2f} m | crash {info.get('crash',False)} out {info.get('out_of_road',False)}"
            viewer.draw(compose(bev,planned_trajectory(expert),states[-1],executed,status,main_view(sim)))
        if on_step is not None:
            from scripts.expert_drive import planned_trajectory
            on_step(dict(bev=bev,state=states[-1],executed_action=executed,
                         episode_id=episode_id,step_id=step,seed=seed,noise_active=noisy,
                         manual=manual,lateral_m=detail['lateral_m']),
                    planned_trajectory(expert,steps=cfg.future_steps,dt=cfg.dt_s))
        _,_,tm,tc,info=sim.step(executed)
        states.append(vehicle_state(sim.agent))
        values=dict(bev=bev,expert_action=action,executed_action=executed,noise_active=noisy,manual=manual,
                    collision=info['crash'],out_of_road=info['out_of_road'],terminated=tm,truncated=tc,lateral_m=detail['lateral_m'])
        for key,value in values.items():buffers[key].append(value)
        if tm or tc:reason='episode_end';break
    n=len(buffers['bev']);keep=min(n,frame_limit)
    states=np.asarray(states,dtype=np.float32)
    data={key:np.asarray(value) for key,value in buffers.items()}
    if not n:return None,dict(reason=reason,stored_frames=0),reset_requested
    traj,valid,fc,fo,good=future_labels(states,data['collision'],data['out_of_road'],data['lateral_m'],data['manual'],cfg.future_steps)
    data.update(state=states[:-1],next_state=states[1:],future_trajectory=traj,future_valid=valid,
                future_collision=fc,future_out_of_road=fo,good=good,
                episode_id=np.full(n,episode_id,np.int32),map_id=np.full(n,provider.geometry_hash,dtype='U64'),
                seed=np.full(n,seed,np.int32),step_id=np.arange(n,dtype=np.int32))
    report=dict(episode_id=episode_id,seed=seed,map_id=provider.geometry_hash,reason=reason,
                simulated_steps=n,stored_frames=keep,
                outcome={k:bool(info.get(k)) for k in ['arrive_dest','crash','out_of_road','max_step']},
                route_completion=float(info.get('route_completion',0)),
                good_frames=int(good[:keep].sum()),noise_frames=int(data['noise_active'][:keep].sum()),
                max_abs_lateral_m=float(np.max(np.abs(data['lateral_m']))))
    return {key:value[:keep] for key,value in data.items()},report,reset_requested


def future_labels_v2(states, terminal_failure, steps=10, stride=2):
    """Sample 0.2s future states; retain perturbation/recovery supervision.

    states contains actual observed states, including the terminal observation.
    Only an irrecoverable terminal observation loses action supervision. Missing
    future states are masked per waypoint, independently of driving quality.
    """
    states=np.asarray(states,np.float32)
    n=len(states)
    trajectory=np.zeros((n,steps,4),np.float32)
    valid=np.zeros((n,steps),bool)
    for t in range(n):
        indices=t+stride*np.arange(1,steps+1)
        keep=indices<n
        future=states[indices[keep]]
        trajectory[t,keep,:2]=world_to_ego(future[:,:2],states[t,:2],states[t,2])
        trajectory[t,keep,2]=future[:,3]
        trajectory[t,keep,3]=wrap_angle(future[:,2]-states[t,2])
        valid[t,keep]=True
    return trajectory,valid,~np.asarray(terminal_failure,bool)


def record_episode_v2(sim,provider,cfg,seed,episode_id,frame_limit,on_step=None,controls=None):
    from dataclasses import replace
    from env.data_v2 import third_person
    expert=NavigationExpert(sim,cfg.expert,cfg.dt_s)
    noise=ActionNoise(cfg.seed+seed,cfg.action_noise)
    rows=[];states=[];info={};reason='collection_budget';tm=tc=False
    for step in range(min(cfg.horizon,frame_limit+cfg.future_steps*cfg.waypoint_stride)):
        if controls:
            controls.wait_until_ready()
            if controls.quit:reason='user_quit';break
            if controls.reset_requested:reason='user_reset';controls.reset_requested=False;break
        state=vehicle_state(sim.agent)
        action,detail=expert.act()
        recovering=noise.recovery_remaining>0
        executed,noisy=noise.apply(action)
        manual=bool(controls and controls.manual)
        if manual:executed=controls.manual_action();noisy=False
        row=dict(bev=provider.capture().raster,state=state,expert_action=action,executed_action=executed,
            noise_active=noisy,recovery_active=bool(not noisy and (recovering or abs(detail['lateral_m'])>.3)),
            manual=manual,lateral_m=detail['lateral_m'],episode_id=episode_id,step_id=step,seed=seed,
            map_id=provider.geometry_hash,transition_valid=True,terminal_failure=False)
        if on_step:
            from scripts.expert_drive import planned_trajectory
            preview=planned_trajectory(expert,steps=cfg.future_steps,dt=cfg.waypoint_dt_s)
            preview=np.column_stack([preview[:,:2],np.full(len(preview),detail['target_speed_m_s']),preview[:,2]])
            if sim.main_camera is not None:row['camera_rgb']=third_person(sim)
            on_step(row,preview)
            row.pop('camera_rgb',None)  # RGB is visual-only; model input remains BEV
        observation,_,tm,tc,info=sim.step(executed)
        provider.update(observation)
        row.update(collision=bool(info['crash']),out_of_road=bool(info['out_of_road']),
                   terminated=tm,truncated=tc)
        rows.append(row);states.append(state)
        if tm or tc:reason='episode_end';break
    if not rows:return None,dict(episode_id=episode_id,seed=seed,reason=reason,stored_frames=0,simulated_steps=0)
    # Retain the terminal image for world/safety training, with no invented action.
    terminal=dict(rows[-1],bev=provider.capture().raster,state=vehicle_state(sim.agent),
        expert_action=np.zeros(2,np.float32),executed_action=np.zeros(2,np.float32),noise_active=False,
        recovery_active=False,step_id=len(rows),transition_valid=False,
        terminal_failure=bool(tm and (info.get('crash') or info.get('out_of_road'))))
    rows.append(terminal);states.append(terminal['state'])
    states=np.asarray(states,np.float32)
    data={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    traj,valid,action_mask=future_labels_v2(states,data['terminal_failure'],cfg.future_steps,cfg.waypoint_stride)
    action_mask &= data['transition_valid']
    data.update(future_trajectory=traj,future_valid=valid,action_mask=action_mask,
        good=action_mask.copy(),next_state=np.concatenate([states[1:],states[-1:]],axis=0))
    # Safety events cover every intermediate transition, not just sampled endpoints.
    for key,event in [('future_collision','collision'),('future_out_of_road','out_of_road')]:
        flags=np.zeros_like(valid)
        for t in range(len(rows)):
            for k in range(cfg.future_steps):
                stop=min(t+(k+1)*cfg.waypoint_stride,len(rows)-1)
                flags[t,k]=data[event][t:stop].any()
        data[key]=flags
    keep=min(len(rows),frame_limit)
    report=dict(episode_id=episode_id,seed=seed,map_id=provider.geometry_hash,reason=reason,
        stored_frames=keep,simulated_steps=len(rows)-1,
        outcome={key:bool(info.get(key,False)) for key in ('arrive_dest','crash','out_of_road','max_step')},
        route_completion=float(info.get('route_completion',0)),
        action_frames=int(action_mask[:keep].sum()),noise_frames=int(data['noise_active'][:keep].sum()),
        recovery_frames=int(data['recovery_active'][:keep].sum()),
        terminal_failure_frames=int(data['terminal_failure'][:keep].sum()),
        max_abs_lateral_m=float(np.max(np.abs(data['lateral_m']))))
    return {key:value[:keep] for key,value in data.items()},report
