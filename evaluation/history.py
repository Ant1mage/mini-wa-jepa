"""Observed-only rolling encoder cache at exact training history intervals."""
import numpy as np
import torch
from mini_wa_jepa.geometry import wrap_angle


class ObservedHistory:
    def __init__(self, control_dt_s=.02, history_dt_s=.1):
        stride=round(history_dt_s/control_dt_s)
        if stride<1 or abs(stride*control_dt_s-history_dt_s)>1e-8:
            raise ValueError('History dt must be an integer number of control steps')
        self.stride=stride;self.control_dt_s=control_dt_s;self.reset()

    def reset(self):
        self.frames={};self.motion={};self.headings={};self.last_step=None

    def append(self,step,encoded,speed,heading):
        if self.last_step is None and step!=0:
            raise ValueError('Reset history must begin at step zero')
        if self.last_step is not None and step!=self.last_step+1:
            raise ValueError('History must reset on episode/step discontinuity')
        if encoded.ndim!=3 or len(encoded)!=1:
            raise ValueError('Expected one actual encoded frame [1,N,D]')
        self.headings[step]=float(heading)
        reference=max(0,step-self.stride)
        yaw=0. if step==0 else float(wrap_angle(heading-self.headings[reference]))/((step-reference)*self.control_dt_s)
        self.motion[step]=np.array([speed/10,yaw],np.float32)
        self.frames[step]=encoded.detach()
        indices=[max(0,step-lag*self.stride) for lag in (3,2,1,0)]
        frames=torch.stack([self.frames[index] for index in indices],dim=1)
        motion=np.stack([self.motion[index] for index in indices])[None]
        keep=max(0,step-3*self.stride)
        for store in (self.frames,self.motion,self.headings):
            for key in list(store):
                if key<keep:del store[key]
        self.last_step=step
        return frames,motion,indices

