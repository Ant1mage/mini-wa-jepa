"""Sequence BEV noise, isolated from teacher/target BEV and simulator data."""
import cv2
import numpy as np

from configs.stage2 import BEVNoiseConfig


def augment_bev(sequence, rng, config=BEVNoiseConfig()):
    result=np.array(sequence,copy=True)
    if not config.enabled:
        return result
    # Object-level dropouts remain coherent through this history clip.
    vehicle_union=np.max(sequence[:,2],axis=0)
    count,labels=cv2.connectedComponents((vehicle_union>0).astype(np.uint8))
    for label in range(1,count):
        if rng.random()<config.missed_vehicle_probability:
            result[:,2,labels==label]=0
    jitter=config.boundary_jitter_pixels
    dx,dy=rng.integers(-jitter,jitter+1,size=2)
    for frame in result:
        frame[1]=cv2.warpAffine(frame[1],np.float32([[1,0,dx],[0,1,dy]]),
                               (frame.shape[2],frame.shape[1]),flags=cv2.INTER_NEAREST)
        if rng.random()<config.flicker_probability:
            frame[1:]=0
    return result
