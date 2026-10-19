import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
import tempfile
import unittest
from pathlib import Path

import numpy as np

from configs.stage2 import BEVNoiseConfig,ActionNoiseConfig
from controller.pure_pursuit import SpeedPID,pursuit_steering
from dataset.augmentation import augment_bev
from dataset.recording import future_labels
from dataset.storage import ShardWriter,ShardReader
from env.expert import ActionNoise
from mini_wa_jepa.geometry import world_to_ego,ego_to_world,ego_to_pixels
from mini_wa_jepa.normalization import Normalizer


class DataTests(unittest.TestCase):
    def test_coordinate_convention_and_roundtrip(self):
        local=world_to_ego([[10,12],[8,10]],[10,10],np.pi/2)
        np.testing.assert_allclose(local,[[2,0],[0,2]],atol=1e-7)
        np.testing.assert_allclose(ego_to_world(local,[10,10],np.pi/2),[[10,12],[8,10]])
        pixels=ego_to_pixels([[2,0],[0,2]])
        self.assertLess(pixels[0,1],31.5);self.assertLess(pixels[1,0],31.5)

    def test_future_label_alignment_failure_and_terminal_mask(self):
        states=np.zeros((7,6),np.float32);states[:,0]=np.arange(7)
        collision=np.array([False,False,False,True,False,False])
        traj,valid,fc,fo,good=future_labels(states,collision,np.zeros(6,bool),np.zeros(6),np.zeros(6,bool),3)
        np.testing.assert_allclose(traj[0,:,0],[1,2,3])
        self.assertTrue(good[0]);self.assertFalse(good[1]);self.assertTrue(fc[1,-1])
        np.testing.assert_array_equal(valid[-1],[True,False,False]);self.assertFalse(good[-1])

    def test_normalization_roundtrip_at_non_default_scale(self):
        normalizer=Normalizer(2.5,[3,4,0.1],[2,3,0.2],2,3)
        trajectory=np.array([[1,2,0.3],[-3,4,-2]],np.float32)
        np.testing.assert_allclose(normalizer.inverse_trajectory(normalizer.trajectory(trajectory)),trajectory,atol=1e-6)
        np.testing.assert_allclose(normalizer.inverse_speed(normalizer.speed([2,7])),[2,7])

    def test_shards_random_access_across_boundaries_and_cache_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=ShardWriter(directory,3)
            writer.add(dict(state=np.arange(14).reshape(7,2),episode_id=np.arange(7)))
            writer.close();reader=ShardReader(directory,1)
            for i in [6,0,3,2,5]:self.assertEqual(reader[i]['episode_id'],i)
            self.assertEqual(len(reader.cache),1);self.assertEqual(len(reader),7)

    def test_noise_disabled_identity_and_target_not_mutated(self):
        data=np.full((4,3,64,64),255,np.uint8)
        result=augment_bev(data,np.random.default_rng(0))
        np.testing.assert_array_equal(result,data);self.assertIsNot(result,data)
        result=augment_bev(data,np.random.default_rng(0),BEVNoiseConfig(True,1.0,1,1.0))
        self.assertEqual(result[:,1:].sum(),0);self.assertTrue(np.all(data==255))

    def test_pursuit_sign_pid_braking_and_deterministic_action_noise(self):
        self.assertGreater(pursuit_steering([5,2],2.5,0.7),0)
        self.assertLess(SpeedPID().step(2,8),0)
        a=ActionNoise(7,ActionNoiseConfig(event_probability=1));b=ActionNoise(7,ActionNoiseConfig(event_probability=1))
        np.testing.assert_array_equal(a.apply(np.zeros(2))[0],b.apply(np.zeros(2))[0])


class ViewerTests(unittest.TestCase):
    def test_keyboard_pause_single_step_reset_manual_and_quit(self):
        from evaluation.visuals import LiveViewer
        import pygame
        viewer=LiveViewer()
        try:
            for key in [pygame.K_SPACE,pygame.K_n,pygame.K_r,pygame.K_m]:
                pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=key))
                viewer.poll()
            self.assertTrue(viewer.paused and viewer.manual and viewer.reset_requested)
            self.assertTrue(viewer.ready());self.assertFalse(viewer.ready())
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_w));viewer.poll()
            self.assertEqual(viewer.manual_action()[1],1)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_q));viewer.poll()
            self.assertTrue(viewer.quit)
        finally:viewer.close()


if __name__=='__main__':unittest.main()
