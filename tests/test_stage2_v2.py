import unittest
import numpy as np
from configs.stage2 import ActionNoiseConfig,Stage2Config
from dataset.recording import future_labels_v2
from env.expert import ActionNoise
from mini_wa_jepa.normalization import Normalizer


class V2DataTests(unittest.TestCase):
    def test_exact_future_times_speed_heading_and_terminal_only_mask(self):
        states=np.zeros((24,6),np.float32)
        states[:,0]=np.arange(24)*.1;states[:,2]=np.arange(24)*.02;states[:,3]=np.arange(24)*.3
        failure=np.zeros(24,bool);failure[-1]=True
        traj,valid,mask=future_labels_v2(states,failure)
        np.testing.assert_allclose(traj[0,:,0],np.arange(1,11)*.2,atol=1e-7)
        np.testing.assert_allclose(traj[0,:,2],np.arange(1,11)*.6,atol=1e-6)
        np.testing.assert_allclose(traj[0,:,3],np.arange(1,11)*.04,atol=1e-6)
        self.assertTrue(mask[:-1].all());self.assertFalse(mask[-1])
        self.assertTrue(valid[0].all());self.assertFalse(valid[-2:].any())
        self.assertTrue(valid[-3,0]);self.assertFalse(valid[-3,1:].any())

    def test_pulse_duration_and_full_authority_recovery(self):
        cfg=ActionNoiseConfig(event_probability=1.,duration_min=4,duration_max=4,recovery_steps=8)
        noise=ActionNoise(7,cfg);action=np.array([.1,.2],np.float32)
        flags=[]
        for _ in range(12):
            output,noisy=noise.apply(action);flags.append(noisy)
            if not noisy:np.testing.assert_array_equal(output,action)
        self.assertEqual(flags,[True]*4+[False]*8)
        self.assertTrue(noise.apply(action)[1])

    def test_four_coordinate_normalization_scales_velocity_but_not_angle(self):
        normalizer=Normalizer(scale=2.,trajectory_mean=[0]*4,trajectory_std=[1]*4)
        trajectory=np.array([[1,2,3,.4]],np.float32)
        np.testing.assert_allclose(normalizer.trajectory(trajectory),[[2,4,6,.4]])
        np.testing.assert_allclose(normalizer.inverse_trajectory(normalizer.trajectory(trajectory)),trajectory)
        self.assertEqual(Stage2Config().waypoint_stride,2)


class BaselineTests(unittest.TestCase):
    def test_model_budget_and_masked_rescue_gradients(self):
        import torch
        from models.baseline import ResNet18Trajectory,masked_trajectory_loss
        model=ResNet18Trajectory()
        count=sum(p.numel() for p in model.parameters())
        self.assertGreaterEqual(count,15_000_000);self.assertLessEqual(count,30_000_000)
        prediction=torch.zeros(3,10,4,requires_grad=True)
        target=torch.ones_like(prediction)
        valid=torch.ones(3,10,dtype=torch.bool);valid[1,5:]=False
        action=torch.tensor([True,True,False])
        loss=masked_trajectory_loss(prediction,target,valid,action);loss.backward()
        self.assertTrue((prediction.grad[0]!=0).all())
        self.assertTrue((prediction.grad[1,:5]!=0).all())
        self.assertEqual(prediction.grad[1,5:].abs().sum(),0)
        self.assertEqual(prediction.grad[2].abs().sum(),0)


if __name__=='__main__':unittest.main()


class StorageResumeTests(unittest.TestCase):
    def test_resume_keeps_previous_shards_and_all_frames(self):
        import tempfile
        from pathlib import Path
        from dataset.storage import ShardWriter,ShardReader
        with tempfile.TemporaryDirectory() as directory:
            writer=ShardWriter(directory,3);writer.add({'state':np.arange(8).reshape(4,2)});writer.close()
            first=(Path(directory)/'shard_0000.npz').read_bytes()
            writer=ShardWriter(directory,3,resume=True)
            writer.add({'state':np.arange(8,14).reshape(3,2)});writer.close()
            reader=ShardReader(directory)
            np.testing.assert_array_equal(reader.metadata('state'),np.arange(14).reshape(7,2))
            self.assertEqual((Path(directory)/'shard_0000.npz').read_bytes(),first)

    def test_worker_merge_renumbers_episode_ids_without_crossing_splits(self):
        import tempfile
        from pathlib import Path
        from dataclasses import replace
        from configs.stage2 import SPLITS
        from dataset.storage import ShardWriter,ShardReader
        from scripts.collect_stage2_parallel import merge
        from mini_wa_jepa.provenance import write_json
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parts=root/'parts'
            for split,spec in SPLITS.items():
                for map_spec in spec['maps']:
                    part=parts/f'{split}_{map_spec}'
                    writer=ShardWriter(part/split)
                    writer.add(dict(state=np.zeros((1,6),np.float32),episode_id=np.zeros(1,np.int32)))
                    writer.close()
                    write_json(part/'manifest.json',dict(schema_version=2,completed=True,
                        git_commit='source',source_tree_sha256='tree',goal_sha256='goal',
                        episodes=[dict(episode_id=0,stored_frames=1,simulated_steps=1)]))
            merge(parts,root/'merged',replace(Stage2Config(),frames=4,validation_frames=2,test_frames=2))
            ids=np.concatenate([ShardReader(root/'merged'/split).metadata('episode_id') for split in SPLITS])
            np.testing.assert_array_equal(ids,np.arange(8))


class TerminationTests(unittest.TestCase):
    def test_full_body_grace_is_counted_once_per_engine_decision(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from env.data_v2 import FullOffroadRule
        env=SimpleNamespace(engine=SimpleNamespace(episode_step=0))
        vehicle=SimpleNamespace(id='vehicle_object_not_agent_id',on_lane=False,position=np.zeros(2),
                                heading=np.array([1.,0.]),heading_theta=0.,LENGTH=4.,WIDTH=2.)
        rule=FullOffroadRule(.1,1.)
        with patch('metadrive.utils.pg.utils.ray_localization',return_value=([],False)):
            for step in range(1,11):
                env.engine.episode_step=step
                self.assertEqual(rule(env,vehicle),step==10)
                self.assertEqual(rule(env,vehicle),step==10)
            env.engine.episode_step=1
            self.assertFalse(rule(env,vehicle))
        with patch('metadrive.utils.pg.utils.ray_localization',return_value=([],True)):
            env.engine.episode_step=12
            self.assertFalse(rule(env,vehicle))
        vehicle.on_lane=True
        self.assertFalse(rule(env,vehicle))
