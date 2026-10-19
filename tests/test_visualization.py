"""Verify time alignment, trajectory overlays, optional imports and report provenance."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from mini_wa_jepa.visualization.reports import load_stage_runs
from mini_wa_jepa.visualization.streaming import log_recorded_frame, make_rerun_logger


class StreamingTests(unittest.TestCase):
    def test_no_flags_do_not_initialize_sdk(self):
        self.assertIsNone(make_rerun_logger(argparse.Namespace(rerun=False, live=False, rerun_save=None), 'test'))

    def test_row_alignment_valid_waypoints_and_real_speed(self):
        logger = Mock()
        row = dict(bev=np.zeros((4, 128, 128), np.uint8), state=np.array([0, 0, 0, 7, 7, 0]),
                   executed_action=np.array([.2, -.3]), episode_id=3, step_id=4, seed=9,
                   noise_active=True, lateral_m=.5, future_trajectory=np.arange(12).reshape(3, 4),
                   future_valid=np.array([True, True, False]))
        log_recorded_frame(logger, row, dt_s=.1, timeline_step=104)
        logger.set_step.assert_called_once_with(104, sim_time_s=10.4)
        logger.log_telemetry.assert_called_once_with(speed_m_s=7, steering=.2, throttle_brake=-.3,
                                                    lateral_error_m=.5)
        np.testing.assert_array_equal(logger.log_trajectories.call_args.kwargs['ground_truth'], row['future_trajectory'][:2])
        self.assertEqual(logger.log_trajectories.call_args.kwargs['size'], 128)
        logger.log_camera.assert_called_once_with(None)

    def test_owned_streams_and_ego_pixel_projection_and_clear(self):
        from mini_wa_jepa.visualization import RerunLogger
        stream_a, stream_b = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(RerunLogger, '_new_stream', side_effect=[stream_a, stream_b]), \
                patch.object(RerunLogger, '_spawn_viewer') as spawn:
            logger = RerunLogger(spawn=True, save_path=Path(directory) / 'both.rrd', viewer_port=19001)
            spawn.assert_called_once_with(stream_b, 19001)
            stream_a.save.assert_called_once()
            logger.set_step(31, 3.1)
            with patch.object(logger, '_log') as log:
                logger.log_trajectories(planned=np.array([[10, 0, 7, 0], [0, 10, 7, 0]]))
                pixels = log.call_args.args[1].strips.as_arrow_array().to_pylist()
                np.testing.assert_allclose(pixels[0], [[63.5, 43.5], [43.5, 63.5]])
                logger.log_trajectories(ground_truth=np.empty((0, 4)))
                self.assertEqual(log.call_args.args[0], 'bev/ground_truth_trajectory')
                self.assertEqual(type(log.call_args.args[1]).__name__, 'Clear')
            logger.close()
            stream_a.flush.assert_called_once()
            stream_b.flush.assert_called_once()

    def test_file_roundtrip_and_existing_artifact_preserved(self):
        from mini_wa_jepa.visualization import RerunLogger
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.rrd'
            with RerunLogger(spawn=False, save_path=path) as logger:
                logger.set_step(0, 0.)
                logger.log_bev(np.zeros((4, 128, 128), np.uint8))
                logger.log_state([0, 0, 0, 5, 5, 0])
                logger.log_latents(predicted=np.zeros((4, 2, 3)), ground_truth=np.zeros((4, 2, 3)))
            self.assertGreater(path.stat().st_size, 1000)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                RerunLogger(spawn=False, save_path=path)
            self.assertEqual(path.read_bytes(), original)

    def test_viewer_uses_bundled_binary_and_matching_transport(self):
        import rerun as rr
        import rerun_cli
        from mini_wa_jepa.visualization import RerunLogger
        process = Mock()
        process.poll.return_value = None
        stream = Mock()
        with patch('mini_wa_jepa.visualization.rerun_logger.subprocess.Popen', return_value=process) as popen, \
                patch('mini_wa_jepa.visualization.rerun_logger.socket.create_connection', return_value=MagicMock()), \
                patch.object(rr, 'connect_tcp', create=True) as tcp, \
                patch.object(rr, 'connect_grpc', create=True) as grpc:
            RerunLogger._spawn_viewer(stream, 19002)
            self.assertEqual(Path(popen.call_args.args[0][0]).parent, Path(rerun_cli.__file__).parent)
            if hasattr(rr, 'set_time'):
                grpc.assert_called_once_with('rerun+http://127.0.0.1:19002/proxy', recording=stream)
                tcp.assert_not_called()
            else:
                tcp.assert_called_once_with('127.0.0.1:19002', recording=stream)
                grpc.assert_not_called()


class ReportTests(unittest.TestCase):
    def test_missing_legacy_invalid_reports_do_not_become_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_stage_runs(directory, 3), ([], [], []))
            path = Path(directory) / 'artifacts/stage3/metrics.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(stage=3, runs=[])))
            rows, sources, errors = load_stage_runs(directory, 3)
            self.assertEqual(rows, [])
            self.assertEqual(sources, [])
            self.assertEqual(len(errors), 1)

    def test_missing_metric_remains_null_and_provenance_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'artifacts/stage3/run/metrics.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(visualization_schema_version=1, spec_version=2, stage=3,
                config={'batch_size': 4}, seeds=[7], git_commit='source', goal_sha256='goal',
                runs=[dict(run_id='run', model='encoder', ablation='VICReg', split='val', seed=7,
                           metrics={'effective_rank_ratio': .8})])))
            rows, sources, errors = load_stage_runs(directory, 3)
            self.assertEqual(errors, [])
            self.assertIsNone(rows[0]['ade_m'])
            self.assertEqual(rows[0]['effective_rank_ratio'], .8)
            self.assertEqual(sources[0]['git_commit'], 'source')
            report = json.loads(path.read_text())
            report['runs'][0]['metrics']['ade_m'] = float('nan')
            path.write_text(json.dumps(report))
            rows, _, errors = load_stage_runs(directory, 3)
            self.assertFalse(rows)
            self.assertEqual(len(errors), 1)


if __name__ == '__main__':
    unittest.main()
