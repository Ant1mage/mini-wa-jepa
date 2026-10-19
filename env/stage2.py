"""Physics-only MetaDrive; no RGB/GL/CUDA rendering dependency in collection."""
def make_data_env(map_spec, seed_start, traffic_density=0.1, horizon=1000, render_mode="physics"):
    from metadrive.envs.metadrive_env import MetaDriveEnv
    class DataEnv(MetaDriveEnv):
        def reset_sensors(self):
            super().reset_sensors()
            if self.config['use_render']:
                # Live viewer owns pacing. Keep one render per decision instead
                # of native ForceFPS rendering all five physics substeps.
                self.engine.force_fps.disable()

        @classmethod
        def default_config(cls):
            config = super().default_config()
            # Native vehicle accepts these parameters, but BaseEnv's strict
            # top-level config does not declare them. Extend via Config API.
            config["vehicle_config"].update(dict(max_engine_force=800.0,
                max_brake_force=150.0, max_steering=40.0, wheel_friction=0.9),
                allow_add_new_key=True)
            return config
    visual = render_mode != "physics"
    extra = dict(sensors={"main_camera": ()}) if visual else {}
    return DataEnv(dict(use_render=render_mode == "live", image_observation=render_mode in ("offscreen","viewer"), image_on_cuda=False,
        map=map_spec, start_seed=seed_start, num_scenarios=100,
        store_map=False, horizon=horizon, traffic_density=traffic_density,
        random_spawn_lane_index=False,
        agent_configs={"default_agent": {"spawn_lane_index": (">", ">>", 1)}},
        vehicle_config={"max_engine_force": 800.0, "max_brake_force": 150.0,
                        "max_steering": 40.0, "wheel_friction": 0.9,
                        "show_navi_mark": False, "image_source": "main_camera" if visual else "rgb_camera"},
        window_size=(960, 540), show_logo=False, show_terrain=visual and render_mode != 'viewer',
        multi_thread_render=False,
        show_interface=False, interface_panel=[], camera_smooth=False,
        log_level=50, **extra))
