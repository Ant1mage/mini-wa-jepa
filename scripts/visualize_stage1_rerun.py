"""Load Stage 1 acceptance BEV samples and telemetry into Rerun."""
import argparse
import json
from pathlib import Path
import numpy as np

from mini_wa_jepa.visualization import RerunLogger


def main():
    parser = argparse.ArgumentParser(description="Stream Stage 1 BEV samples into Rerun")
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="artifacts/restart_stage1/egl_100",
        help="Path to Stage 1 artifacts directory",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        default=True,
        help="Spawn native Rerun viewer window (default True)",
    )
    parser.add_argument(
        "--no-spawn",
        dest="spawn",
        action="store_false",
        help="Do not spawn Rerun viewer automatically",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save .rrd recording",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory {samples_dir} does not exist.")

    npz_files = sorted(samples_dir.glob("seed_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No seed_*.npz files found in {samples_dir}")

    # Load report.json if available
    report_file = samples_dir / "report.json"
    metadata_by_seed = {}
    if report_file.exists():
        try:
            report_data = json.loads(report_file.read_text())
            for ep in report_data.get("episodes", []):
                metadata_by_seed[ep.get("seed")] = ep
        except Exception as e:
            print(f"Warning: could not parse {report_file}: {e}")

    logger = RerunLogger(app_id="mini-wa-jepa-stage1", spawn=args.spawn, save_path=args.save)
    print(f"Streaming {len(npz_files)} Stage 1 sample sets into Rerun timeline...")

    for idx, npz_path in enumerate(npz_files):
        data = np.load(npz_path)
        bev = data["bev"]  # (4, 128, 128) uint8
        native_rgb = data["native_rgb"] if "native_rgb" in data else None

        # Extract seed from filename e.g. seed_026_step_088.npz
        seed_str = npz_path.stem.split("_")[1]
        try:
            seed = int(seed_str)
        except ValueError:
            seed = idx

        logger.set_step(step=idx, sim_time_s=float(idx * 0.5))

        # Log BEV & camera
        logger.log_bev(bev)
        if native_rgb is not None:
            logger.log_camera(native_rgb, name="camera/native_semantic_bev")

        # Log metrics if available
        meta = metadata_by_seed.get(seed, {})
        fps = meta.get("pipeline_fps")
        rss_mb = (meta.get("memory", {}).get("rss_bytes", 0)) / (1024 * 1024)
        cal_err = meta.get("max_calibration_error_pixel")

        logger.log_context(seed=seed)
        logger.log_metrics(dict(pipeline_fps=fps, rss_mb=rss_mb or None,
                                calibration_error_px=cal_err))

    logger.close()
    print("Done! You can scrub through seeds in the Rerun timeline.")


if __name__ == "__main__":
    main()
