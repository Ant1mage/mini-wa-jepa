"""Read measured stage reports for Marimo; never substitute synthetic results."""
import json
import math
from pathlib import Path


METRICS = {
    'ade_m': 'ADE (m)', 'fde_m': 'FDE (m)',
    'route_completion': 'Route completion (0–1)', 'collision_rate': 'Collision rate (0–1)',
    'out_of_road_rate': 'Out-of-road rate (0–1)', 'lateral_error_m': 'Lateral error (m)',
    'inference_ms': 'Inference latency (ms)', 'effective_rank_ratio': 'Effective rank ratio',
    'pipeline_ms_p95': 'Full pipeline P95 (ms)', 'success_rate': 'Success rate (0–1)',
    'latent_mse': 'Latent prediction MSE', 'copy_last_mse': 'Copy-last MSE',
    'prediction_copy_ratio': 'Prediction / copy MSE',
    'normalized_latent_mse': 'Latent MSE / content variance',
    'pooled_effective_rank_ratio': 'Frame-pooled effective rank ratio',
    'std_mean': 'Content channel std mean', 'std_p10': 'Content channel std P10',
    'delta_mse': 'Same-context consequence delta MSE',
    'retrieval_accuracy': 'Candidate consequence retrieval accuracy',
    'wrong_command_mse': 'Wrong-command latent MSE',
}


def load_stage_runs(project_root, stage):
    """Return flattened runs, report metadata and visible validation errors."""
    if stage not in (2, 3, 4, 5, 6):
        raise ValueError('Expected stage 2–6')
    root = Path(project_root)
    paths = sorted(set((root / 'artifacts' / f'stage{stage}').glob('**/metrics.json')) |
                   set((root / 'artifacts' / f'restart_stage{stage}').glob('**/metrics.json')))
    rows, sources, errors = [], [], []
    for path in paths:
        try:
            report = json.loads(path.read_text())
            if not isinstance(report, dict) or not isinstance(report.get('runs'), list):
                raise ValueError('Expected an object with a runs list')
            if (report.get('visualization_schema_version') != 1 or report.get('spec_version') != 2
                    or report.get('stage') != stage):
                raise ValueError('Report must identify visualization_schema_version=1, spec_version=2 and its stage')
            for field in ('config', 'seeds', 'git_commit', 'goal_sha256'):
                if field not in report:
                    raise ValueError(f'Missing provenance: {field}')
            parsed = []
            for run in report['runs']:
                if not isinstance(run, dict) or not isinstance(run.get('metrics'), dict):
                    raise ValueError('Each run must contain a metrics object')
                row = {key: run[key] for key in ('run_id', 'model', 'ablation', 'split', 'seed')}
                row['stage'] = stage
                for field in ('degradation_type', 'degradation_level'):
                    if field in run:
                        row[field] = run[field]
                for key in METRICS:
                    value = run.get('metrics', {}).get(key)
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                              or not math.isfinite(value)):
                        raise ValueError(f'{key} must be a finite measured number or null')
                    row[key] = value
                row['source'] = str(path.relative_to(root))
                parsed.append(row)
            rows.extend(parsed)
            sources.append(dict(path=str(path.relative_to(root)), **{key: report[key] for key in
                                ('config', 'seeds', 'git_commit', 'goal_sha256')}))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
    return rows, sources, errors
