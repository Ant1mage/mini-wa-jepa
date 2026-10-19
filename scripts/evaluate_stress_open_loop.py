"""阶段5冻结模型、匹配测试队列上的完整开环退化评测。"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.stage5 import Stage5Config, CONDITIONS, CHECKPOINTS
from evaluation.degradation import DegradedSequences
from evaluation.stress import FrozenPolicy, cohort_data, require_rocm
from evaluation.stress_visualization import open_samples
from evaluation.latent import paired_episode_bootstrap
from mini_wa_jepa.provenance import provenance, write_json
from scripts.evaluate_joint import physical_metrics, summarize_groups
from training.joint import evaluate


@torch.no_grad()
def evaluate_resnet(policy, loader, data):
    plans, targets, sources, anchors = [], [], [], []
    for raw in loader:
        output = policy.batch(raw)
        plans.append(policy.physical(output['trajectory']))
        targets.append(data.normalizer.inverse_trajectory(raw['trajectory'].numpy()))
        sources.append(raw['observation_sources'].numpy()); anchors.extend(raw['anchor'].numpy())
    anchors = np.asarray(anchors)
    arrays = dict(anchors=anchors, episodes=np.asarray(data.data.episodes[anchors]), full_mask=np.ones(len(anchors), bool),
                  prediction=np.concatenate(plans), target=np.concatenate(targets), observation_sources=np.concatenate(sources),
                  noisy=np.asarray(data.data.noisy[anchors]), recovery=np.asarray(data.data.recovery[anchors]))
    return physical_metrics(arrays), arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/restart_stage5/open_loop'))
    parser.add_argument('--data', type=Path, default=Path(Stage5Config.data))
    parser.add_argument('--smoke-samples', type=int)
    parser.add_argument('--conditions', nargs='*', default=[c.name for c in CONDITIONS])
    parser.add_argument('--models', nargs='*', default=list(CHECKPOINTS), choices=list(CHECKPOINTS))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); config = Stage5Config(); require_rocm()
    torch.set_num_threads(config.cpu_threads); torch.manual_seed(config.seed)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    data = cohort_data(args.data, args.smoke_samples)
    report = dict(**provenance(config), stage=5, completed=False, smoke_samples=args.smoke_samples,
                  conditions=[asdict(c) for c in CONDITIONS if c.name in args.conditions], models={}, paired={},
                  anchors=len(data), episode_count=len(np.unique(data.episodes[data.anchors])),
                  protocol_sha256=hashlib.sha256(Path('docs/STAGE5_PROTOCOL.md').read_bytes()).hexdigest(),
                  dataset_manifest_sha256=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest())
    writer = SummaryWriter(str(args.output/'tensorboard')); started = time.perf_counter()
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for name in args.models:
                policy = FrozenPolicy(name, CHECKPOINTS[name], torch.device('cuda'))
                report['models'][name] = dict(**policy.metadata, conditions={})
                for condition_index, spec in enumerate(CONDITIONS):
                    if spec.name not in args.conditions:
                        continue
                    folder = args.output/name/spec.name
                    if args.resume and (folder/'summary.json').exists():
                        saved_summary = json.loads((folder/'summary.json').read_text())
                        if saved_summary['checkpoint_sha256'] != policy.metadata['checkpoint_sha256']:
                            raise ValueError('Resume checkpoint changed')
                        report['models'][name]['conditions'][spec.name] = saved_summary
                        continue
                    folder.mkdir(parents=True, exist_ok=False)
                    dataset = DegradedSequences(data, spec, config.seed)
                    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=config.workers,
                                        pin_memory=True, shuffle=False, multiprocessing_context='spawn' if config.workers else None)
                    tick = time.perf_counter()
                    if name == 'resnet':
                        metrics, arrays = evaluate_resnet(policy, loader, dataset)
                    else:
                        metrics, arrays = evaluate(policy.model, loader, policy.config, policy.device)
                        arrays['observation_sources'] = dataset.sources[data.history_indices[data.anchors]]
                    for key in ('prediction', 'target', 'ade_m', 'fde_m'):
                        if not np.isfinite(arrays[key]).all():
                            raise FloatingPointError(f'Nonfinite {name}/{spec.name}/{key}')
                    if not np.array_equal(arrays['anchors'], data.anchors) or not arrays['full_mask'].all():
                        raise ValueError('Cohort order or masks changed')
                    np.savez_compressed(folder/'raw.npz', **arrays)
                    parity = {}
                    if spec.name == 'clean':
                        frozen = np.load(f'artifacts/restart_stage4/test_eval/{name}_raw.npz')
                        lookup = dict(zip(frozen['anchors'].tolist(), range(len(frozen['anchors']))))
                        positions = [lookup[int(a)] for a in data.anchors]
                        for key in ('ade_m', 'fde_m'):
                            expected = float(frozen[key][positions].mean())
                            parity[key] = dict(expected=expected, actual=metrics[key], absolute_delta=abs(metrics[key]-expected))
                            if abs(metrics[key]-expected) > 2e-6:
                                raise ValueError(f'Clean parity failure: {name}/{key}/{parity[key]}')
                    evidence = open_samples(policy, dataset, folder, config.visualization_sample_ids, spec, writer, condition_index)
                    summary = dict(metrics=metrics, groups=summarize_groups(arrays, args.data), clean_parity=parity,
                                   seconds=time.perf_counter()-tick, checkpoint_sha256=policy.metadata['checkpoint_sha256'],
                                   condition=asdict(spec), evidence=evidence)
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(f'{name}/{key}', value, condition_index)
                    report['models'][name]['conditions'][spec.name] = summary
                    write_json(folder/'summary.json', summary); write_json(args.output/'report.json', report); writer.flush()
                    print(json.dumps(dict(model=name, condition=spec.name, ade_m=metrics['ade_m'], fde_m=metrics['fde_m'],
                                          seconds=summary['seconds']), ensure_ascii=False), flush=True)
                del policy; torch.cuda.empty_cache()
        if set(args.models) == set(CHECKPOINTS):
            for spec in CONDITIONS:
                if spec.name not in args.conditions:
                    continue
                arrays = {name: np.load(args.output/name/spec.name/'raw.npz') for name in args.models}
                report['paired'][spec.name] = {}
                for other in ('action_only', 'resnet'):
                    report['paired'][spec.name]['joint_minus_'+other] = {
                        key: paired_episode_bootstrap(arrays['joint'][key], arrays[other][key],
                                                      arrays['joint']['episodes'], config.bootstrap_samples, config.seed)
                        for key in ('ade_m', 'fde_m')}
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc(); raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        write_json(args.output/'report.json', report); writer.close()


if __name__ == '__main__':
    main()
