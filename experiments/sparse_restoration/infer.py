"""Evaluate sparse restoration using fixed checkpoints."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from recovery import recover_dataset
from recovery.cache import RecoveryCacheConfig, build_loader, collect_test_pairs, degradation_cfg, plain
from recovery.cache import load_or_create_pairs_cache, load_or_create_method_cache, save_method_cache
from recovery.cache import write_json
from recovery.checkpoints import fixed_ae_paths, resolve_path
from recovery.registry import ADAPTERS, canonical_method, canonical_dataset, resolve_config
from recovery.adapters.ae import _model_cfg
from .evaluate import run_irae_recovery_eval
from utils.utils_metrics import avg_psnr_pytorch, avg_ssim_pytorch


# Reuse the stage evaluator so C, CF and CFO come from one recovery chain.
def evaluate_stages(dataset, ratio, args, pairs=None, cache_config=None):
    if cache_config is not None and pairs is None:
        raise ValueError('Shared-cache export requires paired test inputs.')
    _, _, degradation, config = resolve_config(dataset, degradation_cfg(ratio, 42), 'irae')
    inference = plain(config) | dict(split='test', num_samples=args.num_samples, device=args.device)
    # Translate preset fields to the stage evaluator's inference settings.
    for old, new in [('lambda', 'lambda_init'), ('is_lambda_update', 'update_lambda'),
                     ('lambda_check_interval', 'lambda_update_interval'), ('jinit_stage_iters', 'stage_iterations'),
                     ('lambda_init_gamma', 'aux_lambda_scale')]:
        inference[new] = inference.pop(old)
    loader = build_loader(RecoveryCacheConfig(dataset=dataset, num_workers=0, data_root=getattr(args, 'data_root', None),
                          feature_batch_size=int(config['execution_batch_size'])), 'test', degradation)
    folder = args.output / dataset / str(ratio)
    restored, seen = [], 0
    # Preserve exact sample alignment before exporting full-test recoveries for downstream use.
    def collect(pack):
        nonlocal seen
        count = len(pack['index'])
        selected = slice(seen, seen + count)
        if any(not torch.equal(pack[key], pairs[target][selected]) for key, target in
               [('index', 'indices'), ('clean', 'clean'), ('degraded', 'degraded')]):
            raise ValueError('Stage recovery and shared test pairs are not aligned.')
        if cache_config is not None:
            restored.append(pack['recovered'].cpu())
        seen += count
        if seen % (10 * int(config['execution_batch_size'])) == 0 or seen == len(pairs['indices']):
            print(f'{dataset} ratio={ratio}: {seen}/{len(pairs["indices"])} stage recoveries', flush=True)
    started = time.perf_counter()
    result = run_irae_recovery_eval(eval_loader=loader, ckpt_dir=resolve_path(config['checkpoint']['path']),
        checkpoint_paths=fixed_ae_paths(config['checkpoint']),
        model_cfg=_model_cfg(config), dataset_cfg={'name': dataset}, degradation_cfg=degradation,
        inference_cfg=inference,
        metrics_dir=folder/'metrics', images_dir=folder/'images', model_tag=config['model_id'],
        batch_callback=collect if pairs is not None else None)
    if pairs is not None and seen != len(pairs['indices']):
        raise ValueError('Stage recovery did not cover every paired test sample.')
    if cache_config:
        save_method_cache(cache_config, ratio, 'irae', pairs, torch.cat(restored), config,
                          time.perf_counter()-started, {'producer': 'E06', 'split': 'test'})
    return result, inference


def metric_row(dataset, ratio, method, clean, restored):
    return dict(dataset=dataset, ratio=ratio, method=method, samples=len(clean),
                psnr=float(avg_psnr_pytorch(clean, restored)), ssim=float(avg_ssim_pytorch(clean, restored)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', type=canonical_dataset, default=['mnist', 'cifar10', 'celeba-hq-64'])
    parser.add_argument('--ratios', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--methods', nargs='+', type=canonical_method, choices=list(ADAPTERS), default=['ae', 'irae'], help='Recovery methods; aefp is the public alias for irae.')
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--data-root', help='Dataset root; use a separate call for each dataset when supplying it.')
    parser.add_argument('--include-stages', action='store_true', help='Also compute observation and C/CF metrics for E05 controls.')
    parser.add_argument('--recovery-cache-root', help='Export shared full-test tensors for E07 and qualitative figures.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error('--num-samples must be positive.')
    for name in ('datasets', 'ratios', 'methods'):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f'--{name} must contain unique values.')
    if args.data_root and len(args.datasets) != 1:
        parser.error('--data-root requires exactly one dataset.')
    # Reject unavailable preset combinations before creating output or recovery caches.
    for dataset in args.datasets:
        for ratio in args.ratios:
            for method in args.methods:
                resolve_config(dataset, degradation_cfg(ratio, 42), method)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'status.json', dict(complete=False))
    rows, configs = [], {}
    for dataset in args.datasets:
        for ratio in args.ratios:
            cfg = RecoveryCacheConfig(dataset=dataset, test_num_samples=args.num_samples, data_root=args.data_root)
            if args.recovery_cache_root:
                cfg.recovery_cache_root = args.recovery_cache_root
            pairs = load_or_create_pairs_cache(cfg, ratio)[0] if args.recovery_cache_root else collect_test_pairs(cfg, ratio)
            if args.num_samples is not None and len(pairs['indices']) != args.num_samples:
                raise ValueError('Paired test inputs do not cover the requested sample count.')
            if args.include_stages:
                rows.append(metric_row(dataset, ratio, 'observation', pairs['clean'], pairs['degraded']))
            for method in args.methods:
                if args.include_stages and method == 'irae':
                    result, config = evaluate_stages(dataset, ratio, args, pairs, cfg if args.recovery_cache_root else None)
                    rows.extend(dict(dataset=dataset, ratio=ratio, method=name, samples=len(pairs['indices']),
                                     psnr=row['psnr'], ssim=row['ssim'])
                                for name, row in zip(('irae_C', 'irae_CF', 'irae'), result['stage_rows']))
                elif args.recovery_cache_root:
                    record, _ = load_or_create_method_cache(cfg, ratio, method, pairs, torch.device(args.device))
                    rows.append(metric_row(dataset, ratio, method, pairs['clean'], record['reconstruction']))
                    config = record['resolved_config']
                else:
                    result = recover_dataset(dataset, degradation_cfg(ratio, 42), method,
                                         num_samples=args.num_samples, device=args.device, root=args.data_root)
                    if not torch.equal(result.indices, pairs['indices']) or not torch.equal(result.observation, pairs['degraded']):
                        raise ValueError('Recovery and metric samples are not aligned.')
                    restored = result.reconstruction
                    if not torch.isfinite(restored).all():
                        raise ValueError('Non-finite recovered image.')
                    rows.append(metric_row(dataset, ratio, method, pairs['clean'], restored))
                    config = plain(result.resolved_config)
                configs[f'{dataset}/{ratio}/{method}'] = config
                print(rows[-1], flush=True)
                with (args.output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (args.output / 'config.json').write_text(json.dumps(dict(
                    requested_samples=args.num_samples, partial_test=args.num_samples is not None,
                    include_stages=args.include_stages,
                    presets=configs), indent=2), encoding='utf-8')
    write_json(args.output / 'status.json', dict(complete=True, metric_rows=len(rows)))


if __name__ == '__main__':
    main()
