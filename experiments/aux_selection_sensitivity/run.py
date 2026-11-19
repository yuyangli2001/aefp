"""Evaluate the default auxiliary-selection grid and generate PSNR/SSIM results."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from methods.continuation import irae_solver
from models.loading import build_ae_model, load_model_state
from recovery.adapters.ae import _model_cfg
from utils.utils_dataloader import build_degradation_config, get_degradation_dataloader
from utils.utils_metrics import compute_psnr, compute_ssim

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / 'configs/sensitivity.yaml'
DATASETS = {'mnist': 28, 'cifar10': 32, 'celeba-hq-64': 64}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def make_plan(config):
    grid = dict(config['grid'])
    if grid.pop('exclude', []):
        raise ValueError('Sensitivity runs require a complete budget grid.')
    config = dict(config, grid=grid)
    for name in ('delta_c', 'delta_f'):
        values = grid[name]
        if (not values or len(set(values)) != len(values)
                or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values)):
            raise ValueError(f'{name} must contain unique finite nonnegative budgets.')
    pairs = [(c, f) for c in grid['delta_c'] for f in grid['delta_f']]
    if tuple(grid['default']) not in pairs:
        raise ValueError('The default must belong to the budget grid.')
    if not config['datasets']:
        raise ValueError('Select at least one dataset.')
    if build_degradation_config(config['degradation']).mode != 'sparse':
        raise ValueError('Auxiliary sensitivity requires sparse corruption.')
    recovery = config['recovery']
    steps = recovery['jinit_stage_iters']
    if len(steps) != 3 or any(type(step) is not int or step <= 0 for step in steps):
        raise ValueError('Require three positive integer stage budgets.')
    if not isinstance(recovery['is_lambda_update'], bool):
        raise ValueError('is_lambda_update must be a boolean.')
    models = yaml.safe_load((ROOT / config['checkpoint_manifest']).read_text(encoding='utf-8'))['models']
    datasets = {}
    for name, spec in config['datasets'].items():
        batch_size = spec['recovery_batch_size']
        if name not in DATASETS or type(batch_size) is not int or batch_size <= 0:
            raise ValueError(f'Invalid dataset or recovery batch size: {name}')
        stages = spec['selected_stages']
        for budget in set(grid['delta_c']) | set(grid['delta_f']):
            stage = stages.get(budget)
            if type(stage) is not int or stage < 0:
                raise ValueError(f'Missing or invalid stage for {name}, budget {budget}.')
        cells = [dict(id=f'c{c}_f{f}', delta_c=c, delta_f=f, stages=[stages[c], stages[f]]) for c, f in pairs]
        datasets[name] = dict(spec=spec, checkpoint=models[spec['model_id']], cells=cells)
    return dict(config=config, datasets=datasets)


def make_loader(dataset, spec, degradation, workers, num_samples=None):
    source = 'celeba-hq-256' if dataset == 'celeba-hq-64' else dataset
    return get_degradation_dataloader(source, 'val', degradation,
        batch_size=spec['recovery_batch_size'], num_workers=workers, shuffle=False,
        image_size=DATASETS[dataset],
        sample_indices=list(range(num_samples)) if num_samples is not None else None)


def load_models(dataset, data, cell, device, cache):
    models = []
    for stage in [*cell['stages'], 0]:
        if stage not in cache:
            model = build_ae_model(_model_cfg(dict(dataset=dataset, checkpoint=data['checkpoint']))).to(device)
            rel = f'jinit/cache/stage_{stage}.pth' if stage else 'direct/stage_1.pth'
            load_model_state(model, ROOT/data['checkpoint']['path']/rel, device)
            model.eval().requires_grad_(False)
            cache[stage] = model
        models.append(cache[stage])
    return models


def recover(observation, models, config, spec, device):
    cfg = config['recovery']
    return irae_solver(solver='jinit', observation=observation.to(device), ae_models=models,
        degradation={'mode': 'sparse'}, device=device, lambda_=spec['lambda'],
        lambda_init_gamma=cfg['lambda_init_gamma'], stage_iterations=cfg['jinit_stage_iters'],
        is_lambda_update=cfg['is_lambda_update'], lambda_multiplier=cfg['lambda_multiplier'],
        lambda_check_interval=cfg['lambda_check_interval'], backtracking_rule=cfg['backtracking_rule'],
        sufficient_decrease_constant=cfg['sufficient_decrease_constant'])['reconstruction']


def metrics(clean, restored):
    original = clean.numpy().transpose(0, 2, 3, 1)
    reconstructed = restored.detach().cpu().numpy().transpose(0, 2, 3, 1)
    values = np.asarray([(compute_psnr(a, b), compute_ssim(a, b)) for a, b in zip(original, reconstructed)])
    if values.shape != (len(clean), 2) or not np.isfinite(values).all():
        raise ValueError('Non-finite or incomplete recovery metrics.')
    return values


# Run each budget pair independently, even when it selects the same models.
def evaluate(dataset, data, config, output, device, workers, num_samples, resume):
    from experiments.aux_selection_sensitivity.report import read_cell
    if not resume and any((output / dataset / (cell['id'] + '.npz')).exists() for cell in data['cells']):
        raise FileExistsError('Sensitivity results exist; use --resume or a new output directory.')
    loader = make_loader(dataset, data['spec'], config['degradation'], workers, num_samples)
    expected_count = len(loader.dataset)
    coverage_path = output / dataset / 'coverage.json'
    if expected_count <= 0:
        raise ValueError('Validation loader must contain samples.')
    if coverage_path.exists():
        if json.loads(coverage_path.read_text(encoding='utf-8'))['expected_count'] != expected_count:
            raise ValueError('Validation sample coverage changed; use a new output directory.')
    else:
        if any(coverage_path.parent.glob('*.npz')):
            raise ValueError('Existing cells have no coverage.json; use a new output directory.')
        save_json(coverage_path, dict(expected_count=expected_count))
    expected_indices = np.arange(expected_count)
    cache = {}
    for cell in data['cells']:
        destination = output / dataset / (cell['id'] + '.npz')
        if resume and destination.exists():
            _, saved_indices = read_cell(destination, expected_count)
            if not np.array_equal(saved_indices, expected_indices):
                raise ValueError(f'Cached sample indices differ from the requested validation prefix: {destination}')
            continue
        models = load_models(dataset, data, cell, device, cache)
        values, indices = [], []
        for batch in loader:
            restored = recover(batch['degraded'], models, config, data['spec'], device)
            values.append(metrics(batch['clean'], restored))
            indices.extend(batch['index'].tolist())
        if not np.array_equal(indices, expected_indices):
            raise ValueError('Validation loader returned incomplete or reordered samples.')
        temporary = destination.with_suffix('.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, metrics=np.concatenate(values), indices=indices)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'{dataset} {cell["id"]}: {len(indices)} images', flush=True)


def experiment_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--output', type=Path, default=Path('outputs/aux_selection_sensitivity'))
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS))
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    return parser


def load_config(path, datasets=None):
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if datasets:
        config['datasets'] = {name: config['datasets'][name] for name in datasets}
    return config


def run_experiment(args, num_samples=None):
    from experiments.aux_selection_sensitivity.report import summarize

    if args.workers < 0:
        raise ValueError('Workers must be nonnegative.')
    device = torch.device(args.device)
    plan = make_plan(load_config(args.config, args.datasets))
    config = plan['config']
    plan['num_samples'] = num_samples
    manifest = args.output / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['config']['grid'].get('exclude') == []:
            saved['config']['grid'].pop('exclude')
        if not args.resume or saved != json.loads(json.dumps(plan)):
            raise ValueError('Use --resume with the same configuration, or choose a new output directory.')
    else:
        if any(args.output.glob('*/*.npz')):
            raise ValueError('Existing cells have no manifest.json; use a new output directory.')
        save_json(manifest, plan)
    torch.set_num_threads(4)
    torch.manual_seed(config['degradation']['seed'])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    for name, data in plan['datasets'].items():
        evaluate(name, data, config, args.output, device, args.workers, num_samples, args.resume)
    summarize(args.output)


def main(argv=None):
    run_experiment(experiment_parser(__doc__).parse_args(argv))


if __name__ == '__main__':
    main()
