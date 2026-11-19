"""Generate experiment figures from saved results."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import yaml

from experiments.analytic_ae.common import SynConfig
from experiments.analytic_ae.report import plot_recovery_heatmaps, plot_mismatch_comparison
from experiments.aux_selection_sensitivity.report import plot_grid
from experiments.auxiliary_continuation.visualize import replot_reconstruction_strip
from experiments.capacity_geometry.run_capacity import plot_summary, summarize
from experiments.sparse_restoration.visualize import COMPARISON_PANELS, replot_from_cache

FIGURES = ('fig_syn_recovery', 'fig_mismatch_bias_combined', 'fig_capacity_tradeoff',
           'fig_sparse_restoration_celeba64_r0p2', 'fig_3stage_sample_reconstructions',
           'app_sensitivity_heatmap')
RECOVERY_METHODS = tuple(key for key, _ in COMPARISON_PANELS if key not in ('clean', 'degraded'))


# Check inputs, render to a temporary directory, and save the completed outputs.
def generate(args):
    destination = args.output_dir.resolve()
    sources = {
        'analytic_config': args.analytic_dir / 'resolved_config.json',
        'recovery_cells': args.analytic_dir / 'results_recovery_cells.csv',
        'mismatch_curve': args.analytic_dir / 'results_mismatch_random_curve.csv',
        'tangent_results': args.tangent_results,
        'capacity': args.capacity_csv,
        'endpoints': args.endpoints,
        'sensitivity': args.sensitivity_csv,
        'sensitivity_config': args.sensitivity_config,
    }
    sources.update({f'cache_{name}': args.recovery_dir / f'{name}.pt'
                    for name in ('test_pairs', *RECOVERY_METHODS)})
    if any(path.suffix.lower() == '.pdf' for path in sources.values()):
        raise ValueError('Figure generation requires numerical inputs, never PDFs.')
    if any(path.resolve().is_relative_to(destination) for path in sources.values()):
        raise ValueError('The output directory must not contain numerical inputs.')
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.sample_index < 0:
        raise ValueError('sample_index must be nonnegative.')
    if destination.exists() and any(destination.iterdir()) and not getattr(args, 'overwrite', False):
        raise FileExistsError(f'{destination} is not empty; use --overwrite or a new output directory.')
    config = SynConfig.from_json(sources['analytic_config'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Check every figure before replacing existing files.
    with TemporaryDirectory(prefix='.figures-', dir=destination.parent) as temporary:
        staged = Path(temporary)
        _render(args, staged, config, sources)
        for name in FIGURES:
            for suffix in ('.pdf', '.png'):
                path = staged / (name + suffix)
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f'Missing generated figure: {path.name}')
        code_paths = [Path(__file__), *(Path(__file__).parent / name for name in (
            'analytic_ae/report.py', 'capacity_geometry/run_capacity.py', 'sparse_restoration/visualize.py',
            'auxiliary_continuation/visualize.py', 'aux_selection_sensitivity/report.py')),
            Path(__file__).parents[1] / 'utils/figure_layout.py']
        manifest = dict(complete=True, source_files=[str(path.resolve()) for path in sources.values()],
            code_files=[str(path.resolve()) for path in code_paths], sample_index=args.sample_index,
            pdf_postprocessing=False, inference_rerun=False,
            figures={name: {kind: f'{name}.{kind}' for kind in ('pdf', 'png')} for name in FIGURES})
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / 'figure_manifest.json'
        marker.unlink(missing_ok=True)
        for path in staged.rglob('*'):
            if path.is_file():
                target = destination / path.relative_to(staged)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
        marker.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Generated {len(FIGURES)} experiment figures in {destination}')
    return destination


# Render figures and supporting summaries from saved measurements.
def _render(args, destination, config, sources):
    config = replace(config, output_dir=str(destination))
    plot_recovery_heatmaps(config, pd.read_csv(sources['recovery_cells']))
    random_curve = pd.read_csv(sources['mismatch_curve'])
    tangent = pd.read_csv(sources['tangent_results'])
    plot_mismatch_comparison(config, random_curve, tangent)

    # Aggregate capacity runs before plotting their means and variation.
    capacity_data = destination / 'capacity_data'
    capacity_data.mkdir(exist_ok=True)
    summary = summarize(pd.read_csv(sources['capacity']).to_dict('records'), capacity_data)
    plot_summary(summary, destination / 'capacity_tradeoff.png')

    # Plot cached reconstructions and stage endpoints without running recovery.
    replot_from_cache(args.recovery_dir, destination, args.sample_index, RECOVERY_METHODS)
    replot_reconstruction_strip(sources['endpoints'], destination, force=True)

    # Color the sensitivity grid by PSNR change from its configured default.
    grid = yaml.safe_load(sources['sensitivity_config'].read_text(encoding='utf-8'))['grid']
    plot_grid(pd.read_csv(sources['sensitivity']), grid, destination / 'app_sensitivity_heatmap.pdf')


# Parse input paths and output options.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analytic-dir', type=Path, required=True)
    parser.add_argument('--tangent-results', type=Path, required=True,
                        help='Random tangent-mismatch results from the unified mismatch experiment.')
    parser.add_argument('--capacity-csv', type=Path, required=True)
    parser.add_argument('--recovery-dir', type=Path, required=True)
    parser.add_argument('--endpoints', type=Path, required=True)
    parser.add_argument('--sensitivity-csv', type=Path, required=True)
    parser.add_argument('--sensitivity-config', type=Path,
                        default=Path(__file__).parent / 'aux_selection_sensitivity/configs/sensitivity.yaml')
    parser.add_argument('--sample-index', type=int, default=18)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/figures'))
    parser.add_argument('--overwrite', action='store_true', help='Replace generated files in an existing output directory.')
    return generate(parser.parse_args(argv))


if __name__ == '__main__':
    main()
