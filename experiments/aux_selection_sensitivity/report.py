"""Summarize per-image recovery metrics and plot the budget grid."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def read_cell(path, expected_count=None):
    with np.load(path, allow_pickle=False) as saved:
        values, indices = saved['metrics'], saved['indices']
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices)
            or len(np.unique(indices)) != len(indices) or (indices < 0).any()
            or values.shape != (len(indices), 2) or not np.isfinite(values).all()
            or (expected_count is not None and len(indices) != expected_count)):
        raise ValueError(f'Invalid metrics or sample coverage: {path}')
    return values, indices


# Compare only cells evaluated on the same validation images.
def collect(output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for dataset, data in manifest['datasets'].items():
        coverage = output / dataset / 'coverage.json'
        expected_count = manifest.get('num_samples')
        if coverage.exists():
            measured_count = json.loads(coverage.read_text(encoding='utf-8'))['expected_count']
            if expected_count is not None and measured_count != expected_count:
                raise ValueError(f'Sample coverage differs from the requested count: {dataset}')
            expected_count = measured_count
        elif 'num_samples' in manifest and expected_count is None:
            raise ValueError(f'Missing full-validation coverage metadata: {dataset}')
        reference_indices = None
        for cell in data['cells']:
            values, indices = read_cell(output / dataset / (cell['id'] + '.npz'), expected_count)
            if reference_indices is not None and not np.array_equal(indices, reference_indices):
                raise ValueError(f'Cells use different samples: {dataset}')
            reference_indices = indices
            rows.append(dict(dataset=dataset, cell=cell['id'], delta_c=cell['delta_c'],
                delta_f=cell['delta_f'], c_stage=cell['stages'][0], f_stage=cell['stages'][1],
                samples=len(indices), psnr=float(values[:, 0].mean()), ssim=float(values[:, 1].mean())))
    return manifest, _grid_results(pd.DataFrame(rows), manifest['config']['grid'])


def summarize(output):
    manifest, frame = collect(output)
    destination = output / 'report'
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / 'grid_results.csv', index=False)
    _plot_grid(frame, manifest['config']['grid'], destination / 'app_sensitivity_heatmap.pdf')
    print(f'Reported {len(frame)} cells: {destination}', flush=True)
    return frame


# Heatmap colors show each dataset's PSNR change from its default cell.
def _grid_results(frame, grid):
    if frame.empty:
        raise ValueError('No sensitivity-grid results.')
    if frame.duplicated(['dataset', 'delta_c', 'delta_f']).any():
        raise ValueError('Duplicate sensitivity-grid cells.')
    if not np.isfinite(frame[['psnr', 'samples']].to_numpy()).all():
        raise ValueError('Non-finite sensitivity-grid results.')
    frame = frame.copy()
    expected = {(c, f) for c in grid['delta_c'] for f in grid['delta_f']} if 'delta_c' in grid else None
    c, f = grid['default']
    for dataset, group in frame.groupby('dataset'):
        if expected is not None and set(zip(group.delta_c, group.delta_f)) != expected:
            raise ValueError(f'Incomplete sensitivity grid: {dataset}')
        if group.samples.nunique() != 1 or (group.samples <= 0).any():
            raise ValueError(f'Inconsistent sensitivity sample counts: {dataset}')
        reference = group.loc[(group.delta_c == c) & (group.delta_f == f), 'psnr']
        if reference.empty:
            raise ValueError(f'Missing default sensitivity cell: {dataset}')
        frame.loc[group.index, 'psnr_vs_default'] = group.psnr - reference.iloc[0]
    frame['same_network'] = frame.c_stage == frame.f_stage
    return frame


def plot_grid(frame, grid, path):
    _plot_grid(_grid_results(frame, grid), grid, path)


@plt.rc_context({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                 'pdf.fonttype': 42, 'ps.fonttype': 42})
def _plot_grid(frame, grid, path):
    datasets = frame.dataset.unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 4), squeeze=False)
    bound = max(0.1, float(frame.psnr_vs_default.abs().max()))
    for ax, dataset in zip(axes[0], datasets):
        selected = frame.loc[frame.dataset == dataset]
        delta = selected.pivot(index='delta_c', columns='delta_f', values='psnr_vs_default').reindex(index=grid['delta_c'], columns=grid['delta_f'])
        scores = selected.pivot(index='delta_c', columns='delta_f', values='psnr').reindex_like(delta)
        same = selected.pivot(
            index='delta_c', columns='delta_f', values='same_network').reindex_like(delta)
        im = ax.imshow(delta.to_numpy(), cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
        for i, j in np.ndindex(scores.shape):
            color = 'white' if abs(delta.iloc[i, j]) > 0.6 * bound else 'black'
            mark = r'$\dagger$' if same.iloc[i, j] else ''
            ax.text(j, i, f'{scores.iloc[i, j]:.3f}{mark}', ha='center', va='center', fontsize=9, color=color)
        c, f = grid['default']
        ax.add_patch(Rectangle((grid['delta_f'].index(f) - 0.5, grid['delta_c'].index(c) - 0.5),
                               1, 1, fill=False, edgecolor='black', linewidth=1.5))
        ax.set(title=f'{dataset} (n={int(selected.samples.iloc[0])})', xlabel='Fine budget (dB)', ylabel='Coarse budget (dB)',
               xticks=range(len(delta.columns)), xticklabels=delta.columns,
               yticks=range(len(delta.index)), yticklabels=delta.index)
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.18, wspace=0.5)
    fig.colorbar(im, cax=fig.add_axes([0.88, 0.22, 0.015, 0.62]), label='PSNR change from default (dB)')
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/aux_selection_sensitivity'))
    summarize(parser.parse_args().output)
