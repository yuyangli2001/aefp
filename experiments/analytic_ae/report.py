"""Plot saved analytic recovery and mismatch tables."""

from __future__ import annotations
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .common import SynConfig


# Check coverage before aggregation can hide missing or repeated measurements.
def _require_coverage(frame, keys, expected):
    if frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError(f"Incomplete or duplicated plot data for {', '.join(keys)}.")


def _contour_boundary(
    ax: plt.Axes,
    frame: pd.DataFrame,
    config: SynConfig,
    dimensions: Optional[np.ndarray] = None,
    label: bool = True,
) -> None:
    dimensions = np.sort(frame["dimension"].unique()) if dimensions is None else dimensions
    supports = np.sort(frame["support_size"].unique())
    mu = frame.pivot(index="support_size", columns="dimension", values="mu_actual_mean").reindex(
        index=supports, columns=dimensions
    )
    tau = 1.0 - 2.0 * supports[:, None] / config.m * mu.to_numpy() * dimensions[None, :]
    finite = tau[np.isfinite(tau)]
    if min(tau.shape) < 2:
        return
    if len(finite) and finite.min() <= 0 <= finite.max():
        contour = ax.contour(
            np.arange(len(dimensions)), np.arange(len(supports)), tau, levels=(0.0,), colors="black", linewidths=1.2
        )
        if label:
            ax.clabel(contour, fmt={0.0: "Certified sufficient boundary"}, fontsize=7)


def plot_recovery_heatmaps(config: SynConfig, cells: pd.DataFrame) -> None:
    _require_coverage(cells, ["target_mu", "dimension", "support_size"], {
        (target, dimension, support) for dimension, target in config.recovery_pairs()
        for support in config.support_sizes
    })
    rates = cells[["eta_certificate_rate", "pgd_instance_success_rate"]].to_numpy()
    if not (np.isfinite(rates).all() and ((0 <= rates) & (rates <= 1)).all()
            and np.isfinite(cells["mu_actual_mean"]).all()):
        raise ValueError("Recovery plots require finite coherence and rates in [0, 1].")
    targets = np.sort(cells["target_mu"].unique())
    dimensions = np.asarray(config.dimensions)
    supports = np.asarray(config.support_sizes)
    # Thin y ticks from the top while preserving every heatmap row.
    tick_indices = np.arange(len(supports))[::-2]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#D9D9D9")
    with plt.rc_context(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(len(targets), 2, figsize=(7.4, 5.8), sharex=True, sharey=True, squeeze=False)
        image_handle = None
        for row, target in enumerate(targets):
            frame = cells[cells["target_mu"] == target]
            matrices = (
                frame.pivot(index="support_size", columns="dimension", values="eta_certificate_rate").reindex(
                    index=supports, columns=dimensions
                ),
                frame.pivot(index="support_size", columns="dimension", values="pgd_instance_success_rate").reindex(
                    index=supports, columns=dimensions
                ),
            )
            for column, matrix in enumerate(matrices):
                ax = axes[row, column]
                image_handle = ax.imshow(
                    np.ma.masked_invalid(matrix.to_numpy()),
                    origin="lower",
                    aspect="auto",
                    vmin=0,
                    vmax=1,
                    cmap=cmap,
                )
                _contour_boundary(ax, frame, config, dimensions, label=False)
                ax.text(
                    0.02,
                    0.96,
                    rf"$\mu={target:g}$",
                    transform=ax.transAxes,
                    color="white",
                    fontweight="bold",
                    ha="left",
                    va="top",
                )
                ax.set_xticks(np.arange(len(dimensions)), dimensions, rotation=45, ha="right")
                ax.set_yticks(tick_indices, [f"{supports[i]/config.m:.3f}" for i in tick_indices])
            axes[row, 0].set_ylabel(r"Corruption fraction $\rho_s$")

        axes[0, 0].set_title("Local Minimizer Certification Rate")
        axes[0, 1].set_title("Numerical AEFP Recovery Rate")
        axes[-1, 0].set_xlabel("Tangent dimension $d$")
        axes[-1, 1].set_xlabel("Tangent dimension $d$")
        fig.subplots_adjust(left=0.10, right=0.88, bottom=0.17, top=0.90, wspace=0.08, hspace=0.12)
        colorbar_ax = fig.add_axes((0.90, 0.22, 0.018, 0.62))
        fig.colorbar(image_handle, cax=colorbar_ax, label="Success rate")
        fig.legend(
            handles=(
                Line2D((0,), (0,), color="black", linewidth=1.2, label="Sufficient boundary"),
                Patch(facecolor="#D9D9D9", label="Excluded settings"),
            ),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=False,
            prop={"weight": "normal", "size": 9.5},
        )
        output = config.output_path()
        fig.savefig(output / "fig_syn_recovery.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(output / "fig_syn_recovery.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


MISMATCH_STYLE = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 17,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 18,
    "axes.labelweight": "normal", "xtick.labelsize": 17, "ytick.labelsize": 17,
    "legend.fontsize": 16, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _draw_mismatch_panel(ax, config, frame, kind, title):
    columns = {"set_adversarial": ("support_size", "delta_over_radius", "error_over_radius", "median"),
               "set_random": ("support_size", "delta_over_radius", "mean_error_over_radius", "mean"),
               "tangent": ("s", "epsilon_T_r_star_over_R", "recovery_error_over_R", "mean")}
    support, x_column, y_column, statistic = columns[kind]
    if kind == "tangent":
        _require_coverage(frame, ["s", "trial", "theta_deg"], {
            (size, trial, angle) for size in config.mismatch_support_sizes
            for trial in range(config.mismatch_random_trials_per_subspace)
            for angle in config.tangent_angles
        })
        if not frame["check_pass"].eq(True).all():
            raise ValueError("Tangent mismatch results contain failed numerical checks.")
    elif kind == "set_adversarial":
        seeds = frame["subspace_seed"].unique()
        if len(seeds) != config.mismatch_num_subspace_instances:
            raise ValueError("Adversarial mismatch results have an unexpected number of subspaces.")
        _require_coverage(frame, [support, x_column, "subspace_seed"], {
            (size, ratio, seed)
            for size in config.mismatch_support_sizes for ratio in config.mismatch_delta_over_radius
            for seed in seeds
        })
    else:
        _require_coverage(frame, [support, x_column], {
            (size, ratio) for size in config.mismatch_support_sizes for ratio in config.mismatch_delta_over_radius
        })
    if not np.isfinite(frame[[support, x_column, y_column]].to_numpy()).all() or (frame[y_column] < 0).any():
        raise ValueError("Mismatch curves require finite, nonnegative values.")
    limit = float(frame[x_column].max())
    ax.plot([0, limit], [0, limit], color="0.45", linestyle="--", linewidth=1.0, label="Unit slope")
    for index, (size, group) in enumerate(frame.groupby(support, sort=True)):
        curve = group.groupby(x_column)[y_column].agg(statistic)
        ax.plot(curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float),
                color=("#0072B2", "#D55E00")[index % 2], marker=("o", "s")[index % 2],
                markersize=3.8, linewidth=1.2, label=rf"$\rho_s\approx {size / config.mismatch_m:.2f}$")
    xlabel = (r"Tangent mismatch $\varepsilon_T r^\star/r_{\mathrm{exp}}$" if kind == "tangent"
              else r"Set mismatch $\varepsilon_M/r_{\mathrm{exp}}$")
    ax.set(xlim=(0, limit * 1.02), ylim=(0, None), xlabel=xlabel, title=title,
           ylabel=r"Recovery error $\Vert\hat{\mathbf{x}}-\mathbf{x}^\star\Vert_2/r_{\mathrm{exp}}$")
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_box_aspect(1)
    ax.legend(frameon=False, loc="upper left", borderaxespad=0.35,
              handlelength=1.1, handletextpad=0.4, labelspacing=0.3)


# Plot the median error across subspaces for adversarial set mismatch.
def plot_adversarial_mismatch(config, frame):
    with plt.rc_context(MISMATCH_STYLE):
        fig, ax = plt.subplots(figsize=(4.5, 4.1))
        _draw_mismatch_panel(ax, config, frame, "set_adversarial", "Adversarial Set Mismatch")
        fig.tight_layout(pad=0.6)
        for extension in ("pdf", "png"):
            fig.savefig(config.output_path() / f"fig_mismatch_bias.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_mismatch_comparison(config, random_curve, tangent):
    with plt.rc_context(MISMATCH_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.1))
        _draw_mismatch_panel(axes[0], config, random_curve, "set_random", "Random Set Mismatch")
        _draw_mismatch_panel(axes[1], config, tangent, "tangent", "Random Tangent Mismatch")
        axes[1].set_ylabel("")
        for label, ax in zip(("(a)", "(b)"), axes):
            ax.text(0.98, 0.04, label, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")
        fig.tight_layout(pad=0.6)
        for extension in ("pdf", "png"):
            fig.savefig(config.output_path() / f"fig_mismatch_bias_combined.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
