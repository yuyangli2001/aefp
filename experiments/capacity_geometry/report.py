"""Completeness checks, paired summaries, and publication-ready diagnostics."""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.capacity_geometry import inputs as io


# Assemble only the exact expected shards; missing work is never silently dropped.
def collect_family(run, manifest, frozen, family):
    geometry, recovery, missing = [], [], []
    samples = io.input_samples(run, manifest, "test")
    for spec in (s for s in manifest["models"] if s["family"] == family):
        for index, label, image in samples:
            path = run / family / spec["model_id"] / f"geometry_{index}.npz"
            if not path.exists():
                missing.append(path.relative_to(run).as_posix())
                continue
            rows, data = io.read_shard(path, frozen["id"])
            if len(rows) != len(frozen["thresholds"]) or {r["epsilon"] for r in rows} != set(frozen["thresholds"]):
                raise ValueError(f"Incomplete threshold rows: {path}")
            if any(r["model_id"] != spec["model_id"] or r["sample_index"] != index or r["label"] != label for r in rows):
                raise ValueError(f"Geometry identity mismatch: {path}")
            spectrum = data["spectrum"]
            if spectrum.shape != (image.size,) or not np.isfinite(spectrum).all() or (spectrum < 0).any():
                raise ValueError(f"Invalid spectrum: {path}")

            for row in rows:
                dimension = row["d_hat"]
                if dimension != int((spectrum <= row["epsilon"]).sum()) or not 0 <= dimension <= spec["dimension"]:
                    raise ValueError(f"Geometry dimension mismatch: {path}")
                if dimension == 0:
                    if row["mu_hat"] is not None or row["K_hat"] is not None:
                        raise ValueError(f"Empty geometry must have undefined coherence: {path}")
                elif (not all(value is not None and math.isfinite(value) for value in (row["mu_hat"], row["K_hat"]))
                      or not 1 - 1e-7 <= row["mu_hat"] <= image.size / dimension + 1e-7
                      or not math.isclose(row["K_hat"], row["mu_hat"] * dimension, rel_tol=1e-8, abs_tol=1e-8)):
                    raise ValueError(f"Invalid coherence metrics: {path}")
                if any(row[name] is None or not math.isfinite(row[name]) or row[name] < 0 for name in ("E_recon", "jacobian_frobenius_sq")):
                    raise ValueError(f"Invalid geometry metrics: {path}")
            geometry.extend(rows)
        if family == "capacity":
            for batch_id, batch in io.input_batches(run, manifest, "test"):
                for seed in manifest["config"]["init_seeds"]:
                    path = run / family / spec["model_id"] / f"recovery_batch{batch_id}_init{seed}.npz"
                    if not path.exists():
                        missing.append(path.relative_to(run).as_posix())
                        continue
                    rows, data = io.read_shard(path, frozen["id"])
                    if [r["sample_index"] for r in rows] != batch["indices"].tolist():
                        raise ValueError(f"Recovery batch ordering changed: {path}")
                    if any(r["model_id"] != spec["model_id"] or r["batch_id"] != batch_id or r["init_seed"] != seed for r in rows):
                        raise ValueError(f"Recovery identity mismatch: {path}")
                    names = ("E_rest", "E_init", "G_init", "fixed_point_residual", "realized_support_ratio", "init_distance_ratio", "clip_fraction")
                    if any(r[name] is None or not math.isfinite(r[name]) or r[name] < 0 for r in rows for name in names):
                        raise ValueError(f"Invalid recovery metrics: {path}")
                    if data["recovered"].shape != batch["clean"].shape or not np.isfinite(data["recovered"]).all():
                        raise ValueError(f"Invalid recovered images: {path}")
                    recovery.extend(rows)
    return pd.DataFrame(geometry), pd.DataFrame(recovery), missing


# Average initialization directions before treating a trained model as one replicate.
def aggregate_models(geometry, recovery, specs):
    if geometry.empty or geometry.duplicated(["model_id", "sample_index", "epsilon"]).any():
        raise ValueError("Require unique, nonempty geometry records.")
    metadata = pd.DataFrame([{k: spec[k] for k in ("model_id", "dimension", "seed", "family")} for spec in specs])
    paired = geometry.merge(metadata, on="model_id", validate="many_to_one")
    if not recovery.empty:
        if recovery.duplicated(["model_id", "sample_index", "init_seed"]).any():
            raise ValueError("Duplicate recovery initialization records.")
        keys = ["model_id", "sample_index"]
        if set(map(tuple, geometry[keys].to_numpy())) != set(map(tuple, recovery[keys].to_numpy())):
            raise ValueError("Geometry and recovery must use the same model/image pairs.")
        numeric = ["E_rest", "E_init", "G_init", "fixed_point_residual", "realized_support_ratio", "init_distance_ratio", "clip_fraction"]
        averaged = recovery.groupby(["model_id", "sample_index"], as_index=False)[numeric].mean()
        paired = paired.merge(averaged, on=["model_id", "sample_index"], validate="many_to_one")
    metrics = [c for c in ("E_recon", "d_hat", "mu_hat", "K_hat", "E_rest", "G_init", "E_init",
                           "fixed_point_residual", "jacobian_frobenius_sq", "clean_psnr") if c in paired]
    summaries = []
    keys = ["model_id", "dimension", "seed", "epsilon"]
    for values, group in paired.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row.update({metric: float(group[metric].mean()) for metric in metrics})
        if "E_rest" in row:
            row["restoration_ratio"] = row["E_rest"] / row["E_recon"] if row["E_recon"] > 0 else np.nan
        row.update(sample_count=len(group), valid_geometry_count=int((group.d_hat > 0).sum()),
                   zero_dimension_fraction=float((group.d_hat == 0).mean()))
        summaries.append(row)
    return pd.DataFrame(summaries), paired


def aggregate_seeds(models):
    metrics = [c for c in models if c not in ("model_id", "dimension", "seed", "epsilon")]
    rows = []
    for (dimension, epsilon), group in models.groupby(["dimension", "epsilon"]):
        row = dict(dimension=int(dimension), epsilon=epsilon, model_seed_count=len(group))
        for name in metrics:
            row[name + "_mean"] = float(group[name].mean())
            row[name + "_std"] = float(group[name].std(ddof=1))
            row[name + "_seed_count"] = int(group[name].count())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(paired):
    keys = ["seed", "sample_index", "epsilon"]
    values = ["K_hat", "E_rest", "E_recon", "G_init", "d_hat"]
    reference = paired.loc[paired.dimension == 96, keys + values].rename(columns={v: v + "_reference" for v in values})
    result = paired.merge(reference, on=keys, validate="many_to_one")
    result["common_nonzero"] = (result.d_hat > 0) & (result.d_hat_reference > 0)
    for name in values[:-1]:
        result["delta_" + name] = result[name] - result[name + "_reference"]
    return result


# Save both a shareable vector figure and a lightweight preview.
def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_capacity(models, summary, deltas, epsilon, destination):
    main = models.loc[models.epsilon == epsilon]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric in zip(axes.flat, ("E_recon", "E_rest", "K_hat", "G_init")):
        for seed, group in main.groupby("seed"):
            group = group.sort_values("dimension")
            axis.plot(group.dimension.to_numpy(), group[metric].to_numpy(), "o-", label=f"seed {int(seed)}")
        axis.set(xlabel="Bottleneck dimension", ylabel=metric, xscale="log", xticks=sorted(main.dimension.unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.legend(fontsize=8)
    save_figure(fig, destination / "seed_trajectories")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, ("d_hat", "K_hat", "zero_dimension_fraction")):
        for threshold, group in summary.groupby("epsilon"):
            group = group.sort_values("dimension")
            axis.errorbar(group.dimension.to_numpy(), group[metric + "_mean"].to_numpy(), yerr=group[metric + "_std"].fillna(0).to_numpy(),
                          marker="o", label=f"epsilon={threshold:g}")
        axis.set(xlabel="Bottleneck dimension", ylabel=metric, xscale="log")
        axis.legend(fontsize=8)
    save_figure(fig, destination / "threshold_sensitivity")

    valid = deltas.loc[(deltas.epsilon == epsilon) & deltas.common_nonzero & (deltas.dimension != 96)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, seed in zip(axes, sorted(main.seed.unique())):
        group = valid.loc[valid.seed == seed]
        scatter = axis.scatter(group.delta_K_hat.to_numpy(), group.delta_E_rest.to_numpy(), c=group.delta_E_recon.to_numpy(),
                               cmap="coolwarm", s=7, alpha=0.4)
        axis.axhline(0, color="gray", lw=0.6)
        axis.axvline(0, color="gray", lw=0.6)
        axis.set(title=f"seed {int(seed)}", xlabel="Delta K versus dimension 96", ylabel="Delta restoration error")
        fig.colorbar(scatter, ax=axis, label="Delta reconstruction error")
    save_figure(fig, destination / "paired_changes")


def write_main_table(summary, epsilon, path):
    main = summary.loc[summary.epsilon == epsilon].sort_values("dimension")
    fields = [("d_hat", 2), ("mu_hat", 2), ("K_hat", 2), ("E_recon", 5), ("E_rest", 5), ("restoration_ratio", 3)]
    lines = ["# Capacity and local geometry", "", "Means across training seeds. Each restoration ratio is formed within a model before averaging.", "",
             "| d_b | d_hat | mu_hat | K_hat | E_recon | E_rest | ξ |", "|---:" * 7 + "|"]
    for _, row in main.iterrows():
        values = []
        for name, digits in fields:
            value = row[name + "_mean"]
            values.append(f"{value:.{digits}f}" if math.isfinite(value) else "NA")
        lines.append(f"| {int(row.dimension)} | " + " | ".join(values) + " |")
    lines += ["", "mu/K use nonzero-dimension samples; see CSV files for valid sample and seed counts. Recovery means use the full subset."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_common_auxiliary(auxiliary, destination):
    metrics = ['E_recon', 'd_hat', 'mu_hat', 'K_hat', 'jacobian_frobenius_sq']
    summaries, membership = [], []
    for epsilon, group in auxiliary.groupby('epsilon'):
        dimensions = group.pivot(index='sample_index', columns='model_id', values='d_hat')
        if dimensions.shape[1] != 3 or dimensions.isna().any().any():
            raise ValueError('Incomplete paired C/F/O measurements.')
        valid = (dimensions > 0).all(axis=1)
        membership.extend(dict(epsilon=epsilon, sample_index=int(index), common_nonzero=bool(ok))
                          for index, ok in valid.items())
        for model_id, rows in group.groupby('model_id'):
            subset = rows[rows.sample_index.isin(valid.index[valid])]
            summaries.append(dict(epsilon=epsilon, model_id=model_id,
                sample_count=len(rows), valid_geometry_count=int((rows.d_hat > 0).sum()),
                common_nonzero_count=len(subset),
                **{'common_' + key: float(subset[key].mean()) for key in metrics}))
    pd.DataFrame(summaries).to_csv(destination / 'auxiliary_common_subset.csv', index=False)
    pd.DataFrame(membership).to_csv(destination / 'auxiliary_common_membership.csv', index=False)
    return pd.DataFrame(summaries)


def summarize_run(run, manifest, frozen):
    destination = run / "report"
    destination.mkdir(exist_ok=True)
    # Invalidate previous completion metadata before inspecting the current shards.
    io.write_json(destination / "complete.json", dict(frozen_id=frozen["id"], capacity_complete=False, auxiliary_complete=False))
    geometry, recovery, missing = collect_family(run, manifest, frozen, "capacity")
    auxiliary, _, aux_missing = collect_family(run, manifest, frozen, "auxiliary")
    io.write_json(destination / "progress.json", dict(capacity_complete=not missing, auxiliary_complete=not aux_missing,
                  missing_capacity=missing, missing_auxiliary=aux_missing))
    if missing:
        print(f"Capacity incomplete: {len(missing)} missing shards; no paper table generated.", flush=True)
        return False
    models, paired = aggregate_models(geometry, recovery, manifest["models"])
    summary, deltas = aggregate_seeds(models), paired_deltas(paired)
    for name, frame in (("geometry", geometry), ("recovery", recovery), ("model_summary", models),
                         ("capacity_summary", summary), ("paired", paired), ("paired_deltas", deltas)):
        frame.to_csv(destination / f"{name}.csv", index=False)
    coverage = deltas.groupby(["dimension", "seed", "epsilon"], as_index=False).agg(
        pair_count=("common_nonzero", "size"), common_nonzero_count=("common_nonzero", "sum"))
    coverage.to_csv(destination / "paired_coverage.csv", index=False)
    plot_capacity(models, summary, deltas, frozen["epsilon"], destination)
    write_main_table(summary, frozen["epsilon"], destination / "capacity_table.md")
    if not aux_missing:
        common = write_common_auxiliary(auxiliary, destination)
        aux_models, _ = aggregate_models(auxiliary, pd.DataFrame(), manifest["models"])
        metadata = pd.DataFrame([{k: s[k] for k in ("model_id", "stage", "regularization")}
                                 for s in manifest["models"] if s["family"] == "auxiliary"])
        aux_summary = aux_models.merge(metadata, on="model_id", validate="many_to_one")
        aux_summary.to_csv(destination / "auxiliary_summary.csv", index=False)
        auxiliary.to_csv(destination / "auxiliary_geometry.csv", index=False)
        names = ["model_id", "stage", "regularization", "E_recon", "d_hat", "mu_hat", "K_hat", "zero_dimension_fraction"]
        main_aux = aux_summary.loc[aux_summary.epsilon == frozen["epsilon"], names].copy()
        shared = common.loc[common.epsilon == frozen["epsilon"]].set_index("model_id")
        for metric in ("E_recon", "d_hat", "mu_hat", "K_hat"):
            main_aux[metric] = main_aux.model_id.map(shared["common_" + metric])
        main_aux["common_nonzero_count"] = main_aux.model_id.map(shared.common_nonzero_count)
        names = names + ["common_nonzero_count"]
        lines = ["# C/F/O geometry diagnostic", "", "E_recon, d_hat, mu_hat and K_hat use the common nonzero-dimension subset; zero_dimension_fraction uses all images.", "", "| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
        for _, row in main_aux.iterrows():
            lines.append("| " + " | ".join("NA" if pd.isna(v) else f"{v:.5g}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
        (destination / "auxiliary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    io.write_json(destination / "complete.json", dict(frozen_id=frozen["id"], capacity_complete=True, auxiliary_complete=not aux_missing))
    print(f"Report written to {destination}", flush=True)
    return True
