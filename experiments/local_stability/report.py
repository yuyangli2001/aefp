"""Recompute local-rank diagnostics from raw spectra and residuals."""

import numpy as np
import pandas as pd

from experiments.capacity_geometry import inputs as io
from experiments.local_stability import inputs, metrics
from experiments.local_stability.run import location, perturb, read_measurement

CSV_NAMES = ("anchors", "neighborhood", "model_summary", "capacity_summary")
REPORT_NAMES = tuple(n + ".csv" for n in CSV_NAMES) + ("table.md", "table.tex")
MEANS = ("E_recon", "d_hat", "delta_d", "absolute_dimension_change", "dimension_changed",
         "outside_fraction", "realized_radius")


# Require every model/anchor/radius/direction combination before publishing any summary.
def collect(run, manifest):
    config = manifest["config"]
    arrays = io.read_npz(run / "inputs.npz")
    inputs.validate_inputs(arrays, manifest)
    expected, missing, anchor_rows, rows = [], [], [], []
    key = inputs.execution(run, manifest)["id"] if (run / "execution.json").exists() else None
    for spec in manifest["models"]:
        for index, label, image, directions in zip(arrays["indices"], arrays["labels"], arrays["clean"], arrays["directions"]):
            paths = [location(run, spec, index)] + [location(run, spec, index, r, d)
                     for r in range(len(config["radii"])) for d in range(config["num_directions"])]
            expected.extend(path for path, _ in paths)
            absent = [path.relative_to(run).as_posix() for path, _ in paths if not path.exists()]
            if absent:
                missing.extend(absent)
                continue
            anchor_path, anchor_identity = paths[0]
            anchor = read_measurement(anchor_path, key, anchor_identity, config, image.size, spec["dimension"])
            meta = dict(model_id=spec["model_id"], dimension=spec["dimension"], seed=spec["seed"],
                        sample_index=int(index), label=int(label))
            for epsilon in config["thresholds"]:
                anchor_rows.append(dict(**meta, epsilon=epsilon, **metrics.anchor_metrics(anchor, epsilon)))
            for radius_index, radius in enumerate(config["radii"]):
                for direction_index, direction in enumerate(directions):
                    path, identity = location(run, spec, index, radius_index, direction_index)
                    point = read_measurement(path, key, identity, config, image.size, spec["dimension"])
                    image_perturbed = perturb(image, direction, radius)
                    actual = float(np.linalg.norm(image_perturbed - image) / np.sqrt(image.size))
                    if not np.isclose(actual, radius, rtol=1e-10, atol=1e-14):
                        raise ValueError("Perturbation radius does not match the frozen protocol.")
                    for epsilon in config["thresholds"]:
                        rows.append(dict(**meta, epsilon=epsilon, radius=radius, direction_index=direction_index,
                                         realized_radius=actual, outside_fraction=float(((image_perturbed < 0) | (image_perturbed > 1)).mean()),
                                         **metrics.compare(anchor, point, epsilon)))
    if missing:
        return {}, missing, expected
    anchors, neighborhood = pd.DataFrame(anchor_rows), pd.DataFrame(rows)
    models, capacity = aggregate(neighborhood)
    return dict(zip(CSV_NAMES, (anchors, neighborhood, models, capacity))), [], expected


# Training seeds are the replicate units; zero-rank anchors remain in dimension statistics.
def aggregate(frame):
    summaries = []
    keys = ["model_id", "dimension", "seed", "epsilon", "radius"]
    for values, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update({name: float(group[name].mean()) for name in MEANS})
        row.update(anchor_count=group.sample_index.nunique(), pair_count=len(group),
                   zero_anchor_fraction=float(group.d_hat.eq(0).mean()),
                   delta_d_p95=float(group.delta_d.quantile(.95)))
        summaries.append(row)
    models = pd.DataFrame(summaries)
    names = [n for n in models if n not in keys]
    summaries = []
    for (dimension, epsilon, radius), group in models.groupby(["dimension", "epsilon", "radius"], sort=True):
        row = dict(dimension=int(dimension), epsilon=epsilon, radius=radius, model_seed_count=len(group))
        for name in names:
            row.update({name + "_mean": float(group[name].mean()), name + "_std": float(group[name].std(ddof=1)),
                        name + "_seed_count": int(group[name].count())})
        summaries.append(row)
    return models, pd.DataFrame(summaries)


def tables(summary, manifest):
    config = manifest["config"]
    main = summary.loc[summary.epsilon == config["epsilon"]]

    markdown = ["# Finite-accuracy local-rank diagnostics", "",
                f"Split={config['split']}; anchors={manifest['anchor_count']}/{manifest['pool_count']}; "
                f"directions={config['num_directions']}; epsilon={config['epsilon']}; normalized radii={config['radii']}.", "",
                "Entries are means across training seeds; sample SDs remain in CSV (SD=NA for one seed).",
                "These sampled finite-threshold dimension diagnostics do not certify neighborhood constant rank or an exact fixed-point manifold. See CSV for dimension changes and denominators.", "",
                "| d_b | E_recon | mean d_hat | Delta_d(r1) | Delta_d(r2) |",
                "|---:|---:|---:|---:|---:|"]
    latex = [r"% Entries are means across training seeds; split and sample counts are recorded in table.md.",
             r"\begin{tabular}{@{}c|rrrr@{}}", r"\toprule",
             r"$d_b$ & $E_{\mathrm{recon}}$ & $\overline{\widehat d}_{\epsilon}$ & $\Delta_d(r_1)$ & $\Delta_d(r_2)$ \\", r"\midrule"]
    for dimension, group in main.groupby("dimension", sort=True):
        first, second = [group.loc[group.radius == radius].iloc[0] for radius in config["radii"]]
        cells = [str(int(dimension)), f"{first.E_recon_mean:.4f}", f"{first.d_hat_mean:.2f}",
                 f"{first.delta_d_mean:.3f}", f"{second.delta_d_mean:.3f}"]
        for destination, tex in ((markdown, False), (latex, True)):
            destination.append(" & ".join("$" + c + "$" for c in cells) + r" \\" if tex else "| " + " | ".join(cells) + " |")
    latex += [r"\bottomrule", r"\end{tabular}"]
    return {"table.md": "\n".join(markdown) + "\n", "table.tex": "\n".join(latex) + "\n"}


def summarize(run, manifest):
    destination = run / "report"
    destination.mkdir(parents=True, exist_ok=True)
    # Remove only this reporter's generated publication artifacts before checking completeness.
    for name in (*REPORT_NAMES, "complete.json", "audit.json"):
        (destination / name).unlink(missing_ok=True)
    io.write_json(destination / "progress.json", dict(manifest_id=manifest["id"], complete=False, status="checking"))
    frames, missing, paths = collect(run, manifest)
    io.write_json(destination / "progress.json", dict(manifest_id=manifest["id"], complete=False,
                  status="incomplete" if missing else "writing", expected_shards=len(paths), missing=missing))
    if missing:
        print(f"Incomplete: {len(missing)}/{len(paths)} measurement shards missing.", flush=True)
        return False
    for name, frame in frames.items():
        frame.to_csv(destination / (name + ".csv"), index=False)
    for name, content in tables(frames["capacity_summary"], manifest).items():
        (destination / name).write_text(content, encoding="utf-8")
    checked = paths + [destination / name for name in REPORT_NAMES]
    io.write_json(destination / "complete.json", dict(manifest_id=manifest["id"],
                  execution_id=inputs.execution(run, manifest)["id"], complete=True,
                  model_count=len(manifest["models"]), anchor_count=manifest["anchor_count"],
                  measurement_shards=len(paths), rows={k: len(v) for k, v in frames.items()},
                  artifacts=[p.relative_to(run).as_posix() for p in checked]))
    io.write_json(destination / "progress.json", dict(manifest_id=manifest["id"], complete=True,
                  expected_shards=len(paths), missing=[]))
    print(f"Complete: {len(paths)} measurements; report={destination}", flush=True)
    return True
