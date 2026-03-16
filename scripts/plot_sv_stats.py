#!/usr/bin/env python3
"""
Publication-quality plots of singular value statistics from nanochat training runs.

Reads one or more JSONL files produced by nanochat/sv_stats.py and generates
four figure types suitable for scientific journals and conference papers.

Usage
-----
Single experiment:
  python -m scripts.plot_sv_stats $NANOCHAT_BASE_DIR/sv_stats/d24_sv_stats.jsonl

Compare experiments:
  python -m scripts.plot_sv_stats \\
      results/d12_baseline_muon_sv_stats.jsonl \\
      results/d12_muono_sv_stats.jsonl \\
      results/d12_adamo_sv_stats.jsonl \\
      --labels "Muon (baseline)" "MuonO (ours)" "AdamO (ours)" \\
      --output-dir figures/isometry

Options
-------
  --output-dir DIR    where to save figures (default: ./sv_figures)
  --format pdf|png    file format (default: pdf, best for journals)
  --x-axis step|flops x-axis for time-series plots (default: step)
  --labels LABEL ...  names for each experiment (default: stem of filename)
  --dpi N             DPI for raster outputs (default: 300)

Figures produced
----------------
  1. sv_global_evolution.{ext}   — 5-panel global metrics over training
  2. sv_by_type.{ext}            — condition number per weight-type
  3. sv_layer_heatmap.{ext}      — heatmap (layer × type) at init / mid / final
  4. sv_distribution.{ext}       — violin plot of cond distribution at key steps
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

# Journal/conference style — no LaTeX required, uses matplotlib's mathtext
_STYLE = {
    # Fonts
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "legend.handlelength": 1.5,
    # Layout
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    # Axes
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    # Lines & ticks
    "lines.linewidth": 1.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}

# Paper widths (inches): IEEE / NeurIPS / ICML single- and double-column
COL1 = 3.50   # single column
COL2 = 7.16   # double column

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────

# Paul Tol's vibrant palette — distinguishable for colourblind readers
_EXPERIMENT_COLORS = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#009988",  # teal
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey (fallback)
]

# Per weight-type colours, grouped by role
_TYPE_COLORS = {
    "c_q":         "#4477AA",  # blue   — query
    "c_k":         "#66CCEE",  # sky    — key
    "c_v":         "#228833",  # green  — value
    "c_proj_attn": "#CCBB44",  # olive  — attn output
    "c_fc":        "#EE6677",  # rose   — MLP expand
    "c_proj_mlp":  "#AA3377",  # purple — MLP contract
    "ve_gate":     "#AAAAAA",  # grey   — VE gate (minor)
}

# Readable LaTeX-ish labels for each type
_TYPE_LABELS = {
    "c_q":         r"$W_Q$",
    "c_k":         r"$W_K$",
    "c_v":         r"$W_V$",
    "c_proj_attn": r"$W_O$ (attn)",
    "c_fc":        r"$W_{FC}$",
    "c_proj_mlp":  r"$W_{proj}$ (MLP)",
    "ve_gate":     r"$W_{gate}$",
}

_METRIC_LABELS = {
    "cond":     r"Condition number $\kappa$",
    "eff_rank": r"Effective rank",
    "max_sv2":  r"Max $\sigma^2$",
    "min_sv2":  r"Min $\sigma^2$",
    "mean_sv2": r"Mean $\sigma^2$",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(filepath):
    """Load a JSONL file into a list of step-records."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def global_df(records):
    """Return a DataFrame of global stats + step + flops columns."""
    rows = []
    for r in records:
        row = {"step": r["step"], "flops": r.get("flops", 0.0)}
        row.update(r["global"])
        rows.append(row)
    return pd.DataFrame(rows)


def bytype_df(records):
    """Return a long-format DataFrame of per-type stats over time."""
    rows = []
    for r in records:
        for wtype, stats in r["by_type"].items():
            row = {"step": r["step"], "flops": r.get("flops", 0.0), "type": wtype}
            row.update(stats)
            rows.append(row)
    return pd.DataFrame(rows)


def permatrix_df(records, at_steps=None):
    """Return a DataFrame of per-matrix stats at the given steps (None = all)."""
    rows = []
    for r in records:
        if at_steps is not None and r["step"] not in at_steps:
            continue
        for m in r["per_matrix"]:
            row = {"step": r["step"]}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def pick_checkpoints(records, n=3):
    """Return (n) representative steps: always first and last, plus midpoint(s)."""
    steps = [r["step"] for r in records]
    if len(steps) <= n:
        return steps
    indices = [0] + [round(i * (len(steps) - 1) / (n - 1)) for i in range(1, n - 1)] + [len(steps) - 1]
    return [steps[i] for i in sorted(set(indices))]


def x_values(df, x_axis):
    if x_axis == "flops":
        vals = df["flops"].values
        return vals / 1e18, "Training FLOPs ($\\times 10^{18}$)"
    else:
        return df["step"].values, "Training step"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Global metrics over training
# ─────────────────────────────────────────────────────────────────────────────

def fig_global_evolution(all_records, labels, x_axis, outpath):
    """
    5-panel figure showing global (mean over all matrices) metrics over training.
    One line per experiment, suitable as main results figure.
    """
    metrics = ["cond", "eff_rank", "max_sv2", "min_sv2", "mean_sv2"]
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(COL2, COL2 * 0.55))
    axes = axes.flatten()

    for ax, metric in zip(axes[:len(metrics)], metrics):
        for i, (records, label, color) in enumerate(zip(all_records, labels, _EXPERIMENT_COLORS)):
            df = global_df(records)
            xs, xlabel = x_values(df, x_axis)
            ax.plot(xs, df[metric].values, color=color, label=label,
                    linewidth=1.6, zorder=3)
        ax.set_ylabel(_METRIC_LABELS[metric])
        ax.set_xlabel(xlabel)
        if metric == "cond":
            ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--",
                       label="Perfect isometry ($\\kappa=1$)", zorder=2)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))

    # Hide unused panel (5 metrics, 6 slots)
    axes[-1].set_visible(False)

    # Single shared legend below the figure
    handles, lbls = axes[0].get_legend_handles_labels()
    # Also grab the κ=1 line if present
    all_handles, all_labels = [], []
    seen = set()
    for ax in axes[:len(metrics)]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in seen:
                all_handles.append(hi)
                all_labels.append(li)
                seen.add(li)
    fig.legend(all_handles, all_labels, loc="lower right",
               bbox_to_anchor=(0.98, 0.04), ncol=1, framealpha=0.95)

    fig.suptitle("Singular value statistics over training", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-type condition number and effective rank
# ─────────────────────────────────────────────────────────────────────────────

def fig_by_type(all_records, labels, x_axis, outpath):
    """
    2-row figure: condition number (top) and effective rank (bottom),
    with one line per weight type. Each experiment gets its own column.
    If only one experiment, uses a single column at COL2 width.
    """
    metrics = ["cond", "eff_rank"]
    n_exp = len(all_records)
    ncols = n_exp
    nrows = len(metrics)

    fig_w = min(COL2, COL1 * n_exp) if n_exp > 1 else COL2
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_w * 0.65),
                             sharey="row", sharex="col", squeeze=False)

    for col, (records, exp_label) in enumerate(zip(all_records, labels)):
        df = bytype_df(records)
        xs_all = {}
        for wtype in df["type"].unique():
            sub = df[df["type"] == wtype].sort_values("step")
            xs, xlabel = x_values(sub, x_axis)
            xs_all[wtype] = (xs, sub)

        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            for wtype in sorted(_TYPE_COLORS.keys()):
                if wtype not in xs_all:
                    continue
                xs, sub = xs_all[wtype]
                color = _TYPE_COLORS[wtype]
                lbl = _TYPE_LABELS.get(wtype, wtype)
                ax.plot(xs, sub[metric].values, color=color, label=lbl,
                        linewidth=1.4, zorder=3)

            if metric == "cond":
                ax.axhline(1.0, color="#888888", linewidth=0.7, linestyle="--", zorder=2)

            if col == 0:
                ax.set_ylabel(_METRIC_LABELS[metric])
            if row == nrows - 1:
                ax.set_xlabel(xlabel)
            if row == 0:
                ax.set_title(exp_label, fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))

    # Shared legend for weight types
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=4, framealpha=0.95,
               title="Weight type")

    fig.suptitle("Per weight-type singular value statistics", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Layer × type heatmaps at init / mid / final
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap(ax, df_step, metric, types, cmap, norm, title):
    """Draw a single heatmap panel on ax for one checkpoint."""
    n_layers = df_step["layer"].max() + 1
    n_types = len(types)
    grid = np.full((n_layers, n_types), np.nan)
    for j, wtype in enumerate(types):
        sub = df_step[df_step["type"] == wtype]
        for _, row in sub.iterrows():
            grid[int(row["layer"]), j] = row[metric]

    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest", origin="upper")
    ax.set_xticks(range(n_types))
    ax.set_xticklabels([_TYPE_LABELS.get(t, t) for t in types],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 6)))
    ax.set_ylabel("Layer index")
    ax.set_title(title, fontsize=9)
    return im


def fig_layer_heatmaps(all_records, labels, outpath, metric="cond", n_checkpoints=3):
    """
    Heatmap of `metric` over (layer × weight_type) at n_checkpoints in time.
    One row per experiment, one column per checkpoint.
    """
    types_ordered = ["c_q", "c_k", "c_v", "c_proj_attn", "c_fc", "c_proj_mlp", "ve_gate"]
    n_exp = len(all_records)

    # Gather all values to set a shared colour scale
    all_vals = []
    checkpoints_per_exp = []
    for records in all_records:
        ckpts = pick_checkpoints(records, n_checkpoints)
        checkpoints_per_exp.append(ckpts)
        pm = permatrix_df(records, at_steps=set(ckpts))
        all_vals.extend(pm[metric].dropna().tolist())

    vmin, vmax = np.percentile(all_vals, 2), np.percentile(all_vals, 98)
    if metric == "cond":
        cmap = "RdYlGn_r"   # red=high κ (bad), green=low κ (good near 1)
        norm = Normalize(vmin=max(1.0, vmin), vmax=vmax)
    else:
        cmap = "RdYlGn"     # green=high eff_rank (good)
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(n_exp, n_checkpoints,
                             figsize=(COL2, max(2.0, 2.2 * n_exp)),
                             squeeze=False)

    for row_i, (records, label, ckpts) in enumerate(
            zip(all_records, labels, checkpoints_per_exp)):
        pm = permatrix_df(records, at_steps=set(ckpts))
        for col_i, step in enumerate(ckpts):
            ax = axes[row_i][col_i]
            df_step = pm[pm["step"] == step]
            ck_label = f"Step {step:,}"
            if col_i == 0:
                ck_label = f"{label}\n{ck_label}"
            im = _heatmap(ax, df_step, metric, types_ordered, cmap, norm, ck_label)
            if col_i > 0:
                ax.set_ylabel("")

    # Shared colorbar
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.04, shrink=0.8)
    cbar.set_label(_METRIC_LABELS[metric], fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    title_metric = _METRIC_LABELS[metric].replace("$", "").replace("\\kappa", "κ")
    fig.suptitle(f"Layer × weight-type heatmap: {title_metric}", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Violin / box plots of condition number distribution
# ─────────────────────────────────────────────────────────────────────────────

def fig_distribution(all_records, labels, outpath, metric="cond", n_checkpoints=5):
    """
    Violin plots showing the distribution of `metric` across all weight matrices
    at n_checkpoints during training. One subplot per experiment.
    """
    n_exp = len(all_records)
    fig, axes = plt.subplots(1, n_exp, figsize=(COL2, 2.8), squeeze=False)

    for col_i, (records, label, color) in enumerate(
            zip(all_records, labels, _EXPERIMENT_COLORS)):
        ckpts = pick_checkpoints(records, n_checkpoints)
        pm = permatrix_df(records, at_steps=set(ckpts))
        ax = axes[0][col_i]

        groups = [pm[pm["step"] == s][metric].dropna().tolist() for s in ckpts]
        positions = list(range(len(ckpts)))

        parts = ax.violinplot(groups, positions=positions,
                              showmedians=True, showextrema=True, widths=0.7)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
        for key in ("cmedians", "cmaxes", "cmins", "cbars"):
            parts[key].set_color(color)
            parts[key].set_linewidth(1.0)

        if metric == "cond":
            ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--",
                       label="Perfect isometry ($\\kappa=1$)")

        step_labels = [f"{s:,}" for s in ckpts]
        ax.set_xticks(positions)
        ax.set_xticklabels(step_labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_xlabel("Training step")
        if col_i == 0:
            ax.set_ylabel(_METRIC_LABELS[metric])
        ax.set_title(label, fontsize=9)

    fig.suptitle(f"Distribution of {_METRIC_LABELS.get(metric, metric)} across weight matrices",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(outpath)
    print(f"  Saved: {outpath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality SV statistics plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="+", metavar="FILE",
                        help="JSONL file(s) produced by nanochat/sv_stats.py")
    parser.add_argument("--output-dir", default="sv_figures",
                        help="directory to save figures (default: sv_figures/)")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                        help="output format (default: pdf, best for journals)")
    parser.add_argument("--x-axis", default="step", choices=["step", "flops"],
                        help="x-axis for time-series plots (default: step)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="experiment labels (default: filename stems)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PNG output (default: 300)")
    args = parser.parse_args()

    # Apply publication style globally
    mpl.rcParams.update(_STYLE)
    mpl.rcParams["savefig.dpi"] = args.dpi

    # Resolve labels
    labels = args.labels or [Path(f).stem.replace("_sv_stats", "").replace("_", " ")
                              for f in args.files]
    if len(labels) != len(args.files):
        parser.error(f"Got {len(args.files)} files but {len(labels)} --labels")

    # Load data
    print(f"Loading {len(args.files)} file(s)...")
    all_records = []
    for f, label in zip(args.files, labels):
        recs = load_jsonl(f)
        print(f"  {label}: {len(recs)} checkpoints, "
              f"steps {recs[0]['step']}–{recs[-1]['step']}")
        all_records.append(recs)

    # Output directory
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = args.format

    print(f"\nGenerating figures in {out}/")

    def p(name):
        return out / f"{name}.{ext}"

    # Figure 1: global evolution
    fig_global_evolution(all_records, labels, args.x_axis, p("sv_global_evolution"))

    # Figure 2: per-type
    fig_by_type(all_records, labels, args.x_axis, p("sv_by_type"))

    # Figure 3: layer heatmaps — condition number
    fig_layer_heatmaps(all_records, labels, p("sv_layer_heatmap_cond"),
                       metric="cond", n_checkpoints=3)

    # Figure 4: layer heatmaps — effective rank
    fig_layer_heatmaps(all_records, labels, p("sv_layer_heatmap_effrank"),
                       metric="eff_rank", n_checkpoints=3)

    # Figure 5: violin distributions
    fig_distribution(all_records, labels, p("sv_distribution_cond"),
                     metric="cond", n_checkpoints=5)
    fig_distribution(all_records, labels, p("sv_distribution_effrank"),
                     metric="eff_rank", n_checkpoints=5)

    print(f"\nDone. {len(list(out.glob(f'*.{ext}')))} figures saved to {out}/")


if __name__ == "__main__":
    main()
