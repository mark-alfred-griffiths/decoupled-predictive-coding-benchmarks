#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare λ-cap=0 (vanilla PC) vs λ-cap=1 (IL-ASGN) side-by-side.

- Loads two CSVs produced by plot_pc_equilibrium.py
- Aggregates mean PI steps across seeds
- Produces a faceted figure (rows=rollout_tol, cols=dt) with both λ settings
- Optionally writes a CSV with deltas (lam1 - lam0) averaged over training

Usage:
  python compare_lambda_caps.py \
    --lam0 output/pi_plots_lambda0/pi_equilibration_timeseries.csv \
    --lam1 output/pi_plots_lambda1/pi_equilibration_timeseries.csv \
    --outdir compare_out
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lam0", type=str, required=True,
                   help="CSV from λ-cap=0 run (vanilla PC).")
    p.add_argument("--lam1", type=str, required=True,
                   help="CSV from λ-cap=1 run (IL-ASGN).")
    p.add_argument("--outdir", type=str, default="compare_out",
                   help="Directory to save figures/CSVs.")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--title", type=str, default="PI steps vs training — side-by-side λ comparison")
    p.add_argument("--alpha", type=float, default=0.9, help="Line alpha.")
    p.add_argument("--linewidth", type=float, default=1.8)
    p.add_argument("--markersize", type=float, default=3.0)
    return p.parse_args()


def fmt_tol(x: float) -> str:
    return f"{x:.0e}" if x < 1e-2 else f"{x:g}"


def load_and_tag(csv_path: str, tag: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expected columns (from plot_pc_equilibrium.py):
    # train_step, pi_steps, test_acc, test_loss, dt, rollout_tol, rollout_steps, seed, [oracle_every?]
    # Be forgiving: fill missing expected columns if necessary.
    for col in ["dt", "rollout_tol", "rollout_steps", "train_step", "pi_steps"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")
    df["lambda_cap"] = tag
    return df


def aggregate_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean across seeds per (dt, tol, rollout_steps, train_step, lambda_cap).
    """
    group_cols = ["lambda_cap", "dt", "rollout_tol", "rollout_steps", "train_step"]
    agg = (df.groupby(group_cols, as_index=False)
             .agg(pi_steps_mean=("pi_steps", "mean"),
                  pi_steps_med=("pi_steps", "median"),
                  n_runs=("pi_steps", "count")))
    return agg


def ensure_sorted_unique(vals: List) -> List:
    return sorted(list(dict.fromkeys(vals)))


def plot_facets_side_by_side(agg_df: pd.DataFrame, out_png: Path,
                             title: str, dpi: int,
                             alpha: float, lw: float, ms: float):
    """
    Faceted grid: rows = rollout_tol, cols = dt.
    Inside each facet, overlay time-series for each rollout_steps,
    colored by lambda_cap (lam0 vs lam1).
    """
    dts = ensure_sorted_unique(agg_df["dt"].unique().tolist())
    tols = ensure_sorted_unique(agg_df["rollout_tol"].unique().tolist())
    rsteps = ensure_sorted_unique(agg_df["rollout_steps"].unique().tolist())
    lambdas = ensure_sorted_unique(agg_df["lambda_cap"].unique().tolist())  # ["lam0", "lam1"]

    # Prepare figure grid
    nrows, ncols = len(tols), len(dts)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.4*ncols, 3.2*nrows),
                             sharex=True, sharey=True)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    # Assign colors for λ conditions
    color_map = {"lam0": "#1f77b4", "lam1": "#d62728"}  # blue vs red
    linestyle_map = {rs: "-" if i == 0 else "--" if i == 1 else ":" for i, rs in enumerate(rsteps)}

    for i, tol in enumerate(tols):
        for j, dt in enumerate(dts):
            ax = axes[i, j]
            sub = agg_df[(agg_df["rollout_tol"] == tol) & (agg_df["dt"] == dt)]
            if sub.empty:
                ax.set_visible(False)
                continue

            # plot per rollout_steps and lambda_cap
            for rs in rsteps:
                for lam in lambdas:
                    s = sub[(sub["rollout_steps"] == rs) & (sub["lambda_cap"] == lam)]
                    if s.empty:
                        continue
                    s = s.sort_values("train_step")
                    ax.plot(
                        s["train_step"].values,
                        s["pi_steps_mean"].values,
                        label=f"{lam} • rollout={rs}",
                        color=color_map.get(lam, None),
                        linestyle=linestyle_map.get(rs, "-"),
                        alpha=alpha,
                        linewidth=lw,
                        marker="o",
                        markersize=ms
                    )

            if i == nrows - 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel("mean PI steps")
            ax.set_title(f"dt={dt:g}, tol={fmt_tol(tol)}")
            ax.grid(True, alpha=0.3)

    # Build a single legend above subplots, below title
    # Collect unique labels in order of appearance
    handles_labels = {}
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in handles_labels:
                handles_labels[label] = handle

    if handles_labels:
        fig.legend(
            list(handles_labels.values()),
            list(handles_labels.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=min(len(handles_labels), 4),
            frameon=False
        )

    fig.suptitle(title, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {out_png}")


def write_delta_summary(agg_df: pd.DataFrame, out_csv: Path):
    """
    Compute (lam1 - lam0) average PI steps over training,
    grouped by (dt, tol, rollout_steps). Writes CSV.
    """
    # Pivot so lam0/lam1 become columns
    pivot = (agg_df.pivot_table(index=["dt", "rollout_tol", "rollout_steps", "train_step"],
                                columns="lambda_cap",
                                values="pi_steps_mean")
                   .reset_index())
    if not {"lam0", "lam1"}.issubset(set(pivot.columns)):
        print("[warn] Missing lam0 or lam1 columns in pivot – skipping delta CSV.")
        return

    pivot["delta_lam1_minus_lam0"] = pivot["lam1"] - pivot["lam0"]

    # Average across training steps
    delta = (pivot.groupby(["dt", "rollout_tol", "rollout_steps"], as_index=False)
                   .agg(avg_delta=("delta_lam1_minus_lam0", "mean"),
                        med_delta=("delta_lam1_minus_lam0", "median"),
                        n_points=("delta_lam1_minus_lam0", "count")))
    delta.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df0 = load_and_tag(args.lam0, tag="lam0")
    df1 = load_and_tag(args.lam1, tag="lam1")
    df = pd.concat([df0, df1], ignore_index=True)

    # Aggregate across seeds
    agg = aggregate_mean(df)

    # Plot side-by-side (overlaid) facets
    out_png = outdir / "side_by_side_pi_steps.png"
    plot_facets_side_by_side(
        agg, out_png,
        title=args.title,
        dpi=args.dpi,
        alpha=args.alpha,
        lw=args.linewidth,
        ms=args.markersize
    )

    # CSV with deltas averaged over training (lam1 - lam0)
    out_delta_csv = outdir / "avg_delta_pi_steps_lam1_minus_lam0.csv"
    write_delta_summary(agg, out_delta_csv)


if __name__ == "__main__":
    main()
