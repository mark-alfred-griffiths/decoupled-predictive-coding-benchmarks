#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility to sweep predictive-coding settings and visualise equilibration.

The script is intentionally self-contained so it can be dropped into new
experiments. It will:

* Launch ``train_path_integral_pc.py`` (or a user-specified runner) over a grid of
  ``dt``, ``rollout_tol``, ``rollout_steps``, seeds, and ``lambda_cap``
  values. Logs are cached per grid point so re-running the script will parse
  existing results rather than re-executing runs.
* Parse predictive-coding iteration counts from the logs, **using
  ``PC_steps`` as the primary measure of PC iterations** (falling back to the
  total proposal count or PI metrics if necessary).
* Save a consolidated CSV of the parsed timeseries.
* Generate two figures inspired by the previous plotting utility:
    - 2D "surface" line plots comparing ILASGN (``lambda_cap=1``) against PC
      (``lambda_cap=0``) over training steps, saved per hyperparameter setting.
    - Delta curves showing ``ILASGN - PC`` predictive-coding iterations for
      each hyperparameter setting.

The plotting code is written to be robust to missing points and will emit a
warning instead of failing if a surface cannot be constructed from the
available data.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D plots
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runner",
        type=str,
        default="train_path_integral_pc.py",
        help="Training script to execute for each grid point.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="gpu",
        help="Device passed through to the training script.",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--train-steps", type=int, default=700)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--oracle-every", type=int, nargs="+", default=[1, 3, 5])

    parser.add_argument("--grid-dt", type=float, nargs="+", default=[0.0036])
    parser.add_argument("--grid-tol", type=float, nargs="+", default=[5e-4])
    parser.add_argument("--grid-steps", type=int, nargs="+", default=[8])
    parser.add_argument(
        "--lambda-cap",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
        help="Values of --lambda-cap to sweep (e.g., 1=ILASGN, 0=PC).",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra CLI arguments forwarded to the runner.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output/pi_plots_out",
        help="Directory to place logs, CSV, and figures.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="pi_equilibration_timeseries.csv",
        help="Filename for the aggregated CSV (inside --outdir).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Regex helpers
# -----------------------------------------------------------------------------
STEP_RE = re.compile(r"\[Step\s+(\d+)\]")
PI_RE = re.compile(r"PI_steps=([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)", re.IGNORECASE)
PI_MAX_RE = re.compile(r"PI_max_steps=([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)", re.IGNORECASE)
PC_STEPS_RE = re.compile(r"PC_steps=([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)", re.IGNORECASE)
PC_TOTAL_RE = re.compile(
    r"(?:PC_total|total_proposals)=([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)",
    re.IGNORECASE,
)
ACC_RE = re.compile(r"test_acc=([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
LOSS_RE = re.compile(r"test_loss=([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
LAMBDA_RE = re.compile(r"lambda_gate=([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Command helpers
# -----------------------------------------------------------------------------

def _build_cmd(
    python_bin: str,
    runner: str,
    device: str,
    seed: int,
    train_steps: int,
    eval_every: int,
    oracle_every: int,
    dt: float,
    tol: float,
    steps: int,
    lambda_cap: float,
    extra: str,
) -> List[str]:
    cmd = [
        python_bin,
        runner,
        "--device",
        device,
        "--seed",
        str(seed),
        "--train-steps",
        str(train_steps),
        "--eval-every",
        str(eval_every),
        "--oracle-every",
        str(oracle_every),
        "--dt",
        str(dt),
        "--rollout-tol",
        str(tol),
        "--rollout-steps",
        str(steps),
        "--lambda-cap",
        str(lambda_cap),
    ]
    if extra:
        cmd.extend(shlex.split(extra))
    return cmd


<<<<<<< HEAD
# -----------------------------------------------------------------------------
# Log parsing
# -----------------------------------------------------------------------------
=======
    # parsed values for the current step
    cur_pi_steps: Optional[float] = None       # PI_steps (legacy)
    cur_pi_max: Optional[float] = None         # PI_max_steps
    cur_pc_steps: Optional[float] = None       # PC_steps (actual)
    cur_pc_total: Optional[float] = None       # PC_total / total_proposals (actual)
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e

def _parse_float(value: str) -> float:
    if value.lower() == "nan":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _coerce_count(value: Optional[float]) -> float:
    if value is None:
        return float("nan")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(as_float):
        return float("nan")
    return float(int(round(as_float)))


def parse_log_lines(lines: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    cur_step: Optional[int] = None
    pc_steps: Optional[float] = None
    pc_total: Optional[float] = None
    pi_steps: Optional[float] = None
    pi_max: Optional[float] = None
    acc: Optional[float] = None
    loss: Optional[float] = None
    gate: Optional[float] = None

    def flush_row() -> None:
        nonlocal cur_step, pc_steps, pc_total, pi_steps, pi_max, acc, loss, gate
        if cur_step is None:
            return

<<<<<<< HEAD
        # Prefer the actual executed PC_steps first as requested
        preferred = _coerce_count(pc_steps)
        fallback_total = _coerce_count(pc_total)
        fallback_max = _coerce_count(pi_max)
        fallback_pi = _coerce_count(pi_steps)

        metric = preferred
        if not math.isfinite(metric):
            metric = fallback_total
        if not math.isfinite(metric):
            metric = fallback_max
        if not math.isfinite(metric):
            metric = fallback_pi
=======
        # Prefer real executed effort: chart only the predictive-coding step count
        pc_total = _coerce_count(cur_pc_total)
        pc_steps = _coerce_count(cur_pc_steps)
        pi_max   = _coerce_count(cur_pi_max)
        pi_steps = _coerce_count(cur_pi_steps)

        # Decide the value to chart in 'pc_iterations'
        chosen = float("nan")
        if math.isfinite(pc_steps):
            chosen = pc_steps
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e

        if not math.isfinite(metric) and gate is None:
            return

<<<<<<< HEAD
        rows.append(
            {
                "train_step": cur_step,
                "pc_metric": metric,
                "pc_steps": preferred,
                "pc_total": fallback_total,
                "pi_max_steps": fallback_max,
                "pi_steps": fallback_pi,
                "test_acc": float("nan") if acc is None else acc,
                "test_loss": float("nan") if loss is None else loss,
                "lambda_gate": float("nan") if gate is None else gate,
            }
=======
        row = dict(
            train_step=current_step,
            # series used for plotting
            pc_iterations=chosen,

            # keep raw fields for debugging/alt analysis
            pc_total=pc_total,
            pc_steps=pc_steps,
            pi_max_steps=pi_max,

            test_acc=_maybe_float(cur_acc),
            test_loss=_maybe_float(cur_loss),
            lambda_gate=_maybe_float(cur_gate),
            lambda_gate_raw=_maybe_float(cur_gate_raw),
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e
        )

        cur_step = None
        pc_steps = None
        pc_total = None
        pi_steps = None
        pi_max = None
        acc = None
        loss = None
        gate = None

    for line in lines:
        match_step = STEP_RE.search(line)
        if match_step:
            new_step = int(match_step.group(1))
            # If this is the first time we've seen a step number, start capturing
            # immediately. If we see the same step number twice (logs often emit
            # two lines per step), we keep accumulating without flushing. Only
            # when the step number changes do we flush the previous row.
            if cur_step is None:
                cur_step = new_step
            elif new_step != cur_step:
                flush_row()
                cur_step = new_step

        if cur_step is None:
            continue

        if (m := PC_STEPS_RE.search(line)):
            pc_steps = _parse_float(m.group(1))
        if (m := PC_TOTAL_RE.search(line)):
            pc_total = _parse_float(m.group(1))
        if (m := PI_RE.search(line)):
            pi_steps = _parse_float(m.group(1))
        if (m := PI_MAX_RE.search(line)):
            pi_max = _parse_float(m.group(1))
        if (m := ACC_RE.search(line)):
            acc = _parse_float(m.group(1))
        if (m := LOSS_RE.search(line)):
            loss = _parse_float(m.group(1))
        if (m := LAMBDA_RE.search(line)):
            gate = _parse_float(m.group(1))

    flush_row()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

def run_grid(args: argparse.Namespace) -> pd.DataFrame:
    outdir = Path(args.outdir)
    logdir = outdir / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / args.csv_name
    base_cols = [
        "train_step",
<<<<<<< HEAD
        "pc_metric",
=======
        "pc_iterations",
        "pc_total",
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e
        "pc_steps",
        "pc_total",
        "pi_max_steps",
<<<<<<< HEAD
        "pi_steps",
=======
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e
        "test_acc",
        "test_loss",
        "lambda_gate",
        "seed",
        "dt",
        "rollout_tol",
        "rollout_steps",
        "oracle_every",
        "lambda_cap",
        "log_path",
    ]

    if not csv_path.exists():
        pd.DataFrame(columns=base_cols).to_csv(csv_path, index=False)
        print(f"[INFO] Created CSV header at {csv_path}")

    try:
        combined = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive read
        print(f"[WARN] Failed reading existing CSV {csv_path}: {exc}; starting fresh")
        combined = pd.DataFrame(columns=base_cols)

    python_bin = sys.executable
    parsed_any = False

<<<<<<< HEAD
    for seed, dt, tol, steps, oracle_every, lambda_cap in itertools.product(
        args.seed,
        args.grid_dt,
        args.grid_tol,
        args.grid_steps,
        args.oracle_every,
        args.lambda_cap,
    ):
        param_desc = (
            f"seed={seed}, dt={dt}, tol={tol}, rollout_steps={steps}, "
            f"oracle_every={oracle_every}, lambda_cap={lambda_cap}, "
            f"train_steps={args.train_steps}, eval_every={args.eval_every}, "
            f"device={args.device}, extra={args.extra or ''}"
=======
def _safe_mean(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.mean()) if not s.empty else float("nan")

def _safe_median(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.median()) if not s.empty else float("nan")

def _safe_quantile(series: pd.Series, q: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(np.quantile(s, q))

def _summarise_pi(long_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["lambda_cap", "dt", "rollout_tol", "train_step"]
    summary = (
        long_df.groupby(group_cols, as_index=False)
        .agg(
            pc_iterations_mean=("pc_iterations", _safe_mean),
            pc_iterations_median=("pc_iterations", _safe_median),
            pc_iterations_q25=("pc_iterations", lambda s: _safe_quantile(s, 0.25)),
            pc_iterations_q75=("pc_iterations", lambda s: _safe_quantile(s, 0.75)),
            runs=("pc_iterations", lambda s: int(s.notna().sum())),
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e
        )
        log_name = (
            f"seed{seed}_lam{lambda_cap}_dt{dt}_tol{tol}_steps{steps}_oracle{oracle_every}_dev{args.device}.log"
        )
        log_path = logdir / log_name

        if log_path.exists() and log_path.stat().st_size > 0:
            print(
                f"[INFO] Using cached log {log_name} (use --outdir to change destination) | {param_desc}"
            )
            with log_path.open("r", encoding="utf-8") as fh:
                df = parse_log_lines(fh)
            if df.empty:
                print(f"[WARN] Existing log {log_name} had no parseable rows")
                continue
        else:
            cmd = _build_cmd(
                python_bin,
                args.runner,
                args.device,
                seed,
                args.train_steps,
                args.eval_every,
                oracle_every,
                dt,
                tol,
                steps,
                lambda_cap,
                args.extra,
            )
            env = os.environ.copy()
            if args.device == "cpu":
                env["JAX_PLATFORMS"] = "cpu"
            print(f"[RUN] Launching {args.runner} with params: {param_desc}")
            print(f"[RUN] Command: {shlex.join(cmd)}")
            try:
                with log_path.open("w", encoding="utf-8") as fh:
                    proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                if proc.returncode != 0:
                    print(f"[WARN] Command failed for {log_name}; see log for details", file=sys.stderr)
            except FileNotFoundError as exc:
                print(f"Failed to launch runner: {exc}", file=sys.stderr)
                continue

<<<<<<< HEAD
            with log_path.open("r", encoding="utf-8") as fh:
                df = parse_log_lines(fh)
            if df.empty:
                print(f"[WARN] No parseable rows in {log_path}")
                continue

        df = df.assign(
            seed=seed,
            dt=dt,
            rollout_tol=tol,
            rollout_steps=steps,
            oracle_every=oracle_every,
            lambda_cap=lambda_cap,
            log_path=str(log_path),
=======
            for lam in lambda_vals:
                lam_sub = sub[sub["lambda_cap"] == lam].sort_values("train_step")
                if lam_sub.empty:
                    continue
                color = color_map[lam]
                label = f"λ_cap={lam:g}"
                ax.plot(
                    lam_sub["train_step"].values,
                    lam_sub["pc_iterations_mean"].values,
                    color=color,
                    linewidth=2.2,
                    label=label,
                )
                y1 = lam_sub["pc_iterations_q25"].values
                y2 = lam_sub["pc_iterations_q75"].values
                if np.all(np.isnan(y1)) or np.all(np.isnan(y2)):
                    pass
                else:
                    ax.fill_between(
                        lam_sub["train_step"].values,
                        y1,
                        y2,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                    )

            if i == nrows - 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel("mean PC iterations")

    handles = []
    labels = []
    for lam in lambda_vals:
        handles.append(plt.Line2D([0], [0], color=color_map[lam], linewidth=2.2))
        labels.append(f"λ_cap={lam:g}")

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)

    fig.suptitle(
        "Predictive-coding equilibration vs training step\n"
        "Facets: tolerance × dt | Lines: λ_cap with IQR shading"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = outdir / "faceted_pc_iterations_vs_training.png"
    fig.savefig(out, dpi=220)
    print(f"[saved] {out}")

def plot_surface(long_df: pd.DataFrame, outdir: Path):
    """
    3D surface of mean PC iterations vs training step × dt.

    This version avoids an explicit Python nested loop by pivoting the
    aggregated dataframe directly into a 2D array suitable for plotting.
    """
    if long_df.empty:
        print("[WARN] Nothing to plot for 3D surface.")
        return

    df = long_df
    lam_label = "all λ_cap"
    if "lambda_cap" in df and not df["lambda_cap"].isna().all():
        lam_vals = sorted(df["lambda_cap"].dropna().unique())
        if lam_vals:
            lam_sel = lam_vals[-1]
            df = df[df["lambda_cap"] == lam_sel]
            lam_label = f"λ_cap={lam_sel:g}"

    # Pick the modal tol / rollout_steps / oracle_every to define the slice
    tol_sel = float(df["rollout_tol"].mode().iat[0])
    rs_sel  = int(df["rollout_steps"].mode().iat[0])
    oracle_sel = int(df["oracle_every"].mode().iat[0])

    group_cols = ["dt", "rollout_tol", "rollout_steps", "oracle_every", "train_step"]
    avg = (
        df.groupby(group_cols, as_index=False)
          .agg(pc_iterations_mean=("pc_iterations", "mean"))
    )

    surface_df = avg[
        (avg["rollout_tol"] == tol_sel)
        & (avg["rollout_steps"] == rs_sel)
        & (avg["oracle_every"] == oracle_sel)
    ]
    if surface_df.empty:
        print("[WARN] No rows for selected tol/rollout_steps.")
        return

    # Pivot to a dense grid: rows=train_step, cols=dt
    surface_df = surface_df.sort_values(["train_step", "dt"])
    pivot = surface_df.pivot(index="train_step", columns="dt", values="pc_iterations_mean")

    ts_vals = pivot.index.to_numpy()
    dt_vals = pivot.columns.to_numpy()
    Z = pivot.to_numpy()

    # Meshgrid expects X (dt) and Y (train_step)
    T, TT = np.meshgrid(dt_vals, ts_vals)

    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot_surface(T, TT, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax3d.set_xlabel("dt")
    ax3d.set_ylabel("training step")
    ax3d.set_zlabel("mean PC iterations")
    ax3d.set_title(
        "Surface: mean PC iterations vs training step × dt\n"
        f"(tol={fmt_tol(tol_sel)}, rollout_steps={rs_sel}, oracle_every={oracle_sel}, {lam_label})"
    )
    out = outdir / "surface_pc_iterations_vs_training_dt.png"
    fig.savefig(out, dpi=200)
    print(f"[saved] {out}")

def plot_lambda_difference(long_df: pd.DataFrame, outdir: Path):
    """
    Compare smallest vs largest λ_cap and plot the Δ in PC iterations
    (positive = improvement for larger λ if we compute small − large).
    """
    if long_df.empty or "lambda_cap" not in long_df:
        print("[WARN] Skipping λ-cap difference plot; missing data.")
        return

    summary = _summarise_pi(long_df)
    if summary.empty:
        print("[WARN] No summary rows for λ-cap difference plot.")
        return

    # Normalise λ values and pick extremes
    def _norm(v):
        try:
            return float(v)
        except Exception:
            return v

    lam_vals = sorted(summary["lambda_cap"].dropna().unique(), key=_norm)
    if len(lam_vals) < 2:
        print("[WARN] Need at least two λ_cap values for difference plot.")
        return

    lam_small = lam_vals[0]
    lam_large = lam_vals[-1]

    key_cols = ["dt", "rollout_tol", "train_step"]

    def _slice_for(lam_value: float, suffix: str) -> pd.DataFrame:
        sub = summary[summary["lambda_cap"] == lam_value]
        keep = key_cols + ["pc_iterations_mean", "pc_iterations_q25", "pc_iterations_q75"]
        sub = sub[keep].rename(
            columns={
                "pc_iterations_mean": f"mean{suffix}",
                "pc_iterations_q25": f"q25{suffix}",
                "pc_iterations_q75": f"q75{suffix}",
            }
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e
        )

        parsed_any = True

        dedupe_cols = [
            "train_step",
            "seed",
            "dt",
            "rollout_tol",
            "rollout_steps",
            "oracle_every",
            "lambda_cap",
            "log_path",
            "pc_metric",
            "pc_steps",
            "pc_total",
            "pi_max_steps",
            "pi_steps",
            "test_acc",
            "test_loss",
            "lambda_gate",
        ]
        before_len = len(combined)
        combined = pd.concat([combined, df], ignore_index=True, sort=False)
        existing_cols = [c for c in dedupe_cols if c in combined.columns]
        combined = combined.drop_duplicates(subset=existing_cols, keep="last")
        added = len(combined) - before_len
        combined.to_csv(csv_path, index=False)
        print(
            f"[INFO] Updated {csv_path} after {log_name}; total rows: {len(combined)}; "
            f"added {max(added, 0)} new rows"
        )

    if not parsed_any:
        print(f"[WARN] No data parsed; leaving CSV untouched at {csv_path}")
        return combined

    print(f"Wrote {csv_path} with {len(combined)} rows")
    return combined


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_surface(df: pd.DataFrame, outdir: Path) -> List[Path]:
    def _format_lambda_cap(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else str(value)

    def _format_lambda_caps(values: Iterable[float]) -> str:
        unique_vals = sorted({float(v) for v in values})
        return "-".join(_format_lambda_cap(v) for v in unique_vals) if unique_vals else "unknown"

    surfaces_dir = outdir / "surfaces"
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    output_paths: List[Path] = []
    grouped = df.groupby(["dt", "rollout_tol", "rollout_steps", "oracle_every"])
    for (dt, tol, steps, oracle_every), subset in grouped:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_line = False

        lambda_caps_string = _format_lambda_caps(subset["lambda_cap"].unique())

        styles = {
            1.0: {"label": "ILASGN (λ=1)", "linestyle": "-", "color": "red"},
            0.0: {"label": "PC (λ=0)", "linestyle": ":", "color": "blue"},
        }

        for lam, style in styles.items():
            lam_df = subset[subset["lambda_cap"] == lam]
            if lam_df.empty:
                continue

            line_df = lam_df.groupby("train_step")["pc_metric"].mean().reset_index()
            line_df = line_df.sort_values("train_step")
            ax.plot(
                line_df["train_step"],
                line_df["pc_metric"],
                label=style["label"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.5,
            )
            has_line = True

        if not has_line:
            plt.close(fig)
            print(
                f"[WARN] No surface lines for dt={dt}, tol={tol}, steps={steps}, oracle_every={oracle_every}"
            )
<<<<<<< HEAD
            continue

        ax.set_xlabel("Train step")
        ax.set_ylabel("Mean PC_steps")
        ax.set_title(
            "PC iterations vs. training (ILASGN vs PC)\n"
            f"dt={dt}, tol={tol}, rollout_steps={steps}, oracle_every={oracle_every}, "
            f"lambda_caps={lambda_caps_string}"
=======
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.set_title(f"dt={dt:g}, tol={fmt_tol(tol)}")
            ax.grid(True, alpha=0.3)
            if i == nrows - 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel(f"Δ PC iterations (λ={lam_small:g} − λ={lam_large:g})")

    fig.suptitle(
        "Mean PC-iteration improvement from IL-ASGN over vanilla predictive coding\n"
        "Positive values → IL-ASGN requires fewer PC iterations"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = outdir / "lambda_cap_difference.png"
    fig.savefig(out, dpi=220)
    print(f"[saved] {out}")

def plot_lambda_gate_summary(long_df: pd.DataFrame, outdir: Path):
    if long_df.empty or "lambda_cap" not in long_df or "lambda_gate" not in long_df:
        print("[WARN] Skipping λ-gate summary; missing data.")
        return

    gated = long_df[(long_df["lambda_cap"] > 0) & (long_df["lambda_gate"].notna())]
    if gated.empty:
        print("[WARN] No λ_gate observations for IL-ASGN runs.")
        return

    summary = (
        gated.groupby("train_step", as_index=False)
        .agg(
            gate_mean=("lambda_gate", _safe_mean),
            gate_q25=("lambda_gate", lambda s: _safe_quantile(s, 0.25)),
            gate_q75=("lambda_gate", lambda s: _safe_quantile(s, 0.75)),
            gate_raw_mean=("lambda_gate_raw", _safe_mean),
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e
        )
        ax.legend()
        fig.tight_layout()

        out_name = (
            f"surface_dt{dt}_tol{tol}_steps{steps}_oracle{oracle_every}_lam{lambda_caps_string}.png"
        )
        out_path = surfaces_dir / out_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved surface plot to {out_path}")
        output_paths.append(out_path)

    if not output_paths:
        print("[WARN] No surface plots were generated")
    return output_paths


def plot_delta_curves(df: pd.DataFrame, outdir: Path) -> List[Path]:
    def _format_lambda_cap(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else str(value)

    def _format_lambda_caps(values: Iterable[float]) -> str:
        unique_vals = sorted({float(v) for v in values})
        return "-".join(_format_lambda_cap(v) for v in unique_vals) if unique_vals else "unknown"

    delta_dir = outdir / "delta_surfaces"
    delta_dir.mkdir(parents=True, exist_ok=True)

    output_paths: List[Path] = []
    grouped = df.groupby(["dt", "rollout_tol", "rollout_steps", "oracle_every"])
    for (dt, tol, steps, oracle_every), subset in grouped:
        lambda_caps_string = _format_lambda_caps(subset["lambda_cap"].unique())

        ilasgn = subset[subset["lambda_cap"] == 1.0]
        pc = subset[subset["lambda_cap"] == 0.0]
        if ilasgn.empty or pc.empty:
            print(
                f"[WARN] Missing ILASGN/PC pairing for dt={dt}, tol={tol}, steps={steps}, oracle_every={oracle_every}"
            )
            continue

        ilasgn_grouped = ilasgn.groupby("train_step")["pc_metric"].mean().reset_index()
        pc_grouped = pc.groupby("train_step")["pc_metric"].mean().reset_index()
        merged = pd.merge(
            ilasgn_grouped.rename(columns={"pc_metric": "ilasgn"}),
            pc_grouped.rename(columns={"pc_metric": "pc"}),
            on="train_step",
            how="inner",
        )
        if merged.empty:
            print(
                f"[WARN] No overlapping steps for ILASGN vs PC at dt={dt}, tol={tol}, steps={steps}, oracle_every={oracle_every}"
            )
            continue

        merged = merged.sort_values("train_step")
        merged["delta"] = merged["ilasgn"] - merged["pc"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            merged["train_step"],
            merged["delta"],
            label="ILASGN (λ=1) - PC (λ=0)",
            color="green",
            linewidth=2.5,
        )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Train step")
        ax.set_ylabel("Δ PC_steps (ILASGN - PC)")
        ax.set_title(
            "Delta PC iterations: ILASGN - PC\n"
            f"dt={dt}, tol={tol}, rollout_steps={steps}, oracle_every={oracle_every}, "
            f"lambda_caps={lambda_caps_string}"
        )
        ax.legend()
        fig.tight_layout()

        out_name = (
            f"delta_dt{dt}_tol{tol}_steps{steps}_oracle{oracle_every}_lam{lambda_caps_string}.png"
        )
        out_path = delta_dir / out_name
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved delta curve to {out_path}")
        output_paths.append(out_path)

    if not output_paths:
        print("[WARN] No delta curves were generated")
    return output_paths


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    df = run_grid(args)
    if df.empty:
        print("No data collected; exiting.")
        return

    outdir = Path(args.outdir)
    plot_surface(df, outdir)
    plot_delta_curves(df, outdir)

<<<<<<< HEAD
=======
    combos = list(
        itertools.product(
            args.grid_dt,
            args.grid_tol,
            args.grid_steps,
            args.seed,
            args.oracle_every,
            args.lambda_cap,
        )
    )
    all_long = []

    for dt, tol, steps, seed, oracle_every, lambda_cap in combos:
        print(
            (
                "\n=== RUN seed={seed} dt={dt} tol={tol} steps={steps} oracle_every={oracle_every} "
                "lambda_cap={lambda_cap} dev={dev} ==="
            ).format(
                seed=seed, dt=dt, tol=tol, steps=steps,
                oracle_every=oracle_every, lambda_cap=lambda_cap, dev=args.device
            )
        )
        try:
            df = run_one(
                runner=args.runner,
                preferred_device=args.device,
                seed=seed,
                train_steps=args.train_steps,
                eval_every=args.eval_every,
                oracle_every=oracle_every,
                dt=dt, tol=tol, steps=steps,
                lambda_cap=lambda_cap,
                extra=args.extra,
                logs_dir=logs_dir,
                rerun_empty=args.rerun_empty,
            )
        except Exception as e:
            print(f"[ERROR] seed={seed} dt={dt} tol={tol} steps={steps}: {e}")
            df = pd.DataFrame(columns=["train_step","pc_iterations","test_acc","test_loss"])

        if not df.empty:
            df["dt"] = dt
            df["rollout_tol"] = tol
            df["rollout_steps"] = steps
            df["seed"] = seed
            df["oracle_every"] = oracle_every
            df["lambda_cap"] = lambda_cap
        all_long.append(df)

    long_df = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    # Ensure downstream accessors (e.g., .notna()) always have the expected column even
    # if no rows were parsed (e.g., all logs were empty/absent).
    if "pc_iterations" not in long_df.columns:
        long_df["pc_iterations"] = pd.Series(dtype=float)
    csv_path = outdir / args.csv_name
    long_df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path} (rows={len(long_df)})")
    # quick sanity
    print("rows:", len(long_df), "| with_pc_iterations:", long_df["pc_iterations"].notna().sum())

    plot_faceted_time_series(long_df, outdir)
    plot_surface(long_df, outdir)
    plot_lambda_difference(long_df, outdir)
    plot_lambda_gate_summary(long_df, outdir)

    if not long_df.empty:
        pivot = (
            long_df.groupby(
                [
                    "lambda_cap",
                    "dt",
                    "rollout_tol",
                    "rollout_steps",
                    "oracle_every",
                    "train_step",
                ],
                as_index=False,
            )
            .agg(
                pc_iterations_mean=("pc_iterations", "mean"),
                test_acc_mean=("test_acc", "mean"),
                lambda_gate_mean=("lambda_gate", "mean"),
            )
        )
        pivot_out = outdir / "pivot_mean_pc_iterations.csv"
        pivot.to_csv(pivot_out, index=False)
        print(f"[saved] {pivot_out}")
>>>>>>> 1fd31e0ae33af1c36dc17abcdc89253e4300520e

if __name__ == "__main__":
    main()
