#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid-search path-integral hyperparameters and visualise PC convergence cost.

This script orchestrates a sweep over the path-integral related hyperparameters of
``main_adj.py`` (or another compatible training entrypoint exposing
``--metrics-json``). For every combination in the user-specified grid the script
launches training, collects the per-step ``max_steps`` values from the metrics
JSON, and parses ``lambda_gate`` traces from the textual logs. The aggregated
metrics are written to CSV and rendered in a set of 3D scatter plots showing how
``max_steps`` evolves throughout training as each hyperparameter varies.

Outputs stored under ``--outdir``:

* ``metrics/<combo>.json`` – cached metrics emitted by the runner.
* ``logs/<combo>.log`` – raw stdout/stderr for debugging and ``lambda_gate`` parsing.
* ``aggregated_metrics.csv`` – long-form table combining the sweep results.
* ``max_steps_vs_<param>.png`` – 3D scatter plots (x=param, y=train step,
  z=max_steps, colour-coded by ``lambda_gate`` when available).
* ``lambda_gate_timecourses.png`` – optional line plot summarising
  ``lambda_gate`` trajectories per configuration.

Example usage::

    python sweep_path_integral_grid.py \
        --oracle-every 1 2 4 \
        --dt 0.02 0.05 \
        --rollout-tol 1e-4 5e-4 \
        --rollout-steps 4 8 \
        --train-steps 400 \
        --eval-every 100
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

STEP_RE = re.compile(r"\[Step\s+(\d+)\]")
LAMBDA_RE = re.compile(r"lambda_gate=([+-]?(?:\d+(?:\.\d*)?|\.\d+)|nan)", re.IGNORECASE)
TAIL_LINES = 80


@dataclass(frozen=True)
class SweepPoint:
    oracle_every: int
    dt: float
    rollout_tol: float
    rollout_steps: int

    def as_label(self) -> str:
        return (
            f"oracle{self.oracle_every}_"
            f"dt{_format_float_token(self.dt)}_"
            f"rtol{_format_float_token(self.rollout_tol)}_"
            f"rsteps{self.rollout_steps}"
        )

    def as_dict(self) -> Dict[str, float]:
        return dict(
            oracle_every=self.oracle_every,
            dt=self.dt,
            rollout_tol=self.rollout_tol,
            rollout_steps=self.rollout_steps,
        )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke the training script",
    )
    p.add_argument(
        "--main-script",
        default=Path("main_adj.py"),
        type=Path,
        help="Path to the training script exposing --metrics-json",
    )
    p.add_argument(
        "--oracle-every",
        dest="oracle_every_values",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Grid values for --oracle-every",
    )
    p.add_argument(
        "--dt",
        dest="dt_values",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1],
        help="Grid values for --dt",
    )
    p.add_argument(
        "--rollout-tol",
        dest="rollout_tol_values",
        type=float,
        nargs="+",
        default=[1e-4, 5e-4, 1e-3],
        help="Grid values for --rollout-tol",
    )
    p.add_argument(
        "--rollout-steps",
        dest="rollout_steps_values",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Grid values for --rollout-steps",
    )
    p.add_argument(
        "--train-steps",
        type=int,
        default=400,
        help="Number of optimisation steps per run",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Evaluation interval passed to the runner",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Optional batch size forwarded to the runner",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed forwarded to the training script",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="gpu",
        help="Preferred device for the child process",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("path_integral_sweep"),
        help="Directory to store cached metrics, logs, and plots",
    )
    p.add_argument(
        "--csv-name",
        type=str,
        default="aggregated_metrics.csv",
        help="Filename for the aggregated CSV (stored under --outdir)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run training even if cached metrics/logs exist",
    )
    p.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=[],
        help="Additional flag(s) forwarded to the runner (repeatable)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _format_float_token(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1:
        token = f"{value:.3f}".rstrip("0").rstrip(".")
    else:
        token = f"{value:.3g}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def _build_command(
    python: str,
    script: Path,
    seed: int,
    device: str,
    batch_size: int,
    train_steps: int,
    eval_every: int,
    point: SweepPoint,
    metrics_path: Path,
    extra_args: Sequence[str],
) -> List[str]:
    cmd: List[str] = [
        python,
        str(script),
        f"--seed={seed}",
        f"--device={device}",
        f"--batch-size={batch_size}",
        f"--train-steps={train_steps}",
        f"--eval-every={eval_every}",
        f"--oracle-every={point.oracle_every}",
        f"--dt={point.dt}",
        f"--rollout-tol={point.rollout_tol}",
        f"--rollout-steps={point.rollout_steps}",
        f"--metrics-json={metrics_path}",
    ]
    cmd.extend(extra_args)
    return cmd


def _maybe_read_tail(path: Path, max_lines: int = TAIL_LINES) -> str:
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return "<log unavailable>"
    tail = "".join(lines[-max_lines:])
    return tail


def _run_training(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("============================================================")
    print("Running:")
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_fh:
        result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        tail = _maybe_read_tail(log_path)
        raise RuntimeError(
            f"Runner failed with code {result.returncode}. See {log_path} for full log.\n"
            f"Last {TAIL_LINES} log lines:\n{tail}"
        )


def _parse_lambda_from_log(log_path: Path) -> pd.DataFrame:
    steps: List[int] = []
    lambdas: List[float] = []
    current_step: Optional[int] = None

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        print(f"[WARN] Log file missing: {log_path}")
        return pd.DataFrame(columns=["train_step", "lambda_gate"])

    for line in lines:
        step_match = STEP_RE.search(line)
        if step_match:
            current_step = int(step_match.group(1))

        lambda_match = LAMBDA_RE.search(line)
        if lambda_match and current_step is not None:
            token = lambda_match.group(1)
            if token.lower() != "nan":
                try:
                    value = float(token)
                except ValueError:
                    value = math.nan
            else:
                value = math.nan

            if math.isfinite(value):
                steps.append(current_step)
                lambdas.append(value)
            current_step = None

    if not steps:
        return pd.DataFrame(columns=["train_step", "lambda_gate"])
    df = pd.DataFrame({"train_step": steps, "lambda_gate": lambdas})
    return df.drop_duplicates(subset="train_step")


def _load_metrics(metrics_path: Path) -> pd.DataFrame:
    try:
        with metrics_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        raise RuntimeError(f"Metrics JSON missing: {metrics_path}") from None

    records = payload.get("records", [])
    if not isinstance(records, list):
        raise RuntimeError(f"Unexpected metrics payload in {metrics_path}: {type(records)}")

    if not records:
        return pd.DataFrame(columns=["train_step", "max_steps"])

    df = pd.DataFrame(records)
    if "step" not in df.columns:
        raise RuntimeError(f"Metrics JSON missing 'step' column: {metrics_path}")
    df = df.rename(columns={"step": "train_step"})
    if "max_steps" not in df.columns:
        raise RuntimeError(f"Metrics JSON missing 'max_steps': {metrics_path}")
    return df[["train_step", "max_steps"]]


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------

def _merge_metrics(
    point: SweepPoint,
    metrics_df: pd.DataFrame,
    lambda_df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    merged = metrics_df.merge(lambda_df, on="train_step", how="left")
    for key, value in point.as_dict().items():
        merged[key] = value
    merged["combo"] = label
    return merged


def _plot_max_steps(df: pd.DataFrame, param: str, out_path: Path) -> None:
    subset = df.dropna(subset=[param, "max_steps"]).copy()
    if subset.empty:
        print(f"[WARN] No data available to plot {param} vs max_steps.")
        return

    has_lambda = subset["lambda_gate"].notna().any()
    colour = subset["lambda_gate"].fillna(subset["lambda_gate"].median() if has_lambda else 0.0)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        subset[param],
        subset["train_step"],
        subset["max_steps"],
        c=colour,
        cmap="viridis",
        marker="o",
        edgecolor="k",
        linewidths=0.2,
        alpha=0.85,
    )
    ax.set_xlabel(param)
    ax.set_ylabel("Training step")
    ax.set_zlabel("PC iterations (max_steps)")
    ax.set_title(f"max_steps vs {param}")
    if has_lambda:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
        cbar.set_label("lambda_gate")
    else:
        sc.set_cmap("viridis")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


def _plot_lambda_timecourses(lambda_df: pd.DataFrame, out_path: Path) -> None:
    if lambda_df.empty:
        print("[WARN] No lambda_gate samples found; skipping timecourse plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for combo, group in lambda_df.groupby("combo"):
        group = group.sort_values("train_step")
        ax.plot(
            group["train_step"],
            group["lambda_gate"],
            marker="o",
            linestyle="-",
            label=combo,
            alpha=0.8,
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel("lambda_gate")
    ax.set_title("lambda_gate trajectories across sweep")
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    outdir: Path = args.outdir
    metrics_dir = outdir / "metrics"
    logs_dir = outdir / "logs"
    plots_dir = outdir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    points: List[SweepPoint] = [
        SweepPoint(o, d, rt, rs)
        for o, d, rt, rs in itertools.product(
            args.oracle_every_values,
            args.dt_values,
            args.rollout_tol_values,
            args.rollout_steps_values,
        )
    ]

    if not points:
        raise RuntimeError("Empty sweep grid; supply at least one value per parameter.")

    aggregated_rows: List[pd.DataFrame] = []
    lambda_rows: List[pd.DataFrame] = []

    for point in points:
        label = point.as_label()
        metrics_path = metrics_dir / f"{label}.json"
        log_path = logs_dir / f"{label}.log"

        need_run = args.force or not metrics_path.exists()
        if need_run:
            cmd = _build_command(
                python=args.python,
                script=args.main_script,
                seed=args.seed,
                device=args.device,
                batch_size=args.batch_size,
                train_steps=args.train_steps,
                eval_every=args.eval_every,
                point=point,
                metrics_path=metrics_path,
                extra_args=args.extra_args,
            )
            _run_training(cmd, log_path)
        elif not log_path.exists():
            # The metrics are cached but log missing → re-run to recover lambda trace.
            cmd = _build_command(
                python=args.python,
                script=args.main_script,
                seed=args.seed,
                device=args.device,
                batch_size=args.batch_size,
                train_steps=args.train_steps,
                eval_every=args.eval_every,
                point=point,
                metrics_path=metrics_path,
                extra_args=args.extra_args,
            )
            _run_training(cmd, log_path)
        else:
            print(f"[cache] Using cached metrics/logs for {label}")

        metrics_df = _load_metrics(metrics_path)
        lambda_df = _parse_lambda_from_log(log_path)
        merged = _merge_metrics(point, metrics_df, lambda_df, label)
        aggregated_rows.append(merged)

        if not lambda_df.empty:
            temp = lambda_df.copy()
            temp["combo"] = label
            lambda_rows.append(temp)

    if not aggregated_rows:
        raise RuntimeError("No sweep results collected; aborting.")

    all_metrics = pd.concat(aggregated_rows, ignore_index=True)
    csv_path = outdir / args.csv_name
    all_metrics.to_csv(csv_path, index=False)
    print(f"[data] Wrote aggregated metrics to {csv_path}")

    for param in ("oracle_every", "dt", "rollout_tol", "rollout_steps"):
        plot_path = plots_dir / f"max_steps_vs_{param}.png"
        _plot_max_steps(all_metrics, param, plot_path)

    if lambda_rows:
        lambda_all = pd.concat(lambda_rows, ignore_index=True)
    else:
        lambda_all = pd.DataFrame(columns=["train_step", "lambda_gate", "combo"])
    lambda_plot_path = plots_dir / "lambda_gate_timecourses.png"
    _plot_lambda_timecourses(lambda_all, lambda_plot_path)


if __name__ == "__main__":
    main()
