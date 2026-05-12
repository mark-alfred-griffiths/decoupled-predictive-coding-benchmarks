#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot predictive-coding equilibration using actual PC iterations.

- Runs your training script across a grid of (dt, rollout_tol, rollout_steps, seed, λ_cap)
- Parses logs for **PC_steps=...** (actual PC iterations)
- Writes a long CSV and produces:
  * faceted time-series (rows=tol, cols=dt; lines by λ_cap with IQR shading)
  * Δ plot contrasting smallest vs largest λ_cap (faceted overview only)
  * λ-gate summary (if λ_gate appears in logs)
  * 3D surfaces of mean PC steps vs training step × dt
  * Δ-surfaces (λ=0 − λ=1) vs training step × dt

Robustness improvements:
- Use rounded keys (dt_key, tol_key) to avoid float-equality facet empties.
"""

import argparse
import itertools
import os
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runner", type=str, required=True,
                   help="Path to your IL-ASGN training script.")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu",
                   help="If 'cpu', forces JAX_PLATFORMS=cpu in child process.")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--train-steps", type=int, default=700)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--oracle-every", type=int, nargs="+", default=[1, 3, 5, 8])

    # Sweep grids
    p.add_argument("--grid-dt", type=float, nargs="+", default=[0.05])
    p.add_argument("--grid-tol", type=float, nargs="+", default=[1e-3])
    p.add_argument("--grid-steps", type=int, nargs="+", default=[4])

    p.add_argument("--lambda-cap", type=float, nargs="+", default=[1.0],
                   help="Use 0 for vanilla PC and 1 for IL-ASGN (can sweep multiple).")

    p.add_argument("--extra", type=str, default="",
                   help='Extra flags for the runner, e.g. --extra "--batch-size 256 --width 400"')

    p.add_argument("--outdir", type=str, default="output/pi_plots_out")
    p.add_argument("--csv-name", type=str, default="pi_equilibration_timeseries.csv")
    return p.parse_args()

# -------------------------
# Regex (PC_steps-focused + tolerant)
# -------------------------
STEP_RE = re.compile(r"\[Step\s+(\d+)\]|\bStep\s+(\d+)\b")
PC_STEPS_RE = re.compile(r"\bPC_steps\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
ACC_RE  = re.compile(r"\btest_acc\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
LOSS_RE = re.compile(r"\btest_loss\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
LAMBDA_GATE_RE = re.compile(r"\blambda_gate\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)
LAMBDA_GATE_RAW_RE = re.compile(r"\braw\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)", re.IGNORECASE)
CUDA_MISSING_RE = re.compile(r"No visible GPU devices|Unknown backend: 'gpu'", re.IGNORECASE)

def _f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# -------------------------
# Build command
# -------------------------
def _build_cmd(py: str, runner: str, device: str, seed: int,
               train_steps: int, eval_every: int, oracle_every: int,
               dt: float, tol: float, steps: int, lambda_cap: float,
               extra: str) -> List[str]:
    cmd = [
        py, runner,
        "--device", device,
        "--seed", str(seed),
        "--train-steps", str(train_steps),
        "--eval-every", str(eval_every),
        "--oracle-every", str(oracle_every),
        "--dt", str(dt),
        "--rollout-tol", str(tol),
        "--rollout-steps", str(steps),
        "--lambda-cap", str(lambda_cap),
    ]
    if extra:
        cmd += shlex.split(extra)
    return cmd

# -------------------------
# Log parsing (writes a row for every PC_steps found)
# -------------------------
def _parse_log_lines(lines: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    last_step: Optional[int] = None
    synthetic_step = 0

    for line in lines:
        # discover step if present
        m_step = STEP_RE.search(line)
        if m_step:
            step = int(next(g for g in m_step.groups() if g is not None))
            last_step = step
        else:
            step = last_step

        m_pc = PC_STEPS_RE.search(line)
        if not m_pc:
            continue

        if step is None:
            synthetic_step += 1
            step = synthetic_step

        pc_steps = _f(m_pc.group(1))

        m_acc = ACC_RE.search(line)
        m_loss = LOSS_RE.search(line)
        m_gate = LAMBDA_GATE_RE.search(line)
        m_gate_raw = LAMBDA_GATE_RAW_RE.search(line)

        rows.append({
            "train_step": float(step),                          # numeric
            "pi_steps": float(int(round(pc_steps))),            # coerce to integer count
            "pc_steps": float(int(round(pc_steps))),
            "test_acc": _f(m_acc.group(1)) if m_acc else float("nan"),
            "test_loss": _f(m_loss.group(1)) if m_loss else float("nan"),
            "lambda_gate": _f(m_gate.group(1)) if m_gate else float("nan"),
            "lambda_gate_raw": _f(m_gate_raw.group(1)) if m_gate_raw else float("nan"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = (df.sort_values("train_step")
                .drop_duplicates(subset=["train_step"], keep="last")
                .reset_index(drop=True))
    return df

def _parse_existing_log(log_path: Path) -> pd.DataFrame:
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        df = _parse_log_lines(f)
    print(f"[parse-existing] rows={len(df)} from {log_path.name}")
    return df

# -------------------------
# Run + parse a single job
# -------------------------
def _run_and_parse(cmd: List[str], env: Dict[str, str], log_path: Path) -> pd.DataFrame:
    tail_lines: List[str] = []
    TAIL_N = 120

    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)
        for line in proc.stdout:
            lf.write(line)
            tail_lines.append(line)
            if len(tail_lines) > TAIL_N:
                tail_lines.pop(0)
        ret = proc.wait()

    df = _parse_existing_log(log_path)

    if ret != 0:
        print("\n----- RUNNER LOG TAIL -----")
        print("".join(tail_lines).rstrip())
        print("----- END LOG TAIL -----\n")
        tail_text = "".join(tail_lines)
        if CUDA_MISSING_RE.search(tail_text):
            raise RuntimeError("CUDA missing")
        raise RuntimeError(f"Runner exited with code {ret}. See log: {log_path}")

    return df

def _log_filename(seed: int, dt: float, tol: float, steps: int,
                  oracle_every: int, lambda_cap: float, dev: str) -> str:
    lam = f"{lambda_cap:g}".replace(".", "p").replace("-", "m")
    return f"seed{seed}_lam{lam}_dt{dt}_tol{tol}_steps{steps}_oracle{oracle_every}_dev{dev}.log"

def _legacy_log_filename(seed: int, dt: float, tol: float, steps: int,
                         oracle_every: int, dev: str) -> str:
    return f"seed{seed}_dt{dt}_tol{tol}_steps{steps}_oracle{oracle_every}_dev{dev}.log"

def _infer_lambda_cap_from_df(df: pd.DataFrame) -> Optional[float]:
    if "lambda_gate" not in df:
        return None
    s = df["lambda_gate"].dropna()
    if s.empty:
        return None
    mx = float(s.max())
    if not math.isfinite(mx):
        return None
    return 0.0 if mx <= 1e-6 else 1.0

def run_one(runner: str, preferred_device: str, seed: int,
            train_steps: int, eval_every: int, oracle_every: int,
            dt: float, tol: float, steps: int, lambda_cap: float, extra: str,
            logs_dir: Path) -> pd.DataFrame:
    py = sys.executable

    def env_for(device: str) -> Dict[str, str]:
        e = os.environ.copy()
        e.setdefault("PYTHONIOENCODING", "UTF-8")
        e.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        e.setdefault("JAX_TRACEBACK_FILTERING", "off")
        if device == "cpu":
            e["JAX_PLATFORMS"] = "cpu"
        else:
            e["JAX_PLATFORMS"] = "cuda,cpu"
            e.setdefault("JAX_PJRT_USE_CUDA_PLUGIN", "1")
        return e

    log = logs_dir / _log_filename(seed, dt, tol, steps, oracle_every, lambda_cap, preferred_device)

    if log.exists() and log.stat().st_size > 0:
        print(f"[SKIP] Found existing log → parsing: {log.name}")
        return _parse_existing_log(log)

    # Legacy support (old name w/o λ in filename)
    legacy = logs_dir / _legacy_log_filename(seed, dt, tol, steps, oracle_every, preferred_device)
    if legacy.exists() and legacy.stat().st_size > 0:
        candidate = _parse_existing_log(legacy)
        inferred = _infer_lambda_cap_from_df(candidate)
        if inferred is not None and abs(inferred - lambda_cap) <= 1e-6:
            print(f"[SKIP] Using legacy log for λ_cap={lambda_cap:g}: {legacy.name}")
            return candidate
        else:
            print(f"[INFO] Legacy log {legacy.name} does not match λ_cap={lambda_cap:g}; launching new run.")

    cmd = _build_cmd(py, runner, preferred_device, seed, train_steps, eval_every,
                     oracle_every, dt, tol, steps, lambda_cap, extra)
    try:
        return _run_and_parse(cmd, env_for(preferred_device), log)
    except RuntimeError as e:
        if "CUDA missing" in str(e) and preferred_device != "cpu":
            print("[INFO] GPU missing → retrying on CPU.")
            log_cpu = logs_dir / _log_filename(seed, dt, tol, steps, oracle_every, lambda_cap, "cpu")
            if log_cpu.exists() and log_cpu.stat().st_size > 0:
                print(f"[SKIP] Found existing CPU log → parsing: {log_cpu.name}")
                return _parse_existing_log(log_cpu)
            cpu_cmd = _build_cmd(py, runner, "cpu", seed, train_steps, eval_every,
                                 oracle_every, dt, tol, steps, lambda_cap, extra)
            return _run_and_parse(cpu_cmd, env_for("cpu"), log_cpu)
        raise

# -------------------------
# Helpers
# -------------------------
def fmt_tol(x: float) -> str:
    return f"{x:.0e}" if x < 1e-2 else f"{x:g}"

def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if not s.empty else float("nan")

def _safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if not s.empty else float("nan")

def _safe_quantile(s: pd.Series, q: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.quantile(s, q)) if not s.empty else float("nan")

def _add_rounded_keys(df: pd.DataFrame, digits: int = 12) -> pd.DataFrame:
    """Add dt_key / tol_key rounded columns to stabilise group-by and filtering."""
    if df.empty:
        df["dt_key"] = df.get("dt", pd.Series([], dtype=float))
        df["tol_key"] = df.get("rollout_tol", pd.Series([], dtype=float))
        return df
    df = df.copy()
    df["dt_key"] = np.round(pd.to_numeric(df["dt"], errors="coerce"), digits)
    df["tol_key"] = np.round(pd.to_numeric(df["rollout_tol"], errors="coerce"), digits)
    return df

def _summarise(long_df: pd.DataFrame) -> pd.DataFrame:
    long_df = _add_rounded_keys(long_df)
    group_cols = ["lambda_cap", "dt_key", "tol_key", "train_step"]
    summary = (
        long_df.groupby(group_cols, as_index=False)
        .agg(
            pi_steps_mean=("pi_steps", _safe_mean),
            pi_steps_median=("pi_steps", _safe_median),
            pi_steps_q25=("pi_steps", lambda s: _safe_quantile(s, 0.25)),
            pi_steps_q75=("pi_steps", lambda s: _safe_quantile(s, 0.75)),
            runs=("pi_steps", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            dt=("dt", _safe_mean),                 # keep pretty labels
            rollout_tol=("rollout_tol", _safe_mean)
        )
    )
    return summary

# -------------------------
# Plots
# -------------------------
def plot_faceted_time_series(long_df: pd.DataFrame, outdir: Path):
    if long_df.empty:
        print("[WARN] Nothing to plot for faceted time-series.")
        return

    summary = _summarise(long_df)
    if summary.empty:
        print("[WARN] No summary rows for faceted time-series plot.")
        return

    dts_key = sorted(summary["dt_key"].unique())
    tols_key = sorted(summary["tol_key"].unique())
    lambda_vals = sorted(summary["lambda_cap"].dropna().unique())
    palette = plt.get_cmap("tab10")
    color_map = {lam: palette(i % palette.N) for i, lam in enumerate(lambda_vals)}

    nrows, ncols = len(tols_key), len(dts_key)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.5 * ncols, 3.4 * nrows),
                             sharex=True, sharey=True)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, tol_k in enumerate(tols_key):
        for j, dt_k in enumerate(dts_key):
            ax = axes[i, j]
            sub = summary[(summary["tol_key"] == tol_k) & (summary["dt_key"] == dt_k)]
            if sub.empty:
                ax.set_visible(False)
                continue

            tol_label = fmt_tol(float(sub["rollout_tol"].iloc[0]))
            dt_label = float(sub["dt"].iloc[0])
            ax.set_title(f"dt={dt_label:g}, tol={tol_label}")
            ax.grid(True, alpha=0.3)

            drew = False
            ymax = 0.0
            for lam in lambda_vals:
                lam_sub = sub[sub["lambda_cap"] == lam].sort_values("train_step")
                if lam_sub.empty:
                    continue
                color = color_map[lam]
                ax.plot(lam_sub["train_step"].values,
                        lam_sub["pi_steps_mean"].values,
                        color=color, linewidth=2.2, label=f"λ_cap={lam:g}")
                y1, y2 = lam_sub["pi_steps_q25"].values, lam_sub["pi_steps_q75"].values
                if not (np.all(np.isnan(y1)) or np.all(np.isnan(y2))):
                    ax.fill_between(lam_sub["train_step"].values, y1, y2,
                                    color=color, alpha=0.18, linewidth=0)
                    ymax = max(ymax, np.nanmax(y2))
                else:
                    ymax = max(ymax, np.nanmax(lam_sub["pi_steps_mean"].values))
                drew = True

            if drew:
                lo, hi = ax.get_ylim()
                ax.set_ylim(bottom=0, top=max(hi, ymax * 1.05))
            if i == nrows - 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel("mean PC steps")

    handles = [plt.Line2D([0], [0], color=color_map[lam], linewidth=2.2)
               for lam in lambda_vals]
    labels = [f"λ_cap={lam:g}" for lam in lambda_vals]
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)

    fig.suptitle(
        "Predictive-coding equilibration vs training step\n"
        "Facets: tolerance × dt | Lines: λ_cap with IQR shading"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = outdir / "faceted_pi_steps_vs_training.png"
    fig.savefig(out, dpi=220)
    print(f"[saved] {out}")


def plot_surface(long_df: pd.DataFrame, outdir: Path):
    """
    Loop over all (tol, rollout_steps, oracle_every) and plot mean PC-step surfaces
    vs training step × dt for each λ_cap value (multi-surface per figure).
    """
    if long_df.empty:
        print("[WARN] Nothing to plot for 3D surface.")
        return

    df = _add_rounded_keys(long_df)
    if "lambda_cap" not in df or df["lambda_cap"].isna().all():
        print("[WARN] No λ_cap column; skipping surface plot.")
        return

    # NEW: subdirectory for surfaces
    surfaces_dir = outdir / "surfaces"
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    lam_vals = sorted(df["lambda_cap"].dropna().unique())
    palette = plt.get_cmap("viridis")

    for tol_k in sorted(df["tol_key"].unique()):
        for rs_sel in sorted(df["rollout_steps"].unique()):
            for oracle_sel in sorted(df["oracle_every"].unique()):
                group_cols = ["lambda_cap", "dt_key", "tol_key",
                              "rollout_steps", "oracle_every", "train_step"]
                avg = (
                    df.groupby(group_cols, as_index=False)
                      .agg(pi_steps_mean=("pi_steps", "mean"))
                )
                subset = avg[
                    (avg["tol_key"] == tol_k)
                    & (avg["rollout_steps"] == rs_sel)
                    & (avg["oracle_every"] == oracle_sel)
                ]
                if subset.empty:
                    continue

                ts_vals = sorted(subset["train_step"].unique())
                dt_vals = sorted(subset["dt_key"].unique())

                fig = plt.figure(figsize=(8, 6))
                ax3d = fig.add_subplot(111, projection="3d")

                for i, lam in enumerate(lam_vals):
                    sub = subset[subset["lambda_cap"] == lam]
                    if sub.empty:
                        continue
                    Z = np.full((len(ts_vals), len(dt_vals)), np.nan)
                    for ii, ts in enumerate(ts_vals):
                        for jj, dt_k in enumerate(dt_vals):
                            row = sub[(sub["train_step"] == ts) & (sub["dt_key"] == dt_k)]
                            if not row.empty:
                                Z[ii, jj] = float(row["pi_steps_mean"].iloc[0])
                    T, TT = np.meshgrid(dt_vals, ts_vals)
                    color = palette(i / max(1, len(lam_vals) - 1))
                    ax3d.plot_surface(
                        T, TT, Z,
                        rstride=1, cstride=1,
                        color=color, alpha=0.6, linewidth=0, antialiased=True,
                    )

                ax3d.set_xlabel("dt")
                ax3d.set_ylabel("training step")
                ax3d.set_zlabel("mean PC steps")
                ax3d.set_title(
                    "Surfaces: mean PC steps vs training step × dt\n"
                    f"(tol≈{fmt_tol(tol_k)}, rollout_steps={rs_sel}, oracle_every={oracle_sel})"
                )

                # Manual legend
                from matplotlib.patches import Patch
                handles = [Patch(color=palette(i / max(1, len(lam_vals) - 1)),
                                 label=f"λ_cap={lam:g}") for i, lam in enumerate(lam_vals)]
                ax3d.legend(handles=handles, loc="best", frameon=True)

                out = surfaces_dir / (
                    f"surface_pc_steps_dt_train_multi_lambda_"
                    f"tol{fmt_tol(tol_k)}_rs{rs_sel}_oracle{oracle_sel}.png"
                )
                fig.savefig(out, dpi=220)
                print(f"[saved] {out}")



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
    """
    Aggregate raw PI/PC step data into mean, median, and quartile summaries
    grouped by λ_cap, dt, rollout_tol, and training step.
    """
    group_cols = ["lambda_cap", "dt", "rollout_tol", "train_step"]
    summary = (
        long_df.groupby(group_cols, as_index=False)
        .agg(
            pi_steps_mean=("pi_steps", _safe_mean),
            pi_steps_median=("pi_steps", _safe_median),
            pi_steps_q25=("pi_steps", lambda s: _safe_quantile(s, 0.25)),
            pi_steps_q75=("pi_steps", lambda s: _safe_quantile(s, 0.75)),
            runs=("pi_steps", lambda s: int(s.notna().sum())),
        )
    )
    return summary


def plot_delta_surface(long_df: pd.DataFrame, outdir: Path):
    """
    Loop over all (tol, rollout_steps, oracle_every) and plot Δ-surface for each:
        Δ = mean PC_steps(λ=0) − mean PC_steps(λ=1)
    Positive Δ ⇒ IL-ASGN (λ=1) converges faster than vanilla PC.
    """
    if long_df.empty or "lambda_cap" not in long_df:
        print("[WARN] Skipping Δ-surface; missing data.")
        return

    df = _add_rounded_keys(long_df)

    # NEW: subdirectory for delta surfaces
    delta_dir = outdir / "delta_surfaces"
    delta_dir.mkdir(parents=True, exist_ok=True)

    for tol_k in sorted(df["tol_key"].unique()):
        for rs_sel in sorted(df["rollout_steps"].unique()):
            for oracle_sel in sorted(df["oracle_every"].unique()):

                group_cols = ["lambda_cap", "dt_key", "tol_key",
                              "rollout_steps", "oracle_every", "train_step"]
                avg = (
                    df.groupby(group_cols, as_index=False)
                      .agg(pc_mean=("pi_steps", "mean"))
                )
                sub = avg[
                    (avg["tol_key"] == tol_k)
                    & (avg["rollout_steps"] == rs_sel)
                    & (avg["oracle_every"] == oracle_sel)
                ]
                if sub.empty:
                    continue

                piv = sub.pivot_table(
                    index=["dt_key", "train_step"],
                    columns="lambda_cap",
                    values="pc_mean"
                ).reset_index()

                if 0.0 not in piv.columns or 1.0 not in piv.columns:
                    print(f"[WARN] Need both λ=0 and λ=1 to compute Δ "
                          f"(tol={fmt_tol(tol_k)}, steps={rs_sel}, oracle={oracle_sel}).")
                    continue

                piv["delta"] = piv[0.0] - piv[1.0]   # positive ⇒ IL-ASGN saves steps

                ts_vals = sorted(piv["train_step"].unique())
                dt_vals = sorted(piv["dt_key"].unique())
                Z = np.full((len(ts_vals), len(dt_vals)), np.nan)
                for ii, ts in enumerate(ts_vals):
                    for jj, dt_k in enumerate(dt_vals):
                        row = piv[(piv["train_step"] == ts) & (piv["dt_key"] == dt_k)]
                        if not row.empty:
                            Z[ii, jj] = float(row["delta"].iloc[0])

                # Plot Δ-surface
                T, TT = np.meshgrid(dt_vals, ts_vals)
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                surf = ax.plot_surface(
                    T, TT, Z, rstride=1, cstride=1,
                    cmap="Reds", edgecolor="none",
                    alpha=0.9, antialiased=True,
                )
                ax.contour(T, TT, Z, zdir='z', offset=np.nanmin(Z), cmap="Reds", alpha=0.4)
                cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.08)
                cbar.set_label("Δ PC steps  (λ=0 − λ=1)")

                ax.set_xlabel("dt")
                ax.set_ylabel("training step")
                ax.set_zlabel("Δ PC steps  (λ=0 − λ=1)")
                ax.set_title(
                    "Δ-surface: IL-ASGN reduction in predictive-coding iterations\n"
                    f"(tol≈{fmt_tol(tol_k)}, rollout_steps={rs_sel}, oracle_every={oracle_sel})"
                )

                out = delta_dir / (
                    f"surface_delta_pc_steps_dt_train_"
                    f"tol{fmt_tol(tol_k)}_rs{rs_sel}_oracle{oracle_sel}.png"
                )
                fig.savefig(out, dpi=240)
                print(f"[saved] {out}")


def plot_lambda_difference(long_df: pd.DataFrame, outdir: Path):
    """
    Compare smallest vs largest λ_cap and plot Δ in PC steps.
    (Faceted overview across all (dt, tol); no per-slice loop.)
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
        keep = key_cols + ["pi_steps_mean", "pi_steps_q25", "pi_steps_q75"]
        sub = sub[keep].rename(
            columns={
                "pi_steps_mean": f"mean{suffix}",
                "pi_steps_q25": f"q25{suffix}",
                "pi_steps_q75": f"q75{suffix}",
            }
        )
        return sub

    small = _slice_for(lam_small, "_small")
    large = _slice_for(lam_large, "_large")

    merged = pd.merge(small, large, on=key_cols, how="inner")
    if merged.empty:
        print(
            "[WARN] Skipping λ-cap difference plot; no overlapping dt/tol/train_step "
            f"for λ_cap={lam_small} and λ_cap={lam_large}."
        )
        return

    # Δ = small − large (improvement means fewer steps with larger λ, so Δ>0)
    merged["improvement_mean"] = merged["mean_small"] - merged["mean_large"]
    merged["improvement_lo"] = merged["q25_small"] - merged["q75_large"]
    merged["improvement_hi"] = merged["q75_small"] - merged["q25_large"]

    dts = sorted(merged["dt"].unique())
    tols = sorted(merged["rollout_tol"].unique())
    nrows, ncols = len(tols), len(dts)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.1 * nrows),
        sharex=True,
        sharey=True,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, tol in enumerate(tols):
        for j, dt in enumerate(dts):
            ax = axes[i, j]
            sub = merged[(merged["rollout_tol"] == tol) & (merged["dt"] == dt)]
            if sub.empty:
                ax.set_visible(False)
                continue

            sub = sub.sort_values("train_step")
            ax.plot(
                sub["train_step"].values,
                sub["improvement_mean"].values,
                linewidth=2.2,
            )
            ax.fill_between(
                sub["train_step"].values,
                sub["improvement_lo"].values,
                sub["improvement_hi"].values,
                alpha=0.2,
                linewidth=0,
            )
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
            ax.set_title(f"dt={dt:g}, tol={fmt_tol(tol)}")
            ax.grid(True, alpha=0.3)
            if i == nrows - 1:
                ax.set_xlabel("training step")
            if j == 0:
                ax.set_ylabel(f"Δ PC steps (λ={lam_small:g} − λ={lam_large:g})")

    fig.suptitle(
        "Mean PC-step improvement from IL-ASGN over vanilla predictive coding\n"
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
        )
        .sort_values("train_step")
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(summary["train_step"].values, summary["gate_mean"].values, linewidth=2.4, label="Capped λ_gate")
    ax.fill_between(summary["train_step"].values, summary["gate_q25"].values,
                    summary["gate_q75"].values, alpha=0.2, linewidth=0)
    if summary["gate_raw_mean"].notna().any():
        ax.plot(summary["train_step"].values, summary["gate_raw_mean"].values,
                linewidth=1.8, linestyle="--", label="Raw λ_gate before cap")
    ax.set_xlabel("training step")
    ax.set_ylabel("λ gate value")
    ax.set_title("IL-ASGN gating behaviour across training")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = outdir / "lambda_gate_summary.png"
    fig.savefig(out, dpi=220)
    print(f"[saved] {out}")

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    logs_dir = outdir / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(
        args.grid_dt, args.grid_tol, args.grid_steps,
        args.seed, args.oracle_every, args.lambda_cap
    ))
    all_long = []

    for dt, tol, steps, seed, oracle_every, lambda_cap in combos:
        print(
            "\n=== RUN seed={seed} dt={dt} tol={tol} steps={steps} oracle_every={oracle_every} "
            "lambda_cap={lambda_cap} dev={dev} ===".format(
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
                logs_dir=logs_dir
            )
        except Exception as e:
            print(f"[ERROR] seed={seed} dt={dt} tol={tol} steps={steps}: {e}")
            df = pd.DataFrame(columns=["train_step","pi_steps","test_acc","test_loss"])

        if not df.empty:
            df["dt"] = float(dt)
            df["rollout_tol"] = float(tol)
            df["rollout_steps"] = int(steps)
            df["seed"] = int(seed)
            df["oracle_every"] = int(oracle_every)
            df["lambda_cap"] = float(lambda_cap)
        all_long.append(df)

    long_df = pd.concat(all_long, ignore_index=True) if all_long else pd.DataFrame()
    csv_path = outdir / args.csv_name
    long_df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path} (rows={len(long_df)})")
    print("rows:", len(long_df), "| with_PC_steps:", long_df["pi_steps"].notna().sum())

    plot_faceted_time_series(long_df, outdir)
    plot_surface(long_df, outdir)           # now loops over (tol, steps, oracle)
    plot_delta_surface(long_df, outdir)     # now loops over (tol, steps, oracle)
    plot_lambda_difference(long_df, outdir) # reverted to faceted overview only
    plot_lambda_gate_summary(long_df, outdir)

    if not long_df.empty:
        pivot = (
            _add_rounded_keys(long_df)
            .groupby(["lambda_cap","dt_key","tol_key","rollout_steps","oracle_every","train_step"],
                     as_index=False)
            .agg(
                pc_steps_mean=("pi_steps", "mean"),
                test_acc_mean=("test_acc", "mean"),
                lambda_gate_mean=("lambda_gate", "mean"),
                dt=("dt", _safe_mean),
                rollout_tol=("rollout_tol", _safe_mean),
            )
        )
        pivot_out = outdir / "pivot_mean_pi_steps.csv"
        pivot.to_csv(pivot_out, index=False)
        print(f"[saved] {pivot_out}")

if __name__ == "__main__":
    main()
