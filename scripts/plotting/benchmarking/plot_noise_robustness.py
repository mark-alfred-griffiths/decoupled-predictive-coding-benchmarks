#!/usr/bin/env python3
"""Plot IL-ASGN vs backprop vs PC robustness to input noise and report summary metrics.

Expected metrics layout (matching running.txt):
- Gaussian: output/ilasgn_runs/mnist_gaussian_sigma{level}[_seed{seed}].json, output/backprop_runs/mnist_gaussian_sigma{level}[_seed{seed}].json, output/pc_runs/mnist_gaussian_sigma{level}[_seed{seed}].json
- Salt & pepper: output/ilasgn_runs/mnist_saltpepper_{level}[_seed{seed}].json, output/backprop_runs/mnist_saltpepper_{level}[_seed{seed}].json, output/pc_runs/mnist_saltpepper_{level}[_seed{seed}].json
- Occlusion: output/ilasgn_runs/mnist_occlusion_{level}[_seed{seed}].json, output/backprop_runs/mnist_occlusion_{level}[_seed{seed}].json, output/pc_runs/mnist_occlusion_{level}[_seed{seed}].json

Use the optional `{seed}` placeholder in the metrics patterns when averaging across
multiple seeds (see --seeds).

The script overlays IL-ASGN, Backprop, and Predictive Coding (PC) on shared axes for
each noise type and prints a simple robustness score: mean final-test accuracy
across severities. PC metrics are optional; missing files are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from backprop_vs_pc_vs_ilasgn import _load_metrics, _normalize_accuracies, summarize


def _load_acc(path: Path, label: str) -> float | None:
    if not path.exists():
        print(f"[warning] Missing metrics file for {label}: {path}")
        return None

    try:
        payload = _load_metrics(path)
        summary = _normalize_accuracies(summarize(label, payload))
        return summary.get("final_test_accuracy")
    except Exception as exc:  # noqa: BLE001 - user-facing script, print & continue
        print(f"[warning] Failed to load {label} metrics from {path}: {exc}")
        return None


def _aggregate_over_seeds(
    pattern: Path, label: str, level: float, seeds: List[int]
) -> tuple[float | None, float | None]:
    accs: List[float] = []
    for seed in seeds:
        path = Path(str(pattern).format(level=level, seed=seed))
        acc = _load_acc(path, f"{label} (seed {seed})")
        if acc is not None:
            accs.append(acc)

    if not accs:
        return None, None

    mean = float(np.mean(accs))
    std = float(np.std(accs))
    return mean, std


def collect_noise_curve(
    levels: List[float],
    seeds: List[int],
    ilasgn_pattern: Path,
    backprop_pattern: Path,
    pc_pattern: Path | None = None,
) -> dict[str, dict[str, List[float | None]]]:
    collected: dict[str, dict[str, List[float | None]]] = {
        "IL-ASGN": {"mean": [], "std": []},
        "Backprop": {"mean": [], "std": []},
    }

    if pc_pattern is not None:
        collected["PC"] = {"mean": [], "std": []}

    for level in levels:
        il_mean, il_std = _aggregate_over_seeds(ilasgn_pattern, "IL-ASGN", level, seeds)
        bp_mean, bp_std = _aggregate_over_seeds(backprop_pattern, "Backprop", level, seeds)
        collected["IL-ASGN"]["mean"].append(il_mean)
        collected["IL-ASGN"]["std"].append(il_std)
        collected["Backprop"]["mean"].append(bp_mean)
        collected["Backprop"]["std"].append(bp_std)

        if pc_pattern is not None:
            pc_mean, pc_std = _aggregate_over_seeds(pc_pattern, "PC", level, seeds)
            collected["PC"]["mean"].append(pc_mean)
            collected["PC"]["std"].append(pc_std)

    return collected


def robustness_score(stats: dict[str, List[float | None]]) -> float | None:
    vals = [a for a in stats["mean"] if a is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def plot_noise_axes(
    ax,
    levels: List[float],
    curves: dict[str, dict[str, List[float | None]]],
    title: str,
    xlabel: str,
) -> None:
    markers = {
        "IL-ASGN": "o",
        "Backprop": "s",
        "PC": "^",
    }

    for label, stats in curves.items():
        mean = np.array([m if m is not None else np.nan for m in stats["mean"]], dtype=float)
        std = np.array([s if s is not None else np.nan for s in stats["std"]], dtype=float)
        ax.plot(levels, mean, marker=markers.get(label, "o"), label=label)
        ax.fill_between(levels, mean - std, mean + std, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final test accuracy (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=Path(__file__).resolve().parent,
                        help="Base directory for metric files; relative patterns are resolved from here.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0],
                        help="Seeds to average for each noise level (use {seed} in the metrics patterns when passing multiple seeds).")
    parser.add_argument("--gaussian-levels", nargs="*", type=float, default=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help="Sigma values for Gaussian noise (default matches running.txt sweep).")
    parser.add_argument("--saltpepper-levels", nargs="*", type=float, default=[0.1, 0.2, 0.3],
                        help="Flip probabilities for salt-and-pepper noise.")
    parser.add_argument("--occlusion-levels", nargs="*", type=float, default=[0.1, 0.2, 0.3],
                        help="Block side fractions for occlusion noise.")
    parser.add_argument("--gaussian-ilasgn", default="output/ilasgn_runs/mnist_gaussian_sigma{level}.json",
                        help="Format string for IL-ASGN Gaussian metrics path.")
    parser.add_argument("--gaussian-backprop", default="output/backprop_runs/mnist_gaussian_sigma{level}.json",
                        help="Format string for backprop Gaussian metrics path.")
    parser.add_argument("--gaussian-pc", default="output/pc_runs/mnist_gaussian_sigma{level}.json",
                        help="Format string for predictive-coding Gaussian metrics path (optional).")
    parser.add_argument("--saltpepper-ilasgn", default="output/ilasgn_runs/mnist_saltpepper_{level}.json",
                        help="Format string for IL-ASGN salt-pepper metrics path (level placeholder).")
    parser.add_argument("--saltpepper-backprop", default="output/backprop_runs/mnist_saltpepper_{level}.json",
                        help="Format string for backprop salt-pepper metrics path.")
    parser.add_argument("--saltpepper-pc", default="output/pc_runs/mnist_saltpepper_{level}.json",
                        help="Format string for predictive-coding salt-pepper metrics path (optional).")
    parser.add_argument("--occlusion-ilasgn", default="output/ilasgn_runs/mnist_occlusion_{level}.json",
                        help="Format string for IL-ASGN occlusion metrics path.")
    parser.add_argument("--occlusion-backprop", default="output/backprop_runs/mnist_occlusion_{level}.json",
                        help="Format string for backprop occlusion metrics path.")
    parser.add_argument("--occlusion-pc", default="output/pc_runs/mnist_occlusion_{level}.json",
                        help="Format string for predictive-coding occlusion metrics path (optional).")
    parser.add_argument("--output", type=Path, default=Path("output/pi_plots_out/noise_robustness.png"),
                        help="Where to save the robustness figure.")
    args = parser.parse_args()

    gaussian_levels = args.gaussian_levels
    saltpepper_levels = args.saltpepper_levels
    occlusion_levels = args.occlusion_levels

    # Resolve relative metric paths against the metrics root so the script can be launched
    # from any working directory (e.g., IDE run configs).
    def _resolve(pattern: str) -> Path:
        p = Path(pattern)
        return (args.metrics_root / p) if not p.is_absolute() else p

    ga_il_pattern = _resolve(args.gaussian_ilasgn)
    ga_bp_pattern = _resolve(args.gaussian_backprop)
    ga_pc_pattern = _resolve(args.gaussian_pc) if args.gaussian_pc else None
    sp_il_pattern = _resolve(args.saltpepper_ilasgn)
    sp_bp_pattern = _resolve(args.saltpepper_backprop)
    sp_pc_pattern = _resolve(args.saltpepper_pc) if args.saltpepper_pc else None
    occ_il_pattern = _resolve(args.occlusion_ilasgn)
    occ_bp_pattern = _resolve(args.occlusion_backprop)
    occ_pc_pattern = _resolve(args.occlusion_pc) if args.occlusion_pc else None

    args.output.parent.mkdir(parents=True, exist_ok=True)

    noise_specs = [
        ("Gaussian noise robustness", "Sigma (σ)", gaussian_levels, ga_il_pattern, ga_bp_pattern, ga_pc_pattern),
        ("Salt–pepper noise robustness", "Flip probability (p)", saltpepper_levels, sp_il_pattern, sp_bp_pattern, sp_pc_pattern),
        ("Occlusion noise robustness", "Block fraction", occlusion_levels, occ_il_pattern, occ_bp_pattern, occ_pc_pattern),
    ]

    fig, axes = plt.subplots(1, len(noise_specs), figsize=(6 * len(noise_specs), 4), constrained_layout=True)
    if len(noise_specs) == 1:
        axes = [axes]

    collected = []
    for ax, (title, xlabel, levels, il_pattern, bp_pattern, pc_pattern) in zip(axes, noise_specs):
        curves = collect_noise_curve(levels, args.seeds, il_pattern, bp_pattern, pc_pattern)
        plot_noise_axes(ax, levels, curves, title, xlabel)
        collected.append((title, levels, curves))

    fig.suptitle("IL-ASGN vs Backprop vs PC robustness")
    fig.savefig(args.output, dpi=200)
    print(f"Saved robustness figure to {args.output}")

    def _fmt(score: float | None) -> str:
        return f"{score:.2f}%" if score is not None else "n/a"

    print("\nRobustness summary (mean final test accuracy across severities):")
    for title, levels, curves in collected:
        il_score = robustness_score(curves["IL-ASGN"])
        bp_score = robustness_score(curves["Backprop"])
        pc_score = robustness_score(curves.get("PC", [])) if "PC" in curves else None

        summary_parts = [
            f"IL-ASGN {_fmt(il_score)}",
            f"Backprop {_fmt(bp_score)}",
        ]
        if pc_score is not None:
            summary_parts.append(f"PC {_fmt(pc_score)}")

        print(f"{title.split()[0]} ({levels}): " + " | ".join(summary_parts))
        if il_score is not None and bp_score is not None:
            print(f"  Δ IL-ASGN – Backprop: {il_score - bp_score:+.2f} percentage points")
        if pc_score is not None and il_score is not None:
            print(f"  Δ IL-ASGN – PC: {il_score - pc_score:+.2f} percentage points")


if __name__ == "__main__":
    main()
