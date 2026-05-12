#!/usr/bin/env python3
"""Plot seed-robustness for MNIST, MNIST bottleneck, and CIFAR-10.

The script aggregates metrics from IL-ASGN, backprop, and predictive coding
runs collected across multiple random seeds (default: 0–9). For each dataset
family (MNIST, MNIST bottleneck, CIFAR-10) it produces a three-panel figure
covering Gaussian, salt-and-pepper, and occlusion noise. Lines show the mean
final test accuracy across seeds, and shaded regions denote the seed-wise
standard deviation, making it easy to verify that robustness curves are not
seed-dependent.

Defaults expect per-seed metrics with ``_seed{seed}.json`` suffixes, e.g.
``output/ilasgn_runs/mnist_gaussian_sigma0.1_seed3.json``. Override any dataset/noise
/method template with ``--template mnist.gaussian.pc=custom/path.json``.
Templates are formatted with ``{level}`` and ``{seed}`` placeholders and are
resolved relative to ``--metrics-root`` unless absolute.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from backprop_vs_pc_vs_ilasgn import _load_metrics, _normalize_accuracies, summarize


MethodName = str
NoiseFamily = str
DatasetName = str


@dataclass
class MethodTemplates:
    """Path templates for a noise family.

    Templates are format strings with ``level`` and ``seed`` fields.
    """

    ilasgn: str
    backprop: str
    pc: Optional[str] = None


DEFAULT_TEMPLATES: Dict[DatasetName, Dict[NoiseFamily, MethodTemplates]] = {
    "mnist": {
        "gaussian": MethodTemplates(
            ilasgn="output/ilasgn_runs/mnist_gaussian_sigma{level}_seed{seed}.json",
            backprop="output/backprop_runs/mnist_gaussian_sigma{level}_seed{seed}.json",
            pc="output/pc_runs/mnist_gaussian_sigma{level}_seed{seed}.json",
        ),
        "saltpepper": MethodTemplates(
            ilasgn="output/ilasgn_runs/mnist_saltpepper_{level}_seed{seed}.json",
            backprop="output/backprop_runs/mnist_saltpepper_{level}_seed{seed}.json",
            pc="output/pc_runs/mnist_saltpepper_{level}_seed{seed}.json",
        ),
        "occlusion": MethodTemplates(
            ilasgn="output/ilasgn_runs/mnist_occlusion_{level}_seed{seed}.json",
            backprop="output/backprop_runs/mnist_occlusion_{level}_seed{seed}.json",
            pc="output/pc_runs/mnist_occlusion_{level}_seed{seed}.json",
        ),
    },
    "mnist_bottleneck": {
        "gaussian": MethodTemplates(
            ilasgn="output/ilasgn_runs/mnist_bottleneck_gaussian_sigma{level}_seed{seed}.json",
            backprop="output/backprop_runs/mnist_bottleneck_gaussian_sigma{level}_seed{seed}.json",
            pc="output/pc_runs/mnist_bottleneck_gaussian_sigma{level}_seed{seed}.json",
        ),
        "saltpepper": MethodTemplates(
            ilasgn="output/ilasgn_runs/mnist_bottleneck_saltpepper_{level}_seed{seed}.json",
            backprop="output/backprop_runs/mnist_bottleneck_saltpepper_{level}_seed{seed}.json",
            pc="output/pc_runs/mnist_bottleneck_saltpepper_{level}_seed{seed}.json",
        ),
        "occlusion": MethodTemplates(
            ilasgn="output/ilasgn_runs/mnist_bottleneck_occlusion_{level}_seed{seed}.json",
            backprop="output/backprop_runs/mnist_bottleneck_occlusion_{level}_seed{seed}.json",
            pc="output/pc_runs/mnist_bottleneck_occlusion_{level}_seed{seed}.json",
        ),
    },
    "cifar10": {
        "gaussian": MethodTemplates(
            ilasgn="output/ilasgn_runs/cifar10_gaussian_sigma{level}_seed{seed}.json",
            backprop="output/backprop_runs/cifar10_gaussian_sigma{level}_seed{seed}.json",
            pc="output/pc_runs/cifar10_gaussian_sigma{level}_seed{seed}.json",
        ),
        "saltpepper": MethodTemplates(
            ilasgn="output/ilasgn_runs/cifar10_saltpepper_{level}_seed{seed}.json",
            backprop="output/backprop_runs/cifar10_saltpepper_{level}_seed{seed}.json",
            pc="output/pc_runs/cifar10_saltpepper_{level}_seed{seed}.json",
        ),
        "occlusion": MethodTemplates(
            ilasgn="output/ilasgn_runs/cifar10_occlusion_{level}_seed{seed}.json",
            backprop="output/backprop_runs/cifar10_occlusion_{level}_seed{seed}.json",
            pc="output/pc_runs/cifar10_occlusion_{level}_seed{seed}.json",
        ),
    },
}


X_LABELS: Mapping[NoiseFamily, str] = {
    "gaussian": "Sigma (σ)",
    "saltpepper": "Flip probability (p)",
    "occlusion": "Block fraction",
}

NOISE_ORDER: Sequence[NoiseFamily] = ("gaussian", "saltpepper", "occlusion")

METHOD_LABELS: Mapping[str, MethodName] = {
    "ilasgn": "IL-ASGN",
    "backprop": "Backprop",
    "pc": "PC",
}


def _load_final_accuracy(path: Path, label: str) -> Optional[float]:
    if not path.exists():
        print(f"[warn] Missing metrics for {label}: {path}")
        return None

    try:
        payload = _load_metrics(path)
        summary = _normalize_accuracies(summarize(label, payload))
        return summary.get("final_test_accuracy")
    except Exception as exc:  # noqa: BLE001 - script should not halt on malformed runs
        print(f"[warn] Failed to parse {label} metrics from {path}: {exc}")
        return None


def _resolve_template(root: Path, template: str) -> Path:
    tmpl_path = Path(template)
    return tmpl_path if tmpl_path.is_absolute() else root / tmpl_path


def _masked_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masked = np.ma.array(values, mask=np.isnan(values))
    return masked.mean(axis=1).filled(np.nan), masked.std(axis=1).filled(np.nan)


def _collect_seed_curves(
    levels: Sequence[float],
    seeds: Sequence[int],
    templates: MethodTemplates,
    metrics_root: Path,
    label_prefix: str,
    warn_if_seedless: bool,
) -> Dict[MethodName, List[List[float]]]:
    methods = {
        METHOD_LABELS["ilasgn"]: templates.ilasgn,
        METHOD_LABELS["backprop"]: templates.backprop,
    }
    if templates.pc:
        methods[METHOD_LABELS["pc"]] = templates.pc

    curves: Dict[MethodName, List[List[float]]] = {name: [] for name in methods}

    for level in levels:
        for method_label, template in methods.items():
            if warn_if_seedless and "{seed" not in template and len(seeds) > 1:
                print(
                    f"[warn] Template for {label_prefix} {method_label} lacks '{{seed}}' placeholder; "
                    f"the same file will be reused for all seeds."
                )
                warn_if_seedless = False

            seed_values: List[float] = []
            for seed in seeds:
                path = _resolve_template(metrics_root, template.format(level=level, seed=seed))
                acc = _load_final_accuracy(path, f"{label_prefix} {method_label} (seed {seed})")
                seed_values.append(np.nan if acc is None else acc)
            curves[method_label].append(seed_values)

    return curves


def _plot_noise_panel(ax, levels: Sequence[float], curves: Dict[MethodName, List[List[float]]], title: str, xlabel: str) -> None:
    level_array = np.array(levels, dtype=float)

    for method_label, level_seed_values in curves.items():
        values = np.array(level_seed_values, dtype=float)
        means, stds = _masked_mean_std(values)

        valid = ~np.isnan(means)
        if not np.any(valid):
            print(f"[info] Skipping {method_label} in '{title}' because all values are missing")
            continue

        ax.plot(level_array[valid], means[valid], marker="o", label=method_label)
        ax.fill_between(
            level_array[valid],
            means[valid] - stds[valid],
            means[valid] + stds[valid],
            alpha=0.15,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final test accuracy (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()


def _summarize_seed_spread(
    dataset: str,
    noise: str,
    levels: Sequence[float],
    seeds: Sequence[int],
    curves: Dict[MethodName, List[List[float]]],
) -> None:
    print(f"\n{dataset} – {noise} (seeds {list(seeds)}; levels {list(levels)}):")
    for method_label, level_seed_values in curves.items():
        values = np.array(level_seed_values, dtype=float)
        means, stds = _masked_mean_std(values)

        valid_means = means[~np.isnan(means)]
        valid_stds = stds[~np.isnan(stds)]
        if valid_means.size == 0:
            print(f"  {method_label}: no usable metrics")
            continue

        overall_mean = float(np.mean(valid_means))
        avg_seed_std = float(np.mean(valid_stds)) if valid_stds.size else float("nan")
        max_seed_std = float(np.max(valid_stds)) if valid_stds.size else float("nan")
        print(
            f"  {method_label}: mean(acc) {overall_mean:.2f}% | "
            f"avg seed std {avg_seed_std:.2f}% | max seed std {max_seed_std:.2f}%"
        )


def _apply_template_overrides(
    base: Dict[DatasetName, Dict[NoiseFamily, MethodTemplates]],
    overrides: Iterable[str],
) -> Dict[DatasetName, Dict[NoiseFamily, MethodTemplates]]:
    updated = {ds: {noise: MethodTemplates(**vars(tmpl)) for noise, tmpl in noises.items()} for ds, noises in base.items()}

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Template override must use dataset.noise.method=path syntax: '{override}'")

        key, path = override.split("=", 1)
        parts = key.split(".")
        if len(parts) != 3:
            raise ValueError(f"Template override must specify dataset.noise.method: '{override}'")

        dataset, noise, method = parts
        if dataset not in updated:
            raise ValueError(f"Unknown dataset '{dataset}' in override '{override}'")
        if noise not in updated[dataset]:
            raise ValueError(f"Unknown noise family '{noise}' in override '{override}'")
        if method not in METHOD_LABELS:
            raise ValueError(f"Unknown method '{method}' in override '{override}' (expected ilasgn/backprop/pc)")

        tmpl = updated[dataset][noise]
        setattr(tmpl, method, path)

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["mnist", "mnist_bottleneck", "cifar10"],
        help="Which dataset families to plot (subset of mnist, mnist_bottleneck, cifar10)",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=list(range(10)),
        help="Seeds to aggregate (default: 0 1 2 3 4 5 6 7 8 9)",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory for metric files; relative templates are resolved from here.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pi_plots_out"),
        help="Where to write the per-dataset robustness figures.",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Override a template with dataset.noise.method=path (e.g., mnist.gaussian.pc=custom.json)",
    )
    parser.add_argument(
        "--gaussian-levels",
        nargs="*",
        type=float,
        default=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Sigma values for Gaussian noise.",
    )
    parser.add_argument(
        "--saltpepper-levels",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3],
        help="Flip probabilities for salt-and-pepper noise.",
    )
    parser.add_argument(
        "--occlusion-levels",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3],
        help="Block fractions for occlusion noise.",
    )
    args = parser.parse_args()

    selected_levels = {
        "gaussian": args.gaussian_levels,
        "saltpepper": args.saltpepper_levels,
        "occlusion": args.occlusion_levels,
    }

    templates = _apply_template_overrides(DEFAULT_TEMPLATES, args.template)

    missing_datasets = [ds for ds in args.datasets if ds not in templates]
    if missing_datasets:
        raise SystemExit(f"Unknown dataset(s) requested: {', '.join(missing_datasets)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        print(f"\nGenerating seed-robustness plot for {dataset} (seeds {args.seeds})")
        fig, axes = plt.subplots(1, len(NOISE_ORDER), figsize=(6 * len(NOISE_ORDER), 4), constrained_layout=True)
        fig.suptitle(f"Seed robustness – {dataset}")

        dataset_templates = templates[dataset]
        for ax, noise in zip(axes, NOISE_ORDER):
            levels = selected_levels[noise]
            curves = _collect_seed_curves(
                levels=levels,
                seeds=args.seeds,
                templates=dataset_templates[noise],
                metrics_root=args.metrics_root,
                label_prefix=f"{dataset} {noise}",
                warn_if_seedless=True,
            )
            _plot_noise_panel(ax, levels, curves, f"{noise.title()} noise", X_LABELS[noise])
            _summarize_seed_spread(dataset, noise, levels, args.seeds, curves)

        output_path = args.output_dir / f"{dataset}_seed_robustness.png"
        fig.savefig(output_path, dpi=200)
        print(f"Saved {dataset} robustness figure to {output_path}")


if __name__ == "__main__":
    main()
