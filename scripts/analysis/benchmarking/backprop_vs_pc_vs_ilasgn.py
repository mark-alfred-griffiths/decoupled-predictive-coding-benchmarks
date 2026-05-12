#!/usr/bin/env python3
"""Compare backprop, predictive coding, and IL-ASGN metrics side-by-side.

Run IL-ASGN with ``--metrics-json`` in ``train_path_integral_pc.py``, the backprop baseline
with ``--metrics-json`` in ``backprop_implementation.py``, and predictive-coding
baselines with ``--metrics-json`` in the relevant PC runner (MNIST, bottleneck
MNIST, CIFAR-10, etc.). Then call this helper with the produced JSON files to
print a concise comparison of final and best test accuracy for each method.
Provide any two (or all three) sources; lengths must match across the provided
lists when supplying multiple pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from json import JSONDecodeError
from typing import Any, Dict, Iterable, Optional


def _coerce_payload(obj: Any) -> Dict[str, Any]:
    """Normalize decoded metrics into a dict with optional ``records`` list."""

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, list) and obj and all(isinstance(item, dict) for item in obj):
        return {"records": obj}

    raise JSONDecodeError("Unsupported metrics structure", str(obj), 0)


def _looks_like_full_payload(obj: Any) -> bool:
    return isinstance(obj, dict) and ("records" in obj or "config" in obj or "final_test" in obj)


def _load_metrics(path: Path) -> Dict[str, Any]:
    """Load metrics, tolerating concatenated/JSONL files by taking the last record.

    Handles three shapes:
    1. A single JSON object (standard payload).
    2. Multiple JSON objects concatenated without delimiters (e.g., repeated json.dump).
    3. Newline-delimited JSON objects (JSONL).
    """

    text = path.read_text(encoding="utf-8")

    # Fast path: a single valid JSON object.
    try:
        return _coerce_payload(json.loads(text))
    except JSONDecodeError:
        pass

    # Concatenated objects without separators.
    decoder = json.JSONDecoder()
    idx = 0
    decoded: list[Any] = []
    while idx < len(text):
        try:
            obj, end = decoder.raw_decode(text, idx)
        except JSONDecodeError:
            break
        decoded.append(obj)
        idx = end
        # Skip over any whitespace before the next object.
        while idx < len(text) and text[idx].isspace():
            idx += 1

    if decoded:
        if len(decoded) > 1 and not any(_looks_like_full_payload(obj) for obj in decoded):
            # Treat as a sequence of metric rows (JSONL without newlines).
            try:
                return _coerce_payload(decoded)
            except JSONDecodeError:
                pass
        return _coerce_payload(decoded[-1])

    # Newline-delimited objects: fall back to per-line parsing.
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except JSONDecodeError:
            continue

    if records:
        try:
            return _coerce_payload(records)
        except JSONDecodeError:
            pass

    raise JSONDecodeError("Could not parse metrics file", text, 0)


def _latest(records: Iterable[Dict[str, Any]], keys: Iterable[str]) -> Optional[float]:
    for rec in reversed(list(records)):
        for key in keys:
            value = rec.get(key)
            if value is not None:
                return float(value)
    return None


def _first(records: Iterable[Dict[str, Any]], keys: Iterable[str]) -> Optional[Any]:
    for rec in records:
        for key in keys:
            if rec.get(key) is not None:
                return rec.get(key)
    return None


def _best(records: Iterable[Dict[str, Any]], keys: Iterable[str]) -> Optional[float]:
    vals: list[float] = []
    for rec in records:
        for key in keys:
            if rec.get(key) is not None:
                vals.append(float(rec[key]))
                break
    return max(vals) if vals else None


def summarize(label: str, payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    records = payload.get("records", []) if isinstance(payload.get("records"), list) else []
    final = payload.get("final_test", {}) or payload.get("final_eval", {})

    def _extract_field(keys: list[str]) -> Optional[Any]:
        for source in (payload.get("config", {}), payload):
            for key in keys:
                if source.get(key) is not None:
                    return source.get(key)
        return _first(records, keys)

    dataset = _extract_field(["dataset", "dataset_name", "data", "task", "dataset_id"])
    width = _extract_field(["width", "hidden", "hidden_size"])  # width is optional for some PC runs

    train_accuracy_keys = ["train_accuracy", "train_acc"]
    test_accuracy_keys = ["test_accuracy", "test_acc", "eval_accuracy", "accuracy", "acc"]
    loss_keys = ["test_loss", "eval_loss", "loss", "train_loss"]

    return {
        "label": label,
        "dataset": dataset,
        "width": width,
        "final_train_accuracy": _latest(records, train_accuracy_keys),
        "final_test_accuracy": final.get("accuracy") if final else _latest(records, test_accuracy_keys),
        "final_test_loss": final.get("loss") if final else _latest(records, loss_keys),
        "best_test_accuracy": _best(records, test_accuracy_keys),
    }


def _detect_scale(*values: Optional[float]) -> Optional[str]:
    """Infer whether accuracies are stored as fractions (0–1) or percentages."""

    finite_values = [v for v in values if v is not None]
    if not finite_values:
        return None

    # If any value clearly exceeds 1, treat inputs as percentages already.
    if any(v > 1 for v in finite_values):
        return "percentage"
    return "fraction"


def _normalize_accuracy(value: Optional[float], scale: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    value = float(value)
    if scale == "fraction":
        return value * 100
    return value


def _normalize_accuracies(summary: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    scale = _detect_scale(
        summary.get("final_test_accuracy"),
        summary.get("best_test_accuracy"),
        summary.get("final_train_accuracy"),
    )

    return {
        **summary,
        "scale": scale or "unknown",
        "final_test_accuracy": _normalize_accuracy(summary.get("final_test_accuracy"), scale),
        "best_test_accuracy": _normalize_accuracy(summary.get("best_test_accuracy"), scale),
        "final_train_accuracy": _normalize_accuracy(summary.get("final_train_accuracy"), scale),
    }


def format_row(summary: Dict[str, Any]) -> str:
    def _fmt(value: Optional[float]) -> str:
        return f"{value:6.2f}%" if value is not None else "   n/a"

    return (
        f"{summary['label']:>12} | Dataset: {summary['dataset'] or '?':<8} | "
        f"Width: {summary['width'] or '?':>4} | "
        f"Final test: {_fmt(summary['final_test_accuracy'])} | "
        f"Best test: {_fmt(summary['best_test_accuracy'])} | "
        f"Final train: {_fmt(summary['final_train_accuracy'])} | "
        f"Scale detected: {summary['scale']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ilasgn",
        type=Path,
        nargs="*",
        default=[],
        help="Metrics JSON from train_path_integral_pc.py (one or more files)",
    )
    parser.add_argument(
        "--backprop",
        type=Path,
        nargs="*",
        default=[],
        help="Metrics JSON from backprop_implementation.py (one or more files)",
    )
    parser.add_argument(
        "--pc",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Metrics JSON from predictive-coding runs (MNIST, bottleneck MNIST, CIFAR-10, etc.; "
            "one or more files)"
        ),
    )
    args = parser.parse_args()

    provided = {label: paths for label, paths in {
        "IL-ASGN": args.ilasgn,
        "Backprop": args.backprop,
        "PredictiveCoding": args.pc,
    }.items() if paths}

    if len(provided) < 2:
        raise SystemExit("Provide metrics for at least two methods using --ilasgn/--backprop/--pc")

    lengths = {len(paths) for paths in provided.values()}
    if len(lengths) != 1:
        raise SystemExit("All provided metric lists must have the same length")

    multi = next(iter(lengths)) > 1
    blocks: list[str] = []

    method_order = [(label, paths) for label, paths in (
        ("IL-ASGN", args.ilasgn),
        ("Backprop", args.backprop),
        ("PredictiveCoding", args.pc),
    ) if paths]

    for idx in range(next(iter(lengths))):
        loaded: list[tuple[str, dict[str, float | None]]] = []
        for label, paths in method_order:
            summary = _normalize_accuracies(summarize(label, _load_metrics(paths[idx])))
            loaded.append((label, summary))

        names = " vs ".join(Path(paths[idx]).name for _, paths in method_order)
        header = (
            f"Pair {idx + 1}: {names}" if multi else f"Comparing {names}"
        )

        dataset_labels = {summary["dataset"] for _, summary in loaded if summary.get("dataset")}
        if dataset_labels:
            dataset_label = dataset_labels.pop() if len(dataset_labels) == 1 else ", ".join(sorted(dataset_labels))
            header = f"{header} [Dataset: {dataset_label}]"

        lines = [header, "=" * 80]
        lines.extend(format_row(summary) for _, summary in loaded)

        def maybe_add_delta(label_a: str, summary_a: dict[str, float | None], label_b: str, summary_b: dict[str, float | None]) -> None:
            if summary_a["final_test_accuracy"] is None or summary_b["final_test_accuracy"] is None:
                return
            delta = summary_a["final_test_accuracy"] - summary_b["final_test_accuracy"]
            lines.append(
                f"Δ(final acc) {label_a} - {label_b}: {delta:+.2f} percentage points"
            )

        if len(loaded) >= 2:
            lines.append("-" * 80)
            # Pairwise comparisons across available methods.
            for i in range(len(loaded)):
                for j in range(i + 1, len(loaded)):
                    maybe_add_delta(loaded[i][0], loaded[i][1], loaded[j][0], loaded[j][1])

        blocks.append("\n".join(lines))

    print("Comparison (test accuracy)")
    print("\n\n".join(blocks))
    print(f"\nProcessed {len(blocks)} comparison{'s' if len(blocks) != 1 else ''}.")


if __name__ == "__main__":
    main()
