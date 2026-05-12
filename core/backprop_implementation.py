#!/usr/bin/env python3
"""Train a backprop MLP on CIFAR-10 with the same architecture as IL-ASGN.

This script mirrors the two-layer ReLU MLP used in ``train_path_integral_pc.py`` so you can
compare IL-ASGN against a standard backpropagation baseline without changing the
model. By default it uses the same width (1024), CIFAR-10 preprocessing, and a
matching data pipeline. Use ``--eval-every`` to monitor test accuracy during
training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "mnist"], default="cifar10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="Fraction of training labels to corrupt uniformly at random.")
    parser.add_argument("--num-train", type=int, default=0,
                        help="Optional cap on the number of training examples (0 = full dataset).")
    parser.add_argument("--input-noise-type", choices=["none", "gaussian", "salt_pepper", "occlusion"], default="none",
                        help="Optional input corruption to apply before flattening.")
    parser.add_argument("--input-noise-level", type=float, default=0.0,
                        help="Strength of the input corruption. Interpreted as σ for Gaussian noise, total flip probability for salt-and-pepper noise, or occlusion block side as a fraction of the image size.")
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="Optional path to dump train/test metrics for comparison with IL-ASGN")
    args, unknown = parser.parse_known_args()
    if unknown:
        script_name = Path(__file__).stem
        print(f"[{script_name}] Ignoring unknown CLI args: {unknown}")
    return args


args = parse_args()


# -------------------------
# Device
# -------------------------
def _set_platform_env(device_choice: str):
    if device_choice == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device_choice == "gpu":
        prefer_rocm = any(k in os.environ for k in ("ROCM_PATH", "HIP_VISIBLE_DEVICES", "ROCM_HOME"))
        platform = "rocm" if prefer_rocm else "cuda"
        os.environ["JAX_PLATFORMS"] = f"{platform},cpu"
    else:
        os.environ.pop("JAX_PLATFORMS", None)


_set_platform_env(args.device)


# -------------------------
# Data
# -------------------------
def _make_noise_transform(noise_type: str, noise_level: float):
    if noise_type == "none" or noise_level <= 0:
        return None

    if noise_type == "gaussian":
        def add_gaussian(x: torch.Tensor) -> torch.Tensor:
            return x + noise_level * torch.randn_like(x)

        return transforms.Lambda(add_gaussian)

    if noise_type == "salt_pepper":
        def add_salt_pepper(x: torch.Tensor) -> torch.Tensor:
            mask = torch.rand_like(x)
            x_noisy = x.clone()
            salt = mask < (noise_level / 2.0)
            pepper = (mask >= (noise_level / 2.0)) & (mask < noise_level)
            x_noisy[salt] = 1.0
            x_noisy[pepper] = 0.0
            return x_noisy

        return transforms.Lambda(add_salt_pepper)

    if noise_type == "occlusion":
        def add_occlusion(x: torch.Tensor) -> torch.Tensor:
            _, h, w = x.shape
            block_side = max(1, int(round(noise_level * min(h, w))))
            block_side = min(block_side, h, w)
            if block_side <= 0:
                return x
            top = torch.randint(0, h - block_side + 1, (1,)).item()
            left = torch.randint(0, w - block_side + 1, (1,)).item()
            x_noisy = x.clone()
            x_noisy[:, top:top + block_side, left:left + block_side] = 0.0
            return x_noisy

        return transforms.Lambda(add_occlusion)

    raise ValueError(f"Unknown noise type {noise_type}")


def _mnist_transforms(normalize: bool = True, noise_transform=None):
    transforms_list = [transforms.ToTensor()]
    if normalize:
        transforms_list.append(transforms.Normalize(mean=(0.1307,), std=(0.3081,)))
    if noise_transform is not None:
        transforms_list.append(noise_transform)
    return transforms.Compose(transforms_list)


def _cifar_transforms(train: bool, *, normalize: bool = True, augment: bool = False, noise_transform=None):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    t = []
    if train and augment:
        t.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    t.append(transforms.ToTensor())
    if normalize:
        t.append(transforms.Normalize(mean=mean, std=std))
    if noise_transform is not None:
        t.append(noise_transform)
    return transforms.Compose(t)


def _apply_label_noise(ds, *, n_classes: int, noise_fraction: float, seed: int):
    if noise_fraction <= 0:
        return ds
    if not 0.0 <= noise_fraction <= 1.0:
        raise ValueError("label-noise must be in [0, 1]")

    rng = torch.Generator().manual_seed(seed)
    n = len(ds)
    n_noisy = int(round(noise_fraction * n))
    if n_noisy == 0:
        return ds

    perm = torch.randperm(n, generator=rng)[:n_noisy]
    if hasattr(ds, "targets"):
        targets = torch.as_tensor(ds.targets)
    elif hasattr(ds, "labels"):
        targets = torch.as_tensor(ds.labels)
    else:
        return ds

    noisy = targets.clone()
    noisy[perm] = torch.randint(0, n_classes, (n_noisy,), generator=rng)
    if hasattr(ds, "targets"):
        ds.targets = noisy.tolist()
    else:
        ds.labels = noisy.tolist()
    return ds


def get_loaders(dataset: str, batch_size: int, *, augment: bool, noise_type: str, noise_level: float,
                label_noise: float, num_train: int, seed: int) -> Tuple[DataLoader, DataLoader, int, int]:
    noise_transform = _make_noise_transform(noise_type, noise_level)
    if dataset == "mnist":
        train = datasets.MNIST(str(DEFAULT_DATA_ROOT), train=True, download=True, transform=_mnist_transforms(True, noise_transform))
        test = datasets.MNIST(str(DEFAULT_DATA_ROOT), train=False, download=True, transform=_mnist_transforms(True, noise_transform))
        input_dim, n_classes = 28 * 28, 10
        flattener = lambda x: torch.flatten(x, start_dim=1)  # noqa: E731
    elif dataset in ("cifar10", "cifar100"):
        is100 = dataset == "cifar100"
        ds = datasets.CIFAR100 if is100 else datasets.CIFAR10
        train = ds(str(DEFAULT_DATA_ROOT), train=True, download=True, transform=_cifar_transforms(True, augment=augment, noise_transform=noise_transform))
        test = ds(str(DEFAULT_DATA_ROOT), train=False, download=True, transform=_cifar_transforms(False, augment=False, noise_transform=noise_transform))
        input_dim, n_classes = 32 * 32 * 3, (100 if is100 else 10)
        flattener = lambda x: x  # Keep CHW layout expected by eqx Conv2d  # noqa: E731
    else:
        raise ValueError("Unsupported dataset")

    train = _apply_label_noise(train, n_classes=n_classes, noise_fraction=label_noise, seed=seed)

    if num_train and num_train > 0:
        g = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train), generator=g)[:num_train].tolist()
        train = torch.utils.data.Subset(train, indices)

    def _collate(batch):
        xs, ys = zip(*batch)
        x = flattener(torch.stack(xs, dim=0))
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=_collate)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=_collate)
    return train_loader, test_loader, input_dim, n_classes


# -------------------------
# Model (mirrors train_path_integral_pc.py)
# -------------------------
def _linear_batch(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    """Batch-friendly linear used by IL-ASGN (matches train_path_integral_pc.py)."""

    return x @ linear.weight.T + linear.bias


class MLP(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    hidden_dim: int

    def __init__(self, in_size: int, width: int, out_size: int, key):
        k1, k2 = jax.random.split(key, 2)
        self.lin1 = eqx.nn.Linear(in_size, width, key=k1)
        self.lin2 = eqx.nn.Linear(width, out_size, key=k2)
        self.hidden_dim = width

    def __call__(self, x):
        h = jax.nn.relu(_linear_batch(self.lin1, x))
        return _linear_batch(self.lin2, h)

    def hidden(self, x):
        return jax.nn.relu(_linear_batch(self.lin1, x))

    def from_hidden(self, h):
        return _linear_batch(self.lin2, h)


class CifarConvNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    conv4: eqx.nn.Conv2d
    conv5: eqx.nn.Conv2d
    pool1: eqx.nn.MaxPool2d
    pool2: eqx.nn.MaxPool2d
    classifier: eqx.nn.Linear
    hidden_dim: int

    def __init__(self, out_size: int, base_width: int, key):
        base_width = max(16, int(base_width))
        k1, k2, k3, k4, k5, k_linear = jax.random.split(key, 6)
        self.conv1 = eqx.nn.Conv2d(3, base_width, kernel_size=3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(base_width, base_width, kernel_size=3, padding=1, key=k2)
        self.conv3 = eqx.nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1, key=k3)
        self.conv4 = eqx.nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1, key=k4)
        self.conv5 = eqx.nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1, key=k5)
        self.pool1 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        self.hidden_dim = base_width * 4
        self.classifier = eqx.nn.Linear(self.hidden_dim, out_size, key=k_linear)

    def hidden(self, x: jnp.ndarray) -> jnp.ndarray:
        def _encode_single(img):
            h_single = jax.nn.relu(self.conv1(img))
            h_single = jax.nn.relu(self.conv2(h_single))
            h_single = self.pool1(h_single)
            h_single = jax.nn.relu(self.conv3(h_single))
            h_single = jax.nn.relu(self.conv4(h_single))
            h_single = self.pool2(h_single)
            h_single = jax.nn.relu(self.conv5(h_single))
            return h_single

        h = eqx.filter_vmap(_encode_single, in_axes=0, out_axes=0)(x)
        return jnp.mean(h, axis=(2, 3))

    def from_hidden(self, h: jnp.ndarray) -> jnp.ndarray:
        return _linear_batch(self.classifier, h)

    def __call__(self, x):
        h = self.hidden(x)
        return self.from_hidden(h)


@dataclass
class Metrics:
    loss: float
    accuracy: float


# -------------------------
# Training utilities
# -------------------------
@eqx.filter_jit
def loss_fn(model, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = model(x)
    y_onehot = jax.nn.one_hot(y, num_classes=logits.shape[-1])
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))


@eqx.filter_jit
def compute_accuracy(model, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = model(x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y) * 100.0


@eqx.filter_jit
def train_step(model, opt_state, x: jnp.ndarray, y: jnp.ndarray, optim):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def evaluate(model, loader: Iterable, *, device: jax.typing.ArrayLike) -> Metrics:
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for x_np, y_np in loader:
        x = jnp.asarray(x_np, dtype=jnp.float32)
        y = jnp.asarray(y_np, dtype=jnp.int32)
        total_loss += float(loss_fn(model, x, y))
        total_acc += float(compute_accuracy(model, x, y))
        n_batches += 1
    return Metrics(loss=total_loss / n_batches, accuracy=total_acc / n_batches)


# -------------------------
# Main
# -------------------------

def main():
    torch.manual_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    train_loader, test_loader, input_dim, n_classes = get_loaders(
        args.dataset,
        args.batch_size,
        augment=args.augment,
        noise_type=args.input_noise_type,
        noise_level=args.input_noise_level,
        label_noise=args.label_noise,
        num_train=args.num_train,
        seed=args.seed,
    )
    steps_per_epoch = len(train_loader)
    key, init_key = jax.random.split(key)
    if args.dataset == "mnist":
        model = MLP(input_dim, args.width, n_classes, key=init_key)
    else:
        conv_base = max(16, min(int(args.width), 256))
        model = CifarConvNet(n_classes, conv_base, key=init_key)
    optim = optax.adam(args.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    metrics_history = [] if args.metrics_json else None

    def _maybe_record(step: int, epoch: int, *, train_loss, train_acc, test_loss=None, test_acc=None):
        if metrics_history is None:
            return
        metrics_history.append(
            {
                "step": int(step),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "test_loss": None if test_loss is None else float(test_loss),
                "test_accuracy": None if test_acc is None else float(test_acc),
            }
        )

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        for x_np, y_np in train_loader:
            x = jnp.asarray(x_np, dtype=jnp.float32)
            y = jnp.asarray(y_np, dtype=jnp.int32)
            model, opt_state, batch_loss = train_step(model, opt_state, x, y, optim)

            need_metrics = (global_step % args.log_every == 0) or (metrics_history is not None)
            batch_acc = compute_accuracy(model, x, y) if need_metrics else None
            if global_step % args.log_every == 0:
                print(
                    f"Step {global_step:05d} | Epoch {epoch:03d} | "
                    f"Train loss {float(batch_loss):.4f} | "
                    f"Train acc {(float(batch_acc) if batch_acc is not None else float('nan')):.2f}%"
                )

            if metrics_history is not None:
                _maybe_record(
                    global_step,
                    epoch,
                    train_loss=float(batch_loss),
                    train_acc=float(batch_acc) if batch_acc is not None else float("nan"),
                )
            global_step += 1

        if epoch % args.eval_every == 0:
            metrics = evaluate(model, test_loader, device=x.device)
            print(
                f"[Eval @ epoch {epoch}] Test loss {metrics.loss:.4f} | "
                f"Test acc {metrics.accuracy:.2f}%"
            )
            if metrics_history is not None:
                _maybe_record(
                    global_step,
                    epoch,
                    train_loss=float(batch_loss),
                    train_acc=float(batch_acc) if batch_acc is not None else float("nan"),
                    test_loss=float(metrics.loss),
                    test_acc=float(metrics.accuracy),
                )

    print("Training complete.")
    final_metrics = evaluate(model, test_loader, device=x.device)
    print(
        f"Final test loss {final_metrics.loss:.4f} | "
        f"Final test acc {final_metrics.accuracy:.2f}%"
    )

    if metrics_history is not None and args.metrics_json:
        out_path = Path(args.metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "seed": args.seed,
                "dataset": args.dataset,
                "augment": bool(args.augment),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "eval_every": args.eval_every,
                "log_every": args.log_every,
                "lr": args.lr,
                "width": args.width,
                "input_dim": input_dim,
                "n_classes": n_classes,
                "steps_per_epoch": steps_per_epoch,
                "input_noise_type": args.input_noise_type,
                "input_noise_level": args.input_noise_level,
            },
            "records": metrics_history,
            "final_test": {
                "loss": float(final_metrics.loss),
                "accuracy": float(final_metrics.accuracy),
            },
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics history to {out_path}")


if __name__ == "__main__":
    main()
