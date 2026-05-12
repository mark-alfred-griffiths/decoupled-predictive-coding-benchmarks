#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a vanilla predictive coding network on CIFAR-10 with JPC."""

import os
import sys
import time
import json
import argparse
import importlib.metadata as importlib_metadata
from pathlib import Path
from typing import Iterator, Tuple, Optional

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "core" / "jpc_main"))

_original_version = importlib_metadata.version
try:
    _original_version("jpc")
except importlib_metadata.PackageNotFoundError:
    import importlib.metadata as _importlib_metadata_module

    def _shimmed_version(name: str):
        if name == "jpc":
            return "0.0.0"
        return _original_version(name)

    _importlib_metadata_module.version = _shimmed_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="Device to run JAX on.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=133,
                        help="Training batch size (matches CIFAR-10 baselines in running.txt).")
    parser.add_argument("--eval-batch-size", type=int, default=512,
                        help="Evaluation batch size.")
    parser.add_argument("--train-steps", type=int, default=1500,
                        help="Number of optimisation steps (aligned with CIFAR-10 IL-ASGN budget).")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Logging frequency in optimisation steps.")
    parser.add_argument("--eval-every", type=int, default=300,
                        help="Evaluation frequency in optimisation steps.")
    parser.add_argument("--warmup-steps", type=int, default=1,
                        help="Number of initial steps to exclude from timing metrics to avoid JIT compilation overhead.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--width", type=int, default=128, help="Hidden layer width.")
    parser.add_argument("--depth", type=int, default=3, help="Number of layers including the output layer.")
    parser.add_argument("--activation", choices=[
        "relu", "tanh", "gelu", "silu", "selu", "hard_tanh", "leaky_relu", "linear"
    ], default="relu", help="Activation function for hidden layers.")
    parser.add_argument("--loss", choices=["mse", "ce"], default="ce",
                        help="Loss to use at the output layer.")
    parser.add_argument("--param-type", choices=["sp", "mupc", "ntp"], default="sp",
                        help="Predictive coding parameterisation.")
    parser.add_argument("--solver", choices=["heun", "rk4", "euler", "tsit5"], default="heun",
                        help="Diffrax ODE solver for inference.")
    parser.add_argument("--max-t1", type=float, default=20.0,
                        help="End time of the inference integration interval.")
    parser.add_argument("--dt", type=float, default=0.0,
                        help="Fixed integration step size. Set to 0 for adaptive step size.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for the PID controller.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for the PID controller.")
    parser.add_argument("--record-every", type=int, default=1,
                        help="Store every Nth inference iterate when measuring convergence steps.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay coefficient.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                        help="Global gradient norm clip. Set <=0 to disable.")
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="Optional path to save per-step metrics as JSON Lines.")
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="Probability of replacing a training label with a random class.")
    parser.add_argument("--input-noise-type", choices=["none", "gaussian", "salt_pepper", "occlusion"], default="none",
                        help="Type of corruption to apply to inputs before flattening.")
    parser.add_argument("--input-noise-level", type=float, default=0.0,
                        help="Strength of input corruption. Interpreted as σ for Gaussian noise, total flip probability for salt"
                        "-and-pepper noise, or occlusion block side as a fraction of the image size.")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"), help="Directory to download data to.")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Apply CIFAR-10 data augmentation (random crop + flip). Disable with --no-augment.")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Disable CIFAR-10 data augmentation.")
    return parser.parse_args()


def set_jax_platform(device_choice: str) -> None:
    if device_choice == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device_choice == "gpu":
        prefer_rocm = any(k in os.environ for k in ("ROCM_PATH", "HIP_VISIBLE_DEVICES", "ROCM_HOME"))
        platform = "rocm" if prefer_rocm else "cuda"
        os.environ["JAX_PLATFORMS"] = f"{platform},cpu"
    else:
        os.environ.pop("JAX_PLATFORMS", None)


args = parse_args()
set_jax_platform(args.device)

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import optax
import equinox as eqx

from diffrax import Heun, Tsit5, Euler, Dopri5, PIDController
from jpc import make_mlp, make_pc_step, test_discriminative_pc
try:
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch and torchvision are required to run pc_cifar10_baseline.py."
    ) from exc


def build_solver(name: str):
    if name == "heun":
        return Heun()
    if name == "rk4":
        return Dopri5()
    if name == "euler":
        return Euler()
    if name == "tsit5":
        return Tsit5()
    raise ValueError(f"Unknown solver {name}")


def _make_noise_transform(noise_type: str, noise_level: float):
    if noise_type == "none" or noise_level <= 0:
        return None

    if noise_type == "gaussian":
        def add_gaussian(x: torch.Tensor) -> torch.Tensor:
            return x + noise_level * torch.randn_like(x)

        return transforms.Lambda(add_gaussian)

    if noise_type == "salt_pepper":
        def add_salt_pepper(x: torch.Tensor) -> torch.Tensor:
            salt_pepper_mask = torch.rand_like(x)
            x_noisy = x.clone()
            salt = salt_pepper_mask < (noise_level / 2.0)
            pepper = (salt_pepper_mask >= (noise_level / 2.0)) & (salt_pepper_mask < noise_level)
            x_noisy[salt] = 1.0
            x_noisy[pepper] = 0.0
            return x_noisy

        return transforms.Lambda(add_salt_pepper)

    if noise_type == "occlusion":
        def add_occlusion(x: torch.Tensor) -> torch.Tensor:
            # ``noise_level`` scales the square side length relative to the shortest image dimension.
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


def _cifar_transforms(train: bool, *, augment: bool, noise_transform=None):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform_steps = []
    if train and augment:
        transform_steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transform_steps.append(transforms.ToTensor())
    transform_steps.append(transforms.Normalize(mean, std))
    if noise_transform is not None:
        transform_steps.append(noise_transform)
    transform_steps.append(transforms.Lambda(lambda x: torch.flatten(x)))
    return transforms.Compose(transform_steps)


def prepare_dataloaders(batch_size: int, eval_batch_size: int, data_root: str, augment: bool,
                        noise_type: str, noise_level: float) -> Tuple[DataLoader, DataLoader]:
    noise_transform = _make_noise_transform(noise_type, noise_level)
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True,
                                transform=_cifar_transforms(True, augment=augment, noise_transform=noise_transform))
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True,
                               transform=_cifar_transforms(False, augment=False, noise_transform=noise_transform))

    def collate(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=10).to(torch.float32)
        return imgs.to(torch.float32), one_hot.to(torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, collate_fn=collate)
    return train_loader, test_loader


def numpy_batch(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x, y = batch
    return jnp.asarray(x.numpy()), jnp.asarray(y.numpy())


def evaluate_model(model, test_loader: DataLoader, loss: str, param_type: str) -> Tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0
    total_examples = 0
    for batch in test_loader:
        x, y = numpy_batch(batch)
        loss_val, acc_val = test_discriminative_pc(model=model, output=y, input=x, loss=loss, param_type=param_type)
        loss_val, acc_val = jax.block_until_ready((loss_val, acc_val))
        batch_size = x.shape[0]
        total_loss += float(loss_val) * batch_size
        total_acc += float(acc_val) * batch_size
        total_examples += batch_size
    return total_loss / total_examples, total_acc / total_examples


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    train_loader, test_loader = prepare_dataloaders(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        data_root=args.data_root,
        augment=args.augment,
        noise_type=args.input_noise_type,
        noise_level=args.input_noise_level,
    )
    input_dim = 32 * 32 * 3
    output_dim = 10

    key, model_key = jax.random.split(key)
    model = make_mlp(
        key=model_key,
        input_dim=input_dim,
        width=args.width,
        depth=args.depth,
        output_dim=output_dim,
        act_fn=args.activation,
        use_bias=True,
        param_type=args.param_type,
    )
    skip_model = None

    optim_chain = []
    if args.grad_clip_norm > 0:
        optim_chain.append(optax.clip_by_global_norm(args.grad_clip_norm))
    optim_chain.append(optax.adam(args.lr))
    optimizer = optax.chain(*optim_chain)
    opt_state = optimizer.init((eqx.filter(model, eqx.is_array), None))

    solver = build_solver(args.solver)
    controller = PIDController(rtol=args.rtol, atol=args.atol)
    dt_value: Optional[float] = None if args.dt == 0 else args.dt

    metrics_file = None
    if args.metrics_json is not None:
        metrics_path = Path(args.metrics_json)
        if metrics_path.parent:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", encoding="utf-8")

    train_iter: Iterator = iter(train_loader)
    last_train_acc: float | None = None

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x_batch, y_batch = numpy_batch(batch)

        if args.label_noise > 0:
            num_classes = y_batch.shape[1]
            y_int = jnp.argmax(y_batch, axis=1)
            key, noise_key, label_key = jax.random.split(key, 3)
            mask = jax.random.bernoulli(noise_key, p=float(args.label_noise), shape=y_int.shape)
            n_flip = int(mask.sum())
            if n_flip > 0:
                rand_labels = jax.random.randint(label_key, (n_flip,), minval=0, maxval=num_classes, dtype=y_int.dtype)
                y_int_noisy = y_int.at[mask].set(rand_labels)
                y_batch = jnn.one_hot(y_int_noisy, num_classes, dtype=y_batch.dtype)

        step_start = time.perf_counter()
        result = make_pc_step(
            model=model,
            optim=optimizer,
            opt_state=opt_state,
            output=y_batch,
            input=x_batch,
            loss_id=args.loss,
            param_type=args.param_type,
            ode_solver=solver,
            max_t1=args.max_t1,
            dt=dt_value,
            stepsize_controller=controller,
            weight_decay=args.weight_decay,
            record_activities=True,
            record_every=args.record_every,
            calculate_accuracy=True,
        )
        loss_val = result["loss"] if result["loss"] is not None else jnp.array(0.0)
        jax.block_until_ready(loss_val)
        step_time = time.perf_counter() - step_start

        model = result["model"]
        opt_state = result["opt_state"]
        t_max = int(result["t_max"]) if result["t_max"] is not None else -1
        acc_val = float(result["acc"]) if result["acc"] is not None else float("nan")
        last_train_acc = acc_val
        loss_scalar = float(loss_val)

        metrics = {
            "step": step,
            "loss": loss_scalar,
            "train_acc": acc_val,
            "t_max": t_max,
            "inference_time_sec": None if step <= args.warmup_steps else step_time,
            # Metadata for downstream comparisons
            "dataset": "cifar10",
            "width": args.width,
            "depth": args.depth,
            "label_noise": args.label_noise,
            "input_noise_type": args.input_noise_type,
            "input_noise_level": args.input_noise_level,
            "augment": args.augment,
        }

        if metrics_file is not None:
            json.dump(metrics, metrics_file)
            metrics_file.write("\n")
            metrics_file.flush()

        if step % args.log_every == 0:
            if metrics["inference_time_sec"] is None:
                timing_str = "(warmup)"
            else:
                timing_str = f"{metrics['inference_time_sec']:.4f}s"
            print(
                f"step={step:05d} loss={loss_scalar:.4f} acc={acc_val:.2f}% t_max={t_max} inference_time={timing_str}")

        if step % args.eval_every == 0:
            eval_loss, eval_acc = evaluate_model(model, test_loader, args.loss, args.param_type)
            print(f"  eval: loss={eval_loss:.4f} acc={eval_acc:.2f}%")

            if metrics_file is not None:
                eval_metrics = {
                    "step": step,
                    "split": "eval",
                    "test_loss": float(eval_loss),
                    "test_acc": float(eval_acc),
                    "dataset": "cifar10",
                    "width": args.width,
                    "depth": args.depth,
                    "label_noise": args.label_noise,
                    "input_noise_type": args.input_noise_type,
                    "input_noise_level": args.input_noise_level,
                    "augment": args.augment,
                }
                json.dump(eval_metrics, metrics_file)
                metrics_file.write("\n")
                metrics_file.flush()

    if metrics_file is not None:
        final_loss, final_acc = evaluate_model(model, test_loader, args.loss, args.param_type)
        final_record = {
            "step": args.train_steps,
            "split": "final_eval",
            "train_acc": last_train_acc,
            "test_loss": float(final_loss),
            "test_acc": float(final_acc),
            "dataset": "cifar10",
            "width": args.width,
            "depth": args.depth,
            "label_noise": args.label_noise,
            "input_noise_type": args.input_noise_type,
            "input_noise_level": args.input_noise_level,
            "augment": args.augment,
        }
        json.dump(final_record, metrics_file)
        metrics_file.write("\n")
        metrics_file.flush()
        metrics_file.close()


if __name__ == "__main__":
    main()
