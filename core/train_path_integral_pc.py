#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST/CIFAR in JAX/Equinox with a Path-Integral Oracle and Learned Precisions (stable).

This version uses a discrete predictive-coding rollout (Option B) implemented with
a single lax.scan, so it's reverse-mode/differentiation-safe (no dynamic fori_loop).
PI_max_steps now reflects the actual number of PC iterations until convergence
(or the cap given by --rollout-steps).

It also:
- Caps the oracle blend with --lambda-cap (and logs raw/capped gate values).
- Norm-matches oracle grads to task grad norm before blending.
- Restores verbose per-step logging.
- Logs batch train loss/accuracy every step, and full test loss/accuracy at --eval-every.
- Dataset switch (mnist/cifar10/cifar100), augmentation, input noise, label noise.
- NEW: Decay-based estimator of total PC iterations with adaptive burn-in and log-linear fit fallback.
"""

import os
import math
import json
import argparse
import warnings
import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
from typing import Tuple, Any
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="gpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--grad-accum-steps", type=int, default=0,
                   help="Microbatches to accumulate before optimizer step (0=auto).")
    p.add_argument("--lr", type=float, default=1e-3, help="LR for model params")
    p.add_argument("--lr-prec", type=float, default=5e-3, help="LR for precision params")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--train-steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20,
                   help="Console print cadence in optimizer steps.")
    p.add_argument("--oracle-every", type=int, default=1,
                   help="Recompute oracle gradients every N amortiser steps")

    # Stability knobs
    p.add_argument("--alpha", type=float, default=0.1, help="Legacy weight (unused).")
    p.add_argument("--dt", type=float, default=0.0036, help="Integration step for PC dynamics")
    p.add_argument("--rollout-steps", type=int, default=32, help="Max PC inference iterations")
    p.add_argument("--rollout-tol", type=float, default=2e-4, help="Convergence tolerance")
    p.add_argument("--rollout-burnin", type=int, default=3,
                   help="Min steps before early-stop is allowed (adaptive: clamped to steps-2).")
    p.add_argument("--rollout-max-steps", type=int, default=0,
                   help="(Unused in Option B) compatibility only.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Global grad-norm clip")
    p.add_argument("--lambda-min", type=float, default=1e-3, help="(unused here)")
    p.add_argument("--lambda-max", type=float, default=10.0, help="(unused here)")

    p.add_argument("--pi-init", type=float, default=1.0, help="Softplus pre-activation init")
    p.add_argument("--pi-min", type=float, default=1e-3, help="Min precision value")
    p.add_argument("--pi-max", type=float, default=1e3, help="Max precision value")
    p.add_argument("--pi-logdet-weight", type=float, default=1.0,
                   help="Weight for -0.5*(logdet Pi_y + logdet Pi_x)")
    p.add_argument("--pi-normalise-logdet", action="store_true",
                   help="Divide logdet sums by dimension")

    p.add_argument("--nan-guard", action="store_true",
                   help="Skip parameter updates on NaN/Inf grads")
    p.add_argument("--metrics-json", type=str, default=None,
                   help="Optional path to write per-step training metrics as JSON")

    # Oracle blend cap
    p.add_argument("--lambda-cap", type=float, default=1,
                   help="Max fraction of oracle gradient in convex blend (0..1).")

    # -------- Difficulty controls --------
    p.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"], default="cifar10",
                   help="Choose dataset and difficulty.")
    p.add_argument("--augment", action="store_true", default=True,
                   help="Enable basic augmentation (flips/crops) for CIFAR train set.")
    p.add_argument("--input-noise-std", type=float, default=0.01,
                   help="Add N(0, std^2) Gaussian noise to training inputs.")
    p.add_argument("--input-noise-type", choices=["gaussian", "salt_pepper", "occlusion", "none"], default="gaussian",
                   help="Choose between Gaussian, salt-and-pepper, or occlusion input corruption. Set to none to disable.")
    p.add_argument("--salt-pepper-prob", type=float, default=0.0,
                   help="Total probability of replacing pixels with 0/1 when input-noise-type=salt_pepper.")
    p.add_argument("--occlusion-fraction", type=float, default=0.0,
                   help="Side length of the occlusion block as a fraction of the input height/width when input-noise-type=occlusion.")
    p.add_argument("--label-noise", type=float, default=0.0,
                   help="Fraction of training labels randomly corrupted.")

    return parse_known_args_and_fix(p)



def parse_known_args_and_fix(p):
    # Support both direct run and IPython where extra args may appear
    args, _ = p.parse_known_args()
    return args


args = parse_args()


# -------------------------
# Set JAX platform BEFORE import
# -------------------------
def set_jax_platform_env(device_choice: str):
    if device_choice == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device_choice == "gpu":
        prefer_rocm = any(k in os.environ for k in ("ROCM_PATH", "HIP_VISIBLE_DEVICES", "ROCM_HOME"))
        platform = "rocm" if prefer_rocm else "cuda"
        # Include cpu fallback to allow host callbacks when needed
        os.environ["JAX_PLATFORMS"] = f"{platform},cpu"
    else:
        os.environ.pop("JAX_PLATFORMS", None)


set_jax_platform_env(args.device)

# -------------------------
# JAX / Equinox / Optax
# -------------------------
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from jax.flatten_util import ravel_pytree
from path_integral import PathIntegral, path_integral
import jpc

init_activities_with_ffwd = jpc.init_activities_with_ffwd
solve_inference = jpc.solve_inference
neg_activity_grad = getattr(jpc, "neg_pc_activity_grad", None)
if neg_activity_grad is None:
    neg_activity_grad = getattr(jpc, "neg_activity_grad")

try:
    _solve_inference_params = inspect.signature(solve_inference).parameters
except (ValueError, TypeError):
    # fall back to empty mapping if the callable does not expose a signature
    _solve_inference_params = {}

_SOLVE_INFERENCE_ACCEPTS_VAR_KW = any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in _solve_inference_params.values()
)
_SOLVE_INFERENCE_SUPPORTS_MAX_STEPS = "max_steps" in _solve_inference_params
_SOLVE_INFERENCE_SUPPORTS_MAX_SOLVER_STEPS = "max_solver_steps" in _solve_inference_params
_SOLVE_INFERENCE_SUPPORTS_RETURN_SOLVER_STEPS = (
    "return_solver_steps" in _solve_inference_params or _SOLVE_INFERENCE_ACCEPTS_VAR_KW
)
_SOLVE_INFERENCE_WARNED_MISSING_COUNTS = False
_SOLVE_INFERENCE_WARNED_UNSUPPORTED_KWARGS = set()


def _warn_solver_kwarg_rejection(keyword: str, *, cap_value):
    if keyword not in _SOLVE_INFERENCE_WARNED_UNSUPPORTED_KWARGS:
        msg = "solve_inference() rejected keyword %r" % keyword
        if cap_value is not None:
            msg += "; requested solver cap %s may be ignored" % cap_value
        warnings.warn(msg, RuntimeWarning, stacklevel=4)
        _SOLVE_INFERENCE_WARNED_UNSUPPORTED_KWARGS.add(keyword)


def _call_solve_inference_with_adapters(base_kwargs: dict, max_steps: int):
    global _SOLVE_INFERENCE_SUPPORTS_MAX_STEPS
    global _SOLVE_INFERENCE_SUPPORTS_MAX_SOLVER_STEPS
    global _SOLVE_INFERENCE_SUPPORTS_RETURN_SOLVER_STEPS

    cap_candidates = []
    if _SOLVE_INFERENCE_SUPPORTS_MAX_STEPS:
        cap_candidates.append("max_steps")
    if _SOLVE_INFERENCE_SUPPORTS_MAX_SOLVER_STEPS:
        cap_candidates.append("max_solver_steps")
    for fallback_key in ("max_steps", "max_solver_steps", None):
        if fallback_key not in cap_candidates:
            cap_candidates.append(fallback_key)

    requested_return = _SOLVE_INFERENCE_SUPPORTS_RETURN_SOLVER_STEPS
    last_error = None

    for cap_key in cap_candidates:
        kwargs = dict(base_kwargs)
        if cap_key is not None and max_steps is not None:
            kwargs[cap_key] = max_steps

        while True:
            if requested_return:
                kwargs["return_solver_steps"] = True
            else:
                kwargs.pop("return_solver_steps", None)

            try:
                result = solve_inference(**kwargs)
            except TypeError as exc:  # pragma: no cover - exercised in user envs
                last_error = exc
                message = str(exc)

                if requested_return and "return_solver_steps" in message:
                    _warn_solver_kwarg_rejection("return_solver_steps", cap_value=max_steps)
                    requested_return = False
                    _SOLVE_INFERENCE_SUPPORTS_RETURN_SOLVER_STEPS = False
                    continue

                if cap_key is not None and cap_key in kwargs and f"'{cap_key}'" in message:
                    _warn_solver_kwarg_rejection(cap_key, cap_value=max_steps)
                    if cap_key == "max_steps":
                        _SOLVE_INFERENCE_SUPPORTS_MAX_STEPS = False
                    elif cap_key == "max_solver_steps":
                        _SOLVE_INFERENCE_SUPPORTS_MAX_SOLVER_STEPS = False
                    break

                raise
            else:
                if cap_key == "max_steps":
                    _SOLVE_INFERENCE_SUPPORTS_MAX_STEPS = True
                elif cap_key == "max_solver_steps":
                    _SOLVE_INFERENCE_SUPPORTS_MAX_SOLVER_STEPS = True
                _SOLVE_INFERENCE_SUPPORTS_RETURN_SOLVER_STEPS = requested_return
                return result, cap_key, requested_return

    if last_error is not None:
        raise last_error

    # Should be unreachable but defensive fallback to raw call.
    result = solve_inference(**base_kwargs)
    _SOLVE_INFERENCE_SUPPORTS_RETURN_SOLVER_STEPS = False
    return result, None, False

_SOLVE_INFERENCE_SUPPORTS_MAX_STEPS = "max_steps" in _solve_inference_params
_SOLVE_INFERENCE_SUPPORTS_MAX_SOLVER_STEPS = "max_solver_steps" in _solve_inference_params
from diffrax import (
    PIDController,
    Heun,
)

# -------------------------
# Torch for data I/O
# -------------------------
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -------------------------
# Data utilities (MNIST/CIFAR)
# -------------------------
def _make_noise_transform(noise_type: str, *,
                         gaussian_std: float,
                         salt_pepper_prob: float,
                         occlusion_fraction: float):
    if noise_type == "occlusion" and occlusion_fraction > 0:
        def add_occlusion(x: torch.Tensor) -> torch.Tensor:
            _, h, w = x.shape
            block_side = max(1, int(round(occlusion_fraction * min(h, w))))
            block_side = min(block_side, h, w)
            if block_side <= 0:
                return x
            top = torch.randint(0, h - block_side + 1, (1,)).item()
            left = torch.randint(0, w - block_side + 1, (1,)).item()
            x_noisy = x.clone()
            x_noisy[:, top:top + block_side, left:left + block_side] = 0.0
            return x_noisy

        return transforms.Lambda(add_occlusion)
    return None


def _cifar_transforms(train: bool, normalize=True, augment=False, noise_transform=None):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    t = []
    if train and augment:
        t += [transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip()]
    t += [transforms.ToTensor()]
    if normalize:
        t += [transforms.Normalize(mean=mean, std=std)]
    if noise_transform is not None:
        t.append(noise_transform)
    return transforms.Compose(t)

def _mnist_transforms(normalize=True, noise_transform=None):
    transforms_list = [transforms.ToTensor()]
    if normalize:
        transforms_list.append(transforms.Normalize(mean=(0.1307,), std=(0.3081,)))
    if noise_transform is not None:
        transforms_list.append(noise_transform)
    return transforms.Compose(transforms_list)

def get_loaders_and_dims(dataset: str, batch_size: int, *, pin_memory: bool, augment: bool, noise_transform=None):
    if dataset == "mnist":
        train = datasets.MNIST(str(DEFAULT_DATA_ROOT), train=True, download=True, transform=_mnist_transforms(True, noise_transform=noise_transform))
        test  = datasets.MNIST(str(DEFAULT_DATA_ROOT), train=False, download=True, transform=_mnist_transforms(True, noise_transform=noise_transform))
        input_dim, n_classes = 28 * 28, 10
        def _flatten(x): return torch.flatten(x, start_dim=1)  # (B, 784)
    elif dataset in ("cifar10", "cifar100"):
        is100 = (dataset == "cifar100")
        DS = datasets.CIFAR100 if is100 else datasets.CIFAR10
        train = DS(str(DEFAULT_DATA_ROOT), train=True, download=True,
                   transform=_cifar_transforms(True, True, augment, noise_transform))
        test  = DS(str(DEFAULT_DATA_ROOT), train=False, download=True,
                   transform=_cifar_transforms(False, True, False, noise_transform))
        input_dim, n_classes = 32 * 32 * 3, (100 if is100 else 10)

        def _flatten(x):
            # Keep spatial structure for the ConvNet in CHW layout expected by eqx Conv2d.
            return x  # (B, 3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    def _collate(batch):
        xs, ys = zip(*batch)  # ys are ints from dataset
        x = torch.stack(xs, dim=0)
        x = _flatten(x)
        y_int = torch.tensor(ys, dtype=torch.long)
        y_oh = torch.eye(n_classes, dtype=torch.float32)[y_int]
        return x, y_oh, y_int

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=pin_memory, collate_fn=_collate)
    test_loader  = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False,
                              pin_memory=pin_memory, collate_fn=_collate)
    return train_loader, test_loader, input_dim, n_classes

def to_jax(x, dtype=jnp.float32):
    return jnp.asarray(x, dtype=dtype)


# -------------------------
# Model + Precisions
# -------------------------
def _linear_batch(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
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
        h = self.hidden(x)
        return self.from_hidden(h)

    def hidden(self, x):
        return jax.nn.relu(_linear_batch(self.lin1, x))

    def from_hidden(self, h):
        return _linear_batch(self.lin2, h)

    def pc_layers(self):
        try:
            hidden_cls = _PCHiddenLayer  # type: ignore[name-defined]
            output_cls = _PCOutputLayer  # type: ignore[name-defined]
        except NameError:
            class _PCHiddenLayer(eqx.Module):  # type: ignore[no-redef]
                linear: eqx.nn.Linear

                def __call__(self, x):
                    return jax.nn.relu(_linear_batch(self.linear, x))

            class _PCOutputLayer(eqx.Module):  # type: ignore[no-redef]
                linear: eqx.nn.Linear

                def __call__(self, x):
                    return _linear_batch(self.linear, x)

            globals()["_PCHiddenLayer"] = _PCHiddenLayer
            globals()["_PCOutputLayer"] = _PCOutputLayer
            hidden_cls = _PCHiddenLayer
            output_cls = _PCOutputLayer

        return (hidden_cls(self.lin1), output_cls(self.lin2))


class CifarConvNet(eqx.Module):
    """Shared convolutional encoder/classifier for CIFAR-10 and CIFAR-100."""

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
        """Encode either a single image (C,H,W) or a batch (B,C,H,W)."""

        def _encode_single(img):
            h_single = jax.nn.relu(self.conv1(img))
            h_single = jax.nn.relu(self.conv2(h_single))
            h_single = self.pool1(h_single)
            h_single = jax.nn.relu(self.conv3(h_single))
            h_single = jax.nn.relu(self.conv4(h_single))
            h_single = self.pool2(h_single)
            h_single = jax.nn.relu(self.conv5(h_single))
            return h_single

        # Single image: (C, H, W) – used when JPC vmaps over pc_layers[0]
        if x.ndim == 3:
            # x: (C, H, W)
            h_single = _encode_single(x)  # (C', H', W')
            return jnp.mean(h_single, axis=(1, 2))  # -> (C',)

        # Batch of images: (B, C, H, W) – normal training path
        if x.ndim == 4:
            h = eqx.filter_vmap(_encode_single, in_axes=0, out_axes=0)(x)  # (B, C', H', W')
            return jnp.mean(h, axis=(2, 3))  # (B, C')

        # Anything else is a bug
        raise ValueError(
            f"CifarConvNet.hidden expected (C,H,W) or (B,C,H,W), got shape {x.shape}"
        )

        h = eqx.filter_vmap(_encode_single, in_axes=0, out_axes=0)(x)
        return jnp.mean(h, axis=(2, 3))  # Global average pool over spatial dims

    def from_hidden(self, h: jnp.ndarray) -> jnp.ndarray:
        return _linear_batch(self.classifier, h)

    def __call__(self, x):
        h = self.hidden(x)
        return self.from_hidden(h)


    def pc_layers(self):
        class _PCConvBackbone(eqx.Module):
            backbone: CifarConvNet

            def __call__(self, x):
                # Ensure inputs are in CHW (or BCHW) format before passing
                # them to the ConvNet backbone. JPC's init_activities_with_ffwd
                # may hand us per-sample tensors with different layouts.
                if x.ndim == 3:
                    # Either (C, H, W) or (H, W, C)
                    if x.shape[0] == 3:
                        # Already CHW
                        x_chw = x
                    elif x.shape[-1] == 3:
                        # HWC -> CHW
                        x_chw = jnp.transpose(x, (2, 0, 1))
                    else:
                        raise ValueError(
                            f"Expected 3-channel image with shape (C,H,W) or (H,W,C), got {x.shape}"
                        )
                elif x.ndim == 4:
                    # Either (B, C, H, W) or (B, H, W, C)
                    if x.shape[1] == 3:
                        x_chw = x
                    elif x.shape[-1] == 3:
                        # BHWC -> BCHW
                        x_chw = jnp.transpose(x, (0, 3, 1, 2))
                    else:
                        raise ValueError(
                            f"Expected 3-channel batch with shape (B,3,H,W) or (B,H,W,3), got {x.shape}"
                        )
                else:
                    raise ValueError(
                        f"Unsupported input rank {x.ndim} for CifarConvNet PC backbone; shape={x.shape}"
                    )

                return self.backbone.hidden(x_chw)

        class _PCConvClassifier(eqx.Module):
            classifier: eqx.nn.Linear

            def __call__(self, x):
                return _linear_batch(self.classifier, x)

        return (_PCConvBackbone(self), _PCConvClassifier(self.classifier))


class LearnedPrecisions(eqx.Module):
    rho_y: jnp.ndarray
    rho_x: jnp.ndarray

    def __init__(self, out_dim: int, hid_dim: int, init: float = 1.0, key=None):
        self.rho_y = jnp.ones((out_dim,)) * init
        self.rho_x = jnp.ones((hid_dim,)) * init

    def pi_vectors(self):
        pi_y = jax.nn.softplus(self.rho_y) + 1e-6
        pi_x = jax.nn.softplus(self.rho_x) + 1e-6
        pi_y = jnp.clip(pi_y, a_min=args.pi_min, a_max=args.pi_max)
        pi_x = jnp.clip(pi_x, a_min=args.pi_min, a_max=args.pi_max)
        return pi_y, pi_x

    def logdets(self, normalise: bool = False):
        pi_y, pi_x = self.pi_vectors()
        ld_y = jnp.sum(jnp.log(pi_y))
        ld_x = jnp.sum(jnp.log(pi_x))
        if normalise:
            ld_y = ld_y / pi_y.shape[0]
            ld_x = ld_x / pi_x.shape[0]
        return ld_y, ld_x


# -------------------------
# Free Energy
# -------------------------
def free_energy_components(model, lp: LearnedPrecisions, z, y_onehot,
                           *, logdet_weight: float, normalise_logdet: bool):
    mu, mu_dot = jnp.split(z, 2, axis=1)
    logits = model.from_hidden(mu)
    probs = jax.nn.softmax(logits, axis=-1)
    pi_y, pi_x = lp.pi_vectors()
    e_y = y_onehot - probs
    pred_err = 0.5 * jnp.mean(jnp.sum((e_y ** 2) * pi_y[None, :], axis=1))
    hid_mu = 0.5 * jnp.mean(jnp.sum((mu ** 2) * pi_x[None, :], axis=1))
    hid_vel = 0.5 * jnp.mean(jnp.sum((mu_dot ** 2) * pi_x[None, :], axis=1))
    hid_err = hid_mu + hid_vel
    ld_y, ld_x = lp.logdets(normalise=normalise_logdet)
    logdet_term = -0.5 * logdet_weight * (ld_y + ld_x)
    F = pred_err + hid_err + logdet_term
    return F, pred_err, hid_err, (ld_y, ld_x)


def trapezoid_integral(y, dt, steps_taken):
    """Integrate ``y`` over time; supports arrays and PyTrees of arrays."""

    def _integrate_array(arr):
        T = arr.shape[0]
        n_steps = jnp.minimum(jnp.asarray(steps_taken, jnp.int32), jnp.asarray(T, jnp.int32))

        def single_step(_):
            return arr[0] * dt

        def multi_step(n):
            idx = jnp.arange(T)
            last = jax.lax.dynamic_index_in_dim(arr, n - 1, keepdims=False)
            mask = (idx > 0) & (idx < (n - 1))
            interior_sum = jnp.sum(jnp.where(mask, arr, 0.0))
            return dt * (0.5 * (arr[0] + last) + interior_sum)

        return jax.lax.cond(n_steps <= 1, single_step, multi_step, n_steps)

    if eqx.is_array(y):
        return _integrate_array(y)

    try:
        return jax.tree_util.tree_map(_integrate_array, y)
    except Exception:
        return _integrate_array(jnp.asarray(y))


def _tree_sum(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sum(jnp.stack([jnp.sum(jnp.asarray(leaf)) for leaf in leaves]))


def _tree_sum(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sum(jnp.stack([jnp.sum(jnp.asarray(leaf)) for leaf in leaves]))


def _make_pc_layers(model):
    """Build lightweight wrappers so JPC sees the model as a generative model."""

    pc_layers = getattr(model, "pc_layers", None)
    if callable(pc_layers):
        return pc_layers()

    raise TypeError("Model does not expose pc_layers() for predictive coding")


def _solve_inference_with_cap(
    params,
    activities,
    output,
    *,
    input=None,
    loss_id="mse",
    param_type="sp",
    solver=None,
    max_t1=20,
    dt=None,
    stepsize_controller=None,
    weight_decay=0.0,
    spectral_penalty=0.0,
    activity_decay=0.0,
    record_iters=False,
    record_every=None,
    max_steps=16384,
):
    if solver is None:
        solver = Heun()
    if stepsize_controller is None:
        stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)

    max_steps = max(1, int(max_steps))

    solve_kwargs = dict(
        params=params,
        activities=activities,
        output=output,
        input=input,
        loss_id=loss_id,
        param_type=param_type,
        solver=solver,
        max_t1=max_t1,
        dt=dt,
        stepsize_controller=stepsize_controller,
        weight_decay=weight_decay,
        spectral_penalty=spectral_penalty,
        activity_decay=activity_decay,
        record_iters=record_iters,
        record_every=record_every,
    )

    result, _, requested_return = _call_solve_inference_with_adapters(
        solve_kwargs, max_steps
    )
    if isinstance(result, tuple):
        activities_iters = result[0]
        solver_payload = result[1] if len(result) > 1 else None
    else:
        activities_iters = result
        solver_payload = None

    stats = _coerce_solver_stats(
        solver_payload,
        default_steps=max_steps,
        requested_return=requested_return,
        return_solver_steps=True,
    )

    if _SOLVE_INFERENCE_SUPPORTS_MAX_STEPS:
        solve_kwargs["max_steps"] = max_steps
    elif _SOLVE_INFERENCE_SUPPORTS_MAX_SOLVER_STEPS:
        solve_kwargs["max_solver_steps"] = max_steps
    else:
        if max_steps is not None and max_steps != 16384:
            warnings.warn(
                "solve_inference() does not accept a solver step cap; "
                "requested max_steps=%s will be ignored" % max_steps,
                RuntimeWarning,
            )

    activities_iters, solver_steps = solve_inference(**solve_kwargs)

    if _SOLVE_INFERENCE_SUPPORTS_RETURN_SOLVER_STEPS:
        solve_kwargs["return_solver_steps"] = True

    result = solve_inference(**solve_kwargs)
    if isinstance(result, tuple):
        activities_iters = result[0]
        solver_payload = result[1] if len(result) > 1 else None
    else:
        activities_iters = result
        solver_payload = None

    stats = _coerce_solver_stats(
        solver_payload,
        default_steps=max_steps,
        requested_return=_SOLVE_INFERENCE_SUPPORTS_RETURN_SOLVER_STEPS,
    )
    return activities_iters, stats



# -------------------------
# Discrete predictive coding (Option B, scan-only)
# -------------------------
def _pc_euler_step(acts, params, output, input,
                   loss_id="ce", param_type="sp"):
    grad = neg_activity_grad(
        0.0, acts,
        (params, output, input, loss_id, param_type, 0.0, 0.0, 0.0, None, None)
    )
    return jax.tree_util.tree_map(lambda a, g: a + args.dt * g, acts, grad)


def _activities_delta_norm(old, new):
    diffs = [(n - o) for o, n in zip(jax.tree.leaves(old), jax.tree.leaves(new))]
    sq = [jnp.sum(d * d) for d in diffs] if diffs else []
    return jnp.sqrt(jnp.sum(jnp.stack(sq))) if sq else jnp.asarray(0.0, jnp.float32)


MAX_STEPS_VERSION = 4
"""Bump whenever ``max_steps`` semantics change (see :func:`_count_inference_steps`)."""


def _infer_max_solver_steps(user_cap: int, rollout_steps: int) -> int:
    """Choose a ceiling for the Diffrax integrator.

    ``solve_inference`` aborts once ``max_steps`` proposals have been evaluated.
    The optimal cap depends on how aggressively we integrate the predictive-
    coding dynamics. When the user provides ``--rollout-max-steps`` we trust
    that cap. Otherwise we scale the allowance with the requested number of
    rollout steps to avoid premature termination on longer trajectories.
    """

    user_cap = int(user_cap)
    if user_cap > 0:
        return max(1, user_cap)

    rollout_steps = max(1, int(rollout_steps))
    scaled_cap = 4096 * rollout_steps
    return max(16384, scaled_cap)


def simulate_inference(model, lp: LearnedPrecisions, x, y_onehot,
                       *, steps: int, dt: float, tolerance: float,
                       logdet_weight: float, normalise_logdet: bool,
                       max_solver_steps: int, key):
    """Discrete predictive-coding rollout with early-stop masking (scan-only, VJP-safe).

    - Uses *true* activity deltas for convergence (unmasked).
    - Freezes activities after convergence and zero-pads logged deltas.
    - Tracks steps actually executed (accepted/proposed = steps_taken).
    - Adds decay-based estimate of total PC iterations with adaptive burn-in and
      log-linear fallback when the ratio is ill-defined.
    """
    del key

    if max_solver_steps > 0:
        steps_eff = min(int(steps), int(max_solver_steps))
    else:
        steps_eff = int(steps)
    steps_eff = max(1, steps_eff)

    pc_layers = _make_pc_layers(model)

    # Initialise activities from feed-forward; clamp last activity to y
    init_acts = list(init_activities_with_ffwd(model=pc_layers, input=x))
    init_acts[-1] = y_onehot
    activities0 = tuple(init_acts)

    # Helper: compute free energy from activities and IL-ASGN cross term
    def _F_from_acts(acts):
        mu = acts[0]
        mu_dot = jnp.zeros_like(mu)
        z = jnp.concatenate([mu, mu_dot], axis=1)

        def _F_only(model_in, lp_in, mu_in):
            z_in = jnp.concatenate([mu_in, jnp.zeros_like(mu_in)], axis=1)
            F_val, _, _, _ = free_energy_components(
                model_in, lp_in, z_in, y_onehot,
                logdet_weight=logdet_weight, normalise_logdet=normalise_logdet
            )
            return F_val

        F, pred_err, hid_err, (ld_y, ld_x) = free_energy_components(
            model, lp, z, y_onehot,
            logdet_weight=logdet_weight, normalise_logdet=normalise_logdet
        )

        grad_mu_fn = lambda m, l: jax.grad(lambda mu_in: _F_only(m, l, mu_in))(mu)
        grad_mu, vjp_params = eqx.filter_vjp(grad_mu_fn, model, lp)
        mixed_model, mixed_lp = vjp_params(grad_mu)
        cross_term = (mixed_model, mixed_lp)

        return F, pred_err, hid_err, ld_y, ld_x, cross_term

    # Seed FE with the value at the initial acts
    F0, _, _, _, _, cross0 = _F_from_acts(activities0)

    tol = jnp.asarray(tolerance, jnp.float32)

    # ---- adaptive burn-in (can't stop before this many steps)
    burnin_eff = jnp.minimum(
        jnp.asarray(steps_eff - 1, jnp.int32),         # never exceed steps-1
        jnp.asarray(args.rollout_burnin, jnp.int32)    # user-configured
    )

    # Body: if not done -> one Euler step; else keep state frozen.
    def body(carry, i):
        acts, done, iters, F_prev, cross_prev = carry

        # Candidate next acts
        cand_acts = _pc_euler_step(acts, (pc_layers, None), y_onehot, x)
        F_cand, pred_err_c, hid_err_c, ld_y_c, ld_x_c, cross_cand = _F_from_acts(cand_acts)

        # ---- compute delta BEFORE any masking/freezing (true step size)
        delta_true = _activities_delta_norm(acts, cand_acts)

        # Convergence allowed only after burn-in
        allow_stop = (i + 1) >= burnin_eff
        just_done = (delta_true < tol) & (~done) & allow_stop
        done_next = done | just_done

        # Freeze acts after convergence, otherwise take the candidate
        acts_next = jax.tree_util.tree_map(
            lambda a_keep, a_new: jnp.where(done_next, a_keep, a_new), acts, cand_acts
        )

        # FE sequence: keep last F once converged; else use candidate F
        F_now = jnp.where(done_next, F_prev, F_cand)
        cross_now = jax.tree_util.tree_map(
            lambda prev, new: jnp.where(done_next, prev, new), cross_prev, cross_cand
        )

        # Count iterations performed (don't increment once done)
        iters_next = jnp.where(done, iters, iters + 1)

        # For logging, zero-pad the tail *after* convergence
        delta_masked = jnp.where(done_next, jnp.asarray(0.0, delta_true.dtype), delta_true)

        return (acts_next, done_next, iters_next, F_now, cross_now), (F_now, cross_now, delta_true, delta_masked)

    # Scan for at most ``steps_eff`` iterations; collect Fs and both delta streams
    (acts_final, done_final, steps_taken, F_last, cross_last), (Fs, cross_terms, deltas_true, deltas_masked) = jax.lax.scan(
        body,
        init=(activities0, jnp.asarray(False), jnp.asarray(0, jnp.int32), F0, cross0),
        xs=jnp.arange(steps_eff),
    )

    del done_final, F_last, cross_last  # not needed downstream

    # Terminal state already correct (frozen after done=True)
    mu_final = acts_final[0]
    mu_dot_final = jnp.zeros_like(mu_final)
    z_final = jnp.concatenate([mu_final, mu_dot_final], axis=1)

    # Final energies from terminal activities
    F_final, pred_err, hid_err, (ld_y, ld_x) = free_energy_components(
        model, lp, z_final, y_onehot,
        logdet_weight=logdet_weight, normalise_logdet=normalise_logdet
    )

    # Integrate IL-ASGN cross term up to the actual number of iterations
    ilasgn_path_components = path_integral.trapezoid_from_cross_terms(cross_terms, dt, steps_taken)
    ilasgn_path_tree = ilasgn_path_components.total
    ilasgn_path = _tree_sum(ilasgn_path_tree)

    # Discrete mode: accepted == proposed == iterations performed
    accepted_steps = jnp.asarray(steps_taken, jnp.int32)
    proposed_steps = accepted_steps

    # ---- Decay-based estimate of total steps (beyond acceptance) ----
    T = int(deltas_true.shape[0])
    W_const = min(8, T)
    n_eff = jnp.minimum(steps_taken, jnp.asarray(W_const, jnp.int32))
    start = jnp.maximum(0, steps_taken - jnp.asarray(W_const, jnp.int32))
    win_true = jax.lax.dynamic_slice_in_dim(deltas_true, start, W_const, axis=0)

    # Mask keeps only the first n_eff entries of the window
    idx_i32 = jnp.arange(W_const, dtype=jnp.int32)
    mask = (idx_i32 < n_eff).astype(jnp.float32)

    # Guard tiny/zero values to keep logs finite but do NOT use padded zeros
    y = jnp.maximum(win_true.astype(jnp.float32), 1e-12)
    logy = jnp.log(y)
    x = idx_i32.astype(jnp.float32)

    sum_m = jnp.sum(mask)
    sum_x = jnp.sum(x * mask)
    sum_y = jnp.sum(logy * mask)
    sum_xx = jnp.sum((x * x) * mask)
    sum_xy = jnp.sum((x * logy) * mask)

    den = sum_m * sum_xx - sum_x * sum_x
    valid_fit = (n_eff >= 3) & jnp.isfinite(den) & (jnp.abs(den) > 0)
    slope = jnp.where(valid_fit, (sum_m * sum_xy - sum_x * sum_y) / den, 0.0)

    # r_hat from the last two *true* deltas (need >=2)
    last_idx = jnp.maximum(1, n_eff) - 1
    d_last = jax.lax.dynamic_index_in_dim(win_true, last_idx, keepdims=False)
    d_prev = jax.lax.dynamic_index_in_dim(win_true, jnp.maximum(0, last_idx - 1), keepdims=False)

    # Prefer ratio, but fall back to exp(slope) when the ratio is ill-defined
    r_from_ratio = d_last / (d_prev + 1e-12)
    r_from_slope = jnp.exp(slope)  # because slope ~ d/dx log(delta) ≈ log(r)
    use_ratio = (n_eff >= 2) & jnp.isfinite(r_from_ratio) & (d_prev > 0)
    r_hat = jnp.where(use_ratio, r_from_ratio, r_from_slope)

    tol_f = jnp.asarray(tolerance, y.dtype)
    extra_float = jnp.log(tol_f / (d_last + 1e-12)) / jnp.log(r_hat + 1e-12)

    bad = (r_hat <= 0.0) | (jnp.abs(r_hat - 1.0) < 1e-3) | ~jnp.isfinite(extra_float)
    pi_est_extra = jnp.where((n_eff < 2) | bad, 0.0, jnp.clip(extra_float, 0.0, 1e6))
    pi_est_total = jnp.minimum(
        steps_taken + jnp.asarray(jnp.floor(pi_est_extra), jnp.int32),
        jnp.asarray(T, jnp.int32),
    )

    aux = {
        "Fs": Fs,
        "lams": jnp.full((1,), jnp.nan, dtype=z_final.dtype),
        # keep padded-for-logging deltas:
        "deltas": deltas_masked.astype(jnp.float32),

        "F_final": F_final,
        "ilasgn_path": ilasgn_path,
        "ilasgn_path_tree": ilasgn_path_tree,
        "pred_err": pred_err,
        "hid_err": hid_err,
        "logdet_y": ld_y,
        "logdet_x": ld_x,
        "steps_taken": steps_taken,
        "accepted_steps": accepted_steps,
        "proposed_steps": proposed_steps,

        # decay-fit diagnostics (computed from deltas_true window)
        "pi_decay_r": r_hat.astype(jnp.float32),
        "pi_fit_slope": slope.astype(jnp.float32),
        "pi_est_extra": pi_est_extra.astype(jnp.float32),
        "pi_est_total_f": pi_est_total.astype(jnp.float32),
        "pi_est_valid": valid_fit.astype(jnp.float32),
        "pi_delta_last": d_last.astype(jnp.float32),
        "pi_tol": tol_f.astype(jnp.float32),
        "pi_cap_hit": (steps_taken == jnp.asarray(steps_eff, jnp.int32)).astype(jnp.float32),
        "pi_fit_n_eff": n_eff.astype(jnp.float32),
    }
    return z_final, aux, steps_taken




# -------------------------
# InferenceStats container
# -------------------------
@dataclass
class InferenceStats(eqx.Module):
    accepted: jnp.ndarray
    total: jnp.ndarray

    @classmethod
    def from_arrays(cls, accepted, total):
        accepted_i32 = jnp.asarray(accepted, jnp.int32)
        total_i32 = jnp.asarray(total, jnp.int32)
        total_i32 = jnp.maximum(total_i32, accepted_i32)
        return cls(accepted=accepted_i32, total=total_i32)

    @staticmethod
    def _to_int32(value):
        if value is None:
            return jnp.asarray(0, jnp.int32)
        return jnp.asarray(value, jnp.int32)

    @classmethod
    def from_solver_steps(cls, steps) -> "InferenceStats":
        steps_i32 = cls._to_int32(steps)
        return cls.from_arrays(steps_i32, steps_i32)

    @classmethod
    def zeros(cls):
        z = jnp.asarray(0, jnp.int32)
        return cls(z, z)


def _warn_missing_solver_counts(reason: str):
    global _SOLVE_INFERENCE_WARNED_MISSING_COUNTS
    if not _SOLVE_INFERENCE_WARNED_MISSING_COUNTS:
        warnings.warn(
            "solve_inference() did not return solver step counts (%s); "
            "falling back to an approximate value" % reason,
            RuntimeWarning,
            stacklevel=3,
        )
        _SOLVE_INFERENCE_WARNED_MISSING_COUNTS = True


def _coerce_solver_stats(
    payload,
    *,
    default_steps,
    requested_return: bool,
    **_ignored_kwargs,
) -> "InferenceStats":
    """Best-effort conversion of ``solve_inference`` diagnostics to ``InferenceStats``."""

    if isinstance(payload, InferenceStats):
        return payload

    if isinstance(payload, dict):
        accepted = payload.get("accepted")
        accepted = payload.get("accepted_steps", accepted)
        accepted = payload.get("num_accepted_steps", accepted)
        total = payload.get("total")
        total = payload.get("proposed_steps", total)
        total = payload.get("proposed", total)
        total = payload.get("num_steps", total)
        if accepted is not None or total is not None:
            accepted_val = accepted if accepted is not None else total
            total_val = total if total is not None else accepted_val
            return InferenceStats.from_arrays(accepted_val, total_val)

    if isinstance(payload, (tuple, list)):
        if not payload:
            payload = None
        elif len(payload) == 1:
            payload = payload[0]
        else:
            accepted_val, total_val = payload[0], payload[-1]
            return InferenceStats.from_arrays(accepted_val, total_val)

    if payload is None:
        reason = "API without solver counts" if not requested_return else "no payload returned"
        fallback = None if default_steps is None else jnp.asarray(default_steps, jnp.int32)
        _warn_missing_solver_counts(reason)
        if fallback is None:
            fallback = jnp.asarray(0, jnp.int32)
        return InferenceStats.from_solver_steps(fallback)

    try:
        return InferenceStats.from_solver_steps(payload)
    except Exception:
        try:
            leaves = jax.tree_util.tree_leaves(payload)
        except Exception:
            leaves = ()
        if leaves:
            if len(leaves) == 1:
                return InferenceStats.from_solver_steps(leaves[0])
            accepted_val, total_val = leaves[0], leaves[-1]
            return InferenceStats.from_arrays(accepted_val, total_val)

    fallback = None if default_steps is None else jnp.asarray(default_steps, jnp.int32)
    _warn_missing_solver_counts(f"unrecognised payload type {type(payload)!r}")
    if fallback is None:
        fallback = jnp.asarray(0, jnp.int32)
    return InferenceStats.from_solver_steps(fallback)


# -------------------------
# Oracle objective + grads
# -------------------------
def oracle_objective(model, lp, x, y_onehot, *, steps, dt, tolerance,
                     logdet_weight, normalise_logdet, max_solver_steps, key):
    _, aux, _ = simulate_inference(model, lp, x, y_onehot,
                                   steps=steps, dt=dt, tolerance=tolerance,
                                   logdet_weight=logdet_weight,
                                   normalise_logdet=normalise_logdet,
                                   max_solver_steps=max_solver_steps, key=key)
    return aux["F_final"] - aux["ilasgn_path"], aux


oracle_value_and_grads_model = eqx.filter_value_and_grad(
    lambda m, lp, x, y, *, steps, dt, tolerance, logdet_weight,
    normalise_logdet, max_solver_steps, key:
        oracle_objective(m, lp, x, y, steps=steps, dt=dt, tolerance=tolerance,
                         logdet_weight=logdet_weight,
                         normalise_logdet=normalise_logdet,
                         max_solver_steps=max_solver_steps, key=key),
    has_aux=True
)

oracle_value_and_grads_lp = eqx.filter_value_and_grad(
    lambda lp, m, x, y, *, steps, dt, tolerance, logdet_weight,
    normalise_logdet, max_solver_steps, key:
        oracle_objective(m, lp, x, y, steps=steps, dt=dt, tolerance=tolerance,
                         logdet_weight=logdet_weight,
                         normalise_logdet=normalise_logdet,
                         max_solver_steps=max_solver_steps, key=key),
    has_aux=True
)


# -------------------------
# Task loss / eval
# -------------------------
def task_loss(model, x, y_int):
    logits = model(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y_int).mean()


task_value_and_grads = eqx.filter_value_and_grad(task_loss, has_aux=False)

accuracy = eqx.filter_jit(
    lambda model, x, y_int: jnp.mean(
        (jnp.argmax(model(x), axis=-1) == y_int).astype(jnp.float32)
    )
)


# -------------------------
# Oracle cache
# -------------------------
@dataclass
class OracleCache(eqx.Module):
    grads_model: Any
    grads_lp: Any
    steps_since: jnp.ndarray


# -------------------------
# Main training loop
# -------------------------
def main():
    warnings.simplefilter("ignore")
    torch.manual_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    metrics_history = [] if args.metrics_json else None

    def _maybe_scalar(value):
        if value is None:
            return None
        try:
            leaves = jax.tree_util.tree_leaves(value)
        except Exception:
            leaves = None
        if leaves is not None and len(leaves) > 1:
            try:
                value = _tree_sum(value)
            except Exception:
                pass
        try:
            arr = np.asarray(value)
        except Exception:
            return None
        if arr.size == 0:
            return None
        try:
            scalar = float(arr.reshape(-1)[0])
        except Exception:
            return None
        if math.isnan(scalar):
            return None
        return scalar

    # Device setup
    def _safe_devices(kind: str):
        try:
            return jax.devices(kind)
        except Exception:
            return []

    gpu_devices = _safe_devices("cuda") or _safe_devices("rocm") or _safe_devices("gpu")
    cpu_devices = _safe_devices("cpu")

    if args.device == "gpu":
        if not gpu_devices:
            raise RuntimeError("No CUDA/ROCm backend found.")
        device = gpu_devices[0]
    elif args.device == "cpu":
        if not cpu_devices:
            raise RuntimeError("No CPU backend available.")
        device = cpu_devices[0]
    else:
        device = gpu_devices[0] if gpu_devices else cpu_devices[0]

    print(f"JAX backend     : {jax.default_backend()}")
    print(f"JAX devices     : {jax.devices()}")
    print(f"Selected DEVICE : {device}")

    using_gpu = getattr(device, "platform", "") in ("cuda", "gpu", "rocm")
    log_every = max(1, int(args.log_every))
    eval_every = max(1, int(args.eval_every))

    auto_grad_accum_note = None
    if args.grad_accum_steps <= 0:
        grad_accum_steps = 2 if using_gpu else 1

        if using_gpu:
            if args.dataset == "mnist":
                approx_width = int(args.width)
            else:
                approx_width = max(16, min(int(args.width), 256)) * 4
            approx_batch = int(args.batch_size)
            approx_rollout = max(1, int(args.rollout_steps))

            estimated_bytes = float(approx_batch) * float(approx_width) * float(approx_width)
            estimated_bytes *= float(approx_rollout) * 4.0

            device_mem_limit = None
            try:
                mem_stats_fn = getattr(device, "memory_stats", None)
                if callable(mem_stats_fn):
                    stats = mem_stats_fn()
                    device_mem_limit = float(stats.get("bytes_limit")) if stats else None
            except Exception:
                device_mem_limit = None

            should_reduce = False
            if device_mem_limit:
                threshold = 0.35 * device_mem_limit
                if estimated_bytes >= threshold:
                    should_reduce = True
                    auto_grad_accum_note = (
                        "auto-selected no accumulation to stay below ~35% of GPU memory"
                    )
            if not should_reduce:
                if approx_width >= 768 or approx_batch >= 192:
                    should_reduce = True
                    auto_grad_accum_note = "auto-selected no accumulation for large width/batch"

            if should_reduce:
                grad_accum_steps = 1

        else:
            auto_grad_accum_note = "auto-selected no accumulation on non-GPU backend"
    else:
        grad_accum_steps = max(1, int(args.grad_accum_steps))

    effective_batch_size = args.batch_size * grad_accum_steps
    if grad_accum_steps > 1:
        msg = (f"Using gradient accumulation over {grad_accum_steps} microbatches "
               f"(effective batch size {effective_batch_size}).")
        if auto_grad_accum_note:
            msg += f" {auto_grad_accum_note}."
        print(msg)
    else:
        msg = f"Using batch size {args.batch_size} with no accumulation."
        if auto_grad_accum_note:
            msg += f" ({auto_grad_accum_note})"
        print(msg)

    # Data
    print("Loading dataset...")
    noise_transform = _make_noise_transform(
        args.input_noise_type,
        gaussian_std=args.input_noise_std,
        salt_pepper_prob=args.salt_pepper_prob,
        occlusion_fraction=args.occlusion_fraction,
    )
    train_loader, test_loader, INPUT_DIM, N_CLASSES = get_loaders_and_dims(
        args.dataset,
        args.batch_size,
        pin_memory=using_gpu,
        augment=args.augment,
        noise_transform=noise_transform,
    )
    print("Loading Complete")

    steps_per_epoch = 0
    try:
        steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    except Exception:
        steps_per_epoch = 0

    # Model + LP + Optimisers
    key, k_model = jax.random.split(key)
    if args.dataset == "mnist":
        model = MLP(INPUT_DIM, int(args.width), N_CLASSES, key=k_model)
    else:
        conv_base = max(16, min(int(args.width), 256))
        model = CifarConvNet(N_CLASSES, conv_base, key=k_model)
    lp = LearnedPrecisions(N_CLASSES, model.hidden_dim, init=args.pi_init)

    model_optim = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(args.lr)
    )
    lp_optim = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adam(args.lr_prec)
    )

    model_opt_state = model_optim.init(eqx.filter(model, eqx.is_array))
    lp_opt_state = lp_optim.init(eqx.filter(lp, eqx.is_array))

    zero_model_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                              eqx.filter(model, eqx.is_array))
    zero_lp_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x),
                                           eqx.filter(lp, eqx.is_array))
    oracle_every = max(1, int(args.oracle_every))
    oracle_cache = OracleCache(zero_model_grads, zero_lp_grads,
                               jnp.asarray(oracle_every, jnp.int32))

    # Helper
    def _tree_isfinite(tree):
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_array(leaf)]
        if not leaves:
            return jnp.array(True)
        checks = [jnp.all(jnp.isfinite(x)) for x in leaves]
        return jnp.all(jnp.stack(checks))

    def update_step(model, lp, model_opt_state, lp_opt_state, oracle_cache,
                    x_np, y_oh_np, y_int_np, key):

        # ---- Convert inputs to JAX ----
        x = to_jax(x_np, jnp.float32)
        y_oh = to_jax(y_oh_np, jnp.float32)
        y_int = to_jax(y_int_np, jnp.int32)

        # Oracle recomputation frequency
        freq = jnp.maximum(jnp.asarray(oracle_every, jnp.int32),
                           jnp.asarray(1, jnp.int32))

        # ----- ORACLE CACHE: recompute or reuse -----
        def _recompute(cache):
            (_, aux_model), grads_model = oracle_value_and_grads_model(
                model, lp, x, y_oh,
                steps=args.rollout_steps, dt=args.dt, tolerance=args.rollout_tol,
                logdet_weight=args.pi_logdet_weight,
                normalise_logdet=args.pi_normalise_logdet,
                max_solver_steps=args.rollout_max_steps, key=key
            )
            (_, aux_lp), grads_lp = oracle_value_and_grads_lp(
                lp, model, x, y_oh,
                steps=args.rollout_steps, dt=args.dt, tolerance=args.rollout_tol,
                logdet_weight=args.pi_logdet_weight,
                normalise_logdet=args.pi_normalise_logdet,
                max_solver_steps=args.rollout_max_steps, key=key
            )

            stats_model = InferenceStats.from_arrays(aux_model["accepted_steps"],
                                                     aux_model["proposed_steps"])
            stats_lp = InferenceStats.from_arrays(aux_lp["accepted_steps"],
                                                  aux_lp["proposed_steps"])

            new_cache = OracleCache(grads_model, grads_lp, jnp.asarray(1, jnp.int32))
            return grads_model, grads_lp, new_cache, stats_model, stats_lp

        def _reuse(cache):
            new_steps = jnp.minimum(cache.steps_since + 1, freq)
            new_cache = OracleCache(cache.grads_model, cache.grads_lp, new_steps)
            stats_zero = InferenceStats.zeros()
            return cache.grads_model, cache.grads_lp, new_cache, stats_zero, stats_zero

        grads_oracle_model, grads_oracle_lp, oracle_cache_candidate, stats_model, stats_lp = \
            jax.lax.cond(oracle_cache.steps_since >= freq, _recompute, _reuse, oracle_cache)

        # ----- TASK GRADS -----
        L_task, grads_task_model = task_value_and_grads(model, x, y_int)

        # ----- GATE -----
        gt_vec, _ = ravel_pytree(grads_task_model)
        go_vec, _ = ravel_pytree(grads_oracle_model)

        cos_sim = jnp.dot(gt_vec, go_vec) / (jnp.linalg.norm(gt_vec) *
                                             jnp.linalg.norm(go_vec) + 1e-8)
        lambda_raw = (cos_sim + 1) * 0.5
        lambda_gate = jnp.clip(lambda_raw, 0.0, float(args.lambda_cap))

        # Norm-match oracle grads
        scale = jnp.linalg.norm(gt_vec) / (jnp.linalg.norm(go_vec) + 1e-8)
        grads_oracle_scaled = jax.tree_map(lambda g: g * scale, grads_oracle_model)

        # Blend grads
        blended_model_grads = jax.tree_map(
            lambda gt, go: (1 - lambda_gate) * gt + lambda_gate * go,
            grads_task_model, grads_oracle_scaled
        )

        # ----- NAN GUARD -----
        if args.nan_guard:
            ok = _tree_isfinite(blended_model_grads) & _tree_isfinite(grads_oracle_lp)
        else:
            ok = jnp.array(True)

        # ---- APPLY UPDATES (masked ONLY on array leaves) ----

        def mask_update_leaf(g):
            if not isinstance(g, jnp.ndarray):
                return g  # Non-array leaf unchanged
            return jnp.where(ok, g, jnp.zeros_like(g))

        # Compute raw optax updates
        model_updates, model_opt_state_new = model_optim.update(
            blended_model_grads, model_opt_state, eqx.filter(model, eqx.is_array)
        )
        lp_updates, lp_opt_state_new = lp_optim.update(
            grads_oracle_lp, lp_opt_state, eqx.filter(lp, eqx.is_array)
        )

        # Mask array leaves only
        model_updates_masked = jax.tree_map(mask_update_leaf, model_updates)
        lp_updates_masked = jax.tree_map(mask_update_leaf, lp_updates)

        # Apply masked updates
        model_new = eqx.apply_updates(model, model_updates_masked)
        lp_new = eqx.apply_updates(lp, lp_updates_masked)

        # ----- SIMULATE INFERENCE FOR METRICS -----
        z_final, aux, steps_taken = simulate_inference(
            model_new, lp_new, x, y_oh,
            steps=args.rollout_steps, dt=args.dt, tolerance=args.rollout_tol,
            logdet_weight=args.pi_logdet_weight,
            normalise_logdet=args.pi_normalise_logdet,
            max_solver_steps=args.rollout_max_steps, key=key
        )

        mu_final, _ = jnp.split(z_final, 2, axis=1)
        pc_activity = 0.5 * jnp.mean(jnp.sum(mu_final ** 2, axis=1))
        pc_per_unit = pc_activity / mu_final.shape[1]

        grad_diff = gt_vec - go_vec
        sg_objective = 0.5 * jnp.mean(grad_diff ** 2)
        sg_relmse = jnp.mean(grad_diff ** 2) / (jnp.mean(go_vec ** 2) + 1e-8)

        _, pi_x = lp_new.pi_vectors()
        mean_log_pi_g = jnp.mean(jnp.log(pi_x))

        metrics_stats = InferenceStats.from_arrays(aux["accepted_steps"],
                                                   aux["proposed_steps"])

        max_steps = jnp.max(jnp.stack([stats_model.accepted,
                                       stats_lp.accepted,
                                       metrics_stats.accepted]))
        max_steps_total = jnp.max(jnp.stack([stats_model.total,
                                             stats_lp.total,
                                             metrics_stats.total]))

        dof = z_final.shape[1]
        vfe_per_dof = aux["F_final"] / jnp.maximum(1.0, dof)

        metrics = {
            "F_final": aux["F_final"],
            "ilasgn_path": aux["ilasgn_path"],
            "pred_err": aux["pred_err"],
            "hid_err": aux["hid_err"],
            "logdet_y": aux["logdet_y"],
            "logdet_x": aux["logdet_x"],

            "L_task": L_task,
            "CE_loss": L_task,
            "lambda_gate": lambda_gate,
            "lambda_gate_raw": lambda_raw,
            "sg_objective": sg_objective,
            "sg_relmse": sg_relmse,
            "sg_cosine": cos_sim,

            "mean_log_pi_g": mean_log_pi_g,
            "pc_activity_obj": pc_activity,
            "pc_activity_per_unit": pc_per_unit,
            "vfe_per_dof": vfe_per_dof,

            # PI diagnostics
            "pi_decay_r": aux["pi_decay_r"],
            "pi_fit_slope": aux["pi_fit_slope"],
            "pi_est_extra": aux["pi_est_extra"],
            "pi_est_total_f": aux["pi_est_total_f"],
            "pi_est_valid": aux["pi_est_valid"],
            "pi_delta_last": aux["pi_delta_last"],
            "pi_tol": aux["pi_tol"],
            "pi_cap_hit": aux["pi_cap_hit"],
            "pi_fit_n_eff": aux["pi_fit_n_eff"],

            "oracle_model_steps": stats_model.accepted.astype(jnp.float32),
            "oracle_model_steps_total": stats_model.total.astype(jnp.float32),
            "oracle_lp_steps": stats_lp.accepted.astype(jnp.float32),
            "oracle_lp_steps_total": stats_lp.total.astype(jnp.float32),
            "metrics_steps": metrics_stats.accepted.astype(jnp.float32),
            "metrics_steps_total": metrics_stats.total.astype(jnp.float32),
            "pc_inference_steps": metrics_stats.total.astype(jnp.float32),

            "max_steps": max_steps.astype(jnp.float32),
            "max_steps_total": max_steps_total.astype(jnp.float32),
            "max_steps_version": jnp.asarray(float(MAX_STEPS_VERSION), jnp.float32),

            "nan_guard_ok": ok.astype(jnp.float32),
        }

        # ----- UPDATE ORACLE CACHE (only if ok == True) -----
        def mask_cache(new, old):
            return jax.tree_map(lambda n, o: jnp.where(ok, n, o), new, old)

        oracle_cache_new = OracleCache(
            mask_cache(oracle_cache_candidate.grads_model, oracle_cache.grads_model),
            mask_cache(oracle_cache_candidate.grads_lp, oracle_cache.grads_lp),
            jnp.where(ok, oracle_cache_candidate.steps_since, oracle_cache.steps_since)
        )

        return model_new, lp_new, model_opt_state_new, lp_opt_state_new, oracle_cache_new, metrics

    update_step = eqx.filter_jit(update_step)

    # ---- Simple evaluation helpers (host side) ----
    def _eval_batch_train(model_now, xb_np, yint_np):
        xj = to_jax(xb_np, jnp.float32)
        yj = to_jax(yint_np, jnp.int32)
        loss = float(jax.device_get(task_loss(model_now, xj, yj)))
        acc = float(jax.device_get(accuracy(model_now, xj, yj)))
        return loss, acc

    def _eval_full_test(model_now):
        total = 0
        sum_loss = 0.0
        sum_acc = 0.0
        for xb_t, yb_oh_t, yb_int_t in test_loader:
            xj = to_jax(xb_t.numpy(), jnp.float32)
            yj = to_jax(yb_int_t.numpy(), jnp.int32)
            bs = int(yj.shape[0])
            l = float(jax.device_get(task_loss(model_now, xj, yj)))
            a = float(jax.device_get(accuracy(model_now, xj, yj)))
            sum_loss += l * bs
            sum_acc += a * bs
            total += bs
        if total == 0:
            return float('nan'), float('nan')
        return sum_loss / total, sum_acc / total

    # Warm-up (compile)
    if args.dataset == "mnist":
        dummy_x_shape = (effective_batch_size, INPUT_DIM)
    else:
        # CIFAR loaders keep spatial structure for the ConvNet (C, H, W).
        dummy_x_shape = (effective_batch_size, 3, 32, 32)

    dummy_x  = np.zeros(dummy_x_shape, np.float32)
    dummy_oh = np.zeros((effective_batch_size, N_CLASSES),  np.float32)
    dummy_y  = np.zeros((effective_batch_size,),            np.int32)

    # Prefer real microbatches so the compiler sees true shapes/order.
    try:
        warmup_xs, warmup_yohs, warmup_yints = [], [], []
        warmup_iter = iter(train_loader)
        for _ in range(max(1, grad_accum_steps)):
            xb_t, yb_oh_t, yb_int_t = next(warmup_iter)
            warmup_xs.append(xb_t.numpy())
            warmup_yohs.append(yb_oh_t.numpy())
            warmup_yints.append(yb_int_t.numpy())

        if warmup_xs:
            dummy_x = np.concatenate(warmup_xs, axis=0)
            dummy_oh = np.concatenate(warmup_yohs, axis=0)
            dummy_y = np.concatenate(warmup_yints, axis=0)
    except StopIteration:
        pass
    except Exception:
        # If anything goes wrong, fall back to zero tensors.
        dummy_x  = np.zeros(dummy_x_shape, np.float32)
        dummy_oh = np.zeros((effective_batch_size, N_CLASSES),  np.float32)
        dummy_y  = np.zeros((effective_batch_size,),            np.int32)

    if args.dataset != "mnist":
        # Ensure the ConvNet sees CHW tensors with a batch axis.
        if dummy_x.ndim == 3:
            dummy_x = np.expand_dims(dummy_x, axis=0)
        if dummy_x.ndim == 4 and dummy_x.shape[1] != 3 and dummy_x.shape[-1] == 3:
            dummy_x = np.transpose(dummy_x, (0, 3, 1, 2))
        if dummy_x.ndim != 4 or dummy_x.shape[1] != 3:
            dummy_x = np.zeros(dummy_x_shape, np.float32)

    key, k0 = jax.random.split(key)
    print("About to run first update_step…")
    _ = jax.device_get(update_step(model, lp, model_opt_state, lp_opt_state, oracle_cache,
                                   dummy_x, dummy_oh, dummy_y, k0))
    print("First update_step done.")

    # Training loop with gradient accumulation
    last_vfe_per_dof = float('nan')
    step = 0
    accum_x, accum_yoh, accum_yint = [], [], []

    def flush_accum(force: bool = False) -> bool:
        nonlocal model, lp, model_opt_state, lp_opt_state, oracle_cache
        nonlocal key, step, last_vfe_per_dof

        if not accum_x:
            return False
        if not force and len(accum_x) < grad_accum_steps:
            return False

        xb = np.concatenate(accum_x, axis=0)
        yb_oh = np.concatenate(accum_yoh, axis=0)
        yb_int = np.concatenate(accum_yint, axis=0)

        accum_x.clear()
        accum_yoh.clear()
        accum_yint.clear()

        # ---------- training-time perturbations ----------
        if args.input_noise_type == "gaussian" and args.input_noise_std > 0:
            xb = xb + np.random.normal(0.0, args.input_noise_std, size=xb.shape).astype(np.float32)
        elif args.input_noise_type == "salt_pepper" and args.salt_pepper_prob > 0:
            noise_mask = np.random.rand(*xb.shape)
            salt = noise_mask < (args.salt_pepper_prob / 2.0)
            pepper = (noise_mask >= (args.salt_pepper_prob / 2.0)) & (noise_mask < args.salt_pepper_prob)
            xb = xb.copy()
            xb[salt] = 1.0
            xb[pepper] = 0.0

        if args.label_noise > 0:
            C = yb_oh.shape[1]
            mask = np.random.rand(yb_int.shape[0]) < float(args.label_noise)
            n_flip = int(mask.sum())
            if n_flip > 0:
                rand_labels = np.random.randint(0, C, size=n_flip, dtype=yb_int.dtype)
                yb_int_noisy = yb_int.copy()
                yb_int_noisy[mask] = rand_labels
                yb_oh_noisy = np.eye(C, dtype=np.float32)[yb_int_noisy]
                yb_int, yb_oh = yb_int_noisy, yb_oh_noisy
        # --------------------------------------------------

        key, k1 = jax.random.split(key)
        model, lp, model_opt_state, lp_opt_state, oracle_cache, metrics = update_step(
            model, lp, model_opt_state, lp_opt_state, oracle_cache, xb, yb_oh, yb_int, k1
        )

        step += 1

        # ---- VERBOSE LOGGING (restored; authoritative counters) ----
        m = jax.device_get(metrics)
        metrics_host = m

        def _f(x):
            scalar = _maybe_scalar(x)
            if scalar is None:
                return float('nan')
            return float(scalar)

        # Degrees of freedom for vfe_per_dof (2*WIDTH since z=[mu, mu_dot])
        dof = 2 * model.hidden_dim
        VFE_final = _f(m["F_final"])
        vfe_per_dof = VFE_final / jnp.maximum(1.0, float(dof))
        if math.isfinite(vfe_per_dof) and math.isfinite(last_vfe_per_dof):
            d_vfe_per_dof = vfe_per_dof - last_vfe_per_dof
        else:
            d_vfe_per_dof = 0.0
        if math.isfinite(vfe_per_dof):
            last_vfe_per_dof = vfe_per_dof

        # Batch train metrics each step
        train_loss_batch, train_acc_batch = _eval_batch_train(model, xb, yb_int)

        # Full test metrics on cadence
        test_loss_full = None
        test_acc_full = None
        do_eval = (step % eval_every) == 0
        if do_eval:
            test_loss_full, test_acc_full = _eval_full_test(model)

        il_path_total = _f(m.get("ilasgn_path_total"))
        il_path_model = _f(m.get("ilasgn_path_model_total"))
        il_path_lp = _f(m.get("ilasgn_path_lp_total"))

        log_msg = (
            f"[Step {step:5d}] "
            f"CE_loss={_f(m['CE_loss']):.4f} | "
            f"VFE_final={VFE_final:.4f} (IL_path={_f(m['ilasgn_path']):.4f}) | "
            f"VFE_per_dof={vfe_per_dof:.4f} Delta_VFE_per_dof={d_vfe_per_dof:+.4f} | "
            f"VFE_pred_term={_f(m['pred_err']):.4f} VFE_hidden_term={_f(m['hid_err']):.4f} "
            f"logdet_Pi_y_sum={_f(m['logdet_y']):.2f} logdet_Pi_x_sum={_f(m['logdet_x']):.2f} | "
            f"lambda_gate={_f(m['lambda_gate']):.3f} (raw={_f(m['lambda_gate_raw']):.3f}) | "
            f"SG_objective={_f(m['sg_objective']):.4f} | "
            f"SG_relMSE={_f(m['sg_relmse']):.4f} SG_cosine={_f(m['sg_cosine']):.3f} "
            f"mean_log_Pi_g={_f(m['mean_log_pi_g']):.3f} | "
            f"PC_activity_obj={_f(m['pc_activity_obj']):.4f} "
            f"(per_unit {_f(m['pc_activity_per_unit']):.4e}) | "
            # PI_max_steps reflects *executed* iterations this step
            f"PI_max_steps={_f(m['max_steps']):.0f}"
        )

        # Decay-estimate summary; 'total_proposals' derived from max_steps_total
        log_msg += (
            f" | PI_est_total={_f(m['pi_est_total_f']):.1f} "
            f"PI_est_valid={int(round(_f(m['pi_est_valid'])))} "
            f"(cap_hit={int(round(_f(m['pi_cap_hit'])))} "
            f"delta_last={_f(m['pi_delta_last']):.2e} "
            f"tol={_f(m['pi_tol']):.2e} "
            f"r={_f(m['pi_decay_r']):.3f} "
            f"slope={_f(m['pi_fit_slope']):.3e}) | "
            f"TrainAcc={train_acc_batch:.3f} TrainLoss={train_loss_batch:.4f}"
        )

        postfix = []
        if "max_steps_total" in m:
            # Keep this aligned with scan length; avoids stale mismatches
            postfix.append(f"total_proposals={_f(m['max_steps_total']):.0f}")
        if "oracle_model_steps" in m and "oracle_model_steps_total" in m:
            postfix.append(f"oracle_model={_f(m['oracle_model_steps']):.0f}/{_f(m['oracle_model_steps_total']):.0f}")
        if "oracle_lp_steps" in m and "oracle_lp_steps_total" in m:
            postfix.append(f"oracle_lp={_f(m['oracle_lp_steps']):.0f}/{_f(m['oracle_lp_steps_total']):.0f}")
        if "metrics_steps" in m and "metrics_steps_total" in m:
            postfix.append(f"metrics={_f(m['metrics_steps']):.0f}/{_f(m['metrics_steps_total']):.0f}")
        if postfix:
            log_msg += " (" + ", ".join(postfix) + ")"

        if do_eval:
            log_msg += f" | TestAcc={test_acc_full:.3f} TestLoss={test_loss_full:.4f}"

        log_now = force or (step % log_every) == 0
        if log_now:
            print(log_msg)

        # Optional JSON record
        if metrics_history is not None:
            version_scalar = _maybe_scalar(metrics_host.get("max_steps_version"))
            record = {
                "step": int(step),
                "F_final": _maybe_scalar(metrics_host.get("F_final")),
                "ilasgn_path": _maybe_scalar(metrics_host.get("ilasgn_path")),
                "CE_loss": _maybe_scalar(metrics_host.get("CE_loss")),
                "oracle_model_steps": _maybe_scalar(metrics_host.get("oracle_model_steps")),
                "oracle_model_steps_total": _maybe_scalar(metrics_host.get("oracle_model_steps_total")),
                "oracle_lp_steps": _maybe_scalar(metrics_host.get("oracle_lp_steps")),
                "oracle_lp_steps_total": _maybe_scalar(metrics_host.get("oracle_lp_steps_total")),
                "metrics_steps": _maybe_scalar(metrics_host.get("metrics_steps")),
                "metrics_steps_total": _maybe_scalar(metrics_host.get("metrics_steps_total")),
                "pc_inference_steps": _maybe_scalar(metrics_host.get("pc_inference_steps")),
                "max_steps": _maybe_scalar(metrics_host.get("max_steps")),
                "max_steps_total": _maybe_scalar(metrics_host.get("max_steps_total")),
                "max_steps_version": int(version_scalar) if version_scalar is not None else None,
                # Convenience fields for cross-method comparisons
                "train_loss": _maybe_scalar(train_loss_batch),
                "train_accuracy": _maybe_scalar(train_acc_batch),
                "test_loss": _maybe_scalar(test_loss_full),
                "test_accuracy": _maybe_scalar(test_acc_full),
                "epoch": int(math.ceil(step / max(1, steps_per_epoch))) if steps_per_epoch > 0 else None,
            }
            metrics_history.append(record)

        need_eval = bool(do_eval)
        if log_now or need_eval:

            def _get(key_name):
                value = _maybe_scalar(metrics_host.get(key_name))
                return value if value is not None else float('nan')

            mCE = _get("CE_loss")
            mF = _get("F_final")
            mIL = _get("ilasgn_path")
            mVPD = _get("vfe_per_dof")
            mPE = _get("pred_err")
            mHE = _get("hid_err")
            mLY = _get("logdet_y")
            mLX = _get("logdet_x")
            mLG = _get("lambda_gate")
            mSGO = _get("sg_objective")
            mSGRM = _get("sg_relmse")
            mSGC = _get("sg_cosine")
            mMPI = _get("mean_log_pi_g")
            mPCA = _get("pc_activity_obj")
            mPCU = _get("pc_activity_per_unit")
            mPCI = _get("pc_inference_steps")
            mOMS = _get("oracle_model_steps")
            mOMT = _get("oracle_model_steps_total")
            mOLS = _get("oracle_lp_steps")
            mOLT = _get("oracle_lp_steps_total")
            mMS = _get("metrics_steps")
            mMST = _get("metrics_steps_total")
            mMAX = _get("max_steps")
            mMAXT = _get("max_steps_total")

            if math.isfinite(mVPD) and math.isfinite(last_vfe_per_dof):
                mDVPD = mVPD - last_vfe_per_dof
            else:
                mDVPD = 0.0
            if math.isfinite(mVPD):
                last_vfe_per_dof = mVPD

            test_loss = float('nan')
            test_acc = float('nan')
            if need_eval:
                test_loss_sum, test_acc_sum, n = 0.0, 0.0, 0
                for exb_t, eyb_oh_t, eyb_int_t in test_loader:
                    exb = to_jax(exb_t.numpy())
                    eyl = to_jax(eyb_int_t.numpy(), jnp.int32)
                    eloss = task_loss(model, exb, eyl)
                    eacc = accuracy(model, exb, eyl)
                    eloss, eacc = jax.device_get((eloss, eacc))
                    b = exb_t.shape[0]
                    test_loss_sum += float(eloss) * b
                    test_acc_sum += float(eacc) * b
                    n += b
                if n:
                    test_loss = test_loss_sum / n
                    test_acc = test_acc_sum / n

            log_msg = (
                f"[Step {step:5d}] CE_loss={mCE:.4f} | VFE_final={mF:.4f} (IL_path={mIL:.4f}) | "
                f"VFE_per_dof={mVPD:.4f} Delta_VFE_per_dof={mDVPD:+.4f} | "
                f"VFE_pred_term={mPE:.4f} VFE_hidden_term={mHE:.4f} "
                f"logdet_Pi_y_sum={mLY:.2f} logdet_Pi_x_sum={mLX:.2f} | "
                f"lambda_gate={mLG:.3f} | SG_objective={mSGO:.4f} | "
                f"SG_relMSE={mSGRM:.4f} SG_cosine={mSGC:.3f} mean_log_Pi_g={mMPI:.3f} | "
                f"PC_activity_obj={mPCA:.4f} (per_unit {mPCU:.4e}) | "
                f"PI_max_steps={mMAX:.0f}"
            )

            if mPCI is not None and math.isfinite(mPCI):
                log_msg += f" | PC_steps={mPCI:.0f}"

            details = []
            if mMAXT is not None and math.isfinite(mMAXT):
                details.append(f"total_proposals={mMAXT:.0f}")
            if mOMS is not None and math.isfinite(mOMS):
                if mOMT is not None and math.isfinite(mOMT):
                    details.append(f"oracle_model={mOMS:.0f}/{mOMT:.0f}")
                else:
                    details.append(f"oracle_model={mOMS:.0f}")
            if mOLS is not None and math.isfinite(mOLS):
                if mOLT is not None and math.isfinite(mOLT):
                    details.append(f"oracle_lp={mOLS:.0f}/{mOLT:.0f}")
                else:
                    details.append(f"oracle_lp={mOLS:.0f}")
            if mMS is not None and math.isfinite(mMS):
                if mMST is not None and math.isfinite(mMST):
                    details.append(f"metrics={mMS:.0f}/{mMST:.0f}")
                else:
                    details.append(f"metrics={mMS:.0f}")

            if details:
                log_msg += " (" + ", ".join(details) + ")"

            if mMAXT is not None and math.isfinite(mMAXT):
                log_msg += f" (total_proposals={mMAXT:.0f})"

            if need_eval:
                log_msg += f" | test_loss={test_loss:.4f} test_acc={test_acc:.4f}"

            print(log_msg)


        return step >= args.train_steps

    # Train
    while step < args.train_steps:
        for xb_t, yb_oh_t, yb_int_t in train_loader:
            accum_x.append(xb_t.numpy())
            accum_yoh.append(yb_oh_t.numpy())
            accum_yint.append(yb_int_t.numpy())
            if flush_accum():
                break
        if step >= args.train_steps:
            break
        if flush_accum(force=True):
            break

    print("Training complete.")

    # Persist metrics JSON if requested
    if metrics_history is not None and args.metrics_json:
        out_path = Path(args.metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "seed": args.seed,
                "dataset": args.dataset,
                "augment": bool(args.augment),
                "input_noise_type": args.input_noise_type,
                "input_noise_std": float(args.input_noise_std),
                "salt_pepper_prob": float(args.salt_pepper_prob),
                "occlusion_fraction": float(args.occlusion_fraction),
                "label_noise": float(args.label_noise),
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * (2 if using_gpu and args.grad_accum_steps <= 0 else max(1, args.grad_accum_steps)),
                "grad_accum_steps": (2 if using_gpu and args.grad_accum_steps <= 0 else max(1, args.grad_accum_steps)),
                "train_steps": args.train_steps,
                "eval_every": eval_every,
                "log_every": log_every,
                "oracle_every": args.oracle_every,
                "rollout_steps": args.rollout_steps,
                "rollout_burnin": args.rollout_burnin,
                "rollout_tol": args.rollout_tol,
                "dt": args.dt,
                "alpha": args.alpha,
                "lr": args.lr,
                "lr_prec": args.lr_prec,
                "lambda_cap": args.lambda_cap,
                "input_dim": INPUT_DIM,
                "n_classes": N_CLASSES,
                "width": args.width,
            },
            "records": metrics_history,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics history to {out_path}")


if __name__ == "__main__":
    main()