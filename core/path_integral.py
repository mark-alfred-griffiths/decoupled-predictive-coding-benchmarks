r"""Numerical path-integral utilities for the IL-ASGN research prototype.

This module implements the dynamical-systems perspective outlined in the
accompanying research proposal (see Sections 3.3–3.5).  In that document the
free action is introduced as the time-integral of the Lagrangian where the
Lagrangian corresponds to the variational free energy augmented with control
terms.  The :class:`PathIntegral` helper below mirrors the theoretical
construction:

* :meth:`PathIntegral.compute_path_integral` performs discrete trapezoidal
  quadrature that approximates the free action :math:`\mathcal{F}_\text{path}`
  over either scalars or PyTree-valued Lagrangian traces.

* :meth:`PathIntegral.trapezoid_from_cross_terms` integrates the per-parameter
  IL-ASGN mixed Hessian action :math:`\partial^2 F / (\partial \theta\,\partial \mu)
  \cdot \nabla_\mu F` at each Euler step to provide a structured, per-parameter
  path integral.  This mirrors the theoretical prescription where the
  contribution of :math:`\theta` is mediated through the evolving hidden state
  :math:`\mu(t)` rather than via the free energy values themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import dtypes as jax_dtypes


@dataclass
class ThetaPathIntegral:
    """Decomposition of the :math:`\theta`-aligned path integral contributions."""

    cross_action: Any
    direct_grad: Any
    total: Any


class PathIntegral:
    """Evaluate free-action style path integrals on scalars or PyTrees.

    Parameters
    ----------
    beta:
        Optional scaling factor applied to the integrated quantity.  This allows
        experiments to anneal the regularisation strength without altering the
        implementation.
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def _trapezoid_single(self, values: jnp.ndarray, dt: float, steps_taken: jnp.ndarray):
        """Composite trapezoidal rule for a single leaf array."""
        # Convert possible float0 (integer tangents) to a usable floating dtype
        # before combining with ``dt`` inside differentiation contexts.  Guard
        # ``result_type`` because it cannot accept float0.
        values_dtype = getattr(values, "dtype", None)
        float0_like = False
        if values_dtype is not None:
            names = getattr(values_dtype, "names", None)
            float0_like = values_dtype == jax_dtypes.float0 or (
                names is not None and "float0" in names
            )

        if float0_like:
            values = jnp.zeros_like(values, dtype=jnp.float32)
        else:
            values = jnp.asarray(values, dtype=jnp.result_type(values, jnp.float32))

        dt = jnp.asarray(dt, dtype=values.dtype)

        T = values.shape[0]
        n = jnp.minimum(jnp.asarray(steps_taken, jnp.int32), jnp.asarray(T, jnp.int32))

        def single_step(_):
            return values[0] * dt

        def multi_step(m: jnp.ndarray):
            idx = jnp.arange(T)
            last = jax.lax.dynamic_index_in_dim(values, m - 1, keepdims=False)
            # ``values`` may have arbitrary trailing dimensions (e.g. batches and
            # features).  Construct a mask that only gates the time axis while
            # broadcasting cleanly across the remaining dimensions.
            mask = (idx > 0) & (idx < (m - 1))
            mask = jnp.reshape(mask, (T,) + (1,) * (values.ndim - 1))
            interior_sum = jnp.sum(jnp.where(mask, values, 0.0), axis=0)
            return dt * (0.5 * (values[0] + last) + interior_sum)

        return jax.lax.cond(n <= 1, single_step, multi_step, n)

    def compute_path_integral(self, lagrangian_values: Any, dt: float, steps_taken: jnp.ndarray):
        """Compute a trapezoidal path integral for scalars or PyTrees.

        ``lagrangian_values`` may be a single array with leading dimension
        ``time`` or a PyTree whose leaves share that leading dimension.  The
        integration is applied leafwise, preserving the tree structure while
        truncating to ``steps_taken`` entries.
        """

        if not jax.tree_util.tree_leaves(lagrangian_values):
            return jnp.asarray(0.0)

        return jax.tree_map(lambda v: self._trapezoid_single(v, dt, steps_taken), lagrangian_values)

    def trapezoid_from_cross_terms(
        self, cross_terms: Any, dt: float, steps_taken: jnp.ndarray, theta_grad_final: Any | None = None
    ) -> ThetaPathIntegral:
        """Integrate mixed Hessian actions and optionally add the terminal :math:`\nabla_\theta F`."""

        path_tree = self.compute_path_integral(cross_terms, dt, steps_taken)
        cross_action = jax.tree_map(lambda leaf: self.beta * leaf, path_tree)

        if theta_grad_final is None or not jax.tree_util.tree_leaves(theta_grad_final):
            direct_grad = jax.tree_map(jnp.zeros_like, cross_action)
        else:
            direct_grad = theta_grad_final

        total = jax.tree_map(lambda cross, grad: cross + grad, cross_action, direct_grad)
        return ThetaPathIntegral(cross_action=cross_action, direct_grad=direct_grad, total=total)


# Convenience instance mirroring legacy module-level usage
path_integral = PathIntegral()
