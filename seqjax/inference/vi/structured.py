"""Structured variational approximations for latent trajectories.

This module implements a local Gaussian family with a block-banded Cholesky
factor of the precision, driven by :class:`~seqjax.inference.vi.api.LatentContext`.
"""

from __future__ import annotations

from typing import Any, Type, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.nn import softplus
from jaxtyping import Array, Float, PRNGKeyArray

import seqjax.model.typing as seqjtyping
from .interface import AmortizedVariationalApproximation, LatentContextDims, LatentContext


_LOG_2PI = jnp.log(2.0 * jnp.pi)

def solve_lower_bidiagonal_transpose(
    diag_blocks: Float[Array, "sample_length x_dim x_dim"],
    subdiag_blocks: Float[Array, "sample_length_minus_1 x_dim x_dim"],
    eps: Float[Array, "sample_length x_dim"],
) -> Float[Array, "sample_length x_dim"]:
    """Solve L.T @ residual = eps for residual.

    ``L`` is block lower-bidiagonal with diagonal blocks ``diag_blocks[t]`` and
    subdiagonal blocks ``subdiag_blocks[t - 1]``:

        L[t, t]     = diag_blocks[t]
        L[t, t - 1] = subdiag_blocks[t - 1],  t >= 1

    Therefore ``L.T`` is block upper-bidiagonal.

    Args:
        diag_blocks:
            Array of shape ``(T, x_dim, x_dim)``. Each block should be lower
            triangular with positive diagonal.

        subdiag_blocks:
            Array of shape ``(T - 1, x_dim, x_dim)``. The block
            ``subdiag_blocks[t - 1]`` is ``L[t, t - 1]``.

        eps:
            Array of shape ``(T, x_dim)``.

    Returns:
        residual:
            Array of shape ``(T, x_dim)`` satisfying ``L.T @ residual = eps``.
    """
    last = jax.scipy.linalg.solve_triangular(
        diag_blocks[-1].T,
        eps[-1],
        lower=False,
    )

    def step(
        carry: Float[Array, "x_dim"],
        items,
    ):
        residual_next = carry
        diag_block, subdiag_block_next, eps_t = items

        rhs = eps_t - subdiag_block_next.T @ residual_next

        residual_t = jax.scipy.linalg.solve_triangular(
            diag_block.T,
            rhs,
            lower=False,
        )

        return residual_t, residual_t

    _, residual_reversed_without_last = jax.lax.scan(
        step,
        last,
        (
            diag_blocks[:-1][::-1],
            subdiag_blocks[::-1],
            eps[:-1][::-1],
        ),
    )

    residual = jnp.concatenate(
        [
            residual_reversed_without_last[::-1],
            last[None, :],
        ],
        axis=0,
    )

    return residual


def _lower_with_positive_diagonal(
    raw_block: Float[Array, "... x_dim x_dim"],
) -> Float[Array, "... x_dim x_dim"]:
    """Project raw blocks to lower triangular form with positive diagonals."""
    lower = jnp.tril(raw_block)
    diag_idx = jnp.diag_indices(raw_block.shape[-1])
    lower = lower.at[..., diag_idx[0], diag_idx[1]].set(
        softplus(lower[..., diag_idx[0], diag_idx[1]]) + 1e-4
    )
    return lower


class StructuredPrecisionGaussian[
    LatentT: seqjtyping.Latent,
](AmortizedVariationalApproximation[LatentT]):
    """Block-banded precision-Cholesky Gaussian approximation.

    The precision matrix is represented as ``Σ⁻¹ = L Lᵀ``, where ``L`` is a
    block lower-bidiagonal Cholesky factor. The diagonal blocks ``L[t, t]`` are
    produced from the corresponding hidden input ``h_t``, while sub-diagonal
    blocks ``L[t, t - 1]`` are produced from adjacent inputs ``(h_{t-1}, h_t)``.

    Sampling uses ``x = μ + L^{-T} ε`` with ``ε ~ N(0, I)``. Since ``Lᵀ`` is
    block upper-bidiagonal, this triangular solve proceeds backwards in time.
    """
    target_struct_cls: Type[LatentT]
    latent_context_dims: LatentContextDims
    x_dim: int

    temporal_structure: Literal["mean_field", "bidiagonal"] = eqx.field(static=True)
    within_time_structure: Literal["full", "diagonal"] = eqx.field(static=True)
    chol_subdiagonal_net: Optional[eqx.nn.MLP]
    chol_diagonal_net: eqx.nn.MLP
    mean_net: eqx.nn.MLP

    def __init__(
        self,
        target_struct_cls: Type[LatentT],
        *,
        sample_length: int,
        latent_context_dims: LatentContextDims,
        hidden_dim: int,
        depth: int,
        key: PRNGKeyArray,
        temporal_structure: Literal["mean_field", "bidiagonal"] = "bidiagonal",
        within_time_structure: Literal["full", "diagonal"] = "full",
    ) -> None:
        self.temporal_structure = temporal_structure
        self.within_time_structure = within_time_structure

        self.x_dim = target_struct_cls.flat_dim
        super().__init__(
            target_struct_cls,
            shape=(sample_length, self.x_dim),
            sample_length=sample_length,
        )

        self.target_struct_cls = target_struct_cls
        self.latent_context_dims = latent_context_dims

        diag_key, subdiag_key, mean_key = jrandom.split(key, 3)

        in_dim = (
            self.latent_context_dims.sequence_embedded_context_dim 
            + self.latent_context_dims.parameter_context_dim 
            + self.latent_context_dims.condition_context_dim
        )

        if within_time_structure == "full":
            diag_out_size = self.x_dim * self.x_dim
        elif within_time_structure == "diagonal":
            diag_out_size = self.x_dim
        else:
            raise ValueError(f"Unknown within_time_structure: {within_time_structure}")

        self.chol_diagonal_net = eqx.nn.MLP(
            in_size=in_dim,
            out_size=diag_out_size,
            width_size=hidden_dim,
            depth=depth,
            key=diag_key,
        )

        if temporal_structure == "bidiagonal":
            self.chol_subdiagonal_net = eqx.nn.MLP(
                in_size=2 * in_dim,
                out_size=self.x_dim * self.x_dim,
                width_size=hidden_dim,
                depth=depth,
                key=subdiag_key,
            )
        elif temporal_structure == "mean_field":
            self.chol_subdiagonal_net = None
        else:
            raise ValueError(f"Unknown temporal_structure: {temporal_structure}")

        self.mean_net = eqx.nn.MLP(
            in_size=in_dim,
            out_size=self.x_dim,
            width_size=hidden_dim,
            depth=depth,
            key=mean_key,
        )

    def _make_inputs(
        self,
        condition: LatentContext,
    ) -> Float[Array, "sample_length in_dim"]:
        seq_ctx = condition.sequence_embedded_context
        theta_vec = condition.parameter_context.ravel().reshape(-1)
        cond_vec = condition.condition_context.ravel().reshape(-1)
        theta = jnp.broadcast_to(
            theta_vec, 
            (seq_ctx.shape[0], self.latent_context_dims.parameter_context_dim)
        )
        cond_ctx = jnp.broadcast_to(
            cond_vec, 
            (seq_ctx.shape[0], self.latent_context_dims.condition_context_dim)
        )
        return jnp.concatenate([seq_ctx, theta, cond_ctx], axis=-1)

    def _precision_cholesky_blocks(
        self,
        condition: LatentContext,
    ) -> tuple[
        Float[Array, "sample_length x_dim"],
        Float[Array, "sample_length x_dim x_dim"],
        Float[Array, "sample_length_minus_1 x_dim x_dim"],
    ]:
        local_inputs = self._make_inputs(condition)

        mean = jax.vmap(self.mean_net)(local_inputs)

        diag_raw = jax.vmap(self.chol_diagonal_net)(local_inputs)

        if self.within_time_structure == "full":
            diag_raw = diag_raw.reshape(
                local_inputs.shape[0],
                self.x_dim,
                self.x_dim,
            )
            diag_blocks = _lower_with_positive_diagonal(diag_raw)

        elif self.within_time_structure == "diagonal":
            diag = softplus(diag_raw) + 1e-4
            diag_blocks = jax.vmap(jnp.diag)(diag)

        else:
            raise ValueError(
                f"Unknown within_time_structure: {self.within_time_structure}"
            )

        if self.temporal_structure == "mean_field":
            subdiag_blocks = jnp.zeros(
                (local_inputs.shape[0] - 1, self.x_dim, self.x_dim)
            )

        elif self.temporal_structure == "bidiagonal":
            if self.chol_subdiagonal_net is None:
                raise ValueError(
                    "chol_subdiagonal_net is None but temporal_structure='bidiagonal'"
                )

            pair_inputs = jnp.concatenate(
                [local_inputs[:-1], local_inputs[1:]],
                axis=-1,
            )
            subdiag_blocks = jax.vmap(self.chol_subdiagonal_net)(pair_inputs).reshape(
                pair_inputs.shape[0],
                self.x_dim,
                self.x_dim,
            )

        else:
            raise ValueError(f"Unknown temporal_structure: {self.temporal_structure}")

        return mean, diag_blocks, subdiag_blocks


    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: LatentContext,
        state: Any = None,
    ) -> tuple[LatentT, Float[Array, ""], Any]:
        mean, diag_blocks, subdiag_blocks = self._precision_cholesky_blocks(condition)
        eps = jrandom.normal(key, (self.shape[0], self.x_dim))

        residual = solve_lower_bidiagonal_transpose(
            diag_blocks,
            subdiag_blocks,
            eps,
        )

        x_path = mean + residual

        log_det_l = jnp.sum(
            jnp.log(jnp.diagonal(diag_blocks, axis1=-2, axis2=-1))
        )

        total_dim = self.shape[0] * self.x_dim
        quadratic = jnp.sum(eps**2)

        log_q = (
            -0.5 * total_dim * _LOG_2PI
            + log_det_l
            - 0.5 * quadratic
        )
        sample = self.target_struct_cls.unravel(x_path)
        return sample, log_q, state
