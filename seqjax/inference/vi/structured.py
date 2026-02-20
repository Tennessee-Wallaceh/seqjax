"""Structured variational approximations for latent trajectories.

This module implements a local Gaussian family with a block-banded Cholesky
factor of the covariance, driven by :class:`~seqjax.inference.vi.api.LatentContext`.
"""

from __future__ import annotations

from typing import Type

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.nn import softplus
from jaxtyping import Array, Float, PRNGKeyArray

import seqjax.model.typing as seqjtyping
from .api import AmortizedVariationalApproximation, LatentContext


_LOG_2PI = jnp.log(2.0 * jnp.pi)


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
    """Block-banded Cholesky covariance Gaussian approximation.

    The covariance is represented by a block lower bi-diagonal Cholesky factor
    ``B`` with ``Σ = B Bᵀ``. The diagonal blocks ``B[t,t]`` are produced from
    the corresponding hidden input ``h_t``, while sub-diagonal blocks
    ``B[t,t-1]`` are produced from adjacent inputs ``(h_{t-1}, h_t)``.
    """

    target_struct_cls: Type[LatentT]
    batch_length: int
    buffer_length: int
    context_dim: int
    parameter_dim: int
    condition_dim: int
    x_dim: int
    chol_diagonal_net: eqx.nn.MLP
    chol_subdiagonal_net: eqx.nn.MLP
    mean_net: eqx.nn.MLP

    def __init__(
        self,
        target_struct_cls: Type[LatentT],
        *,
        batch_length: int,
        buffer_length: int,
        context_dim: int,
        parameter_dim: int,
        condition_dim: int,
        hidden_dim: int,
        depth: int,
        key: PRNGKeyArray,
    ) -> None:
        sample_length = 2 * buffer_length + batch_length
        self.x_dim = target_struct_cls.flat_dim
        super().__init__(
            target_struct_cls,
            (sample_length, self.x_dim),
            batch_length,
            buffer_length,
        )
        self.target_struct_cls = target_struct_cls
        self.batch_length = batch_length
        self.buffer_length = buffer_length
        self.context_dim = context_dim
        self.parameter_dim = parameter_dim
        self.condition_dim = condition_dim

        diag_key, subdiag_key, mean_key = jrandom.split(key, 3)
        in_dim = context_dim + parameter_dim + condition_dim
        self.chol_diagonal_net = eqx.nn.MLP(
            in_size=in_dim,
            out_size=self.x_dim * self.x_dim,
            width_size=hidden_dim,
            depth=depth,
            key=diag_key,
        )
        self.chol_subdiagonal_net = eqx.nn.MLP(
            in_size=2 * in_dim,
            out_size=self.x_dim * self.x_dim,
            width_size=hidden_dim,
            depth=depth,
            key=subdiag_key,
        )
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
        theta = jnp.broadcast_to(theta_vec, (seq_ctx.shape[0], self.parameter_dim))
        cond_ctx = jnp.broadcast_to(cond_vec, (seq_ctx.shape[0], self.condition_dim))
        return jnp.concatenate([seq_ctx, theta, cond_ctx], axis=-1)

    def _covariance_cholesky_blocks(
        self,
        condition: LatentContext,
    ) -> tuple[
        Float[Array, "sample_length x_dim"],
        Float[Array, "sample_length x_dim x_dim"],
        Float[Array, "sample_length_minus_1 x_dim x_dim"],
    ]:
        local_inputs = self._make_inputs(condition)

        mean = jax.vmap(self.mean_net)(local_inputs)

        diag_raw = jax.vmap(self.chol_diagonal_net)(local_inputs).reshape(
            local_inputs.shape[0], self.x_dim, self.x_dim
        )
        diag_blocks = _lower_with_positive_diagonal(diag_raw)

        pair_inputs = jnp.concatenate([local_inputs[:-1], local_inputs[1:]], axis=-1)
        subdiag_blocks = jax.vmap(self.chol_subdiagonal_net)(pair_inputs).reshape(
            pair_inputs.shape[0], self.x_dim, self.x_dim
        )
        return mean, diag_blocks, subdiag_blocks

    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: LatentContext,
    ) -> tuple[LatentT, Float[Array, ""]]:
        mean, diag_blocks, subdiag_blocks = self._covariance_cholesky_blocks(condition)

        eps = jrandom.normal(key, (self.shape[0], self.x_dim))

        first = diag_blocks[0] @ eps[0]

        def sample_step(carry: Float[Array, "x_dim"], items):
            diag_block, subdiag_block, noise = items
            x_t = subdiag_block @ carry + diag_block @ noise
            return noise, x_t

        _, tail = jax.lax.scan(
            sample_step,
            eps[0],
            (diag_blocks[1:], subdiag_blocks, eps[1:]),
        )
        centered = jnp.concatenate([first[None, :], tail], axis=0)
        x_path = centered + mean

        residual = x_path - mean
        y0 = jax.scipy.linalg.solve_triangular(diag_blocks[0], residual[0], lower=True)

        def solve_step(carry: Float[Array, "x_dim"], items):
            diag_block, subdiag_block, r_t = items
            rhs = r_t - subdiag_block @ carry
            y_t = jax.scipy.linalg.solve_triangular(diag_block, rhs, lower=True)
            return y_t, y_t

        _, y_tail = jax.lax.scan(
            solve_step,
            y0,
            (diag_blocks[1:], subdiag_blocks, residual[1:]),
        )
        whitened = jnp.concatenate([y0[None, :], y_tail], axis=0)

        log_det_b = jnp.sum(jnp.log(jnp.diagonal(diag_blocks, axis1=-2, axis2=-1)))
        total_dim = self.shape[0] * self.x_dim
        quadratic = jnp.sum(whitened**2)
        log_q = -0.5 * (total_dim * _LOG_2PI + 2.0 * log_det_b + quadratic)

        return self.target_struct_cls.unravel(x_path), log_q