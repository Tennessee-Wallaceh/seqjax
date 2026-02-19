"""Structured variational approximations for latent trajectories.

This module sketches a block tri-diagonal precision Gaussian family that uses
local sequence context from :class:`~seqjax.inference.vi.api.LatentContext`.
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


def _unpack_precision_params(
    params: Float[Array, "sample_length width"],
    x_dim: int,
) -> tuple[Float[Array, "sample_length x_dim"], Float[Array, "sample_length x_dim x_dim"], Float[Array, "sample_length_minus_1 x_dim x_dim"]]:
    """Split MLP outputs into mean, diagonal Cholesky, and off-diagonal blocks."""
    diag_size = x_dim * x_dim
    off_size = x_dim * x_dim

    mean = params[:, :x_dim]
    diag_flat = params[:, x_dim : x_dim + diag_size]
    off_flat = params[:-1, x_dim + diag_size : x_dim + diag_size + off_size]

    diag_raw = diag_flat.reshape(params.shape[0], x_dim, x_dim)
    # lower-triangular block with positive diagonal to ensure SPD precision.
    diag_lower = jnp.tril(diag_raw)
    diag_idx = jnp.diag_indices(x_dim)
    diag_lower = diag_lower.at[:, diag_idx[0], diag_idx[1]].set(
        softplus(diag_lower[:, diag_idx[0], diag_idx[1]]) + 1e-4
    )

    off_blocks = off_flat.reshape(params.shape[0] - 1, x_dim, x_dim)
    return mean, diag_lower, off_blocks


class StructuredPrecisionGaussian[
    LatentT: seqjtyping.Latent,
](AmortizedVariationalApproximation[LatentT]):
    """Block tri-diagonal precision Gaussian approximation.

    The approximation is parameterised through a block bi-diagonal Cholesky
    factor ``L`` of the precision matrix ``Q = L Lᵀ``. Local context from
    ``LatentContext.sequence_embedded_context`` drives per-time-step blocks.
    """

    target_struct_cls: Type[LatentT]
    batch_length: int
    buffer_length: int
    context_dim: int
    parameter_dim: int
    condition_dim: int
    x_dim: int
    trunk: eqx.nn.MLP

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
        out_size = self.x_dim + 2 * self.x_dim * self.x_dim
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
        self.trunk = eqx.nn.MLP(
            in_size=context_dim + parameter_dim + condition_dim,
            out_size=out_size,
            width_size=hidden_dim,
            depth=depth,
            key=key,
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

    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: LatentContext,
    ) -> tuple[LatentT, Float[Array, ""]]:
        local_inputs = self._make_inputs(condition)
        params = jax.vmap(self.trunk)(local_inputs)
        mean, diag_blocks, off_blocks = _unpack_precision_params(params, self.x_dim)

        eps = jrandom.normal(key, (self.shape[0], self.x_dim))

        def solve_step(carry: Float[Array, "x_dim"], items):
            diag_block, off_block, noise = items
            rhs = noise - off_block @ carry
            x = jax.scipy.linalg.solve_triangular(
                diag_block.T,
                rhs,
                lower=False,
            )
            return x, x

        first = jax.scipy.linalg.solve_triangular(
            diag_blocks[0].T,
            eps[0],
            lower=False,
        )
        _, tail = jax.lax.scan(
            solve_step,
            first,
            (diag_blocks[1:], off_blocks, eps[1:]),
        )
        centered = jnp.concatenate([first[None, :], tail], axis=0)
        x_path = centered + mean

        log_det_q = 2.0 * jnp.sum(jnp.log(jnp.diagonal(diag_blocks, axis1=-2, axis2=-1)))
        quadratic = jnp.sum(eps**2)
        total_dim = self.shape[0] * self.x_dim
        log_q = 0.5 * (log_det_q - total_dim * _LOG_2PI - quadratic)

        return self.target_struct_cls.unravel(x_path), log_q
