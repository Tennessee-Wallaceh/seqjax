from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
import optax
import pytest

import seqjax.model.typing as seqjtyping
from seqjax.inference.vi.autoregressive import AmortizedMultivariateAutoregressor
from seqjax.inference.vi.conv_nf import AmortizedConvCoupling
from seqjax.inference.vi.interface import LatentContext
from seqjax.inference.vi.maf import AmortizedMAF
from seqjax.inference.vi.structured import StructuredPrecisionGaussian


class _Latent2D(seqjtyping.Latent):
    x: jnp.ndarray
    _shape_template = OrderedDict(x=jax.ShapeDtypeStruct((2,), jnp.float32))


class _Latent5D(seqjtyping.Latent):
    x: jnp.ndarray
    _shape_template = OrderedDict(x=jax.ShapeDtypeStruct((5,), jnp.float32))


class _Latent1D(seqjtyping.Latent):
    x: jnp.ndarray
    _shape_template = OrderedDict(x=jax.ShapeDtypeStruct((1,), jnp.float32))


class _ObservationContext(seqjtyping.Observation):
    x: jnp.ndarray
    _shape_template = OrderedDict(x=jax.ShapeDtypeStruct((3,), jnp.float32))


class _ConditionContext(seqjtyping.Condition):
    c: jnp.ndarray
    _shape_template = OrderedDict(c=jax.ShapeDtypeStruct((2,), jnp.float32))


class _ParameterContext(seqjtyping.Parameters):
    p: jnp.ndarray
    _shape_template = OrderedDict(p=jax.ShapeDtypeStruct((4,), jnp.float32))


@dataclass(frozen=True)
class _EmbedderStub:
    sequence_embedded_context_dim: int = 6
    parameter_context_dim: int = _ParameterContext.flat_dim
    condition_context_dim: int = _ConditionContext.flat_dim
    embedded_context_dim: int = _ObservationContext.flat_dim


class _LatentApproximation(Protocol):
    def sample_and_log_prob(
        self,
        key: jax.Array,
        condition: LatentContext,
        state: object | None = None,
    ) -> tuple[seqjtyping.Latent, jax.Array, object | None]: ...


def _target_stats(dim: int) -> tuple[jax.Array, jax.Array]:
    mean = jnp.linspace(-0.5, 0.7, dim, dtype=jnp.float32)
    basis = jnp.arange(dim * dim, dtype=jnp.float32).reshape(dim, dim) / float(dim * dim)
    cov = basis @ basis.T + 0.35 * jnp.eye(dim, dtype=jnp.float32)
    return mean, cov


def _build_sequence_context(sample_length: int) -> LatentContext:
    return LatentContext.build_from_sequence_and_embedded(
        sequence_embedded_context=jnp.ones((sample_length, 6), dtype=jnp.float32),
        embedded_context=jnp.linspace(-1.0, 1.0, _ObservationContext.flat_dim, dtype=jnp.float32),
        observations=_ObservationContext.unravel(
            jnp.ones((sample_length, _ObservationContext.flat_dim), dtype=jnp.float32)
        ),
        conditions=_ConditionContext.unravel(
            jnp.zeros((sample_length, _ConditionContext.flat_dim), dtype=jnp.float32)
        ),
        parameters=_ParameterContext.unravel(jnp.array([[0.3, -0.2, 0.1, 0.5]], dtype=jnp.float32)),
    )


def _build_global_context(sample_length: int) -> LatentContext:
    return LatentContext(
        observation_context=_ObservationContext.unravel(
            jnp.ones((_ObservationContext.flat_dim,), dtype=jnp.float32)
        ),
        condition_context=_ConditionContext.unravel(
            jnp.zeros((_ConditionContext.flat_dim,), dtype=jnp.float32)
        ),
        parameter_context=_ParameterContext.unravel(
            jnp.array([0.3, -0.2, 0.1, 0.5], dtype=jnp.float32)
        ),
        embedded_context=jnp.linspace(-1.0, 1.0, _ObservationContext.flat_dim, dtype=jnp.float32),
        sequence_embedded_context=jnp.ones((sample_length, 6), dtype=jnp.float32),
    )


def _sample_loss(
    approximation: _LatentApproximation,
    *,
    key: jax.Array,
    context: LatentContext,
    mean: jax.Array,
    cov: jax.Array,
    samples_per_step: int,
) -> jax.Array:
    keys = jrandom.split(key, samples_per_step)

    def _single_sample(sample_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        sample, log_q, _ = approximation.sample_and_log_prob(sample_key, context)
        x = sample.ravel().reshape(-1)
        log_p = jstats.multivariate_normal.logpdf(x, mean, cov)
        return log_q, log_p

    log_q, log_p = jax.vmap(_single_sample)(keys)
    return jnp.mean(log_q - log_p)


def _fit_and_estimate_kl(
    approximation: _LatentApproximation,
    *,
    key: jax.Array,
    context: LatentContext,
    dim: int,
    train_steps: int = 100,
    train_lr: float = 2e-2,
    samples_per_step: int = 96,
    eval_samples: int = 8192,
) -> float:
    mean, cov = _target_stats(dim)

    optimizer = optax.adam(train_lr)
    opt_state = optimizer.init(eqx.filter(approximation, eqx.is_inexact_array))

    @eqx.filter_jit
    def _update(
        model: _LatentApproximation,
        state: optax.OptState,
        step_key: jax.Array,
    ) -> tuple[_LatentApproximation, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(_sample_loss)(
            model,
            key=step_key,
            context=context,
            mean=mean,
            cov=cov,
            samples_per_step=samples_per_step,
        )
        updates, next_state = optimizer.update(grads, state, model)
        next_model = eqx.apply_updates(model, updates)
        return next_model, next_state, loss

    model = approximation
    loop_key = key
    for _ in range(train_steps):
        loop_key, step_key = jrandom.split(loop_key)
        model, opt_state, _ = _update(model, opt_state, step_key)

    eval_key = jrandom.fold_in(loop_key, 999)
    kl_estimate = _sample_loss(
        model,
        key=eval_key,
        context=context,
        mean=mean,
        cov=cov,
        samples_per_step=eval_samples,
    )
    return float(kl_estimate)


@pytest.mark.parametrize(
    ("latent_cls", "dim", "kl_tolerance"),
    [
        (_Latent2D, 2, 0.12),
        (_Latent5D, 5, 0.35),
    ],
)
def test_latent_approximations_can_fit_known_gaussian(
    latent_cls: type[seqjtyping.Latent],
    dim: int,
    kl_tolerance: float,
) -> None:
    embedder = _EmbedderStub()

    approximations: list[tuple[str, _LatentApproximation, LatentContext]] = [
        (
            "autoregressive",
            AmortizedMultivariateAutoregressor(
                latent_cls,
                sample_length=1,
                embedder=embedder,
                lag_order=1,
                nn_width=16,
                nn_depth=1,
                key=jrandom.key(1),
            ),
            _build_sequence_context(sample_length=1),
        ),
        (
            "structured",
            StructuredPrecisionGaussian(
                latent_cls,
                sample_length=1,
                embedder=embedder,
                hidden_dim=16,
                depth=1,
                key=jrandom.key(2),
            ),
            _build_global_context(sample_length=1),
        ),
        (
            "maf",
            AmortizedMAF(
                latent_cls,
                sample_length=1,
                embedder=embedder,
                key=jrandom.key(3),
                nn_width=16,
                nn_depth=1,
                flow_layers=1,
            ),
            _build_global_context(sample_length=1),
        ),
    ]
    approximations.append(
        (
            "conv-flow",
            AmortizedConvCoupling(
                _Latent1D,
                sample_length=dim,
                embedder=embedder,
                key=jrandom.key(4),
                nn_width=16,
                nn_depth=1,
                flow_layers=1,
                kernel_size=3,
            ),
            _build_global_context(sample_length=dim),
        )
    )

    results: dict[str, float] = {}
    for index, (label, approximation, context) in enumerate(approximations):
        key = jrandom.key(123 + index + 100 * dim)
        results[label] = _fit_and_estimate_kl(
            approximation,
            key=key,
            context=context,
            dim=dim,
        )

    per_approx_tolerance = {
        2: {
            "autoregressive": kl_tolerance,
            "structured": kl_tolerance,
            "maf": 0.20,
            "conv-flow": 0.30,
        },
        5: {
            "autoregressive": 0.20,
            "structured": 0.20,
            "maf": kl_tolerance,
            "conv-flow": 0.90,
        },
    }
    if dim not in per_approx_tolerance:
        raise ValueError(f"No tolerance table configured for dim={dim}.")

    dim_tolerance = per_approx_tolerance[dim]
    violations = {
        label: kl
        for label, kl in results.items()
        if label not in dim_tolerance or (not jnp.isfinite(kl) or kl > dim_tolerance[label])
    }
    if violations:
        formatted = ", ".join(f"{label}={kl:.4f}" for label, kl in sorted(violations.items()))
        raise AssertionError(
            f"Failed Gaussian fit checks in dim={dim} with tolerances {dim_tolerance}: {formatted}. "
            f"All KL estimates: {results}"
        )
