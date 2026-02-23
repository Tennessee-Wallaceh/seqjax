import typing

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .api import SequenceAggregator


AggregationKind = typing.Literal[
    "sequence-flatten",
    "observation-flatten",
    "avg-pool",
    "max-pool",
]


class SequenceFlattenAggregator(eqx.Module):
    output_dim: int = eqx.field(static=True)

    def __init__(self, *, sample_length: int, sequence_dim: int):
        self.output_dim = sample_length * sequence_dim

    def __call__(self, sequence_features: Array, observations) -> Array:
        del observations
        return sequence_features.reshape(-1)


class ObservationFlattenAggregator(eqx.Module):
    output_dim: int = eqx.field(static=True)

    def __init__(self, *, sample_length: int, observation_dim: int):
        self.output_dim = sample_length * observation_dim

    def __call__(self, sequence_features: Array, observations) -> Array:
        del sequence_features
        return observations.ravel()


class AvgPoolAggregator(eqx.Module):
    output_dim: int = eqx.field(static=True)
    pooling: eqx.nn.AdaptiveAvgPool1d

    def __init__(self, *, pool_dim: int, sequence_dim: int):
        self.pooling = eqx.nn.AdaptiveAvgPool1d(pool_dim)
        self.output_dim = pool_dim * sequence_dim

    def __call__(self, sequence_features: Array, observations) -> Array:
        del observations
        pooled = self.pooling(jnp.swapaxes(sequence_features, 0, 1))
        return pooled.flatten()


class MaxPoolAggregator(eqx.Module):
    output_dim: int = eqx.field(static=True)
    pooling: eqx.nn.AdaptiveMaxPool1d

    def __init__(self, *, pool_dim: int, sequence_dim: int):
        self.pooling = eqx.nn.AdaptiveMaxPool1d(pool_dim)
        self.output_dim = pool_dim * sequence_dim

    def __call__(self, sequence_features: Array, observations) -> Array:
        del observations
        pooled = self.pooling(jnp.swapaxes(sequence_features, 0, 1))
        return pooled.flatten()


def build_sequence_aggregator(
    kind: AggregationKind,
    *,
    sample_length: int,
    sequence_dim: int,
    observation_dim: int,
    pool_dim: None | int = None,
) -> SequenceAggregator:
    if kind == "sequence-flatten":
        return SequenceFlattenAggregator(
            sample_length=sample_length,
            sequence_dim=sequence_dim,
        )
    if kind == "observation-flatten":
        return ObservationFlattenAggregator(
            sample_length=sample_length,
            observation_dim=observation_dim,
        )

    if pool_dim is None:
        pool_dim = max(1, int(0.1 * sample_length))

    if kind == "avg-pool":
        return AvgPoolAggregator(pool_dim=pool_dim, sequence_dim=sequence_dim)
    if kind == "max-pool":
        return MaxPoolAggregator(pool_dim=pool_dim, sequence_dim=sequence_dim)

    raise ValueError(
        "aggregation_kind='none' is not supported; "
        "choose one of {'sequence-flatten', 'observation-flatten', 'avg-pool', 'max-pool'}."
    )
