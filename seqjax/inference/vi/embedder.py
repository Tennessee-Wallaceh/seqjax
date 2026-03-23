import equinox as eqx
from jaxtyping import Array, Int
import jax.numpy as jnp
import jax
import typing
from dataclasses import field

from seqjax.model.interface import BayesianSequentialModelProtocol
from .interface import LatentContext, Embedder, SequenceAggregator
from .aggregation import AggregationKind, build_sequence_aggregator


PositionMode = typing.Literal["sample", "sequence"]


class PositionalBasis(typing.Protocol):
    def __call__(
        self,
        positions: Array,
        n_pos_embedding: int,
    ) -> Array: ...


def _fourier_positional_basis(
    positions: Array,
    n_pos_embedding: int,
) -> Array:
    freqs = (2.0 ** jnp.arange(n_pos_embedding)) * jnp.pi
    phase = positions[:, None] * freqs[None, :]
    return jnp.concatenate(
        [positions[:, None], jnp.sin(phase), jnp.cos(phase)],
        axis=-1,
    )


def _validate_position_mode(position_mode: None | PositionMode) -> None:
    if position_mode is not None and position_mode not in ("sample", "sequence"):
        raise ValueError(
            "position_mode must be one of None, 'sample', or 'sequence', "
            + f"got {position_mode!r}"
        )


def _build_position_features(
    *,
    sample_length: int,
    sequence_length: int,
    n_pos_embedding: int,
    position_mode: PositionMode,
    positional_basis: PositionalBasis,
    sequence_start: None | int,
    sample_mode_cache: None | Array,
) -> Array:
    if position_mode == "sample":
        if sample_mode_cache is not None:
            return sample_mode_cache
        positions = (jnp.arange(sample_length, dtype=jnp.float32) + 0.5) / jnp.asarray(
            sample_length,
            dtype=jnp.float32,
        )
        return positional_basis(positions, n_pos_embedding)

    if sequence_start is None:
        raise ValueError("sequence_start must be provided when position_mode='sequence'.")

    positions = (
        jnp.arange(sample_length, dtype=jnp.float32)
        + jnp.asarray(int(sequence_start), dtype=jnp.float32)
        + 0.5
    ) / jnp.asarray(sequence_length, dtype=jnp.float32)
    return positional_basis(positions, n_pos_embedding)


class PositionalEmbedder(Embedder):
    """Produces pure positional sequence features from sample-step indices."""

    n_pos_embedding: int = eqx.field(static=True)
    position_mode: PositionMode = eqx.field(static=True)
    positional_basis: PositionalBasis = eqx.field(static=True)
    pos_context: None | Array

    def __init__(
        self,
        target_posterior: BayesianSequentialModelProtocol,
        sample_length: int,
        sequence_length: int,
        n_pos_embedding: int = 8,
        position_mode: PositionMode = "sample",
        positional_basis: PositionalBasis = _fourier_positional_basis,
    ):
        if sample_length < 1:
            raise ValueError(f"sample_length must be >= 1, got {sample_length}")
        if n_pos_embedding < 1:
            raise ValueError(f"n_pos_embedding must be >= 1, got {n_pos_embedding}")

        self.n_pos_embedding = n_pos_embedding
        self.position_mode = position_mode
        self.positional_basis = positional_basis
        self.pos_context = (
            self.positional_basis(
                (jnp.arange(sample_length, dtype=jnp.float32) + 0.5)
                / jnp.asarray(sample_length, dtype=jnp.float32),
                n_pos_embedding,
            )
            if position_mode == "sample"
            else None
        )

        self.sequence_embedded_context_dim = 1 + 2 * self.n_pos_embedding
        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
            self.embedded_context_dim,
        ) = LatentContext.from_sequence_context_dims(target_posterior, sample_length)

        super().__init__(target_posterior, sample_length, sequence_length)

    def embed(
        self,
        observations,
        conditions,
        parameters,
        *,
        sequence_start: None | int = None,
    ):
        sequence_embedded_context = _build_position_features(
            sample_length=self.sample_length,
            sequence_length=self.sequence_length,
            n_pos_embedding=self.n_pos_embedding,
            position_mode=self.position_mode,
            positional_basis=self.positional_basis,
            sequence_start=sequence_start,
            sample_mode_cache=self.pos_context,
        )

        return LatentContext.build_from_sequence_context(
            sequence_embedded_context,
            observations,
            conditions,
            parameters,
        )


class WindowEmbedder(Embedder):
    """
    Reshapes observation information to a context of appropriate size for
    each step in the batch
    """

    prev_window: int
    post_window: int
    position_mode: None | PositionMode = eqx.field(static=True)
    n_pos_embedding: int = eqx.field(static=True)
    positional_basis: PositionalBasis = eqx.field(static=True, default=_fourier_positional_basis)
    pos_context: None | Array = field(init=False, default=None)

    y_dimension: int = field(init=False)
    window_size: int = field(init=False)
    indexer: Int[Array, "sample_length window_size"] = field(init=False)

    def __init__(
        self,
        target_posterior: BayesianSequentialModelProtocol,
        sample_length: int,
        sequence_length: int,
        prev_window: int,
        post_window: int,
        position_mode: None | PositionMode = None,
        n_pos_embedding: int = 8,
        positional_basis: PositionalBasis = _fourier_positional_basis,
    ):
        self.prev_window = prev_window
        self.post_window = post_window
        self.position_mode = position_mode
        self.n_pos_embedding = n_pos_embedding
        self.positional_basis = positional_basis

        self.window_size = self.prev_window + self.post_window + 1
        self.y_dimension = target_posterior.target.observation_cls.flat_dim
        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
            self.embedded_context_dim,
        ) = LatentContext.from_sequence_context_dims(
            target_posterior,
            sample_length,
        )

        _validate_position_mode(self.position_mode)
        if self.n_pos_embedding < 1:
            raise ValueError(f"n_pos_embedding must be >= 1, got {self.n_pos_embedding}")

        pos_dim = 0 if self.position_mode is None else (1 + 2 * self.n_pos_embedding)
        self.sequence_embedded_context_dim = self.y_dimension * self.window_size + pos_dim

        self.pos_context = None
        if self.position_mode == "sample":
            positions = (jnp.arange(sample_length, dtype=jnp.float32) + 0.5) / jnp.asarray(
                sample_length,
                dtype=jnp.float32,
            )
            self.pos_context = self.positional_basis(positions, self.n_pos_embedding)

        sample_path_ix = jnp.arange(sample_length).reshape(-1, 1)
        self.indexer = (
            jnp.hstack(
                [sample_path_ix - step for step in reversed(range(1, self.prev_window + 1))]
                + [sample_path_ix]
                + [sample_path_ix + step for step in range(1, self.post_window + 1)]
            )
            + self.prev_window
        )

        super().__init__(target_posterior, sample_length, sequence_length)

    def _pad(self, observations):
        return jnp.pad(
            observations,
            (self.prev_window, self.post_window),
            mode="edge",
        )[self.indexer]

    def embed(
        self,
        observations,
        conditions,
        parameters,
        *,
        sequence_start: None | int = None,
    ):
        observation_array = observations.ravel()
        per_dim_context = jax.vmap(self._pad, in_axes=[1])(observation_array)

        sequence_embedded_context = jax.vmap(jnp.ravel)(
            jnp.transpose(per_dim_context, (1, 0, 2))
        )

        if self.position_mode is not None:
            position_features = _build_position_features(
                sample_length=self.sample_length,
                sequence_length=self.sequence_length,
                n_pos_embedding=self.n_pos_embedding,
                position_mode=self.position_mode,
                positional_basis=self.positional_basis,
                sequence_start=sequence_start,
                sample_mode_cache=self.pos_context,
            )
            sequence_embedded_context = jnp.concatenate(
                [sequence_embedded_context, position_features],
                axis=-1,
            )

        return LatentContext.build_from_sequence_context(
            sequence_embedded_context,
            observations,
            conditions,
            parameters,
        )


class RNNEmbedder(Embedder):
    cell_fwd: eqx.nn.GRUCell
    cell_rev: eqx.nn.GRUCell
    hidden: int
    aggregation_kind: AggregationKind = eqx.field(static=True)
    aggregator: SequenceAggregator = eqx.field(static=True)
    position_mode: None | PositionMode = eqx.field(static=True)
    n_pos_embedding: int = eqx.field(static=True)
    positional_basis: PositionalBasis = eqx.field(static=True)
    pos_context: None | Array

    def __init__(
        self,
        target_posterior: BayesianSequentialModelProtocol,
        sample_length: int,
        sequence_length: int,
        hidden: int,
        aggregation_kind: AggregationKind = "observation-flatten",
        position_mode: None | PositionMode = None,
        n_pos_embedding: int = 8,
        positional_basis: PositionalBasis = _fourier_positional_basis,
        *,
        key,
    ):
        y_dim = target_posterior.target.observation_cls.flat_dim
        k1, k2 = jax.random.split(key)
        self.cell_fwd = eqx.nn.GRUCell(y_dim, hidden, key=k1)
        self.cell_rev = eqx.nn.GRUCell(y_dim, hidden, key=k2)
        self.hidden = hidden
        self.aggregation_kind = aggregation_kind
        self.position_mode = position_mode
        self.n_pos_embedding = n_pos_embedding
        self.positional_basis = positional_basis

        _validate_position_mode(self.position_mode)
        if self.n_pos_embedding < 1:
            raise ValueError(f"n_pos_embedding must be >= 1, got {self.n_pos_embedding}")

        pos_dim = 0 if self.position_mode is None else (1 + 2 * self.n_pos_embedding)
        self.sequence_embedded_context_dim = self.hidden * 2 + pos_dim
        self.pos_context = None
        if self.position_mode == "sample":
            positions = (jnp.arange(sample_length, dtype=jnp.float32) + 0.5) / jnp.asarray(
                sample_length,
                dtype=jnp.float32,
            )
            self.pos_context = self.positional_basis(positions, self.n_pos_embedding)

        observation_dim = target_posterior.target.observation_cls.flat_dim
        self.aggregator = build_sequence_aggregator(
            aggregation_kind,
            sample_length=sample_length,
            sequence_dim=self.sequence_embedded_context_dim,
            observation_dim=observation_dim,
        )
        self.embedded_context_dim = self.aggregator.output_dim
        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
        ) = LatentContext.from_sequence_and_embedded_dims(target_posterior, sample_length)

        super().__init__(target_posterior, sample_length, sequence_length)

    def _scan(self, cell, seq):
        def step(carry, x):
            new_carry = cell(x, carry)
            return new_carry, new_carry

        h0 = jnp.zeros((cell.hidden_size,))
        _, hs = jax.lax.scan(step, h0, seq)
        return hs

    def embed(self, observations, conditions, parameters, *, sequence_start: None | int = None):
        seq = observations.ravel()
        h_fwd = self._scan(self.cell_fwd, seq)
        h_rev = self._scan(self.cell_rev, seq[::-1])[::-1]
        sequence_embedded_context = jnp.concatenate([h_fwd, h_rev], axis=-1)

        if self.position_mode is not None:
            position_features = _build_position_features(
                sample_length=self.sample_length,
                sequence_length=self.sequence_length,
                n_pos_embedding=self.n_pos_embedding,
                position_mode=self.position_mode,
                positional_basis=self.positional_basis,
                sequence_start=sequence_start,
                sample_mode_cache=self.pos_context,
            )
            sequence_embedded_context = jnp.concatenate(
                [sequence_embedded_context, position_features],
                axis=-1,
            )

        embedded_context = self.aggregator(sequence_embedded_context, observations)
        return LatentContext.build_from_sequence_and_embedded(
            sequence_embedded_context,
            embedded_context,
            observations,
            conditions,
            parameters,
        )


class ConvResidualBlock(eqx.Module):
    conv_1: eqx.nn.Conv1d
    norm_1: eqx.nn.LayerNorm

    def __init__(
        self,
        hidden: int,
        *,
        kernel_size: int,
        dilation: int=1,
        key,
    ):
        k1, k2 = jax.random.split(key, 2)
        self.conv_1 = eqx.nn.Conv1d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="SAME",
            padding_mode="REPLICATE",
            key=k1,
        )
        self.norm_1 = eqx.nn.LayerNorm(hidden)

    def _norm_over_time(self, norm, x: Array) -> Array:
        return jax.vmap(norm, in_axes=1, out_axes=1)(x)

    def __call__(self, x):
        resid = self._norm_over_time(self.norm_1, x)
        resid = self.conv_1(resid)
        resid = jax.nn.gelu(resid)
        return 0.9 * x + 0.1 * resid
    
class Conv1DEmbedder(Embedder):
    in_proj: eqx.nn.Conv1d
    convs: tuple[ConvResidualBlock, ...]
    hidden: int
    aggregation_kind: AggregationKind = eqx.field(static=True)
    aggregator: SequenceAggregator = eqx.field(static=True)
    position_mode: None | PositionMode = eqx.field(static=True)
    n_pos_embedding: int = eqx.field(static=True)
    positional_basis: PositionalBasis = eqx.field(static=True)
    pos_context: None | Array

    def __init__(
        self,
        target_posterior: BayesianSequentialModelProtocol,
        sample_length: int,
        sequence_length: int,
        hidden: int,
        *,
        kernel_size: int = 5,
        depth: int = 2,
        pool_dim: None | int = None,
        pool_kind: str = "avg",
        position_mode: None | PositionMode = None,
        n_pos_embedding: int = 8,
        positional_basis: PositionalBasis = _fourier_positional_basis,
        key,
    ):
        k_in, *k_layers = jax.random.split(key, depth + 1)

        self.in_proj = eqx.nn.Conv1d(
            in_channels=target_posterior.target.observation_cls.flat_dim,
            out_channels=hidden,
            kernel_size=1,
            padding="SAME",
            padding_mode="REPLICATE",
            key=k_in,
        )

        dilations = tuple(2**i for i in range(depth))
        self.convs = tuple(
            ConvResidualBlock(
                hidden=hidden,
                kernel_size=kernel_size,
                dilation=d,
                key=k,
            )
            for d, k in zip(dilations, k_layers)
        )
        self.hidden = hidden

        if pool_kind == "avg":
            self.aggregation_kind = "avg-pool"
        elif pool_kind == "max":
            self.aggregation_kind = "max-pool"
        else:
            raise ValueError(f"pool_kind: {pool_kind} not supported!")

        self.position_mode = position_mode
        self.n_pos_embedding = n_pos_embedding
        self.positional_basis = positional_basis
        _validate_position_mode(self.position_mode)
        if self.n_pos_embedding < 1:
            raise ValueError(f"n_pos_embedding must be >= 1, got {self.n_pos_embedding}")

        pos_dim = 0 if self.position_mode is None else (1 + 2 * self.n_pos_embedding)
        self.sequence_embedded_context_dim = self.hidden + pos_dim
        self.pos_context = None
        if self.position_mode == "sample":
            positions = (jnp.arange(sample_length, dtype=jnp.float32) + 0.5) / jnp.asarray(
                sample_length,
                dtype=jnp.float32,
            )
            self.pos_context = self.positional_basis(positions, self.n_pos_embedding)

        observation_dim = target_posterior.target.observation_cls.flat_dim
        self.aggregator = build_sequence_aggregator(
            self.aggregation_kind,
            sample_length=sample_length,
            sequence_dim=self.sequence_embedded_context_dim,
            observation_dim=observation_dim,
            pool_dim=pool_dim,
        )
        self.embedded_context_dim = self.aggregator.output_dim
        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
        ) = LatentContext.from_sequence_and_embedded_dims(target_posterior, sample_length)

        super().__init__(target_posterior, sample_length, sequence_length)

    def convolve(self, observations):
        seq = observations.ravel()
        x = jnp.swapaxes(seq, 0, 1)

        x = self.in_proj(x)
        x = jax.nn.gelu(x)
        for conv in self.convs:
            x = conv(x)

        return jnp.swapaxes(x, 0, 1)

    def embed(self, observations, conditions, parameters, *, sequence_start: None | int = None):
        sequence_embedded_context = self.convolve(observations)
        if self.position_mode is not None:
            position_features = _build_position_features(
                sample_length=self.sample_length,
                sequence_length=self.sequence_length,
                n_pos_embedding=self.n_pos_embedding,
                position_mode=self.position_mode,
                positional_basis=self.positional_basis,
                sequence_start=sequence_start,
                sample_mode_cache=self.pos_context,
            )
            sequence_embedded_context = jnp.concatenate(
                [sequence_embedded_context, position_features],
                axis=-1,
            )

        aggregated = self.aggregator(sequence_embedded_context, observations)
        return LatentContext.build_from_sequence_and_embedded(
            sequence_embedded_context,
            aggregated,
            observations,
            conditions,
            parameters,
        )


class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm_1: eqx.nn.LayerNorm
    norm_2: eqx.nn.LayerNorm
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, *, key):
        k_attn, k_in, k_out = jax.random.split(key, 3)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            key=k_attn,
        )
        self.norm_1 = eqx.nn.LayerNorm(hidden_dim)
        self.norm_2 = eqx.nn.LayerNorm(hidden_dim)
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=k_in)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=k_out)

    def __call__(self, x):
        x_norm = jax.vmap(self.norm_1)(x)
        x = x + self.attention(x_norm, x_norm, x_norm)

        ff_in = jax.vmap(self.norm_2)(x)
        ff_hidden = jax.vmap(self.ff_in)(ff_in)
        ff_out = jax.vmap(self.ff_out)(jax.nn.gelu(ff_hidden))
        return x + ff_out


class TransformerEmbedder(Embedder):
    in_proj: eqx.nn.Linear
    blocks: tuple[TransformerBlock, ...]
    hidden: int
    pooling: eqx.nn.AdaptiveAvgPool1d
    position_mode: None | PositionMode = eqx.field(static=True)
    n_pos_embedding: int = eqx.field(static=True)
    positional_basis: PositionalBasis = eqx.field(static=True)
    pos_context: None | Array

    def __init__(
        self,
        target_posterior: BayesianSequentialModelProtocol,
        sample_length: int,
        sequence_length: int,
        hidden: int,
        *,
        depth: int = 2,
        num_heads: int = 2,
        mlp_multiplier: int = 4,
        pool_dim: None | int = None,
        position_mode: None | PositionMode = None,
        n_pos_embedding: int = 8,
        positional_basis: PositionalBasis = _fourier_positional_basis,
        key,
    ):
        y_dim = target_posterior.target.observation_cls.flat_dim
        k_proj, *k_blocks = jax.random.split(key, depth + 1)

        self.in_proj = eqx.nn.Linear(y_dim, hidden, key=k_proj)
        self.blocks = tuple(
            TransformerBlock(
                hidden_dim=hidden,
                num_heads=num_heads,
                mlp_dim=hidden * mlp_multiplier,
                key=k,
            )
            for k in k_blocks
        )
        self.hidden = hidden

        if pool_dim is None:
            pool_dim = max(1, int(0.1 * sample_length))
        self.pooling = eqx.nn.AdaptiveAvgPool1d(pool_dim)

        self.position_mode = position_mode
        self.n_pos_embedding = n_pos_embedding
        self.positional_basis = positional_basis
        _validate_position_mode(self.position_mode)
        if self.n_pos_embedding < 1:
            raise ValueError(f"n_pos_embedding must be >= 1, got {self.n_pos_embedding}")

        pos_dim = 0 if self.position_mode is None else (1 + 2 * self.n_pos_embedding)
        self.sequence_embedded_context_dim = hidden + pos_dim
        self.embedded_context_dim = self.pooling.target_shape[0] * self.sequence_embedded_context_dim

        self.pos_context = None
        if self.position_mode == "sample":
            positions = (jnp.arange(sample_length, dtype=jnp.float32) + 0.5) / jnp.asarray(
                sample_length,
                dtype=jnp.float32,
            )
            self.pos_context = self.positional_basis(positions, self.n_pos_embedding)

        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
        ) = LatentContext.from_sequence_and_embedded_dims(target_posterior, sample_length)

        super().__init__(target_posterior, sample_length, sequence_length)

    def encode(self, observations):
        seq = observations.ravel()
        hidden_sequence = jax.vmap(self.in_proj)(seq)
        for block in self.blocks:
            hidden_sequence = block(hidden_sequence)
        return hidden_sequence

    def embed(self, observations, conditions, parameters, *, sequence_start: None | int = None):
        sequence_embedded_context = self.encode(observations)

        if self.position_mode is not None:
            position_features = _build_position_features(
                sample_length=self.sample_length,
                sequence_length=self.sequence_length,
                n_pos_embedding=self.n_pos_embedding,
                position_mode=self.position_mode,
                positional_basis=self.positional_basis,
                sequence_start=sequence_start,
                sample_mode_cache=self.pos_context,
            )
            sequence_embedded_context = jnp.concatenate(
                [sequence_embedded_context, position_features],
                axis=-1,
            )

        downsampled_embedding = self.pooling(
            jnp.swapaxes(sequence_embedded_context, 0, 1)
        ).flatten()
        return LatentContext.build_from_sequence_and_embedded(
            sequence_embedded_context,
            downsampled_embedding,
            observations,
            conditions,
            parameters,
        )
