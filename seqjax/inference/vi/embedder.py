import equinox as eqx
from jaxtyping import Array, Int
import jax.numpy as jnp
import jax
from dataclasses import field

from seqjax.model.base import BayesianSequentialModel
from .api import LatentContext, Embedder


# class PositionalEmbedder(Embedder):
#     """
#     Reshapes observation information to a context of appropriate size for
#     each step in the batch
#     """
#     sample_length: int
#     pos_context: int 

#     def __init__():
#         seq_length = sequence_embedded_context.shape[0]
#         pos = (jnp.arange(seq_length) + 0.5) / seq_length
                                                    
#         freqs = (2.0 ** jnp.arange(self.n_pos_embedding)) * jnp.pi 

#         x = pos[:, None] * freqs[None, :]
#         self.pos_context = jnp.concatenate(
#             [pos[:, None], jnp.sin(x), jnp.cos(x)],
#             axis=-1
#         )

#     def embed():

class WindowEmbedder(Embedder):
    """
    Reshapes observation information to a context of appropriate size for
    each step in the batch
    """
    prev_window: int
    post_window: int
    
    y_dimension: int = field(init=False)
    window_size: int = field(init=False)
    indexer: Int[Array, "sample_length window_size"] = field(init=False)

    def __post__init__(
        self,
    ):
        self.window_size = self.prev_window + self.post_window + 1
        self.y_dimension = self.target_posterior.target.observation_cls.flat_dim
        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
            self.embedded_context_dim,
        ) = LatentContext.from_sequence_context_dims(
            self.target_posterior, self.sample_length
        )

        self.sequence_embedded_context_dim = self.y_dimension * self.window_size

        # build the indexer, applied to each dimension of y
        # will give the context window
        # the indexer operates on the padded observations
        # padded_y.shape == [prev_window + observation_length + post_window, y_dimension]
        sample_path_ix = jnp.arange(self.sample_length).reshape(-1, 1)
        self.indexer = (
            jnp.hstack(
                [sample_path_ix - step for step in reversed(range(1, self.prev_window + 1))]
                + [sample_path_ix]
                + [sample_path_ix + step for step in range(1, self.post_window + 1)]
            )
            + self.prev_window
        )

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
    ):
        observation_array = observations.ravel()
        per_dim_context = jax.vmap(self._pad, in_axes=[1])(observation_array)
        
        # flip so leading dim is step index, and flatten each step
        sequence_embedded_context = jax.vmap(jnp.ravel)(
            jnp.transpose(per_dim_context, (1, 0, 2))
        )
        return LatentContext.build_from_sequence_context(
            sequence_embedded_context, 
            observations,
            conditions,
            parameters
        )


class RNNEmbedder(Embedder):
    cell_fwd: eqx.nn.GRUCell
    cell_rev: eqx.nn.GRUCell
    hidden: int

    def __init__(
        self, 
        target_posterior: BayesianSequentialModel,
        sample_length: int,
        sequence_length: int,
        hidden: int, 
        *, 
        key
    ):
        y_dim = self.target_posterior.target.observation_cls.flat_dim
        k1, k2 = jax.random.split(key)
        self.cell_fwd = eqx.nn.GRUCell(y_dim, hidden, key=k1)
        self.cell_rev = eqx.nn.GRUCell(y_dim, hidden, key=k2)
        self.hidden = hidden
        super().__init__(target_posterior, sample_length, sequence_length)

    def __post__init__(
        self,
    ):
        self.sequence_embedded_context_dim = (
            self.target_posterior.target.observation_cls.flat_dim
            * self.hidden * 2
        ) 
        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
            self.embedded_context_dim,
        ) = LatentContext.from_sequence_context_dims(
            self.target_posterior, self.sample_length
        )
        
    def _scan(self, cell, seq):
        def step(carry, x):
            new_carry = cell(x, carry)
            return new_carry, new_carry  # output the hidden itself

        h0 = jnp.zeros((cell.hidden_size,))
        _, hs = jax.lax.scan(step, h0, seq)
        return hs  # shape (T, hidden * 2)

    def embed(self, observations, conditions, parameters):
        seq = observations.ravel()  # (T, y_dim)
        h_fwd = self._scan(self.cell_fwd, seq)
        h_rev = self._scan(self.cell_rev, seq[::-1])[::-1]
        sequence_embedded_context = jnp.concatenate([h_fwd, h_rev], axis=-1)  # (T, 2*hidden)
        return LatentContext.build_from_sequence_context(
            sequence_embedded_context, 
            observations,
            conditions,
            parameters
        )


class Conv1DEmbedder(Embedder):
    in_proj: eqx.nn.Conv1d
    convs: tuple[eqx.nn.Conv1d, ...]
    hidden: int
    pooling: eqx.nn.AdaptiveAvgPool1d

    def __init__(
        self,
        target_posterior: BayesianSequentialModel,
        sample_length: int,
        sequence_length: int,
        hidden: int,
        *,
        kernel_size: int = 5,
        depth: int = 2,
        pool_dim: None | int = None,
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

        self.convs = tuple(
            eqx.nn.Conv1d(
                in_channels=hidden,
                out_channels=hidden,
                kernel_size=kernel_size,
                padding="SAME",
                padding_mode="REPLICATE",
                key=k,
            )
            for k in k_layers
        )
        self.hidden = hidden

        if pool_dim is None:
            pool_dim = max(1, int(0.1 * sample_length))
        self.pooling = eqx.nn.AdaptiveAvgPool1d(pool_dim)

        self.sequence_embedded_context_dim = (
            target_posterior.target.observation_cls.flat_dim
            * self.hidden
        )
        
        self.embedded_context_dim = self.pooling.target_shape[0] * self.hidden
        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
        ) = LatentContext.from_sequence_and_embedded_dims(
            target_posterior, sample_length
        )

        super().__init__(target_posterior, sample_length, sequence_length)

    def convolve(self, observations):
        seq = observations.ravel()     # (T, y_dim)
        x = jnp.swapaxes(seq, 0, 1)    # (y_dim, T)

        x = self.in_proj(x)            # (hidden, T)
        x = jax.nn.gelu(x)
        for conv in self.convs:
            x = conv(x)
            x = jax.nn.gelu(x)

        return x
    
    def embed(self, observations, conditions, parameters):
        sequence_embedded_context = self.convolve(observations)
        downsampled_embedding = self.pooling(sequence_embedded_context).flatten()
        return LatentContext.build_from_sequence_and_embedded(
            jnp.swapaxes(sequence_embedded_context, 0, 1),  # (T, hidden)
            downsampled_embedding,
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

    def __init__(
        self,
        target_posterior: BayesianSequentialModel,
        sample_length: int,
        sequence_length: int,
        hidden: int,
        *,
        depth: int = 2,
        num_heads: int = 2,
        mlp_multiplier: int = 4,
        pool_dim: None | int = None,
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

        self.sequence_embedded_context_dim = hidden
        self.embedded_context_dim = self.pooling.target_shape[0] * hidden
        (
            self.observation_context_dim,
            self.condition_context_dim,
            self.parameter_context_dim,
        ) = LatentContext.from_sequence_and_embedded_dims(
            target_posterior, sample_length
        )

        super().__init__(target_posterior, sample_length, sequence_length)

    def encode(self, observations):
        seq = observations.ravel()
        hidden_sequence = jax.vmap(self.in_proj)(seq)
        for block in self.blocks:
            hidden_sequence = block(hidden_sequence)
        return hidden_sequence

    def embed(self, observations, conditions, parameters):
        sequence_embedded_context = self.encode(observations)
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
