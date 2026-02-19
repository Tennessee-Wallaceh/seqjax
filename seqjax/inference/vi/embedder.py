import equinox as eqx
from jaxtyping import Array, Int
import jax.numpy as jnp
import jax

from .api import LatentContext, Embedder

class WindowEmbedder(Embedder):
    """
    Reshapes observation information to a context of appropriate size for
    each step in the batch
    """

    sample_length: int
    prev_window: int
    post_window: int
    y_dimension: int

    window_size: int
    indexer: Int[Array, "sample_length window_size"]

    def __init__(
        self,
        sample_length,
        prev_window,
        post_window,
        y_dimension: int = 1,
    ):
        self.prev_window = prev_window  # take prev_window observations before
        self.post_window = post_window  # take post_window observations after
        self.window_size = prev_window + post_window + 1
        self.y_dimension = y_dimension
        self.context_dimension = y_dimension * self.window_size
        self.sample_length = sample_length

        # build the indexer, applied to each dimension of y
        # will give the context window
        # the indexer operates on the padded observations
        # padded_y.shape == [prev_window + observation_length + post_window, y_dimension]
        sample_path_ix = jnp.arange(sample_length).reshape(-1, 1)
        self.indexer = (
            jnp.hstack(
                [sample_path_ix - step for step in reversed(range(1, prev_window + 1))]
                + [sample_path_ix]
                + [sample_path_ix + step for step in range(1, post_window + 1)]
            )
            + prev_window
        )

    def _pad(self, observations):
        return jnp.pad(observations, (self.prev_window, self.post_window), mode="mean")[
            self.indexer
        ]

    def embed(
        self, 
        observations,
        conditions,
        parameters,
    ):
        observation_array = observations.ravel()
        per_dim_context = jax.vmap(self._pad, in_axes=[1])(observation_array)
        
        # flip so leading dim is step index, and flatten each step
        sequence_embedded_context = jax.vmap(jnp.ravel)(jnp.transpose(per_dim_context, (1, 0, 2)))
        return LatentContext.build_from_sequence_context(
            sequence_embedded_context, 
            observations,
            conditions,
            parameters
        )


class RNNEmbedder(Embedder):
    cell_fwd: eqx.nn.GRUCell
    cell_rev: eqx.nn.GRUCell

    def __init__(self, hidden: int, y_dim: int, *, key):
        k1, k2 = jax.random.split(key)
        self.cell_fwd = eqx.nn.GRUCell(y_dim, hidden, key=k1)
        self.cell_rev = eqx.nn.GRUCell(y_dim, hidden, key=k2)
        self.context_dimension = 2 * hidden  # fwd ⊕ rev

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

    def __init__(
        self,
        hidden: int,
        y_dim: int,
        *,
        kernel_size: int = 5,
        depth: int = 2,
        key,
    ):
        k_in, *k_layers = jax.random.split(key, depth + 1)

        self.in_proj = eqx.nn.Conv1d(
            in_channels=y_dim,
            out_channels=hidden,
            kernel_size=1,
            padding="SAME",
            key=k_in,
        )

        self.convs = tuple(
            eqx.nn.Conv1d(
                in_channels=hidden,
                out_channels=hidden,
                kernel_size=kernel_size,
                padding="SAME",   # centered/offline
                key=k,
            )
            for k in k_layers
        )

        self.context_dimension = hidden

    def embed(self, observations, conditions, parameters):
        seq = observations.ravel()     # (T, y_dim)
        x = jnp.swapaxes(seq, 0, 1)    # (y_dim, T)

        x = self.in_proj(x)            # (hidden, T)
        x = jax.nn.gelu(x)
        for conv in self.convs:
            x = conv(x)
            x = jax.nn.gelu(x)

        sequence_embedded_context = jnp.swapaxes(x, 0, 1)   # (T, hidden)
    
        return LatentContext.build_from_sequence_context(
            sequence_embedded_context, 
            observations,
            conditions,
            parameters
        )