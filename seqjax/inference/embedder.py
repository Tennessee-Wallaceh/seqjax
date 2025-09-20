import equinox as eqx
from jaxtyping import Array, Int, Float
import jax.numpy as jnp
import jax
from abc import abstractmethod

import seqjax.model.typing as seqjtyping


class Embedder(eqx.Module):
    """
    Maps observation sequence to a context vector for
    each point in the batch
    """

    context_dimension: int

    @abstractmethod
    def embed(
        self, observations: seqjtyping.Observation
    ) -> Float[Array, "batch_length context_dimension"]: ...


class WindowEmbedder(Embedder):
    """
    Reshapes observation information to a context of appropriate size for
    each step in the batch
    """

    prev_window: int
    post_window: int
    sample_length: int
    window_size: int
    y_dimension: int
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
        self, observations: seqjtyping.Observation
    ) -> Float[Array, "batch_length x_dimension context_length"]:
        observation_array = observations.ravel(observations)
        per_dim_context = jax.vmap(self._pad, in_axes=[1])(observation_array)
        # flip so leading dim is step index, and flatten each step
        return jax.vmap(jnp.ravel)(jnp.transpose(per_dim_context, (1, 0, 2)))


class NoReshapeEmbedder(Embedder):
    prev_window: int
    post_window: int
    window_size: int
    y_dimension: int
    indexer: Int[Array, "batch_length window_size"]

    def __init__(self, batch_length, prev_window, post_window, y_dimension: int = 1):
        self.prev_window = prev_window  # take prev_window observations before
        self.post_window = post_window  # take post_window observations after
        self.window_size = prev_window + post_window + 1
        self.y_dimension = y_dimension
        self.context_dimension = y_dimension * batch_length

        # build the indexer, applied to each dimension of y
        # will give the context window
        path_ix = jnp.arange(batch_length).reshape(-1, 1)
        self.indexer = (
            jnp.hstack(
                [path_ix - step for step in reversed(range(1, prev_window + 1))]
                + [path_ix]
                + [path_ix + step for step in range(1, post_window + 1)]
            )
            + prev_window
        )

    def _pad(self, observations):
        return jnp.pad(observations, (self.prev_window, self.post_window), mode="mean")[
            self.indexer
        ]

    def embed(
        self, observations: seqjtyping.Observation
    ) -> Float[Array, "batch_length context_dimension"]:
        return observations.ravel(observations)


class SquareDiffEmbedder(WindowEmbedder):
    """
    Idenditcal to pass through, but square the difference in observations
    Useful for latent vol.
    """

    def _pad(self, observations):
        dt = jnp.array(1 / (256 * 8 * 60))
        observations = jnp.abs(
            jnp.log(observations[..., 1:]) - jnp.log(observations[..., :-1])
        ) / jnp.sqrt(dt)
        return jnp.pad(
            jnp.square(observations), (self.prev_window, self.post_window), mode="mean"
        )[self.indexer]


class LogReturnEmbedder(Embedder):
    """
    Computes a annualised log return, from a path of prices.
    Useful for stochastic vol.
    """

    prev_window: int
    post_window: int
    sample_length: int
    window_size: int
    y_dimension: int
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

    def embed(self, observations):
        dt = jnp.array(1 / (256 * 8))
        log_returns = (
            jnp.log(observations[1:]) - jnp.log(observations[:-1])
        ) / jnp.sqrt(dt)
        embedded = jax.vmap(self._pad, in_axes=[1])(log_returns)

        def slice_embed(step_ix):
            return jax.lax.dynamic_slice_in_dim(
                embedded, step_ix[0], self.window_size, axis=1
            )

        return jax.vmap(jnp.ravel)(jax.vmap(slice_embed)(self.indexer))

    def _pad(self, observations):
        return jnp.pad(observations, (self.prev_window, self.post_window), mode="mean")


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
        return hs  # shape (T, hidden)

    def embed(
        self, obs: seqjtyping.Packable
    ) -> Float[Array, "batch_length context_dimension"]:
        seq = obs.ravel(obs)  # (T, y_dim)
        h_fwd = self._scan(self.cell_fwd, seq)
        h_rev = self._scan(self.cell_rev, seq[::-1])[::-1]
        return jnp.concatenate([h_fwd, h_rev], axis=-1)  # (T, 2*hidden)
