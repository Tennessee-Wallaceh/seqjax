# mypy: ignore-errors
from abc import abstractmethod
from typing import Callable, Optional, Literal, Any
import equinox as eqx
import jax.scipy.stats as jstats
from jaxtyping import Shaped, Array, Int, Float, PRNGKeyArray
import jax.numpy as jnp
import jax
from jax.nn import softplus
import jax.random as jrandom


import seqjax
import seqjax.model
import seqjax.model.typing
from seqjax.util import broadcast_pytree, infer_pytree_shape
from seqjax.inference.embedder import Embedder


class Bijection(eqx.Module):
    # maps from one manifold to another
    @abstractmethod
    def transform_and_lad(
        self, x: Float[Array, "batch_length"]
    ) -> Float[Array, "context_length"]:
        pass

    @abstractmethod
    def inverse_and_lad(
        self, x: Float[Array, "batch_length"]
    ) -> Float[Array, "context_length"]:
        pass


class Identity(Bijection):
    def transform_and_lad(self, x: Float[Array, "num_samples x_dim"]) -> tuple[
        Float[Array, "num_samples x_dim"],
        Float[Array, "num_samples"],
    ]:
        return x, jnp.array(0.0)

    def inverse_and_lad(self, x: Float[Array, "num_samples x_dim"]) -> tuple[
        Float[Array, "num_samples x_dim"],
        Float[Array, "num_samples"],
    ]:
        return x, jnp.array(0.0)


def log_dsftpls(x):
    return -jax.nn.softplus(-x)


class Softplus(Bijection):
    def transform_and_lad(self, x: Float[Array, "num_samples x_dim"]) -> tuple[
        Float[Array, "num_samples x_dim"],
        Float[Array, "num_samples"],
    ]:
        return softplus(x), log_dsftpls(x).sum(axis=1)

    def inverse_and_lad(self, y: Float[Array, "num_samples x_dim"]) -> tuple[
        Float[Array, "num_samples x_dim"],
        Float[Array, "num_samples"],
    ]:
        x = inverse_softplus(y)
        return x, -log_dsftpls(x).sum(axis=1)


class Sigmoid(Bijection):
    upper: float
    lower: float

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def transform_and_lad(
        self, x: jnp.ndarray  # Float[Array, "num_samples x_dim"]
    ) -> tuple[
        jnp.ndarray, jnp.ndarray
    ]:  # (Float[Array, "num_samples x_dim"], Float[Array, "num_samples"])
        sig = 1 / (1 + jnp.exp(-x))

        # rescale
        y = self.lower + (self.upper - self.lower) * sig

        lad = jnp.log(self.upper - self.lower) - softplus(-x) - softplus(x)

        return y, lad.sum(axis=1)

    def inverse_and_lad(
        self, y: jnp.ndarray  # Float[Array, "num_samples x_dim"]
    ) -> tuple[
        jnp.ndarray, jnp.ndarray
    ]:  # (Float[Array, "num_samples x_dim"], Float[Array, "num_samples"])

        # rescale
        sig = (y - self.lower) / (self.upper - self.lower)
        x = jax.scipy.special.logit(sig)
        lad = -jnp.log(self.upper - self.lower) - jnp.log(sig) - jnp.log(1 - sig)

        return x, lad


class Constraint(eqx.Module):
    dim: int
    dim_ix: list[list[int]]
    bijections: list[Bijection]

    def __init__(self, dim: int, dim_ix: list[list[int]], bijections: list[Bijection]):
        self.dim_ix = dim_ix
        self.bijections = bijections
        self.dim = dim
        assert len(dim_ix) == len(bijections)

    # maps from one manifold to another
    def transform_and_lad(self, z: Float[Array, "num_samples x_dim"]) -> tuple[
        Float[Array, "num_samples x_dim"],
        Float[Array, "num_samples"],
    ]:
        x = z
        lad = jnp.zeros(x.shape[0])

        for trans_ix, bijection in zip(self.dim_ix, self.bijections):
            dim_z = z[:, trans_ix]
            dim_x, dim_lad = bijection.transform_and_lad(dim_z)
            x = x.at[:, trans_ix].set(dim_x)
            lad += dim_lad

        return x, lad


class MeanField(eqx.Module):
    theta_dim: int
    loc: Float[Array, "theta_dim"]
    _unc_scale: Float[Array, "theta_dim"]

    def __init__(self, theta_dim):
        self.theta_dim = theta_dim
        self.loc = jnp.zeros(theta_dim)
        self._unc_scale = jnp.ones(theta_dim)

    def sample_and_log_prob(self, num_samples, *, key):
        z = jrandom.normal(key, [num_samples, self.theta_dim])
        scale = 1e-3 + softplus(self._unc_scale)
        x = z * scale + self.loc
        log_q_x = jstats.norm.logpdf(x, loc=self.loc, scale=scale)
        return x, jnp.sum(log_q_x, axis=1)


class ParameterModel(eqx.Module):
    dim: int
    base_flow: MeanField
    constraint: Constraint
    parameter_map: list[str]
    target_parameters: Any

    def sample_struct_and_log_prob(self, key, num_samples):
        z_approx, log_q_z = self.base_flow.sample_and_log_prob(
            num_samples,
            key=key,
        )
        x_approx, lad = self.constraint.transform_and_lad(z_approx)
        return self.array_to_struct(x_approx), log_q_z - lad

    def sample_array_and_log_prob(self, key, num_samples):
        z_approx, log_q_z = self.base_flow.sample_and_log_prob(
            num_samples,
            key=key,
        )
        x_approx, lad = self.constraint.transform_and_lad(z_approx)
        return x_approx, log_q_z - lad

    def array_to_struct(self, theta_array):
        # TODO: this suggests that the final dim of theta should always be the dimension
        # of the parameter.
        # If the parameters are mixed size then we still require some special knowledge to go from flat to
        # struct.
        # Shapes are just tuples, should they be stored on the parameter type?
        # Then, if order is preserved, we get a straightforward relation between flat+unpacked versions.
        param_dict = {
            param: theta_array[..., ix] for ix, param in enumerate(self.parameter_map)
        }
        return self.target_parameters(**param_dict)
