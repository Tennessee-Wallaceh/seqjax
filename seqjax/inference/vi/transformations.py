from abc import abstractmethod
from typing import Type, Generic, Callable, Optional, Literal, Any
import typing
import equinox as eqx
import jax.scipy.stats as jstats
from jaxtyping import Shaped, Array, Int, Float, PRNGKeyArray
import jax.numpy as jnp
import jax
from jax.nn import softplus
import jax.random as jrandom
import seqjax.model.typing
import operator
import distrax


class Bijector(eqx.Module):
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


class Identity(Bijector):
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


class ConstrainedRQS(Bijector):
    lower: float
    upper: float
    _unc_params: Any

    def __init__(self, num_bins: int, lower: float, upper: float):
        self._unc_params = jnp.zeros((num_bins * 3 + 1))
        self.lower = lower
        self.upper = upper

    def transform_and_lad(self, x: Float[Array, "num_samples x_dim"]) -> tuple[
        Float[Array, "num_samples x_dim"],
        Float[Array, "num_samples"],
    ]:
        return distrax.RationalQuadraticSpline(
            self._unc_params,
            range_min=self.lower,
            range_max=self.upper,
        ).forward_and_log_det(x)

    def inverse_and_lad(self, x: Float[Array, "num_samples x_dim"]) -> tuple[
        Float[Array, "num_samples x_dim"],
        Float[Array, "num_samples"],
    ]:
        return distrax.RationalQuadraticSpline(
            self._unc_params,
            range_min=self.lower,
            range_max=self.upper,
        ).inverse_and_log_det(x)


def log_dsftpls(x):
    return -jax.nn.softplus(-x)


def inverse_softplus(y: jnp.ndarray) -> jnp.ndarray:
    thresh = 20.0
    large = y >= thresh
    y_large = y + jnp.log1p(-jnp.exp(-y))
    y_small = jnp.log(jnp.expm1(y))
    return jnp.where(large, y_large, y_small)


class Softplus(Bijector):
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


class Sigmoid(Bijector):
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

        return y, lad

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


class Chain(Bijector):
    bijectors: typing.Tuple[
        Bijector, ...
    ]  # keep as pytree (not static) so it's trainable

    def __init__(self, bijectors: typing.Sequence[Bijector]):
        # flatten nested Chains to avoid deep Python loops/pytrees
        flat: list[Bijector] = []
        for b in bijectors:
            if isinstance(b, Chain):
                flat.extend(b.bijectors)
            else:
                flat.append(b)
        self.bijectors = tuple(flat)

    def transform_and_lad(
        self, x: Float[Array, "num_samples ..."]
    ) -> typing.Tuple[Float[Array, "num_samples ..."], Float[Array, "num_samples"]]:
        y = x
        shape = x.shape[0] if x.shape else ()
        lad = jnp.zeros(shape, dtype=x.dtype)
        for b in self.bijectors:
            y, inc = b.transform_and_lad(y)
            lad = lad + inc
        return y, lad

    def inverse_and_lad(
        self, y: Float[Array, "num_samples ..."]
    ) -> typing.Tuple[Float[Array, "num_samples ..."], Float[Array, "num_samples"]]:
        x = y
        shape = x.shape[0] if x.shape else ()
        lad = jnp.zeros(shape, dtype=y.dtype)
        for b in reversed(self.bijectors):
            x, inc = b.inverse_and_lad(x)
            lad = lad + inc
        return x, lad


class FieldwiseBijector[TargetStructT: seqjax.model.typing.Packable](eqx.Module):
    target_struct_cls: type[TargetStructT] = eqx.field(static=True)
    field_bijections: dict[str, Bijector]

    # maps from one manifold to another
    def transform_and_lad(
        self,
        z: TargetStructT,
    ) -> tuple[TargetStructT, Float[Array, "batch"]]:
        x = z
        lad = jnp.zeros(x.batch_shape)  # batch dim

        for field, bijection in self.field_bijections.items():
            leaf = getattr(x, field)
            leaf_z, ld = bijection.transform_and_lad(leaf)
            x = eqx.tree_at(operator.attrgetter(field), x, leaf_z)
            lad = lad + ld

        return x, lad


class FieldwiseBijectorFactory[TargetStructT: seqjax.model.typing.Packable](
    typing.Protocol
):
    def __call__(
        self, target_struct_cls: type[TargetStructT]
    ) -> FieldwiseBijector[TargetStructT]: ...
