from flowjax.bijections import Affine, AbstractBijection
from flowjax.bijections.masked_autoregressive import MaskedAutoregressive as FlowjaxMAF
from flowjax.distributions import (
    Normal as FlowjaxNormal,
    Transformed as FlowjaxTransformed,
)
from flowjax.flows import masked_autoregressive_flow
import equinox as eqx
import seqjax.model.typing as seqjtyping
from seqjax.model.base import SequentialModel
from seqjax.model import simulate, evaluate

import jaxtyping
import jax
import jax.numpy as jnp
import jax.random as jrandom
import typing

from .api import Embedder, LatentContext, VariationalApproximationFactory, UnconditionalVariationalApproximation, AmortizedVariationalApproximation


def _ensure_legacy_key(key: jaxtyping.PRNGKeyArray) -> jaxtyping.PRNGKeyArray:
    """Convert legacy uint32 keys to typed JAX keys for FlowJax."""
    return jrandom.wrap_key_data(jnp.asarray(key, dtype=jnp.uint32))

class MaskedAutoregressiveFlow[
    TargetStructT: seqjtyping.Packable,
](UnconditionalVariationalApproximation[TargetStructT]):
    """Masked autoregressive flow over the flattened parameter space."""

    target_struct_cls: type[TargetStructT]
    base_distribution: FlowjaxNormal
    flow: FlowjaxMAF
    distribution: FlowjaxTransformed

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        super().__init__(target_struct_cls, shape=(target_struct_cls.flat_dim,))
        self.target_struct_cls = target_struct_cls
        dim = target_struct_cls.flat_dim

        loc = jnp.broadcast_to(jnp.asarray(base_loc), (dim,))
        scale = jnp.broadcast_to(jnp.asarray(base_scale), (dim,))
        self.base_distribution = FlowjaxNormal(loc, scale)

        if transformer is None:
            transformer = Affine()

        flow_key = _ensure_legacy_key(key)
        self.flow = FlowjaxMAF(
            flow_key,
            transformer=transformer,
            dim=dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
        )
        distribution = typing.cast(
            FlowjaxTransformed,
            FlowjaxTransformed(self.base_distribution, self.flow),  # type: ignore[arg-type, call-arg]
        )
        self.distribution = distribution

    def sample_and_log_prob(self, key, condition=None):
        flow_key = _ensure_legacy_key(key)
        flat_sample = self.distribution.sample(flow_key)
        log_q = self.distribution.log_prob(flat_sample)
        return self.target_struct_cls.unravel(flat_sample), log_q


class MaskedAutoregressiveFlowFactory[TargetStructT: seqjtyping.Packable](
    VariationalApproximationFactory[TargetStructT, None]
):
    def __init__(
        self,
        *,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        self._key = key
        self._nn_width = nn_width
        self._nn_depth = nn_depth
        self._base_loc = base_loc
        self._base_scale = base_scale
        self._transformer = transformer

    def __call__(
        self, target_struct_cls: type[TargetStructT]
    ) -> MaskedAutoregressiveFlow[TargetStructT]:
        return MaskedAutoregressiveFlow(
            target_struct_cls,
            key=self._key,
            nn_width=self._nn_width,
            nn_depth=self._nn_depth,
            base_loc=self._base_loc,
            base_scale=self._base_scale,
            transformer=self._transformer,
        )


class AmortizedMAF[
    TargetStructT: seqjtyping.Packable,
](AmortizedVariationalApproximation[TargetStructT]):
    """Conditional masked autoregressive flow over buffered latent paths."""

    
    distribution: FlowjaxTransformed
    _flat_sample_dim: int = eqx.field(static=True)

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        sample_length: int,
        embedder: Embedder,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        flow_layers: int = 1,
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:
        
        shape = (sample_length, target_struct_cls.flat_dim)
        super().__init__(
            target_struct_cls,
            shape=shape,
            sample_length=sample_length,
        )
        self.target_struct_cls = target_struct_cls
        flat_sample_dim = sample_length * target_struct_cls.flat_dim
        self._flat_sample_dim = flat_sample_dim

        if transformer is None:
            transformer = Affine()

        loc = jnp.broadcast_to(jnp.asarray(base_loc), (flat_sample_dim,))
        scale = jnp.broadcast_to(jnp.asarray(base_scale), (flat_sample_dim,))

        cond_input_dim = (
            embedder.parameter_context_dim
            + embedder.condition_context_dim
            + embedder.embedded_context_dim
        )

        self.distribution = masked_autoregressive_flow(
            key,
            base_dist=FlowjaxNormal(loc, scale),
            flow_layers=flow_layers,
            transformer=transformer,
            cond_dim=cond_input_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            invert=False
        )
        

    def _build_condition(
        self,
        condition: LatentContext,
    ) -> jaxtyping.Array:
        return jnp.concatenate(
            [
                condition.parameter_context.ravel().flatten(), 
                condition.condition_context.ravel().flatten(), 
                condition.embedded_context, 
            ], axis=0
        )

    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: LatentContext,
    ) -> tuple[TargetStructT, jaxtyping.Scalar]:
        
        cond = self._build_condition(condition)
        flat_sample, log_q = self.distribution.sample_and_log_prob(key, condition=cond)
        reshaped_sample = jnp.reshape(flat_sample, self.shape)
        latent_sample = typing.cast(
            TargetStructT, self.target_struct_cls.unravel(reshaped_sample)
        )
        return latent_sample, log_q


class AmortizedLatentPriorMAF[
    TargetStructT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters  
](AmortizedVariationalApproximation[TargetStructT]):
    """Conditional masked autoregressive flow over buffered latent paths."""

    target_struct_cls: type[TargetStructT]
    target_ssm: SequentialModel[
        TargetStructT,
        ObservationT,
        ConditionT,
        ParametersT  
    ]
    transform: AbstractBijection
    _flat_sample_dim: int = eqx.field(static=True)
    _condition_dim: int = eqx.field(static=True)
    _parameter_dim: int = eqx.field(static=True)
    _context_dim: int = eqx.field(static=True)

    def __init__(
        self,
        target_ssm: SequentialModel[
            TargetStructT,
            ObservationT,
            ConditionT,
            ParametersT  
        ],
        *,
        buffer_length: int,
        batch_length: int,
        context_dim: int,
        parameter_dim: int,
        condition_dim: int,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        flow_layers: int = 1,
    ) -> None:
        self.target_ssm = target_ssm
        target_struct_cls = target_ssm.latent_cls
        sample_length = 2 * buffer_length + batch_length
        shape = (sample_length, target_struct_cls.flat_dim)
        super().__init__(
            target_struct_cls,
            shape=shape,
            batch_length=batch_length,
            buffer_length=buffer_length,
        )
        self.target_struct_cls = target_struct_cls
        flat_sample_dim = sample_length * target_struct_cls.flat_dim
        self._flat_sample_dim = flat_sample_dim
        self._parameter_dim = parameter_dim
        self._context_dim = context_dim
        self._condition_dim = condition_dim

        transformer = Affine()

        cond_input_dim = parameter_dim + context_dim + condition_dim

        # use flowjax constructor, but throw away the base dist
        self.transform = masked_autoregressive_flow(
            key,
            base_dist=FlowjaxNormal(
                jnp.zeros(flat_sample_dim), 
                jnp.ones(flat_sample_dim)
            ),
            flow_layers=flow_layers,
            transformer=transformer,
            cond_dim=cond_input_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            invert=False
        ).bijection
        

    def _flatten_condition(
        self,
        condition: LatentContext,
    ) -> jaxtyping.Array:
        theta_flat = condition.parameter_context.ravel()
        obs_flat = condition.observation_context.ravel().flatten()
        cond_flat = condition.condition_context.ravel()
        return jnp.concatenate(
            [theta_flat, obs_flat, cond_flat], axis=0
        )

    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: LatentContext,
    ) -> tuple[TargetStructT, jaxtyping.Scalar]:
        
        z_path, _ = simulate.simulate(
            key,
            self.target_ssm,
            condition.parameter_context,
            sequence_length=self._flat_sample_dim,
        )

        log_p_z = evaluate.log_prob_x(
            self.target_ssm,
            z_path,
            condition.condition_context,
            condition.parameter_context,
        )

        cond = self._flatten_condition(condition)
        flat_z = z_path.ravel().flatten()
        flat_sample, log_det = self.transform.transform_and_log_det(flat_z, condition=cond)
        reshaped_sample = jnp.reshape(flat_sample, self.shape)
        latent_sample = typing.cast(
            TargetStructT, self.target_struct_cls.unravel(reshaped_sample)
        )
        return latent_sample, log_p_z - log_det


class AmortizedARMAF[
    TargetStructT: seqjtyping.Packable,
](AmortizedVariationalApproximation[TargetStructT]):
    """Conditional masked autoregressive flow over buffered latent paths,
    with an AR(p) final transform (global AR coefficients) along time.
    """

    target_struct_cls: type[TargetStructT]
    distribution: FlowjaxTransformed
    _flat_sample_dim: int = eqx.field(static=True)
    _condition_dim: int = eqx.field(static=True)
    _parameter_dim: int = eqx.field(static=True)
    _context_dim: int = eqx.field(static=True)

    ar_order: int = eqx.field(static=True)
    ar_raw: jaxtyping.Array  # shape (p,)

    def __init__(
        self,
        target_struct_cls: type[TargetStructT],
        *,
        buffer_length: int,
        batch_length: int,
        context_dim: int,
        parameter_dim: int,
        condition_dim: int,
        key: jaxtyping.PRNGKeyArray,
        nn_width: int,
        nn_depth: int,
        flow_layers: int = 1,
        ar_order: int = 1,
        ar_phi_init: jaxtyping.Array | None = None,  # optional init, shape (p,)
        base_loc: jaxtyping.Array | float = 0.0,
        base_scale: jaxtyping.Array | float = 1.0,
        transformer: AbstractBijection | None = None,
    ) -> None:

        sample_length = 2 * buffer_length + batch_length
        shape = (sample_length, target_struct_cls.flat_dim)
        super().__init__(
            target_struct_cls,
            shape=shape,
            batch_length=batch_length,
            buffer_length=buffer_length,
        )
        self.target_struct_cls = target_struct_cls
        flat_sample_dim = sample_length * target_struct_cls.flat_dim
        self._flat_sample_dim = flat_sample_dim
        self._parameter_dim = parameter_dim
        self._context_dim = context_dim
        self._condition_dim = condition_dim

        self.ar_order = int(ar_order)
        if self.ar_order <= 0:
            raise ValueError(f"Expected ar_order >= 1, got {ar_order}.")

        # AR params: raw -> phi. Default uses tanh to keep each coefficient bounded.
        # NOTE: tanh does NOT guarantee AR(p) stability for p>1.
        if ar_phi_init is None:
            # mild init near 0 by default (stable-ish, not near unit-root)
            self.ar_raw = jnp.zeros((self.ar_order,))
        else:
            ar_phi_init = jnp.asarray(ar_phi_init)
            if ar_phi_init.shape != (self.ar_order,):
                raise ValueError(
                    f"Expected ar_phi_init.shape == ({self.ar_order},), got {ar_phi_init.shape}."
                )
            # inverse of tanh: arctanh, with clipping for numerical safety
            clipped = jnp.clip(ar_phi_init, -0.999, 0.999)
            self.ar_raw = jnp.arctanh(clipped)

        if transformer is None:
            transformer = Affine()

        loc = jnp.broadcast_to(jnp.asarray(base_loc), (flat_sample_dim,))
        scale = jnp.broadcast_to(jnp.asarray(base_scale), (flat_sample_dim,))

        cond_input_dim = parameter_dim + sample_length * (context_dim + condition_dim)
        print("COND INPUT:", cond_input_dim)
        print("parameter_dim INPUT:", parameter_dim)
        print("context_dim INPUT:", context_dim)
        print("condition_dim INPUT:", condition_dim)
        key = _ensure_legacy_key(key)

        self.distribution = masked_autoregressive_flow(
            key,
            base_dist=FlowjaxNormal(loc, scale),
            flow_layers=flow_layers,
            transformer=transformer,
            cond_dim=cond_input_dim,
            nn_width=nn_width,
            nn_depth=nn_depth,
            invert=False,
        )

    def _build_condition(
        self,
        condition: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array],
    ) -> jaxtyping.Array:
        theta_context, observation_context, condition_context = condition
        print("theta_context INPUT:", theta_context.shape)
        print("observation_context INPUT:", observation_context.shape)
        print("condition_context INPUT:", condition_context.shape)
        theta_flat = jnp.ravel(theta_context[0])  # only need 1 theta
        obs_flat = jnp.ravel(observation_context)
        cond_flat = jnp.ravel(condition_context)
        return jnp.concatenate([theta_flat, obs_flat, cond_flat], axis=0)

    def _ar_phi(self) -> jaxtyping.Array:
        # bounded coefficients; does not enforce stationarity for p>1
        return jnp.tanh(self.ar_raw)

    def _ar_integrate(self, eps: jaxtyping.Array) -> jaxtyping.Array:
        """eps -> x via causal AR(p) recursion along time.
        eps shape: (T, D). Returns x shape: (T, D).
        Uses zero padding for x_{t-k} when t-k < 0.
        """
        phi = self._ar_phi()  # (p,)
        p = self.ar_order
        T, D = eps.shape

        def step(carry, eps_t):
            # carry holds [x_{t-1}, x_{t-2}, ..., x_{t-p}] in carry[0], carry[1], ...
            ar_part = jnp.tensordot(phi, carry, axes=(0, 0))  # (D,)
            x_t = ar_part + eps_t
            new_carry = jnp.concatenate([x_t[None, :], carry[:-1, :]], axis=0)
            return new_carry, x_t

        init_carry = jnp.zeros((p, D), dtype=eps.dtype)
        _, x = jax.lax.scan(step, init_carry, eps, length=T)
        return x

    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: tuple[jaxtyping.Array, jaxtyping.Array, jaxtyping.Array],
    ) -> tuple[TargetStructT, jaxtyping.Scalar]:
        cond = self._build_condition(condition)
        flow_key = _ensure_legacy_key(key)

        # Interpret flow output as innovations eps (flat), with log_q(eps).
        flat_sample, log_q = self.distribution.sample_and_log_prob(flow_key, condition=cond)

        reshaped_sample = jnp.reshape(flat_sample, self.shape)  # (T, D)

        # Final AR(p) transform: x = AR_integrate(eps).
        # For additive AR(p), logdet = 0 => log_q(x) == log_q(eps).
        reshaped_sample = self._ar_integrate(reshaped_sample)

        latent_sample = typing.cast(
            TargetStructT, self.target_struct_cls.unravel(reshaped_sample)
        )
        return latent_sample, log_q