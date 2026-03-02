"""Stochastic volatility models in log-variance form."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from seqjax.model.base import (
    BayesianSequentialModel,
    SequentialModelBase,
    LatentContext,
    ObservationContext,
    ConditionContext,
)
from seqjax.model.typing import HyperParameters, NoCondition

from .common import (
    LatentVar,
    LogReturnObs,
    LogVarAR,
    LogVarParams,
    LogVarStd,
    StochVarARPrior,
    StochVarFullPrior,
    StochVarPrior,
    lvar_from_ar_only,
    lvar_from_std_only,
)


class SimpleStochasticVar(
    SequentialModelBase[
        LatentVar,
        LogReturnObs,
        NoCondition,
        LogVarParams,
    ]
):
    latent_cls = LatentVar
    observation_cls = LogReturnObs
    condition_cls = NoCondition
    parameter_cls = LogVarParams

    prior_order = 2
    transition_order = 2
    emission_order = 2
    observation_dependency = 0

    def prior_sample(
        self,
        key: PRNGKeyArray,
        conditions: ConditionContext[NoCondition],
        parameters: LogVarParams,
    ) -> LatentContext[LatentVar]:
        del conditions
        key_1, key_2 = jrandom.split(key)
        first = LatentVar(log_var=parameters.long_term_log_var)
        second = LatentVar(
            log_var=parameters.long_term_log_var
            + parameters.std_log_var * jrandom.normal(key_2)
        )
        _ = key_1
        return self.make_latent_context(first, second)

    def prior_log_prob(
        self,
        latent: LatentContext[LatentVar],
        conditions: ConditionContext[NoCondition],
        parameters: LogVarParams,
    ) -> Scalar:
        del conditions
        first, second = latent[-2], latent[-1]
        first_log = jnp.where(
            jnp.isclose(first.log_var, parameters.long_term_log_var),
            0.0,
            -jnp.inf,
        )
        second_log = jstats.norm.logpdf(
            second.log_var,
            loc=parameters.long_term_log_var,
            scale=parameters.std_log_var,
        )
        return first_log + second_log

    def transition_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentVar],
        condition: NoCondition,
        parameters: LogVarParams,
    ) -> LatentVar:
        del condition
        current = latent_history[-1]
        loc = parameters.long_term_log_var + parameters.ar * (
            current.log_var - parameters.long_term_log_var
        )
        return LatentVar(log_var=loc + parameters.std_log_var * jrandom.normal(key))

    def transition_log_prob(
        self,
        latent_history: LatentContext[LatentVar],
        latent: LatentVar,
        condition: NoCondition,
        parameters: LogVarParams,
    ) -> Scalar:
        del condition
        current = latent_history[-1]
        loc = parameters.long_term_log_var + parameters.ar * (
            current.log_var - parameters.long_term_log_var
        )
        return jstats.norm.logpdf(
            latent.log_var,
            loc=loc,
            scale=parameters.std_log_var,
        )

    def emission_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentVar],
        observation_history: ObservationContext[LogReturnObs],
        condition: NoCondition,
        parameters: LogVarParams,
    ) -> LogReturnObs:
        del observation_history, condition
        previous_latent, current_latent = latent_history[-2], latent_history[-1]
        current_var = jnp.exp(current_latent.log_var / 2)
        previous_var = jnp.exp(previous_latent.log_var / 2)
        mean = (
            parameters.ar
            * (current_latent.log_var - parameters.long_term_log_var)
            * previous_var
        )
        scale = current_var * jnp.sqrt(1 - parameters.ar**2)
        return LogReturnObs(log_return=mean + scale * jrandom.normal(key))

    def emission_log_prob(
        self,
        latent_history: LatentContext[LatentVar],
        observation: LogReturnObs,
        observation_history: ObservationContext[LogReturnObs],
        condition: NoCondition,
        parameters: LogVarParams,
    ) -> Scalar:
        del observation_history, condition
        previous_latent, current_latent = latent_history[-2], latent_history[-1]
        current_var = jnp.exp(current_latent.log_var / 2)
        previous_var = jnp.exp(previous_latent.log_var / 2)
        mean = (
            parameters.ar
            * (current_latent.log_var - parameters.long_term_log_var)
            * previous_var
        )
        scale = current_var * jnp.sqrt(1 - parameters.ar**2)
        return jstats.norm.logpdf(observation.log_return, loc=mean, scale=scale)


class SimpleStochasticVarBayesian(
    BayesianSequentialModel[
        LatentVar,
        LogReturnObs,
        NoCondition,
        LogVarParams,
        LogVarStd,
        HyperParameters,
    ]
):
    def __init__(self, ref_params: LogVarParams, hyperparameters: HyperParameters | None = None):
        super().__init__(hyperparameters=hyperparameters)
        self.convert_to_model_parameters = staticmethod(
            partial(lvar_from_std_only, ref_params=ref_params)
        )

    inference_parameter_cls = LogVarStd
    target = SimpleStochasticVar()
    def parameter_prior(self) -> StochVarPrior:
        return StochVarPrior()


class ARStochasticVarBayesian(
    BayesianSequentialModel[
        LatentVar,
        LogReturnObs,
        NoCondition,
        LogVarParams,
        LogVarAR,
        HyperParameters,
    ]
):
    def __init__(self, ref_params: LogVarParams, hyperparameters: HyperParameters | None = None):
        super().__init__(hyperparameters=hyperparameters)
        self.convert_to_model_parameters = staticmethod(
            partial(lvar_from_ar_only, ref_params=ref_params)
        )

    inference_parameter_cls = LogVarAR
    target = SimpleStochasticVar()
    def parameter_prior(self) -> StochVarARPrior:
        return StochVarARPrior()


class StochasticVarBayesian(
    BayesianSequentialModel[
        LatentVar,
        LogReturnObs,
        NoCondition,
        LogVarParams,
        LogVarParams,
        HyperParameters,
    ]
):
    def __init__(self, hyperparameters: HyperParameters | None = None):
        super().__init__(hyperparameters=hyperparameters)

    inference_parameter_cls = LogVarParams
    target = SimpleStochasticVar()
    def parameter_prior(self) -> StochVarFullPrior:
        return StochVarFullPrior()
    convert_to_model_parameters = staticmethod(lambda parameters: parameters)
