"""Simple stochastic volatility model variants."""

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
from seqjax.model.typing import HyperParameters

from .common import (
    LVolStd,
    LatentVol,
    LogReturnObs,
    LogVolRW,
    StdLogVolPrior,
    StochVolParamPrior,
    TimeIncrement,
    lv_to_std_only,
)


class SimpleStochasticVol(
    SequentialModelBase[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolRW,
    ]
):
    latent_cls = LatentVol
    observation_cls = LogReturnObs
    condition_cls = TimeIncrement
    parameter_cls = LogVolRW

    prior_order = 1
    transition_order = 1
    emission_order = 1
    observation_dependency = 0

    def prior_sample(
        self,
        key: PRNGKeyArray,
        conditions: ConditionContext[TimeIncrement],
        parameters: LogVolRW,
    ) -> LatentContext[LatentVol]:
        del key, conditions
        return self.make_latent_context(LatentVol(log_vol=jnp.log(parameters.long_term_vol)))

    def prior_log_prob(
        self,
        latent: LatentContext[LatentVol],
        conditions: ConditionContext[TimeIncrement],
        parameters: LogVolRW,
    ) -> Scalar:
        del conditions
        prior_mean = jnp.log(parameters.long_term_vol)
        return jstats.norm.logpdf(latent[-1].log_vol, loc=prior_mean, scale=0.1)

    def transition_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> LatentVol:
        reversion = jnp.exp(-parameters.mean_reversion * condition.dt)
        loc = (1 - reversion) * jnp.log(parameters.long_term_vol) + reversion * latent_history[-1].log_vol
        scale = parameters.std_log_vol * jnp.sqrt(condition.dt)
        return LatentVol(log_vol=loc + scale * jrandom.normal(key))

    def transition_log_prob(
        self,
        latent_history: LatentContext[LatentVol],
        latent: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Scalar:
        reversion = jnp.exp(-parameters.mean_reversion * condition.dt)
        loc = (1 - reversion) * jnp.log(parameters.long_term_vol) + reversion * latent_history[-1].log_vol
        scale = parameters.std_log_vol * jnp.sqrt(condition.dt)
        return jstats.norm.logpdf(latent.log_vol, loc=loc, scale=scale)

    def emission_sample(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentVol],
        observation_history: ObservationContext[LogReturnObs],
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> LogReturnObs:
        del observation_history, condition, parameters
        return_scale = jnp.exp(latent_history[-1].log_vol)
        return LogReturnObs(log_return=jrandom.normal(key) * return_scale)

    def emission_log_prob(
        self,
        latent_history: LatentContext[LatentVol],
        observation: LogReturnObs,
        observation_history: ObservationContext[LogReturnObs],
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Scalar:
        del observation_history, condition, parameters
        return_scale = jnp.exp(latent_history[-1].log_vol)
        return jstats.norm.logpdf(observation.log_return, loc=0.0, scale=return_scale)


class SimpleStochasticVolBayesianStdLogVol(
    BayesianSequentialModel[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolRW,
        LVolStd,
        HyperParameters,
    ]
):
    def __init__(self, ref_params: LogVolRW, hyperparameters: HyperParameters | None = None):
        super().__init__(hyperparameters=hyperparameters)
        self.convert_to_model_parameters = staticmethod(
            partial(lv_to_std_only, ref_params=ref_params)
        )

    inference_parameter_cls = LVolStd
    target = SimpleStochasticVol()
    def parameter_prior(self) -> StdLogVolPrior:
        return StdLogVolPrior()


class SimpleStochasticVolBayesian(
    BayesianSequentialModel[
        LatentVol,
        LogReturnObs,
        TimeIncrement,
        LogVolRW,
        LogVolRW,
        HyperParameters,
    ]
):
    def __init__(self, hyperparameters: HyperParameters | None = None):
        super().__init__(hyperparameters=hyperparameters)

    inference_parameter_cls = LogVolRW
    target = SimpleStochasticVol()
    def parameter_prior(self) -> StochVolParamPrior:
        return StochVolParamPrior()
    convert_to_model_parameters = staticmethod(lambda parameters: parameters)
