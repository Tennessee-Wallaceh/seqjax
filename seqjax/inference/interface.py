import typing
import jaxtyping

from seqjax.model.base import (
    SequentialModel,
    ParticleType,
    ObservationType,
    ConditionType,
    ParametersType,
    BayesianSequentialModel,
    InferenceParametersType,
)
from seqjax.model.typing import HyperParametersType


class InferenceMethod(typing.Protocol):
    """Callable protocol for Bayesian inference routines.

    This interface represents the minimal call signature shared by the
    Bayesian samplers provided in :mod:`seqjax`. Concrete inference functions
    such as :func:`~seqjax.inference.mcmc.run_nuts` should be partially applied
    with any algorithm-specific configuration before being used through this
    protocol.
    """

    def __call__(
        self,
        target_posterior: BayesianSequentialModel[
            ParticleType,
            ObservationType,
            ConditionType,
            ParametersType,
            InferenceParametersType,
            HyperParametersType,
        ],
        hyperparameters: HyperParametersType,
        key: jaxtyping.PRNGKeyArray,
        observation_path: ObservationType,
        condition_path: ConditionType,
        test_samples: int,
        config: typing.Any,
    ) -> tuple[InferenceParametersType, typing.Any]: ...


def inference_method(f: InferenceMethod) -> InferenceMethod:
    """Adding this decorator ensures that an intended inference method satisfies the protocol"""
    return f


class LatentInferenceMethod(typing.Protocol):
    """Callable protocol for latent path inference routines.

    This interface covers samplers that condition on known model parameters and
    return samples from the posterior ``p(x | y, \theta)``. Concrete inference
    functions such as :func:`~seqjax.inference.mcmc.run_nuts` should be
    partially applied with any algorithm-specific configuration before being
    used through this protocol.
    """

    def __call__(
        self,
        target: SequentialModel[
            ParticleType, ObservationType, ConditionType, ParametersType
        ],
        key: jaxtyping.PRNGKeyArray,
        observations: ObservationType,
        *,
        parameters: ParametersType,
        condition_path: ConditionType | None = None,
        initial_latents: ParticleType | None = None,
        initial_conditions: tuple[ConditionType, ...] | None = None,
        observation_history: tuple[ObservationType, ...] | None = None,
    ) -> typing.Any: ...
