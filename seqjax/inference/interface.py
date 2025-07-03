from __future__ import annotations

from typing import Any, Protocol
from jaxtyping import PRNGKeyArray

from seqjax.model.base import (
    SequentialModel,
    ParticleType,
    ObservationType,
    ConditionType,
    ParametersType,
    ParameterPrior,
)
from seqjax.model.typing import Batched, SequenceAxis, HyperParametersType


class InferenceMethod(Protocol):
    """Callable protocol for Bayesian inference routines.

    This interface represents the minimal call signature shared by the
    Bayesian samplers provided in :mod:`seqjax`. Concrete inference functions
    such as :func:`~seqjax.inference.mcmc.run_nuts` should be partially applied
    with any algorithm-specific configuration before being used through this
    protocol.
    """

    def __call__(
        self,
        target: SequentialModel[
            ParticleType, ObservationType, ConditionType, ParametersType
        ],
        key: PRNGKeyArray,
        observations: Batched[ObservationType, SequenceAxis],
        *,
        parameter_prior: (
            ParameterPrior[ParametersType, HyperParametersType] | None
        ) = None,
        condition_path: Batched[ConditionType, SequenceAxis] | None = None,
        initial_latents: Batched[ParticleType, SequenceAxis] | None = None,
        hyper_parameters: HyperParametersType | None = None,
        initial_conditions: tuple[ConditionType, ...] | None = None,
        observation_history: tuple[ObservationType, ...] | None = None,
    ) -> Any: ...


class LatentInferenceMethod(Protocol):
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
        key: PRNGKeyArray,
        observations: Batched[ObservationType, SequenceAxis],
        *,
        parameters: ParametersType,
        condition_path: Batched[ConditionType, SequenceAxis] | None = None,
        initial_latents: Batched[ParticleType, SequenceAxis] | None = None,
        initial_conditions: tuple[ConditionType, ...] | None = None,
        observation_history: tuple[ObservationType, ...] | None = None,
    ) -> Any: ...
