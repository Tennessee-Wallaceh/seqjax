from __future__ import annotations

from functools import partial

from jaxtyping import Array

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
)

from .base import SMCSampler, proposal_from_transition
from .resampling import (
    conditional_resample,
    multinomial_resample_from_log_weights,
    gumbel_resample_from_log_weights,
)


class BootstrapParticleFilter(
    SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType]
):
    """Classical bootstrap particle filter."""

    def __init__(
        self,
        target: SequentialModel[
            ParticleType, ObservationType, ConditionType, ParametersType
        ],
        num_particles: int,
        ess_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            target=target,
            proposal=proposal_from_transition(target.transition),  # type: ignore[arg-type]
            resampler=partial(
                conditional_resample,
                resampler=multinomial_resample_from_log_weights,
                esse_threshold=ess_threshold,
            ),
            num_particles=num_particles,
        )


class AuxiliaryParticleFilter(
    SMCSampler[ParticleType, ObservationType, ConditionType, ParametersType]
):
    """Bootstrap particle filter with auxiliary resampling."""

    def __init__(
        self,
        target: SequentialModel[
            ParticleType, ObservationType, ConditionType, ParametersType
        ],
        num_particles: int,
    ) -> None:
        super().__init__(
            target=target,
            proposal=proposal_from_transition(target.transition),  # type: ignore[arg-type]
            resampler=partial(
                conditional_resample,
                resampler=gumbel_resample_from_log_weights,
                esse_threshold=0.5,
            ),
            num_particles=num_particles,
        )

    def _resample_log_weights(
        self,
        log_w: Array,
        particles: tuple[ParticleType, ...],
        observation_history: tuple[ObservationType, ...],
        observation: ObservationType,
        condition: ConditionType,
        params: ParametersType,
    ) -> Array:
        obs_hist = (
            observation_history[-self.target.emission.observation_dependency :]
            if self.target.emission.observation_dependency > 0
            else ()
        )
        resample_particles = particles[-self.target.emission.order :]
        return log_w + self.emission_logp(
            resample_particles, obs_hist, observation, condition, params
        )
