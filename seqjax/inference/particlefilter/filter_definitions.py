from functools import partial

from jaxtyping import Array

from typing import Callable

from seqjax.model.base import (
    SequentialModel,
)
import seqjax.model.typing as seqjtyping
from .base import SMCSampler, proposal_from_transition
from .resampling import (
    conditional_resample,
    multinomial_resample_from_log_weights,
)


class BootstrapParticleFilter[
    ParticleT: seqjtyping.Latent,
    InitialParticleT: tuple[seqjtyping.Latent, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    SMCSampler[
        ParticleT,
        InitialParticleT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
        ConditionT,
        ParametersT,
    ]
):
    """Classical bootstrap particle filter."""

    def __init__(
        self,
        target: SequentialModel[
            ParticleT,
            InitialParticleT,
            ObservationParticleHistoryT,
            ObservationT,
            ObservationHistoryT,
            ConditionHistoryT,
            ConditionT,
            ParametersT,
        ],
        num_particles: int,
        ess_threshold: float = 0.5,
        target_parameters: Callable = lambda x: x,
    ) -> None:
        super().__init__(
            target=target,
            proposal=proposal_from_transition(target.transition, target_parameters),  # type: ignore[arg-type]
            resampler=partial(
                conditional_resample,
                resampler=multinomial_resample_from_log_weights,
                esse_threshold=ess_threshold,
            ),
            num_particles=num_particles,
        )


class AuxiliaryParticleFilter[
    ParticleT: seqjtyping.Latent,
    InitialParticleT: tuple[seqjtyping.Latent, ...],
    TransitionParticleHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationParticleHistoryT: tuple[seqjtyping.Latent, ...],
    ObservationT: seqjtyping.Observation,
    ObservationHistoryT: tuple[seqjtyping.Observation, ...],
    ConditionHistoryT: tuple[seqjtyping.Condition, ...],
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    SMCSampler[
        ParticleT,
        InitialParticleT,
        ObservationParticleHistoryT,
        ObservationT,
        ObservationHistoryT,
        ConditionHistoryT,
        ConditionT,
        ParametersT,
    ]
):
    """Bootstrap particle filter with auxiliary resampling."""

    def __init__(
        self,
        target: SequentialModel[
            ParticleT,
            InitialParticleT,
            ObservationParticleHistoryT,
            ObservationT,
            ObservationHistoryT,
            ConditionHistoryT,
            ConditionT,
            ParametersT,
        ],
        num_particles: int,
    ) -> None:
        super().__init__(
            target=target,
            proposal=proposal_from_transition(target.transition),  # type: ignore[arg-type]
            resampler=partial(
                conditional_resample,
                resampler=multinomial_resample_from_log_weights,
                esse_threshold=0.5,
            ),
            num_particles=num_particles,
        )

    def _resample_log_weights(
        self,
        log_w: Array,
        particles: tuple[ParticleT, ...],
        observation_history: tuple[ObservationT, ...],
        observation: ObservationT,
        condition: ConditionT,
        params: ParametersT,
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
