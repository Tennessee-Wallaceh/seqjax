from dataclasses import dataclass
import typing

import jax
from jaxtyping import Array, PyTree

from seqjax.model import interface as model_interface
import seqjax.model.typing as seqjtyping


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FilterContext[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    LatentLengthT: int,
    ObservationLengthT: int,
]:
    """Histories retained by a particle filter.

    Filter histories may be longer than the suffixes required by the model.
    Model and proposal contexts are explicit projections of this context.
    """

    particles: model_interface.LatentContext[ParticleT, LatentLengthT]
    observations: model_interface.ObservedHistoryContext[
        ObservationT,
        ConditionT,
        ObservationLengthT,
    ]

    @property
    def values(self) -> tuple[ParticleT, ...]:
        """Return retained particles for compatibility with history consumers."""

        return self.particles.values

    @property
    def length(self) -> LatentLengthT:
        return self.particles.length

    def __getitem__(self, lag_index: int) -> ParticleT:
        return self.particles[lag_index]

    def with_particles(
        self,
        particles: model_interface.LatentContext[ParticleT, LatentLengthT],
    ) -> typing.Self:
        return type(self)(particles=particles, observations=self.observations)

    def append(
        self,
        particle: ParticleT,
        observation: ObservationT | None = None,
        condition: ConditionT | None = None,
    ) -> typing.Self:
        """Append a particle, and optionally its aligned observed item."""

        if (observation is None) != (condition is None):
            raise ValueError(
                "observation and condition must either both be provided or both omitted"
            )

        observations = self.observations
        if observation is not None and condition is not None:
            observations = observations.append_observation(observation, condition)

        return type(self)(
            particles=self.particles.append(particle),
            observations=observations,
        )

    def latent_context[ContextLengthT: int](
        self,
        length: ContextLengthT,
    ) -> model_interface.LatentContext[ParticleT, ContextLengthT]:
        if length > self.particles.length:
            raise ValueError(
                "Cannot project latent context of length "
                f"{length} from filter latent context of length "
                f"{self.particles.length}"
            )

        values = () if length == 0 else self.particles.values[-length:]
        return model_interface.LatentContext.from_values(*values, length=length)

    def observed_context[ContextLengthT: int](
        self,
        length: ContextLengthT,
    ) -> model_interface.ObservedHistoryContext[
        ObservationT,
        ConditionT,
        ContextLengthT,
    ]:
        if length > self.observations.length:
            raise ValueError(
                "Cannot project observed context of length "
                f"{length} from filter observed context of length "
                f"{self.observations.length}"
            )

        values = () if length == 0 else self.observations.values[-length:]
        return model_interface.ObservedHistoryContext.from_values(
            *values,
            length=length,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FilterData[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    InferenceParameterT: seqjtyping.Parameters,
    FilterLatentHistoryLength: int = int,
    FilterObservationHistoryLength: int = int,
]:
    """Data produced by one filtering step."""

    step_ix: int
    start_log_w: Array
    resampled_log_w: Array
    log_w: Array
    log_z_inc: Array

    particles: FilterContext[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ]
    ancestor_ix: Array
    log_w_inc: Array
    resampled_particles: FilterContext[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ]

    observation: ObservationT
    condition: ConditionT
    inference_parameters: InferenceParameterT


class Recorder(typing.Protocol):
    """Produce a recorded value from the current filtering step."""

    def __call__(self, filter_data: FilterData) -> PyTree: ...
