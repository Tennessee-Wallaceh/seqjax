from dataclasses import dataclass
import typing

import jax
from jaxtyping import Array, PyTree, PRNGKeyArray

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
class WeightedPopulation[ContextT]:
    context: ContextT
    normalized_log_weights: Array

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FilterData[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParameterT: seqjtyping.Parameters,
    FilterLatentContextLength: int = int,
    FilterObservationContextLength: int = int,
]:
    """Data encapsulating one filtering step."""

    step_ix: int

    incoming: WeightedPopulation[FilterContext[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentContextLength,
        FilterObservationContextLength
    ]]
    selected: WeightedPopulation[FilterContext[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentContextLength,
        FilterObservationContextLength
    ]]
    updated: WeightedPopulation[FilterContext[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentContextLength,
        FilterObservationContextLength
    ]]

    ancestor_ix: Array # incoming_ix_for_selected
    selection_log_z_adjustment: Array
    log_z_inc: Array

    observation: ObservationT
    condition: ConditionT
    parameters: ParameterT


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AncestorSample:
    ancestor_ix: Array
    normalized_log_weights: Array

class Resampler(typing.Protocol):
    """Return a weighted particle representation of the supplied log probabilities."""

    def __call__(
        self,
        key: PRNGKeyArray,
        normalized_log_weights: Array,
        num_resample: int,
    ) -> AncestorSample: ...


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AncestorSelectionResult[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    FilterLatentContextLength: int,
    FilterObservationContextLength: int,
]:
    selected: WeightedPopulation[FilterContext[
        ParticleT,
        ObservationT,
        ConditionT,
        FilterLatentContextLength,
        FilterObservationContextLength
    ]]
    ancestor_ix: Array
    selection_log_z_adjustment: Array 


class AncestorSelection[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    FilterLatentContextLength: int,
    FilterObservationContextLength: int,
](typing.Protocol):

    @property
    def latent_context_length(self) -> int: ...

    @property
    def observation_context_length(self) -> int: ...

    def __call__(
        self,
        key: PRNGKeyArray,
        incoming: WeightedPopulation[FilterContext[
            ParticleT,
            ObservationT,
            ConditionT,
            FilterLatentContextLength,
            FilterObservationContextLength
        ]],
        observation: ObservationT,
        parameters: ParametersT,
        condition: ConditionT,
        num_particles: int,
    ) -> AncestorSelectionResult[
        ParticleT, ObservationT, ConditionT,
        FilterLatentContextLength, FilterObservationContextLength,
    ]: ...


@dataclass(frozen=True)
class ProposalResult[ParticleT: seqjtyping.Latent]:
    particles: ParticleT
    log_prob: Array

class Proposal[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    FilterLatentHistoryLength: int,
    FilterObservationHistoryLength: int,
](typing.Protocol):
    """
    Draw new latent states and evaluate their conditional proposal density.
    """

    @property
    def latent_context_length(self) -> int: ...

    @property
    def observation_context_length(self) -> int: ...

    def __call__(
        self,
        key: PRNGKeyArray,
        context: FilterContext[
            ParticleT,
            ObservationT,
            ConditionT,
            FilterLatentHistoryLength,
            FilterObservationHistoryLength,
        ],
        observation: ObservationT,
        parameters: ParametersT,
        condition: ConditionT,
        num_particles: int,
    ) -> ProposalResult[
        ParticleT,
    ]: ...



class Recorder(typing.Protocol):
    """Produce a recorded value from the current filtering step."""

    def __call__(self, filter_data: FilterData) -> PyTree: ...
