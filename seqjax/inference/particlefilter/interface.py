from dataclasses import dataclass
from typing import Protocol, Any, overload

import jax
from jaxtyping import Array

from seqjax.model.interface import (
    FixedLengthHistoryContext
)
import seqjax.model.typing as seqjtyping

@jax.tree_util.register_dataclass
class ProposalContext[ParticleT: seqjtyping.Latent](
    FixedLengthHistoryContext[ParticleT],
):
    """Concrete observation history context."""

@jax.tree_util.register_dataclass
class FilterContext[ParticleT: seqjtyping.Latent](
    FixedLengthHistoryContext[ParticleT],
):
    """Concrete observation history context."""

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FilterData[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    InferenceParameterT: seqjtyping.Parameters,
]:
    """
    Encapsulates the data arising from a filter step
    """
    step_ix: int
    start_log_w: Array
    resampled_log_w: Array
    log_w: Array
    log_z_inc: Array

    particles: FilterContext[ParticleT]
    ancestor_ix: Array
    log_w_inc: Array
    resampled_particles: FilterContext[ParticleT]

    observation: ObservationT
    condition: ConditionT
    inference_parameters: InferenceParameterT

class Recorder(Protocol):
    """
    The filtering recorder is just a function producing some output from the current filter data.
    """

    def __call__(self, filter_data: FilterData) -> Any: ...
