import typing
from dataclasses import dataclass, field

from seqjax.inference.particlefilter import SMCSampler
from seqjax.inference.particlefilter.resampling import (
    Resampler,
    multinomial_resample_from_log_weights,
)
from seqjax.inference.particlefilter.base import TransitionProposal
from seqjax.model.interface import SequentialModelProtocol
from seqjax.model import typing as seqjtyping

"""
Filter configurations
"""
ProposalKind = typing.Literal["model-transition"]

"""
Resampling methods
"""
ResampleKind = typing.Literal["multinomial"]

resample_registry: dict[ResampleKind, Resampler] = {
    "multinomial": multinomial_resample_from_log_weights
}

"""
Filter
"""
FilterKind = typing.Literal["bootstrap"]


@dataclass
class BootstrapFilterConfig[
    FilterLatentHistoryLength: int,
    FilterObservationHistoryLength: int,
]:
    label: FilterKind = field(init=False, default="bootstrap")
    proposal: ProposalKind = field(init=False, default="model-transition")
    resample: ResampleKind
    num_particles: int
    latent_context_length: FilterLatentHistoryLength | None = None
    observation_context_length: FilterObservationHistoryLength | None = None


registry = {"bootstrap": BootstrapFilterConfig}


def build_filter[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    ModelLatentContextLength: int,
    ModelObservationContextLength: int,
    FilterLatentHistoryLength: int,
    FilterObservationHistoryLength: int,
](
    target_ssm: SequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        ModelLatentContextLength,
        ModelObservationContextLength,
    ],
    config: BootstrapFilterConfig[
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ],
):
    return SMCSampler(
        target=target_ssm,
        proposal=TransitionProposal(target_ssm),
        resampler=resample_registry[config.resample],
        num_particles=config.num_particles,
        latent_context_length=config.latent_context_length,
        observation_context_length=config.observation_context_length,
    )
