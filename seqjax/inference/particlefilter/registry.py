import typing
from dataclasses import dataclass, field

from seqjax.inference.particlefilter import SMCSampler
from seqjax.inference.particlefilter.resampling import (
    Resampler,
    multinomial_resample_from_log_weights,
    systematic_resample_from_log_weights,
    no_resample,
)
from seqjax.inference.particlefilter.base import (
    AuxiliaryTransitionProposal,
    TransitionProposal,
)
from seqjax.model.interface import SequentialModelProtocol
from seqjax.model import typing as seqjtyping

"""
Filter configurations
"""
ProposalKind = typing.Literal["model-transition", "auxiliary-transition"]

"""
Resampling methods
"""
ResampleKind = typing.Literal["multinomial", "systematic", "none"]

resample_registry: dict[ResampleKind, Resampler] = {
    "multinomial": multinomial_resample_from_log_weights,
    "systematic": systematic_resample_from_log_weights,
    "none": no_resample,
}

"""
Filter
"""
FilterKind = typing.Literal["bootstrap", "auxiliary"]


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


@dataclass
class AuxiliaryFilterConfig[
    FilterLatentHistoryLength: int,
    FilterObservationHistoryLength: int,
]:
    label: FilterKind = field(init=False, default="auxiliary")
    proposal: ProposalKind = field(init=False, default="auxiliary-transition")
    resample: ResampleKind
    num_particles: int
    latent_context_length: FilterLatentHistoryLength | None = None
    observation_context_length: FilterObservationHistoryLength | None = None


registry = {
    "bootstrap": BootstrapFilterConfig,
    "auxiliary": AuxiliaryFilterConfig,
}


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
    config: BootstrapFilterConfig | AuxiliaryFilterConfig,
):
    if not isinstance(config, (BootstrapFilterConfig, AuxiliaryFilterConfig)):
        raise TypeError(
            f"Unsupported particle-filter configuration object: {type(config).__name__}"
        )

    try:
        resampler = resample_registry[config.resample]
    except KeyError as error:
        raise ValueError(f"Unsupported resampler: {config.resample!r}") from error

    if isinstance(config, BootstrapFilterConfig):
        proposal_cls = TransitionProposal
    else:
        proposal_cls = AuxiliaryTransitionProposal

    proposal = proposal_cls(
        target=target_ssm,
        resampler=resampler,
        latent_context_length=target_ssm.latent_context_length,
        observation_context_length=target_ssm.observation_context_length,
    )

    return SMCSampler(
        target=target_ssm,
        proposal=proposal,
        num_particles=config.num_particles,
        latent_context_length=config.latent_context_length,
        observation_context_length=config.observation_context_length,
    )
