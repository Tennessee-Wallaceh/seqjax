import typing
from dataclasses import dataclass, field

from seqjax.inference.particlefilter import SMCSampler
from seqjax.inference.particlefilter import interface as pf_interface
from seqjax.inference.particlefilter.resampling import (
    multinomial_resample_from_log_weights,
    systematic_resample_from_log_weights,
    no_resample,
)
from seqjax.inference.particlefilter.ancestor_selection import (
    LookaheadAncestorSelection,
    ResamplingAncestorSelection,
)
from seqjax.inference.particlefilter.proposals import (
    TransitionProposal,
)
from seqjax.model.interface import SequentialModelProtocol
from seqjax.model import typing as seqjtyping


ResampleKind = typing.Literal["multinomial", "systematic", "none"]
FilterKind = typing.Literal["bootstrap", "auxiliary"]

resample_registry: dict[ResampleKind, pf_interface.Resampler] = {
    "multinomial": multinomial_resample_from_log_weights,
    "systematic": systematic_resample_from_log_weights,
    "none": no_resample,
}


@dataclass
class BootstrapFilterConfig[
    FilterLatentHistoryLength: int,
    FilterObservationHistoryLength: int,
]:
    label: FilterKind = field(init=False, default="bootstrap")
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
    config: (
        BootstrapFilterConfig[
            FilterLatentHistoryLength, FilterObservationHistoryLength
        ]
        | AuxiliaryFilterConfig[
            FilterLatentHistoryLength, FilterObservationHistoryLength
        ]
    ),
) -> SMCSampler[
    ParticleT,
    ObservationT,
    ConditionT,
    ParametersT,
    ModelLatentContextLength,
    ModelObservationContextLength,
    FilterLatentHistoryLength,
    FilterObservationHistoryLength,
]:
    if not isinstance(config, (BootstrapFilterConfig, AuxiliaryFilterConfig)):
        raise TypeError(
            "Unsupported particle-filter configuration object: "
            f"{type(config).__name__}"
        )

    try:
        resampler = resample_registry[config.resample]
    except KeyError as error:
        raise ValueError(f"Unsupported resampler: {config.resample!r}") from error

    proposal = TransitionProposal(
        target=target_ssm,
        latent_context_length=max(1, target_ssm.latent_context_length),
        observation_context_length=target_ssm.observation_context_length,   
    )

    ancestor_selection: pf_interface.AncestorSelection[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        FilterLatentHistoryLength,
        FilterObservationHistoryLength,
    ]
    if isinstance(config, BootstrapFilterConfig):
        ancestor_selection = ResamplingAncestorSelection(
            resampler=resampler,
            latent_context_length=max(1, target_ssm.latent_context_length),
            observation_context_length=target_ssm.observation_context_length,   
        )
    else:
        ancestor_selection = LookaheadAncestorSelection(
            target=target_ssm,
            resampler=resampler,
            latent_context_length=max(1, target_ssm.latent_context_length),
            observation_context_length=target_ssm.observation_context_length,
        )

    return SMCSampler(
        target=target_ssm,
        proposal=proposal,
        ancestor_selection=ancestor_selection,
        num_particles=config.num_particles,
    )