import typing
from dataclasses import dataclass, field

from seqjax.inference.particlefilter import SMCSampler
from seqjax.inference.particlefilter.resampling import Resampler, multinomial_resample_from_log_weights
from seqjax.inference.particlefilter.base import TransitionProposal
from seqjax.model.interface import (
    BayesianSequentialModelProtocol,
)
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
class BootstrapFilterConfig:
    label: FilterKind = field(init=False, default="bootstrap")
    proposal: ProposalKind = field(init=False, default="model-transition")
    resample: ResampleKind
    num_particles: int

registry = {
    "bootstrap": BootstrapFilterConfig
}

def build_filter[
    ParticleT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
](
    target_posterior: BayesianSequentialModelProtocol[
        ParticleT,
        ObservationT,
        ConditionT,
        ParametersT,
        InferenceParametersT,
        typing.Any,
    ], 
    config: BootstrapFilterConfig
):
    return SMCSampler(
        target=target_posterior.target,
        proposal=TransitionProposal(target_posterior),
        resampler=resample_registry[config.resample],
        num_particles=config.num_particles,
        parameterization=target_posterior.parameterization
    )