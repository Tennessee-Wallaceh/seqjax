import typing
from dataclasses import dataclass, field
from functools import partial

from seqjax.inference.particlefilter import SMCSampler
from seqjax.inference.particlefilter import SMCSampler
from seqjax.inference.particlefilter.resampling import multinomial_resample_from_log_weights
from seqjax.inference.particlefilter.base import TransitionProposal

"""
Filter configurations

smc = SMCSampler(
    target=target_posterior.target,
    proposal=TransitionProposal(target_posterior),
    resampler=conditional_resample,
    num_particles=3000,
)

config = {
    filter="bootstrap",
    resampler="multinomial",
    num_particles=5000
}

"""
ProposalKind = typing.Literal["model-transition"]

"""
Resampling methods
"""
ResampleKind = typing.Literal["multinomial"]

resample_registry = {
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

def _build_filter(target_posterior, config: BootstrapFilterConfig):
    return SMCSampler(
        target=target_posterior.target,
        proposal=TransitionProposal(target_posterior),
        resampler=resample_registry[config.resample],
        num_particles=config.num_particles,
    )