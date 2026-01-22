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

    def from_dict(cls, config_dict):
        return cls(
            **config_dict
        )


def _build_filter(target_posterior, config: BootstrapFilterConfig):
    return SMCSampler(
        target=target_posterior.target,
        proposal=TransitionProposal(target_posterior),
        resampler=multinomial_resample_from_log_weights,
        num_particles=config.num_particles,
    )