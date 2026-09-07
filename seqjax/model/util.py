import typing

import seqjax.model.typing as seqjtyping
from . import interface
from seqjax.inference.particlefilter.interface import ProposalContext, FilterContext
    
@typing.overload
def add_history[LatentT: seqjtyping.Latent](
    context: interface.LatentContext[LatentT],
    new_value: LatentT,
) -> interface.LatentContext[LatentT]: ...

@typing.overload
def add_history[ObservationT: seqjtyping.Observation](
    context: interface.ObservationContext[ObservationT],
    new_value: ObservationT,
) -> interface.ObservationContext[ObservationT]: ...

@typing.overload
def add_history[ConditionT: seqjtyping.Condition](
    context: interface.ConditionContext[ConditionT],
    new_value: ConditionT,
) -> interface.ConditionContext[ConditionT]: ...

@typing.overload
def add_history[ParticleT: seqjtyping.Latent](
    context: ProposalContext[ParticleT],
    new_value: ParticleT,
) -> ProposalContext[ParticleT]: ...

@typing.overload
def add_history[ParticleT: seqjtyping.Latent](
    context: FilterContext[ParticleT],
    new_value: ParticleT,
) -> FilterContext[ParticleT]: ...

def add_history(
    context: interface.FixedLengthHistoryContext,
    new_value,
):
    new_history = (*context.values, new_value)
    new_context = new_history[len(new_history) - context.length:]
    return type(context).from_values(*new_context, length=context.length)
    
