import typing

import seqjax.model.typing as seqjtyping
from seqjax import util
from . import interface
from seqjax.inference.particlefilter.interface import ProposalContext, FilterContext

def slice_prior_context[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    model: interface.SequentialModelProtocol[LatentT, ObservationT, ConditionT, ParametersT],
    condition_sequence: ConditionT,
) -> interface.ConditionContext[ConditionT]:
    if isinstance(condition_sequence, seqjtyping.NoCondition):
        return model.condition_context(())
    else:
        return model.condition_context(tuple(
            util.dynamic_index_pytree_in_dim(
                condition_sequence,
                ix,
                0,
            )
            for ix in range(model.prior_order)
        ))
    

def initial_context[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    model: interface.SequentialModelProtocol[LatentT, ObservationT, ConditionT, ParametersT],
    condition_sequence: ConditionT,
) -> ConditionT:
    if isinstance(condition_sequence, seqjtyping.NoCondition):
        return condition_sequence
    else:
        return util.index_pytree(condition_sequence, model.prior_order)
    
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
    
