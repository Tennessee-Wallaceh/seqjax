import typing

import seqjax.model.typing as seqjtyping
import jax.random as jrandom
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Scalar
import jax.scipy.stats as jstats
from .interface import LatentContext, ConditionContext

class TransitionSample[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> LatentT: ...
    
class TransitionLogProb[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    def __call__(
        self,
        latent_history: LatentContext[LatentT],
        latent: LatentT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar: ...

def gaussian_loc_scale_transition[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
](
    loc_scale: typing.Callable[
        [LatentContext[LatentT], ConditionT, ParametersT],
        tuple[Scalar, Scalar],
    ], 
    latent_t: type[LatentT]
) -> tuple[
    TransitionSample[LatentT, ConditionT, ParametersT], 
    TransitionLogProb[LatentT, ConditionT, ParametersT]
]:
    
    def sample(
        key: PRNGKeyArray,
        latent_history: LatentContext[LatentT],
        condition: ConditionT,
        parameters: ParametersT,
    ) -> LatentT:
        loc_x, scale_x = loc_scale(latent_history, condition, parameters)
        eps = jrandom.normal(key)
        next_x = loc_x + eps * scale_x
        return latent_t.unravel(next_x)

    def log_prob(
        latent_history: LatentContext[LatentT],
        latent: LatentT,
        condition: ConditionT,
        parameters: ParametersT,
    ) -> Scalar:
        loc_x, scale_x = loc_scale(latent_history, condition, parameters)
        x = latent.ravel()
        lp = jstats.norm.logpdf(x, loc=loc_x, scale=scale_x)
        return jnp.sum(lp)
    
    return sample, log_prob