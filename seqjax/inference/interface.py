import typing
import jaxtyping
import seqjax.model.typing as seqjtyping
from seqjax.model.base import (
    BayesianSequentialModel,
)


class InferenceMethod[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](typing.Protocol):
    """Callable protocol for Bayesian inference routines.

    This interface represents the minimal call signature shared by the
    Bayesian samplers provided in :mod:`seqjax`. Concrete inference functions
    such as :func:`~seqjax.inference.mcmc.run_nuts` should be partially applied
    with any algorithm-specific configuration before being used through this
    protocol.
    """

    def __call__(
        self,
        target_posterior: BayesianSequentialModel[
            LatentT,
            ObservationT,
            ConditionT,
            ParametersT,
            InferenceParametersT,
            HyperParametersT,
        ],
        hyperparameters: HyperParametersT,
        key: jaxtyping.PRNGKeyArray,
        observation_path: ObservationT,
        condition_path: ConditionT,
        test_samples: int,
        config: typing.Any,
        tracker: typing.Any = None,  # optional for logging
    ) -> tuple[InferenceParametersT, typing.Any]: ...


def inference_method(f: InferenceMethod) -> InferenceMethod:
    """Adding this decorator ensures that an intended inference method satisfies the protocol"""
    return f
