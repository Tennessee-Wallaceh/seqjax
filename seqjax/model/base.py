"""Protocol typing for sequential models.

Condition and Parameters are separated. Parameters remain static over time while
conditions vary.

Use ``SequentialModel`` to group pure functions that operate on the same
``Latent`` and ``Emission`` types. ``Prior``, ``Transition`` and ``Emission``
provide additional structure and are typically paired in use.

Specific dependency structures can be expressed by defining custom Prior, Transition
and Emission protocols. The default ones assume first order Markovian structure
(i.e. only depending on the previous latent state) and emissions depending only
on the current latent state.
This requires a fair amount of boilerplate, but allows for nice typing without resorting to
metaclasses.

Alternatives:
- Fully explicit history for generic Transitions etc, model def becomes even more verbose
- Abstract base classes with metaclass magic to enforce structure, but less static typing support
"""

from typing import Callable
import typing

from jaxtyping import PRNGKeyArray, Scalar

import seqjax.model.typing as seqjtyping


class ParameterPrior[
    ParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
](typing.Protocol):
    """Parameter prior specified as utility for specifying Bayesian models."""

    sample: Callable[
        [
            PRNGKeyArray,
            HyperParametersT,
        ],
        ParametersT,
    ]

    log_prob: Callable[
        [
            ParametersT,
            HyperParametersT,
        ],
        Scalar,
    ]


class Prior1[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    """Prior must define density + sampling up to the start state t=0.
    As such it receives conditions up to t=0, length corresponding to order.

    This could be over a number of latents if the dependency structure of the model requires.

    For example if the transition is (x_t1, x_t2) -> x_t3 then the prior must produce 2 latents (x_t-1, x_t0).
    Even if the transition is (x_t1,) -> x_t2, if emission is (x_t1, x_t2) -> y_t2 then
    the prior must specify p(x_t-1, x_t0), so that p(y_t0 | x_t-1, x_t0) can be evaluated.
    """

    order: typing.ClassVar[typing.Literal[1]] = 1

    sample: Callable[
        [
            PRNGKeyArray,
            ConditionT,
            ParametersT,
        ],
        tuple[LatentT],
    ]

    log_prob: Callable[
        [
            tuple[LatentT],
            ConditionT,
            ParametersT,
        ],
        Scalar,
    ]


class Prior2[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    """Prior must define density + sampling up to the start state t=0.
    As such it receives conditions up to t=0, length corresponding to order.

    This could be over a number of latents if the dependency structure of the model requires.

    For example if the transition is (x_t1, x_t2) -> x_t3 then the prior must produce 2 latents (x_t-1, x_t0).
    Even if the transition is (x_t1,) -> x_t2, if emission is (x_t1, x_t2) -> y_t2 then
    the prior must specify p(x_t-1, x_t0), so that p(y_t0 | x_t-1, x_t0) can be evaluated.
    """

    order: typing.ClassVar[typing.Literal[2]] = 2

    sample: Callable[
        [
            PRNGKeyArray,
            tuple[ConditionT, ConditionT],
            ParametersT,
        ],
        tuple[LatentT, LatentT],
    ]

    log_prob: Callable[
        [
            tuple[LatentT, LatentT],
            tuple[ConditionT, ConditionT],
            ParametersT,
        ],
        Scalar,
    ]


type Prior[
    L: seqjtyping.Latent,
    C: seqjtyping.Condition | seqjtyping.NoCondition,
    P: seqjtyping.Parameters,
] = Prior1[L, C, P] | Prior2[L, C, P]


class Transition1[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    order: typing.ClassVar[typing.Literal[1]] = 1

    sample: Callable[
        [
            PRNGKeyArray,
            tuple[LatentT],
            ConditionT,
            ParametersT,
        ],
        LatentT,
    ]

    log_prob: Callable[
        [
            tuple[LatentT],
            LatentT,
            ConditionT,
            ParametersT,
        ],
        Scalar,
    ]


class Transition2[
    LatentT: seqjtyping.Latent,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    order: typing.ClassVar[typing.Literal[2]] = 2

    sample: Callable[
        [
            PRNGKeyArray,
            tuple[LatentT, LatentT],
            ConditionT,
            ParametersT,
        ],
        LatentT,
    ]

    log_prob: Callable[
        [
            tuple[LatentT, LatentT],
            LatentT,
            ConditionT,
            ParametersT,
        ],
        Scalar,
    ]


type Transition[
    L: seqjtyping.Latent,
    C: seqjtyping.Condition | seqjtyping.NoCondition,
    P: seqjtyping.Parameters,
] = Transition1[L, C, P] | Transition2[L, C, P]


class EmissionO1D0[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    order: typing.ClassVar[typing.Literal[1]] = 1
    observation_dependency: typing.ClassVar[typing.Literal[0]] = 0

    sample: Callable[
        [
            PRNGKeyArray,
            tuple[LatentT],
            tuple[()],
            ConditionT,
            ParametersT,
        ],
        ObservationT,
    ]

    log_prob: Callable[
        [
            tuple[LatentT],
            tuple[()],
            ObservationT,
            ConditionT,
            ParametersT,
        ],
        Scalar,
    ]


class EmissionO2D0[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
](typing.Protocol):
    order: typing.ClassVar[typing.Literal[2]] = 2
    observation_dependency: typing.ClassVar[typing.Literal[0]] = 0

    sample: Callable[
        [
            PRNGKeyArray,
            tuple[LatentT, LatentT],
            tuple[()],
            ConditionT,
            ParametersT,
        ],
        ObservationT,
    ]

    log_prob: Callable[
        [
            tuple[LatentT, LatentT],
            tuple[()],
            ObservationT,
            ConditionT,
            ParametersT,
        ],
        Scalar,
    ]


type Emission[
    L: seqjtyping.Latent,
    O: seqjtyping.Observation,
    C: seqjtyping.Condition | seqjtyping.NoCondition,
    P: seqjtyping.Parameters,
] = EmissionO1D0[L, O, C, P] | EmissionO2D0[L, O, C, P]


class SequentialModel[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
]:
    latent_cls: type[LatentT]
    observation_cls: type[ObservationT]
    parameter_cls: type[ParametersT]
    condition_cls: type[ConditionT]
    prior: Prior[LatentT, ConditionT, ParametersT]
    transition: Transition[LatentT, ConditionT, ParametersT]
    emission: Emission[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ]


class BayesianSequentialModel[
    LatentT: seqjtyping.Latent,
    ObservationT: seqjtyping.Observation,
    ConditionT: seqjtyping.Condition | seqjtyping.NoCondition,
    ParametersT: seqjtyping.Parameters,
    InferenceParametersT: seqjtyping.Parameters,
    HyperParametersT: seqjtyping.HyperParameters,
]:
    inference_parameter_cls: type[InferenceParametersT]
    target: SequentialModel[
        LatentT,
        ObservationT,
        ConditionT,
        ParametersT,
    ]
    parameter_prior: ParameterPrior[InferenceParametersT, HyperParametersT]
    target_parameter: Callable[[InferenceParametersT], ParametersT]
