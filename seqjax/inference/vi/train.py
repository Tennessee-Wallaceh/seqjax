import time
from typing import (
    Any,
    Callable,
    Iterator,
    MutableMapping,
    Protocol,
    TypeVar,
    TypeAlias,
)
import typing

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import jaxtyping
import optax  # type: ignore[import-untyped]
from jaxtyping import PyTree
from jax.sharding import Sharding

from tqdm.auto import trange  # type: ignore[import-untyped]
from tqdm.notebook import trange as nbtrange  # type: ignore[import-untyped]

from seqjax.model.base import BayesianSequentialModel
from seqjax.inference.vi.base import (
    SSMVariationalApproximation,
    BufferedSSMVI,
)
import seqjax.model.typing as seqjtyping


class _ProgressIterator(Protocol):
    def __iter__(self) -> Iterator[int]: ...

    def set_postfix(self, *args: Any, **kwargs: Any) -> None: ...


ParticleT = TypeVar("ParticleT", bound=seqjtyping.Particle)
InitialParticleT = TypeVar("InitialParticleT", bound=tuple[seqjtyping.Particle, ...])
TransitionParticleHistoryT = TypeVar(
    "TransitionParticleHistoryT", bound=tuple[seqjtyping.Particle, ...]
)
ObservationParticleHistoryT = TypeVar(
    "ObservationParticleHistoryT", bound=tuple[seqjtyping.Particle, ...]
)
ObservationT = TypeVar("ObservationT", bound=seqjtyping.Observation)
ObservationHistoryT = TypeVar(
    "ObservationHistoryT", bound=tuple[seqjtyping.Observation, ...]
)
ConditionHistoryT = TypeVar("ConditionHistoryT", bound=tuple[seqjtyping.Condition, ...])
ConditionT = TypeVar("ConditionT", bound=seqjtyping.Condition)
ParametersT = TypeVar("ParametersT", bound=seqjtyping.Parameters)
InferenceParametersT = TypeVar("InferenceParametersT", bound=seqjtyping.Parameters)
HyperParametersT = TypeVar("HyperParametersT", bound=seqjtyping.HyperParameters)

SSMApproximationT = SSMVariationalApproximation[
    ParticleT,
    InitialParticleT,
    TransitionParticleHistoryT,
    ObservationParticleHistoryT,
    ObservationT,
    ObservationHistoryT,
    ConditionHistoryT,
    ConditionT,
    ParametersT,
    InferenceParametersT,
    HyperParametersT,
]
BufferedApproximationT = BufferedSSMVI[
    ParticleT,
    InitialParticleT,
    TransitionParticleHistoryT,
    ObservationParticleHistoryT,
    ObservationT,
    ObservationHistoryT,
    ConditionHistoryT,
    ConditionT,
    ParametersT,
    InferenceParametersT,
    HyperParametersT,
]
TargetModelT = BayesianSequentialModel[
    ParticleT,
    InitialParticleT,
    TransitionParticleHistoryT,
    ObservationParticleHistoryT,
    ObservationT,
    ObservationHistoryT,
    ConditionHistoryT,
    ConditionT,
    ParametersT,
    InferenceParametersT,
    HyperParametersT,
]

TrainableModuleT = TypeVar("TrainableModuleT", bound=eqx.Module)
StaticModuleT = TypeVar("StaticModuleT", bound=eqx.Module)
OptStateT = TypeVar("OptStateT")

LoggedArray = jaxtyping.Float[jaxtyping.Array, "..."]
LoggedValue = float | int | LoggedArray
TrackerLogRow = MutableMapping[str, LoggedValue]

ArrayTree: TypeAlias = PyTree[LoggedArray]

LossFunction = Callable[
    [
        TrainableModuleT,
        StaticModuleT,
        ObservationT,
        ConditionT | None,
        jaxtyping.PRNGKeyArray,
    ],
    jaxtyping.Scalar,
]

CompiledStepFn = Callable[
    [
        TrainableModuleT,
        StaticModuleT,
        OptStateT,
        ObservationT,
        ConditionT | None,
        jaxtyping.PRNGKeyArray,
    ],
    tuple[jaxtyping.Scalar, TrainableModuleT, OptStateT],
]

LossAndGradFn = Callable[
    [
        TrainableModuleT,
        StaticModuleT,
        ObservationT,
        ConditionT | None,
        jaxtyping.PRNGKeyArray,
    ],
    tuple[jaxtyping.Scalar, TrainableModuleT],
]


def loss_buffered_neg_elbo(
    trainable: TrainableModuleT,
    static: StaticModuleT,
    observations: ObservationT,
    conditions: ConditionT | None,
    key: jaxtyping.PRNGKeyArray,
    target_posterior: TargetModelT,
    num_context: int,
    samples_per_context: int,
) -> jaxtyping.Scalar:
    approximation = typing.cast(SSMApproximationT, eqx.combine(trainable, static))

    return approximation.estimate_loss(
        observations,
        typing.cast(ConditionT, conditions),
        key,
        num_context,
        samples_per_context,
        target_posterior,
        None,
    )


def loss_pre_train_neg_elbo(
    trainable: TrainableModuleT,
    static: StaticModuleT,
    observations: ObservationT,
    conditions: ConditionT | None,
    key: jaxtyping.PRNGKeyArray,
    target_posterior: TargetModelT,
    num_context: int,
    samples_per_context: int,
) -> jaxtyping.Scalar:
    approximation = typing.cast(BufferedApproximationT, eqx.combine(trainable, static))

    return approximation.estimate_pretrain_loss(
        observations,
        typing.cast(ConditionT, conditions),
        key,
        num_context,
        samples_per_context,
        target_posterior,
        None,
    )


class LocalTracker:
    rows: list[TrackerLogRow]

    def __init__(self) -> None:
        self.rows = []

    def log(self, data: TrackerLogRow) -> None:
        self.rows.append(data)


@eqx.filter_jit
def sample_theta_qs(
    static: StaticModuleT,
    trainable: TrainableModuleT,
    key: jaxtyping.PRNGKeyArray,
    metric_samples: int,
) -> tuple[ArrayTree, ArrayTree, ArrayTree]:
    model = typing.cast(SSMApproximationT, eqx.combine(static, trainable))
    parameter_keys = jrandom.split(key, metric_samples)
    theta, _ = jax.vmap(model.parameter_approximation.sample_and_log_prob)(
        parameter_keys, None
    )
    qs = jax.tree_util.tree_map(
        lambda x: jnp.quantile(x, jnp.array([0.05, 0.95])), theta
    )
    means = jax.tree_util.tree_map(lambda x: jnp.mean(x), theta)
    return (
        typing.cast(ArrayTree, qs),
        typing.cast(ArrayTree, means),
        typing.cast(ArrayTree, theta),
    )


class DefaultTracker:
    record_interval: int
    metric_samples: int
    elapsed_time_s: float
    update_rows: list[TrackerLogRow]
    checkpoint_samples: list[tuple[float, ArrayTree]]
    train_phase_start_time: float

    def __init__(self, record_interval: int = 5, metric_samples: int = 100) -> None:
        self.record_interval = record_interval
        self.metric_samples = metric_samples
        self.elapsed_time_s = 0.0
        self.update_rows = []
        self.checkpoint_samples = []
        self.train_phase_start_time = 0.0

    def start_run(self) -> None:
        self.train_phase_start_time = time.time()

    def track_step(
        self,
        static: StaticModuleT,
        trainable: TrainableModuleT,
        opt_step: int,
        loss: jaxtyping.Scalar,
        key: jaxtyping.PRNGKeyArray,
        loop: _ProgressIterator,
    ) -> None:
        if (opt_step + 1) % self.record_interval == 0:
            jax.device_get(loss)  # sync

            # increment elapsed time by the time of this train phase
            self.elapsed_time_s += time.time() - self.train_phase_start_time

            update: TrackerLogRow = {
                "step": int(opt_step + 1),
                "loss": float(loss),
                "elapsed_time_s": self.elapsed_time_s,
                # "phase_time_s": elapsed_time_s - time_offset,
                # "loss_label": loss_label,
            }

            qs, means, theta = sample_theta_qs(
                static, trainable, key, self.metric_samples
            )
            self.checkpoint_samples.append((self.elapsed_time_s, theta))
            _reads = []
            for param in static.parameter_approximation.target_struct_cls.fields():
                update[f"{param}_q05"] = getattr(qs, param)[0]
                update[f"{param}_q95"] = getattr(qs, param)[1]
                update[f"{param}_mean"] = getattr(means, param)
                _reads.append(f"{param}: {update[f'{param}_mean']:.2f}")
            mean_str = " , ".join(_reads)

            self.update_rows.append(update)

            loop.set_postfix({"loss:": f"{loss.item():.3f} {mean_str}"})

            # start the train phase timer
            self.train_phase_start_time = time.time()


def train(
    model: SSMApproximationT,
    observations: ObservationT,
    conditions: ConditionT | None,
    target: TargetModelT,
    *,
    key: jaxtyping.PRNGKeyArray,
    optim: optax.GradientTransformation,
    run_tracker: DefaultTracker | None,
    num_steps: int = 1000,
    filter_spec: PyTree[bool] | None = None,
    observations_per_step: int = 5,
    samples_per_context: int = 10,
    pre_train: bool = False,
    device_sharding: Sharding | None = None,
    nb_context: bool = False,
) -> SSMApproximationT:
    # set up record if needed
    if run_tracker is None:
        run_tracker = DefaultTracker()
    run_tracker = typing.cast(DefaultTracker, run_tracker)

    # optimizer initailisation
    if filter_spec is None:
        filter_spec = jtu.tree_map(lambda leaf: eqx.is_inexact_array(leaf), model)

    trainable, static = eqx.partition(model, filter_spec)
    opt_state = optim.init(trainable)

    # loss configuration
    base_loss_fn: Callable[..., jaxtyping.Scalar]
    base_loss_fn = loss_pre_train_neg_elbo if pre_train else loss_buffered_neg_elbo
    loss_fn = typing.cast(
        LossFunction,
        partial(
            base_loss_fn,
            target_posterior=target,
            num_context=samples_per_context,
            samples_per_context=observations_per_step,
        ),
    )

    loss_and_grad: LossAndGradFn = jax.value_and_grad(loss_fn)

    # main training step
    step_keys = jrandom.split(key, num_steps)

    def make_step(
        trainable_in: TrainableModuleT,
        static_in: StaticModuleT,
        opt_state_in: OptStateT,
        observations_in: ObservationT,
        conditions_in: ConditionT | None,
        key_in: jaxtyping.PRNGKeyArray,
    ) -> tuple[jaxtyping.Scalar, TrainableModuleT, OptStateT]:
        loss, grads = loss_and_grad(
            trainable_in,
            static_in,
            observations_in,
            typing.cast(ConditionT, conditions_in),
            key_in,
        )
        updates, opt_state_next = optim.update(grads, opt_state_in, params=trainable_in)
        trainable_next = eqx.apply_updates(trainable_in, updates)
        return (
            loss,
            typing.cast(TrainableModuleT, trainable_next),
            typing.cast(OptStateT, opt_state_next),
        )

    # compile
    compiled_make_step = typing.cast(
        CompiledStepFn,
        typing.cast(Any, eqx.filter_jit(make_step))
        .lower(trainable, static, opt_state, observations, conditions, step_keys[0])
        .compile(),
    )

    # train loop
    loop: _ProgressIterator
    if nb_context:
        loop = nbtrange(num_steps, position=1)
    else:
        loop = trange(num_steps, position=1)

    # init step (to get tracking info)
    loss, _trainable, _opt_state = compiled_make_step(
        trainable, static, opt_state, observations, conditions, step_keys[0]
    )

    run_tracker.start_run()
    run_tracker.track_step(static, _trainable, -1, loss, step_keys[0], loop)

    for opt_step in loop:
        loss, trainable, opt_state = compiled_make_step(
            trainable, static, opt_state, observations, conditions, step_keys[opt_step]
        )

        run_tracker.track_step(
            static, trainable, opt_step, loss, step_keys[opt_step], loop
        )

    return typing.cast(SSMApproximationT, eqx.combine(static, trainable))
