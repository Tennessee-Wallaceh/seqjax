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

from tqdm.auto import tqdm  # type: ignore[import-untyped]

from seqjax.model.base import BayesianSequentialModel
from seqjax.inference.interface import InferenceDataset
from seqjax.inference.vi.base import (
    SSMVariationalApproximation,
    BufferedSSMVI,
)
from seqjax.inference.vi.sampling import VISamplingKwargs
import seqjax.model.typing as seqjtyping

DEFAULT_SYNC_INTERVAL_S = 10


class _ProgressIterator(Protocol):
    def __iter__(self) -> Iterator[int]: ...

    def set_postfix(self, *args: Any, **kwargs: Any) -> None: ...


ParticleT = TypeVar("ParticleT", bound=seqjtyping.Latent)
ObservationT = TypeVar("ObservationT", bound=seqjtyping.Observation)
ConditionT = TypeVar("ConditionT", bound=seqjtyping.Condition)
ParametersT = TypeVar("ParametersT", bound=seqjtyping.Parameters)
InferenceParametersT = TypeVar("InferenceParametersT", bound=seqjtyping.Parameters)
HyperParametersT = TypeVar("HyperParametersT", bound=seqjtyping.HyperParameters)

SSMApproximationT = SSMVariationalApproximation[
    ParticleT,
    ObservationT,
    ConditionT,
    ParametersT,
    InferenceParametersT,
    HyperParametersT,
]
BufferedApproximationT = BufferedSSMVI[
    ParticleT,
    ObservationT,
    ConditionT,
    ParametersT,
    InferenceParametersT,
    HyperParametersT,
]


class SupportsPretrainLoss(Protocol):
    def estimate_pretrain_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        target_posterior: "TargetModelT",
        hyperparameters: HyperParametersT | None,
    ) -> jaxtyping.Scalar: ...


class SupportsPriorFitLoss(Protocol):
    def estimate_prior_fit_loss(
        self,
        dataset: InferenceDataset[ObservationT, ConditionT],
        key: jaxtyping.PRNGKeyArray,
        sample_kwargs: VISamplingKwargs,
        target_posterior: "TargetModelT",
        hyperparameters: HyperParametersT | None,
    ) -> jaxtyping.Scalar: ...


TargetModelT = BayesianSequentialModel[
    ParticleT,
    ObservationT,
    ConditionT,
    ParametersT,
    InferenceParametersT,
    HyperParametersT,
]

TrainableModuleT = TypeVar("TrainableModuleT", bound=SSMVariationalApproximation)
StaticModuleT = TypeVar("StaticModuleT", bound=SSMVariationalApproximation)
OptStateT = TypeVar("OptStateT")

LoggedArray = jaxtyping.Float[jaxtyping.Array, "..."]
LoggedValue = float | int | LoggedArray | str
TrackerLogRow = MutableMapping[str, LoggedValue]

ArrayTree: TypeAlias = PyTree[LoggedArray]

LossFunction = Callable[
    [
        TrainableModuleT,
        StaticModuleT,
        InferenceDataset[ObservationT, ConditionT],
        jaxtyping.PRNGKeyArray,
    ],
    jaxtyping.Scalar,
]

CompiledStepFn = Callable[
    [
        TrainableModuleT,
        StaticModuleT,
        OptStateT,
        InferenceDataset[ObservationT, ConditionT],
        jaxtyping.PRNGKeyArray,
    ],
    tuple[jaxtyping.Scalar, TrainableModuleT, OptStateT],
]

LossAndGradFn = Callable[
    [
        TrainableModuleT,
        StaticModuleT,
        InferenceDataset[ObservationT, ConditionT],
        jaxtyping.PRNGKeyArray,
    ],
    tuple[jaxtyping.Scalar, TrainableModuleT],
]


def loss_buffered_neg_elbo(
    trainable: TrainableModuleT,
    static: StaticModuleT,
    dataset: InferenceDataset[ObservationT, ConditionT],
    key: jaxtyping.PRNGKeyArray,
    target_posterior: TargetModelT,
    *,
    sample_kwargs: VISamplingKwargs,
) -> jaxtyping.Scalar:
    approximation = typing.cast(SSMApproximationT, eqx.combine(trainable, static))

    return approximation.estimate_loss(
        dataset,
        key,
        sample_kwargs,
        target_posterior,
        None,
    )


def loss_pre_train_neg_elbo(
    trainable: TrainableModuleT,
    static: StaticModuleT,
    dataset: InferenceDataset[ObservationT, ConditionT],
    key: jaxtyping.PRNGKeyArray,
    target_posterior: TargetModelT,
    *,
    sample_kwargs: VISamplingKwargs,
) -> jaxtyping.Scalar:
    approximation = typing.cast(
        SupportsPretrainLoss,
        eqx.combine(trainable, static),
    )

    return approximation.estimate_pretrain_loss(
        dataset,
        key,
        sample_kwargs,
        target_posterior,
        None,
    )


def loss_pre_train_prior(
    trainable: TrainableModuleT,
    static: StaticModuleT,
    dataset: InferenceDataset[ObservationT, ConditionT],
    key: jaxtyping.PRNGKeyArray,
    target_posterior: TargetModelT,
    *,
    sample_kwargs: VISamplingKwargs,
) -> jaxtyping.Scalar:
    approximation = typing.cast(
        SupportsPriorFitLoss,
        eqx.combine(trainable, static),
    )
    return approximation.estimate_prior_fit_loss(
        dataset,
        key,
        sample_kwargs,
        target_posterior,
        None,
    )


LossLables = typing.Literal["elbo", "pretrain", "param-prior"]
losses: dict[LossLables, Callable] = {
    "elbo": loss_buffered_neg_elbo,
    "pretrain": loss_pre_train_neg_elbo,
    "param-prior": loss_pre_train_prior,
}

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


class Tracker:
    metric_samples: int
    elapsed_time_s: float
    update_rows: list[TrackerLogRow]
    checkpoint_samples: list[tuple[float, ArrayTree]]
    train_phase_start_time: float
    record_trigger: Callable[[int, float], bool]
    custom_record_fcns: list[
        Callable[
            [
                TrackerLogRow,
                typing.Any,
                typing.Any,
                int,
                jaxtyping.Scalar,
                str,
                jaxtyping.PRNGKeyArray,
            ],
            tuple[list[str], list[LoggedValue]],
        ]
    ]
    postfix: dict[typing.Any, typing.Any]

    def __init__(
        self,
        metric_samples: int = 100,
        record_trigger=None,
        custom_record_fcns=None,
    ) -> None:
        self.update_rows = []
        self.checkpoint_samples = []
        self.metric_samples = metric_samples

        # configure any custom record functions
        self.record_trigger = record_trigger or (lambda step, elapsed_time_s: True)
        self.custom_record_fcns = custom_record_fcns or []
        self.postfix = {}

    def track_step(
        self,
        elapsed_time_s: float,
        opt_step: int,
        static: StaticModuleT,
        trainable: TrainableModuleT,
        loss: jaxtyping.Scalar,
        key: jaxtyping.PRNGKeyArray,
        loop: _ProgressIterator,
        loss_label: LossLables,
        force_record: bool = False,
    ) -> dict[typing.Any, typing.Any]:
        if force_record or self.record_trigger(opt_step, elapsed_time_s):
            update: TrackerLogRow = {
                "step": int(opt_step + 1),
                "loss": float(loss),
                "elapsed_time_s": elapsed_time_s,
                "loss_kind": loss_label,
            }

            qs, means, theta = sample_theta_qs(
                static, trainable, key, self.metric_samples
            )
            self.checkpoint_samples.append((elapsed_time_s, theta))
            _reads = []
            for param in static.parameter_approximation.target_struct_cls.fields():
                update[f"{param}_q05"] = getattr(qs, param)[0]
                update[f"{param}_q95"] = getattr(qs, param)[1]
                update[f"{param}_mean"] = getattr(means, param)
                _reads.append(f"{param}: {update[f'{param}_mean']:.2f}")
            mean_str = " , ".join(_reads)

            for fcn in self.custom_record_fcns:
                out = fcn(update, static, trainable, opt_step, loss, loss_label, key)
                if out is None:
                    continue
                labels, values = out
                for label, value in zip(labels, values):
                    update[label] = value

            self.update_rows.append(update)

            self.postfix = {"loss:": f"{loss.item():.3f} {mean_str}"}

        return self.postfix


def train(
    model: SSMApproximationT,
    dataset: InferenceDataset[ObservationT, ConditionT],
    target: TargetModelT,
    *,
    key: jaxtyping.PRNGKeyArray,
    optim: optax.GradientTransformation,
    run_tracker: Tracker | None,
    num_steps: int | None = 1000,
    filter_spec: PyTree[bool] | None = None,
    sample_kwargs: VISamplingKwargs,
    loss_label: LossLables = 'elbo',
    nb_context: bool = False,
    initial_opt_state: OptStateT | None = None,
    time_limit_s: int | None = None,
    sync_interval_s: float | None = None,
) -> tuple[SSMApproximationT, OptStateT]:
    # check for valid set up
    if num_steps is None and time_limit_s is None:
        raise Exception(
            "Variational fitting requires either num_steps or time_limit_s is set!"
        )

    if sync_interval_s is None:
        sync_interval_s = DEFAULT_SYNC_INTERVAL_S

    # set up tracker if needed
    if run_tracker is None:
        run_tracker = Tracker()

    # optimizer initailisation
    if filter_spec is None:
        filter_spec = jtu.tree_map(lambda leaf: eqx.is_inexact_array(leaf), model)

    trainable, static = eqx.partition(model, filter_spec)

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    else:
        opt_state = optim.init(trainable)

    # loss configuration
    base_loss_fn: Callable[..., jaxtyping.Scalar]
    base_loss_fn = losses[loss_label]
    loss_fn = typing.cast(
        LossFunction,
        partial(
            base_loss_fn,
            target_posterior=target,
            sample_kwargs=sample_kwargs,
        ),
    )

    loss_and_grad: LossAndGradFn = jax.value_and_grad(loss_fn)

    # main training step
    def make_step(
        trainable_in: TrainableModuleT,
        static_in: StaticModuleT,
        opt_state_in: OptStateT,
        dataset_in: InferenceDataset[ObservationT, ConditionT],
        key_in: jaxtyping.PRNGKeyArray,
    ) -> tuple[jaxtyping.Scalar, TrainableModuleT, OptStateT]:
        loss, grads = loss_and_grad(
            trainable_in,
            static_in,
            dataset_in,
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
        .lower(trainable, static, opt_state, dataset, key)
        .compile(),
    )

    # train loop
    loop: _ProgressIterator
    loop = tqdm(bar_format="{desc} | elapsed {elapsed} | {postfix}")

    # init step (to get tracking info)
    loss, _trainable, _ = compiled_make_step(
        trainable, static, opt_state, dataset, key
    )

    tracker_postfix = run_tracker.track_step(
        -1, -1, static, _trainable, loss, key, loop, loss_label,
    )

    elapsed_time_s = 0.0
    phase_start = time.perf_counter()

    opt_step = 0
    while True:
        subkey, key = jrandom.split(key)
        loss, trainable, opt_state = compiled_make_step(
            trainable, static, opt_state, dataset, subkey
        )
        sync_now = (time.perf_counter() - phase_start) > sync_interval_s

        if sync_now:
            # sync now
            loss.block_until_ready()
            t_after = time.perf_counter()
            elapsed_time_s += t_after - phase_start

            # we may want expensive operations during the run,
            # so need to account for this, so only start the timer after the step has
            # been tracked
            tracker_postfix = run_tracker.track_step(
                elapsed_time_s,
                opt_step,
                static,
                trainable,
                loss,
                subkey,
                loop,
                loss_label,
            )

            if time_limit_s and elapsed_time_s > time_limit_s:
                print("Stopping due to time limit")
                break

            if num_steps and opt_step > num_steps:
                print("Stopping due to step limit")
                break

            if num_steps is not None:
                tracker_postfix["iter"] = (
                    f"{100 * opt_step / num_steps:.0f}% ({num_steps})"
                )

            if time_limit_s is not None:
                tracker_postfix["time"] = (
                    f"{100 * elapsed_time_s / time_limit_s:.0f}% ({time_limit_s / 60:.0f}m)"
                )

            tracker_postfix["rate"] = f"{opt_step / elapsed_time_s:.1f} it/s"

            loop.set_postfix(tracker_postfix)

            phase_start = time.perf_counter()

        opt_step += 1

    # add final record
    run_tracker.track_step(
        elapsed_time_s,
        opt_step + 1,
        static,
        trainable,
        loss,
        subkey,
        loop,
        loss_label,
        force_record=True,
    )

    return typing.cast(SSMApproximationT, eqx.combine(static, trainable)), opt_state
