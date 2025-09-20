import time
from typing import Optional, Any, Iterator, Protocol
import typing

import jaxtyping

from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu


import equinox as eqx

from tqdm.auto import trange  # type: ignore[import-untyped]
from tqdm.notebook import trange as nbtrange  # type: ignore[import-untyped]


class _ProgressIterator(Protocol):
    def __iter__(self) -> Iterator[int]: ...

    def set_postfix(self, *args: Any, **kwargs: Any) -> None: ...


from seqjax.model.base import BayesianSequentialModel
from seqjax.inference.vi.base import (
    SSMVariationalApproximation,
    BufferedSSMVI,
)
import seqjax.model.typing


def loss_buffered_neg_elbo(
    trainable,
    static,
    observations,
    conditions,
    key,
    target_posterior,
    num_context,
    samples_per_context,
):
    # build full model for sampling
    approximation: SSMVariationalApproximation = eqx.combine(trainable, static)

    return approximation.estimate_loss(
        observations,
        conditions,
        key,
        num_context,
        samples_per_context,
        target_posterior,
        None,
    )


def loss_pre_train_neg_elbo(
    trainable,
    static,
    observations,
    conditions,
    key,
    target_posterior,
    num_context,
    samples_per_context,
):
    # build full model for sampling
    approximation: BufferedSSMVI = eqx.combine(trainable, static)

    return approximation.estimate_pretrain_loss(
        observations,
        conditions,
        key,
        num_context,
        samples_per_context,
        target_posterior,
        None,
    )


class LocalTracker:
    def __init__(self):
        self.rows = []

    def log(self, data):
        self.rows.append(data)


@eqx.filter_jit
def sample_theta_qs(static, trainable, key, metric_samples):
    model: SSMVariationalApproximation = eqx.combine(static, trainable)
    parameter_keys = jrandom.split(key, metric_samples)
    theta, _ = jax.vmap(model.parameter_approximation.sample_and_log_prob)(
        parameter_keys, None
    )
    qs = jax.tree_util.tree_map(
        lambda x: jnp.quantile(x, jnp.array([0.05, 0.95])), theta
    )
    means = jax.tree_util.tree_map(lambda x: jnp.mean(x), theta)
    return qs, means, theta


class DefaultTracker:
    def __init__(self, record_interval=100, metric_samples=100):
        self.record_interval = record_interval
        self.metric_samples = metric_samples
        self.elapsed_time_s = 0
        self.update_rows = []
        self.checkpoint_samples = []

    def start_run(self):
        self.train_phase_start_time = time.time()

    def track_step(self, static, trainable, opt_step, loss, key, loop):
        if (opt_step + 1) % self.record_interval == 0:
            jax.device_get(loss)  # sync

            # increment elapsed time by the time of this train phase
            self.elapsed_time_s += time.time() - self.train_phase_start_time

            update = {
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
    model: SSMVariationalApproximation,
    observations: seqjax.model.typing.Observation,
    conditions: Optional[seqjax.model.typing.Condition],
    target: BayesianSequentialModel,
    *,
    key,
    optim,
    run_tracker: DefaultTracker,
    num_steps=1000,
    filter_spec=None,
    observations_per_step: int = 5,
    samples_per_context: int = 10,
    pre_train: bool = False,
    device_sharding: Optional[Any] = None,
    nb_context=False,
) -> SSMVariationalApproximation:
    # set up record if needed
    if run_tracker is None:
        run_tracker = DefaultTracker()

    # optimizer initailisation
    if filter_spec is None:
        filter_spec = jtu.tree_map(lambda leaf: eqx.is_inexact_array(leaf), model)

    trainable, static = eqx.partition(model, filter_spec)
    opt_state = optim.init(trainable)

    # loss configuration
    if pre_train:
        loss_fn = loss_pre_train_neg_elbo
    else:
        loss_fn = loss_buffered_neg_elbo

    loss_and_grad = jax.value_and_grad(
        partial(
            loss_fn,
            target_posterior=target,
            num_context=samples_per_context,
            samples_per_context=observations_per_step,
        )
    )

    # main training step
    step_keys = jrandom.split(key, num_steps)

    def make_step(trainable, static, opt_state, observations, conditions, key):
        loss, grads = loss_and_grad(trainable, static, observations, conditions, key)
        updates, opt_state = optim.update(grads, opt_state, params=trainable)
        trainable = eqx.apply_updates(trainable, updates)
        return loss, trainable, opt_state

    # compile
    compiled_make_step: typing.Callable[
        [
            eqx.Module,
            eqx.Module,
            typing.Any,
            seqjax.model.typing.Observation,
            Optional[seqjax.model.typing.Condition],
            jaxtyping.PRNGKeyArray,
        ],
        tuple[jaxtyping.Scalar, eqx.Module, eqx.Module],
    ] = (
        typing.cast(
            typing.Any, eqx.filter_jit(make_step)
        )  # mypy can't infer appropriate form
        .lower(trainable, static, opt_state, observations, conditions, step_keys[0])
        .compile()
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

    return eqx.combine(static, trainable)
