import time
from typing import Optional, Any

from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
from tqdm.auto import trange
from tqdm.notebook import trange as nbtrange
from wandb.sdk.wandb_run import Run

# from seqjax.model.evaluate import log_p_joint, log_p_x, log_p_y_given_x
from seqjax import util
from seqjax.inference.vi.base import (
    SSMVariationalApproximation,
)
from seqjax.model.evaluate import buffered_log_p_joint
import seqjax.model.typing
from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    ParameterPrior,
    SequentialModel,
    BayesianSequentialModel,
)
from seqjax.model.typing import (
    Batched,
    SequenceAxis,
    SampleAxis,
    HyperParametersType,
    InferenceParametersType,
)


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
    )


def loss_buffered_neg_elbo_path(
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

    theta_q, _, x_q, log_q_x_path, start_ix, latent_scaling = (
        approximation.joint_sample_and_log_prob(
            observations,
            conditions,
            key,
            num_context,
            samples_per_context,
        )
    )

    corresponding_observations = jax.vmap(
        util.dynamic_slice_pytree, in_axes=[None, 0, None]
    )(observations, start_ix, approximation.latent_approximation.shape[0])

    corresponding_conditions = jax.vmap(
        util.dynamic_slice_pytree, in_axes=[None, 0, None]
    )(conditions, start_ix, approximation.latent_approximation.shape[0])

    buffered_theta_q = approximation.buffer_params(jax.lax.stop_gradient(theta_q))

    batched_log_p_joint = jax.vmap(
        partial(buffered_log_p_joint, target_posterior.target),
        in_axes=[0, None, None, 0],
    )
    batched_log_p_joint = jax.vmap(batched_log_p_joint, in_axes=[0, 0, 0, 0])
    log_p_x_y_path = batched_log_p_joint(
        x_q,
        corresponding_observations,
        corresponding_conditions,
        target_posterior.target_parameter(buffered_theta_q),
    )

    return jnp.mean(log_q_x_path - log_p_x_y_path)


def loss_parameter_prior_fit(
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

    param_keys, latent_keys = jrandom.split(key)
    theta_keys = jrandom.split(param_keys, (num_context, samples_per_context))
    prior_theta = jax.vmap(jax.vmap(target_posterior.parameter_prior.sample))(
        theta_keys, None
    )

    theta_q, _, x_q, log_q_x_path, start_ix, latent_scaling = (
        approximation.joint_sample_and_log_prob(
            observations,
            conditions,
            key,
            num_context,
            samples_per_context,
        )
    )

    corresponding_observations = jax.vmap(
        util.dynamic_slice_pytree, in_axes=[None, 0, None]
    )(observations, start_ix, approximation.latent_approximation.shape[0])

    corresponding_conditions = jax.vmap(
        util.dynamic_slice_pytree, in_axes=[None, 0, None]
    )(conditions, start_ix, approximation.latent_approximation.shape[0])

    buffered_theta_q = approximation.buffer_params(jax.lax.stop_gradient(theta_q))

    batched_log_p_joint = jax.vmap(
        partial(buffered_log_p_joint, target_posterior.target),
        in_axes=[0, None, None, 0],
    )
    batched_log_p_joint = jax.vmap(batched_log_p_joint, in_axes=[0, 0, 0, 0])
    log_p_x_y_path = batched_log_p_joint(
        x_q,
        corresponding_observations,
        corresponding_conditions,
        target_posterior.target_parameter(buffered_theta_q),
    )

    return jnp.mean(log_q_x_path - log_p_x_y_path)


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
        parameter_keys
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
                _reads.append(f'{param}: {update[f"{param}_mean"]:.2f}')
            mean_str = " , ".join(_reads)

            self.update_rows.append(update)

            loop.set_postfix({f"loss:": f"{loss.item():.3f} {mean_str}"})

            # start the train phase timer
            self.train_phase_start_time = time.time()


def train(
    model: SSMVariationalApproximation,
    observations: seqjax.model.typing.Observation,
    conditions: Optional[seqjax.model.typing.Condition],
    target,
    *,
    key,
    optim,
    run_tracker: DefaultTracker,
    num_steps=1000,
    filter_spec=None,
    observations_per_step: int = 5,
    samples_per_context: int = 10,
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
    loss_and_grad = jax.value_and_grad(
        partial(
            loss_buffered_neg_elbo,
            target_posterior=target,
            num_context=samples_per_context,
            samples_per_context=observations_per_step,
        )
    )

    # main training step
    step_keys = jrandom.split(key, num_steps)

    def make_step(trainable, static, opt_state, observations, conditions, key):
        loss, grads = loss_and_grad(trainable, static, observations, conditions, key)
        updates, opt_state = optim.update(grads, opt_state)
        trainable = eqx.apply_updates(trainable, updates)
        return loss, trainable, opt_state

    # compile
    make_step = (
        eqx.filter_jit(make_step)
        .lower(trainable, static, opt_state, observations, conditions, step_keys[0])
        .compile()
    )

    # train loop
    if nb_context:
        loop = nbtrange(num_steps, position=1)
    else:
        loop = trange(num_steps, position=1)

    run_tracker.start_run()
    for opt_step in loop:
        loss, trainable, opt_state = make_step(
            trainable, static, opt_state, observations, conditions, step_keys[opt_step]
        )

        run_tracker.track_step(
            static, trainable, opt_step, loss, step_keys[opt_step], loop
        )

    return eqx.combine(static, trainable)
