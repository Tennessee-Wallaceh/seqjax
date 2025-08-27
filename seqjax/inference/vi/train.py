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

    theta_q, log_q_theta, x_q, log_q_x_path, start_ix, latent_scaling = (
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

    log_p_theta = jax.vmap(
        lambda x: target_posterior.parameter_prior.log_prob(x, None)
    )(theta_q)

    batched_log_p_joint = jax.vmap(
        partial(buffered_log_p_joint, target_posterior.target),
        in_axes=[0, None, None, 0],
    )
    batched_log_p_joint = jax.vmap(batched_log_p_joint, in_axes=[0, 0, 0, 0])
    log_p_x_y_path_no_x = batched_log_p_joint(
        jax.lax.stop_gradient(x_q),
        corresponding_observations,
        corresponding_conditions,
        target_posterior.target_parameter(approximation.buffer_params(theta_q)),
    )

    log_p_x_y_path_no_theta = batched_log_p_joint(
        x_q,
        corresponding_observations,
        corresponding_conditions,
        target_posterior.target_parameter(
            approximation.buffer_params(jax.lax.stop_gradient(theta_q))
        ),
    )

    # apply scaling for each index and sum down the sequence axis
    # latent_scaling is (num_context, sample_length) so add extra axis in middle
    # (num_context, 1, sample_length) to broadcast
    path_neg_elbo = jnp.sum(
        jnp.expand_dims(latent_scaling, axis=1) * (log_q_x_path - log_p_x_y_path_no_x),
        axis=-1,
    )

    x_path_neg_elbo = jnp.sum(log_q_x_path - log_p_x_y_path_no_theta, axis=-1)
    neg_elbo = (log_q_theta - log_p_theta) + 0.5 * path_neg_elbo + 0.5 * x_path_neg_elbo

    return jnp.mean(neg_elbo)


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


class LocalTracker:
    def __init__(self):
        self.rows = []

    def log(self, data):
        self.rows.append(data)


def train(
    model: SSMVariationalApproximation,
    observations: seqjax.model.typing.Observation,
    conditions: Optional[seqjax.model.typing.Condition],
    target,
    *,
    key,
    optim,
    run_tracker: Optional[Run | LocalTracker] = None,
    num_steps=1000,
    record_interval=100,
    filter_spec=None,
    observations_per_step: int = 5,
    samples_per_context: int = 10,
    metric_samples: int = 1000,
    device_sharding: Optional[Any] = None,
    nb_context=False,
    loss_label="buffered-neg-elbo",
    time_offset=0,
    loop_label="",
) -> tuple[SSMVariationalApproximation, Any, Any, int]:

    # set up record if needed
    if run_tracker is None:
        run_tracker = LocalTracker()

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
    path_loss_and_grad = jax.value_and_grad(
        partial(
            loss_buffered_neg_elbo_path,
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

    def sample_theta_qs(trainable, static, key):
        model: SSMVariationalApproximation = eqx.combine(trainable, static)
        parameter_keys = jrandom.split(key, metric_samples)
        theta, _ = jax.vmap(model.parameter_approximation.sample_and_log_prob)(
            parameter_keys
        )
        qs = jax.tree_util.tree_map(
            lambda x: jnp.quantile(x, jnp.array([0.05, 0.95])), theta
        )
        means = jax.tree_util.tree_map(lambda x: jnp.mean(x), theta)
        return qs, means

    # compile
    make_step = (
        eqx.filter_jit(make_step)
        .lower(trainable, static, opt_state, observations, conditions, step_keys[0])
        .compile()
    )
    sample_theta_qs = (
        eqx.filter_jit(sample_theta_qs).lower(trainable, static, step_keys[0]).compile()
    )

    # train loop
    if nb_context:
        loop = nbtrange(num_steps, position=1)
    else:
        loop = trange(num_steps, position=1)

    train_phase_start_time = time.time()
    elapsed_time_s = time_offset
    print(f"elapsed_s: {elapsed_time_s}")

    mean_str = ""
    for opt_step in loop:
        loss, trainable, opt_state = make_step(
            trainable, static, opt_state, observations, conditions, step_keys[opt_step]
        )
        loss = loss.item()

        if (opt_step + 1) % record_interval == 0:
            jax.device_get(loss)  # sync

            # increment elapsed time by the time of this train phase
            elapsed_time_s += time.time() - train_phase_start_time

            update = {
                "step": int(opt_step + 1),
                loss_label: float(loss),
                "elapsed_time_s": elapsed_time_s,
                "phase_time_s": elapsed_time_s - time_offset,
                "loss_label": loss_label,
            }

            qs, means = sample_theta_qs(trainable, static, step_keys[opt_step])

            _reads = []
            for param in static.parameter_approximation.target_struct_cls.fields():
                update[f"{param}_q05"] = getattr(qs, param)[0]
                update[f"{param}_q95"] = getattr(qs, param)[1]
                update[f"{param}_mean"] = getattr(means, param)
                _reads.append(f'{param}: {update[f"{param}_mean"]:.2f}')
            mean_str = " , ".join(_reads)

            # ess_eff, psis_k, prior_fit, data_fit = compute_vi_metrics(
            #     trainable, static, key
            # )
            # update["ess"] = float(ess_eff)
            # update["k_hat"] = float(psis_k)
            # update["prior_fit"] = float(prior_fit)
            # update["data_fit"] = float(data_fit)

            run_tracker.log(update)

            # start the train phase timer
            train_phase_start_time = time.time()

        loop.set_postfix({f"{loop_label}| {loss_label}": f"{loss:.3f} {mean_str}"})

    model = eqx.combine(static, trainable)
    return model, opt_state, run_tracker, elapsed_time_s
