import time 
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

from seqjax.inference.vi import interface
from seqjax.inference.interface import InferenceDataset
from seqjax.inference.vi.sampling import VISamplingKwargs
from seqjax.inference.vi.embedder import Embedder
import seqjax.model.typing as seqjtyping
import seqjax.model.interface as model_interface
from seqjax.model.evaluate import log_prob_joint

import numpy as np
import arviz as az
from scipy.special import logsumexp

DEFAULT_SYNC_INTERVAL_S = 5

LatentT = typing.TypeVar("LatentT", bound=seqjtyping.Latent)
ObservationT = typing.TypeVar("ObservationT", bound=seqjtyping.Observation)
ConditionT = typing.TypeVar("ConditionT", bound=seqjtyping.Condition)
ParametersT = typing.TypeVar("ParametersT", bound=seqjtyping.Parameters)
OptStateT = typing.TypeVar("OptStateT")
TrainableModuleT = typing.TypeVar("TrainableModuleT")


def compute_log_iw(
    trainable,
    key,
    *,
    static,
    sample_kwargs,
    y,
    c,
    context,
    target,
    params,
):
    approximation = eqx.combine(trainable, static)

    x, loq_q_x, _ = jax.vmap(
        approximation.sample_and_log_prob,
        in_axes=(0, None, None),
    )(
        jrandom.split(key, sample_kwargs["mc-samples"]),
        context,
        None
    )

    log_p_x_y = jax.vmap(
        log_prob_joint,
        in_axes=(None, 0, None, None, None)
    )(
        target,
        x,
        y,
        c,
        params,
    )

    return log_p_x_y - loq_q_x


def log_ess_np(log_iw, axis=-1):
    """
    ESS from unnormalised log importance weights.

    Parameters
    ----------
    log_iw:
        Unnormalised log importance weights, log p(x) - log q(x).
    axis:
        Sample axis.

    Returns
    -------
    ess:
        Absolute effective sample size.
    rel_ess:
        Relative effective sample size, ESS / n.
    """
    log_iw = np.asarray(log_iw)

    n = log_iw.shape[axis]
    log_ess = (
        2.0 * logsumexp(log_iw, axis=axis)
        - logsumexp(2.0 * log_iw, axis=axis)
    )

    ess = np.exp(log_ess)
    rel_ess = ess / n

    return ess, rel_ess


def elbo_estimate_np(log_iw, axis=-1):
    """
    Monte Carlo ELBO estimate under q.

    Since log_iw = log p(x, y) - log q(x), or log p(x) - log q(x)
    depending on context, the simple VI/IS ELBO estimate is:

        E_q[log_iw]

    This is not the log marginal likelihood estimate. The IS estimate of
    log normalising constant would be:

        logmeanexp(log_iw)
    """
    log_iw = np.asarray(log_iw)
    return np.mean(log_iw, axis=axis)


def logz_is_estimate_np(log_iw, axis=-1):
    """
    Self-normalised-free importance sampling estimate of log Z:

        log mean_i exp(log_iw_i)

    This estimates log E_q[p/q], not E_q[log p/q].
    It is upward-biased on the log scale for finite N, but consistent.
    """
    log_iw = np.asarray(log_iw)
    n = log_iw.shape[axis]

    return logsumexp(log_iw, axis=axis) - np.log(n)


def importance_diagnostics(log_iw, axis=-1):
    """
    Diagnostics for unnormalised log importance weights.

    Uses ArviZ PSIS rather than a homebrew GPD fit.

    Parameters
    ----------
    log_iw:
        Array of log importance weights.
    axis:
        Sample axis over which to run PSIS / ESS / ELBO.

    Returns
    -------
    dict with:
        k_hat:
            Pareto k estimate from ArviZ.
        ess_raw:
            ESS of raw importance weights.
        rel_ess_raw:
            ESS / n for raw importance weights.
        ess_psis:
            ESS of Pareto-smoothed weights.
        rel_ess_psis:
            ESS / n for Pareto-smoothed weights.
        elbo:
            mean(log_iw).
        logz_is:
            logmeanexp(log_iw).
        log_iw_psis:
            Pareto-smoothed log weights.
    """
    log_iw = np.asarray(log_iw)

    ess_raw, rel_ess_raw = log_ess_np(log_iw, axis=axis)

    log_iw_psis, k_hat = az.psislw(log_iw)

    ess_psis, rel_ess_psis = log_ess_np(log_iw_psis, axis=axis)

    elbo = elbo_estimate_np(log_iw, axis=axis)
    logz_is = logz_is_estimate_np(log_iw, axis=axis)

    return {
        "k_hat": k_hat,
        "ess_raw": ess_raw,
        "rel_ess_raw": rel_ess_raw,
        "ess_psis": ess_psis,
        "rel_ess_psis": rel_ess_psis,
        "elbo": elbo,
        "logz_is": logz_is,
        "elbo_gap_estimate": logz_is - elbo,
    }



def loss_neg_elbo(
    trainable,
    key,
    *,
    static,
    sample_kwargs,
    y,
    c,
    context,
    target,
    params,
):
    approximation = eqx.combine(trainable, static)

    x, loq_q_x, _ = jax.vmap(
        approximation.sample_and_log_prob,
        in_axes=(0, None, None),
    )(
        jrandom.split(key, sample_kwargs["mc-samples"]),
        context,
        None
    )

    log_p_x_y = jax.vmap(
        log_prob_joint,
        in_axes=(None, 0, None, None, None)
    )(
        target,
        x,
        y,
        c,
        params,
    )

    return jnp.mean(loq_q_x - log_p_x_y)

LossLabels = typing.Literal["elbo"]
losses: dict[LossLabels, typing.Any] = {
    "elbo": loss_neg_elbo,
}

class LatentFitTracker:
    def __init__(
        self,
        record_trigger=None,
        custom_record_fcns=None,
    ):
        self.update_rows = []
        self.record_trigger = record_trigger or (lambda step, elapsed_time_s: True)
        self.custom_record_fcns = custom_record_fcns or []

    def record(self, opt_step, elapsed_time_s,  loss, force_record=False):
        if force_record or self.record_trigger(opt_step, elapsed_time_s):
            update = {
                "loss": float(loss),
                "opt_step": opt_step,
                "elapsed_time_s": elapsed_time_s,
            }
            for fcn in self.custom_record_fcns:
                out = fcn(update)
                if out is None:
                    continue
                labels, values = out
                for label, value in zip(labels, values):
                    update[label] = value

            self.update_rows.append(update)

        return {"loss": float(loss)}



def train(
    model: interface.VariationalApproximation[LatentT, ConditionT],
    embedder: Embedder,
    dataset: InferenceDataset[ObservationT, ConditionT],
    target: model_interface.SequentialModelProtocol,
    params: ParametersT,
    *,
    key: jaxtyping.PRNGKeyArray,
    optim: optax.GradientTransformation,
    run_tracker: None,
    num_steps: int | None = 1000,
    time_limit_s: int | None = None,
    filter_spec: PyTree[bool] | None = None,
    sample_kwargs: VISamplingKwargs,
    loss_label: LossLabels = 'elbo',
    initial_opt_state: OptStateT | None = None,
    model_state: typing.Any = None,
    sync_interval_s: float = DEFAULT_SYNC_INTERVAL_S,
    nb_context: bool = False,
    compiled_steps: int = 1,
    unroll: int = 1,
) -> tuple[
    interface.VariationalApproximation[LatentT, ConditionT], 
    OptStateT, 
    typing.Any
]:
    if num_steps is None and time_limit_s is None:
        raise Exception(
            "Variational fitting requires either num_steps or time_limit_s is set!"
        )

    # # set up tracker if needed
    if run_tracker is None:
        run_tracker = LatentFitTracker()

    # optimizer initailisation
    if filter_spec is None:
        filter_spec = jtu.tree_map(lambda leaf: eqx.is_inexact_array(leaf), model)

    trainable, static = eqx.partition(model, filter_spec)

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    else:
        opt_state = optim.init(trainable)

    #TODO: implement for multi sequence
    y, c = dataset.single_sequence()

    context, _ = embedder.embed(
        y, c,
        seqjtyping.NoParam(), 
        None
    )
    
    base_loss_fn = losses[loss_label]
    loss_fn = partial(
        base_loss_fn,
        static=static,
        y=y,
        c=c,
        context=context,
        target=target,
        params=params,
        sample_kwargs=sample_kwargs,
    )

    loss_and_grad = jax.value_and_grad(loss_fn)

    def make_step(
        trainable_in: TrainableModuleT,
        opt_state_in: OptStateT,
        key_in: jaxtyping.PRNGKeyArray,
    ) -> tuple[jaxtyping.Scalar, TrainableModuleT, OptStateT, typing.Any]:
        loss, grads = loss_and_grad(
            trainable_in,
            key_in,
        )
        updates, opt_state_next = optim.update(grads, opt_state_in, params=trainable_in)
        trainable_next = eqx.apply_updates(trainable_in, updates)
        return (
            loss,
            typing.cast(TrainableModuleT, trainable_next),
            typing.cast(OptStateT, opt_state_next),
        )

    def k_steps(trainable_in, opt_state_in, key_in):
        def body(carry, _):
            trainable, opt_state, key = carry
            key, subkey = jrandom.split(key)

            loss, grads = loss_and_grad(trainable, subkey)
            updates, opt_state = optim.update(grads, opt_state, params=trainable)
            trainable = eqx.apply_updates(trainable, updates)

            return (trainable, opt_state, key), loss

        (trainable_out, opt_state_out, key_out), losses = jax.lax.scan(
            body,
            (trainable_in, opt_state_in, key_in),
            xs=None,
            length=compiled_steps,
            unroll=unroll
        )
        return losses[-1], trainable_out, opt_state_out, key_out

    compiled_make_step = eqx.filter_jit(k_steps).lower(
        trainable, opt_state, key, 
    ).compile()

    loop = tqdm(bar_format="{desc} | elapsed {elapsed} | {postfix}")

    # init step (to get tracking info)
    loss, _, _, _ = compiled_make_step(
        trainable, opt_state, key, 
    )
    elapsed_time_s = 0.0
    phase_start = time.perf_counter()

    run_tracker.record(-1, -1, loss)

    opt_step = 0
    while True:
        loss, trainable, opt_state, key = compiled_make_step(
            trainable, opt_state, key, 
        )
        sync_now = (time.perf_counter() - phase_start) > sync_interval_s

        if sync_now:
            # sync now
            loss.block_until_ready()
            t_after = time.perf_counter()
            elapsed_time_s += t_after - phase_start

            if time_limit_s and elapsed_time_s > time_limit_s:
                print("Stopping due to time limit")
                break

            if num_steps and opt_step >= num_steps:
                print("Stopping due to step limit")
                break
            
            tracker_postfix = run_tracker.record(opt_step, elapsed_time_s, loss)

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


        opt_step += compiled_steps

    log_iw = compute_log_iw(
        trainable,
        key,
        static=static,
        sample_kwargs={"mc-samples": 10000},
        y=y,
        c=c,
        context=context,
        target=target,
        params=params,
    )
    iw_diagnostics = importance_diagnostics(log_iw)

    return eqx.combine(static, trainable), opt_state, run_tracker, iw_diagnostics