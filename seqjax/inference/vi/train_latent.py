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

DEFAULT_SYNC_INTERVAL_S = 5

LatentT = typing.TypeVar("LatentT", bound=seqjtyping.Latent)
ObservationT = typing.TypeVar("ObservationT", bound=seqjtyping.Observation)
ConditionT = typing.TypeVar("ConditionT", bound=seqjtyping.Condition)
ParametersT = typing.TypeVar("ParametersT", bound=seqjtyping.Parameters)
OptStateT = typing.TypeVar("OptStateT")
TrainableModuleT = typing.TypeVar("TrainableModuleT")

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
    def __init__(self):
        self.update_rows = []

    def record(self, opt_step, elapsed_time_s,  loss):
        self.update_rows.append({
            "loss": float(loss),
            "opt_step": opt_step,
            "elapsed_time_s": elapsed_time_s,
        })

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

    return eqx.combine(static, trainable), opt_state, run_tracker