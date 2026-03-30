import time 
import typing

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
import seqjax.model.typing as seqjtyping


LatentT = typing.TypeVar("LatentT", bound=seqjtyping.Latent)
ObservationT = typing.TypeVar("ObservationT", bound=seqjtyping.Observation)
ConditionT = typing.TypeVar("ConditionT", bound=seqjtyping.Condition)
ParametersT = typing.TypeVar("ParametersT", bound=seqjtyping.Parameters)

class Loss(typing.Protocol):
    def __call__(self,):
        pass 

def loss_neg_elbo(
    trainable: TrainableModuleT,
    static: StaticModuleT,
    dataset: InferenceDataset[ObservationT, ConditionT],
    key: jaxtyping.PRNGKeyArray,
    state: typing.Any,
    *,
    sample_kwargs: VISamplingKwargs,
) -> tuple[jaxtyping.Scalar, typing.Any]:
    approximation = eqx.combine(trainable, static)


LossLabels = typing.Literal["elbo"]
losses: dict[LossLabels, Callable] = {
    "elbo": loss_neg_elbo,
}


def train(
    model: interface.VariationalApproximation[LatentT, ConditionT],
    dataset: InferenceDataset[ObservationT, ConditionT],
    target: TargetModelT,
    *,
    key: jaxtyping.PRNGKeyArray,
    optim: optax.GradientTransformation,
    run_tracker: Tracker | None,
    num_steps: int | None = 1000,
    time_limit_s: int | None = None,
    filter_spec: PyTree[bool] | None = None,
    sample_kwargs: VISamplingKwargs,
    loss_label: LossLables = 'elbo',
    initial_opt_state: OptStateT | None = None,
    model_state: typing.Any = None,
    sync_interval_s: float = DEFAULT_SYNC_INTERVAL_S,
    nb_context: bool = False,
) -> tuple[
    interface.VariationalApproximation[LatentT, ConditionT], 
    OptStateT, 
    typing.Any
]:
    if num_steps is None and time_limit_s is None:
        raise Exception(
            "Variational fitting requires either num_steps or time_limit_s is set!"
        )

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

    loss_fn = typing.cast(
        LossFunction,
        partial(
            base_loss_fn,
            sample_kwargs=sample_kwargs,
        ),
    )

    loss_and_grad: LossAndGradFn = jax.value_and_grad(loss_fn, has_aux=True)

    def make_step(
        trainable_in: TrainableModuleT,
        static_in: StaticModuleT,
        opt_state_in: OptStateT,
        dataset_in: InferenceDataset[ObservationT, ConditionT],
        key_in: jaxtyping.PRNGKeyArray,
        state_in: typing.Any,
    ) -> tuple[jaxtyping.Scalar, TrainableModuleT, OptStateT, typing.Any]:
        (loss, next_state), grads = loss_and_grad(
            trainable_in,
            static_in,
            dataset_in,
            key_in,
            state_in,
        )
        updates, opt_state_next = optim.update(grads, opt_state_in, params=trainable_in)
        trainable_next = eqx.apply_updates(trainable_in, updates)
        return (
            loss,
            typing.cast(TrainableModuleT, trainable_next),
            typing.cast(OptStateT, opt_state_next),
            next_state,
        )

    compiled_make_step = typing.cast(
        CompiledStepFn,
        typing.cast(Any, eqx.filter_jit(make_step))
        .lower(trainable, static, opt_state, dataset, key, model_state)
        .compile(),
    )