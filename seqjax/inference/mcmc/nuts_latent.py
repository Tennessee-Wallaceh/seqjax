from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jaxtyping

import blackjax  # type: ignore

from seqjax.inference.interface import InferenceDataset
from seqjax.inference.mcmc.nuts import inference_loop_multiple_chains
from seqjax.model import evaluate
from seqjax.model.interface import SequentialModelProtocol
from seqjax.model.simulate import simulate
import seqjax.model.typing as seqjtyping
from seqjax.util import pytree_shape


LatentT = TypeVar("LatentT", bound=seqjtyping.Latent)
ObservationT = TypeVar("ObservationT", bound=seqjtyping.Observation)
ConditionT = TypeVar("ConditionT", bound=seqjtyping.Condition)
ParametersT = TypeVar("ParametersT", bound=seqjtyping.Parameters)


class LatentNUTSTracker(Protocol):
    def on_block_end(
        self,
        *,
        elapsed_time_s: float,
        samples_taken: int,
    ) -> None: ...


@dataclass
class LatentNUTSConfig:
    fixed_parameters: seqjtyping.Parameters
    step_size: float = 1e-3
    num_adaptation: int = 1000
    num_warmup: int = 1000
    num_steps: int | None = 5000
    sample_block_size: int = 1000
    downsample_stride: int = 1
    inverse_mass_matrix: Any | None = None
    num_chains: int = 1
    max_time_s: float | None = None
    initial_latents: seqjtyping.Latent | None = None

    def __post_init__(self) -> None:
        positive_fields = [
            "step_size",
            "num_adaptation",
            "num_warmup",
            "sample_block_size",
            "num_chains",
            "downsample_stride",
        ]
        if self.num_steps is not None:
            positive_fields.append("num_steps")

        for field in positive_fields:
            value = getattr(self, field)
            if value <= 0:
                raise ValueError(f"{field} must be > 0, got {value}.")


@dataclass(frozen=True)
class LatentNUTSDiagnostics:
    block_times_s: list[tuple[float, int]]


def _validate_dataset_and_latents(
    *,
    latents: seqjtyping.Latent,
    dataset: InferenceDataset[ObservationT, ConditionT],
) -> None:
    latent_shape = pytree_shape(latents)[0]
    if len(latent_shape) == 0:
        raise ValueError(
            "Latent state must include at least sequence and time axes; "
            f"received leaf shape {latent_shape}."
        )

    if latent_shape[0] != dataset.num_sequences:
        raise ValueError(
            "Latent state must include a leading num_sequences axis. "
            f"Expected {dataset.num_sequences}, got {latent_shape[0]}."
        )

    if len(latent_shape) < 2:
        raise ValueError(
            "Latent state must include a time axis after num_sequences. "
            f"Received leaf shape {latent_shape}."
        )

    expected_latent_length = dataset.sequence_length
    if latent_shape[1] != expected_latent_length:
        raise ValueError(
            "Latent path length must match dataset.sequence_length for latent NUTS. "
            f"Expected {expected_latent_length}, got {latent_shape[1]}."
        )


def run_latent_nuts[
    LatentPathT: seqjtyping.Latent,
    ObservationPathT: seqjtyping.Observation,
    ConditionPathT: seqjtyping.Condition,
    ModelParametersT: seqjtyping.Parameters,
](
    target: SequentialModelProtocol[
        LatentPathT,
        ObservationPathT,
        ConditionPathT,
        ModelParametersT,
    ],
    key: jaxtyping.PRNGKeyArray,
    dataset: InferenceDataset[ObservationPathT, ConditionPathT],
    config: LatentNUTSConfig,
    tracker: LatentNUTSTracker | None = None,
) -> tuple[
    LatentPathT,
    LatentNUTSDiagnostics,
]:
    if config.num_steps is None and config.max_time_s is None:
        raise ValueError(
            "LatentNUTSConfig requires at least one stopping criterion: "
            "num_steps or max_time_s."
        )

    observations = dataset.observations
    conditions = dataset.conditions
    num_sequences = dataset.num_sequences
    fixed_parameters = config.fixed_parameters
    
    def logdensity(latents: LatentPathT) -> jaxtyping.Scalar:
        _validate_dataset_and_latents(latents=latents, dataset=dataset)

        condition_in_axes = None if isinstance(conditions, seqjtyping.NoCondition) else 0

        log_like = jax.vmap(
            lambda latent_path, observation_path, condition_path: evaluate.log_prob_joint(
                target,
                latent_path,
                observation_path,
                condition_path,
                fixed_parameters,
            ),
            in_axes=(0, 0, condition_in_axes),
        )(latents, observations, conditions)

        return jnp.sum(log_like)

    def initial_latents(sample_key: jaxtyping.PRNGKeyArray) -> LatentPathT:
        if config.initial_latents is not None:
            init_latents = config.initial_latents
            _validate_dataset_and_latents(latents=init_latents, dataset=dataset)
            return init_latents

        condition_in_axes = None if isinstance(conditions, seqjtyping.NoCondition) else 0
        simulation_keys = jrandom.split(sample_key, num_sequences)

        simulated_latents, _ = jax.vmap(
            lambda sim_key, condition_path: simulate(
                sim_key,
                target,
                fixed_parameters,
                dataset.sequence_length,
                condition=condition_path,
            ),
            in_axes=(0, condition_in_axes)
        )(simulation_keys, conditions)

        _validate_dataset_and_latents(latents=simulated_latents, dataset=dataset)
        return simulated_latents

    warmup_key, init_key, sample_key = jrandom.split(key, 3)

    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity,
        initial_step_size=config.step_size,
    )
    (_, nuts_config), _ = warmup.run(
        warmup_key,
        initial_latents(init_key),
        num_steps=config.num_adaptation,
    )

    nuts = blackjax.nuts(logdensity, **nuts_config)

    chain_inits = jax.vmap(initial_latents)(jrandom.split(init_key, config.num_chains))
    initial_states = jax.vmap(nuts.init)(chain_inits)

    warmup_states, _ = inference_loop_multiple_chains(
        warmup_key,
        nuts.step,
        initial_states,
        num_samples=config.num_warmup,
        num_chains=config.num_chains,
    )
    jax.block_until_ready(warmup_states)

    current_states = warmup_states
    latent_blocks: list[LatentPathT] = []
    block_times_s: list[tuple[float, int]] = []

    samples_taken = 0
    inference_time_start = time.time()
    next_sample_key = sample_key

    while True:
        sample_key, next_sample_key = jrandom.split(next_sample_key)

        current_states, paths = inference_loop_multiple_chains(
            sample_key,
            nuts.step,
            current_states,
            num_samples=config.sample_block_size,
            num_chains=config.num_chains,
        )
        jax.block_until_ready(current_states)

        raw_latent_block = paths.position

        raw_keep = config.sample_block_size
        if config.num_steps is not None:
            remaining = config.num_steps - samples_taken
            raw_keep = min(raw_keep, remaining)

        stride = config.downsample_stride
        global_offset = samples_taken % stride
        block_offset = (-global_offset) % stride

        latent_block = jax.tree_util.tree_map(
            lambda x: x[block_offset:raw_keep:stride, ...],
            raw_latent_block,
        )
        latent_blocks.append(latent_block)

        samples_taken += raw_keep
        elapsed_time_s = time.time() - inference_time_start
        block_times_s.append((elapsed_time_s, samples_taken))

        if tracker is not None:
            tracker.on_block_end(
                elapsed_time_s=elapsed_time_s,
                samples_taken=samples_taken,
            )

        if config.max_time_s and elapsed_time_s > config.max_time_s:
            break

        if config.num_steps and samples_taken >= config.num_steps:
            break

    latent_samples = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *latent_blocks,
    )

    return latent_samples, LatentNUTSDiagnostics(block_times_s=block_times_s)
