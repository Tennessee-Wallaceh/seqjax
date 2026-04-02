import jax.numpy as jnp
import jax.random as jrandom
import pytest

from seqjax.inference.interface import ObservationDataset
from seqjax.inference.mcmc.nuts_latent import LatentNUTSConfig, run_latent_nuts
from seqjax.model import ar
from seqjax.model import simulate
import seqjax.model.typing as seqjtyping


def _build_dataset(sequence_length: int = 16) -> tuple[ObservationDataset, ar.ARParameters]:
    true_params = ar.ARParameters(
        ar=jnp.array(0.6),
        observation_std=jnp.array(0.4),
        transition_std=jnp.array(0.3),
    )
    _, observations = simulate.simulate(
        jrandom.PRNGKey(0),
        ar.ar_model,
        true_params,
        sequence_length=sequence_length,
        condition=seqjtyping.NoCondition(),
    )
    dataset = ObservationDataset.from_single_sequence(
        observation_path=observations,
        condition_path=seqjtyping.NoCondition(),
    )
    return dataset, true_params


def test_run_latent_nuts_returns_samples() -> None:
    dataset, fixed_params = _build_dataset(sequence_length=12)

    config = LatentNUTSConfig(
        fixed_parameters=fixed_params,
        step_size=1e-2,
        num_adaptation=20,
        num_warmup=20,
        num_steps=24,
        sample_block_size=12,
        num_chains=2,
        downsample_stride=2,
    )

    latent_samples, diagnostics = run_latent_nuts(
        target=ar.ar_model,
        key=jrandom.PRNGKey(7),
        dataset=dataset,
        config=config,
    )

    assert latent_samples.batch_shape[0] == 12
    assert len(diagnostics.block_times_s) > 0


def test_run_latent_nuts_requires_stopping_criterion() -> None:
    dataset, fixed_params = _build_dataset(sequence_length=8)

    with pytest.raises(ValueError, match="stopping criterion"):
        run_latent_nuts(
            target=ar.ar_model,
            key=jrandom.PRNGKey(1),
            dataset=dataset,
            config=LatentNUTSConfig(
                fixed_parameters=fixed_params,
                num_steps=None,
                max_time_s=None,
            ),
        )


def test_run_latent_nuts_rejects_initial_latents_without_sequence_axis() -> None:
    dataset, fixed_params = _build_dataset(sequence_length=10)
    bad_initial_latents, _ = simulate.simulate(
        jrandom.PRNGKey(99),
        ar.ar_model,
        fixed_params,
        sequence_length=dataset.sequence_length,
        condition=seqjtyping.NoCondition(),
    )

    with pytest.raises(ValueError, match="leading num_sequences axis"):
        run_latent_nuts(
            target=ar.ar_model,
            key=jrandom.PRNGKey(2),
            dataset=dataset,
            config=LatentNUTSConfig(
                fixed_parameters=fixed_params,
                num_adaptation=5,
                num_warmup=5,
                num_steps=10,
                sample_block_size=5,
                initial_latents=bad_initial_latents,
            ),
        )
