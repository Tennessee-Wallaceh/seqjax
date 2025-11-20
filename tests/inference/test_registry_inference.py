"""Tests covering all registered inference methods."""

from collections.abc import Iterator
from functools import partial

import jax.numpy as jnp
import jax.random as jrandom
import pytest

from _pytest.mark.structures import ParameterSet

from tqdm import auto as tqdm_auto
from tqdm import notebook as tqdm_notebook

from seqjax.inference import (
    mcmc,
    particlefilter,
    pmcmc,
    registry as inference_registry,
    sgld,
    vi,
)
from seqjax.model import ar, registry as model_registry, simulate


class _DummyTqdmIterator(Iterator[int]):
    """Minimal iterator with ``set_postfix`` to satisfy ``tqdm`` usage in tests."""

    def __init__(self, n: int) -> None:
        self._iter = iter(range(int(n)))

    def __iter__(self) -> "_DummyTqdmIterator":
        return self

    def __next__(self) -> int:
        return next(self._iter)

    def set_postfix(self, *_args, **_kwargs) -> None:  # pragma: no cover - behaviourless stub
        """Match the API used in :mod:`seqjax.inference.vi.train`."""

@pytest.fixture(autouse=True)
def _stub_tqdm_trange(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``tqdm`` progress bars with a deterministic iterator."""

    def _trange(n: int, *_args, **_kwargs) -> _DummyTqdmIterator:
        return _DummyTqdmIterator(n)

    monkeypatch.setattr(tqdm_auto, "trange", _trange)
    monkeypatch.setattr(tqdm_notebook, "trange", _trange)


INFERENCE_TEST_SETUPS: dict[str, tuple[object, int]] = {
    "NUTS": (
        mcmc.NUTSConfig(step_size=1e-1, num_adaptation=5, num_warmup=5, num_chains=1),
        10,
    ),
    "buffer-vi": (
        vi.BufferedVIConfig(
            optimization=vi.run.AdamOpt(lr=1e-2, total_steps=2),
            buffer_length=2,
            batch_length=4,
            parameter_field_bijections={"ar": "sigmoid"},
            observations_per_step=1,
            samples_per_context=1,
            control_variate=False,
            pre_training_steps=0,
            latent_approximation=vi.run.AutoregressiveLatentApproximation(
                nn_width=4,
                nn_depth=1,
            ),
        ),
        200,
    ),
    "full-vi": (
        vi.FullVIConfig(
            optimization=vi.run.AdamOpt(lr=1e-2, total_steps=2),
            parameter_field_bijections={"ar": "sigmoid"},
            observations_per_step=1,
            samples_per_context=1,
        ),
        200,
    ),
    "particle-mcmc": (
        pmcmc.ParticleMCMCConfig(
            particle_filter=particlefilter.BootstrapParticleFilter(
                target=ar.AR1Target(),
                num_particles=8,
            ),
            mcmc=mcmc.RandomWalkConfig(step_size=0.05),
            initial_parameter_guesses=3,
        ),
        20,
    ),
    "full-sgld": (
        sgld.SGLDConfig(
            particle_filter=particlefilter.BootstrapParticleFilter(
                target=ar.AR1Target(),
                num_particles=8,
                target_parameters=partial(
                    ar.fill_parameter,
                    ref_params=model_registry.parameter_settings["ar"]["base"],
                ),
            ),
            step_size=1e-3,
            num_samples=20,
            initial_parameter_guesses=3,
        ),
        20,
    ),
}


def test_full_sgld_inference_name_includes_particle_count() -> None:
    """The SGLD inference name should encode both samples and particles."""

    config = sgld.SGLDConfig(
        particle_filter=particlefilter.BootstrapParticleFilter(
            target=ar.AR1Target(),
            num_particles=5,
        ),
        num_samples=7,
    )
    inference = inference_registry.FullSGLDInference(method="full-sgld", config=config)

    assert inference.name == "full-sgld-n7-p5"


def _iter_registry_entries() -> list[ParameterSet]:
    entries: list[ParameterSet] = []
    for index, (method_label, inference_fn) in enumerate(
        inference_registry.inference_functions.items()
    ):
        config_entry = INFERENCE_TEST_SETUPS.get(method_label)
        if config_entry is None:
            raise AssertionError(f"Missing test configuration for inference method {method_label!r}")
        config, test_samples = config_entry
        entries.append(
            pytest.param(
                method_label,
                inference_fn,
                config,
                test_samples,
                index,
                id=method_label,
            )
        )
    return entries


@pytest.fixture(scope="module")
def ar1_problem() -> tuple[ar.AR1Bayesian, ar.NoisyEmission]:
    """Simulate a short AR(1) path that can be reused across inference tests."""

    sequence_length = 10
    parameters = model_registry.parameter_settings["ar"]["base"]
    posterior = ar.AR1Bayesian(parameters)
    _latents, observations = simulate.simulate(
        jrandom.PRNGKey(0),
        posterior.target,
        condition=None,
        parameters=parameters,
        sequence_length=sequence_length,
    )
    assert observations.batch_shape[0] == sequence_length
    return posterior, observations


@pytest.mark.parametrize(
    "method_label,inference_fn,config,test_samples,case_index",
    _iter_registry_entries(),
)
def test_registered_inference_methods_can_run(
    method_label: str,
    inference_fn,
    config,
    test_samples: int,
    case_index: int,
    ar1_problem: tuple[ar.AR1Bayesian, ar.NoisyEmission],
) -> None:
    """Every registered inference routine should execute on the AR(1) example."""

    posterior, observations = ar1_problem
    key = jrandom.fold_in(jrandom.PRNGKey(2024), case_index)

    parameter_samples, extra_data = inference_fn(
        posterior,
        hyperparameters=None,
        key=key,
        observation_path=observations,
        condition_path=None,
        test_samples=test_samples,
        config=config,
    )

    assert isinstance(parameter_samples, ar.AROnlyParameters)

    ar_samples = jnp.asarray(parameter_samples.ar)
    assert ar_samples.shape[-1] == test_samples
    assert ar_samples.size > 0
    assert bool(jnp.isfinite(ar_samples).all())

    assert isinstance(extra_data, tuple)
