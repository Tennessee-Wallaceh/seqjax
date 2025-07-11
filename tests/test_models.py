# ruff: noqa: E402
import pytest

# mark requires jax
jax = pytest.importorskip("jax")
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats


from seqjax import simulate, evaluate
from seqjax.model.ar import AR1Target, ARParameters
from seqjax.model.linear_gaussian import LinearGaussianSSM, LGSSMParameters
from seqjax.model.stochastic_vol import SimpleStochasticVol, LogVolRW, TimeIncrement
from seqjax.model.sir import SIRModel, SIRParameters, SIRPrior, SIRState
from seqjax.model.poisson_ssm import PoissonSSM, PoissonSSMParameters
from seqjax.model.hmm import HiddenMarkovModel, HMMParameters
from seqjax.model.base import Prior, Transition, Emission, SequentialModel
from seqjax.util import pytree_shape
from tests.test_typing import (
    DummyParticle,
    DummyObservation,
    DummyCondition,
    DummyParameters,
)
from typing import ClassVar
from jaxtyping import PRNGKeyArray, Scalar


def test_ar1_target_simulate_length() -> None:
    key = jax.random.PRNGKey(0)
    params = ARParameters()
    latent, obs, x_hist, y_hist = simulate.simulate(
        key, AR1Target, None, params, sequence_length=3
    )

    assert latent.x.shape == (3,)
    assert obs.y.shape == (3,)
    logp = evaluate.log_prob_joint(
        AR1Target,
        latent,
        obs,
        None,
        params,
        x_history=x_hist,
    )
    assert jnp.shape(logp) == ()


def test_linear_gaussian_simulate_length() -> None:
    key = jax.random.PRNGKey(0)
    params = LGSSMParameters()
    latent, obs, x_hist, y_hist = simulate.simulate(
        key, LinearGaussianSSM, None, params, sequence_length=3
    )

    assert latent.x.shape == (3, 2)
    assert obs.y.shape == (3, 2)
    logp = evaluate.log_prob_joint(
        LinearGaussianSSM,
        latent,
        obs,
        None,
        params,
        x_history=x_hist,
    )
    assert jnp.shape(logp) == ()


def test_simple_stochastic_vol_simulate_length() -> None:
    key = jax.random.PRNGKey(0)
    params = LogVolRW(
        std_log_vol=jnp.array(0.1),
        mean_reversion=jnp.array(0.1),
        long_term_vol=jnp.array(1.0),
    )
    cond = TimeIncrement(jnp.array([1.0, 1.0, 1.0, 1.0]))
    latent, obs, x_hist, y_hist = simulate.simulate(
        key, SimpleStochasticVol, cond, params, sequence_length=3
    )

    assert latent.log_vol.shape == (3,)
    assert obs.log_return.shape == (3,)
    assert pytree_shape(x_hist)[0][0] == 0
    assert pytree_shape(y_hist)[0][0] == 0


def test_sir_simulate_length() -> None:
    key = jrandom.PRNGKey(0)
    params = SIRParameters(
        infection_rate=jnp.array(0.1),
        recovery_rate=jnp.array(0.05),
        population=jnp.array(100.0),
    )
    latent, obs, _, _ = simulate.simulate(
        key, SIRModel, None, params, sequence_length=3
    )

    assert latent.s.shape == (3,)
    assert obs.new_cases.shape == (3,)


def test_sir_prior_log_prob_checks_initial_state() -> None:
    params = SIRParameters(
        infection_rate=jnp.array(0.1),
        recovery_rate=jnp.array(0.05),
        population=jnp.array(100.0),
    )
    s0 = params.population - 1
    correct = SIRState(s=s0, i=jnp.array(1.0), r=jnp.array(0.0))
    wrong = SIRState(s=s0 - 1, i=jnp.array(2.0), r=jnp.array(0.0))

    logp_ok = SIRPrior.log_prob((correct, correct), (None, None), params)
    logp_bad_first = SIRPrior.log_prob((wrong, correct), (None, None), params)
    logp_bad_second = SIRPrior.log_prob((correct, wrong), (None, None), params)

    assert jnp.array_equal(logp_ok, jnp.array(0.0))
    assert jnp.isneginf(logp_bad_first)
    assert jnp.isneginf(logp_bad_second)


def test_poisson_ssm_simulate_length() -> None:
    key = jax.random.PRNGKey(0)
    params = PoissonSSMParameters()
    latent, obs, x_hist, y_hist = simulate.simulate(
        key, PoissonSSM, None, params, sequence_length=3
    )

    assert latent.log_rate.shape == (3,)
    assert obs.count.shape == (3,)

def test_hmm_simulate_length() -> None:
    key = jax.random.PRNGKey(0)
    params = HMMParameters(
        initial_probs=jnp.array([0.6, 0.4]),
        transition_matrix=jnp.array([[0.7, 0.3], [0.2, 0.8]]),
        emission_probs=jnp.array([[0.9, 0.1], [0.2, 0.8]]),
    )
    latent, obs, x_hist, y_hist = simulate.simulate(
        key, HiddenMarkovModel, None, params, sequence_length=3
    )

    assert latent.z.shape == (3,)
    assert obs.y.shape == (3,)
    logp = evaluate.log_prob_joint(
        HiddenMarkovModel,
        latent,
        obs,
        None,
        params,
        x_history=x_hist,
    )

    assert jnp.shape(logp) == ()


class Prior1(Prior[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[DummyCondition],
        parameters: DummyParameters,
    ) -> tuple[DummyParticle]:
        return (DummyParticle(jrandom.normal(key)),)

    @staticmethod
    def log_prob(
        particle: tuple[DummyParticle],
        conditions: tuple[DummyCondition],
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Transition1(Transition[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[DummyParticle],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyParticle:
        return DummyParticle(jrandom.normal(key))

    @staticmethod
    def log_prob(
        particle_history: tuple[DummyParticle],
        particle: DummyParticle,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Emission1(
    Emission[DummyParticle, DummyObservation, DummyCondition, DummyParameters]
):
    order: ClassVar[int] = 1
    observation_dependency: ClassVar[int] = 0

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[DummyParticle],
        observation_history: tuple[()],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyObservation:
        return DummyObservation(jrandom.normal(key))

    @staticmethod
    def log_prob(
        particle: tuple[DummyParticle],
        observation_history: tuple[()],
        observation: DummyObservation,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Target1(
    SequentialModel[DummyParticle, DummyObservation, DummyCondition, DummyParameters]
):
    prior = Prior1()
    transition = Transition1()
    emission = Emission1()


class Prior2(Prior[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 2

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[DummyCondition, DummyCondition],
        parameters: DummyParameters,
    ) -> tuple[DummyParticle, DummyParticle]:
        k1, k2 = jrandom.split(key)
        return (
            DummyParticle(jrandom.normal(k1)),
            DummyParticle(jrandom.normal(k2)),
        )

    @staticmethod
    def log_prob(
        particle: tuple[DummyParticle, DummyParticle],
        conditions: tuple[DummyCondition, DummyCondition],
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Transition2(Transition[DummyParticle, DummyCondition, DummyParameters]):
    order: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[DummyParticle],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyParticle:
        return DummyParticle(jrandom.normal(key))

    @staticmethod
    def log_prob(
        particle_history: tuple[DummyParticle],
        particle: DummyParticle,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Emission2(
    Emission[DummyParticle, DummyObservation, DummyCondition, DummyParameters]
):
    order: ClassVar[int] = 2
    observation_dependency: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[DummyParticle, DummyParticle],
        observation_history: tuple[DummyObservation],
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> DummyObservation:
        return DummyObservation(jrandom.normal(key))

    @staticmethod
    def log_prob(
        particle: tuple[DummyParticle, DummyParticle],
        observation_history: tuple[DummyObservation],
        observation: DummyObservation,
        condition: DummyCondition,
        parameters: DummyParameters,
    ) -> Scalar:
        return jnp.array(0.0)


class Target2(
    SequentialModel[DummyParticle, DummyObservation, DummyCondition, DummyParameters]
):
    prior = Prior2()
    transition = Transition2()
    emission = Emission2()


@pytest.mark.parametrize(
    "target,obs_dep,seq_len,cond_len,should_fail",
    [
        (Target1, 0, 3, 3, False),
        (Target2, 1, 3, 4, False),
        (Target2, 1, 3, 3, True),
    ],
)
def test_simulate_dependency_lengths(
    target,
    obs_dep,
    seq_len,
    cond_len,
    should_fail,
) -> None:
    key = jax.random.PRNGKey(1)
    params = DummyParameters(
        reference_emission=tuple(
            DummyObservation(jnp.array(0.0)) for _ in range(obs_dep)
        )
    )
    condition = DummyCondition(jnp.ones(cond_len))
    if should_fail:
        with pytest.raises(jax.errors.JaxRuntimeError):
            simulate.simulate(key, target, condition, params, seq_len)
    else:
        latent, obs, x_hist, y_hist = simulate.simulate(
            key, target, condition, params, seq_len
        )
        assert pytree_shape(latent)[0][0] == seq_len
        assert pytree_shape(obs)[0][0] == seq_len
        assert pytree_shape(x_hist)[0][0] == target.prior.order - 1
        assert pytree_shape(y_hist)[0][0] == obs_dep


def test_ar1_joint_log_prob_closed_form() -> None:
    key = jrandom.PRNGKey(0)
    params = ARParameters(
        ar=jnp.array(0.5),
        observation_std=jnp.array(1.0),
        transition_std=jnp.array(0.3),
    )
    latents, observations, x_hist, _ = simulate.simulate(
        key, AR1Target, None, params, sequence_length=3
    )

    x = latents.x
    y = observations.y

    manual_logp = jstats.norm.logpdf(x[0], loc=0.0, scale=params.transition_std)
    manual_logp += jstats.norm.logpdf(y[0], loc=x[0], scale=params.observation_std)
    manual_logp += jnp.sum(
        jstats.norm.logpdf(x[1:], loc=params.ar * x[:-1], scale=params.transition_std)
    )
    manual_logp += jnp.sum(
        jstats.norm.logpdf(y[1:], loc=x[1:], scale=params.observation_std)
    )

    eval_logp = evaluate.log_prob_joint(
        AR1Target,
        latents,
        observations,
        None,
        params,
        x_history=x_hist,
    )
    assert jnp.allclose(manual_logp, eval_logp)
