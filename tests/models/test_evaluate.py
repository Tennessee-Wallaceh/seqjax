import jax
import jax.numpy as jnp
from typing import ClassVar


from collections import OrderedDict
from seqjax.model import evaluate
import seqjax.model.typing as seqjtyping
from seqjax.model.base import SequentialModel, Prior, Transition, Emission
from jaxtyping import PRNGKeyArray, Scalar


"""
Packables
"""


class TestLatent(seqjtyping.Latent):
    x: Scalar
    _shape_template: ClassVar = OrderedDict(
        x=jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
    )


class TestObs(seqjtyping.Observation):
    y: Scalar
    _shape_template: ClassVar = OrderedDict(
        y=jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
    )


class TestParam(seqjtyping.Parameters):
    theta: Scalar
    _shape_template: ClassVar = OrderedDict(
        theta=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32),
    )


"""
Priors
"""


def prior_order_1_sample(
    key: PRNGKeyArray,
    conditions: tuple[seqjtyping.NoCondition],
    parameters: TestParam,
) -> tuple[TestLatent]:
    """Sample the initial latent value."""
    return (TestLatent(x=jnp.array(0)),)


def prior_order_1_log_prob(
    latent: tuple[TestLatent],
    conditions: tuple[seqjtyping.NoCondition],
    parameters: TestParam,
) -> Scalar:
    """Evaluate the prior log-density."""
    assert len(latent) == 1
    return latent[0].x + parameters.theta


prior_order_1 = Prior[tuple[TestLatent], tuple[seqjtyping.NoCondition], TestParam](
    order=1,
    sample=prior_order_1_sample,
    log_prob=prior_order_1_log_prob,
)


def prior_order_2_sample(
    key: PRNGKeyArray,
    conditions: tuple[seqjtyping.NoCondition, seqjtyping.NoCondition],
    parameters: TestParam,
) -> tuple[TestLatent, TestLatent]:
    """Sample the initial latent value."""
    return (TestLatent(x=jnp.array(-1)), TestLatent(x=jnp.array(0)))


def prior_order_2_log_prob(
    latent: tuple[TestLatent, TestLatent],
    conditions: tuple[seqjtyping.NoCondition, seqjtyping.NoCondition],
    parameters: TestParam,
) -> Scalar:
    """Evaluate the prior log-density."""
    latent_m1, latent_0 = latent
    return latent_m1.x + latent_0.x + parameters.theta


prior_order_2 = Prior[
    tuple[TestLatent, TestLatent],
    tuple[seqjtyping.NoCondition, seqjtyping.NoCondition],
    TestParam,
](
    order=2,
    sample=prior_order_2_sample,
    log_prob=prior_order_2_log_prob,
)

"""
Transitions
"""


def transition_order_1_sample(
    key: PRNGKeyArray,
    latent_history: tuple[TestLatent],
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> TestLatent:
    """Sample the next latent state."""
    (last_latent,) = latent_history
    return TestLatent(x=last_latent.x + 1)


def transition_order_1_log_prob(
    latent_history: tuple[TestLatent],
    latent: TestLatent,
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> Scalar:
    """Return the transition log-density."""
    (last_latent,) = latent_history
    return latent.x + last_latent.x + parameters.theta


transition_order_1 = Transition[
    tuple[TestLatent],
    TestLatent,
    seqjtyping.NoCondition,
    TestParam,
](
    order=1,
    sample=transition_order_1_sample,
    log_prob=transition_order_1_log_prob,
)


def transition_order_2_sample(
    key: PRNGKeyArray,
    latent_history: tuple[TestLatent, TestLatent],
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> TestLatent:
    """Sample the next latent state."""
    (
        latent_m1,
        last_latent,
    ) = latent_history
    return TestLatent(x=last_latent.x + 1)


def transition_order_2_log_prob(
    latent_history: tuple[TestLatent, TestLatent],
    latent: TestLatent,
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> Scalar:
    """Return the transition log-density."""
    (
        latent_m1,
        last_latent,
    ) = latent_history
    return latent.x + last_latent.x + parameters.theta


transition_order_2 = Transition[
    tuple[TestLatent, TestLatent],
    TestLatent,
    seqjtyping.NoCondition,
    TestParam,
](
    order=2,
    sample=transition_order_2_sample,
    log_prob=transition_order_2_log_prob,
)

"""
Emissions
"""


def emission_order_1_sample(
    key: PRNGKeyArray,
    latent: tuple[TestLatent],
    observation_history: tuple[()],
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> TestObs:
    """Sample an observation."""
    (current_latent,) = latent
    y = current_latent.x
    return TestObs(y=y)


def emission_order_1_log_prob(
    latent: tuple[TestLatent],
    observation: TestObs,
    observation_history: tuple[()],
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> Scalar:
    """Return the emission log-density."""
    (current_latent,) = latent
    return observation.y + current_latent.x + parameters.theta


emission_order_1 = Emission[
    tuple[TestLatent],
    seqjtyping.NoCondition,
    TestObs,
    TestParam,
](
    sample=emission_order_1_sample,
    log_prob=emission_order_1_log_prob,
    order=1,
)


def emission_order_2_sample(
    key: PRNGKeyArray,
    latent: tuple[TestLatent, TestLatent],
    observation_history: tuple[()],
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> TestObs:
    """Sample an observation."""
    (last_latent, current_latent) = latent
    y = current_latent.x
    return TestObs(y=y)


def emission_order_2_log_prob(
    latent: tuple[TestLatent, TestLatent],
    observation: TestObs,
    observation_history: tuple[()],
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> Scalar:
    """Return the emission log-density."""
    (last_latent, current_latent) = latent
    return observation.y + current_latent.x + last_latent.x + parameters.theta


emission_order_2 = Emission[
    tuple[TestLatent, TestLatent],
    seqjtyping.NoCondition,
    TestObs,
    TestParam,
](
    sample=emission_order_2_sample,
    log_prob=emission_order_2_log_prob,
    order=2,
)


def emission_order_1_dep_1_sample(
    key: PRNGKeyArray,
    latent: tuple[TestLatent],
    observation_history: tuple[TestObs],
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> TestObs:
    """Sample an observation."""
    (current_latent,) = latent
    (last_obs,) = observation_history
    y = current_latent.x
    return TestObs(y=y)


def emission_order_1_dep_1_log_prob(
    latent: tuple[TestLatent],
    observation: TestObs,
    observation_history: tuple[TestObs],
    condition: seqjtyping.NoCondition,
    parameters: TestParam,
) -> Scalar:
    """Return the emission log-density."""
    (current_latent,) = latent
    (last_obs,) = observation_history
    return observation.y + current_latent.x + last_obs.y + parameters.theta


emission_order_1_dep_1 = Emission[
    tuple[TestLatent], seqjtyping.NoCondition, TestObs, TestParam, tuple[TestObs]
](
    sample=emission_order_1_dep_1_sample,
    log_prob=emission_order_1_dep_1_log_prob,
    order=1,
    observation_dependency=1,
)


class Target_TO2_EO1(
    SequentialModel[
        TestLatent,
        TestObs,
        seqjtyping.NoCondition,
        TestParam,
        tuple[TestLatent, TestLatent],
        tuple[seqjtyping.NoCondition, seqjtyping.NoCondition],
        tuple[TestLatent, TestLatent],
        tuple[TestLatent],
    ]
):
    latent_cls = TestLatent
    observation_cls = TestObs
    parameter_cls = TestParam
    condition_cls = seqjtyping.NoCondition

    prior = prior_order_2
    transition = transition_order_2
    emission = emission_order_1

    reference_emission = ()


class Target_TO1_EO1_ED1(
    SequentialModel[
        TestLatent,
        TestObs,
        seqjtyping.NoCondition,
        TestParam,
        tuple[TestLatent],
        tuple[seqjtyping.NoCondition],
        tuple[TestLatent],
        tuple[TestLatent],
        tuple[TestObs],
    ]
):
    latent_cls = TestLatent
    observation_cls = TestObs
    parameter_cls = TestParam
    condition_cls = seqjtyping.NoCondition

    prior = prior_order_1
    transition = transition_order_1
    emission = emission_order_1_dep_1
    reference_emission = (TestObs(jnp.array(-1)),)


class Target_TO2_EO2(
    SequentialModel[
        TestLatent,
        TestObs,
        seqjtyping.NoCondition,
        TestParam,
        tuple[TestLatent, TestLatent],
        tuple[seqjtyping.NoCondition, seqjtyping.NoCondition],
        tuple[TestLatent, TestLatent],
        tuple[TestLatent, TestLatent],
    ]
):
    latent_cls = TestLatent
    observation_cls = TestObs
    parameter_cls = TestParam
    condition_cls = seqjtyping.NoCondition

    prior = prior_order_2
    transition = transition_order_2
    emission = emission_order_2

    reference_emission = ()


class Target_TO1_EO2(
    SequentialModel[
        TestLatent,
        TestObs,
        seqjtyping.NoCondition,
        TestParam,
        tuple[TestLatent, TestLatent],
        tuple[seqjtyping.NoCondition, seqjtyping.NoCondition],
        tuple[TestLatent],
        tuple[TestLatent, TestLatent],
    ]
):
    latent_cls = TestLatent
    observation_cls = TestObs
    parameter_cls = TestParam
    condition_cls = seqjtyping.NoCondition

    prior = prior_order_2
    transition = transition_order_1
    emission = emission_order_2

    reference_emission = ()


def test_TO2_EO2():
    x_path = TestLatent(jnp.arange(-1, 5))
    y_path = TestObs(jnp.arange(0, 5))
    target = Target_TO2_EO2()
    params = TestParam(theta=jnp.array(0.0))

    prior_conditions = evaluate.slice_prior_conditions(
        seqjtyping.NoCondition(), target.prior
    )

    assert prior_conditions == (seqjtyping.NoCondition(), seqjtyping.NoCondition())

    prior_latents = evaluate.slice_prior_latent(x_path, target.prior)

    assert prior_latents == (
        TestLatent(jnp.array(-1)),
        TestLatent(jnp.array(0)),
    )

    transition_histories = evaluate.slice_transition_latent_history(
        x_path,
        target.transition,
        target.prior,
    )

    assert transition_histories == (
        TestLatent(jnp.array([-1, 0, 1, 2])),
        TestLatent(jnp.array([0, 1, 2, 3])),
    )

    emission_histories = evaluate.slice_emission_latent_history(
        x_path,
        target.emission,
        target.prior,
    )

    # The transition histories includes the final latent ix 4 to
    # evaluate y[4] | x[4], x[3]
    assert emission_histories == (
        TestLatent(jnp.array([-1, 0, 1, 2, 3])),
        TestLatent(jnp.array([0, 1, 2, 3, 4])),
    )

    out = evaluate.log_prob_x(target, x_path, seqjtyping.NoCondition(), params)
    # test log probs just sum latent values
    # so we get prior -1 + 0
    # then (0 + 1) + (1 + 2) + (2 + 3) + (3 + 4)
    # = 15
    assert out == jnp.array(15)

    emission_observation_history = evaluate.slice_emission_observation_history(
        y_path,
        target.emission,
    )

    assert emission_observation_history == ()

    out = evaluate.log_prob_y_given_x(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    # (-1 + 0 + 0) + (0 + 1 + 1) + ... + (3 + 4 + 4)
    assert out == jnp.array(25)

    log_p_joint = evaluate.log_prob_joint(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    assert log_p_joint == jnp.array(40)


def test_TO1_EO1_ED1():
    x_path = TestLatent(jnp.arange(0, 5))
    y_path = TestObs(jnp.arange(-1, 5))
    target = Target_TO1_EO1_ED1()
    params = TestParam(theta=jnp.array(0.0))

    prior_conditions = evaluate.slice_prior_conditions(
        seqjtyping.NoCondition(), target.prior
    )

    assert prior_conditions == (seqjtyping.NoCondition(),)

    prior_latents = evaluate.slice_prior_latent(x_path, target.prior)

    assert prior_latents == (TestLatent(jnp.array(0)),)

    transition_histories = evaluate.slice_transition_latent_history(
        x_path,
        target.transition,
        target.prior,
    )

    assert transition_histories == (TestLatent(jnp.array([0, 1, 2, 3])),)

    emission_histories = evaluate.slice_emission_latent_history(
        x_path,
        target.emission,
        target.prior,
    )

    # The transition histories includes the final latent ix 4 to
    # evaluate y[4] | x[4], x[3]
    assert emission_histories == (TestLatent(jnp.array([0, 1, 2, 3, 4])),)

    out = evaluate.log_prob_x(target, x_path, seqjtyping.NoCondition(), params)
    # test log probs just sum latent values
    # prior 0
    # then (0 + 1) + (1 + 2) + (2 + 3) + (3 + 4)
    # = 16
    assert out == jnp.array(16)

    emission_observation_history = evaluate.slice_emission_observation_history(
        y_path,
        target.emission,
    )

    assert emission_observation_history == (TestObs(jnp.array([-1, 0, 1, 2, 3])),)

    out = evaluate.log_prob_y_given_x(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    # (-1 + 0 + 0) + (0 + 1 + 1) + ... + (3 + 4 + 4)
    assert out == jnp.array(25)

    log_p_joint = evaluate.log_prob_joint(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    assert log_p_joint == jnp.array(41)


def test_TO2_EO1():
    x_path = TestLatent(jnp.arange(-1, 5))
    y_path = TestObs(jnp.arange(0, 5))
    target = Target_TO2_EO1()
    params = TestParam(theta=jnp.array(0.0))

    prior_conditions = evaluate.slice_prior_conditions(
        seqjtyping.NoCondition(), target.prior
    )

    assert prior_conditions == (seqjtyping.NoCondition(), seqjtyping.NoCondition())

    prior_latents = evaluate.slice_prior_latent(x_path, target.prior)

    assert prior_latents == (
        TestLatent(jnp.array(-1)),
        TestLatent(jnp.array(0)),
    )

    transition_histories = evaluate.slice_transition_latent_history(
        x_path,
        target.transition,
        target.prior,
    )

    assert transition_histories == (
        TestLatent(jnp.array([-1, 0, 1, 2])),
        TestLatent(jnp.array([0, 1, 2, 3])),
    )

    emission_histories = evaluate.slice_emission_latent_history(
        x_path,
        target.emission,
        target.prior,
    )

    # The transition histories includes the final latent ix 4 to
    # evaluate y[4] | x[4]
    assert emission_histories == (TestLatent(jnp.array([0, 1, 2, 3, 4])),)

    out = evaluate.log_prob_x(target, x_path, seqjtyping.NoCondition(), params)
    # test log probs just sum latent values
    # prior -1
    # then (0 + 1) + (1 + 2) + (2 + 3) + (3 + 4)
    # = 15
    assert out == jnp.array(15)

    emission_observation_history = evaluate.slice_emission_observation_history(
        y_path,
        target.emission,
    )

    assert emission_observation_history == ()

    out = evaluate.log_prob_y_given_x(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    # 2 * (1 + ... + 4)
    assert out == jnp.array(20)

    log_p_joint = evaluate.log_prob_joint(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    assert log_p_joint == jnp.array(35)


def test_TO1_EO2():
    x_path = TestLatent(jnp.arange(-1, 5))
    y_path = TestObs(jnp.arange(0, 5))
    target = Target_TO1_EO2()
    params = TestParam(theta=jnp.array(0.0))

    prior_conditions = evaluate.slice_prior_conditions(
        seqjtyping.NoCondition(), target.prior
    )

    assert prior_conditions == (seqjtyping.NoCondition(), seqjtyping.NoCondition())

    prior_latents = evaluate.slice_prior_latent(x_path, target.prior)

    assert prior_latents == (
        TestLatent(jnp.array(-1)),
        TestLatent(jnp.array(0)),
    )

    transition_histories = evaluate.slice_transition_latent_history(
        x_path,
        target.transition,
        target.prior,
    )

    assert transition_histories == (TestLatent(jnp.array([0, 1, 2, 3])),)

    emission_histories = evaluate.slice_emission_latent_history(
        x_path,
        target.emission,
        target.prior,
    )

    # The transition histories includes the final latent ix 4 to
    # evaluate y[4] | x[4],  x[3]
    assert emission_histories == (
        TestLatent(jnp.array([-1, 0, 1, 2, 3])),
        TestLatent(jnp.array([0, 1, 2, 3, 4])),
    )

    out = evaluate.log_prob_x(target, x_path, seqjtyping.NoCondition(), params)
    # -(1 + 1 + 2 + 3)  + (1 + 2 + 3+ 4)
    assert out == jnp.array(15)

    emission_observation_history = evaluate.slice_emission_observation_history(
        y_path,
        target.emission,
    )

    assert emission_observation_history == ()

    out = evaluate.log_prob_y_given_x(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    # (-1 + 0 + 0) + (0 + 1 + 1) + (1 + 2 + 2) + (2 +  3+ 3) + (3 + 4+ 4)
    assert out == jnp.array(25)

    log_p_joint = evaluate.log_prob_joint(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    assert log_p_joint == jnp.array(40)


def test_auto_batching():
    x_path = TestLatent(jnp.arange(-1, 5))
    y_path = TestObs(jnp.arange(0, 5))
    target = Target_TO1_EO2()

    # non batched case
    params = TestParam(theta=jnp.array(1.0))
    # we provide a param for each [0, ..., T]
    params_seq = TestParam(theta=1.0 * jnp.ones_like(y_path.y))

    log_p_x = evaluate.log_prob_x(target, x_path, seqjtyping.NoCondition(), params)
    log_p_x_seq = evaluate.log_prob_x(
        target, x_path, seqjtyping.NoCondition(), params_seq
    )
    assert log_p_x == log_p_x_seq

    log_p_y = evaluate.log_prob_y_given_x(
        target, x_path, y_path, seqjtyping.NoCondition(), params
    )
    log_p_y_seq = evaluate.log_prob_y_given_x(
        target, x_path, y_path, seqjtyping.NoCondition(), params_seq
    )
    assert log_p_y == log_p_y_seq

    # try a new param seq
    new_params_seq = TestParam(theta=jnp.arange(len(y_path.y)))
    log_p_x_new_seq = evaluate.log_prob_x(
        target, x_path, seqjtyping.NoCondition(), new_params_seq
    )
    log_p_y_new_seq = evaluate.log_prob_y_given_x(
        target, x_path, y_path, seqjtyping.NoCondition(), new_params_seq
    )

    # we should just accumulate the path
    new_contribution = jnp.sum(jnp.arange(len(y_path.y))) - len(y_path.y)
    assert log_p_y_new_seq == new_contribution + log_p_y
    assert log_p_x_new_seq == new_contribution + log_p_x
