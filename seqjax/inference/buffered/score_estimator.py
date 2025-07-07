"""
Implement estimators in
Particle Approximations of the Score and Observed
Information Matrix for Parameter Estimation in State Space Models With Linear
Computational Cost

https://eprints.whiterose.ac.uk/id/eprint/83198/9/2_pdfsam_J_of_Comp_Graph_Stats_2015.pdf
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

from seqjax.model.base import (
    ConditionType,
    ObservationType,
    ParametersType,
    ParticleType,
    SequentialModel,
    ParameterPrior,
    HyperParametersType,
)
from seqjax.model.typing import Batched, SequenceAxis
from seqjax.inference.particlefilter import SMCSampler
from .buffered import _run_segment


def _step_joint(
    target,
    parameters,
    particle_history,
    particle,
    observation_history,
    observation,
    condition,
):
    return target.emission.log_prob(
        particle, observation_history, observation, condition, parameters
    ) + target.transition.log_prob(particle_history, particle, condition, parameters)


def run_score_estimator(
    target: SequentialModel[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    particle_filter: SMCSampler[
        ParticleType, ObservationType, ConditionType, ParametersType
    ],
    key,
):

    #
    step_score = jax.grad(partial(_step_joint, target=target))

    # TODO: Accumulate the step score
    run_filter(particle_filter, recorders=())

    # TODO: Build the overall score estimator from recorder return
