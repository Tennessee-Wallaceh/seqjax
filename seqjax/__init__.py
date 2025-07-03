"""Utilities for sequential probabilistic models built on JAX."""

# Re-export frequently used modules and classes from the package root for
# convenience.  Users can simply ``import seqjax`` and access these components
# without needing to know the underlying module structure.

# simulation and evaluation helpers
from .model import evaluate, simulate
from .model.visualise import graph_model

# base model interfaces
from .model.base import Emission, Prior, SequentialModel, Transition
from .inference.particlefilter import Proposal

# simple inference utilities
from .inference.particlefilter import BootstrapParticleFilter, AuxiliaryParticleFilter
from .inference.mcmc import NUTSConfig, run_nuts, run_bayesian_nuts
from .inference.interface import InferenceMethod, LatentInferenceMethod
from .inference import (
    BufferedConfig,
    run_buffered_filter,
    BufferedSGLDConfig,
    run_buffered_sgld,
)
from .inference.pmcmc import RandomWalkConfig, ParticleMCMCConfig, run_particle_mcmc
from .inference.kalman import run_kalman_filter

__all__ = [
    "simulate",
    "evaluate",
    "Prior",
    "Transition",
    "Proposal",
    "Emission",
    "SequentialModel",
    "graph_model",
    "BootstrapParticleFilter",
    "AuxiliaryParticleFilter",
    "NUTSConfig",
    "run_nuts",
    "run_bayesian_nuts",
    "InferenceMethod",
    "LatentInferenceMethod",
    "BufferedConfig",
    "run_buffered_filter",
    "BufferedSGLDConfig",
    "run_buffered_sgld",
    "RandomWalkConfig",
    "ParticleMCMCConfig",
    "run_particle_mcmc",
    "run_kalman_filter",
]
from .inference.autoregressive_vi import (
    Sampler,
    Autoregressor,
    RandomAutoregressor,
    AmortizedUnivariateAutoregressor,
    AmortizedResidualUnivariateAutoregressor,
    AmortizedMultivariateAutoregressor,
    AmortizedMultivariateIsotropicAutoregressor,
)

__all__ += [
    "Sampler",
    "Autoregressor",
    "RandomAutoregressor",
    "AmortizedUnivariateAutoregressor",
    "AmortizedResidualUnivariateAutoregressor",
    "AmortizedMultivariateAutoregressor",
    "AmortizedMultivariateIsotropicAutoregressor",
]
