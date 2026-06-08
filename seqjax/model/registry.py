"""Central registry for sequential models, parameter presets, and conditions.

The registries in this module let experiments and tests discover the models
that ship with :mod:`seqjax` without having to know the implementation
details.  To register a new model:

1. Add its label to :data:`SequentialModelLabel`.
2. Insert the model class into :data:`sequential_models` under the same label.
3. Provide any reusable parameter presets in :data:`parameter_settings` using
   nested dictionaries where the outer key matches the model label and the
   inner keys name the presets (e.g. ``"base"``).
4. If the model requires structured conditioning data, register a callable in
   :data:`condition_generators` that produces a pytree of conditions for a
   given sequence length.

Labels must be consistent across all three registries so that
``sequential_models[label]``, ``parameter_settings[label]``, and
``condition_generators[label]`` refer to the same model family.
"""

from dataclasses import dataclass
import typing

import jax.numpy as jnp

from . import (
    interface, 
    ar, 
    ar_bayesian, 
    stochastic_vol,
    linear_gaussian_bayesian, 
    linear_gaussian,
    double_well,
    double_well_bayesian,
)

from .typing import Parameters, HyperParameters, NoHyper

PosteriorFactory = typing.Callable[
    [HyperParameters], 
    interface.BayesianSequentialModelProtocol
]
SequentialModelLabel = typing.Literal[
    "ar", 
    "svar",
    "lgssm-5d",
    "double-well",
]

BayesianModelLabel = typing.Literal[
    "ar-aronly",
    "ar-full",
    "svar-full",
    "lgssm-5d-full",
    "double-well-ebonly",
]

# Maps each model label to its ``SequentialModel`` implementation. The keys
# must appear in ``SequentialModelLabel`` and are typically accessed via
# ``sequential_models[label]``. When adding a new model, extend
# ``SequentialModelLabel`` and add the class here with the same label.
sequential_models: dict[SequentialModelLabel, interface.SequentialModelProtocol] = {
    "ar": ar.ar_model,
    "svar": stochastic_vol.simple_var.simple_stochastic_var_model,
    "lgssm-5d": linear_gaussian.lgssm(dim=5),
}

# Factories that create a ``BayesianSequentialModel`` for each target model
# label. These factories are typically called with hyperparameters to
# construct the posterior used by inference algorithms.
posterior_factories: dict[BayesianModelLabel, PosteriorFactory] = {
    "ar-aronly": typing.cast(PosteriorFactory, ar_bayesian.ar_only),
    "ar-full": typing.cast(PosteriorFactory, ar_bayesian.ar_full),
    "svar-full": stochastic_vol.simple_var.svar_full,
    "lgssm-5d-full": linear_gaussian_bayesian.lgssm_full,
    "double-well-ebonly": double_well_bayesian.eb_only,
}

# Predefined parameter presets for each model. The outer keys mirror
# ``sequential_models`` and thus use ``SequentialModelLabel``. Inner keys are
# free-form preset names (e.g. ``"base"``) that experiments reference with
# ``parameter_settings[label][preset]``. Add new presets under the appropriate
# model label and keep labels consistent across both dictionaries.
parameter_settings: dict[BayesianModelLabel, dict[str, Parameters]] = {
    "ar-full": {
        "base": ar.ARParameters(
            ar=jnp.array(0.8),
            observation_std=jnp.array(0.1),
            transition_std=jnp.array(0.5),
        ),
        "lower-ar": ar.ARParameters(
            ar=jnp.array(0.5),
            observation_std=jnp.array(0.1),
            transition_std=jnp.array(0.5),
        ),
    },
    "ar-aronly": {
        "base": ar.ARParameters(
            ar=jnp.array(0.8),
            observation_std=jnp.array(0.1),
            transition_std=jnp.array(0.5),
        ),
    },
    "svar-full": {
        # base is ~ equity like daily sampling
        # annual log_vol_std ~ 0.2 * 16.
        "base": stochastic_vol.simple_var.LogVarParams(
            ar=jnp.array(0.90),
            std_log_var=jnp.array(0.40),
            long_term_log_var=2 * jnp.log(jnp.array(0.16)),
        ),
        # high-freq is similar to a minute level sampling
        # much lower std log var + higher ar
        # higher long term var as this is similar to crypto
        "high-freq": stochastic_vol.simple_var.LogVarParams(
            ar=jnp.array(0.95),
            std_log_var=jnp.array(0.10),
            long_term_log_var=2 * jnp.log(jnp.array(0.60)),
        ),
    },
    "lgssm-5d-full": {
        "base": linear_gaussian.make_lgssm_parameters_cls(dim=5)(),
    },
    "double-well": {
        "base": double_well.DoubleWellParams(
            energy_barrier=jnp.array(2.0),
            transition_std=jnp.array(1.0),
            observation_std=jnp.array(0.2),
        )
    }
}

hyperparameter_settings: dict[BayesianModelLabel, dict[str, HyperParameters]] = {
    "ar-full": {
        "base": NoHyper(),
    },
    "ar-aronly": {
        "base": ar_bayesian.FixedARParams(
            fixed_observation_std=parameter_settings["ar-aronly"]["base"].observation_std,
            fixed_transition_std=parameter_settings["ar-aronly"]["base"].transition_std,
        ),
    },
    "svar-full": {
        "base": stochastic_vol.simple_var.LogVarPriorHyper(
            # ar ~ Uniform(-1, 1)
            ar_mean=jnp.array(0.0),
            ar_std=jnp.sqrt(jnp.array(1.0 / 3.0)),

            # broad prior over long-run annual vol
            # mean 16%, but large uncertainty
            long_term_vol_mean=jnp.array(0.16),
            long_term_vol_std=jnp.array(0.30),

            # broad-ish prior over log-vol innovation scale
            # centred near 0.5, allows values around 0.1–1+ without being absurd
            std_log_var_mean=jnp.array(0.5),
            std_log_var_std=jnp.array(0.5),
        ),
        "high-freq": stochastic_vol.simple_var.LogVarPriorHyper(
            # high persistence expected, but not fixed
            ar_mean=jnp.array(0.85),
            ar_std=jnp.array(0.25),

            # high long-run annual vol, but fairly broad
            long_term_vol_mean=jnp.array(0.60),
            long_term_vol_std=jnp.array(0.30),

            # we expect high-frequency latent vol
            # to probably be smoother per step
            std_log_var_mean=jnp.array(0.15),
            std_log_var_std=jnp.array(0.20),
        ),
    },
    "lgssm-5d-full": {
        "base": linear_gaussian_bayesian.LGSSMHyperParameters(dim=5)
    },
    "double-well": {
        "base": double_well_bayesian.FixedEBParameters(
            fixed_observation_std=parameter_settings["double-well"]["base"].observation_std,
            fixed_transition_std=parameter_settings["double-well"]["base"].transition_std,
        )
    }
}

ConditionGenerator = typing.Callable[[int], typing.Any]

# Optional mapping of model labels to callables that generate condition
# sequences for simulations and likelihood evaluations.
condition_generators: dict[SequentialModelLabel, ConditionGenerator] = {
    "double-well": double_well.make_unit_time_increments,
}

@dataclass(kw_only=True, frozen=True, slots=True)
class RealDataConfig:
    dataset_name: str
    target_model_label: BayesianModelLabel
    sequence_length: int
    num_sequences: int = 1
    hyperparameters: str

    @property
    def target(self):
        return self.posterior.target

    @property
    def posterior(self) -> interface.BayesianSequentialModelProtocol:
        return posterior_factories[self.target_model_label](
            hyperparameter_settings[self.target_model_label][self.hyperparameters]
        )
    
@dataclass(kw_only=True, frozen=True, slots=True)
class SyntheticDataConfig:
    target_model_label: BayesianModelLabel
    generative_parameter_label: str
    sequence_length: int
    seed: int
    num_sequences: int = 1

    @property
    def dataset_name(self):
        dataset_name = self.target_model_label
        dataset_name += f"-{self.generative_parameter_label}"
        dataset_name += f"-d{self.seed}"
        dataset_name += f"-l{self.sequence_length}"
        dataset_name += f"-n{self.num_sequences}"
        return dataset_name

    @property
    def generative_parameters(self):
        return parameter_settings[self.target_model_label][
            self.generative_parameter_label
        ]
    
    @property
    def target(self):
        return self.posterior.target

    @property
    def posterior(self) -> interface.BayesianSequentialModelProtocol:
        return posterior_factories[self.target_model_label](
            hyperparameter_settings[self.target_model_label]["base"]
        )

    
DataConfig = RealDataConfig | SyntheticDataConfig
