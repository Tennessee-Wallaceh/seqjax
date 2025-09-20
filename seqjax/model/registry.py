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

from dataclasses import dataclass, field
from functools import partial
import typing

import jax.numpy as jnp

from . import ar, double_well
from .base import SequentialModel
from .typing import Parameters

SequentialModelLabel = typing.Literal["ar", "double_well"]

# Maps each model label to its ``SequentialModel`` implementation. The keys
# must appear in ``SequentialModelLabel`` and are typically accessed via
# ``sequential_models[label]``. When adding a new model, extend
# ``SequentialModelLabel`` and add the class here with the same label.
sequential_models: dict[SequentialModelLabel, type[SequentialModel]] = {
    "ar": ar.AR1Target,
    "double_well": double_well.DoubleWellTarget,
}

# Predefined parameter presets for each model. The outer keys mirror
# ``sequential_models`` and thus use ``SequentialModelLabel``. Inner keys are
# free-form preset names (e.g. ``"base"``) that experiments reference with
# ``parameter_settings[label][preset]``. Add new presets under the appropriate
# model label and keep labels consistent across both dictionaries.
parameter_settings: dict[SequentialModelLabel, dict[str, Parameters]] = {
    "ar": {
        "base": ar.ARParameters(
            ar=jnp.array(0.8),
            observation_std=jnp.array(0.1),
            transition_std=jnp.array(0.5),
        ),
        "lower_ar": ar.ARParameters(
            ar=jnp.array(0.5),
            observation_std=jnp.array(0.1),
            transition_std=jnp.array(0.5),
        ),
    },
    "double_well": {
        "base": double_well.DoubleWellParams(
            energy_barrier=jnp.array(0.5),
            observation_std=jnp.array(1.0),
            transition_std=jnp.array(0.5),
        ),
    },
}

ConditionGenerator = typing.Callable[[int], typing.Any]

# Optional mapping of model labels to callables that generate condition
# sequences for simulations and likelihood evaluations.
condition_generators: dict[SequentialModelLabel, ConditionGenerator] = {
    "double_well": partial(double_well.make_unit_time_increments, dt=1.0),
}


@dataclass(kw_only=True, frozen=True, slots=True)
class DataConfig:
    target_model_label: SequentialModelLabel
    generative_parameter_label: str
    sequence_length: int
    seed: int

    @property
    def dataset_name(self):
        dataset_name = self.target_model_label
        dataset_name += f"-{self.generative_parameter_label}"
        dataset_name += f"-d{self.seed}"
        dataset_name += f"-l{self.sequence_length}"
        return dataset_name

    @property
    def target(self):
        return sequential_models[self.target_model_label]

    @property
    def generative_parameters(self):
        return parameter_settings[self.target_model_label][
            self.generative_parameter_label
        ]


@dataclass(kw_only=True, frozen=True, slots=True)
class ARDataConfig(DataConfig):
    target_model_label: SequentialModelLabel = field(default="ar", init=False)
