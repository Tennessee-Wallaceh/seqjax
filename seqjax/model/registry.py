"""
Central registry of available sequential models and predefined parameter presets.

The module maintains two dictionaries with matching keys:

* ``sequential_models`` maps a short string label to the ``SequentialModel``
  subclass that implements the model.
* ``parameter_settings`` maps the same label to a mapping of preset names to
  parameter dataclass instances.

Experiments first retrieve a model class via ``sequential_models[label]`` and
then choose a preset through ``parameter_settings[label][preset]``.  Using the
same label across both dictionaries guarantees that models and presets remain
compatible.

Registering a new model type and parameter presets
-------------------------------------------------
1. Implement a ``SequentialModel`` subclass and its associated parameter
   dataclass.
2. Pick a unique string label for the model.
3. Add the class to ``sequential_models`` under that label.
4. Create one or more parameter presets and add them to
   ``parameter_settings`` under the same label.
5. (Optional) expose a ``DataConfig`` helper if the model needs specialised
   configuration handling.
"""

import jax.numpy as jnp

import typing
from . import ar
from dataclasses import dataclass, field
from .base import SequentialModel
from .typing import Parameters

SequentialModelLabel = typing.Literal["ar"]
sequential_models: dict[SequentialModelLabel, type[SequentialModel]] = {
    "ar": ar.AR1Target
}
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
    }
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
