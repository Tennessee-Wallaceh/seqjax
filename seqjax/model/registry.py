"""
Submodule mapping static configuration to code objects, corresponding to particular models +
parameter settings.
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
