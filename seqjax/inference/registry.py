""" """

import typing
from dataclasses import dataclass
from functools import partial

from seqjax.inference import InferenceMethod, pmcmc, mcmc, particlefilter, sgld, vi

inference_functions = {
    "NUTS": mcmc.run_bayesian_nuts,
    "buffer-vi": vi.run_buffered_vi,
    "full-vi": vi.run_full_path_vi,
}


@dataclass
class NUTSInference:
    method: typing.Literal["NUTS"]
    config: mcmc.NUTSConfig

    @property
    def name(self):
        return f"NUTS-c{self.config.num_chains}-w{self.config.num_warmup}"


@dataclass
class BufferVI:
    method: typing.Literal["buffer-vi"]
    config: vi.BufferedVIConfig

    @property
    def name(self) -> str:
        name = f"buffer-vi-b{self.config.buffer_length}-m{self.config.batch_length}"
        for param, bijector_label in self.config.parameter_field_bijections.items():
            name += f"-{param}_{bijector_label}"

        if self.config.control_variate:
            name += "-cv"

        return name


@dataclass
class FullVI:
    method: typing.Literal["full-vi"]
    config: vi.FullVIConfig

    @property
    def name(self):
        name = f"full-vi"
        for param, bijector_label in self.config.parameter_field_bijections.items():
            name += f"-{param}_{bijector_label}"
        return name


InferenceConfig = NUTSInference


def build_inference(i_config: InferenceConfig, target_model) -> InferenceMethod:
    return partial(inference_functions[i_config.method], config=i_config.config)
