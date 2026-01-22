""" """

import typing
from dataclasses import dataclass

from seqjax.inference import InferenceMethod, mcmc, pmcmc, vi, sgld

inference_functions: dict[str, InferenceMethod] = {
    "NUTS": mcmc.run_bayesian_nuts,
    "buffer-vi": vi.run_buffered_vi,
    "full-vi": vi.run_full_path_vi,
    "particle-mcmc": pmcmc.run_particle_mcmc,
    "full-sgld": sgld.run_full_sgld_mcmc,
    "buffer-sgld": sgld.run_buffer_sgld_mcmc,
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
    config: vi.registry.BufferedVIConfig

    @property
    def name(self) -> str:
        name = f"buffer-vi-b{self.config.buffer_length}-m{self.config.batch_length}"
        for param, bijector_label in self.config.parameter_field_bijections.items():
            name += f"-{param}_{bijector_label}"

        if self.config.control_variate:
            name += "-cv"

        name += f"-{str(self.config.optimization)}"

        name += f"-E_{self.config.embedder.label}"

        return name


@dataclass
class FullVI:
    method: typing.Literal["full-vi"]
    config: vi.registry.FullVIConfig

    @property
    def name(self):
        name = "full-vi"
        for param, bijector_label in self.config.parameter_field_bijections.items():
            name += f"-{param}_{bijector_label}"

        name += f"-{str(self.config.optimization)}"

        name += f"-E_{self.config.embedder.label}"

        return name


@dataclass
class ParticleMCMCInference:
    method: typing.Literal["particle-mcmc"]
    config: pmcmc.ParticleMCMCConfig

    @property
    def name(self) -> str:
        name = "particle-mcmc"

        num_particles = self.config.particle_filter_config.num_particles
        if num_particles is not None:
            name += f"-p{num_particles}"

        step_size = self.config.mcmc.step_size
        if step_size is not None:
            name += f"-ss{step_size:.3g}"

        return name


@dataclass
class FullSGLDInference:
    method: typing.Literal["full-sgld"]
    config: sgld.SGLDConfig

    @property
    def name(self) -> str:
        num_particles = self.config.particle_filter_config.num_particles
        name = f"full-sgld-n{self.config.num_samples}-p{num_particles}"
        return name


@dataclass
class BufferSGLDInference:
    method: typing.Literal["buffer-sgld"]
    config: sgld.BufferedSGLDConfig

    @property
    def name(self) -> str:
        num_particles = self.config.particle_filter_config.num_particles
        name = f"full-sgld-n{self.config.num_samples}-p{num_particles}"
        name += f"-b{self.config.buffer_length}-m{self.config.batch_length}-ss{self.config.step_size}"
        return name


InferenceConfig = (
    NUTSInference 
    | BufferVI 
    | FullVI 
    | ParticleMCMCInference 
    | FullSGLDInference
    | BufferSGLDInference
)


def from_dict(config_dict: dict[str, typing.Any]) -> InferenceConfig:
    method = config_dict["method"]
    if method == "NUTS":
        return NUTSInference(
            method="NUTS",
            config=mcmc.NUTSConfig.from_dict(config_dict["config"]),
        )
    elif method == "buffer-vi":
        return BufferVI(
            method="buffer-vi",
            config=vi.BufferedVIConfig.from_dict(config_dict["config"]),
        )
    elif method == "full-vi":
        return FullVI(
            method="full-vi",
            config=vi.FullVIConfig.from_dict(config_dict["config"]),
        )
    elif method == "particle-mcmc":
        return ParticleMCMCInference(
            method="particle-mcmc",
            config=pmcmc.ParticleMCMCConfig.from_dict(config_dict["config"]),
        )
    elif method == "full-sgld":
        return FullSGLDInference(
            method="full-sgld",
            config=sgld.SGLDConfig.from_dict(config_dict["config"]),
        )
    elif method == "buffer-sgld":
        return FullSGLDInference(
            method="full-sgld",
            config=sgld.BufferedSGLDConfig.from_dict(config_dict["config"]),
        )
    else:
        raise ValueError(f"Unknown inference method: {method}")


def build_inference(i_config: InferenceConfig) -> InferenceMethod:
    return inference_functions[i_config.method]
