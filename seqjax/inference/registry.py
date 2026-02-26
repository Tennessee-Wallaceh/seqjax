import typing
from dataclasses import dataclass

from seqjax.inference import (
    InferenceMethod,
    mcmc,
    pmcmc,
    sgld,
    vi,
)

InferenceName = typing.Literal[
    "NUTS",
    "buffer-vi",
    "full-vi",
    "particle-mcmc",
    "full-sgld",
    "buffer-sgld",
]

def default_name[MethodConfig](label: InferenceName, config: MethodConfig) -> str:
    return label

@dataclass(frozen=True)
class InferenceSpec[MethodConfig]:
    label: InferenceName
    config_cls: type[MethodConfig]
    run: InferenceMethod
    name_fn: typing.Callable[[InferenceName, MethodConfig], str] = default_name

    def build_config(self, config_dict):
        config = self.config_cls(**config_dict)
        return InferenceConfig(
            label=self.label,
            config=config,
            run=self.run,
            name=self.name_fn(self.label, config),
        )
    
@dataclass(frozen=True)
class InferenceConfig[MethodConfig]:
    label: InferenceName
    config: MethodConfig
    run: InferenceMethod
    name: str 

inference_registry: dict[InferenceName, InferenceSpec] = {}

def register_inference[MethodConfig](
    method: InferenceName,
    *,
    config_cls: type[MethodConfig],
    run: InferenceMethod,
    name_fn: typing.Callable[[InferenceName, MethodConfig], str] = default_name,
) -> None:
    if method in inference_registry:
        raise ValueError(f"Duplicate inference method registration: {method!r}")
    inference_registry[method] = InferenceSpec(
        label=method,
        config_cls=config_cls,
        run=run,
        name_fn=name_fn,
    )

register_inference(
    "NUTS",
    config_cls=mcmc.NUTSConfig,
    run=mcmc.run_bayesian_nuts,
    name_fn=lambda _, config: (
        f"NUTS-c{config.num_chains}-w{config.num_warmup}"
    )
)

register_inference(
    "buffer-vi",
    config_cls=vi.registry.BufferedVIConfig,
    run=vi.run.run_buffered_vi,
    name_fn=lambda label, config: (
        f"{label}-B{config.buffer_length}-M{config.batch_length}"
        f"-MC{config.samples_per_context}-BS{config.num_context_per_sequence}"
        # + "".join(
        #     f"-{param}_{bijector_label}"
        #     for param, bijector_label in config.parameter_field_bijections.items()
        # )
        # + ("-cv" if config.control_variate else "")
        + f"-{str(config.optimization)}"
        + f"-E_{config.embedder.label}"
    ),
)

register_inference(
    "full-vi",
    config_cls=vi.registry.FullVIConfig,
    run=vi.run.run_full_path_vi,
    # name_fn=lambda config: (
    #     "full-vi"
    #     + "".join(
    #         f"-{param}_{bijector_label}"
    #         for param, bijector_label in config.parameter_field_bijections.items()
    #     )
    #     + f"-{str(config.optimization)}"
    #     + f"-E_{config.embedder.label}"
    # ),
)

register_inference(
    "particle-mcmc",
    config_cls=pmcmc.ParticleMCMCConfig,
    run=pmcmc.run_particle_mcmc,
    # name_fn=lambda config: (
    #     "particle-mcmc"
    #     + (
    #         f"-p{config.particle_filter_config.num_particles}"
    #         if config.particle_filter_config.num_particles is not None
    #         else ""
    #     )
    #     + (
    #         f"-ss{config.mcmc.step_size:.3g}"
    #         if config.mcmc.step_size is not None
    #         else ""
    #     )
    # ),
)

register_inference(
    "full-sgld",
    config_cls=sgld.SGLDConfig,
    run=sgld.run_full_sgld_mcmc,
    # name_fn=lambda config: f"full-sgld-p{config.particle_filter_config.num_particles}",
)

register_inference(
    "buffer-sgld",
    config_cls=sgld.BufferedSGLDConfig,
    run=sgld.run_buffer_sgld_mcmc,
    # name_fn=lambda config: (
    #     f"buffer-sgld-p{config.particle_filter_config.num_particles}"
    #     f"-b{config.buffer_length}-m{config.batch_length}-ss{config.step_size:.3g}"
    # ),
)
