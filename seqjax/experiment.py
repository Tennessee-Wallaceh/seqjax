from dataclasses import dataclass
from typing import Any, Protocol


from seqjax.inference import registry as inference_registry
from seqjax.model import registry as model_registry


class ResultProcessor(Protocol):
    """Protocol for experiment-specific result processing."""

    def process(
        self,
        run: Any,
        config: "ExperimentConfig",
        param_samples: Any,
        extra_data: Any,
        x_path: Any,
        y_path: Any,
        condition: Any,
    ) -> None: ...


@dataclass
class ExperimentConfig:
    data_config: model_registry.DataConfig
    test_samples: int
    fit_seed: int
    inference: inference_registry.InferenceConfig

    @property
    def posterior_factory(self) -> model_registry.PosteriorFactory:
        return self.data_config.posterior_factory

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ExperimentConfig":
        data_config = model_registry.DataConfig.from_dict(config_dict["data_config"])
        inference = inference_registry.from_dict(config_dict["inference"])
        return cls(
            data_config=data_config,
            test_samples=config_dict["test_samples"],
            fit_seed=config_dict["fit_seed"],
            inference=inference,
        )
