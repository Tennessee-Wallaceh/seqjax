import typing
import optax  # type: ignore
from dataclasses import dataclass, field

"""
Optimization configurations
"""


@dataclass
class CosineOpt:
    label: str = field(init=False, default="cosine-sched")
    warmup_steps: int = 0
    decay_steps: int = 5000
    peak_lr: float = 1e-2
    end_lr: float = 1e-5
    total_steps: int = 10_000
    time_limit_s: int | None = None

    def __repr__(self) -> str:
        return f"{self.label}({self.peak_lr:.0e},{self.end_lr:.0e},{self.warmup_steps},{self.decay_steps})"

    @classmethod
    def from_dict(cls, config_dict: dict[str, typing.Any]) -> "CosineOpt":
        return cls(
            warmup_steps=config_dict["warmup_steps"],
            decay_steps=config_dict["decay_steps"],
            peak_lr=config_dict["peak_lr"],
            end_lr=config_dict["end_lr"],
            total_steps=config_dict["total_steps"],
            time_limit_s=config_dict["time_limit_s"],
        )


@dataclass
class AdamOpt:
    label: str = field(init=False, default="adam-plain")
    lr: float = 1e-3
    total_steps: int = 500_000
    time_limit_s: int | None = 60 * 60 * 2  # default to 2 hour limit

    def __repr__(self) -> str:
        return f"{self.label}({self.lr:.0e})"

    @classmethod
    def from_dict(cls, config_dict: dict[str, typing.Any]) -> "AdamOpt":
        return cls(
            lr=config_dict["lr"],
            total_steps=config_dict["total_steps"],
            time_limit_s=config_dict["time_limit_s"],
        )


OptConfigLabels = typing.Literal["cosine-sched", "adam-plain"]
OptConfig = CosineOpt | AdamOpt

registry: dict[OptConfigLabels, OptConfig] = {
    "cosine-sched": CosineOpt(),
    "adam-plain": AdamOpt(),
}


def build_optimizer(optimization_config) -> optax.GradientTransformation:
    if isinstance(optimization_config, AdamOpt):
        optim = optax.apply_if_finite(
            optax.adam(optimization_config.lr), max_consecutive_errors=100
        )
    elif isinstance(optimization_config, CosineOpt):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=optimization_config.peak_lr,
            peak_value=optimization_config.peak_lr,
            warmup_steps=optimization_config.warmup_steps,
            decay_steps=optimization_config.decay_steps,
            end_value=optimization_config.end_lr,
        )

        optim = optax.apply_if_finite(
            optax.adam(learning_rate=schedule), max_consecutive_errors=100
        )
    else:
        raise Exception(f"Unknown Optimizer: {optimization_config.optimization}")

    return optim
