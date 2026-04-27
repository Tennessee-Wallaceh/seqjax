from __future__ import annotations

from typing import Any, Protocol

import equinox as eqx

from seqjax import io


class LatentFitArtifactSink(Protocol):
    """Protocol for persisting latent-fit outputs."""

    def save(
        self,
        *,
        run_name: str,
        fitted_approximation: eqx.Module,
        optimization_state: Any,
        tracker_rows: list[dict[str, Any]],
    ) -> None: ...


class WandbLatentFitArtifactSink:
    """Persist latent-fit outputs as W&B artifacts."""

    def __init__(self, run: io.WandbRun) -> None:
        self.run = run

    def save(
        self,
        *,
        run_name: str,
        fitted_approximation: eqx.Module,
        optimization_state: Any,
        tracker_rows: list[dict[str, Any]],
    ) -> None:
        io.save_model_artifact(
            self.run,
            f"{run_name}-latent-approximation",
            fitted_approximation,
        )
        io.save_python_artifact(
            self.run,
            f"{run_name}-latent-fit-metadata",
            "run_output",
            [
                ("latent_fit_updates", tracker_rows),
                ("latent_opt_state", optimization_state),
            ],
        )
