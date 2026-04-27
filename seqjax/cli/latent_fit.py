from __future__ import annotations

from typing import Any, Protocol

import equinox as eqx
import jax
import jax.random as jrandom
from seqjax.model.evaluate import log_prob_joint
from seqjax import io
import numpy as np
import arviz as az
from scipy.special import logsumexp

def compute_log_iw(
    trainable,
    key,
    *,
    static,
    sample_kwargs,
    y,
    c,
    context,
    target,
    params,
):
    approximation = eqx.combine(trainable, static)

    x, loq_q_x, _ = jax.vmap(
        approximation.sample_and_log_prob,
        in_axes=(0, None, None),
    )(
        jrandom.split(key, sample_kwargs["mc-samples"]),
        context,
        None
    )

    log_p_x_y = jax.vmap(
        log_prob_joint,
        in_axes=(None, 0, None, None, None)
    )(
        target,
        x,
        y,
        c,
        params,
    )

    return log_p_x_y - loq_q_x


def log_ess_np(log_iw, axis=-1):
    """
    ESS from unnormalised log importance weights.

    Parameters
    ----------
    log_iw:
        Unnormalised log importance weights, log p(x) - log q(x).
    axis:
        Sample axis.

    Returns
    -------
    ess:
        Absolute effective sample size.
    rel_ess:
        Relative effective sample size, ESS / n.
    """
    log_iw = np.asarray(log_iw)

    n = log_iw.shape[axis]
    log_ess = (
        2.0 * logsumexp(log_iw, axis=axis)
        - logsumexp(2.0 * log_iw, axis=axis)
    )

    ess = np.exp(log_ess)
    rel_ess = ess / n

    return ess, rel_ess


def elbo_estimate_np(log_iw, axis=-1):
    """
    Monte Carlo ELBO estimate under q.

    Since log_iw = log p(x, y) - log q(x), or log p(x) - log q(x)
    depending on context, the simple VI/IS ELBO estimate is:

        E_q[log_iw]

    This is not the log marginal likelihood estimate. The IS estimate of
    log normalising constant would be:

        logmeanexp(log_iw)
    """
    log_iw = np.asarray(log_iw)
    return np.mean(log_iw, axis=axis)


def logz_is_estimate_np(log_iw, axis=-1):
    """
    Self-normalised-free importance sampling estimate of log Z:

        log mean_i exp(log_iw_i)

    This estimates log E_q[p/q], not E_q[log p/q].
    It is upward-biased on the log scale for finite N, but consistent.
    """
    log_iw = np.asarray(log_iw)
    n = log_iw.shape[axis]

    return logsumexp(log_iw, axis=axis) - np.log(n)


def importance_diagnostics(log_iw, axis=-1):
    """
    Diagnostics for unnormalised log importance weights.

    Uses ArviZ PSIS rather than a homebrew GPD fit.

    Parameters
    ----------
    log_iw:
        Array of log importance weights.
    axis:
        Sample axis over which to run PSIS / ESS / ELBO.

    Returns
    -------
    dict with:
        k_hat:
            Pareto k estimate from ArviZ.
        ess_raw:
            ESS of raw importance weights.
        rel_ess_raw:
            ESS / n for raw importance weights.
        ess_psis:
            ESS of Pareto-smoothed weights.
        rel_ess_psis:
            ESS / n for Pareto-smoothed weights.
        elbo:
            mean(log_iw).
        logz_is:
            logmeanexp(log_iw).
        log_iw_psis:
            Pareto-smoothed log weights.
    """
    log_iw = np.asarray(log_iw)

    ess_raw, rel_ess_raw = log_ess_np(log_iw, axis=axis)

    log_iw_psis, k_hat = az.psislw(log_iw)

    ess_psis, rel_ess_psis = log_ess_np(log_iw_psis, axis=axis)

    elbo = elbo_estimate_np(log_iw, axis=axis)
    logz_is = logz_is_estimate_np(log_iw, axis=axis)

    return {
        "k_hat": k_hat,
        "ess_raw": ess_raw,
        "rel_ess_raw": rel_ess_raw,
        "ess_psis": ess_psis,
        "rel_ess_psis": rel_ess_psis,
        "elbo": elbo,
        "logz_is": logz_is,
        "elbo_gap_estimate": logz_is - elbo,
    }


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
