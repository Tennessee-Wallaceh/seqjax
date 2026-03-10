from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import PRNGKeyArray, Scalar

from .types import (
    LVolStd,
    LatentVol,
    LogVarAR,
    LogVarParams,
    LogVarStd,
    LogVolRW,
    LogVolWithSkew,
    TimeIncrement,
)


class RandomWalkParameters(typing.Protocol):
    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_vol: Scalar


def random_walk_loc_scale(
    prev_latent: LatentVol,
    condition: TimeIncrement,
    parameters: RandomWalkParameters,
) -> tuple[Scalar, Scalar]:
    move_scale = jnp.sqrt(condition.dt) * parameters.std_log_vol
    move_loc = prev_latent.log_vol + condition.dt * parameters.mean_reversion * (
        jnp.log(parameters.long_term_vol) - prev_latent.log_vol
    )
    return move_loc, move_scale


def skew_return_mean_and_scale(
    last_latent: LatentVol,
    current_latent: LatentVol,
    condition: TimeIncrement,
    parameters: LogVolWithSkew,
) -> tuple[Scalar, Scalar]:
    dt = condition.dt
    current_vol = jnp.exp(current_latent.log_vol)
    current_var = jnp.exp(2 * current_latent.log_vol)

    log_vol_mean, _ = random_walk_loc_scale(last_latent, condition, parameters)

    return_mean = -0.5 * dt * current_var
    return_mean += (
        parameters.skew
        * (current_vol / parameters.std_log_vol)
        * (current_latent.log_vol - log_vol_mean)
    )

    return_scale = jnp.sqrt(dt) * current_vol * jnp.sqrt(1 - parameters.skew**2)
    return return_mean, return_scale


def lv_to_std_only(lv_only: LVolStd, ref_params: LogVolRW) -> LogVolRW:
    return LogVolRW(
        std_log_vol=lv_only.std_log_vol,
        long_term_vol=jnp.ones_like(lv_only.std_log_vol) * ref_params.long_term_vol,
        mean_reversion=jnp.ones_like(lv_only.std_log_vol) * ref_params.mean_reversion,
    )


def lvar_from_std_only(lv_only: LogVarStd, ref_params: LogVarParams) -> LogVarParams:
    return LogVarParams(
        std_log_var=lv_only.std_log_var,
        ar=jnp.ones_like(lv_only.std_log_var) * ref_params.ar,
        long_term_log_var=jnp.ones_like(lv_only.std_log_var) * ref_params.long_term_log_var,
    )


def lvar_from_ar_only(ar_only: LogVarAR, ref_params: LogVarParams) -> LogVarParams:
    return LogVarParams(
        std_log_var=jnp.ones_like(ar_only.ar) * ref_params.std_log_var,
        ar=ar_only.ar,
        long_term_log_var=jnp.ones_like(ar_only.ar) * ref_params.long_term_log_var,
    )


class StochVolParamPrior:
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: typing.Any) -> LogVolRW:
        _ = hyperparameters
        std_key, mr_key, ltv_key = jrandom.split(key, 3)

        std_mean = jnp.array(3.0)
        std_scale = jnp.array(1.5)
        std_lower = (0.0 - std_mean) / std_scale
        std_log_vol = std_mean + std_scale * jrandom.truncated_normal(
            std_key, lower=std_lower, upper=jnp.inf
        )

        mr_mean = jnp.array(10.0)
        mr_scale = jnp.array(10.0)
        mr_lower = (0.0 - mr_mean) / mr_scale
        mean_reversion = mr_mean + mr_scale * jrandom.truncated_normal(
            mr_key, lower=mr_lower, upper=jnp.inf
        )

        long_term_vol = jnp.exp(
            jnp.array(-2.0) + jnp.array(0.5) * jrandom.normal(ltv_key)
        )

        return LogVolRW(
            std_log_vol=std_log_vol,
            mean_reversion=mean_reversion,
            long_term_vol=long_term_vol,
        )

    @staticmethod
    def log_prob(parameters: LogVolRW, hyperparameters: typing.Any) -> Scalar:
        _ = hyperparameters

        def truncated_normal_logpdf(x: Scalar, mean: float, scale: float) -> Scalar:
            alpha = -mean / scale
            normalization = 1 - jstats.norm.cdf(alpha)
            z = (x - mean) / scale
            log_numerator = jstats.norm.logpdf(z) - jnp.log(scale)
            return jnp.where(x >= 0.0, log_numerator - jnp.log(normalization), -jnp.inf)

        std_log_vol_lpdf = truncated_normal_logpdf(parameters.std_log_vol, 3.0, 0.5)
        mean_reversion_lpdf = truncated_normal_logpdf(parameters.mean_reversion, 10.0, 10.0)

        long_term_lpdf = jstats.norm.logpdf(
            jnp.log(parameters.long_term_vol),
            loc=jnp.array(-2.0),
            scale=jnp.array(0.5),
        )

        return std_log_vol_lpdf + mean_reversion_lpdf + long_term_lpdf


class StdLogVolPrior:
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: typing.Any) -> LVolStd:
        base_p = StochVolParamPrior.sample(key, hyperparameters)
        return LVolStd(std_log_vol=base_p.std_log_vol)

    @staticmethod
    def log_prob(parameters: LVolStd, hyperparameters: typing.Any) -> Scalar:
        _ = hyperparameters
        alpha = -3.0 / 0.5
        normalization = 1 - jstats.norm.cdf(alpha)
        z = (parameters.std_log_vol - 3.0) / 0.5
        log_numerator = jstats.norm.logpdf(z) - jnp.log(0.5)
        return jnp.where(
            parameters.std_log_vol >= 0.0,
            log_numerator - jnp.log(normalization),
            -jnp.inf,
        )


class StochVarPrior:
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: typing.Any) -> LogVarStd:
        _ = hyperparameters
        return LogVarStd(std_log_var=10 / jrandom.gamma(key, 10))

    @staticmethod
    def log_prob(parameters: LogVarStd, hyperparameters: typing.Any) -> Scalar:
        _ = hyperparameters
        return (
            jstats.gamma.logpdf(1 / parameters.std_log_var, 10, scale=1 / 10)
            - 2 * jnp.log(parameters.std_log_var)
        )


class StochVarARPrior:
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: typing.Any) -> LogVarAR:
        _ = hyperparameters
        return LogVarAR(ar=jrandom.uniform(key, minval=-1, maxval=1))

    @staticmethod
    def log_prob(parameters: LogVarAR, hyperparameters: typing.Any) -> Scalar:
        _ = hyperparameters
        return jstats.uniform.logpdf(parameters.ar, loc=-1.0, scale=2.0)


class StochVarFullPrior:
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: typing.Any) -> LogVarParams:
        _ = hyperparameters
        k1, k2, k3 = jrandom.split(key, 3)
        long_term_log_var_mean = 2 * jnp.log(jnp.array(0.16))
        return LogVarParams(
            std_log_var=10 / jrandom.gamma(k1, 10),
            ar=jrandom.uniform(k2, minval=-1, maxval=1),
            long_term_log_var=long_term_log_var_mean + jrandom.normal(k3),
        )

    @staticmethod
    def log_prob(parameters: LogVarParams, hyperparameters: typing.Any) -> Scalar:
        _ = hyperparameters
        long_term_log_var_mean = 2 * jnp.log(jnp.array(0.16))
        return (
            jstats.gamma.logpdf(1 / parameters.std_log_var, 10, scale=1 / 10)
            - 2 * jnp.log(parameters.std_log_var)
            + jstats.uniform.logpdf(parameters.ar, loc=-1.0, scale=2.0)
            + jstats.norm.logpdf(
                parameters.long_term_log_var,
                loc=long_term_log_var_mean,
            )
        )


class SkewStochVolParamPrior:
    @staticmethod
    def sample(key: PRNGKeyArray, hyperparameters: typing.Any) -> LogVolWithSkew:
        base_key, skew_key = jrandom.split(key)
        base = StochVolParamPrior.sample(base_key, hyperparameters)
        skew = jrandom.uniform(
            skew_key,
            shape=(),
            minval=jnp.array(-1.0, dtype=jnp.float32),
            maxval=jnp.array(1.0, dtype=jnp.float32),
        )
        return LogVolWithSkew(
            std_log_vol=base.std_log_vol,
            mean_reversion=base.mean_reversion,
            long_term_vol=base.long_term_vol,
            skew=skew,
        )

    @staticmethod
    def log_prob(parameters: LogVolWithSkew, hyperparameters: typing.Any) -> Scalar:
        base = LogVolRW(
            std_log_vol=parameters.std_log_vol,
            mean_reversion=parameters.mean_reversion,
            long_term_vol=parameters.long_term_vol,
        )
        skew_log_prob = jnp.where(
            (parameters.skew >= -1.0) & (parameters.skew <= 1.0),
            -jnp.log(2.0),
            -jnp.inf,
        )
        return StochVolParamPrior.log_prob(base, hyperparameters) + skew_log_prob
