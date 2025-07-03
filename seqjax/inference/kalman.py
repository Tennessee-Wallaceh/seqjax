import jax
import jax.numpy as jnp
from jaxtyping import Array

from seqjax.model.linear_gaussian import LGSSMParameters, VectorObservation


def _gaussian_logpdf(x: Array, mean: Array, cov: Array) -> Array:
    dim = x.shape[0]
    diff = x - mean
    solve = jnp.linalg.solve(cov, diff)
    log_det = jnp.linalg.slogdet(cov)[1]
    return -0.5 * (dim * jnp.log(2.0 * jnp.pi) + log_det + diff @ solve)


def run_kalman_filter(
    parameters: LGSSMParameters,
    observations: VectorObservation,
    *,
    initial_mean: Array | None = None,
    initial_cov: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Run a Kalman filtering pass for a linear Gaussian state space model."""

    A = parameters.transition_matrix
    C = parameters.emission_matrix
    Q = jnp.diag(parameters.transition_noise_scale ** 2)
    R = jnp.diag(parameters.emission_noise_scale ** 2)

    state_dim = A.shape[0]
    if initial_mean is None:
        initial_mean = jnp.zeros(state_dim)
    if initial_cov is None:
        initial_cov = Q

    y = observations.y

    def step(carry, obs):
        mean_prev, cov_prev, log_like_prev = carry
        mean_pred = A @ mean_prev
        cov_pred = A @ cov_prev @ A.T + Q

        innov = obs - C @ mean_pred
        S = C @ cov_pred @ C.T + R
        K = jnp.linalg.solve(S, cov_pred @ C.T).T

        mean = mean_pred + K @ innov
        cov = cov_pred - K @ C @ cov_pred

        log_like = log_like_prev + _gaussian_logpdf(obs, C @ mean_pred, S)

        return (mean, cov, log_like), (mean, cov, log_like)

    _, hist = jax.lax.scan(
        step, (initial_mean, initial_cov, jnp.array(0.0)), y
    )

    means, covs, log_marginal = hist
    return means, covs, log_marginal
