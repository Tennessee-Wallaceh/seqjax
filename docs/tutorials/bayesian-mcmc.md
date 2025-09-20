# Bayesian inference with NUTS

SeqJAX integrates with [BlackJAX](https://blackjax-devs.github.io/blackjax/)
to provide Hamiltonian Monte Carlo kernels for sequential models. The
[`seqjax.inference.mcmc`](../api.md#mcmc-kernels) module exposes a convenient
interface that handles warm-up, multiple chains, and posterior bookkeeping. In
this tutorial we infer the AR(1) autoregressive coefficient using the
No-U-Turn Sampler (NUTS).

## Define the Bayesian model

The [`AR1Bayesian`](../api.md#seqjaxmodelar) helper wraps the AR(1) state space
model together with a prior over the autoregressive coefficient. We first
simulate a synthetic data set that plays the role of observed measurements.

```python
import jax.numpy as jnp
import jax.random as jrandom
from seqjax import simulate
from seqjax.model.ar import AR1Target, ARParameters, AR1Bayesian

key = jrandom.key(0)
true_parameters = ARParameters(
    ar=jnp.array(0.8),
    observation_std=jnp.array(0.5),
    transition_std=jnp.array(0.3),
)
model = AR1Target()
posterior = AR1Bayesian(true_parameters)

_, observations, _, _ = simulate.simulate(
    key,
    model,
    condition=None,
    parameters=true_parameters,
    sequence_length=200,
)
```

The resulting `posterior` object satisfies the
[`BayesianSequentialModel`](../api.md#seqjaxmodelbase) interface required by
all inference routines in `seqjax.inference`.

## Configure the NUTS sampler

[`NUTSConfig`](../api.md#mcmc-kernels) controls the step size, warm-up length,
and number of chains. The configuration below keeps the defaults but increases
the number of warm-up steps to obtain a well-tuned mass matrix.

```python
from seqjax.inference.mcmc import NUTSConfig

nuts_config = NUTSConfig(
    step_size=1e-2,
    num_adaptation=500,
    num_warmup=1000,
    num_chains=2,
)
```

The sampler will draw `test_samples` posterior samples in total, split evenly
across the requested number of chains.

## Run `run_bayesian_nuts`

The [`run_bayesian_nuts`](../api.md#mcmc-kernels) function executes warm-up and
sampling in one call. It returns the posterior samples over parameters together
with auxiliary diagnostics.

```python
from seqjax.inference.mcmc import run_bayesian_nuts

parameter_samples, (draw_times_s, latent_samples) = run_bayesian_nuts(
    target_posterior=posterior,
    hyperparameters=None,
    key=jrandom.key(1),
    observation_path=observations,
    condition_path=None,
    test_samples=2000,
    config=nuts_config,
)
```

The `parameter_samples` PyTree matches the structure of
`posterior.inference_parameter_cls`. For the AR(1) example it is an
`AROnlyParameters` object where the `ar` field stores a one-dimensional array of
size `test_samples`.

```python
ar_draws = parameter_samples.ar  # shape (test_samples,)
mean_estimate = jnp.mean(ar_draws)
credible_region = jnp.quantile(ar_draws, jnp.array([0.05, 0.95]))
```

The second output contains two diagnostics:

- `draw_times_s` approximates the cumulative runtime of each posterior sample.
  It is handy for plotting per-iteration wall-clock times.
- `latent_samples` stores the sampled trajectories for each chain. For the AR(1)
  model it has shape `(samples_per_chain, num_chains, sequence_length)`.

## Where to go next

- The [particle filtering tutorial](particle-filtering.md) shows how to compute
  filtering distributions using the same sequential model.
- The [variational inference tutorial](variational-inference.md) covers the
  amortised inference API in [`seqjax.inference.vi`](../api.md#variational-inference).

For additional kernels, such as random-walk Metropolis or particle MCMC, see
the [`seqjax.inference.mcmc` API documentation](../api.md#mcmc-kernels) and the
[`seqjax.inference.pmcmc` module reference](../api.md#particle-mcmc).
