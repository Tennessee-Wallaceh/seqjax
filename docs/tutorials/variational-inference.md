# Variational inference with amortised buffers

Variational inference (VI) approximates the posterior with a parameterised
family of distributions that can be optimised efficiently. SeqJAX offers two
approaches via the [`seqjax.inference.vi`](../api.md#variational-inference)
package: a buffered amortised approximation for long sequences and a full-path
model that conditions on every observation simultaneously. This tutorial
focuses on the buffered variant because it scales gracefully to streaming
scenarios.

## Prepare data and a posterior model

We reuse the AR(1) example from the previous tutorials. The simulated data and
[`AR1Bayesian`](../api.md#seqjaxmodelar) posterior provide the inputs required
by the VI routines.

```python
import jax.numpy as jnp
import jax.random as jrandom
from seqjax import simulate
from seqjax.model.ar import AR1Target, ARParameters, AR1Bayesian

key = jrandom.PRNGKey(0)
true_parameters = ARParameters(
    ar=jnp.array(0.7),
    observation_std=jnp.array(0.6),
    transition_std=jnp.array(0.3),
)
model = AR1Target()
posterior = AR1Bayesian(true_parameters)

_, observations, _, _ = simulate.simulate(
    key,
    model,
    condition=None,
    parameters=true_parameters,
    sequence_length=400,
)
```

A longer sequence is useful here because the amortised approximation learns a
neural network that conditions on sliding windows of the observation stream.

## Configure the buffered VI approximation

[`BufferedVIConfig`](../api.md#variational-inference) tunes the optimiser,
context window sizes, and bijective transforms applied to constrained
parameters. The example below keeps most defaults, but specifies a sigmoid
bijection so that the AR coefficient remains in $(-1, 1)$ and reduces the
network width to speed up the demonstration run.

```python
from seqjax.inference.vi import BufferedVIConfig, run

vi_config = BufferedVIConfig(
    optimization=run.AdamOpt(lr=5e-3, total_steps=500),
    buffer_length=20,
    batch_length=30,
    parameter_field_bijections={"ar": "sigmoid"},
    observations_per_step=5,
    samples_per_context=5,
    latent_approximation=run.AutoregressiveLatentApproximation(
        nn_width=16,
        nn_depth=2,
    ),
)
```

The `buffer_length` and `batch_length` settings control how many observations
are available when sampling mini-batches from the stream, while
`observations_per_step` and `samples_per_context` dictate the stochastic
estimator used for the evidence lower bound (ELBO).

## Optimise with `run_buffered_vi`

[`run_buffered_vi`](../api.md#variational-inference) fits the approximation and
returns posterior samples drawn from the learned variational family. The
function expects `test_samples` to be at least 100 because the evaluation stage
uses 100 contexts with `test_samples / 100` samples each.

```python
from seqjax.inference.vi import run_buffered_vi

parameter_samples, (buffer_start, latent_posterior, tracker) = run_buffered_vi(
    target_posterior=posterior,
    hyperparameters=None,
    key=jrandom.PRNGKey(1),
    observation_path=observations,
    condition_path=None,
    test_samples=500,
    config=vi_config,
)
```

The returned `parameter_samples` PyTree matches the structure of
`posterior.inference_parameter_cls`, but each field is flattened into a vector
of posterior draws. For AR(1) this means `parameter_samples.ar` has shape
`(test_samples,)` and can be analysed just like MCMC output.

The tuple accompanying the samples contains useful diagnostics:

- `buffer_start` is an array recording the starting index of each buffered
  context that contributed to the amortised latent approximation.
- `latent_posterior` contains samples of the latent states produced by the
  amortised autoregressive network. The array has shape `(num_contexts,
  samples_per_context, latent_dim)`.
- `tracker` is a [`DefaultTracker`](../api.md#variational-inference) instance.
  Inspect `tracker.update_rows` to view recorded ELBO estimates and posterior
  quantiles during training.

```python
ar_variational_draws = parameter_samples.ar
last_update = tracker.update_rows[-1]
print("Final ELBO estimate:", last_update["loss"])
```

## Related resources

- The [particle filtering](particle-filtering.md) and
  [Bayesian MCMC](bayesian-mcmc.md) tutorials explore alternative inference
  strategies using the same sequential model.
- To operate on the entire observation path instead of buffered windows, see
  [`run_full_path_vi`](../api.md#variational-inference).
