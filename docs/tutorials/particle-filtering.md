# Particle filtering with SeqJAX

Sequential Monte Carlo (SMC) methods approximate posterior distributions over
latent trajectories by propagating and resampling a population of particles.
SeqJAX exposes a particle filtering API in
[`seqjax.inference.particlefilter`](../api.md#particle-filtering) that works
with any `SequentialModel`. This tutorial walks through fitting the autoregressive
AR(1) example from [`seqjax.model.ar`](../api.md#seqjaxmodelar) to show the
workflow end to end.

## Simulate a reference data set

We start by simulating a short path from the AR(1) model. The
[`simulate.simulate`](../api.md#seqjaxmodelsimulate) helper generates both the
latent trajectory and the noisy observations that we will feed to the filter.

```python
import jax.numpy as jnp
import jax.random as jrandom
from seqjax import simulate
from seqjax.model.ar import AR1Target, ARParameters

key = jrandom.key(0)
true_parameters = ARParameters(
    ar=jnp.array(0.8),
    observation_std=jnp.array(1.0),
    transition_std=jnp.array(0.5),
)
model = AR1Target()

_, observations = simulate.simulate(
    key,
    model,
    condition=None,
    parameters=true_parameters,
    sequence_length=100,
)
```

The `observations` PyTree is compatible with every inference routine, so the
same object can be reused in the later MCMC and VI tutorials.

## Configure the particle filter

The [`BootstrapParticleFilter`](../api.md#particle-filtering) class wraps the
standard algorithm that proposes particles from the transition model. The
filter needs a target model, the number of particles, and an optional effective
sample size (ESS) threshold that triggers resampling.

```python
from seqjax.inference import particlefilter

smc = particlefilter.BootstrapParticleFilter(
    target=model,
    num_particles=256,
    ess_threshold=0.6,
)
```

SeqJAX also provides an [`AuxiliaryParticleFilter`](../api.md#particle-filtering)
that adapts proposal weights using the emission model. Both variants share the
same interface, so swapping them only requires changing the class name.

## Run the filter and collect diagnostics

[`particlefilter.run_filter`](../api.md#particle-filtering) performs the SMC
pass. You can request arbitrary diagnostics by passing recorders from
[`particlefilter.recorders`](../api.md#particle-filtering) via the `recorders`
argument. The example below tracks the weighted particle mean and the ESS at
each time step.

```python
log_weights, particle_states, recorder_history = particlefilter.run_filter(
    smc,
    key=jrandom.key(1),
    parameters=true_parameters,
    observation_path=observations,
    recorders=(
        particlefilter.current_particle_mean,
        particlefilter.effective_sample_size,
    ),
)

mean_path, ess_path = recorder_history
state_mean = mean_path.x        # shape (sequence_length,)
normalised_ess = ess_path        # shape (sequence_length,)
```

The `particle_states` output contains the final collection of particles, while
`mean_path` and `normalised_ess` store the recorder values across time. Each
recorder returns a PyTree that matches the latent particle structure, so the
above code simply extracts the scalar state from the `LatentValue` dataclass.

## Next steps

Particle filtering pairs well with other inference routines:

- Follow the [Bayesian MCMC tutorial](bayesian-mcmc.md) to infer parameters and
  latent states jointly with Hamiltonian Monte Carlo.
- Continue to the [variational inference tutorial](variational-inference.md) to
  amortise posterior inference with neural networks.

Refer to the [`seqjax.inference.particlefilter` API
reference](../api.md#particle-filtering) for the full list of recorders,
resampling schemes, and low-level utilities.
