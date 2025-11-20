# Getting Started

This short tutorial demonstrates a simple autoregressive model using `seqjax`.

```python
import jax.random as jrandom
from seqjax.model.ar import AR1Target, ARParameters
from seqjax import simulate

parameters = ARParameters(
    ar=0.8,
    observation_std=1.0,
    transition_std=0.5,
)
model = AR1Target()

key = jrandom.key(0)

latent_path, observation_path = simulate.simulate(
    key, model, condition=None, parameters=parameters, sequence_length=5,
)
print(observation_path)
```

## Next steps

Ready to move beyond simulation? Dive into the advanced tutorials:

- [Particle filtering](particle-filtering.md) demonstrates how to track latent
  state distributions with sequential Monte Carlo.
- [Bayesian MCMC](bayesian-mcmc.md) walks through running Hamiltonian Monte
  Carlo with [`seqjax.inference.mcmc`](../api.md#mcmc-kernels).
- [Variational inference](variational-inference.md) explains the amortised
  inference utilities in [`seqjax.inference.vi`](../api.md#variational-inference).
