import jax
import equinox as eqx
import jax.numpy as jnp

class EMAParamNorm(eqx.Module):
    index: eqx.nn.StateIndex
    momentum: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, feature_dim: int, momentum: float = 0.99, eps: float = 1e-5):
        init_state = {
            "mean": jnp.zeros((feature_dim,)),
            "var": jnp.ones((feature_dim,)),
            "initialized": jnp.array(False),
        }
        self.index = eqx.nn.StateIndex(init_state)
        self.momentum = momentum
        self.eps = eps

    def __call__(self, x, state: eqx.nn.State, *, update_stats: bool):
        stats = state.get(self.index)
        mean = stats["mean"]
        var = stats["var"]
        initialized = stats["initialized"]

        x_norm = jax.lax.cond(
            initialized,
            lambda _: (x - mean) / jnp.sqrt(var + self.eps),
            lambda _: x,
            operand=None,
        )

        x2 = jnp.reshape(x, (-1, x.shape[-1]))
        batch_mean = jnp.mean(x2, axis=0)
        batch_var = jnp.var(x2, axis=0)
        new_stats = jax.lax.cond(
            initialized,
            lambda _: {
                "mean": self.momentum * mean + (1.0 - self.momentum) * batch_mean,
                "var": self.momentum * var + (1.0 - self.momentum) * batch_var,
                "initialized": jnp.array(True),
            },
            lambda _: {
                "mean": batch_mean,
                "var": batch_var,
                "initialized": jnp.array(True),
            },
            operand=None,
        )
    
        if update_stats:
            state = state.set(self.index, new_stats)
            
        return x_norm, state