import jax
import jax.random as jrandom
import blackjax

def inference_loop_multiple_chains(
    rng_key, kernel, initial_state, num_samples, num_chains
):

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, infos = jax.vmap(kernel)(keys, states)
        return states, (states, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos


def run_mcmc(logdensity_fn, seed=0):
    rng_key = jrandom.PRNGKey(seed)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

    warmup = blackjax.window_adaptation(
        blackjax.nuts, 
        logdensity_fn,
        initial_step_size=1e-7,
        target_acceptance_rate=0.9,
        is_mass_matrix_diagonal=False
    ) 

    num_warmup = 5000
    (initial_state, kernel_params), warmup_trace = warmup.run(
        warmup_key,
        initial_position,
        num_steps=num_warmup,
    )
