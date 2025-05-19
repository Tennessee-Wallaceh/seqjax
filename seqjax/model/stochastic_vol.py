from typing import Union, ClassVar
from jaxtyping import Scalar, PRNGKeyArray, Float, Array
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.random as jrandom

from seqjax.model.base import (
    Target,
    Particle,
    Observation,
    Condition,
    Parameters,
    Transition,
    Prior,
    Emission,
    HyperParameters,
    ParameterPrior,
)

"""
All values are in annualised terms.
"""


# Latent Particles
class LatentVol(Particle):
    log_vol: Scalar

class Underlying(Observation):
    underlying: Scalar

# parameters
class LogVolRW(Parameters):
    std_log_vol: Scalar

    # initial values
    reference_emission: tuple[Underlying] = (Underlying(jnp.array(3000.)),)

class LogVolWithSkew(Parameters):
    std_log_vol: Scalar
    mean_reversion: Scalar
    long_term_log_vol: Scalar
    return_drift: Scalar
    skew: Scalar  # correlation between random variations

    # initial values
    reference_emission: tuple[Underlying] = (Underlying(jnp.array(3000.)),)
    

LogVolRandomWalks = Union[LogVolRW,  LogVolWithSkew]

class TimeIncrement(Condition):
    dt: Scalar  # time since last observation


class HalfCauchyStds(ParameterPrior[LogVolRW, HyperParameters]):
    @staticmethod
    def sample(key, hyperparameters):
        pass
    
    @staticmethod
    def log_p(parameteters, hyperparameters):
        log_2 = jnp.log(jnp.array(2.0))
        log_p_theta = jstats.norm.logpdf(parameteters.std_log_vol) + log_2
        return log_p_theta
    

class GaussianStart(Prior[LatentVol, TimeIncrement, LogVolRandomWalks]):
    order: ClassVar[int] = 2

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        conditions: tuple[TimeIncrement, TimeIncrement],
        parameters: LogVolRandomWalks
    ) -> tuple[LatentVol, LatentVol]:
        mu = jnp.array(-2.)
        sigma = jnp.array(0.5)

        start_key, trans_key = jrandom.split(key)
        start_lv = LatentVol(log_vol=mu + sigma * jrandom.normal(start_key))
        next_lv  = RandomWalk.sample(
            trans_key,
            (start_lv,),
            conditions[1],
            parameters,
        )
        return start_lv, next_lv

    @staticmethod
    def log_p(
        particle: tuple[LatentVol, LatentVol],
        conditions: tuple[TimeIncrement, TimeIncrement],
        parameters: LogVolRandomWalks
    ) -> Scalar:
        start_lv, next_lv = particle
        mu = jnp.array(-2.)
        sigma = jnp.array(0.5)

        base_log_p = jstats.norm.logpdf(
            start_lv.log_vol,
            loc=mu,
            scale=sigma,
        )
        rw_log_p = RandomWalk.log_p(
            (start_lv,),
            next_lv,
            conditions[1],
            parameters,
        )
        return base_log_p + rw_log_p
    
class NonCenteredRandomWalk(Transition[LatentVol, TimeIncrement, LogVolRandomWalks]):
    order: ClassVar[int] = 1

    @staticmethod
    def loc_scale(
        particle_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRandomWalks
    ):
        move_scale = jnp.sqrt(condition.dt) * parameters.std_log_vol
        prev_particle, = particle_history # unpack
        move_loc = prev_particle.log_vol
        return move_loc, move_scale
    
    @staticmethod
    def sample_innovation(key: PRNGKeyArray):
        return jrandom.normal(key)
    
    @staticmethod
    def apply_innovation(eps: LatentVol, loc: Scalar, scale: Scalar):
        next_log_vol = loc + scale * eps.log_vol
        return LatentVol(log_vol=next_log_vol)
    
    @staticmethod
    def log_p_innovation(eps, loc, scale):
        return jstats.norm.logpdf(eps.log_vol) - jnp.log(scale)

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ) -> LatentVol:
        loc, scale = RandomWalk.loc_scale(particle_history, condition, parameters)
        return NonCenteredRandomWalk.apply_innovation(
            LatentVol(NonCenteredRandomWalk.sample_innovation(key)),
            loc,
            scale,
        )

    @staticmethod
    def log_p(
        particle_history: tuple[LatentVol],
        particle: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ) -> Scalar:
        loc, scale = RandomWalk.loc_scale(particle_history, condition, parameters)
        eps_particle = LatentVol((particle.log_vol - loc) / scale)
        return NonCenteredRandomWalk.log_p_innovation(eps_particle, loc, scale)
    
class RandomWalk(Transition[LatentVol, TimeIncrement, LogVolRandomWalks]):
    order: ClassVar[int] = 1

    @staticmethod
    def loc_scale(
        particle_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRandomWalks
    ):
        move_scale = jnp.sqrt(condition.dt) * parameters.std_log_vol
        prev_particle, = particle_history # unpack
        move_loc = prev_particle.log_vol
        return move_loc, move_scale

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle_history: tuple[LatentVol],
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ) -> LatentVol:
        loc, scale = RandomWalk.loc_scale(particle_history, condition, parameters)
        return LatentVol(loc + scale * jrandom.normal(key))

    @staticmethod
    def log_p(
        particle_history: tuple[LatentVol],
        particle: LatentVol,
        condition: TimeIncrement,
        parameters: LogVolRandomWalks,
    ) -> Scalar:
        loc, scale = RandomWalk.loc_scale(particle_history, condition, parameters)
        return jstats.norm.logpdf(particle.log_vol, loc=loc, scale=scale)
    
class LogReturn(Emission[LatentVol, Underlying, TimeIncrement, LogVolRW]):
    order: ClassVar[int] = 2 # depends on last particle and particle now
    observation_dependency: ClassVar[int] = 1

    @staticmethod
    def sample(
        key: PRNGKeyArray,
        particle: tuple[LatentVol, LatentVol],
        observation_history: tuple[Underlying],
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Underlying:
        last_particle, _ = particle
        prev_observation, = observation_history
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(last_particle.log_vol)
        log_return = jrandom.normal(key) * return_scale
        return Underlying(underlying=prev_observation.underlying * jnp.exp(log_return))

    @staticmethod
    def log_p(
        particle: tuple[LatentVol, LatentVol],
        observation_history: tuple[Underlying],
        observation: Underlying,
        condition: TimeIncrement,
        parameters: LogVolRW,
    ) -> Scalar:
        last_particle, _ = particle
        prev_observation, = observation_history
        return_scale = jnp.sqrt(condition.dt) * jnp.exp(last_particle.log_vol)
        log_return = jnp.log(observation.underlying) - jnp.log(
            prev_observation.underlying
        )
        return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)
    

# class SkewRandomWalk(Transition[LatentVol, LastUnderlying, LogVolWithSkew]):
#     @staticmethod
#     def sample(
#         key: PRNGKeyArray,
#         particle: LatentVol,
#         condition: LastUnderlying,
#         parameters: LogVolWithSkew,
#     ) -> LatentVol:
#         dt = condition.dt
#         eps_log_vol = jrandom.normal(key)

#         log_vol_mean = particle.current_log_vol + dt * (
#             parameters.mean_reversion
#             * (parameters.long_term_log_vol - particle.current_log_vol)
#         )

#         noise = parameters.std_log_vol * jnp.sqrt(dt) * eps_log_vol
#         next_log_vol = log_vol_mean + noise

#         return LatentVol(
#             current_log_vol=next_log_vol,
#             last_log_vol=particle.current_log_vol,
#         )

    # @staticmethod
    # def log_p(
    #     particle: LatentVol,
    #     next_particle: LatentVol,
    #     condition: LastUnderlying,
    #     parameters: LogVolWithSkew,
    # ) -> Scalar:
    #     dt = condition.dt

    #     log_vol_mean = particle.current_log_vol + dt * (
    #         parameters.mean_reversion
    #         * (parameters.long_term_log_vol - particle.current_log_vol)
    #     )

    #     # next_particle.last_log_vol == particle.current_log_vol
    #     return jstats.norm.logpdf(
    #         next_particle.current_log_vol,
    #         loc=log_vol_mean,
    #         scale=parameters.std_log_vol * jnp.sqrt(dt),
    #     )


# class LogReturn(Emission[LatentVol, LastUnderlying, Underlying, LogVolWithSkew]):
#     order = 1
#     latent_dependency = 1

#     @staticmethod
#     def sample(
#         key: PRNGKeyArray,
#         particle: Tuple[LatentVol, LatentVol],
#         prev_observation: Tuple[Underlying],
#         condition: LastUnderlying,
#         parameters: LogVolRW,
#     ) -> Underlying:
#         last_particle, _ = particle
#         prev_observation, = prev_observation
#         return_scale = jnp.sqrt(condition.dt) * jnp.exp(last_particle.log_vol)
#         log_return = jrandom.normal(key) * return_scale
#         return Underlying(underlying=prev_observation.underlying * jnp.exp(log_return))

#     @staticmethod
#     def log_p(
#         particle: Tuple[LatentVol, LatentVol],
#         prev_observation: Tuple[Underlying],
#         observation: Underlying,
#         condition: LastUnderlying,
#         parameters: LogVolRW,
#     ):
#         last_particle, _ = particle
#         prev_observation, = prev_observation
#         return_scale = jnp.sqrt(condition.dt) * jnp.exp(last_particle.log_vol)
#         log_return = jnp.log(observation.underlying) - jnp.log(
#             prev_observation.underlying
#         )
#         return jstats.norm.logpdf(log_return, loc=0.0, scale=return_scale)


# class SkewLogReturn(Emission[LatentVol, LastUnderlying, Underlying, LogVolWithSkew]):
#     @staticmethod
#     def return_mean_and_scale(
#         particle: LatentVol,
#         condition: LastUnderlying,
#         parameters: LogVolWithSkew,
#     ) -> tuple[Scalar, Scalar]:
#         dt = condition.dt

#         last_vol = jnp.exp(particle.last_log_vol)
#         last_var = jnp.exp(2 * particle.last_log_vol)

#         log_vol_mean = particle.last_log_vol + dt * (
#             parameters.mean_reversion
#             * (parameters.long_term_log_vol - particle.last_log_vol)
#         )

#         return_mean = -0.5 * dt * last_var
#         return_mean -= (
#             parameters.skew
#             * (last_vol / parameters.std_log_vol)
#             * (particle.current_log_vol - log_vol_mean)
#         )

#         return_scale = jnp.sqrt(condition.dt) * last_vol * (1 - parameters.skew**2)

#         return return_mean, return_scale

#     @staticmethod
#     def sample(
#         key: PRNGKeyArray,
#         particle: LatentVol,
#         condition: LastUnderlying,
#         parameters: LogVolWithSkew,
#     ) -> Underlying:

#         return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
#             particle, condition, parameters
#         )

#         log_return = jrandom.normal(key) * return_scale + return_mean

#         return Underlying(underlying=condition.last_underlying * jnp.exp(log_return))

#     @staticmethod
#     def log_p(
#         particle: LatentVol,
#         observation: Underlying,
#         condition: LastUnderlying,
#         parameters: LogVolWithSkew,
#     ) -> Scalar:
#         return_mean, return_scale = SkewLogReturn.return_mean_and_scale(
#             particle, condition, parameters
#         )

#         log_return = jnp.log(observation.underlying) - jnp.log(
#             condition.last_underlying
#         )

#         return jstats.norm.logpdf(log_return, loc=return_mean, scale=return_scale)


class SimpleStochasticVol(Target[LatentVol, Underlying, TimeIncrement, LogVolRW]):
    # particle_type = LatentVol
    prior = GaussianStart()
    transition = RandomWalk()
    emission = LogReturn()



# class SkewStochasticVol(Target[LatentVol, LastUnderlying, Underlying, LogVolWithSkew]):
#     particle_type = LatentVol
    
#     prior = UniformStart()
#     transition = SkewRandomWalk()
#     emission = SkewLogReturn()

#     @staticmethod
#     def emission_to_condition(
#         observation: Underlying,
#         condition: LastUnderlying,
#         parameters: LogVolRW,
#     ):
#         return LastUnderlying(
#             dt=condition.dt,
#             last_underlying=observation.underlying,
#         )

#     @staticmethod
#     def reference_emission(parameters: LogVolWithSkew):
#         return Underlying(underlying=parameters.initial_underlying)
