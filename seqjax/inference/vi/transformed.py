import typing

import equinox as eqx
import jaxtyping
import seqjax.model.typing
from seqjax.inference.vi.interface import (
    VariationalApproximation,
    VariationalApproximationFactory,
    UnconditionalVariationalApproximation,
)
from seqjax.inference.vi.transformations import (
    FieldwiseBijector,
    FieldwiseBijectorFactory,
)


class TransformedApproximation[T: seqjax.model.typing.Packable, C](
    VariationalApproximation[T, C]
):
    base: VariationalApproximation[T, C]
    constraint: FieldwiseBijector
    target_struct_cls: type[T] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, base: VariationalApproximation, constraint: FieldwiseBijector):
        if base.target_struct_cls != constraint.target_struct_cls:
            raise ValueError("Base and Constraint mismatch")
        self.base = base
        self.constraint = constraint
        self.target_struct_cls = base.target_struct_cls
        self.shape = base.shape

    def sample_and_log_prob(
        self,
        key: jaxtyping.PRNGKeyArray,
        condition: C,
        state: typing.Any = None,
    ) -> tuple[T, jaxtyping.Scalar, typing.Any]:
        theta_z, log_q_z, next_state = self.base.sample_and_log_prob(
            key,
            condition,
            state,
        )
        theta_x, lad = self.constraint.transform_and_lad(theta_z)
        log_q_x = log_q_z - lad
        return theta_x, log_q_x, next_state


# overload for unconditional approximation
@typing.overload
def transform_approximation[
    T: seqjax.model.typing.Packable,
](
    target_struct_class: type[T],
    base: VariationalApproximationFactory[T, None],
    constraint: FieldwiseBijectorFactory[T],
) -> UnconditionalVariationalApproximation[T]: ...


@typing.overload
def transform_approximation[
    T: seqjax.model.typing.Packable,
    C,
](
    target_struct_class: type[T],
    base: VariationalApproximationFactory[T, C],
    constraint: FieldwiseBijectorFactory[T],
) -> VariationalApproximation[T, C]: ...


def transform_approximation[
    T: seqjax.model.typing.Packable,
    C,
](
    target_struct_class: type[T],
    base: VariationalApproximationFactory[T, C],
    constraint: FieldwiseBijectorFactory[T],
) -> VariationalApproximation[T, C]:
    return TransformedApproximation(
        base(target_struct_class),
        constraint(target_struct_class),
    )
