from seqjax.inference.vi.base import (
    VariationalApproximation,
    VariationalApproximationFactory,
)
from seqjax.inference.vi.transformations import (
    FieldwiseBijector,
    FieldwiseBijectorFactory,
)


class TransformedApproximation(VariationalApproximation):
    base: VariationalApproximation
    constraint: FieldwiseBijector

    def __init__(self, base: VariationalApproximation, constraint: FieldwiseBijector):
        if base.target_struct_cls != constraint.target_struct_cls:
            raise ValueError("Base and Constraint mismatch")
        self.base = base
        self.constraint = constraint
        self.target_struct_cls = base.target_struct_cls
        self.shape = base.shape

    def sample_and_log_prob(self, key, condition=None):
        theta_z, log_q_z = self.base.sample_and_log_prob(key, condition)
        theta_x, lad = self.constraint.transform_and_lad(theta_z)
        log_q_x = log_q_z - lad
        return theta_x, log_q_x


def transform_approximation(
    target_struct_class,
    base: VariationalApproximationFactory,
    constraint: FieldwiseBijectorFactory,
) -> VariationalApproximation:
    return TransformedApproximation(
        base(target_struct_class),
        constraint(target_struct_class),
    )
