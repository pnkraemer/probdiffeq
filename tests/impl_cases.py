"""Test cases for state-space model implementations."""

from probdiffeq.backend import testing
from probdiffeq.ssm import cubature, recipes


@testing.case(id="IsoTS0", tags=["nd"])
def case_ts0_iso():
    def impl_factory(*, num_derivatives, ode_shape):
        return recipes.ts0_iso(num_derivatives=num_derivatives)

    return impl_factory


@testing.case(id="BlockDiagTS0", tags=["nd"])
def case_ts0_blockdiag():
    return recipes.ts0_blockdiag


@testing.case(id="DenseTS1", tags=["nd"])
def case_ts1_dense():
    return recipes.ts1_dense


@testing.case(id="ScalarTS0", tags=["scalar"])
def case_ts0_scalar():
    def impl_factory(*, num_derivatives, ode_shape=()):
        return recipes.ts0_scalar(num_derivatives=num_derivatives)

    return impl_factory


@testing.case(id="DenseTS0", tags=["nd"])
def case_ts0_dense():
    return recipes.ts0_dense


@testing.case(id="DenseSLR1(Default)", tags=["nd"])
def case_slr1_dense_default():
    def impl_factory(**kwargs):
        return recipes.slr1_dense(**kwargs)

    return impl_factory


@testing.case(id="DenseSLR1(ThirdOrderSpherical)", tags=["nd"])
def case_slr1_dense_sci():
    def impl_factory(**kwargs):
        cube_fn = cubature.third_order_spherical
        return recipes.slr1_dense(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


@testing.case(id="DenseSLR1(UnscentedTransform)", tags=["nd"])
def case_slr1_dense_ut():
    def impl_factory(**kwargs):
        cube_fn = cubature.unscented_transform
        return recipes.slr1_dense(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


@testing.case(id="DenseSLR1(GaussHermite)", tags=["nd"])
def case_slr1_dense_gh():
    def impl_factory(**kwargs):
        cube_fn = cubature.gauss_hermite
        return recipes.slr1_dense(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


# todo: parametrize with different cubature rules
@testing.case(id="DenseSLR0(Default)", tags=["nd"])
def case_slr0_dense_default():
    def impl_factory(**kwargs):
        return recipes.slr0_dense(**kwargs)

    return impl_factory


@testing.case(id="DenseSLR0(ThirdOrderSpherical)", tags=["nd"])
def case_slr0_dense_sci():
    def impl_factory(**kwargs):
        cube_fn = cubature.third_order_spherical
        return recipes.slr0_dense(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


@testing.case(id="DenseSLR0(UnscentedTransform)", tags=["nd"])
def case_slr0_dense_ut():
    def impl_factory(**kwargs):
        cube_fn = cubature.unscented_transform
        return recipes.slr0_dense(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


@testing.case(id="DenseSLR0(GaussHermite)", tags=["nd"])
def case_slr0_dense_gh():
    def impl_factory(**kwargs):
        cube_fn = cubature.gauss_hermite
        return recipes.slr0_dense(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


# todo: parametrize with different cubature rules
@testing.case(id="BlockDiagSLR1", tags=["nd"])
def case_slr1_blockdiag():
    def impl_factory(**kwargs):
        return recipes.slr1_blockdiag(**kwargs)

    return impl_factory
