"""Test cases for implementations."""

from probdiffeq.backend import testing
from probdiffeq.implementations import cubature, recipes


@testing.case(id="IsoTS0", tags=["nd"])
def case_ts0_iso():
    def impl_factory(*, num_derivatives, ode_shape):
        return recipes.IsoTS0.from_params(num_derivatives=num_derivatives)

    return impl_factory


@testing.case(id="BlockDiagTS0", tags=["nd"])
def case_ts0_blockdiag():
    return recipes.BlockDiagTS0.from_params


@testing.case(id="DenseTS1", tags=["nd"])
def case_ts1_dense():
    return recipes.DenseTS1.from_params


@testing.case(id="DenseTS0", tags=["scalar"])
def case_ts0_scalar():
    def impl_factory(*, num_derivatives, ode_shape=()):
        return recipes.ScalarTS0.from_params(num_derivatives=num_derivatives)

    return impl_factory


@testing.case(id="DenseTS0", tags=["nd"])
def case_ts0_dense():
    return recipes.DenseTS0.from_params


@testing.case(id="DenseSLR1(Default)", tags=["nd"])
def case_slr1_dense_default():
    def impl_factory(**kwargs):
        return recipes.DenseSLR1.from_params(**kwargs)

    return impl_factory


@testing.case(id="DenseSLR1(ThirdOrderSpherical)", tags=["nd"])
def case_slr1_dense_sci():
    def impl_factory(**kwargs):
        cube_fn = cubature.ThirdOrderSpherical.from_params
        return recipes.DenseSLR1.from_params(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


@testing.case(id="DenseSLR1(UnscentedTransform)", tags=["nd"])
def case_slr1_dense_ut():
    def impl_factory(**kwargs):
        cube_fn = cubature.UnscentedTransform.from_params
        return recipes.DenseSLR1.from_params(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


@testing.case(id="DenseSLR1(GaussHermite)", tags=["nd"])
def case_slr1_dense_gh():
    def impl_factory(**kwargs):
        cube_fn = cubature.GaussHermite.from_params
        return recipes.DenseSLR1.from_params(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


# todo: parametrize with different cubature rules
@testing.case(id="DenseSLR0(Default)", tags=["nd"])
def case_slr0_dense_default():
    def impl_factory(**kwargs):
        return recipes.DenseSLR0.from_params(**kwargs)

    return impl_factory


@testing.case(id="DenseSLR0(ThirdOrderSpherical)", tags=["nd"])
def case_slr0_dense_sci():
    def impl_factory(**kwargs):
        cube_fn = cubature.ThirdOrderSpherical.from_params
        return recipes.DenseSLR0.from_params(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


@testing.case(id="DenseSLR0(UnscentedTransform)", tags=["nd"])
def case_slr0_dense_ut():
    def impl_factory(**kwargs):
        cube_fn = cubature.UnscentedTransform.from_params
        return recipes.DenseSLR0.from_params(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


@testing.case(id="DenseSLR0(GaussHermite)", tags=["nd"])
def case_slr0_dense_gh():
    def impl_factory(**kwargs):
        cube_fn = cubature.GaussHermite.from_params
        return recipes.DenseSLR0.from_params(cubature_rule_fn=cube_fn, **kwargs)

    return impl_factory


# todo: parametrize with different cubature rules
@testing.case(id="BlockDiagSLR1", tags=["nd"])
def case_slr1_blockdiag():
    def impl_factory(**kwargs):
        return recipes.BlockDiagSLR1.from_params(**kwargs)

    return impl_factory
