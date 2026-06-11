"""Tests for integrated Wiener processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


def test_init_diffuse_derivatives() -> None:
    ssm = probdiffeq.state_space_model_dense()
    tcoeffs = [2.0, 3.0]
    init, _prior = ssm.prior_wiener_integrated(
        tcoeffs, diffuse_derivatives=3, diffuse_eps=123.0
    )
    assert testing.allclose(init.mean, [2.0, 3.0, 0.0, 0.0, 0.0])
    assert testing.allclose(init.std, [0.0, 0.0, 123.0, 123.0, 123.0])


def test_init_is_exact_dense() -> None:
    ssm = probdiffeq.state_space_model_dense()
    tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True])]
    init, _prior = ssm.prior_wiener_integrated(
        tcoeffs, is_exact=isdiff, inexact_eps=0.123
    )

    [m], [s] = init.mean, init.std
    assert testing.allclose(m, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s, np.asarray([0.0, 0.123, 0.0]))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray(False)]  # wrong shape
        _ = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff)


def test_init_is_exact_blockdiag() -> None:
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True]), np.asarray([False, False, True])]
    ssm = probdiffeq.state_space_model_blockdiag()
    init, _iwp = ssm.prior_wiener_integrated(
        tcoeffs, is_exact=isdiff, inexact_eps=0.123
    )

    [m1, m2], [s1, s2] = init.mean, init.std
    assert testing.allclose(m1, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(m2, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s1, np.asarray([0.0, 0.123, 0.0]))
    assert testing.allclose(s2, np.asarray([0.123, 0.123, 0.0]))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray(False)]  # wrong shape
        _ = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff)


def test_init_is_exact_isotropic() -> None:
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray(True), np.asarray(False)]
    ssm = probdiffeq.state_space_model_isotropic()
    init, _ssm = ssm.prior_wiener_integrated(
        tcoeffs, is_exact=isdiff, inexact_eps=0.123
    )

    [m1, m2], [s1, s2] = init.mean, init.std
    assert testing.allclose(m1, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(m2, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s1, np.asarray(0.0))
    assert testing.allclose(s2, np.asarray(0.123))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray([True, False, True])]  # wrong shape
        _ = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff)
