"""Tests for prior variable initialisation."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


def test_init_diffuse_derivatives() -> None:
    """Assert that diffuse initialisation sets the mean to known values and the std to eps for unknown derivatives."""
    ssm = probdiffeq.state_space_model_dense()
    tcoeffs = [2.0, 3.0]
    iwp = ssm.prior_wiener_integrated(tcoeffs, diffuse_derivatives=3, diffuse_eps=123.0)
    assert testing.allclose(iwp.init.mean, [2.0, 3.0, 0.0, 0.0, 0.0])
    assert testing.allclose(iwp.init.std, [0.0, 0.0, 123.0, 123.0, 123.0])


def test_init_is_exact_dense() -> None:
    """Assert that exact initialisation zeros the std for known Taylor coefficients in the dense model."""
    ssm = probdiffeq.state_space_model_dense()
    tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True])]
    iwp = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff, inexact_eps=0.123)

    [m], [s] = iwp.init.mean, iwp.init.std
    assert testing.allclose(m, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s, np.asarray([0.0, 0.123, 0.0]))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray(False)]  # wrong shape
        _ = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff)


def test_init_is_exact_blockdiag() -> None:
    """Assert that exact initialisation zeros the std for known Taylor coefficients in the blockdiag model."""
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True]), np.asarray([False, False, True])]
    ssm = probdiffeq.state_space_model_blockdiag()
    iwp = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff, inexact_eps=0.123)

    [m1, m2], [s1, s2] = iwp.init.mean, iwp.init.std
    assert testing.allclose(m1, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(m2, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s1, np.asarray([0.0, 0.123, 0.0]))
    assert testing.allclose(s2, np.asarray([0.123, 0.123, 0.0]))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray(False)]  # wrong shape
        _ = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff)


def test_init_is_exact_isotropic() -> None:
    """Assert that exact initialisation zeros the std for known Taylor coefficients in the isotropic model."""
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray(True), np.asarray(False)]
    ssm = probdiffeq.state_space_model_isotropic()
    iwp = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff, inexact_eps=0.123)

    [m1, m2], [s1, s2] = iwp.init.mean, iwp.init.std
    assert testing.allclose(m1, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(m2, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s1, np.asarray(0.0))
    assert testing.allclose(s2, np.asarray(0.123))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray([True, False, True])]  # wrong shape
        _ = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff)
