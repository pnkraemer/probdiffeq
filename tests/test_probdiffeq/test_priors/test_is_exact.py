"""Tests for the 'is_exact' flags in prior construction."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, linalg, np, testing


@testing.case
def case_ssm_dense():
    """Construct a dense SSM."""
    return probdiffeq.state_space_model_dense()


@testing.case
def case_ssm_isotropic():
    """Construct an isotropic SSM."""
    return probdiffeq.state_space_model_isotropic()


@testing.case
def case_ssm_blockdiag():
    """Construct a blockdiagonal SSM."""
    return probdiffeq.state_space_model_blockdiag()


@testing.case
@testing.parametrize_with_cases("ssm", cases=".", prefix="case_ssm_")
def case_prior_iwp(ssm):
    """Construct an integrated Wiener process."""
    return ssm.prior_wiener_integrated


@testing.case
def case_prior_ioup():
    """Construct an integrated OU process."""
    ssm = probdiffeq.state_space_model_dense()
    return func.partial(ssm.prior_ornstein_uhlenbeck_integrated, lambda s: s)


@testing.case
def case_prior_matern():
    """Construct a Matern process."""
    ssm = probdiffeq.state_space_model_dense()
    return func.partial(ssm.prior_matern, 1.0)


@testing.parametrize_with_cases("prior", cases=".", prefix="case_prior_")
def test_is_exact_handles_plain_booleans(prior) -> None:
    """Assert that exact initialisation zeros the std for known Taylor coefficients in the isotropic model."""
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]

    # All SSMs should handle plain booleans and promote them internally
    isdiff = [True, False]
    iwp = prior(tcoeffs, is_exact=isdiff, inexact_eps=0.123)

    [m1, m2], [s1, s2] = iwp.init.mean, iwp.init.std
    assert testing.allclose(m1, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(m2, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(linalg.vector_norm(s1), 0)
    assert linalg.vector_norm(s2) > 0


def test_isotropic_is_exact_rejects_array_valued_flags() -> None:
    """Assert that exact initialisation zeros the std for known Taylor coefficients in the isotropic model."""
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]

    ssm = probdiffeq.state_space_model_isotropic()

    # All SSMs should handle plain booleans and promote them internally
    isdiff = [True, False]
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


def test_dense_is_exact_accepts_array_valued_flags() -> None:
    """Assert that exact initialisation zeros the std for known Taylor coefficients in the dense model."""
    ssm = probdiffeq.state_space_model_dense()
    tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True])]
    iwp = ssm.prior_wiener_integrated(tcoeffs, is_exact=isdiff, inexact_eps=0.123)

    [m], [s] = iwp.init.mean, iwp.init.std
    assert testing.allclose(m, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s, np.asarray([0.0, 0.123, 0.0]))


def test_blockdiag_is_exact_accepts_array_valued_flags() -> None:
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
