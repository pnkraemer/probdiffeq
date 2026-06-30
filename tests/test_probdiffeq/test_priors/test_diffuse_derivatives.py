"""Tests for diffuse-derivative handling."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, np, testing


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
def test_diffuse_derivatives_modify_mean_and_std_correctly(prior) -> None:
    """Assert that diffuse initialisation sets the mean to known values and the std to eps for unknown derivatives."""
    tcoeffs = [2.0, 3.0]
    iwp = prior(tcoeffs, diffuse_derivatives=3, diffuse_eps=123.0)
    assert testing.allclose(iwp.init.mean, [2.0, 3.0, 0.0, 0.0, 0.0])
    assert testing.allclose(iwp.init.std, [0.0, 0.0, 123.0, 123.0, 123.0])


def test_exponential_prior_verification_respects_diffuse_derivatives():
    """The jet-coords comparison in the exponential prior should respect diffuse_derivatives.

    See issue #894.
    """
    tcoeffs = [np.ones((3,))]  # one coefficient - rest via diffuse derivatives

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=3)
    def vf_linear(u, du, ddu, /):
        del u
        del du
        return ddu

    ssm = probdiffeq.state_space_model_dense()

    msg = "does not match the Taylor"
    with testing.raises(TypeError, match=msg):  # should this be a value error?
        _ = ssm.prior_exponential(vf_linear, tcoeffs)  # fails

    _ = ssm.prior_exponential(vf_linear, tcoeffs, diffuse_derivatives=2)  # passes
