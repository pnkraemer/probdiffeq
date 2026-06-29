"""Tests for exponential priors."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, linalg, np, random, testing, tree


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
def test_exponential_prior_matches_ioup(ssm_factory):
    """Assert that the exponential prior with IOUP drift matches prior_ornstein_uhlenbeck_integrated."""
    u = np.ones((4,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u, u, u, u]

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=4)
    def vf_exponential(u, du, ddu, dddu, /):
        del u
        del du
        del ddu
        return linop_ioup(dddu)

    def linop_ioup(x):
        return M @ x

    ssm = ssm_factory()
    exponential = ssm.prior_exponential(vf_exponential, tcoeffs)
    ioup = ssm.prior_ornstein_uhlenbeck_integrated(linop_ioup, tcoeffs)

    scale = 12.3456
    dt = 0.123456
    cond1 = func.jit(exponential.transition)(dt=dt, output_scale=scale)
    cond2 = func.jit(ioup.transition)(dt=dt, output_scale=scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
def test_exponential_prior_matches_iwp(ssm_factory):
    """Assert that the exponential prior with zero drift matches the IWP."""
    u = np.ones((2,))
    tcoeffs = [u, u, u, u]

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=4)
    def vf_linear(u, du, ddu, dddu, /):
        del du
        del ddu
        del dddu
        return np.zeros_like(u)

    ssm = ssm_factory()
    exponential = ssm.prior_exponential(vf_linear, tcoeffs)
    iwp = ssm.prior_wiener_integrated(tcoeffs)

    scale = 12.3456
    dt = 0.123456

    cond1 = func.jit(exponential.transition)(dt=dt, output_scale=scale)
    cond2 = func.jit(iwp.transition)(dt=dt, output_scale=scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
def test_exponential_raises_error_if_vf_linear_is_bad(ssm_factory):
    """Assert that a mismatched Taylor-coefficient count raises a TypeError."""
    u = np.ones((3,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 2

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=3)
    def vf_linear(u, du, ddu, /):
        del u
        del du
        return M @ ddu.ravel()

    ssm = ssm_factory()
    with testing.raises(TypeError, match="Taylor coefficients"):
        _ = ssm.prior_exponential(vf_linear, tcoeffs)

    # Sanity check: equal order is fine
    tcoeffs = [u] * 3
    ssm = ssm_factory()
    _ = ssm.prior_exponential(vf_linear, tcoeffs)


@testing.parametrize("ode_shape", [(), (3,), (3, 3)])
@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
@testing.parametrize("sign", [1, -1])
def test_exponential_transition_as_expected(ode_shape, ssm_factory, sign):
    """Follow Proposition 1 in https://arxiv.org/abs/2305.14978."""
    ssm = ssm_factory()
    u = np.ones(ode_shape)
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 3

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=3)
    def vf_linear(u, du, ddu, /):
        del u
        del du
        return M @ ddu.ravel()

    exponential = ssm.prior_exponential(vf_linear, tcoeffs)

    dt = sign * 0.123456
    output_scale = 1.0
    cond = func.jit(exponential.transition)(dt=dt, output_scale=output_scale)
    cond = cond.preconditioner_apply()
    A_received = cond.A

    (d,) = tree.ravel_pytree(u)[0].shape
    assert testing.allclose(A_received[-d:, -d:], linalg.expm(M * np.eye(1) * dt))

    ssm = probdiffeq.state_space_model_dense()
    iwp = ssm.prior_wiener_integrated(tcoeffs[:-1])
    cond = func.jit(iwp.transition)(dt=dt, output_scale=1.0)
    cond = cond.preconditioner_apply()
    phi_iwp_smaller = cond.A
    assert testing.allclose(A_received[:-d, :-d], phi_iwp_smaller)


@testing.parametrize(
    "ssm_factory",
    [probdiffeq.state_space_model_isotropic, probdiffeq.state_space_model_blockdiag],
)
def test_exponential_not_implemented_for_isotropic_or_blockdiag(ssm_factory):
    """Assert that the exponential prior raises NotImplementedError for non-dense models."""
    ssm = ssm_factory()

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=3)
    def vf_linear(u, du, ddu, /):
        del du
        del ddu
        return np.zeros_like(u)

    u = np.ones((2,))
    tcoeffs = [u, u, u]
    with testing.raises(NotImplementedError, match="reach out"):
        _ = ssm.prior_exponential(vf_linear, tcoeffs)
