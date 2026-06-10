"""Tests for exponential priors."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, linalg, np, random, testing, tree


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
def test_exponential_prior_matches_ioup(ssm_factory):
    u = np.ones((4,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u, u, u, u]

    def vf_exponential(u, du, ddu, dddu):
        del u
        del du
        del ddu
        return linop_ioup(dddu)

    def linop_ioup(x):
        return M @ x

    ssm = ssm_factory()
    _init, exponential = ssm.prior_exponential(vf_exponential, tcoeffs)
    _init, ioup = ssm.prior_ornstein_uhlenbeck_integrated(linop_ioup, tcoeffs)

    scale = 12.3456
    dt = 0.123456
    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(ioup)(dt, scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
def test_exponential_prior_matches_iwp(ssm_factory):
    u = np.ones((2,))
    tcoeffs = [u, u, u, u]

    def vf_linear(u, du, ddu, dddu):
        del du
        del ddu
        del dddu
        return np.zeros_like(u)

    ssm = ssm_factory()
    _init, exponential = ssm.prior_exponential(vf_linear, tcoeffs)
    _init, iwp = ssm.prior_wiener_integrated(tcoeffs)

    scale = 12.3456
    dt = 0.123456

    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(iwp)(dt, scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
def test_exponential_raises_error_if_vf_linear_is_bad(ssm_factory):
    u = np.ones((3,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 2

    def vf_linear(u, du, ddu):
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
def test_exponential_transition_as_expected(ode_shape, ssm_factory):
    """Follow Proposition 1 in https://arxiv.org/abs/2305.14978."""
    ssm = ssm_factory()
    u = np.ones(ode_shape)
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 3

    def vf_linear(u, du, ddu):
        del u
        del du
        return M @ ddu.ravel()

    _init, exponential = ssm.prior_exponential(vf_linear, tcoeffs)

    dt = 0.123456
    cond = func.jit(exponential)(dt)
    cond = cond.preconditioner_apply()
    A_received = cond.A

    (d,) = tree.ravel_pytree(u)[0].shape
    assert testing.allclose(A_received[-d:, -d:], linalg.expm(M * np.eye(1) * dt))

    ssm = probdiffeq.state_space_model_dense()
    _init, iwp = ssm.prior_wiener_integrated(tcoeffs[:-1])
    cond = func.jit(iwp)(dt)
    cond = cond.preconditioner_apply()
    phi_iwp_smaller = cond.A
    assert testing.allclose(A_received[:-d, :-d], phi_iwp_smaller)


@testing.parametrize(
    "ssm_factory",
    [probdiffeq.state_space_model_isotropic, probdiffeq.state_space_model_blockdiag],
)
def test_exponential_not_implemented_for_isotropic_or_blockdiag(ssm_factory):

    ssm = ssm_factory()

    def vf_linear(u, du, ddu):
        del du
        del ddu
        return np.zeros_like(u)

    u = np.ones((2,))
    tcoeffs = [u, u, u]
    with testing.raises(NotImplementedError, match="reach out"):
        _ = ssm.prior_exponential(vf_linear, tcoeffs)
