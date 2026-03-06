"""Tests for exponential priors."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, linalg, np, random, testing, tree


def test_exponential_prior_matches_matern32():
    u = np.ones((4,))
    tcoeffs = [u, u]

    ell = 0.3456

    def vf_exponential(u, du):
        return -(ell**2) * u - 2 * ell * du

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    exponential = probdiffeq.prior_exponential(vf_exponential, ssm=ssm)

    matern = probdiffeq.prior_matern(ell, ssm=ssm)

    scale = 12.3456
    dt = 0.123456
    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(matern)(dt, scale)
    assert testing.allclose(cond1, cond2)


def test_exponential_prior_matches_matern52():
    u = np.ones((4,))
    tcoeffs = [u, u, u]

    ell = 0.3456

    def vf_exponential(u, du, ddu):
        return -(ell**3) * u - 3 * ell**2 * du - 3 * ell * ddu

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    exponential = probdiffeq.prior_exponential(vf_exponential, ssm=ssm)

    matern = probdiffeq.prior_matern(ell, ssm=ssm)

    scale = 12.3456
    dt = 0.123456
    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(matern)(dt, scale)
    assert testing.allclose(cond1, cond2)


def test_exponential_prior_matches_ioup():
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

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    exponential = probdiffeq.prior_exponential(vf_exponential, ssm=ssm)

    ioup = probdiffeq.prior_ornstein_uhlenbeck_integrated(linop_ioup, ssm=ssm)

    scale = 12.3456
    dt = 0.123456
    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(ioup)(dt, scale)
    assert testing.allclose(cond1, cond2)


def test_exponential_prior_matches_oscillator():
    u = np.ones((4,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u, u, u, u]

    def vf_exponential(u, du, ddu, dddu):
        del u
        del du
        del dddu
        return linop_ioup(ddu)

    def linop_ioup(x):
        return M @ x

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    exponential = probdiffeq.prior_exponential(vf_exponential, ssm=ssm)

    ioup = probdiffeq.prior_oscillator(linop_ioup, ssm=ssm)

    scale = 12.3456
    dt = 0.123456
    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(ioup)(dt, scale)
    assert testing.allclose(cond1, cond2)


def test_exponential_prior_matches_iwp():
    u = np.ones((2,))
    tcoeffs = [u, u, u, u]

    def vf_linear(u, du, ddu, dddu):
        del du
        del ddu
        del dddu
        return np.zeros_like(u)

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    exponential = probdiffeq.prior_exponential(vf_linear, ssm=ssm)
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)

    scale = 12.3456
    dt = 0.123456

    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(iwp)(dt, scale)
    assert testing.allclose(cond1, cond2)


def test_exponential_raises_error_if_vf_linear_is_bad():
    u = np.ones((3,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 2

    def vf_linear(u, du, ddu):
        del u
        del du
        return M @ ddu.ravel()

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    with testing.raises(TypeError, match="Taylor coefficients"):
        _ = probdiffeq.prior_exponential(vf_linear, ssm=ssm)

    # Sanity check: equal order is fine
    tcoeffs = [u] * 3
    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    _ = probdiffeq.prior_exponential(vf_linear, ssm=ssm)


@testing.parametrize("ode_shape", [(), (3,), (3, 3)])
def test_exponential_transition_as_expected(ode_shape):
    """Follow Proposition 1 in https://arxiv.org/abs/2305.14978."""
    u = np.ones(ode_shape)
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 3

    def vf_linear(u, du, ddu):
        del u
        del du
        return M @ ddu.ravel()

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    exponential = probdiffeq.prior_exponential(vf_linear, ssm=ssm)

    dt = 0.123456
    cond = func.jit(exponential)(dt)
    cond = ssm.conditional.preconditioner_apply(cond)
    A_received = cond.A

    (d,) = tree.ravel_pytree(u)[0].shape
    assert testing.allclose(A_received[-d:, -d:], linalg.expm(M * np.eye(1) * dt))

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs[:-1], ssm_fact="dense")
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)
    cond = func.jit(iwp)(dt)
    cond = ssm.conditional.preconditioner_apply(cond)
    phi_iwp_smaller = cond.A
    assert testing.allclose(A_received[:-d, :-d], phi_iwp_smaller)


@testing.parametrize("ssm_fact", ["isotropic", "blockdiag"])
def test_exponential_not_implemented_for_isotropic_or_blockdiag(ssm_fact):
    u = np.ones((2,))
    tcoeffs = [u, u, u]

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=ssm_fact)

    def vf_linear(u, du, ddu):
        del du
        del ddu
        return np.zeros_like(u)

    with testing.raises(NotImplementedError, match="reach out"):
        _ = probdiffeq.prior_exponential(vf_linear, ssm=ssm)
