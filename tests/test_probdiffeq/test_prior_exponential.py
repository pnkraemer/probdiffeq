"""Tests for exponential priors."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, linalg, np, random, testing, tree


@testing.parametrize("ssm_fact", ["dense"])
def test_exponential_prior_matches_ioup(ssm_fact):
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

    ssm = probdiffeq.ssm_taylor(ssm_fact=ssm_fact)
    _init, exponential = probdiffeq.prior_exponential(vf_exponential, tcoeffs, ssm=ssm)
    _init, ioup = probdiffeq.prior_ornstein_uhlenbeck_integrated(
        linop_ioup, tcoeffs, ssm=ssm
    )

    scale = 12.3456
    dt = 0.123456
    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(ioup)(dt, scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("ssm_fact", ["dense"])
def test_exponential_prior_matches_iwp(ssm_fact):
    u = np.ones((2,))
    tcoeffs = [u, u, u, u]

    def vf_linear(u, du, ddu, dddu):
        del du
        del ddu
        del dddu
        return np.zeros_like(u)

    ssm = probdiffeq.ssm_taylor(ssm_fact=ssm_fact)
    _init, exponential = probdiffeq.prior_exponential(vf_linear, tcoeffs, ssm=ssm)
    _init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)

    scale = 12.3456
    dt = 0.123456

    cond1 = func.jit(exponential)(dt, scale)
    cond2 = func.jit(iwp)(dt, scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("ssm_fact", ["dense"])
def test_exponential_raises_error_if_vf_linear_is_bad(ssm_fact):
    u = np.ones((3,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 2

    def vf_linear(u, du, ddu):
        del u
        del du
        return M @ ddu.ravel()

    ssm = probdiffeq.ssm_taylor(ssm_fact=ssm_fact)
    with testing.raises(TypeError, match="Taylor coefficients"):
        _ = probdiffeq.prior_exponential(vf_linear, tcoeffs, ssm=ssm)

    # Sanity check: equal order is fine
    tcoeffs = [u] * 3
    ssm = probdiffeq.ssm_taylor(ssm_fact=ssm_fact)
    _ = probdiffeq.prior_exponential(vf_linear, tcoeffs, ssm=ssm)


@testing.parametrize("ode_shape", [(), (3,), (3, 3)])
@testing.parametrize("ssm_fact", ["dense"])
def test_exponential_transition_as_expected(ode_shape, ssm_fact):
    """Follow Proposition 1 in https://arxiv.org/abs/2305.14978."""
    ssm = probdiffeq.ssm_taylor(ssm_fact=ssm_fact)
    u = np.ones(ode_shape)
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 3

    def vf_linear(u, du, ddu):
        del u
        del du
        return M @ ddu.ravel()

    _init, exponential = probdiffeq.prior_exponential(vf_linear, tcoeffs, ssm=ssm)

    dt = 0.123456
    cond = func.jit(exponential)(dt)
    cond = ssm.conditional.preconditioner_apply(cond)
    A_received = cond.A

    (d,) = tree.ravel_pytree(u)[0].shape
    assert testing.allclose(A_received[-d:, -d:], linalg.expm(M * np.eye(1) * dt))

    ssm = probdiffeq.ssm_taylor(ssm_fact="dense")
    _init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs[:-1], ssm=ssm)
    cond = func.jit(iwp)(dt)
    cond = ssm.conditional.preconditioner_apply(cond)
    phi_iwp_smaller = cond.A
    assert testing.allclose(A_received[:-d, :-d], phi_iwp_smaller)


@testing.parametrize("ssm_fact", ["isotropic", "blockdiag"])
def test_exponential_not_implemented_for_isotropic_or_blockdiag(ssm_fact):

    ssm = probdiffeq.ssm_taylor(ssm_fact=ssm_fact)

    def vf_linear(u, du, ddu):
        del du
        del ddu
        return np.zeros_like(u)

    u = np.ones((2,))
    tcoeffs = [u, u, u]
    with testing.raises(NotImplementedError, match="reach out"):
        _ = probdiffeq.prior_exponential(vf_linear, tcoeffs, ssm=ssm)
