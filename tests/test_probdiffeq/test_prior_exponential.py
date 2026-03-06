"""Tests for IOUP."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, linalg, np, random, testing, tree


def test_ioup_reduces_to_iwp():
    u = np.ones((2,))
    M = np.zeros((2, 2))
    tcoeffs = [u, u, u, u, u]

    def vf_linear(u):
        return np.zeros_like(u)

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    ioup = probdiffeq.prior_exponential(vf_linear, ssm=ssm)
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)

    scale = 12.3456
    dt = 0.123456

    cond1 = func.jit(ioup)(dt, scale)
    cond2 = func.jit(iwp)(dt, scale)
    assert testing.allclose(cond1, cond2)


def test_exponential_raises_error_if_vf_linear_is_bad():
    u = np.ones((3,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * 2

    def vf_linear(ddu, du, u):
        return M @ u.ravel()

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    with testing.raises(TypeError, match="Taylor coefficients"):
        ioup = probdiffeq.prior_exponential(vf_linear, ssm=ssm)

    # Sanity check: equal order is fine
    tcoeffs = [u] * 3
    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    _ = probdiffeq.prior_exponential(vf_linear, ssm=ssm)


@testing.parametrize("shape", [(), (3,), (3, 3)])
@testing.parametrize("n", [2, 4])
def test_ioup_transition_as_expected(shape, n):
    """Follow Proposition 1 in https://arxiv.org/abs/2305.14978."""
    u = np.ones(shape)
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs = [u] * n

    def vf_linear(u):
        return M @ u.ravel()

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    ioup = probdiffeq.prior_exponential(vf_linear, ssm=ssm)

    dt = 0.123456
    cond = func.jit(ioup)(dt)
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
def test_ioup_not_implemented_for_isotropic_or_blockdiag(ssm_fact):
    u = np.ones((2,))
    M = np.zeros((2, 2))
    tcoeffs = [u, u, u, u, u]

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=ssm_fact)

    def vf_linear(u):
        return np.zeros_like(u)

    with testing.raises(NotImplementedError, match="reach out"):
        _ = probdiffeq.prior_exponential(vf_linear, ssm=ssm)
