"""Tests for IOUP."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, linalg, np, random, testing


def test_ioup_reduces_to_iwp():
    u = np.ones((2,))
    M = np.zeros((2, 2))
    tcoeffs = [u, u, u, u, u]

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    ioup = probdiffeq.prior_ioup(ssm=ssm, rate=M)
    iwp = probdiffeq.prior_iwp(ssm=ssm)

    scale = 12.3456
    dt = 0.123456

    cond1 = func.jit(ioup)(dt, scale)
    cond2 = func.jit(iwp)(dt, scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("shape", [(), (3,)])
@testing.parametrize("n", [2, 4])
def test_ioup_transition_as_expected(shape, n):
    """Follow Proposition 1 in https://arxiv.org/abs/2305.14978."""
    u = np.ones(shape)
    M = random.normal(random.prng_key(seed=1), shape=(*shape, *shape))
    tcoeffs = [u] * n

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")
    ioup = probdiffeq.prior_ioup(ssm=ssm, rate=M)

    dt = 0.123456
    cond = func.jit(ioup)(dt)
    cond = ssm.conditional.preconditioner_apply(cond)
    A_received = cond.A

    (d,) = shape if len(shape) == 1 else (1,)
    assert testing.allclose(A_received[-d:, -d:], linalg.expm(M * np.eye(1) * dt))

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs[:-1], ssm_fact="dense")
    iwp = probdiffeq.prior_iwp(ssm=ssm)
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

    with testing.raises(NotImplementedError, match="reach out"):
        _ = probdiffeq.prior_ioup(ssm=ssm, rate=M)


@testing.parametrize("shape", [(2, 2)])
def test_ioup_not_implemented_for_matrix_valued_dense(shape):
    u = np.ones(shape)
    tcoeffs = [u, u, u, u, u]

    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="dense")

    with testing.raises(NotImplementedError, match="implemented"):
        M = np.zeros((2, 2))  # irrelevant
        _ = probdiffeq.prior_ioup(ssm=ssm, rate=M)
