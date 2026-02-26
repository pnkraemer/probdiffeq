"""Tests for IOUP."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


def test_ioup_reduces_to_iwp():
    u = np.ones((2,))
    M = np.zeros((2, 2))
    tcoeffs = [u, u, u]

    init, ioup, ssm = probdiffeq.prior_ornstein_uhlenbeck_integrated(tcoeffs, M=M)
    _init, iwp, _ssm = probdiffeq.prior_wiener_integrated(tcoeffs)

    scale = 12.3456
    dt = 0.123456

    cond1 = ioup(dt, scale)
    cond2 = iwp(dt, scale)

    print(cond1.A)
    print(cond2.A)
    print(cond1.noise.mean)
    print(cond2.noise.mean)
    print(cond1.noise.cholesky)
    print(cond2.noise.cholesky)

    assert testing.allclose(cond1, cond2)
