"""Tests for integrated Ornstein-Uhlenbeck processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, np, random, testing


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
@testing.parametrize("dt", [1.234, -1.234])
def test_ioup_matches_exponential_prior(ssm_factory, dt):
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
    cond1 = func.jit(exponential.transition)(dt=dt, output_scale=scale)
    cond2 = func.jit(ioup.transition)(dt=dt, output_scale=scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
@testing.parametrize("dt", [1.234, -1.234])
def test_ioup_diffuse_matches_exponential_diffuse_prior(ssm_factory, dt):
    """Assert that the exponential prior with IOUP drift matches prior_ornstein_uhlenbeck_integrated."""
    u = np.ones((4,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs_mean = [u, u, u, u]
    tcoeffs_std = [u, 0.1 * u, 0.2 * u, 0.3 * u]

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=4)
    def vf_exponential(u, du, ddu, dddu, /):
        del u
        del du
        del ddu
        return linop_ioup(dddu)

    def linop_ioup(x):
        return M @ x

    ssm = ssm_factory()
    exponential = ssm.prior_exponential_diffuse(
        vf_exponential, tcoeffs_mean, tcoeffs_std
    )
    ioup = ssm.prior_ornstein_uhlenbeck_integrated_diffuse(
        linop_ioup, tcoeffs_mean, tcoeffs_std
    )

    scale = 12.3456
    cond1 = func.jit(exponential.transition)(dt=dt, output_scale=scale)
    cond2 = func.jit(ioup.transition)(dt=dt, output_scale=scale)
    assert testing.allclose(cond1, cond2)
