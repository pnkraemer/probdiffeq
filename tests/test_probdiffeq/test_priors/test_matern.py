"""Tests for integrated Ornstein-Uhlenbeck processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, np, random, testing


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
@testing.parametrize("dt", [1.234, -1.234])
@testing.parametrize("length_scale", [0.567])
def test_matern_matches_exponential_prior(ssm_factory, dt, length_scale):
    """Assert that the exponential prior with IOUP drift matches prior_ornstein_uhlenbeck_integrated."""
    u = np.ones((4,))
    tcoeffs = [u, u, u]

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=3)
    def vf_exponential(u, du, ddu, /):
        ell = np.sqrt(2 * 2.5) / length_scale
        return -(ell**3) * u - 3 * ell**2 * du - 3 * ell * ddu

    ssm = ssm_factory()
    exponential = ssm.prior_exponential(vf_exponential, tcoeffs)
    matern = ssm.prior_matern(length_scale, tcoeffs)

    scale = 12.3456
    cond1 = func.jit(exponential.transition)(dt=dt, output_scale=scale)
    cond2 = func.jit(matern.transition)(dt=dt, output_scale=scale)
    assert testing.allclose(cond1, cond2)


@testing.parametrize("ssm_factory", [probdiffeq.state_space_model_dense])
@testing.parametrize("dt", [1.234, -1.234])
@testing.parametrize("length_scale", [0.567])
def test_matern_diffuse_matches_exponential_diffuse_prior(
    ssm_factory, dt, length_scale
):
    """Assert that the exponential prior with IOUP drift matches prior_ornstein_uhlenbeck_integrated."""
    u = np.ones((4,))
    M = random.normal(random.prng_key(seed=1), shape=(u.size, u.size))
    tcoeffs_mean = [u, u, u]
    tcoeffs_std = [u, M @ u, M @ M @ u]

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_tcoeffs_in_args=3)
    def vf_exponential(u, du, ddu, /):
        ell = np.sqrt(2 * 2.5) / length_scale
        return -(ell**3) * u - 3 * ell**2 * du - 3 * ell * ddu

    ssm = ssm_factory()
    exponential = ssm.prior_exponential_diffuse(
        vf_exponential, tcoeffs_mean, tcoeffs_std
    )
    matern = ssm.prior_matern_diffuse(length_scale, tcoeffs_mean, tcoeffs_std)

    scale = 12.3456
    cond1 = func.jit(exponential.transition)(dt=dt, output_scale=scale)
    cond2 = func.jit(matern.transition)(dt=dt, output_scale=scale)
    assert testing.allclose(cond1, cond2)
