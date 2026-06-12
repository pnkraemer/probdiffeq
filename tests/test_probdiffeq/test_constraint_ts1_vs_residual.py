from probdiffeq import probdiffeq
from probdiffeq.backend import np, random, testing, tree


@testing.parametrize("seed", [1, 2])
def test_residual_matches_ts1(seed: int):
    """Assert that residual-based constraints and corresponding TS1 versions match."""

    @probdiffeq.residual_velocity
    def f(u, du, /, *, t):
        """Evaluate the residual corresponding to the ODE."""
        return du - vf(u, t=t)

    @probdiffeq.ode
    def vf(y, *, t):
        """Evaluate the ODE vector field."""
        del t
        return 2 * y * (1 - y)

    ssm = probdiffeq.state_space_model_dense()
    ode_ts1 = ssm.constraint_ode_ts1(vf)
    residual = ssm.constraint_residual(f)

    rv = _create_random_variable(ssm, seed=seed)
    x = ode_ts1.init_linearization()
    y = residual.init_linearization()

    fx, x = ode_ts1.linearize(rv, x, damp=0.0, t=0.0)
    fy, y = residual.linearize(rv, y, damp=0.0, t=0.0)

    assert testing.allclose(fx.A, fy.A)
    assert testing.allclose(fx.noise.mean_flat, fy.noise.mean_flat)
    assert testing.allclose(fx.noise.cholesky_flat, fy.noise.cholesky_flat)


@testing.parametrize("seed", [1, 2])
@testing.parametrize("lift_by", [1, 2])
def test_residual_matches_ts1_jet_lift(seed: int, lift_by: int):
    """Assert that residual-based constraints and corresponding TS1 versions match."""

    @probdiffeq.residual_velocity
    def f(u, du, /, *, t):
        """Evaluate the residual corresponding to the ODE."""
        return du - vf(u, t=t)

    @probdiffeq.ode
    def vf(y, *, t):
        """Evaluate the ODE vector field."""
        del t
        return 2 * y * (1 - y)

    ssm = probdiffeq.state_space_model_dense()
    vf_lifted = probdiffeq.ode_jet_lift(vf, lift_by=lift_by)
    ode_ts1 = ssm.constraint_ode_ts1(vf_lifted)
    residual_lifted = probdiffeq.residual_jet_lift(f, lift_by=lift_by)
    residual = ssm.constraint_residual(residual_lifted)

    rv = _create_random_variable(ssm, seed=seed)
    x = ode_ts1.init_linearization()
    y = residual.init_linearization()

    fx, x = ode_ts1.linearize(rv, x, damp=0.0, t=0.0)
    fy, y = residual.linearize(rv, y, damp=0.0, t=0.0)

    assert testing.allclose(fx.A, fy.A)
    assert testing.allclose(fx.noise.mean_flat, fy.noise.mean_flat)
    assert testing.allclose(fx.noise.cholesky_flat, fy.noise.cholesky_flat)


def _create_random_variable(ssm, seed):
    tcoeffs = [np.ones(())] * 5  # values irrelevant
    iwp = ssm.prior_wiener_integrated(tcoeffs, is_exact=False, inexact_eps=1.0)
    rv = iwp.transition(dt=0.1, output_scale=1.0)

    key = random.prng_key(seed=seed)
    noise_flat, unravel = tree.ravel_pytree(rv.noise)
    noise_flat = random.normal(key, shape=noise_flat.shape)
    return unravel(noise_flat)
