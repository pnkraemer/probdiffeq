from probdiffeq import probdiffeq
from probdiffeq.backend import np, random, testing, tree


def test_residual_constraints():
    """Assert that residual-based constraints and corresponding TS1 versions match."""

    @probdiffeq.residual_velocity
    def f(u, du, /, *, t):
        """Evaluate a custom residual for the harmonic oscillator."""
        return du - vf(u, t=t)

    @probdiffeq.ode
    def vf(y, *, t):
        """Evaluate the harmonic oscillator as a 2nd-order problem."""
        return 2 * y * (1 - y)

    ssm = probdiffeq.state_space_model_dense()
    ode_ts1 = ssm.constraint_ode_ts1(vf)
    residual = ssm.constraint_residual(f)

    rv = create_random_variable(ssm)
    x = ode_ts1.init_linearization()
    y = residual.init_linearization()

    fx, x = ode_ts1.linearize(rv, x, damp=0.0, t=0.0)
    fy, y = residual.linearize(rv, y, damp=0.0, t=0.0)

    assert testing.allclose(fx.A, fy.A)
    assert testing.allclose(fx.noise.mean_flat, fy.noise.mean_flat)
    assert testing.allclose(fx.noise.cholesky_flat, fy.noise.cholesky_flat)


def create_random_variable(ssm, seed=1):
    tcoeffs = [np.ones(())] * 5  # values irrelevant
    iwp = ssm.prior_wiener_integrated(tcoeffs, is_exact=False, inexact_eps=1.0)
    rv = iwp.transition(dt=0.1, output_scale=1.0)

    key = random.prng_key(seed=1)
    noise_flat, unravel = tree.ravel_pytree(rv.noise)
    noise_flat = random.normal(key, shape=noise_flat.shape)
    noise = unravel(noise_flat)
    return noise
