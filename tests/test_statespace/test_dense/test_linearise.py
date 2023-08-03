def test_slr0_inexact_but_calibrated(setup, noise=1e-1):
    vf, x0 = setup
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)

    b = linearise.slr0(vf, rv0, cubature_rule=cubature.gauss_hermite(input_shape=(1,)))

    error = jnp.abs(b.mean - vf(rv0.mean)) / jnp.abs(b.cov_sqrtm_lower)
    assert 0.1 < error < 10.0


def test_slr1_inexact_but_calibrated(setup, noise=1e-1):
    vf, x0 = setup
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)
    cubature_rule = cubature.gauss_hermite(input_shape=(1,))
    A, b = linearise.slr1(vf, rv0, cubature_rule=cubature_rule)

    error = jnp.abs(A(x0) + b.mean - vf(rv0.mean)) / jnp.abs(b.cov_sqrtm_lower)
    assert 0.5 < error < 2.0


def test_ode_constraint_1st_noisy_inexact_but_calibrated(setup, noise=1e-1):
    vf, x0 = setup
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)

    cubature_rule = cubature.gauss_hermite(input_shape=(1,))
    fun = functools.partial(linearise.slr1, cubature_rule=cubature_rule)

    ode_linearise = linearise.ode_constraint_1st_noisy(fun, ode_shape=(1,), ode_order=1)
    A, b = ode_linearise(vf, rv0)

    rv = variables.DenseNormal(jnp.stack([x0] * 3), jnp.eye(3), target_shape=(3, 1))
    linearisation = A(x0) + b.mean
    truth = rv.mean[1] - vf(rv.mean[0])
    standard_deviation = rv.mean[1] - vf(rv.mean[0])

    error_abs = jnp.abs((linearisation - truth) / standard_deviation)
    error_rel = error_abs / jnp.abs(b.cov_sqrtm_lower)
    assert 0.5 < error_rel < 2.0


def test_linearisation_allclose(linearisation, truth):
    return jnp.allclose(linearisation, truth)
