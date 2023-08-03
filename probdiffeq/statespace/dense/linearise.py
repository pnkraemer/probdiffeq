"""Linearisation."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace.dense import variables

# todo:
#  statistical linear regression (zeroth order)
#  statistical linear regression (cov-free)
#  statistical linear regression (Jacobian)
#  statistical linear regression (Bayesian cubature)


def ts0(fn, m):
    """Linearise a function with a zeroth-order Taylor series."""
    return fn(m)


def ts1(fn, m):
    """Linearise a function with a first-order Taylor series."""
    b, jvp = jax.linearize(fn, m)
    return jvp, b - jvp(m)


def slr1(fn, x, *, cubature_rule):
    """Linearise a function with first-order statistical linear regression."""
    # Create sigma-points
    pts_centered = cubature_rule.points @ x.cov_sqrtm_lower.T
    pts = x.mean[None, :] + pts_centered
    pts_centered_normed = pts_centered * cubature_rule.weights_sqrtm[:, None]

    # Evaluate the nonlinear function
    fx = jax.vmap(fn)(pts)
    fx_mean = cubature_rule.weights_sqrtm**2 @ fx
    fx_centered = fx - fx_mean[None, :]
    fx_centered_normed = fx_centered * cubature_rule.weights_sqrtm[:, None]

    # Compute statistical linear regression matrices
    _, (cov_sqrtm_cond, linop_cond) = _sqrt_util.revert_conditional_noisefree(
        R_X_F=pts_centered_normed, R_X=fx_centered_normed
    )
    mean_cond = fx_mean - linop_cond @ x.mean
    rv_cond = variables.DenseNormal(mean_cond, cov_sqrtm_cond.T, target_shape=None)
    return lambda v: linop_cond @ v, rv_cond


def slr0(fn, x, *, cubature_rule):
    """Linearise a function with zeroth-order statistical linear regression.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    # Create sigma-points
    pts_centered = cubature_rule.points @ x.cov_sqrtm_lower.T
    pts = x.mean[None, :] + pts_centered

    # Evaluate the nonlinear function
    fx = jax.vmap(fn)(pts)
    fx_mean = cubature_rule.weights_sqrtm**2 @ fx
    fx_centered = fx - fx_mean[None, :]
    fx_centered_normed = fx_centered * cubature_rule.weights_sqrtm[:, None]

    cov_sqrtm = _sqrt_util.triu_via_qr(fx_centered_normed)

    return variables.DenseNormal(fx_mean, cov_sqrtm.T, target_shape=None)


def ode_constraint_0th(linearise_fun, /, *, ode_shape, ode_order):
    def linearise_fun_wrapped(fun, mean):
        if mean.shape != ode_shape:
            raise ValueError

        select = functools.partial(_select_derivative, ode_shape=ode_shape)

        a0 = functools.partial(select, i=slice(0, ode_order))
        a1 = functools.partial(select, i=ode_order)

        fx = linearise_fun(fun, a0(mean))
        return _autobatch_linop(a1), -fx

    return linearise_fun_wrapped


def ode_constraint_0th_noisy(fun, rv, /, *, ode_shape, ode_order, linearise_fun):
    if rv.mean.shape != ode_shape:
        raise ValueError

    if ode_order > 1:
        raise ValueError

    # Projection functions
    select = functools.partial(_select_derivative, ode_shape=ode_shape)
    a0 = _autobatch_linop(functools.partial(select, i=0))
    a1 = _autobatch_linop(functools.partial(select, i=1))

    # Extract the linearisation point
    m0, r_0_nonsquare = a0(rv.mean), a0(rv.cov_sqrtm_lower)
    r_0_square = _sqrt_util.triu_via_qr(r_0_nonsquare.T)
    linearisation_pt = variables.DenseNormal(m0, r_0_square.T, target_shape=None)

    # Gather the variables and return
    noise = linearise_fun(fun, linearisation_pt)
    bias = variables.DenseNormal(
        -noise.mean, noise.cov_sqrtm_lower, target_shape=noise.target_shape
    )
    return a1, bias


def ode_constraint_1st(fun, mean, /, *, ode_shape, ode_order, linearise_fun=ts1):
    if mean.shape != ode_shape:
        raise ValueError

    select = functools.partial(_select_derivative, ode_shape=ode_shape)

    a0 = functools.partial(select, i=slice(0, ode_order))
    a1 = functools.partial(select, i=ode_order)

    fx, jvp = linearise_fun(fun, a0(mean))

    def A(x):
        return a1(x) - jvp(a0(x))

    rx = a1(mean) - fx
    return _autobatch_linop(A), rx - A(mean)


def ode_constraint_1st_noisy(linearise_fun, /, *, ode_shape, ode_order):
    def new(fun, rv):
        if rv.mean.shape != ode_shape:
            raise ValueError

        if ode_order > 1:
            raise ValueError

        # Projection functions
        select = functools.partial(_select_derivative, ode_shape=ode_shape)
        a0 = _autobatch_linop(functools.partial(select, i=0))
        a1 = _autobatch_linop(functools.partial(select, i=1))

        # Extract the linearisation point
        m0, r_0_nonsquare = a0(rv.mean), a0(rv.cov_sqrtm_lower)
        r_0_square = _sqrt_util.triu_via_qr(r_0_nonsquare.T)
        linearisation_pt = variables.DenseNormal(m0, r_0_square.T, target_shape=None)

        # Gather the variables and return
        J, noise = linearise_fun(fun, linearisation_pt)

        def A(x):
            return a1(x) - J(a0(x))

        mean, cov_lower = noise.mean, noise.cov_sqrtm_lower
        bias = variables.DenseNormal(-mean, cov_lower, target_shape=noise.target_shape)
        return _autobatch_linop(A), bias

    return new


def _select_derivative(x, i, *, ode_shape):
    (d,) = ode_shape
    x_reshaped = jnp.reshape(x, (-1, d), order="F")
    return x_reshaped[i, ...]


def _autobatch_linop(fun):
    def fun_(x):
        if jnp.ndim(x) > 1:
            return jax.vmap(fun_, in_axes=1, out_axes=1)(x)
        return fun(x)

    return fun_
