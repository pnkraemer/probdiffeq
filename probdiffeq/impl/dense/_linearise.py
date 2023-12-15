"""Linearisation."""


from probdiffeq.backend import functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _linearise
from probdiffeq.impl.dense import _normal
from probdiffeq.util import cholesky_util, linop_util


class LinearisationBackend(_linearise.LinearisationBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def ode_taylor_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            a0 = functools.partial(self._select_dy, idx_or_slice=slice(0, ode_order))
            a1 = functools.partial(self._select_dy, idx_or_slice=ode_order)

            if np.shape(a0(mean)) != (expected_shape := (ode_order, *self.ode_shape)):
                msg = f"{np.shape(a0(mean))} != {expected_shape}"
                raise ValueError(msg)

            fx = ts0(fun, a0(mean))
            linop = linop_util.parametrised_linop(lambda v, _p: _autobatch_linop(a1)(v))
            return linop, -fx

        return linearise_fun_wrapped

    def ode_taylor_1st(self, ode_order):
        def new(fun, mean, /):
            a0 = functools.partial(self._select_dy, idx_or_slice=slice(0, ode_order))
            a1 = functools.partial(self._select_dy, idx_or_slice=ode_order)

            if np.shape(a0(mean)) != (expected_shape := (ode_order, *self.ode_shape)):
                msg = f"{np.shape(a0(mean))} != {expected_shape}"
                raise ValueError(msg)

            jvp, fx = ts1(fun, a0(mean))

            @_autobatch_linop
            def A(x):
                x1 = a1(x)
                x0 = a0(x)
                return x1 - jvp(x0)

            linop = linop_util.parametrised_linop(lambda v, _p: A(v))
            return linop, -fx

        return new

    def ode_statistical_1st(self, cubature_fun):
        cubature_rule = cubature_fun(input_shape=self.ode_shape)
        linearise_fun = functools.partial(slr1, cubature_rule=cubature_rule)

        def new(fun, rv, /):
            # Projection functions
            a0 = _autobatch_linop(functools.partial(self._select_dy, idx_or_slice=0))
            a1 = _autobatch_linop(functools.partial(self._select_dy, idx_or_slice=1))

            # Extract the linearisation point
            m0, r_0_nonsquare = a0(rv.mean), a0(rv.cholesky)
            r_0_square = cholesky_util.triu_via_qr(r_0_nonsquare.T)
            linearisation_pt = _normal.Normal(m0, r_0_square.T)

            # Gather the variables and return
            J, noise = linearise_fun(fun, linearisation_pt)

            def A(x):
                return a1(x) - J @ a0(x)

            linop = linop_util.parametrised_linop(lambda v, _p: A(v))

            mean, cov_lower = noise.mean, noise.cholesky
            bias = _normal.Normal(-mean, cov_lower)
            return linop, bias

        return new

    def ode_statistical_0th(self, cubature_fun):
        cubature_rule = cubature_fun(input_shape=self.ode_shape)
        linearise_fun = functools.partial(slr0, cubature_rule=cubature_rule)

        def new(fun, rv, /):
            # Projection functions
            a0 = _autobatch_linop(functools.partial(self._select_dy, idx_or_slice=0))
            a1 = _autobatch_linop(functools.partial(self._select_dy, idx_or_slice=1))

            # Extract the linearisation point
            m0, r_0_nonsquare = a0(rv.mean), a0(rv.cholesky)
            r_0_square = cholesky_util.triu_via_qr(r_0_nonsquare.T)
            linearisation_pt = _normal.Normal(m0, r_0_square.T)

            # Gather the variables and return
            noise = linearise_fun(fun, linearisation_pt)
            mean, cov_lower = noise.mean, noise.cholesky
            bias = _normal.Normal(-mean, cov_lower)
            linop = linop_util.parametrised_linop(lambda v, _p: a1(v))
            return linop, bias

        return new

    def _select_dy(self, x, idx_or_slice):
        (d,) = self.ode_shape
        x_reshaped = np.reshape(x, (-1, d), order="F")
        return x_reshaped[idx_or_slice, ...]


def _autobatch_linop(fun):
    def fun_(x):
        if np.ndim(x) > 1:
            return functools.vmap(fun_, in_axes=1, out_axes=1)(x)
        return fun(x)

    return fun_


def ts0(fn, m):
    return fn(m)


def ts1(fn, m):
    b, jvp = functools.linearize(fn, m)
    return jvp, b - jvp(m)


def slr1(fn, x, *, cubature_rule):
    """Linearise a function with first-order statistical linear regression."""
    # Create sigma-points
    pts_centered = cubature_rule.points @ x.cholesky.T
    pts = x.mean[None, :] + pts_centered
    pts_centered_normed = pts_centered * cubature_rule.weights_sqrtm[:, None]

    # Evaluate the nonlinear function
    fx = functools.vmap(fn)(pts)
    fx_mean = cubature_rule.weights_sqrtm**2 @ fx
    fx_centered = fx - fx_mean[None, :]
    fx_centered_normed = fx_centered * cubature_rule.weights_sqrtm[:, None]

    # Compute statistical linear regression matrices
    _, (cov_sqrtm_cond, linop_cond) = cholesky_util.revert_conditional_noisefree(
        R_X_F=pts_centered_normed, R_X=fx_centered_normed
    )
    mean_cond = fx_mean - linop_cond @ x.mean
    rv_cond = _normal.Normal(mean_cond, cov_sqrtm_cond.T)
    return linop_cond, rv_cond


def slr0(fn, x, *, cubature_rule):
    """Linearise a function with zeroth-order statistical linear regression.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    # Create sigma-points
    pts_centered = cubature_rule.points @ x.cholesky.T
    pts = x.mean[None, :] + pts_centered

    # Evaluate the nonlinear function
    fx = functools.vmap(fn)(pts)
    fx_mean = cubature_rule.weights_sqrtm**2 @ fx
    fx_centered = fx - fx_mean[None, :]
    fx_centered_normed = fx_centered * cubature_rule.weights_sqrtm[:, None]

    cov_sqrtm = cholesky_util.triu_via_qr(fx_centered_normed)

    return _normal.Normal(fx_mean, cov_sqrtm.T)
