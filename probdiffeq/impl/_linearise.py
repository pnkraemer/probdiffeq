from probdiffeq.backend import abc, functools, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Callable
from probdiffeq.impl import _conditional, _normal
from probdiffeq.util import cholesky_util


class LinearisationBackend(abc.ABC):
    @abc.abstractmethod
    def ode_taylor_0th(self, ode_order: int, damp: float) -> _normal.Normal:
        raise NotImplementedError

    @abc.abstractmethod
    def ode_taylor_1st(self, ode_order: int, damp: float) -> _normal.Normal:
        raise NotImplementedError

    @abc.abstractmethod
    def ode_statistical_1st(
        self, cubature_fun: Callable, damp: float
    ) -> _normal.Normal:
        raise NotImplementedError

    @abc.abstractmethod
    def ode_statistical_0th(
        self, cubature_fun: Callable, damp: float
    ) -> _normal.Normal:
        raise NotImplementedError


class DenseLinearisation(LinearisationBackend):
    def __init__(self, ode_shape, unravel):
        self.ode_shape = ode_shape
        self.unravel = unravel

    def ode_taylor_0th(self, ode_order, damp: float):
        def linearise_fun_wrapped(fun, rv):
            mean = rv.mean

            def a1(m):
                return tree_util.ravel_pytree(self.unravel(m)[ode_order])[0]

            fx = tree_util.ravel_pytree(fun(*self.unravel(mean)[:ode_order]))[0]
            linop = functools.jacrev(a1)(mean)

            cov_lower = damp * np.eye(len(fx))
            bias = _normal.Normal(-fx, cov_lower)
            to_latent = np.ones(linop.shape[1])
            to_observed = np.ones(linop.shape[0])
            return _conditional.LatentCond(
                linop, bias, to_latent=to_latent, to_observed=to_observed
            )

        return linearise_fun_wrapped

    def ode_taylor_1st(self, ode_order, damp):
        def new(fun, rv, /):
            mean = rv.mean

            def constraint(m):
                a1 = tree_util.ravel_pytree(self.unravel(m)[ode_order])[0]
                a0 = tree_util.ravel_pytree(fun(*self.unravel(m)[:ode_order]))[0]
                return a1 - a0

            fx = constraint(mean)
            linop = functools.jacrev(constraint)(mean)
            fx = fx - linop @ mean

            cov_lower = damp * np.eye(len(fx))
            bias = _normal.Normal(fx, cov_lower)
            to_latent = np.ones(linop.shape[1])
            to_observed = np.ones(linop.shape[0])
            return _conditional.LatentCond(
                linop, bias, to_latent=to_latent, to_observed=to_observed
            )

        return new

    def ode_statistical_1st(self, cubature_fun, damp: float):
        cubature_rule = cubature_fun(input_shape=self.ode_shape)
        linearise_fun = functools.partial(self.slr1, cubature_rule=cubature_rule)

        def new(fun, rv, /):
            def select_0(s):
                return tree_util.ravel_pytree(self.unravel(s)[0])

            def select_1(s):
                return tree_util.ravel_pytree(self.unravel(s)[1])

            m0, unravel = select_0(rv.mean)

            extract_ = functools.vmap(lambda s: select_0(s)[0], in_axes=1, out_axes=1)
            r_0_nonsquare = extract_(rv.cholesky)

            # # Extract the linearisation point
            r_0_square = cholesky_util.triu_via_qr(r_0_nonsquare.T)
            linearisation_pt = _normal.Normal(m0, r_0_square.T)

            def vf_flat(u):
                return tree_util.ravel_pytree(fun(unravel(u)))[0]

            # Gather the variables and return
            J, noise = linearise_fun(vf_flat, linearisation_pt)
            mean, cov_lower = noise.mean, noise.cholesky

            def A(x):
                return select_1(x)[0] - J @ select_0(x)[0]

            linop = functools.jacrev(A)(rv.mean)

            # Include the damping term. (TODO: use a single qr?)
            damping = damp * np.eye(len(cov_lower))
            stack = np.concatenate((cov_lower.T, damping.T))
            cov_lower = cholesky_util.triu_via_qr(stack).T
            bias = _normal.Normal(-mean, cov_lower)
            to_latent = np.ones(linop.shape[1])
            to_observed = np.ones(linop.shape[0])
            return _conditional.LatentCond(
                linop, bias, to_latent=to_latent, to_observed=to_observed
            )

        return new

    def ode_statistical_0th(self, cubature_fun, damp: float):
        cubature_rule = cubature_fun(input_shape=self.ode_shape)
        linearise_fun = functools.partial(self.slr0, cubature_rule=cubature_rule)

        def new(fun, rv, /):
            def select_0(s):
                return tree_util.ravel_pytree(self.unravel(s)[0])

            m0, unravel = select_0(rv.mean)

            extract_ = functools.vmap(lambda s: select_0(s)[0], in_axes=1, out_axes=1)
            r_0_nonsquare = extract_(rv.cholesky)

            # # Extract the linearisation point
            r_0_square = cholesky_util.triu_via_qr(r_0_nonsquare.T)
            linearisation_pt = _normal.Normal(m0, r_0_square.T)

            def vf_flat(u):
                return tree_util.ravel_pytree(fun(unravel(u)))[0]

            # Gather the variables and return
            noise = linearise_fun(vf_flat, linearisation_pt)
            mean, cov_lower = noise.mean, noise.cholesky

            # Include the damping term. (TODO: use a single qr?)
            damping = damp * np.eye(len(cov_lower))
            stack = np.concatenate((cov_lower.T, damping.T))
            cov_lower = cholesky_util.triu_via_qr(stack).T

            def select_1(s):
                return tree_util.ravel_pytree(self.unravel(s)[1])

            linop = functools.jacrev(lambda s: select_1(s)[0])(rv.mean)
            bias = _normal.Normal(-mean, cov_lower)
            to_latent = np.ones(linop.shape[1])
            to_observed = np.ones(linop.shape[0])
            return _conditional.LatentCond(
                linop, bias, to_latent=to_latent, to_observed=to_observed
            )

        return new

    @staticmethod
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

    @staticmethod
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


class IsotropicLinearisation(LinearisationBackend):
    def __init__(self, unravel):
        self.unravel = unravel

    def ode_taylor_1st(self, ode_order, damp: float):
        raise NotImplementedError

    def ode_taylor_0th(self, ode_order, damp: float):
        def linearise_fun_wrapped(fun, rv):
            mean = rv.mean

            def a1(m):
                return m[[ode_order], ...]

            linop = functools.jacrev(a1)(mean[..., 0])
            fx = tree_util.ravel_pytree(fun(*self.unravel(mean)[:ode_order]))[0]

            cov_lower = damp * np.eye(1)
            bias = _normal.Normal(-fx, cov_lower)

            to_latent = np.ones((linop.shape[1],))
            to_observed = np.ones((linop.shape[0],))
            return _conditional.LatentCond(
                linop, bias, to_latent=to_latent, to_observed=to_observed
            )

        return linearise_fun_wrapped

    def ode_statistical_0th(self, cubature_fun, damp: float):
        raise NotImplementedError

    def ode_statistical_1st(self, cubature_fun, damp: float):
        raise NotImplementedError


class BlockDiagLinearisation(LinearisationBackend):
    def ode_taylor_0th(self, ode_order, damp: float):
        def linearise_fun_wrapped(fun, rv):
            mean = rv.mean
            m0 = mean[:, :ode_order]
            fx = self.ts0(fun, m0.T)

            def a1(s):
                return s[[ode_order], ...]

            @functools.vmap
            def lo(s):
                return _jac_materialize(lambda v, _p: a1(v), inputs=s)

            linop = lo(mean)
            d, *_ = linop.shape
            cov_lower = damp * np.ones((d, 1, 1))
            bias = _normal.Normal(-fx[:, None], cov_lower)

            to_latent = np.ones((linop.shape[2],))
            to_observed = np.ones((linop.shape[1],))
            return _conditional.LatentCond(
                linop, bias, to_latent=to_latent, to_observed=to_observed
            )

        return linearise_fun_wrapped

    def ode_taylor_1st(self, ode_order, damp: float):
        raise NotImplementedError

    def ode_statistical_0th(self, cubature_fun, damp: float):
        raise NotImplementedError

    def ode_statistical_1st(self, cubature_fun, damp: float):
        raise NotImplementedError

    @staticmethod
    def ts0(fn, m):
        return fn(m)


def _jac_materialize(func, /, *, inputs, params=None):
    return functools.jacrev(lambda v: func(v, params))(inputs)
