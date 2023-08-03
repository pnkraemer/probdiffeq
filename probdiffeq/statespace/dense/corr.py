"""Corrections."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _corr, cubature
from probdiffeq.statespace.dense import linearise, variables


def taylor_order_zero(*, ode_shape, ode_order):
    _tmp = functools.partial(linearise.ode_constraint_0th, ode_shape=ode_shape)
    linearise_fun = functools.partial(_tmp, ode_order=ode_order)
    return _DenseODEConstraint(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fun=linearise_fun,
        string_repr=f"<TS0 with ode_order={ode_order}>",
    )


def taylor_order_one(*, ode_shape, ode_order):
    _tmp = functools.partial(linearise.ode_constraint_1st, ode_shape=ode_shape)
    linearise_fun = functools.partial(_tmp, ode_order=ode_order)
    return _DenseODEConstraint(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fun=linearise_fun,
        string_repr=f"<TS1 with ode_order={ode_order}>",
    )


def statistical_order_zero(
    ode_shape,
    ode_order,
    cubature_rule_fn=cubature.third_order_spherical,
):
    cubature_rule = cubature_rule_fn(input_shape=ode_shape)
    linearise_fn = functools.partial(linearise.slr0, cubature_rule=cubature_rule)
    return _DenseSLR0(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fn=linearise_fn,
    )


def statistical_order_one(
    ode_shape,
    ode_order,
    cubature_rule_fn=cubature.third_order_spherical,
):
    cubature_rule = cubature_rule_fn(input_shape=ode_shape)
    linearise_fn = functools.partial(linearise.slr1, cubature_rule=cubature_rule)
    return _DenseSLR1(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fn=linearise_fn,
    )


@jax.tree_util.register_pytree_node_class
class _DenseODEConstraint(_corr.Correction):
    def __init__(self, ode_shape, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    # todo: move outside class
    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_shape, self.linearise, self.string_repr
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape, lin, string_repr = aux
        return cls(
            ode_order=ode_order,
            ode_shape=ode_shape,
            linearise_fun=lin,
            string_repr=string_repr,
        )

    def init(self, ssv, /):
        m_like = jnp.zeros(self.ode_shape)
        chol_like = jnp.zeros(self.ode_shape + self.ode_shape)
        obs_like = variables.DenseNormal(m_like, chol_like, target_shape=None)
        return ssv, obs_like

    def estimate_error(self, ssv: variables.DenseSSV, corr, /, vector_field, t, p):
        def f_wrapped(s):
            return vector_field(*s, t=t, p=p)

        A, b = self.linearise(f_wrapped, ssv.hidden_state.mean)
        observed = variables.marginalise_deterministic(ssv.hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, ssv, corr, /, vector_field, t, p):
        A, b = corr
        observed, (cor, _gn) = variables.revert_deterministic(ssv.hidden_state, (A, b))

        u = jnp.reshape(cor.mean, ssv.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, cor, target_shape=ssv.target_shape)
        return ssv, observed

    def extract(self, ssv, corr, /):
        return ssv


def _estimate_error(observed, /):
    mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros_like(observed.mean))
    output_scale = mahalanobis_norm / jnp.sqrt(observed.mean.size)
    error_estimate_unscaled = observed.marginal_stds()
    return output_scale * error_estimate_unscaled


@jax.tree_util.register_pytree_node_class
class _DenseSLR0(_corr.Correction):
    """Zeroth-order statistical linear regression in state-space models \
     with dense covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def __init__(self, ode_shape, ode_order, linearise_fn):
        if ode_order > 1:
            raise ValueError
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape
        self.linearise_fn = linearise_fn

        # Selection matrices
        fn, fn_vect = _select_derivative, _select_derivative_vect
        select = functools.partial(fn, ode_shape=self.ode_shape)
        select_vect = functools.partial(fn_vect, ode_shape=self.ode_shape)
        self.e0 = functools.partial(select, i=0)
        self.e1 = functools.partial(select, i=1)
        self.e0_vect = functools.partial(select_vect, i=0)
        self.e1_vect = functools.partial(select_vect, i=self.ode_order)

    def __repr__(self):
        return f"<SLR0 with ode_order={self.ode_order}>"

    def init(self, ssv, /):
        m_like = jnp.zeros(self.ode_shape)
        chol_like = jnp.zeros(self.ode_shape + self.ode_shape)
        obs_like = variables.DenseNormal(m_like, chol_like, target_shape=None)
        return ssv, obs_like

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = ()
        aux = self.ode_order, self.ode_shape, self.linearise_fn
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape, linearise_fn = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape, linearise_fn=linearise_fn)

    def estimate_error(self, ssv: variables.DenseSSV, corr, /, vector_field, t, p):
        # Compute the linearisation point
        m_0 = self.e0(ssv.hidden_state.mean)
        r_0 = self.e0_vect(ssv.hidden_state.cov_sqrtm_lower).T
        r_0_square = _sqrt_util.triu_via_qr(r_0)
        lin_pt = variables.DenseNormal(m_0, r_0_square.T, target_shape=ssv.target_shape)

        # todo: higher-order ODEs
        def f_wrapped(s):
            return vector_field(s, t=t, p=p)

        # Apply statistical linear regression to the ODE vector field
        noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)
        cache = (f_wrapped,)

        # Compute the marginal observation
        m_1 = self.e1(ssv.hidden_state.mean)
        r_1 = self.e1_vect(ssv.hidden_state.cov_sqrtm_lower).T
        m_marg = m_1 - noise.mean
        l_marg = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(r_1, noise.cov_sqrtm_lower.T)
        ).T
        marginals = variables.DenseNormal(m_marg, l_marg, target_shape=None)

        # Compute output scale and error estimate
        mahalanobis_norm = marginals.mahalanobis_norm(jnp.zeros_like(m_marg))
        output_scale = mahalanobis_norm / jnp.sqrt(m_marg.size)
        error_estimate_unscaled = marginals.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return error_estimate, marginals, cache

    def complete(self, ssv, corr, /, vector_field, t, p):
        # Select the required derivatives
        m_0 = self.e0(ssv.hidden_state.mean)
        m_1 = self.e1(ssv.hidden_state.mean)
        r_0 = self.e0_vect(ssv.hidden_state.cov_sqrtm_lower).T
        r_1 = self.e1_vect(ssv.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrt_util.triu_via_qr(r_0)
        lin_pt = variables.DenseNormal(m_0, r_0_square.T, target_shape=ssv.target_shape)

        # Apply statistical linear regression to the ODE vector field
        (f_wrapped, *_) = corr
        noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = ssv.hidden_state.cov_sqrtm_lower
        HL = r_1.T
        r_marg, (r_bw, gain) = _sqrt_util.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )
        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - noise.mean
        marginals = variables.DenseNormal(m_marg, r_marg.T, target_shape=None)

        # Compute the corrected mean and gather the correction
        m_bw = ssv.hidden_state.mean - gain @ m_marg
        rv = variables.DenseNormal(m_bw, r_bw.T, target_shape=ssv.target_shape)
        u = m_bw.reshape(ssv.target_shape, order="F")[0, :]
        corrected = variables.DenseSSV(u, rv, target_shape=ssv.target_shape)
        return corrected, marginals

    def extract(self, ssv, corr, /):
        return ssv


@jax.tree_util.register_pytree_node_class
class _DenseSLR1(_corr.Correction):
    def __init__(self, ode_shape, ode_order, linearise_fn):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape
        self.linearise_fn = linearise_fn

        # Selection matrices
        fn, fn_vect = _select_derivative, _select_derivative_vect
        select = functools.partial(fn, ode_shape=self.ode_shape)
        select_vect = functools.partial(fn_vect, ode_shape=self.ode_shape)
        self.e0 = functools.partial(select, i=0)
        self.e1 = functools.partial(select, i=1)
        self.e0_vect = functools.partial(select_vect, i=0)
        self.e1_vect = functools.partial(select_vect, i=self.ode_order)

    def __repr__(self):
        return f"<SLR1 with ode_order={self.ode_order}>"

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = ()
        aux = self.ode_order, self.ode_shape, self.linearise_fn
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape, linearise_fn = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape, linearise_fn=linearise_fn)

    def init(self, ssv, /):
        m_like = jnp.zeros(self.ode_shape)
        chol_like = jnp.zeros(self.ode_shape + self.ode_shape)
        obs_like = variables.DenseNormal(m_like, chol_like, target_shape=None)
        return ssv, obs_like

    def estimate_error(self, ssv: variables.DenseSSV, corr, /, vector_field, t, p):
        # Compute the linearisation point
        m_0 = self.e0(ssv.hidden_state.mean)
        r_0 = self.e0_vect(ssv.hidden_state.cov_sqrtm_lower).T
        r_0_square = _sqrt_util.triu_via_qr(r_0)
        lin_pt = variables.DenseNormal(m_0, r_0_square.T, target_shape=ssv.target_shape)

        # todo: higher-order ODEs
        def f_wrapped(s):
            return vector_field(s, t=t, p=p)

        # Apply statistical linear regression to the ODE vector field
        linop, noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)
        cache = (f_wrapped,)

        # Compute the marginal observation
        m_1 = self.e1(ssv.hidden_state.mean)
        r_1 = self.e1_vect(ssv.hidden_state.cov_sqrtm_lower).T
        m_marg = m_1 - (linop @ m_0 + noise.mean)
        l_marg = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(r_1, r_0_square @ linop.T, noise.cov_sqrtm_lower.T)
        ).T
        marginals = variables.DenseNormal(m_marg, l_marg, target_shape=None)

        # Compute output scale and error estimate
        mahalanobis_norm = marginals.mahalanobis_norm(jnp.zeros_like(m_marg))
        output_scale = mahalanobis_norm / jnp.sqrt(m_marg.size)
        error_estimate_unscaled = marginals.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return error_estimate, marginals, cache

    def complete(self, ssv, corr, /, vector_field, t, p):
        # Select the required derivatives
        m_0 = self.e0(ssv.hidden_state.mean)
        m_1 = self.e1(ssv.hidden_state.mean)
        r_0 = self.e0_vect(ssv.hidden_state.cov_sqrtm_lower).T
        r_1 = self.e1_vect(ssv.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrt_util.triu_via_qr(r_0)
        lin_pt = variables.DenseNormal(m_0, r_0_square.T, target_shape=ssv.target_shape)

        # Apply statistical linear regression to the ODE vector field
        (f_wrapped, *_) = corr
        H, noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = ssv.hidden_state.cov_sqrtm_lower
        HL = r_1.T - H @ r_0.T
        r_marg, (r_bw, gain) = _sqrt_util.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )

        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - (H @ m_0 + noise.mean)
        marginals = variables.DenseNormal(m_marg, r_marg.T, target_shape=None)

        # Compute the corrected mean and gather the correction
        m_bw = ssv.hidden_state.mean - gain @ m_marg
        rv = variables.DenseNormal(m_bw, r_bw.T, target_shape=ssv.target_shape)
        u = m_bw.reshape(ssv.target_shape, order="F")[0, :]
        corrected = variables.DenseSSV(u, rv, target_shape=ssv.target_shape)

        # Return the results
        return corrected, marginals

    def extract(self, ssv, corr, /):
        return ssv


def _select_derivative_vect(x, i, *, ode_shape):
    def select_fn(s):
        return _select_derivative(s, i, ode_shape=ode_shape)

    select = jax.vmap(select_fn, in_axes=1, out_axes=1)
    return select(x)


def _select_derivative(x, i, *, ode_shape):
    (d,) = ode_shape
    x_reshaped = jnp.reshape(x, (-1, d), order="F")
    return x_reshaped[i, ...]
