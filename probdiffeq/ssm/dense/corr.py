"""Corrections."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.ssm import _collections, cubature
from probdiffeq.ssm.dense import _vars, linearise


def taylor_order_zero(*args, **kwargs):
    return _DenseTaylorZerothOrder(*args, **kwargs)


def taylor_order_one(*args, **kwargs):
    return _DenseTaylorFirstOrder(*args, **kwargs)


def statistical_order_zero(
    ode_shape,
    ode_order,
    cubature_rule_fn=cubature.third_order_spherical,
):
    cubature_rule = cubature_rule_fn(input_shape=ode_shape)
    linearise_fn = functools.partial(linearise.slr0, cubature_rule=cubature_rule)
    return _DenseStatisticalZerothOrder(
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
    return _DenseStatisticalFirstOrder(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fn=linearise_fn,
    )


@jax.tree_util.register_pytree_node_class
class _DenseTaylorZerothOrder(_collections.AbstractCorrection):
    def __init__(self, ode_shape, ode_order):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        # Turn this into an argument if other linearisation functions apply
        self.linearise_fn = linearise.ts0

        # Selection matrices
        fn, fn_vect = _select_derivative, _select_derivative_vect
        select = functools.partial(fn, ode_shape=self.ode_shape)
        select_vect = functools.partial(fn_vect, ode_shape=self.ode_shape)
        self.e0 = functools.partial(select, i=slice(0, self.ode_order))
        self.e1 = functools.partial(select, i=self.ode_order)
        self.e1_vect = functools.partial(select_vect, i=self.ode_order)

    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape)

    def begin_correction(self, x: _vars.DenseStateSpaceVar, /, vector_field, t, p):
        m0 = self.e0(x.hidden_state.mean)
        m1 = self.e1(x.hidden_state.mean)
        cov_sqrtm_lower = self.e1_vect(x.hidden_state.cov_sqrtm_lower)

        def f_wrapped(s):
            """Evaluate the ODE vector field at (x, Dx, D^2x, ...).

            The time-point and the parameter are fixed.
            If the vector field depends on x and Dx, this wrapper receives the
            stack (x, Dx) as an input (instead of, e.g., a tuple).
            """
            return vector_field(*s, t=t, p=p)

        fx = self.linearise_fn(fn=f_wrapped, m=m0)

        b = m1 - fx
        l_obs_raw = _sqrt_util.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        observed = _vars.DenseNormal(b, l_obs_raw)

        mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros_like(b))
        output_scale = mahalanobis_norm / jnp.sqrt(b.size)
        error_estimate_unscaled = observed.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return error_estimate, output_scale, (b,)

    def complete_correction(self, extrapolated, cache):
        ext = extrapolated  # alias for readability
        l_obs_nonsquare = self.e1_vect(ext.hidden_state.cov_sqrtm_lower)

        # Compute correction according to ext -> obs
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=ext.hidden_state.cov_sqrtm_lower.T
        )

        # Gather observation terms
        (b,) = cache
        observed = _vars.DenseNormal(mean=b, cov_sqrtm_lower=r_obs.T)

        # Gather correction terms
        m_cor = ext.hidden_state.mean - gain @ b
        cor = _vars.DenseNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        _shape = ext.target_shape
        corrected = _vars.DenseStateSpaceVar(cor, cache=None, target_shape=_shape)
        return observed, corrected


@jax.tree_util.register_pytree_node_class
class _DenseTaylorFirstOrder(_collections.AbstractCorrection):
    def __init__(self, ode_shape, ode_order):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        # Selection matrices
        select = functools.partial(_select_derivative, ode_shape=self.ode_shape)
        self.e0 = functools.partial(select, i=slice(0, self.ode_order))
        self.e1 = functools.partial(select, i=self.ode_order)

    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape)

    def begin_correction(self, x: _vars.DenseStateSpaceVar, /, vector_field, t, p):
        def ode_residual(s):
            x0 = self.e0(s)
            x1 = self.e1(s)
            fx0 = vector_field(*x0, t=t, p=p)
            return x1 - fx0

        # Linearise the ODE residual (not the vector field!)
        jvp_fn, (b,) = linearise.ts1(fn=ode_residual, m=x.hidden_state.mean)

        # Evaluate sqrt(cov) -> J @ sqrt(cov)
        jvp_fn_vect = jax.vmap(jvp_fn, in_axes=1, out_axes=1)
        cov_sqrtm_lower = jvp_fn_vect(x.hidden_state.cov_sqrtm_lower)

        # Gather the observed variable
        l_obs_raw = _sqrt_util.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        observed = _vars.DenseNormal(b, l_obs_raw)

        # Extract the output scale and the error estimate
        mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros_like(b))
        output_scale = mahalanobis_norm / jnp.sqrt(b.size)
        error_estimate_unscaled = observed.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return error_estimate, output_scale, (jvp_fn, (b,))

    def complete_correction(self, extrapolated: _vars.DenseStateSpaceVar, cache):
        # Assign short-named variables for readability
        ext = extrapolated

        # Evaluate sqrt(cov) -> J @ sqrt(cov)
        jvp_fn, (b,) = cache
        jvp_fn_vect = jax.vmap(jvp_fn, in_axes=1, out_axes=1)
        l_obs_nonsquare = jvp_fn_vect(ext.hidden_state.cov_sqrtm_lower)

        # Compute the correction matrices
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=ext.hidden_state.cov_sqrtm_lower.T
        )

        # Gather the observed variable
        observed = _vars.DenseNormal(mean=b, cov_sqrtm_lower=r_obs.T)

        # Gather the corrected variable
        m_cor = ext.hidden_state.mean - gain @ b
        rv = _vars.DenseNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        _shape = ext.target_shape
        corrected = _vars.DenseStateSpaceVar(rv, cache=None, target_shape=_shape)

        # Return the results
        return observed, corrected


@jax.tree_util.register_pytree_node_class
class _DenseStatisticalZerothOrder(_collections.AbstractCorrection):
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

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = ()
        aux = self.ode_order, self.ode_shape, self.linearise_fn
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape, linearise_fn = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape, linearise_fn=linearise_fn)

    def begin_correction(self, x: _vars.DenseStateSpaceVar, /, vector_field, t, p):
        # Compute the linearisation point
        m_0 = self.e0(x.hidden_state.mean)
        r_0 = self.e0_vect(x.hidden_state.cov_sqrtm_lower).T
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # todo: higher-order ODEs
        def f_wrapped(s):
            return vector_field(s, t=t, p=p)

        # Apply statistical linear regression to the ODE vector field
        noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)
        cache = (f_wrapped,)

        # Compute the marginal observation
        m_1 = self.e1(x.hidden_state.mean)
        r_1 = self.e1_vect(x.hidden_state.cov_sqrtm_lower).T
        m_marg = m_1 - noise.mean
        l_marg = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(r_1, noise.cov_sqrtm_lower.T)
        ).T
        marginals = _vars.DenseNormal(m_marg, l_marg)

        # Compute output scale and error estimate
        mahalanobis_norm = marginals.mahalanobis_norm(jnp.zeros_like(m_marg))
        output_scale = mahalanobis_norm / jnp.sqrt(m_marg.size)
        error_estimate_unscaled = marginals.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return error_estimate, output_scale, cache

    def complete_correction(self, extrapolated, cache):
        # Select the required derivatives
        _x = extrapolated  # readability in current code block
        m_0 = self.e0(_x.hidden_state.mean)
        m_1 = self.e1(_x.hidden_state.mean)
        r_0 = self.e0_vect(_x.hidden_state.cov_sqrtm_lower).T
        r_1 = self.e1_vect(_x.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # Apply statistical linear regression to the ODE vector field
        f_wrapped, *_ = cache
        noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = extrapolated.hidden_state.cov_sqrtm_lower
        HL = r_1.T
        r_marg, (r_bw, gain) = _sqrt_util.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )
        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - noise.mean
        marginals = _vars.DenseNormal(m_marg, r_marg.T)

        # Compute the corrected mean and gather the correction
        m_bw = extrapolated.hidden_state.mean - gain @ m_marg
        rv = _vars.DenseNormal(m_bw, r_bw.T)
        _shape = extrapolated.target_shape
        corrected = _vars.DenseStateSpaceVar(rv, cache=None, target_shape=_shape)

        # Return the results
        return marginals, corrected


@jax.tree_util.register_pytree_node_class
class _DenseStatisticalFirstOrder(_collections.AbstractCorrection):
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

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = ()
        aux = self.ode_order, self.ode_shape, self.linearise_fn
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape, linearise_fn = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape, linearise_fn=linearise_fn)

    def begin_correction(self, x: _vars.DenseStateSpaceVar, /, vector_field, t, p):
        # Compute the linearisation point
        m_0 = self.e0(x.hidden_state.mean)
        r_0 = self.e0_vect(x.hidden_state.cov_sqrtm_lower).T
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # todo: higher-order ODEs
        def f_wrapped(s):
            return vector_field(s, t=t, p=p)

        # Apply statistical linear regression to the ODE vector field
        linop, noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)
        cache = (f_wrapped,)

        # Compute the marginal observation
        m_1 = self.e1(x.hidden_state.mean)
        r_1 = self.e1_vect(x.hidden_state.cov_sqrtm_lower).T
        m_marg = m_1 - (linop @ m_0 + noise.mean)
        l_marg = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(r_1, r_0_square @ linop.T, noise.cov_sqrtm_lower.T)
        ).T
        marginals = _vars.DenseNormal(m_marg, l_marg)

        # Compute output scale and error estimate
        mahalanobis_norm = marginals.mahalanobis_norm(jnp.zeros_like(m_marg))
        output_scale = mahalanobis_norm / jnp.sqrt(m_marg.size)
        error_estimate_unscaled = marginals.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return error_estimate, output_scale, cache

    def complete_correction(self, extrapolated, cache):
        # Select the required derivatives
        _x = extrapolated  # readability in current code block
        m_0 = self.e0(_x.hidden_state.mean)
        m_1 = self.e1(_x.hidden_state.mean)
        r_0 = self.e0_vect(_x.hidden_state.cov_sqrtm_lower).T
        r_1 = self.e1_vect(_x.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # Apply statistical linear regression to the ODE vector field
        f_wrapped, *_ = cache
        H, noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = extrapolated.hidden_state.cov_sqrtm_lower
        HL = r_1.T - H @ r_0.T
        r_marg, (r_bw, gain) = _sqrt_util.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )

        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - (H @ m_0 + noise.mean)
        marginals = _vars.DenseNormal(m_marg, r_marg.T)

        # Compute the corrected mean and gather the correction
        m_bw = extrapolated.hidden_state.mean - gain @ m_marg
        rv = _vars.DenseNormal(m_bw, r_bw.T)
        _shape = extrapolated.target_shape
        corrected = _vars.DenseStateSpaceVar(rv, cache=None, target_shape=_shape)

        # Return the results
        return marginals, corrected


def _select_derivative_vect(x, i, *, ode_shape):
    def select_fn(s):
        return _select_derivative(s, i, ode_shape=ode_shape)

    select = jax.vmap(select_fn, in_axes=1, out_axes=1)
    return select(x)


def _select_derivative(x, i, *, ode_shape):
    (d,) = ode_shape
    x_reshaped = jnp.reshape(x, (-1, d), order="F")
    return x_reshaped[i, ...]
