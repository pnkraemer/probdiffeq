"""Corrections."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq import cubature as cubature_module
from probdiffeq.implementations import _collections, _sqrtm
from probdiffeq.implementations.dense import _vars

# todo:
#  statistical linear regression (zeroth order)
#  statistical linear regression (cov-free)
#  statistical linear regression (Jacobian)
#  statistical linear regression (Bayesian cubature)


def linearise_ts0(*, fn, m):
    """Linearise a function with a zeroth-order Taylor series."""
    return fn(m)


def linearise_ts1(*, fn, m):
    """Linearise a function with a first-order Taylor series."""
    b, jvp_fn = jax.linearize(fn, m)
    return jvp_fn, (b,)


def linearise_slr1(*, fn, x, cubature_rule):
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
    _, (cov_sqrtm_cond, linop_cond) = _sqrtm.revert_conditional_noisefree(
        R_X_F=pts_centered_normed, R_X=fx_centered_normed
    )
    mean_cond = fx_mean - linop_cond @ x.mean
    return linop_cond, _vars.DenseNormal(mean_cond, cov_sqrtm_cond.T)


def linearise_slr0(*, fn, x, cubature_rule):
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
    return _vars.DenseNormal(fx_mean, fx_centered_normed.T)


@jax.tree_util.register_pytree_node_class
class DenseTaylorZerothOrder(_collections.AbstractCorrection):
    def __init__(self, ode_shape, ode_order):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        # Turn this into an argument if other linearisation functions apply
        self.linearise_fn = linearise_ts0

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
        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        observed = _vars.DenseNormal(b, l_obs_raw)

        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (b,)

    def complete_correction(self, extrapolated, cache):
        ext = extrapolated  # alias for readability
        l_obs_nonsquare = self.e1_vect(ext.hidden_state.cov_sqrtm_lower)

        # Compute correction according to ext -> obs
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=ext.hidden_state.cov_sqrtm_lower.T
        )

        # Gather observation terms
        (b,) = cache
        observed = _vars.DenseNormal(mean=b, cov_sqrtm_lower=r_obs.T)

        # Gather correction terms
        m_cor = ext.hidden_state.mean - gain @ b
        cor = _vars.DenseNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        corrected = _vars.DenseStateSpaceVar(cor, target_shape=ext.target_shape)
        return observed, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class DenseTaylorFirstOrder(_collections.AbstractCorrection):
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
        jvp_fn, (b,) = linearise_ts1(fn=ode_residual, m=x.hidden_state.mean)

        # Evaluate sqrt(cov) -> J @ sqrt(cov)
        jvp_fn_vect = jax.vmap(jvp_fn, in_axes=1, out_axes=1)
        cov_sqrtm_lower = jvp_fn_vect(x.hidden_state.cov_sqrtm_lower)

        # Gather the observed variable
        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        obs = _vars.DenseNormal(b, l_obs_raw)

        # Extract the output scale and the error estimate
        output_scale_sqrtm = obs.norm_of_whitened_residual_sqrtm()
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))

        # Return the scaled error estimate and the other quantities
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (jvp_fn, (b,))

    def complete_correction(self, extrapolated: _vars.DenseStateSpaceVar, cache):
        # Assign short-named variables for readability
        ext = extrapolated

        # Evaluate sqrt(cov) -> J @ sqrt(cov)
        jvp_fn, (b,) = cache
        jvp_fn_vect = jax.vmap(jvp_fn, in_axes=1, out_axes=1)
        l_obs_nonsquare = jvp_fn_vect(ext.hidden_state.cov_sqrtm_lower)

        # Compute the correction matrices
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=ext.hidden_state.cov_sqrtm_lower.T
        )

        # Gather the observed variable
        observed = _vars.DenseNormal(mean=b, cov_sqrtm_lower=r_obs.T)

        # Gather the corrected variable
        m_cor = ext.hidden_state.mean - gain @ b
        rv = _vars.DenseNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        corrected = _vars.DenseStateSpaceVar(rv, target_shape=ext.target_shape)

        # Return the results
        return observed, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class DenseStatisticalZerothOrder(_collections.AbstractCorrection):
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

    @classmethod
    def from_params(cls, ode_shape, ode_order, cubature=None):
        if cubature is None:
            make_rule_fn = cubature_module.ThirdOrderSpherical.from_params
            cubature = make_rule_fn(input_shape=ode_shape)

        linearise_fn = functools.partial(linearise_slr0, cubature_rule=cubature)
        return cls(ode_shape=ode_shape, ode_order=ode_order, linearise_fn=linearise_fn)

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
        r_0_square = _sqrtm.sqrtm_to_upper_triangular(R=r_0)
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
        l_marg = _sqrtm.sum_of_sqrtm_factors(R_stack=(r_1, noise.cov_sqrtm_lower.T)).T
        marginals = _vars.DenseNormal(m_marg, l_marg)

        # Compute output scale and error estimate
        output_scale_sqrtm = marginals.norm_of_whitened_residual_sqrtm()
        l_obs = marginals.cov_sqrtm_lower
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs, l_obs))

        # Return scaled error estimate and other quantities
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated, cache):
        # Select the required derivatives
        _x = extrapolated  # readability in current code block
        m_0 = self.e0(_x.hidden_state.mean)
        m_1 = self.e1(_x.hidden_state.mean)
        r_0 = self.e0_vect(_x.hidden_state.cov_sqrtm_lower).T
        r_1 = self.e1_vect(_x.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrtm.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # Apply statistical linear regression to the ODE vector field
        f_wrapped, *_ = cache
        noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = extrapolated.hidden_state.cov_sqrtm_lower
        HL = r_1.T
        r_marg, (r_bw, gain) = _sqrtm.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )
        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - noise.mean
        marginals = _vars.DenseNormal(m_marg, r_marg.T)

        # Compute the corrected mean and gather the correction
        m_bw = extrapolated.hidden_state.mean - gain @ m_marg
        rv = _vars.DenseNormal(m_bw, r_bw.T)
        _shape = extrapolated.target_shape
        corrected = _vars.DenseStateSpaceVar(rv, target_shape=_shape)

        # Return the results
        return marginals, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class DenseStatisticalFirstOrder(_collections.AbstractCorrection):
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

    @classmethod
    def from_params(cls, ode_shape, ode_order, cubature=None):
        if cubature is None:
            make_rule_fn = cubature_module.ThirdOrderSpherical.from_params
            cubature = make_rule_fn(input_shape=ode_shape)

        linearise_fn = functools.partial(linearise_slr1, cubature_rule=cubature)
        return cls(ode_shape=ode_shape, ode_order=ode_order, linearise_fn=linearise_fn)

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
        r_0_square = _sqrtm.sqrtm_to_upper_triangular(R=r_0)
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
        l_marg = _sqrtm.sum_of_sqrtm_factors(
            R_stack=(r_1, r_0_square @ linop.T, noise.cov_sqrtm_lower.T)
        ).T
        marginals = _vars.DenseNormal(m_marg, l_marg)

        # Compute output scale and error estimate
        output_scale_sqrtm = marginals.norm_of_whitened_residual_sqrtm()
        l_obs = marginals.cov_sqrtm_lower
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs, l_obs))

        # Return scaled error estimate and other quantities
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated, cache):
        # Select the required derivatives
        _x = extrapolated  # readability in current code block
        m_0 = self.e0(_x.hidden_state.mean)
        m_1 = self.e1(_x.hidden_state.mean)
        r_0 = self.e0_vect(_x.hidden_state.cov_sqrtm_lower).T
        r_1 = self.e1_vect(_x.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrtm.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # Apply statistical linear regression to the ODE vector field
        f_wrapped, *_ = cache
        H, noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = extrapolated.hidden_state.cov_sqrtm_lower
        HL = r_1.T - H @ r_0.T
        r_marg, (r_bw, gain) = _sqrtm.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )

        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - (H @ m_0 + noise.mean)
        marginals = _vars.DenseNormal(m_marg, r_marg.T)

        # Compute the corrected mean and gather the correction
        m_bw = extrapolated.hidden_state.mean - gain @ m_marg
        rv = _vars.DenseNormal(m_bw, r_bw.T)
        _shape = extrapolated.target_shape
        corrected = _vars.DenseStateSpaceVar(rv, target_shape=_shape)

        # Return the results
        return marginals, (corrected, gain)


def _select_derivative_vect(x, i, *, ode_shape):
    def select_fn(s):
        return _select_derivative(s, i, ode_shape=ode_shape)

    select = jax.vmap(select_fn, in_axes=1, out_axes=1)
    return select(x)


def _select_derivative(x, i, *, ode_shape):
    (d,) = ode_shape
    x_reshaped = jnp.reshape(x, (-1, d), order="F")
    return x_reshaped[i, ...]
