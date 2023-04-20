"""Corrections."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _corr, cubature
from probdiffeq.statespace.dense import _vars, linearise


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
class _DenseTaylorZerothOrder(_corr.Correction):
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

    def __repr__(self):
        return f"<TS0 with ode_order={self.ode_order}>"

    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape)

    def init(self, ssv, /):
        m_like = jnp.zeros(self.ode_shape)
        chol_like = jnp.zeros(self.ode_shape + self.ode_shape)
        obs_like = _vars.DenseNormal(m_like, chol_like, target_shape=None)
        return ssv, obs_like

    def begin(self, ssv: _vars.DenseSSV, corr, /, vector_field, t, p):
        m0 = self.e0(ssv.hidden_state.mean)
        m1 = self.e1(ssv.hidden_state.mean)
        cov_sqrtm_lower = self.e1_vect(ssv.hidden_state.cov_sqrtm_lower)

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
        observed = _vars.DenseNormal(b, l_obs_raw, target_shape=None)

        mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros_like(b))
        output_scale = mahalanobis_norm / jnp.sqrt(b.size)
        error_estimate_unscaled = observed.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return ssv, (error_estimate, output_scale, (b,))

    def complete(self, ssv, corr, /, vector_field, t, p):
        l_obs_nonsquare = self.e1_vect(ssv.hidden_state.cov_sqrtm_lower)

        # Compute correction according to ext -> obs
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=ssv.hidden_state.cov_sqrtm_lower.T
        )

        # Gather observation terms
        *_, (b,) = corr
        observed = _vars.DenseNormal(mean=b, cov_sqrtm_lower=r_obs.T, target_shape=None)

        # Gather correction terms
        m_cor = ssv.hidden_state.mean - gain @ b
        cor = _vars.DenseNormal(
            mean=m_cor, cov_sqrtm_lower=r_cor.T, target_shape=ssv.target_shape
        )
        u = m_cor.reshape(ssv.target_shape, order="F")[0, :]
        ssv = _vars.DenseSSV(u, cor, target_shape=ssv.target_shape)
        return ssv, observed

    def extract(self, ssv, corr, /):
        return ssv


@jax.tree_util.register_pytree_node_class
class _DenseTaylorFirstOrder(_corr.Correction):
    def __init__(self, ode_shape, ode_order):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        # Selection matrices
        select = functools.partial(_select_derivative, ode_shape=self.ode_shape)
        self.e0 = functools.partial(select, i=slice(0, self.ode_order))
        self.e1 = functools.partial(select, i=self.ode_order)

    def __repr__(self):
        return f"<TS1 with ode_order={self.ode_order}>"

    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape)

    def init(self, ssv, /):
        m_like = jnp.zeros(self.ode_shape)
        chol_like = jnp.zeros(self.ode_shape + self.ode_shape)
        obs_like = _vars.DenseNormal(m_like, chol_like, target_shape=None)
        return ssv, obs_like

    def begin(self, ssv: _vars.DenseSSV, corr, /, vector_field, t, p):
        def ode_residual(s):
            x0 = self.e0(s)
            x1 = self.e1(s)
            fx0 = vector_field(*x0, t=t, p=p)
            return x1 - fx0

        # Linearise the ODE residual (not the vector field!)
        jvp_fn, (b,) = linearise.ts1(fn=ode_residual, m=ssv.hidden_state.mean)

        # Evaluate sqrt(cov) -> J @ sqrt(cov)
        jvp_fn_vect = jax.vmap(jvp_fn, in_axes=1, out_axes=1)
        cov_sqrtm_lower = jvp_fn_vect(ssv.hidden_state.cov_sqrtm_lower)

        # Gather the observed variable
        l_obs_raw = _sqrt_util.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        observed = _vars.DenseNormal(b, l_obs_raw, target_shape=None)

        # Extract the output scale and the error estimate
        mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros_like(b))
        output_scale = mahalanobis_norm / jnp.sqrt(b.size)
        error_estimate_unscaled = observed.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return ssv, (error_estimate, output_scale, (jvp_fn, (b,)))

    def complete(self, ssv: _vars.DenseSSV, corr, /, vector_field, t, p):
        # Evaluate sqrt(cov) -> J @ sqrt(cov)
        *_, (jvp_fn, (b,)) = corr
        jvp_fn_vect = jax.vmap(jvp_fn, in_axes=1, out_axes=1)
        l_obs_nonsquare = jvp_fn_vect(ssv.hidden_state.cov_sqrtm_lower)

        # Compute the correction matrices
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=ssv.hidden_state.cov_sqrtm_lower.T
        )

        # Gather the observed variable
        observed = _vars.DenseNormal(mean=b, cov_sqrtm_lower=r_obs.T, target_shape=None)

        # Gather the corrected variable
        m_cor = ssv.hidden_state.mean - gain @ b
        rv = _vars.DenseNormal(
            mean=m_cor, cov_sqrtm_lower=r_cor.T, target_shape=ssv.target_shape
        )
        u = m_cor.reshape(ssv.target_shape, order="F")[0, :]
        ssv = _vars.DenseSSV(u, rv, target_shape=ssv.target_shape)
        return ssv, observed

    def extract(self, ssv, _corr, /):
        return ssv


@jax.tree_util.register_pytree_node_class
class _DenseStatisticalZerothOrder(_corr.Correction):
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
        obs_like = _vars.DenseNormal(m_like, chol_like, target_shape=None)
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

    def begin(self, ssv: _vars.DenseSSV, corr, /, vector_field, t, p):
        # Compute the linearisation point
        m_0 = self.e0(ssv.hidden_state.mean)
        r_0 = self.e0_vect(ssv.hidden_state.cov_sqrtm_lower).T
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T, target_shape=ssv.target_shape)

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
        marginals = _vars.DenseNormal(m_marg, l_marg, target_shape=None)

        # Compute output scale and error estimate
        mahalanobis_norm = marginals.mahalanobis_norm(jnp.zeros_like(m_marg))
        output_scale = mahalanobis_norm / jnp.sqrt(m_marg.size)
        error_estimate_unscaled = marginals.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return ssv, (error_estimate, output_scale, cache)

    def complete(self, ssv, corr, /, vector_field, t, p):
        # Select the required derivatives
        m_0 = self.e0(ssv.hidden_state.mean)
        m_1 = self.e1(ssv.hidden_state.mean)
        r_0 = self.e0_vect(ssv.hidden_state.cov_sqrtm_lower).T
        r_1 = self.e1_vect(ssv.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T, target_shape=ssv.target_shape)

        # Apply statistical linear regression to the ODE vector field
        *_, (f_wrapped, *_) = corr
        noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = ssv.hidden_state.cov_sqrtm_lower
        HL = r_1.T
        r_marg, (r_bw, gain) = _sqrt_util.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )
        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - noise.mean
        marginals = _vars.DenseNormal(m_marg, r_marg.T, target_shape=None)

        # Compute the corrected mean and gather the correction
        m_bw = ssv.hidden_state.mean - gain @ m_marg
        rv = _vars.DenseNormal(m_bw, r_bw.T, target_shape=ssv.target_shape)
        u = m_bw.reshape(ssv.target_shape, order="F")[0, :]
        corrected = _vars.DenseSSV(u, rv, target_shape=ssv.target_shape)
        return corrected, marginals

    def extract(self, ssv, corr, /):
        return ssv


@jax.tree_util.register_pytree_node_class
class _DenseStatisticalFirstOrder(_corr.Correction):
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
        obs_like = _vars.DenseNormal(m_like, chol_like, target_shape=None)
        return ssv, obs_like

    def begin(self, ssv: _vars.DenseSSV, corr, /, vector_field, t, p):
        # Compute the linearisation point
        m_0 = self.e0(ssv.hidden_state.mean)
        r_0 = self.e0_vect(ssv.hidden_state.cov_sqrtm_lower).T
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T, target_shape=ssv.target_shape)

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
        marginals = _vars.DenseNormal(m_marg, l_marg, target_shape=None)

        # Compute output scale and error estimate
        mahalanobis_norm = marginals.mahalanobis_norm(jnp.zeros_like(m_marg))
        output_scale = mahalanobis_norm / jnp.sqrt(m_marg.size)
        error_estimate_unscaled = marginals.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled

        # Return scaled error estimate and other quantities
        return ssv, (error_estimate, output_scale, cache)

    def complete(self, ssv, corr, /, vector_field, t, p):
        # Select the required derivatives
        m_0 = self.e0(ssv.hidden_state.mean)
        m_1 = self.e1(ssv.hidden_state.mean)
        r_0 = self.e0_vect(ssv.hidden_state.cov_sqrtm_lower).T
        r_1 = self.e1_vect(ssv.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T, target_shape=ssv.target_shape)

        # Apply statistical linear regression to the ODE vector field
        *_, (f_wrapped, *_) = corr
        H, noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = ssv.hidden_state.cov_sqrtm_lower
        HL = r_1.T - H @ r_0.T
        r_marg, (r_bw, gain) = _sqrt_util.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )

        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - (H @ m_0 + noise.mean)
        marginals = _vars.DenseNormal(m_marg, r_marg.T, target_shape=None)

        # Compute the corrected mean and gather the correction
        m_bw = ssv.hidden_state.mean - gain @ m_marg
        rv = _vars.DenseNormal(m_bw, r_bw.T, target_shape=ssv.target_shape)
        u = m_bw.reshape(ssv.target_shape, order="F")[0, :]
        corrected = _vars.DenseSSV(u, rv, target_shape=ssv.target_shape)

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
