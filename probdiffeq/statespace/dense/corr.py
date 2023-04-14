"""Corrections."""
import abc
import functools

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _collections, cubature
from probdiffeq.statespace.dense import _vars, linearise


def taylor_order_zero(*args, **kwargs):
    return _DenseTaylorZerothOrder(*args, linearise_fn=linearise.ts0, **kwargs)


def taylor_order_one(*args, **kwargs):
    return _DenseTaylorFirstOrder(*args, linearise_fn=linearise.ts1, **kwargs)


def statistical_order_zero(
    ode_shape,
    ode_order,
    cubature_rule_fn=cubature.unscented_transform,
):
    if ode_order > 1:
        raise ValueError

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
    cubature_rule_fn=cubature.unscented_transform,
):
    if ode_order > 1:
        raise ValueError

    cubature_rule = cubature_rule_fn(input_shape=ode_shape)
    linearise_fn = functools.partial(linearise.slr1, cubature_rule=cubature_rule)
    return _DenseStatisticalFirstOrder(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fn=linearise_fn,
    )


class _DenseCorrection(_collections.AbstractCorrection, abc.ABC):
    def __init__(self, ode_shape, ode_order, linearise_fn):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        self.linearise_fn = linearise_fn

        # Selection matrices
        select = functools.partial(_select_derivative, ode_shape=self.ode_shape)
        select_vect = functools.partial(
            _select_derivative_vect, ode_shape=self.ode_shape
        )
        self.e0 = functools.partial(select, i=slice(0, self.ode_order))
        self.e1 = functools.partial(select, i=self.ode_order)
        self.e0_vect = functools.partial(select_vect, i=slice(0, self.ode_order))
        self.e1_vect = functools.partial(select_vect, i=self.ode_order)

    def __repr__(self):
        name = self.__class__.__name__
        return f"<{name} with ode_order={self.ode_order}>"

    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_shape, self.linearise_fn
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape, linearise_fn = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape, linearise_fn=linearise_fn)

    def init(self, x, /):
        m_like = jnp.zeros(self.ode_shape)
        cholesky_like = jnp.zeros(self.ode_shape + self.ode_shape)
        observed_like = _vars.DenseNormal(mean=m_like, cov_sqrtm_lower=cholesky_like)
        error_estimate = jnp.zeros(self.ode_shape)
        return _vars.DenseSSV(
            observed_state=observed_like,
            error_estimate=error_estimate,
            hidden_state=x.hidden_state,
            hidden_shape=x.hidden_shape,
            cache_extra=x.cache_extra,
            backward_model=x.backward_model,
            output_scale_dynamic=None,
            cache_corr=None,
        )

    @abc.abstractmethod
    def begin(self, x: _vars.DenseSSV, /, vector_field, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, x: _vars.DenseSSV, /, vector_field, t, p):
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class _DenseTaylorZerothOrder(_DenseCorrection):
    def begin(self, x: _vars.DenseSSV, /, vector_field, t, p):
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

        return _vars.DenseSSV(
            output_scale_dynamic=output_scale,
            error_estimate=error_estimate,
            cache_corr=(b,),
            hidden_state=x.hidden_state,
            hidden_shape=x.hidden_shape,
            cache_extra=x.cache_extra,
            backward_model=x.backward_model,
            observed_state=None,
        )

    def complete(self, x: _vars.DenseSSV, /, vector_field, t, p):
        l_obs_nonsquare = self.e1_vect(x.hidden_state.cov_sqrtm_lower)

        # Compute correction according to ext -> obs
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=x.hidden_state.cov_sqrtm_lower.T
        )

        # Gather observation terms
        (b,) = x.cache_corr
        observed = _vars.DenseNormal(mean=b, cov_sqrtm_lower=r_obs.T)

        # Gather correction terms
        m_cor = x.hidden_state.mean - gain @ b
        corrected = _vars.DenseNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)

        return _vars.DenseSSV(
            corrected,
            observed_state=observed,
            hidden_shape=x.hidden_shape,
            error_estimate=x.error_estimate,
            backward_model=x.backward_model,
            output_scale_dynamic=None,
            cache_extra=None,
            cache_corr=None,
        )


@jax.tree_util.register_pytree_node_class
class _DenseTaylorFirstOrder(_DenseCorrection):
    def begin(self, x: _vars.DenseSSV, /, vector_field, t, p):
        def ode_residual(s):
            x0 = self.e0(s)
            x1 = self.e1(s)
            fx0 = vector_field(*x0, t=t, p=p)
            return x1 - fx0

        # Linearise the ODE residual (not the vector field!)
        # todo: can we pass jvp_fn around in a cache?
        #  If not, we have to rethink the line below.:
        jvp_fn, (b,) = self.linearise_fn(fn=ode_residual, m=x.hidden_state.mean)

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

        return _vars.DenseSSV(
            output_scale_dynamic=output_scale,
            error_estimate=error_estimate,
            cache_corr=(jvp_fn, (b,)),
            hidden_state=x.hidden_state,
            hidden_shape=x.hidden_shape,
            cache_extra=x.cache_extra,
            backward_model=x.backward_model,
            observed_state=None,
        )

    def complete(self, x: _vars.DenseSSV, /, vector_field, t, p):
        # Evaluate sqrt(cov) -> J @ sqrt(cov)
        jvp_fn, (b,) = x.cache_corr

        jvp_fn_vect = jax.vmap(jvp_fn, in_axes=1, out_axes=1)
        l_obs_nonsquare = jvp_fn_vect(x.hidden_state.cov_sqrtm_lower)

        # Compute the correction matrices
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=x.hidden_state.cov_sqrtm_lower.T
        )

        # Gather the observed variable
        observed = _vars.DenseNormal(mean=b, cov_sqrtm_lower=r_obs.T)

        # Gather the corrected variable
        m_cor = x.hidden_state.mean - gain @ b
        corrected = _vars.DenseNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)

        return _vars.DenseSSV(
            corrected,
            observed_state=observed,
            hidden_shape=x.hidden_shape,
            error_estimate=x.error_estimate,
            backward_model=x.backward_model,
            output_scale_dynamic=None,
            cache_extra=None,
            cache_corr=None,
        )


@jax.tree_util.register_pytree_node_class
class _DenseStatisticalZerothOrder(_DenseCorrection):
    """Zeroth-order statistical linear regression in state-space models \
     with dense covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def begin(self, x: _vars.DenseSSV, /, vector_field, t, p):
        # Compute the linearisation point
        m_0 = self.e0(x.hidden_state.mean)[0]
        r_0 = self.e0_vect(x.hidden_state.cov_sqrtm_lower)[0]
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # todo: higher-order ODEs
        def f_wrapped(s):
            return vector_field(s, t=t, p=p)

        # Apply statistical linear regression to the ODE vector field
        noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

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

        return _vars.DenseSSV(
            output_scale_dynamic=output_scale,
            error_estimate=error_estimate,
            hidden_state=x.hidden_state,
            hidden_shape=x.hidden_shape,
            cache_extra=x.cache_extra,
            backward_model=x.backward_model,
            cache_corr=None,
            observed_state=None,
        )

    def complete(self, x, /, vector_field, t, p):
        # Select the required derivatives
        m_0 = self.e0(x.hidden_state.mean)[0]
        r_0 = self.e0_vect(x.hidden_state.cov_sqrtm_lower)[0]
        m_1 = self.e1(x.hidden_state.mean)
        r_1 = self.e1_vect(x.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # Apply statistical linear regression to the ODE vector field
        # todo: higher-order ODEs
        def f_wrapped(s):
            return vector_field(s, t=t, p=p)

        noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = x.hidden_state.cov_sqrtm_lower
        HL = r_1.T
        r_marg, (r_bw, gain) = _sqrt_util.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )
        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - noise.mean
        observed = _vars.DenseNormal(m_marg, r_marg.T)

        # Compute the corrected mean and gather the correction
        m_bw = x.hidden_state.mean - gain @ m_marg
        corrected = _vars.DenseNormal(m_bw, r_bw.T)
        return _vars.DenseSSV(
            corrected,
            observed_state=observed,
            hidden_shape=x.hidden_shape,
            error_estimate=x.error_estimate,
            backward_model=x.backward_model,
            output_scale_dynamic=None,
            cache_extra=None,
            cache_corr=None,
        )


@jax.tree_util.register_pytree_node_class
class _DenseStatisticalFirstOrder(_DenseCorrection):
    def begin(self, x: _vars.DenseSSV, /, vector_field, t, p):
        # Compute the linearisation point
        m_0 = self.e0(x.hidden_state.mean)[0]  # only first-order ODEs
        r_0 = self.e0_vect(x.hidden_state.cov_sqrtm_lower)[0]
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # todo: higher-order ODEs
        def f_wrapped(s):
            return vector_field(s, t=t, p=p)

        # Apply statistical linear regression to the ODE vector field
        linop, noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

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

        return _vars.DenseSSV(
            output_scale_dynamic=output_scale,
            error_estimate=error_estimate,
            hidden_state=x.hidden_state,
            hidden_shape=x.hidden_shape,
            cache_extra=x.cache_extra,
            backward_model=x.backward_model,
            cache_corr=None,
            observed_state=None,
        )

    def complete(self, x, /, vector_field, t, p):
        # todo: higher-order ODEs
        def f_wrapped(s):
            return vector_field(s, t=t, p=p)

        # Select the required derivatives
        m_0 = self.e0(x.hidden_state.mean)[0]
        r_0 = self.e0_vect(x.hidden_state.cov_sqrtm_lower)[0]
        m_1 = self.e1(x.hidden_state.mean)
        r_1 = self.e1_vect(x.hidden_state.cov_sqrtm_lower).T

        # Extract the linearisation point
        r_0_square = _sqrt_util.sqrtm_to_upper_triangular(R=r_0)
        lin_pt = _vars.DenseNormal(m_0, r_0_square.T)

        # Apply statistical linear regression to the ODE vector field
        H, noise = self.linearise_fn(fn=f_wrapped, x=lin_pt)

        # Compute the sigma-point correction of the ODE residual
        L = x.hidden_state.cov_sqrtm_lower
        HL = r_1.T - H @ r_0.T
        r_marg, (r_bw, gain) = _sqrt_util.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )

        # Compute the marginal mean and gather the marginals
        m_marg = m_1 - (H @ m_0 + noise.mean)
        observed = _vars.DenseNormal(m_marg, r_marg.T)

        # Compute the corrected mean and gather the correction
        m_bw = x.hidden_state.mean - gain @ m_marg
        corrected = _vars.DenseNormal(m_bw, r_bw.T)

        return _vars.DenseSSV(
            corrected,
            observed_state=observed,
            hidden_shape=x.hidden_shape,
            error_estimate=x.error_estimate,
            backward_model=x.backward_model,
            output_scale_dynamic=None,
            cache_extra=None,
            cache_corr=None,
        )


def _select_derivative_vect(x, i, *, ode_shape):
    def select_fn(s):
        return _select_derivative(s, i, ode_shape=ode_shape)

    select = jax.vmap(select_fn, in_axes=1, out_axes=1)
    return select(x)


def _select_derivative(x, i, *, ode_shape):
    (d,) = ode_shape
    x_reshaped = jnp.reshape(x, (-1, d), order="F")
    return x_reshaped[i, ...]
