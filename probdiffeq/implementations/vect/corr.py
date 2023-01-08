"""Vectorised corrections."""

import jax
import jax.numpy as jnp

from probdiffeq import cubature as cubature_module
from probdiffeq.implementations import _collections, _sqrtm
from probdiffeq.implementations.vect import _vars


@jax.tree_util.register_pytree_node_class
class VectTaylorZerothOrder(_collections.AbstractCorrection):
    def __init__(self, ode_shape, ode_order):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape)

    def begin_correction(self, x: _vars.VectStateSpaceVar, /, vector_field, t, p):
        m0 = self._select_derivative(x.hidden_state.mean, slice(0, self.ode_order))
        m1 = self._select_derivative(x.hidden_state.mean, self.ode_order)
        b = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = self._select_derivative_vect(
            x.hidden_state.cov_sqrtm_lower, 1
        )

        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        observed = _vars.VectNormal(b, l_obs_raw)
        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (b,)

    def complete_correction(self, extrapolated, cache):
        (b,) = cache
        m_ext, l_ext = (
            extrapolated.hidden_state.mean,
            extrapolated.hidden_state.cov_sqrtm_lower,
        )

        l_obs_nonsquare = self._select_derivative_vect(l_ext, self.ode_order)
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=l_ext.T
        )
        m_cor = m_ext - gain @ b

        _shape = extrapolated.target_shape
        observed = _vars.VectNormal(mean=b, cov_sqrtm_lower=r_obs.T)
        rv = _vars.VectNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        corrected = _vars.VectStateSpaceVar(rv, target_shape=_shape)
        return observed, (corrected, gain)

    def _select_derivative_vect(self, x, i):
        select = jax.vmap(
            lambda s: self._select_derivative(s, i), in_axes=1, out_axes=1
        )
        return select(x)

    def _select_derivative(self, x, i):
        (d,) = self.ode_shape
        x_reshaped = jnp.reshape(x, (-1, d), order="F")
        return x_reshaped[i, ...]


@jax.tree_util.register_pytree_node_class
class VectTaylorFirstOrder(_collections.AbstractCorrection):
    def __init__(self, ode_shape, ode_order):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape)

    def begin_correction(self, x: _vars.VectStateSpaceVar, /, vector_field, t, p):
        vf_p = jax.tree_util.Partial(self._res, vector_field=vector_field, t=t, p=p)
        b, fn = jax.linearize(vf_p, x.hidden_state.mean)

        cov_sqrtm_lower = self._cov_sqrtm_lower(
            cache=(b, fn), cov_sqrtm_lower=x.hidden_state.cov_sqrtm_lower
        )

        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        output_scale_sqrtm = _vars.VectNormal(
            b, l_obs_raw
        ).norm_of_whitened_residual_sqrtm()
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (b, fn)

    def complete_correction(self, extrapolated: _vars.VectStateSpaceVar, cache):
        b, _ = cache

        m_ext, l_ext = (
            extrapolated.hidden_state.mean,
            extrapolated.hidden_state.cov_sqrtm_lower,
        )

        l_obs_nonsquare = self._cov_sqrtm_lower(cache=cache, cov_sqrtm_lower=l_ext)

        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=l_ext.T
        )
        m_cor = m_ext - gain @ b

        shape = extrapolated.target_shape
        observed = _vars.VectNormal(mean=b, cov_sqrtm_lower=r_obs.T)
        corrected = _vars.VectNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        return observed, (_vars.VectStateSpaceVar(corrected, target_shape=shape), gain)

    def _cov_sqrtm_lower(self, cache, cov_sqrtm_lower):
        _, fn = cache
        return jax.vmap(fn, in_axes=1, out_axes=1)(cov_sqrtm_lower)

    def _res(self, x, vector_field, t, p):
        x0 = self._select_derivative(x, slice(0, self.ode_order))
        x1 = self._select_derivative(x, self.ode_order)
        fx0 = vector_field(*x0, t=t, p=p)
        return x1 - fx0

    def _select_derivative_vect(self, x, i):
        select = jax.vmap(
            lambda s: self._select_derivative(s, i), in_axes=1, out_axes=1
        )
        return select(x)

    def _select_derivative(self, x, i):
        (d,) = self.ode_shape
        x_reshaped = jnp.reshape(x, (-1, d), order="F")
        return x_reshaped[i, ...]


@jax.tree_util.register_pytree_node_class
class VectMomentMatching(_collections.AbstractCorrection):
    def __init__(self, ode_shape, ode_order, cubature):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape
        self.cubature = cubature

    @classmethod
    def from_params(cls, ode_shape, ode_order):
        sci_fn = cubature_module.ThirdOrderSpherical.from_params
        cubature = sci_fn(input_shape=ode_shape)
        return cls(ode_shape=ode_shape, ode_order=ode_order, cubature=cubature)

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature,)
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature,) = children
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape, cubature=cubature)

    def begin_correction(self, x: _vars.VectStateSpaceVar, /, vector_field, t, p):

        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))
        cache = (vmap_f,)

        # 1. x -> (e0x, e1x)
        R_X = x.hidden_state.cov_sqrtm_lower.T
        L0 = self._select_derivative_vect(R_X.T, 0)
        r_marg1_x = _sqrtm.sqrtm_to_upper_triangular(R=L0.T)
        m_marg1_x = self._select_derivative(x.hidden_state.mean, 0)
        m_marg1_y = self._select_derivative(x.hidden_state.mean, 1)

        # 2. (x, y) -> (f(x), y)
        x_centered = self.cubature.points @ r_marg1_x
        sigma_points = m_marg1_x[None, :] + x_centered
        fx = vmap_f(sigma_points)
        m_marg2 = self.cubature.weights_sqrtm**2 @ fx
        fx_centered = fx - m_marg2[None, :]
        fx_centered_normed = fx_centered * self.cubature.weights_sqrtm[:, None]

        # 3. (x, y) -> y - x (last one)
        m_marg = m_marg1_y - m_marg2
        R1 = self._select_derivative_vect(R_X.T, 1).T
        l_marg = _sqrtm.sum_of_sqrtm_factors(R1=R1, R2=fx_centered_normed).T

        # Summarise
        marginals = _vars.VectNormal(m_marg, l_marg)
        output_scale_sqrtm = marginals.norm_of_whitened_residual_sqrtm()

        # Compute error estimate
        l_obs = marginals.cov_sqrtm_lower
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs, l_obs))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated, cache):
        # The correction step for the cubature Kalman filter implementation
        # is quite complicated. The reason is that the observation model
        # is x -> e1(x) - f(e0(x)), i.e., a composition of a linear/nonlinear/linear
        # model, and that we _only_ want to cubature-linearise the nonlinearity.
        # So what we do is that we compute marginals, gains, and posteriors
        # for each of the three transitions and merge them in the end.
        # This uses the fewest sigma-points possible, and ultimately should
        # lead to the fastest, most stable implementation.

        # Compute the linearisation as in
        # Eq. (9) in https://arxiv.org/abs/2102.00514
        H, noise = self._linearize(x=extrapolated, cache=cache)

        # Compute the CKF correction
        L = extrapolated.hidden_state.cov_sqrtm_lower
        L0 = self._select_derivative_vect(L, 0)
        L1 = self._select_derivative_vect(L, 1)
        HL = L1 - H @ L0
        r_marg, (r_bw, gain) = _sqrtm.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )

        # Catch up the marginals
        x = extrapolated  # alias for readability in this code-block
        x0 = self._select_derivative(x.hidden_state.mean, 0)
        x1 = self._select_derivative(x.hidden_state.mean, 1)
        m_marg = x1 - (H @ x0 + noise.mean)
        shape = extrapolated.target_shape
        marginals = _vars.VectNormal(m_marg, r_marg.T)

        # Catch up the correction and return result
        m_bw = extrapolated.hidden_state.mean - gain @ m_marg
        rv = _vars.VectNormal(m_bw, r_bw.T)
        corrected = _vars.VectStateSpaceVar(rv, target_shape=shape)
        return marginals, (corrected, gain)

    def _linearize(self, x, cache):
        vmap_f, *_ = cache

        # Create sigma points
        m_0 = self._select_derivative(x.hidden_state.mean, 0)
        r_0 = self._select_derivative_vect(x.hidden_state.cov_sqrtm_lower, 0).T
        r_0_square = _sqrtm.sqrtm_to_upper_triangular(R=r_0)
        pts_centered = self.cubature.points @ r_0_square
        pts = m_0[None, :] + pts_centered

        # Evaluate the vector-field
        fx = vmap_f(pts)
        fx_mean = self.cubature.weights_sqrtm**2 @ fx
        fx_centered = fx - fx_mean[None, :]

        # Revert the transition to get H and Omega
        # This is a pure sqrt-implementation of
        # Eq. (9) in https://arxiv.org/abs/2102.00514
        # It seems to be different to Section VI.B in
        # https://arxiv.org/pdf/2207.00426.pdf,
        # because the implementation below avoids sqrt-down-dates
        pts_centered_normed = pts_centered * self.cubature.weights_sqrtm[:, None]
        fx_centered_normed = fx_centered * self.cubature.weights_sqrtm[:, None]
        # todo: with R_X_F = r_0_square, we would save a qr decomposition, right?
        #  (but would it still be valid?)
        _, (r_Om, H) = _sqrtm.revert_conditional_noisefree(
            R_X_F=pts_centered_normed, R_X=fx_centered_normed
        )

        # Catch up the transition-mean and return the result
        d = fx_mean - H @ m_0
        return H, _vars.VectNormal(d, r_Om.T)

    def _select_derivative_vect(self, x, i):
        select = jax.vmap(
            lambda s: self._select_derivative(s, i), in_axes=1, out_axes=1
        )
        return select(x)

    def _select_derivative(self, x, i):
        (d,) = self.ode_shape
        x_reshaped = jnp.reshape(x, (-1, d), order="F")
        return x_reshaped[i, ...]


@jax.tree_util.register_pytree_node_class
class VectMomentMatchingZerothOrder(_collections.AbstractCorrection):
    """Zeroth-order moment matching."""

    def __init__(self, ode_shape, ode_order, cubature):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape
        self.cubature = cubature

    @classmethod
    def from_params(cls, ode_shape, ode_order):
        sci_fn = cubature_module.ThirdOrderSpherical.from_params
        cubature = sci_fn(input_shape=ode_shape)
        return cls(ode_shape=ode_shape, ode_order=ode_order, cubature=cubature)

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature,)
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature,) = children
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape, cubature=cubature)

    def begin_correction(self, x: _vars.VectStateSpaceVar, /, vector_field, t, p):
        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))
        cache = (vmap_f,)

        # Construct sigma-points
        m_0 = self._select_derivative(x.hidden_state.mean, 0)
        r_0 = self._select_derivative_vect(x.hidden_state.cov_sqrtm_lower, 0).T
        r_0_square = _sqrtm.sqrtm_to_upper_triangular(R=r_0)
        pts_centered = self.cubature.points @ r_0_square
        pts = m_0[None, :] + pts_centered

        # Evaluate the vector-field
        fx = vmap_f(pts)

        # Compute zeroth order approximation
        fx_mean = self.cubature.weights_sqrtm**2 @ fx

        # Complete estimation
        m1 = self._select_derivative(x.hidden_state.mean, self.ode_order)
        b = m1 - fx_mean
        cov_sqrtm_lower = self._select_derivative_vect(
            x.hidden_state.cov_sqrtm_lower, 1
        )

        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        observed = _vars.VectNormal(b, l_obs_raw)
        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated, cache):
        vmap_f, *_ = cache
        m_ext, l_ext = (
            extrapolated.hidden_state.mean,
            extrapolated.hidden_state.cov_sqrtm_lower,
        )

        # Create sigma points
        m_0 = self._select_derivative(m_ext, 0)
        r_0 = self._select_derivative_vect(l_ext, 0).T
        r_0_square = _sqrtm.sqrtm_to_upper_triangular(R=r_0)
        pts_centered = self.cubature.points @ r_0_square
        pts = m_0[None, :] + pts_centered

        # Evaluate the vector-field
        fx = vmap_f(pts)

        # Compute zeroth order approximation
        m_1 = self._select_derivative(m_ext, 1)
        b = m_1 - self.cubature.weights_sqrtm**2 @ fx

        # Complete correction
        l_obs_nonsquare = self._select_derivative_vect(l_ext, self.ode_order)
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=l_ext.T
        )
        m_cor = m_ext - gain @ b

        _shape = extrapolated.target_shape
        observed = _vars.VectNormal(mean=b, cov_sqrtm_lower=r_obs.T)
        rv = _vars.VectNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        corrected = _vars.VectStateSpaceVar(rv, target_shape=_shape)
        return observed, (corrected, gain)

    def _select_derivative_vect(self, x, i):
        select = jax.vmap(
            lambda s: self._select_derivative(s, i), in_axes=1, out_axes=1
        )
        return select(x)

    def _select_derivative(self, x, i):
        (d,) = self.ode_shape
        x_reshaped = jnp.reshape(x, (-1, d), order="F")
        return x_reshaped[i, ...]
