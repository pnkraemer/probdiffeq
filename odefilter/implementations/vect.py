"""State-space models with dense covariance structure   ."""
import functools

import jax
import jax.numpy as jnp

from odefilter import cubature as cubature_module
from odefilter.implementations import _collections, _ibm_util, _sqrtm


@jax.tree_util.register_pytree_node_class
class VectNormal(_collections.StateSpaceVariable):
    """Vector-normal distribution.

    You can think of this as a traditional multivariate normal distribution.
    But in fact, it is more of a matrix-normal distribution.
    This means that the mean vector is a (d*n,)-shaped array but
    represents a (d,n)-shaped matrix.
    """

    def __init__(self, mean, cov_sqrtm_lower, target_shape):
        self.mean = mean  # (n,) shape
        self.cov_sqrtm_lower = cov_sqrtm_lower  # (n, n) shape
        self.target_shape = target_shape

    def tree_flatten(self):
        children = self.mean, self.cov_sqrtm_lower
        aux = (self.target_shape,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        mean, cov_sqrtm_lower = children
        (target_shape,) = aux
        return cls(mean, cov_sqrtm_lower, target_shape=target_shape)

    # todo: extract _whiten() method?!

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower

        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, (m_obs - u), lower=False)

        x1 = jnp.dot(res_white, res_white.T)
        x2 = jnp.linalg.slogdet(l_obs)[1] ** 2
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, obs_pt, lower=False)
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def condition_on_qoi_observation(self, u, /, observation_std):
        hc = self._select_derivative_vect(self.cov_sqrtm_lower, 0)
        m_obs = self._select_derivative(self.mean, 0)

        r_yx = observation_std * jnp.eye(u.shape[0])
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=self.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = self.mean - gain @ (m_obs - u)

        obs = VectNormal(m_obs, r_obs.T, target_shape=self.target_shape)
        cor = VectNormal(m_cor, r_cor.T, target_shape=self.target_shape)
        return obs, (cor, gain)

    def extract_qoi(self):
        if self.mean.ndim == 1:
            return self._select_derivative(self.mean, i=0)
        return jax.vmap(self._select_derivative, in_axes=(0, None))(self.mean, 0)

    def extract_qoi_from_sample(self, u, /):
        if u.ndim == 1:
            return u.reshape(self.target_shape, order="F")[0, ...]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def _select_derivative_vect(self, x, i):
        fn = functools.partial(self._select_derivative, i=i)
        select = jax.vmap(fn, in_axes=1, out_axes=1)
        return select(x)

    def _select_derivative(self, x, i):
        x_reshaped = jnp.reshape(x, self.target_shape, order="F")
        return x_reshaped[i, ...]

    def scale_covariance(self, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return VectNormal(
                mean=self.mean,
                cov_sqrtm_lower=scale_sqrtm * self.cov_sqrtm_lower,
                target_shape=self.target_shape,
            )
        return VectNormal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * self.cov_sqrtm_lower,
            target_shape=self.target_shape,
        )

    # automatically batched because of numpy's broadcasting rules?
    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    def Ax_plus_y(self, A, x, y):
        return A @ x + y

    @property
    def sample_shape(self):
        return self.mean.shape


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

    def begin_correction(self, x: VectNormal, /, vector_field, t, p):
        m0 = self._select_derivative(x.mean, slice(0, self.ode_order))
        m1 = self._select_derivative(x.mean, self.ode_order)
        b = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = self._select_derivative_vect(x.cov_sqrtm_lower, 1)

        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        observed = VectNormal(b, l_obs_raw, target_shape=x.target_shape)
        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (b,)

    def complete_correction(self, extrapolated, cache):
        (b,) = cache
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower

        l_obs_nonsquare = self._select_derivative_vect(l_ext, self.ode_order)
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=l_ext.T
        )
        m_cor = m_ext - gain @ b

        _shape = extrapolated.target_shape
        observed = VectNormal(mean=b, cov_sqrtm_lower=r_obs.T, target_shape=_shape)
        corrected = VectNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T, target_shape=_shape)
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

    def begin_correction(self, x: VectNormal, /, vector_field, t, p):
        vf_partial = jax.tree_util.Partial(
            self._residual, vector_field=vector_field, t=t, p=p
        )
        b, fn = jax.linearize(vf_partial, x.mean)

        cov_sqrtm_lower = self._cov_sqrtm_lower(
            cache=(b, fn), cov_sqrtm_lower=x.cov_sqrtm_lower
        )

        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        output_scale_sqrtm = VectNormal(
            b, l_obs_raw, target_shape=x.target_shape
        ).norm_of_whitened_residual_sqrtm()
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (b, fn)

    def complete_correction(self, extrapolated, cache):
        b, _ = cache

        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower

        l_obs_nonsquare = self._cov_sqrtm_lower(cache=cache, cov_sqrtm_lower=l_ext)

        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=l_ext.T
        )
        m_cor = m_ext - gain @ b

        shape = extrapolated.target_shape
        observed = VectNormal(mean=b, cov_sqrtm_lower=r_obs.T, target_shape=shape)
        corrected = VectNormal(mean=m_cor, cov_sqrtm_lower=r_cor.T, target_shape=shape)
        return observed, (corrected, gain)

    def _cov_sqrtm_lower(self, cache, cov_sqrtm_lower):
        _, fn = cache
        return jax.vmap(fn, in_axes=1, out_axes=1)(cov_sqrtm_lower)

    def _residual(self, x, vector_field, t, p):
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
        sci_fn = cubature_module.SphericalCubatureIntegration.from_params
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

    def begin_correction(self, x: VectNormal, /, vector_field, t, p):

        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))
        cache = (vmap_f,)

        # 1. x -> (e0x, e1x)
        R_X = x.cov_sqrtm_lower.T
        L0 = self._select_derivative_vect(R_X.T, 0)
        r_marg1_x = _sqrtm.sqrtm_to_upper_triangular(R=L0.T)
        m_marg1_x = self._select_derivative(x.mean, 0)
        m_marg1_y = self._select_derivative(x.mean, 1)

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
        marginals = VectNormal(m_marg, l_marg, target_shape=x.target_shape)
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
        L = extrapolated.cov_sqrtm_lower
        L0 = self._select_derivative_vect(L, 0)
        L1 = self._select_derivative_vect(L, 1)
        HL = L1 - H @ L0
        r_marg, (r_bw, gain) = _sqrtm.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )

        # Catch up the marginals
        x = extrapolated  # alias for readability in this code-block
        x0 = self._select_derivative(x.mean, 0)
        x1 = self._select_derivative(x.mean, 1)
        m_marg = x1 - (H @ x0 + noise.mean)
        shape = extrapolated.target_shape
        marginals = VectNormal(m_marg, r_marg.T, target_shape=shape)

        # Catch up the backward noise and return result
        m_bw = extrapolated.mean - gain @ m_marg
        backward_noise = VectNormal(m_bw, r_bw.T, target_shape=shape)
        return marginals, (backward_noise, gain)

    def _linearize(self, x, cache):
        vmap_f, *_ = cache

        # Create sigma points
        m_0 = self._select_derivative(x.mean, 0)
        r_0 = self._select_derivative_vect(x.cov_sqrtm_lower, 0).T
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
        return H, VectNormal(d, r_Om.T, target_shape=x.target_shape)

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
class VectConditional(_collections.AbstractConditional):
    def __init__(self, transition, noise):
        self.transition = transition
        self.noise = noise

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)

    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        shape = self.noise.target_shape
        return VectNormal(m, self.noise.cov_sqrtm_lower, target_shape=shape)

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return VectConditional(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

        shape = self.noise.target_shape
        noise = VectNormal(mean=xi, cov_sqrtm_lower=Xi, target_shape=shape)
        return VectConditional(g, noise=noise)

    def marginalise(self, rv, /):
        # Pull into preconditioned space
        m0_p = rv.mean
        l0_p = rv.cov_sqrtm_lower

        # Apply transition
        m_new_p = self.transition @ m0_p + self.noise.mean
        l_new_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.transition @ l0_p).T, R2=self.noise.cov_sqrtm_lower.T
        ).T

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        return VectNormal(m_new, l_new, target_shape=rv.target_shape)


@jax.tree_util.register_pytree_node_class
class VectIBM(_collections.AbstractExtrapolation):
    def __init__(self, a, q_sqrtm_lower, num_derivatives, ode_shape):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

        self.num_derivatives = num_derivatives
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"a={self.a}, q={self.q_sqrtm_lower}"
        args2 = f"num_derivatives={self.num_derivatives}"
        args3 = f"ode_shape={self.ode_shape}"
        return f"{name}({args1}, {args2}, {args3})"

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        aux = self.num_derivatives, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        a, q_sqrtm_lower = children
        n, d = aux
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower, num_derivatives=n, ode_shape=d)

    @classmethod
    def from_params(cls, ode_shape, num_derivatives):
        """Create a strategy from hyperparameters."""
        assert len(ode_shape) == 1
        (d,) = ode_shape
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        eye_d = jnp.eye(d)
        return cls(
            a=jnp.kron(eye_d, a),
            q_sqrtm_lower=jnp.kron(eye_d, q_sqrtm),
            num_derivatives=num_derivatives,
            ode_shape=ode_shape,
        )

    def init_corrected(self, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        if taylor_coefficients[0].shape != self.ode_shape:
            msg = "The solver's ODE dimension does not match the initial condition."
            raise ValueError(msg)

        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return VectNormal(
            mean=m0_corrected,
            cov_sqrtm_lower=c_sqrtm0_corrected,
            target_shape=m0_matrix.shape,
        )

    def init_error_estimate(self):
        return jnp.zeros(self.ode_shape)  # the initialisation is error-free

    def begin_extrapolation(self, m0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * m0
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        (d,) = self.ode_shape
        shape = (self.num_derivatives + 1, d)
        extrapolated = VectNormal(m_ext, q_sqrtm, target_shape=shape)
        return extrapolated, (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, dt):
        p, p_inv = _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        (d,) = self.ode_shape
        p = jnp.tile(p, d)
        p_inv = jnp.tile(p_inv, d)
        return p, p_inv

    def complete_extrapolation(self, linearisation_pt, l0, cache, output_scale_sqrtm):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.mean
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(output_scale_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p

        shape = linearisation_pt.target_shape
        return VectNormal(mean=m_ext, cov_sqrtm_lower=l_ext, target_shape=shape)

    def revert_markov_kernel(self, linearisation_pt, cache, l0, output_scale_sqrtm):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.mean

        l0_p = p_inv[:, None] * l0
        r_ext_p, (r_bw_p, g_bw_p) = _sqrtm.revert_conditional(
            R_X_F=(self.a @ l0_p).T,
            R_X=l0_p.T,
            R_YX=(output_scale_sqrtm * self.q_sqrtm_lower).T,
        )
        l_ext_p, l_bw_p = r_ext_p.T, r_bw_p.T
        m_bw_p = m0_p - g_bw_p @ m_ext_p

        # Un-apply the pre-conditioner.
        # The backward models remains preconditioned, because
        # we do backward passes in preconditioner-space.
        l_ext = p[:, None] * l_ext_p
        m_bw = p * m_bw_p
        l_bw = p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]

        shape = linearisation_pt.target_shape
        backward_noise = VectNormal(mean=m_bw, cov_sqrtm_lower=l_bw, target_shape=shape)
        bw_model = VectConditional(g_bw, noise=backward_noise)
        extrapolated = VectNormal(mean=m_ext, cov_sqrtm_lower=l_ext, target_shape=shape)
        return extrapolated, bw_model

    def init_conditional(self, rv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=rv_proto)
        return VectConditional(op, noise=noi)

    def _init_backward_transition(self):
        (d,) = self.ode_shape
        k = (self.num_derivatives + 1) * d
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        return VectNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
            target_shape=rv_proto.target_shape,
        )

    def init_output_scale_sqrtm(self):
        return 1.0
