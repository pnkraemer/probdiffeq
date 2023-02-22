"""Implementations for scalar initial value problems."""

import jax
import jax.numpy as jnp

from probdiffeq import cubature as cubature_module
from probdiffeq.implementations import _collections, _ibm_util, _sqrtm

# todo: make public and split into submodules


@jax.tree_util.register_pytree_node_class
class ScalarNormal(_collections.AbstractNormal):
    # Normal RV. Shapes (), (). No QOI.

    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return m + l_sqrtm * base

    def Ax_plus_y(self, A, x, y):
        return A * x + y

    def condition_on_qoi_observation(self, u, /, observation_std):
        raise NotImplementedError

    def extract_qoi(self):
        raise NotImplementedError

    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    def scale_covariance(self, scale_sqrtm):
        return ScalarNormal(self.mean, scale_sqrtm * self.cov_sqrtm_lower)

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = (m_obs - u) / l_obs
        x1 = l_obs**2
        x2 = jnp.dot(res_white, res_white.T)
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = obs_pt / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm


@jax.tree_util.register_pytree_node_class
class StateSpaceVar(_collections.StateSpaceVar):
    # Normal RV. Shapes (n,), (n,n); zeroth state is the QOI.

    def extract_qoi(self):
        return self.hidden_state.mean[..., 0]

    def condition_on_qoi_observation(self, u, /, observation_std):
        if self.hidden_state.cov_sqrtm_lower.ndim > 2:
            return jax.vmap(
                StateSpaceVar.condition_on_qoi_observation, in_axes=(0, 0, None)
            )(self, u, observation_std)

        hc = self.hidden_state.cov_sqrtm_lower[0]
        m_obs = self.hidden_state.mean[0]

        r_yx = observation_std  # * jnp.eye(1)
        r_obs_mat, (r_cor, gain_mat) = _sqrtm.revert_conditional(
            R_X=self.hidden_state.cov_sqrtm_lower.T,
            R_X_F=hc[:, None],
            R_YX=r_yx[None, None],
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))

        m_cor = self.hidden_state.mean - gain * (m_obs - u)

        obs = ScalarNormal(m_obs, r_obs.T)
        cor = Normal(m_cor, r_cor.T)
        return obs, (StateSpaceVar(cor), gain)

    def extract_qoi_from_sample(self, u, /):
        if u.ndim == 1:
            return u[0]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def scale_covariance(self, scale_sqrtm):
        return StateSpaceVar(
            self.hidden_state.scale_covariance(scale_sqrtm=scale_sqrtm)
        )

    def marginal_nth_derivative(self, n):
        if self.hidden_state.mean.ndim > 1:
            # if the variable has batch-axes, vmap the result
            fn = StateSpaceVar.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.hidden_state.mean.shape[0]:
            msg = f"The {n}th derivative not available in the state-space variable."
            raise ValueError(msg)

        mean = self.hidden_state.mean[n]
        cov_sqrtm_lower_nonsquare = self.hidden_state.cov_sqrtm_lower[n, :]
        cov_sqrtm_lower = _sqrtm.sqrtm_to_upper_triangular(
            R=cov_sqrtm_lower_nonsquare[:, None]
        ).T
        return ScalarNormal(mean=mean, cov_sqrtm_lower=jnp.reshape(cov_sqrtm_lower, ()))


@jax.tree_util.register_pytree_node_class
class Normal(_collections.AbstractNormal):
    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, (m_obs - u), lower=False)
        x1 = jnp.linalg.slogdet(l_obs)[1] ** 2
        x2 = jnp.dot(res_white, res_white.T)
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, obs_pt, lower=False)
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def scale_covariance(self, scale_sqrtm):
        return Normal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[..., None, None] * self.cov_sqrtm_lower,
        )

    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    def Ax_plus_y(self, A, x, y):
        return A @ x + y


@jax.tree_util.register_pytree_node_class
class TaylorZerothOrder(_collections.AbstractCorrection):
    def begin_correction(self, x: StateSpaceVar, /, vector_field, t, p):
        m0, m1 = self.select_derivatives(x.hidden_state)
        fx = vector_field(*m0, t=t, p=p)
        cache, observed = self.marginalise_observation(fx, m1, x.hidden_state)

        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()
        error_estimate = observed.cov_sqrtm_lower
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def marginalise_observation(self, fx, m1, x):
        b = m1 - fx
        cov_sqrtm_lower = x.cov_sqrtm_lower[self.ode_order, :]
        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower[:, None])
        l_obs = jnp.reshape(l_obs_raw, ())
        observed = ScalarNormal(b, l_obs)
        cache = (b,)
        return cache, observed

    def select_derivatives(self, x):
        m0, m1 = x.mean[: self.ode_order], x.mean[self.ode_order]
        return m0, m1

    def complete_correction(self, extrapolated, cache):
        (b,) = cache
        m_ext, l_ext = (
            extrapolated.hidden_state.mean,
            extrapolated.hidden_state.cov_sqrtm_lower,
        )

        l_obs_nonsquare = l_ext[self.ode_order, :]
        r_obs_mat, (r_cor, gain_mat) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare[:, None], R_X=l_ext.T
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))
        m_cor = m_ext - gain * b

        observed = ScalarNormal(mean=b, cov_sqrtm_lower=r_obs.T)
        corrected = StateSpaceVar(Normal(mean=m_cor, cov_sqrtm_lower=r_cor.T))
        return observed, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class StatisticalFirstOrder(_collections.AbstractCorrection):
    def __init__(self, ode_order, cubature):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        self.cubature = cubature

    @classmethod
    def from_params(cls, ode_order):
        sci_fn = cubature_module.ThirdOrderSpherical.from_params
        cubature = sci_fn(input_shape=())
        return cls(ode_order=ode_order, cubature=cubature)

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature,)
        aux = (self.ode_order,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature,) = children
        (ode_order,) = aux
        return cls(ode_order=ode_order, cubature=cubature)

    def begin_correction(self, x: Normal, /, vector_field, t, p):
        raise NotImplementedError

    def calibrate(
        self, fx_mean: jax.Array, fx_centered_normed: jax.Array, extrapolated: Normal
    ):
        # Extract shapes
        (S,) = fx_centered_normed.shape
        (n,) = extrapolated.mean.shape

        # Marginal mean
        m_marg = extrapolated.mean[1] - fx_mean

        # Marginal covariance
        R1 = jnp.reshape(extrapolated.cov_sqrtm_lower[1, :], (n, 1))
        R2 = jnp.reshape(fx_centered_normed, (S, 1))
        std_marg_mat = _sqrtm.sum_of_sqrtm_factors(R_stack=(R1, R2))
        std_marg = jnp.reshape(std_marg_mat, ())

        # Extract error estimate and output scale from marginals
        marginals = ScalarNormal(m_marg, std_marg)
        output_scale_sqrtm = marginals.norm_of_whitened_residual_sqrtm()
        error_estimate = std_marg
        return error_estimate, output_scale_sqrtm

    def complete_correction(self, extrapolated, cache):
        raise NotImplementedError

    def linearize(self, rv, vmap_f):
        # Create sigma points
        pts, _, pts_centered_normed = self.transform_sigma_points(rv.hidden_state)

        # Evaluate the vector-field
        fx = vmap_f(pts)
        fx_mean, _, fx_centered_normed = self.center(fx)

        # Complete linearization
        return self.linearization_matrices(
            fx_centered_normed, fx_mean, pts_centered_normed, rv
        )

    def transform_sigma_points(self, rv: Normal):
        # Extract square-root of covariance (-> std-dev.)
        L0_nonsq = rv.cov_sqrtm_lower[0, :]
        r_marg1_x_mat = _sqrtm.sqrtm_to_upper_triangular(R=L0_nonsq[:, None])
        r_marg1_x = jnp.reshape(r_marg1_x_mat, ())

        # Multiply and shift the unit-points
        m_marg1_x = rv.mean[0]
        sigma_points_centered = self.cubature.points * r_marg1_x[None]
        sigma_points = m_marg1_x[None] + sigma_points_centered

        # Scale the shifted points with square-root weights
        _w = self.cubature.weights_sqrtm
        sigma_points_centered_normed = sigma_points_centered * _w
        return sigma_points, sigma_points_centered, sigma_points_centered_normed

    def center(self, fx):
        fx_mean = self.cubature.weights_sqrtm**2 @ fx
        fx_centered = fx - fx_mean[None]
        fx_centered_normed = fx_centered * self.cubature.weights_sqrtm
        return fx_mean, fx_centered, fx_centered_normed

    def linearization_matrices(
        self, fx_centered_normed, fx_mean, pts_centered_normed, rv
    ):
        # Revert the transition to get H and Omega
        # This is a pure sqrt-implementation of
        # Eq. (9) in https://arxiv.org/abs/2102.00514
        # It seems to be different to Section VI.B in
        # https://arxiv.org/pdf/2207.00426.pdf,
        # because the implementation below avoids sqrt-down-dates
        # pts_centered_normed = pts_centered * self.cubature.weights_sqrtm[:, None]
        # todo: with R_X_F = r_0_square, we would save a qr decomposition, right?
        #  (but would it still be valid?)
        _, (std_noi_mat, linop_mat) = _sqrtm.revert_conditional_noisefree(
            R_X_F=pts_centered_normed[:, None], R_X=fx_centered_normed[:, None]
        )
        std_noi = jnp.reshape(std_noi_mat, ())
        linop = jnp.reshape(linop_mat, ())

        # Catch up the transition-mean and return the result
        m_noi = fx_mean - linop * rv.mean[0]
        return linop, ScalarNormal(m_noi, std_noi)

    def complete_correction_post_linearize(self, linop, extrapolated, noise):
        # Compute the cubature-correction
        L0, L1 = extrapolated.cov_sqrtm_lower[0, :], extrapolated.cov_sqrtm_lower[1, :]
        HL = L1 - linop * L0
        std_marg_mat, (r_bw, gain_mat) = _sqrtm.revert_conditional(
            R_X=extrapolated.cov_sqrtm_lower.T,
            R_X_F=HL[:, None],
            R_YX=noise.cov_sqrtm_lower[None, None],
        )

        # Reshape the matrices into appropriate scalar-valued versions
        (n,) = extrapolated.mean.shape
        std_marg = jnp.reshape(std_marg_mat, ())
        gain = jnp.reshape(gain_mat, (n,))

        # Catch up the marginals
        x0, x1 = extrapolated.mean[0], extrapolated.mean[1]
        m_marg = x1 - (linop * x0 + noise.mean)
        obs = ScalarNormal(m_marg, std_marg)

        # Catch up the backward noise and return result
        m_bw = extrapolated.mean - gain * m_marg
        cor = StateSpaceVar(Normal(m_bw, r_bw.T))
        return obs, (cor, gain)


@jax.tree_util.register_pytree_node_class
class Conditional(_collections.AbstractConditional):
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
        if self.transition.ndim > 2:
            return jax.vmap(Conditional.__call__)(self, x)

        m = self.transition @ x + self.noise.mean
        return StateSpaceVar(Normal(m, self.noise.cov_sqrtm_lower))

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return Conditional(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        if self.transition.ndim > 2:
            return jax.vmap(Conditional.merge_with_incoming_conditional)(self, incoming)

        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R_stack=((A @ D_sqrtm).T, B_sqrtm.T)).T

        noise = Normal(mean=xi, cov_sqrtm_lower=Xi)
        return Conditional(g, noise=noise)

    def marginalise(self, rv, /):
        # Todo: this auto-batch is a bit hacky,
        #  but single-handedly replaces the entire BatchConditional class
        if rv.hidden_state.mean.ndim > 1:
            return jax.vmap(Conditional.marginalise)(self, rv)

        m0, l0 = rv.hidden_state.mean, rv.hidden_state.cov_sqrtm_lower

        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrtm.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
        ).T

        return StateSpaceVar(Normal(m_new, l_new))


@jax.tree_util.register_pytree_node_class
class IBM(_collections.AbstractExtrapolation):
    def __init__(self, a, q_sqrtm_lower):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"a={self.a}, q={self.q_sqrtm_lower}"
        return f"{name}({args1})"

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower = children
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower)

    @classmethod
    def from_params(cls, num_derivatives):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def init_corrected(self, taylor_coefficients):
        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)

        rv = Normal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)
        return StateSpaceVar(rv)

    def init_error_estimate(self):
        return jnp.zeros(())

    def begin_extrapolation(self, p0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * p0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        extrapolated = Normal(m_ext, q_sqrtm)
        return StateSpaceVar(extrapolated), (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )

    def complete_extrapolation(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.hidden_state.mean
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ (p_inv[:, None] * p0.hidden_state.cov_sqrtm_lower)).T,
                (output_scale_sqrtm * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p
        return StateSpaceVar(Normal(mean=m_ext, cov_sqrtm_lower=l_ext))

    def revert_markov_kernel(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.hidden_state.mean

        l0_p = p_inv[:, None] * p0.hidden_state.cov_sqrtm_lower
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

        backward_noise = Normal(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = Conditional(g_bw, noise=backward_noise)
        extrapolated = Normal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return StateSpaceVar(extrapolated), bw_model

    def init_conditional(self, ssv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=ssv_proto.hidden_state)
        return Conditional(op, noise=noi)

    def _init_backward_transition(self):
        k = self.num_derivatives + 1
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        mean = jnp.zeros_like(rv_proto.mean)
        cov_sqrtm_lower = jnp.zeros_like(rv_proto.cov_sqrtm_lower)
        return Normal(mean, cov_sqrtm_lower)

    def init_output_scale_sqrtm(self):
        return 1.0
