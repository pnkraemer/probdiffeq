"""Implementations for scalar initial value problems."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probdiffeq import _sqrt_util
from probdiffeq.implementations import _collections, _ibm_util, cubature

# todo: make public and split into submodules


@jax.tree_util.register_pytree_node_class
class NormalQOI(_collections.AbstractNormal):
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
        return NormalQOI(self.mean, scale_sqrtm * self.cov_sqrtm_lower)

    def logpdf(self, u, /):
        x1 = self.marginal_stds() ** 2  # logdet
        x2 = self.mahalanobis_norm(u) ** 2
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def marginal_stds(self):
        return jnp.abs(self.cov_sqrtm_lower)

    def mahalanobis_norm(self, u, /):
        res_white_scalar = self.residual_white(u)
        return res_white_scalar

    def residual_white(self, u, /):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = (obs_pt - u) / l_obs
        return res_white


@jax.tree_util.register_pytree_node_class
class StateSpaceVar(_collections.StateSpaceVar):
    # Normal RV. Shapes (n,), (n,n); zeroth state is the QOI.

    def extract_qoi(self):
        return self.hidden_state.mean[..., 0]

    def observe_qoi(self, observation_std):
        # what is this for? batched calls? If so, that seems wrong.
        #  the scalar state should not worry about the context it is called in.
        if self.hidden_state.cov_sqrtm_lower.ndim > 2:
            fn = StateSpaceVar.observe_qoi
            fn_vmap = jax.vmap(fn, in_axes=(0, None), out_axes=(0, 0))
            return fn_vmap(self, observation_std)

        hc = self.hidden_state.cov_sqrtm_lower[0]
        m_obs = self.hidden_state.mean[0]

        r_yx = observation_std  # * jnp.eye(1)
        r_obs_mat, (r_cor, gain_mat) = _sqrt_util.revert_conditional(
            R_X=self.hidden_state.cov_sqrtm_lower.T,
            R_X_F=hc[:, None],
            R_YX=r_yx[None, None],
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))

        m_cor = self.hidden_state.mean - gain * m_obs

        obs = NormalQOI(m_obs, r_obs.T)
        cor = NormalHiddenState(m_cor, r_cor.T)
        return obs, ConditionalQOI(gain, cor)

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
        cov_sqrtm_lower = _sqrt_util.sqrtm_to_upper_triangular(
            R=cov_sqrtm_lower_nonsquare[:, None]
        ).T
        return NormalQOI(mean=mean, cov_sqrtm_lower=jnp.reshape(cov_sqrtm_lower, ()))


@jax.tree_util.register_pytree_node_class
class NormalHiddenState(_collections.AbstractNormal):
    def logpdf(self, u, /):
        raise NotImplementedError

    def mahalanobis_norm(self, u, /):
        raise NotImplementedError

    def scale_covariance(self, scale_sqrtm):
        return NormalHiddenState(
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
        mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros(()))
        output_scale_sqrtm = mahalanobis_norm / jnp.sqrt(m1.size)
        error_estimate_unscaled = observed.marginal_stds()
        error_estimate = output_scale_sqrtm * error_estimate_unscaled
        return error_estimate, output_scale_sqrtm, cache

    def marginalise_observation(self, fx, m1, x):
        b = m1 - fx
        cov_sqrtm_lower = x.cov_sqrtm_lower[self.ode_order, :]
        l_obs_raw = _sqrt_util.sqrtm_to_upper_triangular(R=cov_sqrtm_lower[:, None])
        l_obs = jnp.reshape(l_obs_raw, ())
        observed = NormalQOI(b, l_obs)
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
        r_obs_mat, (r_cor, gain_mat) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare[:, None], R_X=l_ext.T
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))
        m_cor = m_ext - gain * b
        observed = NormalQOI(mean=b, cov_sqrtm_lower=r_obs.T)
        corrected = StateSpaceVar(
            NormalHiddenState(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        )
        return observed, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class StatisticalFirstOrder(_collections.AbstractCorrection):
    def __init__(self, ode_order, cubature_rule):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        self.cubature_rule = cubature_rule

    @classmethod
    def from_params(cls, ode_order):
        sci_fn = cubature.ThirdOrderSpherical.from_params
        cubature_rule = sci_fn(input_shape=())
        return cls(ode_order=ode_order, cubature_rule=cubature_rule)

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature_rule,)
        aux = (self.ode_order,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature_rule,) = children
        (ode_order,) = aux
        return cls(ode_order=ode_order, cubature_rule=cubature_rule)

    def begin_correction(self, x: NormalHiddenState, /, vector_field, t, p):
        raise NotImplementedError

    def calibrate(
        self,
        fx_mean: ArrayLike,
        fx_centered_normed: ArrayLike,
        extrapolated: NormalHiddenState,
    ):
        fx_mean = jnp.asarray(fx_mean)
        fx_centered_normed = jnp.asarray(fx_centered_normed)

        # Extract shapes
        (S,) = fx_centered_normed.shape
        (n,) = extrapolated.mean.shape

        # Marginal mean
        m_marg = extrapolated.mean[1] - fx_mean

        # Marginal covariance
        R1 = jnp.reshape(extrapolated.cov_sqrtm_lower[1, :], (n, 1))
        R2 = jnp.reshape(fx_centered_normed, (S, 1))
        std_marg_mat = _sqrt_util.sum_of_sqrtm_factors(R_stack=(R1, R2))
        std_marg = jnp.reshape(std_marg_mat, ())

        # Extract error estimate and output scale from marginals
        marginals = NormalQOI(m_marg, std_marg)
        mahalanobis_norm = marginals.mahalanobis_norm(jnp.zeros(()))
        output_scale_sqrtm = mahalanobis_norm / jnp.sqrt(m_marg.size)

        error_estimate_unscaled = marginals.marginal_stds()
        error_estimate = error_estimate_unscaled * output_scale_sqrtm
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

    def transform_sigma_points(self, rv: NormalHiddenState):
        # Extract square-root of covariance (-> std-dev.)
        L0_nonsq = rv.cov_sqrtm_lower[0, :]
        r_marg1_x_mat = _sqrt_util.sqrtm_to_upper_triangular(R=L0_nonsq[:, None])
        r_marg1_x = jnp.reshape(r_marg1_x_mat, ())

        # Multiply and shift the unit-points
        m_marg1_x = rv.mean[0]
        sigma_points_centered = self.cubature_rule.points * r_marg1_x[None]
        sigma_points = m_marg1_x[None] + sigma_points_centered

        # Scale the shifted points with square-root weights
        _w = self.cubature_rule.weights_sqrtm
        sigma_points_centered_normed = sigma_points_centered * _w
        return sigma_points, sigma_points_centered, sigma_points_centered_normed

    def center(self, fx):
        fx_mean = self.cubature_rule.weights_sqrtm**2 @ fx
        fx_centered = fx - fx_mean[None]
        fx_centered_normed = fx_centered * self.cubature_rule.weights_sqrtm
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
        # pts_centered_normed = pts_centered * self.cubature_rule.weights_sqrtm[:, None]
        _, (std_noi_mat, linop_mat) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=pts_centered_normed[:, None], R_X=fx_centered_normed[:, None]
        )
        std_noi = jnp.reshape(std_noi_mat, ())
        linop = jnp.reshape(linop_mat, ())

        # Catch up the transition-mean and return the result
        m_noi = fx_mean - linop * rv.mean[0]
        return linop, NormalQOI(m_noi, std_noi)

    def complete_correction_post_linearize(self, linop, extrapolated, noise):
        # Compute the cubature-correction
        L0, L1 = (
            extrapolated.cov_sqrtm_lower[0, :],
            extrapolated.cov_sqrtm_lower[1, :],
        )
        HL = L1 - linop * L0
        std_marg_mat, (r_bw, gain_mat) = _sqrt_util.revert_conditional(
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
        obs = NormalQOI(m_marg, std_marg)

        # Catch up the backward noise and return result
        m_bw = extrapolated.mean - gain * m_marg
        cor = StateSpaceVar(NormalHiddenState(m_bw, r_bw.T))
        return obs, (cor, gain)


@jax.tree_util.register_pytree_node_class
class _Conditional(_collections.AbstractConditional):
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


@jax.tree_util.register_pytree_node_class
class ConditionalHiddenState(_Conditional):
    def __call__(self, x, /):
        if self.transition.ndim > 2:
            return jax.vmap(ConditionalHiddenState.__call__)(self, x)

        m = self.transition @ x + self.noise.mean
        return StateSpaceVar(NormalHiddenState(m, self.noise.cov_sqrtm_lower))

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return ConditionalHiddenState(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        if self.transition.ndim > 2:
            fn = ConditionalHiddenState.merge_with_incoming_conditional
            return jax.vmap(fn)(self, incoming)

        A = self.transition
        (b, B_sqrtm_lower) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((A @ D_sqrtm).T, B_sqrtm_lower.T)
        ).T

        noise = NormalHiddenState(mean=xi, cov_sqrtm_lower=Xi)
        return ConditionalHiddenState(g, noise=noise)

    def marginalise(self, rv, /):
        # Todo: this auto-batch is a bit hacky,
        #  but single-handedly replaces the entire BatchConditional class
        if rv.hidden_state.mean.ndim > 1:
            return jax.vmap(ConditionalHiddenState.marginalise)(self, rv)

        m0, l0 = rv.hidden_state.mean, rv.hidden_state.cov_sqrtm_lower

        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
        ).T

        return StateSpaceVar(NormalHiddenState(m_new, l_new))


@jax.tree_util.register_pytree_node_class
class ConditionalQOI(_Conditional):
    def __call__(self, x, /):
        if self.transition.ndim > 1:
            return jax.vmap(ConditionalQOI.__call__)(self, x)
        m = self.transition * x + self.noise.mean
        return StateSpaceVar(NormalHiddenState(m, self.noise.cov_sqrtm_lower))


@jax.tree_util.register_pytree_node_class
class IBM(_collections.AbstractExtrapolation):
    def __init__(self, a, q_sqrtm_lower, preconditioner_scales, preconditioner_powers):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

        self.preconditioner_scales = preconditioner_scales
        self.preconditioner_powers = preconditioner_powers

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"a={self.a}, q={self.q_sqrtm_lower}"
        args2 = f"preconditioner_scales={self.preconditioner_scales}"
        args3 = f"preconditioner_powers={self.preconditioner_powers}"
        return f"{name}({args1}, {args2}, {args3})"

    def tree_flatten(self):
        children = (
            self.a,
            self.q_sqrtm_lower,
            self.preconditioner_scales,
            self.preconditioner_powers,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower, scales, powers = children
        return cls(
            a=a,
            q_sqrtm_lower=q_sqrtm_lower,
            preconditioner_scales=scales,
            preconditioner_powers=powers,
        )

    @classmethod
    def from_params(cls, num_derivatives):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        _tmp = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
        scales, powers = _tmp
        return cls(
            a=a,
            q_sqrtm_lower=q_sqrtm,
            preconditioner_scales=scales,
            preconditioner_powers=powers,
        )

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def init_hidden_state(self, taylor_coefficients):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)

        m0_matrix = jnp.stack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)

        rv = NormalHiddenState(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)
        return StateSpaceVar(rv)

    def init_error_estimate(self):
        return jnp.zeros(())

    def begin_extrapolation(self, p0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * p0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        extrapolated = NormalHiddenState(m_ext, q_sqrtm)
        return StateSpaceVar(extrapolated), (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )

    def complete_extrapolation(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.hidden_state.mean
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ (p_inv[:, None] * p0.hidden_state.cov_sqrtm_lower)).T,
                (output_scale_sqrtm * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p
        return StateSpaceVar(NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext))

    def revert_markov_kernel(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.hidden_state.mean

        l0_p = p_inv[:, None] * p0.hidden_state.cov_sqrtm_lower
        r_ext_p, (r_bw_p, g_bw_p) = _sqrt_util.revert_conditional(
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

        backward_noise = NormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = ConditionalHiddenState(g_bw, noise=backward_noise)
        extrapolated = NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return StateSpaceVar(extrapolated), bw_model

    def init_conditional(self, ssv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=ssv_proto.hidden_state)
        return ConditionalHiddenState(op, noise=noi)

    def _init_backward_transition(self):
        k = self.num_derivatives + 1
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        mean = jnp.zeros_like(rv_proto.mean)
        cov_sqrtm_lower = jnp.zeros_like(rv_proto.cov_sqrtm_lower)
        return NormalHiddenState(mean, cov_sqrtm_lower)

    def init_output_scale_sqrtm(self):
        return 1.0
