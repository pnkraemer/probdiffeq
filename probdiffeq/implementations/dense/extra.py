"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _ibm_util, _sqrtm
from probdiffeq.implementations.dense import _vars

# todo: init_corrected should be init_hidden_state?


@jax.tree_util.register_pytree_node_class
class DenseConditional(_collections.AbstractConditional):
    def __init__(self, transition, noise, target_shape):
        self.transition = transition
        self.noise = noise
        self.target_shape = target_shape

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"transition={self.transition}, noise={self.noise}"
        args2 = f"target_shape={self.target_shape}"
        return f"{name}({args1}, {args2})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = (self.target_shape,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        transition, noise = children
        (target_shape,) = aux
        return cls(transition=transition, noise=noise, target_shape=target_shape)

    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        cond = _vars.DenseNormal(m, self.noise.cov_sqrtm_lower)
        return _vars.DenseStateSpaceVar(cond, target_shape=self.target_shape)

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        shape = self.target_shape
        return DenseConditional(self.transition, noise=noise, target_shape=shape)

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R_stack=((A @ D_sqrtm).T, B_sqrtm.T)).T

        noise = _vars.DenseNormal(mean=xi, cov_sqrtm_lower=Xi)
        return DenseConditional(g, noise=noise, target_shape=self.target_shape)

    def marginalise(self, rv, /):
        # Pull into preconditioned space
        m0_p = rv.hidden_state.mean
        l0_p = rv.hidden_state.cov_sqrtm_lower

        # Apply transition
        m_new_p = self.transition @ m0_p + self.noise.mean
        l_new_p = _sqrtm.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0_p).T, self.noise.cov_sqrtm_lower.T)
        ).T

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        marg = _vars.DenseNormal(m_new, l_new)
        return _vars.DenseStateSpaceVar(marg, target_shape=rv.target_shape)


@jax.tree_util.register_pytree_node_class
class DenseIBM(_collections.AbstractExtrapolation):
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
        if taylor_coefficients[0].shape != self.ode_shape:
            msg = "The solver's ODE dimension does not match the initial condition."
            raise ValueError(msg)

        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        corr = _vars.DenseNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)
        return _vars.DenseStateSpaceVar(corr, target_shape=m0_matrix.shape)

    def init_error_estimate(self):
        return jnp.zeros(self.ode_shape)  # the initialisation is error-free

    def begin_extrapolation(self, p0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * p0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        (d,) = self.ode_shape
        shape = (self.num_derivatives + 1, d)
        extrapolated = _vars.DenseNormal(m_ext, q_sqrtm)
        ssv = _vars.DenseStateSpaceVar(extrapolated, target_shape=shape)
        return ssv, (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, dt):
        p, p_inv = _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        (d,) = self.ode_shape
        p = jnp.tile(p, d)
        p_inv = jnp.tile(p_inv, d)
        return p, p_inv

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

        shape = linearisation_pt.target_shape
        rv = _vars.DenseNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return _vars.DenseStateSpaceVar(rv, target_shape=shape)

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

        shape = linearisation_pt.target_shape
        backward_noise = _vars.DenseNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = DenseConditional(g_bw, noise=backward_noise, target_shape=shape)
        rv = _vars.DenseNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        extrapolated = _vars.DenseStateSpaceVar(rv, target_shape=shape)
        return extrapolated, bw_model

    def init_conditional(self, ssv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=ssv_proto.hidden_state)
        return DenseConditional(op, noise=noi, target_shape=ssv_proto.target_shape)

    def _init_backward_transition(self):
        (d,) = self.ode_shape
        k = (self.num_derivatives + 1) * d
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        return _vars.DenseNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )

    def init_output_scale_sqrtm(self):
        return 1.0
