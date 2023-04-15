"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _extra, _ibm_util
from probdiffeq.statespace.dense import _conds, _vars


def ibm_dense(ode_shape, num_derivatives):
    assert len(ode_shape) == 1
    (d,) = ode_shape
    a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
    eye_d = jnp.eye(d)

    _tmp = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
    scales, powers = _tmp
    return _DenseIBM(
        a=jnp.kron(eye_d, a),
        q_sqrtm_lower=jnp.kron(eye_d, q_sqrtm),
        num_derivatives=num_derivatives,
        ode_shape=ode_shape,
        preconditioner_scales=scales,
        preconditioner_powers=powers,
    )


@jax.tree_util.register_pytree_node_class
class _DenseIBM(_extra.Extrapolation):
    def __init__(
        self,
        *,
        num_derivatives,
        ode_shape,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_derivatives = num_derivatives
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

    @property
    def target_shape(self):
        return (self.num_derivatives + 1,) + self.ode_shape

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        args3 = f"ode_shape={self.ode_shape}"
        return f"<Dense IBM with {args2}, {args3}>"

    def tree_flatten(self):
        children = (
            self.a,
            self.q_sqrtm_lower,
            self.preconditioner_scales,
            self.preconditioner_powers,
        )
        aux = self.num_derivatives, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        a, q_sqrtm_lower, scales, powers = children
        n, d = aux
        return cls(
            a=a,
            q_sqrtm_lower=q_sqrtm_lower,
            num_derivatives=n,
            ode_shape=d,
            preconditioner_powers=powers,
            preconditioner_scales=scales,
        )

    def solution_from_tcoeffs_without_reversal(self, taylor_coefficients, /):
        c_sqrtm0_corrected, m0_corrected = self._stack_tcoeffs(taylor_coefficients)
        return _vars.DenseNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def solution_from_tcoeffs_with_reversal(self, taylor_coefficients, /):
        c_sqrtm0_corrected, m0_corrected = self._stack_tcoeffs(taylor_coefficients)
        rv = _vars.DenseNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)
        cond = self._init_conditional(rv_proto=rv)
        return rv, cond

    def _init_conditional(self, rv_proto):
        (d,) = self.ode_shape
        k = (self.num_derivatives + 1) * d

        op = jnp.eye(k)
        noi = _vars.DenseNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )
        return _conds.DenseConditional(op, noise=noi, target_shape=self.target_shape)

    def _stack_tcoeffs(self, taylor_coefficients):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        if taylor_coefficients[0].shape != self.ode_shape:
            msg = "The solver's ODE dimension does not match the initial condition."
            raise ValueError(msg)
        m0_matrix = jnp.stack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return c_sqrtm0_corrected, m0_corrected

    # todo: we really have two different extrapolation models in one
    #  with two completely disjoint sets of init(), extract(), begin(), and complete()
    #  methods. This should be changed very soon.

    def init_without_reversal(self, rv, /):
        return _vars.DenseSSV(
            rv,
            hidden_shape=self.target_shape,
            observed_state=None,
            output_scale_dynamic=None,
            error_estimate=None,
            cache_extra=None,
            cache_corr=None,
            backward_model=None,
        )

    def init_with_reversal(self, rv, cond, /):
        return _vars.DenseSSV(
            rv,
            backward_model=cond,
            hidden_shape=self.target_shape,
            observed_state=None,
            output_scale_dynamic=None,
            error_estimate=None,
            cache_extra=None,
            cache_corr=None,
        )

    def init_with_reversal_and_reset(self, rv, _cond, /):
        cond = self._init_conditional(rv_proto=rv)
        return _vars.DenseSSV(
            rv,
            backward_model=cond,
            hidden_shape=self.target_shape,
            observed_state=None,
            output_scale_dynamic=None,
            error_estimate=None,
            cache_extra=None,
            cache_corr=None,
        )

    def begin(self, s0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        ext = _vars.DenseNormal(m_ext, q_sqrtm)
        return _vars.DenseSSV(
            ext,
            cache_extra=(m_ext_p, m0_p, p, p_inv),
            hidden_shape=s0.hidden_shape,
            output_scale_dynamic=None,
            cache_corr=None,
            backward_model=None,
            # todo: The below should not be necessary
            #  but currently, it is: because of pytree-shape stability in interpolation
            observed_state=jax.tree_util.tree_map(jnp.zeros_like, s0.observed_state),
            error_estimate=jax.tree_util.tree_map(jnp.zeros_like, s0.error_estimate),
        )

    def _assemble_preconditioner(self, dt):
        p, p_inv = _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )
        (d,) = self.ode_shape
        p = jnp.tile(p, d)
        p_inv = jnp.tile(p_inv, d)
        return p, p_inv

    def complete_without_reversal(self, state, /, state_previous, output_scale):
        _, _, p, p_inv = state.cache_extra
        m_ext = state.hidden_state.mean
        l0 = state_previous.hidden_state.cov_sqrtm_lower

        l0_p = p_inv[:, None] * l0
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ l0_p).T,
                (output_scale * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p

        rv = _vars.DenseNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return _vars.DenseSSV(
            rv,
            hidden_shape=state.hidden_shape,
            error_estimate=state.error_estimate,
            cache_corr=state.cache_corr,
            observed_state=state.observed_state,  # usually None?
            backward_model=None,
            output_scale_dynamic=None,
            cache_extra=None,
        )

    def complete_with_reversal(self, state, /, state_previous, output_scale):
        m_ext_p, m0_p, p, p_inv = state.cache_extra
        m_ext = state.hidden_state.mean
        l0 = state_previous.hidden_state.cov_sqrtm_lower

        l0_p = p_inv[:, None] * l0
        r_ext_p, (r_bw_p, g_bw_p) = _sqrt_util.revert_conditional(
            R_X_F=(self.a @ l0_p).T,
            R_X=l0_p.T,
            R_YX=(output_scale * self.q_sqrtm_lower).T,
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

        backward_noise = _vars.DenseNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = _conds.DenseConditional(
            g_bw, noise=backward_noise, target_shape=state.hidden_shape
        )
        rv = _vars.DenseNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

        return _vars.DenseSSV(
            rv,
            backward_model=bw_model,
            hidden_shape=state.hidden_shape,
            error_estimate=state.error_estimate,
            cache_corr=state.cache_corr,
            observed_state=state.observed_state,  # usually None?
            output_scale_dynamic=None,
            cache_extra=None,
        )

    def extract_with_reversal(self, s, /):
        return s.hidden_state, s.backward_model

    def extract_without_reversal(self, s, /):
        return s.hidden_state

    # todo: private (remove sub-functions)
    def promote_output_scale(self, output_scale):
        return output_scale

    def replace_backward_model(self, s, /, backward_model):
        return _vars.DenseSSV(
            s.hidden_state,
            backward_model=backward_model,  # new
            hidden_shape=s.hidden_shape,
            observed_state=s.observed_state,
            error_estimate=s.error_estimate,
            output_scale_dynamic=s.output_scale_dynamic,
            cache_extra=s.cache_extra,
            cache_corr=s.cache_corr,
        )

    def duplicate_with_unit_backward_model(self, s, /):
        unit_bw_model = self._init_conditional(rv_proto=s.hidden_state)
        return self.replace_backward_model(s, unit_bw_model)
