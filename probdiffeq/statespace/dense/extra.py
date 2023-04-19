"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq import _collections, _sqrt_util
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

    def filter_solution_from_tcoeffs(self, taylor_coefficients, /):
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
        return _vars.DenseNormal(
            mean=m0_corrected,
            cov_sqrtm_lower=c_sqrtm0_corrected,
            target_shape=self.target_shape,
        )

    def smoother_solution_from_tcoeffs(self, taylor_coefficients, /):
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
        rv = _vars.DenseNormal(
            mean=m0_corrected,
            cov_sqrtm_lower=c_sqrtm0_corrected,
            target_shape=self.target_shape,
        )
        conds = self.smoother_init_conditional(rv_proto=rv)
        return _collections.MarkovSequence(init=rv, backward_model=conds)

    def filter_init(self, sol, /):
        ssv = _vars.DenseSSV(sol, target_shape=self.target_shape)
        extra = None
        return ssv, extra

    def filter_extract(self, ssv, _extra, /):
        return ssv.hidden_state

    def smoother_init(self, sol: _collections.MarkovSequence, /):
        ssv = _vars.DenseSSV(sol.init, target_shape=self.target_shape)
        extra = sol.backward_model
        return ssv, extra

    def smoother_extract(self, ssv, extra, /) -> _collections.MarkovSequence:
        rv = ssv.hidden_state
        cond = extra
        return _collections.MarkovSequence(init=rv, backward_model=cond)

    def init_error_estimate(self):
        return jnp.zeros(self.ode_shape)  # the initialisation is error-free

    def filter_begin(self, ssv, extra, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * ssv.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        l0 = ssv.hidden_state.cov_sqrtm_lower

        (d,) = self.ode_shape
        shape = (self.num_derivatives + 1, d)
        ext = _vars.DenseNormal(m_ext, q_sqrtm, target_shape=shape)
        cache = (m_ext_p, m0_p, p, p_inv, l0)
        ssv = _vars.DenseSSV(ext, target_shape=shape)
        return ssv, cache

    def smoother_begin(self, ssv, extra, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * ssv.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        l0 = ssv.hidden_state.cov_sqrtm_lower

        (d,) = self.ode_shape
        shape = (self.num_derivatives + 1, d)
        ext = _vars.DenseNormal(m_ext, q_sqrtm, target_shape=shape)
        cache = (extra, m_ext_p, m0_p, p, p_inv, l0)
        ssv = _vars.DenseSSV(ext, target_shape=shape)
        return ssv, cache

    def _assemble_preconditioner(self, dt):
        p, p_inv = _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )
        (d,) = self.ode_shape
        p = jnp.tile(p, d)
        p_inv = jnp.tile(p_inv, d)
        return p, p_inv

    def filter_complete(self, ssv, extra, /, output_scale):
        _, _, p, p_inv, l0 = extra
        m_ext = ssv.hidden_state.mean
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ (p_inv[:, None] * l0)).T,
                (output_scale * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p

        shape = ssv.target_shape
        rv = _vars.DenseNormal(mean=m_ext, cov_sqrtm_lower=l_ext, target_shape=shape)
        ssv = _vars.DenseSSV(rv, target_shape=shape)
        return ssv, None

    def smoother_complete(self, ssv, extra, /, output_scale):
        _, m_ext_p, m0_p, p, p_inv, l0 = extra
        # todo: move this to cache? it may be modified by correction models
        m_ext = ssv.hidden_state.mean

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

        backward_noise = _vars.DenseNormal(
            mean=m_bw, cov_sqrtm_lower=l_bw, target_shape=self.target_shape
        )
        bw_model = _conds.DenseConditional(
            g_bw, noise=backward_noise, target_shape=self.target_shape
        )
        rv = _vars.DenseNormal(
            mean=m_ext, cov_sqrtm_lower=l_ext, target_shape=self.target_shape
        )
        ext = _vars.DenseSSV(rv, target_shape=self.target_shape)
        return ext, bw_model

    def smoother_init_conditional(self, rv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=rv_proto)
        return _conds.DenseConditional(op, noise=noi, target_shape=self.target_shape)

    def _init_backward_transition(self):
        (d,) = self.ode_shape
        k = (self.num_derivatives + 1) * d
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        return _vars.DenseNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
            target_shape=rv_proto.target_shape,
        )

    # todo: the below two are a bit of an init/extract pair.
    #  It seems like this should happen in a "calibration" algorithm, right?
    #  Probably on ivpsolver-level. But the difficulty will be that
    #  implementing these two functions will require the shape of the state-space.
    def promote_output_scale(self, output_scale):
        return output_scale

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale
