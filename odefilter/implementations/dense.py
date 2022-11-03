"""State-space models with dense covariance structure   ."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.lax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax.tree_util import register_pytree_node_class

from odefilter import _control_flow
from odefilter import cubature as cubature_module
from odefilter.implementations import _correction, _extrapolation, _ibm_util, _sqrtm


class _Normal(NamedTuple):
    """Random variable with a normal distribution."""

    mean: Array  # (k,) shape
    cov_sqrtm_lower: Array  # (k,k) shape


@register_pytree_node_class
class _DenseCorrection(_correction.Correction):
    def __init__(self, *, ode_dimension, ode_order=1):
        super().__init__(ode_order=ode_order)
        self.ode_dimension = ode_dimension

    def tree_flatten(self):
        children = ()
        aux = self.ode_order, self.ode_dimension
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        ode_order, ode_dimension = aux
        return cls(ode_order=ode_order, ode_dimension=ode_dimension)

    def evidence_sqrtm(self, *, observed):
        obs_pt, l_obs = observed.mean, observed.cov_sqrtm_lower
        res_white = jsp.linalg.solve_triangular(l_obs.T, obs_pt, lower=False)
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def _e0(self, x):
        x_reshaped = jnp.reshape(x, (-1, self.ode_dimension), order="F")
        return x_reshaped[: self.ode_order, ...]

    def _e1(self, x):
        x_reshaped = jnp.reshape(x, (-1, self.ode_dimension), order="F")
        return x_reshaped[self.ode_order, ...]

    def _e0v(self, x):
        return jax.vmap(lambda s: self._e0(s), in_axes=1, out_axes=1)(x)

    def _e1v(self, x):
        return jax.vmap(lambda s: self._e1(s), in_axes=1, out_axes=1)(x)


@register_pytree_node_class
class TaylorZerothOrder(_DenseCorrection):
    def begin_correction(self, x: _Normal, /, *, vector_field, t, p):
        m0, m1 = self._e0(x.mean), self._e1(x.mean)
        b = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = self._e1v(x.cov_sqrtm_lower)

        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        output_scale_sqrtm = self.evidence_sqrtm(observed=_Normal(b, l_obs_raw))
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (b,)

    def complete_correction(self, *, extrapolated, cache):
        (b,) = cache
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower

        l_obs_nonsquare = self._e1v(l_ext)
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=l_ext.T
        )
        l_obs, l_cor = r_obs.T, r_cor.T

        m_cor = m_ext - gain @ b
        observed = _Normal(mean=b, cov_sqrtm_lower=l_obs)
        corrected = _Normal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, gain)


@register_pytree_node_class
class TaylorFirstOrder(_DenseCorrection):
    """Extended Kalman filter correction."""

    def begin_correction(self, x: _Normal, /, *, vector_field, t, p):
        vf_partial = jax.tree_util.Partial(
            self._residual, vector_field=vector_field, t=t, p=p
        )
        b, fn = jax.linearize(vf_partial, x.mean)

        cov_sqrtm_lower = self._cov_sqrtm_lower(
            cache=(b, fn), cov_sqrtm_lower=x.cov_sqrtm_lower
        )

        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        output_scale_sqrtm = self.evidence_sqrtm(observed=_Normal(b, l_obs_raw))
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (b, fn)

    def complete_correction(self, *, extrapolated, cache):
        b, _ = cache

        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower

        l_obs_nonsquare = self._cov_sqrtm_lower(cache=cache, cov_sqrtm_lower=l_ext)

        r_obs, (r_cor, gain) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare.T, R_X=l_ext.T
        )
        l_obs, l_cor = r_obs.T, r_cor.T

        m_cor = m_ext - gain @ b
        observed = _Normal(mean=b, cov_sqrtm_lower=l_obs)
        corrected = _Normal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, gain)

    def _cov_sqrtm_lower(self, *, cache, cov_sqrtm_lower):
        _, fn = cache
        return jax.vmap(fn, in_axes=1, out_axes=1)(cov_sqrtm_lower)

    def _residual(self, x, *, vector_field, t, p):
        x0, x1 = self._e0(x), self._e1(x)
        fx0 = vector_field(*x0, t=t, p=p)
        return x1 - fx0


@register_pytree_node_class
class MomentMatching(_DenseCorrection):
    def __init__(self, *, ode_dimension, cubature=None, ode_order=1):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order, ode_dimension=ode_dimension)

        if cubature is None:
            self.cubature = cubature_module.SphericalCubatureIntegration.from_params(
                ode_dimension=ode_dimension
            )
        else:
            self.cubature = cubature

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature,)
        aux = self.ode_order, self.ode_dimension
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature,) = children
        ode_order, ode_dimension = aux
        return cls(ode_order=ode_order, ode_dimension=ode_dimension, cubature=cubature)

    def begin_correction(self, x: _Normal, /, *, vector_field, t, p):

        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))
        e0v = jax.vmap(self._e0, in_axes=1, out_axes=1)
        e1v = jax.vmap(self._e1, in_axes=1, out_axes=1)
        cache = (vmap_f, e0v, e1v)

        # 1. x -> (e0x, e1x)
        R_X = x.cov_sqrtm_lower.T
        r_marg1_x = _sqrtm.sqrtm_to_upper_triangular(R=e0v(R_X.T).T)
        m_marg1_x, m_marg1_y = self._e0(x.mean), self._e1(x.mean)

        # 2. (x, y) -> (f(x), y)
        x_centered = self.cubature.points @ r_marg1_x
        sigma_points = m_marg1_x[None, :] + x_centered
        fx = vmap_f(sigma_points)
        m_marg2 = self.cubature.weights_sqrtm**2 @ fx
        fx_centered = fx - m_marg2[None, :]
        fx_centered_normed = fx_centered * self.cubature.weights_sqrtm[:, None]

        # 3. (x, y) -> y - x (last one)
        m_marg = m_marg1_y - m_marg2
        l_marg = _sqrtm.sum_of_sqrtm_factors(R1=e1v(R_X.T).T, R2=fx_centered_normed).T

        # Summarise
        marginals = _Normal(m_marg, l_marg)
        output_scale_sqrtm = self.evidence_sqrtm(observed=marginals)

        # Compute error estimate
        l_obs = marginals.cov_sqrtm_lower
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs, l_obs))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, *, extrapolated, cache):
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
        (_, e0v, e1v), L = cache, extrapolated.cov_sqrtm_lower
        HL = e1v(L) - H @ e0v(L)
        r_marg, (r_bw, gain) = _sqrtm.revert_conditional(
            R_X_F=HL.T, R_X=L.T, R_YX=noise.cov_sqrtm_lower.T
        )

        # Catch up the marginals
        x = extrapolated  # alias for readability in this code-block
        m_marg = self._e1(x.mean) - (H @ self._e0(x.mean) + noise.mean)
        marginals = _Normal(m_marg, r_marg.T)

        # Catch up the backward noise and return result
        m_bw = extrapolated.mean - gain @ m_marg
        backward_noise = _Normal(m_bw, r_bw.T)
        return marginals, (backward_noise, gain)

    def _linearize(self, *, x, cache):
        vmap_f, e0v, _ = cache

        # Create sigma points
        m_0 = self._e0(x.mean)
        r_0 = e0v(x.cov_sqrtm_lower).T
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
        return H, _Normal(d, r_Om.T)

    # Are we overwriting a method here?

    def _e1(self, x):
        return x.reshape((-1, self.ode_dimension), order="F")[1, ...]

    def _e0(self, x):
        return x.reshape((-1, self.ode_dimension), order="F")[0, ...]


@register_pytree_node_class
@dataclass(frozen=True)
class IBM(_extrapolation.Extrapolation):
    """Handle dense covariances."""

    a: Array
    q_sqrtm_lower: Array

    num_derivatives: int
    ode_dimension: int

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        aux = self.num_derivatives, self.ode_dimension
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        a, q_sqrtm_lower = children
        n, d = aux
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower, num_derivatives=n, ode_dimension=d)

    @classmethod
    def from_params(cls, *, ode_dimension, num_derivatives=4):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        eye_d = jnp.eye(ode_dimension)
        return cls(
            a=jnp.kron(eye_d, a),
            q_sqrtm_lower=jnp.kron(eye_d, q_sqrtm),
            num_derivatives=num_derivatives,
            ode_dimension=ode_dimension,
        )

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        if taylor_coefficients[0].shape[0] != self.ode_dimension:
            msg = "The solver's ODE dimension does not match the initial condition."
            raise ValueError(msg)

        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return _Normal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def init_error_estimate(self):  # noqa: D102
        return jnp.zeros((self.ode_dimension,))  # the initialisation is error-free

    def begin_extrapolation(self, m0, /, *, dt):  # noqa: D102
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * m0
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        return _Normal(m_ext, q_sqrtm), (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, *, dt):  # noqa: D102
        p, p_inv = _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        p = jnp.tile(p, self.ode_dimension)
        p_inv = jnp.tile(p_inv, self.ode_dimension)
        return p, p_inv

    def complete_extrapolation(  # noqa: D102
        self, *, linearisation_pt, l0, cache, output_scale_sqrtm
    ):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.mean
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(output_scale_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return _Normal(mean=m_ext, cov_sqrtm_lower=l_ext)

    def revert_markov_kernel(  # noqa: D102
        self, *, linearisation_pt, cache, l0, output_scale_sqrtm
    ):
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

        backward_op = g_bw
        backward_noise = _Normal(mean=m_bw, cov_sqrtm_lower=l_bw)
        extrapolated = _Normal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    def condense_backward_models(
        self, *, transition_init, noise_init, transition_state, noise_state
    ):

        A = transition_init
        (b, B_sqrtm) = noise_init.mean, noise_init.cov_sqrtm_lower

        C = transition_state
        (d, D_sqrtm) = (noise_state.mean, noise_state.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

        noise = _Normal(mean=xi, cov_sqrtm_lower=Xi)
        return noise, g

    def extract_sol(self, *, rv):  # noqa: D102
        if rv.mean.ndim == 1:
            return rv.mean.reshape((-1, self.ode_dimension), order="F")[0, ...]
        return jax.vmap(self.extract_sol)(rv=rv)

    def init_backward_transition(self):  # noqa: D102
        k = (self.num_derivatives + 1) * self.ode_dimension
        return jnp.eye(k)

    def init_backward_noise(self, *, rv_proto):  # noqa: D102
        return _Normal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )

    def init_output_scale_sqrtm(self):
        return 1.0

    def scale_covariance(self, *, rv, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return _Normal(
                mean=rv.mean, cov_sqrtm_lower=scale_sqrtm * rv.cov_sqrtm_lower
            )
        return _Normal(
            mean=rv.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * rv.cov_sqrtm_lower,
        )

    def marginalise_backwards(self, *, init, linop, noise):
        def body_fun(carry, x):
            op, noi = x
            out = self.marginalise_model(init=carry, linop=op, noise=noi)
            return out, out

        # Initial condition does not matter
        bw_models = jax.tree_util.tree_map(lambda x: x[1:, ...], (linop, noise))

        _, rvs = _control_flow.scan_with_init(
            f=body_fun, init=init, xs=bw_models, reverse=True
        )
        return rvs

    def marginalise_model(self, *, init, linop, noise):
        # todo: add preconditioner?

        # Pull into preconditioned space
        m0_p = init.mean
        l0_p = init.cov_sqrtm_lower

        # Apply transition
        m_new_p = linop @ m0_p + noise.mean
        l_new_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(linop @ l0_p).T, R2=noise.cov_sqrtm_lower.T
        ).T

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        return _Normal(mean=m_new, cov_sqrtm_lower=l_new)

    def sample_backwards(self, init, linop, noise, base_samples):
        def body_fun(carry, x):
            op, noi = x
            out = op @ carry + noi
            return out, out

        linop_sample, noise_ = jax.tree_util.tree_map(
            lambda x: x[1:, ...], (linop, noise)
        )
        noise_sample = self._transform_samples(noise_, base_samples[..., :-1, :])
        init_sample = self._transform_samples(init, base_samples[..., -1, :])

        # todo: should we use an associative scan here?
        _, samples = _control_flow.scan_with_init(
            f=body_fun, init=init_sample, xs=(linop_sample, noise_sample), reverse=True
        )
        return samples

    # automatically batched because of numpy's broadcasting rules?
    def _transform_samples(self, rvs, base):
        m, l_sqrtm = rvs.mean, rvs.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    def extract_mean_from_marginals(self, mean):
        if mean.ndim == 1:
            return mean.reshape((-1, self.ode_dimension), order="F")[0, ...]
        return jax.vmap(self.extract_mean_from_marginals)(mean)


def negative_marginal_log_likelihood(*, h, sigmas, data, posterior):

    bw_models = jax.tree_util.tree_map(lambda x: x[1:, ...], posterior.backward_model)
    init = jax.tree_util.tree_map(lambda x: x[-1, ...], posterior.init)

    def filter_step(carry, x):

        rv, num_data, mll = carry

        # Read
        bw_model, sigma, y = x
        A = bw_model.transition
        m_noise, l_noise = bw_model.noise.mean, bw_model.noise.cov_sqrtm_lower

        # Extrapolate
        m_ext = A @ rv.mean + m_noise
        l_ext = _sqrtm.sum_of_sqrtm_factors(
            R1=(A @ rv.cov_sqrtm_lower).T, R2=l_noise.T
        ).T

        # Correct
        hc = jax.vmap(h, in_axes=1, out_axes=1)(l_ext)
        m_obs = h(m_ext)
        d = y.shape[0]
        r_yx = sigma * jnp.eye(d)
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=l_ext.T, R_YX=r_yx
        )
        m_cor = m_ext - gain @ (m_obs - y)

        # Compute marginal log likelihood and go
        # todo: extract a log-likelihood function
        res_white = jsp.linalg.solve_triangular(r_obs, (m_obs - y), lower=False)
        x1 = jnp.dot(res_white, res_white.T)
        _, x2 = jnp.linalg.slogdet(r_obs)
        x2 = x2**2
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        mll_new = 0.5 * (x1 + x2 + x3)
        mll_updated = (num_data * mll + mll_new) / (num_data + 1)
        return (_Normal(m_cor, r_cor.T), num_data + 1, mll_updated), None

    # todo: this should return a Filtering posterior or a smoothing posterior
    #  which could then be plotted. Right?
    #  (We might also want some dense-output/checkpoint kind of thing here)
    # todo: we should reuse the extrapolation model implementations.
    #  But this only works if the ODE posterior uses the preconditioner (I think).
    # todo: we should allow proper noise, and proper information functions.
    #  But it is not clear which data structure that should be.
    #
    (_, _, mll), _ = jax.lax.scan(
        f=filter_step,
        init=(init, 0, 0.0),
        xs=(bw_models, sigmas[:-1], data[:-1]),
        reverse=True,
    )
    return mll
