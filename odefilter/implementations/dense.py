"""State-space models with dense covariance structure   ."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from jax.tree_util import register_pytree_node_class

from odefilter import _control_flow
from odefilter.implementations import _ibm, _implementation, _sqrtm


class MultivariateNormal(NamedTuple):
    """Random variable with a normal distribution."""

    mean: Array  # (k,) shape
    cov_sqrtm_lower: Array  # (k,k) shape


class _DenseInformationCommon(_implementation.Information):
    def evidence_sqrtm(self, *, observed):
        obs_pt, l_obs = observed.mean, observed.cov_sqrtm_lower
        res_white = jsp.linalg.solve_triangular(l_obs.T, obs_pt, lower=False)
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def _residual(self, x, *, t, p):
        x_reshaped = jnp.reshape(x, (-1, self.ode_dimension), order="F")

        x1 = x_reshaped[self.ode_order, ...]
        fx0 = self.f(*x_reshaped[: self.ode_order, ...], t=t, p=p)
        return x1 - fx0


@register_pytree_node_class
class EK1(_DenseInformationCommon):
    """Extended Kalman filter information."""

    def __init__(self, f, /, *, ode_order, ode_dimension):
        super().__init__(f, ode_order=ode_order)
        self.ode_dimension = ode_dimension

    def tree_flatten(self):
        children = ()
        aux = self.f, self.ode_order, self.ode_dimension
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        f, ode_order, ode_dimension = aux
        return cls(f, ode_order=ode_order, ode_dimension=ode_dimension)

    def begin_correction(self, x: MultivariateNormal, /, *, t, p):
        b, fn = jax.linearize(jax.tree_util.Partial(self._residual, t=t, p=p), x.mean)

        cov_sqrtm_lower = self._cov_sqrtm_lower(
            cache=(b, fn), cov_sqrtm_lower=x.cov_sqrtm_lower
        )

        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower.T).T
        output_scale_sqrtm = self.evidence_sqrtm(
            observed=MultivariateNormal(b, l_obs_raw)
        )
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
        observed = MultivariateNormal(mean=b, cov_sqrtm_lower=l_obs)
        corrected = MultivariateNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, gain)

    def _cov_sqrtm_lower(self, *, cache, cov_sqrtm_lower):
        _, fn = cache
        return jax.vmap(fn, in_axes=1, out_axes=1)(cov_sqrtm_lower)


@register_pytree_node_class
class CK1(_DenseInformationCommon):
    """Cubature Kalman filter information."""

    def __init__(
        self, f, /, *, sigma_weights_sqrtm, unit_sigma_points, ode_order, ode_dimension
    ):
        if ode_order > 1:
            raise ValueError

        super().__init__(f, ode_order=ode_order)
        self.ode_dimension = ode_dimension

        self.unit_sigma_points = unit_sigma_points
        self.sigma_weights_sqrtm = sigma_weights_sqrtm

    @classmethod
    def from_spherical_cubature_integration(cls, f, /, ode_order, ode_dimension):
        rule = _spherical_cubature_params(dim=ode_order * ode_dimension)
        return cls(
            f,
            ode_order=ode_order,
            ode_dimension=ode_dimension,
            unit_sigma_points=rule.points,
            sigma_weights_sqrtm=rule.weights_sqrtm,
        )

    def tree_flatten(self):
        children = (self.unit_sigma_points, self.sigma_weights_sqrtm)
        aux = self.f, self.ode_order, self.ode_dimension
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        pts, weights_sqrtm = children
        f, ode_order, ode_dimension = aux
        return cls(
            f,
            ode_order=ode_order,
            ode_dimension=ode_dimension,
            unit_sigma_points=pts,
            sigma_weights_sqrtm=weights_sqrtm,
        )

    def begin_correction(self, x: MultivariateNormal, /, *, t, p):
        # Getting it down to 2*d sigma points...

        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(self.f, t=t, p=p))
        e0v = jax.vmap(self._e0, in_axes=1, out_axes=1)
        e1v = jax.vmap(self._e1, in_axes=1, out_axes=1)
        cache = (vmap_f, e0v, e1v)

        # 1. x -> (e0x, e1x)
        R_X = x.cov_sqrtm_lower.T
        r_marg1_x = _sqrtm.sqrtm_to_upper_triangular(R=e0v(R_X.T).T)
        r_marg1_y = _sqrtm.sqrtm_to_upper_triangular(R=e1v(R_X.T).T)
        m_marg1_x, m_marg1_y = self._e0(x.mean), self._e1(x.mean)

        # 2. (x, y) -> (f(x), y)
        x_centered = self.unit_sigma_points @ r_marg1_x
        sigma_points = m_marg1_x[None, :] + x_centered
        fx = vmap_f(sigma_points)
        m_marg2 = self.sigma_weights_sqrtm**2 @ fx
        fx_centered = fx - m_marg2[None, :]
        fx_normed = fx_centered * self.sigma_weights_sqrtm[:, None]
        r_marg2 = _sqrtm.sqrtm_to_upper_triangular(R=fx_normed)

        # 3. (x, y) -> y - x (last one)
        m_marg = m_marg1_y - m_marg2
        l_marg = _sqrtm.sum_of_sqrtm_factors(R1=r_marg1_y, R2=r_marg2).T

        # Summarise
        marginals = MultivariateNormal(m_marg, l_marg)
        output_scale_sqrtm = self.evidence_sqrtm(observed=marginals)

        # Compute error estimate
        l_obs = marginals.cov_sqrtm_lower
        error_estimate = jnp.sqrt(jnp.einsum("nj,nj->n", l_obs, l_obs))
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, *, extrapolated, cache):
        # This is the cubature filter, optimised to quadrature-linearise
        # only the operator (x, y) -> y - f(x)
        # We can use even fewer sigma-points, but that is future work.
        (vmap_f, e0v, e1v) = cache

        # 1. x -> (e0x, e1x)
        marg1, (bw1, gain1) = self._step_1(x=extrapolated, cache=cache)

        # 2. (x, y) -> (f(x), y)
        marg2, (bw2, gain2) = self._step_2(x=extrapolated, marg1=marg1, cache=cache)

        # 3. (x, y) -> y - x
        marg3, (bw3, gain3) = self._step_3(marg2=marg2, cache=cache)

        # Condense backward models
        bw_transition = gain1 @ gain2 @ gain3
        bw_noise_mean = gain1 @ (gain2 @ bw3.mean + bw2.mean) + bw1.mean
        x = _sqrtm.sum_of_sqrtm_factors(
            R1=bw3.cov_sqrtm_lower.T @ gain2.T, R2=bw2.cov_sqrtm_lower.T
        )
        bw_noise_cov_sqrtm = _sqrtm.sum_of_sqrtm_factors(
            R1=x @ gain1.T, R2=bw1.cov_sqrtm_lower.T
        )
        bw_noise = MultivariateNormal(bw_noise_mean, bw_noise_cov_sqrtm.T)

        marginals = marg3
        return marginals, (bw_noise, bw_transition)

    def _step_1(self, *, x, cache):
        (vmap_f, e0v, e1v) = cache

        R_X = x.cov_sqrtm_lower.T  # (n, n)
        R_X_F = jnp.hstack((e0v(R_X.T).T, e1v(R_X.T).T))  # (n, 2*d)
        r_marg1, (r_bw1, gain1) = _sqrtm.revert_conditional_noisefree(
            R_X_F=R_X_F, R_X=R_X
        )  # (2*d, 2*d), (n, n), (n, 2*d)
        m_marg1 = jnp.hstack((self._e0(x.mean), self._e1(x.mean)))  # (2*d,)
        m_bw1 = x.mean - gain1 @ m_marg1

        marginals = MultivariateNormal(m_marg1, r_marg1.T)
        backward_noise = MultivariateNormal(m_bw1, r_bw1.T)
        return marginals, (backward_noise, gain1)

    def _step_2(self, *, x, marg1, cache):

        (vmap_f, e0v, e1v) = cache

        R_X = x.cov_sqrtm_lower.T

        r_marg1_x = _sqrtm.sqrtm_to_upper_triangular(R=e0v(R_X.T).T)
        m_marg1_x = self._e0(x.mean)
        m_marg1_y = self._e1(x.mean)
        x_centered = self.unit_sigma_points @ r_marg1_x
        x_normed = x_centered * self.sigma_weights_sqrtm[:, None]  # (S, 2*d)
        sigma_points = m_marg1_x[None, :] + x_centered
        fx = vmap_f(sigma_points)
        m_marg2 = self.sigma_weights_sqrtm**2 @ fx
        fx_centered = fx - m_marg2[None, :]
        fx_normed = fx_centered * self.sigma_weights_sqrtm[:, None]
        r_marg2_x, (r_bw2_x, gain2_x) = _sqrtm.revert_conditional_noisefree(
            R_X_F=fx_normed, R_X=x_normed
        )
        m_bw2_x = m_marg1_x - gain2_x @ m_marg2

        gain2 = jsp.linalg.block_diag(gain2_x, jnp.eye(*gain2_x.shape))
        m_bw2 = jnp.hstack((m_bw2_x, jnp.zeros_like(m_bw2_x)))
        r_bw2 = jsp.linalg.block_diag(r_bw2_x, jnp.zeros_like(r_bw2_x))

        # marginal: H := crosscov @ Pinv; multiply the original cov
        # with (H, I) from left and right
        # Eq. (9) in https://arxiv.org/abs/2102.00514
        H = (
            jnp.linalg.inv(r_marg1_x.T @ r_marg1_x)
            @ gain2_x
            @ (r_marg2_x.T @ r_marg2_x)
        )
        HH = jsp.linalg.block_diag(H, jnp.eye(*H.shape))
        r_marg2 = marg1.cov_sqrtm_lower.T @ HH  # todo: what about the transpose?!
        m_marg2_full = jnp.hstack((m_marg2, m_marg1_y))

        marginals = MultivariateNormal(m_marg2_full, r_marg2.T)
        backward_noise = MultivariateNormal(m_bw2, r_bw2.T)

        return marginals, (backward_noise, gain2)

    def _step_3(self, *, marg2, cache):

        m0, m1 = jnp.split(marg2.mean, indices_or_sections=2, axis=0)
        r0, r1 = jnp.split(marg2.cov_sqrtm_lower.T, indices_or_sections=2, axis=1)

        mean = m1 - m0
        R_X_F = r1 - r0
        r_marg, (r_bw3, gain3) = _sqrtm.revert_conditional_noisefree(
            R_X_F=R_X_F, R_X=marg2.cov_sqrtm_lower.T
        )
        m_bw = marg2.mean - gain3 @ mean

        marginals = MultivariateNormal(mean, r_marg.T)
        backward_noise = MultivariateNormal(m_bw, r_bw3.T)

        return marginals, (backward_noise, gain3)

    def _e1(self, x):
        return x.reshape((-1, self.ode_dimension), order="F")[1, ...]

    def _e0(self, x):
        return x.reshape((-1, self.ode_dimension), order="F")[0, ...]


class _PositiveCubatureRule(NamedTuple):
    """Cubature rule with positive weights."""

    points: Array
    weights_sqrtm: Array


def _spherical_cubature_params(*, dim):
    eye_d = jnp.eye(dim) * jnp.sqrt(dim)
    pts = jnp.vstack((eye_d, -1 * eye_d))
    weights_sqrtm = jnp.ones((2 * dim,)) / jnp.sqrt(2.0 * dim)
    return _PositiveCubatureRule(points=pts, weights_sqrtm=weights_sqrtm)


@register_pytree_node_class
@dataclass(frozen=True)
class DenseImplementation(_implementation.Implementation):
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
    def from_num_derivatives(cls, *, num_derivatives, ode_dimension):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm.system_matrices_1d(num_derivatives=num_derivatives)
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
        return MultivariateNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def init_error_estimate(self):  # noqa: D102
        return jnp.zeros((self.ode_dimension,))  # the initialisation is error-free

    def begin_extrapolation(self, m0, /, *, dt):  # noqa: D102
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * m0
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        return MultivariateNormal(m_ext, q_sqrtm), (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, *, dt):  # noqa: D102
        p, p_inv = _ibm.preconditioner_diagonal(
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
        return MultivariateNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

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
        backward_noise = MultivariateNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        extrapolated = MultivariateNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    # todo: this must not operate on the backward model data structure
    def condense_backward_models(self, *, bw_init, bw_state):
        A = bw_init.transition
        (b, B_sqrtm) = bw_init.noise.mean, bw_init.noise.cov_sqrtm_lower
        C = bw_state.transition
        (d, D_sqrtm) = (bw_state.noise.mean, bw_state.noise.cov_sqrtm_lower)
        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T
        noise = MultivariateNormal(mean=xi, cov_sqrtm_lower=Xi)
        return noise, g

    def extract_sol(self, *, rv):  # noqa: D102
        if rv.mean.ndim == 1:
            return rv.mean.reshape((-1, self.ode_dimension), order="F")[0, ...]
        return jax.vmap(self.extract_sol)(rv=rv)

    def init_backward_transition(self):  # noqa: D102
        k = (self.num_derivatives + 1) * self.ode_dimension
        return jnp.eye(k)

    def init_backward_noise(self, *, rv_proto):  # noqa: D102
        return MultivariateNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )

    def init_output_scale_sqrtm(self):
        return 1.0

    def scale_covariance(self, *, rv, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return MultivariateNormal(
                mean=rv.mean, cov_sqrtm_lower=scale_sqrtm * rv.cov_sqrtm_lower
            )
        return MultivariateNormal(
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

        return MultivariateNormal(mean=m_new, cov_sqrtm_lower=l_new)

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
