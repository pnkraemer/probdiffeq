"""State-space models with dense covariance structure   ."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import register_pytree_node_class

from odefilter import _control_flow
from odefilter.implementations import _ibm, _implementation, _sqrtm


class MultivariateNormal(NamedTuple):
    """Random variable with a normal distribution."""

    mean: Any  # (k,) shape
    cov_sqrtm_lower: Any  # (k,k) shape


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

        l_obs = _sqrtm.sqrtm_to_upper_triangular(R=l_obs_nonsquare.T).T
        observed = MultivariateNormal(mean=b, cov_sqrtm_lower=l_obs)

        crosscov = l_ext @ l_obs_nonsquare.T
        gain = jsp.linalg.cho_solve((l_obs, True), crosscov.T).T

        m_cor = m_ext - gain @ b
        l_cor = l_ext - gain @ l_obs_nonsquare
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
    def from_spherical_cubature_integration(
        cls, f, /, ode_order, ode_dimension, num_derivatives
    ):
        # todo: make more efficient!
        #  the number of derivatives in the prior should _not_ be relevant
        k = ode_dimension * 2
        rule = _spherical_cubature_params(dim=ode_order * k)
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
        cache = (vmap_f,)

        marginals, _ = self.complete_correction(cache=cache, extrapolated=x)
        output_scale_sqrtm = self.evidence_sqrtm(observed=marginals)
        error_estimate = jnp.sqrt(
            jnp.einsum("nj,nj->n", marginals.cov_sqrtm_lower, marginals.cov_sqrtm_lower)
        )
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, *, extrapolated, cache):
        # This is the cubature filter, optimised to quadrature-linearise
        # only the operator (x, y) -> y - f(x)
        # We can use even fewer sigma-points, but that is future work.

        (vmap_f,) = cache
        e1v = jax.vmap(self._e1, in_axes=1, out_axes=1)  # cov
        e0v = jax.vmap(self._e0, in_axes=1, out_axes=1)  # cov

        # 1. x -> (E0x, E1x)
        x = extrapolated
        R_X = x.cov_sqrtm_lower.T  # (n, n)
        R_X_F = jnp.hstack((e0v(R_X.T).T, e1v(R_X.T).T))  # (n, 2*d)
        R_YX = jnp.zeros((2 * self.ode_dimension, 2 * self.ode_dimension))  # (2*d, 2*d)
        print(R_X.shape, R_X_F.shape, R_YX.shape)
        r_marg1, (r_bw1, gain1) = _sqrtm.revert_gauss_markov_correlation(
            R_X_F=R_X_F, R_X=R_X, R_YX=R_YX
        )  # (2*d, 2*d), (n, n), (n, 2*d)
        m_marg1 = jnp.hstack((self._e0(x.mean), self._e1(x.mean)))  # (2*d,)
        m_bw1 = x.mean - gain1 @ m_marg1

        # 2. (x,y) -> (y - f(x))
        x_centered = self.unit_sigma_points @ r_marg1  # (S, 2*d)
        x_normed = x_centered * self.sigma_weights_sqrtm[:, None]  # (S, 2*d)
        sigma_points = m_marg1[None, :] + x_centered  # (1, 2*d) + (S, 2*d) = (S, 2*d)
        s0, s1 = jnp.split(
            sigma_points, indices_or_sections=2, axis=1
        )  # (S, d), (S, d)
        fx = s1 - vmap_f(s0)  # (S, d)
        b = self.sigma_weights_sqrtm**2 @ fx  # (S,) @ (S, d) = (d,)
        fx_centered = fx - b[None, :]  # (S, d) - (1, d) = (S, d)
        fx_normed = fx_centered * self.sigma_weights_sqrtm[:, None]  # (S, d)
        R_X_F = fx_normed  # (S, d)
        R_X = x_normed  # (S, 2*d)
        R_YX = jnp.zeros((self.ode_dimension, self.ode_dimension))  # (d, d)
        marginals2, (r_bw2, gain2) = _sqrtm.revert_gauss_markov_correlation(
            R_X_F=R_X_F, R_X=R_X, R_YX=R_YX
        )  # (d, d), (2*d, 2*d), (2*d, d)
        m_marg2 = b
        m_bw2 = m_marg1 - gain2 @ m_marg2

        # Condense models
        marginal_mean, marginal_cov_sqrtm = b, marginals2
        bw_transition = gain1 @ gain2
        bw_noise_mean = gain1 @ m_bw2 + m_bw1
        bw_noise_cov_sqrtm = _sqrtm.sum_of_sqrtm_factors(R1=r_bw2 @ gain1.T, R2=r_bw1).T

        # Return values
        marginals = MultivariateNormal(marginal_mean, marginal_cov_sqrtm)
        bw_noise = MultivariateNormal(bw_noise_mean, bw_noise_cov_sqrtm)
        return marginals, (bw_noise, bw_transition)

    def _e1(self, x):
        return x.reshape((-1, self.ode_dimension), order="F")[1, ...]

    def _e0(self, x):
        return x.reshape((-1, self.ode_dimension), order="F")[0, ...]


class _PositiveCubatureRule(NamedTuple):
    points: Any
    weights_sqrtm: Any


def _spherical_cubature_params(*, dim):
    eye_d = jnp.eye(dim) * jnp.sqrt(dim)
    pts = jnp.vstack((eye_d, -1 * eye_d))
    weights_sqrtm = jnp.sqrt(jnp.ones((2 * dim,)) / (2 * dim))
    return _PositiveCubatureRule(points=pts, weights_sqrtm=weights_sqrtm)


@register_pytree_node_class
@dataclass(frozen=True)
class DenseImplementation(_implementation.Implementation):
    """Handle dense covariances."""

    a: Any
    q_sqrtm_lower: Any

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
        r_ext_p, (r_bw_p, g_bw_p) = _sqrtm.revert_gauss_markov_correlation(
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
