"""ODE filter strategy implementations."""

from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float

from odefilter import sqrtm


class IsotropicNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, "n d"]
    cov_sqrtm_lower: Float[Array, "n n"]


class MultivariateNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, " k"]
    cov_sqrtm_lower: Float[Array, "k k"]


class IsotropicImplementation(eqx.Module):
    """Handle isotropic covariances."""

    a: Any
    q_sqrtm_lower: Any

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    @property
    def num_derivatives(self):
        """Number of derivatives in the state-space model."""  # noqa: D401
        return self.a.shape[0] - 1

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_corrected = jnp.vstack(taylor_coefficients)
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return IsotropicNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    @staticmethod
    def init_error_estimate():  # noqa: D102
        return jnp.empty(())

    def init_backward_transition(self):  # noqa: D102
        return jnp.eye(*self.a.shape)

    def init_backward_noise(self, *, rv_proto):  # noqa: D102
        shape_m = rv_proto.mean.shape
        shape_l = rv_proto.cov_sqrtm_lower.shape
        return IsotropicNormal(
            mean=jnp.zeros(shape_m), cov_sqrtm_lower=jnp.zeros(shape_l)
        )

    def assemble_preconditioner(self, *, dt):  # noqa: D102
        return preconditioner_diagonal(dt=dt, num_derivatives=self.num_derivatives)

    def extrapolate_mean(self, m0, /, *, p, p_inv):  # noqa: D102
        m0_p = p_inv[:, None] * m0
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        return m_ext, m_ext_p, m0_p

    def estimate_error(self, *, linear_fn, m_obs, p_inv):  # noqa: D102
        l_obs_raw = linear_fn(p_inv[:, None] * self.q_sqrtm_lower)
        c_obs_raw = jnp.dot(l_obs_raw, l_obs_raw)
        res_white = m_obs / jnp.sqrt(c_obs_raw)
        diffusion_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white) / res_white.size)
        error_estimate = diffusion_sqrtm * jnp.sqrt(c_obs_raw)
        return diffusion_sqrtm, error_estimate

    def complete_extrapolation(  # noqa: D102
        self, *, m_ext, l0, p_inv, p, diffusion_sqrtm
    ):
        l_ext_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return IsotropicNormal(m_ext, l_ext)

    def revert_markov_kernel(  # noqa: D102
        self, *, m_ext, l0, p, p_inv, diffusion_sqrtm, m0_p, m_ext_p
    ):
        l0_p = p_inv[:, None] * l0
        r_ext_p, (r_bw_p, g_bw_p) = sqrtm.revert_gauss_markov_correlation(
            R_X_F=(self.a @ l0_p).T,
            R_X=l0_p.T,
            R_YX=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        )
        l_ext_p, l_bw_p = r_ext_p.T, r_bw_p.T
        m_bw_p = m0_p - g_bw_p @ m_ext_p

        # Un-apply the pre-conditioner
        l_ext = p[:, None] * l_ext_p
        m_bw, l_bw = p[:, None] * m_bw_p, p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]
        backward_op = g_bw
        backward_noise = IsotropicNormal(m_bw, l_bw)
        extrapolated = IsotropicNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    @staticmethod
    def final_correction(*, extrapolated, linear_fn, m_obs):  # noqa: D102
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs = linear_fn(l_ext)  # shape (n,)
        c_obs = jnp.dot(l_obs, l_obs)
        g = (l_ext @ l_obs.T) / c_obs  # shape (n,)
        m_cor = m_ext - g[:, None] * m_obs[None, :]
        l_cor = l_ext - g[:, None] * l_obs[None, :]
        corrected = IsotropicNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return corrected, (corrected.mean[0])

    @staticmethod
    def condense_backward_models(*, bw_init, bw_state):  # noqa: D102
        A = bw_init.transition
        (b, B_sqrtm) = bw_init.noise.mean, bw_init.noise.cov_sqrtm_lower

        C = bw_state.transition
        (d, D_sqrtm) = bw_state.noise.mean, bw_state.noise.cov_sqrtm_lower

        g = A @ C
        xi = A @ d + b
        Xi = sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

        noise = IsotropicNormal(mean=xi, cov_sqrtm_lower=Xi)
        return noise, g

    @staticmethod
    def marginalise_backwards(*, init, backward_model):
        """Compute marginals of a markov sequence."""

        def body_fun(carry, x):
            linop, noise = x.transition, x.noise
            out = IsotropicImplementation.marginalise_model_isotropic(
                init=carry, linop=linop, noise=noise
            )
            return out, out

        _, rvs = jax.lax.scan(f=body_fun, init=init, xs=backward_model, reverse=False)
        return rvs

    @staticmethod
    def marginalise_model_isotropic(*, init, linop, noise):
        """Marginalise the output of a linear model."""
        # Apply transition
        m_new = jnp.dot(linop, init.mean) + noise.mean
        l_new = sqrtm.sum_of_sqrtm_factors(
            R1=jnp.dot(linop, init.cov_sqrtm_lower).T, R2=noise.cov_sqrtm_lower.T
        ).T

        return IsotropicNormal(mean=m_new, cov_sqrtm_lower=l_new)


class DenseImplementation(eqx.Module):
    """Handle dense covariances."""

    a: Any
    q_sqrtm_lower: Any

    num_derivatives: int
    ode_dimension: int

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives, ode_dimension):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = system_matrices_1d(num_derivatives=num_derivatives)
        eye_d = jnp.eye(ode_dimension)
        return cls(
            a=jnp.kron(eye_d, a),
            q_sqrtm_lower=jnp.kron(eye_d, q_sqrtm),
            num_derivatives=num_derivatives,
            ode_dimension=ode_dimension,
        )

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return MultivariateNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def init_error_estimate(self):  # noqa: D102
        return jnp.empty((self.ode_dimension,))

    def init_backward_transition(self):  # noqa: D102
        raise NotImplementedError

    def assemble_preconditioner(self, *, dt):  # noqa: D102
        p, p_inv = preconditioner_diagonal(dt=dt, num_derivatives=self.num_derivatives)
        p = jnp.tile(p, self.ode_dimension)
        p_inv = jnp.tile(p_inv, self.ode_dimension)
        return p, p_inv

    def extrapolate_mean(self, m0, /, *, p, p_inv):  # noqa: D102
        m0_p = p_inv * m0
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        return m_ext, m_ext_p, m0_p

    def estimate_error(self, *, linear_fn, m_obs, p_inv):  # noqa: D102
        l_obs_nonsquare = jax.vmap(linear_fn, in_axes=1, out_axes=1)(
            p_inv[:, None] * self.q_sqrtm_lower
        )
        l_obs_raw = sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T
        res_white = jsp.linalg.solve_triangular(l_obs_raw.T, m_obs, lower=False)
        diffusion_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        error_estimate = diffusion_sqrtm * jnp.sqrt(
            jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw)
        )
        return diffusion_sqrtm, error_estimate

    def complete_extrapolation(  # noqa: D102
        self, *, m_ext, l0, p_inv, p, diffusion_sqrtm
    ):
        l_ext_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return MultivariateNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

    def revert_markov_kernel(  # noqa: D102
        self, *, m_ext, l0, p, p_inv, diffusion_sqrtm, m0_p, m_ext_p
    ):
        raise NotImplementedError

    def final_correction(self, *, extrapolated, linear_fn, m_obs):  # noqa: D102
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs_nonsquare = jax.vmap(linear_fn, in_axes=1, out_axes=1)(l_ext)

        l_obs = sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T
        crosscov = l_ext @ l_obs_nonsquare.T
        gain = jsp.linalg.cho_solve((l_obs, True), crosscov.T).T

        m_cor = m_ext - gain @ m_obs
        l_cor = l_ext - gain @ l_obs_nonsquare
        corrected = MultivariateNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        u = m_cor.reshape((-1, self.ode_dimension), order="F")[0]
        return corrected, u


def system_matrices_1d(*, num_derivatives):
    """Construct the IBM system matrices."""
    x = jnp.arange(num_derivatives + 1)

    A_1d = jnp.flip(_pascal(x)[0])  # no idea why the [0] is necessary...
    Q_1d = jnp.flip(_hilbert(x))
    return A_1d, jnp.linalg.cholesky(Q_1d)


def preconditioner(*, dt, num_derivatives):
    """Construct the IBM preconditioner."""
    p, p_inv = preconditioner_diagonal(dt=dt, num_derivatives=num_derivatives)
    return jnp.diag(p), jnp.diag(p_inv)


def preconditioner_diagonal(*, dt, num_derivatives):
    """Construct the diagonal of the IBM preconditioner."""
    powers = jnp.arange(num_derivatives, -1, -1)

    scales = _factorial(powers)
    powers = powers + 0.5

    scaling_vector = (jnp.abs(dt) ** powers) / scales
    scaling_vector_inv = (jnp.abs(dt) ** (-powers)) * scales

    return scaling_vector, scaling_vector_inv


@partial(jax.vmap, in_axes=(0, None), out_axes=(0, 0))
def preconditioner_diagonal_batched(dts, num_derivatives):
    """Compute the diagonal preconditioner, but for a number of time-steps at once."""
    return preconditioner_diagonal(dt=dts, num_derivatives=num_derivatives)


def _hilbert(a):
    return 1 / (a[:, None] + a[None, :] + 1)


def _pascal(a):
    return _binom(a[:, None], a[None, :])


def _batch_gram(k):
    k_vmapped_x = jax.vmap(k, in_axes=(0, None), out_axes=-1)
    k_vmapped_xy = jax.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=-1)
    return jax.jit(k_vmapped_xy)


@_batch_gram
def _binom(n, k):
    a = _factorial(n)
    b = _factorial(n - k)
    c = _factorial(k)
    return a / (b * c)


def _factorial(n):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))
