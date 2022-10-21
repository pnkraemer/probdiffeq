"""Inference via fixed-point smoothing."""

from dataclasses import dataclass

import jax
import jax.tree_util

from odefilter.strategies import _interface, _markov


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicFixedPointSmoother(_interface.DynamicSmootherCommon):
    """Smoother implementation with dynamic calibration (time-varying diffusion)."""

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )

        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(rv_proto=corrected)
        backward_model = _markov.BackwardModel(
            transition=backward_transition,
            noise=backward_noise,
        )
        sol = self.implementation.extract_sol(rv=corrected)

        solution = _markov.Posterior(
            t=t0,
            t_previous=t0,
            u=sol,
            marginals_filtered=corrected,
            marginals=None,
            diffusion_sqrtm=1.0,
            backward_model=backward_model,
        )

        error_estimate = self.implementation.init_error_estimate()
        return solution, error_estimate

    @jax.jit
    def step_fn(self, *, state, info_op, dt, parameters):
        """Step."""
        # Assemble preconditioner
        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.marginals_filtered.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation and estimate error.
        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)
        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p=p
        )
        error_estimate = dt * diffusion_sqrtm * error_estimate

        # Complete extrapolation and condense backward models
        x = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=state.marginals_filtered.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        extrapolated, (backward_noise, backward_op) = x
        bw_increment = _markov.BackwardModel(
            transition=backward_op, noise=backward_noise
        )
        noise, gain = self.implementation.condense_backward_models(
            bw_state=bw_increment,
            bw_init=state.backward_model,
        )
        backward_model = _markov.BackwardModel(transition=gain, noise=noise)

        # Final observation
        _, (corrected, _) = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Extract and return solution
        sol = self.implementation.extract_sol(rv=corrected)
        smoothing_solution = _markov.Posterior(
            t=state.t + dt,
            t_previous=state.t_previous,  # condensing the models...
            u=sol,
            marginals_filtered=corrected,
            marginals=None,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model,
        )

        return smoothing_solution, error_estimate

    def _case_right_corner(self, *, s0, s1, t):  # s1.t == t
        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?

        backward_model1 = s1.backward_model
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.backward_model, bw_state=backward_model1
        )
        backward_model1 = _markov.BackwardModel(transition=g0, noise=noise0)
        solution = _markov.Posterior(
            t=t,
            t_previous=s0.t_previous,  # condensed the model...
            u=s1.u,
            marginals_filtered=s1.marginals_filtered,
            marginals=None,
            backward_model=backward_model1,
            diffusion_sqrtm=s1.diffusion_sqrtm,
        )

        accepted = self._duplicate_with_unit_backward_model(state=solution, t=t)
        previous = accepted

        return accepted, solution, previous

    def _case_interpolate(self, *, s0, s1, t):  # noqa: D102
        # A fixed-point smoother interpolates almost like a smoother.
        # The key difference is that when interpolating from s0.t to t,
        # the backward models in s0.t and the incoming model are condensed into one.
        # The reasoning is that the previous model "knows how to get to the
        # quantity of interest", and this is what we are interested in.
        # The rest remains the same as for the smoother.

        # Use the s1.diffusion as a diffusion over the interval.
        # Filtering/smoothing solutions are right-including intervals.
        diffusion_sqrtm = s1.diffusion_sqrtm

        # From s0.t to t
        extrapolated0, bw0 = self._interpolate_from_to_fn(
            rv=s0.marginals_filtered, diffusion_sqrtm=diffusion_sqrtm, t=t, t0=s0.t
        )
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.backward_model, bw_state=bw0
        )
        backward_model0 = _markov.BackwardModel(transition=g0, noise=noise0)
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = _markov.Posterior(
            t=t,
            t_previous=s0.t_previous,  # condensed the model...
            u=sol,
            marginals_filtered=extrapolated0,
            marginals=None,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model0,
        )

        # new model! no condensing...
        previous = self._duplicate_with_unit_backward_model(state=solution, t=t)

        # From t to s1.t
        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, diffusion_sqrtm=diffusion_sqrtm, t=s1.t, t0=t
        )
        accepted = _markov.Posterior(
            t=s1.t,
            t_previous=t,  # new model! No condensing...
            u=s1.u,
            marginals_filtered=s1.marginals_filtered,
            marginals=None,
            diffusion_sqrtm=diffusion_sqrtm,
            backward_model=backward_model1,
        )
        return accepted, solution, previous

    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError
