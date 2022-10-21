"""Inference via fixed-point smoothing."""

from dataclasses import dataclass

import jax
import jax.tree_util

from odefilter.strategies import _common


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DynamicFixedPointSmoother(_common.DynamicSmootherCommon):
    """Smoother implementation with dynamic calibration (time-varying output-scale)."""

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise."""
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )

        backward_transition = self.implementation.init_backward_transition()
        backward_noise = self.implementation.init_backward_noise(rv_proto=corrected)
        backward_model = _common.BackwardModel(
            transition=backward_transition,
            noise=backward_noise,
        )
        sol = self.implementation.extract_sol(rv=corrected)

        posterior = _common.MarkovSequence(
            init=corrected, backward_model=backward_model
        )
        solution = _common.Solution(
            t=t0,
            t_previous=t0,
            u=sol,
            posterior=posterior,
            marginals=None,
            output_scale_sqrtm=1.0,
            num_data_points=1.0,
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
            state.posterior.init.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation and estimate error.
        m_obs, linear_fn = info_op(x=m_ext, t=state.t + dt, p=parameters)
        output_scale_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p=p
        )
        error_estimate = dt * output_scale_sqrtm * error_estimate

        # Complete extrapolation and condense backward models
        extrapolated = self._complete_extrapolation(
            bw_model_previous=state.posterior.backward_model,
            l0=state.posterior.init.cov_sqrtm_lower,
            m0_p=m0_p,
            m_ext=m_ext,
            m_ext_p=m_ext_p,
            output_scale_sqrtm=output_scale_sqrtm,
            p=p,
            p_inv=p_inv,
        )

        # Final observation
        _, (corrected, _) = self._final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )

        # Extract and return solution
        sol = self.implementation.extract_sol(rv=corrected.init)
        smoothing_solution = _common.Solution(
            t=state.t + dt,
            t_previous=state.t_previous,  # condensing the models...
            u=sol,
            posterior=corrected,
            marginals=None,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=state.num_data_points + 1.0,
        )

        return smoothing_solution, error_estimate

    def _final_correction(self, *, extrapolated, linear_fn, m_obs):
        a, (corrected, b) = self.implementation.final_correction(
            extrapolated=extrapolated.init, linear_fn=linear_fn, m_obs=m_obs
        )
        corrected_seq = _common.MarkovSequence(
            init=corrected, backward_model=extrapolated.backward_model
        )
        return a, (corrected_seq, b)

    def _complete_extrapolation(
        self,
        *,
        bw_model_previous,
        l0,
        m0_p,
        m_ext,
        m_ext_p,
        output_scale_sqrtm,
        p,
        p_inv
    ):
        x = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=l0,
            p=p,
            p_inv=p_inv,
            output_scale_sqrtm=output_scale_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        extrapolated, (backward_noise, backward_op) = x
        bw_increment = _common.BackwardModel(
            transition=backward_op, noise=backward_noise
        )
        noise, gain = self.implementation.condense_backward_models(
            bw_state=bw_increment,
            bw_init=bw_model_previous,
        )
        backward_model = _common.BackwardModel(transition=gain, noise=noise)
        return _common.MarkovSequence(init=extrapolated, backward_model=backward_model)

    def _case_right_corner(self, *, s0, s1, t):  # s1.t == t
        # can we guarantee that the backward model in s1 is the
        # correct backward model to get from s0 to s1?

        backward_model1 = s1.posterior.backward_model
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.posterior.backward_model, bw_state=backward_model1
        )
        backward_model1 = _common.BackwardModel(transition=g0, noise=noise0)
        posterior1 = _common.MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )
        solution = _common.Solution(
            t=t,
            t_previous=s0.t_previous,  # condensed the model...
            u=s1.u,
            posterior=posterior1,
            marginals=None,
            output_scale_sqrtm=s1.output_scale_sqrtm,
            num_data_points=s1.num_data_points,
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

        # Use the s1.output-scale as a output-scale over the interval.
        # Filtering/smoothing solutions are right-including intervals.
        output_scale_sqrtm = s1.output_scale_sqrtm

        # From s0.t to t
        extrapolated0, bw0 = self._interpolate_from_to_fn(
            rv=s0.posterior.init,
            output_scale_sqrtm=output_scale_sqrtm,
            t=t,
            t0=s0.t,
        )
        noise0, g0 = self.implementation.condense_backward_models(
            bw_init=s0.posterior.backward_model, bw_state=bw0
        )
        backward_model0 = _common.BackwardModel(transition=g0, noise=noise0)
        posterior0 = _common.MarkovSequence(
            init=extrapolated0, backward_model=backward_model0
        )
        sol = self.implementation.extract_sol(rv=extrapolated0)
        solution = _common.Solution(
            t=t,
            t_previous=s0.t_previous,  # condensed the model...
            u=sol,
            posterior=posterior0,
            marginals=None,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )

        # new model! no condensing...
        previous = self._duplicate_with_unit_backward_model(state=solution, t=t)

        # From t to s1.t
        _, backward_model1 = self._interpolate_from_to_fn(
            rv=extrapolated0, output_scale_sqrtm=output_scale_sqrtm, t=s1.t, t0=t
        )
        posterior1 = _common.MarkovSequence(
            init=s1.posterior.init, backward_model=backward_model1
        )
        accepted = _common.Solution(
            t=s1.t,
            t_previous=t,  # new model! No condensing...
            u=s1.u,
            posterior=posterior1,
            marginals=None,
            output_scale_sqrtm=output_scale_sqrtm,
            num_data_points=s1.num_data_points,
        )
        return accepted, solution, previous

    def offgrid_marginals(self, *, t, state, state_previous):
        raise NotImplementedError
