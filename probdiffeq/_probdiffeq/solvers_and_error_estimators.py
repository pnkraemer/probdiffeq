"""Solvers."""

from probdiffeq import ssm_impl
from probdiffeq._probdiffeq import constraints, estimators_and_loss_functions, utilities
from probdiffeq.backend import func, linalg, np, structs, tree
from probdiffeq.backend.typing import Any, Array, Callable, Generic, TypeVar

__all__ = [
    "ErrorEstimator",
    "ProbabilisticSolution",
    "ProbabilisticSolver",
    "error_norm_rms_then_scale",
    "error_norm_scale_then_rms",
    "error_residual_std",
    "error_state_std",
    "solver",
    "solver_dynamic",
    "solver_mle",
]


N = TypeVar("N", bound=ssm_impl.AbstractTreeNormal)
"""A type-variable to describe normal distributions.

Used to type marginals, for example.
"""

T = TypeVar(
    "T",
    bound=estimators_and_loss_functions.MarkovSequence | ssm_impl.AbstractTreeNormal,
)
"""A type-variable to describe posterior distributions."""


@tree.register_dataclass
@structs.dataclass
class ProbabilisticSolution(Generic[N, T]):
    """A datastructure for probabilistic solutions of differential equations."""

    t: Array
    """The current time-step."""

    u: N
    """The current ODE solution estimate."""

    solution_full: T
    """The current posterior estimate."""

    # Todo: merge 'output_scale' and 'auxiliary' and "fun_evals"?
    output_scale: Any
    """The current output scale."""

    num_steps: Array
    """The number of steps taken until the current point."""

    auxiliary: Any
    """Auxiliary states.

    For instance, random keys for Hutchinson-based
    diagonal linearisation in the correction,
    or running means of the MLE calibration.
    """

    fun_evals: Any
    """Function evaluations.

    Used to cache observation models between solver steps
    and error estimates.
    """


class ProbabilisticSolver:
    """An interface for probabilistic differential equation solvers.

    Related:
    [`solver`](#probdiffeq.probdiffeq.solver),
    [`solver_mle`](#probdiffeq.probdiffeq.solver_mle),
    [`solver_dynamic`](#probdiffeq.probdiffeq.solver_dynamic).

    """

    def __init__(
        self,
        *,
        strategy: estimators_and_loss_functions.MarkovStrategy,
        prior: Callable,
        constraint: constraints.Constraint,
        constraint_init: constraints.Constraint | None,
    ) -> None:
        self.strategy = strategy
        self.prior = prior
        self.constraint = constraint
        self.constraint_init = constraint_init

    @property
    def is_suitable_for_offgrid_marginals(self):
        """Whether the solver admits offgrid-marginal calculation.

        Excludes fixed-point smoothers, for example.
        """
        return self.strategy.is_suitable_for_offgrid_marginals

    @property
    def is_suitable_for_save_at(self):
        """Whether the solver admits adaptive time-stepping with checkpoints.

        Excludes fixed-interval smoothers, for example.
        """
        return self.strategy.is_suitable_for_save_at

    @property
    def is_suitable_for_save_every_step(self):
        """Whether the solver admits adaptive time-stepping without checkpoints.

        Excludes fixed-point smoothers, for example.
        """
        return self.strategy.is_suitable_for_save_every_step

    def init(self, t, init, *, damp: float) -> ProbabilisticSolution:
        """Initialize the probabilistic solution."""
        raise NotImplementedError

    def step(self, state: ProbabilisticSolution, *, dt: float, damp: float):
        """Perform a step."""
        raise NotImplementedError

    def userfriendly_output(
        self, *, solution0: ProbabilisticSolution, solution: ProbabilisticSolution
    ) -> ProbabilisticSolution:
        """Make the solutions 'user-friendly'.

        This may include calibration, calculation of marginals, and other things.
        """
        raise NotImplementedError

    def offgrid_marginals(self, t, *, solution):
        """Compute off-grid marginals via jax.numpy.searchsorted.

        !!! warning
            The time-point 't' may not be an element in the solution grid.
            Otherwise, anything can happen and the solution will be incorrect.
            At the moment, we do not check this.

        !!! warning
            The time-point 't' must also be strictly in (t0, t1).
            It must not lie outside the interval, and it must not coincide
            with the interval boundaries.
            At the moment, we do not check this.
        """
        assert t.shape == solution.t[0].shape
        # side="left" and side="right" are equivalent
        # because we _assume_ that the point sets are disjoint.
        index = np.searchsorted(solution.t, t)

        # Extract the LHS

        def _extract_previous(pytree):
            return tree.tree_map(lambda s: s[index - 1, ...], pytree)

        posterior_t0 = _extract_previous(solution.solution_full)
        t0 = _extract_previous(solution.t)

        # Extract the RHS

        def _extract(pytree):
            return tree.tree_map(lambda s: s[index, ...], pytree)

        t1 = _extract(solution.t)
        output_scale = _extract(solution.output_scale)

        # Take the marginals because we need the t1-value to be informed
        # about *all* datapoints
        u_t1 = _extract(solution.u)
        _, posterior_t1 = self.strategy.init_posterior(u=u_t1)

        if not self.is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        transition_t0_t = self.prior(t - t0, output_scale)
        transition_t_t1 = self.prior(t1 - t, output_scale)
        (estimate, _posterior), _interp_res = self.strategy.interpolate(
            posterior_t0=posterior_t0,
            posterior_t1=posterior_t1,
            transition_t0_t=transition_t0_t,
            transition_t_t1=transition_t_t1,
        )
        return estimate

    def interpolate(
        self, *, t, interp_from: ProbabilisticSolution, interp_to: ProbabilisticSolution
    ):
        """Interpolate between two solution objects."""
        # Domain is (t0, t1]; thus, take the output scale from interp_to
        output_scale = interp_to.output_scale
        transition_t0_t = self.prior(t - interp_from.t, output_scale)
        transition_t_t1 = self.prior(interp_to.t - t, output_scale)

        # Interpolate
        tmp = self.strategy.interpolate(
            posterior_t0=interp_from.solution_full,
            posterior_t1=interp_to.solution_full,
            transition_t0_t=transition_t0_t,
            transition_t_t1=transition_t_t1,
        )
        (estimate, interpolated), step_and_interpolate_from = tmp

        step_from = ProbabilisticSolution(
            t=interp_to.t,
            # New:
            solution_full=step_and_interpolate_from.step_from,
            # Old:
            u=interp_to.u,
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )

        interpolated = ProbabilisticSolution(
            t=t,
            # New:
            solution_full=interpolated,
            u=estimate,
            # Taken from the rhs point
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )

        interp_from = ProbabilisticSolution(
            t=t,
            # New:
            solution_full=step_and_interpolate_from.interp_from,
            # Old:
            u=interp_from.u,
            output_scale=interp_from.output_scale,
            auxiliary=interp_from.auxiliary,
            num_steps=interp_from.num_steps,
            fun_evals=interp_from.fun_evals,
        )

        interp_res = utilities.InterpResult(
            step_from=step_from, interp_from=interp_from
        )
        return interpolated, interp_res

    def interpolate_at_t1(
        self, *, t, interp_from: ProbabilisticSolution, interp_to: ProbabilisticSolution
    ):
        """Interpolate the solution near a checkpoint."""
        del t
        tmp = self.strategy.interpolate_at_t1(posterior_t1=interp_to.solution_full)
        (estimate, interpolated), step_and_interpolate_from = tmp

        prev = ProbabilisticSolution(
            t=interp_to.t,
            # New
            solution_full=step_and_interpolate_from.interp_from,
            # Old
            u=interp_from.u,
            output_scale=interp_from.output_scale,
            auxiliary=interp_from.auxiliary,
            num_steps=interp_from.num_steps,
            fun_evals=interp_from.fun_evals,
        )
        sol = ProbabilisticSolution(
            t=interp_to.t,
            # New:
            solution_full=interpolated,
            u=estimate,
            # Old:
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )
        acc = ProbabilisticSolution(
            t=interp_to.t,
            # New:
            solution_full=step_and_interpolate_from.step_from,
            # Old
            u=interp_to.u,
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )
        return sol, utilities.InterpResult(step_from=acc, interp_from=prev)


class solver_mle(ProbabilisticSolver):
    """Create a solver that uses maximum-likelihood calibration for the output scale.

    Related:
    [`ProbabilisticSolver`](#probdiffeq.probdiffeq.ProbabilisticSolver).

    """

    def __init__(
        self,
        *,
        constraint: constraints.Constraint,
        prior: Callable,
        strategy: estimators_and_loss_functions.MarkovStrategy,
        constraint_init: constraints.Constraint | None = None,
        correct_asymptotic_underconfidence: bool = True,
    ) -> None:
        super().__init__(
            strategy=strategy,
            prior=prior,
            constraint=constraint,
            constraint_init=constraint_init,
        )
        self.correct_asymptotic_underconfidence = correct_asymptotic_underconfidence

    def init(self, t, u, *, damp) -> ProbabilisticSolution:
        u_pred, prediction = self.strategy.init_posterior(u=u)
        cstate = self.constraint.init_linearization()

        prototype = u.prototype_output_scale_calibrated()
        output_scale_prior = np.ones_like(prototype)

        # Update
        lin_fun = func.partial(self.constraint.linearize, damp=damp, t=t)
        fx, _cstate = func.eval_shape(lin_fun, u_pred, cstate)
        fx = tree.tree_map(np.zeros_like, fx)

        if self.constraint_init is not None:
            cstate_init = self.constraint_init.init_linearization()
            fx_init, _cstate = self.constraint_init.linearize(
                u_pred, cstate_init, damp=damp, t=t
            )
            zeros = tree.tree_map(np.zeros_like, fx_init.noise.mean)
            output_scale_running, updates = (
                fx_init.bayes_rule_and_residual_white_rms_tree(
                    zeros, u_pred, solve_triu=linalg.lstsq_svd
                )
            )

            u, posterior = self.strategy.apply_updates(prediction, updates=updates)
            num_data = 1.0

        else:
            u, posterior = u_pred, prediction

            output_scale_running = np.zeros_like(output_scale_prior)
            num_data = 0.0

        auxiliary = (cstate, output_scale_running, num_data)

        return ProbabilisticSolution(
            t=t,
            u=u,
            solution_full=posterior,
            auxiliary=auxiliary,
            output_scale=output_scale_prior,
            num_steps=0,
            fun_evals=fx,
        )

    def step(self, state, *, dt: float, damp: float):
        # Discretize
        prototype = state.u.prototype_output_scale_calibrated()
        output_scale = np.ones_like(prototype)
        transition = self.prior(dt, output_scale)

        # Predict
        u, prediction = self.strategy.predict(
            posterior=state.solution_full, transition=transition
        )

        # Linearize
        (lin_state, output_scale_running, num_data) = state.auxiliary
        fx, cstate = self.constraint.linearize(
            u, state=lin_state, damp=damp, t=state.t + dt
        )

        # Do the full correction step
        zeros = tree.tree_map(np.zeros_like, fx.noise.mean)
        new_term, updates = fx.bayes_rule_and_residual_white_rms_tree(
            zeros, u, solve_triu=linalg.solve_triu
        )
        u, posterior = self.strategy.apply_updates(
            prediction=prediction, updates=updates
        )

        # Calibrate the output scale: c^2 = w_1 * a^2 + w_2 * b^2
        x1 = np.sqrt(num_data / (num_data + 1)) * output_scale_running
        x2 = np.sqrt(1 / (num_data + 1)) * new_term
        output_scale_running = np.hypot(x1, x2)

        # Return the state
        auxiliary = (cstate, output_scale_running, num_data + 1)
        return ProbabilisticSolution(
            t=state.t + dt,
            u=u,
            solution_full=posterior,
            output_scale=state.output_scale,
            auxiliary=auxiliary,
            num_steps=state.num_steps + 1,
            fun_evals=fx,
        )

    def userfriendly_output(
        self, *, solution0: ProbabilisticSolution, solution: ProbabilisticSolution
    ) -> ProbabilisticSolution:
        assert solution.t.ndim > 0

        # This is the MLE solver, so we take the calibrated scale
        _, output_scale, _ = solution.auxiliary
        ones = np.ones_like(output_scale)
        output_scale = output_scale[-1]

        # Improve the calibration like in other Gaussian process models.
        #   ODE priors are generally not as smooth as the ODE solutions,
        #   which means that their uncertainty is often a bit too large.
        #   See e.g. the "asymptotic underconfidence" derivations
        #   in https://arxiv.org/abs/2001.10965
        if self.correct_asymptotic_underconfidence:
            output_scale = output_scale / np.sqrt(solution.num_steps[-1])

        # Finalize the solution with the calibrated output scale
        init = solution0.solution_full
        posterior = solution.solution_full
        estimate, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        output_scale = ones * output_scale[None, ...]
        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbabilisticSolution(
            t=ts,
            u=estimate,
            solution_full=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


class solver_dynamic(ProbabilisticSolver):
    """Create a solver that calibrates the output scale dynamically.

    Related:
    [`ProbabilisticSolver`](#probdiffeq.probdiffeq.ProbabilisticSolver).
    """

    def __init__(
        self,
        *,
        strategy: estimators_and_loss_functions.MarkovStrategy,
        prior: Callable,
        constraint: constraints.Constraint,
        constraint_init: constraints.Constraint | None = None,
        re_linearize_after_calibration=False,
        stop_gradient_through_calibration=True,
    ) -> None:
        super().__init__(
            strategy=strategy,
            prior=prior,
            constraint=constraint,
            constraint_init=constraint_init,
        )
        self.re_linearize_after_calibration = re_linearize_after_calibration
        self.stop_gradient_through_calibration = stop_gradient_through_calibration

    def init(self, t, u, *, damp) -> ProbabilisticSolution:
        u_pred, prediction = self.strategy.init_posterior(u=u)
        lin_state = self.constraint.init_linearization()

        prototype = u_pred.prototype_output_scale_calibrated()
        output_scale = np.ones_like(prototype)
        lin_fun = func.partial(self.constraint.linearize, damp=damp, t=t)
        fx, _lin_state = func.eval_shape(lin_fun, u_pred, lin_state)
        fx = tree.tree_map(np.zeros_like, fx)

        if self.constraint_init is not None:
            cstate_init = self.constraint_init.init_linearization()
            fx_init, _cstate = self.constraint_init.linearize(
                u_pred, cstate_init, damp=damp, t=t
            )
            zeros = tree.tree_map(np.zeros_like, fx_init.noise.mean)
            updates = fx_init.bayes_rule_tree(
                zeros, u_pred, solve_triu=linalg.lstsq_svd
            )
            u, posterior = self.strategy.apply_updates(prediction, updates=updates)
        else:
            u, posterior = u_pred, prediction
        return ProbabilisticSolution(
            t=t,
            u=u,
            solution_full=posterior,
            auxiliary=lin_state,
            output_scale=output_scale,
            num_steps=0,
            fun_evals=fx,
        )

    def step(self, state: ProbabilisticSolution, *, dt: float, damp: float):
        lin_state = state.auxiliary

        # Calibrate the output scale
        prototype = state.u.prototype_output_scale_calibrated()
        ones = np.ones_like(prototype)
        transition = self.prior(dt, ones)
        mean = state.u.mean_flat
        u = transition.apply_flat(mean)

        # Linearize

        fx, lin_state = self.constraint.linearize(
            u, state=lin_state, damp=damp, t=state.t + dt
        )
        observed = fx.marginalise(u)
        zeros = tree.tree_map(np.zeros_like, fx.noise.mean)
        output_scale = observed.residual_white_rms_tree(zeros)

        if self.stop_gradient_through_calibration:
            output_scale = func.stop_gradient(output_scale)

        # Do the full extrapolation with the calibrated output scale
        # (Includes re-discretisation)
        transition = self.prior(dt, output_scale)
        u, prediction = self.strategy.predict(
            state.solution_full, transition=transition
        )

        # Relinearize
        if self.re_linearize_after_calibration:
            fx, lin_state = self.constraint.linearize(
                u, state=lin_state, damp=damp, t=state.t + dt
            )

        # Complete the update
        zeros = tree.tree_map(np.zeros_like, fx.noise.mean)
        updates = fx.bayes_rule_tree(zeros, u, solve_triu=linalg.solve_triu)
        u, posterior = self.strategy.apply_updates(prediction, updates=updates)

        # Return solution
        return ProbabilisticSolution(
            t=state.t + dt,
            u=u,
            solution_full=posterior,
            num_steps=state.num_steps + 1,
            auxiliary=lin_state,
            output_scale=output_scale,
            fun_evals=fx,  # return the initial linearization
        )

    def userfriendly_output(
        self, *, solution: ProbabilisticSolution, solution0: ProbabilisticSolution
    ):
        # This is the dynamic solver,
        # and all covariances have been calibrated already
        ones = np.ones_like(solution.output_scale)
        output_scale = ones[-1, ...]

        init = solution0.solution_full
        posterior = solution.solution_full
        estimate, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        # TODO: stack the calibrated output scales?
        output_scale = ones
        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbabilisticSolution(
            t=ts,
            u=estimate,
            solution_full=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


class solver(ProbabilisticSolver):
    """Create a solver that does not calibrate the output scale automatically.

    This is the text-book implementation of probabilistic solvers.
    It is typically used in parameter estimation:

    - In combination with gradient-based optimisation of the output scale.
    - In combination with diffusion tempering.

    See the tutorials for example applications.


    Related:
    [`ProbabilisticSolver`](#probdiffeq.probdiffeq.ProbabilisticSolver).

    """

    def __init__(
        self,
        *,
        constraint: constraints.Constraint,
        prior: Callable,
        strategy: estimators_and_loss_functions.MarkovStrategy,
        constraint_init: constraints.Constraint | None = None,
    ) -> None:
        super().__init__(
            strategy=strategy,
            prior=prior,
            constraint=constraint,
            constraint_init=constraint_init,
        )

    def init(self, t: Array, u, *, damp) -> ProbabilisticSolution:
        u_pred, prediction = self.strategy.init_posterior(u=u)

        if self.constraint_init is not None:
            cstate_init = self.constraint_init.init_linearization()
            fx_init, _cstate = self.constraint_init.linearize(
                u_pred, cstate_init, damp=damp, t=t
            )
            zeros = tree.tree_map(np.zeros_like, fx_init.noise.mean)
            updates = fx_init.bayes_rule_tree(
                zeros, u_pred, solve_triu=linalg.lstsq_svd
            )
            u, posterior = self.strategy.apply_updates(prediction, updates=updates)

        else:
            u, posterior = u_pred, prediction

        cstate = self.constraint.init_linearization()
        lin_fun = func.partial(self.constraint.linearize, damp=damp, t=t)
        fx, _cstate = func.eval_shape(lin_fun, u_pred, cstate)
        fx = tree.tree_map(np.zeros_like, fx)

        prototype = u.prototype_output_scale_calibrated()
        output_scale = np.ones_like(prototype)
        return ProbabilisticSolution(
            t=t,
            u=u,
            solution_full=posterior,
            num_steps=0,
            auxiliary=cstate,
            output_scale=output_scale,
            fun_evals=fx,
        )

    def step(self, state: ProbabilisticSolution, *, dt, damp):
        # Discretize
        output_scale = np.ones_like(state.output_scale)
        transition = self.prior(dt, output_scale)

        # Predict
        u_pred, prediction = self.strategy.predict(
            state.solution_full, transition=transition
        )

        u = u_pred

        # Linearize
        fx, auxiliary = self.constraint.linearize(
            u, state.auxiliary, damp=damp, t=state.t + dt
        )
        # Update
        zeros = tree.tree_map(np.zeros_like, fx.noise.mean)
        updates = fx.bayes_rule_tree(zeros, u_pred, solve_triu=linalg.solve_triu)
        u, posterior = self.strategy.apply_updates(prediction, updates=updates)

        # Return solution
        return ProbabilisticSolution(
            t=state.t + dt,
            u=u,
            solution_full=posterior,
            output_scale=output_scale,
            auxiliary=auxiliary,
            num_steps=state.num_steps + 1,
            fun_evals=fx,
        )

    def userfriendly_output(
        self, *, solution0: ProbabilisticSolution, solution: ProbabilisticSolution
    ) -> ProbabilisticSolution:
        assert solution.t.ndim > 0

        # This is the uncalibrated solver, so scale=1
        ones = np.ones_like(solution.output_scale)
        output_scale = np.ones_like(solution.output_scale[-1])

        init = solution0.solution_full
        posterior = solution.solution_full
        u, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        output_scale = ones * output_scale[None, ...]

        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbabilisticSolution(
            t=ts,
            u=u,
            solution_full=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


def error_norm_scale_then_rms(*, norm_order=None) -> Callable:
    """Normalize an error by scaling followed by computing the root-mean-square norm.

    This is the recommended approach, and there is no reason to choose
    [`error_norm_rms_then_scale`](#probdiffeq.probdiffeq.error_norm_rms_then_scale),
    in situations where the present function applies.
    However, there are situations where it doesn't apply, for example,
    in residual-based error estimators for root constraints whose pytree
    structure differs from that of the target Taylor coefficients.

    See the custom information operator tutorial for details.
    """

    def normalize(error_abs, reference, atol, rtol):
        scale = atol + rtol * np.abs(reference)
        error_rel = error_abs / scale
        return rms(error_rel)

    def rms(s):
        return linalg.vector_norm(s, order=norm_order) / np.sqrt(s.size)

    return normalize


def error_norm_rms_then_scale(norm_order=None) -> Callable:
    """Normalize an error by computing the root-mean-square norm followed by scaling.

    Use this for residual-based error estimators in combination
    with custom root constraints.

    See the custom information operator tutorial for details.
    """

    def normalize(error_abs, reference, atol, rtol):
        norm_abs = rms(error_abs)
        norm_ref = rms(reference)
        return norm_abs / (atol + rtol * norm_ref)

    def rms(s):
        return linalg.vector_norm(s, order=norm_order) / np.sqrt(s.size)

    return normalize


class ErrorEstimator:
    """An interface for error estimators in probabilistic solvers.

    Related:
    [`error_residual_std`](#probdiffeq.probdiffeq.error_residual_std).

    """

    def init_error(self):
        """Initialize the error-estimation state."""
        raise NotImplementedError

    def estimate_error_norm(
        self,
        state: tuple,
        previous: ProbabilisticSolution,
        proposed: ProbabilisticSolution,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ):
        """Estimate the error norm.

        The error norm is a single scalar that already includes:

        - Absolute and relative tolerances
        - Error contraction rates

        In the acceptance/rejection step, this error norm is compared
        to one to determine whether a step has been successful.
        """
        raise NotImplementedError


class error_residual_std(ErrorEstimator):
    r"""Construct an error estimator based on a local residual's standard deviation.

    This is the common error estimate, proposed by Schober et al. (2019),
    extended by Bosch et al. (2021) to different linearization and calibration modes,
    and then generalised to state-space model factorisations by
    Krämer, Bosch, and Schmidt et al. (2022).
    Please consider citing these papers in your work if you use any of
    these error estimates.

    ??? note "BibTex for Schober et al. (2019)"
        ```bibtex
        @article{schober2019probabilistic,
            title={A probabilistic model for the numerical
            solution of initial value problems},
            author={Schober, Michael and S{\"a}rkk{\"a}, Simo and Hennig, Philipp},
            journal={Statistics and Computing},
            volume={29},
            number={1},
            pages={99--122},
            year={2019},
            publisher={Springer}
        }
        ```

    ??? note "BibTex for Bosch et al. (2021)"
        ```bibtex
            @inproceedings{bosch2021calibrated,
                title={Calibrated adaptive probabilistic ODE solvers},
                author={Bosch, Nathanael and Hennig, Philipp and Tronarp, Filip},
                booktitle={International Conference on
                Artificial Intelligence and Statistics},
                pages={3466--3474},
                year={2021},
                organization={PMLR}
            }
        ```

    ??? note "BibTex for Krämer, Bosch, and Schmidt et al. (2022)"
        ```bibtex
            @inproceedings{kramer2022probabilistic,
                title={Probabilistic ODE solutions in millions of dimensions},
                author={Kr{\"a}mer, Nicholas and Bosch, Nathanael and
                Schmidt, Jonathan and Hennig, Philipp},
                booktitle={International Conference on Machine Learning},
                pages={11634--11649},
                year={2022},
                organization={PMLR}
            }
        ```

    Related:
    [`ErrorEstimator`](#probdiffeq.probdiffeq.ErrorEstimator).

    """

    def __init__(
        self,
        *,
        constraint: constraints.Constraint,
        prior: Any,
        error_norm: Callable | None = None,
        re_linearize_before_error: bool = False,  # cache by default
        error_per_unit_step: bool = False,
    ) -> None:
        if error_norm is None:
            error_norm = error_norm_scale_then_rms()

        self.error_norm = error_norm
        self.constraint = constraint
        self.prior = prior
        self.re_linearize_before_error = re_linearize_before_error
        self.error_per_unit_step = error_per_unit_step

    def init_error(self):
        return self.constraint.init_linearization()

    def estimate_error_norm(
        self,
        state,
        previous: ProbabilisticSolution,
        proposed: ProbabilisticSolution,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ) -> tuple[float, tuple]:
        # Discretize; The output scale is set to one
        # since the error is multiplied with a local scale estimate anyway
        prototype = proposed.u.prototype_output_scale_calibrated()
        output_scale = np.ones_like(prototype)
        transition = self.prior(dt, output_scale)

        # Extrapolate from the zero-error state
        mean = previous.u.mean_flat
        rv = transition.apply_flat(mean)

        # Optionally: re-linearize
        if self.re_linearize_before_error:
            linearized, state = self.constraint.linearize(
                rv, state, damp=damp, t=proposed.t
            )
        else:
            linearized = proposed.fun_evals

        # Extract the local residual std from the linearization
        observed = linearized.marginalise(rv)
        zeros = tree.tree_map(np.zeros_like, linearized.noise.mean)
        output_scale = observed.residual_white_rms_tree(zeros)
        observed = observed.rescale_cholesky(output_scale)
        error = observed.std
        error, _ = tree.ravel_pytree(error)

        # Compute a reference
        previous_leaves = tree.tree_leaves_depth_one(previous.u.mean)
        proposed_leaves = tree.tree_leaves_depth_one(proposed.u.mean)
        error_contraction_rate = len(previous_leaves)
        u0, _ = tree.ravel_pytree(previous_leaves[0])
        u1, _ = tree.ravel_pytree(proposed_leaves[0])
        reference = np.maximum(np.abs(u0), np.abs(u1))

        # Turn the unscaled absolute error into a relative one.
        # This is a generalisation of the typical residual-based
        # error estimates for probabilistic solvers in the sense that
        # it respects higher-order information. For first-order problems,
        # it is identical to Schober et al, Bosch et al., and so on.
        # For higher-order problems it is closer to Taylor-series based
        # (non-probabilistic) ODE solvers; for example, refer to
        # Tan et al. (2026; https://arxiv.org/pdf/2602.04086).
        n = self.constraint.residual_order - 1
        if self.error_per_unit_step:
            n += 1

        error_abs = error * dt**n / np.factorial(n)
        error_norm = self.error_norm(error_abs, reference, atol=atol, rtol=rtol)

        # Scale the error norm with the error contraction rate and return
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, state


class error_state_std(ErrorEstimator):
    r"""Construct an error estimator based on a state's standard deviation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """

    # TODO: make the experimental-warning into a decorator
    def __init__(
        self,
        *,
        constraint: constraints.Constraint,
        prior: Any,
        error_norm: Callable | None = None,
        re_linearize_before_error: bool = False,  # cache by default
        derivative_idx: int = 0,
        error_per_unit_step: bool = False,
    ) -> None:
        if error_norm is None:
            error_norm = error_norm_scale_then_rms()

        self.error_norm = error_norm
        self.constraint = constraint
        self.prior = prior
        self.re_linearize_before_error = re_linearize_before_error
        self.derivative_idx = derivative_idx
        self.error_per_unit_step = error_per_unit_step

    def init_error(self):
        return self.constraint.init_linearization()

    def estimate_error_norm(
        self,
        state,
        previous: ProbabilisticSolution,
        proposed: ProbabilisticSolution,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ) -> tuple[float, tuple]:
        # Discretize; The output scale is set to one
        # since the error is multiplied with a local scale estimate anyway
        prototype = proposed.u.prototype_output_scale_calibrated()
        output_scale = np.ones_like(prototype)
        transition = self.prior(dt, output_scale)

        # Extrapolate from the zero-error state
        mean = previous.u.mean_flat
        rv = transition.apply_flat(mean)

        mean = previous.u.mean
        mean_leaves = tree.tree_leaves_depth_one(mean)
        error_contraction_rate = len(mean_leaves)

        # Optionally: re-linearize
        if self.re_linearize_before_error:
            linearized, state = self.constraint.linearize(
                rv, state, damp=damp, t=proposed.t
            )
        else:
            linearized = proposed.fun_evals

        # Extract the local residual std from the linearization
        zeros = tree.tree_map(np.zeros_like, linearized.noise.mean)
        output_scale, conditional = linearized.bayes_rule_and_residual_white_rms_tree(
            zeros, rv, solve_triu=linalg.solve_triu
        )

        # Measure error on the n-th state (usually, n=0 because why not)
        n = self.derivative_idx

        # *New:* Go back into solution space
        std = conditional.std[n]
        error, _ = tree.ravel_pytree(std)
        error = output_scale * error
        error, _ = tree.ravel_pytree(error)  # TODO: this line is useless?

        # Compute a reference
        u0, _ = tree.ravel_pytree(previous.u.mean[n])
        u1, _ = tree.ravel_pytree(proposed.u.mean[n])
        reference = np.maximum(np.abs(u0), np.abs(u1))

        # Increase the order (after selecting the states)
        if self.error_per_unit_step:
            n += 1

        # Turn the unscaled absolute error into a relative one.
        error_abs = error * dt**n / np.factorial(n)
        error_norm = self.error_norm(error_abs, reference, atol=atol, rtol=rtol)

        # Scale the error norm with the error contraction rate and return
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, state
