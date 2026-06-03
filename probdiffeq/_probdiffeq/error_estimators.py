"""Error estimators."""

from probdiffeq import ssm_impl
from probdiffeq._probdiffeq import constraints, solvers
from probdiffeq.backend import linalg, np, tree
from probdiffeq.backend.typing import Any, Callable

__all__ = [
    "ErrorEstimator",
    "error_norm_rms_then_scale",
    "error_norm_scale_then_rms",
    "error_residual_std",
    "error_state_std",
]


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
        previous: solvers.ProbabilisticSolution,
        proposed: solvers.ProbabilisticSolution,
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
        ssm: ssm_impl.FactSsmImpl,
        error_norm: Callable | None = None,
        re_linearize_before_error: bool = False,  # cache by default
        error_per_unit_step: bool = False,
    ) -> None:
        if error_norm is None:
            error_norm = error_norm_scale_then_rms()

        self.error_norm = error_norm
        self.constraint = constraint
        self.prior = prior
        self.ssm = ssm
        self.re_linearize_before_error = re_linearize_before_error
        self.error_per_unit_step = error_per_unit_step

    def init_error(self):
        return self.constraint.init_linearization()

    def estimate_error_norm(
        self,
        state,
        previous: solvers.ProbabilisticSolution,
        proposed: solvers.ProbabilisticSolution,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ) -> tuple[float, tuple]:
        # Discretize; The output scale is set to one
        # since the error is multiplied with a local scale estimate anyway
        prototype = self.ssm.prior.prototype_output_scale_calibrated(proposed.u)
        output_scale = np.ones_like(prototype)
        transition = self.prior(dt, output_scale)

        # Extrapolate from the zero-error state
        mean = previous.u.mean_flat
        rv = self.ssm.conditional.apply_flat(mean, transition)

        # Optionally: re-linearize
        if self.re_linearize_before_error:
            linearized, state = self.constraint.linearize(
                rv, state, damp=damp, t=proposed.t
            )
        else:
            linearized = proposed.fun_evals

        # Extract the local residual std from the linearization
        observed = self.ssm.conditional.marginalise(rv, linearized)
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
        ssm: ssm_impl.FactSsmImpl,
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
        self.ssm = ssm
        self.re_linearize_before_error = re_linearize_before_error
        self.derivative_idx = derivative_idx
        self.error_per_unit_step = error_per_unit_step

    def init_error(self):
        return self.constraint.init_linearization()

    def estimate_error_norm(
        self,
        state,
        previous: solvers.ProbabilisticSolution,
        proposed: solvers.ProbabilisticSolution,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ) -> tuple[float, tuple]:
        # Discretize; The output scale is set to one
        # since the error is multiplied with a local scale estimate anyway
        prototype = self.ssm.prior.prototype_output_scale_calibrated(
            template=proposed.u
        )
        output_scale = np.ones_like(prototype)
        transition = self.prior(dt, output_scale)

        # Extrapolate from the zero-error state
        mean = previous.u.mean_flat
        rv = self.ssm.conditional.apply_flat(mean, transition)

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
        output_scale, conditional = (
            self.ssm.conditional.bayes_rule_and_residual_white_rms_tree(
                zeros, rv, linearized, solve_triu=linalg.solve_triu
            )
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
