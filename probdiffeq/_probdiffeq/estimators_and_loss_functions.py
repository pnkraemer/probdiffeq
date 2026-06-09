"""Estimation strategies and loss functions."""

from probdiffeq import ssm_impl
from probdiffeq._probdiffeq import utilities
from probdiffeq.backend import flow, func, linalg, np, random, structs, tree
from probdiffeq.backend.typing import Any, Callable, Generic, Sequence, TypeVar

__all__ = [
    "MarkovSequence",
    "MarkovStrategy",
    "loss_lml_terminal_values",
    "loss_lml_timeseries",
    "strategy_filter",
    "strategy_smoother_fixedinterval",
    "strategy_smoother_fixedpoint",
]


def loss_lml_terminal_values(*, tcoeff_index=0):
    """Construct a log-marginal-likelihood loss for the terminal value."""

    def loss(u, /, *, marginals, std):
        u = tree.tree_map(np.asarray, u)
        std = tree.tree_map(np.asarray, std)
        std_expected = marginals.std[tcoeff_index]

        msg = "The standard deviation container differs from what was expected."
        msg += f" Expected: shape={tree.tree_map(np.shape, std_expected)}."
        msg += f" Received: shape={tree.tree_map(np.shape, std)}."
        msg += f" For reference: data-shape={tree.tree_map(np.shape, u)}."

        try:
            shapes_equal = tree.tree_map(
                lambda a, b: a.shape == b.shape, std, std_expected
            )
        except Exception as error:
            raise ValueError(msg) from error

        if not tree.tree_all(shapes_equal):
            raise ValueError(msg)

        model = marginals.to_derivative(tcoeff_index, std)
        marg = model.marginalise(marginals)

        # Wrap into list because blockdiag & isotropic models
        # expect sequences of states (even if the length is one)
        return marg.logpdf_tree([u])

    return loss


def loss_lml_timeseries(
    *, average_pdfs: bool = True, tcoeff_index=0, solve_triu=linalg.lstsq_svd
):
    """Construct a log-marginal-likelihood loss for a time-series."""

    def loss(u, /, *, posterior, std):
        if not isinstance(posterior, MarkovSequence):
            msg = "The datatype of the posterior is not as expected."
            msg += f" Expected: {MarkovSequence}."
            msg += f" Received: {type(posterior)}."
            msg += " Did you perhaps use a filter instead of a smoother"
            msg += ", or mean to use a different loss?"
            raise TypeError(msg)

        u = tree.tree_map(np.asarray, u)
        std = tree.tree_map(np.asarray, std)
        std_expected = posterior.marginal.std[tcoeff_index]

        msg = "The standard deviation container differs from what was expected."
        msg += f" Expected: shape={tree.tree_map(np.shape, std_expected)}."
        msg += f" Received: shape={tree.tree_map(np.shape, std)}."
        msg += f" For reference: data-shape={tree.tree_map(np.shape, u)}."

        try:
            shapes_equal = tree.tree_map(
                lambda a, b: a.shape == b.shape, std, std_expected
            )
        except Exception as error:
            raise ValueError(msg) from error

        if not tree.tree_all(shapes_equal):
            raise ValueError(msg)

        # Remove the filtering distributions from the posterior
        posterior = posterior.remove_filtering_distributions()

        def make_model(s):
            return posterior.marginal.to_derivative(tcoeff_index, s)

        model = func.vmap(make_model)(std)

        # Use solve_triu=lstsq because for noise-free observations, the initial state
        # of the ODE solution tends to be noise-free,
        # which clashes and returns NaNs if we use exact solvers.
        return posterior.evaluate_lml(
            [u], model=model, average_pdfs=average_pdfs, solve_triu=solve_triu
        )

    return loss


C = TypeVar("C", bound=Sequence)
"""A type-variable to describe sequences.

Used to type Taylor coefficients, for example.
"""

N = TypeVar("N", bound=ssm_impl.AbstractTreeNormal)
"""A type-variable to describe normal distributions.

Used to type marginals, for example.
"""


@tree.register_dataclass
@structs.dataclass
class MarkovSequence(Generic[N]):
    """A datastructure for Markov sequences as batches of joint distributions.

    This is the output type of smoother-based estimators.
    (Filter-based estimators do not return specialised types.)
    """

    marginal: N
    """The marginal distribution."""

    conditional: Any
    """The conditional distribution."""

    reverse: bool = structs.dataclass_field(metadata={"static": True})
    """The direction of factorisations."""

    @classmethod
    def from_grid(cls, init, discretize, *, grid, reverse: bool):
        marginal = init
        conditional = func.vmap(discretize)(np.diff(grid))
        return cls(marginal, conditional, reverse=reverse)

    def rescale_cholesky(self, factor, /):
        marg = self.marginal.rescale_cholesky(factor)
        cond = self.conditional.rescale_noise(factor)
        return MarkovSequence(marg, cond, reverse=self.reverse)

    def evaluate_marginals(self):
        """Extract the (time-)marginals from a Markov sequence.

        This is only needed in combination with smoothing-based strategies.
        """
        if self.marginal.mean_flat.ndim == self.conditional.noise.mean_flat.ndim:
            # TODO: do this in the postprocessing of the solver...
            markov_seq = self.remove_filtering_distributions()
            return markov_seq.evaluate_marginals()

        def step(x, cond):
            extrapolated = cond.marginalise(x)
            return extrapolated, extrapolated

        _, marginals = flow.scan(
            step, init=self.marginal, xs=self.conditional, reverse=self.reverse
        )

        if self.reverse:
            # Append the terminal marginal to the computed ones
            return tree.tree_array_append(marginals, self.marginal)

        return tree.tree_array_prepend(self.marginal, marginals)

    def evaluate_lml(self, u, *, model, average_pdfs: bool, solve_triu: Callable):
        assert self.reverse

        # Process the terminal value
        u0 = tree.tree_map(lambda s: s[-1], u)
        model0 = tree.tree_map(lambda s: s[-1], model)
        pdf0, updated = model0.bayes_rule_and_logpdf_tree(
            u0, self.marginal, solve_triu=solve_triu
        )

        # Process the remaining values
        def body(rv_and_logpdf, prior_and_observation_and_data):
            rv, logpdf, num_data = rv_and_logpdf
            prior, observation, data = prior_and_observation_and_data

            predicted = prior.marginalise(rv)

            logpdf_n, corrected = observation.bayes_rule_and_logpdf_tree(
                data, predicted, solve_triu=solve_triu
            )

            # The mean of the PDFs (as opposed to their sum) usually
            # leads to LML values that are more "human-readable"
            # (ie magnitude O(1) instead O(N)). This is technically not
            # a log-marginal-likelihood, but much nicer to work with when
            # optimising the loss.
            if average_pdfs:
                logpdf1 = (logpdf * num_data + logpdf_n) / (num_data + 1)
            else:
                logpdf1 = logpdf + logpdf_n

            return (corrected, logpdf1, num_data + 1), ()

        u1 = tree.tree_map(lambda s: s[:-1], u)
        model1 = tree.tree_map(lambda s: s[:-1], model)
        init = (updated, pdf0, 1)
        xs = (self.conditional, model1, u1)
        (_, pdf, _), _ = flow.scan(body, init=init, xs=xs, reverse=self.reverse)
        return pdf

    def remove_filtering_distributions(self):
        """Discard all intermediate filtering solutions from a Markov sequence.

        This function is useful to convert a smoothing-solution into a Markov sequence
        that is compatible with sampling or marginalisation.
        """
        # There should be one marginal and many conditionals. If not, remove marginals
        if self.marginal.mean_flat.ndim == self.conditional.noise.mean_flat.ndim:
            marginal_idx = -1 if self.reverse else 0
            init = tree.tree_map(lambda x: x[marginal_idx, ...], self.marginal)
            return MarkovSequence(init, self.conditional, reverse=self.reverse)
        return self

    def sample(self, key, *, shape: tuple = ()):
        """Sample from a Markov sequence."""
        # If the MarkovSequence carries unnecessary filtering marginals, remove them
        if self.marginal.mean_flat.ndim == self.conditional.noise.mean_flat.ndim:
            markov_seq = self.remove_filtering_distributions()
            return markov_seq.sample(key, shape=shape)

        # If many samples are required, vmap over recursive calls to sample()
        if len(shape) > 0:
            n, *shape_remaining = shape
            keys = random.split(key, num=n)
            sample_partial = func.partial(self.sample, shape=shape_remaining)
            return func.vmap(sample_partial)(keys)

        # Compute a single sample from the Markov sequence

        def body(sample_and_key, cond):
            smp_flat, key = sample_and_key

            # Propagate the previous sample
            predicted = cond.apply_flat(smp_flat)

            # Sample the propagated variable
            key, subkey = random.split(key, num=2)
            smp_flat = predicted.sample_flat(subkey)

            # Unravel the sample
            smp_tree = predicted.tree_flatten.unflatten_array(smp_flat)
            return (smp_flat, key), smp_tree

        # Loop over the conditionals
        key, subkey = random.split(key, num=2)
        sample0_flat = self.marginal.sample_flat(subkey)
        init = (sample0_flat, key)
        xs = self.conditional
        _, samples = flow.scan(body, init=init, xs=xs, reverse=self.reverse)

        sample0_tree = self.marginal.tree_flatten.unflatten_array(sample0_flat)
        if self.reverse:
            return tree.tree_array_append(samples, sample0_tree)
        return tree.tree_array_prepend(sample0_tree, samples)


T = TypeVar("T", bound=MarkovSequence | ssm_impl.AbstractTreeNormal)
"""A type-variable to describe posterior distributions."""


class MarkovStrategy(Generic[T]):
    """An interface for estimation strategies in Markovian state-space models.

    Related:
    [`strategy_filter`](#probdiffeq.probdiffeq.strategy_filter),
    [`strategy_smoother_fixedpoint`](#probdiffeq.probdiffeq.strategy_smoother_fixedpoint),
    [`strategy_smoother_fixedinterval`](#probdiffeq.probdiffeq.strategy_smoother_fixedinterval).
    """

    def __init__(
        self,
        is_suitable_for_save_at: int,
        is_suitable_for_save_every_step: int,
        is_suitable_for_offgrid_marginals: int,
    ) -> None:
        self.is_suitable_for_save_at = is_suitable_for_save_at
        self.is_suitable_for_save_every_step = is_suitable_for_save_every_step
        self.is_suitable_for_offgrid_marginals = is_suitable_for_offgrid_marginals

    def init_posterior(self, *, u) -> T:
        """Initialize a posterior distribution."""
        raise NotImplementedError

    def predict(self, posterior: T, *, transition) -> tuple:
        """Make a prediction."""
        raise NotImplementedError

    def apply_updates(self, prediction: T, *, updates) -> tuple:
        """Apply updates to a prediction."""
        raise NotImplementedError

    def interpolate(
        self, *, posterior_t0: T, posterior_t1: T, transition_t0_t, transition_t_t1
    ) -> tuple[tuple, utilities.InterpResult[T]]:
        """Interpolate between two points."""
        raise NotImplementedError

    def interpolate_at_t1(
        self, *, posterior_t1: T
    ) -> tuple[tuple, utilities.InterpResult[T]]:
        """Interpolate at a checkpoint."""
        raise NotImplementedError

    def finalize(self, *, posterior0: T, posterior: T, output_scale) -> T:
        """Finalize the posterior before returning a solution."""
        raise NotImplementedError


class strategy_smoother_fixedinterval(MarkovStrategy[MarkovSequence]):
    """Construct a fixed-interval smoother.

    Use this strategy for fixed steps.
    For adaptive steps, consider using a fixed-point smoother instead.


    Related:
    [`MarkovStrategy`](#probdiffeq.probdiffeq.MarkovStrategy).
    """

    def __init__(self) -> None:
        super().__init__(
            is_suitable_for_save_at=False,
            is_suitable_for_save_every_step=True,
            is_suitable_for_offgrid_marginals=True,
        )

    def init_posterior(self, *, u):
        cond = u.identity_conditional()
        posterior = MarkovSequence(marginal=u, conditional=cond, reverse=True)
        return u, posterior

    def predict(self, posterior: MarkovSequence, *, transition) -> tuple:
        marginals, cond = transition.revert(
            posterior.marginal, solve_triu=linalg.solve_triu
        )
        posterior = MarkovSequence(
            marginal=marginals, conditional=cond, reverse=posterior.reverse
        )

        return marginals, posterior

    def apply_updates(self, prediction, *, updates):
        posterior = MarkovSequence(
            updates, prediction.conditional, reverse=prediction.reverse
        )
        marginals = updates
        return marginals, posterior

    def finalize(
        self, *, posterior0: MarkovSequence, posterior: MarkovSequence, output_scale
    ):
        prototype = posterior0.marginal.prototype_output_scale_calibrated()
        assert output_scale.shape == prototype.shape
        posterior0 = posterior0.rescale_cholesky(output_scale)
        posterior = posterior.rescale_cholesky(output_scale)

        # Marginalise
        marginals = posterior.evaluate_marginals()

        # Prepend the initial condition to the filtering distributions
        init = tree.tree_array_prepend(posterior0.marginal, posterior.marginal)
        posterior = MarkovSequence(
            marginal=init, conditional=posterior.conditional, reverse=posterior.reverse
        )

        # Extract targets
        return marginals, posterior

    def interpolate(
        self,
        *,
        posterior_t0: MarkovSequence,
        posterior_t1: MarkovSequence,
        transition_t0_t,
        transition_t_t1,
    ):
        """Interpolate between two Markov sequences.

        Here is how a smoother interpolates:

        - Extrapolate from t0 to t, which gives the filtering distribution
          and the backward transition from t to t0.
        - Extrapolate from t to t1, which gives another filtering distribution
          and the backward transition from t1 to t.
        - Apply the new t1-to-t backward transition to the posterior
          to compute the interpolation.

        This intermediate result is informed about its "right-hand side" datum.
        Note how in probdiffeq, all solver subintervals include their right-hand
        side time-point: in other words, they are (t0, t1].

        Specifically, interpolation is not equal to computing offgrid marginals.
        Interpolation always assumes that the current subinterval is the right-most
        subinterval, which is the case during the forward pass.
        After the simulation, if there are observations > t1,
        which happens when computing offgrid-marginals, do not use `interpolate()`.
        """
        # Extrapolate from t0 to t, and from t to t1.

        _, extrapolated_t = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        _, extrapolated_t1 = self.predict(
            posterior=extrapolated_t, transition=transition_t_t1
        )

        # Marginalise backwards from t1 to t to obtain the interpolated solution.
        marginal_t1 = posterior_t1.marginal
        conditional_t1_to_t = extrapolated_t1.conditional
        rv_at_t = conditional_t1_to_t.marginalise(marginal_t1)
        solution_at_t = MarkovSequence(
            rv_at_t, extrapolated_t.conditional, reverse=extrapolated_t.reverse
        )

        # The state at t1 gets a new backward model;
        # (it must remember how to get back to t, not to t0).
        solution_at_t1 = MarkovSequence(
            marginal_t1, conditional_t1_to_t, reverse=extrapolated_t.reverse
        )
        interp_res = utilities.InterpResult(
            step_from=solution_at_t1, interp_from=solution_at_t
        )

        # Extract targets
        marginals = solution_at_t.marginal
        return (marginals, solution_at_t), interp_res

    def interpolate_at_t1(self, posterior_t1):
        marginals = posterior_t1.marginal

        interp_res = utilities.InterpResult(
            step_from=posterior_t1, interp_from=posterior_t1
        )
        return (marginals, posterior_t1), interp_res


class strategy_filter(MarkovStrategy):
    """Construct a filter.

    Filters work with all timestepping strategies.
    However, the uncertainties are not informed by the full
    timeseries, which makes them visually less pleasing.
    Filter solutions also do not admit computing log-marginal
    likelihoods or joint sampling from the posterior distribution.
    For these use-cases, use smoothers instead.

    Related:
    [`MarkovStrategy`](#probdiffeq.probdiffeq.MarkovStrategy).
    """

    def __init__(self) -> None:
        super().__init__(
            is_suitable_for_save_at=True,
            is_suitable_for_save_every_step=True,
            is_suitable_for_offgrid_marginals=True,
        )

    def init_posterior(self, *, u):
        return u, u

    def predict(self, posterior: T, *, transition) -> tuple:
        marginals = transition.marginalise(posterior)
        return marginals, marginals

    def apply_updates(self, prediction, *, updates):
        del prediction
        marginals = updates
        return marginals, marginals

    def finalize(self, *, posterior0, posterior, output_scale):
        expected = posterior0.prototype_output_scale_calibrated()
        assert output_scale.shape == expected.shape

        # No rescaling because no calibration at the initial step
        posterior0 = posterior0.rescale_cholesky(output_scale)

        # Calibrate
        posterior = posterior.rescale_cholesky(output_scale)

        # Stack
        posterior = tree.tree_array_prepend(posterior0, posterior)

        marginals = posterior
        return marginals, posterior

    def interpolate(
        self, *, posterior_t0, posterior_t1, transition_t0_t, transition_t_t1
    ):
        del transition_t_t1
        _, interpolated = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        marginals = interpolated
        interp_res = utilities.InterpResult(
            step_from=posterior_t1, interp_from=interpolated
        )
        return (marginals, interpolated), interp_res

    def interpolate_at_t1(self, *, posterior_t1):
        marginals = posterior_t1
        interp_res = utilities.InterpResult(
            step_from=posterior_t1, interp_from=posterior_t1
        )
        return (marginals, posterior_t1), interp_res


class strategy_smoother_fixedpoint(MarkovStrategy[MarkovSequence]):
    r"""Construct a fixedpoint-smoother.

    Fixed-point smoothers are the method of choice for adaptive
    time-stepping in probabilistic differential equation solvers.

    Concretely, we implement the fixedpoint smoother by Krämer (2025a).
    Applied to probabilistic solvers, this leads to the algorithm
    by Krämer (2025b). Please consider citing both papers if you use
    fixed-point smoothers and ODE solvers in your work.


    ??? note "BibTex for Krämer (2025a)"
        ```bibtex
        @article{kramer2025numerically,
            title={Numerically Robust Fixed-Point Smoothing Without State Augmentation},
            author={Kr{\"a}mer, Nicholas},
            year={2025},
            journal={Transactions on Machine Learning Research}
        }
        ```

    ??? note "BibTex for Krämer (2025b)"
        ```bibtex
            @InProceedings{kramer2024adaptive,
            title     = {Adaptive Probabilistic ODE Solvers Without Adaptive
            Memory Requirements},
            author    = {Kr\"{a}mer, Nicholas},
            booktitle = {Proceedings of the First International Conference
            on Probabilistic Numerics},
            pages     = {12--24},
            year      = {2025},
            editor    = {Kanagawa, Motonobu and Cockayne, Jon and Gessner,
            Alexandra and Hennig, Philipp},
            volume    = {271},
            series    = {Proceedings of Machine Learning Research},
            publisher = {PMLR},
            url       = {https://proceedings.mlr.press/v271/kramer25a.html}
        }
        ```
    Related:
    [`MarkovStrategy`](#probdiffeq.probdiffeq.MarkovStrategy).


    """

    def __init__(self) -> None:
        super().__init__(
            is_suitable_for_save_at=True,
            is_suitable_for_save_every_step=False,
            is_suitable_for_offgrid_marginals=False,
        )

    def init_posterior(self, *, u):
        cond = u.identity_conditional()
        posterior = MarkovSequence(u, cond, reverse=True)
        return u, posterior

    def predict(self, posterior: MarkovSequence, *, transition) -> tuple:
        rv = posterior.marginal
        bw0 = posterior.conditional
        marginals, cond = transition.revert(rv, solve_triu=linalg.solve_triu)
        cond = bw0.merge(cond)
        predicted = MarkovSequence(marginals, cond, reverse=posterior.reverse)

        return marginals, predicted

    def apply_updates(self, prediction: MarkovSequence, *, updates):
        posterior = MarkovSequence(
            updates, prediction.conditional, reverse=prediction.reverse
        )
        marginals = updates
        return marginals, posterior

    def finalize(
        self, *, posterior0: MarkovSequence, posterior: MarkovSequence, output_scale
    ):
        expected = posterior0.marginal.prototype_output_scale_calibrated()
        assert output_scale.shape == expected.shape
        posterior0 = posterior0.rescale_cholesky(output_scale)
        posterior = posterior.rescale_cholesky(output_scale)

        # Marginalise
        marginals = posterior.evaluate_marginals()

        # Prepend the initial condition to the filtering distributions
        init = tree.tree_array_prepend(posterior0.marginal, posterior.marginal)
        posterior = MarkovSequence(
            marginal=init, conditional=posterior.conditional, reverse=posterior.reverse
        )

        # Extract targets
        return marginals, posterior

    def interpolate_at_t1(self, *, posterior_t1: MarkovSequence):
        cond_identity = posterior_t1.marginal.identity_conditional()
        resume_from = MarkovSequence(
            posterior_t1.marginal,
            conditional=cond_identity,
            reverse=posterior_t1.reverse,
        )
        interp_res = utilities.InterpResult(
            step_from=resume_from, interp_from=resume_from
        )

        interpolated = posterior_t1
        marginals = interpolated.marginal
        return (marginals, interpolated), interp_res

    def interpolate(
        self,
        *,
        posterior_t0: MarkovSequence,
        posterior_t1: MarkovSequence,
        transition_t0_t,
        transition_t_t1,
    ):
        """Interpolate between two Markov sequences.

        Assuming `state_t0` has seen $n$ collocation points,
        and `state_t1` has seen $n+1$ collocation points,
        then interpolation at time $t$ is computed as follows:

        1. Extrapolate from $t_0$ to $t$. This yields:
            - the marginal at $t$ given $n$ observations.
            - the backward transition from $t$ to $t_0$ given $n$ observations.

        2. Extrapolate from $t$ to $t_1$. This yields:
            - the marginal at $t_1$ given $n$ observations
              (in contrast,`state_t1` has seen $n+1$ observations)
            - the backward transition from $t_1$ to $t$ given $n$ observations.

        3. Apply the backward transition from $t_1$ to $t$
        to the marginal inside `state_t1`
        to obtain the marginal at $t$ given $n+1$ observations. Similarly,
        the interpolated solution inherits all auxiliary info from the $t_1$ state.

        ---------------------------------------------------------------------

        All comments from fixed-interval smoother interpolation apply.

        ---------------------------------------------------------------------

        Difference to standard smoother interpolation:

        In the fixed-point smoother, backward transitions are modified
        to ensure that future operations remain correct.
        Denote the location of the fixed-point with $t_f$. Then,
        the backward transition at $t$ is merged with that at $t_0$.
        This preserves knowledge of how to move from $t$ to $t_f$.

        Then, `t` becomes the new fixed-point location. To ensure
        that future operations "find their way back to $t$":

        - Subsequent interpolations do not continue from the raw
        interpolated value. Instead, they continue from a nearly
        identical state where the backward transition is replaced
        by the identity.

        - Subsequent solver steps do not continue from the initial $t_1$
        state. Instead, they continue from a version whose backward
        model is replaced with the `t-to-t1` transition.


        ---------------------------------------------------------------------

        As a result, each interpolation must return three distinct states:

        1. the interpolated solution,
        2. the state to continue interpolating from,
        3. the state to continue solver stepping from.

        These are intentionally different in the fixed-point smoother.
        """
        # Note to myself: Don't attempt to remove any of them.
        # They're all important. You will break the code (again) :).

        # Extrapolate from t0 to t, and from t to t1.
        # This yields all building blocks.
        _, extrapolated_t = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        conditional_id = posterior_t0.marginal.identity_conditional()
        previous_new = MarkovSequence(
            extrapolated_t.marginal, conditional_id, reverse=extrapolated_t.reverse
        )
        _, extrapolated_t1 = self.predict(
            posterior=previous_new, transition=transition_t_t1
        )

        # Marginalise from t1 to t to obtain the interpolated solution.
        marginal_t1 = posterior_t1.marginal
        conditional_t1_to_t = extrapolated_t1.conditional
        rv_at_t = conditional_t1_to_t.marginalise(marginal_t1)

        # Return the right combination of marginals and conditionals.
        interpolated = MarkovSequence(
            rv_at_t, extrapolated_t.conditional, reverse=extrapolated_t.reverse
        )
        step_from = MarkovSequence(
            posterior_t1.marginal,
            conditional=conditional_t1_to_t,
            reverse=posterior_t1.reverse,
        )
        interp_res = utilities.InterpResult(
            step_from=step_from, interp_from=previous_new
        )

        marginals = interpolated.marginal
        return (marginals, interpolated), interp_res
