"""Markov processes."""

from probdiffeq import ssm_impl
from probdiffeq.backend import flow, func, np, random, structs, tree
from probdiffeq.backend.typing import Any, Callable, Generic, Sequence, TypeVar

__all__ = ["MarkovSequence"]

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

    def evaluate_marginals(self, *, ssm):
        """Extract the (time-)marginals from a Markov sequence.

        This is only needed in combination with smoothing-based strategies.
        """
        if self.marginal.mean_flat.ndim == self.conditional.noise.mean_flat.ndim:
            # TODO: do this in the postprocessing of the solver...
            markov_seq = self.remove_filtering_distributions()
            return markov_seq.evaluate_marginals(ssm=ssm)

        def step(x, cond):
            extrapolated = ssm.conditional.marginalise(x, cond)
            return extrapolated, extrapolated

        _, marginals = flow.scan(
            step, init=self.marginal, xs=self.conditional, reverse=self.reverse
        )

        if self.reverse:
            # Append the terminal marginal to the computed ones
            return tree.tree_array_append(marginals, self.marginal)

        return tree.tree_array_prepend(self.marginal, marginals)

    def evaluate_lml(
        self,
        u,
        *,
        model,
        ssm: ssm_impl.FactSsmImpl,
        average_pdfs: bool,
        solve_triu: Callable,
    ):
        assert self.reverse

        # Process the terminal value
        u0 = tree.tree_map(lambda s: s[-1], u)
        model0 = tree.tree_map(lambda s: s[-1], model)
        pdf0, updated = ssm.conditional.bayes_rule_and_logpdf_tree(
            u0, self.marginal, model0, solve_triu=solve_triu
        )

        # Process the remaining values
        def body(rv_and_logpdf, prior_and_observation_and_data):
            rv, logpdf, num_data = rv_and_logpdf
            prior, observation, data = prior_and_observation_and_data

            predicted = ssm.conditional.marginalise(rv, prior)

            logpdf_n, corrected = ssm.conditional.bayes_rule_and_logpdf_tree(
                data, predicted, observation, solve_triu=solve_triu
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

    def sample(self, key, *, ssm: ssm_impl.FactSsmImpl, shape: tuple = ()):
        """Sample from a Markov sequence."""
        # If the MarkovSequence carries unnecessary filtering marginals, remove them
        if self.marginal.mean_flat.ndim == self.conditional.noise.mean_flat.ndim:
            markov_seq = self.remove_filtering_distributions()
            return markov_seq.sample(key, ssm=ssm, shape=shape)

        # If many samples are required, vmap over recursive calls to sample()
        if len(shape) > 0:
            n, *shape_remaining = shape
            keys = random.split(key, num=n)
            sample_partial = func.partial(self.sample, ssm=ssm, shape=shape_remaining)
            return func.vmap(sample_partial)(keys)

        # Compute a single sample from the Markov sequence

        def body(sample_and_key, cond):
            smp_flat, key = sample_and_key

            # Propagate the previous sample
            predicted = ssm.conditional.apply_flat(smp_flat, cond)

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
