"""Loss functions, including log-marginal likelihoods."""

from probdiffeq import ssm_impl
from probdiffeq.backend import func, linalg, np, tree
from probdiffeq.backend.typing import Sequence, TypeVar

__all__ = ["loss_lml_terminal_values", "loss_lml_timeseries"]

C = TypeVar("C", bound=Sequence)
"""A type-variable to describe sequences.

Used to type Taylor coefficients, for example.
"""

N = TypeVar("N", bound=ssm_impl.AbstractTreeNormal)
"""A type-variable to describe normal distributions.

Used to type marginals, for example.
"""


def loss_lml_terminal_values(*, ssm: ssm_impl.FactSsmImpl, tcoeff_index=0):
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

        model = ssm.prior.to_derivative(tcoeff_index, std, template=marginals)
        marg = ssm.conditional.marginalise(marginals, model)

        # Wrap into list because blockdiag & isotropic models
        # expect sequences of states (even if the length is one)
        return marg.logpdf_tree([u])

    return loss


def loss_lml_timeseries(
    *,
    ssm: ssm_impl.FactSsmImpl,
    average_pdfs: bool = True,
    tcoeff_index=0,
    solve_triu=linalg.lstsq_svd,
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
            return ssm.prior.to_derivative(tcoeff_index, s, template=posterior.marginal)

        model = func.vmap(make_model)(std)

        # Use solve_triu=lstsq because for noise-free observations, the initial state
        # of the ODE solution tends to be noise-free,
        # which clashes and returns NaNs if we use exact solvers.
        return posterior.evaluate_lml(
            [u], model=model, ssm=ssm, average_pdfs=average_pdfs, solve_triu=solve_triu
        )

    return loss
