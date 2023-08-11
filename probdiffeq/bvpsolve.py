"""BVP solving functionality.

!!! warning "Warning: highly EXPERIMENTAL!"
    This module is highly experimental.
    There is no guarantee that it works correctly.
    It might be deleted tomorrow
    and without any deprecation policy.

"""

import jax

from probdiffeq.solvers import markov
from probdiffeq.solvers.statespace import correction, extrapolation


def solve_separable_affine_2nd(
    ode, bconds, prior: markov.MarkovSeqPreconFwd, *, bcond_nugget=1e-6
) -> markov.MarkovSeqPreconFwd:
    """Solve an affine, 2nd-order BVP with separable, affine boundary conditions.

    Currently restricted to scalar problems.
    """
    prior_bridge = _constrain_bconds_affine_separable(
        bconds, prior, nugget=bcond_nugget
    )
    return _constrain_ode_affine_2nd(ode, prior_bridge)


def _constrain_bconds_affine_separable(
    bconds, prior: markov.MarkovSeqPreconFwd, *, nugget
) -> markov.MarkovSeqPreconRev:
    """Constrain a discrete prior to satisfy boundary conditions."""
    bcond_first, bcond_second = bconds

    # First boundary condition

    _, (init, _) = correction.correct_affine_qoi_noisy(
        prior.init, bcond_first, stdev=nugget
    )

    # Loop over time
    final, conditionals = jax.lax.scan(
        lambda a, b: extrapolation.extrapolate_precon_with_reversal(a, *b),
        init=init,
        xs=(prior.conditional, prior.preconditioner),
        reverse=False,
    )

    # Second boundary condition
    _, (final, _) = correction.correct_affine_qoi_noisy(
        final, bcond_second, stdev=nugget
    )

    # Return reverse-Markov sequence
    return markov.MarkovSeqPreconRev(
        init=final, conditional=conditionals, preconditioner=prior.preconditioner
    )


def _constrain_ode_affine_2nd(
    vf, prior: markov.MarkovSeqPreconRev
) -> markov.MarkovSeqPreconFwd:
    As, bs = vf

    # First ODE constraint
    _, (final, _) = correction.correct_affine_ode_2nd(
        prior.init, affine=(As[-1], bs[-1])
    )

    # Run the extrapolate-correct loop (which includes the final ODE constraint)

    def step(rv_carry, ssm):
        observation_model, (transition, precon) = ssm
        rv_ext, reversal = extrapolation.extrapolate_precon_with_reversal(
            rv_carry, conditional=transition, preconditioner=precon
        )
        _, (rv_corrected, _) = correction.correct_affine_ode_2nd(
            rv_ext, affine=observation_model
        )
        return rv_corrected, reversal

    correction_remaining = (As[:-1], bs[:-1])
    extrapolation = (prior.conditional, prior.preconditioner)
    rv, transitions = jax.lax.scan(
        step, init=final, xs=(correction_remaining, extrapolation), reverse=True
    )
    return markov.MarkovSeqPreconFwd(
        init=rv,
        conditional=transitions,
        preconditioner=prior.preconditioner,
    )
