"""BVP solver."""

import jax

from probdiffeq import _markov
from probdiffeq.statespace.scalar import corr, extra


def solve(vf, bcond, prior: _markov.MarkovSeqPreconFwd) -> _markov.MarkovSeqPreconFwd:
    """Solve a BVP.

    Improvements:

    - This function solves linear problems. Make it expect linear problems
    - the discretised IBM prior should not be in here, but in the statespace module
    - solve the bridge-nugget problem: it should not be necessary
    - how do we generalise to multidimensional problems?
    - how do we generalise to nonlinear problems?
    - how do we generalise to non-separable BCs?
    - how would mesh refinement work?
    - how would parameter estimation work?
    - which of the new functions in the statespace are actually required?
    - do we always use preconditioning for everything?
    - what is a clean solution for the reverse=True/False choices?
    """
    prior_bridge = constrain_bcond(bcond, prior)
    return constrain_ode(vf, prior_bridge)


def constrain_bcond(
    bcond, prior: _markov.MarkovSeqPreconFwd
) -> _markov.MarkovSeqPreconRev:
    """Constrain a discrete prior to satisfy boundary conditions."""
    bcond_first, bcond_second = bcond

    # First boundary condition
    _, (init, _) = corr.correct_affine_qoi_noisy(prior.init, affine=bcond_first)

    # Extrapolate
    final, conditionals = jax.lax.scan(
        lambda a, b: extra.extrapolate_precon_with_reversal(a, *b),
        init=init,
        xs=(prior.conditional, prior.preconditioner),
        reverse=False,
    )

    # Second boundary condition
    _, (final_bcond, _) = corr.correct_affine_qoi_noisy(final, affine=bcond_second)

    # Return reversed Markov sequence
    return _markov.MarkovSeqPreconRev(
        init=final_bcond, conditional=conditionals, preconditioner=prior.preconditioner
    )


def constrain_ode(vf, prior: _markov.MarkovSeqPreconRev) -> _markov.MarkovSeqPreconFwd:
    As, bs = vf

    # Initialise on the right end
    _, (final, _) = corr.correct_affine_ode_2nd(prior.init, affine=(As[-1], bs[-1]))
    correction_remaining = (As[:-1], bs[:-1])

    # Run the extrapolate-correct Kalman filter
    def step(rv_carry, ssm):
        observation_model, (transition, precon) = ssm
        rv_ext, reversal = extra.extrapolate_precon_with_reversal(
            rv_carry, conditional=transition, preconditioner=precon
        )
        _, (rv_corrected, _) = corr.correct_affine_ode_2nd(
            rv_ext, affine=observation_model
        )
        return rv_corrected, reversal

    extrapolation = (prior.conditional, prior.preconditioner)
    rv, transitions = jax.lax.scan(
        step, init=final, xs=(correction_remaining, extrapolation), reverse=True
    )
    return _markov.MarkovSeqPreconFwd(
        init=rv,
        conditional=transitions,
        preconditioner=prior.preconditioner,
    )
