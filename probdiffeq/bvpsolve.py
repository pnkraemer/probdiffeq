"""BVP solver."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq import _markov
from probdiffeq.statespace.scalar import corr, extra, linearise, variables


def solve(
    vf, bcond, grid, *, num_derivatives=4, output_scale=1.0
) -> _markov.MarkovSeqPreconFwd:
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
    prior = ibm_prior_discretised(
        grid, num_derivatives=num_derivatives, output_scale=output_scale
    )
    prior_bridge = bridge(bcond, prior, num_derivatives=num_derivatives)
    return constrain_with_ode(vf, grid, prior_bridge, num_derivatives=num_derivatives)


def ibm_prior_discretised(
    grid, *, num_derivatives, output_scale
) -> _markov.MarkovSeqPreconFwd:
    """Construct the discrete transition densities of an IBM prior."""
    init = variables.standard_normal(num_derivatives + 1, output_scale=output_scale)

    transitions = functools.partial(
        extra.ibm_discretise,
        num_derivatives=num_derivatives,
        output_scale=output_scale,
    )
    transition, precon = jax.vmap(transitions)(jnp.diff(grid))

    return _markov.MarkovSeqPreconFwd(
        init=init, transition=transition, preconditioner=precon
    )


def bridge(
    bcond, prior: _markov.MarkovSeqPreconFwd, *, num_derivatives
) -> _markov.MarkovSeqPreconRev:
    """Bridge a discrete prior according to separable boundary conditions."""
    _shape = (num_derivatives + 1,)
    correction_bcond = _correction_model_bcond(bcond, state_shape=_shape)
    return _constrain_boundary(prior, correction_bcond)


def _correction_model_bcond(bcond, *, state_shape):
    """Transform the boundary condition into constraints on the full state."""
    g0, g1 = bcond

    unimportant_value = jnp.ones(state_shape)
    A_left = linearise.ts1(lambda x: g0(x[0, ...]), unimportant_value)
    A_right = linearise.ts1(lambda x: g1(x[0, ...]), unimportant_value)
    return A_left, A_right


def _constrain_boundary(
    prior: _markov.MarkovSeqPreconFwd, correction
) -> _markov.MarkovSeqPreconRev:
    """Constrain a discrete prior to satisfy boundary conditions.

    This algorithm runs a reverse-time Kalman filter.
    """
    # init, transitions, precons, reverse = prior

    H_first, H_second = correction

    # Initialise (we usually filter in reverse)
    _, (rv_corrected, _) = corr.correct_affine_qoi_noisy(prior.init, H_first)

    # Run the reverse-time Kalman filter
    def step(rv_carry, transition):
        system, precon = transition
        return extra.extrapolate_with_reversal_precon(
            rv_carry, transition=system, precon=precon, output_scale=1.0
        )

    rv_init, transitions = jax.lax.scan(
        step,
        init=rv_corrected,
        xs=(prior.transition, prior.preconditioner),
        reverse=False,
    )

    # Constrain on the remaining end
    _, (rv_corr, _) = corr.correct_affine_qoi_noisy(rv_init, H_second)

    # Return solution
    return _markov.MarkovSeqPreconRev(
        init=rv_corr, transition=transitions, preconditioner=prior.preconditioner
    )


def constrain_with_ode(
    vf, grid, prior: _markov.MarkovSeqPreconRev, *, num_derivatives
) -> _markov.MarkovSeqPreconFwd:
    # Linearise ODE
    state_shape = (num_derivatives + 1,)
    correction_model = _correction_model_ode(vf, grid, state_shape=state_shape)

    # Constrain the states and return the result
    return _kalmanfilter(correction_model, prior)


def _correction_model_ode(vf, mesh, *, state_shape):
    """Linearise the ODE vector field as a function of the state."""

    def residual(x):
        return x[2] - vf(x[0])

    @jax.vmap
    def lin(m):
        return linearise.ts1_matrix(residual, m)

    unimportant_values = jnp.ones(mesh.shape + state_shape)
    return lin(unimportant_values)


def _kalmanfilter(
    correction, prior: _markov.MarkovSeqPreconRev
) -> _markov.MarkovSeqPreconFwd:
    jac, (bias,) = correction

    # Initialise on the right end and run backwards
    _, (rv_corrected, _) = corr.correct_affine_qoi_matrix(
        prior.init, (jac[-1], (bias[-1],))
    )
    correction_remaining = jax.tree_util.tree_map(lambda x: x[:-1], correction)

    # Run the extrapolate-correct Kalman filter
    def step(rv_carry, ssm):
        observation_model, transition_and_precon = ssm
        (transition, precon) = transition_and_precon
        jac, (bias,) = observation_model

        rv_ext, (rv_rev, gain) = extra.extrapolate_with_reversal_precon(
            rv_carry, transition=transition, precon=precon, output_scale=1.0
        )
        _, (rv, _) = corr.correct_affine_qoi_matrix(rv_ext, (jac, (bias,)))
        return rv, (rv_rev, gain)

    extrapolation = (prior.transition, prior.preconditioner)
    rv, transitions = jax.lax.scan(
        step,
        init=rv_corrected,
        xs=(correction_remaining, extrapolation),
        reverse=True,
    )
    return _markov.MarkovSeqPreconFwd(
        init=rv,
        transition=transitions,
        preconditioner=prior.preconditioner,
    )
