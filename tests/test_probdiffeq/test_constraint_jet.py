"""Assert that all jet-linearisation constraints are correct."""

from probdiffeq import probdiffeq
from probdiffeq.backend import linalg, np, structs, testing
from probdiffeq.backend.typing import Array, Callable, Literal


@structs.dataclass
class Root:
    """Different APIs for one and the same root-finding problem.

    Its purpose is to enable different jet-constraints to solve the same
    problem but work with different roots.
    """

    root: Callable
    root_dae_algebraic: Callable
    root_dae_differential: Callable
    ode_vf: Callable
    t0: float
    u0: Array


@testing.fixture(name="root")
def fixture_root_sir():

    def root(u, du, *, t):
        linear = imex_linear(u, du, t=t)
        nonlinear = imex_nonlinear(u, du, t=t)
        return linear + nonlinear

    def imex_linear(u, du, *, t):
        del t
        del u
        return du

    def imex_nonlinear(u, du, *, t):
        del du
        return -vf(u, t=t)

    def vf(y, *, t):
        del t
        # infection and recovery rates
        beta, gamma = 2.0, 0.5

        # 'noqa' because "I" is unambiguous in an SIR model...
        S, I, _R = y  # noqa: E741

        # Dynamics
        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I
        f2 = gamma * I
        return np.stack([f0, f1, f2])

    def dae_algebraic(u, /, *, t):
        del t
        N = 1.0  # total population
        return u[0] + u[1] + u[2] - N

    def dae_differential(u, du, /, *, t):
        del t
        beta, gamma = 2.0, 0.5
        S, I, _R = u  # noqa: E741 ("I" is a good variable name in an SIR model)

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I

        F1 = du[0] - f0
        F2 = du[1] - f1

        return np.stack([F1, F2])

    u0 = np.asarray([0.99, 0.01, 0.0])
    t0 = 0.0
    return Root(
        root=root,
        root_dae_differential=dae_differential,
        root_dae_algebraic=dae_algebraic,
        ode_vf=vf,
        t0=t0,
        u0=u0,
    )


@testing.fixture(name="expected")
@testing.parametrize("derivatives", [1, 5])
def fixture_expected(root, derivatives):
    @probdiffeq.ode
    def vf(y, /, *, t):
        return root.ode_vf(y, t=t)

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=derivatives)
    return jetexpand(vf, [root.u0], t=root.t0)


def case_jet_dae_iterated(root):
    nlstsq = probdiffeq.wlstsq_nc_gauss_newton(maxiter=50, tol=1e-10)
    linearization = probdiffeq.linearization_map(nlstsq)

    def constraint_residual(ssm, lift_by: int):
        differential = probdiffeq.residual(root.root_dae_differential)
        differential = probdiffeq.jet_lift(differential, lift_by=lift_by)
        algebraic = probdiffeq.residual(root.root_dae_algebraic)
        algebraic = probdiffeq.jet_lift(algebraic, lift_by=lift_by + 1)
        dae = probdiffeq.dae(differential, algebraic)

        return probdiffeq.constraint_dae(dae, ssm=ssm, linearization=linearization)

    return constraint_residual


def case_jet_constraint_iterated(root):
    nlstsq = probdiffeq.wlstsq_nc_gauss_newton(maxiter=50, tol=1e-10)
    linearization = probdiffeq.linearization_map(nlstsq)

    def constraint_residual(ssm, lift_by):
        implicit = probdiffeq.residual(root.root)
        implicit = probdiffeq.jet_lift(implicit, lift_by=lift_by)
        return probdiffeq.constraint_residual(
            implicit, ssm=ssm, linearization=linearization
        )

    return constraint_residual


@testing.parametrize_with_cases("jet_factory", cases=".", prefix="case_jet_")
@testing.parametrize("lift_by", [0, "max"])
@testing.parametrize("ssm_fact", ["dense"])
def test_posterior_linearisation_matches_closed_form_recursion(
    root: Root,
    jet_factory: Callable,
    expected: list,
    lift_by: int | Literal["max"],
    ssm_fact,
):
    derivatives = len(expected) - 1
    ssm = probdiffeq.state_space_model(ssm_fact=ssm_fact)
    init, _iwp = probdiffeq.prior_wiener_integrated(
        [root.u0], diffuse_derivatives=derivatives, ssm=ssm
    )

    if lift_by == "max":
        lift_by = len(expected) - 2
    else:
        lift_by = lift_by
    constraint = jet_factory(ssm=ssm, lift_by=lift_by)

    cstate = constraint.init_linearization()
    fx, cstate = constraint.linearize(init, cstate, damp=0.0, t=root.t0)

    _observed, reverted = ssm.conditional.revert(init, fx, solve_triu=linalg.lstsq_svd)

    updated = ssm.conditional.apply_flat(0.0, reverted)
    received = updated.mean
    if lift_by == "max":
        assert testing.allclose(received, expected)
    else:
        assert testing.allclose(received[: 2 + lift_by], expected[: 2 + lift_by])


@testing.parametrize_with_cases("jet_factory", cases=".", prefix="case_jet_")
@testing.parametrize("wrong_lift_by", [-1, 4, 100])
@testing.parametrize("ssm_fact", ["dense"])
def test_wrong_lift_by_raises_error(
    root: Root, jet_factory: Callable, wrong_lift_by, ssm_fact
):
    # 5 Taylor coefficients + root-orders of 2 (constraints depend on u and du).
    ssm = probdiffeq.state_space_model(ssm_fact=ssm_fact)
    init, _ = probdiffeq.prior_wiener_integrated(
        [root.u0], diffuse_derivatives=4, ssm=ssm
    )
    constraint = jet_factory(ssm=ssm, lift_by=wrong_lift_by)

    cstate = constraint.init_linearization()
    with testing.raises(ValueError, match="Received"):
        _ = constraint.linearize(init, cstate, damp=0.0, t=root.t0)
