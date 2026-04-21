"""Assert that all jet-linearisation constraints are correct."""

from probdiffeq import diffeqjet, probdiffeq
from probdiffeq.backend import linalg, np, structs, testing
from probdiffeq.backend.typing import Array, Callable, Literal
from probdiffeq.util import nlstsq_util


@structs.dataclass
class Root:
    """Different APIs for one and the same root-finding problem.

    Its purpose is to enable different jet-constraints to solve the same
    problem but work with different roots.
    """

    root: Callable
    root_imex_linear: Callable
    root_imex_nonlinear: Callable
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
        root_imex_linear=imex_linear,
        root_imex_nonlinear=imex_nonlinear,
        root_dae_differential=dae_differential,
        root_dae_algebraic=dae_algebraic,
        ode_vf=vf,
        t0=t0,
        u0=u0,
    )


@testing.fixture(name="expected")
@testing.parametrize("derivatives", [1, 5])
def fixture_expected(root, derivatives):
    def vf_autonomous(y, /):
        return root.ode_vf(y, t=root.t0)

    return diffeqjet.odejet_padded_scan(vf_autonomous, [root.u0], num=derivatives)


def case_jet_iterated_dae(root):
    nlstsq = nlstsq_util.nlstsq_constrained_gauss_newton(maxiter=50, tol=1e-10)

    def constraint(ssm, jet_order):
        return probdiffeq.constraint_jet_dae(
            differential=root.root_dae_differential,
            algebraic=root.root_dae_algebraic,
            ssm=ssm,
            nlstsq=nlstsq,
            jet_order_differential=jet_order,
            jet_order_algebraic=jet_order + 1 if jet_order != "max" else "max",
        )

    return constraint


def case_jet_iterated_imex(root):
    nlstsq = nlstsq_util.nlstsq_constrained_gauss_newton(maxiter=50, tol=1e-10)

    def constraint(ssm, jet_order):
        return probdiffeq.constraint_jet_imex(
            implicit=root.root_imex_linear,
            explicit=root.root_imex_nonlinear,
            ssm=ssm,
            nlstsq=nlstsq,
            jet_order_implicit=jet_order,
            jet_order_explicit=jet_order,
        )

    return constraint


def case_jet_iterated(root):
    nlstsq = nlstsq_util.nlstsq_constrained_gauss_newton(maxiter=50, tol=1e-10)

    def constraint(ssm, jet_order):
        return probdiffeq.constraint_jet(
            root.root, ssm=ssm, nlstsq=nlstsq, jet_order=jet_order
        )

    return constraint


@testing.parametrize_with_cases("jet_factory", cases=".", prefix="case_jet_")
@testing.parametrize("jet_order", [0, "max"])
def test_posterior_linearisation_matches_closed_form_recursion(
    root: Root, jet_factory: Callable, expected: list, jet_order: int | Literal["max"]
):
    derivatives = len(expected) - 1
    init, ssm = probdiffeq.ssm_taylor([root.u0], diffuse_derivatives=derivatives)
    constraint = jet_factory(ssm=ssm, jet_order=jet_order)

    cstate = constraint.init_linearization()
    fx, cstate = constraint.linearize(init.marginals, cstate, damp=0.0, t=root.t0)

    _observed, reverted = ssm.conditional.revert(
        init.marginals, fx, solve_triu=linalg.lstsq_svd
    )

    updated = ssm.conditional.apply(0.0, reverted)
    received = updated.mean_tree()
    if jet_order == "max":
        assert testing.allclose(received, expected)
    else:
        assert testing.allclose(received[: 2 + jet_order], expected[: 2 + jet_order])


@testing.parametrize_with_cases("jet_factory", cases=".", prefix="case_jet_")
@testing.parametrize("wrong_jet_order", [-1, 4, 100])
def test_wrong_jet_order_raises_error(
    root: Root, jet_factory: Callable, wrong_jet_order
):
    # 5 Taylor coefficients + root-orders of 2 (constraints depend on u and du).
    init, ssm = probdiffeq.ssm_taylor([root.u0], diffuse_derivatives=4)
    constraint = jet_factory(ssm=ssm, jet_order=wrong_jet_order)

    cstate = constraint.init_linearization()
    with testing.raises(ValueError, match="Received"):
        _ = constraint.linearize(init.marginals, cstate, damp=0.0, t=root.t0)


@testing.parametrize_with_cases("jet_factory", cases=".", prefix="case_jet_")
def test_max_jet_order_is_tight(root: Root, jet_factory: Callable):
    init, ssm = probdiffeq.ssm_taylor([root.u0], diffuse_derivatives=4)
    c1 = jet_factory(ssm=ssm, jet_order="max")
    cstate = c1.init_linearization()
    output1 = c1.linearize(init.marginals, cstate, damp=0.0, t=root.t0)

    # 5 Taylor coefficients (u, du, ddu, dddu)
    # and a root-order of 2 (constraints depend on u and du)
    # implies the max feasible jet order is 3
    c2 = jet_factory(ssm=ssm, jet_order=3)
    cstate = c2.init_linearization()
    output2 = c2.linearize(init.marginals, cstate, damp=0.0, t=root.t0)

    # The outputs should be identical
    assert testing.allclose(output1, output2)
