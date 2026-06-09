"""Tests for output scales."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, np, structs, testing, tree
from probdiffeq.backend.typing import Callable, Literal


@structs.dataclass
class ScaleShapeRules:
    """Configuration for the output-scale test cases."""

    ssm_fact: Literal["dense", "isotropic", "blockdiag"]  # noqa: F821
    prior: Callable
    ode: tuple
    base: tuple
    calibrated: tuple
    base_baddies: list[tuple]
    calibrated_baddies: list[tuple]


def case_scale_rules_iwp_dense() -> ScaleShapeRules:
    return ScaleShapeRules(
        ssm_fact="dense",
        prior=probdiffeq.prior_wiener_integrated,
        ode=(1, 1),
        base=(1, 1),
        calibrated=(),
        base_baddies=[(), (1,)],
        calibrated_baddies=[(1,), (1, 1)],
    )


def case_scale_rules_ioup_dense() -> ScaleShapeRules:
    def linop(s):
        return tree.tree_map(np.zeros_like, s)

    prior = func.partial(probdiffeq.prior_ornstein_uhlenbeck_integrated, linop)
    return ScaleShapeRules(
        ssm_fact="dense",
        prior=prior,
        ode=(1,),
        base=(1,),
        calibrated=(),
        base_baddies=[(), (1, 1)],
        calibrated_baddies=[(1,), (1, 1)],
    )


def case_scale_rules_iwp_blockdiag() -> ScaleShapeRules:
    return ScaleShapeRules(
        ssm_fact="blockdiag",
        prior=probdiffeq.prior_wiener_integrated,
        ode=(1, 1),
        base=(1, 1),
        calibrated=(1,),
        base_baddies=[(), (1,)],
        calibrated_baddies=[(), (1, 1)],
    )


def case_scale_rules_iwp_isotropic() -> ScaleShapeRules:
    return ScaleShapeRules(
        ssm_fact="isotropic",
        prior=probdiffeq.prior_wiener_integrated,
        ode=(1, 1),
        base=(),
        calibrated=(),
        base_baddies=[(1,), (1, 1)],
        calibrated_baddies=[(1,), (1, 1)],
    )


@testing.parametrize_with_cases("rules", cases=".", prefix="case_scale_rules_")
def test_output_scales_covariances_scaled_correctly_default(rules: ScaleShapeRules):
    # Test that the transition covariances are scaled correctly
    ssm = probdiffeq.state_space_model(ssm_fact=rules.ssm_fact)

    # 1d problem, but "unusual" shapes. Values don't matter.
    tcoeffs = [np.ones(rules.ode), np.ones(rules.ode)]
    _init, iwp = rules.prior(tcoeffs, ssm=ssm)
    cond = iwp(1.0)
    Q_expected = 1.0 / np.asarray([[3.0, 2.0], [2.0, 1.0]])
    _, cov = cond.noise.to_multivariate_normal()
    assert testing.allclose(cov, Q_expected)


@testing.parametrize_with_cases("rules", cases=".", prefix="case_scale_rules_")
def test_output_scales_covariances_scaled_correctly_custom(rules: ScaleShapeRules):
    # Test that the transition covariances are scaled correctly
    ssm = probdiffeq.state_space_model(ssm_fact=rules.ssm_fact)

    # 1d problem, but "unusual" shapes. Values don't matter.
    tcoeffs = [np.ones(rules.ode), np.ones(rules.ode)]
    scale = 123.45 * np.ones(rules.base)
    _init, iwp = rules.prior(tcoeffs, output_scale=scale, ssm=ssm)

    cond = iwp(1.0, 9.876 * np.ones(rules.calibrated))
    Q_expected = (9.876 * 123.45) ** 2.0 * 1.0 / np.asarray([[3.0, 2.0], [2.0, 1.0]])
    _, cov = cond.noise.to_multivariate_normal()
    assert testing.allclose(cov, Q_expected)


@testing.parametrize_with_cases("rules", cases=".", prefix="case_scale_rules_")
def test_output_scales_wrong_shape_raises_error_at_construction(rules: ScaleShapeRules):
    ssm = probdiffeq.state_space_model(ssm_fact=rules.ssm_fact)

    # Sanity check: assert that the same error does not happen with the correct shape
    tcoeffs = [np.ones(rules.ode), np.ones(rules.ode)]
    _ = rules.prior(tcoeffs, output_scale=np.ones(rules.base), ssm=ssm)

    # Test that for the wrong shape or type, an error is raised during construction
    for shapes in rules.base_baddies:
        scale = 123.45 * np.ones(shapes)
        with testing.raises(ValueError, match="wrong shape"):
            _ = rules.prior(tcoeffs, output_scale=scale, ssm=ssm)


@testing.parametrize_with_cases("rules", cases=".", prefix="case_scale_rules_")
def test_output_scales_wrong_shape_raises_error_at_calling(rules: ScaleShapeRules):
    ssm = probdiffeq.state_space_model(ssm_fact=rules.ssm_fact)

    tcoeffs = [np.ones(rules.ode), np.ones(rules.ode)]
    _init, iwp = rules.prior(tcoeffs, ssm=ssm)

    # Sanity check: assert that the same error does not happen with the correct shape
    _ = iwp(1.0, np.ones(rules.calibrated))

    for shapes in rules.calibrated_baddies:
        with testing.raises(ValueError, match="wrong shape"):
            _ = iwp(1.0, np.ones(shapes))


@testing.parametrize_with_cases("rules", cases=".", prefix="case_scale_rules_")
def test_output_scales_wrong_type_raises_error(rules: ScaleShapeRules):
    ssm = probdiffeq.state_space_model(ssm_fact=rules.ssm_fact)

    # Sanity check: assert that the same error does not happen with the correct shape
    tcoeffs = [{"u": np.ones(rules.ode)}, {"u": np.ones(rules.ode)}]

    # output scale should inherit pytree structure
    if rules.ssm_fact == "isotropic":
        scale_good = np.ones(rules.base)  # bad: not a dict
        scale_bad = {"u": scale_good}  # good: dict
    else:
        scale_bad = np.ones(rules.base)  # bad: not a dict
        scale_good = {"u": scale_bad}  # good: dict

    _ = rules.prior(tcoeffs, output_scale=scale_good, ssm=ssm)

    # Test that for the wrong shape or type, an error is raised during construction
    with testing.raises(TypeError, match="unexpected PyTree structure"):
        _ = rules.prior(tcoeffs, output_scale=scale_bad, ssm=ssm)
