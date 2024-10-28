"""State-space model implementations."""

from probdiffeq.backend import containers, functools, tree_util
from probdiffeq.impl import _conditional, _linearise, _normal, _prototypes, _stats


@containers.dataclass
class FactImpl:
    """Implementation of factorized state-space models."""

    name: str
    prototypes: _prototypes.PrototypeBackend
    normal: _normal.NormalBackend
    stats: _stats.StatsBackend
    linearise: _linearise.LinearisationBackend
    conditional: _conditional.ConditionalBackend
    transform: _conditional.TransformBackend

    # To assert a valid tree_equal of solutions, the factorisations
    # must be comparable.
    def __eq__(self, other):
        if isinstance(other, FactImpl):
            return self.name == other.name
        return False


def choose(which: str, /, *, tcoeffs_like) -> FactImpl:
    """Choose a state-space model implementation."""
    if which == "dense":
        return _select_dense(tcoeffs_like=tcoeffs_like)
    if which == "isotropic":
        return _select_isotropic(tcoeffs_like=tcoeffs_like)
    if which == "blockdiag":
        return _select_blockdiag(tcoeffs_like=tcoeffs_like)

    msg1 = f"Implementation '{which}' unknown. "
    msg2 = "Choose one out of {'dense', 'isotropic', 'blockdiag'}."
    raise ValueError(msg1 + msg2)


def _select_dense(*, tcoeffs_like) -> FactImpl:
    ode_shape = tcoeffs_like[0].shape
    flat, unravel = tree_util.ravel_pytree(tcoeffs_like)

    num_derivatives = len(tcoeffs_like) - 1

    prototypes = _prototypes.DensePrototype(ode_shape=ode_shape)
    normal = _normal.DenseNormal(ode_shape=ode_shape)
    linearise = _linearise.DenseLinearisation(ode_shape=ode_shape, unravel=unravel)
    stats = _stats.DenseStats(ode_shape=ode_shape, unravel=unravel)
    conditional = _conditional.DenseConditional(
        ode_shape=ode_shape,
        num_derivatives=num_derivatives,
        unravel=unravel,
        flat_shape=flat.shape,
    )
    transform = _conditional.DenseTransform()
    return FactImpl(
        name="dense",
        linearise=linearise,
        transform=transform,
        conditional=conditional,
        normal=normal,
        prototypes=prototypes,
        stats=stats,
    )


def _select_isotropic(*, tcoeffs_like) -> FactImpl:
    ode_shape = tcoeffs_like[0].shape
    num_derivatives = len(tcoeffs_like) - 1

    tcoeffs_tree_only = tree_util.tree_map(lambda *_a: 0.0, tcoeffs_like)
    _, unravel_tree = tree_util.ravel_pytree(tcoeffs_tree_only)
    unravel = functools.vmap(unravel_tree, in_axes=1, out_axes=0)

    prototypes = _prototypes.IsotropicPrototype(ode_shape=ode_shape)
    normal = _normal.IsotropicNormal(ode_shape=ode_shape)
    stats = _stats.IsotropicStats(ode_shape=ode_shape, unravel=unravel)
    linearise = _linearise.IsotropicLinearisation()
    conditional = _conditional.IsotropicConditional(
        ode_shape=ode_shape, num_derivatives=num_derivatives, unravel_tree=unravel_tree
    )
    transform = _conditional.IsotropicTransform()
    return FactImpl(
        name="isotropic",
        prototypes=prototypes,
        normal=normal,
        stats=stats,
        linearise=linearise,
        conditional=conditional,
        transform=transform,
    )


def _select_blockdiag(*, tcoeffs_like) -> FactImpl:
    ode_shape = tcoeffs_like[0].shape
    num_derivatives = len(tcoeffs_like) - 1

    tcoeffs_tree_only = tree_util.tree_map(lambda *_a: 0.0, tcoeffs_like)
    _, unravel_tree = tree_util.ravel_pytree(tcoeffs_tree_only)
    unravel = functools.vmap(unravel_tree)

    prototypes = _prototypes.BlockDiagPrototype(ode_shape=ode_shape)
    normal = _normal.BlockDiagNormal(ode_shape=ode_shape)
    stats = _stats.BlockDiagStats(ode_shape=ode_shape, unravel=unravel)
    linearise = _linearise.BlockDiagLinearisation()
    conditional = _conditional.BlockDiagConditional(
        ode_shape=ode_shape, num_derivatives=num_derivatives, unravel_tree=unravel_tree
    )
    transform = _conditional.BlockDiagTransform(ode_shape=ode_shape)
    return FactImpl(
        name="blockdiag",
        prototypes=prototypes,
        normal=normal,
        stats=stats,
        linearise=linearise,
        conditional=conditional,
        transform=transform,
    )
