"""State-space model implementations."""

from probdiffeq.backend import containers, functools, tree_util
from probdiffeq.backend.typing import Callable
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

    num_derivatives: int
    unravel: Callable

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
    ode_shape = tree_util.ravel_pytree(tcoeffs_like[0])[0].shape
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
    return FactImpl(
        name="dense",
        linearise=linearise,
        conditional=conditional,
        normal=normal,
        prototypes=prototypes,
        stats=stats,
        num_derivatives=len(tcoeffs_like) - 1,
        unravel=unravel,
    )


def _select_isotropic(*, tcoeffs_like) -> FactImpl:
    ode_shape = tree_util.ravel_pytree(tcoeffs_like[0])[0].shape
    num_derivatives = len(tcoeffs_like) - 1

    tcoeffs_tree_only = tree_util.tree_map(lambda *_a: 0.0, tcoeffs_like)
    _, unravel_tree = tree_util.ravel_pytree(tcoeffs_tree_only)

    leaves, _ = tree_util.tree_flatten(tcoeffs_like)
    _, unravel_leaf = tree_util.ravel_pytree(leaves[0])

    def unravel(z):
        tree = functools.vmap(unravel_tree, in_axes=1, out_axes=0)(z)
        return tree_util.tree_map(unravel_leaf, tree)

    prototypes = _prototypes.IsotropicPrototype(ode_shape=ode_shape)
    normal = _normal.IsotropicNormal(ode_shape=ode_shape)
    stats = _stats.IsotropicStats(ode_shape=ode_shape, unravel=unravel)
    linearise = _linearise.IsotropicLinearisation(unravel=unravel)
    conditional = _conditional.IsotropicConditional(
        ode_shape=ode_shape, num_derivatives=num_derivatives, unravel_tree=unravel_tree
    )
    return FactImpl(
        name="isotropic",
        prototypes=prototypes,
        normal=normal,
        stats=stats,
        linearise=linearise,
        conditional=conditional,
        num_derivatives=len(tcoeffs_like) - 1,
        unravel=unravel,
    )


def _select_blockdiag(*, tcoeffs_like) -> FactImpl:
    ode_shape = tree_util.ravel_pytree(tcoeffs_like[0])[0].shape
    num_derivatives = len(tcoeffs_like) - 1

    tcoeffs_tree_only = tree_util.tree_map(lambda *_a: 0.0, tcoeffs_like)
    _, unravel_tree = tree_util.ravel_pytree(tcoeffs_tree_only)

    leaves, _ = tree_util.tree_flatten(tcoeffs_like)
    _, unravel_leaf = tree_util.ravel_pytree(leaves[0])

    def unravel(z):
        tree = functools.vmap(unravel_tree, in_axes=0, out_axes=0)(z)
        return tree_util.tree_map(unravel_leaf, tree)

    prototypes = _prototypes.BlockDiagPrototype(ode_shape=ode_shape)
    normal = _normal.BlockDiagNormal(ode_shape=ode_shape)
    stats = _stats.BlockDiagStats(ode_shape=ode_shape, unravel=unravel)
    linearise = _linearise.BlockDiagLinearisation(unravel=unravel)
    conditional = _conditional.BlockDiagConditional(
        ode_shape=ode_shape, num_derivatives=num_derivatives, unravel_tree=unravel_tree
    )
    return FactImpl(
        name="blockdiag",
        prototypes=prototypes,
        normal=normal,
        stats=stats,
        linearise=linearise,
        conditional=conditional,
        num_derivatives=len(tcoeffs_like) - 1,
        unravel=unravel,
    )
