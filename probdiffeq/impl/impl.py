"""State-space model implementations."""

from probdiffeq.backend import func, structs, tree
from probdiffeq.backend.typing import Callable
from probdiffeq.impl import _conditional, _normal, _prototypes


@structs.dataclass
class FactImpl:
    """Implementation of factorized state-space models."""

    name: str
    prototypes: _prototypes.PrototypeBackend
    normal: _normal.Normal
    linearize: _conditional.LinearizationFactoryBackend
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
    ode_shape = tree.ravel_pytree(tcoeffs_like[0])[0].shape
    flat, unravel = tree.ravel_pytree(tcoeffs_like)

    num_derivatives = len(tcoeffs_like) - 1

    prototypes = _prototypes.DensePrototype(ode_shape=ode_shape)
    normal = _normal.NormalDense
    linearize = _conditional.DenseLinearizationFactory(
        ode_shape=ode_shape, unravel=unravel
    )
    conditional = _conditional.DenseConditional(
        ode_shape=ode_shape,
        num_derivatives=num_derivatives,
        unravel=unravel,
        flat_shape=flat.shape,
    )
    return FactImpl(
        name="dense",
        linearize=linearize,
        conditional=conditional,
        normal=normal,
        prototypes=prototypes,
        num_derivatives=len(tcoeffs_like) - 1,
        unravel=unravel,
    )


def _select_isotropic(*, tcoeffs_like) -> FactImpl:
    ode_shape = tree.ravel_pytree(tcoeffs_like[0])[0].shape
    num_derivatives = len(tcoeffs_like) - 1

    tcoeffs_tree_only = tree.tree_map(lambda *_a: 0.0, tcoeffs_like)
    _, unravel_tree = tree.ravel_pytree(tcoeffs_tree_only)

    leaves, tree_structure = tree.tree_flatten(tcoeffs_like)
    _, unravel_leaf = tree.ravel_pytree(leaves[0])

    def unravel(z):
        pytree = func.vmap(unravel_tree, in_axes=1, out_axes=0)(z)
        return tree.tree_map(unravel_leaf, pytree)

    prototypes = _prototypes.IsotropicPrototype(ode_shape=ode_shape)
    normal = _normal.NormalIso
    linearize = _conditional.IsotropicLinearizationFactory(unravel=unravel)
    conditional = _conditional.IsotropicConditional(
        ode_shape=ode_shape,
        num_derivatives=num_derivatives,
        unravel_tree=unravel_tree,
        tree_structure=tree_structure,
    )
    return FactImpl(
        name="isotropic",
        prototypes=prototypes,
        normal=normal,
        linearize=linearize,
        conditional=conditional,
        num_derivatives=len(tcoeffs_like) - 1,
        unravel=unravel,
    )


def _select_blockdiag(*, tcoeffs_like) -> FactImpl:
    ode_shape = tree.ravel_pytree(tcoeffs_like[0])[0].shape
    num_derivatives = len(tcoeffs_like) - 1

    tcoeffs_tree_only = tree.tree_map(lambda *_a: 0.0, tcoeffs_like)
    _, unravel_tree = tree.ravel_pytree(tcoeffs_tree_only)

    leaves, treedef = tree.tree_flatten(tcoeffs_like)
    _, unravel_leaf = tree.ravel_pytree(leaves[0])

    def unravel(z):
        pytree = func.vmap(unravel_tree, in_axes=0, out_axes=0)(z)
        return tree.tree_map(unravel_leaf, pytree)

    prototypes = _prototypes.BlockDiagPrototype(ode_shape=ode_shape)
    normal = _normal.NormalBlockDiag  # (ode_shape=ode_shape)
    linearize = _conditional.BlockDiagLinearizationFactory(unravel=unravel)
    conditional = _conditional.BlockDiagConditional(
        ode_shape=ode_shape,
        num_derivatives=num_derivatives,
        unravel_tree=unravel_tree,
        treedef=treedef,
        unravel_leaf=unravel_leaf,
    )
    return FactImpl(
        name="blockdiag",
        prototypes=prototypes,
        normal=normal,
        linearize=linearize,
        conditional=conditional,
        num_derivatives=len(tcoeffs_like) - 1,
        unravel=unravel,
    )
