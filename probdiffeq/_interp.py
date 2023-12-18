"""Interpolation utilities."""

from probdiffeq.backend import containers, tree_util
from probdiffeq.backend.typing import Generic, TypeVar

T = TypeVar("T")
"""A type-variable corresponding to the posterior-type used in interpolation."""

# todo: rename to: solution, step_from, interpolate_from?
#  in general, this object should not be necessary...


@tree_util.register_pytree_node_class
@containers.dataclass
class InterpRes(Generic[T]):
    accepted: T
    """The new 'accepted' field.

    At time `max(t, s1.t)`. Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.
    """

    solution: T
    """The new 'solution' field.

    At time `t`. This is the interpolation result.
    """

    previous: T
    """The new `previous_solution` field.

    At time `t`. Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.

    The difference between `solution` and `previous` emerges in save_at* modes.
    One belongs to the just-concluded time interval, and the other belongs to
    the to-be-started time interval.
    Concretely, this means that one has a unit backward model and the other
    remembers how to step back to the previous state.
    """

    # make it look like a namedtuple.
    #  we cannot use normal named tuples because we want to use a type-variable
    #  and namedtuples don't support that.
    #  this is a bit ugly, but it does not really matter...
    def __iter__(self):
        return iter(containers.dataclass_astuple(self))

    def tree_flatten(self):
        aux = ()
        children = self.previous, self.solution, self.accepted
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        prev, sol, acc = children
        return cls(previous=prev, solution=sol, accepted=acc)
