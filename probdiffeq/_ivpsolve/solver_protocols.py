from probdiffeq.backend.typing import Any, Array, Generic, Protocol, TypeVar

__all__ = ["Solution", "Solver"]
T_contra = TypeVar("T_contra", contravariant=True)
T = TypeVar("T")
S = TypeVar("S")


class Solution(Protocol, Generic[S]):
    """An protocol that defines expected solution types."""

    t: Array
    """The timepoints that the solution is expressed on."""

    u: S
    """The IVP solution state.

    This is the usually the same type as the initial condition.
    """

    num_steps: int
    """The number of steps taken by the solver."""

    solution_full: Any
    """A full description of the solution (beyond 'u', e.g. for dense outputs)."""


# Revisit this dependent typing one Python >=3.12 is enforced
# Concretely, Something like Solver[T, S: Solution[T]](Protocol):...
# can now be written.


class Solver(Protocol, Generic[T_contra, S]):
    """An protocol that defines expected solver types."""

    def init(self, t, u: T_contra, *, damp: float) -> S:
        """Initialise the solver's state."""

    def step(self, state: S, *, dt: float, damp: float) -> S:
        """Perform a step."""

    def interpolate(self, *, t, interp_from: S, interp_to: S) -> Any:
        """Interpolate between two solver states."""

    def interpolate_at_t1(self, *, t, interp_from: S, interp_to: S) -> Any:
        """Interpolate close to a checkpoint."""

    @property
    def is_suitable_for_save_at(self) -> bool:
        """Whether or not the solver can be used with a certain style of adaptive time-stepping."""

    @property
    def is_suitable_for_save_every_step(self) -> bool:
        """Whether or not the solver can be used with a certain style of adaptive time-stepping."""

    def userfriendly_output(self, *, solution: S, solution0: S) -> S:
        """Postprocess the solution before returning."""
