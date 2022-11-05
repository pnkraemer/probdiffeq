"""Implementations."""


from typing import Generic, TypeVar

E = TypeVar("E")  # think: Extrapolation style
C = TypeVar("C")  # think: Correction style


class Implementation(Generic[C, E]):
    """Implementations.

    Mostly a container for an extrapolation method and a correction method.
    """

    def __init__(self, *, correction: C, extrapolation: E):
        self.correction = correction
        self.extrapolation = extrapolation

    def tree_flatten(self):
        children = (self.correction, self.extrapolation)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        correction, extrapolation = children
        return cls(correction=correction, extrapolation=extrapolation)
