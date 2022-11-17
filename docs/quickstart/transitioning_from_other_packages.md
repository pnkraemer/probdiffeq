# Transitioning from other packages

Here is how you get started with ``probdiffeqs`` if you have experience with other (ProbNum-)ODE solver packages in Python/Julia.

The most similar packages to ``probdiffeqs`` are

* Tornadox, because it also implements ProbNum-ODE solvers in JAX, albeit tornadox is not nearly as efficient as the ``probdiffeqs`` code and lacks almost all features implemented here.
* ProbNumDiffEq.jl, because it also implements performant, usable ODE filters. It does not provide the breadth of features that ``probdiffeqs`` provide; especially, the factorised state-space models are not implemented there.
* ProbNum, because it also implements ProbNum ODE solvers, and its compositionality structure is very similar to that of ``probdiffeqs``.
* Diffrax, because it also provides performant ODE solvers in JAX. ``probdiffeqs`` are not _quite_ as performant as Diffrax' solvers, but that is because they compute strictly more: we always return posterior distributions instead of point estimates.
But we are sufficiently close for a user to not notice the differences in speed.

``probdiffeqs`` draws a lot of inspiration from those code bases, without those,
it wouldn't exist.


## Transitioning from Scipy
TBD

## Transitioning from ProbNum
TBD

## Transitioning from Tornadox
TBD

## Transitioning from Diffrax
TBD

## Transitioning from ProbNumDiffEq.jl
TBD
