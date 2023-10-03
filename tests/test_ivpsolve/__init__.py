"""Test the correctness of the solution routines.

There is a base-configuration:

```python
solver = test_util.generate_solver()
solution = solve_and_save_every_step(lotka_volterra, solver)
```

If this approximation is accurate (measured in error comparing to e.g. diffrax),
the solvers work (in principle).

To guarantee that the other solution routines also work, we do the following:

* simulate_terminal_values() with the same arguments
should yield the same terminal value as the base case.
* solve_and_save_at() should be identical to
interpolating the solve_and_save_every_step results
* solve_fixed_grid() should be identical
if the fixed grid is the solution grid of solve_and_save_every_step

If these tests pass, and assuming that interpolation is correct,
the solution routines must be correct.
As a result, we can run all other tests with either one of the solution routines
without a loss of generality.
"""

# todo: we use solve_while_loop separately in each file.
#  We could reuse those simulations and cut the runtime in half, but
#  at the price of making the tests more complicated.
