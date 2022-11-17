# Iterative algorithms

ODE solvers are nested iterative algorithms:

* Step iteratively through time; at each time-step:
* Reject steps iteratively until a desired acceptance criterion is met; at each step attempt:
* Iterate through some linearisations/stages to complete the step attempt.

To combine this nested logic of iteration with JAX' functional programming paradigm,
``probdiffeqs`` use an init-apply-extract mechanism
just like many other JAX-based simulation libraries ([Optax](https://optax.readthedocs.io/en/latest/index.html), [Blackjax](https://blackjax-devs.github.io/blackjax/), etc.).

Algorithms in the ``probdiffeqs`` package are collections of parametrised
functions (thanks @[Equinox](https://docs.kidger.site/equinox/)!).
One such collection roughly implements the ``alg`` bit of:
```python
def iterative_alg(*problem, alg, cond_fun):
    """Iterative, stateless computation in the ``probdiffeqs`` package."""
    state = alg.init_fn(*problem)

    while cond_fun(state):
        state = alg.apply_fn(*problem, state=state)

    return alg.extract_fn(state)

```
To be more specific:
`init_fn` and `extract_fn` work pretty much always exactly as stated here,
but `apply_fn` is often called `step_fn`, sometimes called `attempt_step_fn`,
and so on. There is no unified naming scheme on the `apply_fn` level.
