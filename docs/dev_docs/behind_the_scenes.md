# Behind the scenes

How are the `odefilter` solvers implemented, and why are they implemented the way they are?



## States, algorithms, parameters and factories

The `odefilter` solvers and all subroutines they rely on are accessible via factory methods:
```python
solver, solver_params = ek0()
controller, control_params = step.pi_control()
algo, params = inits.taylor_mode()
# etc.
```
If we pass options, these structures are nested:
```python
solver, solver_params = ek0(step_control=step.pi_control())
# solver.step_control contains the controller
# solver_params.step_control contains the control_params

solver, solver_params = ek0(init=inits.taylor_mode(), step_control=step.pi_control())
# solver.step_control contains the controller
# solver.inits contains the initialisation routine
# solver_params.step_control contains the control_params
# solver_params.inits contains the initialisation routine parameters
```

The algorithms are strictly separated from the parameters.

_**Why?**_

* Just-in-time compilation: We recompile when algorithms change, but not when algorithm parameters change. This is easy to implement if algorithms and algorithm parameters are separate.
* Model tuning: improving the simulation models requires parameter optimisation, which requires gradients. Differentiating a function with respect to algorithm parameters is easy to implement if algorithms and algorithm parameters are separate.
* Complexity: It is easy to understand how much memory an algorithm consumes if all memory consumption is isolated in a single data structure.

Algorithms become pure functions, parameters are valid pytrees. JAX' power unfolds.

For the same reason are the algorithms stateless, too.
The commonly behave like:
```python
state = init_fn(problem, params)
state = update_fn(state, problem, params)
```
And details are explained in the next section.


## Init, update, and extract

Iterative algorithms in JAX often follow the

```python
init_fn, update_fn, extract_fn = algorithm_factory(params)

state = init_fn(x, *relevant_args)
state = update_fn(state, *relevant_args)
qoi = extract_fn(state, *relevant_args)
```
pattern ([see here](https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html)).
This is implemented in one way or another by BlackJax, Flax, Optax, and many other JAX-based simulation codes.

Here, we follow a similar pattern:
As explained in the previous section, algorithms in `odefilter` behave like
```python
state = init_fn(problem, params)
state = update_fn(state, problem, params)
```
For example, we solve ODEs as
```python
solver, solver_params = algorithm_factory(params)

state = solver.init_fn(ode_problem, solver_params)
while state.t < ode_problem.tmax:
    state = solver.perform_step_fn(state, ode_problem, solver_params)
solver.extract_fn(state, ode_problem, solver_params)
```
where the solver, which implements `init_fn`, `perform_step_fn`, and `extract_fn`, replaces the `init_fn, update_fn, extract_fn` triple;
and the `perform_step_fn` updates the state given the ODE problem and the solver parameters.
Sometimes, there will be slight nuances with respect to the `problem` variable.


But even inside the ODE solver, algorithms perform this pattern.
For example, the adaptive step-selection does
```python
alg, params = adaptive(solver=ek0(), atol=1e-4, rtol=1e-4)
state = alg.init_fn(problem, params)  # proposes first step size
state = alg.update_fn(state, problem, params)  # Proposes the next dt
```


This is very similar to the init-update-extract pattern, but makes a stronger distinction between a solver-state and a solver-parameter.

_**Why?**_

Because states change over time, but parameters do not, and if we strictly separate them, JAX' function transformations (e.g. `vmap`)
and control flow (e.g. `scan`) immediately do the right thing.
