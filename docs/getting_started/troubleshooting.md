# Troubleshooting

## General troubleshooting

If you encounter unexpected issues, please ensure you have the latest version of JAX installed. 
If you're not already using [virtual environments](https://docs.python.org/3/tutorial/venv.html), now might be a good time to start, as they can help manage dependencies more effectively.

With these points covered, try to execute some of the examples in Probdiffeq's documentation, for example [the easy example](https://pnkraemer.github.io/probdiffeq/examples_quickstart/easy_example/).
If these examples work $-$ great! If not, reach out. 

Unlike many other JAX-based scientific computing libraries, probdiffeq works best with double precision. 
This is because, during solver initialization, it computes the Cholesky factor of a Hilbert matrix (with somewhere between 2-12 rows), which needs high precision.

## Long compilation times

If a solution routine takes an unexpectedly long time to compile but runs quickly afterward, the issue might be related to how Taylor coefficients are computed. 
Some functions in `probdiffeq.taylor` unroll a small loop, which can slow down compilation.  

To avoid this, try using `probdiffeq.taylor.taylor.odejet_padded_scan()`, which replaces loop unrolling with a scan.  

If the problem persists, consider:  

- Reducing the number of derivatives (if appropriate for your problem).  
- Switching to a different Taylor-coefficient routine, such as a Runge-Kutta starter with `probdiffeq.taylor.taylor.runge_kutta_starter()`.  

For $\nu < 5$, using a Runge-Kutta starter should maintain solver performance. However, for higher-order methods (e.g., \(\nu = 9\)), `taylor_fn=taylor.odejet_fn` appears to be the best choice.  


## Taylor-derivative routines yield NaNs

If you encounter unexpected NaNs while estimating Taylor derivative routines, the issue might come from the vector field itself!
For instance, in the Pleiades problem, there's a term like $\|x\|^2 / (\|x\|^2 + \|y\|^2)$, which can have differentiability issues near zero, depending on how it's implemented. 
See [this issue (external)](https://github.com/pnkraemer/diffeqzoo/issues/126) for more details.
In some cases, the fix is as simple as wrapping the quotient in `jax.numpy.nan_to_num`. 
You can also check out [Probdiffeq's Pleiades benchmark](https://github.com/pnkraemer/probdiffeq/blob/main/docs/benchmarks/pleiades/run_pleiades.py) for a concrete example.

## Other problems
Your problem is not discussed here? Feel free to reach out $-$ opening an issue is a great way to get help!
