# Troubleshooting

## Long compilation times

If a solution routine takes surprisingly long to compile but then executes quickly, 
it may be due to the choice of Taylor-coefficient computation.
Some functions in `probdiffeq.taylor` unroll a (small) loop.
To avoid this, use `probdiffeq.taylor.autodiff.taylor_mode()` 
(which is implemented with a scan).
If the problem persists, reduce the number of derivatives 
(if that is appropriate for your integration problem)
or switch to a different Taylor-coefficient routine.
For example, use a Runge-Kutta starter `probdiffeq.taylor.estim.make_runge_kutta_starter()`.
For $\nu < 5$, switching to Runge-Kutta starters should preserve performance of the solvers.
High-order methods, e.g. $\nu = 9$ seem to rely on `taylor_fn=taylor.taylor_mode_fn`.


## Other problems
Your problem is not discussed here? Please open an issue! 
