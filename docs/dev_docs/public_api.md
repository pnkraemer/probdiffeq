# Private and public API

All public functions and classes that are in the online documentation 
are considered public API.
At the moment, this affects the following:

* `ivpsolve.py`
* `adaptive.py`
* `taylor/*`
* `solvers/*`
* `impl.impl.select()`

Exceptions to this rule are all functions and class that are 
marked as `warning: highly experimental`, e.g., `taylor.autodiff.taylor_mode_doubling`.


Everything else (`backend`, `util`, `impl`) is not public and breaking changes here will not necessarily increase the version.
