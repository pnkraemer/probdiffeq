# Private and public API

All public functions and classes that are in the online documentation 
are considered public API.
At the moment, this affects the following:

* `ivpsolve.py`
* `adaptive.py`
* `taylor/*`
* `ivpsolvers.py`
* `stats.py`
* `impl.impl.select()`

Exceptions to this rule are all functions and class that are 
marked as `warning: highly experimental`, e.g., `ivpsolve.solve_adaptive_save_at`.


Everything else (e.g. `backend`, `util`, `impl`) is not public and breaking changes here will not necessarily increase the version.
