# Private and public API

All public functions and classes in the following modules and packages are considered public API:

* `ivpsolvers.py`
* `ivpsolve.py`
* `solution.py`
* `taylor.py`
* `controls.py`
* `strategies.filters.py`
* `strategies.smoothers.py`
* `implementations.recipes.py`
* `implementations.cubature.py`

Exceptions from this rule are all functions and classes  that are not marked as `warning: highly experimental`,
e.g., `taylor.taylor_mode_doubling_fn`.

Breaking changes in these public modules are officially considered breaking changes.
This means that the minor version number is increased (there has not been a major version yet).
It also means that an entry in the  changelog is warranted, and if deprecation policies are introduced in the future, it would apply to these module.

Everything else is either considered private or experimental.
For example, `implementations.dense.*` is accessible from the standard namespace, but not considered public API.
Changes to this code are treated as bugfixes, breaking or not: 
patch-version increases, changelog entries are optional, and deprecation policy is not considered necessary.


