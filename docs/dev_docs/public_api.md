# Private and public API

All public functions and class-creators in the following modules and packages are considered public API:

* `ivpsolvers.py`
* `ivpsolve.py`
* `solution.py`
* `taylor.py`
* `controls.py`
* `strategies.filters.py`
* `strategies.smoothers.py`
* `ssm.recipes.py`
* `ssm.cubature.py`

Exceptions of this rule are all functions and class-creators that are 
marked as `warning: highly experimental`, e.g., `taylor.taylor_mode_doubling_fn`.


Breaking changes in these public modules are officially considered breaking changes.
This means that the minor version number is increased according the the rules of semantic versioning
(there has not been a major version yet).
It also means that an entry in the  changelog is warranted, and if deprecation policies are introduced in the future, it would apply to these module.

Everything else is either considered private or experimental.
For example, `ssm.dense.*` is accessible from the standard namespace, but not considered public API.
Changes to this code are treated as bugfixes, breaking or not: 
patch-version increases, changelog entries are optional, and deprecation policy is not considered necessary.


