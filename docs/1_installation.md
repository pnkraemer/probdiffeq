# Installation

Install the latest release from PyPI:

```bash
pip install probdiffeq
```

> This assumes [JAX](https://jax.readthedocs.io/en/latest/) is already installed.  

To install with JAX (CPU backend):  

```bash
pip install probdiffeq[cpu]
```

⚠️ **Note:** This is an active research project. Expect rough edges and breaking API changes.




**Versioning**

Probdiffeq follows **0.MINOR.PATCH** until its first stable release:  
- **PATCH** → bugfixes & new features  
- **MINOR** → breaking changes  

See [semantic versioning](https://semver.org/).
Notably, Probdiffeq's API is not guaranteed to be stable, but we do our best to follow the versioning scheme so that downstream projects remain reproducible.
