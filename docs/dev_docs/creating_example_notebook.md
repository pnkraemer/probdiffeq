# Creating an example notebook

To embed a new example notebook into the docs, follow the steps:

1. Create a jupyter notebook, preferably in `docs/examples/` or `docs/advanced_examples/` and fill it with content.
   If you are wondering which folder is more appropriate: if your notebook introduces an external dependency (for example, an optimisation or sampling library), it is an advanced example.
2. Pair the notebook with a markdown version of the notebook via jupytext
3. Include the notebook into the docs by mentioning it in the `nav` section of `mkdocs.yml`
4. If the notebook is not in `examples/` or in `advanced_examples/`, consider updating the makefile
5. Enjoy.

The same steps kind-of apply to the benchmarks, too.
