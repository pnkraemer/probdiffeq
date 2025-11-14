# Creating an new example/benchmark

Probdiffeq hosts numerous tutorials and benchmarks that demonstrate the library.
The differences between examples and benchmarks are minimal. Generally, they are all jupyter notebooks (paired to Markdown files via jupytext for version control) and demonstrate one functionality. The difference between an example and a benchmark is that the examples show *what* probdiffeq offers,
and the benchmarks demonstrate *how well* probdiffeq offers it, e.g. in comparison to other solver libraries. Each tutorial or benchmark runs in under a minute.
New examples are welcome! To create a new example or benchmark, follow these steps:



1. Create a new script in the corresponding subdirectory of the `docs/` directory. The resulting path should look like: `docs/examples_benchmarks/benchmark-name.ipynb` or `docs/examples_advanced/example-name.ipynb` or similar. Choose a meaningful name for your benchmark, e.g. `scipy-comparison-hires` or `demonstrate-calibration`. In case you are wondering which examples-subfolder is most appropriate: if your notebook introduces an external dependency (for example, an optimisation or sampling library), then it is an advanced tutorial. 



2. Create a Jupyter notebook that contains the example or benchmark. This notebook should execute the script and plot the results. The run itself not take too long (think, less much than a minute), otherwise the continuous integration (which verifies the examples and benchmarks run correctly) grows out of hand.

3. Link the notebook to a markdown file via jupytext (for better version control):

      ```
      jupytext --set-formats ipynb,py:light  <new-benchmark-notebook.ipynb>
      ```

      If not done already, run `pip install .[doc]` to install dependencies like jupytext.

4. Include the notebook (the .ipynb file) into the docs by mentioning it in the `nav:` entry in  `mkdocs.yml`. Check whether the markdown script is covered by the rules in `mkdocs.yml` under `exclude:` -- if not, include it here.
5. Mention the new benchmark in the makefile (`benchmarks-run`). T
6. If the example requires access to an external dependency, mention it under the `doc` optional dependencies in the `pyproject.toml`.
