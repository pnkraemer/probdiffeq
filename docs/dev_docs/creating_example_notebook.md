# Creating an example notebook


## Tutorial


To create a new example notebook and include it in the documentation, follow the steps:

1. Create a jupyter notebook, preferably in `docs/examples_*/` and fill it with content.
   In case you are wondering which subfolder is most appropriate: 
   if your notebook introduces an external dependency 
   (for example, an optimisation or sampling library), 
   do not place it next to the solver-configuration notebooks.
2. Pair the notebook with a markdown version of the notebook via jupytext
3. Include the notebook into the docs by mentioning it in the `nav` section of `mkdocs.yml`
4. Update the makefile to enjoy formatting and linting
5. Enjoy.


## Benchmark

1. Create a new folder in the `docs/benchmarks/` directory
2. Create the benchmark script. Usually, the execution is in a python script and the plotting in a jupyter notebook.
3. Link the (plotting-)notebook to a markdown file (for better version control). 
4. Include the (plotting-)notebook into the docs via `mkdocs.yml`. Mention the markdown and python script in the same folder under `mkdocs.yml -> exclude`
5. Mention the new benchmark in the makefile (`clean`, `format`, `run-benchmarks`, `dry-run-benchmarks`). A dry-run is for checking that the code functions properly. The benchmark run itself should not take less than a minute, otherwise the whole benchmark suite grows out of hand.
6. Mention the juypter caches and potential data-storage in the `.gitignore`
