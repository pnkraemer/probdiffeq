# Creating an example notebook

To create a new example notebook or benchmark and include it in the documentation, follow the steps:

1. Create a jupyter notebook, preferably in `docs/examples_*/` and fill it with content.
   In case you are wondering which subfolder is most appropriate: 
   if your notebook introduces an external dependency 
   (for example, an optimisation or sampling library), 
   do not place it next to the solver-configuration notebooks.
2. Pair the notebook with a markdown version of the notebook via jupytext
3. Include the notebook into the docs by mentioning it in the `nav` section of `mkdocs.yml`
4. Update the makefile to enjoy formatting and linting
5. Enjoy.
