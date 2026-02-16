# Creating a new example/benchmark

Probdiffeq hosts numerous tutorials and benchmarks that demonstrate the library. The differences between examples and benchmarks are minimal: they are all Jupyter notebooks (paired to `py:light` files via jupytext for version control) and each demonstrates one functionality. Examples show *what* probdiffeq offers, while benchmarks show *how well* it performs, often compared to other solver libraries. Each tutorial or benchmark should run in under a minute. New contributions are welcome!

## Steps

1. **Create the script:**  
   Create a new Jupyter notebook in the appropriate subdirectory of `docs/`. Example paths include:
   - `docs/examples_benchmarks/benchmark-name.ipynb`
   - `docs/examples_advanced/example-name.ipynb`  
   Choose a meaningful name (e.g., `work-precision-hires`, `demonstrate-calibration`). The notebook should run the full example/benchmark and produce its plots. Ensure execution time stays well below one minute to keep CI manageable.

   If your example requires external dependencies (e.g., sampling or optimization libraries), place it in `examples_advanced`. If it is a benchmark, place it in `examples_benchmarks`. Otherwise, place it in
   `examples_basic`.

2. **Sync to py:light:**  
   Install documentation dependencies and pre-commit hooks if you haven't already:
   ```
   pip install .[doc,format-and-lint]
   pre-commit install
   ```
   Link the notebook to a py:light script using jupytext (preferred for version control and formatting):
   ```
   jupytext --set-formats ipynb,py:light <new-notebook.ipynb>
   ```
   Or link all notebooks at once:
   ```
   jupytext --set-formats ipynb,py:light docs/examples_*/*.ipynb
   ```
   Notebooks placed correctly according to the directory structure will be included by the previous command.
   
3. **Docs:**  
   Add the new `.ipynb` file to the documentation navigation in `mkdocs.yml` under `nav:`.  
   Ensure the corresponding script is excluded under `mkdocs.yml -> exclude:`; if needed, add it there.

4. **Makefile:**  
   Add the new example or benchmark to the appropriate Makefile target (e.g., `examples-and-benchmarks`).

5. **Pyproject.toml:**  
   If your example requires external dependencies, list them under the `doc` optional dependencies in `pyproject.toml`.

6. **Pull request:**  
   Commit the new notebook (the pre-commit hook will handle formatting and linting). Open a pull request and you're done.
