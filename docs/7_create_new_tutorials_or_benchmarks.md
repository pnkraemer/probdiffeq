# Create new tutorials or benchmarks

Probdiffeq hosts numerous tutorials and benchmarks that demonstrate the library. The differences between examples and benchmarks are minimal: they are all Python scripts (which become `Jupyter notebook` files in the final docs) and each demonstrates one functionality. 

- Tutorials show *what* probdiffeq offers

- Benchmarks show *how well* it performs, often compared to other solver libraries. 

Each tutorial or benchmark should run in under a minute, most run in a few seconds. New contributions are welcome!

### Steps

1. **Create the script:**  
  Create a new  notebook in the appropriate subdirectory of `tutorials/` or `benchmarks/`.
  Choose a meaningful name (e.g., `benchmarks/work-precision-hires.py`, `tutorials/demonstrate-calibration.py`). 
  The tutorials show up in the documentation according to the alphabetic order in the `tutorials/` and `benchmarks` directories.
    
2. **Fill the script:** 
  Write the benchmark/tutorial code. Ensure the execution time stays well below one minute to keep CI manageable.
 
3. **Write documentation:**
  The module docstring will become the title and description of the notebook, so choose a good one. 
    
4. **Pull request:**  
