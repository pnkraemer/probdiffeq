"""Plot the results from the HIRES benchmark."""
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq.util.doc_util import notebook

results = jnp.load(os.path.dirname(__file__) + "/results.npy", allow_pickle=True)[()]


plt.rcParams.update(notebook.plot_config())
for label, wp in results.items():
    plt.loglog(wp["work"], wp["precision"], label=label)

plt.grid("both")
plt.legend()
plt.show()
