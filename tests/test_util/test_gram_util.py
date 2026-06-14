"""Test the matrix-exponential and finite-horizon Gramian utilities."""

from probdiffeq.backend import func, linalg, np, random, testing
from probdiffeq.util import gram_util, test_util


def case_method_pade_legendre_13():
    """Use the 13th-order Pade-Legendre method."""
    return gram_util.pade_and_legendre_13()


def case_method_pade_legendre_9():
    """Use the 9th-order Pade-Legendre method."""
    return gram_util.pade_and_legendre_9()


def case_method_pade_legendre_7():
    """Use the 7th-order Pade-Legendre method."""
    return gram_util.pade_and_legendre_7()


def case_method_pade_legendre_5():
    """Use the 5th-order Pade-Legendre method."""
    return gram_util.pade_and_legendre_5()


def case_method_pade_legendre_3():
    """Use the 3rd-order Pade-Legendre method."""
    return gram_util.pade_and_legendre_3()


@testing.parametrize("seed", [1])
@testing.parametrize("nrows", [8], ids=["size8"])
@testing.parametrize("ncols", [2, 8], ids=["tall", "square"])
@testing.parametrize("use_triu", [True, False], ids=["triu", "dense"])
@testing.parametrize_with_cases("pade_legendre", ".", prefix="case_method_")
def test_exp_gram_cholesky_matches_matrix_fraction_baseline(
    seed, nrows, ncols, pade_legendre, use_triu
):
    """Assert that exp_gram_cholesky matches the matrix-fraction baseline for both tall and square B."""
    # Define the problem
    key = random.prng_key(seed=seed)
    A = 0.01 * random.normal(key, shape=(nrows, nrows))
    B = random.normal(key, shape=(nrows, ncols))

    if use_triu:
        A = np.triu(A)
        algorithm = gram_util.exp_gram_cholesky(
            pade_legendre=pade_legendre, solve=linalg.solve_triu
        )
    else:
        algorithm = gram_util.exp_gram_cholesky(
            pade_legendre=pade_legendre, solve=linalg.solve_lu
        )

    # Run our algorithm
    eA2, L2 = func.jit(algorithm)(A, B)

    # Evaluate a baseline
    algorithm = test_util.exp_gram_matrix_fraction()
    eA1, G1 = func.jit(algorithm)(A, B)

    # Choose a tolerance based on the dtype.
    tol = np.sqrt(np.finfo_eps(G1.dtype))
    assert testing.allclose(eA1, eA2, atol=tol, rtol=tol)
    assert testing.allclose(G1, L2 @ L2.T, atol=tol, rtol=tol)
    assert testing.allclose(L2, np.tril(L2), atol=tol, rtol=tol)
