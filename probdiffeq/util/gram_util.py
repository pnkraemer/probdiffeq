"""Utilities for matrix exponentials and (finite-horizon) Gramians."""

from probdiffeq.backend import flow, linalg, np, structs
from probdiffeq.backend.typing import Array, Callable


def exp_gram_matrix_fraction(expm=linalg.expm) -> Callable:
    """Compute matrix exponential and finite-horizon Gramian via matrix fractions."""

    def compute(A, B):
        n = A.shape[0]
        M = np.block([[A, B @ B.T], [np.zeros_like(A), -A.T]])
        E = expm(M)
        eA = E[:n, :n]
        Sigma = E[:n, n:] @ eA.T
        return eA, Sigma

    return compute


@structs.dataclass
class PadeLegendre:
    """Pade-Legendre combinations for matrix exponentials and Gramians."""

    q: int
    eta_fp64: Array
    eta_fp32: Array
    init: Callable


def exp_gram_cholesky(*, pade_legendre: PadeLegendre, solve: Callable) -> Callable:
    r"""Compute matrix exponential and Cholesky factor of a finite-horizon Gramian.

    Concretely, for matrices A and B, compute the matrix exponential $e^A$ and
    the Cholesky factor of $G(A, B) = \int_0^1 e^{s A} B B^s e^{s A^s} ds.$

    The algorithm combines doubling, Pade approximations, and Legendre polynomials
    as proposed by Stillfjord and Tronarp: https://arxiv.org/abs/2310.13462.
    """

    def compute(A: Array, B: Array):
        # Initialise the approximation
        eA, U, s = _exp_gram_cholesky_init(
            A, B, pade_legendre=pade_legendre, solve=solve
        )

        # Run doubling loop (in case num > 0, otherwise the while-loop does nothing)
        _s, (eA, U) = flow.while_loop(
            lambda c: c[0] < s, _exp_gram_cholesky_double, init=(0, (eA, U))
        )

        # Turn U into valid triangular factor.
        # It is already triangular but let's enforce nonnegative
        # diagonals to mimic actualy Cholesky factors.
        scale = np.sign(linalg.diagonal(U))
        scale = np.where(scale == 0.0, 1.0, scale)
        U = U * scale[None, :]

        # Return exponential and U
        return eA, U

    return compute


def _exp_gram_cholesky_double(carry):
    i, (eA, U) = carry
    stack = np.concatenate((U, eA @ U), axis=-1)
    U = linalg.qr_r(stack.T).T  # triangularize
    eA = eA @ eA
    return i + 1, (eA, U)


def _exp_gram_cholesky_init(A, B, pade_legendre: PadeLegendre, solve: Callable):
    # Estimate the required number of doublings
    n, _n = A.shape
    A_L1 = linalg.matrix_norm(A, order=1)
    eta = pade_legendre.eta_fp64 if A.dtype == "float64" else pade_legendre.eta_fp32
    s1 = np.log2(A_L1 / eta)
    s2 = np.log2((n - 1) / pade_legendre.q)
    num = np.maximum(s1, s2)
    num = np.ceil(num)
    num = np.maximum(0.0, num)

    # Scale A and B and initialise the approximation
    A = A / (2**num)
    B = B / np.sqrt(2**num)
    eA, S = pade_legendre.init(A, B, solve=solve)
    return eA, S, num


def pade_and_legendre_3() -> PadeLegendre:
    """Construct a Pade/Legendre approximation of order 3."""
    pade_coeffs = (120.0, 60.0, 12.0, 1.0)
    legendre_coeffs = [[120, 0, 2, 0], [0, 60, 0, 0], [0, 0, 10, 0], [0, 0, 0, 1]]
    legendre_norms = [1, 3, 5, 7]

    def init(A, B, *, solve=linalg.solve_lu):
        b = np.asarray(pade_coeffs, dtype=A.dtype)
        M, N = A.shape
        ident = np.eye(M, N, dtype=A.dtype)
        A2 = linalg.vector_dot(A, A)
        U = linalg.vector_dot(A, (b[3] * A2 + b[1] * ident))
        V = b[2] * A2 + b[0] * ident
        eA = solve(V - U, V + U)

        C = np.asarray(legendre_coeffs, dtype=A.dtype)
        P = A2
        L_even = [P @ B * C[0, 2] + B * C[0, 0], P @ B * C[2, 2]]
        L_odd = [A @ B * C[1, 1], A @ P @ B * C[3, 3]]

        Ls = []
        for l_i, l_o in zip(L_even, L_odd):
            Ls.append(l_i)
            Ls.append(l_o)

        sqr_norms = np.asarray(legendre_norms, dtype=A.dtype)
        Ls = [ell / np.sqrt(i) for i, ell in zip(sqr_norms, Ls)]
        rhs = np.concatenate(Ls, axis=-1)

        L = solve(V - U, rhs)
        L = linalg.qr_r(L.T).T

        # Constant output shape
        cholesky = np.zeros_like(A).at[: L.shape[0], : L.shape[1]].set(L)

        return eA, cholesky

    return PadeLegendre(init=init, q=3, eta_fp64=0.0006794818550677766, eta_fp32=0.048)


def pade_and_legendre_5():
    """Construct a Pade/Legendre approximation of order 5."""
    pade_coeffs = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    legendre_coeffs = [
        [30240, 0, 840, 0, 2, 0],
        [0, 15120, 0, 168, 0, 0],
        [0, 0, 2520, 0, 10, 0],
        [0, 0, 0, 252, 0, 0],
        [0, 0, 0, 0, 18, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    legendre_norms = [1, 3, 5, 7, 9, 11]

    def init(A, B, *, solve=linalg.solve_lu):
        M, N = A.shape
        ident = np.eye(M, N, dtype=A.dtype)
        A2 = linalg.vector_dot(A, A)
        A4 = linalg.vector_dot(A2, A2)

        b = np.asarray(pade_coeffs, dtype=A.dtype)
        U = linalg.vector_dot(A, b[5] * A4 + b[3] * A2 + b[1] * ident)
        V = b[4] * A4 + b[2] * A2 + b[0] * ident
        eA = solve(V - U, V + U)

        C = np.asarray(legendre_coeffs, dtype=A.dtype)
        sqr_norms = np.asarray(legendre_norms, dtype=A.dtype)
        C = 1 / np.sqrt(sqr_norms[:, None]) * C

        P = A2
        zeros = np.zeros_like(B)
        L_even = [P @ B * C[0, 2] + B * C[0, 0], P @ B * C[2, 2], zeros]
        L_odd = [P @ B * C[1, 3] + B * C[1, 1], P @ B * C[3, 3], zeros]

        for k in [2]:  # todo: triple-check
            L_even = [ell + P @ B * C[2 * i, 2 * k] for i, ell in enumerate(L_even)]
            L_odd = [
                ell + P @ B * C[2 * i + 1, 2 * k + 1] for i, ell in enumerate(L_odd)
            ]

        L_odd = [A @ ell for ell in L_odd]

        Ls = []
        for l_i, l_o in zip(L_even, L_odd):
            Ls.append(l_i)
            Ls.append(l_o)

        rhs = np.concatenate(Ls, axis=-1)

        L = solve(V - U, rhs)
        L = linalg.qr_r(L.T).T

        # Constant output shape
        cholesky = np.zeros_like(A).at[: L.shape[0], : L.shape[1]].set(L)

        return eA, cholesky

    return PadeLegendre(
        init=init, q=5, eta_fp64=0.021803366063031033, eta_fp32=0.61190283
    )


def pade_and_legendre_7():
    """Construct a Pade/Legendre approximation of order 7."""
    pade_coeffs = (
        17297280.0,
        8648640.0,
        1995840.0,
        277200.0,
        25200.0,
        1512.0,
        56.0,
        1.0,
    )
    legendre_coeffs = [
        [17297280, 0, 554400, 0, 3024, 0, 2, 0],
        [0, 8648640, 0, 133056, 0, 324, 0, 0],
        [0, 0, 1441440, 0, 11880, 0, 10, 0],
        [0, 0, 0, 144144, 0, 616, 0, 0],
        [0, 0, 0, 0, 10296, 0, 18, 0],
        [0, 0, 0, 0, 0, 572, 0, 0],
        [0, 0, 0, 0, 0, 0, 26, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ]
    legendre_norms = [1, 3, 5, 7, 9, 11, 13, 15]

    def init(A, B, *, solve=linalg.solve_lu):
        b = np.asarray(pade_coeffs, dtype=A.dtype)
        M, N = A.shape
        ident = np.eye(M, N, dtype=A.dtype)
        A2 = linalg.vector_dot(A, A)
        A4 = linalg.vector_dot(A2, A2)
        A6 = linalg.vector_dot(A4, A2)
        U = linalg.vector_dot(A, b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
        V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        eA = solve(V - U, V + U)

        C = np.asarray(legendre_coeffs, dtype=A.dtype)
        P = A2
        zeros = np.zeros_like(B)
        L_even = [P @ B * C[0, 2] + B * C[0, 0], P @ B * C[2, 2], zeros, zeros]
        L_odd = [P @ B * C[1, 3] + B * C[1, 1], P @ B * C[3, 3], zeros, zeros]

        for k in [2, 3]:
            P = A2 @ P
            L_even = [ell + P @ B * C[2 * i, 2 * k] for i, ell in enumerate(L_even)]
            L_odd = [
                ell + P @ B * C[2 * i + 1, 2 * k + 1] for i, ell in enumerate(L_odd)
            ]

        L_odd = [A @ ell for ell in L_odd]

        Ls = []
        for l_i, l_o in zip(L_even, L_odd):
            Ls.append(l_i)
            Ls.append(l_o)

        sqr_norms = np.asarray(legendre_norms, dtype=A.dtype)
        Ls = [ell / np.sqrt(i) for i, ell in zip(sqr_norms, Ls)]
        rhs = np.concatenate(Ls, axis=-1)

        L = solve(V - U, rhs)
        L = linalg.qr_r(L.T).T

        # Constant output shape
        cholesky = np.zeros_like(A).at[: L.shape[0], : L.shape[1]].set(L)

        return eA, cholesky

    return PadeLegendre(
        init=init, q=7, eta_fp64=0.13306928209763041, eta_fp32=1.5580196
    )


def pade_and_legendre_9():
    """Construct a Pade/Legendre approximation of order 9."""
    pade_coeffs = (
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
        1.0,
    )
    legendre_coeffs = [
        [17643225600, 0, 605404800, 0, 4324320, 0, 7920, 0, 2, 0],
        [0, 8821612800, 0, 155675520, 0, 617760, 0, 528, 0, 0],
        [0, 0, 1470268800, 0, 15444000, 0, 34320, 0, 10, 0],
        [0, 0, 0, 147026880, 0, 960960, 0, 1092, 0, 0],
        [0, 0, 0, 0, 10501920, 0, 42120, 0, 18, 0],
        [0, 0, 0, 0, 0, 583440, 0, 1320, 0, 0],
        [0, 0, 0, 0, 0, 0, 26520, 0, 26, 0],
        [0, 0, 0, 0, 0, 0, 0, 1020, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 34, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    legendre_norms = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    def init(A, B, *, solve=linalg.solve_lu):
        b = np.asarray(pade_coeffs, dtype=A.dtype)
        M, N = A.shape
        ident = np.eye(M, N, dtype=A.dtype)
        A2 = linalg.vector_dot(A, A)
        A4 = linalg.vector_dot(A2, A2)
        A6 = linalg.vector_dot(A4, A2)
        A8 = linalg.vector_dot(A6, A2)
        U = linalg.vector_dot(
            A, b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident
        )
        V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
        eA = solve(V - U, V + U)

        C = np.asarray(legendre_coeffs, dtype=A.dtype)
        P = A2
        zeros = np.zeros_like(B)
        L_even = [P @ B * C[0, 2] + B * C[0, 0], P @ B * C[2, 2], zeros, zeros, zeros]
        L_odd = [P @ B * C[1, 3] + B * C[1, 1], P @ B * C[3, 3], zeros, zeros, zeros]

        for k in [2, 3, 4]:
            P = A2 @ P
            L_even = [ell + P @ B * C[2 * i, 2 * k] for i, ell in enumerate(L_even)]
            L_odd = [
                ell + P @ B * C[2 * i + 1, 2 * k + 1] for i, ell in enumerate(L_odd)
            ]

        L_odd = [A @ ell for ell in L_odd]

        Ls = []
        for l_i, l_o in zip(L_even, L_odd):
            Ls.append(l_i)
            Ls.append(l_o)

        sqr_norms = np.asarray(legendre_norms, dtype=A.dtype)
        Ls = [ell / np.sqrt(i) for i, ell in zip(sqr_norms, Ls)]
        rhs = np.concatenate(Ls, axis=-1)

        L = solve(V - U, rhs)
        L = linalg.qr_r(L.T).T

        # Constant output shape
        cholesky = np.zeros_like(A).at[: L.shape[0], : L.shape[1]].set(L)

        return eA, cholesky

    return PadeLegendre(init=init, q=9, eta_fp64=0.41098379928173995, eta_fp32=2.801222)


def pade_and_legendre_13():
    """Construct a Pade/Legendre approximation of order 13."""
    # fmt: off
    pade_coeffs = (
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
        129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0,
        1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0
    )
    legendre_coeffs= [
        [64764752532480000, 0, 2374707592857600, 0, 21118941043200, 0, 67044257280, 0, 81681600, 0, 32760, 0, 2, 0],
        [0, 32382376266240000, 0, 647647525324800, 0, 3620389893120, 0, 7449361920, 0, 5569200, 0, 1080, 0, 0],
        [0, 0, 5397062711040000, 0, 69390806284800, 0, 260727667200, 0, 352716000, 0, 153000, 0, 10, 0],
        [0, 0, 0, 539706271104000, 0, 4797389076480, 0, 12443820480, 0, 10852800, 0, 2380, 0, 0],
        [0, 0, 0, 0, 38550447936000, 0, 245321032320, 0, 439538400, 0, 232560, 0, 18, 0],
        [0, 0, 0, 0, 0, 2141691552000, 0, 9884730240, 0, 11938080, 0, 3344, 0, 0],
        [0, 0, 0, 0, 0, 0, 97349616000, 0, 324498720, 0, 248976, 0, 26, 0],
        [0, 0, 0, 0, 0, 0, 0, 3744216000, 0, 8809920, 0, 3780, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 124807200, 0, 197064, 0, 34, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3670800, 0, 3496, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96600, 0, 42, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2300, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ]
    # fmt: on

    legendre_norms = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

    def init(A, B, *, solve=linalg.solve_lu):
        b = np.asarray(pade_coeffs, dtype=A.dtype)

        n, m = A.shape
        ident = np.eye(n, m, dtype=A.dtype)
        zeros = np.zeros_like(B)
        A2 = linalg.vector_dot(A, A)
        A4 = linalg.vector_dot(A2, A2)
        A6 = linalg.vector_dot(A4, A2)

        # Initialise the matrix exponential estimate with
        # a PadeLegendre approximation
        U = linalg.vector_dot(
            A,
            linalg.vector_dot(A6, b[13] * A6 + b[11] * A4 + b[9] * A2)
            + b[7] * A6
            + b[5] * A4
            + b[3] * A2
            + b[1] * ident,
        )
        V = (
            linalg.vector_dot(A6, b[12] * A6 + b[10] * A4 + b[8] * A2)
            + b[6] * A6
            + b[4] * A4
            + b[2] * A2
            + b[0] * ident
        )
        eA = solve(V - U, V + U)

        # Initialise the Gramian
        C = np.asarray(legendre_coeffs, dtype=A.dtype)
        L_even = [
            A2 @ B * C[0, 2] + B * C[0, 0],
            A2 @ B * C[2, 2],
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        ]
        L_odd = [
            A2 @ B * C[1, 3] + B * C[1, 1],
            A2 @ B * C[3, 3],
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        ]

        P = A2
        for k in [2, 3, 4, 5, 6]:
            P = A2 @ P
            L_even = [ell + P @ B * C[2 * i, 2 * k] for i, ell in enumerate(L_even)]
            L_odd = [
                ell + P @ B * C[2 * i + 1, 2 * k + 1] for i, ell in enumerate(L_odd)
            ]

        L_odd = [A @ ell for ell in L_odd]

        Ls = []
        for l_i, l_o in zip(L_even, L_odd):
            Ls.append(l_i)
            Ls.append(l_o)

        sqr_norms = np.asarray(legendre_norms, dtype=A.dtype)
        Ls = [ell / np.sqrt(i) for i, ell in zip(sqr_norms, Ls)]
        rhs = np.concatenate(Ls, axis=-1)

        L = solve(V - U, rhs)
        L = linalg.qr_r(L.T).T

        # Constant output shape
        cholesky = np.zeros_like(A).at[: L.shape[0], : L.shape[1]].set(L)

        return eA, cholesky

    return PadeLegendre(init=init, eta_fp64=1.579470165477942, q=13, eta_fp32=5.7639575)
