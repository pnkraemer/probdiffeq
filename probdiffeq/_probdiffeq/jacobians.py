"""Jacobian matrix handling."""

from probdiffeq.backend import func, linalg, random

__all__ = [
    "Jacobian",
    "jacobian_materialize",
    "jacobian_monte_carlo_fwd",
    "jacobian_monte_carlo_rev",
]


class Jacobian:
    """An interface for working with Jacobian matrices."""

    def init_jacobian_handler(self):
        """Initialize the handler state.

        For example, if the handler uses stochastic sampling,
        this initialisation would create a random key.
        """
        raise NotImplementedError

    def materialize_dense(
        self, fun, x, state, /, *, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        """Materialize a dense Jacobian.

        This is typically used for first-order linearization in dense
        state-space models.
        """
        raise NotImplementedError

    def calculate_trace_along_d(
        self, fun, x, state, /, *, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        """Calculate the trace of a Jacobian.

        This is typically used for first-order linearization in isotropic
        state-space models.
        """
        raise NotImplementedError

    def calculate_diagonal_along_d(
        self, fun, x, state, /, *, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        """Calculate the diagonal of a Jacobian.

        This is typically used for first-order linearization in block-diagonal
        state-space models.
        """
        raise NotImplementedError


class jacobian_materialize(Jacobian):
    """Construct a handler that always materialized Jacobian matrices.

    Use this Jacobian if the dimension of the problem is relatively small.
    """

    def __init__(self, *, jacfun=func.jacfwd) -> None:
        self.jacfun = jacfun

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(jacfun={self.jacfun})"

    def init_jacobian_handler(self):
        return ()

    def materialize_dense(
        self, fun, x, state, /, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        if x.shape != (num_tcoeffs * d,):
            msg = "This function expects a flat array for 'x'. "
            msg += f"Expected: x.shape = {(num_tcoeffs * d,)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)

        fx = fun(x, **fun_kwargs)
        dfx = func.jacfwd(lambda s: fun(s, **fun_kwargs))(x)
        return fx, dfx, state

    def calculate_trace_along_d(
        self, fun, x, state, /, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        if x.shape != (num_tcoeffs, d):
            msg = "This function expects an nxd array for 'x'. "
            msg += f"Expected: x.shape = {(num_tcoeffs, d)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)

        fx = fun(x, **fun_kwargs)

        dfx = func.jacfwd(lambda s: fun(s, **fun_kwargs))(x)
        dfx_trace = linalg.trace(dfx, axis1=0, axis2=-1)
        return fx, dfx_trace, state

    def calculate_diagonal_along_d(
        self, fun, x, state, /, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        if x.shape != (d, num_tcoeffs):
            msg = "This function expects a dxn array for 'x'. "
            msg += f"Expected: x.shape = {(d, num_tcoeffs)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)
        fx = fun(x)
        dfx = func.jacfwd(lambda s: fun(s, **fun_kwargs))(x)
        dfx_diagonal = linalg.diagonal(dfx, axis1=0, axis2=1)
        return fx, dfx_diagonal.T, state


class jacobian_monte_carlo_fwd(Jacobian):
    """Construct a handler that uses stochastic trace estimation for traces/diagonals.

    Use a Hutchinson handler if the dimension of the problem is large.

    This implementation uses **forward-mode** automatic differentiation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """

    def __init__(self, *, seed=1, num_probes=10) -> None:
        self.seed = seed
        self.num_probes = num_probes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(seed={self.seed}, num_probes={self.num_probes})"
        )

    def init_jacobian_handler(self):
        return random.prng_key(seed=self.seed)

    def materialize_dense(
        self, fun, x, state, /, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        if x.shape != (num_tcoeffs * d,):
            msg = "This function expects a flat array for 'x'. "
            msg += f"Expected: x.shape = {(num_tcoeffs * d,)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)

        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x, **fun_kwargs)
        dfx = func.jacfwd(lambda s: fun(s, **fun_kwargs))(x)
        return fx, dfx, state

    def calculate_trace_along_d(
        self, fun, x, key, /, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        if x.shape != (num_tcoeffs, d):
            msg = "This function expects an nxd array for 'x'. "
            msg += f"Expected: x.shape = {(num_tcoeffs, d)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)

        fx, Jvp = func.linearize(lambda s: fun(s, **fun_kwargs), x)

        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)
        J_trace = func.vmap(lambda s: linalg.vector_dot(s, Jvp(s)))(v)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal_along_d(self, fun, x, key, /, num_tcoeffs, d, **fun_kwargs):
        if x.shape != (d, num_tcoeffs):
            msg = "This function expects a dxn array for 'x'. "
            msg += f"Expected: x.shape = {(d, num_tcoeffs)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)

        fx, Jvp = func.linearize(lambda s: fun(s, **fun_kwargs), x)

        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)

        # shape: (s, d, n)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        # shape: (s, d)
        Jv = func.vmap(Jvp)(v)

        # shape: (s, d, n)
        vJv = v * Jv[..., None]

        # shape: (d, n)
        J_diagonal = vJv.mean(axis=0)
        return fx, J_diagonal, key


class jacobian_monte_carlo_rev(Jacobian):
    """Construct a handler that uses stochastic trace estimation for traces/diagonals.

    Use a Hutchinson handler if the dimension of the problem is large.

    This implementation uses **reverse-mode** automatic differentiation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """

    def __init__(self, *, seed=1, num_probes=10) -> None:
        self.seed = seed
        self.num_probes = num_probes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(seed={self.seed}, num_probes={self.num_probes})"
        )

    def init_jacobian_handler(self):
        return random.prng_key(seed=self.seed)

    def materialize_dense(
        self, fun, x, state, /, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        if x.shape != (num_tcoeffs * d,):
            msg = "This function expects a flat array for 'x'. "
            msg += f"Expected: x.shape = {(num_tcoeffs * d,)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)

        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x, **fun_kwargs)
        dfx = func.jacrev(lambda s: fun(s, **fun_kwargs))(x)
        return fx, dfx, state

    def calculate_trace_along_d(
        self, fun, x, key, /, num_tcoeffs: int, d: int, **fun_kwargs
    ):
        if x.shape != (num_tcoeffs, d):
            msg = "This function expects an nxd array for 'x'. "
            msg += f"Expected: x.shape = {(num_tcoeffs, d)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)

        fx, vjp = func.vjp(lambda s: fun(s, **fun_kwargs), x)

        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *(x[0].shape))

        # v.shape: (s, d)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        # vjpx.shape: (s, n, d)
        (vjpx,) = func.vmap(vjp)(v)

        # Einsum along the "d" axis (trace = sum of diagonals)
        J_trace = linalg.einsum("snd,sd->sn", vjpx, v)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal_along_d(
        self, fun, x, key, /, num_tcoeffs: int, d: int, **fun_kwargs
    ):

        if x.shape != (d, num_tcoeffs):
            msg = "This function expects a dxn array for 'x'. "
            msg += f"Expected: x.shape = {(d, num_tcoeffs)}. "
            msg += f"Received: x.shape = {x.shape}. "
            raise ValueError(msg)

        fx, vjp = func.vjp(lambda s: fun(s, **fun_kwargs), x)

        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *(x[:, 0].shape))

        # shape: (s, d)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        # shape: (s, d, n)
        (vjpx,) = func.vmap(vjp)(v)

        # shape: (s, d, n)
        vJv = vjpx * v[..., None]

        # shape: (d, n)
        J_diagonal = vJv.mean(axis=0)
        return fx, J_diagonal, key
