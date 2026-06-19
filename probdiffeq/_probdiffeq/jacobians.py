"""Jacobian matrix handling."""

from probdiffeq.backend import func, linalg, np, random, structs, tree
from probdiffeq.backend.typing import Array

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

    def materialize_dense(self, fun, x, state, /, **fun_kwargs):
        """Materialize a dense Jacobian.

        This is typically used for first-order linearization in dense
        state-space models.
        """
        raise NotImplementedError

    def calculate_trace_along_d(self, fun, x, state, /, **fun_kwargs):
        """Calculate the trace of a Jacobian.

        This is typically used for first-order linearization in isotropic
        state-space models.
        """
        raise NotImplementedError

    def calculate_diagonal_along_d(self, fun, x, state, /, **fun_kwargs):
        """Calculate the diagonal of a Jacobian.

        This is typically used for first-order linearization in block-diagonal
        state-space models.
        """
        raise NotImplementedError

    def _verify_fun_and_x(self, fun, x):
        fx_like = func.eval_shape(fun, x)

        # Be strict with the input types and shapes (because previous versions
        # of this code were lose *and different*.
        msg = "'fun' must map an (n, d) array to an (m, d) array."
        msg += f" Received: f(x).shape = {tree.tree_map(np.shape, fx_like)}"
        msg += f" for x.shape == {tree.tree_map(np.shape, x)}. "

        # fx is the result of func.eval_shape, so if it is a
        # ShapeDtypeStruct, the function would have returned an array
        # (which is expected). If it returns something else (eg. list, tuple),
        # we raise an error.
        x_is_array = isinstance(x, Array)
        fx_is_array_like = isinstance(fx_like, structs.ShapeDtypeStruct)
        if not x_is_array or not fx_is_array_like:
            raise TypeError(msg)

        if x.ndim != 2 or fx_like.ndim != 2:
            raise ValueError(msg)

        n_in, d = x.shape
        n_out, d2 = fx_like.shape
        if d != d2:
            raise ValueError(msg)

        return n_in, n_out, d


class jacobian_materialize(Jacobian):
    """Construct a handler that always materialized Jacobian matrices.

    Use this Jacobian if the dimension of the problem is relatively small.
    """

    def __init__(self, *, jacfun=func.jacrev) -> None:
        self.jacfun = jacfun

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(jacfun={self.jacfun})"

    def init_jacobian_handler(self):
        return ()

    def materialize_dense(self, fun, x, state, /, **fun_kwargs):
        _ = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)

        fx = fun(x, **fun_kwargs)
        dfx = func.jacfwd(lambda s: fun(s, **fun_kwargs))(x)
        return fx, dfx, state

    def calculate_trace_along_d(self, fun, x, state, /, **fun_kwargs):
        _ = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)

        fx = fun(x, **fun_kwargs)
        dfx = func.jacfwd(lambda s: fun(s, **fun_kwargs))(x)
        dfx_trace = linalg.trace(dfx, axis1=1, axis2=3)
        return fx, dfx_trace, state

    def calculate_diagonal_along_d(self, fun, x, state, /, **fun_kwargs):
        _ = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)
        fx = fun(x, **fun_kwargs)
        dfx = func.jacfwd(lambda s: fun(s, **fun_kwargs))(x)
        dfx_diagonal = linalg.einsum("mdnd->dmn", dfx)
        return fx, dfx_diagonal, state


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

    def materialize_dense(self, fun, x, state, /, **fun_kwargs):
        _ = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)

        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x, **fun_kwargs)
        dfx = func.jacfwd(lambda s: fun(s, **fun_kwargs))(x)
        return fx, dfx, state

    def calculate_trace_along_d(self, fun, x, key, /, **fun_kwargs):
        n_in, _n_out, d = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)

        fx, Jvp = func.linearize(lambda s: fun(s, **fun_kwargs), x)

        key, subkey = random.split(key, num=2)

        # (s, n_in, d)
        v = random.rademacher(subkey, shape=(self.num_probes, n_in, d), dtype=x.dtype)

        # (s, n_out, d)
        Jv = func.vmap(Jvp)(v)

        # (s, n_out, n_in)
        vJv = linalg.einsum("smd,snd->snm", v, Jv)

        # (n_out, n_in)
        J_trace = np.mean(vJv, axis=0)
        return fx, J_trace, key

    def calculate_diagonal_along_d(self, fun, x, key, /, **fun_kwargs):
        n_in, _n_out, d = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)

        fx, Jvp = func.linearize(lambda s: fun(s, **fun_kwargs), x)

        key, subkey = random.split(key, num=2)

        # (s, n_in, d)
        v = random.rademacher(subkey, shape=(self.num_probes, n_in, d), dtype=x.dtype)

        # (s, n_out, d)
        Jv = func.vmap(Jvp)(v)

        # (s, n_out, n_in, d)
        vJv = v[:, None, :, :] * Jv[:, :, None, :]

        # (n_in, n_out, d)
        J_diagonal = np.mean(vJv, axis=0)

        # (d, n_in, n_out)
        J_diagonal = np.transpose(J_diagonal, axes=(2, 0, 1))
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

    def materialize_dense(self, fun, x, state, /, **fun_kwargs):
        _ = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)

        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x, **fun_kwargs)
        dfx = func.jacrev(lambda s: fun(s, **fun_kwargs))(x)
        return fx, dfx, state

    def calculate_trace_along_d(self, fun, x, key, /, **fun_kwargs):
        _n_in, n_out, d = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)

        fx, vjp = func.vjp(lambda s: fun(s, **fun_kwargs), x)

        key, subkey = random.split(key, num=2)

        # (s, n_out, d)
        v = random.rademacher(subkey, shape=(self.num_probes, n_out, d), dtype=x.dtype)

        # (s, n_in, d)
        (vjpx,) = func.vmap(vjp)(v)

        # (s, n_in, n_out)
        J_trace = linalg.einsum("snd,smd->smn", vjpx, v)

        # (n_in, n_out)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal_along_d(self, fun, x, key, /, **fun_kwargs):

        _n_in, n_out, d = self._verify_fun_and_x(lambda s: fun(s, **fun_kwargs), x)
        fx, vjp = func.vjp(lambda s: fun(s, **fun_kwargs), x)

        key, subkey = random.split(key, num=2)

        # (s, n_out, d)
        v = random.rademacher(subkey, shape=(self.num_probes, n_out, d), dtype=x.dtype)

        # (s, n_in, d)
        (vjpx,) = func.vmap(vjp)(v)

        # (s, n_out, n_in, d)
        vJv = vjpx[:, None, :, :] * v[:, :, None, :]

        # (n_out, n_in, d)
        J_diagonal = np.mean(vJv, axis=0)

        # (d, n_out, n_in)
        J_diagonal = np.transpose(J_diagonal, axes=(2, 0, 1))
        return fx, J_diagonal, key
