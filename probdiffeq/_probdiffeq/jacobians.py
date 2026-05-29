from probdiffeq.backend import func, linalg, random

__all__ = [
    "JacobianHandler",
    "jacobian_hutchinson_fwd",
    "jacobian_hutchinson_rev",
    "jacobian_materialize",
]


class JacobianHandler:
    """An interface for working with Jacobian matrices."""

    def init_jacobian_handler(self):
        """Initialize the handler state.

        For example, if the handler uses stochastic sampling,
        this initialisation would create a random key.
        """
        raise NotImplementedError

    def materialize_dense(self, fun, x, state, /):
        """Materialize a dense Jacobian.

        This is typically used for first-order linearization in dense
        state-space models.
        """
        raise NotImplementedError

    def calculate_trace(self, fun, x, state, /):
        """Calculate the trace of a Jacobian.

        This is typically used for first-order linearization in isotropic
        state-space models.
        """
        raise NotImplementedError

    def calculate_diagonal(self, fun, x, state, /):
        """Calculate the diagonal of a Jacobian.

        This is typically used for first-order linearization in block-diagonal
        state-space models.
        """
        raise NotImplementedError


class jacobian_materialize(JacobianHandler):
    """Construct a handler that always materialized Jacobian matrices.

    Use this Jacobian if the dimension of the problem is relatively small.
    """

    def __init__(self, *, jacfun=func.jacfwd) -> None:
        self.jacfun = jacfun

    def init_jacobian_handler(self):
        return ()

    def materialize_dense(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        return fx, dfx, ()

    def calculate_trace(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        dfx_trace = linalg.trace(dfx)
        return fx, dfx_trace, ()

    def calculate_diagonal(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        dfx_diagonal = linalg.diagonal(dfx)
        return fx, dfx_diagonal, ()


class jacobian_hutchinson_fwd(JacobianHandler):
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

    def init_jacobian_handler(self):
        return random.prng_key(seed=self.seed)

    def materialize_dense(self, fun, x, state, /):
        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        return fx, dfx, state

    def calculate_trace(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, Jvp = func.linearize(fun, x)
        J_trace = func.vmap(lambda s: linalg.vector_dot(s, Jvp(s)))(v)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, Jvp = func.linearize(fun, x)
        vJv = func.vmap(lambda s: s * Jvp(s))(v)
        J_diagonal = vJv.mean(axis=0)
        return fx, J_diagonal, key


class jacobian_hutchinson_rev(JacobianHandler):
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

    def init_jacobian_handler(self):
        return random.prng_key(seed=self.seed)

    def materialize_dense(self, fun, x, state, /):
        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x)
        dfx = func.jacrev(fun)(x)
        return fx, dfx, state

    def calculate_trace(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, vjp = func.vjp(fun, x)
        J_trace = func.vmap(lambda s: linalg.vector_dot(s, vjp(s)[0]))(v)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, vjp = func.vjp(fun, x)
        vJv = func.vmap(lambda s: s * vjp(s)[0])(v)
        J_diagonal = vJv.mean(axis=0)
        return fx, J_diagonal, key
