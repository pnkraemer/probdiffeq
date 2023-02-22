"""Interface for estimation strategies."""

import abc

import jax


@jax.tree_util.register_pytree_node_class
class Strategy(abc.ABC):
    """Inference strategy interface."""

    def __init__(self, implementation):
        self.implementation = implementation

    def __repr__(self):
        args = f"implementation={self.implementation}"
        return f"{self.__class__.__name__}({args})"

    @abc.abstractmethod
    def init_posterior(self, *, taylor_coefficients):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol_terminal_value(self, *, posterior):
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(self, *, p0, p1, t, t0, t1, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def case_interpolate(self, *, p0, rv1, t, t0, t1, scale_sqrtm):  # noqa: D102
        raise NotImplementedError

    @abc.abstractmethod
    def marginals(self, *, posterior):  # todo: rename to marginalise?
        raise NotImplementedError

    @abc.abstractmethod
    def marginals_terminal_value(self, *, posterior):  # todo: rename to marginalise?
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(
        self, *, t, marginals, posterior_previous, t0, t1, scale_sqrtm
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, key, *, posterior, shape):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, *, posterior, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, linearisation_pt, cache, *, output_scale_sqrtm, posterior_previous
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, *, extrapolated, cache_obs):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (implementation,) = children
        return cls(implementation=implementation)

    def init_error_estimate(self):
        return self.implementation.extrapolation.init_error_estimate()

    def init_output_scale_sqrtm(self):
        return self.implementation.extrapolation.init_output_scale_sqrtm()

    def begin_correction(self, linearisation_pt, *, vector_field, t, p):
        return self.implementation.correction.begin_correction(
            linearisation_pt, vector_field=vector_field, t=t, p=p
        )
