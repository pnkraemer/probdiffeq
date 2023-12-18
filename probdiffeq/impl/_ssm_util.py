"""SSM utilities."""

from probdiffeq.backend import abc


class SSMUtilBackend(abc.ABC):
    @abc.abstractmethod
    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply(self, rv, p, /):
        raise NotImplementedError

    @abc.abstractmethod
    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        raise NotImplementedError

    @abc.abstractmethod
    def ibm_transitions(self, num_derivatives, output_scale=None):
        raise NotImplementedError

    # TODO: rename to avoid confusion with conditionals?
    @abc.abstractmethod
    def update_mean(self, mean, x, /, num):
        raise NotImplementedError

    # TODO: move those to random.py and cond.py instead?

    @abc.abstractmethod
    def identity_conditional(self, num_derivatives_per_ode_dimension, /):
        raise NotImplementedError

    @abc.abstractmethod
    def standard_normal(self, num_derivatives_per_ode_dimension, /, output_scale):
        raise NotImplementedError
