"""SSM utilities."""

import abc


class SSMUtilBackEnd(abc.ABC):
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
    def ibm_transitions(self, num_derivatives, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def update_mean(self, mean, x, /, num):
        raise NotImplementedError

    # todo: move those to random.py and cond.py instead?

    @abc.abstractmethod
    def identity_conditional(self, ndim):
        raise NotImplementedError

    @abc.abstractmethod
    def standard_normal(self, ndim, output_scale):
        raise NotImplementedError
