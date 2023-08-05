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
    def ibm_transitions(self, num_derivatives, output_scale):
        raise NotImplementedError
