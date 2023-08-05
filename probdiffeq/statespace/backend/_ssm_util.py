"""SSM utilities."""

import abc


class SSMUtilBackEnd(abc.ABC):
    @abc.abstractmethod
    def stack_tcoeffs(self, tcoeffs, /, num_derivatives):
        raise NotImplementedError
