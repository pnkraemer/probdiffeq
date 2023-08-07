from probdiffeq.backend import _ssm_util


class SSMUtilBackend(_ssm_util.SSMUtilBackend):
    def ibm_transitions(self, num_derivatives, output_scale):
        raise NotImplementedError

    def identity_conditional(self, ndim):
        raise NotImplementedError

    def normal_from_tcoeffs(self, tcoeffs, /, num_derivatives):
        raise NotImplementedError

    def preconditioner_apply(self, rv, p, /):
        raise NotImplementedError

    def preconditioner_apply_cond(self, cond, p, p_inv, /):
        raise NotImplementedError

    def standard_normal(self, ndim, output_scale):
        raise NotImplementedError

    def update_mean(self, mean, x, /, num):
        raise NotImplementedError
