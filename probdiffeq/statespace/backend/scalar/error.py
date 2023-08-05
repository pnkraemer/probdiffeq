from probdiffeq.statespace.backend import error


class ErrorBackEnd(error.ErrorBackEnd):
    def estimate(self, observed, /):
        mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros(()))
        output_scale = mahalanobis_norm
        error_estimate_unscaled = observed.marginal_stds()
        return output_scale * error_estimate_unscaled
