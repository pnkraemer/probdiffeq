from probdiffeq.backend import func, linalg, np


def blockdiag_cholesky_from_ensembles(ensembles_smd, bias: bool):
    S, _n, _d = ensembles_smd.shape

    # Center the ensembles
    ensembles_smd -= ensembles_smd.mean(axis=0, keepdims=True)

    def ensemble_to_sample_cholesky(s):
        """Compute a sample Cholesky factor from ensembles."""
        num, _n = s.shape
        s = s / np.sqrt(num) if bias else s / np.sqrt(num - 1)
        return linalg.qr_r(s).T

    # The QR decomposition is why we assume S >= n,
    # so let's check it briefly:
    _S, m, _d = ensembles_smd.shape
    if m > S:
        msg = "The function requires at least as many ensembles as Taylor coefficients."
        msg += f" Received: S={S} < m={m}, which violates this assumption."
        raise ValueError(msg)

    # Assume ensembles are shape (S, n, d), so we batch along d
    # but since we also want the output to be (d, n, n), the out_axes is 0.
    transform = func.vmap(ensemble_to_sample_cholesky, in_axes=-1, out_axes=0)
    return transform(ensembles_smd)
