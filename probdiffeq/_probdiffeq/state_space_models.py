"""State space models."""

from probdiffeq import ssm_impl

__all__ = ["state_space_model"]


def state_space_model(ssm_fact="dense"):
    """Construct an implementation of a factorised state-space model."""
    if ssm_fact == "dense":
        return ssm_impl.FactSsmImpl.from_dense()

    if ssm_fact == "blockdiag":
        return ssm_impl.FactSsmImpl.from_blockdiag()

    if ssm_fact == "isotropic":
        return ssm_impl.FactSsmImpl.from_isotropic()
    raise ValueError(ssm_fact)
