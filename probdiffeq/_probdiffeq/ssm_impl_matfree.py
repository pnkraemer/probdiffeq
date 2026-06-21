"""Matrix-free extensions for blockdiagonal models."""

from probdiffeq._probdiffeq import problems, ssm_impl_api, taylor_points
from probdiffeq._probdiffeq import ssm_impl_blockdiag as blockdiag
from probdiffeq.backend import func, linalg, np, random, tree
from probdiffeq.backend.typing import Array, Callable, Sequence, TypeVar

__all__ = ["state_space_model_matfree"]


C = TypeVar("C", bound=Sequence)
"""A type-variable for Sequence types.

For example, this variable is used to type Taylor coefficients.
"""

# TODO: remove unnecessary function evaluations
# TODO: remove info_operator duplication between Ts1 and Linop
# TODO: include matfree TS1 in hires example
# TODO: Implement iterated linearisation
# TODO: try Hutchinson instead of ensembles
# TODO: Implement residual-based solvers
# TODO: Implement a benchmark for high-dimensional, stiff problems.


class Linop:
    def __init__(self, *, batch, n_in, n_out, d_in, d_out):
        self.batch = batch
        self.n_in = n_in
        self.n_out = n_out
        self.d_in = d_in
        self.d_out = d_out

    def __repr__(self):
        name = self.__class__.__name__
        args = f"batch={self.batch}, n_out={self.n_out}, d_out={self.d_out}, n_in={self.n_in}, d_in={self.d_in})"
        return f"{name}({args})"


class MatfreeLinopODEConstraint(Linop):
    def __init__(self, *, ode, tree_flatten, x, t):
        self.ode = ode
        self.tree_flatten = tree_flatten
        self.t = t
        self.x = x

        *batch, n, d = x.shape
        super().__init__(batch=batch, n_in=n, n_out=1, d_in=d, d_out=d)

    def init_jvp(self):
        _, jvp = func.linearize(self._information_operator, self.x)
        return jvp

    def matvec_dn(self, vec_dn, *, jvp):
        vec_nd = vec_dn.T
        return self.matvec_nd(vec_nd, jvp=jvp).T

    def vecmat_flat(self, w, *, jvp):
        v = np.ones((self.n_in * self.d_in,))
        vecmat = func.linear_transpose(lambda s: self.matvec_flat(s, jvp=jvp), v)
        (vm,) = vecmat(w)
        return vm

    def matvec_flat(self, vec_flat, *, jvp):
        vec = vec_flat.reshape((self.n_in, self.d_in))
        return self.matvec_nd(vec, jvp=jvp).reshape((-1,))

    def matvec_nd(self, vec, *, jvp):
        x1 = vec[np.asarray([self.ode.num_tcoeffs_in_args])]
        fx0 = jvp(vec)
        return x1 - fx0

    def _information_operator(self, s: Array) -> Array:
        # Move to latent space (arg is (n, d) but latent space is (d, n))
        s = s.T

        # Extract all tcoeffs
        jet_coords = self.tree_flatten.unflatten_array(s)

        # Extract relevant tcoeffs ("jet-coordinates")
        jet_coords = jet_coords[: self.ode.num_tcoeffs_in_args]

        # Evaluate the actual vector field
        fs0 = self.ode.vector_field(jet_coords=jet_coords, t=self.t)

        # Bring back into (m, d) form.
        return tree.ravel_pytree(fs0)[0][None, :]

    @classmethod
    def _register_as_pytree(cls) -> None:
        """Register this class (or a subclass) as a JAX pytree."""

        def flatten(linop):
            children = linop.t, linop.x
            aux = (linop.ode, linop.tree_flatten)
            return children, aux

        def unflatten(aux, children):
            (t, x) = children
            ode, tree_flatten = aux
            return cls(ode=ode, tree_flatten=tree_flatten, t=t, x=x)

        tree.register_pytree_node(cls, flatten, unflatten)


MatfreeLinopODEConstraint._register_as_pytree()


class MatfreeLinopLstSq(Linop):
    def __init__(self, *, cholesky_x, linop_ode_constraint: MatfreeLinopODEConstraint):
        # Swap 'in' and 'out' because LstSq is an inverse operator
        super().__init__(
            batch=linop_ode_constraint.batch,
            n_in=linop_ode_constraint.n_out,
            n_out=linop_ode_constraint.n_in,
            d_in=linop_ode_constraint.d_out,
            d_out=linop_ode_constraint.d_in,
        )
        self.cholesky_x = cholesky_x
        self.linop_ode_constraint = linop_ode_constraint

    def init_jvp(self):
        return self.linop_ode_constraint.init_jvp()

    def matvec_dn(self, vec_dn, *, params):
        vec_flat = vec_dn.T.reshape((self.n_in * self.d_in,))
        Avec_flat = self.matvec_flat(vec_flat, params=params)
        return Avec_flat.reshape((self.n_out, self.d_out)).T

    def matvec_nd(self, vec, *, params):
        vec_flat = vec.reshape((self.n_in * self.d_in,))
        Avec_flat = self.matvec_flat(vec_flat, params=params)
        return Avec_flat.reshape((self.n_out, self.d_out))

    def matvec_flat(self, vec, *, params):
        mean_x, jvp = params

        def vecmat(s):

            # vector-Jacobian product
            Ats = self.linop_ode_constraint.vecmat_flat(s, jvp=jvp)

            # Cholesky-vector products
            return MatfreeLinopLstSq.materialized_vecmat_dn(self.cholesky_x, Ats)

        # TODO: turn "D" into a damping factor?
        x0 = mean_x.T.reshape((-1,)) if mean_x is not None else None
        tol = 1e-16
        lstsq_sol = linalg.lstsq_lsmr(vecmat, vec, x0=x0, atol=tol, btol=tol, ctol=tol)
        return MatfreeLinopLstSq.materialized_matvec_dn(self.cholesky_x, lstsq_sol)

    @staticmethod
    def materialized_vecmat_dn(M, v):
        d, n, _m = M.shape
        v_dn = v.reshape((n, d)).T
        # transpose matvec, not matvec:
        Mv = linalg.einsum("...mn,...m->...n", M, v_dn)
        return Mv.T.reshape((-1,))

    @staticmethod
    def materialized_matvec_dn(M, v):
        d, n, _m = M.shape
        v_dn = v.reshape((n, d)).T
        Mv = linalg.einsum("...nm,...m->...n", M, v_dn)
        return Mv.T.reshape((-1,))


class MatfreeLatentCond(ssm_impl_api.AbstractLatentCond):
    def __init__(self, A, noise, key, num_ensembles, bias):
        super().__init__(A, noise=noise, to_latent=None, to_observed=None)
        self.key = key
        self.num_ensembles = num_ensembles
        self.bias = bias

    def apply_flat(self, x, /):
        jvp = self.A.init_jvp()
        y = self.A.matvec_nd(x.T, params=(None, jvp)).T
        m = y + self.noise.mean_flat
        c = self.noise.cholesky_flat
        tree_flatten = self.noise.tree_flatten
        return blockdiag.BlockDiagNormal(m, c, tree_flatten)

    def marginalise(self, rv, /):
        # Observed mean
        obs_mean = self.A.matvec_dn(rv.mean_flat) + self.noise.mean_flat

        # TODO: we currently ignore the noise
        keys = random.split(self.key, num=self.num_ensembles)
        ensembles = func.vmap(rv.sample_flat)(keys)  # d, n
        ensembles = np.transpose(ensembles, axes=(0, 2, 1))  # n, d
        ensembles_mv = func.vmap(self.A.matvec_nd)(ensembles)
        C = blockdiag_cholesky_from_ensembles(ensembles_mv, bias=self.bias)
        return blockdiag.BlockDiagNormal(obs_mean, C, self.noise.tree_flatten)

    def revert(self, rv, /, *, solve_triu: Callable):
        del solve_triu  # unused

        # Observed mean
        jvp = self.A.init_jvp()
        obs_mean = self.A.matvec_dn(rv.mean_flat, jvp=jvp) + self.noise.mean_flat

        # Gain & conditioning
        K = MatfreeLinopLstSq(cholesky_x=rv.cholesky_flat, linop_ode_constraint=self.A)

        # Posterior mean:
        z = self.A.matvec_dn(rv.mean_flat, jvp=jvp) + self.noise.mean_flat
        cond_mean = rv.mean_flat - K.matvec_dn(z, params=(None, jvp))

        # Posterior covariance via probing/ensembles
        keys = random.split(self.key, num=self.num_ensembles)
        ensembles = func.vmap(rv.sample_flat)(keys)  # d, n
        ensembles = np.transpose(ensembles, axes=(0, 2, 1))  # n, d

        def matvec_nd(p_nd):
            Ap_nd = self.A.matvec_nd(p_nd, jvp=jvp)
            KAp_nd = K.matvec_nd(Ap_nd, params=(cond_mean, jvp))
            return p_nd - KAp_nd, Ap_nd

        # Posteriors
        ensembles_p, ensembles_o = func.vmap(matvec_nd)(ensembles)
        cond_chol = blockdiag_cholesky_from_ensembles(ensembles_p, bias=self.bias)
        noise = blockdiag.BlockDiagNormal(cond_mean, cond_chol, rv.tree_flatten)

        # Backward linear operator
        _key, subkey = random.split(self.key, num=2)
        cond = MatfreeLatentCond(
            K, noise, key=subkey, num_ensembles=self.num_ensembles, bias=self.bias
        )

        # Marginals
        obs_chol = blockdiag_cholesky_from_ensembles(ensembles_o, bias=self.bias)
        observed = blockdiag.BlockDiagNormal(
            obs_mean, obs_chol, self.noise.tree_flatten
        )

        # Group and return
        return observed, cond

    def merge(self, other, /):
        raise NotImplementedError

    def preconditioner_apply(self, /):
        raise NotImplementedError

    @classmethod
    def _register_as_pytree(cls) -> None:
        """Register this class (or a subclass) as a JAX pytree."""

        def flatten(linop):
            children = (linop.A, linop.noise, linop.key)
            aux = linop.num_ensembles, linop.bias
            return children, aux

        def unflatten(aux, children):
            (A, noise, key) = children
            num_ensembles, bias = aux
            return cls(A, noise, key=key, num_ensembles=num_ensembles, bias=bias)

        tree.register_pytree_node(cls, flatten, unflatten)


MatfreeLatentCond._register_as_pytree()


class MatfreeOdeTs1(ssm_impl_api.AbstractOde):
    """Dense ODE linearization via TS1 (first-degree Taylor series: evaluate the residual and its Jacobian at the linearization point)."""

    # Sample count? See p. 6 in https://arxiv.org/abs/2606.08203

    def __init__(self, ode, key, num_ensembles, bias):
        super().__init__(ode=ode)
        self.key = key
        self.num_ensembles = num_ensembles
        self.bias = bias

    def init_linearization(self):
        # todo: we should move the trace estimation outside
        # the jacobian object. Then, this bit here becomes simpler?
        # For example, we could even move the information operator
        # to a method in the ODE, then call func.linearize() / vjp()
        # in the constraints here, and then call the trace/diagonal estimators
        # on this linearisation with a uniform API.
        jac = self.ode.jacobian.init_jacobian_handler()
        return self.key, jac

    def linearize(self, rv, state, *, damp: float, t: float):

        # Read n and d so that we can turn latent arrays into
        # (n, d) arrays, which Jacobians require
        m_tree = rv.mean
        n = len(m_tree)
        d = rv.mean_flat.size // n

        # Materialize the Jacobian
        m0 = rv.mean_flat.T
        fun = func.partial(self.information_operator, tree_flatten=rv.tree_flatten, t=t)
        fx, JVP = func.linearize(fun, m0)

        # Flatten fx and J correctly (from [m, d, n, d] to [md,nd])
        m, d = fx.shape
        fx = fx.reshape((m * d,))

        # Complete the expressions for bias and linop

        fx = JVP(rv.mean_flat.T) - fx
        # E1 = projection_e1(rv.mean_flat.T)

        # Flatten fx into the correct pytree structure
        f0 = blockdiag.BlockDiagNormal.from_dirac(
            [m_tree[self.ode.num_tcoeffs_in_args]], damp=damp
        )
        fx = f0.tree_flatten.unflatten_array(fx.T)

        # Collect all quantities and return
        noise = blockdiag.BlockDiagNormal.from_dirac(fx, damp=damp)

        # n_out, d_out, n_in, d_in = E1.shape
        linop = MatfreeLinopODEConstraint(
            ode=self.ode, tree_flatten=rv.tree_flatten, t=t, x=m0
        )

        key, jac = state
        key, subkey = random.split(key, num=2)
        cond = MatfreeLatentCond(
            linop, noise, key=subkey, num_ensembles=self.num_ensembles, bias=self.bias
        )

        return cond, (key, jac)

    # Rewrite the vector field as one that maps
    # Arrays to arrays.
    # Maps (n, d) to (1, d) to conform the Jacobian API
    def information_operator(self, s: Array, *, tree_flatten, t) -> Array:
        # Move to latent space (arg is (n, d) but latent space is (d, n))
        s = s.T

        # Extract all tcoeffs
        jet_coords = tree_flatten.unflatten_array(s)

        # Extract relevant tcoeffs ("jet-coordinates")
        jet_coords = jet_coords[: self.ode.num_tcoeffs_in_args]

        # Evaluate the actual vector field
        fs0 = self.ode.vector_field(jet_coords=jet_coords, t=t)

        # Bring back into (m, d) form.
        return tree.ravel_pytree(fs0)[0][None, :]


class state_space_model_matfree(ssm_impl_api.StateSpaceModel):
    """Implementation of matrix-free extensions for block-diagonal SSMs."""

    def __init__(self, *, key, num_ensembles, bias=False):
        self.key = key
        self.num_ensembles = num_ensembles
        self.bias = bias

    def prior_wiener_integrated(self, *args, **kwargs):
        bd = blockdiag.state_space_model_blockdiag()
        return bd.prior_wiener_integrated(*args, **kwargs)

    def prior_wiener_integrated_diffuse(self, *args, **kwargs):
        bd = blockdiag.state_space_model_blockdiag()
        return bd.prior_wiener_integrated_diffuse(*args, **kwargs)

    def prior_exponential(
        self,
        ode,
        tcoeffs_mean: C,
        /,
        *,
        is_exact: C | bool = True,
        inexact_eps: float = 1e-6,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        raise NotImplementedError

    def prior_exponential_diffuse(
        self,
        ode,
        tcoeffs_mean: C,
        tcoeffs_std: C,
        /,
        *,
        diffuse_derivatives: int = 0,
        diffuse_eps: float = 1.0,
        output_scale: Array | None = None,
    ):
        raise NotImplementedError

    def constraint_ode_ts0(self, ode: problems.JetOde, /):
        raise NotImplementedError

    def constraint_ode_ts1(self, ode: problems.JetOde, /):
        if not isinstance(ode, problems.JetOde):
            raise TypeError(ode)

        return MatfreeOdeTs1(
            ode=ode, key=self.key, num_ensembles=self.num_ensembles, bias=self.bias
        )

    def constraint_residual(
        self,
        residual: problems.JetResidual,
        *,
        taylor_point: taylor_points.TaylorPoint | None = None,
    ):
        raise NotImplementedError


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
