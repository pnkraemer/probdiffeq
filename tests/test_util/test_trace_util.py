import collections
import string

from probdiffeq.backend import func, linalg, np, random, testing


@testing.parametrize("probes", [100_000])
@testing.parametrize("seed", range(20))
def test_trace(probes, seed):
    key = random.prng_key(seed=seed)
    A = random.normal(key, shape=(5, 6, 7, 5))

    # Traces
    trace = linalg.einsum("ijki->jk", A)
    traceest, sem = stocheinsum("ijki->jk", A, key=key, probes=probes)

    error = trace - traceest
    error_magnitude = np.amax(np.abs(error))

    assert error_magnitude < 0.1

    # Diagonals
    diag = linalg.einsum("ijki->jik", A)
    diagest, sem = stocheinsum("ijki->jik", A, key=key, probes=probes)

    error = diag - diagest
    error_magnitude = np.amax(np.abs(error))

    assert error_magnitude < 0.1


def stocheinsum(subscripts, operand, *, key, probes: int):
    """stocheinsum: a minimal stochastic einsum.

    Same notation as einsum, but a *repeated* input index is a trace loop: instead
    of forming that diagonal we cut the loop with a random probe z (E[z z^T] = I)
    and let einsum do the rest. Running example: "ijki->jk" on A of shape (2,3,4,2).
    """
    A = operand
    in_sub, out_sub = (s.strip() for s in subscripts.split("->"))
    # "ijki->jk"  ->  in_sub = "ijki", out_sub = "jk"

    # Record every position each label sits at in the input.
    # "ijki"  ->  {i:[0,3], j:[1], k:[2]}
    pos = collections.defaultdict(list)
    for p, ch in enumerate(in_sub):
        pos[ch].append(p)

    # A label appearing 2+ times is a trace loop to estimate; here just "i".
    # If nothing repeats there's no loop, so fall back to exact einsum.
    repeated = [ch for ch, ps in pos.items() if len(ps) >= 2]
    if not repeated:
        return linalg.einsum(subscripts, A)

    # Pick dummy letters not already used, to rename the legs of each loop.
    # used = {i,j,k}; fresh yields 'a','b',... (skipping i,j,k)
    fresh = (c for c in string.ascii_letters if c not in set(in_sub) | set(out_sub))

    # Rename each occurrence of a looped label to its own dummy leg, and remember
    # each loop's size so we can draw a probe of the right length.
    # "ijki" -> "ajkb"; loops = [(2, ['a','b'])]   (i sat at positions 0 and 3, dim 2)
    new_in, new_out = list(in_sub), list(out_sub)
    loops = []
    for ch in repeated:
        size, legs = A.shape[pos[ch][0]], []
        for p in pos[ch]:
            legs.append(next(fresh))
            new_in[p] = legs[-1]
        if ch in out_sub:  # 'ii->i' style: keep one leg alive
            new_out[out_sub.index(ch)] = legs[0]
        loops.append((size, legs))

    # Build the rewritten einsum: original tensor + one probe per leg.
    # "ajkb,a,b->jk"
    spec = f"{''.join(new_in)},{','.join(d for _, legs in loops for d in legs)}->{''.join(new_out)}"

    # One probe draw: a Rademacher z per loop (SAME z on every leg of that loop),
    # then a plain einsum. E_z[ einsum("ajkb,a,b->jk", A, z, z) ] = sum_i A[i,j,k,i].
    def sample(k):
        operands = [A]
        for (size, legs), kk in zip(loops, random.split(k, len(loops))):
            z = random.rademacher(kk, (size,), dtype=A.dtype)
            operands += [z] * len(legs)
        return linalg.einsum(spec, *operands)

    # Average sample() over probes independent draws; free indices j,k pass through,
    # so the result keeps the exact einsum's shape (3, 4).
    keys = random.split(key, probes)
    samples = func.vmap(sample)(keys)
    mean = np.mean(samples, axis=0)
    sem = np.std(samples, axis=0) / np.sqrt(probes)
    return mean, sem
