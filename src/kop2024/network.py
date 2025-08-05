import math
from functools import partial
import jax
import jax.numpy as jnp
from kop2024.helpers import monomial_exponents, product


def coefficients(m, n, k):
    d = 4  # TODO: generalize
    assert m >= n

    # all ways of splitting a set of 4 elements into two disjoint subsets
    split_idx = (
        # left set indices
        (0, 0, 0, 1, 1, 2),
        (1, 2, 3, 2, 3, 3),
        # right set indices
        (2, 1, 1, 0, 0, 0),
        (3, 3, 2, 3, 2, 1),
    )

    exponents = monomial_exponents(n, d)

    # compute all indices to sum relative of H for extracting all monomial coefficients
    # as described in section 2 of the paper
    parts = [[], [], [], []]
    for exp in exponents:
        for i, s in enumerate(split_idx):
            parts[i].extend([exp[i] for i in s])
    parts = tuple(
        tuple(s) for s in parts
    )  # make it hashable to pass as static arg to jax.jit

    # compute normlization factors according to section 2
    norm = tuple(
        product(math.factorial(exp.count(a)) for a in set(exp)) for exp in exponents
    )

    @partial(jax.jit, static_argnames=["parts", "norm"])
    def network(A, B, parts, norm):
        d = 4
        n = A.shape[1]
        num_monomials = math.comb(n + d - 1, n - 1)
        split_count = math.comb(d, d // 2)
        H = jnp.einsum("ikj,jl->ilk", B, A)
        H = H + jnp.transpose(H, (1, 0, 2))
        H = jnp.einsum("ijp,klp->ijkl", H, H)
        return jnp.sum(
            H[*parts].reshape(num_monomials, split_count), axis=1
        ) / jnp.array(norm)

    return exponents, partial(network, parts=parts, norm=norm)
