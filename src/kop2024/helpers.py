import functools
import operator
import itertools
import jax.numpy as jnp


def product(iterable):
    return functools.reduce(operator.mul, iterable, 1.0)


def monomial_exponents(n: int, d: int) -> list[tuple[int]]:
    """
    Generate a list of all monomials in n variables of degree exactly d.
    """

    if n > 1:
        # use the "stars and bars" method
        out = []
        for c in itertools.combinations(range(d + n - 1), n - 1):
            # compute the n-tuple for which that assigns an exponent to each variable
            indicator = tuple(
                [c[0]]
                + [c[i + 1] - c[i] - 1 for i in range(n - 2)]
                + [d + n - 2 - c[-1]]
            )

            # save each appearing variable with its index according to its multiplicity
            variables = []
            for i, a in enumerate(indicator):
                if a != 0:
                    variables.extend(a * [i])
            out.append(tuple(variables))
        return out
    else:
        # for n=1 there is exactly one exponent
        return [tuple(d * [0])]


def extract_sos_summands(A, B):
    coef_matrix = jnp.einsum("plj,jk->lkp", B, A)
    n = coef_matrix.shape[1]
    k = coef_matrix.shape[0]
    exponents = monomial_exponents(n, 2)

    out = []
    for h in range(k):
        summand = {}
        for e in exponents:
            if e[0] == e[1]:
                summand[e] = float(coef_matrix[h, *e])
            else:
                summand[e] = float(coef_matrix[h, *e]) + float(
                    coef_matrix[h, e[1], e[0]]
                )
        out.append(summand)

    return out


def add_polynomials(p, q):
    out = {e: v for e, v in p.items()}
    for e, v in q.items():
        if e not in out:
            out[e] = v
        else:
            out[e] += v
    return out


def multiply_polynomials(p, q):
    out = {}
    for (pe, pv), (qe, qv) in itertools.product(p.items(), q.items()):
        e = tuple(sorted(list(pe) + list(qe)))
        v = pv * qv
        if e not in out:
            out[e] = v
        else:
            out[e] += v
    return out
