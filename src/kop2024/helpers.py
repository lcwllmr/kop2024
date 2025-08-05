import functools
import operator
import itertools


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
