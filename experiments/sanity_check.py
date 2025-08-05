import jax
import jax.numpy as jnp
from kop2024 import network
from kop2024.helpers import extract_sos_summands, add_polynomials, multiply_polynomials


m, n, k = 6, 4, 2

key = jax.random.key(0)
key, subkey = jax.random.split(key)
A = jax.random.normal(subkey, (m, n))
key, subkey = jax.random.split(key)
B = jax.random.normal(subkey, (n, k, m))

exponents, f = network.coefficients(m, n, k)
coef_vector = f(A, B)

p = {e: float(v) for e, v in zip(exponents, coef_vector)}
g1, g2 = extract_sos_summands(A, B)
pbar = add_polynomials(multiply_polynomials(g1, g1), multiply_polynomials(g2, g2))

for e in exponents:
    assert jnp.isclose(p[e], pbar[e])
