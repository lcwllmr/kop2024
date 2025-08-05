import jax
from kop2024 import network


m, n, k = 6, 4, 2


key = jax.random.key(0)
key, subkey = jax.random.split(key)
A = jax.random.normal(subkey, (m, n))
key, subkey = jax.random.split(key)
B = jax.random.normal(subkey, (n, k, m))


exponents, f = network.coefficients(m, n, k)
coef_vector = f(A, B)

print({e: float(v) for e, v in zip(exponents, coef_vector)})
