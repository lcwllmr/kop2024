import time
import jax
import jax.numpy as jnp
import optax
from kop2024 import network


key = jax.random.key(0)
m, n, k = 40, 30, 10


def build_random_A_B():
    global key
    key, subkey = jax.random.split(key)
    A = jax.random.normal(subkey, (m, n))
    key, subkey = jax.random.split(key)
    B = jax.random.normal(subkey, (n, k, m))
    return A, B


tik = time.time()
_, coef = network.coefficients(m, n, k)
A_star, B_star = build_random_A_B()
p = coef(A_star, B_star)
tok = time.time()
print(
    f"Compiled coefficient extractor and built target polynomial in {tok - tik:4f} seconds"
)


@jax.jit
def loss_function(params):
    A, B = params
    p_bar = coef(A, B)
    return jnp.mean(optax.l2_loss(p_bar, p))


optimizer = optax.adam(1e-1)
A, B = build_random_A_B()
params = [A, B]
opt_state = optimizer.init(params)

it = 0
loss = loss_function(params)
it_times = []

# run warm-up to get correct timings
grads = jax.grad(loss_function)(params)
updates, opt_state = optimizer.update(grads, opt_state, params)
_ = optax.apply_updates(params, updates)

while loss > 1e-7:
    it += 1

    tik = time.time()
    grads = jax.grad(loss_function)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    tok = time.time()
    it_times.append(tok - tik)

    loss = loss_function(params)
    if it % 50 == 0:
        mean_time = jnp.mean(jnp.array(it_times))
        print(f"it={it} loss={loss:4e} avg_time={mean_time:.4f}s")
        it_times = []

print(f"Terminated after {it} iterations with loss {loss:4e}")
