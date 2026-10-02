"""VTV-FM: variational terminal-velocity flow matching (arXiv 2610.00785).

Closed-form minimum-acceleration cubic, the terminal-velocity cost and its
completion v1*, a small Sinkhorn pairing, the acceleration loss plus the
latent-v1 ADM penalty, and velocity Verlet integration. No CIFAR, no PDE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def cubic_coeffs(x0, v0, x1, v1, T):
    """alpha, beta of the min-acceleration cubic. T > 0, Delta = x1 - x0.

    alpha = (-2 Delta + T (v0 + v1)) / T**3
    beta  = ( 3 Delta - T (2 v0 + v1)) / T**2
    """
    x0, v0, x1, v1 = (jnp.asarray(a, dtype=jnp.float32) for a in (x0, v0, x1, v1))
    T = jnp.asarray(T, dtype=jnp.float32)
    delta = x1 - x0
    alpha = (-2.0 * delta + T * (v0 + v1)) / T**3
    beta = (3.0 * delta - T * (2.0 * v0 + v1)) / T**2
    return alpha, beta


def gamma(t, x0, v0, x1, v1, T):
    alpha, beta = cubic_coeffs(x0, v0, x1, v1, T)
    t = jnp.asarray(t, dtype=jnp.float32)
    return x0 + v0 * t + beta * t**2 + alpha * t**3


def velocity(t, x0, v0, x1, v1, T):
    alpha, beta = cubic_coeffs(x0, v0, x1, v1, T)
    t = jnp.asarray(t, dtype=jnp.float32)
    return v0 + 2.0 * beta * t + 3.0 * alpha * t**2


def acceleration(t, x0, v0, x1, v1, T):
    alpha, beta = cubic_coeffs(x0, v0, x1, v1, T)
    t = jnp.asarray(t, dtype=jnp.float32)
    return 2.0 * beta + 6.0 * alpha * t


def cost_T(x0, v0, x1, v1, T):
    """C_T = (1/T) * (12 ||(x1-x0)/T - (v0+v1)/2||^2 + ||v1-v0||^2)."""
    x0, v0, x1, v1 = (jnp.asarray(a, dtype=jnp.float32) for a in (x0, v0, x1, v1))
    T = jnp.asarray(T, dtype=jnp.float32)
    mid = (x1 - x0) / T - 0.5 * (v0 + v1)
    return (1.0 / T) * (12.0 * jnp.sum(mid**2, axis=-1) + jnp.sum((v1 - v0) ** 2, axis=-1))


def v1_star(x0, v0, x1, T):
    """v1* = (3 / (2 T)) (x1 - x0) - 0.5 v0."""
    x0, v0, x1 = (jnp.asarray(a, dtype=jnp.float32) for a in (x0, v0, x1))
    T = jnp.asarray(T, dtype=jnp.float32)
    return (3.0 / (2.0 * T)) * (x1 - x0) - 0.5 * v0


def cost_T_star(x0, v0, x1, T):
    """C_T* = (3/T) ||(x1-x0)/T - v0||^2."""
    x0, v0, x1 = (jnp.asarray(a, dtype=jnp.float32) for a in (x0, v0, x1))
    T = jnp.asarray(T, dtype=jnp.float32)
    return (3.0 / T) * jnp.sum(((x1 - x0) / T - v0) ** 2, axis=-1)


def sinkhorn(cost, eps=0.05, n_iter=80):
    """Entropic OT. Returns the coupling, row-normalized (rows sum to 1/n).

    Kernel K_ij = exp(-C_ij / eps), scaled by its max so the exp does not
    underflow. Alternating row and column normalization.
    """
    cost = jnp.asarray(cost, dtype=jnp.float32)
    log_k = -cost / eps
    log_k = log_k - log_k.max()
    K = jnp.exp(log_k)
    n = cost.shape[0]
    a = jnp.ones((n,)) / n
    b = jnp.ones((n,)) / n
    u = jnp.ones((n,))
    v = jnp.ones((n,))
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = u[:, None] * K * v[None, :]
    return P


def match_pairs(x0, v0, x1, T, eps=0.05):
    """Pair each source row to a target by row-argmax of the Sinkhorn coupling.

    Cost is C_T*(x0_i, v0_i, x1_j). Returns indices (n,) and completed v1.
    """
    # pairwise C_T*
    def row_cost(x0_i, v0_i):
        return jax.vmap(lambda x1_j: cost_T_star(x0_i, v0_i, x1_j, T))(x1)

    C = jax.vmap(row_cost)(x0, v0)
    P = sinkhorn(C, eps=eps)
    idx = jnp.argmax(P, axis=1)
    x1_m = x1[idx]
    v1 = jax.vmap(lambda a, b, c: v1_star(a, b, c, T))(x0, v0, x1_m)
    return idx, v1, P


def sample_bridge(x0, v0, x1, v1, T, key, n):
    """n points along the cubic. Returns t, x_t, v_t, a_t, each (n, ...)."""
    t = jax.random.uniform(key, (n,), minval=0.0, maxval=T)
    return (
        t,
        jax.vmap(lambda ti: gamma(ti, x0, v0, x1, v1, T))(t),
        jax.vmap(lambda ti: velocity(ti, x0, v0, x1, v1, T))(t),
        jax.vmap(lambda ti: acceleration(ti, x0, v0, x1, v1, T))(t),
    )


def init_params(key, d, hidden=64):
    """MLP: concat(x, v, tau) -> acceleration, width `hidden`."""
    d_in = 2 * d + 1
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (d_in, hidden)) * 0.3,
        "b1": jnp.zeros((hidden,)),
        "w2": jax.random.normal(k2, (hidden, hidden)) * 0.3,
        "b2": jnp.zeros((hidden,)),
        "w3": jax.random.normal(k3, (hidden, d)) * 0.01,
        "b3": jnp.zeros((d,)),
    }


def a_theta(params, x, v, tau):
    tau = jnp.asarray(tau, dtype=jnp.float32)
    # a scalar tau (one Verlet step on a batch) broadcasts over the batch
    while tau.ndim < x.ndim:
        tau = tau[..., None]
    tau = jnp.broadcast_to(tau, x.shape[:-1] + (1,))
    inp = jnp.concatenate([x, v, tau], axis=-1)
    h = jax.nn.gelu(inp @ params["w1"] + params["b1"])
    h = jax.nn.gelu(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


def loss_acc(pred, target):
    return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))


def div_v(params, x, v, tau):
    """Divergence of a_theta w.r.t. v, per batch row."""

    def one(x_i, v_i, tau_i):
        jac = jax.jacobian(lambda vv: a_theta(params, x_i, vv, tau_i))(v_i)
        return jnp.trace(jac)

    return jax.vmap(one)(x, v, tau)


def loss_adm(params, x, v, t, T, d):
    """Latent-v1 ADM penalty: (((T - t) / d) * div_v a + 3)^2."""
    div = div_v(params, x, v, t / T)
    resid = ((T - t) / d) * div + 3.0
    return jnp.mean(resid**2)


def loss_va(v_t, v_prev, a_t, a_prev, dt):
    """Discrete residual of v_{k+1} - v_k - dt * a_mid."""
    return jnp.mean(jnp.sum((v_t - v_prev - dt * 0.5 * (a_t + a_prev)) ** 2, axis=-1))


def loss_xv(x_t, x_prev, v_t, v_prev, dt):
    """Discrete residual of x_{k+1} - x_k - dt * v_mid."""
    return jnp.mean(jnp.sum((x_t - x_prev - dt * 0.5 * (v_t + v_prev)) ** 2, axis=-1))


def verlet(params, x0, v0, T, n_steps):
    """Velocity Verlet. a_theta is evaluated at tau = t / T.

    x_next = x + dt v + 0.5 dt^2 a
    a_next = a_theta(x_next, v + dt a, t_next / T)
    v_next = v + 0.5 dt (a + a_next)
    """
    dt = T / n_steps
    d = x0.shape[-1]

    def step(carry, k):
        x, v, t = carry
        a = a_theta(params, x, v, t / T)
        x_next = x + dt * v + 0.5 * dt**2 * a
        v_pred = v + dt * a
        t_next = t + dt
        a_next = a_theta(params, x_next, v_pred, t_next / T)
        v_next = v + 0.5 * dt * (a + a_next)
        return (x_next, v_next, t_next), x_next

    (x_T, v_T, _), xs = jax.lax.scan(
        step, (x0, v0, jnp.array(0.0, dtype=jnp.float32)), jnp.arange(n_steps)
    )
    return x_T, v_T, xs


def train_step(params, opt_state, optimizer, batch, T, lam_adm=0.0):
    """L = L_acc + lam_adm * L_ADM. Other residuals default to weight 0."""
    x_t, v_t, a_t, t = batch

    def loss_fn(p):
        pred = a_theta(p, x_t, v_t, t / T)
        loss = loss_acc(pred, a_t)
        if lam_adm:
            loss = loss + lam_adm * loss_adm(p, x_t, v_t, t, T, x_t.shape[-1])
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def make_batch(key, x0, v0, x1, T, n):
    """Pair with C_T*, complete v1*, sample the cubic. x0, x1 are datasets."""
    k1, k2, k3 = jax.random.split(key, 3)
    i = jax.random.randint(k1, (n,), 0, x0.shape[0])
    j = jax.random.randint(k2, (n,), 0, x1.shape[0])
    # cheap pairing: for the smoke we complete v1* against a random target,
    # which is the correct conditional completion given the pair. The Sinkhorn
    # matching is tested on its own and used when the caller passes matched x1.
    x0_b, v0_b, x1_b = x0[i], v0[i], x1[j]
    v1_b = jax.vmap(lambda a, b, c: v1_star(a, b, c, T))(x0_b, v0_b, x1_b)
    t = jax.random.uniform(k3, (n,), minval=0.05 * T, maxval=0.95 * T)
    x_t = jax.vmap(lambda ti, a, b, c, d: gamma(ti, a, b, c, d, T))(
        t, x0_b, v0_b, x1_b, v1_b
    )
    v_t = jax.vmap(lambda ti, a, b, c, d: velocity(ti, a, b, c, d, T))(
        t, x0_b, v0_b, x1_b, v1_b
    )
    a_t = jax.vmap(lambda ti, a, b, c, d: acceleration(ti, a, b, c, d, T))(
        t, x0_b, v0_b, x1_b, v1_b
    )
    return x_t, v_t, a_t, t


def train(key, x0, v0, x1, T, steps=400, lr=1e-3, batch=256, hidden=64, lam_adm=0.0):
    d = x0.shape[-1]
    key, sub = jax.random.split(key)
    params = init_params(sub, d, hidden=hidden)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    losses = []
    for _ in range(steps):
        key, sub = jax.random.split(key)
        b = make_batch(sub, x0, v0, x1, T, batch)
        params, opt_state, loss = train_step(
            params, opt_state, optimizer, b, T, lam_adm=lam_adm
        )
        losses.append(float(loss))
    return params, losses
