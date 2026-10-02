"""Equation checks for the VTV-FM slice. Seeded. Not tautologies."""

import jax
import jax.numpy as jnp
import numpy as np

import vtv_fm


def test_boundaries_T_not_one():
    T = 2.5
    x0 = jnp.array([0.3, -1.1])
    v0 = jnp.array([0.4, 0.2])
    x1 = jnp.array([-0.7, 1.4])
    v1 = jnp.array([0.1, -0.5])
    np.testing.assert_allclose(vtv_fm.gamma(0.0, x0, v0, x1, v1, T), x0, atol=1e-5)
    np.testing.assert_allclose(vtv_fm.gamma(T, x0, v0, x1, v1, T), x1, atol=1e-5)
    np.testing.assert_allclose(vtv_fm.velocity(0.0, x0, v0, x1, v1, T), v0, atol=1e-5)
    np.testing.assert_allclose(vtv_fm.velocity(T, x0, v0, x1, v1, T), v1, atol=1e-5)


def test_velocity_acceleration_match_finite_differences():
    T = 1.7
    x0 = jnp.array([0.2, 0.5, -0.4])
    v0 = jnp.array([-0.3, 0.1, 0.6])
    x1 = jnp.array([1.0, -0.2, 0.3])
    v1 = jnp.array([0.4, -0.7, 0.2])
    ts = jnp.linspace(0.1, T - 0.1, 9)
    eps = 1e-3
    for t in ts:
        g_p = vtv_fm.gamma(float(t) + eps, x0, v0, x1, v1, T)
        g_m = vtv_fm.gamma(float(t) - eps, x0, v0, x1, v1, T)
        fd_v = (g_p - g_m) / (2 * eps)
        np.testing.assert_allclose(
            vtv_fm.velocity(t, x0, v0, x1, v1, T), fd_v, atol=1e-3
        )
        v_p = vtv_fm.velocity(float(t) + eps, x0, v0, x1, v1, T)
        v_m = vtv_fm.velocity(float(t) - eps, x0, v0, x1, v1, T)
        fd_a = (v_p - v_m) / (2 * eps)
        np.testing.assert_allclose(
            vtv_fm.acceleration(t, x0, v0, x1, v1, T), fd_a, atol=1e-3
        )


def test_cost_matches_trapezoid_of_acceleration_energy():
    T = 2.5
    x0 = jnp.array([0.0, 1.0])
    v0 = jnp.array([0.3, -0.2])
    x1 = jnp.array([1.5, -0.4])
    v1 = jnp.array([-0.1, 0.6])
    n = 4001
    ts = jnp.linspace(0.0, T, n)
    acc = jax.vmap(lambda t: vtv_fm.acceleration(t, x0, v0, x1, v1, T))(ts)
    energy = jnp.sum(acc**2, axis=-1)
    # C_T equals the integral of ||gamma''||^2, not the mean. The 1/T in the
    # closed form is already the right normalization; dividing the trapezoid
    # by T again is a double count.
    trap = jnp.trapezoid(energy, ts)
    closed = vtv_fm.cost_T(x0, v0, x1, v1, T)
    rel = abs(float(trap) - float(closed)) / float(closed)
    assert rel < 1e-3, (float(trap), float(closed), rel)


def test_v1_star_minimizes_and_matches_cost_star():
    T = 1.3
    x0 = jnp.array([0.2, -0.5])
    v0 = jnp.array([0.4, 0.1])
    x1 = jnp.array([1.1, 0.7])
    v_star = vtv_fm.v1_star(x0, v0, x1, T)
    c_star = float(vtv_fm.cost_T_star(x0, v0, x1, T))
    c_at = float(vtv_fm.cost_T(x0, v0, x1, v_star, T))
    np.testing.assert_allclose(c_at, c_star, rtol=1e-5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = jnp.array(rng.normal(size=2).astype(np.float32))
        assert float(vtv_fm.cost_T(x0, v0, x1, v, T)) >= c_at - 1e-4


def test_decomposition():
    T = 2.5
    rng = np.random.default_rng(1)
    for _ in range(10):
        x0, v0, x1, v1 = (
            jnp.array(rng.normal(size=3).astype(np.float32)) for _ in range(4)
        )
        left = vtv_fm.cost_T(x0, v0, x1, v1, T)
        v_star = vtv_fm.v1_star(x0, v0, x1, T)
        right = vtv_fm.cost_T_star(x0, v0, x1, T) + (4.0 / T) * jnp.sum(
            (v1 - v_star) ** 2
        )
        np.testing.assert_allclose(left, right, rtol=1e-5)


def test_sinkhorn_puts_mass_on_cheap_pairs():
    # two clusters: sources near 0 pair cheaply with targets near 0
    x0 = jnp.array([[0.0, 0.0], [0.1, -0.1], [5.0, 5.0], [5.2, 4.8]])
    v0 = jnp.zeros_like(x0)
    x1 = jnp.array([[0.05, 0.0], [5.1, 5.0], [-0.05, 0.1], [4.9, 5.2]])
    T = 1.0
    idx, v1, P = vtv_fm.match_pairs(x0, v0, x1, T, eps=0.05)
    # rows sum to 1/n (the balanced marginal), not to 1
    np.testing.assert_allclose(P.sum(axis=1), 0.25, atol=1e-3)
    # cheap targets for the first cluster are columns 0 and 2
    assert int(idx[0]) in (0, 2)
    assert int(idx[1]) in (0, 2)
    assert int(idx[2]) in (1, 3)
    assert int(idx[3]) in (1, 3)
    # completed v1 equals v1* of the matched target
    for i in range(4):
        expect = vtv_fm.v1_star(x0[i], v0[i], x1[int(idx[i])], T)
        np.testing.assert_allclose(v1[i], expect, atol=1e-5)


def test_verlet_constant_acceleration_matches_quadratic():
    # a field that ignores state and returns a constant
    a_const = jnp.array([0.3, -0.2])

    def constant_field(params, x, v, tau):
        return jnp.broadcast_to(a_const, x.shape)

    x0 = jnp.array([[0.0, 1.0], [0.5, -0.5]])
    v0 = jnp.array([[0.1, 0.0], [-0.2, 0.4]])
    T = 1.6
    n = 64
    # patch a_theta
    orig = vtv_fm.a_theta
    vtv_fm.a_theta = constant_field
    try:
        x_T, v_T, _ = vtv_fm.verlet(None, x0, v0, T, n)
    finally:
        vtv_fm.a_theta = orig
    np.testing.assert_allclose(x_T, x0 + v0 * T + 0.5 * a_const * T**2, atol=1e-4)
    np.testing.assert_allclose(v_T, v0 + a_const * T, atol=1e-4)


def test_divergence_and_adm_zero_field():
    d = 3
    M = jnp.array([[0.2, 0.0, 0.1], [-0.3, 0.5, 0.0], [0.0, 0.2, -0.4]])

    def linear(params, x, v, tau):
        return v @ M.T

    orig = vtv_fm.a_theta
    vtv_fm.a_theta = linear
    try:
        x = jnp.zeros((4, d))
        v = jnp.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -0.2, 0.5], [0.0, 0.0, 0.0]]
        )
        tau = jnp.zeros((4,))
        div = vtv_fm.div_v(None, x, v, tau)
        np.testing.assert_allclose(div, jnp.trace(M), atol=1e-5)
    finally:
        vtv_fm.a_theta = orig

    # a field with div_v = -3 d / (T - t) zeros L_ADM
    T = 2.0
    t = jnp.array([0.5, 1.0, 1.5])

    def killing(params, x, v, tau):
        # div_v of s(tau)*v is d*s. Want ((T-t)/d)*div + 3 = 0, so s = -3/(T-t).
        scale = -3.0 / (T * (1.0 - tau[..., None]))
        return v * scale

    vtv_fm.a_theta = killing
    try:
        x = jax.random.normal(jax.random.PRNGKey(0), (3, d))
        v = jax.random.normal(jax.random.PRNGKey(1), (3, d))
        loss = float(vtv_fm.loss_adm(None, x, v, t, T, d))
    finally:
        vtv_fm.a_theta = orig
    assert loss < 1e-8


def test_smoke_learns():
    key = jax.random.PRNGKey(0)
    T = 1.0
    d = 2
    n_data = 512
    k1, k2 = jax.random.split(key)
    x0 = jax.random.normal(k1, (n_data, d)) * 0.3
    v0 = jnp.zeros_like(x0)
    x1 = jax.random.normal(k2, (n_data, d)) * 0.3 + jnp.array([1.5, -1.0])
    target_mean = x1.mean(axis=0)

    key, sub = jax.random.split(key)
    fresh = vtv_fm.init_params(sub, d, hidden=64)
    b0 = vtv_fm.make_batch(jax.random.PRNGKey(1), x0, v0, x1, T, 256)
    l0 = float(vtv_fm.loss_acc(vtv_fm.a_theta(fresh, b0[0], b0[1], b0[3] / T), b0[2]))

    params, losses = vtv_fm.train(
        key, x0, v0, x1, T, steps=300, lr=2e-3, batch=256, hidden=64
    )
    b1 = vtv_fm.make_batch(jax.random.PRNGKey(1), x0, v0, x1, T, 256)
    l1 = float(vtv_fm.loss_acc(vtv_fm.a_theta(params, b1[0], b1[1], b1[3] / T), b1[2]))

    src = jax.random.normal(jax.random.PRNGKey(2), (256, d)) * 0.3
    x_hat, _, _ = vtv_fm.verlet(params, src, jnp.zeros_like(src), T, n_steps=32)
    x_fresh, _, _ = vtv_fm.verlet(fresh, src, jnp.zeros_like(src), T, n_steps=32)
    err = float(jnp.linalg.norm(x_hat.mean(0) - target_mean))
    err_fresh = float(jnp.linalg.norm(x_fresh.mean(0) - target_mean))
    print(
        f"smoke L_acc {l0:.4f} -> {l1:.4f}  "
        f"mean err {err_fresh:.4f} -> {err:.4f}  "
        f"train {losses[0]:.3f} -> {losses[-1]:.3f}"
    )
    assert l1 < l0 / 2.0
    assert err < err_fresh
