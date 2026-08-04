import jax
import jax.numpy as jnp
import numpy as np

from hax.networks.reconsiren import (
    _assignment_probabilities,
    _head_balance_loss,
    _jitter_rotations,
)


def test_adaptive_responsibilities_are_affine_loss_invariant():
    losses = jnp.array([[0.1, 0.4, 0.8], [2.0, 1.0, 3.0]], dtype=jnp.float32)
    expected = _assignment_probabilities(losses, temperature=0.7, adaptive=True)
    transformed = _assignment_probabilities(37.0 * losses + 11.0,
                                            temperature=0.7, adaptive=True)

    np.testing.assert_allclose(transformed, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(expected.sum(axis=1)), 1.0, atol=1e-6)
    assert np.all(np.asarray(expected) > 0.0)


def test_head_balance_penalizes_collapsed_usage():
    uniform = jnp.full((8, 4), 0.25)
    collapsed = jax.nn.one_hot(jnp.zeros((8,), dtype=jnp.int32), 4)

    np.testing.assert_allclose(_head_balance_loss(uniform), 0.0, atol=1e-7)
    assert float(_head_balance_loss(collapsed)) > 1.0


def test_rotation_jitter_is_optional_and_stays_on_so3():
    rotations = jnp.broadcast_to(jnp.eye(3), (4, 6, 3, 3))
    key = jax.random.PRNGKey(3)

    unchanged = _jitter_rotations(rotations, key, 0.0)
    jittered = _jitter_rotations(rotations, key, 12.0)

    np.testing.assert_allclose(unchanged, rotations, atol=1e-6)
    assert not np.allclose(np.asarray(jittered), np.asarray(rotations))
    products = jnp.matmul(jnp.swapaxes(jittered, -1, -2), jittered)
    expected_identity = jnp.broadcast_to(jnp.eye(3), products.shape)
    np.testing.assert_allclose(products, expected_identity, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.det(jittered), 1.0, rtol=1e-5, atol=1e-5)
