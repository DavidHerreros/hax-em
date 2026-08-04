import jax
import jax.numpy as jnp
import numpy as np

from hax.networks.reconsiren import (
    _assignment_probabilities,
    _bound_candidate_view_directions,
    _candidate_coverage_loss,
    _fibonacci_sphere_directions,
    _rotation_matrices_from_rotvec,
)


def test_adaptive_responsibilities_are_affine_loss_invariant():
    losses = jnp.array([[0.1, 0.4, 0.8], [2.0, 1.0, 3.0]], dtype=jnp.float32)
    expected = _assignment_probabilities(losses, temperature=0.7, adaptive=True)
    transformed = _assignment_probabilities(37.0 * losses + 11.0,
                                            temperature=0.7, adaptive=True)

    np.testing.assert_allclose(transformed, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(expected.sum(axis=1)), 1.0, atol=1e-6)
    assert np.all(np.asarray(expected) > 0.0)


def test_candidate_coverage_penalizes_anchor_quantization():
    dense = _fibonacci_sphere_directions(256)
    anchors = _fibonacci_sphere_directions(16)
    quantized = jnp.repeat(anchors, 16, axis=0)
    empty_bank = jnp.zeros((512, 3), dtype=jnp.float32)
    key = jax.random.PRNGKey(4)

    dense_loss = _candidate_coverage_loss(
        dense, empty_bank, 0, key, n_bins=128, kappa=32.0,
        bank_samples=128, bank_mix=0.5)
    quantized_loss = _candidate_coverage_loss(
        quantized, empty_bank, 0, key, n_bins=128, kappa=32.0,
        bank_samples=128, bank_mix=0.5)

    assert float(quantized_loss) > float(dense_loss)
    assert float(dense_loss) >= -1e-6


def test_candidate_coverage_has_finite_direction_gradients():
    directions = _fibonacci_sphere_directions(32)
    bank = _fibonacci_sphere_directions(64)

    def loss_fn(raw_directions):
        unit_directions = raw_directions / jnp.linalg.norm(
            raw_directions, axis=-1, keepdims=True)
        return _candidate_coverage_loss(
            unit_directions, bank, bank.shape[0], jax.random.PRNGKey(6),
            n_bins=64, kappa=24.0, bank_samples=32, bank_mix=0.5)

    gradients = jax.grad(loss_fn)(directions)
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert float(jnp.linalg.norm(gradients)) > 0.0


def test_candidate_view_direction_is_bounded_and_rotation_stays_valid():
    anchors = jnp.eye(3)[None, ...]
    proposed = _rotation_matrices_from_rotvec(
        jnp.deg2rad(jnp.array([[[80.0, 0.0, 0.0]]])))
    bounded = _bound_candidate_view_directions(proposed, anchors, 20.0)

    anchor_direction = anchors[0, :, 2]
    direction = bounded[0, 0, :, 2]
    deviation = jnp.rad2deg(jnp.arccos(jnp.clip(jnp.dot(anchor_direction, direction), -1, 1)))
    assert float(deviation) <= 20.001
    np.testing.assert_allclose(
        bounded @ jnp.swapaxes(bounded, -1, -2),
        jnp.broadcast_to(jnp.eye(3), bounded.shape), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.det(bounded), 1.0, rtol=1e-5, atol=1e-5)
