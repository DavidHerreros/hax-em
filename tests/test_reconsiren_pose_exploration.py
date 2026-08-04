import jax
import jax.numpy as jnp
import numpy as np

from hax.networks.reconsiren import (
    _assignment_probabilities,
    _bound_candidate_view_directions,
    _jitter_rotations,
    _local_rotation_proposals,
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


def test_local_proposals_include_current_pose_and_valid_rotations():
    rotations = jnp.broadcast_to(jnp.eye(3), (2, 4, 3, 3))
    proposals = _local_rotation_proposals(
        rotations, jax.random.PRNGKey(4), n_proposals=3, jitter_degrees=8.0)

    assert proposals.shape == (2, 4, 3, 3, 3)
    np.testing.assert_allclose(proposals[:, :, 0], rotations, atol=1e-6)
    products = jnp.matmul(jnp.swapaxes(proposals, -1, -2), proposals)
    np.testing.assert_allclose(
        products, jnp.broadcast_to(jnp.eye(3), products.shape), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.det(proposals), 1.0, rtol=1e-5, atol=1e-5)


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
