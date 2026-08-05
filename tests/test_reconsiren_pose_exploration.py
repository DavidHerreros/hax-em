import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from hax.networks.reconsiren import (
    _PoseDiagnosticsTracker,
    _TemporalCandidateScoreBank,
    ReconSIREN,
    _assignment_probabilities,
    _blend_candidate_scores_with_history,
    _bound_candidate_view_directions,
    _candidate_coverage_loss,
    _candidate_reconstruction_losses,
    _fibonacci_sphere_directions,
    _rotation_matrices_from_rotvec,
    _select_candidates_with_head_hysteresis,
    _symmetry_aware_rotation_change_degrees,
    _top_two_candidate_diagnostics,
    train_step_reconsiren,
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


def test_ncc_candidate_scoring_is_projection_scale_and_offset_invariant():
    target = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1)
    candidate = target[..., 0][:, None, ...]
    transformed = 23.0 * candidate - 71.0
    images = jnp.stack([candidate[:, 0], transformed[:, 0]], axis=1)
    ctf = jnp.ones((1, 8, 5), dtype=jnp.float32)

    losses = _candidate_reconstruction_losses(
        images, target, ctf, None, scoring="ncc")

    np.testing.assert_allclose(losses, np.zeros((1, 2)), atol=1e-5)


def test_ncc_candidate_scoring_prefers_shape_over_amplitude():
    target = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1)
    correlated = 0.01 * target[..., 0]
    anticorrelated = -100.0 * target[..., 0]
    images = jnp.stack([correlated, anticorrelated], axis=1)
    ctf = jnp.ones((1, 8, 5), dtype=jnp.float32)

    losses = _candidate_reconstruction_losses(
        images, target, ctf, None, scoring="ncc")

    assert float(losses[0, 0]) < float(losses[0, 1])


def test_candidate_head_hysteresis_keeps_only_ambiguous_previous_winner():
    losses = jnp.array([
        [0.10, 0.11, 0.50],
        [0.10, 0.30, 0.50],
        [0.20, 0.10, 0.30],
    ])
    head_indices = jnp.broadcast_to(jnp.arange(3), losses.shape)
    previous_heads = jnp.array([1, 1, 7])

    selected, retained, accepted, available, contested, advantage = (
        _select_candidates_with_head_hysteresis(
            losses, head_indices, previous_heads, threshold_std=0.25))

    np.testing.assert_array_equal(selected, np.array([1, 0, 1]))
    np.testing.assert_array_equal(retained, np.array([True, False, False]))
    np.testing.assert_array_equal(accepted, np.array([False, True, False]))
    np.testing.assert_array_equal(available, np.array([True, True, False]))
    np.testing.assert_array_equal(contested, np.array([True, True, False]))
    assert float(advantage[0]) < 0.25 < float(advantage[1])


def test_temporal_candidate_scores_prefer_consistent_head():
    losses = jnp.array([[0.10, 0.09, 0.50]], dtype=jnp.float32)
    head_indices = jnp.array([[0, 1, 2]])
    historical_scores = jnp.array([[-1.0, 0.5, 1.5]], dtype=jnp.float32)
    historical_valid = jnp.ones_like(historical_scores, dtype=bool)

    blended, current, valid = _blend_candidate_scores_with_history(
        losses, head_indices, historical_scores, historical_valid,
        history_weight=0.5)

    assert int(jnp.argmin(current, axis=1)[0]) == 1
    assert int(jnp.argmin(blended, axis=1)[0]) == 0
    np.testing.assert_array_equal(valid, np.ones((1, 3), dtype=bool))


def test_temporal_candidate_score_bank_uses_ema_and_initializes_unseen_heads():
    bank = _TemporalCandidateScoreBank(n_particles=2, n_heads=3)
    bank.update(np.array([0]), np.array([[0, 2]]),
                np.array([[1.0, -1.0]]), decay=0.8)
    bank.update(np.array([0]), np.array([[0, 1]]),
                np.array([[0.0, 2.0]]), decay=0.8)

    scores, valid = bank.batch(np.array([0]))
    np.testing.assert_allclose(scores, np.array([[0.8, 2.0, -1.0]]), atol=1e-6)
    np.testing.assert_array_equal(valid, np.ones((1, 3), dtype=bool))


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


def test_top_two_candidate_diagnostics_report_scale_aware_margin():
    losses = jnp.array([[0.2, 0.5, 0.8], [3.0, 1.0, 2.0]], dtype=jnp.float32)
    (best, absolute, relative, standardized,
     median_normalized, score_entropy) = _top_two_candidate_diagnostics(losses)

    np.testing.assert_array_equal(best, np.array([0, 1]))
    np.testing.assert_allclose(absolute, np.array([0.3, 1.0]), atol=1e-6)
    np.testing.assert_allclose(relative, np.array([0.6, 0.5]), atol=1e-6)
    np.testing.assert_allclose(
        standardized, np.array([1.2247449, 1.2247449]), atol=1e-6)
    np.testing.assert_allclose(median_normalized, np.ones(2), atol=1e-6)
    assert np.all(np.asarray(score_entropy) > 0.0)
    assert np.all(np.asarray(score_entropy) < 1.0)

    shifted_scaled = _top_two_candidate_diagnostics(19.0 * losses + 100.0)
    np.testing.assert_allclose(shifted_scaled[3], standardized, atol=1e-5)
    np.testing.assert_allclose(shifted_scaled[4], median_normalized, atol=1e-5)
    np.testing.assert_allclose(shifted_scaled[5], score_entropy, atol=1e-5)


def test_rotation_change_uses_closest_symmetry_equivalent_pose():
    identity = np.eye(3, dtype=np.float32)
    half_turn_z = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    symmetries = np.stack([identity, half_turn_z])
    previous = identity[None, ...]
    equivalent_current = half_turn_z[None, ...]

    change = _symmetry_aware_rotation_change_degrees(
        previous, equivalent_current, symmetries)

    np.testing.assert_allclose(change, 0.0, atol=1e-4)


def test_pose_diagnostics_tracker_compares_same_particles_across_epochs():
    identity = np.eye(3, dtype=np.float32)
    tracker = _PoseDiagnosticsTracker(2, identity[None, ...])
    labels = np.array([0, 1])
    rotations_epoch_zero = np.stack([identity, identity])
    tracker.update(labels, rotations_epoch_zero, np.array([0, 1]), np.array([0, 1]),
                   np.array([0.2, 0.3]), np.array([0.1, 0.2]),
                   np.array([0.5, 0.6]), np.array([0.7, 0.8]),
                   np.array([0.9, 0.85]), epoch=0)

    ten_degrees = np.asarray(_rotation_matrices_from_rotvec(
        jnp.deg2rad(jnp.array([[10.0, 0.0, 0.0]]))))[0]
    rotations_epoch_one = np.stack([ten_degrees, identity])
    tracker.update(labels, rotations_epoch_one, np.array([2, 1]), np.array([2, 1]),
                   np.array([0.4, 0.5]), np.array([0.3, 0.4]),
                   np.array([0.7, 0.8]), np.array([0.9, 1.0]),
                   np.array([0.8, 0.75]), epoch=1)
    summary = tracker.summary()

    np.testing.assert_allclose(summary["pose_change_mean_degrees"], 5.0, atol=1e-3)
    np.testing.assert_allclose(summary["head_switch_fraction"], 0.5, atol=1e-6)
    np.testing.assert_allclose(summary["comparison_coverage_fraction"], 1.0, atol=1e-6)


def test_train_step_can_return_pose_diagnostics_without_changing_metrics():
    model = ReconSIREN(
        coords=jnp.zeros((4, 3)), values=jnp.full((4,), 0.01),
        xsize=16, sr=1.0, bank_size=32, ctf_type=None,
        num_components=3, optimization_profile="aggressive",
        pose_head_rank=2, pose_spatial_pool=4, consensus_parameterization="direct",
        render_chunk_size=0, candidate_chunk_size=0, coarse_topk=3,
        coarse_gaussians=0, heterogeneity_profile="legacy",
        rngs=nnx.Rngs(3))
    pose_params = nnx.All(nnx.Param, nnx.PathContains("encoder_pose"))
    volume_params = nnx.All(nnx.Param, nnx.PathContains("delta_volume_decoder"))
    het_params = nnx.All(
        nnx.Param,
        (nnx.PathContains("encoder_het"), nnx.PathContains("delta_het_decoder")))
    optimizers = (
        nnx.Optimizer(model, optax.adam(1e-4), wrt=pose_params),
        nnx.Optimizer(model, optax.adam(1e-4), wrt=volume_params),
        nnx.Optimizer(model, optax.adam(1e-4), wrt=het_params),
    )
    graphdef, state = nnx.split((model, *optimizers))
    images = jax.random.normal(jax.random.PRNGKey(7), (2, 16, 16, 1))
    labels = jnp.array([0, 1])

    loss, metrics, diagnostics, _, _ = train_step_reconsiren(
        graphdef, state, images, labels, {}, jax.random.PRNGKey(8),
        assignment_mode="hard", uniform_scope="off",
        previous_candidate_heads=jnp.array([1, 2]),
        apply_candidate_hysteresis=True, candidate_hysteresis_std=0.25,
        historical_candidate_scores=jnp.array([
            [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], dtype=jnp.float32),
        historical_candidate_valid=jnp.ones((2, 3), dtype=bool),
        apply_candidate_temporal_scoring=True, candidate_temporal_weight=0.5,
        train_heterogeneity=False, return_metrics=True,
        return_pose_diagnostics=True)

    assert np.isfinite(float(loss))
    assert len(metrics) == 25
    assert np.all(np.isfinite(np.asarray(metrics)))
    np.testing.assert_allclose(metrics[12], 1.0, atol=1e-5)
    assert 0.0 <= float(metrics[16]) <= 1.0
    np.testing.assert_allclose(metrics[21], 1.0, atol=1e-5)
    (winner_rotations, selected_heads, best_heads, absolute, relative,
     standardized, median_normalized, score_entropy,
     evaluated_heads, current_standardized_scores) = diagnostics
    assert winner_rotations.shape == (2, 3, 3)
    assert (selected_heads.shape == best_heads.shape == absolute.shape == relative.shape
            == standardized.shape == median_normalized.shape == score_entropy.shape == (2,))
    assert evaluated_heads.shape == current_standardized_scores.shape == (2, 3)
