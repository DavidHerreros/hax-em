import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from hax.networks.reconsiren import (
    _PoseDiagnosticsTracker,
    ReconSIREN,
    _candidate_coverage_loss,
    _candidate_reconstruction_losses,
    _consensus_multiscale_size_and_weight,
    _fibonacci_sphere_directions,
    _symmetry_aware_rotation_change_degrees,
    _top_two_candidate_diagnostics,
    train_step_reconsiren,
)


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


def test_low_frequency_candidate_loss_suppresses_high_frequency_error():
    axis = jnp.linspace(-1.0, 1.0, 16)
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    target_2d = jnp.exp(-4.0 * (xx ** 2 + yy ** 2))
    target_2d = (target_2d - jnp.mean(target_2d)) / jnp.std(target_2d)
    checkerboard = ((-1.0) ** (jnp.arange(16)[:, None] + jnp.arange(16)[None, :]))
    images = jnp.stack([target_2d, target_2d + 0.5 * checkerboard], axis=0)[None]
    target = target_2d[None, ..., None]
    ctf = jnp.ones((1, 32, 17), dtype=jnp.float32)

    full = _candidate_reconstruction_losses(images, target, ctf, None)
    low = _candidate_reconstruction_losses(
        images, target, ctf, None, scoring_size=8)

    assert float(full[0, 1] - full[0, 0]) > 0.2
    assert float(low[0, 1] - low[0, 0]) < 0.05


def test_consensus_multiscale_curriculum_uses_discrete_stages_and_decays_weight():
    scales = (0.25, 0.5, 0.75)
    assert _consensus_multiscale_size_and_weight(100, 0, 10, 30, scales) == (25, 1.0)
    size, weight = _consensus_multiscale_size_and_weight(100, 100, 10, 30, scales)
    assert size == 50
    np.testing.assert_allclose(weight, 2 / 3)
    size, weight = _consensus_multiscale_size_and_weight(100, 200, 10, 30, scales)
    assert size == 75
    np.testing.assert_allclose(weight, 1 / 3)
    assert _consensus_multiscale_size_and_weight(100, 300, 10, 30, scales) == (100, 0.0)


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

    angle = np.deg2rad(10.0)
    ten_degrees = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)]], dtype=np.float32)
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
        consensus_multiscale_size=8, consensus_multiscale_weight=0.5,
        train_heterogeneity=False, return_metrics=True,
        return_pose_diagnostics=True)

    assert np.isfinite(float(loss))
    assert len(metrics) == 19
    assert np.all(np.isfinite(np.asarray(metrics)))
    np.testing.assert_allclose(metrics[12], 1.0, atol=1e-5)
    assert 0.0 <= float(metrics[16]) <= 1.0
    assert float(metrics[17]) >= 0.0
    np.testing.assert_allclose(metrics[18], 0.5 * (metrics[0] + metrics[17]))
    (winner_rotations, selected_heads, best_heads, absolute, relative,
     standardized, median_normalized, score_entropy) = diagnostics
    assert winner_rotations.shape == (2, 3, 3)
    assert (selected_heads.shape == best_heads.shape == absolute.shape == relative.shape
            == standardized.shape == median_normalized.shape == score_entropy.shape == (2,))
