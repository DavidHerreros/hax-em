import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from hax.networks.reconsiren import (
    _PoseCurriculumController,
    ReconSIREN,
    _assignment_probabilities,
    _candidate_coverage_loss,
    _candidate_reconstruction_losses,
    _consensus_multiscale_size_and_weight,
    _fibonacci_sphere_directions,
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


def test_pose_curriculum_controller_advances_on_the_epoch_schedule():
    controller = _PoseCurriculumController(
        320, (0.25, 0.5, 0.75), min_epochs=1, max_epochs=1)

    assert controller.scoring_size == 80
    assert controller.observe_epoch()
    assert controller.scoring_size == 160
    assert controller.observe_epoch()
    assert controller.scoring_size == 240
    assert controller.multiscale_weight_multiplier > 0.0
    assert controller.observe_epoch()
    assert controller.is_final
    assert controller.scoring_size == 320
    assert controller.multiscale_weight_multiplier == 0.0
    assert not controller.observe_epoch()


def test_pose_curriculum_controller_respects_epoch_bounds_and_roundtrip():
    controller = _PoseCurriculumController(100, (0.5,), min_epochs=2, max_epochs=3)
    assert not controller.observe_epoch()
    assert not controller.observe_epoch()
    assert controller.observe_epoch()
    assert controller.is_final

    source = _PoseCurriculumController(100, (0.25, 0.5, 0.75), max_epochs=1)
    source.observe_epoch()
    restored = _PoseCurriculumController(100, (0.25, 0.5, 0.75))
    restored.load_state_dict(source.state_dict())
    assert restored.stage == source.stage == 1
    assert restored.scoring_size == 50


def test_assignment_probabilities_are_scale_invariant_and_sharpen():
    losses = jnp.array([[0.2, 0.5, 0.8], [3.0, 1.0, 2.0]], dtype=jnp.float32)

    probs = _assignment_probabilities(losses, 0.3)
    rescaled = _assignment_probabilities(19.0 * losses + 100.0, 0.3)
    np.testing.assert_allclose(probs, rescaled, atol=1e-5)
    np.testing.assert_allclose(np.sum(np.asarray(probs), axis=1), np.ones(2),
                               atol=1e-6)

    cold = _assignment_probabilities(losses, 1e-4)
    np.testing.assert_array_equal(np.argmax(cold, axis=1), np.array([0, 1]))
    np.testing.assert_allclose(np.max(cold, axis=1), np.ones(2), atol=1e-5)


def test_pose_curriculum_controller_temperature_matches_stage():
    controller = _PoseCurriculumController(
        100, (0.25, 0.5, 0.75), max_epochs=1, temperatures=(0.3, 0.15, 0.05))

    assert controller.temperature == 0.3
    controller.observe_epoch()
    assert controller.temperature == 0.15
    controller.observe_epoch()
    assert controller.temperature == 0.05
    controller.observe_epoch()
    assert controller.is_final
    assert controller.temperature == 0.0

    broadcast = _PoseCurriculumController(100, (0.25, 0.5), temperatures=(0.0,))
    assert broadcast.temperature == 0.0
    assert broadcast.temperatures == (0.0, 0.0)


def _tiny_reconsiren_split():
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
    return nnx.split((model, *optimizers))


def test_train_step_accepts_consensus_multiscale_blending():
    graphdef, state = _tiny_reconsiren_split()
    images = jax.random.normal(jax.random.PRNGKey(7), (2, 16, 16, 1))
    labels = jnp.array([0, 1])

    loss, metrics, _, _ = train_step_reconsiren(
        graphdef, state, images, labels, {}, jax.random.PRNGKey(8),
        consensus_multiscale_size=8, consensus_multiscale_weight=0.5,
        train_heterogeneity=False, return_metrics=True)

    assert np.isfinite(float(loss))
    assert len(metrics) == 2
    assert np.all(np.isfinite(np.asarray(metrics)))


def test_train_step_accepts_low_frequency_candidate_scoring():
    graphdef, state = _tiny_reconsiren_split()
    images = jax.random.normal(jax.random.PRNGKey(9), (2, 16, 16, 1))
    labels = jnp.array([0, 1])

    loss, metrics, _, _ = train_step_reconsiren(
        graphdef, state, images, labels, {}, jax.random.PRNGKey(10),
        candidate_scoring_size=8,
        train_heterogeneity=False, return_metrics=True)

    assert np.isfinite(float(loss))
    assert len(metrics) == 2
    assert np.all(np.isfinite(np.asarray(metrics)))


def test_train_step_supports_sampled_assignment():
    graphdef, state = _tiny_reconsiren_split()
    images = jax.random.normal(jax.random.PRNGKey(11), (2, 16, 16, 1))
    labels = jnp.array([0, 1])

    loss, metrics, _, _ = train_step_reconsiren(
        graphdef, state, images, labels, {}, jax.random.PRNGKey(12),
        tau=0.3, assignment_mode="sampled", candidate_scoring_size=8,
        train_heterogeneity=False, return_metrics=True)

    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(np.asarray(metrics)))
