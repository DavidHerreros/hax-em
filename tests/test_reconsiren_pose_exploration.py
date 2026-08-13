import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from hax.utils import candidate_coverage_loss
from hax.networks.reconsiren import (
    ReconSIREN,
    generate_spherical_rotations,
    train_step_reconsiren,
)


def _sphere_directions(n):
    """Fibonacci-lattice directions, as used for the coverage bins."""
    return jnp.asarray(generate_spherical_rotations(n)[:, :, 2], dtype=jnp.float32)


def test_candidate_coverage_penalizes_anchor_quantization():
    dense = _sphere_directions(256)
    anchors = _sphere_directions(16)
    quantized = jnp.repeat(anchors, 16, axis=0)
    bins = _sphere_directions(128)
    empty_bank = jnp.zeros((512, 3), dtype=jnp.float32)
    key = jax.random.PRNGKey(4)

    dense_loss = candidate_coverage_loss(
        dense, bins, empty_bank, 0, key, kappa=32.0,
        bank_samples=128, bank_mix=0.5)
    quantized_loss = candidate_coverage_loss(
        quantized, bins, empty_bank, 0, key, kappa=32.0,
        bank_samples=128, bank_mix=0.5)

    assert float(quantized_loss) > float(dense_loss)
    assert float(dense_loss) >= -1e-6


def test_candidate_coverage_has_finite_direction_gradients():
    directions = _sphere_directions(32)
    bank = _sphere_directions(64)
    bins = _sphere_directions(64)

    def loss_fn(raw_directions):
        unit_directions = raw_directions / jnp.linalg.norm(
            raw_directions, axis=-1, keepdims=True)
        return candidate_coverage_loss(
            unit_directions, bins, bank, bank.shape[0], jax.random.PRNGKey(6),
            kappa=24.0, bank_samples=32, bank_mix=0.5)

    gradients = jax.grad(loss_fn)(directions)
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert float(jnp.linalg.norm(gradients)) > 0.0


def _tiny_reconsiren_split():
    model = ReconSIREN(
        coords=jnp.zeros((4, 3)), values=jnp.full((4,), 0.01),
        xsize=16, sr=1.0, bank_size=32, ctf_type=None,
        num_components=3,
        consensus_parameterization="direct",
        render_chunk_size=0, candidate_chunk_size=0,
        heterogeneity_profile="legacy",
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


def test_train_step_runs_with_hard_assignment():
    graphdef, state = _tiny_reconsiren_split()
    images = jax.random.normal(jax.random.PRNGKey(9), (2, 16, 16, 1))
    labels = jnp.array([0, 1])

    loss, metrics, _, _ = train_step_reconsiren(
        graphdef, state, images, labels, {}, jax.random.PRNGKey(10),
        train_heterogeneity=False, return_metrics=True)

    assert np.isfinite(float(loss))
    assert len(metrics) == 2
    assert np.all(np.isfinite(np.asarray(metrics)))


def test_train_step_supports_stochastic_warm_up():
    graphdef, state = _tiny_reconsiren_split()
    images = jax.random.normal(jax.random.PRNGKey(11), (2, 16, 16, 1))
    labels = jnp.array([0, 1])

    loss, metrics, _, _ = train_step_reconsiren(
        graphdef, state, images, labels, {}, jax.random.PRNGKey(12),
        tau=1e-3, use_tau=True,
        train_heterogeneity=False, return_metrics=True)

    assert np.isfinite(float(loss))
    assert np.all(np.isfinite(np.asarray(metrics)))
