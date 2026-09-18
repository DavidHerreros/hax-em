"""Rigid-gauge fixing: pin a decoder's per-particle output to a canonical frame so a pose error cannot be absorbed by the decoder."""

import numpy as np
import jax
import jax.numpy as jnp

from .geometric_losses import _closest_rotation_polar

__all__ = ["RigidGauge", "build_density_rigid_basis", "gauge_displacement_field", "gauge_density_field"]

_EPS = 1e-8

# Redescending cutoff (units of the mean residual): moving points drop out of the frame fit
_TUKEY_CUTOFF = 1.0
_WEIGHT_FLOOR = 1e-3


class RigidGauge:
    """Static holder (keep out of NNX state) for the rest geometry a gauge fix needs."""

    def __init__(self, mode=None, rest_coords=None, basis=None, weights=None):
        self.mode = mode
        self.rest_coords = None if rest_coords is None else np.asarray(rest_coords, dtype=np.float32)
        self.basis = None if basis is None else np.asarray(basis, dtype=np.float32)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)

    @property
    def available(self):
        if self.mode == "displacement":
            return self.rest_coords is not None
        if self.mode == "density":
            return self.basis is not None
        return False


def build_density_rigid_basis(rest_coords, reference_values, volume_size, blur_sigma=1.0):
    """Linearized rigid density-perturbation fields, (N, 1, 6), or None without a reference density."""
    values = np.asarray(reference_values, dtype=np.float64)
    if not np.any(values != 0.0):
        return None

    idx = np.rint(np.asarray(rest_coords, dtype=np.float64)).astype(np.int64)
    volume_size = int(volume_size)
    if (idx < 0).any() or (idx >= volume_size).any():
        return None

    # Decoder coords are (x, y, z) = reversed volume axes
    ax0, ax1, ax2 = idx[:, 2], idx[:, 1], idx[:, 0]

    volume = np.zeros((volume_size,) * 3, dtype=np.float64)
    np.add.at(volume, (ax0, ax1, ax2), values)

    if blur_sigma and blur_sigma > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
            volume = gaussian_filter(volume, sigma=float(blur_sigma))
        except ImportError:
            pass

    g_ax0, g_ax1, g_ax2 = np.gradient(volume)
    grad = np.stack([g_ax2[ax0, ax1, ax2], g_ax1[ax0, ax1, ax2], g_ax0[ax0, ax1, ax2]], axis=1)
    if not np.any(grad != 0.0):
        return None

    centered = np.asarray(rest_coords, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)

    basis = np.concatenate([-grad, -np.cross(centered, grad)], axis=1)[:, None, :]   # (N, 1, 6)

    # Unit-norm columns (conditioning only)
    norms = np.sqrt((basis ** 2).sum(axis=(0, 1), keepdims=True))
    return (basis / np.maximum(norms, _EPS)).astype(np.float32)


def _rigid_fraction(rigid, field):
    """Removed rigid motion relative to the field (0 when the field is numerically zero)."""
    field_norm = jnp.sqrt(jnp.sum(jnp.square(field)))
    rigid_norm = jnp.sqrt(jnp.sum(jnp.square(rigid)))
    return jnp.where(field_norm > 1e-6, rigid_norm / (field_norm + _EPS), 0.0)


def _robust_weights(base_weights, residual):
    """Tukey-style redescending reweighting (detached): points that move leave the frame fit."""
    magnitude = jnp.sqrt(jnp.sum(residual ** 2, axis=-1) + _EPS)          # (B, N)
    scale = _TUKEY_CUTOFF * jnp.mean(magnitude, axis=-1, keepdims=True) + _EPS
    u = jnp.clip(magnitude / scale, 0.0, 1.0)
    weights = base_weights * (_WEIGHT_FLOOR + (1.0 - _WEIGHT_FLOOR) * jnp.square(1.0 - jnp.square(u)))
    return jax.lax.stop_gradient(weights)


def _superpose(deformed, rest, weights):
    """Weighted Kabsch superposition of the deformed cloud (B, N, 3) back onto the rest cloud (N, 3)."""
    w = weights[..., None]
    w_sum = jnp.sum(w, axis=1, keepdims=True) + _EPS

    mu_deformed = jnp.sum(w * deformed, axis=1, keepdims=True) / w_sum
    mu_rest = jnp.sum(w * rest[None], axis=1, keepdims=True) / w_sum

    p = deformed - mu_deformed
    q = rest[None] - mu_rest

    # Polar factor instead of SVD: smooth gradient for near-isotropic clouds
    cross_covariance = jnp.einsum("bni,bnj->bij", w * p, q)
    rotation = _closest_rotation_polar(cross_covariance)
    return jnp.einsum("bij,bmj->bmi", rotation, p) + mu_rest


def gauge_displacement_field(delta_coords, rest_coords, base_weights, irls_iters=1):
    """Exact gauge of a displacement field (B, N, 3): returns (gauged field, rigid_fraction)."""
    deformed = rest_coords[None] + delta_coords

    # Reweight from the residual of a plain fit, so a common rigid part never reads as motion
    weights = jnp.broadcast_to(base_weights[None, :], delta_coords.shape[:2])
    aligned = _superpose(deformed, rest_coords, weights)
    for _ in range(int(irls_iters)):
        weights = _robust_weights(base_weights[None, :], aligned - rest_coords[None])
        aligned = _superpose(deformed, rest_coords, weights)

    gauged = aligned - rest_coords[None]
    return gauged, _rigid_fraction(aligned - deformed, delta_coords)


def _fit_rigid_density(field, basis, weights, ridge=1e-6):
    """Weighted least-squares fit of a density field (B, N, 1) by the linearized rigid basis (N, 1, K)."""
    b, n, m = field.shape
    k = basis.shape[-1]

    gram_points = jnp.einsum("imk,iml->ikl", basis, basis)              # (N, K, K)
    gram = jnp.einsum("bi,ikl->bkl", weights, gram_points)              # (B, K, K)
    rhs = jnp.einsum("bim,imk->bk", weights[..., None] * field, basis)  # (B, K)

    # Ridge scaled to the Gram, so a degenerate reference stays solvable
    scale = jnp.einsum("bkk->b", gram)[:, None, None] / k
    gram = gram + ridge * scale * jnp.eye(k, dtype=gram.dtype)[None]

    coeffs = jnp.linalg.solve(gram, rhs[..., None])[..., 0]
    return (coeffs @ basis.reshape(n * m, k).T).reshape(b, n, m)


def gauge_density_field(delta_values, basis, base_weights, irls_iters=1):
    """First-order gauge of a density field (B, N, 1): returns (gauged field, rigid_fraction)."""
    weights = jnp.broadcast_to(base_weights[None, :], delta_values.shape[:2])
    if int(irls_iters) > 0:
        weights = _robust_weights(base_weights[None, :], delta_values)

    rigid = _fit_rigid_density(delta_values, basis, weights)
    for _ in range(int(irls_iters) - 1):
        weights = _robust_weights(base_weights[None, :], delta_values - rigid)
        rigid = _fit_rigid_density(delta_values, basis, weights)

    return delta_values - rigid, _rigid_fraction(rigid, delta_values)
