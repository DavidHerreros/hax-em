"""Rigid-gauge fixing for heterogeneity decoders.

A rigid motion of the decoded object is indistinguishable, in the reconstruction
loss, from a refinement of the image pose: rotating the decoded density by ``R``
and leaving the pose alone produces exactly the same projection as leaving the
density alone and refining the pose by ``R``. The pose refinement therefore lives
in a flat direction of the data term, and the decoder is free to absorb it. It
usually will: every geometric prior in these models (distance preservation,
repulsion, ARAP, L1 on the amplitudes) is invariant to a global rigid motion of
the decoded object, so nothing charges the decoder for absorbing the pose, while
the pose head is explicitly anchored to the identity. The cost is not
reconstruction quality but the latent space, which ends up encoding residual
alignment error instead of conformation.

Fixing the gauge -- pinning the decoder's output to one canonical frame -- leaves
the pose head as the only thing that can explain a pose error. How that is done
depends on which field the decoder actually varies per particle:

* **Displacement decoders** (mass transport). The gauge is fixed by rigidly
  superposing the deformed point cloud back onto the rest cloud (a weighted
  Kabsch alignment). Because the correction applied is an *isometry of the
  deformed cloud*, every internal distance, hinge angle and relative domain
  motion survives exactly: what changes is only the frame the motion is
  reported in. (A linear projection onto the 6 rigid displacement fields of the
  *rest* geometry is the small-motion approximation of this, and it is not safe
  here -- it shears large domain motions, because ``omega x c_rest`` stops being
  the right rigid field once the cloud has moved.)

* **Density decoders** (fixed points, learned amplitudes). There is no cloud to
  superpose, so the gauge is fixed in the linearized form, which is the only one
  available: a small rigid motion perturbs the density by
  ``dv_i = -(t + omega x c_i) . grad V(c_i)``, a 6-dimensional linear subspace
  spanned by the reference density's gradients, and that subspace is projected
  out. This needs a reference density to differentiate, so it is unavailable on
  fully ab-initio runs.

In both cases the *frame* is chosen by the weights of the fit, and that is the
knob that decides whether the gauge hides motion:

* Mass (or uniform) weights give the mass-weighted optimal superposition. Under
  a large domain motion this is the frame in which the "stationary" domain
  appears to counter-rotate, and the pose head ends up absorbing a slice of the
  conformational change.
* Robust weights (``irls_iters > 0``) redescend against the residual, so points
  that actually move are pushed out of the fit and the frame is set by the
  least-moving part of the structure -- the rigid core. That is what a pose
  refinement is supposed to mean, and it keeps domain motions in the deformation
  field where they belong.

Cost is a 3x3 (or 6x6) solve per particle plus a couple of matmuls against the
point set, which is negligible next to the per-particle splat the decoder already
runs.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .geometric_losses import _closest_rotation_polar

__all__ = ["RigidGauge", "build_density_rigid_basis", "gauge_displacement_field",
           "gauge_density_field", "gauge_displacement_transform", "apply_gauge_to_points"]

_EPS = 1e-8

# Redescending cutoff, in units of the mean residual magnitude. Points whose
# post-alignment displacement exceeds this are dropped from the frame fit
# entirely, which is what lets a moving domain stop dragging the frame around.
_TUKEY_CUTOFF = 1.0


class RigidGauge:
    """Static holder for the geometry a rigid-gauge fix needs.

    Kept deliberately outside the NNX state (assign it with ``nnx.static``): it is
    a deterministic function of the rest geometry, so it is rebuilt from the saved
    config on reload, and keeping it out of the state means checkpoints written
    before the gauge existed still restore.

    ``mode`` is ``"displacement"`` (mass transport: holds the rest coords) or
    ``"density"`` (fixed points: holds the ``(N, 1, 6)`` linearized basis).
    ``weights`` is the ``(N,)`` mean-normalized base weight of each point.
    """

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
    """Linearized rigid density-perturbation fields on a fixed point set.

    Under a small rigid motion of the object, the density at the (fixed) sample
    points changes by ``dv_i = -(t + omega x c_i) . grad V(c_i)``, so the subspace
    is spanned by the three components of ``-grad V`` and the three components of
    ``-(c_i x grad V(c_i))``.

    ``rest_coords`` is ``(N, 3)`` in the decoder's ``(x, y, z)`` convention (i.e.
    reversed volume axes, as built by the HetSIREN CLI); in the fixed-point modes
    it holds integer voxel indices. ``reference_values`` is the reference density
    sampled at those points.

    Returns ``(N, 1, 6)``, or ``None`` when there is no reference density to
    differentiate -- on a fully ab-initio run there is simply no anchor, and the
    density gauge is unavailable.
    """
    values = np.asarray(reference_values, dtype=np.float64)
    if not np.any(values != 0.0):
        return None

    idx = np.rint(np.asarray(rest_coords, dtype=np.float64)).astype(np.int64)
    volume_size = int(volume_size)
    if (idx < 0).any() or (idx >= volume_size).any():
        return None

    # Decoder coords are (x, y, z) = reversed volume axes.
    ax0, ax1, ax2 = idx[:, 2], idx[:, 1], idx[:, 0]

    volume = np.zeros((volume_size,) * 3, dtype=np.float64)
    np.add.at(volume, (ax0, ax1, ax2), values)

    # A light blur before differencing: the reference is a sparse splat, and raw
    # central differences on it are dominated by the sampling, not by the density.
    if blur_sigma and blur_sigma > 0.0:
        try:
            from scipy.ndimage import gaussian_filter
            volume = gaussian_filter(volume, sigma=float(blur_sigma))
        except ImportError:
            pass

    g_ax0, g_ax1, g_ax2 = np.gradient(volume)
    grad = np.stack([g_ax2[ax0, ax1, ax2],          # d/d coords_x  (= volume axis 2)
                     g_ax1[ax0, ax1, ax2],          # d/d coords_y
                     g_ax0[ax0, ax1, ax2]], axis=1)  # d/d coords_z
    if not np.any(grad != 0.0):
        return None

    centered = np.asarray(rest_coords, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)

    basis = np.concatenate([-grad, -np.cross(centered, grad)], axis=1)   # (N, 6)
    basis = basis[:, None, :]                                            # (N, 1, 6)

    # Unit-norm columns: conditioning only, the span is unchanged.
    norms = np.sqrt((basis ** 2).sum(axis=(0, 1), keepdims=True))
    return (basis / np.maximum(norms, _EPS)).astype(np.float32)


def build_displacement_rigid_basis(rest_coords, weights):
    """Weighted-orthonormal basis of the 6 global rigid modes of a point cloud.

    Returns ``(6, N, 3)``: three unit translations and three infinitesimal rotations
    about the weighted centroid, Gram-Schmidt orthonormalised under the weighted inner
    product ``<a, b>_w = sum_n w_n a_n . b_n``. Degenerate modes (a planar or collinear
    cloud) collapse to zero and simply project nothing out.

    This gauges a *static* consensus correction: a small global rigid motion of the whole
    rest cloud is a flat direction of the reconstruction loss (indistinguishable from
    refining every image pose by the inverse -- the pose<->conformation degeneracy, one
    level up), so left free it drifts and the anchored image-pose head chases it. Projecting
    the learnable Δc0 onto the complement of this basis removes exactly that mode.

    A *linear* projection is used here, unlike the per-particle deformation gauge which must
    use the nonlinear (Kabsch) superposition: the per-particle motion is large, so ``omega x
    c_rest`` stops being the right rigid field once a domain has swung and a linear projection
    would shear it. The consensus correction, by contrast, is small and global (it is removed
    incrementally every step, so it never accumulates a large rotation), so the linearisation
    is accurate -- and, crucially, a constant linear operator has no polar decomposition or
    ``sqrt``-at-zero in its gradient, whereas the Kabsch gauge's gradient is singular at the
    zero-initialised Δc0 and turns the loss into NaN on the first step.
    """
    rest = np.asarray(rest_coords, dtype=np.float64)                     # (N, 3)
    w = np.asarray(weights, dtype=np.float64)                            # (N,)
    n = rest.shape[0]
    mu = (w[:, None] * rest).sum(0) / max(float(w.sum()), 1e-12)
    centered = rest - mu

    raw = []
    for k in range(3):                                                   # translations
        t = np.zeros((n, 3)); t[:, k] = 1.0
        raw.append(t)
    for k in range(3):                                                   # rotations about centroid
        axis = np.zeros(3); axis[k] = 1.0
        raw.append(np.cross(np.broadcast_to(axis, (n, 3)), centered))

    def wdot(a, b):
        return float((w[:, None] * a * b).sum())

    basis = []
    for v in raw:
        for u in basis:
            v = v - wdot(v, u) * u
        norm2 = wdot(v, v)
        basis.append(v / np.sqrt(norm2) if norm2 > 1e-8 else np.zeros_like(v))
    return np.stack(basis).astype(np.float32)                           # (6, N, 3)


def _rigid_fraction(rigid, field):
    """How much of the decoder's field was rigid, as a fraction of the field.

    Guarded: the decoder's coordinate head is zero-initialized, so the field is exactly
    zero at the first step and an unguarded ratio reports numerical dust as a huge
    fraction. Below that floor there is no field to be rigid, so the answer is 0.
    """
    field_norm = jnp.sqrt(jnp.sum(jnp.square(field)))
    rigid_norm = jnp.sqrt(jnp.sum(jnp.square(rigid)))
    return jnp.where(field_norm > 1e-6, rigid_norm / (field_norm + _EPS), 0.0)


def _robust_weights(base_weights, residual):
    """Redescending reweighting: drop the points that move out of the frame fit.

    A Tukey-style cutoff at ``_TUKEY_CUTOFF`` times the mean residual magnitude. A
    bounded reweighting (Cauchy, ``1 / (1 + r^2)``) is not enough here: when half
    the structure swings through a hinge it still leaves the moving half with
    enough weight to drag the frame with it, which is exactly the "the gauge hides
    the motion" failure. Redescending to zero is what pins the frame to the core.

    Detached -- these weights choose the frame, they are not something to optimize
    through.
    """
    magnitude = jnp.sqrt(jnp.sum(residual ** 2, axis=-1) + _EPS)          # (B, N)
    scale = _TUKEY_CUTOFF * jnp.mean(magnitude, axis=-1, keepdims=True) + _EPS
    u = jnp.clip(magnitude / scale, 0.0, 1.0)
    weights = base_weights * jnp.square(1.0 - jnp.square(u))
    return jax.lax.stop_gradient(weights)


def _superpose_transform(deformed, rest, weights):
    """The rigid transform of the weighted superposition, without applying it.

    Returns ``(rotation, mu_deformed, mu_rest)`` with shapes ``(B, 3, 3)``, ``(B, 1, 3)``,
    ``(B, 1, 3)``. Exposed separately from ``_superpose`` so the SAME frame fitted on one
    point set can be applied to another -- which is what lets a field defined on the
    decoder's Gaussians be evaluated, in the identical gauge, at an arbitrary set of query
    points (see ``HetSIREN.decode_field_at``). Refitting the frame on the query points
    instead would put the two fields in different frames.
    """
    w = weights[..., None]                                            # (B, N, 1)
    w_sum = jnp.sum(w, axis=1, keepdims=True) + _EPS                  # (B, 1, 1)

    mu_deformed = jnp.sum(w * deformed, axis=1, keepdims=True) / w_sum
    mu_rest = jnp.sum(w * rest[None], axis=1, keepdims=True) / w_sum

    p = deformed - mu_deformed                                        # (B, N, 3)
    q = rest[None] - mu_rest                                          # (B, N, 3)

    # Kabsch: with H = sum_n w p_n q_n^T = U S V^T, the rotation taking p onto q is
    # V U^T -- which is exactly what the polar iteration returns. It is used here in
    # preference to an SVD not for speed (only one 3x3 per particle reaches this) but
    # for the gradient: the orthogonal polar factor is smooth wherever H is
    # invertible, whereas U and V individually are not when the point cloud's inertia
    # tensor has degenerate singular values -- which is the normal case for a
    # near-isotropic particle, and would make the SVD gradient blow up.
    cross_covariance = jnp.einsum("bni,bnj->bij", w * p, q)           # (B, 3, 3)
    rotation = _closest_rotation_polar(cross_covariance)              # (B, 3, 3)

    return rotation, mu_deformed, mu_rest


def _apply_superposition(deformed, rotation, mu_deformed, mu_rest):
    """Apply a fitted superposition transform to any point set. ``deformed`` is ``(B, M, 3)``."""
    return jnp.einsum("bij,bmj->bmi", rotation, deformed - mu_deformed) + mu_rest


def _superpose(deformed, rest, weights):
    """Weighted rigid superposition of the deformed cloud back onto the rest cloud.

    ``deformed`` ``(B, N, 3)``, ``rest`` ``(N, 3)``, ``weights`` ``(B, N)``.
    Returns the aligned cloud, ``(B, N, 3)``. The correction is a rotation plus a
    translation applied to the whole cloud, i.e. an isometry, so internal geometry
    is preserved exactly.
    """
    return _apply_superposition(deformed, *_superpose_transform(deformed, rest, weights))


def gauge_displacement_field(delta_coords, rest_coords, base_weights, irls_iters=1):
    """Fix the rigid gauge of a deformation field, exactly.

    ``delta_coords`` ``(B, N, 3)`` is the decoder's displacement of the rest points
    (same units as ``rest_coords``). The deformed cloud is rigidly superposed back
    onto the rest cloud and the displacement is recomputed from there, so the
    correction is an isometry and no internal motion is lost -- only the frame
    changes. ``irls_iters`` redescends the fit weights against the residual so the
    frame is set by the rigid core rather than by the mass-weighted compromise.

    Returns ``(gauged_delta_coords, rigid_fraction)``, where ``rigid_fraction`` is
    the size of the removed rigid motion relative to the field -- i.e. how much of
    what the decoder just produced was pose rather than conformation.
    """
    deformed = rest_coords[None] + delta_coords

    # Seed the reweighting from the RAW field, not from a first uniform-weight
    # superposition. The raw displacement magnitude is what says "who moved"; a
    # uniform Kabsch has already smeared the motion over the whole structure
    # (a hinge leaves both domains equally misplaced), and no reweighting can
    # recover the distinction from there.
    weights = jnp.broadcast_to(base_weights[None, :], delta_coords.shape[:2])
    if int(irls_iters) > 0:
        weights = _robust_weights(base_weights[None, :], delta_coords)

    aligned = _superpose(deformed, rest_coords, weights)
    for _ in range(int(irls_iters) - 1):
        weights = _robust_weights(base_weights[None, :], aligned - rest_coords[None])
        aligned = _superpose(deformed, rest_coords, weights)

    gauged = aligned - rest_coords[None]
    return gauged, _rigid_fraction(aligned - deformed, delta_coords)


def gauge_displacement_transform(delta_coords, rest_coords, base_weights, irls_iters=1):
    """The rigid-gauge transform that ``gauge_displacement_field`` would apply.

    Same fit, same robust reweighting, but returns ``(rotation, mu_deformed, mu_rest)``
    instead of the gauged field, so the identical frame can be applied to a different set
    of points via ``apply_gauge_to_points``. This is what makes a dense evaluation of the
    deformation field (at reconstruction voxels, say) live in the same gauge as the sparse
    field the decoder was trained with -- refitting the frame on the dense set would not,
    because the robust weights would then be driven by a different point distribution.
    """
    deformed = rest_coords[None] + delta_coords
    weights = jnp.broadcast_to(base_weights[None, :], delta_coords.shape[:2])
    if int(irls_iters) > 0:
        weights = _robust_weights(base_weights[None, :], delta_coords)

    transform = _superpose_transform(deformed, rest_coords, weights)
    for _ in range(int(irls_iters) - 1):
        aligned = _apply_superposition(deformed, *transform)
        weights = _robust_weights(base_weights[None, :], aligned - rest_coords[None])
        transform = _superpose_transform(deformed, rest_coords, weights)
    return transform


def apply_gauge_to_points(query_rest, query_delta, transform):
    """Gauge a displacement field sampled at ``query_rest`` with a transform fitted elsewhere.

    ``query_rest`` is ``(M, 3)`` (or ``(B, M, 3)``), ``query_delta`` ``(B, M, 3)``. Returns the
    gauged displacement at those points, in the same units.
    """
    rest = query_rest[None] if query_rest.ndim == 2 else query_rest
    aligned = _apply_superposition(rest + query_delta, *transform)
    return aligned - rest


def _fit_rigid_density(field, basis, weights, ridge=1e-6):
    """Weighted least-squares fit of a density field by the linearized rigid basis.

    ``field`` ``(B, N, 1)``, ``basis`` ``(N, 1, 6)``, ``weights`` ``(B, N)``.
    """
    b, n, m = field.shape
    k = basis.shape[-1]

    # Contracting the basis with itself first keeps the Gram an (N, K, K) constant
    # instead of materializing a per-particle (B, N, M, K) tensor.
    gram_points = jnp.einsum("imk,iml->ikl", basis, basis)              # (N, K, K)
    gram = jnp.einsum("bi,ikl->bkl", weights, gram_points)              # (B, K, K)
    rhs = jnp.einsum("bim,imk->bk", weights[..., None] * field, basis)  # (B, K)

    # Ridge scaled to the Gram itself, so a symmetric or degenerate reference (where
    # some rigid directions are barely represented in the density) stays solvable.
    scale = jnp.einsum("bkk->b", gram)[:, None, None] / k
    gram = gram + ridge * scale * jnp.eye(k, dtype=gram.dtype)[None]

    coeffs = jnp.linalg.solve(gram, rhs[..., None])[..., 0]             # (B, K)
    return (coeffs @ basis.reshape(n * m, k).T).reshape(b, n, m)


def density_gauge_magnitude(delta_block):
    """Per-point residual magnitude of a density block, ``(B, C, 1) -> (B, C)``.

    Split out of ``_robust_weights`` so a chunked caller can accumulate the same mean
    over blocks that the whole-cloud path takes in one reduction.
    """
    return jnp.sqrt(jnp.sum(delta_block ** 2, axis=-1) + _EPS)


def density_gauge_scale(magnitude_sum, n_points):
    """The Tukey scale from an accumulated ``sum`` of point magnitudes, ``(B, 1)``."""
    return _TUKEY_CUTOFF * (magnitude_sum / n_points) + _EPS


def density_gauge_block_weights(base_weights_block, delta_block, scale):
    """``_robust_weights`` for one block, given the globally accumulated ``scale``."""
    u = jnp.clip(density_gauge_magnitude(delta_block) / scale, 0.0, 1.0)
    return jax.lax.stop_gradient(base_weights_block * jnp.square(1.0 - jnp.square(u)))


def density_gauge_block_normals(delta_block, basis_block, weights_block):
    """One block's contribution to the ``(B, K, K)`` Gram and ``(B, K)`` RHS."""
    gram_points = jnp.einsum("imk,iml->ikl", basis_block, basis_block)
    gram = jnp.einsum("bi,ikl->bkl", weights_block, gram_points)
    rhs = jnp.einsum("bim,imk->bk", weights_block[..., None] * delta_block, basis_block)
    return gram, rhs


def density_gauge_solve(gram, rhs, ridge=1e-6):
    """Solve the accumulated normal equations for the rigid coefficients, ``(B, K)``.

    Identical ridge policy to ``_fit_rigid_density``, so an accumulated solve and a
    whole-cloud solve differ only by summation order.
    """
    k = gram.shape[-1]
    scale = jnp.einsum("bkk->b", gram)[:, None, None] / k
    gram = gram + ridge * scale * jnp.eye(k, dtype=gram.dtype)[None]
    return jnp.linalg.solve(gram, rhs[..., None])[..., 0]


def density_gauge_apply(coeffs, basis_block):
    """Evaluate the fitted rigid field on one block, ``(B, C, 1)``."""
    n, m, k = basis_block.shape
    return (coeffs @ basis_block.reshape(n * m, k).T).reshape(coeffs.shape[0], n, m)


def rigid_fraction_from_norms(rigid_sq, field_sq):
    """``_rigid_fraction`` from accumulated sums of squares."""
    field_norm = jnp.sqrt(field_sq)
    return jnp.where(field_norm > 1e-6, jnp.sqrt(rigid_sq) / (field_norm + _EPS), 0.0)


def gauge_density_field(delta_values, basis, base_weights, irls_iters=1):
    """Fix the rigid gauge of a density field, to first order.

    ``delta_values`` ``(B, N, 1)`` is the decoder's per-particle density change on
    the fixed points. With no cloud to superpose, the only gauge available is the
    linearized one: project out the 6-dimensional subspace of density perturbations
    that a small rigid motion would produce. That is exactly the regime a pose
    *refinement* lives in.

    Returns ``(gauged_delta_values, rigid_fraction)``.
    """
    # Seeded from the raw field for the same reason as the displacement gauge: the
    # size of the density change is what identifies the parts that are actually
    # changing, and a first unweighted fit would already have spread them out.
    weights = jnp.broadcast_to(base_weights[None, :], delta_values.shape[:2])
    if int(irls_iters) > 0:
        weights = _robust_weights(base_weights[None, :], delta_values)

    rigid = _fit_rigid_density(delta_values, basis, weights)
    for _ in range(int(irls_iters) - 1):
        weights = _robust_weights(base_weights[None, :], delta_values - rigid)
        rigid = _fit_rigid_density(delta_values, basis, weights)

    return delta_values - rigid, _rigid_fraction(rigid, delta_values)
