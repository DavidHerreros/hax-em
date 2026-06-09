from __future__ import annotations

import sys
from functools import partial
from typing import Optional, Tuple
from contextlib import closing

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax

from hax.generators import MetaDataGenerator, extract_columns
from hax.utils import *

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


class GaussianVolumeModel(nnx.Module):
    """Sum of isotropic 3D Gaussians sharing a single learnable sigma.

    The density at voxel ``v`` is
        sum_i  A_i * exp( -|| v - mu_i ||^2 / (2 * sigma^2) )
    where positions ``mu_i`` are stored in voxel coordinates.

    Positive quantities (amplitudes, sigma) are reparameterised in log-space
    so the optimisation is unconstrained.
    """

    @save_config
    def __init__(
        self,
        positions: jnp.ndarray,
        amplitudes: jnp.ndarray,
        sigma: float,
        positivity_mode: str = "relu",
        *,
        rngs: nnx.Rngs,
    ):
        positions = jnp.asarray(positions, dtype=jnp.float32)
        amplitudes = jnp.asarray(amplitudes, dtype=jnp.float32)

        self.positions = nnx.Param(positions)

        if positivity_mode == "relu":
            self.positivity_fn = nnx.relu

            self.log_amplitudes = nnx.Param(
                jnp.maximum(amplitudes, 1e-8)
            )
            self.log_sigma = nnx.Param(
                jnp.asarray(float(sigma), dtype=jnp.float32)
            )

            self.log_a = nnx.Param(1.0)
            self.b = nnx.Param(0.0)

        elif positivity_mode == "exp":
            self.positivity_fn = jnp.exp

            self.log_amplitudes = nnx.Param(
                jnp.log(jnp.maximum(amplitudes, 1e-8))
            )
            self.log_sigma = nnx.Param(
                jnp.asarray(np.log(float(sigma)), dtype=jnp.float32)
            )

            self.log_a = nnx.Param(0.0)
            self.b = nnx.Param(0.0)

        else:
            raise ValueError("Positivity mode not recognized")


    # ----- accessors -----
    def get_positions(self) -> jnp.ndarray:
        """Gaussian centers in voxel coordinates, shape (N, 3)."""
        return self.positions.get_value()

    def get_amplitudes(self) -> jnp.ndarray:
        """Per-Gaussian positive amplitudes, shape (N,)."""
        amplitudes = nnx.relu(self.get_global_scale() * self.positivity_fn(self.log_amplitudes.get_value()) + self.get_global_bias())
        return amplitudes

    def get_sigma(self) -> jnp.ndarray:
        """Shared positive isotropic sigma (scalar, voxels)."""
        return self.positivity_fn(self.log_sigma.get_value())

    def num_gaussians(self) -> int:
        return int(self.positions.get_value().shape[0])

    def get_global_scale(self):
        return self.positivity_fn(self.log_a.get_value())

    def get_global_bias(self):
        return self.b.get_value()

    # ----- rendering -----
    def render(
        self,
        grid_shape: Tuple[int, int, int],
        kernel_radius: Optional[int] = None,
    ) -> jnp.ndarray:
        """Render the Gaussian mixture onto a 3D voxel grid.

        ``kernel_radius`` controls the side of the local cube where each
        Gaussian is evaluated. If ``None`` it is set to ~3.5 * sigma so we
        capture > 99.95% of every Gaussian's mass.
        """
        if kernel_radius is None:
            kernel_radius = max(
                2, int(np.ceil(3.5 * float(self.get_sigma())))
            )
        return _render_gaussians(
            self.get_positions(),
            self.get_amplitudes(),
            self.get_sigma(),
            tuple(int(s) for s in grid_shape),
            int(kernel_radius),
        )

    def render_projections(
            self,
            rotations,
            shifts,
            grid_shape: Tuple[int, int, int],
            image_shape: Optional[Tuple[int, int]] = None,
            projection_axis: int = 2,
            kernel_radius: Optional[int] = None,
    ) -> jnp.ndarray:
        return render_gaussian_projections(
            self.get_positions(),
            self.get_amplitudes(),
            self.get_sigma(),
            rotations,
            shifts,
            image_shape=image_shape,
            grid_shape=grid_shape,
            projection_axis=projection_axis,
            kernel_radius=kernel_radius,
        )


@partial(jax.jit, static_argnums=(3, 4))
def _render_gaussians(
    positions: jnp.ndarray,
    amplitudes: jnp.ndarray,
    sigma: jnp.ndarray,
    grid_shape: Tuple[int, int, int],
    kernel_radius: int,
) -> jnp.ndarray:
    """Render isotropic Gaussians on a 3D grid via local-cube scatter-add.

    Each Gaussian contributes only to a cube of side (2*kernel_radius + 1)
    around its nearest voxel. Contributions that fall outside the grid are
    silently dropped, so Gaussians whose centers drift out of the box stop
    influencing the loss --- they then show up in the prune step.
    """
    # No Gaussians: return zeros (static shape check so this works under jit).
    if positions.shape[0] == 0:
        return jnp.zeros(grid_shape, dtype=jnp.float32)

    centers_int = jnp.floor(positions + 0.5).astype(jnp.int32)            # (N, 3)
    frac = positions - centers_int.astype(positions.dtype)                # (N, 3)

    axis = jnp.arange(-kernel_radius, kernel_radius + 1)
    dx, dy, dz = jnp.meshgrid(axis, axis, axis, indexing="ij")
    patch_off = jnp.stack([dx, dy, dz], axis=-1).astype(positions.dtype)  # (k,k,k,3)

    # distance from each patch voxel center to the Gaussian's exact center
    diff = patch_off[None] - frac[:, None, None, None, :]                 # (N,k,k,k,3)
    dist_sq = jnp.sum(diff * diff, axis=-1)                               # (N,k,k,k)
    values = (
        amplitudes[:, None, None, None]
        * jnp.exp(-dist_sq / (2.0 * sigma * sigma))
    )

    global_coords = (
        centers_int[:, None, None, None, :]
        + patch_off[None].astype(jnp.int32)
    )                                                                     # (N,k,k,k,3)

    coords_flat = global_coords.reshape(-1, 3)
    values_flat = values.reshape(-1)

    gs = jnp.array(grid_shape, dtype=jnp.int32)
    in_bounds = jnp.all((coords_flat >= 0) & (coords_flat < gs), axis=-1)
    values_flat = jnp.where(in_bounds, values_flat, 0.0)
    coords_flat = jnp.clip(coords_flat, 0, gs - 1)

    out = jnp.zeros(grid_shape, dtype=positions.dtype)
    out = out.at[coords_flat[:, 0],
                 coords_flat[:, 1],
                 coords_flat[:, 2]].add(values_flat)
    return out


def render_gaussian_volume(
    positions,
    amplitudes,
    sigma,
    grid_shape: Tuple[int, int, int],
    kernel_radius: Optional[int] = None,
) -> jnp.ndarray:
    """Public utility: render arbitrary Gaussian parameters onto a 3D grid."""
    positions = jnp.asarray(positions, dtype=jnp.float32)
    amplitudes = jnp.asarray(amplitudes, dtype=jnp.float32)
    sigma_arr = jnp.asarray(float(sigma), dtype=jnp.float32)
    if kernel_radius is None:
        kernel_radius = max(2, int(np.ceil(3.5 * float(sigma))))
    return _render_gaussians(
        positions, amplitudes, sigma_arr,
        tuple(int(s) for s in grid_shape), int(kernel_radius),
    )


@partial(jax.jit, static_argnums=(4, 5, 6))
def _render_gaussian_projections(
        positions: jnp.ndarray,  # (N, 3)
        amplitudes: jnp.ndarray,  # (N,)
        sigma: jnp.ndarray,  # scalar
        rotations_shifts: Tuple[jnp.ndarray, jnp.ndarray],
        image_shape: Tuple[int, int],
        kernel_radius: int,
        projection_axis: int,
) -> jnp.ndarray:
    """Render batched 2D projections via local-cube scatter-add.

    See ``render_gaussian_projections`` for the public, validated entry point.
    Returns ``(B, H, W)``. Projections falling outside the image are
    silently dropped (out-of-frame Gaussians contribute zero).
    """
    rotations, shifts = rotations_shifts  # (B,3,3) (B,3)
    B = rotations.shape[0]
    H, W = image_shape
    N = positions.shape[0]

    if N == 0:
        return jnp.zeros((B, H, W), dtype=jnp.float32)

    # Transform coordinated
    factor = 0.5 * image_shape[0]
    plane_axes = tuple(a for a in (0, 1, 2) if a != projection_axis)
    positions = jnp.stack((positions[..., 2], positions[..., 1], positions[..., 0]), axis=-1)
    positions = (positions - factor) / factor
    centers_2d = jnp.einsum("bij,nj->bni", rotations, positions)[..., list(plane_axes)]
    centers_2d = (factor * centers_2d + factor) - shifts[:, None, :]
    centers_2d = jnp.stack((centers_2d[..., 1], centers_2d[..., 0]), axis=-1)

    # 2D amplitude after analytic integration along the projection axis.
    sqrt_2pi = jnp.sqrt(jnp.asarray(2.0 * np.pi, dtype=positions.dtype))
    amp2d = amplitudes * sigma * sqrt_2pi  # (N,)

    # Local-cube splat per (batch, gaussian)
    centers_int = jnp.floor(centers_2d + 0.5).astype(jnp.int32)  # (B,N,2)
    frac = centers_2d - centers_int.astype(centers_2d.dtype)  # (B,N,2)

    axis = jnp.arange(-kernel_radius, kernel_radius + 1)
    du, dv = jnp.meshgrid(axis, axis, indexing="ij")
    patch_off = jnp.stack([du, dv], axis=-1).astype(positions.dtype)  # (k,k,2)

    # distance from each patch pixel to the exact center (no broadcast over B yet)
    diff = patch_off[None, None] - frac[:, :, None, None, :]  # (B,N,k,k,2)
    dist_sq = jnp.sum(diff * diff, axis=-1)  # (B,N,k,k)
    values = (
            amp2d[None, :, None, None]
            * jnp.exp(-dist_sq / (2.0 * sigma * sigma))
    )  # (B,N,k,k)

    global_coords = (
            centers_int[:, :, None, None, :]
            + patch_off[None, None].astype(jnp.int32)
    )  # (B,N,k,k,2)

    # Add a batch index channel so each (b, u, v) lands in image b.
    batch_idx = jnp.broadcast_to(
        jnp.arange(B, dtype=jnp.int32)[:, None, None, None],
        (B, N, 2 * kernel_radius + 1, 2 * kernel_radius + 1),
    )  # (B,N,k,k)

    coords_flat = global_coords.reshape(-1, 2)  # (P,2)
    values_flat = values.reshape(-1)  # (P,)
    batch_flat = batch_idx.reshape(-1)  # (P,)

    img_shape_arr = jnp.array(image_shape, dtype=jnp.int32)
    in_bounds = jnp.all(
        (coords_flat >= 0) & (coords_flat < img_shape_arr),
        axis=-1,
    )
    values_flat = jnp.where(in_bounds, values_flat, 0.0)
    coords_flat = jnp.clip(coords_flat, 0, img_shape_arr - 1)

    out = jnp.zeros((B, H, W), dtype=positions.dtype)
    out = out.at[batch_flat, coords_flat[:, 0], coords_flat[:, 1]].add(values_flat)
    return out


def render_gaussian_projections(
        positions,
        amplitudes,
        sigma,
        rotations,
        shifts,
        image_shape: Optional[Tuple[int, int]] = None,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        projection_axis: int = 2,
        kernel_radius: Optional[int] = None,
) -> jnp.ndarray:
    """Render batched 2D projections of a Gaussian volume.

    For each batch element ``b``, every Gaussian center ``mu_i`` is mapped
    to ``R_b @ mu_i + t_b`` (after optional centering, see below), then
    projected by dropping ``projection_axis``. Because the 3D Gaussians
    are isotropic, the line integral is exact:

        projected_amp = amp * sigma * sqrt(2 * pi)

    so the result is a true continuous-domain projection (no slab
    integration error), sampled onto the requested image grid.

    Parameters
    ----------
    positions : (N, 3) array
        Gaussian centers in voxel coordinates (axis0, axis1, axis2 order).
    amplitudes : (N,) array
        Per-Gaussian 3D amplitudes (the same ``A`` used by the volume
        renderer). The 2D amplitudes are computed internally.
    sigma : float
        Shared isotropic Gaussian sigma (voxels).
    rotations : (B, 3, 3) array
        Per-image rotation matrices applied to the (centered) coordinates.
        No orthogonality check is performed.
    shifts : (B, 3) array
        Per-image translations applied **after** rotation, in voxels, in
        the same (axis0, axis1, axis2) order as ``positions``. The
        component along ``projection_axis`` is ignored (it has no
        effect on the projection).
    image_shape : (H, W) tuple, optional
        Output image size. If None, taken from ``grid_shape`` by dropping
        ``projection_axis``.
    grid_shape : (D0, D1, D2) tuple, optional
        Shape of the original 3D grid the model was fitted on. Used to
        derive ``image_shape`` and the rotation center when those are
        not given. Required if ``image_shape`` is None.
    projection_axis : int, default 2
        Which axis to integrate along (0, 1, or 2). The two remaining
        axes become the image axes in order, i.e. for axis=2 the image
        is (axis0, axis1); for axis=0 the image is (axis1, axis2); etc.
    kernel_radius : int, optional
        Half-side of the local pixel square used per Gaussian. Defaults
        to ``ceil(3.5 * sigma)``.

    Returns
    -------
    images : (B, H, W) jnp.ndarray
        Stack of 2D projections.
    """
    positions = jnp.asarray(positions, dtype=jnp.float32)
    amplitudes = jnp.asarray(amplitudes, dtype=jnp.float32)
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float32)
    rotations = jnp.asarray(rotations, dtype=jnp.float32)
    shifts = jnp.asarray(shifts, dtype=jnp.float32)

    assert rotations.ndim == 3 and rotations.shape[1:] == (3, 3), \
        "rotations must have shape (B, 3, 3)"
    assert shifts.ndim == 2 and shifts.shape[1] == 2, \
        "shifts must have shape (B, 2)"
    assert rotations.shape[0] == shifts.shape[0], \
        "rotations and shifts must share the same batch dimension"
    assert projection_axis in (0, 1, 2), "projection_axis must be 0, 1, or 2"

    if image_shape is None:
        if grid_shape is None:
            raise ValueError(
                "either image_shape or grid_shape must be provided"
            )
        gs = tuple(int(s) for s in grid_shape)
        image_shape = tuple(s for i, s in enumerate(gs) if i != projection_axis)
    image_shape = (int(image_shape[0]), int(image_shape[1]))

    if kernel_radius is None:
        kernel_radius = max(2, int(np.ceil(3.5 * sigma_arr)))

    return _render_gaussian_projections(
        positions,
        amplitudes,
        sigma_arr,
        (rotations, shifts),
        image_shape,
        int(kernel_radius),
        int(projection_axis),
    )


def _estimate_background(
    volume: np.ndarray, mask: Optional[np.ndarray]
) -> Tuple[float, float]:
    """Estimate background mean and standard deviation."""
    if mask is not None:
        bg = volume[~mask]
        if bg.size < 8:
            bg = volume
    else:
        # without a mask, take the lower half of the histogram as "background"
        thr = np.quantile(volume, 0.5)
        bg = volume[volume < thr]
        if bg.size < 8:
            bg = volume
    return float(np.mean(bg)), float(np.std(bg))


def _sample_seed_positions(
    rng: np.random.Generator,
    volume: np.ndarray,
    mask: Optional[np.ndarray],
    noise_threshold: float,
    n_samples: int,
) -> np.ndarray:
    """Sample voxel-coordinate seed positions inside the mask (if given),
    weighted by the local volume value above the noise threshold so we
    naturally concentrate Gaussians on signal and not background."""
    candidate = mask if mask is not None else (volume > noise_threshold)
    if not candidate.any():
        candidate = np.ones_like(volume, dtype=bool)

    idxs = np.argwhere(candidate)                                      # (M, 3)
    weights = np.maximum(volume[candidate] - noise_threshold, 0.0).astype(np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()

    chosen = rng.choice(idxs.shape[0], size=n_samples, replace=True, p=weights)
    positions = idxs[chosen].astype(np.float32)
    # sub-voxel jitter so coincident seeds split apart during optimisation
    positions += rng.normal(scale=0.5, size=positions.shape).astype(np.float32)
    return positions


def _is_inside(
    positions: np.ndarray,
    grid_shape: Tuple[int, int, int],
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Boolean: True where positions sit inside the grid (and mask, if any)."""
    gs = np.asarray(grid_shape)
    inside = np.all((positions >= 0) & (positions <= gs - 1), axis=-1)
    if mask is not None:
        rounded = np.clip(np.round(positions).astype(int), 0, gs - 1)
        inside &= mask[rounded[:, 0], rounded[:, 1], rounded[:, 2]]
    return inside


def _merge_close_gaussians(
    positions: np.ndarray,
    amplitudes: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Greedy non-maximum suppression on Gaussian centers.

    For every Gaussian (visited in order of decreasing amplitude) drop any
    other Gaussian whose center lies within ``radius`` voxels of it. Used
    after the amplitude prune to remove redundant Gaussians that have
    collapsed onto the same density peak.

    Returns a boolean ``keep`` mask of shape (N,).
    """
    n = positions.shape[0]
    keep = np.ones(n, dtype=bool)
    if n < 2 or radius <= 0:
        return keep
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return keep                                            # silently skip

    tree = cKDTree(positions)
    order = np.argsort(-amplitudes)
    for idx in order:
        if not keep[idx]:
            continue
        for nb in tree.query_ball_point(positions[idx], r=float(radius)):
            if nb != idx and keep[nb]:
                keep[nb] = False
    return keep


def fit_gaussian_splat(
    volume,
    mask=None,
    *,
    initial_sigma: float = 1.5,
    min_sigma: float = 0.8,
    initial_num_gaussians: Optional[int] = None,
    max_num_gaussians: int = 200_000,
    learning_rate: float = 5e-3,
    max_iterations: int = 5_000,
    min_iterations: int = 400,
    densify_every: int = 100,
    prune_every: int = 100,
    densify_until: Optional[int] = None,
    densify_fraction: float = 0.05,
    max_new_per_densify: Optional[int] = None,
    prune_amplitude_quantile: float = 0.02,
    noise_amp_multiplier: float = 1.0,
    merge_radius_factor: float = 0.5,
    noise_std_multiplier: float = 3.0,
    noise_floor_multiplier: float = 1.0,
    improvement_tol: float = 0.01,
    improvement_patience: int = 3,
    loss_degradation_tol: float = 0.05,
    convergence_window: int = 300,
    convergence_tol: float = 5e-4,
    seed: int = 0,
    verbose: bool = True,
    quiet: bool = False,
) -> GaussianVolumeModel:
    """Fit a sum of isotropic 3D Gaussians to a CryoEM volume.

    All Gaussians share a single learnable isotropic sigma. The function
    automatically discovers the minimum number of Gaussians, their positions,
    amplitudes and the shared sigma, and stops when convergence is detected.

    Parameters
    ----------
    volume : (D, H, W) array
        CryoEM density map.
    mask : (D, H, W) array, optional
        Soft or binary mask delimiting the protein signal. Strongly
        recommended for noisy maps -- it is used both for seeding and as a
        region constraint: any Gaussian whose center drifts outside the mask
        is pruned.
    initial_sigma : float
        Starting value (voxels) for the shared isotropic sigma.
    min_sigma : float
        Lower bound on sigma (voxels). Enforced via projected gradient after
        every optimiser step. Prevents the model from degenerating into
        sub-voxel "delta-like" Gaussians, which is a known failure mode of
        3D-GS-style fitting when the optimiser has too much capacity.
    initial_num_gaussians : int, optional
        Number of Gaussians to seed. If None, derived from the mask volume
        and ``initial_sigma`` (deliberately undershooting so the densify
        loop grows into the answer).
    max_num_gaussians : int
        Hard cap to prevent runaway densification.
    learning_rate : float
        Adam learning rate (positions, log_amplitudes, log_sigma).
    max_iterations, min_iterations : int
        Bounds on the optimisation loop.
    densify_every, prune_every : int
        Frequency (in iterations) of densify / prune passes.
    densify_until : int, optional
        Stop densifying after this iteration. Defaults to 80 % of
        ``max_iterations`` so the model has room to settle into refinement.
    densify_fraction : float
        Per-pass densification rate as a fraction of the current count
        (kept for backward compatibility, capped by ``max_new_per_densify``).
    max_new_per_densify : int, optional
        Absolute upper bound on Gaussians added per densify pass. Critical
        for stability -- prevents the multiplicative growth that turns
        ``densify_fraction`` into exponential blowup. Defaults to
        ``max(64, 5 % of initial_num_gaussians)``.
    prune_amplitude_quantile : float
        Fraction of Gaussians (sorted by amplitude) eligible for quantile
        pruning. Only applied while densification is active -- after freeze,
        only the absolute noise-relative threshold applies, which prevents
        the prune from slowly destroying the model in the refinement phase.
    noise_amp_multiplier : float
        Absolute amplitude prune threshold = ``noise_amp_multiplier * bg_std``.
        Gaussians whose amplitudes drop below this contribute density below
        the background noise level and are pruned. Unlike the quantile
        rule this is a physically-meaningful absolute threshold that
        stabilises naturally.
    merge_radius_factor : float
        After the amplitude prune, run a greedy non-maximum suppression that
        drops any Gaussian within ``merge_radius_factor * sigma`` of a
        higher-amplitude one. Set to 0 to disable.
    noise_std_multiplier : float
        Signal threshold is ``bg_mean + multiplier * bg_std``. This is what
        makes densification ignore background fluctuations.
    noise_floor_multiplier : float
        Freeze densification (no more new Gaussians) once
        ``sqrt(MSE) < noise_floor_multiplier * bg_std``. At that point we
        are fitting at the noise level and adding capacity would only
        overfit. Set to 0 to disable.
    improvement_tol : float
        Relative loss improvement required between consecutive densify
        passes to consider the pass productive (default 1 %).
    improvement_patience : int
        After this many consecutive unproductive densify passes,
        densification is frozen permanently.
    loss_degradation_tol : float
        After densification freezes, the loss at that moment is recorded.
        If subsequent steps push the loss above
        ``loss_at_freeze * (1 + loss_degradation_tol)`` the run stops
        immediately (defends against destructive over-pruning).
    convergence_window : int
        Sliding window (iterations) over which the loss range is checked.
    convergence_tol : float
        If (max - min) / mean of the loss inside the window drops below this
        AND no topology change has happened inside the window, we declare
        convergence.
    seed : int
        RNG seed.
    verbose : bool
        Print progress. Defaults to True. Set to False to silence everything,
        including the progress bar.
    quiet : bool
        Print *only* the tqdm progress bar with its live postfix metrics
        (loss, N, sigma, phase). Suppresses the [init] header, all event
        lines (densify/prune/merge/freeze/convergence/stop) and the [done]
        summary. Has no effect if ``verbose=False``. Useful in notebooks
        and CLIs where you want a clean single-line live indicator.

    Returns
    -------
    GaussianVolumeModel
        Trained Flax NNX module containing the fitted parameters.
    """
    # ---- prep -------------------------------------------------------------
    volume_np = np.asarray(volume, dtype=np.float32)
    assert volume_np.ndim == 3, "volume must be a 3D array"
    grid_shape = tuple(int(s) for s in volume_np.shape)

    if mask is not None:
        mask_np = np.asarray(mask) > 0.5
        assert mask_np.shape == volume_np.shape, "mask must match volume shape"
    else:
        mask_np = None

    target_volume = jnp.asarray(volume_np)

    # ---- background / noise threshold ------------------------------------
    bg_mean, bg_std = _estimate_background(volume_np, mask_np)
    if bg_std <= 0:
        bg_std = float(np.std(volume_np)) + 1e-6
    noise_threshold = bg_mean + noise_std_multiplier * bg_std
    noise_floor_mse = (noise_floor_multiplier * bg_std) ** 2

    if verbose and not quiet:
        print(f"[init] background mean={bg_mean:.4g}  std={bg_std:.4g}  "
              f"signal_threshold={noise_threshold:.4g}")
        if noise_floor_multiplier > 0:
            print(f"[init] densification freezes when MSE < {noise_floor_mse:.4g} "
                  f"(noise-floor MSE)")

    # ---- initial gaussian count ------------------------------------------
    rng_np = np.random.default_rng(seed)

    if initial_num_gaussians is None:
        if mask_np is not None:
            sig_vox = int(mask_np.sum())
        else:
            sig_vox = int((volume_np > noise_threshold).sum())
        voxels_per_g = max(1, int((2 * initial_sigma) ** 3))
        initial_num_gaussians = max(64, sig_vox // voxels_per_g // 4)
        initial_num_gaussians = min(initial_num_gaussians, max_num_gaussians // 4)
    initial_num_gaussians = int(initial_num_gaussians)

    # Absolute densify cap (linear growth instead of multiplicative).
    if max_new_per_densify is None:
        max_new_per_densify = max(64, int(0.05 * initial_num_gaussians))
    max_new_per_densify = int(max_new_per_densify)

    init_positions = _sample_seed_positions(
        rng_np, volume_np, mask_np, noise_threshold, initial_num_gaussians,
    )
    rounded = np.clip(np.round(init_positions).astype(int), 0,
                      np.asarray(grid_shape) - 1)
    init_amplitudes = np.maximum(
        volume_np[rounded[:, 0], rounded[:, 1], rounded[:, 2]] - bg_mean,
        bg_std,
    ).astype(np.float32)

    if verbose and not quiet:
        print(f"[init] seeding {initial_num_gaussians} Gaussians, "
              f"sigma={initial_sigma}, min_sigma={min_sigma}, "
              f"max_new_per_densify={max_new_per_densify}")

    # ---- model + optimiser ------------------------------------------------
    if initial_sigma < min_sigma:
        initial_sigma = float(min_sigma)
    log_min_sigma = float(np.log(min_sigma))

    model = GaussianVolumeModel(
        positions=jnp.asarray(init_positions),
        amplitudes=jnp.asarray(init_amplitudes),
        sigma=float(initial_sigma),
        rngs=nnx.Rngs(seed),
    )

    def _build_tx():
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate),
        )

    params_filter = nnx.All(nnx.Param, (nnx.PathContains('positions'), nnx.PathContains('log_amplitudes'),
                                               nnx.PathContains('log_sigma')))
    optimizer = nnx.Optimizer(model, _build_tx(), wrt=params_filter)

    # ---- jitted train step ------------------------------------------------
    @partial(nnx.jit, static_argnames=("grid_shape", "kernel_radius"))
    def train_step(model, optimizer, target, grid_shape, kernel_radius):
        def loss_fn(m):
            rendered = m.render(grid_shape, kernel_radius)
            return jnp.mean((rendered - target) ** 2)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        grads, _ = grads.split(params_filter, ...)
        optimizer.update(model, grads)
        # Projected gradient: clamp log_sigma to satisfy sigma >= min_sigma.
        model.log_sigma.value = jnp.maximum(model.log_sigma.value, log_min_sigma)
        return loss

    @partial(nnx.jit, static_argnames=("grid_shape", "kernel_radius"))
    def render_jit(model, grid_shape, kernel_radius):
        return model.render(grid_shape, kernel_radius)

    if densify_until is None:
        densify_until = int(0.8 * max_iterations)

    def _kernel_radius_for(sigma_val: float) -> int:
        return max(2, int(np.ceil(3.5 * sigma_val)))

    kernel_radius = _kernel_radius_for(initial_sigma)

    # ---- main loop --------------------------------------------------------
    losses: list = []
    last_topology_change = 0
    n_history: list = []                       # (step, N) for growth check
    best_loss_for_densify = float("inf")
    best_overall_loss = float("inf")           # all-time minimum
    no_improve_count = 0
    densify_frozen = False
    freeze_reason = ""
    freeze_step: Optional[int] = None          # step at which freeze fired
    converged = False

    # ---- progress bar / logger --------------------------------------------
    # Event lines (densify / prune / merge / freeze / stop) go through ``say``
    # which routes through tqdm.write when a bar is active so the bar stays
    # at the bottom of the terminal. The per-step status (loss, N, sigma,
    # phase) lives in the bar's postfix instead of being re-printed every
    # 100 steps.
    use_bar = verbose and _HAS_TQDM
    pbar = (
        tqdm(
            total=max_iterations,
            file=sys.stdout,
            ascii=" >=",
            colour="green",
            bar_format=(
                "{l_bar}{bar:24}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
            dynamic_ncols=True,
            leave=True,
        )
        if use_bar else None
    )

    def say(msg: str) -> None:
        if not verbose or quiet:
            return
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg)

    def update_postfix() -> None:
        if pbar is None:
            return
        cur_sigma = float(model.get_sigma())
        cur_n = model.num_gaussians()
        if densify_frozen:
            phase = "frozen"
        elif step >= densify_until:
            phase = "refine"
        else:
            phase = "grow"
        # Inline metric line; tqdm renders this after the bar.
        pbar.set_postfix_str(
            f"loss={loss_val:.3e} | N={cur_n:>5d} | "
            f"σ={cur_sigma:.3f} | {phase}",
            refresh=False,
        )

    print(f"\n{bcolors.OKCYAN}###### Starting Adaptive Grid Fit on {grid_shape} volume... ######{bcolors.ENDC}")

    for step in range(max_iterations):
        loss_val = float(train_step(
            model, optimizer, target_volume, grid_shape, kernel_radius,
        ))
        losses.append(loss_val)
        if loss_val < best_overall_loss:
            best_overall_loss = loss_val

        new_kr = _kernel_radius_for(float(model.get_sigma()))
        if new_kr != kernel_radius:
            kernel_radius = new_kr

        # ---- noise-floor stop ----------------------------------------
        if (not densify_frozen
                and noise_floor_multiplier > 0
                and loss_val < noise_floor_mse):
            densify_frozen = True
            freeze_step = step
            freeze_reason = (f"reached noise floor "
                             f"(MSE={loss_val:.3g} < {noise_floor_mse:.3g})")
            say(f"[step {step+1:5d}] freeze densify: {freeze_reason}")

        do_densify = (
            (step + 1) % densify_every == 0
            and step < densify_until
            and not densify_frozen
            and model.num_gaussians() < max_num_gaussians
        )
        do_prune = ((step + 1) % prune_every == 0)

        if do_densify or do_prune:
            cur_pos = np.asarray(model.get_positions())
            cur_amp = np.asarray(model.get_amplitudes())
            cur_sigma = float(model.get_sigma())
            topology_changed = False

            # ---- PRUNE: out-of-box, out-of-mask, low amplitude ----
            if do_prune and cur_pos.shape[0] > 16:
                keep = _is_inside(cur_pos, grid_shape, mask_np)
                # Absolute noise-relative threshold: Gaussians below this
                # contribute density at or below the background noise level.
                amp_thresh = noise_amp_multiplier * bg_std
                # Add the quantile rule on top only while densification is
                # still active. After freeze, applying it forever would
                # slowly eat into useful Gaussians (long-tail destruction).
                if not densify_frozen and prune_amplitude_quantile > 0:
                    q_thresh = float(
                        np.quantile(cur_amp, prune_amplitude_quantile)
                    )
                    amp_thresh = max(amp_thresh, q_thresh)
                keep &= cur_amp > amp_thresh
                if keep.sum() < cur_pos.shape[0]:
                    say(f"[step {step+1:5d}] prune:   "
                        f"{cur_pos.shape[0]:>5d} → {int(keep.sum()):>5d}")
                    cur_pos = cur_pos[keep]
                    cur_amp = cur_amp[keep]
                    topology_changed = True

                # ---- MERGE: drop near-duplicate Gaussians ----
                if merge_radius_factor > 0 and cur_pos.shape[0] > 1:
                    merge_keep = _merge_close_gaussians(
                        cur_pos, cur_amp,
                        radius=merge_radius_factor * cur_sigma,
                    )
                    if merge_keep.sum() < cur_pos.shape[0]:
                        say(f"[step {step+1:5d}] merge:   "
                            f"{cur_pos.shape[0]:>5d} → "
                            f"{int(merge_keep.sum()):>5d}")
                        cur_pos = cur_pos[merge_keep]
                        cur_amp = cur_amp[merge_keep]
                        topology_changed = True

            # ---- DENSIFY: add Gaussians where residual exceeds noise ----
            if do_densify and cur_pos.shape[0] < max_num_gaussians:
                tmp_model = GaussianVolumeModel(
                    positions=jnp.asarray(cur_pos),
                    amplitudes=jnp.asarray(cur_amp),
                    sigma=cur_sigma,
                    rngs=nnx.Rngs(seed + step),
                )
                rendered_np = np.asarray(
                    render_jit(tmp_model, grid_shape, kernel_radius)
                )
                residual = volume_np - rendered_np
                cand = residual > noise_threshold
                if mask_np is not None:
                    cand &= mask_np
                if cand.any():
                    # absolute cap dominates -- prevents multiplicative blowup
                    n_new = int(densify_fraction * max(cur_pos.shape[0], 1))
                    n_new = min(n_new, max_new_per_densify)
                    n_new = max(n_new, 8)
                    n_new = min(n_new, max_num_gaussians - cur_pos.shape[0])
                    if n_new > 0:
                        cand_idxs = np.argwhere(cand)
                        w = np.maximum(residual[cand], 0.0).astype(np.float64)
                        if w.sum() > 0:
                            w /= w.sum()
                            chosen = rng_np.choice(
                                cand_idxs.shape[0], size=n_new,
                                replace=True, p=w,
                            )
                            new_pos = cand_idxs[chosen].astype(np.float32)
                            new_pos += rng_np.normal(
                                scale=0.5, size=new_pos.shape
                            ).astype(np.float32)
                            r = np.clip(np.round(new_pos).astype(int), 0,
                                        np.asarray(grid_shape) - 1)
                            new_amp = np.maximum(
                                residual[r[:, 0], r[:, 1], r[:, 2]],
                                bg_std,
                            ).astype(np.float32)
                            cur_pos = np.concatenate([cur_pos, new_pos])
                            cur_amp = np.concatenate([cur_amp, new_amp])
                            topology_changed = True
                            say(f"[step {step+1:5d}] densify: "
                                f"+{n_new:<4d}      → {cur_pos.shape[0]:>5d}")

                # ---- no-improvement patience ----
                # Did this densify pass meaningfully improve the loss?
                if loss_val < best_loss_for_densify * (1.0 - improvement_tol):
                    best_loss_for_densify = loss_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= improvement_patience:
                        densify_frozen = True
                        freeze_step = step
                        freeze_reason = (
                            f"no loss improvement in {no_improve_count} "
                            f"densify passes"
                        )
                        say(f"[step {step+1:5d}] freeze densify: "
                            f"{freeze_reason}")

            # ---- rebuild model + optimiser if anything changed ----
            if topology_changed:
                model = GaussianVolumeModel(
                    positions=jnp.asarray(cur_pos),
                    amplitudes=jnp.asarray(cur_amp),
                    sigma=cur_sigma,
                    rngs=nnx.Rngs(seed + step + 1),
                )
                optimizer = nnx.Optimizer(model, _build_tx(), wrt=params_filter)
                last_topology_change = step
                n_history.append((step, model.num_gaussians()))

        # ---- convergence check ----------------------------------------
        # Loss must be stable in window AND topology not currently changing
        # AND (either) densification has been frozen so no growth will resume.
        if (
            step >= min_iterations
            and len(losses) >= convergence_window
            and step - last_topology_change >= convergence_window // 2
        ):
            recent = np.asarray(losses[-convergence_window:])
            denom = max(float(np.mean(recent)), 1e-12)
            rel_range = (recent.max() - recent.min()) / denom

            # N stability: change in N over the convergence window
            n_now = model.num_gaussians()
            n_then = n_now
            for s, n in reversed(n_history):
                if s < step - convergence_window:
                    n_then = n
                    break
            n_rel_change = abs(n_now - n_then) / max(n_now, 1)

            if rel_range < convergence_tol and (densify_frozen or n_rel_change < 0.02):
                converged = True
                say(f"[step {step+1:5d}] converged "
                    f"(rel_range={rel_range:.3g} < {convergence_tol:.3g}, "
                    f"N_rel_change={n_rel_change:.3g}, "
                    f"frozen={densify_frozen})")
                break

        # ---- loss degradation early stop -----------------------------
        # After freeze, the only remaining levers are prune and parameter
        # refinement. Detect destructive pruning by comparing a smoothed
        # post-freeze loss against the best loss ever seen.
        #
        # Three grace conditions:
        #   (i)   ``grace_after_freeze`` steps must have elapsed since
        #         freeze (the loss can drop substantially via parameter
        #         refinement during this period -- give it room).
        #   (ii)  ``topology_settle_grace`` steps since the last topology
        #         change (so prune/merge transients can dissipate).
        #   (iii) Enough history to fill the smoothing window.
        #
        # Median (rather than mean) of the smoothing window makes the
        # check robust to isolated spikes after each optimiser rebuild.
        grace_after_freeze = 300
        topology_settle_grace = 50
        smoothing_window = 100
        if (densify_frozen
                and freeze_step is not None
                and loss_degradation_tol > 0
                and step >= freeze_step + grace_after_freeze
                and step - last_topology_change >= topology_settle_grace
                and len(losses) >= smoothing_window):
            smoothed_loss = float(np.median(losses[-smoothing_window:]))
            if smoothed_loss > best_overall_loss * (1.0 + loss_degradation_tol):
                say(f"[step {step+1:5d}] stop: median loss has "
                    f"degraded {(smoothed_loss/best_overall_loss - 1)*100:.1f}% "
                    f"from best ({smoothed_loss:.4g} > "
                    f"{best_overall_loss:.4g})")
                converged = True
                break

        # ---- progress bar tick ---------------------------------------
        if pbar is not None:
            update_postfix()
            pbar.update(1)
        elif verbose and not quiet and (step + 1) % 100 == 0:
            # plain-text fallback (no tqdm installed)
            print(f"[step {step+1:5d}] loss={loss_val:.6g}  "
                  f"N={model.num_gaussians()}  "
                  f"sigma={float(model.get_sigma()):.4g}")

    # ---- close progress bar ----------------------------------------------
    if pbar is not None:
        update_postfix()
        pbar.refresh()
        pbar.close()

    if not converged:
        say(f"[done] max_iterations reached: loss={losses[-1]:.6g}  "
            f"N={model.num_gaussians()}  "
            f"sigma={float(model.get_sigma()):.4g}")

    # ---- final cleanup pass ----------------------------------------------
    cur_pos = np.asarray(model.get_positions())
    cur_amp = np.asarray(model.get_amplitudes())
    cur_sigma = float(model.get_sigma())
    keep = _is_inside(cur_pos, grid_shape, mask_np)
    if cur_amp.size > 0:
        # Same noise-relative threshold as in the loop -- avoids destroying
        # useful Gaussians in the final pass.
        keep &= cur_amp > (noise_amp_multiplier * bg_std)
    cur_pos = cur_pos[keep]
    cur_amp = cur_amp[keep]

    if merge_radius_factor > 0 and cur_pos.shape[0] > 1:
        merge_keep = _merge_close_gaussians(
            cur_pos, cur_amp, radius=merge_radius_factor * cur_sigma,
        )
        cur_pos = cur_pos[merge_keep]
        cur_amp = cur_amp[merge_keep]

    final_model = GaussianVolumeModel(
        positions=jnp.asarray(cur_pos),
        amplitudes=jnp.asarray(cur_amp),
        sigma=cur_sigma,
        rngs=nnx.Rngs(seed + 99999),
    )

    if verbose and not quiet:
        print(f"[done] final number of Gaussians = {final_model.num_gaussians()}  "
              f"sigma={cur_sigma:.4g}")

    return final_model


def fit_weights_to_images(model, md_path, mmap_output_dir, sr, batch_size=256, learning_rate=0.0001, num_epochs=3, is_global=True, uses_ctf=True):
    # Global vs local
    if is_global:
        # Init Optimizer (nnx.Optimizer automatically tracks model params)
        params_filter = nnx.All(nnx.Param, (nnx.PathContains('log_a'), nnx.PathContains('b')))
        optimizer = nnx.Optimizer(model, optax.sgd(learning_rate), wrt=params_filter)

    else:
        # Init Optimizer (nnx.Optimizer automatically tracks model params)
        params_filter = nnx.All(nnx.Param, nnx.PathContains('log_amplitudes'))
        optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=params_filter)

    @partial(nnx.jit, static_argnames=("grid_shape", "kernel_radius"))
    def train_step(model, optimizer, target, rotations, shifts, ctf, grid_shape, kernel_radius):
        def loss_fn(m):
            rendered = m.render_projections(rotations, shifts, grid_shape=grid_shape, kernel_radius=kernel_radius)
            rendered = ctfFilter(rendered, ctf, pad_factor=2)
            return jnp.mean((rendered - target) ** 2)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        grads, _ = grads.split(params_filter, ...)
        optimizer.update(model, grads)
        return loss

    def smooth(loss_history, window=50):
        return np.convolve(loss_history, np.ones(window) / window, mode='valid')

    # Prepare metadata
    generator = MetaDataGenerator(md_path)
    md_columns = extract_columns(generator.md)

    # Grid shape
    xsize = generator.md.getMetaDataImage(0).shape[0]
    grid_shape = (xsize, xsize, xsize)

    # Kernel radius
    def _kernel_radius_for(sigma_val: float) -> int:
        return max(2, int(np.ceil(3.5 * sigma_val)))
    kernel_radius = _kernel_radius_for(model.get_sigma())

    # Early stopping patience (steps)
    patience = 200

    # Grain dataset
    if mmap_output_dir is not None:
        load_to_ram = False
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                             precision=np.float16, group_size=1, shard_size=10000)
    else:
        load_to_ram = True
    data_loader = generator.return_grain_dataset(batch_size=batch_size, shuffle="global",
                                                 num_epochs=None, num_workers=8, num_threads=1, load_to_ram=load_to_ram)
    steps_per_epoch = int(len(generator.md) / batch_size)

    loss_history = []

    if is_global:
        print(f"\n{bcolors.OKCYAN}###### Adjusting gaussian weights to images (Global version)... ######{bcolors.ENDC}")
    else:
        print(f"\n{bcolors.OKCYAN}###### Adjusting gaussian weights to images (Local version)... ######{bcolors.ENDC}")

    pbar = tqdm(range(num_epochs * steps_per_epoch), desc="Adjusting weights", file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

    with closing(iter(data_loader)) as iter_data_loader:
        for _ in pbar:
            (target, labels) = next(iter_data_loader)
            # --- TRAIN STEP ---
            euler_angles = md_columns["euler_angles"][labels]
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

            shifts = md_columns["shifts"][labels]

            if uses_ctf:
                defocusU = md_columns["ctfDefocusU"][labels]
                defocusV = md_columns["ctfDefocusV"][labels]
                defocusAngle = md_columns["ctfDefocusAngle"][labels]
                cs = md_columns["ctfSphericalAberration"][labels]
                kv = md_columns["ctfVoltage"][labels][0]
                ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv, sr,
                                 [2 * xsize, int(2 * 0.5 * xsize + 1)],
                                 rotations.shape[0], True)
            else:
                ctf = jnp.ones(
                    [rotations.shape[0], 2 * xsize, int(2 * 0.5 * xsize + 1)], dtype=jnp.float32)

            loss_val = train_step(model, optimizer, target[..., 0], rotations, shifts, ctf, grid_shape, kernel_radius)

            loss_history.append(loss_val)

            # Progress bar update  (TQDM)
            if len(loss_history) > 1000:
                pbar.set_postfix_str(
                    f"| Loss: {sum(loss_history[-1000:]) / 1000:.6f}")
            else:
                pbar.set_postfix_str(
                    f"| Loss: {sum(loss_history) / len(loss_history):.6f}")

            smoothed = smooth(np.array(loss_history))
            best = min(smoothed)

            # if smoothed[-1] > best and (len(smoothed) - np.argmin(smoothed)) > patience:
            #     break

    return model
