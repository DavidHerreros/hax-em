#!/usr/bin/env python


import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import dm_pix

import numpy as np
from functools import partial

from einops import rearrange

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from hax.utils import *
from hax.layers import *


# How many latents to encode before clustering them into the intermediate
# heterogeneous volumes
LATENTS_FOR_CLUSTERING = 2048


def generate_sphere_points(n):
    """
    Generates N points uniformly distributed within a unit sphere.
    """
    # 1. Randomly sample azimuthal and polar angles
    # Phi: [0, 2π]
    phi = np.random.uniform(0, 2 * np.pi, n)

    # Theta: [0, π]
    # Use inverse cosine to correct for the area element on the sphere
    cos_theta = np.random.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)

    # 2. Randomly sample radius
    # U is uniform [0, 1]. We take the cube root to account for volume scaling.
    u = np.random.uniform(0, 1, n)
    r = u ** (1. / 3.)

    # 3. Convert Spherical to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack([x, y, z], axis=-1)


def generate_cylinder_points(n, radius=1.0, height=1.0):
    """
    Generates N points uniformly distributed within a cylinder,
    centered at the origin along the Z axis.

    Parameters
    ----------
    n      : number of points
    radius : cylinder radius
    height : total height (spans [-height/2, height/2])
    """
    # 1. Sample azimuthal angle uniformly [0, 2π]
    phi = np.random.uniform(0, 2 * np.pi, n)

    # 2. Sample radial distance
    # Area element in polar coords is r·dr·dφ, so PDF ∝ r.
    # Integrating: F(r) = r²/R²  →  r = R·√u  (square root, not cube root,
    # because only the disk cross-section needs correction — not the full volume)
    u = np.random.uniform(0, 1, n)
    r = radius * np.sqrt(u)

    # 3. Sample height uniformly along the axis
    z = np.random.uniform(-height / 2, height / 2, n)

    # 4. Convert polar → Cartesian (no spherical theta needed here)
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return np.stack([x, y, z], axis=-1)


def generate_spherical_rotations(n: int) -> np.ndarray:
    """
    Generates N (3, 3) rotation matrices that distribute evenly over a sphere.
    When applied to the Z-axis vector [0, 0, 1]^T, the resulting vectors
    form a Fibonacci lattice on the unit sphere.
    """
    # 1. Generate Fibonacci sphere points
    indices = np.arange(0, n, dtype=float)
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio

    # Using the standard offset formulation avoids placing points exactly on the poles
    # (z = 1 or z = -1), which neatly prevents division-by-zero errors later.
    z = 1.0 - (2.0 * indices + 1.0) / n
    radius = np.sqrt(1.0 - z ** 2)
    theta = 2.0 * np.pi * indices / phi

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # 2. Build the rotation matrices analytically
    # We construct the matrix that aligns [0,0,1] to [x,y,z] with zero twist.
    denom = 1.0 + z

    R = np.zeros((n, 3, 3))

    R[:, 0, 0] = 1.0 - (x ** 2) / denom
    R[:, 0, 1] = -(x * y) / denom
    R[:, 0, 2] = x

    R[:, 1, 0] = -(x * y) / denom
    R[:, 1, 1] = 1.0 - (y ** 2) / denom
    R[:, 1, 2] = y

    R[:, 2, 0] = -x
    R[:, 2, 1] = -y
    R[:, 2, 2] = z

    return R


def sliced_wasserstein_sphere(
    directions: jax.Array,      # (N, 3) unit vectors from R[:,:,2]
    rng: jax.Array,
    n_projections: int = 64,
) -> jax.Array:
    """
    Sliced Wasserstein distance between direction samples and
    a uniform distribution over S².
    """
    n = directions.shape[0]
    rng_proj, rng_prior = jax.random.split(rng)

    # Sample reference uniform directions on S²
    raw = jax.random.normal(rng_prior, shape=(n, 3))
    uniform_sphere = raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)

    # Random projection directions (also on S²)
    raw_proj = jax.random.normal(rng_proj, shape=(n_projections, 3))
    proj_dirs = raw_proj / jnp.linalg.norm(raw_proj, axis=-1, keepdims=True)

    # Project both sets onto each direction: (n_projections, N)
    d_proj = jnp.einsum("pd,nd->pn", proj_dirs, directions)
    u_proj = jnp.einsum("pd,nd->pn", proj_dirs, uniform_sphere)

    # Sort and compute L2 distance between sorted projections
    d_sorted = jnp.sort(d_proj, axis=-1)
    u_sorted = jnp.sort(u_proj, axis=-1)

    return jnp.mean((d_sorted - u_sorted) ** 2)


def repulsion_loss(
        directions: jax.Array,  # (N, 3) unit vectors
        s: float = 2.0,  # Riesz exponent: higher = more local repulsion
        eps: float = 1e-6,  # numerical safety
) -> jax.Array:
    """
    Riesz s-energy: penalizes pairs of directions that are too close on S².

    E = (1/N²) * sum_{i≠j} 1 / ||d_i - d_j||^s

    s=1  → Coulomb potential (long range, global)
    s=2  → stronger local repulsion
    s→∞  → only nearest neighbor matters (purely local)

    Minimizing this energy = maximizing the spread of points on S²,
    which is the classical Tammes/Thomson problem.
    """
    # Pairwise Euclidean distances on S²
    diff = directions[:, None, :] - directions[None, :, :]  # (N, N, 3)
    sq_dist = jnp.sum(diff ** 2, axis=-1)  # (N, N)

    # Mask diagonal to avoid self-repulsion
    mask = 1.0 - jnp.eye(directions.shape[0])
    energy = mask / (sq_dist + eps) ** (s / 2.0)

    return jnp.mean(energy)


def _assignment_probabilities(losses, temperature, adaptive=True, eps=1e-6):
    """Turn per-candidate losses into stable responsibilities.

    In adaptive mode ``temperature`` is dimensionless: every particle's loss
    differences are divided by their own standard deviation first.  This makes
    the exploration schedule insensitive to map amplitude, box size and CTF
    mode, unlike the historical absolute ``tau``.
    """
    centered = losses - jnp.min(losses, axis=1, keepdims=True)
    if adaptive:
        scale = jnp.std(centered, axis=1, keepdims=True)
        centered = centered / jnp.maximum(scale, eps)
    temperature = jnp.maximum(jnp.asarray(temperature, dtype=losses.dtype), eps)
    return jax.nn.softmax(-centered / temperature, axis=1)


def _top_two_candidate_diagnostics(losses, eps=1e-8):
    """Return complementary confidence diagnostics for candidate competition."""
    if losses.shape[1] < 2:
        best_indices = jnp.zeros(losses.shape[0], dtype=jnp.int32)
        zeros = jnp.zeros(losses.shape[0], dtype=losses.dtype)
        return best_indices, zeros, zeros, zeros, zeros, zeros

    negative_top_two, top_two_indices = jax.lax.top_k(-losses, 2)
    best_losses = -negative_top_two[:, 0]
    second_losses = -negative_top_two[:, 1]
    absolute_margin = jnp.maximum(second_losses - best_losses, 0.0)
    # Historical fraction by which the best candidate improves over the
    # runner-up.  It is retained for continuity but includes any common loss
    # baseline, unlike the standardized diagnostics below.
    relative_margin = absolute_margin / jnp.maximum(jnp.abs(second_losses), eps)
    candidate_scale = jnp.std(losses, axis=1)
    standardized_margin = absolute_margin / jnp.maximum(candidate_scale, eps)
    median_losses = jnp.median(losses, axis=1)
    median_separation = jnp.maximum(median_losses - best_losses, 0.0)
    median_normalized_margin = absolute_margin / jnp.maximum(median_separation, eps)

    # Unlike assignment entropy, this remains informative after switching to
    # hard winners because it is always derived from the full loss landscape.
    centered_losses = losses - best_losses[:, None]
    score_probabilities = jax.nn.softmax(
        -centered_losses / jnp.maximum(candidate_scale[:, None], eps), axis=1)
    score_entropy = -jnp.sum(
        score_probabilities * jnp.log(jnp.maximum(score_probabilities, eps)), axis=1)
    score_entropy = score_entropy / jnp.maximum(
        jnp.log(jnp.asarray(losses.shape[1], dtype=losses.dtype)), eps)
    return (top_two_indices[:, 0], absolute_margin, relative_margin,
            standardized_margin, median_normalized_margin, score_entropy)


def _symmetry_aware_rotation_change_degrees(previous, current, symmetries):
    """Minimum SO(3) geodesic change over equivalent symmetry operations."""
    previous = np.asarray(previous)
    current = np.asarray(current)
    symmetries = np.asarray(symmetries)
    equivalent_previous = np.einsum("sji,njk->nsik", symmetries, previous)
    relative = np.einsum("nji,nsjk->nsik", current, equivalent_previous)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
                     -1.0, 1.0)
    return np.rad2deg(np.min(np.arccos(cosine), axis=1))


class _PoseDiagnosticsTracker:
    """Host-side per-particle history used only by opt-in diagnostics."""

    def __init__(self, n_particles, symmetries):
        self.symmetries = np.asarray(symmetries)
        self.previous_rotations = np.zeros((n_particles, 3, 3), dtype=np.float32)
        self.previous_heads = np.full(n_particles, -1, dtype=np.int32)
        self.previous_epochs = np.full(n_particles, -1, dtype=np.int32)
        self.epoch = None
        self._reset_epoch()

    def _reset_epoch(self):
        self.absolute_margins = []
        self.relative_margins = []
        self.standardized_margins = []
        self.median_normalized_margins = []
        self.score_entropies = []
        self.selected_is_top1 = []
        self.pose_changes = []
        self.head_switches = []
        self.comparable_count = 0
        self.observed_count = 0

    def update(self, labels, rotations, selected_heads, best_heads,
               absolute_margins, relative_margins, standardized_margins,
               median_normalized_margins, score_entropies, epoch):
        if self.epoch != epoch:
            self.epoch = epoch
            self._reset_epoch()

        labels = np.asarray(labels, dtype=np.int64)
        rotations = np.asarray(rotations)
        selected_heads = np.asarray(selected_heads, dtype=np.int32)
        best_heads = np.asarray(best_heads, dtype=np.int32)
        self.absolute_margins.append(np.asarray(absolute_margins))
        self.relative_margins.append(np.asarray(relative_margins))
        self.standardized_margins.append(np.asarray(standardized_margins))
        self.median_normalized_margins.append(np.asarray(median_normalized_margins))
        self.score_entropies.append(np.asarray(score_entropies))
        self.selected_is_top1.append(selected_heads == best_heads)
        self.observed_count += labels.size

        # Compare only consecutive epochs.  This prevents a particle omitted by
        # a dropped final batch from contributing a multi-epoch change.
        comparable = self.previous_epochs[labels] == epoch - 1
        if np.any(comparable):
            comparable_labels = labels[comparable]
            changes = _symmetry_aware_rotation_change_degrees(
                self.previous_rotations[comparable_labels], rotations[comparable],
                self.symmetries)
            self.pose_changes.append(changes)
            self.head_switches.append(
                self.previous_heads[comparable_labels] != selected_heads[comparable])
            self.comparable_count += comparable_labels.size

        self.previous_rotations[labels] = rotations
        self.previous_heads[labels] = selected_heads
        self.previous_epochs[labels] = epoch

    def summary(self):
        absolute = np.concatenate(self.absolute_margins)
        relative = np.concatenate(self.relative_margins)
        standardized = np.concatenate(self.standardized_margins)
        median_normalized = np.concatenate(self.median_normalized_margins)
        score_entropy = np.concatenate(self.score_entropies)
        selected_is_top1 = np.concatenate(self.selected_is_top1)
        summary = {
            "absolute_margin_mean": float(np.mean(absolute)),
            "relative_margin_mean": float(np.mean(relative)),
            "relative_margin_median": float(np.median(relative)),
            "relative_margin_p10": float(np.percentile(relative, 10.0)),
            "relative_margin_below_1pct": float(np.mean(relative < 0.01)),
            "relative_margin_below_5pct": float(np.mean(relative < 0.05)),
            "relative_margin_below_10pct": float(np.mean(relative < 0.10)),
            "standardized_margin_mean": float(np.mean(standardized)),
            "standardized_margin_median": float(np.median(standardized)),
            "standardized_margin_p10": float(np.percentile(standardized, 10.0)),
            "standardized_margin_below_0_1": float(np.mean(standardized < 0.1)),
            "standardized_margin_below_0_25": float(np.mean(standardized < 0.25)),
            "standardized_margin_below_0_5": float(np.mean(standardized < 0.5)),
            "median_normalized_margin_mean": float(np.mean(median_normalized)),
            "median_normalized_margin_median": float(np.median(median_normalized)),
            "candidate_score_entropy_mean": float(np.mean(score_entropy)),
            "selected_is_top1_fraction": float(np.mean(selected_is_top1)),
            "comparison_coverage_fraction": (
                float(self.comparable_count / self.observed_count)
                if self.observed_count else 0.0),
        }
        if self.pose_changes:
            changes = np.concatenate(self.pose_changes)
            switches = np.concatenate(self.head_switches)
            summary.update({
                "pose_change_mean_degrees": float(np.mean(changes)),
                "pose_change_median_degrees": float(np.median(changes)),
                "pose_change_p90_degrees": float(np.percentile(changes, 90.0)),
                "pose_change_over_5deg_fraction": float(np.mean(changes > 5.0)),
                "pose_change_over_15deg_fraction": float(np.mean(changes > 15.0)),
                "pose_change_over_30deg_fraction": float(np.mean(changes > 30.0)),
                "head_switch_fraction": float(np.mean(switches)),
            })
        return summary


class _PoseCurriculumController:
    """Advance the candidate scoring resolution when winner churn settles.

    Unlike a fixed epoch schedule, each low-pass stage is held until the
    per-particle winners are stable at that resolution: the frequency band
    widens only once it stops changing the decisions it feeds.
    """

    def __init__(self, image_size, scales, switch_threshold=0.05,
                 pose_threshold_degrees=5.0, min_epochs=1, max_epochs=10,
                 temperatures=None):
        self.image_size = int(image_size)
        self.scales = tuple(float(scale) for scale in scales)
        self.switch_threshold = float(switch_threshold)
        self.pose_threshold_degrees = float(pose_threshold_degrees)
        self.min_epochs = max(1, int(min_epochs))
        self.max_epochs = max(0, int(max_epochs))
        if temperatures is None:
            temperatures = (0.0,) * len(self.scales)
        temperatures = tuple(float(value) for value in temperatures)
        if len(temperatures) == 1:
            temperatures = temperatures * len(self.scales)
        if len(temperatures) != len(self.scales):
            raise ValueError(
                "curriculum temperatures must match the number of scales")
        self.temperatures = temperatures
        self.stage = 0
        self.epochs_in_stage = 0

    @property
    def temperature(self):
        """Dimensionless assignment temperature; the final stage is always hard."""
        if self.is_final:
            return 0.0
        return self.temperatures[self.stage]

    @property
    def is_final(self):
        return self.stage >= len(self.scales)

    @property
    def scoring_size(self):
        if self.is_final:
            return self.image_size
        return min(self.image_size,
                   max(8, int(round(self.image_size * self.scales[self.stage]))))

    @property
    def multiscale_weight_multiplier(self):
        if self.is_final:
            return 0.0
        return 1.0 - self.stage / (len(self.scales) + 1)

    def observe_epoch(self, summary):
        """Consume one finished epoch's pose-diagnostics summary; return True on advance."""
        if self.is_final:
            return False
        self.epochs_in_stage += 1
        if self.epochs_in_stage < self.min_epochs:
            return False
        # The first observed epoch has no cross-epoch comparison yet; the
        # defaults keep the stage until churn is actually measured.
        stable = (
            summary.get("head_switch_fraction", 1.0) <= self.switch_threshold
            and summary.get("pose_change_median_degrees", 180.0)
            <= self.pose_threshold_degrees)
        capped = 0 < self.max_epochs <= self.epochs_in_stage
        if stable or capped:
            self.stage += 1
            self.epochs_in_stage = 0
            return True
        return False

    def state_dict(self):
        return {"stage": int(self.stage),
                "epochs_in_stage": int(self.epochs_in_stage)}

    def load_state_dict(self, state):
        self.stage = min(max(0, int(state.get("stage", 0))), len(self.scales))
        self.epochs_in_stage = max(0, int(state.get("epochs_in_stage", 0)))


def _fibonacci_sphere_directions(n_directions, dtype=jnp.float32):
    """Deterministic equal-area bin centers on S2."""
    indices = jnp.arange(n_directions, dtype=dtype) + 0.5
    z = 1.0 - 2.0 * indices / n_directions
    radius = jnp.sqrt(jnp.maximum(1.0 - z ** 2, 0.0))
    golden_ratio = (1.0 + jnp.sqrt(jnp.asarray(5.0, dtype=dtype))) / 2.0
    theta = 2.0 * jnp.pi * indices / golden_ratio
    return jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta), z], axis=-1)


def _random_rotation_matrix(key, dtype=jnp.float32):
    """Sample a Haar-uniform rotation through a normalized quaternion."""
    quaternion = jax.random.normal(key, (4,), dtype=dtype)
    quaternion = quaternion / jnp.maximum(jnp.linalg.norm(quaternion), 1e-8)
    w, x, y, z = quaternion
    return jnp.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=dtype)


def _soft_spherical_occupancy(directions, bin_directions, kappa):
    """Soft histogram over equal-area spherical bins."""
    logits = jnp.asarray(kappa, dtype=directions.dtype) * jnp.einsum(
        "nd,kd->nk", directions, bin_directions)
    return jnp.mean(jax.nn.softmax(logits, axis=-1), axis=0)


def _candidate_coverage_loss(directions, memory_bank, bank_count, key,
                             n_bins=256, kappa=32.0, bank_samples=1024,
                             bank_mix=0.5, eps=1e-8):
    """KL-to-uniform loss for current and historical candidate directions.

    The historical occupancy is detached and mixed with equal normalized mass,
    so a large bank cannot dilute gradients from the current batch.  Randomly
    rotating the equal-area grid avoids imprinting fixed bin boundaries.
    """
    rotation_key, sample_key = jax.random.split(key)
    bins = _fibonacci_sphere_directions(n_bins, dtype=directions.dtype)
    bins = jnp.einsum(
        "ij,nj->ni", _random_rotation_matrix(rotation_key, directions.dtype), bins)
    current_occupancy = _soft_spherical_occupancy(directions, bins, kappa)

    occupancy = current_occupancy
    if bank_samples > 0:
        valid_count = jnp.maximum(jnp.asarray(bank_count, dtype=jnp.int32), 1)
        indices = jax.random.randint(
            sample_key, (bank_samples,), minval=0, maxval=valid_count)
        historical = jax.lax.stop_gradient(memory_bank[indices])
        historical_occupancy = _soft_spherical_occupancy(historical, bins, kappa)
        effective_mix = jnp.asarray(bank_mix, directions.dtype) * (
            jnp.asarray(bank_count) > 0).astype(directions.dtype)
        occupancy = ((1.0 - effective_mix) * current_occupancy
                     + effective_mix * historical_occupancy)

    occupancy = occupancy / jnp.maximum(jnp.sum(occupancy), eps)
    return jnp.sum(occupancy * jnp.log(jnp.maximum(occupancy * n_bins, eps)))


class LowRankLinear(nnx.Module):
    """Independent low-rank replacement for a square dense layer.

    Every pose-hypothesis member owns both factors; no weights are shared between
    hypotheses.  ``rank=0`` keeps the legacy dense layer for old checkpoints.
    """

    def __init__(self, features, rank=0, *, rngs: nnx.Rngs, dtype=jnp.bfloat16):
        self.rank = int(rank or 0)
        if self.rank > 0:
            self.down = Linear(features, self.rank, rngs=rngs, dtype=dtype, use_bias=False)
            self.up = Linear(self.rank, features, rngs=rngs, dtype=dtype)
        else:
            self.dense = Linear(features, features, rngs=rngs, dtype=dtype)

    def __call__(self, x):
        if self.rank > 0:
            return self.up(self.down(x))
        return self.dense(x)


class PoseHead(nnx.Module):
    def __init__(self, is_refine=False, low_rank=0, predict_shift_delta=False, *, rngs: nnx.Rngs):
        if is_refine:
            kernel_init = nnx.initializers.zeros_init()
            bias_init = nnx.initializers.zeros_init()
        else:
            kernel_init=nnx.initializers.normal(1e-2)
            bias_init=rot6d_perturbation_init(N=1, sigma=5.0, mode="bias")
            # kernel_init=nnx.initializers.normal(1e-4)
            # bias_init=rot6d_perturbation_init(N=1, sigma=1.0, mode="bias")
            # kernel_init=rot6d_perturbation_init(N=1, sigma=1.0, mode="weight")
            # bias_init=nnx.initializers.zeros_init()
            # kernel_init = jax.nn.initializers.normal(stddev=1e-4)
            # bias_init = nnx.initializers.zeros_init()

        hidden_layers = []
        for _ in range(3):
            if low_rank > 0:
                hidden_layers.append(LowRankLinear(1024, rank=low_rank, rngs=rngs, dtype=jnp.bfloat16))
            else:
                # Keep the exact legacy state-tree path for old checkpoints.
                hidden_layers.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers = nnx.List(hidden_layers)
        self.pose_layer = Linear(1024, 6, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init)
        self.predict_shift_delta = bool(predict_shift_delta)
        if self.predict_shift_delta:
            # Zero-init so per-candidate shifts start exactly at the shared
            # trunk shift and only diverge when the data asks for it.
            self.shift_layer = Linear(1024, 2, rngs=rngs,
                                      kernel_init=nnx.initializers.zeros_init(),
                                      bias_init=nnx.initializers.zeros_init())

    def __call__(self, x):
        for layer in self.hidden_layers:
            # x = nnx.gelu(x + layer(x))
            x = nnx.gelu(layer(x))
        pose = self.pose_layer(x)
        if self.predict_shift_delta:
            return jnp.concat([pose, self.shift_layer(x)], axis=-1)
        return pose


class PoseHeadEnsemble(nnx.Module):
    def __init__(self, num_members, is_refine=False, low_rank=0, predict_shift_delta=False, *, rngs: nnx.Rngs):
        key = rngs.params()
        member_keys = jax.random.split(key, num_members)

        @nnx.vmap(in_axes=(0), out_axes=0)
        def make_member(key):
            return PoseHead(is_refine=is_refine, low_rank=low_rank,
                            predict_shift_delta=predict_shift_delta, rngs=nnx.Rngs(key))

        self.ensemble = make_member(member_keys)

    def __call__(self, x):
        @nnx.vmap(in_axes=(0, None), out_axes=1)
        def forward(model, x):
            return model(x)
        return forward(self.ensemble, x)


class EncoderPose(nnx.Module):
    def __init__(self, input_dim, pyramid_levels=4, num_components=18, refine_current_assignment=False,
                 use_anchor_rotations=True, low_rank=0, spatial_pool=1, per_candidate_shifts=False,
                 *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 64  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 3))
        self.pyramid_levels = pyramid_levels
        self.num_components = num_components
        self.refine_current_assignment = refine_current_assignment
        self.use_anchor_rotations = use_anchor_rotations
        self.spatial_pool = max(1, int(spatial_pool))
        self.per_candidate_shifts = bool(per_candidate_shifts)

        # Hidden layers
        hidden_layers_conv = [Conv(self.pyramid_levels, 64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16)]
        hidden_layers_conv.append(Conv(64, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

        hidden_layers_conv.append(Conv(64, 128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(Conv(128, 128, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

        hidden_layers_conv.append(Conv(128, 256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(Conv(256, 256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(Conv(256, 256, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

        hidden_layers_conv.append(Conv(256, 512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(Conv(512, 512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(Conv(512, 512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_conv = nnx.List(hidden_layers_conv)

        linear_spatial = max(1, self.out_conv_dim // self.spatial_pool)
        hidden_layers_linear = [Linear(linear_spatial * linear_spatial * 512, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        hidden_layers_linear.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_linear.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_layers_linear.append(Linear(1024, 8, rngs=rngs))
        self.hidden_layers_linear = nnx.List(hidden_layers_linear)

        # Anchor rotations
        self.anchor_rotations = jnp.array(generate_spherical_rotations(num_components))

        # Layers to 9D rotation
        self.ensemble_6d_heads = PoseHeadEnsemble(num_members=num_components, is_refine=False,
                                                 low_rank=low_rank,
                                                 predict_shift_delta=self.per_candidate_shifts,
                                                 rngs=rngs)

        # Layers to shifts
        hidden_shifts = [Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        hidden_shifts.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        hidden_shifts.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        if refine_current_assignment:
            hidden_shifts.append(Linear(1024, 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init()))
        else:
            hidden_shifts.append(Linear(1024, 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init()))
        self.hidden_shifts = nnx.List(hidden_shifts)

    def __call__(self, x):
        # Resize images
        x = jax.image.resize(x, (x.shape[0], self.input_conv_dim, self.input_conv_dim, 1), method="bilinear")

        # Pyramid filter
        pyramid_levels_imgs = []
        for i in range(self.pyramid_levels):
            scale_factor = 2 ** i
            if scale_factor == 1:
                processed_level = x
            else:
                new_size = max(1, x.shape[1] // scale_factor)
                downsampled = jax.image.resize(x, (x.shape[0], new_size, new_size, 1), method='bilinear')
                upsampled = jax.image.resize(downsampled, (x.shape[0], x.shape[1], x.shape[1], 1), method='bilinear')
                processed_level = upsampled
            pyramid_levels_imgs.append(processed_level)
        x = jnp.concat(pyramid_levels_imgs, axis=-1)

        # Convolutional hidden layers
        for layer in self.hidden_layers_conv:
            if layer.in_features == layer.out_features and 1 in layer.strides:
                x = nnx.gelu(x + layer(x))
            else:
                x = nnx.gelu(layer(x))

        if self.spatial_pool > 1:
            x = nnx.avg_pool(x, window_shape=(self.spatial_pool, self.spatial_pool),
                             strides=(self.spatial_pool, self.spatial_pool), padding="VALID")

        # Linear hidden layers
        x = rearrange(x, 'b h w c -> b (h w c)')
        for layer in self.hidden_layers_linear[:-1]:
            if layer.in_features == layer.out_features:
                x = nnx.gelu(x + layer(x))
            else:
                x = nnx.gelu(layer(x))
        x = self.hidden_layers_linear[-1](x)

        # First output: rotation matrices
        head_outputs = self.ensemble_6d_heads(x)
        if self.per_candidate_shifts:
            rotations_6d, shift_deltas = head_outputs[..., :6], head_outputs[..., 6:]
        else:
            rotations_6d, shift_deltas = head_outputs, None

        rotations_6d = rotations_6d.reshape(x.shape[0] * self.num_components, 6)
        if self.refine_current_assignment:
            identity_6d = jnp.array([1., 0., 0., 0., 1., 0.])[None, ...].repeat(rotations_6d.shape[0], axis=0)
            rotations_6d = identity_6d + rotations_6d

        a1, a2 = jnp.split(rotations_6d, 2, axis=-1)
        b1 = a1 / jnp.clip(jnp.linalg.norm(a1, axis=-1, keepdims=True), a_min=1e-6)
        a2_ortho = a2 - jnp.sum(a2 * b1, axis=-1, keepdims=True) * b1
        b2 = a2_ortho / jnp.clip(jnp.linalg.norm(a2_ortho, axis=-1, keepdims=True), a_min=1e-6)
        b3 = jnp.cross(b1, b2, axis=-1)
        rotations = jnp.stack([b1, b2, b3], axis=-1)
        rotations = rotations.reshape(x.shape[0], self.num_components, 3, 3)
        if self.use_anchor_rotations and not self.refine_current_assignment:
            rotations = jnp.einsum('bnhk,nkw->bnhw', rotations, self.anchor_rotations)

        # Third output: in plane shifts
        in_plane_shifts = nnx.gelu(self.hidden_shifts[0](x))
        for layer in self.hidden_shifts[1:-1]:
            in_plane_shifts = nnx.gelu(in_plane_shifts + layer(in_plane_shifts))
        in_plane_shifts = self.hidden_shifts[-1](in_plane_shifts)

        # Per-candidate deltas ride on the shared trunk shift, so the winner
        # competition can co-select an orientation with its matching shift.
        if shift_deltas is not None:
            in_plane_shifts = in_plane_shifts[:, None, :] + shift_deltas
        else:
            # Broadcast shifts to euler angles shape
            in_plane_shifts = jnp.broadcast_to(in_plane_shifts[:, None, :], (in_plane_shifts.shape[0], self.num_components, 2))

        return rotations, in_plane_shifts


class EncoderHet(nnx.Module):
    def __init__(self, input_dim, lat_dim=8, architecture="legacy", encoder_size=64,
                 *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = int(encoder_size) if architecture == "resize" else 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 4))
        self.architecture = architecture

        if architecture == "convstem":
            # Bring arbitrary box sizes to the legacy 64x64 grid without the
            # O(box^2 * 4096) flattened projection.  The final resize handles
            # non-powers of two and boxes below 64.
            n_stem = max(0, int(np.floor(np.log2(max(self.input_dim, 1) / self.input_conv_dim))))
            stem = []
            channels = 1
            for i in range(n_stem):
                out_channels = min(16, 4 * (2 ** i))
                stem.append(Conv(channels, out_channels, kernel_size=(5, 5), strides=(2, 2),
                                 padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
                channels = out_channels
            stem.append(Conv(channels, 1, kernel_size=(1, 1), strides=(1, 1),
                             padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.stem = nnx.List(stem)
            hidden_layers_conv = []
        elif architecture == "legacy":
            hidden_layers_conv = [
                Linear(self.input_dim * self.input_dim, self.input_conv_dim * self.input_conv_dim, rngs=rngs,
                       dtype=jnp.bfloat16)]
        elif architecture == "resize":
            # Heterogeneity is inferred from an anti-aliased image while the
            # reconstruction loss remains at full resolution.  This avoids the
            # O(box^2 * 4096) legacy projection (hundreds of millions of weights
            # for 256/320 boxes) without inserting a learned downsampling stage.
            hidden_layers_conv = []
        else:
            raise ValueError(f"Unknown ReconSIREN heterogeneity encoder architecture: {architecture}")

        hidden_layers_conv.append(
            Conv(1, 4, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(
            Conv(4, 8, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(
            Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(
            Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(
            Conv(8, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(
            Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(
            Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_conv.append(
            Conv(16, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_conv = nnx.List(hidden_layers_conv)

        hidden_layers_linear = [Linear(16 * self.out_conv_dim * self.out_conv_dim, 256, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(3):
            hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

        hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        for _ in range(2):
            hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

        self.hidden_layers_linear = nnx.List(hidden_layers_linear)
        self.mean_x = Linear(256, lat_dim, rngs=rngs)
        self.logstd_x = Linear(256, lat_dim, rngs=rngs)

    def sample_gaussian(self, mean, logstd, *, rngs):
        return logstd * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, *, rngs=None):
        if self.architecture == "legacy":
            x = rearrange(x, 'b h w c -> b (h w c)')
            x = nnx.leaky_relu(self.hidden_layers_conv[0](x))
            x = rearrange(x, 'b (h w c) -> b h w c', h=self.input_conv_dim,
                          w=self.input_conv_dim, c=1)
            conv_layers = self.hidden_layers_conv[1:]
        elif self.architecture == "convstem":
            for layer in self.stem:
                x = nnx.leaky_relu(layer(x))
            if x.shape[1] != self.input_conv_dim or x.shape[2] != self.input_conv_dim:
                x = jax.image.resize(
                    x, (x.shape[0], self.input_conv_dim, self.input_conv_dim, 1), method="bilinear")
            conv_layers = self.hidden_layers_conv
        else:
            x = jax.image.resize(
                x, (x.shape[0], self.input_conv_dim, self.input_conv_dim, 1),
                method="lanczos3", antialias=True)
            conv_layers = self.hidden_layers_conv

        for layer in conv_layers:
            if layer.in_features != layer.out_features:
                x = nnx.leaky_relu(layer(x))  # or nnx.relu
            else:
                aux = layer(x)
                if aux.shape[1] == x.shape[1]:
                    x = nnx.leaky_relu(x + aux)  # or nnx.relu
                else:
                    x = nnx.leaky_relu(aux)  # or nnx.relu

        x = rearrange(x, 'b h w c -> b (h w c)')

        for layer in self.hidden_layers_linear:
            if layer.in_features != layer.out_features:
                x = nnx.leaky_relu(layer(x))  # or nnx.relu
            else:
                x = nnx.leaky_relu(x + layer(x))  # or nnx.relu

        mean = self.mean_x(x)
        logstd = self.logstd_x(x)
        # sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean

        return mean, mean, logstd


class DeltaVolumeDecoder(nnx.Module):
    def __init__(self, coords, values, volume_size, learn_delta_volume=True,
                 parameterization="network", *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.learn_delta_volume = learn_delta_volume
        self.parameterization = parameterization
        self.n_gaussians = coords.shape[0]
        self.factor = 0.5 * volume_size
        self.coords = coords[None, ...]
        self.reference_values = values[None, ...]

        if parameterization == "direct":
            variable_type = nnx.Param if learn_delta_volume else nnx.Variable
            self.delta_coords = variable_type(jnp.zeros_like(coords))
            self.delta_values = variable_type(jnp.zeros_like(values))
            return
        if parameterization != "network":
            raise ValueError(f"Unknown consensus parameterization: {parameterization}")

        if jnp.all(self.reference_values == 0):
            kernel_init = nnx.initializers.glorot_uniform()
        else:
            kernel_init = nnx.initializers.zeros_init()

        hidden_linear = [
            Siren2Linear(in_features=self.n_gaussians * 3, out_features=1024, rngs=rngs, dtype=jnp.bfloat16, is_first=True,
                         w0=1.0, s=0.0, c=1.0)]
        for _ in range(3):
            hidden_linear.append(
                Siren2Linear(in_features=1024, out_features=1024, rngs=rngs, dtype=jnp.bfloat16, is_first=False,
                             custom_init=True, is_residual=False, w0=1.0, s=0.0, c=1.0))
        if learn_delta_volume:
            hidden_linear.append(Linear(in_features=1024, out_features=4 * self.n_gaussians, rngs=rngs, kernel_init=kernel_init))
        else:
            hidden_linear.append(Linear(in_features=1024, out_features=3 * self.n_gaussians, rngs=rngs, kernel_init=kernel_init))
        self.hidden_linear = nnx.List(hidden_linear)


    def __call__(self):
        if self.parameterization == "direct":
            if self.learn_delta_volume:
                delta_coords = self.delta_coords.get_value()[None, ...]
                delta_values = self.delta_values.get_value()[None, ...]
            else:
                delta_coords = jnp.zeros_like(self.coords)
                delta_values = jnp.zeros_like(self.reference_values)
            values = nnx.relu(self.reference_values + delta_values)
            coords = self.factor * (self.coords + delta_coords)
            return coords, values

        x = self.coords.flatten()[None, ...]

        if self.learn_delta_volume:
            # Decode voxel values
            x = self.hidden_linear[0](x)
            for layer in self.hidden_linear[1:-1]:
                x = layer(x)
            x = self.hidden_linear[-1](x)

              # Extract delta_coords and values
            x = jnp.reshape(x, (x.shape[0], self.n_gaussians, 4))
            delta_coords, delta_values = x[..., :3], x[..., 3]

            # Recover volume values (TODO: Check if applying ReLu is really needed)
            values = nnx.relu(self.reference_values + delta_values)
        else:
            # Extract delta_coords
            delta_coords = jnp.reshape(x, (x.shape[0], self.n_gaussians, 3))

            # Recover volume values (TODO: Check if applying ReLu is really needed)
            values = nnx.relu(self.reference_values)

        # Recover coords (non-normalized)
        coords = self.factor * (self.coords + delta_coords)

        return coords, values

    def decode_volume(self, coords_values=None, filter=True, sigma=1.0, analytic=False):
        # Decode volume values
        if coords_values is not None:
            coords, values = coords_values
        else:
            coords, values = self.__call__()

        # Displace coordinates
        coords = coords + self.factor

        # Place values on grid
        grids = jnp.zeros((values.shape[0], self.volume_size, self.volume_size, self.volume_size))

        # Scatter volume
        bposf = jnp.floor(coords)
        bposi = bposf.astype(jnp.int32)
        bposf = coords - bposf

        bamp0 = values * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp1 = values * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp2 = values * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp3 = values * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp4 = values * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp5 = values * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp6 = values * (bposf[:, :, 0]) * (bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp7 = values * (bposf[:, :, 0]) * (bposf[:, :, 1]) * (bposf[:, :, 2])

        bamp = jnp.concat([bamp0, bamp1, bamp2, bamp3, bamp4, bamp5, bamp6, bamp7], axis=1)
        bposi = jnp.concat([bposi, bposi + jnp.array((1, 0, 0)), bposi + jnp.array((0, 1, 0)), bposi + jnp.array((0, 0, 1)),
                           bposi + jnp.array((0, 1, 1)), bposi + jnp.array((1, 0, 1)), bposi + jnp.array((1, 1, 0)), bposi + jnp.array((1, 1, 1))], axis=1)

        def scatter_volume(vol, bpos_i, bamp_i):
            return vol.at[bpos_i[..., 2], bpos_i[..., 1], bpos_i[..., 0]].add(bamp_i)

        grids = jax.vmap(scatter_volume, in_axes=(0, None, 0))(grids, bposi, bamp)

        # Filter volume
        if filter:
            low_pass = low_pass_3d_analytic if analytic else low_pass_3d
            grids = jax.vmap(low_pass, in_axes=(0, None))(grids, sigma)

        return grids

class HetVolumeDecoder(nnx.Module):
    def __init__(self, coords, values, n_gaussians, lat_dim, volume_size,
                 residual_to_consensus=False, center_decoder=False,
                 coordinate_scale=1.0, amplitude_scale=1.0, small_final_init=False,
                 *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.n_gaussians = n_gaussians
        self.coords = coords[None, ...]
        self.reference_values = values[None, ...]
        self.residual_to_consensus = bool(residual_to_consensus)
        self.center_decoder = bool(center_decoder)
        self.coordinate_scale = float(coordinate_scale)
        self.amplitude_scale = float(amplitude_scale)

        # Indices to (normalized) coords
        self.factor = 0.5 * volume_size

        hidden = [
            Siren2Linear(in_features=lat_dim, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=True,
                         w0=30.0, s=0.0, c=1.0)]
        for _ in range(4):
            hidden.append(
                Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=False,
                             custom_init=True, is_residual=True, w0=1.0, s=0.0, c=6.0))
        final_init = (nnx.initializers.normal(1e-4) if small_final_init
                      else nnx.initializers.glorot_uniform())
        hidden.append(Linear(in_features=8, out_features=4 * n_gaussians, rngs=rngs,
                             kernel_init=final_init,
                             bias_init=nnx.initializers.zeros_init()))
        self.hidden = nnx.List(hidden)

    def _decode_deltas(self, x):
        x = self.hidden[0](x)
        for layer in self.hidden[1:-1]:
            x = layer(x)
        x = self.hidden[-1](x)
        return jnp.reshape(x, (x.shape[0], self.n_gaussians, 4))

    def __call__(self, x, base_coords=None, base_values=None):
        deltas = self._decode_deltas(x)
        if self.center_decoder:
            # Remove the latent-independent decoder path.  z=0 is therefore the
            # current consensus exactly, while all conformational changes must
            # be explained through a latent-dependent residual.
            deltas = deltas - self._decode_deltas(jnp.zeros_like(x))
        delta_coords, delta_values = deltas[..., :3], deltas[..., 3]

        if self.residual_to_consensus:
            if base_coords is None or base_values is None:
                raise ValueError("Consensus-relative heterogeneity requires base coordinates and values")
            coords = base_coords + self.factor * self.coordinate_scale * delta_coords
            values = nnx.relu(base_values + self.amplitude_scale * delta_values)
        else:
            coords = self.factor * (self.coords + delta_coords)
            values = nnx.relu(self.reference_values + delta_values)

        return coords, values

    def decode_volume(self, x, filter=True, sigma=1.0, base_coords=None, base_values=None,
                      analytic=False):
        # Decode volume values
        coords, values = self.__call__(x, base_coords=base_coords, base_values=base_values)

        # Displace coordinates
        coords = coords + self.factor

        # Place values on grid
        grids = jnp.zeros((x.shape[0], self.volume_size, self.volume_size, self.volume_size))

        # Scatter volume
        bposf = jnp.floor(coords)
        bposi = bposf.astype(jnp.int32)
        bposf = coords - bposf

        bamp0 = values * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp1 = values * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp2 = values * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp3 = values * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp4 = values * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp5 = values * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp6 = values * (bposf[:, :, 0]) * (bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp7 = values * (bposf[:, :, 0]) * (bposf[:, :, 1]) * (bposf[:, :, 2])

        bamp = jnp.concat([bamp0, bamp1, bamp2, bamp3, bamp4, bamp5, bamp6, bamp7], axis=1)
        bposi = jnp.concat([bposi, bposi + jnp.array((1, 0, 0)), bposi + jnp.array((0, 1, 0)), bposi + jnp.array((0, 0, 1)),
                           bposi + jnp.array((0, 1, 1)), bposi + jnp.array((1, 0, 1)), bposi + jnp.array((1, 1, 0)), bposi + jnp.array((1, 1, 1))], axis=1)

        def scatter_volume(vol, bpos_i, bamp_i):
            return vol.at[bpos_i[..., 2], bpos_i[..., 1], bpos_i[..., 0]].add(bamp_i)

        grids = jax.vmap(scatter_volume, in_axes=(0, 0, 0))(grids, bposi, bamp)

        # Filter volume
        if filter:
            low_pass = low_pass_3d_analytic if analytic else low_pass_3d
            grids = jax.vmap(low_pass, in_axes=(0, None))(grids, sigma)

        return grids

class PhysDecoder:
    def __init__(self, xsize, render_chunk_size=0, fused_envelope=False):
        self.xsize = xsize
        self.render_chunk_size = int(render_chunk_size or 0)
        self.fused_envelope = bool(fused_envelope)

    def _scatter(self, values, coords, xsize, rotations_flat, shifts_flat, dtype):
        # Volume factor
        factor = 0.5 * xsize

        # Apply rotation matrices
        coords = jnp.matmul(coords, rearrange(rotations_flat, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts_flat[:, None, :] + factor

        # Scatter image
        B = rotations_flat.shape[0]
        c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
        images = jnp.zeros((B, xsize, xsize), dtype=dtype)

        bposf = jnp.floor(c_sampling)
        bposi = bposf.astype(jnp.int32)
        bposf = c_sampling - bposf

        bamp0 = values * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp1 = values * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp2 = values * (bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp3 = values * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp = jnp.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposi = jnp.concat([bposi, bposi + jnp.array((1, 0)), bposi + jnp.array((1, 1)), bposi + jnp.array((0, 1))], axis=1)

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        return jax.vmap(scatter_img)(images, bposi, bamp)

    def _scatter_chunked(self, values, coords, xsize, rotations_flat, shifts_flat, dtype):
        """Accumulate Gaussian projections in bounded-memory point blocks."""
        block = self.render_chunk_size
        n_points = coords.shape[1]
        n_blocks, tail = divmod(n_points, block)
        images = jnp.zeros((rotations_flat.shape[0], xsize, xsize), dtype=dtype)

        def render_block(values_b, coords_b):
            return self._scatter(values_b, coords_b, xsize, rotations_flat, shifts_flat, dtype)

        if n_blocks:
            def indexed_block(i):
                start = i * block
                values_b = jax.lax.dynamic_slice_in_dim(values, start, block, axis=1)
                coords_b = jax.lax.dynamic_slice_in_dim(coords, start, block, axis=1)
                return render_block(values_b, coords_b)

            indexed_block = jax.checkpoint(indexed_block)
            images, _ = jax.lax.scan(
                lambda acc, i: (acc + indexed_block(i), None), images, jnp.arange(n_blocks))

        if tail:
            start = n_blocks * block
            images = images + jax.checkpoint(render_block)(values[:, start:], coords[:, start:])
        return images

    def __call__(self, x, values, coords, xsize, rotations, shifts, ctf, ctf_type, std,
                 filter=True, render_size=None):
        render_size = xsize if render_size is None else int(render_size)
        scale = render_size / xsize

        rotations_flat = rearrange(rotations, "b n m d -> (b n) m d")
        shifts_flat = rearrange(shifts, "b n m -> (b n) m") * scale
        coords = coords * scale

        blocked = 0 < self.render_chunk_size < coords.shape[1]
        scatter = self._scatter_chunked if blocked else self._scatter
        images = scatter(values, coords, render_size, rotations_flat, shifts_flat, x.dtype)

        apply_ctf = ctf_type in ["apply", "wiener", "squared"]
        if apply_ctf:
            ctf = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf = rearrange(ctf, "b n w h -> (b n) w h")

        if self.fused_envelope:
            # Analytic splat envelope and CTF in a single Fourier pass: exact for
            # any std (the tap kernel below truncates past +-4 px) and one fewer
            # filtering pass per projection.
            images = gaussianCTFFilter(images, sigma=std * scale if filter else None,
                                       ctf=ctf if apply_ctf else None, pad_factor=2)
        else:
            # Gaussian filter (needed by forward interpolation)
            if filter:
                images = dm_pix.gaussian_blur(images[..., None], std * scale, kernel_size=9)[..., 0]

            # Apply CTF
            if apply_ctf:
                images = ctfFilter(images, ctf, pad_factor=2)

        images = rearrange(images, "(b n) w h -> b n w h", b=rotations.shape[0], n=rotations.shape[1])

        return images

class ReconSIREN(nnx.Module):

    @save_config
    def __init__(self, coords, values, xsize, sr, bank_size=1024, ctf_type="apply", lat_dim=8, sigma=1.0,
                 symmetry_group="c1", refine_current_assignment=False, learn_delta_volume=True, num_components=18,
                 use_anchor_rotations=True, optimization_profile="legacy", pose_head_rank=None,
                 pose_spatial_pool=None,
                 het_encoder_architecture=None,
                 consensus_parameterization=None, render_chunk_size=None,
                 candidate_chunk_size=None, coarse_topk=None, coarse_scale=None, coarse_gaussians=None,
                 heterogeneity_profile="legacy", het_encoder_size=None,
                 het_residual_to_consensus=None, het_center_decoder=None,
                 het_coordinate_scale=1.0, het_amplitude_scale=1.0,
                 het_loss_scales=None, het_loss_weights=None, het_mask_radius=None,
                 het_normalize_target=None, het_variance_weight=None,
                 het_covariance_weight=None, het_min_std=0.1,
                 het_start_epoch=None, het_freeze_consensus=None,
                 het_latent_bank_size=2048,
                 fused_envelope=False, sigma_bounds=None, per_candidate_shifts=False,
                 *, rngs: nnx.Rngs, **kwargs):
        super(ReconSIREN, self).__init__()
        aggressive = optimization_profile == "aggressive"
        anti_collapse = heterogeneity_profile == "anti_collapse"
        if optimization_profile not in ("legacy", "aggressive"):
            raise ValueError("optimization_profile must be 'legacy' or 'aggressive'")
        if heterogeneity_profile not in ("legacy", "anti_collapse"):
            raise ValueError("heterogeneity_profile must be 'legacy' or 'anti_collapse'")
        pose_head_rank = (128 if aggressive else 0) if pose_head_rank is None else int(pose_head_rank)
        pose_spatial_pool = (4 if aggressive else 1) if pose_spatial_pool is None else int(pose_spatial_pool)
        if het_encoder_architecture is None:
            het_encoder_architecture = "resize" if anti_collapse else ("convstem" if aggressive else "legacy")
        consensus_parameterization = ("direct" if aggressive else "network") if consensus_parameterization is None else consensus_parameterization
        render_chunk_size = (2048 if aggressive else 0) if render_chunk_size is None else int(render_chunk_size)
        candidate_chunk_size = (3 if aggressive else 0) if candidate_chunk_size is None else int(candidate_chunk_size)
        coarse_topk = (4 if aggressive else num_components) if coarse_topk is None else int(coarse_topk)
        coarse_scale = (0.5 if aggressive else 1.0) if coarse_scale is None else float(coarse_scale)
        coarse_gaussians = (2048 if aggressive else 0) if coarse_gaussians is None else int(coarse_gaussians)
        default_het_size = min(128, max(16, (int(xsize) // 16) * 16))
        het_encoder_size = (default_het_size if anti_collapse else 64) if het_encoder_size is None else int(het_encoder_size)
        het_residual_to_consensus = anti_collapse if het_residual_to_consensus is None else bool(het_residual_to_consensus)
        het_center_decoder = anti_collapse if het_center_decoder is None else bool(het_center_decoder)
        het_mask_radius = (0.45 if anti_collapse else 0.0) if het_mask_radius is None else float(het_mask_radius)
        het_normalize_target = anti_collapse if het_normalize_target is None else bool(het_normalize_target)
        het_variance_weight = (1e-2 if anti_collapse else 0.0) if het_variance_weight is None else float(het_variance_weight)
        het_covariance_weight = (1e-3 if anti_collapse else 0.0) if het_covariance_weight is None else float(het_covariance_weight)
        het_start_epoch = (5 if anti_collapse else 0) if het_start_epoch is None else int(het_start_epoch)
        het_freeze_consensus = anti_collapse if het_freeze_consensus is None else bool(het_freeze_consensus)

        if het_loss_scales is None:
            het_loss_scales = tuple(dict.fromkeys(
                [size for size in (64, 128) if size < xsize] + [int(xsize)]))
        else:
            het_loss_scales = tuple(int(size) for size in het_loss_scales)
        if any(size <= 0 or size > xsize for size in het_loss_scales):
            raise ValueError("heterogeneity loss scales must be in [1, xsize]")
        if het_loss_weights is None:
            if anti_collapse and len(het_loss_scales) == 3:
                het_loss_weights = (0.5, 0.3, 0.2)
            else:
                het_loss_weights = tuple(1.0 for _ in het_loss_scales)
        else:
            het_loss_weights = tuple(float(weight) for weight in het_loss_weights)
        if len(het_loss_scales) != len(het_loss_weights) or not het_loss_scales:
            raise ValueError("heterogeneity loss scales and weights must have equal non-zero length")
        weight_sum = sum(het_loss_weights)
        if weight_sum <= 0.0 or any(weight < 0.0 for weight in het_loss_weights):
            raise ValueError("heterogeneity loss weights must be non-negative and sum to > 0")
        het_loss_weights = tuple(weight / weight_sum for weight in het_loss_weights)
        if het_encoder_architecture == "resize" and (het_encoder_size < 16 or het_encoder_size % 16):
            raise ValueError("het_encoder_size must be a positive multiple of 16 for the resize encoder")
        if not 0.0 <= het_mask_radius <= 0.5:
            raise ValueError("het_mask_radius must be in [0, 0.5]")

        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.optimization_profile = optimization_profile
        self.heterogeneity_profile = heterogeneity_profile
        if coarse_scale <= 0.0 or coarse_scale > 1.0:
            raise ValueError("coarse_scale must be in (0, 1]")
        self.coarse_topk = max(1, min(int(coarse_topk), int(num_components)))
        self.coarse_scale = float(coarse_scale)
        self.coarse_gaussians = max(0, int(coarse_gaussians))
        self.candidate_chunk_size = max(0, int(candidate_chunk_size))
        self.het_loss_scales = het_loss_scales
        self.het_loss_weights = het_loss_weights
        self.het_mask_radius = het_mask_radius
        self.het_normalize_target = het_normalize_target
        self.het_variance_weight = het_variance_weight
        self.het_covariance_weight = het_covariance_weight
        self.het_min_std = float(het_min_std)
        self.het_start_epoch = max(0, het_start_epoch)
        self.het_freeze_consensus = het_freeze_consensus
        self.symmetry_matrices = symmetry_matrices(symmetry_group)
        self.refine_current_assignment = refine_current_assignment
        self.learn_delta_volume = learn_delta_volume
        self.fused_envelope = bool(fused_envelope)
        if sigma_bounds is not None:
            lo, hi = float(sigma_bounds[0]), float(sigma_bounds[1])
            if not 0.0 < lo < hi:
                raise ValueError("sigma_bounds must satisfy 0 < min < max")
            sigma_bounds = (lo, hi)
        self.sigma_bounds = sigma_bounds
        self.encoder_pose = EncoderPose(self.xsize, num_components=num_components, refine_current_assignment=refine_current_assignment,
                                        use_anchor_rotations=use_anchor_rotations, low_rank=pose_head_rank,
                                        spatial_pool=pose_spatial_pool,
                                        per_candidate_shifts=per_candidate_shifts, rngs=rngs)
        self.encoder_het = EncoderHet(self.xsize, lat_dim=lat_dim,
                                      architecture=het_encoder_architecture,
                                      encoder_size=het_encoder_size, rngs=rngs)
        self.delta_volume_decoder = DeltaVolumeDecoder(coords=coords, values=values, volume_size=self.xsize,
                                                       learn_delta_volume=learn_delta_volume,
                                                       parameterization=consensus_parameterization, rngs=rngs)
        self.delta_het_decoder = HetVolumeDecoder(
            coords=coords, values=values, n_gaussians=coords.shape[0], lat_dim=lat_dim,
            volume_size=self.xsize, residual_to_consensus=het_residual_to_consensus,
            center_decoder=het_center_decoder, coordinate_scale=het_coordinate_scale,
            amplitude_scale=het_amplitude_scale, small_final_init=anti_collapse,
            rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize, render_chunk_size=render_chunk_size,
                                        fused_envelope=self.fused_envelope)

        # Gaussian std. When bounds are active the parameter stores a logit and
        # the width lives on a sigmoid between them: a plain clip would zero the
        # gradient at the bound and freeze sigma there for good.
        sigma_init = jnp.asarray(sigma, dtype=jnp.float32)
        if self.sigma_bounds is None:
            self.log_std = nnx.Param(jnp.log(sigma_init))
        else:
            lo, hi = self.sigma_bounds
            frac = jnp.clip((sigma_init - lo) / (hi - lo), 1e-3, 1.0 - 1e-3)
            self.log_std = nnx.Param(jnp.log(frac) - jnp.log1p(-frac))

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.memory_bank = MemoryBank(
            array_init=jnp.zeros((bank_size, 3), dtype=jnp.float32))
        winner_init = jax.random.normal(rngs.params(), (bank_size, 3))
        winner_init = winner_init / jnp.linalg.norm(
            winner_init, axis=-1, keepdims=True)
        self.winner_memory_bank = MemoryBank(
            array_init=winner_init)
        self.candidate_bank_count = nnx.Variable(jnp.array(0, dtype=jnp.int32))
        self.winner_bank_count = nnx.Variable(jnp.array(0, dtype=jnp.int32))
        if anti_collapse:
            latent_bank_size = max(1, int(het_latent_bank_size))
            self.latent_memory_bank = MemoryBank(
                array_init=jnp.zeros((latent_bank_size, lat_dim), dtype=jnp.float32))
            self.latent_bank_count = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    def __call__(self, x, rngs: nnx.Rngs = None, **kwargs):
        # TODO: Return only best angles
        return self.encoder_pose(x, rngs=rngs)
    
    def get_std(self):
        raw = self.log_std.get_value()
        if self.sigma_bounds is None:
            return jnp.exp(raw)
        lo, hi = self.sigma_bounds
        return lo + (hi - lo) * jax.nn.sigmoid(raw)

    def decode_image(self, x, labels, md, ctf_type=None):
        # Precompute batch CTFs
        if self.ctf_type not in (None, "None"):
            defocusU = md["ctfDefocusU"][labels]
            defocusV = md["ctfDefocusV"][labels]
            defocusAngle = md["ctfDefocusAngle"][labels]
            cs = md["ctfSphericalAberration"][labels]
            kv = md["ctfVoltage"][labels][0]
            ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                             self.sr, [2 * self.xsize, int(2 * 0.5 * self.xsize + 1)],
                             x.shape[0], True)
        else:
            ctf = jnp.ones([x.shape[0], 2 * self.xsize, int(2.0 * 0.5 * self.xsize + 1)], dtype=x.dtype)

        if self.ctf_type in ["apply", "squared"]:
            # Wiener filter
            x = prepare_image_cryocrab(x, ctf)
            # x = prepare_image_wiener(x, ctf)

        # Encode images
        rotations, shifts = self(x)

        # Decode volume
        coords, values = self.delta_volume_decoder()

        # Generate projections
        images_corrected = self.phys_decoder(x, values, coords, self.xsize, rotations, shifts, ctf,
                                             ctf_type, self.get_std())

        return images_corrected

    def decode_het_volume(self, x, filter=True):
        if x.ndim == 4:
            _, x, _ = self.encoder_het(x)
        elif x.ndim == 3:
            _, x, _ = self.encoder_het(x[None, ...])
        elif x.ndim == 1:
            x = x[None, ...]

        base_coords = base_values = None
        if self.delta_het_decoder.residual_to_consensus:
            base_coords, base_values = self.delta_volume_decoder()
        vol = self.delta_het_decoder.decode_volume(
            x, filter=filter, sigma=self.get_std(),
            base_coords=base_coords, base_values=base_values,
            analytic=self.fused_envelope)

        return vol


def _candidate_reconstruction_losses(images, targets, ctf, ctf_type,
                                     normalize_target=True, return_prepared=False,
                                     scoring_size=None):
    """Per-particle, per-candidate loss without materialising target copies."""
    target = targets[..., 0] if targets.shape[-1] == 1 else targets
    predicted = images[..., 0] if images.shape[-1] == 1 else images
    n_candidates = predicted.shape[1]

    if ctf_type == "wiener":
        target = wiener2DFilter(target, ctf, pad_factor=2)
        ctf_candidates = jnp.broadcast_to(
            ctf[:, None, :], (ctf.shape[0], n_candidates, ctf.shape[1], ctf.shape[2]))
        predicted = wiener2DFilter(
            rearrange(predicted, "b n w h -> (b n) w h"),
            rearrange(ctf_candidates, "b n w h -> (b n) w h"), pad_factor=2)
        predicted = rearrange(predicted, "(b n) w h -> b n w h",
                              b=target.shape[0], n=n_candidates)
    elif ctf_type == "squared":
        target = ctfFilter(target, ctf, pad_factor=2)
        ctf_candidates = jnp.broadcast_to(
            ctf[:, None, :], (ctf.shape[0], n_candidates, ctf.shape[1], ctf.shape[2]))
        predicted = ctfFilter(
            rearrange(predicted, "b n w h -> (b n) w h"),
            rearrange(ctf_candidates, "b n w h -> (b n) w h"), pad_factor=2)
        predicted = rearrange(predicted, "(b n) w h -> b n w h",
                              b=target.shape[0], n=n_candidates)

    if scoring_size is not None and scoring_size < target.shape[-1]:
        predicted = _resize_candidate_images(predicted, scoring_size)
        target = _resize_candidate_images(target[:, None, ...], scoring_size)[:, 0]

    # The legacy path normalized identical target copies independently.  Taking
    # the same reduction once per particle produces the same value and lets
    # broadcasting remain a view throughout the fused subtraction.
    if normalize_target:
        target = standard_normalization(target)
    losses = jnp.mean(
        jnp.square(predicted - target[:, None, ...]), axis=(-2, -1))
    if return_prepared:
        return losses, predicted, target[:, None, ...]
    return losses


def _prepare_heterogeneity_images(images, targets, ctf, ctf_type, normalize_target):
    """Apply the legacy CTF-loss convention and return candidate-shaped arrays."""
    target = targets[..., 0] if targets.shape[-1] == 1 else targets
    predicted = images[..., 0] if images.shape[-1] == 1 else images
    n_candidates = predicted.shape[1]

    if ctf_type == "wiener":
        target = wiener2DFilter(target, ctf, pad_factor=2)
        ctf_candidates = jnp.broadcast_to(
            ctf[:, None, :], (ctf.shape[0], n_candidates, ctf.shape[1], ctf.shape[2]))
        predicted = wiener2DFilter(
            rearrange(predicted, "b n w h -> (b n) w h"),
            rearrange(ctf_candidates, "b n w h -> (b n) w h"), pad_factor=2)
        predicted = rearrange(predicted, "(b n) w h -> b n w h",
                              b=target.shape[0], n=n_candidates)
    elif ctf_type == "squared":
        target = ctfFilter(target, ctf, pad_factor=2)
        ctf_candidates = jnp.broadcast_to(
            ctf[:, None, :], (ctf.shape[0], n_candidates, ctf.shape[1], ctf.shape[2]))
        predicted = ctfFilter(
            rearrange(predicted, "b n w h -> (b n) w h"),
            rearrange(ctf_candidates, "b n w h -> (b n) w h"), pad_factor=2)
        predicted = rearrange(predicted, "(b n) w h -> b n w h",
                              b=target.shape[0], n=n_candidates)

    if normalize_target:
        target = standard_normalization(target)
    return predicted, target[:, None, ...]


def _resize_candidate_images(images, size):
    if images.shape[-1] == size and images.shape[-2] == size:
        return images
    flat = rearrange(images, "b n h w -> (b n) h w 1")
    flat = jax.image.resize(
        flat, (flat.shape[0], size, size, 1), method="lanczos3", antialias=True)
    return rearrange(flat[..., 0], "(b n) h w -> b n h w", b=images.shape[0])


def _circular_loss_mask(size, radius, dtype):
    if radius <= 0.0:
        return jnp.ones((size, size), dtype=dtype)
    axis = (jnp.arange(size, dtype=jnp.float32) + 0.5) / size - 0.5
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    return (xx * xx + yy * yy <= radius * radius).astype(dtype)


def _heterogeneity_reconstruction_loss(images, targets, ctf, ctf_type,
                                       scales, weights, mask_radius,
                                       normalize_target):
    """Masked multiresolution reconstruction loss for conformational residuals."""
    predicted, target = _prepare_heterogeneity_images(
        images, targets, ctf, ctf_type, normalize_target)
    loss = jnp.asarray(0.0, dtype=predicted.dtype)
    for size, weight in zip(scales, weights):
        predicted_level = _resize_candidate_images(predicted, size)
        target_level = _resize_candidate_images(target, size)
        mask = _circular_loss_mask(size, mask_radius, predicted.dtype)
        squared = jnp.square(predicted_level - target_level) * mask[None, None, ...]
        loss = loss + weight * jnp.sum(squared) / (
            predicted.shape[0] * predicted.shape[1] * jnp.maximum(jnp.sum(mask), 1.0))
    return loss


def _latent_variance_covariance_loss(latent, minimum_std, bank=None, bank_count=None):
    """VICReg-style anti-collapse statistics, optionally backed by a frozen bank."""
    latent_f32 = latent.astype(jnp.float32)
    if bank is None:
        total = jnp.asarray(latent.shape[0], dtype=jnp.float32)
        mean = jnp.mean(latent_f32, axis=0)
        centered = latent_f32 - mean
        covariance = centered.T @ centered / jnp.maximum(total - 1.0, 1.0)
        variance = jnp.mean(jnp.square(centered), axis=0)
    else:
        bank_f32 = jax.lax.stop_gradient(bank.astype(jnp.float32))
        valid = (jnp.arange(bank.shape[0]) < bank_count).astype(jnp.float32)[:, None]
        total = latent.shape[0] + jnp.sum(valid)
        mean = (jnp.sum(latent_f32, axis=0) + jnp.sum(bank_f32 * valid, axis=0)) / total
        centered = latent_f32 - mean
        bank_centered = (bank_f32 - mean) * valid
        covariance = (centered.T @ centered + bank_centered.T @ bank_centered) / jnp.maximum(total - 1.0, 1.0)
        variance = (jnp.sum(jnp.square(centered), axis=0)
                    + jnp.sum(jnp.square(bank_centered), axis=0)) / total

    std = jnp.sqrt(variance + 1e-6)
    variance_loss = jnp.mean(jnp.square(jax.nn.relu(minimum_std - std)))
    off_diagonal = covariance - jnp.diag(jnp.diag(covariance))
    covariance_loss = jnp.mean(jnp.square(off_diagonal))
    return variance_loss, covariance_loss, jnp.mean(std)


def _whitened_reconstruction_loss(predicted, target, whitening_filter):
    """Noise-whitened MSE so every frequency shell carries comparable gradient.

    The plain real-space MSE is dominated by the low-frequency shells where the
    cryo-EM signal (and the coloured noise) concentrates, leaving essentially no
    gradient pressure on the high-resolution shells. Dividing both images by the
    dataset noise amplitude spectrum equalises the per-shell SNR before the
    residual is taken.

    ``predicted`` is candidate-shaped ``(B, N, H, W)``, ``target`` is
    ``(B, 1, H, W)`` and ``whitening_filter`` lies on the unshifted ``fft2``
    grid (see :func:`hax.utils.whitening_filter_2d`).
    """
    predicted_white = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(predicted) * whitening_filter))
    target_white = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(target) * whitening_filter))
    # Dimensionless residual: normalising by the whitened target power keeps the
    # loss on the same O(1) scale as the plain normalized MSE it blends with.
    scale = jnp.sqrt(jnp.mean(jnp.square(target_white), axis=(-2, -1), keepdims=True)) + 1e-8
    return jnp.mean(jnp.square((predicted_white - target_white) / scale), axis=(-2, -1))


def _sharpen_gaussian_envelope(volume, sigma, reg=0.02):
    """Wiener inverse of the splat envelope ``exp(-2 pi^2 sigma^2 f^2)``.

    The rendered map always carries the Gaussian splat envelope, so its
    amplitudes fall off like a B-factor even when the fitted point cloud holds
    sharper structure. This divides the envelope back out with a bounded-gain
    Wiener filter (max boost ~ ``1 / (2 * sqrt(reg))``), the same operation as
    conventional post-hoc map sharpening.
    """
    shape = volume.shape
    fz = jnp.fft.fftfreq(shape[0])
    fy = jnp.fft.fftfreq(shape[1])
    fx = jnp.fft.rfftfreq(shape[2])
    f_sq = (fz[:, None, None] ** 2 + fy[None, :, None] ** 2
            + fx[None, None, :] ** 2)
    sigma_sq = jnp.square(jnp.asarray(sigma, jnp.float32)).reshape(())
    envelope = jnp.exp(-2.0 * jnp.pi ** 2 * sigma_sq * f_sq)
    gain = (1.0 + reg) * envelope / (jnp.square(envelope) + reg)
    return jnp.fft.irfftn(jnp.fft.rfftn(volume) * gain, s=shape)


def _estimate_particle_extent(images, threshold=0.1, margin=1.15):
    """Estimate the particle radius in pixels from raw images alone.

    Orientation-free: the per-pixel variance across the batch carries the
    particle signal (projections change with pose) on top of a flat noise
    floor taken from the outermost radial shells. Needs no reference volume
    and no mask. Returns ``None`` when no clear extent stands out.
    """
    x = np.asarray(images, np.float32)
    if x.ndim == 4:
        x = x[..., 0]
    x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-8)
    variance = x.var(axis=0)

    h, w = variance.shape
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2).astype(np.int32)
    n_shells = int(r.max()) + 1
    profile = np.bincount(r.ravel(), weights=variance.ravel(), minlength=n_shells)
    counts = np.bincount(r.ravel(), minlength=n_shells)
    profile = profile / np.maximum(counts, 1)

    max_radius = min(h, w) // 2
    if max_radius < 8:
        return None
    profile = profile[:max_radius]
    outer = profile[int(0.85 * max_radius):]
    noise_floor = np.median(outer)
    excess = profile - noise_floor
    # Refuse rather than hallucinate: the variance peak must stand well clear
    # of the outer-shell scatter (robust MAD scale) to count as a particle.
    noise_scale = 1.4826 * np.median(np.abs(outer - noise_floor))
    peak = excess.max()
    if peak <= 5.0 * max(noise_scale, 1e-12):
        return None
    above = np.flatnonzero(excess > threshold * peak)
    if above.size == 0:
        return None
    radius = float(above.max()) * margin
    return float(np.clip(radius, 4.0, 0.95 * max_radius))


def _geometry_prior_losses(coords, values, neighbor_indices, sigma):
    """kNN spacing and amplitude-smoothness priors on the consensus cloud.

    Spacing: a mass-weighted penalty when an edge stretches past ~2 sigma (the
    overlap limit for continuous rendered density) or crowds below ~0.7 sigma
    (redundant stacking on blobs). Smoothness: a graph Laplacian on amplitudes
    so neighbouring mass-carrying points render at similar brightness and a
    single iso-surface threshold traces the whole chain. The edge weights and
    normalisations are stop-gradient so neither term can be cheated by simply
    shrinking amplitudes.
    """
    positions = coords[0]
    amplitudes = values[0]
    neighbor_positions = positions[neighbor_indices]  # (N, k, 3)
    distances = jnp.sqrt(jnp.sum(jnp.square(
        positions[:, None, :] - neighbor_positions), axis=-1) + 1e-12)
    sigma = jax.lax.stop_gradient(jnp.mean(sigma))

    edge_weights = jnp.sqrt(amplitudes[:, None] * amplitudes[neighbor_indices] + 1e-12)
    edge_weights = jax.lax.stop_gradient(edge_weights / (jnp.mean(edge_weights) + 1e-12))
    # The gap term applies to the NEAREST neighbour only (neighbor_indices must
    # be distance-sorted): connectivity means d_nn < ~2 sigma. Demanding it of
    # all k neighbours would penalise chain topology itself and squeeze
    # filaments into clumps. Crowding applies to every neighbour.
    gap = jax.nn.relu(distances[:, 0] - 2.0 * sigma) / sigma
    crowd = jax.nn.relu(0.7 * sigma - distances) / sigma
    spacing_loss = (jnp.mean(edge_weights[:, 0] * jnp.square(gap))
                    + jnp.mean(edge_weights * jnp.square(crowd)))

    amplitude_scale = jax.lax.stop_gradient(jnp.mean(amplitudes) + 1e-12)
    smoothness_loss = jnp.mean(edge_weights * jnp.square(
        (amplitudes[:, None] - amplitudes[neighbor_indices]) / amplitude_scale))
    return spacing_loss, smoothness_loss


def _support_loss(coords, values, center, radius, sigma):
    """Mass outside the shrink-wrapped spherical support, in units of sigma.

    The support is derived from the converging cloud itself (no mask needed);
    the term is zero for any point inside it, so it only suppresses dust.
    """
    positions = coords[0]
    amplitudes = values[0]
    distances = jnp.sqrt(jnp.sum(jnp.square(positions - center[None, :]), axis=-1) + 1e-12)
    sigma = jax.lax.stop_gradient(jnp.mean(sigma))
    outside = jax.nn.relu(distances - radius) / sigma
    total_mass = jax.lax.stop_gradient(jnp.sum(amplitudes) + 1e-12)
    return jnp.sum(amplitudes * outside) / total_mass


@partial(jax.jit, donate_argnums=(1,))
def recycle_dead_points_reconsiren(graphdef, state, key, dead_fraction=0.05,
                                   new_value_fraction=0.25):
    """Teleport amplitude-dead Gaussians next to mass-carrying ones.

    Fixed-N counterpart of Gaussian-splatting densify/prune: points whose
    amplitude collapsed below ``dead_fraction`` of the mean are resampled next
    to donors drawn proportionally to mass (one splat width of jitter) and
    restart at a small amplitude, relocating capacity onto the structure
    without changing any array shape. Only valid for the 'direct' consensus
    parameterization. Stale Adam moments of moved points are left in place:
    the inflated second moment just makes their first few updates cautious.
    """
    model, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)
    decoder = model.delta_volume_decoder

    reference_coords = decoder.coords[0]
    reference_values = decoder.reference_values[0]
    delta_coords = decoder.delta_coords.get_value()
    delta_values = decoder.delta_values.get_value()

    values = jax.nn.relu(reference_values + delta_values)
    mean_value = jnp.mean(values)
    dead = values < dead_fraction * mean_value

    donor_key, jitter_key = jax.random.split(jax.random.fold_in(key, 1))
    donors = jax.random.categorical(
        donor_key, jnp.log(values + 1e-12), shape=(values.shape[0],))
    positions = reference_coords + delta_coords
    sigma_normalized = jnp.mean(model.get_std()) / decoder.factor
    jitter = sigma_normalized * jax.random.normal(jitter_key, positions.shape)
    new_delta_coords = positions[donors] + jitter - reference_coords
    new_delta_values = new_value_fraction * mean_value - reference_values

    decoder.delta_coords.value = jnp.where(dead[:, None], new_delta_coords, delta_coords)
    decoder.delta_values.value = jnp.where(dead, new_delta_values, delta_values)

    state = nnx.state((model, optimizer_pose, optimizer_volume, optimizer_het))
    return state, jnp.sum(dead)


def _gather_candidates(values, indices):
    return values[jnp.arange(values.shape[0])[:, None], indices]


def _score_candidates(model, x, values, coords, rotations, shifts, ctf,
                      scoring_size=None, std=None):
    """Render candidates once and return curriculum and full-resolution losses."""
    chunk = model.candidate_chunk_size
    n_candidates = rotations.shape[1]
    scoring_size = model.xsize if scoring_size is None else scoring_size
    std = model.get_std() if std is None else std
    if chunk <= 0 or chunk >= n_candidates:
        images = model.phys_decoder(
            x, values, coords, model.xsize, rotations, shifts, ctf,
            model.ctf_type, std)
        full_losses = _candidate_reconstruction_losses(
            images, x, ctf, model.ctf_type)
        if scoring_size >= model.xsize:
            return full_losses, full_losses
        scoring_losses = _candidate_reconstruction_losses(
            images, x, ctf, model.ctf_type, scoring_size=scoring_size)
        return scoring_losses, full_losses

    scoring_losses = []
    full_losses = []
    for start in range(0, n_candidates, chunk):
        images = model.phys_decoder(
            x, values, coords, model.xsize,
            rotations[:, start:start + chunk], shifts[:, start:start + chunk],
            ctf, model.ctf_type, std)
        full_chunk = _candidate_reconstruction_losses(
            images, x, ctf, model.ctf_type)
        full_losses.append(full_chunk)
        if scoring_size >= model.xsize:
            scoring_losses.append(full_chunk)
        else:
            scoring_losses.append(_candidate_reconstruction_losses(
                images, x, ctf, model.ctf_type, scoring_size=scoring_size))
    return (jnp.concatenate(scoring_losses, axis=1),
            jnp.concatenate(full_losses, axis=1))


def _consensus_multiscale_size_and_weight(
        image_size, hard_step, steps_per_epoch, curriculum_epochs, scales):
    """Return a discrete loss size and a smooth curriculum multiplier."""
    curriculum_steps = float(curriculum_epochs) * int(steps_per_epoch)
    if curriculum_steps <= 0.0 or hard_step >= curriculum_steps:
        return int(image_size), 0.0
    progress = max(float(hard_step), 0.0) / curriculum_steps
    stage = min(int(progress * len(scales)), len(scales) - 1)
    size = min(int(image_size), max(8, int(round(image_size * scales[stage]))))
    return size, 1.0 - progress


@partial(jax.jit, static_argnames=("use_tau", "assignment_mode",
                                   "apply_candidate_coverage", "candidate_coverage_bins",
                                   "candidate_bank_samples",
                                   "candidate_scoring_size",
                                   "consensus_multiscale_size",
                                   "apply_loss_whitening", "apply_amplitude_l1",
                                   "apply_geometry_priors", "apply_support",
                                   "train_pose_volume", "train_heterogeneity", "return_metrics",
                                   "return_pose_diagnostics"),
         donate_argnums=(1,))
def train_step_reconsiren(graphdef, state, x, labels, md, key, tau=0.0001,
                          use_tau=False, assignment_mode="hard", lambda_uniform=0.1,
                          apply_candidate_coverage=False,
                          candidate_coverage_weight=0.0,
                          candidate_coverage_bins=256,
                          candidate_coverage_kappa=32.0,
                          candidate_bank_samples=1024,
                          candidate_bank_mix=0.5,
                          candidate_scoring_size=0,
                          consensus_multiscale_size=0,
                          consensus_multiscale_weight=0.0,
                          apply_loss_whitening=False,
                          whiten_weight=0.0,
                          whiten_filter=None,
                          apply_amplitude_l1=False,
                          amplitude_l1_weight=0.0,
                          extra_blur=0.0,
                          apply_geometry_priors=False,
                          spacing_weight=0.0,
                          smoothness_weight=0.0,
                          neighbor_indices=None,
                          apply_support=False,
                          support_center=None,
                          support_radius=0.0,
                          support_weight=0.0,
                          train_pose_volume=True, train_heterogeneity=True,
                          return_metrics=False, return_pose_diagnostics=False):
    model, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)
    scoring_size = (model.xsize if candidate_scoring_size <= 0
                    else min(int(candidate_scoring_size), model.xsize))
    multiscale_size = (model.xsize if consensus_multiscale_size <= 0
                       else min(int(consensus_multiscale_size), model.xsize))

    # Random keys
    key, coverage_key, swd_key, choice_key, distributions_key = jax.random.split(key, 5)

    def loss_fn(model, x):
        # Correct CTF in images for encoder if needed
        if model.ctf_type in ["apply", "squared"]:
            x_ctf_corrected = prepare_image_cryocrab(x, ctf)
            # x_ctf_corrected = prepare_image_wiener(x, ctf)
        else:
            x_ctf_corrected = x

        # Pose/consensus are always evaluated because they provide the selected
        # orientation and the residual base for the heterogeneity-only stage.
        rotations, shifts = model.encoder_pose(x_ctf_corrected)
        coords, values = model.delta_volume_decoder()

        # Coarse-to-fine render width: extra blur in quadrature on top of the
        # learned splat, annealed to zero as the pose curriculum finishes, so
        # the cloud cannot burn in fine detail while poses are still coarse.
        std_eff = jnp.sqrt(jnp.square(model.get_std()) + jnp.square(extra_blur))

        anchor_deviation = jnp.asarray(0.0, dtype=rotations.dtype)
        if (model.encoder_pose.use_anchor_rotations
                and not model.encoder_pose.refine_current_assignment):
            anchor_directions = model.encoder_pose.anchor_rotations[:, :, 2]
            direction_dot = jnp.sum(
                rotations[..., :, 2] * anchor_directions[None, ...], axis=-1)
            anchor_deviation = jnp.mean(jnp.rad2deg(
                jnp.arccos(jnp.clip(direction_dot, -1.0, 1.0))))

        # Refine current assignment (if provided)
        # rotations = jnp.matmul(rotations, current_rotations[:, None, :, :])
        rotations = jnp.matmul(current_rotations[:, None, :, :], rotations)  # TODO: The two options seem to work?
        shifts = current_shifts[:, None, :] + shifts

        # Random symmetry matrices
        random_indices = jax.random.choice(choice_key, jnp.arange(model.symmetry_matrices.shape[0]), shape=(rotations.shape[0],))
        rotations = jnp.matmul(jnp.transpose(model.symmetry_matrices[random_indices], (0, 2, 1))[:, None, :, :], rotations)

        # Low-frequency curriculum scoring evaluates all candidates.  The
        # optional coarse screen remains available once full-resolution scoring
        # resumes, and is disabled during legacy stochastic exploration.
        rotations_eval, shifts_eval = rotations, shifts
        candidate_head_indices = jnp.broadcast_to(
            jnp.arange(rotations.shape[1], dtype=jnp.int32), rotations.shape[:2])
        if (not use_tau and assignment_mode == "hard"
                and scoring_size >= model.xsize
                and model.coarse_topk < rotations.shape[1]):
            screen_size = max(8, int(round(model.xsize * model.coarse_scale)))
            x_screen = jax.image.resize(
                x, (x.shape[0], screen_size, screen_size, x.shape[-1]), method="bilinear")
            coords_screen, values_screen = coords, values
            if 0 < model.coarse_gaussians < coords.shape[1]:
                keep = model.coarse_gaussians
                idx = (jnp.arange(keep) * coords.shape[1]) // keep
                coords_screen = coords[:, idx]
                values_screen = values[:, idx] * (coords.shape[1] / keep)
            coarse_images = model.phys_decoder(
                x_screen, jax.lax.stop_gradient(values_screen),
                jax.lax.stop_gradient(coords_screen), model.xsize,
                jax.lax.stop_gradient(rotations), jax.lax.stop_gradient(shifts),
                coarse_ctf, model.ctf_type, jax.lax.stop_gradient(std_eff),
                render_size=screen_size)
            coarse_losses = _candidate_reconstruction_losses(
                coarse_images, x_screen, coarse_ctf, model.ctf_type)
            _, top_indices = jax.lax.top_k(-coarse_losses, model.coarse_topk)
            rotations_eval = _gather_candidates(rotations, top_indices)
            shifts_eval = _gather_candidates(shifts, top_indices)
            candidate_head_indices = top_indices

        # The global competition never carries gradients through candidate
        # scoring.  Only its selected pose is rerendered into the consensus.
        candidate_losses, full_candidate_losses = _score_candidates(
            model, x, jax.lax.stop_gradient(values), jax.lax.stop_gradient(coords),
            jax.lax.stop_gradient(rotations_eval), jax.lax.stop_gradient(shifts_eval),
            ctf, scoring_size=scoring_size, std=jax.lax.stop_gradient(std_eff))

        # Candidate responsibilities are used only to pick a single global
        # winner.  Categorical exploration therefore remains safe for volume
        # amplitudes while allowing the winning head to vary over time.
        if use_tau:
            responsibilities = jax.nn.softmax(-candidate_losses / tau, axis=1)
            min_indices = jax.random.categorical(
                key, jnp.log(jnp.maximum(responsibilities, 1e-12)), axis=-1)
        elif assignment_mode == "sampled":
            # Dimensionless temperature: the winner is drawn from margin-aware
            # responsibilities so symmetry breaks gradually instead of at an
            # abrupt argmin switch.
            responsibilities = _assignment_probabilities(candidate_losses, tau)
            min_indices = jax.random.categorical(
                key, jnp.log(jnp.maximum(responsibilities, 1e-12)), axis=-1)
        else:
            min_indices = jnp.argmin(candidate_losses, axis=1)
            responsibilities = jax.nn.one_hot(
                min_indices, candidate_losses.shape[1], dtype=candidate_losses.dtype)

        scoring_best_indices = jnp.argmin(candidate_losses, axis=1)
        full_best_indices = jnp.argmin(full_candidate_losses, axis=1)
        low_frequency_full_agreement = jnp.mean(
            scoring_best_indices == full_best_indices)
        full_scale = jnp.maximum(jnp.std(full_candidate_losses, axis=1), 1e-8)
        batch_indices = jnp.arange(x.shape[0])
        low_frequency_full_disadvantage_std = jnp.mean(
            (full_candidate_losses[batch_indices, scoring_best_indices]
             - full_candidate_losses[batch_indices, full_best_indices]) / full_scale)

        if return_pose_diagnostics:
            (best_indices, absolute_margin, relative_margin, standardized_margin,
             median_normalized_margin, candidate_score_entropy) = (
                _top_two_candidate_diagnostics(candidate_losses))
        else:
            best_indices = jnp.argmin(candidate_losses, axis=1)
            diagnostic_zeros = jnp.zeros(candidate_losses.shape[0], dtype=x.dtype)
            (absolute_margin, relative_margin, standardized_margin,
             median_normalized_margin, candidate_score_entropy) = (diagnostic_zeros,) * 5
        selected_head_indices = candidate_head_indices[batch_indices, min_indices]
        best_head_indices = candidate_head_indices[batch_indices, best_indices]

        rotations_selected = rotations_eval[batch_indices, min_indices][:, None, ...]
        shifts_selected = shifts_eval[batch_indices, min_indices][:, None, ...]
        projection_rms = jnp.asarray(0.0, dtype=x.dtype)
        normalized_target_rms = jnp.asarray(0.0, dtype=x.dtype)
        low_frequency_recon_loss = jnp.asarray(0.0, dtype=x.dtype)
        reconstruction_objective = jnp.asarray(0.0, dtype=x.dtype)
        if train_pose_volume:
            selected_images = model.phys_decoder(
                x, values, coords, model.xsize, rotations_selected, shifts_selected,
                ctf, model.ctf_type, std_eff)
            # The prepared pair is free (already-computed intermediates); the
            # whitened loss and the diagnostics both reuse it.
            selected_losses, selected_predicted, selected_target = (
                _candidate_reconstruction_losses(
                    selected_images, x, ctf, model.ctf_type,
                    return_prepared=True))
            recon_loss = selected_losses.mean()
            if return_pose_diagnostics:
                projection_rms = jnp.sqrt(jnp.mean(jnp.square(selected_predicted)))
                normalized_target_rms = jnp.sqrt(jnp.mean(jnp.square(selected_target)))
            full_band_recon_loss = recon_loss
            if apply_loss_whitening:
                whitened_recon_loss = _whitened_reconstruction_loss(
                    selected_predicted, selected_target, whiten_filter).mean()
                effective_whiten_weight = jnp.clip(
                    jnp.asarray(whiten_weight, dtype=recon_loss.dtype), 0.0, 1.0)
                full_band_recon_loss = (
                    (1.0 - effective_whiten_weight) * recon_loss
                    + effective_whiten_weight * whitened_recon_loss)
            low_frequency_recon_loss = recon_loss
            effective_multiscale_weight = jnp.clip(
                jnp.asarray(consensus_multiscale_weight, dtype=recon_loss.dtype),
                0.0, 1.0)
            if multiscale_size < model.xsize:
                low_frequency_recon_loss = _candidate_reconstruction_losses(
                    selected_images, x, ctf, model.ctf_type,
                    scoring_size=multiscale_size).mean()
            reconstruction_objective = (
                (1.0 - effective_multiscale_weight) * full_band_recon_loss
                + effective_multiscale_weight * low_frequency_recon_loss)
        else:
            recon_loss = full_candidate_losses[batch_indices, min_indices].mean()
            low_frequency_recon_loss = recon_loss
            reconstruction_objective = recon_loss

        consensus_amplitude_mean = jnp.asarray(0.0, dtype=x.dtype)
        consensus_amplitude_rms = jnp.asarray(0.0, dtype=x.dtype)
        consensus_amplitude_max = jnp.asarray(0.0, dtype=x.dtype)
        consensus_active_amplitude_fraction = jnp.asarray(0.0, dtype=x.dtype)
        if return_pose_diagnostics:
            consensus_amplitude_mean = jnp.mean(values)
            consensus_amplitude_rms = jnp.sqrt(jnp.mean(jnp.square(values)))
            consensus_amplitude_max = jnp.max(values)
            consensus_active_amplitude_fraction = jnp.mean(values > 1e-8)

        min_indices_het = jnp.argmin(candidate_losses, axis=1)
        rotations_het = rotations_eval[jnp.arange(x.shape[0]), min_indices_het, :][:, None, ...]
        shifts_het = shifts_eval[jnp.arange(x.shape[0]), min_indices_het, :][:, None, ...]

        latent_dim = model.encoder_het.mean_x.out_features
        latent = jnp.zeros((x.shape[0], latent_dim), dtype=x.dtype)
        recon_het_loss = jnp.asarray(0.0, dtype=x.dtype)
        variance_loss = jnp.asarray(0.0, dtype=x.dtype)
        covariance_loss = jnp.asarray(0.0, dtype=x.dtype)
        latent_std = jnp.asarray(0.0, dtype=x.dtype)
        coordinate_rms = jnp.asarray(0.0, dtype=x.dtype)
        amplitude_rms = jnp.asarray(0.0, dtype=x.dtype)

        if train_heterogeneity:
            _, latent, _ = model.encoder_het(x_ctf_corrected, rngs=distributions_key)
            base_coords = base_values = None
            if model.delta_het_decoder.residual_to_consensus:
                base_coords = jax.lax.stop_gradient(coords)
                base_values = jax.lax.stop_gradient(values)
            coords_het, values_het = model.delta_het_decoder(
                latent, base_coords=base_coords, base_values=base_values)
            images_het = model.phys_decoder(
                x, values_het, coords_het, model.xsize,
                jax.lax.stop_gradient(rotations_het), jax.lax.stop_gradient(shifts_het),
                ctf, model.ctf_type, std_eff)[:, 0, ...]

            if model.heterogeneity_profile == "legacy":
                recon_het_loss = _candidate_reconstruction_losses(
                    images_het[:, None, ...], x, ctf, model.ctf_type,
                    normalize_target=False).mean()
                variance_loss, covariance_loss, latent_std = _latent_variance_covariance_loss(
                    latent, model.het_min_std)
                reference_coords = model.delta_het_decoder.factor * model.delta_het_decoder.coords
                reference_values = model.delta_het_decoder.reference_values
            else:
                recon_het_loss = _heterogeneity_reconstruction_loss(
                    images_het[:, None, ...], x, ctf, model.ctf_type,
                    model.het_loss_scales, model.het_loss_weights,
                    model.het_mask_radius, model.het_normalize_target)
                variance_loss, covariance_loss, latent_std = _latent_variance_covariance_loss(
                    latent, model.het_min_std,
                    bank=model.latent_memory_bank.get(),
                    bank_count=model.latent_bank_count.get_value())
                reference_coords, reference_values = base_coords, base_values

            coordinate_rms = jnp.sqrt(jnp.mean(jnp.square(coords_het - reference_coords)))
            amplitude_rms = jnp.sqrt(jnp.mean(jnp.square(values_het - reference_values)))
        
        # Keep proposal and selected-pose distributions separate.  The legacy
        # SWD remains available, while the higher-resolution bank-aware loss
        # below distinguishes dense coverage from repeated anchor locations.
        rotations_flat = rearrange(rotations, "b n w h -> (b n) w h")
        candidate_directions = rotations_flat[:, :, 2]
        winner_directions = rotations_selected[:, 0, :, 2]

        loss_uniform = jnp.asarray(0.0, dtype=recon_loss.dtype)
        if train_pose_volume:
            loss_swd = sliced_wasserstein_sphere(
                candidate_directions, rng=swd_key, n_projections=64)
            loss_uniform = lambda_uniform * loss_swd

        candidate_coverage_loss = jnp.asarray(0.0, dtype=recon_loss.dtype)
        if apply_candidate_coverage and train_pose_volume:
            candidate_coverage_loss = _candidate_coverage_loss(
                candidate_directions, model.memory_bank.get(),
                model.candidate_bank_count.get_value(), coverage_key,
                n_bins=candidate_coverage_bins,
                kappa=candidate_coverage_kappa,
                bank_samples=candidate_bank_samples,
                bank_mix=candidate_bank_mix)

        normalized_entropy = -jnp.mean(jnp.sum(
            responsibilities * jnp.log(jnp.maximum(responsibilities, 1e-12)), axis=1))
        normalized_entropy = normalized_entropy / jnp.maximum(
            jnp.log(jnp.asarray(responsibilities.shape[1], responsibilities.dtype)), 1.0)

        loss = jnp.asarray(0.0, dtype=recon_loss.dtype)
        if train_pose_volume:
            loss = (loss + 0.5 * reconstruction_objective + loss_uniform
                    + candidate_coverage_weight * candidate_coverage_loss)
        if train_heterogeneity:
            loss = (loss + 0.5 * recon_het_loss
                    + model.het_variance_weight * variance_loss
                    + model.het_covariance_weight * covariance_loss)
        if apply_amplitude_l1:
            # Values are ReLU'd, so the mean is the L1 density prior: it shrinks
            # noise-fitted background mass toward zero without touching coords.
            amplitude_l1 = jnp.asarray(0.0, dtype=recon_loss.dtype)
            if train_pose_volume:
                amplitude_l1 = amplitude_l1 + jnp.mean(values)
            if train_heterogeneity:
                amplitude_l1 = amplitude_l1 + jnp.mean(values_het)
            loss = loss + amplitude_l1_weight * amplitude_l1
        if train_pose_volume and apply_geometry_priors:
            spacing_loss, smoothness_loss = _geometry_prior_losses(
                coords, values, neighbor_indices, std_eff)
            loss = (loss + spacing_weight * spacing_loss
                    + smoothness_weight * smoothness_loss)
        if train_pose_volume and apply_support:
            loss = loss + support_weight * _support_loss(
                coords, values, support_center, support_radius, std_eff)
        metrics = (recon_loss, recon_het_loss, loss_uniform,
                   candidate_coverage_loss, normalized_entropy, anchor_deviation,
                   variance_loss, covariance_loss, latent_std,
                   coordinate_rms, amplitude_rms, projection_rms,
                   normalized_target_rms, consensus_amplitude_mean,
                   consensus_amplitude_rms, consensus_amplitude_max,
                   consensus_active_amplitude_fraction,
                   low_frequency_recon_loss,
                   reconstruction_objective,
                   low_frequency_full_agreement,
                   low_frequency_full_disadvantage_std)
        pose_diagnostics = tuple(jax.lax.stop_gradient(value) for value in (
            rotations_selected[:, 0], selected_head_indices, best_head_indices,
            absolute_margin, relative_margin, standardized_margin,
            median_normalized_margin, candidate_score_entropy))
        return loss, (metrics, candidate_directions, winner_directions, latent,
                      pose_diagnostics)

    # Optimizer parameters
    params_pose = nnx.All(nnx.Param, nnx.PathContains('encoder_pose'))
    params_volume = nnx.All(nnx.Param, nnx.PathContains('delta_volume_decoder'))
    params_het = nnx.All(nnx.Param, (nnx.PathContains('encoder_het'), nnx.PathContains('delta_het_decoder')))

    if model.refine_current_assignment:
        # Precompute batch aligments
        current_euler_angles = md["euler_angles"][labels]
        current_rotations = euler_matrix_batch(current_euler_angles[..., 0], current_euler_angles[..., 1], current_euler_angles[..., 2])

        # Precompute batch shifts
        current_shifts = md["shifts"][labels]
    else:
        current_rotations = jnp.tile(jnp.eye(3)[None, ...], (x.shape[0], 1, 1))
        current_shifts = jnp.zeros((x.shape[0], 2))

    # Precompute batch CTFs
    if model.ctf_type not in (None, "None"):
        defocusU = md["ctfDefocusU"][labels]
        defocusV = md["ctfDefocusV"][labels]
        defocusAngle = md["ctfDefocusAngle"][labels]
        cs = md["ctfSphericalAberration"][labels]
        kv = md["ctfVoltage"][labels][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         model.sr, [2 * model.xsize, int(2 * 0.5 * model.xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], 2 * model.xsize, int(2.0 * 0.5 * model.xsize + 1)], dtype=x.dtype)

    coarse_ctf = ctf
    if (not use_tau and assignment_mode == "hard"
            and scoring_size >= model.xsize
            and model.coarse_topk < model.encoder_pose.num_components):
        screen_size = max(8, int(round(model.xsize * model.coarse_scale)))
        if model.ctf_type not in (None, "None"):
            coarse_ctf = computeCTF(
                defocusU, defocusV, defocusAngle, cs, kv,
                model.sr / (screen_size / model.xsize),
                [2 * screen_size, screen_size + 1], x.shape[0], True)
        else:
            coarse_ctf = jnp.ones(
                [x.shape[0], 2 * screen_size, screen_size + 1], dtype=x.dtype)

    if model.ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, (params_pose, params_volume, params_het)), has_aux=True)
    (loss, (metrics, candidate_directions, winner_directions, latent,
            pose_diagnostics)), grads_combined = grad_fn(model, x)

    grads_pose, grads_volume, grads_het = grads_combined.split(params_pose, params_volume, params_het)

    if train_pose_volume:
        optimizer_pose.update(model, grads_pose)
        optimizer_volume.update(model, grads_volume)
    if train_heterogeneity:
        optimizer_het.update(model, grads_het)

    if train_pose_volume:
        model.memory_bank.enqueue(jax.lax.stop_gradient(candidate_directions))
        model.winner_memory_bank.enqueue(jax.lax.stop_gradient(winner_directions))
        model.candidate_bank_count.value = jnp.minimum(
            model.memory_bank.buffer_size,
            model.candidate_bank_count.get_value() + candidate_directions.shape[0])
        model.winner_bank_count.value = jnp.minimum(
            model.winner_memory_bank.buffer_size,
            model.winner_bank_count.get_value() + winner_directions.shape[0])
    if train_heterogeneity and model.heterogeneity_profile == "anti_collapse":
        model.latent_memory_bank.enqueue(jax.lax.stop_gradient(latent))
        model.latent_bank_count.value = jnp.minimum(
            model.latent_memory_bank.buffer_size,
            model.latent_bank_count.get_value() + latent.shape[0])

    state = nnx.state((model, optimizer_pose, optimizer_volume, optimizer_het))

    if return_metrics:
        if return_pose_diagnostics:
            return loss, metrics, pose_diagnostics, state, key
        return loss, metrics, state, key
    return loss, metrics[0], state, key


@jax.jit
def validation_step_reconsiren(graphdef, state, x, labels, md, key):
    model, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)

    # Random keys
    key, choice_key, distributions_key = jax.random.split(key, 3)

    def loss_fn(model, x):
        # Correct CTF in images for encoder if needed
        if model.ctf_type in ["apply", "squared"]:
            x_ctf_corrected = prepare_image_cryocrab(x, ctf)
            # x_ctf_corrected = prepare_image_wiener(x, ctf)
        else:
            x_ctf_corrected = x

        # Get euler angles and shifts
        rotations, shifts = model.encoder_pose(x_ctf_corrected)

        # Decode volume
        coords, values = model.delta_volume_decoder()

        # Refine current assignment (if provided)
        # rotations = jnp.matmul(rotations, current_rotations[:, None, :, :])
        rotations = jnp.matmul(current_rotations[:, None, :, :], rotations)  # TODO: The two options seem to work?
        shifts = current_shifts[:, None, :] + shifts

        # Random symmetry matrices
        random_indices = jax.random.choice(choice_key, jnp.arange(model.symmetry_matrices.shape[0]), shape=(rotations.shape[0],))
        rotations = jnp.matmul(jnp.transpose(model.symmetry_matrices[random_indices], (0, 2, 1))[:, None, :, :], rotations)

        # Generate projections
        images_corrected = model.phys_decoder(x, values, coords, model.xsize, rotations, shifts, ctf, model.ctf_type, model.get_std())

        # Losses
        images_corrected_loss = images_corrected[..., 0] if images_corrected.shape[-1] == 1 else images_corrected
        x_loss = x[..., 0] if x.shape[-1] == 1 else x

        # Consider CTF if Wiener/Squared mode (only for loss)
        if model.ctf_type == "wiener":
            ctf_broadcasted = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf_broadcasted = rearrange(ctf_broadcasted, "b n w h -> (b n) w h")

            x_loss = wiener2DFilter(x_loss, ctf, pad_factor=2)

            images_corrected_loss = rearrange(images_corrected_loss, "b n w h -> (b n) w h")
            images_corrected_loss = wiener2DFilter(images_corrected_loss, ctf_broadcasted, pad_factor=2)
            images_corrected_loss = rearrange(images_corrected_loss, "(b n) w h -> b n w h")
        elif model.ctf_type == "squared":
            ctf_broadcasted = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf_broadcasted = rearrange(ctf_broadcasted, "b n w h -> (b n) w h")

            x_loss = ctfFilter(x_loss, ctf, pad_factor=2)

            images_corrected_loss = rearrange(images_corrected_loss, "b n w h -> (b n) w h")
            images_corrected_loss = ctfFilter(images_corrected_loss, ctf_broadcasted, pad_factor=2)
            images_corrected_loss = rearrange(images_corrected_loss, "(b n) w h -> b n w h")

        # Broadcast input images to right size
        x_loss = jnp.broadcast_to(x_loss[:, None, ...], (x_loss.shape[0], images_corrected.shape[1], x_loss.shape[1], x_loss.shape[2]))

        x_flat = rearrange(x_loss, "b n w h -> (b n) w h")
        images_corrected_flat = rearrange(images_corrected_loss, "b n w h -> (b n) w h")
        x_flat = standard_normalization(x_flat)
        recon_loss = dm_pix.mse(images_corrected_flat[..., None], x_flat[..., None])
        recon_loss = rearrange(recon_loss, "(b n) -> b n", b=images_corrected_loss.shape[0], n=images_corrected_loss.shape[1])

        # Get minimum indices
        min_indices = jnp.argmin(recon_loss, axis=1)

        # Index losses and rotations based on extracted indices
        recon_loss = recon_loss[jnp.arange(images_corrected.shape[0]), min_indices].mean()

        return recon_loss

    if model.refine_current_assignment:
        # Precompute batch aligments
        current_euler_angles = md["euler_angles"][labels]
        current_rotations = euler_matrix_batch(current_euler_angles[..., 0], current_euler_angles[..., 1], current_euler_angles[..., 2])

        # Precompute batch shifts
        current_shifts = md["shifts"][labels]
    else:
        current_rotations = jnp.tile(jnp.eye(3)[None, ...], (x.shape[0], 1, 1))
        current_shifts = jnp.zeros((x.shape[0], 2))

    # Precompute batch CTFs
    if model.ctf_type not in (None, "None"):
        defocusU = md["ctfDefocusU"][labels]
        defocusV = md["ctfDefocusV"][labels]
        defocusAngle = md["ctfDefocusAngle"][labels]
        cs = md["ctfSphericalAberration"][labels]
        kv = md["ctfVoltage"][labels][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         model.sr, [2 * model.xsize, int(2 * 0.5 * model.xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], 2 * model.xsize, int(2.0 * 0.5 * model.xsize + 1)], dtype=x.dtype)

    if model.ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

    loss = loss_fn(model, x)

    return loss


@jax.jit
def predict_angular_assignment_step_reconsiren(graphdef, state, x, labels, md, key):
    model = nnx.merge(graphdef, state)

    distributions_key, key = jax.random.split(key, 2)

    # Recover alignments in metadata if refining them
    if model.refine_current_assignment:
        # Precompute batch aligments
        current_euler_angles = md["euler_angles"][labels]
        current_rotations = euler_matrix_batch(current_euler_angles[..., 0], current_euler_angles[..., 1], current_euler_angles[..., 2])

        # Precompute batch shifts
        current_shifts = md["shifts"][labels]
    else:
        current_rotations = jnp.tile(jnp.eye(3)[None, ...], (x.shape[0], 1, 1))
        current_shifts = jnp.zeros((x.shape[0], 2))

    # Precompute batch CTFs
    if model.ctf_type not in (None, "None"):
        defocusU = md["ctfDefocusU"][labels]
        defocusV = md["ctfDefocusV"][labels]
        defocusAngle = md["ctfDefocusAngle"][labels]
        cs = md["ctfSphericalAberration"][labels]
        kv = md["ctfVoltage"][labels][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         model.sr, [2 * model.xsize, int(2 * 0.5 * model.xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], 2 * model.xsize, int(2.0 * 0.5 * model.xsize + 1)], dtype=x.dtype)

    coarse_ctf = ctf
    if model.coarse_topk < model.encoder_pose.num_components:
        screen_size = max(8, int(round(model.xsize * model.coarse_scale)))
        if model.ctf_type not in (None, "None"):
            coarse_ctf = computeCTF(
                defocusU, defocusV, defocusAngle, cs, kv,
                model.sr / (screen_size / model.xsize),
                [2 * screen_size, screen_size + 1], x.shape[0], True)
        else:
            coarse_ctf = jnp.ones(
                [x.shape[0], 2 * screen_size, screen_size + 1], dtype=x.dtype)

    if model.ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

    if model.ctf_type in ["apply", "squared"]:
        x_ctf_corrected = prepare_image_cryocrab(x, ctf)
        # x_ctf_corrected = prepare_image_wiener(x, ctf)
    else:
        x_ctf_corrected = x

    # Get euler angles and shifts
    rotations, shifts = model.encoder_pose(x_ctf_corrected)
    _, latent, _ = model.encoder_het(x_ctf_corrected, rngs=distributions_key)

    # Decode volume
    coords, values = model.delta_volume_decoder()

    # Refine current assignment (if provided)
    # rotations = jnp.matmul(rotations, current_rotations[:, None, :, :])
    rotations = jnp.matmul(current_rotations[:, None, :, :], rotations)  # TODO: The two options seem to work?
    shifts = current_shifts[:, None, :] + shifts

    if model.coarse_topk < rotations.shape[1]:
        screen_size = max(8, int(round(model.xsize * model.coarse_scale)))
        x_screen = jax.image.resize(
            x, (x.shape[0], screen_size, screen_size, x.shape[-1]), method="bilinear")
        coords_screen, values_screen = coords, values
        if 0 < model.coarse_gaussians < coords.shape[1]:
            keep = model.coarse_gaussians
            idx = (jnp.arange(keep) * coords.shape[1]) // keep
            coords_screen = coords[:, idx]
            values_screen = values[:, idx] * (coords.shape[1] / keep)
        coarse_images = model.phys_decoder(
            x_screen, values_screen, coords_screen, model.xsize, rotations, shifts,
            coarse_ctf, model.ctf_type, model.get_std(), render_size=screen_size)
        coarse_losses = _candidate_reconstruction_losses(
            coarse_images, x_screen, coarse_ctf, model.ctf_type)
        _, top_indices = jax.lax.top_k(-coarse_losses, model.coarse_topk)
        rotations = _gather_candidates(rotations, top_indices)
        shifts = _gather_candidates(shifts, top_indices)

    recon_loss, _ = _score_candidates(
        model, x, values, coords, rotations, shifts, ctf)

    # Get minimum indices
    min_indices = jnp.argmin(recon_loss, axis=1)

    # Index shifts and rotations based on extracted indices
    rotations = rotations[jnp.arange(x.shape[0]), min_indices, :]
    shifts = shifts[jnp.arange(x.shape[0]), min_indices, :]

    return rotations, shifts, latent

euler_from_matrix_batch = jax.vmap(jax.jit(euler_from_matrix))

def xmippEulerFromMatrix(matrix):
    return -jnp.rad2deg(euler_from_matrix_batch(matrix))


def main():
    import os
    import sys
    import json
    import shutil
    from tqdm import tqdm
    import random
    import numpy as np
    import argparse
    import matplotlib.pyplot as plt
    from xmipp_metadata.image_handler import ImageHandler
    import optax
    from contextlib import closing
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, extract_columns
    from hax.metrics import JaxSummaryWriter, TrainingLogger
    from hax.networks import VolumeAdjustment, train_step_volume_adjustment
    from hax.programs import fit_volume, adjust_weights_to_images
    from hax.programs.gaussian_volume_fitting import get_cosine_reg_strength

    from hax.cli import common_args as ca

    def comma_separated_ints(value):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())

    def comma_separated_floats(value):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_vol(parser,
               help="If provided, the neural network will start from this volume when assigning the angles and shifts to the images.")
    ca.add_mask(parser,
                help=f"ReconSIREN reconstruction mask (the mask provided must be binary - "
                     f"{bcolors.WARNING}NOTE{bcolors.ENDC}: since this is a reconstruction mask, it should be defined such that it covers the "
                     f"volume were the motions of interest are expected to happen)")
    parser.add_argument("--num_gaussians", required=False, type=int, default=None,
                        help=f"Number of Gaussians in the point cloud. With a reference volume the Gaussians are fitted to it before "
                             f"training (default 10000). Fully {bcolors.ITALIC}ab initio{bcolors.ENDC} (no --vol) the default is derived from the "
                             f"estimated particle extent so the cloud can tile the particle at the splat width; set this parameter to "
                             f"override either default (e.g. from the number of residues in your protein).")
    ca.add_load_images_to_ram(parser)
    ca.add_sr(parser)
    parser.add_argument("--do_not_learn_volume", action="store_true",
                        help="When this parameter is provided, ReconSIREN will just learn an angular assignment with shifts without learning any map. This is usually useful when a reference volume with "
                             "high resolution is provided (e.g. coming from an atomic model) and no refinement of the map is needed.")
    parser.add_argument("--refine_current_assignment", action="store_true",
                        help=f"If your input metadata has already and angular assignment and shifts, you can provide this option to refine those angles instead of finding an {bcolors.ITALIC}ab initio{bcolors.ENDC} "
                             f"alignment.")
    parser.add_argument("--do_not_use_anchor_rotations", action="store_true",
                        help=f"By default the poses proposed by the encoder are composed with a fixed set of anchor rotations spread on a spherical grid (one per pose hypothesis), which "
                             f"biases the {bcolors.ITALIC}ab initio{bcolors.ENDC} pose search to cover orientation space more evenly. Provide this option to disable that composition.")
    ca.add_symmetry_group(parser)
    parser.add_argument("--num_components", required=False, type=int, default=18,
                        help=f"Number of candidate pose hypotheses the pose encoder proposes per image during the {bcolors.ITALIC}ab initio{bcolors.ENDC} search. "
                             f"For every image the network evaluates this many orientations (anchored on a spherical grid), renders a projection for each and keeps the "
                             f"best-matching one. A larger value covers orientation space more densely, making the pose search more robust to local minima, but increases GPU "
                             f"memory and compute roughly linearly (this is the main driver of ReconSIREN's training footprint). Set it lower to fit a smaller GPU at the cost of a "
                             f"coarser pose search. Default: 18.")
    parser.add_argument("--optimization_profile", choices=("legacy", "aggressive"), default="aggressive",
                        help="Execution/model profile. aggressive enables the optimized architecture and "
                             "coarse-to-fine pose scoring; legacy preserves the historical architecture. "
                             "Old checkpoints load as legacy automatically.")
    parser.add_argument("--pose_head_rank", type=int, default=None,
                        help="Rank of each independent factorized pose-head layer; 0 uses dense layers. "
                             "Profile default: aggressive=128, legacy=0.")
    parser.add_argument("--pose_spatial_pool", type=int, default=None,
                        help="Average-pooling factor before the pose dense trunk. "
                             "Profile default: aggressive=4, legacy=1.")
    parser.add_argument("--candidate_coverage_epochs", type=float, default=10.0,
                        help="Epochs during which the bank-aware candidate sphere-coverage loss is active; "
                             "0 disables it.")
    parser.add_argument("--candidate_coverage_weight", type=float, default=0.0,
                        help="Weight of the bank-aware spherical occupancy KL; 0 disables it.")
    parser.add_argument("--candidate_coverage_bins", type=int, default=256,
                        help="Number of equal-area soft occupancy bins on the projection sphere.")
    parser.add_argument("--candidate_coverage_kappa", type=float, default=32.0,
                        help="Concentration of soft spherical-bin assignments; larger values resolve "
                             "smaller angular gaps.")
    parser.add_argument("--candidate_bank_samples", type=int, default=1024,
                        help="Historical candidate directions sampled per coverage update; 0 ignores history.")
    parser.add_argument("--candidate_bank_mix", type=float, default=0.5,
                        help="Historical occupancy fraction in [0,1); current candidates retain the "
                             "remaining mass so their gradients are not diluted by bank size.")
    parser.add_argument("--consensus_multiscale_epochs", type=float, default=0.0,
                        help="Hard-assignment epochs using a blended full- and low-frequency loss on the "
                             "selected consensus projection. 0 keeps the legacy full-resolution objective.")
    parser.add_argument("--consensus_multiscale_scales", type=comma_separated_floats,
                        default=(0.25, 0.5, 0.75),
                        help="Comma-separated image-size fractions used in equal stages by the selected "
                             "consensus reconstruction curriculum.")
    parser.add_argument("--consensus_multiscale_weight", type=float, default=0.5,
                        help="Initial low-frequency weight in the selected reconstruction objective. The "
                             "weight decays linearly to zero while retaining the complementary full-resolution term.")
    parser.add_argument("--pose_curriculum", choices=("adaptive", "fixed"), default="adaptive",
                        help=f"Candidate scoring-resolution schedule. {bcolors.BOLD}adaptive{bcolors.ENDC} ranks "
                             "candidates at progressively higher low-pass resolutions and advances a stage only "
                             "once per-particle winners are stable at the current one (this also drives the "
                             f"consensus multiscale loss). {bcolors.BOLD}fixed{bcolors.ENDC} preserves the "
                             "full-resolution scoring and epoch-scheduled multiscale loss.")
    parser.add_argument("--candidate_frequency_scales", type=comma_separated_floats,
                        default=(0.25, 0.5, 0.75),
                        help="Comma-separated image-size fractions used as adaptive curriculum stages before "
                             "full resolution is restored.")
    parser.add_argument("--pose_curriculum_switch_threshold", type=float, default=0.05,
                        help="Winner head-switch fraction at or below which a curriculum stage may advance.")
    parser.add_argument("--pose_curriculum_pose_threshold", type=float, default=5.0,
                        help="Median winner pose change (degrees) at or below which a curriculum stage may advance.")
    parser.add_argument("--pose_curriculum_min_epochs", type=int, default=1,
                        help="Minimum epochs spent in each curriculum stage before it may advance.")
    parser.add_argument("--pose_curriculum_max_epochs", type=int, default=10,
                        help="Epochs after which a curriculum stage advances even if winners are still "
                             "churning; 0 waits for stability indefinitely.")
    parser.add_argument("--pose_curriculum_temperatures", type=comma_separated_floats,
                        default=(0.3, 0.15, 0.05),
                        help="Dimensionless assignment temperatures per adaptive curriculum stage (a single "
                             "value applies to every stage). Winners are sampled from margin-aware "
                             "responsibilities instead of taken by argmin; the final full-resolution stage "
                             "is always hard. 0 disables sampling for clean A/B runs.")
    parser.add_argument("--pose_diagnostics", action="store_true",
                        help="Track per-particle winner rotations and top-two loss margins. Adds a small "
                             "device-to-host transfer per batch but does not affect training.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional reproducible seed for Gaussian-cloud and network initialization.")
    parser.add_argument("--heterogeneity_profile", choices=("legacy", "anti_collapse"),
                        default="anti_collapse",
                        help="Heterogeneity training profile. anti_collapse enables staged residual "
                             "training, resized encoding, multiscale masked loss and latent statistics; "
                             "legacy preserves the historical objective and decoder.")
    parser.add_argument("--het_encoder_architecture", choices=("legacy", "convstem", "resize"), default=None,
                        help="Heterogeneity encoder input projection. Profile default: "
                             "anti_collapse=resize; legacy CLI profile=legacy.")
    parser.add_argument("--het_encoder_size", type=int, default=None,
                        help="Anti-aliased encoder image size for the resize architecture. "
                             "Must be a multiple of 16; anti-collapse default: min(128, box size).")
    parser.add_argument("--het_disable_consensus_residual", action="store_true",
                        help="Disable decoding heterogeneous states as residuals from the learned consensus.")
    parser.add_argument("--het_disable_decoder_centering", action="store_true",
                        help="Disable subtracting decoder(z=0) from heterogeneous states.")
    parser.add_argument("--het_coordinate_scale", type=float, default=1.0,
                        help="Multiplier for heterogeneous coordinate residuals.")
    parser.add_argument("--het_amplitude_scale", type=float, default=1.0,
                        help="Multiplier for heterogeneous amplitude residuals.")
    parser.add_argument("--het_loss_scales", type=comma_separated_ints, default=None,
                        help="Comma-separated reconstruction sizes; default for a full-size anti-collapse "
                             "run is 64,128,full.")
    parser.add_argument("--het_loss_weights", type=comma_separated_floats, default=None,
                        help="Comma-separated weights matching --het_loss_scales; normalized internally.")
    parser.add_argument("--het_mask_radius", type=float, default=None,
                        help="Circular heterogeneity-loss radius as a box fraction in [0, 0.5].")
    parser.add_argument("--het_disable_target_normalization", action="store_true",
                        help="Disable the per-particle target standardization used by consensus training.")
    parser.add_argument("--het_variance_weight", type=float, default=None,
                        help="Weight of the latent standard-deviation floor penalty.")
    parser.add_argument("--het_covariance_weight", type=float, default=None,
                        help="Weight of the off-diagonal latent covariance penalty.")
    parser.add_argument("--het_min_std", type=float, default=0.1,
                        help="Minimum latent standard deviation targeted by the variance penalty.")
    parser.add_argument("--het_start_epoch", type=int, default=None,
                        help="First epoch that trains heterogeneity, providing a consensus-only warm-up. "
                             "Profile defaults: legacy heterogeneity=0, anti-collapse=5.")
    parser.add_argument("--het_train_consensus", action="store_true",
                        help="Continue training pose and consensus after heterogeneity training begins.")
    parser.add_argument("--het_latent_bank_size", type=int, default=2048,
                        help="Number of prior latent vectors used for stable variance/covariance statistics.")
    parser.add_argument("--consensus_parameterization", choices=("network", "direct"), default=None,
                        help="Consensus Gaussian delta parameterization. Profile default: aggressive=direct.")
    parser.add_argument("--render_chunk_size", type=int, default=None,
                        help="Gaussians per rematerialized scatter block; 0 disables point chunking. "
                             "Profile default: aggressive=2048, legacy=0.")
    parser.add_argument("--candidate_chunk_size", type=int, default=None,
                        help="Full-resolution pose candidates scored together; 0 scores all together. "
                             "Profile default: aggressive=3, legacy=0.")
    parser.add_argument("--coarse_topk", type=int, default=None,
                        help="Coarse-ranked candidates evaluated at full resolution. "
                             "Profile default: aggressive=4, legacy=all.")
    parser.add_argument("--coarse_scale", type=float, default=None,
                        help="Linear image scale for coarse pose ranking. Profile default: aggressive=0.5.")
    parser.add_argument("--coarse_gaussians", type=int, default=None,
                        help="Deterministic Gaussian subset for coarse ranking; 0 uses the full cloud. "
                             "Profile default: aggressive=2048, legacy=all.")
    parser.add_argument("--no_fused_envelope", action="store_true",
                        help="Disable the fused analytic Gaussian-envelope + CTF Fourier filter and go back to "
                             "the legacy truncated 9-tap spatial blur followed by a separate CTF pass.")
    parser.add_argument("--sigma_bounds", type=str, default="auto",
                        help="Bounds 'min,max' (in voxels) for the learned splat width. Bounding stops the "
                             "optimizer from inflating sigma to hide pose error (which blurs the map) and from "
                             "collapsing it below the splat sampling limit. 'auto' derives the bounds from the "
                             "fitted initial width.")
    parser.add_argument("--no_sigma_bounds", action="store_true",
                        help="Keep the legacy unbounded learned splat width.")
    parser.add_argument("--whiten_loss_weight", type=float, default=0.5,
                        help="Blend weight in [0,1] for the noise-whitened consensus reconstruction loss. The "
                             "dataset noise spectrum is estimated once from the particle solvent corners; "
                             "whitening equalises the per-frequency-shell SNR so high-resolution shells receive "
                             "real gradient instead of being drowned by the low-frequency power. Only active at "
                             "the final (full-resolution) pose-curriculum stage. 0 disables.")
    parser.add_argument("--amplitude_l1", type=float, default=0.0,
                        help="Weight of an L1 prior on Gaussian amplitudes (consensus and heterogeneous) that "
                             "suppresses noise-fitted background dust. Off by default: measure its scale against "
                             "the reconstruction loss on your dataset before trusting a non-zero value.")
    parser.add_argument("--no_per_candidate_shifts", action="store_true",
                        help="Disable the per-candidate in-plane shift deltas and go back to one shared shift "
                             "broadcast to every pose hypothesis.")
    parser.add_argument("--no_sharpened_map", action="store_true",
                        help="Do not write the additional envelope-sharpened map in predict mode.")
    parser.add_argument("--sharpened_map_reg", type=float, default=0.02,
                        help="Wiener regularizer for the envelope-sharpened map (bounds the maximum gain to "
                             "~1/(2*sqrt(reg))).")
    parser.add_argument("--disable_quality_features", action="store_true",
                        help=f"Master switch that turns off every quality feature at once (fused envelope, sigma "
                             f"bounds, loss whitening, amplitude L1, per-candidate shifts, sharpened map) for "
                             f"A/B testing against the previous behaviour.")
    parser.add_argument("--no_extent_estimation", action="store_true",
                        help=f"Disable the {bcolors.ITALIC}ab initio{bcolors.ENDC} particle-extent estimation from the raw images "
                             f"(per-pixel variance excess over the noise floor) and fall back to the fixed "
                             f"quarter-box initialization ball and the default Gaussian count.")
    parser.add_argument("--support_weight", type=float, default=1.0,
                        help="Weight of the shrink-wrap support penalty: mass outside a spherical support "
                             "re-derived every epoch from the converging cloud itself (no mask needed). The "
                             "term is zero inside the support, so it only suppresses background dust. 0 disables.")
    parser.add_argument("--spacing_prior_weight", type=float, default=0.05,
                        help="Weight of the kNN spacing prior: penalizes mass-carrying neighbours further apart "
                             "than ~2 splat widths (density visually fragments) or closer than ~0.7 (redundant "
                             "stacking). Active only at the final full-resolution curriculum stage. 0 disables.")
    parser.add_argument("--amplitude_smoothness_weight", type=float, default=0.01,
                        help="Weight of the kNN amplitude-smoothness prior (graph Laplacian on Gaussian masses) "
                             "so one iso-surface threshold traces the whole chain instead of beading. Active only "
                             "at the final full-resolution curriculum stage. 0 disables.")
    parser.add_argument("--knn_neighbors", type=int, default=6,
                        help="Neighbours per point in the kNN graph used by the spacing/smoothness priors "
                             "(refreshed on the host every epoch).")
    parser.add_argument("--no_point_recycling", action="store_true",
                        help="Disable periodic recycling of amplitude-dead Gaussians next to mass-carrying ones "
                             "(fixed-N densify/prune). Recycling requires the 'direct' consensus parameterization.")
    parser.add_argument("--recycle_every", type=int, default=5,
                        help="Epoch cadence for point recycling after the warm-up (also triggered when the pose "
                             "curriculum advances a stage).")
    parser.add_argument("--recycle_dead_fraction", type=float, default=0.05,
                        help="A point is considered dead when its amplitude falls below this fraction of the "
                             "mean amplitude.")
    parser.add_argument("--cloud_blur_max", type=float, default=1.5,
                        help="Maximum extra render blur (voxels, added in quadrature to the learned splat width) "
                             "at the start of training, annealed to zero as the pose curriculum reaches full "
                             "resolution: the cloud stays coarse while poses are coarse. 0 disables.")
    parser.add_argument("--no_equalized_map", action="store_true",
                        help="Do not write the amplitude-equalized tracing map in predict mode.")
    parser.add_argument("--equalized_map_gamma", type=float, default=0.5,
                        help="Gamma compression applied to Gaussian masses for the equalized tracing map "
                             "(1 reproduces the physical map; smaller values flatten contrast along the chain so "
                             "a single ChimeraX threshold shows the whole structure).")
    parser.add_argument("--disable_geometry_features", action="store_true",
                        help=f"Master switch that turns off every {bcolors.ITALIC}ab initio{bcolors.ENDC} geometry feature at once "
                             f"(extent estimation, shrink-wrap support, spacing/smoothness priors, point "
                             f"recycling, cloud blur curriculum, equalized map) for A/B testing.")
    ca.add_ctf_type(parser)
    ca.add_mode(parser)
    ca.add_epochs(parser)
    ca.add_batch_size(parser)
    ca.add_learning_rate(parser)
    ca.add_dataset_split_fraction(parser)
    ca.add_output_path(parser)
    ca.add_logging_args(parser)
    ca.add_reload(parser,
                  help="Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data).")
    ca.add_ssd_scratch_folder(parser)
    args = ca.parse_with_config(parser)
    # None keeps the selected profile's default; the store_true overrides only
    # opt out of individual anti-collapse behaviours.
    args.het_residual_to_consensus = (False if args.het_disable_consensus_residual else None)
    args.het_center_decoder = (False if args.het_disable_decoder_centering else None)
    args.het_normalize_target = (False if args.het_disable_target_normalization else None)
    args.het_freeze_consensus = (False if args.het_train_consensus else None)
    if args.heterogeneity_profile == "legacy" and args.het_encoder_architecture is None:
        # A single profile switch should reproduce the historical heterogeneity
        # architecture even when execution optimizations remain enabled.
        args.het_encoder_architecture = "legacy"
    if args.candidate_coverage_epochs < 0.0:
        parser.error("--candidate_coverage_epochs must be non-negative")
    if args.candidate_coverage_bins < 2:
        parser.error("--candidate_coverage_bins must be at least 2")
    if args.candidate_bank_samples < 0:
        parser.error("--candidate_bank_samples must be non-negative")
    if args.candidate_coverage_kappa <= 0.0:
        parser.error("--candidate_coverage_kappa must be positive")
    if not 0.0 <= args.candidate_bank_mix < 1.0:
        parser.error("--candidate_bank_mix must be in [0,1)")
    if args.candidate_coverage_weight < 0.0:
        parser.error("--candidate_coverage_weight must be non-negative")
    if args.consensus_multiscale_epochs < 0.0:
        parser.error("--consensus_multiscale_epochs must be non-negative")
    if not 0.0 <= args.consensus_multiscale_weight <= 1.0:
        parser.error("--consensus_multiscale_weight must be in [0,1]")
    if (not args.consensus_multiscale_scales
            or any(scale <= 0.0 or scale > 1.0
                   for scale in args.consensus_multiscale_scales)):
        parser.error("--consensus_multiscale_scales values must be in (0,1]")
    if any(right <= left for left, right in zip(
            args.consensus_multiscale_scales,
            args.consensus_multiscale_scales[1:])):
        parser.error("--consensus_multiscale_scales must be strictly increasing")
    if (not args.candidate_frequency_scales
            or any(scale <= 0.0 or scale > 1.0
                   for scale in args.candidate_frequency_scales)):
        parser.error("--candidate_frequency_scales values must be in (0,1]")
    if any(right <= left for left, right in zip(
            args.candidate_frequency_scales,
            args.candidate_frequency_scales[1:])):
        parser.error("--candidate_frequency_scales must be strictly increasing")
    if not 0.0 <= args.pose_curriculum_switch_threshold <= 1.0:
        parser.error("--pose_curriculum_switch_threshold must be in [0,1]")
    if args.pose_curriculum_pose_threshold < 0.0:
        parser.error("--pose_curriculum_pose_threshold must be non-negative")
    if args.pose_curriculum_min_epochs < 1:
        parser.error("--pose_curriculum_min_epochs must be at least 1")
    if args.pose_curriculum_max_epochs < 0:
        parser.error("--pose_curriculum_max_epochs must be non-negative")
    if (args.pose_curriculum_max_epochs
            and args.pose_curriculum_max_epochs < args.pose_curriculum_min_epochs):
        parser.error("--pose_curriculum_max_epochs must be 0 or >= --pose_curriculum_min_epochs")
    if (not args.pose_curriculum_temperatures
            or any(value < 0.0 for value in args.pose_curriculum_temperatures)):
        parser.error("--pose_curriculum_temperatures values must be non-negative")
    if len(args.pose_curriculum_temperatures) not in (
            1, len(args.candidate_frequency_scales)):
        parser.error("--pose_curriculum_temperatures must be a single value or one "
                     "per --candidate_frequency_scales entry")
    if args.disable_quality_features:
        args.no_fused_envelope = True
        args.no_sigma_bounds = True
        args.whiten_loss_weight = 0.0
        args.amplitude_l1 = 0.0
        args.no_per_candidate_shifts = True
        args.no_sharpened_map = True
    if args.disable_geometry_features:
        args.no_extent_estimation = True
        args.support_weight = 0.0
        args.spacing_prior_weight = 0.0
        args.amplitude_smoothness_weight = 0.0
        args.no_point_recycling = True
        args.cloud_blur_max = 0.0
        args.no_equalized_map = True
    if args.support_weight < 0.0:
        parser.error("--support_weight must be non-negative")
    if args.spacing_prior_weight < 0.0 or args.amplitude_smoothness_weight < 0.0:
        parser.error("--spacing_prior_weight and --amplitude_smoothness_weight must be non-negative")
    if args.knn_neighbors < 1:
        parser.error("--knn_neighbors must be at least 1")
    if args.recycle_every < 1:
        parser.error("--recycle_every must be at least 1")
    if not 0.0 < args.recycle_dead_fraction < 1.0:
        parser.error("--recycle_dead_fraction must be in (0,1)")
    if args.cloud_blur_max < 0.0:
        parser.error("--cloud_blur_max must be non-negative")
    if not 0.0 < args.equalized_map_gamma <= 1.0:
        parser.error("--equalized_map_gamma must be in (0,1]")
    if args.num_gaussians is not None and args.num_gaussians < 1:
        parser.error("--num_gaussians must be positive")
    if not 0.0 <= args.whiten_loss_weight <= 1.0:
        parser.error("--whiten_loss_weight must be in [0,1]")
    if args.amplitude_l1 < 0.0:
        parser.error("--amplitude_l1 must be non-negative")
    if args.sharpened_map_reg <= 0.0:
        parser.error("--sharpened_map_reg must be positive")
    explicit_sigma_bounds = None
    if not args.no_sigma_bounds and args.sigma_bounds != "auto":
        try:
            explicit_sigma_bounds = tuple(float(v) for v in args.sigma_bounds.split(","))
        except ValueError:
            parser.error("--sigma_bounds must be 'auto' or 'min,max'")
        if len(explicit_sigma_bounds) != 2 or not 0.0 < explicit_sigma_bounds[0] < explicit_sigma_bounds[1]:
            parser.error("--sigma_bounds must satisfy 0 < min < max")
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        random.seed(args.seed)
        np.random.seed(args.seed % (2 ** 32))

    # Matplotlib plot style
    plt.style.use('dark_background')  # This sets many defaults for a dark theme
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'black'
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'

    # Check that training and validation fractions add up to one
    ca.validate_dataset_split_fraction(args.dataset_split_fraction)

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Prepare grain dataset
    if not args.load_images_to_ram and args.mode in ["train", "predict"]:
        mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=True, num_workers=4,
                                             precision=np.float16, group_size=1, shard_size=10000)
    else:
        mmap_output_dir = None

    # Preprocess volume (and mask)
    xsize = generator.md.getMetaDataImage(0).shape[0]
    if args.vol is not None:
        vol = ImageHandler(args.vol).getData()
    else:
        vol = np.zeros((xsize, xsize, xsize))

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    else:
        mask = ImageHandler().createCircularMask(boxSize=xsize, radius=int(0.25 * xsize), is3D=True)

    # Ab initio particle extent from the raw images (no reference, no mask):
    # per-pixel variance across a probe of particles rises inside the particle
    # and stays at the noise floor outside it.
    extent_radius_px = None
    if args.vol is None and args.mode == "train" and not args.no_extent_estimation:
        n_probe = int(min(256, len(generator.md)))
        probe_indices = np.linspace(0, len(generator.md) - 1, n_probe).astype(int)
        probe = np.stack([np.squeeze(generator.md.getMetaDataImage(int(index)))
                          for index in probe_indices])
        extent_radius_px = _estimate_particle_extent(probe)
        if extent_radius_px is not None:
            print(f"{bcolors.OKCYAN}Estimated particle radius from {n_probe} images: "
                  f"{extent_radius_px:.1f} px ({2.0 * extent_radius_px / xsize:.0%} of the box "
                  f"as diameter){bcolors.ENDC}")
        else:
            print(f"{bcolors.WARNING}Could not estimate the particle extent from the images; "
                  f"falling back to the fixed quarter-box initialization{bcolors.ENDC}")

    # Resolve the Gaussian budget: explicit value > extent-derived (ab initio) > 10000.
    if args.num_gaussians is not None:
        num_gaussians = args.num_gaussians
    elif args.vol is None and extent_radius_px is not None:
        # Tile the estimated support at ~1.6x the initial 1-voxel splat width:
        # close enough for neighbouring Gaussians to overlap into continuous
        # density, sparse enough not to waste points on redundancy.
        spacing_target = 1.6
        support_volume = (4.0 / 3.0) * np.pi * extent_radius_px ** 3
        num_gaussians = int(np.clip(support_volume / spacing_target ** 3, 5000, 40000))
        print(f"{bcolors.OKCYAN}Auto Gaussian budget from the estimated extent: "
              f"{num_gaussians} points{bcolors.ENDC}")
    else:
        num_gaussians = 10000

    # Initialize Gaussian positions
    fit_path = os.path.join(args.output_path, "Gaussian_volume_fitting")
    if args.vol is not None:
        if not os.path.isdir(os.path.join(fit_path)):
            # Mask preparation
            if args.mask is not None:
                mask_fit = mask
            else:
                mask_fit = ImageHandler().generateMask(inputFn=vol, boxsize=64)

            # Consensus volume
            model, _, _ = fit_volume(vol * mask_fit, mask=mask_fit, iterations=20000, learning_rate=0.001, n_init=num_gaussians, fixed_gaussians=True)

            # Adjust to images
            # model, _ = adjust_weights_to_images(model, args.md, mmap_output_dir, args.sr, learning_rate=0.0001,
            #                                     num_epochs=5, is_global=True, ctf_type=args.ctf_type)

            # Save model
            NeuralNetworkCheckpointer.save(model, fit_path)

            # Save volume
            vol = np.array(model(place_deltas=True))
            vol_splatted = np.array(model())
            ImageHandler().write(vol_splatted, os.path.join(args.output_path, "consensus_volume.mrc"), overwrite=True)
            ImageHandler().write(vol, os.path.join(args.output_path, "consensus_volume_deltas.mrc"), overwrite=True)
        else:
            model = NeuralNetworkCheckpointer.load(checkpoint_path=fit_path)

        # Prepare network (HetSIREN)
        factor = 0.5 * generator.md.getMetaDataImage(0).shape[0]
        coords = np.array(factor * model.means.get_value() + factor)
        coords = np.stack([coords[..., 2], coords[..., 1], coords[..., 0]], axis=1)
        coords = (coords - factor) / factor
        values = np.array(jax.nn.relu(model.weights.get_value()))
        sigma = jax.nn.relu(model.sigma_param.get_value())
    else:
        # Initialization ball sized to the measured particle, not a fixed guess.
        if extent_radius_px is not None:
            ball_radius = float(np.clip(extent_radius_px / (0.5 * xsize), 0.1, 0.9))
        else:
            ball_radius = 0.25
        coords = ball_radius * jnp.array(generate_sphere_points(num_gaussians) + np.random.normal(0, 0.1, (num_gaussians, 3)))
        # coords = generate_cylinder_points(num_gaussians, radius=0.25, height=1.0) + np.random.normal(0, 0.1, (num_gaussians, 3))
        values = jnp.full((num_gaussians,), 0.01)
        sigma = 1.0


    # # If exists, clean MMAP
    # if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

    # Bounds for the learned splat width. 'auto' anchors them to the fitted
    # initial width: enough head-room to adapt, not enough to blur the map into
    # hiding pose error or to collapse below the splat sampling limit.
    if args.no_sigma_bounds:
        sigma_bounds = None
    elif explicit_sigma_bounds is not None:
        sigma_bounds = explicit_sigma_bounds
    else:
        sigma_init_value = float(np.mean(np.asarray(sigma)))
        sigma_bounds = (min(0.5, 0.75 * sigma_init_value),
                        max(2.0, 1.5 * sigma_init_value))

    # Startup connectivity check: with N points tiling the support, mean point
    # spacing must stay below ~2 splat widths or the rendered density cannot be
    # continuous no matter how well training goes. Ab initio users have no
    # reference map to eyeball, so say it up front.
    if args.vol is None and args.mode == "train":
        radius_check = extent_radius_px if extent_radius_px is not None else 0.25 * xsize
        implied_spacing = ((4.0 / 3.0) * np.pi * radius_check ** 3 / num_gaussians) ** (1.0 / 3.0)
        sigma_now = float(np.mean(np.asarray(sigma)))
        sigma_max = sigma_bounds[1] if sigma_bounds is not None else 2.0 * sigma_now
        print(f"{bcolors.OKCYAN}Cloud geometry: {num_gaussians} points, implied spacing "
              f"{implied_spacing:.2f} px, splat width {sigma_now:.2f} px (max {sigma_max:.2f}){bcolors.ENDC}")
        if implied_spacing > 2.0 * sigma_max:
            print(f"{bcolors.WARNING}WARNING: implied point spacing exceeds twice the maximum splat "
                  f"width - the rendered density cannot be continuous. Increase --num_gaussians to "
                  f"~{int((4.0 / 3.0) * np.pi * radius_check ** 3 / (1.6 * sigma_now) ** 3)} or widen "
                  f"--sigma_bounds.{bcolors.ENDC}")

    # Random keys
    rng_seed = args.seed if args.seed is not None else random.randint(0, 2 ** 32 - 1)
    rng = jax.random.PRNGKey(rng_seed)
    rng, model_key, choice_key = jax.random.split(rng, 3)

    # Prepare network (ReconSIREN)
    reconsiren = ReconSIREN(coords, values, xsize, args.sr, ctf_type=args.ctf_type, symmetry_group=args.symmetry_group,
                            refine_current_assignment=args.refine_current_assignment, lat_dim=8, sigma=sigma,
                            bank_size=10000, learn_delta_volume=not args.do_not_learn_volume,
                            num_components=args.num_components,
                            use_anchor_rotations=not args.do_not_use_anchor_rotations,
                            optimization_profile=args.optimization_profile,
                            pose_head_rank=args.pose_head_rank,
                            pose_spatial_pool=args.pose_spatial_pool,
                            het_encoder_architecture=args.het_encoder_architecture,
                            consensus_parameterization=args.consensus_parameterization,
                            render_chunk_size=args.render_chunk_size,
                            candidate_chunk_size=args.candidate_chunk_size,
                            coarse_topk=args.coarse_topk,
                            coarse_scale=args.coarse_scale,
                            coarse_gaussians=args.coarse_gaussians,
                            heterogeneity_profile=args.heterogeneity_profile,
                            het_encoder_size=args.het_encoder_size,
                            het_residual_to_consensus=args.het_residual_to_consensus,
                            het_center_decoder=args.het_center_decoder,
                            het_coordinate_scale=args.het_coordinate_scale,
                            het_amplitude_scale=args.het_amplitude_scale,
                            het_loss_scales=args.het_loss_scales,
                            het_loss_weights=args.het_loss_weights,
                            het_mask_radius=args.het_mask_radius,
                            het_normalize_target=args.het_normalize_target,
                            het_variance_weight=args.het_variance_weight,
                            het_covariance_weight=args.het_covariance_weight,
                            het_min_std=args.het_min_std,
                            het_start_epoch=args.het_start_epoch,
                            het_freeze_consensus=args.het_freeze_consensus,
                            het_latent_bank_size=args.het_latent_bank_size,
                            fused_envelope=not args.no_fused_envelope,
                            sigma_bounds=sigma_bounds,
                            per_candidate_shifts=not args.no_per_candidate_shifts,
                            rngs=nnx.Rngs(model_key))

    # Reload network
    if args.reload is not None:
        reconsiren = NeuralNetworkCheckpointer.load(os.path.join(args.reload, "ReconSIREN"))


    # Train network
    if args.mode == "train":

        reconsiren.train()

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "ReconSIREN_metrics"))

        # Jitted functions for volume prediction
        @nnx.jit
        def decode_volume(model):
            return model.delta_volume_decoder.decode_volume(sigma=model.get_std(),
                                                            analytic=model.fused_envelope)

        # Consensus point cloud (voxel-centered coords, amplitudes) for the
        # host-side kNN graph and shrink-wrap support updates.
        @nnx.jit
        def decode_cloud(model):
            return model.delta_volume_decoder()

        # Decode volume
        @nnx.jit
        def decode_het_volume(model, x):
            return model.decode_het_volume(x)

        def write_volume(volume, path):
            """Write a map (runs on the logging thread)."""
            ImageHandler().write(volume, path, overwrite=True)

        def write_het_volumes(volumes, out_dir):
            """Write the per-cluster heterogeneous maps (runs on the logging thread)."""
            for idx, volume in enumerate(volumes, start=1):
                ImageHandler().write(volume, os.path.join(out_dir, f"reconsiren_hetmap_{idx:02d}.mrc"),
                                     overwrite=True)

        # Prepare data loader
        data_loader_train, data_loader_val = generator.return_grain_dataset(batch_size=args.batch_size,
                                                                            shuffle="global", num_epochs=None,
                                                                            num_workers=-1, num_threads=1,
                                                                            split_fraction=args.dataset_split_fraction,
                                                                            load_to_ram=args.load_images_to_ram)
        steps_per_epoch = int(int(args.dataset_split_fraction[0] * len(generator.md)) / args.batch_size)
        steps_per_val = int(int(args.dataset_split_fraction[1] * len(generator.md)) / args.batch_size)

        # Example of training data for Tensorboard
        with closing(iter(data_loader_train)) as iter_data_loader:
            x_example, labels_example = next(iter_data_loader)
            whiten_filter = None
            if args.whiten_loss_weight > 0.0:
                # One-time dataset noise profile from the particle solvent
                # corners; the whitened consensus loss reuses this filter at
                # every step.
                noise_psd = estimate_noise_psd(jnp.asarray(x_example, jnp.float32))
                whiten_filter = whitening_filter_2d(noise_psd, (xsize, xsize))
                print(f"{bcolors.OKCYAN}Estimated the noise power spectrum for loss "
                      f"whitening from {x_example.shape[0]} particles{bcolors.ENDC}")
            x_example = jax.vmap(min_max_scale)(x_example)
            writer.add_images("Training data batch", x_example, dataformats="NHWC")

        # Learning rate scheduler
        # total_steps = args.epochs * len(data_loader)
        # lr_schedule_pose = CosineAnnealingScheduler.getScheduler(peak_value=4. * args.learning_rate, total_steps=total_steps, warmup_frac=0.1, init_value=args.learning_rate, end_value=0.0)
        # lr_schedule_volume = CosineAnnealingScheduler.getScheduler(peak_value=4. * 1e-3, total_steps=total_steps, warmup_frac=0.1, init_value=1e-3, end_value=0.0)
        # lr_schedule_het = CosineAnnealingScheduler.getScheduler(peak_value=4. * 1e-3, total_steps=total_steps, warmup_frac=0.1, init_value=1e-3, end_value=0.0)

        # Optimizers (ReconSIREN)
        params_pose = nnx.All(nnx.Param, nnx.PathContains('encoder_pose'))
        params_volume = nnx.All(nnx.Param, nnx.PathContains('delta_volume_decoder'))
        params_het = nnx.All(nnx.Param, (nnx.PathContains('encoder_het'), nnx.PathContains('delta_het_decoder')))
        optimizer_pose = nnx.Optimizer(reconsiren,  optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(args.learning_rate, eps=1e-6)), wrt=params_pose)
        optimizer_volume = nnx.Optimizer(reconsiren, optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(learning_rate=1e-4, eps=1e-6)), wrt=params_volume)
        optimizer_het = nnx.Optimizer(reconsiren, optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(learning_rate=1e-4, eps=1e-6)), wrt=params_het)
        heterogeneity_profile = reconsiren.heterogeneity_profile
        het_start_epoch = reconsiren.het_start_epoch
        het_freeze_consensus = reconsiren.het_freeze_consensus
        # Geometry feature state (host side): the kNN graph and shrink-wrap
        # support are refreshed from the live cloud at every epoch boundary.
        geometry_priors_enabled = ((args.spacing_prior_weight > 0.0
                                    or args.amplitude_smoothness_weight > 0.0)
                                   and not args.do_not_learn_volume)
        support_enabled = args.support_weight > 0.0 and not args.do_not_learn_volume
        recycling_enabled = (not args.no_point_recycling
                             and not args.do_not_learn_volume
                             and reconsiren.delta_volume_decoder.parameterization == "direct")
        if (not args.no_point_recycling and not args.do_not_learn_volume
                and not recycling_enabled):
            print(f"{bcolors.WARNING}Point recycling requires the 'direct' consensus "
                  f"parameterization; disabled for this run{bcolors.ENDC}")
        knn_indices = None
        support_center_value = None
        support_radius_value = None
        # The adaptive curriculum is gated on winner stability, so it needs the
        # per-particle tracker even when the extra TensorBoard panels are off.
        pose_diagnostics_enabled = (args.pose_diagnostics
                                    or args.pose_curriculum == "adaptive")
        pose_diagnostics_tracker = (
            _PoseDiagnosticsTracker(len(generator.md), reconsiren.symmetry_matrices)
            if pose_diagnostics_enabled else None)
        pose_curriculum_controller = (
            _PoseCurriculumController(
                xsize, args.candidate_frequency_scales,
                switch_threshold=args.pose_curriculum_switch_threshold,
                pose_threshold_degrees=args.pose_curriculum_pose_threshold,
                min_epochs=args.pose_curriculum_min_epochs,
                max_epochs=args.pose_curriculum_max_epochs,
                temperatures=args.pose_curriculum_temperatures)
            if args.pose_curriculum == "adaptive" else None)
        curriculum_state_path = os.path.join(
            args.output_path, "ReconSIREN_CHECKPOINT", "pose_curriculum.json")
        graphdef, state = nnx.split((reconsiren, optimizer_pose, optimizer_volume, optimizer_het))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"),
                                                                                        optimizer_pose, optimizer_volume, optimizer_het)
            print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
            if (pose_curriculum_controller is not None
                    and os.path.isfile(curriculum_state_path)):
                with open(curriculum_state_path) as fid:
                    pose_curriculum_controller.load_state_dict(json.load(fid))
        else:
            resume_epoch = 0

        # Logging cadence + background offload of the host-side logging work.
        logger = TrainingLogger(image_every=args.log_images_every,
                                landscape_every=args.log_landscape_every,
                                checkpoint_every=args.log_checkpoint_every,
                                steps_per_epoch=steps_per_epoch,
                                time_budget=args.log_time_budget,
                                background=not args.log_sync).start()

        # Training loop (ReconSIREN)
        training_volume_log = " / volume" if not args.do_not_learn_volume else ""
        print(f"{bcolors.OKCYAN}\n###### Training angular assignment / shifts{training_volume_log} / heterogeneity... ######")

        i = 0
        pbar = tqdm(range(resume_epoch * steps_per_epoch, args.epochs * steps_per_epoch), file=sys.stdout, ascii=" >=",
                    colour="green",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        with closing(iter(data_loader_train)) as iter_data_loader_train, closing(iter(data_loader_val)) as iter_data_loader_val:
            for total_steps in pbar:
                (x, labels) = next(iter_data_loader_train)
                epoch_index = total_steps // steps_per_epoch

                if total_steps % steps_per_epoch == 0:
                    # Let the adaptive curriculum consume the finished epoch's
                    # winner-stability summary before the tracker resets.
                    stage_advanced = False
                    if (pose_curriculum_controller is not None
                            and pose_diagnostics_tracker is not None
                            and pose_diagnostics_tracker.absolute_margins
                            and total_steps > 1500):
                        if pose_curriculum_controller.observe_epoch(
                                pose_diagnostics_tracker.summary()):
                            stage_advanced = True
                            new_size = pose_curriculum_controller.scoring_size
                            print(f"\n{bcolors.OKCYAN}Pose curriculum advanced to stage "
                                  f"{pose_curriculum_controller.stage}: scoring candidates "
                                  f"at {new_size}/{xsize} pixels{bcolors.ENDC}")

                    # Refresh the geometry state from the live cloud: recycle
                    # dead points on schedule, then rebuild the kNN graph and
                    # the shrink-wrap support from the updated positions.
                    if recycling_enabled and total_steps > 1500 and (
                            stage_advanced or epoch_index % args.recycle_every == 0):
                        state, n_recycled = recycle_dead_points_reconsiren(
                            graphdef, state, rng, args.recycle_dead_fraction)
                        rng, _ = jax.random.split(rng)
                        n_recycled = int(n_recycled)
                        if n_recycled:
                            print(f"\n{bcolors.OKCYAN}Recycled {n_recycled} dead Gaussians "
                                  f"onto the structure{bcolors.ENDC}")
                    if geometry_priors_enabled or support_enabled:
                        reconsiren_cloud, _, _, _ = nnx.merge(graphdef, state)
                        cloud_coords, cloud_values = decode_cloud(reconsiren_cloud)
                        cloud_coords = np.asarray(cloud_coords[0], np.float32)
                        cloud_values = np.asarray(cloud_values[0], np.float32)
                        if geometry_priors_enabled:
                            n_neighbors = int(min(args.knn_neighbors, cloud_coords.shape[0] - 1))
                            nn_graph = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cloud_coords)
                            _, nn_idx = nn_graph.kneighbors(cloud_coords)
                            knn_indices = jnp.asarray(nn_idx[:, 1:], dtype=jnp.int32)
                        if support_enabled:
                            mass = np.maximum(cloud_values, 0.0)
                            total_mass = float(mass.sum())
                            if total_mass > 0.0:
                                center = (mass[:, None] * cloud_coords).sum(axis=0) / total_mass
                                distances = np.linalg.norm(cloud_coords - center[None, :], axis=1)
                                order = np.argsort(distances)
                                cumulative = np.cumsum(mass[order]) / total_mass
                                quantile_index = int(min(np.searchsorted(cumulative, 0.99),
                                                         distances.shape[0] - 1))
                                # Margin over the 99%-mass radius; floored so the
                                # support can never collapse onto the cloud core.
                                support_radius_value = float(max(
                                    1.15 * distances[order][quantile_index],
                                    4.0 * float(np.mean(np.asarray(sigma)))))
                                support_center_value = jnp.asarray(center, dtype=jnp.float32)

                    total_loss = 0
                    total_recon_loss = 0
                    total_recon_het_loss = 0
                    total_candidate_coverage_loss = 0
                    total_assignment_entropy = 0
                    total_anchor_deviation = 0
                    total_variance_loss = 0
                    total_covariance_loss = 0
                    total_latent_std = 0
                    total_coordinate_rms = 0
                    total_amplitude_rms = 0
                    total_projection_rms = 0
                    total_normalized_target_rms = 0
                    total_consensus_amplitude_mean = 0
                    total_consensus_amplitude_rms = 0
                    total_consensus_amplitude_max = 0
                    total_consensus_active_amplitude_fraction = 0
                    total_low_frequency_recon_loss = 0
                    total_reconstruction_objective = 0
                    total_frequency_full_agreement = 0
                    total_frequency_full_disadvantage_std = 0
                    projection_diagnostic_steps = 0
                    total_validation_loss = 0

                    # For progress bar (TQDM)
                    step = 1
                    step_validation = 1
                    pbar.set_description(f"Epoch {int(total_steps / steps_per_epoch + 1)}/{args.epochs}")

                    if logger.should("images", i):
                        pbar.set_postfix_str(f"{bcolors.WARNING}Generating intermediate results...{bcolors.ENDC}")

                        with logger.section():
                            # Example of predicted data for Tensorboard
                            reconsiren, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)
                            volume = decode_volume(reconsiren)
                            middle_slize = int(np.round(0.5 * volume.shape[-1]))
                            slice_xy, slice_xz, slice_yz = (min_max_scale(volume[0, middle_slize, :, :]),
                                                            min_max_scale(volume[0, :, middle_slize, :]),
                                                            min_max_scale(volume[0, :, :, middle_slize]))
                            slices = np.stack([slice_xy, slice_xz, slice_yz], axis=0)[..., None]
                            volume = np.array(volume)

                        logger.submit(write_volume, volume,
                                      os.path.join(args.output_path, "reconsiren_map_intermediate.mrc"))
                        logger.submit(writer.add_images, "Predicted volume (slices)", slices,
                                      dataformats="NHWC", global_step=i)

                    if logger.should("landscape", i):
                        with logger.section():
                            reconsiren, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)

                            # Plot candidates and winners independently.  Counts
                            # avoid displaying the zero-initialized bank tail.
                            angular_banks = (
                                ("candidates", reconsiren.memory_bank.get(),
                                 reconsiren.candidate_bank_count.get_value()),
                                ("winners", reconsiren.winner_memory_bank.get(),
                                 reconsiren.winner_bank_count.get_value()),
                            )
                            for scope, bank, count in angular_banks:
                                n_valid = int(count)
                                if n_valid == 0:
                                    continue
                                directions = np.array(bank[:n_valid])
                                dir_x, dir_y, dir_z = (
                                    directions[:, 0], directions[:, 1], directions[:, 2])
                                beta = jnp.arccos(jnp.clip(dir_z, -1.0, 1.0))
                                alpha = jnp.arctan2(dir_y, dir_x)
                                euler_angles = jnp.stack([alpha, beta], axis=-1)
                                fig, _ = plot_angular_distribution(euler_angles)
                                writer.add_figure(
                                    f"Angular distribution density ({scope})",
                                    fig, global_step=i)

                            decoded_centers = []
                            if epoch_index >= het_start_epoch:
                                n_latent_steps = int(min(
                                    steps_per_epoch,
                                    np.ceil(LATENTS_FOR_CLUSTERING / args.batch_size)))
                                latents = []
                                graphdef_aux, state_aux = nnx.split(reconsiren)
                                for _ in range(n_latent_steps):
                                    (x_latent, labels_latent) = next(iter_data_loader_train)
                                    _, _, latent = predict_angular_assignment_step_reconsiren(
                                        graphdef_aux, state_aux, x_latent, labels_latent,
                                        md_columns, rng)
                                    latents.append(np.array(latent))
                                latents = np.concatenate(latents, axis=0)
                                n_clusters = int(min(10, latents.shape[0]))
                                kmeans = KMeans(n_clusters=n_clusters).fit(latents)
                                decoded_centers = [
                                    np.array(decode_het_volume(reconsiren, center[None, ...]))
                                    for center in kmeans.cluster_centers_]

                        if decoded_centers:
                            logger.submit(write_het_volumes, decoded_centers, args.output_path)

                    # Save checkpoint model
                    if logger.should("checkpoint", i):
                        with logger.section():
                            NeuralNetworkCheckpointer.save_intermediate(graphdef, state,
                                                                        os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"),
                                                                        epoch=i, wait=False)
                            if pose_curriculum_controller is not None:
                                os.makedirs(os.path.dirname(curriculum_state_path),
                                            exist_ok=True)
                                with open(curriculum_state_path, "w") as fid:
                                    json.dump(pose_curriculum_controller.state_dict(), fid)

                    i += 1

                # Preserve the historical 1,500-step stochastic warm-up.  The
                # selected-projection curriculum begins with hard assignment.
                if total_steps <= 1500:
                    tau = 1e-3
                    use_tau = True
                    assignment_mode = "hard"
                    candidate_scoring_size = xsize
                    consensus_multiscale_size = xsize
                    consensus_multiscale_weight = 0.0
                else:
                    tau = 0.0
                    use_tau = False
                    assignment_mode = "hard"
                    if pose_curriculum_controller is not None:
                        # Score candidates in the stability-gated band and keep
                        # the consensus multiscale loss on the same band.
                        candidate_scoring_size = pose_curriculum_controller.scoring_size
                        consensus_multiscale_size = candidate_scoring_size
                        consensus_multiscale_weight = (
                            args.consensus_multiscale_weight
                            * pose_curriculum_controller.multiscale_weight_multiplier)
                        tau = pose_curriculum_controller.temperature
                        assignment_mode = "sampled" if tau > 0.0 else "hard"
                    else:
                        candidate_scoring_size = xsize
                        (consensus_multiscale_size,
                         multiscale_weight_multiplier) = _consensus_multiscale_size_and_weight(
                            xsize, total_steps - 1501, steps_per_epoch,
                            args.consensus_multiscale_epochs,
                            args.consensus_multiscale_scales)
                        consensus_multiscale_weight = (
                            args.consensus_multiscale_weight
                            * multiscale_weight_multiplier)
                uniform_weight = 0.1

                # Whitening only makes sense once poses are scored at full
                # resolution: before that the boosted shells are pure noise.
                # It ramps in as the multiscale curriculum ramps out.
                whiten_weight_step = 0.0
                if (whiten_filter is not None and not use_tau
                        and candidate_scoring_size >= xsize):
                    whiten_weight_step = (args.whiten_loss_weight
                                          * (1.0 - consensus_multiscale_weight))
                apply_loss_whitening = whiten_weight_step > 0.0

                # Connectivity priors also wait for full-resolution scoring:
                # equalising or gap-closing a cloud that still has wrong poses
                # would only make garbage look connected.
                apply_geometry_priors_step = (
                    geometry_priors_enabled and knn_indices is not None
                    and not use_tau and candidate_scoring_size >= xsize)
                apply_support_step = (support_enabled
                                      and support_radius_value is not None)

                # Coarse-to-fine cloud: extra render blur annealed with the
                # pose curriculum, so cloud detail waits for pose stability.
                extra_blur_step = 0.0
                if args.cloud_blur_max > 0.0:
                    if total_steps <= 1500:
                        coarse_fraction = 1.0
                    elif pose_curriculum_controller is not None:
                        n_stages = len(pose_curriculum_controller.scales)
                        coarse_fraction = 1.0 - (
                            min(pose_curriculum_controller.stage, n_stages) / n_stages)
                    elif args.consensus_multiscale_weight > 0.0:
                        coarse_fraction = (consensus_multiscale_weight
                                           / args.consensus_multiscale_weight)
                    else:
                        coarse_fraction = 0.0
                    extra_blur_step = args.cloud_blur_max * coarse_fraction

                coverage_steps = args.candidate_coverage_epochs * steps_per_epoch
                apply_candidate_coverage = (
                    coverage_steps > 0 and total_steps < coverage_steps
                    and args.candidate_coverage_weight > 0.0)
                candidate_coverage_weight = (
                    args.candidate_coverage_weight
                    if apply_candidate_coverage else 0.0)
                train_heterogeneity = epoch_index >= het_start_epoch
                train_pose_volume = not (
                    heterogeneity_profile == "anti_collapse"
                    and train_heterogeneity and het_freeze_consensus)
                train_result = train_step_reconsiren(
                    graphdef, state, x, labels, md_columns, rng,
                    lambda_uniform=uniform_weight, tau=tau, use_tau=use_tau,
                    assignment_mode=assignment_mode,
                    apply_candidate_coverage=apply_candidate_coverage,
                    candidate_coverage_weight=candidate_coverage_weight,
                    candidate_coverage_bins=args.candidate_coverage_bins,
                    candidate_coverage_kappa=args.candidate_coverage_kappa,
                    candidate_bank_samples=args.candidate_bank_samples,
                    candidate_bank_mix=args.candidate_bank_mix,
                    candidate_scoring_size=candidate_scoring_size,
                    consensus_multiscale_size=consensus_multiscale_size,
                    consensus_multiscale_weight=consensus_multiscale_weight,
                    apply_loss_whitening=apply_loss_whitening,
                    whiten_weight=whiten_weight_step,
                    whiten_filter=whiten_filter,
                    apply_amplitude_l1=args.amplitude_l1 > 0.0,
                    amplitude_l1_weight=args.amplitude_l1,
                    extra_blur=extra_blur_step,
                    apply_geometry_priors=apply_geometry_priors_step,
                    spacing_weight=args.spacing_prior_weight,
                    smoothness_weight=args.amplitude_smoothness_weight,
                    neighbor_indices=knn_indices if apply_geometry_priors_step else None,
                    apply_support=apply_support_step,
                    support_center=support_center_value if apply_support_step else None,
                    support_radius=support_radius_value if apply_support_step else 0.0,
                    support_weight=args.support_weight,
                    train_pose_volume=train_pose_volume,
                    train_heterogeneity=train_heterogeneity,
                    return_metrics=True,
                    return_pose_diagnostics=pose_diagnostics_enabled)
                if pose_diagnostics_enabled:
                    loss, metrics, pose_diagnostics, state, rng = train_result
                    (winner_rotations, selected_heads, best_heads,
                     absolute_margins, relative_margins, standardized_margins,
                     median_normalized_margins, score_entropies) = pose_diagnostics
                    pose_diagnostics_tracker.update(
                        labels, winner_rotations, selected_heads, best_heads,
                        absolute_margins, relative_margins, standardized_margins,
                        median_normalized_margins, score_entropies, epoch_index)
                else:
                    loss, metrics, state, rng = train_result
                (recon_loss, recon_het_loss, loss_uniform, candidate_coverage_loss,
                 assignment_entropy, anchor_deviation, variance_loss,
                 covariance_loss, latent_std, coordinate_rms, amplitude_rms,
                 projection_rms, normalized_target_rms, consensus_amplitude_mean,
                 consensus_amplitude_rms, consensus_amplitude_max,
                 consensus_active_amplitude_fraction,
                 low_frequency_recon_loss,
                 reconstruction_objective,
                 frequency_full_agreement,
                 frequency_full_disadvantage_std) = metrics
                total_loss += loss
                total_recon_loss += recon_loss
                total_recon_het_loss += recon_het_loss
                total_candidate_coverage_loss += candidate_coverage_loss
                total_assignment_entropy += assignment_entropy
                total_anchor_deviation += anchor_deviation
                total_variance_loss += variance_loss
                total_covariance_loss += covariance_loss
                total_latent_std += latent_std
                total_coordinate_rms += coordinate_rms
                total_amplitude_rms += amplitude_rms
                total_consensus_amplitude_mean += consensus_amplitude_mean
                total_consensus_amplitude_rms += consensus_amplitude_rms
                total_consensus_amplitude_max += consensus_amplitude_max
                total_consensus_active_amplitude_fraction += consensus_active_amplitude_fraction
                total_low_frequency_recon_loss += low_frequency_recon_loss
                total_reconstruction_objective += reconstruction_objective
                total_frequency_full_agreement += frequency_full_agreement
                total_frequency_full_disadvantage_std += frequency_full_disadvantage_std
                if train_pose_volume:
                    total_projection_rms += projection_rms
                    total_normalized_target_rms += normalized_target_rms
                    projection_diagnostic_steps += 1

                # Summary writer (training loss)
                if logger.should_log_scalars(step):
                    mean_loss = float(total_loss) / step
                    mean_recon_loss = float(total_recon_loss) / step
                    mean_recon_het_loss = float(total_recon_het_loss) / step

                    writer.add_scalar('Training loss (ReconSIREN)',
                                      mean_loss,
                                      i * steps_per_epoch + step)

                    writer.add_scalars('Reconstruction loss (ReconSIREN)',
                                       {"consensus": mean_recon_loss,
                                        "heterogeneity": mean_recon_het_loss},
                                       i * steps_per_epoch + step)

                    writer.add_scalars('Heterogeneity diagnostics (ReconSIREN)',
                                       {"latent_std": float(total_latent_std) / step,
                                        "coordinate_rms": float(total_coordinate_rms) / step,
                                        "amplitude_rms": float(total_amplitude_rms) / step,
                                        "variance_penalty": float(total_variance_loss) / step,
                                        "covariance_penalty": float(total_covariance_loss) / step},
                                       i * steps_per_epoch + step)

                    writer.add_scalars('Pose exploration diagnostics (ReconSIREN)',
                                       {"assignment_entropy": float(total_assignment_entropy) / step,
                                        "candidate_coverage_kl": float(total_candidate_coverage_loss) / step,
                                        "mean_anchor_deviation_degrees": float(total_anchor_deviation) / step,
                                        "temperature": float(tau),
                                        "candidate_coverage_weight": float(candidate_coverage_weight)},
                                       i * steps_per_epoch + step)

                    writer.add_scalars(
                        'Selected reconstruction curriculum (ReconSIREN)',
                        {"full_resolution_loss": mean_recon_loss,
                         "low_frequency_loss": (
                             float(total_low_frequency_recon_loss) / step),
                         "blended_objective": (
                             float(total_reconstruction_objective) / step),
                         "low_frequency_weight": float(consensus_multiscale_weight),
                         "loss_resolution_pixels": float(consensus_multiscale_size),
                         "loss_resolution_fraction": (
                             float(consensus_multiscale_size) / xsize),
                         "active": float(consensus_multiscale_weight > 0.0)},
                        i * steps_per_epoch + step)

                    curriculum_scalars = {
                        "scoring_resolution_pixels": float(candidate_scoring_size),
                        "scoring_resolution_fraction": (
                            float(candidate_scoring_size) / xsize),
                        "low_frequency_full_winner_agreement": (
                            float(total_frequency_full_agreement) / step),
                        "selected_full_loss_disadvantage_std": (
                            float(total_frequency_full_disadvantage_std) / step),
                        "active": float(candidate_scoring_size < xsize)}
                    if pose_curriculum_controller is not None:
                        curriculum_scalars["stage"] = float(
                            pose_curriculum_controller.stage)
                        curriculum_scalars["epochs_in_stage"] = float(
                            pose_curriculum_controller.epochs_in_stage)
                    writer.add_scalars(
                        'Candidate frequency curriculum (ReconSIREN)',
                        curriculum_scalars,
                        i * steps_per_epoch + step)

                    if args.pose_diagnostics and pose_diagnostics_tracker is not None:
                        pose_summary = pose_diagnostics_tracker.summary()
                        writer.add_scalars(
                            'Winner standardized confidence (ReconSIREN)',
                            {"margin_over_candidate_std_mean": pose_summary["standardized_margin_mean"],
                             "margin_over_candidate_std_median": pose_summary["standardized_margin_median"],
                             "margin_over_candidate_std_p10": pose_summary["standardized_margin_p10"],
                             "margin_below_0.1_std": pose_summary["standardized_margin_below_0_1"],
                             "margin_below_0.25_std": pose_summary["standardized_margin_below_0_25"],
                             "margin_below_0.5_std": pose_summary["standardized_margin_below_0_5"],
                             "candidate_score_entropy": pose_summary["candidate_score_entropy_mean"]},
                            i * steps_per_epoch + step)
                        writer.add_scalars(
                            'Winner margin over available separation (ReconSIREN)',
                            {"mean": pose_summary["median_normalized_margin_mean"],
                             "median": pose_summary["median_normalized_margin_median"]},
                            i * steps_per_epoch + step)
                        writer.add_scalars(
                            'Winner loss confidence (ReconSIREN)',
                            {"relative_margin_mean": pose_summary["relative_margin_mean"],
                             "relative_margin_median": pose_summary["relative_margin_median"],
                             "relative_margin_p10": pose_summary["relative_margin_p10"],
                             "ambiguous_below_1pct": pose_summary["relative_margin_below_1pct"],
                             "ambiguous_below_5pct": pose_summary["relative_margin_below_5pct"],
                             "ambiguous_below_10pct": pose_summary["relative_margin_below_10pct"],
                             "selected_is_top1_fraction": pose_summary["selected_is_top1_fraction"]},
                            i * steps_per_epoch + step)
                        writer.add_scalar(
                            'Winner top1-top2 absolute loss margin (ReconSIREN)',
                            pose_summary["absolute_margin_mean"],
                            i * steps_per_epoch + step)
                        writer.add_scalars(
                            'Consensus Gaussian amplitude scale (ReconSIREN)',
                            {"mean": float(total_consensus_amplitude_mean) / step,
                             "rms": float(total_consensus_amplitude_rms) / step,
                             "max": float(total_consensus_amplitude_max) / step,
                             "active_fraction": float(total_consensus_active_amplitude_fraction) / step},
                            i * steps_per_epoch + step)
                        if projection_diagnostic_steps:
                            mean_projection_rms = (
                                float(total_projection_rms) / projection_diagnostic_steps)
                            mean_target_rms = (
                                float(total_normalized_target_rms) / projection_diagnostic_steps)
                            writer.add_scalars(
                                'Selected projection scale (ReconSIREN)',
                                {"prediction_rms": mean_projection_rms,
                                 "normalized_target_rms": mean_target_rms,
                                 "prediction_to_target_rms": (
                                     mean_projection_rms / max(mean_target_rms, 1e-8))},
                                i * steps_per_epoch + step)
                        if "pose_change_mean_degrees" in pose_summary:
                            writer.add_scalars(
                                'Winner pose change degrees (ReconSIREN)',
                                {"mean": pose_summary["pose_change_mean_degrees"],
                                 "median": pose_summary["pose_change_median_degrees"],
                                 "p90": pose_summary["pose_change_p90_degrees"]},
                                i * steps_per_epoch + step)
                            writer.add_scalars(
                                'Winner instability fractions (ReconSIREN)',
                                {"pose_change_over_5deg": pose_summary["pose_change_over_5deg_fraction"],
                                 "pose_change_over_15deg": pose_summary["pose_change_over_15deg_fraction"],
                                 "pose_change_over_30deg": pose_summary["pose_change_over_30deg_fraction"],
                                 "head_switch": pose_summary["head_switch_fraction"],
                                 "comparison_coverage": pose_summary["comparison_coverage_fraction"]},
                                i * steps_per_epoch + step)

                    # Progress bar update  (TQDM)
                    stage = "joint" if train_pose_volume and train_heterogeneity else (
                        "heterogeneity" if train_heterogeneity else "consensus")
                    pbar.set_postfix_str(
                        f"stage={stage} | loss={mean_loss:.5f} | "
                        f"cons={mean_recon_loss:.5f} | het={mean_recon_het_loss:.5f}")

                # # Summary writer (validation loss)  FIXME: This fails with StopIteration
                # if step % int(np.ceil(0.5 * steps_per_epoch)) == 0:
                #     # Run validation step
                #     pbar.set_postfix_str(f"{bcolors.WARNING}Running validation step...{bcolors.ENDC}")
                #     for _ in range(steps_per_val):
                #         (x_validation, labels_validation) = next(iter_data_loader_val)
                #         loss_validation = validation_step_reconsiren(graphdef, state, x_validation,
                #                                                      labels_validation,
                #                                                      md_columns, rng)
                #         total_validation_loss += loss_validation
                #         step_validation += 1
                #
                #     writer.add_scalars('Reconstruction loss (ReconSIREN)',
                #                        {"validation": total_validation_loss / step_validation},
                #                        i * steps_per_epoch + step)

                step += 1

        reconsiren, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)

        # Let the background logging thread and the asynchronous checkpoint write finish
        # before the process moves on -- in particular before the checkpoint folder is
        # removed below.
        logger.close()
        NeuralNetworkCheckpointer.wait_for_pending()

        # Save model
        NeuralNetworkCheckpointer.save(reconsiren, os.path.join(args.output_path, "ReconSIREN"))

        # Remove checkpoint (cadence may be disabled or not yet reached, so it may not exist)
        checkpoint_dir = os.path.join(args.output_path, "ReconSIREN_CHECKPOINT")
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

    elif args.mode == "predict":  # TODO: Save angles here

        reconsiren.eval()

        # Prepare data loader
        data_loader = generator.return_grain_dataset(batch_size=args.batch_size, shuffle=False, num_epochs=1,
                                                     num_workers=-1, load_to_ram=args.load_images_to_ram)
        steps_per_epoch = int(np.ceil(len(generator.md) / args.batch_size))

        # Jitted functions for volume prediction. The exported map must use the
        # learned splat width, not the decode_volume default of 1.0.
        decode_volume = jax.jit(lambda: reconsiren.delta_volume_decoder.decode_volume(
            sigma=reconsiren.get_std(), analytic=reconsiren.fused_envelope))
        decode_het_volume = jax.jit(reconsiren.decode_het_volume)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting angular assignment / shifts... ######")

        # For progress bar (TQDM)
        pbar = tqdm(data_loader, file=sys.stdout, ascii=" >=", colour="green", total=steps_per_epoch,
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        graphdef, state = nnx.split(reconsiren)
        md_pred = generator.md
        latents = []
        for (x, labels) in pbar:
            rotations, shifts, latent = predict_angular_assignment_step_reconsiren(
                graphdef, state, x, labels, md_columns, rng)

            # Convert rotation to Euler angles in Xmipp format
            euler_angles = xmippEulerFromMatrix(rotations)

            # Convert to Numpy
            euler_angles, shifts = np.array(euler_angles), np.array(shifts)

            # Store in metadata
            md_pred[labels, 'angleRot'] = euler_angles[:, 0]
            md_pred[labels, 'angleTilt'] = euler_angles[:, 1]
            md_pred[labels, 'anglePsi'] = euler_angles[:, 2]
            md_pred[labels, 'shiftX'] = shifts[:, 0]
            md_pred[labels, 'shiftY'] = shifts[:, 1]

            # Save latents to list
            latents.append(latent)

        # Save latents to metadata
        latents = np.concatenate(latents, axis=0)
        md_pred[:, 'latent_space'] = np.asarray([",".join(np.char.mod('%f', item)) for item in latents])

        md_pred.write(os.path.join(args.output_path, "predicted_pose_shifts" + os.path.splitext(args.md)[1]),
                      updateImagePaths=True)

        # Predict volume
        print(f"{bcolors.OKCYAN}\n###### Predicting volume... ######")

        decoded_volume = decode_volume()
        ImageHandler().write(np.array(decoded_volume), os.path.join(args.output_path, "reconsiren_map.mrc"), overwrite=True)

        if not args.no_sharpened_map:
            # Companion map with the known splat envelope divided back out
            # (bounded-gain Wiener); the standard map above stays untouched.
            sharpened = _sharpen_gaussian_envelope(jnp.asarray(decoded_volume[0]),
                                                   reconsiren.get_std(),
                                                   reg=args.sharpened_map_reg)
            ImageHandler().write(np.array(sharpened),
                                 os.path.join(args.output_path, "reconsiren_map_sharpened.mrc"),
                                 overwrite=True)

        if not args.no_equalized_map:
            # Tracing companion: gamma-compressed masses flatten the contrast
            # along the structure so one iso-surface threshold shows the whole
            # chain. Deliberately decoupled from true occupancy - read topology
            # here, read confidence in the physical map.
            cloud_coords, cloud_values = reconsiren.delta_volume_decoder()
            masses = np.asarray(cloud_values[0], np.float32)
            positive = masses[masses > 0.0]
            if positive.size:
                mean_mass = float(positive.mean())
                equalized_masses = np.where(
                    masses > 0.02 * mean_mass,
                    mean_mass * (masses / mean_mass) ** args.equalized_map_gamma,
                    0.0).astype(np.float32)
                equalized = reconsiren.delta_volume_decoder.decode_volume(
                    coords_values=(cloud_coords, jnp.asarray(equalized_masses)[None, ...]),
                    sigma=reconsiren.get_std(), analytic=reconsiren.fused_envelope)
                ImageHandler().write(np.array(equalized[0]),
                                     os.path.join(args.output_path, "reconsiren_map_equalized.mrc"),
                                     overwrite=True)

        # Predict heterogeneous states
        kmeans = KMeans(n_clusters=20).fit(latents)
        centers = kmeans.cluster_centers_
        idx = 1
        for center in centers:
            decoded = decode_het_volume(center[None, ...])
            ImageHandler().write(np.array(decoded), os.path.join(args.output_path, f"reconsiren_hetmap_{idx:02d}.mrc"), overwrite=True)
            idx += 1

    # If exists, clean MMAP
    # if not args.load_images_to_ram and os.path.isdir(generator.mmap_output_dir):
    #     shutil.rmtree(generator.mmap_output_dir)
