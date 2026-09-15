import jax
from jax import numpy as jnp
from functools import partial


MORTON_BITS = 10


def _spread_bits(v):
    # Insert two zero bits between the bits of a 10-bit integer
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def morton_codes(points, shift=0.0):
    """Z-order codes of points (N, 3), after normalizing them to the unit cube and shifting by a cell fraction"""
    lower = jnp.min(points, axis=0)
    extent = jnp.maximum(jnp.max(points, axis=0) - lower, 1e-6)
    cells = 2 ** MORTON_BITS
    grid = (points - lower) / extent * (cells - 1) + shift
    grid = jnp.clip(jnp.floor(grid), 0, cells - 1).astype(jnp.uint32)
    return (_spread_bits(grid[:, 0]) | (_spread_bits(grid[:, 1]) << 1)
            | (_spread_bits(grid[:, 2]) << 2))


def _window_candidates(points, valid, k, window, shift):
    n = points.shape[0]
    order = jnp.argsort(morton_codes(points, shift))
    sorted_points = points[order]
    offsets = jnp.concatenate([jnp.arange(-window, 0), jnp.arange(1, window + 1)])
    candidates = jnp.clip(jnp.arange(n)[:, None] + offsets[None, :], 0, n - 1)
    distances = jnp.sum(jnp.square(sorted_points[:, None, :] - sorted_points[candidates]), axis=-1)
    original = order[candidates]
    reject = (original == order[:, None]) | ~valid[original]
    distances = jnp.where(reject, jnp.inf, distances)
    best, idx = jax.lax.top_k(-distances, k)
    # Back to the original order
    unsorted = jnp.zeros((n, k), jnp.int32).at[order].set(jnp.take_along_axis(original, idx, axis=1))
    unsorted_dist = jnp.zeros((n, k), distances.dtype).at[order].set(-best)
    return unsorted, unsorted_dist


@partial(jax.jit, static_argnames=("k", "window", "n_shifts"))
def approximate_knn(points, values=None, k=6, window=32, n_shifts=3):
    """Approximate k nearest neighbours of a point cloud (N, 3) via Morton-sorted sliding windows with shifted
    sorting (shifts of 1.5 mean point spacings). Points with values <= 0 are never returned as neighbours.
    Returns indices (N, k)."""
    points = jax.lax.stop_gradient(points)
    valid = (jnp.ones(points.shape[0], bool) if values is None
             else jax.lax.stop_gradient(values) > 0.0)
    spacing_cells = 2 ** MORTON_BITS / points.shape[0] ** (1.0 / 3.0)
    indices, distances = [], []
    for s in range(n_shifts):
        idx, dist = _window_candidates(points, valid, k, window, 1.5 * s * spacing_cells)
        indices.append(idx)
        distances.append(dist)
    if n_shifts == 1:
        return indices[0]
    indices = jnp.concatenate(indices, axis=1)
    distances = jnp.concatenate(distances, axis=1)
    # Drop duplicates found by several shifts, then keep the k closest
    duplicate = (indices[:, :, None] == indices[:, None, :]) & (jnp.arange(indices.shape[1])[None, None, :] < jnp.arange(indices.shape[1])[None, :, None])
    distances = jnp.where(jnp.any(duplicate, axis=-1), jnp.inf, distances)
    _, best = jax.lax.top_k(-distances, k)
    return jnp.take_along_axis(indices, best, axis=1)
