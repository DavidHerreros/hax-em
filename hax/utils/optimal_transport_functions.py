import jax.numpy as jnp
from jax import vmap, jit


# --- 1. Core: 1D Wasserstein Distance ---

@jit
def _wasserstein_1d(
        u_values: jnp.ndarray, v_values: jnp.ndarray,
        u_weights: jnp.ndarray, v_weights: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the 1D Wasserstein-1 distance between two weighted empirical distributions.

    This implementation is a JAX-native version of scipy.stats.wasserstein_distance
    and is JIT-compatible.
    """

    # 1. Get all, sorted values
    # We concatenate and sort. This gives a static shape (M + N,)
    # which JIT can handle. Duplicates are fine, as jnp.diff(all_values)
    # will just result in 0-length intervals.
    all_values_concat = jnp.concatenate([u_values, v_values])
    all_values = jnp.sort(all_values_concat)

    # 2. Sort input values and reorder weights
    u_sorter = jnp.argsort(u_values)
    u_values_sorted = u_values[u_sorter]
    u_weights_sorted = u_weights[u_sorter]

    v_sorter = jnp.argsort(v_values)
    v_values_sorted = v_values[v_sorter]
    v_weights_sorted = v_weights[v_sorter]

    # 3. Compute CDF indices
    # Find the index in the sorted values where each `all_values` point would be inserted
    u_cdf_indices = jnp.searchsorted(u_values_sorted, all_values, side='right')
    v_cdf_indices = jnp.searchsorted(v_values_sorted, all_values, side='right')

    # 4. Evaluate CDFs at all_values
    # We pad with a 0. at the beginning for the 0-th index of cumsum
    u_cdf_padded = jnp.concatenate([jnp.array([0.]), jnp.cumsum(u_weights_sorted)])
    v_cdf_padded = jnp.concatenate([jnp.array([0.]), jnp.cumsum(v_weights_sorted)])

    u_cdf_eval = u_cdf_padded[u_cdf_indices]
    v_cdf_eval = v_cdf_padded[v_cdf_indices]

    # 5. Integrate |CDF_u - CDF_v|
    # The distance is the area between the two CDFs
    cdf_diff = jnp.abs(u_cdf_eval - v_cdf_eval)
    intervals = jnp.diff(all_values)

    # distance = sum( |CDF_u(x) - CDF_v(x)| * dx )
    distance = jnp.sum(cdf_diff[:-1] * intervals)
    return distance


# --- 2. Helper: SWD for a Single Pair ---

@jit
def _sliced_wasserstein_pair(
        coords_i: jnp.ndarray, intensities_i: jnp.ndarray,
        coords_j: jnp.ndarray, intensities_j: jnp.ndarray,
        directions: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the Sliced Wasserstein Distance (SWD) between two point clouds.

    Args:
        coords_i, coords_j: (M, 3) coordinate arrays.
        intensities_i, intensities_j: (M,) intensity arrays.
        directions: (N_proj, 3) array of projection vectors.
    """
    # 1. Project points onto all directions
    # 'mk,pk->pm' -> (M, 3) @ (N_proj, 3).T = (M, N_proj) -> transpose
    proj_i = jnp.einsum('mk,pk->pm', coords_i, directions)  # (N_proj, M)
    proj_j = jnp.einsum('mk,pk->pm', coords_j, directions)  # (N_proj, M)

    # 2. Normalize intensities (weights) to sum to 1
    intens_i_norm = intensities_i / jnp.sum(intensities_i)
    intens_j_norm = intensities_j / jnp.sum(intensities_j)

    # 3. Compute 1D Wasserstein for each projection
    # We vmap _wasserstein_1d over the 'p' (N_proj) dimension.
    # in_axes=(0, 0, None, None) means:
    #   - vmap over axis 0 of proj_i
    #   - vmap over axis 0 of proj_j
    #   - broadcast intens_i_norm (use the same array for all)
    #   - broadcast intens_j_norm (use the same array for all)
    all_1d_distances = vmap(_wasserstein_1d, in_axes=(0, 0, None, None))(
        proj_i, proj_j, intens_i_norm, intens_j_norm
    )

    # 4. Average over all projections
    return jnp.mean(all_1d_distances)


# --- 3. Main Function: SWD Matrix ---

@jit
def compute_swd_matrix(
        coords: jnp.ndarray,
        intensities: jnp.ndarray,
        rotation_matrices: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the (B, B) Sliced Wasserstein Distance matrix for a batch of point clouds.

    Args:
        coords: (B, M, 3) array of coordinates.
        intensities: (B, M) array of corresponding intensities (weights).
        rotation_matrices: (B', 3, 3) array of rotation matrices to define projections.
                           We use all 3 axes of each matrix, so N_proj = B' * 3.

    Returns:
        swd_matrix: (B, B) matrix where swd_matrix[i, j] is the SWD between
                    point cloud i and point cloud j.
    """

    # 1. Define projection directions from rotation matrices.
    # We use the 3 axes (columns) of each rotation matrix as a projection vector.
    # (B', 3, 3) -> transpose axes (0, 2, 1) -> (B', 3, 3) [N_rot, axes, coords]
    # -> reshape to (B' * 3, 3)
    directions = rotation_matrices.transpose(0, 2, 1).reshape(-1, 3)

    # N_proj = B' * 3

    # 2. Define the function for a single row of the matrix.
    # This function computes the SWD between one element (i) and all other elements (j).
    def compute_row(coords_i, intensities_i):
        # vmap over the second set of arguments (coords_j, intensities_j)
        # We broadcast coords_i, intensities_i, and directions.
        return vmap(
            _sliced_wasserstein_pair,
            in_axes=(None, None, 0, 0, None)  # (c_i, i_i, c_j, i_j, dirs)
        )(coords_i, intensities_i, coords, intensities, directions)

    # 3. vmap the row function over the first set of arguments (coords_i, intensities_i)
    # This "double vmap" creates the (B, B) outer-product-like computation.
    swd_matrix = vmap(compute_row, in_axes=(0, 0))(coords, intensities)

    return swd_matrix