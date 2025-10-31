import jax.random
from functools import partial
import jax.numpy as jnp
from jax import vmap, jit
from ott.tools.sliced import sliced_wasserstein


# --- 1. Helper: SWD for a Single Pair (using OTT) ---

@partial(jit, static_argnames=("number_of_directions",))
def _sliced_wasserstein_pair_ott(
        coords_i: jnp.ndarray, intensities_i: jnp.ndarray,
        coords_j: jnp.ndarray, intensities_j: jnp.ndarray,
        number_of_directions: int,
        key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Computes the SWD between two weighted point clouds using JAX OTT.

    Args:
        coords_i, coords_j: (M, 3) coordinate arrays.
        intensities_i, intensities_j: (M,) intensity arrays.
        number_of_directions: number of projections.
        key: Random key to sample the directions
    """

    # 1. Normalize intensities (weights) to sum to 1
    # OTT's `sliced_wasserstein_distance` expects probability measures.
    intens_i_norm = intensities_i / jnp.sum(intensities_i)
    intens_j_norm = intensities_j / jnp.sum(intensities_j)

    # 2. Call the OTT function
    # It handles all the projection, 1D Wasserstein, and averaging internally.
    swd = sliced_wasserstein(
        x=coords_i,
        y=coords_j,
        a=intens_i_norm,
        b=intens_j_norm,
        n_proj=number_of_directions,
        rng=key
    )

    return swd[0]


# --- 2. Main Function: SWD Matrix (using OTT) ---

@partial(jit, static_argnames=("number_of_directions",))
def compute_swd_matrix(
        coords: jnp.ndarray,
        intensities: jnp.ndarray,
        number_of_directions: int,
        key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Computes the (B, B) Sliced Wasserstein Distance matrix using JAX OTT.

    Args:
        coords: (B, M, 3) array of coordinates.
        intensities: (B, M) array of corresponding intensities (weights).
        number_of_directions: number of projections.
        key: Random key to sample the directions

    Returns:
        swd_matrix: (B, B) matrix where swd_matrix[i, j] is the SWD between
                    point cloud i and point cloud j.
    """

    # 1. Define the function for a single row of the matrix.
    # This function computes the SWD between one element (i) and all other elements (j).
    def compute_row(coords_i, intensities_i):
        # vmap over the second set of arguments (coords_j, intensities_j)
        # We broadcast coords_i, intensities_i, and directions.
        return vmap(
            _sliced_wasserstein_pair_ott,
            in_axes=(None, None, 0, 0, None, None)  # (c_i, i_i, c_j, i_j, dirs)
        )(coords_i, intensities_i, coords, intensities, number_of_directions, key)

    # 2 vmap the row function over the first set of arguments (coords_i, intensities_i)
    # This "double vmap" creates the (B, B) outer-product-like computation.
    swd_matrix = vmap(compute_row, in_axes=(0, 0))(coords, intensities)

    return swd_matrix