import sys
from tqdm import tqdm
from contextlib import closing

import pynndescent
from cuml.neighbors.nearest_neighbors import NearestNeighbors

import jax
import jax.numpy as jnp

from functools import partial

import numpy as np
from sklearn.neighbors import KDTree

from hax.generators import NumpyGenerator
from hax.utils.loggers import bcolors


# def estimate_noise_stddev(images, patch_size):
#   """
#   Estimates the standard deviation of noise in images by looking at corner patches.
#
#   Args:
#     images: A JAX array of shape (B, X, X) representing a batch of images.
#     patch_size: The size of the square patches to extract from each corner.
#
#   Returns:
#     A JAX array representing the estimated standard deviation of the noise.
#     If averaging across all patches and images, it will be a scalar.
#   """
#   B, X, _ = images.shape
#
#   # Define corner patch coordinates
#   # Top-left
#   tl_patch = images[:, :patch_size, :patch_size]
#   # Top-right
#   tr_patch = images[:, :patch_size, X-patch_size:]
#   # Bottom-left
#   bl_patch = images[:, X-patch_size:, :patch_size]
#   # Bottom-right
#   br_patch = images[:, X-patch_size:, X-patch_size:]
#
#   # Concatenate all patches along a new dimension (e.g., axis 1)
#   # New shape will be (B, 4, patch_size, patch_size)
#   all_patches = jnp.stack([tl_patch, tr_patch, bl_patch, br_patch], axis=1)
#
#   # Calculate the standard deviation for each patch
#   # You can then decide how to aggregate these:
#   # 1. Stddev per patch:
#   # patch_stddevs = jnp.std(all_patches, axis=(2, 3)) # Shape: (B, 4)
#   # 2. Stddev by concatenating all noise pixels:
#   #    Flatten the patches and then compute std.
#   #    This treats all selected corner pixels as samples of the noise.
#   noise_pixels = all_patches.reshape(B, -1) # Flatten patches for each image
#   # Alternative: concatenate all pixels from all patches across the batch
#   # noise_pixels_all_batch = all_patches.reshape(-1)
#   # overall_noise_stddev = jnp.std(noise_pixels_all_batch)
#   # return overall_noise_stddev
#
#   # For this example, let's calculate stddev for all noise pixels per image,
#   # then average these stddevs.
#   per_image_noise_stddev = jnp.std(noise_pixels, axis=1) # Shape: (B,)
#
#   # You can then average these if you want a single value for the batch:
#   average_noise_stddev = jnp.mean(per_image_noise_stddev)
#   return average_noise_stddev
#
#   # Or return the stddev for each image
#   # return per_image_noise_stddev


@partial(jax.jit, static_argnames=['radius_fraction'])
def estimate_noise_stddev(images, radius_fraction=0.5):
    """
        Measures the mean and standard deviation of noise in regions outside a
        central circumference for a batch of square images.

        This version is JIT-compatible and avoids dynamic indexing.

        Args:
            images (jax.Array): A JAX array of shape (B, X, X), where B is the
                                batch size and X is the height/width of the images.
            radius_fraction (float, optional): The radius of the central circle,
                                               expressed as a fraction of the image
                                               width (X). Defaults to 0.5.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple containing two JAX arrays:
                - means: An array of shape (B,) with the mean of the noise for each image.
                - stds: An array of shape (B,) with the standard deviation of the
                        noise for each image.
        """
    if images.ndim != 3 or images.shape[1] != images.shape[2]:
        raise ValueError(f"Input images must have shape (B, X, X), but got {images.shape}")

    image_size = images.shape[1]
    radius_pixels = radius_fraction * image_size

    coords = jnp.arange(image_size) - (image_size - 1) / 2.0
    xx, yy = jnp.meshgrid(coords, coords, indexing='ij')
    distance_from_center = jnp.sqrt(xx ** 2 + yy ** 2)
    outside_circle_mask = distance_from_center > radius_pixels

    # --- JIT-COMPATIBLE MASKED STATISTICS CALCULATION ---
    # This function is now safe to use inside jax.vmap and jax.jit
    def _calculate_stats_single_image(image):
        """Calculates stats for one (X, X) image using JIT-safe methods."""
        # Count the number of valid pixels from the mask (as a float).
        num_noise_pixels = jnp.sum(outside_circle_mask)

        # --- Calculate Mean ---
        # 1. Use jnp.where to set all pixels *inside* the circle to 0.0.
        #    The shape of the array remains (X, X).
        masked_image = jnp.where(outside_circle_mask, image, 0.0)
        # 2. Sum the entire array. The sum is only over the noise pixels.
        image_sum = jnp.sum(masked_image)
        # 3. Divide by the count of noise pixels.
        mean_val = image_sum / num_noise_pixels

        # --- Calculate Standard Deviation ---
        # A numerically stable way is std = sqrt(E[X^2] - E[X]^2)
        # 1. Get the sum of the squares of the noise pixels.
        image_sq_sum = jnp.sum(jnp.where(outside_circle_mask, jnp.power(image, 2), 0.0))
        # 2. Calculate the mean of the squares.
        mean_of_squares = image_sq_sum / num_noise_pixels
        # 3. Calculate variance and then standard deviation.
        variance = mean_of_squares - jnp.power(mean_val, 2)
        # Use jnp.maximum to prevent sqrt of a negative number due to float precision.
        std_val = jnp.sqrt(jnp.maximum(0.0, variance))

        return mean_val, std_val

    # Vectorize the JIT-safe function across the batch dimension.
    means, stds = jax.vmap(_calculate_stats_single_image)(images)

    return means, stds

def filter_latent_space(space, thr=1.0, k=10, return_ids=False, batch_size=64):
    # Prepare data loader
    data_loader = NumpyGenerator(space).return_grain_dataset(batch_size=batch_size, preShuffle=False, shuffle=False,
                                                             num_epochs=1, num_workers=0)
    steps_per_epoch = int(np.ceil(space.shape[0] / batch_size))

    # Compute distance distributions
    distribution = []

    # Prepare NN search
    if jax.default_backend() == "cpu":
        searcher = pynndescent.NNDescent(space)
        searcher.prepare()
        nn_fn = lambda x: searcher.query(x, k=k)[1]
    elif jax.default_backend() == "gpu":
        searcher = NearestNeighbors(n_neighbors=k)
        searcher.fit(space)
        nn_fn = lambda x: searcher.kneighbors(x)[0]
    else:
        raise ValueError(f"Backend {jax.default_backend()} not supported")

    print(f"{bcolors.OKCYAN}\n###### Filtering latents... ######")
    pbar = tqdm(range(steps_per_epoch), desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    with closing(iter(data_loader)) as iter_data_loader:
        for _ in pbar:
            (z, _) = next(iter_data_loader)
            distances = nn_fn(z)
            distribution.append(jnp.mean(distances[:, 1:], axis=1))
        distribution = jnp.hstack(distribution)

    # Compute Z-Scores
    z_scores = jnp.abs((distribution - jnp.mean(distribution)) / jnp.std(distribution))
    ids = jnp.arange(space.shape[0])[z_scores < thr]

    if return_ids:
        return ids
    else:
        return space[ids]

@partial(jax.jit, static_argnames=['k', 'block_size'])
def batched_knn(dataset: jnp.ndarray,
                queries: jnp.ndarray,
                k: int = 1000,
                block_size: int = 100_000):
    """
    dataset: [N, D]
    queries: [B, D]  with B <= 64
    returns: inds [B, k], dists [B, k]
    """
    N, D = dataset.shape
    B    = queries.shape[0]

    # Update block size
    block_size = len(dataset) if len(dataset) < block_size else block_size

    # Precompute query squared‐norms
    q_norm = jnp.sum(queries**2, axis=1)   # [B]

    # Buffers for “running” top‐k (we store negative d2 so we can use top_k for min)
    init_d = jnp.full((B, k), -jnp.inf)
    init_i = jnp.zeros((B, k), dtype=jnp.int32)

    def body(idx, state):
        best_d, best_i = state

        # compute dynamic start index
        start = idx * block_size

        # grab a block of size [block_size, D] via dynamic_slice
        block = jax.lax.dynamic_slice(dataset,
                                      start_indices=(start, 0),
                                      slice_sizes=(block_size, D))

        # squared norms of block points
        x_norm = jnp.sum(block**2, axis=1)              # [block_size]

        # distances:  (‖x‖² + ‖q‖² − 2 x·q)ᵀ → [B, block_size]
        dots = block @ queries.T                        # [block_size, B]
        d2   = (x_norm[:, None] + q_norm[None, :] - 2*dots).T

        # mask out-of-range rows on the last block
        valid = jnp.arange(block_size) < jnp.minimum(block_size, N - start)
        d2 = jnp.where(valid[None, :], d2, jnp.inf)

        # local top-k (negate d2 to find smallest)
        neg_d2, local_idx = jax.lax.top_k(-d2, k)           # [B, k] each
        abs_idx = local_idx + start                     # absolute indices in dataset

        # merge with global best
        all_d = jnp.concatenate([best_d, neg_d2], axis=1)  # [B, 2k]
        all_i = jnp.concatenate([best_i, abs_idx], axis=1) # [B, 2k]
        new_d, pick = jax.lax.top_k(all_d, k)                 # pick top k of the 2k
        new_i = jnp.take_along_axis(all_i, pick, axis=1)

        return (new_d, new_i)

    num_blocks = (N + block_size - 1) // block_size
    final_d, final_i = jax.lax.fori_loop(0, num_blocks, body, (init_d, init_i))

    # final_d stores negated squared-distances; flip sign and sqrt
    return final_i, jnp.sqrt(-final_d)

@jax.jit
def rigid_registration(A: jnp.ndarray, B: jnp.ndarray):
    """
    Compute the rigid-body transform (R, t) that best aligns B to A:
        A ≈ B @ R.T + t

    Args:
      A: jnp.ndarray, shape (..., N, D) — target point cloud
      B: jnp.ndarray, shape (..., N, D) — source point cloud (to be aligned)

    Returns:
      R: jnp.ndarray, shape (..., D, D) — optimal rotation matrix
      t: jnp.ndarray, shape (..., D)   — optimal translation vector
      B_aligned: jnp.ndarray, shape (..., N, D) — B rotated and translated
    """
    # 1) Compute centroids
    mu_A = jnp.mean(A, axis=-2)
    mu_B = jnp.mean(B, axis=-2)

    # 2) Center the clouds
    A_centered = A - mu_A[..., None, :]
    B_centered = B - mu_B[..., None, :]

    # 3) Covariance matrix
    H = jnp.swapaxes(B_centered, -2, -1) @ A_centered

    # 4) SVD of covariance
    U, S, Vt = jnp.linalg.svd(H, full_matrices=False)
    V = jnp.swapaxes(Vt, -2, -1)
    Ut = jnp.swapaxes(U, -2, -1)

    # 5) Ensure a proper rotation (no reflection)
    det = jnp.linalg.det(V @ Ut)
    sign = jnp.where(det < 0, -1.0, 1.0)
    V = V.at[..., :, -1].set(V[..., :, -1] * sign[..., None])

    # 6) Compute rotation
    R = V @ Ut
    # R = jnp.swapaxes(R, -2, -1)

    # 7) Compute translation
    t = mu_A - jnp.einsum('...ij,...j->...i', R, mu_B)

    # 8) Apply transform
    B_aligned = jnp.einsum('...ij,...nj->...ni', R, B) + t[..., None, :]

    return R, t, B_aligned

def estimate_envelopes(
    projections: jnp.ndarray,   # [B, M, M, 1], already CTF-corrupted but *unpadded*
    ctf_maps:    jnp.ndarray,   # [B, P, Q, 1], real-FFT CTF (|CTF(k)|) with padding
    pixel_size:  float,         # Å per pixel in real space
    pad_shape:   tuple[int,int],# (pad_h, pad_w) for zero-padding before rfft2
    k_min:       float,         # low-freq cutoff (cycles/Å) for fitting
    k_max:       float          # high-freq cutoff (cycles/Å)
) -> jnp.ndarray:
    """
    Returns: envelopes [B, P, Q, 1] where each envelope[b] = exp(-B_b * k^2 /4).
    """

    B, M, _, _ = projections.shape
    pad_h, pad_w = pad_shape
    # drop trailing channel dim
    imgs = projections[...,0]    # [B, M, M]
    ctf  = ctf_maps[...,0]       # [B, P, Q]

    # precompute frequency grid for the *padded* FFT
    fy = jnp.fft.fftfreq(pad_h, d=pixel_size)[:, None]
    fx = jnp.fft.rfftfreq(pad_w, d=pixel_size)[None, :]
    k2 = fx**2 + fy**2           # [P, Q]
    k  = jnp.sqrt(k2)

    # flatten and mask for linear fit
    k2_flat    = k2.ravel()
    mask       = (k > k_min) & (k < k_max)
    mask_flat  = mask.ravel()

    def per_image(img, ctf_map):
        # 1) rFFT² → PSD
        ft      = jnp.fft.rfft2(img, s=(pad_h, pad_w), norm="ortho")
        psd2d   = jnp.abs(ft)**2                  # [P, Q]

        # 2) remove CTF^2 to isolate envelope
        psd_env = psd2d / (ctf_map**2 + 1e-12)

        # 3) linear regression: log(psd_env) ≃ -B/4 * k^2 + const
        y = jnp.log(psd_env.ravel()[mask_flat])
        x = k2_flat[mask_flat]
        mx, my = jnp.mean(x), jnp.mean(y)
        cov_xy = jnp.mean((x - mx)*(y - my))
        var_x  = jnp.mean((x - mx)**2)
        slope  = cov_xy / (var_x + 1e-12)
        B_fact = -4.0 * slope

        # 4) rebuild 2D envelope and re-add channel dim
        env2d = jnp.exp(-B_fact * k2 / 4.0)        # [P, Q]
        return env2d[..., None]                   # [P, Q, 1]

    # vectorize over the batch
    vmapped = jax.jit(jax.vmap(per_image, in_axes=(0,0)))
    return vmapped(imgs, ctf)                     # [B, P, Q, 1]

@jax.jit
def sparse_finite_3D_differences(values, inds, vol_dim):
    """
    Computes spatial finite differences from sparse data in a memory-efficient way.

    This version uses sorting and searching to avoid creating a large (N, N) matrix,
    making it suitable for a large number of sparse points.

    Args:
      values: A JAX array of shape (B, N) with values at each coordinate.
      inds: A JAX array of shape (B, N, 3) with integer coordinates (x, y, z).
      vol_dim: The side dimension of the conceptual dense volume (X).

    Returns:
      A tuple of (diff_x, diff_y, diff_z), each with shape (B, N).
    """

    if inds.ndim == 2:
        in_axes = (0, None)
    elif inds.ndim == 3:
        in_axes = (0, 0)
    else:
        raise ValueError("Input inds expected to have shape (N,3) or (B,N,3)")

    # A vmapped function to process each item in the batch.
    @partial(jax.vmap, in_axes=in_axes)
    def _single_batch_diff(vals, coords):
        # vals shape: (N,), coords shape: (N, 3)
        num_points = coords.shape[0]

        # 1. Linearize 3D coordinates into 1D for efficient sorting and searching.
        # The formula ensures a unique ID for each coordinate within the volume bounds.
        linear_coords = coords[:, 0] + coords[:, 1] * vol_dim + coords[:, 2] * vol_dim * vol_dim

        # 2. Get the permutation that sorts the coordinates.
        perm = jnp.argsort(linear_coords)
        sorted_linear_coords = linear_coords[perm]
        sorted_vals = vals[perm]

        # Helper function to get the difference along one axis
        def _get_diff(axis_offset):
            # Calculate the linear offset for the given axis (e.g., [1, 0, 0] -> 1)
            linear_offset = axis_offset[0] + axis_offset[1] * vol_dim + axis_offset[2] * vol_dim * vol_dim

            # For each original point, calculate its neighbor's target linear coordinate.
            target_neighbor_coords = linear_coords + linear_offset

            # 3. Search for the neighbors in the sorted list.
            # `searchsorted` finds the indices where the neighbors *would be inserted* to
            # maintain order. This is the most efficient step, running in O(N log N).
            found_indices = jnp.searchsorted(sorted_linear_coords, target_neighbor_coords)

            # 4. Verify that the found indices correspond to actual neighbors.
            # Clip indices to prevent out-of-bounds errors.
            found_indices_clipped = jnp.clip(found_indices, 0, num_points - 1)

            # Get the linear coordinates from the sorted list at the found positions.
            retrieved_coords = sorted_linear_coords[found_indices_clipped]

            # An actual match occurs only if the retrieved coordinate is the one we were looking for.
            is_match = (retrieved_coords == target_neighbor_coords)

            # Get the corresponding values from the sorted values array.
            # If there was no match, the value is zeroed out.
            neighbor_values = sorted_vals[found_indices_clipped] * is_match

            # The `neighbor_values` array is now correctly ordered to align with the
            # original `vals` array, so we can subtract directly.
            return neighbor_values - vals

        # Calculate difference for each axis
        diff_x = _get_diff(jnp.array([1, 0, 0]))
        diff_y = _get_diff(jnp.array([0, 1, 0]))
        diff_z = _get_diff(jnp.array([0, 0, 1]))

        return diff_x, diff_y, diff_z

    return _single_batch_diff(values, inds)


def build_graph_from_coordinates(
    centers,
    k_knn: int = 6,
    k_spacing: int = 2,
    radius_factor: float = 1.5,
    undirected: bool = False,
):
    """
    Memory-efficient model-free graph construction using sklearn KDTree.

    Parameters
    ----------
    centers : jnp.ndarray or np.ndarray, shape (N, 3)
        Pseudo-atom centers.
    k_spacing : int, default=2
        Number of nearest neighbours to estimate local spacing (2 in DynaMight).
        We will query k+1 neighbors because the first one is the point itself.
    k_knn : int, default=6
        Number of nearest neighbours to estimate connection graph (6 in DynaMight).
        We will query k+1 neighbors because the first one is the point itself.
    radius_factor : float, default=1.5
        Multiplier for the mean k-NN distance to set the radius.
    leaf_size : int, default=40
        KDTree leaf_size parameter (controls speed/memory trade-off).
    metric : str, default="euclidean"
        Distance metric for KDTree.
    undirected : bool, default=False
        If True, keep only one edge per undirected pair (i < j).
        If False, keep both directions (i->j and j->i).

    Returns
    -------
    edge_index : jnp.ndarray, shape (2, E), dtype=int32
        Edge list [i, j] for the graph.
    cutoff : jnp.ndarray, scalar
        Radius used to connect nodes.
    """
    # Convert to numpy for sklearn
    if isinstance(centers, jnp.ndarray):
        centers_np = np.asarray(centers)
    else:
        centers_np = np.asarray(centers, dtype=np.float32)

    # Build KDTree on CPU
    tree = KDTree(centers_np)

    # Estimate mean k-NN distance (excluding self)
    dists, idxs = tree.query(centers_np, k=k_spacing + 1)
    knn_dists = dists[:, 1 : k_spacing + 1]  # (N, k)
    mean_k = knn_dists.mean()        # scalar float
    cutoff = radius_factor * mean_k

    # Radius neighbors graph
    ind_array = tree.query_radius(centers_np, r=cutoff, return_distance=False)

    edges_i = []
    edges_j = []

    if undirected:
        # Only keep edges with j > i to avoid duplicates
        for i, neigh in enumerate(ind_array):
            for j in neigh:
                if j == i:
                    continue
                if j > i:
                    edges_i.append(i)
                    edges_j.append(j)
    else:
        # Keep directed edges (i -> j), excluding self
        for i, neigh in enumerate(ind_array):
            for j in neigh:
                if j == i:
                    continue
                edges_i.append(i)
                edges_j.append(j)

    edges_i = np.asarray(edges_i, dtype=np.int32)
    edges_j = np.asarray(edges_j, dtype=np.int32)

    # Compute edge weights
    if edges_i.size == 0:
        edge_index_np = np.zeros((2, 0), dtype=np.int32)
        edge_weights_np = np.zeros((0,), dtype=np.float32)
    else:
        edge_index_np = np.stack([edges_i, edges_j], axis=0)

        # Get distances in consensus model
        diffs = centers_np[edges_i] - centers_np[edges_j]
        dists = np.linalg.norm(diffs, axis=-1)

        # Gaussian weighting
        sigma = mean_k
        edge_weights_np = np.exp(-(dists ** 2.) / (2. * sigma ** 2.))

        # Compute Degree Normalization (Optional but stabilizes training)
        # Ensures total weight per node is roughly consistent
        # unique, counts = np.unique(edges_i, return_counts=True)
        # degree_map = dict(zip(unique, counts))
        # node_degrees = np.array([degree_map.get(idx, 1) for idx in edges_i])
        # edge_weights_np = edge_weights_np / node_degrees

    # Build KNN graph (for Outlier Loss)
    dists_knn, idxs_knn = tree.query(centers_np, k=k_knn + 1)
    edges_i_knn = np.repeat(np.arange(centers_np.shape[0]), k_knn)
    edges_j_knn = idxs_knn[:, 1:].flatten()
    edge_index_knn_np = np.stack([edges_i_knn, edges_j_knn], axis=0)

    # Convert back to JAX
    edge_index = jnp.asarray(edge_index_np)
    edge_weights = jnp.asarray(edge_weights_np, dtype=jnp.float32)
    consensus_distances = jnp.asarray(dists, dtype=jnp.float32)
    edge_index_knn = jnp.asarray(edge_index_knn_np)
    cutoff_jnp = jnp.asarray(cutoff, dtype=centers.dtype if isinstance(centers, jnp.ndarray) else jnp.float32)

    return edge_index, edge_weights, consensus_distances, consensus_distances.mean(), edge_index_knn, cutoff_jnp


def sample_mask_points(mask, N):
    """
    Selects N random points from a binary mask without replacement.

    Args:
        mask (np.ndarray): The input boolean or binary mask (M, M, M).
        N (int): The number of points to select.

    Returns:
        np.ndarray: A new mask of the same shape with only the N selected points.
    """
    # 1. Find the flat indices of all non-zero elements (points equal to 1)
    flat_indices = np.flatnonzero(mask)

    # Safety check: ensure we have enough points to sample
    if len(flat_indices) < N:
        raise ValueError(f"Mask only has {len(flat_indices)} points, cannot sample {N}.")

    # 2. Randomly select N indices without replacement
    # explicit generator is used for better reproducibility control,
    # but np.random.choice works fine too.
    rng = np.random.default_rng()
    selected_indices = rng.choice(flat_indices, size=N, replace=False)

    # 3. Create an empty flat volume
    out_flat = np.zeros(mask.size, dtype=mask.dtype)

    # 4. Set the selected points to 1
    out_flat[selected_indices] = 1

    # 5. Reshape back to the original (M, M, M) dimensions
    return out_flat.reshape(mask.shape)


def safe_norm(x, axis=-1, eps=1e-8):
    return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + eps)
