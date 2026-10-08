import os
import numpy as np
from tqdm import tqdm
import sys
from functools import partial
from contextlib import closing

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from einops import rearrange
import dm_pix
import pynndescent
from scipy.spatial import cKDTree
from hax.utils.fsc import hermitian_multiplicity, _shell_index, _shell_fsc
from cuml.neighbors.nearest_neighbors import NearestNeighbors
from xmipp_metadata.image_handler import ImageHandler

from hax.utils import *
from hax.generators import MetaDataGenerator, extract_columns


# --- 1. EFFICIENT GRID MODEL (Splat -> Blur) ---
# (Helper functions remain functional JAX as they are stateless math)

def splat_weights_trilinear(grid_shape, means, weights):
    factor = 0.5 * grid_shape
    grid_coords = (means * factor) + factor

    base_coords = jax.lax.stop_gradient(jnp.floor(grid_coords))
    remainders = grid_coords - base_coords
    base_indices = base_coords.astype(jnp.int32)

    rx, ry, rz = remainders[:, 0], remainders[:, 1], remainders[:, 2]
    cx, cy, cz = 1.0 - rx, 1.0 - ry, 1.0 - rz

    w = [
        weights * cx * cy * cz, weights * rx * cy * cz,
        weights * cx * ry * cz, weights * rx * ry * cz,
        weights * cx * cy * rz, weights * rx * cy * rz,
        weights * cx * ry * rz, weights * rx * ry * rz
    ]

    ix, iy, iz = base_indices[:, 0], base_indices[:, 1], base_indices[:, 2]

    idx_z = jnp.concatenate([iz, iz, iz, iz, iz + 1, iz + 1, iz + 1, iz + 1])
    idx_y = jnp.concatenate([iy, iy, iy + 1, iy + 1, iy, iy, iy + 1, iy + 1])
    idx_x = jnp.concatenate([ix, ix + 1, ix, ix + 1, ix, ix + 1, ix, ix + 1])

    all_weights = jnp.concatenate(w)

    grid = jnp.zeros((grid_shape, grid_shape, grid_shape), dtype=jnp.float32)
    grid = grid.at[(idx_z, idx_y, idx_x)].add(all_weights)

    return grid

def splat_weights(grid_shape, means, weights):
    factor = 0.5 * grid_shape
    grid_coords = (means * factor) + factor
    base_indices = jax.lax.stop_gradient(jnp.round(grid_coords).astype(jnp.int32))
    grid = jnp.zeros((grid_shape, grid_shape, grid_shape), dtype=jnp.float32)
    grid = grid.at[(base_indices[:, 2], base_indices[:, 1], base_indices[:, 0])].add(weights)
    return grid


def splat_weights_bilinear(grid_shape, means, weights, sigma, rotations, shifts, ctf, is_xyz=False):
    factor = 0.5 * grid_shape

    # Rotate means
    if not is_xyz:
        means = jnp.stack([means[:, 2], means[:, 1], means[:, 0]], axis=1)[None, ...]
    else:
        means = means[None, ...]
    means = jnp.matmul(means, rearrange(rotations, "b r c -> b c r"))

    # Apply shifts
    grid_coords = factor * means[..., :-1] - shifts[:, None, :] + factor

    # From XY to YX
    grid_coords = jnp.stack([grid_coords[..., 1], grid_coords[..., 0]], axis=2)

    # Scatter grids (splatting)
    base_indices = jax.lax.stop_gradient(jnp.floor(grid_coords).astype(jnp.int32))
    remainders = grid_coords - base_indices

    # Scatter grids
    offsets = jnp.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
    ])

    def get_corner_weight(offset, remainder):
        terms = jnp.where(offset == 1, remainder, 1.0 - remainder)
        return jnp.prod(terms)

    corner_weights = jax.vmap(
        lambda o: jax.vmap(
            lambda r_b: jax.vmap(
                lambda r: get_corner_weight(o, r),
            )(r_b),
            in_axes=(1,)
        )(remainders)
    )(offsets).T

    values_to_add = corner_weights * weights[None, :, None]

    def scatter_single_grid(grid, base_indices, values_to_add, offsets):
        scatter_indices = (base_indices[:, None, :] + offsets).reshape(-1, 2)
        scatter_values = values_to_add.reshape(-1)
        scatter_indices = jnp.clip(scatter_indices, 0, grid_shape - 1)
        grid = grid.at[tuple(scatter_indices.T)].add(scatter_values)
        return grid

    scatter_grids = jax.vmap(scatter_single_grid, in_axes=(0, 0, 0, None))
    grids = jnp.zeros((rotations.shape[0], grid_shape, grid_shape), dtype=jnp.float32)
    images = scatter_grids(grids, base_indices, values_to_add, offsets)

    # Apply filter
    images = dm_pix.gaussian_blur(images[..., None], sigma, kernel_size=9)[..., 0]
    # images = FastVariableBlur2D((grid_shape, grid_shape))(images[..., None], sigma)[..., 0]

    # Apply CTF
    return ctfFilter(images, ctf, pad_factor=1 if grid_shape > 256 else 2)
    #return images

@partial(jax.jit, static_argnames=("image_size", "apix",))
def global_gaussian_splat(image_size, coords_px, apix, sigma_angstrom, grid, rots, shifts_px, ctfs, point_weights=1.0):
    factor = 0.5 * image_size

    coords_rots = jnp.einsum('bni,bji->bnj', coords_px, rots)
    coords_2d = factor * coords_rots[..., :2] - shifts_px[:, None, :] + factor

    def single_projection_splat(coords_px, sigma_ang, pw_i):
        coords_px_int = jax.lax.stop_gradient(jnp.round(coords_px).astype(jnp.int32))

        grid_x = grid[:, 1]
        grid_y = grid[:, 0]

        shifted_coords_x = coords_px_int[:, 0:1] + grid_x[None, :]
        shifted_coords_y = coords_px_int[:, 1:2] + grid_y[None, :]

        diff_x = shifted_coords_x.astype(jnp.float32) - coords_px[:, 0:1]
        diff_y = shifted_coords_y.astype(jnp.float32) - coords_px[:, 1:2]
        dist_sq = diff_x ** 2 + diff_y ** 2

        var_px = (sigma_ang / apix) ** 2
        if var_px.ndim == 1:
            var_px = var_px[:, None]

        pw = pw_i[:, None] if hasattr(pw_i, 'ndim') and pw_i.ndim == 1 else pw_i

        amplitude = 1.0 / (2.0 * jnp.pi * var_px + 1e-6)
        weights = amplitude * jnp.exp(-dist_sq / (2.0 * var_px + 1e-6)) * pw

        out_of_bounds = (shifted_coords_x < 0) | (shifted_coords_x >= image_size) | \
                        (shifted_coords_y < 0) | (shifted_coords_y >= image_size) | \
                        (weights < 1e-4)

        target_idx = shifted_coords_y * image_size + shifted_coords_x
        safe_weights = jnp.where(out_of_bounds, 0.0, weights)
        safe_idx = jnp.where(out_of_bounds, image_size * image_size, target_idx).astype(jnp.int32)

        img_flat = jnp.zeros((image_size * image_size) + 1, dtype=jnp.float32)
        img_flat = img_flat.at[safe_idx.reshape(-1)].add(safe_weights.reshape(-1))
        return img_flat[:-1].reshape(image_size, image_size)

    sigma_axes = 0 if hasattr(sigma_angstrom, 'ndim') and sigma_angstrom.ndim == 2 else None
    pw_axes = 0 if hasattr(point_weights, 'ndim') and point_weights.ndim == 2 else None
    images = jax.vmap(single_projection_splat, in_axes=(0, sigma_axes, pw_axes))(coords_2d, sigma_angstrom, point_weights)
    return ctfFilter(images, ctfs, pad_factor=1 if image_size > 256 else 2)
    #return images


@partial(jax.jit, static_argnames=("image_size", "apix",))
def anisotropic_gaussian_splat(image_size, coords_px, apix, sigma_3d, grid, rots, shifts_px, ctfs, point_weights=1.0):
    factor = 0.5 * image_size

    coords_rots = jnp.einsum('bni,bji->bnj', coords_px, rots)
    coords_2d = factor * coords_rots[..., :2] - shifts_px[:, None, :] + factor

    sigma_3d_rotated = jnp.einsum('brc,bncd,bkd->bnrk', rots, sigma_3d, rots)
    sigma_2d_px = sigma_3d_rotated[:, :, :2, :2] / (apix ** 2)

    def single_projection_splat(c_px, sig_2d, pw_i):
        c_px_int = jax.lax.stop_gradient(jnp.round(c_px).astype(jnp.int32))
        grid_x, grid_y = grid[:, 1], grid[:, 0]

        shifted_coords_x = c_px_int[:, 0:1] + grid_x[None, :]
        shifted_coords_y = c_px_int[:, 1:2] + grid_y[None, :]
        diff_x = shifted_coords_x.astype(jnp.float32) - c_px[:, 0:1]
        diff_y = shifted_coords_y.astype(jnp.float32) - c_px[:, 1:2]

        a, b, c, d = sig_2d[:, 0, 0:1], sig_2d[:, 0, 1:2], sig_2d[:, 1, 0:1], sig_2d[:, 1, 1:2]

        b_sym = 0.5 * (b + c)
        a_f = a + (1.0 / 12.0)
        d_f = d + (1.0 / 12.0)

        det = (a_f * d_f) - (b_sym ** 2)
        inv_00, inv_01, inv_11 = d_f / det, -b_sym / det, a_f / det

        dist_sq = (diff_x ** 2) * inv_00 + 2.0 * diff_x * diff_y * inv_01 + (diff_y ** 2) * inv_11

        pw = pw_i[:, None] if hasattr(pw_i, 'ndim') and pw_i.ndim == 1 else pw_i

        amplitude = 1.0 / (2.0 * jnp.pi * jnp.sqrt(det))
        weights = amplitude * jnp.exp(-0.5 * dist_sq) * pw

        out_of_bounds = (shifted_coords_x < 0) | (shifted_coords_x >= image_size) | \
                        (shifted_coords_y < 0) | (shifted_coords_y >= image_size) | (weights < 1e-4)

        target_idx = shifted_coords_y * image_size + shifted_coords_x
        safe_weights = jnp.where(out_of_bounds, 0.0, weights)
        safe_idx = (target_idx % (image_size * image_size)).astype(jnp.int32)

        img_flat = jnp.zeros(image_size * image_size, dtype=jnp.float32)
        img_flat = img_flat.at[safe_idx.reshape(-1)].add(safe_weights.reshape(-1))
        return img_flat.reshape(image_size, image_size)

    pw_axes = 0 if hasattr(point_weights, 'ndim') and point_weights.ndim == 2 else None
    images = jax.vmap(single_projection_splat, in_axes=(0, 0, pw_axes))(coords_2d, sigma_2d_px, point_weights)
    return ctfFilter(images, ctfs, pad_factor=1 if image_size > 256 else 2)


def get_outlier_mask(means, k=8, std_dev_mult=1.5):
    """
    Args:
        means: (N, 3) array of positions.
        nn_fn: A callable with signature `dists, idx = nn_fn(points, k)`.
               It should return distances sorted nearest-to-farthest.
        k: Number of real neighbors to check.
        std_dev_mult: Threshold strictness (lower = removes more).
    """
    # Prepare NN search
    if jax.default_backend() == "cpu":
        searcher = pynndescent.NNDescent(means)
        searcher.prepare()
        nn_fn = lambda x: jnp.array(searcher.query(x, k=k + 1)[1])
    elif jax.default_backend() == "gpu":
        searcher = NearestNeighbors(n_neighbors=k + 1)
        searcher.fit(means)
        nn_fn = lambda x: jnp.from_dlpack(searcher.kneighbors(x)[0])
    else:
        raise ValueError(f"Backend {jax.default_backend()} not supported")

    # 1. Query k + 1 neighbors (because the 1st is the point itself)
    # nn_fn is expected to return (distances, indices) or just distances
    neighbor_dists = nn_fn(means)

    # 2. Slice and Average
    # neighbor_dists shape is (N, k+1)
    # We slice [:, 1:] to remove the self-match at index 0
    real_neighbor_dists = neighbor_dists[:, 1:]

    # Average distance to these neighbors
    avg_dist = jnp.mean(real_neighbor_dists, axis=1)

    # 3. Statistical Thresholding
    global_mean = jnp.mean(avg_dist)
    global_std = jnp.std(avg_dist)

    threshold = global_mean + (std_dev_mult * global_std)

    # Returns: True (Keep), False (Prune)
    return avg_dist < threshold


def get_cosine_reg_strength(step, total_steps, start_val, end_val):
    # 1. Normalized progress (0.0 to 1.0)
    progress = jnp.clip(step / total_steps, 0.0, 1.0)

    # 2. Cosine curve (goes from 1.0 down to 0.0)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * progress))

    # 3. Invert it (goes from 0.0 up to 1.0)
    inverted_cosine = 1.0 - cosine_decay
    current_val = start_val + (inverted_cosine * (end_val - start_val))
    return current_val


KIRKLAND_A = jnp.array([
    [0.0878, 0.2860, 0.4446, 0.1415, 0.0401],  # C
    [0.1166, 0.3541, 0.5332, 0.1610, 0.0455],  # N
    [0.1476, 0.4289, 0.6186, 0.1764, 0.0494],  # O
    [0.4137, 1.0963, 1.1578, 0.2974, 0.0825]  # S
], dtype=jnp.float32)

KIRKLAND_B = jnp.array([
    [0.5190, 1.8385, 5.8643, 17.514, 45.456],  # C
    [0.4284, 1.4883, 4.5422, 12.981, 31.956],  # N
    [0.3643, 1.2335, 3.6335, 9.9405, 23.473],  # O
    [0.2070, 0.7765, 2.7095, 8.8471, 23.947]  # S
], dtype=jnp.float32)


@partial(jax.jit, static_argnames=("image_size", "apix"))
def universal_fourier_splat(image_size, coords_px, apix, b_factors, elements_idx, rots, shifts_px, ctfs):
    freq_1d = jnp.fft.fftfreq(image_size, d=apix)
    fy, fx = jnp.meshgrid(freq_1d, freq_1d, indexing='ij')
    k_sq_2d = fx ** 2 + fy ** 2

    factor = 0.5 * image_size
    coords_rots = jnp.einsum('bni,bji->bnj', coords_px, rots)
    coords_2d_px = factor * coords_rots[..., :2] - shifts_px[:, None, :] + factor
    coords_2d_ang = (coords_2d_px - factor) * apix

    def compute_element_form_factor(a_params, b_params):
        return jnp.sum(a_params[:, None, None] * jnp.exp(-b_params[:, None, None] * k_sq_2d[None, :, :] / (4.0 * jnp.pi ** 2)), axis=0)
    f_E_images = jax.vmap(compute_element_form_factor)(KIRKLAND_A, KIRKLAND_B)

    def single_fourier_projection(args):
        c_ang, b_facs = args

        V_x = jnp.exp(- (b_facs[:, None] / 4.0) * (freq_1d[None, :] ** 2) - 2j * jnp.pi * c_ang[:, 0:1] * freq_1d[None, :])
        V_y = jnp.exp(- (b_facs[:, None] / 4.0) * (freq_1d[None, :] ** 2) - 2j * jnp.pi * c_ang[:, 1:2] * freq_1d[None, :])

        def sum_for_element(elem_id):
            mask = (elements_idx == elem_id).astype(jnp.float32)
            V_x_masked = V_x * mask[:, None]
            return jnp.einsum('jy, jx -> yx', V_y, V_x_masked)

        S_E_images = jax.vmap(sum_for_element)(jnp.arange(4))
        F_img = jnp.sum(f_E_images * S_E_images, axis=0)

        return jnp.fft.ifftshift(jnp.real(jnp.fft.ifft2(F_img)))

    images = jax.lax.map(single_fourier_projection, (coords_2d_ang, b_factors))

    pad_factor = 1 if image_size > 256 else 2
    return ctfFilter(images, ctfs, pad_factor=pad_factor)


class FastVariableBlur3D(nnx.Module):
    def __init__(self, shape: tuple[int, int, int]):
        """
        Args:
            shape: (Depth, Height, Width) of the input volume.
        """
        self.d, self.h, self.w = shape

        # 1. Precompute Frequency Grid Coordinates
        # We use broadcasting to create the grid implicitly (saves memory).
        # fz shape: (D, 1, 1)
        fz = jnp.fft.fftfreq(self.d)[:, None, None]
        # fy shape: (1, H, 1)
        fy = jnp.fft.fftfreq(self.h)[None, :, None]
        # fx shape: (1, 1, W/2 + 1) - rfft saves half the space on the last dim
        fx = jnp.fft.rfftfreq(self.w)[None, None, :]

        # 2. Precompute Squared Frequency Radius
        # Broadcasting automatically expands this to (D, H, W/2+1)
        self.f_sq = fz ** 2 + fy ** 2 + fx ** 2

    def __call__(self, x: jax.Array, sigma: float, no_ringing=False) -> jax.Array:
        """
        Args:
            x: Input volume batch (Batch, Depth, Height, Width, Channel) -> NDHWC
            sigma: The blur strength (pixels/voxels).
        """
        if not no_ringing:
            # 3. Generate Gaussian Mask on-the-fly
            # Formula: exp(-2 * pi^2 * sigma^2 * (u^2 + v^2 + w^2))
            mask = jnp.exp(-2 * jnp.pi ** 2 * sigma ** 2 * self.f_sq)
        else:
            # 3. Generate Gaussian Mask on-the-fly
            # DFT of the Gaussian sampled on the voxel grid (separable). The continuous transfer
            # exp(-2 * pi^2 * sigma^2 * f^2) cut at Nyquist gives sinc ripples (-2.6% at sigma 0.45)
            def kernel_1d(n):
                t = jnp.fft.fftfreq(n) * n
                k = jnp.exp(-0.5 * t ** 2 / sigma ** 2)
                return k / jnp.sum(k)

            gz = jnp.fft.fft(kernel_1d(self.d)).real
            gy = jnp.fft.fft(kernel_1d(self.h)).real
            gx = jnp.fft.rfft(kernel_1d(self.w)).real
            mask = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]

        # 4. RFFTN (Real -> Complex, N-dimensional)
        # We perform FFT over axes 1 (D), 2 (H), 3 (W).
        # Batch (0) and Channel (4) are preserved automatically.
        spectrum = jnp.fft.rfftn(x, axes=(1, 2, 3))

        # 5. Apply Mask
        # Expand mask dimensions to match spectrum:
        # Mask is (D, H, W_half) -> (1, D, H, W_half, 1) for broadcasting
        filtered_spectrum = spectrum * mask[None, ..., None]

        # 6. IRFFTN (Complex -> Real, N-dimensional)
        # We must explicitly specify 's' (shape) to ensure the output matches
        # the input dimensions exactly (avoids truncation on odd sizes).
        return jnp.fft.irfftn(
            filtered_spectrum,
            s=(self.d, self.h, self.w),
            axes=(1, 2, 3)
        )

        # (Batch, D, H, W, Channels)
        return jnp.transpose(out, (0, 2, 3, 4, 1))


# --- 2. FLAX NNX MODEL ---

class GaussianSplatModel(nnx.Module):

    @save_config
    def __init__(self, grid_size, sigma=1.0, n_init=None, manual_init=None, *, rngs: nnx.Rngs):
        self.grid_size = grid_size

        # Define Parameters using nnx.Param
        if n_init is not None:
            self.means = nnx.Param(
                jax.random.normal(rngs.params(), (n_init, 3)) * 0.1
            )
            self.weights = nnx.Param(
                # jnp.zeros(n_init)  # Pre-softplus 0.0 -> ~0.7
                jnp.abs(jax.random.normal(rngs.params(), (n_init,)))
            )
            self.sigma_param = nnx.Param(
                jnp.array([sigma])  # Global blur sigma
            )
        elif manual_init is not None:
            self.means = nnx.Param(
                jnp.array(manual_init["means"], dtype=jnp.float32)
            )
            self.weights = nnx.Param(
                jnp.array(manual_init["weights"], dtype=jnp.float32)
            )
            self.sigma_param = nnx.Param(
                jnp.array([sigma])  # Global blur sigma
            )
        else:
            raise ValueError("Provide either n_init or manual_init")

        # Gaussian filter
        self.gaussian_filter_3d = FastVariableBlur3D((grid_size, grid_size, grid_size))

    def update_config(self):
        if self.config["manual_init"] is None:
            self.config["n_init"] = np.array(self.means.get_value()).shape[0]
        else:
            self.config["manual_init"]["means"] = np.array(self.means.get_value())
            self.config["manual_init"]["weights"] = np.array(self.weights.get_value())

    def __call__(self, **kwargs):
        # Forward pass logic
        means = self.means.get_value()
        weights = nnx.relu(self.weights.get_value())
        sigma = jax.nn.softplus(self.sigma_param.get_value())

        if "projection_parameters" in kwargs.keys():
            projection_parameters = kwargs.pop("projection_parameters")

            # Precompute batch aligments
            euler_angles = projection_parameters["euler_angles"]
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

            # Precompute batch shifts
            shifts = projection_parameters["shifts"]
            sr = projection_parameters.get("sr", 1.0)
            pad_factor = 1 if self.grid_size > 256 else 2
            #if "ctfDefocusU" in projection_parameters.keys():
            #    defocusU = projection_parameters["ctfDefocusU"]
            #    defocusV = projection_parameters["ctfDefocusV"]
            #    defocusAngle = projection_parameters["ctfDefocusAngle"]
            #    cs = projection_parameters["ctfSphericalAberration"]
            #    kv = projection_parameters["ctfVoltage"][0]
            #    ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
            #                     projection_parameters["sr"],
            #                     [pad_factor * self.grid_size, int(pad_factor * 0.5 * self.grid_size + 1)],
            #                     rotations.shape[0], True)
            #else:
            #    ctf = jnp.ones(
            #        [rotations.shape[0], pad_factor * self.grid_size, int(pad_factor * 0.5 * self.grid_size + 1)],
            #        dtype=means.dtype)

            # Precompute batch CTFs
            pad_factor = 1 if self.grid_size > 256 else 2
            if "ctfDefocusU" in projection_parameters.keys():
                defocusU = projection_parameters["ctfDefocusU"]
                defocusV = projection_parameters["ctfDefocusV"]
                defocusAngle = projection_parameters["ctfDefocusAngle"]
                cs = projection_parameters["ctfSphericalAberration"]
                kv = projection_parameters["ctfVoltage"][0]
                ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                                 projection_parameters["sr"],
                                 [pad_factor * self.grid_size, int(pad_factor * 0.5 * self.grid_size + 1)],
                                 rotations.shape[0], True)
            else:
                ctf = jnp.ones(
                    [rotations.shape[0], pad_factor * self.grid_size, int(pad_factor * 0.5 * self.grid_size + 1)],
                    dtype=means.dtype)
            if "is_xyz" in kwargs.keys():
                means = kwargs.pop("means")
                sr = projection_parameters["sr"]
                if "grid" in kwargs.keys():
                    weights = kwargs.pop("weights") if "weights" in kwargs.keys() else 1.0
                    final_images = anisotropic_gaussian_splat(self.grid_size, means, sr,
                                                         kwargs.pop("sigma"), kwargs.pop("grid"),
                                                         rotations, shifts, ctf, point_weights=weights)
                else:
                    #in_axes_m = 0 if means.ndim == 3 else None
                    #in_axes_w = 0 if (hasattr(weights, 'ndim') and weights.ndim == 2) else None

                    #final_images = jax.vmap(
                    #    lambda m, w, r, sh, c: splat_weights_bilinear(
                    #        self.grid_size, m, w, sigma, r[None, ...], sh[None, ...], c[None, ...], is_xyz=True
                    #    )[0],
                    #    in_axes=(in_axes_m, in_axes_w, 0, 0, 0)
                    #)(means, weights, rotations, shifts, ctf)

                    b_factors = kwargs.pop("b_factors")
                    elements_idx = kwargs.pop("elements_idx")
                    final_images = universal_fourier_splat(
                        image_size=self.grid_size,
                        coords_px=means,
                        apix=sr,
                        b_factors=b_factors,
                        elements_idx=elements_idx,
                        rots=rotations,
                        shifts_px=shifts,
                        ctfs=ctf
                    )
            else:
                final_images = splat_weights_bilinear(self.grid_size, means, weights, sigma, rotations, shifts, ctf)
            return final_images

        else:
            if kwargs.pop("place_deltas", False):
                return splat_weights(self.grid_size, means, weights)
            else:
                final_vol = splat_weights_trilinear(self.grid_size, means, weights)
                return self.gaussian_filter_3d(final_vol[None, ..., None], sigma,
                                               no_ringing=True if "no_ringing" in kwargs.keys() else False)[0, ..., 0]


class GlobalAdjustment(nnx.Module):

    def __init__(self):
        self.a = nnx.Param(1.0)
        self.b = nnx.Param(0.0)

    def __call__(self, x):
        return jax.nn.softplus(self.a.get_value()) * x + self.b.get_value()

# --- 3. ADAPTIVE LOGIC (NNX Compatible) ---

def adapt_gaussians(model, grads, grad_threshold, prune_threshold, lr=None, optimizer=None):
    """
    Modifies the model structure (adds/removes params) and re-initializes optimizer.
    """
    means = model.means.get_value()
    weights = model.weights.get_value()
    sigma = nnx.relu(model.sigma_param.get_value())

    # 1. SPLIT LOGIC
    grad_means = grads.means.get_value()  # Access gradient of means from NNX grads object
    grad_norms = jnp.linalg.norm(grad_means, axis=-1)

    split_mask = grad_norms > grad_threshold
    n_split = jnp.sum(split_mask)

    # 2. PRUNE LOGIC
    actual_weights = nnx.relu(weights)
    keep_mask = actual_weights > prune_threshold

    # Filter arrays
    means = means[keep_mask]
    weights = weights[keep_mask]
    split_mask = split_mask[keep_mask]
    do_not_split_mask = np.logical_not(split_mask)

    if n_split > 0:
        # print(f"   -> Splitting {n_split} gaussians...")
        parent_means = means[split_mask]
        parent_weights = weights[split_mask]

        # Perturb means
        noise = np.random.normal(0, 0.0001, parent_means.shape)
        new_means = parent_means + noise
        old_means = parent_means - noise
        # Halve weights
        new_weights = old_weights = 0.5 * parent_weights * np.exp((np.linalg.norm(noise, axis=-1) ** 2.) / (2. * sigma ** 2.))

        # Append
        means = jnp.concatenate([means[do_not_split_mask], old_means, new_means])
        weights = jnp.concatenate([weights[do_not_split_mask], old_weights, new_weights])

    # print(f"   -> Count: {model.means.get_value().shape[0]} -> {means.shape[0]}")

    # 3. UPDATE MODEL PARAMETERS
    # In NNX, we can directly assign new arrays to the params
    model.means = nnx.Param(means)
    model.weights = nnx.Param(weights)

    # 4. RESET OPTIMIZER
    # Because the shape of parameters changed, the old optimizer state (momentum, etc.)
    # is invalid. We must re-create the optimizer wrapper for the new model structure.
    # Note: This loses momentum history, which is standard in Gaussian Splatting adaptive steps.
    if optimizer is not None:
        new_optimizer = nnx.Optimizer(model, optimizer.tx, wrt=nnx.Param)
    elif lr is not None:
        new_optimizer = nnx.Optimizer(model, optax.adamw(lr), wrt=nnx.Param)
    else:
        raise ValueError("Either optimizer of lr must be specified")

    return new_optimizer

@partial(jax.jit, static_argnames=("update", "min_sigma"))
def training_step_volume(graphdef, state, target, update=True, min_sigma=0.0):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target):
        recon = model()

        recon_loss = jnp.mean((recon - target) ** 2.)

        l1_loss = 0.001 * jnp.mean(jnp.abs(recon))

        # Floor on the splat width.
        #
        # Nothing else in this loss opposes a narrower Gaussian: with the positions free,
        # shrinking sigma always improves a pointwise volume MSE, because a delta placed on
        # a voxel reproduces that voxel exactly. Left alone the fit runs sigma down until it
        # is a fraction of a voxel, and the map it produces is then a comb of near-delta
        # spikes rather than a continuous density. That is invisible here -- the fit
        # reproduces the reference to FSC > 0.85 either way -- but it is not invisible
        # downstream: HetSIREN inherits this sigma as its render width, and at a fraction of
        # a voxel a sub-voxel displacement of a Gaussian moves essentially its whole
        # amplitude from one voxel to the next, so any deformation the network learns is
        # rendered as voxel-scale speckle instead of as structure.
        #
        # A one-sided hinge, not a pull towards a target: above the floor the penalty and
        # its gradient are exactly zero, so a genuinely broader fit is left alone; it only
        # acts when the fit tries to collapse below the width the point spacing can support.
        sigma = nnx.relu(model.sigma_param.get_value()).mean()
        sigma_loss = jnp.square(jax.nn.relu(min_sigma - sigma))

        # diff_x = recon[1:, :, :] - recon[:-1, :, :]
        # diff_y = recon[:, 1:, :] - recon[:, :-1, :]
        # diff_z = recon[:, :, 1:] - recon[:, :, :-1]
        # l1_grad_loss = 0.00001 * jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        # l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        # Boundary violation loss
        means = model.means.get_value()
        violation = jax.nn.relu(jnp.abs(means) - 0.9)
        boundary_loss = jnp.sum(violation ** 2.)

        return recon_loss + l1_loss + boundary_loss + sigma_loss

    loss_val, grads = nnx.value_and_grad(loss_fn)(model, target)

    # Apply updates directly to the model state managed by optimizer
    if update:
        optimizer.update(model, grads)
        state = nnx.state((model, optimizer))

    return loss_val, grads, state

@partial(jax.jit, static_argnames=("update",))
def training_step_images(graphdef, state, target, projection_parameters, sigma_reg, update=True):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target):
        recon_images = model(projection_parameters=projection_parameters)
        # recon_vol = model()

        recon_loss = jnp.mean((recon_images - target) ** 2.)

        l1_loss = 0.001 * jnp.abs(model.weights.get_value()).mean()

        # l1_loss = jnp.mean(jnp.abs(recon_vol))
        #
        # diff_x = recon_vol[1:, :, :] - recon_vol[:-1, :, :]
        # diff_y = recon_vol[:, 1:, :] - recon_vol[:, :-1, :]
        # diff_z = recon_vol[:, :, 1:] - recon_vol[:, :, :-1]
        # l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        # l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        sigma_loss = jnp.square(1.0 - nnx.relu(model.sigma_param.get_value()).mean())

        # Boundary violation loss
        means = model.means.get_value()
        violation = jax.nn.relu(jnp.abs(means) - 0.9)
        boundary_loss = jnp.sum(violation ** 2.)

        # return recon_loss + l1_loss + 0.01 * (l1_grad_loss + l2_grad_loss) + sigma_reg * sigma_loss
        return recon_loss + sigma_reg * sigma_loss + l1_loss + boundary_loss

    loss_val, grads = nnx.value_and_grad(loss_fn)(model, target)

    # Apply updates directly to the model state managed by optimizer
    if update:
        optimizer.update(model, grads)
        state = nnx.state((model, optimizer))

    return loss_val, grads, state


@partial(jax.jit, static_argnames=("is_xyz", "ctf_type"))
def training_step_local_adjustment(graphdef, state, target, projection_parameters, ctf_type, is_xyz=False):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target):
        if not is_xyz:
            recon_images = model(projection_parameters=projection_parameters)
        else:
            recon_images = model(is_xyz=is_xyz, means=model.means.get_value(), weights=model.weights.get_value(),
                                 projection_parameters=projection_parameters)
        recon_loss = jnp.mean((recon_images - target) ** 2.)
        return recon_loss

    pad_factor = 1 if model.grid_size > 256 else 2
    if ctf_type in ["wiener", "precorrect"]:
        defocusU = projection_parameters.pop("ctfDefocusU")
        defocusV = projection_parameters.pop("ctfDefocusV")
        defocusAngle = projection_parameters.pop("ctfDefocusAngle")
        cs = projection_parameters.pop("ctfSphericalAberration")
        kv = projection_parameters.pop("ctfVoltage")[0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         projection_parameters["sr"],
                         [pad_factor * model.grid_size, int(pad_factor * 0.5 * model.grid_size + 1)],
                         target.shape[0], True)
        target = wiener2DFilter(jnp.squeeze(target), ctf)

    loss_val, grads = nnx.value_and_grad(loss_fn)(model, target)

    params_filter = nnx.All(nnx.Param, nnx.PathContains('weights'))
    grads, _ = grads.split(params_filter, ...)

    # Apply updates directly to the model state managed by optimizer
    optimizer.update(model, grads)
    state = nnx.state((model, optimizer))

    return loss_val, state


@partial(jax.jit, static_argnames=("grid_size", "is_xyz", "ctf_type"))
def training_step_global_adjustment(graphdef, state, target, projection_parameters, means, weights, sigma, grid_size, ctf_type,
                                    is_xyz=False):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target, means, weights, sigma):
        # Forward pass logic
        weights = nnx.relu(model(weights))

        # Precompute batch aligments
        euler_angles = projection_parameters["euler_angles"]
        rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

        # Precompute batch shifts
        shifts = projection_parameters["shifts"]

        # Precompute batch CTFs
        pad_factor = 1 if grid_size > 256 else 2
        if "ctfDefocusU" in projection_parameters.keys():
            defocusU = projection_parameters["ctfDefocusU"]
            defocusV = projection_parameters["ctfDefocusV"]
            defocusAngle = projection_parameters["ctfDefocusAngle"]
            cs = projection_parameters["ctfSphericalAberration"]
            kv = projection_parameters["ctfVoltage"][0]
            ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                             projection_parameters["sr"],
                             [pad_factor * grid_size, int(pad_factor * 0.5 * grid_size + 1)],
                             rotations.shape[0], True)
        else:
            ctf = jnp.ones(
                [rotations.shape[0], pad_factor * grid_size, int(pad_factor * 0.5 * grid_size + 1)],
                dtype=means.dtype)

        if ctf_type in ["wiener", "precorrect"]:
            target = wiener2DFilter(jnp.squeeze(target), ctf)
            ctf = jnp.ones_like(ctf)

        recon_images = splat_weights_bilinear(grid_size, means, weights, sigma, rotations, shifts, ctf, is_xyz=is_xyz)
        recon_loss = jnp.mean((recon_images - target) ** 2.)
        return recon_loss

    loss_val, grads = nnx.value_and_grad(loss_fn)(model, target, means, weights, sigma)

    # Apply updates directly to the model state managed by optimizer
    optimizer.update(model, grads)
    state = nnx.state((model, optimizer))

    return loss_val, state

def fit_volume(target_vol, mask=None, iterations=5000, learning_rate=0.01, densify_interval=500, grad_threshold=1e-5,
               n_init=2500, fixed_gaussians=False, min_sigma=None):
    """Fit a Gaussian mixture to ``target_vol``.

    ``min_sigma`` is a floor (in voxels) on the splat width -- see the hinge in
    ``training_step_volume`` for why the fit needs one. ``None`` (the default) derives it
    from the point cloud as HALF the median nearest-neighbour spacing, which is the width
    at which a row of Gaussians is already flat to ~1% (the ripple falls off as
    exp(-2 pi^2 sigma^2 / s^2)) -- i.e. the narrowest width that still tiles the volume
    without opening gaps between the points. Pass a float to set it explicitly, or 0.0 to
    reproduce the previous (unconstrained) behaviour.
    """
    # Grid size
    grid_size = target_vol.shape[0]

    if mask is not None:
        # Extract mask coords
        mask_sampled = sample_mask_points(mask, n_init)
        inds = np.asarray(np.where(mask_sampled > 0.0)).T
        values = target_vol[inds[:, 0], inds[:, 1], inds[:, 2]]
        factor = 0.5 * target_vol.shape[0]
        coords = (inds - factor) / factor
        manual_init = {"means": coords, "weights": values}

        # Init Model
        rngs = nnx.Rngs(42)
        model = GaussianSplatModel(manual_init=manual_init, grid_size=grid_size, rngs=rngs)
    else:
        # Init Model
        rngs = nnx.Rngs(42)
        model = GaussianSplatModel(n_init=n_init, grid_size=grid_size, rngs=rngs)
        mask = jnp.zeros_like(target_vol)

    # Init Optimizer (nnx.Optimizer automatically tracks model params)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)

    # Splat-width floor, in voxels. Derived from the spacing the points actually have, so it
    # tracks n_init and the size of the molecule rather than being a fixed guess.
    #
    # Measured on the INITIAL cloud, which errs on the side of too small a floor on both
    # init paths: from a mask the points already sit on the structure and (with
    # fixed_gaussians) barely move, while densification only makes them denser; from
    # ``n_init`` they start as a compact blob at the centre and spread outwards. Either way
    # the final spacing is >= this one, so the floor never over-constrains a fit.
    if min_sigma is None:
        pts = np.asarray(model.means.get_value()) * (0.5 * grid_size)   # normalized -> voxels
        nn_d = cKDTree(pts).query(pts, k=2)[0][:, 1]
        min_sigma = float(0.5 * np.median(nn_d))
    min_sigma = float(max(min_sigma, 0.0))

    loss_history = []
    k_history = []

    print(f"\n{bcolors.OKCYAN}###### Starting Adaptive Grid Fit on {grid_size}^3 volume... ######{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}Splat width floor: sigma >= {min_sigma:.3f} voxels{bcolors.ENDC}"
          if min_sigma > 0 else f"{bcolors.WARNING}Splat width floor disabled (min_sigma=0){bcolors.ENDC}")

    graphdef, state = nnx.split((model, optimizer))
    pbar = tqdm(range(iterations), desc="Fitting volume", file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

    for i in pbar:
        # --- TRAIN STEP ---
        loss_val, grads, state = training_step_volume(graphdef, state, target_vol, update=True,
                                                      min_sigma=min_sigma)

        model, _ = nnx.merge(graphdef, state)
        loss_history.append(loss_val)
        k_history.append(model.means.get_value().shape[0])
        s = float(nnx.relu(model.sigma_param.get_value())[0])

        # Progress bar update  (TQDM)
        pbar.set_postfix_str(f"| Loss: {loss_val:.6f} | K: {model.means.get_value().shape[0]:04d} | Sigma: {s:.3f}")

        # --- ADAPTIVE STEP ---
        if i > 0 and i % densify_interval == 0 and not fixed_gaussians:
            # Prune threshold
            signal_std = jnp.std(target_vol, where=(mask == 1))
            signal_mean = jnp.mean(target_vol, where=(mask == 1))
            prune_threshold = signal_mean - signal_std

            # We pass the optimizer because we might need to replace it
            model, optimizer = nnx.merge(graphdef, state)
            optimizer = adapt_gaussians(model, grads, optimizer=optimizer, grad_threshold=grad_threshold, prune_threshold=prune_threshold)
            graphdef, state = nnx.split((model, optimizer))


    model, _ = nnx.merge(graphdef, state)

    # FINAL PRUNING
    means = model.means.get_value()
    weights = model.weights.get_value()

    if not fixed_gaussians:
        # Prune threshold
        signal_std = jnp.std(target_vol, where=(mask == 1))
        signal_mean = jnp.mean(target_vol, where=(mask == 1))
        prune_threshold = signal_mean - signal_std

        # Signal-aware pruning
        actual_weights = nnx.relu(weights)
        keep_mask = actual_weights > prune_threshold
        cc_mask = get_outlier_mask(means[keep_mask])

        # Filter arrays
        means = means[keep_mask][cc_mask]
        weights = weights[keep_mask][cc_mask]

    # Set final means and weights
    model.means = nnx.Param(means)
    model.weights = nnx.Param(weights)
    model.sigma_param = nnx.Param(model.sigma_param.get_value())

    # Update config file
    model.update_config()

    return model, k_history, loss_history

def fit_images(md_path, mmap_output_dir, sr, vol=None, mask=None, batch_size=256, learning_rate=0.01,
               densify_interval=200, save_partial=True, n_init=2500, max_gaussians=50000):
    # Prepare metadata
    generator = MetaDataGenerator(md_path)
    md_columns = extract_columns(generator.md)

    # Grid size
    grid_size = generator.md.getMetaDataImage(0).shape[1]

    # Gaussian splatting class
    if vol is not None:
        if mask is None:
            mask = ImageHandler().generateMask(vol, boxsize=64)

        # Extract mask coords
        mask = sample_mask_points(mask, n_init)
        inds = np.asarray(np.where(mask > 0.0)).T
        values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
        factor = 0.5 * vol.shape[0]
        coords = (inds - factor) / factor
        manual_init = {"means": coords, "weights": values}

        # Init Model
        rngs = nnx.Rngs(42)
        model = GaussianSplatModel(manual_init=manual_init, grid_size=grid_size, rngs=rngs)

    else:
        # Init Model
        rngs = nnx.Rngs(42)
        model = GaussianSplatModel(n_init=n_init, grid_size=grid_size, rngs=rngs)

    # Grain dataset
    generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                         precision=np.float16, group_size=1, shard_size=10000)
    data_loader = generator.return_grain_dataset(batch_size=batch_size, shuffle="global",
                                                 num_epochs=None, num_workers=8, num_threads=1)
    steps_per_epoch = int(len(generator.md) / batch_size)

    # Init Optimizer (nnx.Optimizer automatically tracks model params)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)

    loss_history = []
    k_history = []

    print(f"\n{bcolors.OKCYAN}###### Starting Adaptive Grid Fit on {grid_size}^3 volume... ######{bcolors.ENDC}")

    graphdef, state = nnx.split((model, optimizer))

    pbar = tqdm(range(2 * steps_per_epoch), desc="Fitting volume", file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    with closing(iter(data_loader)) as iter_data_loader:
        for i in pbar:
            (x, _, labels) = next(iter_data_loader)
            # --- TRAIN STEP ---
            projection_parameters = {"euler_angles": md_columns["euler_angles"][labels],
                                     "shifts": md_columns["shifts"][labels]}
            if "ctfDefocusU" in md_columns.keys():
                ctf_parameters = {"ctfDefocusU": md_columns["ctfDefocusU"][labels],
                                  "ctfDefocusV": md_columns["ctfDefocusV"][labels],
                                  "ctfDefocusAngle": md_columns["ctfDefocusAngle"][labels],
                                  "ctfSphericalAberration": md_columns["ctfSphericalAberration"][labels],
                                  "ctfVoltage": md_columns["ctfVoltage"][labels],
                                  "sr": sr}
                projection_parameters = dict(projection_parameters, **ctf_parameters)

            if i % densify_interval == 0:
                sigma_reg_strength = get_cosine_reg_strength(i, 2 * steps_per_epoch, 0.0, 0.01)

            loss_val, grads, state = training_step_images(graphdef, state, x[..., 0], projection_parameters, sigma_reg_strength, update=True)

            model, _ = nnx.merge(graphdef, state)
            loss_history.append(loss_val)
            k_history.append(model.means.get_value().shape[0])
            s = float(jax.nn.softplus(model.sigma_param.get_value())[0])

            # Progress bar update  (TQDM)
            if len(loss_history) > 1000:
                pbar.set_postfix_str(f"| Loss: {sum(loss_history[-1000:]) / 1000:.6f} | K: {model.means.get_value().shape[0]:04d} | Sigma: {s:.3f}")
            else:
                pbar.set_postfix_str(f"| Loss: {sum(loss_history) / len(loss_history):.6f} | K: {model.means.get_value().shape[0]:04d} | Sigma: {s:.3f}")

            # --- ADAPTIVE STEP ---
            if i > 0 and i % densify_interval == 0:
                model, optimizer = nnx.merge(graphdef, state)
                optimizer = adapt_gaussians(model, grads, max_gaussians=max_gaussians, lr=learning_rate)
                graphdef, state = nnx.split((model, optimizer))

            # --- SAVE PARTIAL ---
            if i % (densify_interval // 10 - 1) == 0 and save_partial:
                path = os.path.dirname(md_path)
                partial_volume = splat_volume(graphdef, state)
                ImageHandler().write(partial_volume, os.path.join(path, "volume_gmm.mrc"))

    model, _ = nnx.merge(graphdef, state)

    # FINAL PRUNING
    means = model.means.get_value()
    weights = model.weights.get_value()

    # Prune threshold
    corner_slice = x[:, :10, :10]
    prune_threshold = jnp.mean(corner_slice) + (2.0 * jnp.std(corner_slice))

    # Pruning mask
    actual_weights = nnx.relu(weights)
    keep_mask = actual_weights > prune_threshold
    cc_mask = get_outlier_mask(means[keep_mask])

    # Filter arrays
    means = means[keep_mask][cc_mask]
    weights = weights[keep_mask][cc_mask]

    # Set final means and weights
    model.means = nnx.Param(means)
    model.weights = nnx.Param(weights)

    # Update config file
    model.update_config()

    return model, k_history, loss_history


def adjust_weights_to_images(model, md_path, mmap_output_dir, sr, batch_size=256, learning_rate=0.01, num_epochs=3,
                             is_global=False, is_xyz=False, ctf_type="apply"):
    # Prepare metadata
    generator = MetaDataGenerator(md_path)
    md_columns = extract_columns(generator.md)

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

    # Global vs local
    if is_global:
        model_global_adjustment = GlobalAdjustment()

        # Init Optimizer (nnx.Optimizer automatically tracks model params)
        optimizer = nnx.Optimizer(model_global_adjustment, optax.adamw(learning_rate), wrt=nnx.Param)
        graphdef, state = nnx.split((model_global_adjustment, optimizer))

        # Prepare gaussian params
        means = model.means.get_value()
        weights = model.weights.get_value()
        sigma = jax.nn.softplus(model.sigma_param.get_value())
        grid_size = model.grid_size

    else:
        # Init Optimizer (nnx.Optimizer automatically tracks model params)
        params_filter = nnx.All(nnx.Param, nnx.PathContains('weights'))
        optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=params_filter)
        graphdef, state = nnx.split((model, optimizer))

    loss_history = []

    if is_global:
        print(f"\n{bcolors.OKCYAN}###### Adjusting gaussian weights to images (Global version)... ######{bcolors.ENDC}")
    else:
        print(f"\n{bcolors.OKCYAN}###### Adjusting gaussian weights to images (Local version)... ######{bcolors.ENDC}")

    pbar = tqdm(range(num_epochs * steps_per_epoch), desc="Adjusting weights", file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    with closing(iter(data_loader)) as iter_data_loader:
        for _ in pbar:
            (x, _, labels) = next(iter_data_loader)
            # --- TRAIN STEP ---
            projection_parameters = {"euler_angles": md_columns["euler_angles"][labels],
                                     "shifts": md_columns["shifts"][labels]}
            if ctf_type in ["apply", "wiener", "squared", "precorrect"]:
                ctf_parameters = {"ctfDefocusU": md_columns["ctfDefocusU"][labels],
                                  "ctfDefocusV": md_columns["ctfDefocusV"][labels],
                                  "ctfDefocusAngle": md_columns["ctfDefocusAngle"][labels],
                                  "ctfSphericalAberration": md_columns["ctfSphericalAberration"][labels],
                                  "ctfVoltage": md_columns["ctfVoltage"][labels],
                                  "sr": sr}
                projection_parameters = dict(projection_parameters, **ctf_parameters)

            if is_global:
                loss_val, state = training_step_global_adjustment(graphdef, state, x[..., 0], projection_parameters,
                                                                  means, weights, sigma, grid_size=grid_size, is_xyz=is_xyz, ctf_type=ctf_type)
            else:
                loss_val, state = training_step_local_adjustment(graphdef, state, x[..., 0], projection_parameters, is_xyz=is_xyz, ctf_type=ctf_type)

            loss_history.append(loss_val)

            # Progress bar update  (TQDM)
            if len(loss_history) > 1000:
                pbar.set_postfix_str(
                    f"| Loss: {sum(loss_history[-1000:]) / 1000:.6f}")
            else:
                pbar.set_postfix_str(
                    f"| Loss: {sum(loss_history) / len(loss_history):.6f}")

    if is_global:
        model_global_adjustment, _ = nnx.merge(graphdef, state)
        model.weights = nnx.Param(model_global_adjustment(weights))
    else:
        model, _ = nnx.merge(graphdef, state)

    return model, loss_history


@jax.jit
def splat_volume(graphdef, state):
    model = nnx.merge(graphdef, state)
    if isinstance(model, tuple):
        model = model[0]
    return model()