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
from cuml.neighbors.nearest_neighbors import NearestNeighbors
from xmipp_metadata.image_handler import ImageHandler

from hax.utils import *
from hax.generators import MetaDataGenerator, extract_columns



# --- 1. EFFICIENT GRID MODEL (Splat -> Blur) ---
# (Helper functions remain functional JAX as they are stateless math)

def splat_weights_trilinear(grid_shape, means, weights):
    factor = 0.5 * grid_shape
    grid_coords = (means * factor) + factor
    base_indices = jax.lax.stop_gradient(jnp.floor(grid_coords).astype(jnp.int32))
    remainders = grid_coords - base_indices

    offsets = jnp.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ])

    def get_corner_weight(offset, remainder):
        terms = jnp.where(offset == 1, remainder, 1.0 - remainder)
        return jnp.prod(terms)

    corner_weights = jax.vmap(
        lambda o: jax.vmap(lambda r: get_corner_weight(o, r))(remainders)
    )(offsets).T

    values_to_add = corner_weights * weights[:, None]

    scatter_indices = (base_indices[:, None, :] + offsets).reshape(-1, 3)
    scatter_values = values_to_add.reshape(-1)
    # scatter_indices = jnp.clip(scatter_indices, 0, grid_shape - 1)

    grid = jnp.zeros((grid_shape, grid_shape, grid_shape), dtype=jnp.float32)
    grid = grid.at[tuple(scatter_indices.T)].add(scatter_values)
    return grid

def splat_weights(grid_shape, means, weights):
    factor = 0.5 * grid_shape
    grid_coords = (means * factor) + factor
    base_indices = jax.lax.stop_gradient(jnp.round(grid_coords).astype(jnp.int32))
    grid = jnp.zeros((grid_shape, grid_shape, grid_shape), dtype=jnp.float32)
    grid = grid.at[tuple(base_indices.T)].add(weights)
    return grid

def splat_weights_bilinear(grid_shape, means, weights, sigma, rotations, shifts, ctf):
    factor = 0.5 * grid_shape

    # Rotate means
    means = jnp.stack([means[:, 2], means[:, 1], means[:, 0]], axis=1)[None, ...]
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
    pad_factor = 1 if grid_shape > 256 else 2
    images = ctfFilter(images, ctf, pad_factor=pad_factor)

    return images


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

    # 4. Scale to target range
    current_val = start_val + (inverted_cosine * (end_val - start_val))
    return current_val


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

    def __call__(self, x: jax.Array, sigma: float) -> jax.Array:
        """
        Args:
            x: Input volume batch (Batch, Depth, Height, Width, Channel) -> NDHWC
            sigma: The blur strength (pixels/voxels).
        """
        # 3. Generate Gaussian Mask on-the-fly
        # Formula: exp(-2 * pi^2 * sigma^2 * (u^2 + v^2 + w^2))
        mask = jnp.exp(-2 * jnp.pi ** 2 * sigma ** 2 * self.f_sq)

        # 4. RFFTN (Real -> Complex, N-dimensional)
        # We perform FFT over axes 1 (D), 2 (H), 3 (W).
        # Batch (0) and Channel (4) are preserved automatically.
        spectrum = jnp.fft.rfftn(x, axes=(1, 2, 3))

        # 5. Apply Mask
        # Expand mask dimensions to match spectrum:
        # Mask is (D, H, W_half) -> (1, D, H, W_half, 1) for broadcasting
        mask = mask[None, ..., None]
        filtered_spectrum = spectrum * mask

        # 6. IRFFTN (Complex -> Real, N-dimensional)
        # We must explicitly specify 's' (shape) to ensure the output matches
        # the input dimensions exactly (avoids truncation on odd sizes).
        return jnp.fft.irfftn(
            filtered_spectrum,
            s=(self.d, self.h, self.w),
            axes=(1, 2, 3)
        )


# --- 2. FLAX NNX MODEL ---

class GaussianSplatModel(nnx.Module):

    @save_config
    def __init__(self, grid_size, n_init=None, manual_init=None, *, rngs: nnx.Rngs):
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
                jnp.array([1.0])  # Global blur sigma
            )
        elif manual_init is not None:
            self.means = nnx.Param(
                jnp.array(manual_init["means"], dtype=jnp.float32)
            )
            self.weights = nnx.Param(
                jnp.array(manual_init["weights"], dtype=jnp.float32)
            )
            self.sigma_param = nnx.Param(
                jnp.array([1.0])  # Global blur sigma
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
        sigma = nnx.relu(self.sigma_param.get_value())

        if "projection_parameters" in kwargs.keys():
            projection_parameters = kwargs.pop("projection_parameters")

            # Precompute batch aligments
            euler_angles = projection_parameters["euler_angles"]
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

            # Precompute batch shifts
            shifts = projection_parameters["shifts"]

            # Precompute batch CTFs
            pad_factor = 1 if self.grid_size > 256 else 2
            if "ctfDefocusU" in projection_parameters.keys():
                defocusU = projection_parameters["ctfDefocusU"]
                defocusV = projection_parameters["ctfDefocusV"]
                defocusAngle = projection_parameters["ctfDefocusAngle"]
                cs = projection_parameters["ctfSphericalAberration"]
                kv = projection_parameters["ctfVoltage"][0]
                ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                                 projection_parameters["sr"], [pad_factor * self.grid_size, int(pad_factor * 0.5 * self.grid_size + 1)],
                                 rotations.shape[0], True)
            else:
                ctf = jnp.ones([rotations.shape[0], pad_factor * self.grid_size, int(pad_factor * 0.5 * self.grid_size + 1)],
                               dtype=means.dtype)

            final_images = splat_weights_bilinear(self.grid_size, means, weights, sigma, rotations, shifts, ctf)
            return final_images

        else:
            if not kwargs.pop("place_deltas", False):
                final_vol = splat_weights_trilinear(self.grid_size, means, weights)
                final_vol = self.gaussian_filter_3d(final_vol[None, ..., None], sigma)[0, ..., 0]
                # final_vol = fast_gaussian_filter_3d(final_vol[..., None], sigma, radius=9)[..., 0]
            else:
                final_vol = splat_weights(self.grid_size, means, weights)
            return final_vol
        
        
class GlobalAdjustment(nnx.Module):

    def __init__(self):
        self.a = nnx.Param(1.0)
        self.b = nnx.Param(0.0)
        
    def __call__(self, x):
        return nnx.relu(self.a.get_value()) * x + self.b.get_value()
    

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


# Define Loss Function for NNX
@partial(jax.jit, static_argnames=("update",))
def training_step_volume(graphdef, state, target, update=True):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target):
        recon = model()

        recon_loss = jnp.mean((recon - target) ** 2.)

        l1_loss = 0.001 * jnp.mean(jnp.abs(recon))

        # diff_x = recon[1:, :, :] - recon[:-1, :, :]
        # diff_y = recon[:, 1:, :] - recon[:, :-1, :]
        # diff_z = recon[:, :, 1:] - recon[:, :, :-1]
        # l1_grad_loss = 0.00001 * jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        # l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        # Boundary violation loss
        means = model.means.get_value()
        violation = jax.nn.relu(jnp.abs(means) - 0.9)
        boundary_loss = jnp.sum(violation ** 2.)

        return recon_loss + l1_loss + boundary_loss

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


@jax.jit
def training_step_local_adjustment(graphdef, state, target, projection_parameters):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target):
        recon_images = model(projection_parameters=projection_parameters)

        recon_loss = jnp.mean((recon_images - target) ** 2.)
        return recon_loss

    loss_val, grads = nnx.value_and_grad(loss_fn)(model, target)

    params_filter = nnx.All(nnx.Param, nnx.PathContains('weights'))
    grads, _ = grads.split(params_filter, ...)

    # Apply updates directly to the model state managed by optimizer
    optimizer.update(model, grads)
    state = nnx.state((model, optimizer))

    return loss_val, state


@partial(jax.jit, static_argnames=("grid_size",))
def training_step_global_adjustment(graphdef, state, target, projection_parameters, means, weights, sigma, grid_size):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target, means, weights, sigma):
        # Forward pass logic
        weights = nnx.relu(model(weights))
        sigma = nnx.relu(sigma)

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

        recon_images = splat_weights_bilinear(grid_size, means, weights, sigma, rotations, shifts, ctf)

        recon_loss = jnp.mean((recon_images - target) ** 2.)
        return recon_loss

    loss_val, grads = nnx.value_and_grad(loss_fn)(model, target, means, weights, sigma)

    # Apply updates directly to the model state managed by optimizer
    optimizer.update(model, grads)
    state = nnx.state((model, optimizer))

    return loss_val, state


def fit_volume(target_vol, mask=None, iterations=5000, learning_rate=0.01, densify_interval=500, grad_threshold=1e-5,
               n_init=2500, fixed_gaussians=False):
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

    loss_history = []
    k_history = []

    print(f"\n{bcolors.OKCYAN}###### Starting Adaptive Grid Fit on {grid_size}^3 volume... ######{bcolors.ENDC}")


    graphdef, state = nnx.split((model, optimizer))
    pbar = tqdm(range(iterations), desc="Fitting volume", file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

    for i in pbar:
        # --- TRAIN STEP ---
        loss_val, grads, state = training_step_volume(graphdef, state, target_vol, update=True)

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

    # Update config file
    model.update_config()

    return model, k_history, loss_history


def fit_images(md_path, mmap_output_dir, sr, vol=None, mask=None, batch_size=256, learning_rate=0.01,
               densify_interval=200, grad_threshold=1e-5, save_partial=True, n_init=2500):
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
            (x, labels) = next(iter_data_loader)
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
            s = float(nnx.relu(model.sigma_param.get_value())[0])

            # Progress bar update  (TQDM)
            if len(loss_history) > 1000:
                pbar.set_postfix_str(f"| Loss: {sum(loss_history[-1000:]) / 1000:.6f} | K: {model.means.get_value().shape[0]:04d} | Sigma: {s:.3f}")
            else:
                pbar.set_postfix_str(f"| Loss: {sum(loss_history) / len(loss_history):.6f} | K: {model.means.get_value().shape[0]:04d} | Sigma: {s:.3f}")

            # --- ADAPTIVE STEP ---
            if i > 0 and i % densify_interval == 0:
                # We pass the optimizer because we might need to replace it
                # lr = get_cosine_reg_strength(i, len_dataset // 3, learning_rate, 0.001)

                # Prune threshold
                corner_slice = x[:, :10, :10]
                prune_threshold = jnp.mean(corner_slice) + (2.0 * jnp.std(corner_slice))

                model, optimizer = nnx.merge(graphdef, state)
                optimizer = adapt_gaussians(model, grads, lr=learning_rate, grad_threshold=grad_threshold, prune_threshold=0.1 * prune_threshold)
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


def adjust_weights_to_images(model, md_path, mmap_output_dir, sr, batch_size=256, learning_rate=0.01, num_epochs=3, is_global=False):
    # Prepare metadata
    generator = MetaDataGenerator(md_path)
    md_columns = extract_columns(generator.md)

    # Grain dataset
    generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                         precision=np.float16, group_size=1, shard_size=10000)
    data_loader = generator.return_grain_dataset(batch_size=batch_size, shuffle="global",
                                                 num_epochs=None, num_workers=8, num_threads=1)
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
        sigma = model.sigma_param.get_value()
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
            (x, labels) = next(iter_data_loader)
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

            if is_global:
                loss_val, state = training_step_global_adjustment(graphdef, state, x[..., 0], projection_parameters,
                                                                  means, weights, sigma, grid_size=grid_size)
            else:
                loss_val, state = training_step_local_adjustment(graphdef, state, x[..., 0], projection_parameters)

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