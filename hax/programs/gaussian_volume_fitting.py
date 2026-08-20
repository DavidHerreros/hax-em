"""Fit a volume with a sum of isotropic Gaussians sharing one width.

This is the single Gaussian-fitting implementation in the package: it produces the point
cloud (positions, amplitudes, one shared sigma) that HetSIREN / Zernike3Deep / ReconSIREN
deform, and it must speak their convention exactly -- amplitudes are the *masses* of
**unit-integral** Gaussians and ``sigma`` is the width of the normalized kernel that
``PhysDecoder.finish`` and ``decode_volume`` apply after a trilinear mass splat. Rendering
here is the same two steps (``splat_weights_trilinear`` then ``FastVariableBlur3D``), which
is why the fit and the network agree, and why the render costs ``O(box^3 log box)``
regardless of how many Gaussians there are.

**Sigma and point spacing are one parameter, not two.** A row of Gaussians spaced ``s``
apart is flat to ~1% once ``sigma >= s/2`` (the ripple falls off as
``exp(-2 pi^2 sigma^2 / s^2)``), so anything narrower opens gaps; meanwhile the render
multiplies the spectrum by ``exp(-2 pi^2 sigma^2 k^2)``, so anything wider destroys
resolution. Left free against a pointwise volume MSE, sigma always wins by collapsing --
a delta on a voxel reproduces that voxel exactly -- and the result reproduces the
reference beautifully while being useless to deform: at a fraction of a voxel a sub-voxel
displacement moves a Gaussian's whole amplitude between neighbouring voxels, so learned
motion renders as speckle rather than as structure.

The fix is to stop treating sigma as free. It is *derived* from the spacing that ``N``
points imply over the mask (:func:`_sigma_for_count`), so the only real degree of freedom
is the point count, and the resolution the mixture can carry is a monotone function of it.
That turns "the fewest Gaussians that reproduce the map" into a one-dimensional search,
which is what :func:`fit_volume_adaptive` brackets.

The proportionality constant is ``sigma_factor``, and the tiling argument's value of 0.5 is
what it is set to -- but only because that is what measured best end to end, not because
the argument settles it. Held at fixed ``N`` on three references, 0.5 was best on a smooth
simulated phantom while both real maps preferred ~0.35 (ribosome 18.3 A vs 21.2 A). An
attempt to exploit that by measuring the multiplier per map -- fit briefly at each candidate
on a cheap probe cloud, keep the best -- made every dataset *worse*, because the best
multiplier turns out to depend on ``N`` as well as on the map, so a value probed at one
count does not transfer to another. Tuning this properly needs a joint search over count
and width; a cheap one-dimensional probe is not sound, and 0.5 stands until then.

Two quantitative notes, both measured rather than derived, because the obvious derivations
are wrong. Closed-form sizing from the Gaussian envelope (retain fraction ``a`` of the
amplitude at resolution ``d``, hence ``sigma <= sqrt(ln(1/a) / (2 pi^2 (sr/d)^2))``) is a
*bound*, not an estimate: it treats the render as reference-times-envelope, while the
fitted masses partly invert that envelope, so the achievable width came out ~4x larger than
predicted. And the cost of resolution is not cubic: measured across three references,
``sigma ~ d^0.58`` and ``N ~ d^-1.88``, so relaxing the target from 4 to 8 A buys roughly
3.7x fewer points, not 8x.

Because a mixture is *always* improved by more and narrower Gaussians, the search needs a
floor on the width to have an answer at all; that floor, not a voxel count, is what caps
``N``.

Two things this deliberately does not gate on, both of which were measured and rejected:

* the fitting loss -- real-space MSE is dominated by the low frequencies, where a cryoEM
  map holds nearly all its power, so the fit plateaus long before the high shells are
  right;
* the FSC -- it normalises each map by its own power, so it barely responds to the pure
  amplitude loss that too wide a Gaussian inflicts. A one-voxel blur removes 99.8% of a
  map's Nyquist amplitude and still scores FSC ~0.7 there.

The gate is :func:`~hax.utils.shell_relative_error` instead, which is exactly ``|1 - c|``
when the fit comes out as ``c`` times the reference in a shell.

:func:`fit_volume` is the fixed-``N`` path, used when the caller states the count.
"""

import math
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
from cuml.neighbors.nearest_neighbors import NearestNeighbors

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
        # pynndescent returns (indices, distances); cuML below returns (distances, indices).
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


@partial(jax.jit, static_argnames=("grid_size", "ctf_type"))
def image_affine_stats(graphdef, state, target, projection_parameters, grid_size, ctf_type):
    """Closed-form per-image contrast/bias fit between the current model projection and the images.

    For every image ``i`` in the batch this solves the 1-D least-squares problem

        target_i  ~=  a_i * P_i  +  b_i

    where ``P_i`` is the CTF-applied projection of the *current* volume (linear in the Gaussian
    weights) and ``(a_i, b_i)`` are the per-image contrast (slope) and background (offset). The
    solution is the ordinary regression estimate ``a_i = Cov(P_i, T_i) / Var(P_i)`` obtained from a
    handful of pixel sums -- no optimizer, no epochs. ``var_p`` is returned so the host can drop
    ill-posed images (flat/empty projections) before aggregating the per-image slopes robustly.

    Returns ``(a, b, var_p)`` each of shape ``[B]``.
    """
    model = nnx.merge(graphdef, state)
    if isinstance(model, tuple):
        model = model[0]

    means = model.means.get_value()
    weights = nnx.relu(model.weights.get_value())
    sigma = nnx.relu(model.sigma_param.get_value())

    # Batch alignments
    euler_angles = projection_parameters["euler_angles"]
    rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    shifts = projection_parameters["shifts"]

    # Batch CTFs. The padded/unpadded switch is the one every other CTF site in the package
    # uses -- the delocalization the padding contains is a fixed physical length, so past a
    # large enough box doubling it again only quadruples the FFT for nothing.
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

    # The forward model has to predict the image *as stored*, or the slope comes out biased.
    #   wiener/precorrect : the images are raw observations that the network deconvolves at
    #                       train time, so deconvolve them here too and compare against an
    #                       unmodulated projection.
    #   premultiplied     : whatever extracted the particles already multiplied them by their
    #                       CTF, so the stored pixels carry CTF^2 * P(V).
    #   apply / None      : predict CTF * P(V) as-is.
    if ctf_type in ["wiener", "precorrect"]:
        target = wiener2DFilter(jnp.squeeze(target), ctf)
        ctf = jnp.ones_like(ctf)
    elif ctf_type == "premultiplied":
        ctf = ctf ** 2

    proj = splat_weights_bilinear(grid_size, means, weights, sigma, rotations, shifts, ctf)

    # Per-image ordinary least squares in image space: target ~= a * proj + b
    proj = proj.reshape(proj.shape[0], -1).astype(jnp.float32)
    tgt = target.reshape(target.shape[0], -1).astype(jnp.float32)
    n = proj.shape[1]

    sum_p = jnp.sum(proj, axis=1)
    sum_t = jnp.sum(tgt, axis=1)
    sum_pp = jnp.sum(proj * proj, axis=1)
    sum_pt = jnp.sum(proj * tgt, axis=1)

    var_p = sum_pp / n - (sum_p / n) ** 2
    cov_pt = sum_pt / n - (sum_p / n) * (sum_t / n)

    a = cov_pt / jnp.where(var_p > 0, var_p, 1.0)
    b = (sum_t - a * sum_p) / n
    return a, b, var_p


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


# --- 4. ADAPTIVE (AUTOMATIC N) FIT -------------------------------------------------------


def _seed_uniform(mask_bool, n, rng):
    """``n`` positions drawn uniformly over the masked voxels, in voxel coordinates.

    Uniform, deliberately, and *not* weighted by density. One shared sigma is only valid
    where the point spacing is uniform: seeding proportional to density packs points onto
    the strongest features and thins them everywhere else, so a single width is
    simultaneously too wide for the dense regions and too narrow to tile the sparse ones.
    The density is carried by the amplitudes, which are free.
    """
    idxs = np.argwhere(mask_bool)
    if idxs.shape[0] == 0:
        raise ValueError("The mask given to the Gaussian fit is empty.")
    chosen = rng.choice(idxs.shape[0], size=int(n), replace=int(n) > idxs.shape[0])
    pos = idxs[chosen].astype(np.float32)
    # Jitter inside the voxel each point stands for, so coincident draws separate.
    pos += rng.uniform(-0.5, 0.5, size=pos.shape).astype(np.float32)
    return pos


def _mass_matched_weights(field, positions_voxels, mask_bool):
    """Amplitudes for a fresh cloud, matched in total mass to ``field``.

    The amplitudes are the *masses* of unit-integral Gaussians, so the render's integral is
    simply their sum. Seeding them with the field's VALUE at each point -- the obvious thing,
    and what this used to do -- is wrong by the peak of a unit-integral Gaussian,
    ``(2 pi)^(3/2) sigma^3``: a factor of 8 at sigma = 0.8 px and 130 at sigma = 2. That
    factor grows with sigma, i.e. with how FEW Gaussians there are, so a coarse cloud starts
    furthest from its answer and needs the most steps to walk back -- which reads as the
    coarse fits being physically incapable when they are merely unconverged, and sends the
    outer search chasing a threshold that is an artefact of its own initialisation.

    Distributing the field's integral in proportion to its value is exact in mass and free
    of sigma entirely.
    """
    idx = np.clip(np.rint(positions_voxels).astype(int), 0, np.asarray(field.shape) - 1)
    w = np.clip(field[idx[:, 0], idx[:, 1], idx[:, 2]], 0.0, None).astype(np.float32)

    target_mass = float(np.clip(field, 0.0, None)[mask_bool].sum())
    sampled_mass = float(w.sum())
    if sampled_mass <= 0:
        # Every seed landed on a zero of the field (an all-negative residual, say). Spread
        # the mass evenly rather than returning a cloud of zeros the optimizer cannot leave:
        # a zero amplitude has a zero gradient through the render.
        return np.full(w.shape, target_mass / max(w.shape[0], 1), np.float32)
    return w * (target_mass / sampled_mass)


def _sigma_for_count(n_mask_voxels, n_gaussians, sigma_factor=0.5, min_sigma=0.0):
    """Splat width implied by fitting ``n_gaussians`` into ``n_mask_voxels``.

    Deliberately a function of the *intended* spacing rather than of the spacing the
    fitted cloud ends up with. Measuring the cloud instead looks more faithful and is a
    trap: MSE fitting happily piles several points onto the same density peak, and a width
    re-derived from that collapsed cloud shrinks towards zero, which rewards the collapse
    and lands right back on the sub-voxel comb this whole design exists to avoid. Measured
    on the phantom, the realised median nearest-neighbour spacing came out at 0.004 voxels
    for a cloud whose intended spacing was 5.4.

    ``min_sigma`` is applied to the WIDTH, not converted into a cap on the count, and that
    distinction is load-bearing. Deriving a maximum count from the width floor instead
    (``N_max = V / (min_sigma / sigma_factor)^3``) couples the two: calibrating the width
    multiplier downwards then also slashes the count ceiling, and since fidelity is driven
    far more by the count than by the width, the net effect is a large regression. Measured
    on the ribosome, that coupling cut the ceiling from 68857 to 23618 points and took the
    fit from 9.6 A to 25.1 A.

    Flooring the width instead keeps the search well posed. Below the floor the width
    shrinks with the spacing; at the floor it stops, and adding further points simply makes
    the cloud denser at a fixed width, which strictly enlarges what the mixture can
    represent. So fidelity is non-decreasing in ``N`` everywhere, which is exactly the
    monotonicity the outer bracketing search assumes. Without the floor it is *not*
    monotone -- past the best width, more Gaussians make the fit worse.
    """
    spacing = (float(n_mask_voxels) / max(1.0, float(n_gaussians))) ** (1.0 / 3.0)
    return max(float(min_sigma), float(sigma_factor) * spacing), spacing


def _param_filter():
    """Which parameters the fixed-width fit is allowed to move."""
    return nnx.All(nnx.Param, (nnx.PathContains('means'), nnx.PathContains('weights')))


@partial(jax.jit, static_argnames=("update",))
def training_step_fixed_width(graphdef, state, target, update=True):
    """One step of the fixed-width fit: only positions and amplitudes move.

    ``sigma`` is deliberately absent from the optimizer's parameter filter. It is not that
    a width penalty was omitted -- it is that against a pointwise volume MSE there is no
    penalty that makes a free sigma behave, because a narrower Gaussian always fits better
    (a delta on a voxel reproduces that voxel exactly) right down to the sub-voxel comb that
    ruins the downstream deformation. The width is a function of the point spacing, so the
    caller derives it and pins it here.
    """
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, target):
        recon = model()
        recon_loss = jnp.mean((recon - target) ** 2.)

        # One-sided hinge keeping points inside the box. Summed, not averaged: a point that
        # has left the grid contributes nothing to the reconstruction and so has no other
        # gradient pulling it back, and the strength of that pull should not depend on how
        # many well-behaved points it is averaged against.
        means = model.means.get_value()
        violation = jax.nn.relu(jnp.abs(means) - 0.98)
        boundary_loss = jnp.sum(violation ** 2.)

        return recon_loss + boundary_loss

    loss_val, grads = nnx.value_and_grad(loss_fn)(model, target)

    if update:
        grads, _ = grads.split(_param_filter(), ...)
        optimizer.update(model, grads)
        state = nnx.state((model, optimizer))

    return loss_val, state


def _build_fixed_width_model(grid_size, positions_voxels, weights, sigma):
    """A ``GaussianSplatModel`` at the given cloud, with its width pinned."""
    factor = 0.5 * grid_size
    coords = (np.asarray(positions_voxels, np.float32) - factor) / factor
    model = GaussianSplatModel(manual_init={"means": coords, "weights": np.asarray(weights, np.float32)},
                               grid_size=grid_size, rngs=nnx.Rngs(0))
    model.sigma_param = nnx.Param(jnp.asarray([float(sigma)], dtype=jnp.float32))
    return model


def _refine_cloud(target_norm, positions_voxels, weights, sigma, iterations, position_lr_voxels,
                  quiet):
    """Adam on positions and amplitudes at a fixed point count and fixed width.

    One parameter shape, so one XLA compilation for the whole call. This is the single most
    expensive part of the fit and the reason the outer search moves ``N`` geometrically
    instead of by a few points at a time: every change of ``N`` retraces and recompiles a
    whole-volume render, so ~8 rounds cost 8 compilations where a per-100-step densify loop
    costs a couple of hundred.

    """
    grid_size = int(target_norm.shape[0])
    model = _build_fixed_width_model(grid_size, positions_voxels, weights, sigma)

    # A single learning rate serves both parameter groups because Adam's step is ~lr
    # regardless of gradient magnitude and the caller has normalized the target to unit
    # signal std: amplitudes are then O(1), like the normalized coordinates. Expressed in
    # voxels so it means the same thing at any box size.
    lr = float(position_lr_voxels) / (0.5 * grid_size)
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=max(1, int(iterations)), alpha=0.05)
    params_filter = _param_filter()
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=params_filter)

    graphdef, state = nnx.split((model, optimizer))
    target_j = jnp.asarray(target_norm)

    pbar = tqdm(range(int(iterations)), desc=f"Fitting N={int(np.asarray(weights).shape[0])}", file=sys.stdout,
                ascii=" >=", colour="green", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                disable=quiet)
    loss_val = None
    for _ in pbar:
        loss_val, state = training_step_fixed_width(graphdef, state, target_j, update=True)
        pbar.set_postfix_str(f"| Loss: {float(loss_val):.6g}")
    pbar.close()

    model, _ = nnx.merge(graphdef, state)
    factor = 0.5 * grid_size
    positions = np.asarray(model.means.get_value()) * factor + factor
    weights_out = np.asarray(model.weights.get_value())
    return positions, weights_out, float(loss_val) if loss_val is not None else float("nan")


def _render_cloud(grid_size, positions_voxels, weights, sigma):
    """Render a cloud to a volume through exactly the model's own forward pass."""
    model = _build_fixed_width_model(grid_size, positions_voxels, weights, sigma)
    graphdef, state = nnx.split(model)
    return np.asarray(splat_volume(graphdef, state))


def fit_volume_adaptive(target_vol, mask, sr, resolution=None, shell_error_tol=0.3,
                        sigma_factor=0.5, min_sigma=0.8, max_rounds=8, max_gaussians=None,
                        iterations_per_round=3000, position_lr_voxels=0.2,
                        prune_rel_threshold=0.05, seed=0, quiet=False):
    """Find the fewest Gaussians that reproduce ``target_vol`` without losing resolution.

    The search is one-dimensional. ``sigma`` is not a free parameter -- it is
    ``sigma_factor`` times the spacing that ``N`` points imply over the mask (see
    :func:`_sigma_for_count`) -- so the point count alone sets how much resolution the
    mixture carries, monotonically. The loop fits at a count, measures how far out the
    render still reproduces the reference, and brackets the smallest count that clears the
    goal, searching *downward* as readily as upward: stopping at the first count that works
    would answer "a count that works", not "the fewest".

    **What "reproduces" means here.** The gate is the relative error per shell
    (:func:`~hax.utils.shell_relative_error`), ``e(k) <= shell_error_tol``, not the FSC and
    not the fitting loss. Both of the obvious choices are wrong for this question:

    * real-space MSE is dominated by the low frequencies, where a cryoEM map holds nearly
      all its power, so the fit sits on a converged-looking plateau while the high shells
      are still badly wrong;
    * FSC normalises each map by its own power, so it barely responds to a *pure amplitude*
      loss -- which is precisely what too wide a Gaussian inflicts. Blurring a map by one
      voxel removes 99.8% of its Nyquist amplitude and still scores FSC ~0.7 there.

    ``e(k)`` is exactly ``|1 - c|`` when the fit comes out as ``c`` times the reference in a
    shell, so ``shell_error_tol`` reads directly as "every shell's amplitude is right to
    within this fraction".

    Shells where the reference has no power are skipped, not failed: a map that came out of
    ``reconstruct_volume`` is FSC-denoised, hence identically zero past its own resolution.
    Pass ``resolution`` (Angstrom) to stop at a coarser target than the map's own limit --
    measured, ``N ~ d^-1.9``, so relaxing it from 4 to 8 A is roughly a 3.7x smaller cloud.

    ``sigma_factor`` sets the width as a multiple of the point spacing. The default of 0.5 is
    the value that measured best end to end; see the module docstring for why it is not
    tuned per map even though the best value demonstrably varies.

    ``min_sigma`` floors the render width and thereby caps the count. It is the one place
    the downstream shows through: a mixture is always reproduced better by more, narrower
    Gaussians, right down to one delta per voxel, and that limit is worthless to deform.
    When the floor is what stopped the search the run says so, with the resolution actually
    reached -- an unfiltered reference with power out to Nyquist genuinely cannot be matched
    there by anything that is still a continuous density.

    Returns ``(model, info)``; ``info["history"]`` records what each count bought.
    """
    from hax.utils import (volume_fsc, shell_relative_error, shell_resolution,
                           live_shell_limit)

    target_vol = np.asarray(target_vol, np.float32)
    grid_size = int(target_vol.shape[0])
    mask_bool = np.asarray(mask) > 0.5
    n_mask = int(mask_bool.sum())
    if n_mask == 0:
        raise ValueError("The mask given to the Gaussian fit is empty.")

    rng = np.random.default_rng(seed)

    # Normalize to unit signal std so amplitudes land at O(1), the same scale the normalized
    # coordinates live on -- that is what lets one Adam learning rate serve both. Splatting
    # is linear in the weights, so undoing it at the end is exact rather than approximate.
    signal_std = float(np.std(target_vol[mask_bool]))
    if not np.isfinite(signal_std) or signal_std <= 0:
        signal_std = float(np.std(target_vol)) or 1.0
    target_norm = target_vol / signal_std

    sigma_factor = float(sigma_factor)

    # Ceiling on the count. This is a RESOURCE limit, deliberately independent of the width
    # floor: the two used to be tied (max_gaussians derived from min_sigma), which made
    # calibrating the width downwards silently shrink the search space and cost far more
    # fidelity than the better width won back. The width is floored in _sigma_for_count
    # instead, where it belongs.
    #
    # One Gaussian per four masked voxels. Not a physical bound -- the honest bound is one
    # per voxel, which is no model at all -- but the point at which the cloud stops being
    # affordable downstream: every per-particle tensor in HetSIREN's decode and render chain
    # scales with this count.
    if max_gaussians is None:
        max_gaussians = n_mask // 4
    max_gaussians = int(np.clip(max_gaussians, 64, n_mask))

    # How far out the reference itself carries signal -- the goal, unless the caller asked
    # for something coarser.
    _, power_ref0 = shell_relative_error(np.zeros_like(target_norm), target_norm, mask=mask_bool)
    d_ref, last_live = live_shell_limit(power_ref0, grid_size, sr)
    goal = float(resolution) if resolution is not None else d_ref
    goal = max(goal, 2.0 * sr)

    # Start AT the ceiling and search downward for the smallest count that still meets the
    # goal.
    #
    # This looks wasteful -- the first round is the most expensive one -- and an earlier
    # version duly started at a quarter of the ceiling instead, on the reasoning that the
    # answer would usually be far below it. Measured on three references, that reasoning was
    # wrong: for real maps the answer really is near the ceiling, because fidelity keeps
    # improving with the count right up to it, and starting low left the large counts
    # unevaluated. Six of nine configurations came out worse.
    #
    # (The value here used to come from a closed-form sizing formula, which clipped to the
    # ceiling in 12 of 12 configurations. That formula was a bound rather than an estimate --
    # it assumes the Gaussian envelope alone shapes the spectrum, while the fitted masses
    # partly invert it, so it over-asked by ~64x. It has been removed; starting at the
    # ceiling is what it was doing in practice, only stated honestly.)
    n_current = int(max_gaussians)

    print(f"\n{bcolors.OKCYAN}###### Adaptive Gaussian fit on a {grid_size}^3 volume "
          f"({n_mask} masked voxels, {sr:.3f} A/px) ######{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}Reference carries signal to {d_ref:.2f} A; fitting to {goal:.2f} A "
          f"at a per-shell amplitude error <= {shell_error_tol:.0%}. Starting at N = {n_current} "
          f"(sigma_factor {sigma_factor:.2f}, width floored at {min_sigma:.2f} px, "
          f"ceiling {max_gaussians} points).{bcolors.ENDC}")

    history = []
    positions = _seed_uniform(mask_bool, n_current, rng)
    weights = _mass_matched_weights(target_norm, positions, mask_bool)

    best_pass = None   # smallest N that met the goal -- the answer we are looking for
    best_any = None    # best resolution seen, used only if nothing ever meets the goal
    fail_hi = None     # largest count known to MISS the goal
    pass_lo = None     # smallest count known to MEET it
    for round_idx in range(int(max_rounds)):
        n_current = int(positions.shape[0])
        sigma, spacing = _sigma_for_count(n_mask, n_current, sigma_factor, min_sigma)

        positions, weights, loss = _refine_cloud(
            target_norm, positions, weights, sigma, iterations_per_round, position_lr_voxels, quiet)

        render = _render_cloud(grid_size, positions, weights, sigma)
        err, power_ref = shell_relative_error(render, target_norm, mask=mask_bool)
        d_fit, _ = shell_resolution(err, power_ref, grid_size, sr,
                                    threshold=shell_error_tol, ascending_is_bad=True)
        fsc, _ = volume_fsc(render, target_norm, mask=mask_bool)
        d_fsc, _ = shell_resolution(fsc, power_ref, grid_size, sr, threshold=0.5,
                                    ascending_is_bad=False)

        passed = d_fit <= goal + 1e-6
        history.append({"round": round_idx, "n": n_current, "sigma": float(sigma),
                        "spacing": float(spacing), "loss": loss, "resolution": d_fit,
                        "fsc_resolution": d_fsc, "reference_resolution": d_ref,
                        "passed": bool(passed)})

        print(f"{bcolors.OKGREEN if passed else bcolors.WARNING}"
              f"  round {round_idx}: N = {n_current:>7d}  sigma = {sigma:.2f} px  "
              f"spacing = {spacing:.2f} px  ->  reproduced to {d_fit:.2f} A "
              f"(goal {goal:.2f} A, FSC 0.5 at {d_fsc:.2f} A){bcolors.ENDC}")

        snapshot = (positions.copy(), weights.copy(), float(sigma), d_fit, n_current)
        if passed and (best_pass is None or n_current < best_pass[4]):
            best_pass = snapshot
        if best_any is None or d_fit < best_any[3]:
            best_any = snapshot

        if passed:
            pass_lo = n_current if pass_lo is None else min(pass_lo, n_current)
        else:
            fail_hi = n_current if fail_hi is None else max(fail_hi, n_current)

        if round_idx == max_rounds - 1:
            break

        # Where to look next. The search is bracketing, not a plain Newton step: the model
        # a Newton step would use (resolution ~ sigma ~ N^(-1/3)) is only roughly true, and
        # a step built on it oscillated between the ceiling and a quarter of it instead of
        # converging. Bracketing cannot do that -- once one count has passed and a smaller
        # one has failed, the answer is between them and the interval only shrinks.
        #
        # Searching downward at all is the point: stopping at the first count that clears
        # the goal answers "a count that works", not "the fewest", and the sizing formula is
        # conservative often enough that overshoot is the common case.
        if pass_lo is not None and fail_hi is not None and fail_hi < pass_lo:
            n_next = int(round(math.sqrt(float(fail_hi) * float(pass_lo))))
        elif pass_lo is not None:
            n_next = int(math.ceil(pass_lo / 2.0))          # everything tried passed: go down
        else:
            n_next = int(math.ceil(n_current * 2.0))        # nothing has passed yet: go up
        n_next = int(np.clip(n_next, 64, max_gaussians))

        if n_next == n_current or abs(n_next - n_current) < 0.2 * n_current:
            # The bracket is tight enough that another round would re-measure what we have.
            if pass_lo is None and n_current >= max_gaussians:
                print(f"{bcolors.WARNING}  Stopped at the point-count ceiling (N = {max_gaussians}, "
                      f"one Gaussian per 4 masked voxels): the reference has detail this cloud "
                      f"cannot carry, so the fit reaches {d_fit:.2f} A rather than {goal:.2f} A. "
                      f"Pass --fit_resolution {d_fit:.1f} to make that explicit, raise "
                      f"max_gaussians to spend more points, or accept the loss.{bcolors.ENDC}")
            break

        if n_next > n_current:
            # Grow, warm-starting from the cloud already fitted. The added points are seeded
            # uniformly (see _seed_uniform) and carry the *residual*, mass-matched, so they
            # start out explaining exactly what the previous count could not.
            residual = np.clip(target_norm - render, 0.0, None)
            extra = _seed_uniform(mask_bool, n_next - n_current, rng)
            positions = np.concatenate([positions, extra])
            weights = np.concatenate([weights, _mass_matched_weights(residual, extra, mask_bool)])
        else:
            # Shrink by dropping points at random rather than by amplitude. A random subset
            # of a uniform cloud is still uniform, which is the property the single shared
            # width depends on; keeping the strongest points instead would concentrate them
            # on the dense regions and leave the rest too sparse for any one sigma to tile.
            # Scaling the survivors up by the same fraction keeps the total mass, so the
            # smaller cloud starts at the right density rather than at n_next/n_current of it.
            keep = rng.choice(n_current, size=n_next, replace=False)
            positions = positions[keep]
            weights = weights[keep] * (float(n_current) / float(n_next))

    if best_pass is not None:
        positions, weights, sigma, d_fit, _ = best_pass
    else:
        positions, weights, sigma, d_fit, _ = best_any
        print(f"{bcolors.WARNING}  No count reached {goal:.2f} A; keeping the best "
              f"({d_fit:.2f} A).{bcolors.ENDC}")

    # ---- final prune ----------------------------------------------------------------
    # "Minimum number of Gaussians" is not only about where the growth loop stopped: within
    # a given N some points end up carrying nothing. Drop those and re-measure, keeping the
    # prune only if it cost no resolution. sigma is deliberately NOT re-derived from the
    # smaller count -- the surviving points sit where they were, so the spacing that matters
    # is unchanged and re-deriving would widen the splats for no reason.
    positive = weights[weights > 0]
    if prune_rel_threshold > 0 and positive.size > 0:
        keep = weights > prune_rel_threshold * float(np.median(positive))
        if 0 < int(keep.sum()) < weights.shape[0]:
            p_pruned, w_pruned = positions[keep], weights[keep]
            render_p = _render_cloud(grid_size, p_pruned, w_pruned, sigma)
            err_p, power_p = shell_relative_error(render_p, target_norm, mask=mask_bool)
            d_pruned, _ = shell_resolution(err_p, power_p, grid_size, sr,
                                           threshold=shell_error_tol, ascending_is_bad=True)
            if d_pruned <= d_fit + 1e-6:
                print(f"{bcolors.OKGREEN}  Prune: {weights.shape[0]} -> {int(keep.sum())} Gaussians "
                      f"at the same {d_pruned:.2f} A.{bcolors.ENDC}")
                positions, weights, d_fit = p_pruned, w_pruned, d_pruned
            else:
                print(f"{bcolors.WARNING}  Prune rejected: it would cost resolution "
                      f"({d_pruned:.2f} A vs {d_fit:.2f} A).{bcolors.ENDC}")

    # Undo the normalization: exact, because the render is linear in the weights.
    weights = weights * signal_std

    model = _build_fixed_width_model(grid_size, positions, weights, sigma)
    model.update_config()

    print(f"{bcolors.OKGREEN}Adaptive fit done: {positions.shape[0]} Gaussians, sigma = {sigma:.2f} px "
          f"({sigma * sr:.2f} A), reproducing the reference to {d_fit:.2f} A "
          f"(per-shell amplitude error <= {shell_error_tol:.0%}).{bcolors.ENDC}")

    return model, {"history": history, "n_gaussians": int(positions.shape[0]), "sigma": float(sigma),
                   "resolution": d_fit, "shell_error_tol": float(shell_error_tol),
                   "goal": goal, "signal_std": signal_std}


def _build_projection_parameters(md_columns, labels, sr, ctf_type):
    projection_parameters = {"euler_angles": md_columns["euler_angles"][labels],
                             "shifts": md_columns["shifts"][labels]}
    if ctf_type in ["apply", "wiener", "precorrect", "premultiplied"]:
        ctf_parameters = {"ctfDefocusU": md_columns["ctfDefocusU"][labels],
                          "ctfDefocusV": md_columns["ctfDefocusV"][labels],
                          "ctfDefocusAngle": md_columns["ctfDefocusAngle"][labels],
                          "ctfSphericalAberration": md_columns["ctfSphericalAberration"][labels],
                          "ctfVoltage": md_columns["ctfVoltage"][labels],
                          "sr": sr}
        projection_parameters = dict(projection_parameters, **ctf_parameters)
    return projection_parameters


def adjust_weights_to_images(model, md_path, mmap_output_dir, sr, batch_size=256, learning_rate=0.01, num_epochs=3,
                             is_global=True, ctf_type="apply", max_samples=8192):
    """Match the contrast of the model's projections to the experimental images.

    Estimates a single positive contrast scale ``a`` (and background ``b``) that best maps the current
    volume's projections onto the images, then rescales the Gaussian weights by ``a``. Because
    splatting is *linear* in the weights, ``a * P(V)`` is exactly the projection of the rescaled
    volume, so this is solved in **closed form**: each image gives an independent regression slope
    ``a_i = Cov(P_i, T_i) / Var(P_i)``, and the global scale is the *median* of the per-image slopes
    (robust to junk/outlier particles). This needs a single streaming pass over a subsample of at most
    ``max_samples`` images -- no optimizer, no epochs -- and is both faster and more accurate than
    gradient descent on the scalar. ``learning_rate``/``num_epochs``/``is_global`` are ignored and kept
    only so existing call sites keep working.

    Note on the background ``b``: the image-space offset is a per-image DC/solvent level and is *not*
    representable in a single shared volume, so it is estimated for diagnostics but not baked into the
    weights. Adding it to the per-Gaussian amplitudes instead -- which is what the deleted
    ``fit_weights_to_images`` did -- projects to a mask-shaped blob rather than to a flat DC, so it
    cannot represent the background it is meant to absorb and distorts the reference in exchange.

    Getting this right matters more than its size suggests: HetSIREN pins the density scale *once*,
    here, and its training loss carries no global gray level. A reference whose amplitude is wrong by
    a constant leaves an error the occupancy head can cancel in one cheap direction and the motion
    head cannot, so the optimizer spends its budget on occupancy and the deformation never forms.

    Returns ``(model, info)`` with ``info`` a diagnostics dict.
    """
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

    print(f"\n{bcolors.OKCYAN}###### Adjusting gaussian weights to images (Global, closed-form)... ######{bcolors.ENDC}")

    graphdef, state = nnx.split(model)
    grid_size = model.grid_size

    # Only a subsample of images is needed to estimate two global scalars.
    n_target = min(max_samples, len(generator.md)) if max_samples is not None else len(generator.md)
    n_steps = max(1, int(np.ceil(n_target / batch_size)))

    a_all, var_all, b_all = [], [], []
    pbar = tqdm(range(n_steps), desc="Fitting contrast", file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    with closing(iter(data_loader)) as iter_data_loader:
        for _ in pbar:
            (x, labels) = next(iter_data_loader)
            projection_parameters = _build_projection_parameters(md_columns, labels, sr, ctf_type)
            a, b, var_p = image_affine_stats(graphdef, state, x[..., 0], projection_parameters,
                                             grid_size=grid_size, ctf_type=ctf_type)
            a_all.append(np.asarray(a))
            b_all.append(np.asarray(b))
            var_all.append(np.asarray(var_p))

    a_all = np.concatenate(a_all)
    b_all = np.concatenate(b_all)
    var_all = np.concatenate(var_all)

    # Drop ill-posed images (flat projection) and non-finite slopes before aggregating.
    valid = np.isfinite(a_all) & np.isfinite(b_all) & (var_all > 1e-8 * np.median(var_all[var_all > 0]))
    if not np.any(valid):
        print(f"{bcolors.WARNING}No valid projections to fit contrast; leaving weights unchanged.{bcolors.ENDC}")
        return model, {"scale": 1.0, "offset": 0.0, "n_used": 0}

    scale = float(np.median(a_all[valid]))
    offset = float(np.median(b_all[valid]))
    scale = max(scale, 0.0)  # contrast is non-negative (matches the previous relu on a)

    print(f"{bcolors.OKGREEN}Contrast scale a = {scale:.4f} (median of {int(valid.sum())} images), "
          f"background b = {offset:.4g} [not applied to shared volume].{bcolors.ENDC}")

    model.weights = nnx.Param(scale * model.weights.get_value())
    return model, {"scale": scale, "offset": offset, "n_used": int(valid.sum()),
                   "a_per_image": a_all[valid]}


@jax.jit
def splat_volume(graphdef, state):
    model = nnx.merge(graphdef, state)
    if isinstance(model, tuple):
        model = model[0]
    return model()