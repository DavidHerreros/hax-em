#!/usr/bin/env python


import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import dm_pix

import numpy as np
from functools import partial

from einops import rearrange

from sklearn.cluster import KMeans

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


def lowpass_sigma_from_resolution(resolution_A, sr):
    """Gaussian width, in pixels, that is at half amplitude at ``resolution_A``.

    A Gaussian of width ``sigma`` multiplies the spectrum by ``exp(-2 pi^2 sigma^2 k^2)``,
    so asking for half amplitude at ``k = sr / d`` gives
    ``sigma = sqrt(ln 2 / (2 pi^2)) * d / sr``. Stating the band-limit in Angstrom rather
    than in pixels keeps the schedule meaningful across box sizes and sampling rates --
    the same 40 A start is the same physical resolution whatever the box.
    """
    return float(np.sqrt(np.log(2.0) / (2.0 * np.pi ** 2)) * resolution_A / sr)


def lowpass_sigma_schedule(step, total_steps, sigma_start, frac=0.6):
    """Cosine anneal of the comparison band-limit, ``sigma_start`` -> 0 over ``frac`` of training.

    This is coarse-to-fine pose search, applied on the model side. The width of the
    correlation peak in orientation scales with the resolution the projections are
    compared at, so a search that starts at full Nyquist has a basin of attraction far
    narrower than the anchor spacing and the pose head never leaves its initialisation.
    Starting wide makes that basin large enough to fall into and then narrows it.

    A cosine is used rather than a linear ramp because it is flat at both ends: the
    opening plateau covers the soft-selection (``use_tau``) window for free, and the
    closing plateau stops the band-limit from still moving while the poses settle.

    The schedule targets *zero* extra width, not the splat width. What is annealed is a
    band-limit applied identically to the renders and to the targets; the mixture's own
    ``sigma`` is part of the forward model and is left alone (see the caller, which
    composes the two as ``sqrt(sigma_splat^2 + sigma_lp^2)``).
    """
    if sigma_start <= 0.0 or total_steps <= 0:
        return 0.0
    ramp = max(1.0, frac * total_steps)
    t = min(1.0, step / ramp)
    return float(0.5 * sigma_start * (1.0 + np.cos(np.pi * t)))


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


class PoseHead(nnx.Module):
    def __init__(self, is_refine=False, *, rngs: nnx.Rngs):
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
            hidden_layers.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers = nnx.List(hidden_layers)
        self.pose_layer = Linear(1024, 6, rngs=rngs, kernel_init=kernel_init, bias_init=bias_init)

    def __call__(self, x):
        for layer in self.hidden_layers:
            # x = nnx.gelu(x + layer(x))
            x = nnx.gelu(layer(x))
        return self.pose_layer(x)


class PoseHeadEnsemble(nnx.Module):
    def __init__(self, num_members, is_refine=False, *, rngs: nnx.Rngs):
        key = rngs.params()
        member_keys = jax.random.split(key, num_members)

        @nnx.vmap(in_axes=(0), out_axes=0)
        def make_member(key):
            return PoseHead(is_refine=is_refine, rngs=nnx.Rngs(key))

        self.ensemble = make_member(member_keys)

    def __call__(self, x):
        @nnx.vmap(in_axes=(0, None), out_axes=1)
        def forward(model, x):
            return model(x)
        return forward(self.ensemble, x)


class EncoderPose(nnx.Module):
    def __init__(self, input_dim, pyramid_levels=4, num_components=18, refine_current_assignment=False,
                 use_anchor_rotations=True, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 64  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 3))
        self.pyramid_levels = pyramid_levels
        self.num_components = num_components
        self.refine_current_assignment = refine_current_assignment
        self.use_anchor_rotations = use_anchor_rotations

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

        hidden_layers_linear = [Linear(self.out_conv_dim * self.out_conv_dim * 512, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        hidden_layers_linear.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers_linear.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_layers_linear.append(Linear(1024, 8, rngs=rngs))
        self.hidden_layers_linear = nnx.List(hidden_layers_linear)

        # Anchor rotations
        self.anchor_rotations = jnp.array(generate_spherical_rotations(num_components))

        # Layers to 9D rotation
        self.ensemble_6d_heads = PoseHeadEnsemble(num_members=num_components, is_refine=False, rngs=rngs)

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

        # Linear hidden layers
        x = rearrange(x, 'b h w c -> b (h w c)')
        for layer in self.hidden_layers_linear[:-1]:
            if layer.in_features == layer.out_features:
                x = nnx.gelu(x + layer(x))
            else:
                x = nnx.gelu(layer(x))
        x = self.hidden_layers_linear[-1](x)

        # First output: rotation matrices
        rotations_6d = self.ensemble_6d_heads(x)

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

        # Broadcast shifts to euler angles shape
        in_plane_shifts = jnp.broadcast_to(in_plane_shifts[:, None, :], (in_plane_shifts.shape[0], self.num_components, 2))

        return rotations, in_plane_shifts


class EncoderHet(nnx.Module):
    def __init__(self, input_dim, lat_dim=8, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 4))
        # The stack used to open with a dense Linear(box^2, 64^2) -- a learned resampler from
        # the full image to the working resolution. It was 268M parameters at box 256, 45% of
        # the whole model, and 419M at box 320, which is what made the heterogeneity encoder
        # the single largest thing on the device and made it grow with the box. It is now a
        # jax.image.resize, which costs nothing and is exactly what EncoderPose already does
        # at its own input.
        hidden_layers_conv = [
            Conv(1, 4, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16)]
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
        # Resample to the working resolution (see __init__ for what this replaces).
        x = jax.image.resize(x, (x.shape[0], self.input_conv_dim, self.input_conv_dim, 1), method="bilinear")

        for layer in self.hidden_layers_conv:
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
    def __init__(self, coords, values, volume_size, learn_delta_volume=True, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.learn_delta_volume = learn_delta_volume
        self.n_gaussians = coords.shape[0]
        self.factor = 0.5 * volume_size
        self.coords = coords[None, ...]
        self.reference_values = values[None, ...]

        # The consensus deltas are held directly rather than decoded by an MLP.
        #
        # The MLP that used to sit here took no argument: it read `self.coords`, which is a
        # plain array and not a parameter, so every layer consumed a constant and the whole
        # stack evaluated to a constant. It was 218M parameters at 30k Gaussians -- 37% of
        # the model -- to express 4N = 120k free numbers, and it carried no inductive bias
        # to justify the cost: all 3N coordinates were flattened into a single vector, so
        # unlike a coordinate network evaluated per point there was no per-point structure
        # and no smoothness prior over the cloud.
        #
        # A parameter of the same shape as the output is exactly as expressive. It is not
        # optimiser-neutral -- a deep over-parameterisation conditions Adam differently --
        # so treat this as a change to be measured, not a pure refactor.
        shape = (1, self.n_gaussians, 4 if learn_delta_volume else 3)
        if jnp.all(self.reference_values == 0):
            # No reference to perturb, so the deltas have to start somewhere.
            self.deltas = nnx.Param(0.01 * jax.random.normal(rngs.params(), shape))
        else:
            # Matches the old zeros-initialised readout: start on the reference volume.
            self.deltas = nnx.Param(jnp.zeros(shape))

    def __call__(self):
        if self.learn_delta_volume:
            deltas = self.deltas.get_value()
            delta_coords, delta_values = deltas[..., :3], deltas[..., 3]

            # Recover volume values (TODO: Check if applying ReLu is really needed)
            values = nnx.relu(self.reference_values + delta_values)
        else:
            # Positions stay on the reference cloud. The old code reshaped the flattened
            # *coordinates* into delta_coords here, which doubled every coordinate and put
            # the cloud outside the box; nothing fed a gradient back, so the deltas were
            # frozen either way and zero is what freezing was meant to mean.
            delta_coords = jnp.zeros_like(self.coords)

            # Recover volume values (TODO: Check if applying ReLu is really needed)
            values = nnx.relu(self.reference_values)

        # Recover coords (non-normalized)
        coords = self.factor * (self.coords + delta_coords)

        return coords, values

    def decode_volume(self, coords_values=None, filter=True, sigma=1.0):
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
            grids = jax.vmap(low_pass_3d, in_axes=(0, None))(grids, sigma)

        return grids

class HetVolumeDecoder(nnx.Module):
    def __init__(self, coords, values, n_gaussians, lat_dim, volume_size, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.n_gaussians = n_gaussians
        self.coords = coords[None, ...]
        self.reference_values = values[None, ...]

        # Indices to (normalized) coords
        self.factor = 0.5 * volume_size

        hidden = [
            Siren2Linear(in_features=lat_dim, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=True,
                         w0=30.0, s=0.0, c=1.0)]
        for _ in range(4):
            hidden.append(
                Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=False,
                             custom_init=True, is_residual=True, w0=1.0, s=0.0, c=6.0))
        hidden.append(Linear(in_features=8, out_features=4 * n_gaussians, rngs=rngs,
                                    kernel_init=nnx.initializers.glorot_uniform()))
        self.hidden = nnx.List(hidden)

    def __call__(self, x):
        # Decode coords
        x = self.hidden[0](x)
        for layer in self.hidden[1:-1]:
            x = layer(x)
        x = self.hidden[-1](x)

        # Extract delta_coords and values
        x = jnp.reshape(x, (x.shape[0], self.n_gaussians, 4))
        delta_coords, delta_values = x[..., :3], x[..., 3]

        # Recover coords (non-normalized)
        coords = self.factor * (self.coords + delta_coords)

        # Recover volume values
        values = nnx.relu(self.reference_values + delta_values)

        return coords, values

    def decode_volume(self, x, filter=True, sigma=1.0):
        # Decode volume values
        coords, values = self.__call__(x)

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
            grids = jax.vmap(low_pass_3d, in_axes=(0, None))(grids, sigma)

        return grids

class PhysDecoder:
    def __init__(self, xsize):
        self.xsize = xsize

    def __call__(self, x, values, coords, xsize, rotations, shifts, ctf, ctf_type, std, filter=True):
        # Volume factor
        factor = 0.5 * xsize

        # Flatten rotations and shifts
        rotations_flat = rearrange(rotations, "b n m d -> (b n) m d")
        shifts_flat = rearrange(shifts, "b n m -> (b n) m")

        # Apply rotation matrices
        coords = jnp.matmul(coords, rearrange(rotations_flat, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts_flat[:, None, :] + factor

        # Scatter image
        B = rotations_flat.shape[0]
        c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
        images = jnp.zeros((B, xsize, xsize), dtype=x.dtype)

        bposf = jnp.floor(c_sampling)
        bposi = bposf.astype(jnp.int32)
        bposf = c_sampling - bposf

        # Accumulate the four bilinear corners one at a time rather than concatenating them
        # first: the concatenated form materialises the amplitudes at 4x and the indices at
        # 8x the point count in one go, which at 30k Gaussians and a full hypothesis sweep
        # is the largest buffer in the render.
        fx, fy = bposf[:, :, 0], bposf[:, :, 1]

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        scatter = jax.vmap(scatter_img)
        for offset, weight in (((0, 0), (1.0 - fx) * (1.0 - fy)),
                               ((1, 0), fx * (1.0 - fy)),
                               ((1, 1), fx * fy),
                               ((0, 1), (1.0 - fx) * fy)):
            images = scatter(images, bposi + jnp.array(offset), values * weight)

        # Splat envelope and CTF, in one Fourier pass. The envelope has to be the exact
        # exp(-2 pi^2 sigma^2 k^2) rather than a 9-tap kernel: the annealed width starts
        # around 5-7 px, where 9 taps span well under one sigma and the render comes out
        # far sharper than asked for -- which is precisely the band-limit the anneal
        # exists to impose.
        if ctf_type in ["apply", "wiener", "squared"]:
            ctf = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf = rearrange(ctf, "b n w h -> (b n) w h")
            images = gaussianCTFFilter(images, sigma=std if filter else None, ctf=ctf)
        else:
            images = gaussianCTFFilter(images, sigma=std if filter else None, ctf=None)

        images = rearrange(images, "(b n) w h -> b n w h", b=rotations.shape[0], n=rotations.shape[1])

        return images

class ReconSIREN(nnx.Module):

    @save_config
    def __init__(self, coords, values, xsize, sr, bank_size=1024, ctf_type="apply", lat_dim=8, sigma=1.0,
                 symmetry_group="c1", refine_current_assignment=False, learn_delta_volume=True, num_components=18,
                 use_anchor_rotations=True, *, rngs: nnx.Rngs):
        super(ReconSIREN, self).__init__()
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.symmetry_matrices = symmetry_matrices(symmetry_group)
        self.refine_current_assignment = refine_current_assignment
        self.learn_delta_volume = learn_delta_volume
        self.encoder_pose = EncoderPose(self.xsize, num_components=num_components, refine_current_assignment=refine_current_assignment,
                                        use_anchor_rotations=use_anchor_rotations, rngs=rngs)
        self.encoder_het = EncoderHet(self.xsize, lat_dim=lat_dim, rngs=rngs)
        self.delta_volume_decoder = DeltaVolumeDecoder(coords=coords, values=values, volume_size=self.xsize, learn_delta_volume=learn_delta_volume, rngs=rngs)
        self.delta_het_decoder = HetVolumeDecoder(coords=coords, values=values, n_gaussians=coords.shape[0], lat_dim=lat_dim, volume_size=self.xsize, rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize)

        # Gaussian std
        self.log_std = nnx.Param(jnp.log(sigma))

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        raw = jax.random.normal(rngs.params(), (bank_size, 3))
        array_init = raw / jnp.linalg.norm(raw, axis=-1, keepdims=True)
        self.memory_bank = MemoryBank(array_init=array_init)

    def __call__(self, x, rngs: nnx.Rngs = None, **kwargs):
        # TODO: Return only best angles
        return self.encoder_pose(x, rngs=rngs)
    
    def get_std(self):
        return jnp.exp(self.log_std.get_value())

    def decode_image(self, x, labels, md, ctf_type=None):
        # Precompute batch CTFs
        if self.ctf_type is not None:
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

        # Decode het volume
        vol = self.delta_het_decoder.decode_volume(x, filter=filter, sigma=self.get_std())

        return vol


@partial(jax.jit, static_argnames=("use_tau", "use_lowpass"))
def train_step_reconsiren(graphdef, state, x, labels, md, key, tau=0.0001, use_tau=False, lambda_uniform=0.1,
                          sigma_lp=0.0, use_lowpass=False):
    """One optimisation step.

    ``sigma_lp`` is the extra band-limit (in pixels) applied to *both* the renders and the
    targets this step -- see :func:`lowpass_sigma_schedule`. It is traced, so it can move
    every step without forcing a recompile; ``use_lowpass`` is static and only says whether
    the feature is on at all, so a run that does not use it pays nothing.
    """
    model, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)

    # Random keys
    key, swd_key, uniform_key, choice_key, distributions_key = jax.random.split(key, 5)

    def loss_fn(model, x):
        # Correct CTF in images for encoder if needed
        if model.ctf_type in ["apply", "squared"]:
            x_ctf_corrected = prepare_image_cryocrab(x, ctf)
            # x_ctf_corrected = prepare_image_wiener(x, ctf)
        else:
            x_ctf_corrected = x

        # Get euler angles and shifts
        rotations, shifts = model.encoder_pose(x_ctf_corrected)
        sample, latent, logstd = model.encoder_het(x_ctf_corrected, rngs=distributions_key)

        # Decode volume
        coords, values = model.delta_volume_decoder()

        # Decode het volume
        coords_het, values_het = model.delta_het_decoder(latent)

        # Refine current assignment (if provided)
        # rotations = jnp.matmul(rotations, current_rotations[:, None, :, :])
        rotations = jnp.matmul(current_rotations[:, None, :, :], rotations)  # TODO: The two options seem to work?
        shifts = current_shifts[:, None, :] + shifts

        # Random symmetry matrices
        random_indices = jax.random.choice(choice_key, jnp.arange(model.symmetry_matrices.shape[0]), shape=(rotations.shape[0],))
        rotations = jnp.matmul(jnp.transpose(model.symmetry_matrices[random_indices], (0, 2, 1))[:, None, :, :], rotations)

        # Render width for this step: the mixture's own splat width, widened by the
        # annealed band-limit. Two Gaussians compose in quadrature, and both are diagonal
        # in Fourier space, so this is exactly "render, then low-pass" at no extra cost --
        # and it leaves the model's sigma (the tiling floor its point count implies)
        # untouched, which keeps the anneal a property of the *comparison* rather than of
        # the volume.
        sigma_render = jnp.sqrt(model.get_std() ** 2 + sigma_lp ** 2) if use_lowpass else model.get_std()

        # Selection pass. The loss below gathers exactly one hypothesis per particle, so
        # the other num_components - 1 renders receive no cotangent -- their forward
        # residuals are kept alive through the backward pass for nothing. Cutting every
        # differentiable input here puts the whole sweep in the primal-only part of the
        # trace, so nothing is retained, and the winner is re-rendered with gradients
        # below. Same arithmetic, one render's worth of activations instead of
        # num_components. (The selection itself is never differentiable: argmin has no
        # gradient, and jax.random.categorical does not propagate one through its logits.)
        images_corrected = model.phys_decoder(x, jax.lax.stop_gradient(values), jax.lax.stop_gradient(coords),
                                              model.xsize, jax.lax.stop_gradient(rotations),
                                              jax.lax.stop_gradient(shifts), ctf, model.ctf_type,
                                              jax.lax.stop_gradient(sigma_render))

        # Losses
        images_corrected_loss = images_corrected[..., 0] if images_corrected.shape[-1] == 1 else images_corrected
        x_loss_nb = x[..., 0] if x.shape[-1] == 1 else x

        # Consider CTF if Wiener/Squared mode (only for loss)
        if model.ctf_type == "wiener":
            ctf_broadcasted = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf_broadcasted = rearrange(ctf_broadcasted, "b n w h -> (b n) w h")

            x_loss_nb = wiener2DFilter(x_loss_nb, ctf, pad_factor=2)

            images_corrected_loss = rearrange(images_corrected_loss, "b n w h -> (b n) w h")
            images_corrected_loss = wiener2DFilter(images_corrected_loss, ctf_broadcasted, pad_factor=2)
            images_corrected_loss = rearrange(images_corrected_loss, "(b n) w h -> b n w h")
        elif model.ctf_type == "squared":
            ctf_broadcasted = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf_broadcasted = rearrange(ctf_broadcasted, "b n w h -> (b n) w h")

            x_loss_nb = ctfFilter(x_loss_nb, ctf, pad_factor=2)

            images_corrected_loss = rearrange(images_corrected_loss, "b n w h -> (b n) w h")
            images_corrected_loss = ctfFilter(images_corrected_loss, ctf_broadcasted, pad_factor=2)
            images_corrected_loss = rearrange(images_corrected_loss, "(b n) w h -> b n w h")

        # Band-limit the targets by the same envelope the renders carry. Attenuating only
        # the renders would leave the data's high frequencies -- which the mixture cannot
        # produce at any sigma -- in the MSE as a term that is identical across the pose
        # hypotheses, so it adds variance to the argmin without telling it anything.
        #
        # standard_normalization works off per-image statistics, so normalising the
        # (B, H, W) stack and then broadcasting is identical to broadcasting and then
        # normalising -- and it band-limits B images rather than B * num_components.
        x_pose_target = standard_normalization(x_loss_nb)
        if use_lowpass:
            x_pose_target = gaussianCTFFilter(x_pose_target, sigma=sigma_lp, ctf=None)
            x_het_target = gaussianCTFFilter(x_loss_nb, sigma=sigma_lp, ctf=None)
        else:
            x_het_target = x_loss_nb

        # Broadcast input images to right size
        x_loss = jnp.broadcast_to(x_pose_target[:, None, ...], (x_pose_target.shape[0], images_corrected.shape[1], x_pose_target.shape[1], x_pose_target.shape[2]))

        x_flat = rearrange(x_loss, "b n w h -> (b n) w h")
        images_corrected_flat = rearrange(images_corrected_loss, "b n w h -> (b n) w h")

        # Bandpass (TODO: Make optional to membrane proteins only)
        # x_flat = bandpass_filter(x_flat, pixel_size_A=model.sr, highpass_A=50.)
        # images_corrected_flat = bandpass_filter(images_corrected_flat, pixel_size_A=model.sr, highpass_A=50.)

        recon_loss = dm_pix.mse(images_corrected_flat[..., None], x_flat[..., None])
        recon_loss = rearrange(recon_loss, "(b n) -> b n", b=images_corrected_loss.shape[0], n=images_corrected_loss.shape[1])

        # Get minimum indices
        if use_tau:
            selection_logits = -recon_loss / tau
            min_indices = jax.random.categorical(key, selection_logits, axis=-1)
        else:
            min_indices = jnp.argmin(recon_loss, axis=1)

        # Heterogeneity
        min_indices_het = jnp.argmin(recon_loss, axis=1)
        rotations_het = rotations[jnp.arange(images_corrected.shape[0]), min_indices_het, :][:, None, ...]
        shifts_het = shifts[jnp.arange(images_corrected.shape[0]), min_indices_het, :][:, None, ...]
        # TODO: Test stop gradient in rotations_het and shifts_het
        images_het = model.phys_decoder(x, values_het, coords_het, model.xsize, jax.lax.stop_gradient(rotations_het),
                                        jax.lax.stop_gradient(shifts_het), ctf, model.ctf_type, sigma_render)[:, 0, ...]
        images_het_loss = images_het[..., 0] if images_het.shape[-1] == 1 else images_het
        if model.ctf_type == "wiener":
            images_het_loss = wiener2DFilter(images_het_loss, ctf, pad_factor=2)
        elif model.ctf_type == "squared":
            images_het_loss = ctfFilter(images_het_loss, ctf, pad_factor=2)
        # x_loss_nb = standard_normalization(x_loss_nb)

        # Bandpass (TODO: Make optional to membran proteins only)
        # x_loss_nb = bandpass_filter(x_loss_nb, pixel_size_A=model.sr, highpass_A=50.)
        # images_het_loss = bandpass_filter(images_het_loss, pixel_size_A=model.sr, highpass_A=50.)

        recon_het_loss = dm_pix.mse(images_het_loss[..., None], x_het_target[..., None]).mean()

        # Gradient pass: re-render only the hypothesis the selection just picked. Gathering
        # the winner keeps the path back to the pose encoder, so this carries exactly the
        # gradient the indexed loss used to carry -- and it is numerically the same number,
        # since the selection pass rendered these very poses.
        rotations_sel = rotations[jnp.arange(images_corrected.shape[0]), min_indices, :][:, None, ...]
        shifts_sel = shifts[jnp.arange(images_corrected.shape[0]), min_indices, :][:, None, ...]
        images_sel = model.phys_decoder(x, values, coords, model.xsize, rotations_sel, shifts_sel,
                                        ctf, model.ctf_type, sigma_render)[:, 0, ...]
        images_sel_loss = images_sel[..., 0] if images_sel.shape[-1] == 1 else images_sel
        if model.ctf_type == "wiener":
            images_sel_loss = wiener2DFilter(images_sel_loss, ctf, pad_factor=2)
        elif model.ctf_type == "squared":
            images_sel_loss = ctfFilter(images_sel_loss, ctf, pad_factor=2)

        recon_loss = dm_pix.mse(images_sel_loss[..., None], x_pose_target[..., None]).mean()
        recon_loss_all = 0.5 * (recon_loss + recon_het_loss)
        
        # Viewing directions from rotations
        rotations = rearrange(rotations, "b n w h -> (b n) w h")
        directions = rotations[:, :, 2]

        # An L1 term on the amplitudes and a VAE KL term used to be computed here and never
        # reached `loss`; :func:`repulsion_loss` was computed and multiplied by 0.0. None of
        # them were free -- `0.0 * x` is not eliminated (0 * NaN is NaN), and the repulsion
        # term materialises a (batch * num_components)^2 x 3 pairwise tensor. Reinstate them
        # by adding them to `loss`, not by computing them and discarding the result.

        # Decoupling (TODO: In the future this will be for missing angles like TF implementation)

        # Uniform angular distribution loss
        loss_swd = sliced_wasserstein_sphere(directions, rng=key, n_projections=64)
        loss_uniform = lambda_uniform * loss_swd

        loss = (recon_loss_all + 1.0 * loss_uniform)
        return loss, (recon_loss, loss_uniform, directions)

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
    if model.ctf_type is not None:
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

    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, (params_pose, params_volume, params_het)), has_aux=True)
    (loss, (recon_loss, loss_uniform, directions)), grads_combined = grad_fn(model, x)

    grads_pose, grads_volume, grads_het = grads_combined.split(params_pose, params_volume, params_het)

    optimizer_pose.update(model, grads_pose)
    optimizer_volume.update(model, grads_volume)
    optimizer_het.update(model, grads_het)

    # Update memory bank
    model.memory_bank.enqueue(directions)

    state = nnx.state((model, optimizer_pose, optimizer_volume, optimizer_het))

    return loss, recon_loss, state, key


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
    if model.ctf_type is not None:
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
    if model.ctf_type is not None:
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

    # Generate projections
    images_corrected = model.phys_decoder(x, values, coords, model.xsize, rotations, shifts, ctf, model.ctf_type, model.get_std())

    # Losses
    images_corrected_loss = images_corrected[..., 0] if images_corrected.shape[-1] == 1 else images_corrected
    x_loss = x[..., 0] if x.shape[-1] == 1 else x

    # Consider CTF if Wiener/Squared mode (only for loss)
    if model.ctf_type == "wiener":
        ctf_broadcasted = jnp.broadcast_to(ctf[:, None, :],
                                           (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
        ctf_broadcasted = rearrange(ctf_broadcasted, "b n w h -> (b n) w h")

        x_loss = wiener2DFilter(x_loss, ctf, pad_factor=2)

        images_corrected_loss = rearrange(images_corrected_loss, "b n w h -> (b n) w h")
        images_corrected_loss = wiener2DFilter(images_corrected_loss, ctf_broadcasted, pad_factor=2)
        images_corrected_loss = rearrange(images_corrected_loss, "(b n) w h -> b n w h")
    elif model.ctf_type == "squared":
        ctf_broadcasted = jnp.broadcast_to(ctf[:, None, :],
                                           (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
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

    # Index shifts and rotations based on extracted indices
    rotations = rotations[jnp.arange(images_corrected.shape[0]), min_indices, :]
    shifts = shifts[jnp.arange(images_corrected.shape[0]), min_indices, :]

    return rotations, shifts, latent

euler_from_matrix_batch = jax.vmap(jax.jit(euler_from_matrix))

def xmippEulerFromMatrix(matrix):
    return -jnp.rad2deg(euler_from_matrix_batch(matrix))


def main():
    import os
    import sys
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

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_vol(parser,
               help="If provided, the neural network will start from this volume when assigning the angles and shifts to the images.")
    ca.add_mask(parser,
                help=f"ReconSIREN reconstruction mask (the mask provided must be binary - "
                     f"{bcolors.WARNING}NOTE{bcolors.ENDC}: since this is a reconstruction mask, it should be defined such that it covers the "
                     f"volume were the motions of interest are expected to happen)")
    parser.add_argument("--num_gaussians", required=False, type=int, default=10000,
                        help="Before training the network, HetSIREN will try to fit a set of Gaussians in the reference volume to recreate it. "
                            "The default criterium is to automatically determine the number of Gaussians neede to reproduce the reference volume "
                            "with high-fidelity. However, if you prefer to fix the number of Gaussians in advance based on your own criterium (e.g., "
                            "the number of residues in your protein), you can set this parameter. When set, the HetSIREN will fit this fixed number of Gaussians "
                            "so that the reproduce the reference volume as well as possible.")
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
    parser.add_argument("--lowpass_start", required=False, type=float, default=None,
                        help=f"Resolution (in Angstrom) at which the {bcolors.ITALIC}ab initio{bcolors.ENDC} pose search starts, annealed to full "
                             f"resolution as training proceeds. Projections and images are compared through a Gaussian band-limit that starts here "
                             f"and is removed on a cosine schedule (see --lowpass_frac); the same envelope is applied to both sides, so this narrows "
                             f"what the loss can see rather than what the map can hold. This exists because the width of the correlation peak in "
                             f"orientation scales with the resolution the projections are compared at: searching at full Nyquist from step one leaves "
                             f"a basin of attraction much narrower than the spacing of the pose anchors, and the pose encoder never leaves its "
                             f"initialisation (the giveaway is an angular distribution with exactly --num_components distinct directions). Values "
                             f"around 30-40 A are a reasonable start. {bcolors.WARNING}NOTE{bcolors.ENDC}: off by default, so runs behave as before "
                             f"unless you ask for it.")
    parser.add_argument("--lowpass_frac", required=False, type=float, default=0.6,
                        help="Fraction of the total training over which --lowpass_start is annealed away (default: 0.6). The remaining fraction trains "
                             "at full resolution. The schedule is a cosine, so it is flat at both ends: it holds wide over the early exploration phase "
                             "and stops moving before the poses settle. Ignored unless --lowpass_start is given.")
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
            model, _, _ = fit_volume(vol * mask_fit, mask=mask_fit, iterations=20000, learning_rate=0.001, n_init=args.num_gaussians, fixed_gaussians=True)

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
        coords = 0.25 * jnp.array(generate_sphere_points(args.num_gaussians) + np.random.normal(0, 0.1, (args.num_gaussians, 3)))
        # coords = generate_cylinder_points(args.num_gaussians, radius=0.25, height=1.0) + np.random.normal(0, 0.1, (args.num_gaussians, 3))
        values = jnp.full((args.num_gaussians,), 0.01)
        sigma = 1.0


    # # If exists, clean MMAP
    # if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key, choice_key = jax.random.split(rng, 3)

    # Prepare network (ReconSIREN)
    reconsiren = ReconSIREN(coords, values, xsize, args.sr, ctf_type=args.ctf_type, symmetry_group=args.symmetry_group,
                            refine_current_assignment=args.refine_current_assignment, lat_dim=8, sigma=sigma,
                            bank_size=10000, learn_delta_volume=not args.do_not_learn_volume,
                            num_components=args.num_components,
                            use_anchor_rotations=not args.do_not_use_anchor_rotations, rngs=nnx.Rngs(model_key))

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
            return model.delta_volume_decoder.decode_volume(sigma=model.get_std())

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
        graphdef, state = nnx.split((reconsiren, optimizer_pose, optimizer_volume, optimizer_het))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"),
                                                                                        optimizer_pose, optimizer_volume, optimizer_het)
            print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
        else:
            resume_epoch = 0

        # Coarse-to-fine schedule for the pose search. The band-limit is expressed in
        # Angstrom on the CLI and converted once here, so it means the same thing whatever
        # the box and sampling rate.
        total_training_steps = args.epochs * steps_per_epoch
        use_lowpass = args.lowpass_start is not None
        sigma_lp_start = lowpass_sigma_from_resolution(args.lowpass_start, args.sr) if use_lowpass else 0.0
        if use_lowpass:
            print(f"{bcolors.OKCYAN}\nPose search band-limit: {args.lowpass_start:.1f} A "
                  f"(sigma {sigma_lp_start:.2f} px) annealed to full resolution over "
                  f"{100 * args.lowpass_frac:.0f}% of training{bcolors.ENDC}")

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

                if total_steps % steps_per_epoch == 0:
                    total_loss = 0
                    total_recon_loss = 0
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

                            # Plot angular distribution
                            directions = np.array(reconsiren.memory_bank.get())
                            dir_x, dir_y, dir_z = directions[:, 0], directions[:, 1], directions[:, 2]
                            beta = jnp.arccos(jnp.clip(dir_z, -1.0, 1.0))
                            alpha = jnp.arctan2(dir_y, dir_x)
                            euler_angles = jnp.stack([alpha, beta], axis=-1)
                            fig, _ = plot_angular_distribution(euler_angles)
                            writer.add_figure("Angular distribution density", fig, global_step=i)

                            # Predict some heterogeneous volumes
                            n_latent_steps = int(min(steps_per_epoch,
                                                     np.ceil(LATENTS_FOR_CLUSTERING / args.batch_size)))
                            latents = []
                            graphdef_aux, state_aux = nnx.split(reconsiren)
                            for _ in range(n_latent_steps):
                                (x_latent, labels_latent) = next(iter_data_loader_train)
                                _, _, latent = predict_angular_assignment_step_reconsiren(graphdef_aux, state_aux,
                                                                                          x_latent, labels_latent,
                                                                                          md_columns, rng)
                                latents.append(np.array(latent))
                            latents = np.concatenate(latents, axis=0)
                            n_clusters = int(min(10, latents.shape[0]))
                            kmeans = KMeans(n_clusters=n_clusters).fit(latents)
                            decoded_centers = [np.array(decode_het_volume(reconsiren, center[None, ...]))
                                               for center in kmeans.cluster_centers_]

                        logger.submit(write_het_volumes, decoded_centers, args.output_path)

                    # Save checkpoint model
                    if logger.should("checkpoint", i):
                        with logger.section():
                            NeuralNetworkCheckpointer.save_intermediate(graphdef, state,
                                                                        os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"),
                                                                        epoch=i, wait=False)

                    i += 1

                if total_steps <= 1500:
                    tau = 1e-3
                    use_tau = True
                else:
                    tau = 0.0
                    use_tau = False
                sigma_lp = jnp.float32(lowpass_sigma_schedule(total_steps, total_training_steps,
                                                              sigma_lp_start, args.lowpass_frac))
                loss, recon_loss, state, rng = train_step_reconsiren(graphdef, state, x, labels, md_columns, rng, lambda_uniform=0.1, tau=tau, use_tau=use_tau,
                                                                     sigma_lp=sigma_lp, use_lowpass=use_lowpass)
                total_loss += loss
                total_recon_loss += recon_loss

                # Summary writer (training loss)
                if logger.should_log_scalars(step):
                    mean_loss = float(total_loss) / step
                    mean_recon_loss = float(total_recon_loss) / step

                    writer.add_scalar('Training loss (ReconSIREN)',
                                      mean_loss,
                                      i * steps_per_epoch + step)

                    writer.add_scalars('Reconstruction loss (ReconSIREN)',
                                       {"train": mean_recon_loss},
                                       i * steps_per_epoch + step)

                    if use_lowpass:
                        writer.add_scalar('Pose search band-limit (sigma, px)',
                                          float(sigma_lp),
                                          i * steps_per_epoch + step)

                    # Progress bar update  (TQDM)
                    lp_str = f" | lowpass={float(sigma_lp):.2f}px" if use_lowpass else ""
                    pbar.set_postfix_str(f"loss={mean_loss:.5f} | recon_loss={mean_recon_loss:.5f}{lp_str}")

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

        # Jitted functions for volume prediction
        decode_volume = jax.jit(reconsiren.delta_volume_decoder.decode_volume)
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
            rotations, shifts, latent = predict_angular_assignment_step_reconsiren(graphdef, state, x, labels, md_columns, rng)

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
