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

# Parameters for sphere-coverage loss
CANDIDATE_COVERAGE_EPOCHS = 10.0
CANDIDATE_COVERAGE_BINS = 256
CANDIDATE_COVERAGE_KAPPA = 32.0
CANDIDATE_BANK_SAMPLES = 1024
CANDIDATE_BANK_MIX = 0.5

# Learning rates of the consensus volume and heterogeneity encoder/decoder
VOLUME_LEARNING_RATE = 3e-3
AMPLITUDE_LEARNING_RATE = 1e-3
HET_LEARNING_RATE = 1e-4

# Heterogeneity latent statistics
HET_MIN_STD = 0.1
HET_LATENT_BANK_SIZE = 2048

# Extra render blur during the early warm-up phase, to avoid dusty collapse
CLOUD_BLUR_WARMUP = 1.5

# Ab initio geometry features
KNN_NEIGHBORS = 6
RECYCLE_DEAD_FRACTION = 0.05

# Companion maps written next to the physical map
SHARPENED_MAP_REG = 0.02
EQUALIZED_MAP_GAMMA = 0.5


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
        self.shift_layer = Linear(1024, 2, rngs=rngs,
                                  kernel_init=nnx.initializers.zeros_init(),
                                  bias_init=nnx.initializers.zeros_init())

    def __call__(self, x):
        for layer in self.hidden_layers:
            # x = nnx.gelu(x + layer(x))
            x = nnx.gelu(layer(x))
        pose = self.pose_layer(x)
        return jnp.concat([pose, self.shift_layer(x)], axis=-1)


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
        head_outputs = self.ensemble_6d_heads(x)
        rotations_6d, shift_deltas = head_outputs[..., :6], head_outputs[..., 6:]

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

        # Per-candidate deltas further modify the shared in-plane shift to improve its
        # accuracy for a given image
        in_plane_shifts = in_plane_shifts[:, None, :] + shift_deltas

        return rotations, in_plane_shifts


class EncoderHet(nnx.Module):
    def __init__(self, input_dim, lat_dim=8, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.out_conv_dim = -(-self.input_dim // (2 ** 4))

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
    def __init__(self, coords, values, volume_size, learn_delta_volume=True, parameterization="network", *, rngs: nnx.Rngs):
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
            grids = jax.vmap(low_pass_3d_analytic, in_axes=(0, None))(grids, sigma)

        return grids


class HetVolumeDecoder(nnx.Module):
    def __init__(self, coords, values, n_gaussians, lat_dim, volume_size,
                 residual_to_consensus=False, center_decoder=False,
                 small_final_init=False,
                 *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.n_gaussians = n_gaussians
        self.coords = coords[None, ...]
        self.reference_values = values[None, ...]
        self.residual_to_consensus = bool(residual_to_consensus)
        self.center_decoder = bool(center_decoder)

        # Indices to (normalized) coords
        self.factor = 0.5 * volume_size

        hidden = [
            Siren2Linear(in_features=lat_dim, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=True,
                         w0=30.0, s=0.0, c=1.0)]
        for _ in range(4):
            hidden.append(
                Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=False,
                             custom_init=True, is_residual=True, w0=1.0, s=0.0, c=6.0))
        final_init = (nnx.initializers.normal(1e-4) if small_final_init else nnx.initializers.glorot_uniform())
        hidden.append(Linear(in_features=8, out_features=4 * n_gaussians, rngs=rngs, kernel_init=final_init, bias_init=nnx.initializers.zeros_init()))
        self.hidden = nnx.List(hidden)

    def decode_deltas(self, x):
        x = self.hidden[0](x)
        for layer in self.hidden[1:-1]:
            x = layer(x)
        x = self.hidden[-1](x)
        return jnp.reshape(x, (x.shape[0], self.n_gaussians, 4))

    def __call__(self, x, base_coords=None, base_values=None):
        deltas = self.decode_deltas(x)
        if self.center_decoder:
            # Let z=0 be the consensus map, so any latent vector represents the deviation to be
            # considered to get a given heterogeneous state
            deltas = deltas - self.decode_deltas(jnp.zeros_like(x))
        delta_coords, delta_values = deltas[..., :3], deltas[..., 3]

        if self.residual_to_consensus:
            if base_coords is None or base_values is None:
                raise ValueError("Consensus-relative heterogeneity requires base coordinates and values")
            coords = base_coords + self.factor * delta_coords
            values = nnx.relu(base_values + delta_values)
        else:
            coords = self.factor * (self.coords + delta_coords)
            values = nnx.relu(self.reference_values + delta_values)

        return coords, values

    def decode_volume(self, x, filter=True, sigma=1.0, base_coords=None, base_values=None):
        coords, values = self.__call__(x, base_coords=base_coords, base_values=base_values)
        return splat_cloud_volumes(coords + self.factor, values, self.volume_size,
                                    filter=filter, sigma=sigma)


class PhysDecoder:
    def __init__(self, xsize):
        self.xsize = xsize

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

    def __call__(self, x, values, coords, xsize, rotations, shifts, ctf, ctf_type, std,
                 filter=True):
        rotations_flat = rearrange(rotations, "b n m d -> (b n) m d")
        shifts_flat = rearrange(shifts, "b n m -> (b n) m")

        images = self._scatter(values, coords, xsize, rotations_flat, shifts_flat, x.dtype)

        apply_ctf = ctf_type in ["apply", "wiener", "squared"]
        if apply_ctf:
            ctf = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf = rearrange(ctf, "b n w h -> (b n) w h")

        # Apply CTF + low pass filter
        images = gaussianCTFFilter(images, sigma=std if filter else None,
                                   ctf=ctf if apply_ctf else None, pad_factor=2)

        images = rearrange(images, "(b n) w h -> b n w h", b=rotations.shape[0], n=rotations.shape[1])

        return images

class ReconSIREN(nnx.Module):

    @save_config
    def __init__(self, coords, values, xsize, sr, bank_size=1024, ctf_type="apply", lat_dim=8, sigma=1.0,
                 symmetry_group="c1", refine_current_assignment=False, learn_delta_volume=True, num_components=18,
                 use_anchor_rotations=True, consensus_parameterization=None, heterogeneity_profile="legacy",
                 het_start_epoch=None, sigma_min=0.0, *, rngs: nnx.Rngs, **kwargs):
        super(ReconSIREN, self).__init__()
        anti_collapse = heterogeneity_profile == "anti_collapse"
        if heterogeneity_profile not in ("legacy", "anti_collapse"):
            raise ValueError("heterogeneity_profile must be 'legacy' or 'anti_collapse'")
        consensus_parameterization = "direct" if consensus_parameterization is None else consensus_parameterization
        het_residual_to_consensus = anti_collapse
        het_center_decoder = anti_collapse
        het_mask_radius = 0.45 if anti_collapse else 0.0
        het_normalize_target = anti_collapse
        het_variance_weight = 1e-2 if anti_collapse else 0.0
        het_covariance_weight = 1e-3 if anti_collapse else 0.0
        het_start_epoch = (5 if anti_collapse else 0) if het_start_epoch is None else int(het_start_epoch)

        # Multiresolution heterogeneity reconstruction loss
        het_loss_scales = tuple(dict.fromkeys(
            [size for size in (64, 128) if size < xsize] + [int(xsize)]))
        if anti_collapse and len(het_loss_scales) == 3:
            het_loss_weights = (0.5, 0.3, 0.2)
        else:
            het_loss_weights = tuple(1.0 for _ in het_loss_scales)
        weight_sum = sum(het_loss_weights)
        het_loss_weights = tuple(weight / weight_sum for weight in het_loss_weights)

        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.heterogeneity_profile = heterogeneity_profile
        self.het_loss_scales = het_loss_scales
        self.het_loss_weights = het_loss_weights
        self.het_mask_radius = het_mask_radius
        self.het_normalize_target = het_normalize_target
        self.het_variance_weight = het_variance_weight
        self.het_covariance_weight = het_covariance_weight
        self.het_min_std = HET_MIN_STD
        self.het_start_epoch = max(0, het_start_epoch)
        self.symmetry_matrices = symmetry_matrices(symmetry_group)
        self.refine_current_assignment = refine_current_assignment
        self.learn_delta_volume = learn_delta_volume
        self.encoder_pose = EncoderPose(self.xsize, num_components=num_components, refine_current_assignment=refine_current_assignment,
                                        use_anchor_rotations=use_anchor_rotations, rngs=rngs)
        self.encoder_het = EncoderHet(self.xsize, lat_dim=lat_dim, rngs=rngs)
        self.delta_volume_decoder = DeltaVolumeDecoder(coords=coords, values=values, volume_size=self.xsize,
                                                       learn_delta_volume=learn_delta_volume,
                                                       parameterization=consensus_parameterization, rngs=rngs)
        self.delta_het_decoder = HetVolumeDecoder(coords=coords, values=values, n_gaussians=coords.shape[0],
                                                  lat_dim=lat_dim, volume_size=self.xsize, residual_to_consensus=het_residual_to_consensus,
                                                  center_decoder=het_center_decoder, small_final_init=anti_collapse, rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize)

        # Gaussian std, floored so splats can't collapse below sigma_min
        self.sigma_min = float(sigma_min)
        init = jnp.maximum(jnp.asarray(sigma, jnp.float32) - self.sigma_min, 0.05)
        self.raw_std = nnx.Param(jnp.log(jnp.expm1(init)))

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.memory_bank = MemoryBank(array_init=jnp.zeros((bank_size, 3), dtype=jnp.float32))

        winner_init = jax.random.normal(rngs.params(), (bank_size, 3))
        winner_init = winner_init / jnp.linalg.norm(winner_init, axis=-1, keepdims=True)
        self.winner_memory_bank = MemoryBank(array_init=winner_init)

        self.candidate_bank_count = nnx.Variable(jnp.array(0, dtype=jnp.int32))
        self.winner_bank_count = nnx.Variable(jnp.array(0, dtype=jnp.int32))

        if anti_collapse:
            self.latent_memory_bank = MemoryBank(array_init=jnp.zeros((HET_LATENT_BANK_SIZE, lat_dim), dtype=jnp.float32))
            self.latent_bank_count = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    def __call__(self, x, rngs: nnx.Rngs = None, **kwargs):
        # TODO: Return only best angles
        return self.encoder_pose(x, rngs=rngs)
    
    def get_std(self):
        return self.sigma_min + jax.nn.softplus(self.raw_std.get_value())

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

    def _het_latent(self, x):
        if x.ndim == 4:
            _, x, _ = self.encoder_het(x)
        elif x.ndim == 3:
            _, x, _ = self.encoder_het(x[None, ...])
        elif x.ndim == 1:
            x = x[None, ...]
        return x

    def decode_het_cloud(self, x):
        """Heterogeneous point cloud (coords, masses) for a latent or an image."""
        x = self._het_latent(x)
        base_coords = base_values = None
        if self.delta_het_decoder.residual_to_consensus:
            base_coords, base_values = self.delta_volume_decoder()
        return self.delta_het_decoder(x, base_coords=base_coords, base_values=base_values)

    def decode_het_volume(self, x, filter=True):
        x = self._het_latent(x)

        base_coords = base_values = None
        if self.delta_het_decoder.residual_to_consensus:
            base_coords, base_values = self.delta_volume_decoder()
        vol = self.delta_het_decoder.decode_volume(
            x, filter=filter, sigma=self.get_std(),
            base_coords=base_coords, base_values=base_values)

        return vol


def candidate_reconstruction_losses(images, targets, ctf, ctf_type,
                                     normalize_target=True, return_prepared=False):
    """
    Compute the representation loss between the predicted images (images) and the experimental images (targets)

    Additionally, this function applies any additional CTF/normalization based on ctf_type and normalize_target
    """
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
    losses = jnp.mean(
        jnp.square(predicted - target[:, None, ...]), axis=(-2, -1))
    if return_prepared:
        return losses, predicted, target[:, None, ...]
    return losses


def prepare_heterogeneity_images(images, targets, ctf, ctf_type, normalize_target):
    """
    This is just a function to apply any additional CTF/normalization to prepare the experimental images (targets)
    and predicted heterogeneity images (images) to compute losses from them
    """
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


def resize_candidate_images(images, size):
    """
    This is a small helper function to resize images considering the two batch dimensions in ReconSIREN. Useful to
    compute multi-resolution related losses
    """
    if images.shape[-1] == size and images.shape[-2] == size:
        return images
    flat = rearrange(images, "b n h w -> (b n) h w 1")
    flat = jax.image.resize(
        flat, (flat.shape[0], size, size, 1), method="lanczos3", antialias=True)
    return rearrange(flat[..., 0], "(b n) h w -> b n h w", b=images.shape[0])


def circular_loss_mask(size, radius, dtype):
    """
    Function to compute on the fly a circular mask with a given radius. Used mainly to focus the losses on the
    regions where there is protein in the experimental images
    """
    if radius <= 0.0:
        return jnp.ones((size, size), dtype=dtype)
    axis = (jnp.arange(size, dtype=jnp.float32) + 0.5) / size - 0.5
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    return (xx * xx + yy * yy <= radius * radius).astype(dtype)


def heterogeneity_reconstruction_loss(images, targets, ctf, ctf_type,
                                       scales, weights, mask_radius,
                                       normalize_target, return_prepared=False):
    """
    Function to prepare the predicted heterogeneity images (images) and experimental images (targets) and compute
    representation loss from them.

    Additionally, the losses can follow a multi-resolution approximation to improve convergence.
    """
    predicted, target = prepare_heterogeneity_images(
        images, targets, ctf, ctf_type, normalize_target)
    loss = jnp.asarray(0.0, dtype=predicted.dtype)
    for size, weight in zip(scales, weights):
        predicted_level = resize_candidate_images(predicted, size)
        target_level = resize_candidate_images(target, size)
        mask = circular_loss_mask(size, mask_radius, predicted.dtype)
        squared = jnp.square(predicted_level - target_level) * mask[None, None, ...]
        loss = loss + weight * jnp.sum(squared) / (
            predicted.shape[0] * predicted.shape[1] * jnp.maximum(jnp.sum(mask), 1.0))
    if return_prepared:
        return loss, predicted, target
    return loss


@partial(jax.jit, donate_argnums=(1,))
def recycle_dead_points_reconsiren(graphdef, state, key,
                                   dead_fraction=RECYCLE_DEAD_FRACTION,
                                   new_value_fraction=0.25):
    """Move amplitude-dead Gaussians next to mass-carrying ones. This allows all Gaussians to contribute to the structure"""
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

    het_decoder = model.delta_het_decoder
    if isinstance(het_decoder, HetVolumeDecoder):
        readout = het_decoder.hidden[-1]
        dead_outputs = jnp.repeat(dead, 4)
        kernel = readout.kernel.get_value()
        readout.kernel.value = jnp.where(dead_outputs[None, :], jnp.zeros_like(kernel), kernel)
        if readout.bias is not None:
            bias = readout.bias.get_value()
            readout.bias.value = jnp.where(dead_outputs, jnp.zeros_like(bias), bias)

    state = nnx.state((model, optimizer_pose, optimizer_volume, optimizer_het))
    return state, jnp.sum(dead)


def volume_optimizer_transform(parameterization, volume_lr, coords_lr=None,
                               amplitude_lr=None):
    """Optax transform for the consensus volume decoder"""
    import optax

    if parameterization != "direct":
        return optax.chain(optax.clip_by_global_norm(1.0),
                           optax.adamw(volume_lr, eps=1e-6))

    coords_lr = volume_lr if coords_lr is None else coords_lr
    amplitude_lr = volume_lr if amplitude_lr is None else amplitude_lr

    def label_params(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: ("amplitude" if "delta_values" in jax.tree_util.keystr(path)
                             else "coords"),
            params)

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {"coords": optax.adamw(coords_lr, eps=1e-6),
             "amplitude": optax.adamw(amplitude_lr, eps=1e-6)},
            label_params))


def score_candidates(model, x, values, coords, rotations, shifts, ctf, std=None):
    """Render every candidate once and return its reconstruction loss."""
    std = model.get_std() if std is None else std
    images = model.phys_decoder(
        x, values, coords, model.xsize, rotations, shifts, ctf,
        model.ctf_type, std)
    return candidate_reconstruction_losses(images, x, ctf, model.ctf_type)


@partial(jax.jit, static_argnames=("use_tau",
                                   "apply_candidate_coverage",
                                   "apply_loss_whitening",
                                   "apply_geometry_priors", "apply_support",
                                   "train_heterogeneity", "return_metrics"),
         donate_argnums=(1,))
def train_step_reconsiren(graphdef, state, x, labels, md, key, tau=0.0001, use_tau=False, lambda_uniform=0.1,
                          apply_candidate_coverage=False, candidate_coverage_weight=0.0, apply_loss_whitening=False,
                          whiten_weight=0.0, whiten_filter=None, apply_geometry_priors=False, spacing_weight=0.0,
                          smoothness_weight=0.0, neighbor_indices=None, apply_support=False, support_center=None,
                          support_radius=0.0, support_weight=0.0, train_heterogeneity=True, return_metrics=False,
                          extra_blur=0.0):
    model, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)

    # Random keys
    key, coverage_key, swd_key, choice_key, distributions_key = jax.random.split(key, 5)

    def loss_fn(model, x):
        # Correct CTF in images for encoder if needed
        if model.ctf_type in ["apply", "squared"]:
            x_ctf_corrected = prepare_image_cryocrab(x, ctf)
            # x_ctf_corrected = prepare_image_wiener(x, ctf)
        else:
            x_ctf_corrected = x

        # Get candidate poses and shifts
        rotations, shifts = model.encoder_pose(x_ctf_corrected)

        # Decode consensus values and coords
        coords, values = model.delta_volume_decoder()

        std_eff = jnp.sqrt(jnp.square(model.get_std()) + jnp.square(extra_blur))

        # Refine current assignment (if provided)
        # rotations = jnp.matmul(rotations, current_rotations[:, None, :, :])
        rotations = jnp.matmul(current_rotations[:, None, :, :], rotations)  # TODO: The two options seem to work?
        shifts = current_shifts[:, None, :] + shifts

        # Random symmetry matrices
        random_indices = jax.random.choice(choice_key, jnp.arange(model.symmetry_matrices.shape[0]), shape=(rotations.shape[0],))
        rotations = jnp.matmul(jnp.transpose(model.symmetry_matrices[random_indices], (0, 2, 1))[:, None, :, :], rotations)

        rotations_eval, shifts_eval = rotations, shifts

        # Compute all candidate losses without gradient (as only the winner will need gradient - computed later on)
        candidate_losses = score_candidates(
            model, x, jax.lax.stop_gradient(values), jax.lax.stop_gradient(coords),
            jax.lax.stop_gradient(rotations_eval), jax.lax.stop_gradient(shifts_eval),
            ctf, std=jax.lax.stop_gradient(std_eff))

        # Pick a winner candidate
        if use_tau:
            responsibilities = jax.nn.softmax(-candidate_losses / tau, axis=1)
            min_indices = jax.random.categorical(
                key, jnp.log(jnp.maximum(responsibilities, 1e-12)), axis=-1)
        else:
            min_indices = jnp.argmin(candidate_losses, axis=1)

        batch_indices = jnp.arange(x.shape[0])

        # Compute losses for winner (with gradients)
        rotations_selected = rotations_eval[batch_indices, min_indices][:, None, ...]
        shifts_selected = shifts_eval[batch_indices, min_indices][:, None, ...]
        selected_images = model.phys_decoder(
            x, values, coords, model.xsize, rotations_selected, shifts_selected,
            ctf, model.ctf_type, std_eff)
        selected_losses, selected_predicted, selected_target = (
            candidate_reconstruction_losses(
                selected_images, x, ctf, model.ctf_type,
                return_prepared=True))
        recon_loss = selected_losses.mean()
        reconstruction_objective = recon_loss

        # Consider the case where whitening is requested
        if apply_loss_whitening:
            whitened_recon_loss = whitened_reconstruction_loss(
                selected_predicted, selected_target, whiten_filter).mean()
            effective_whiten_weight = jnp.clip(
                jnp.asarray(whiten_weight, dtype=recon_loss.dtype), 0.0, 1.0)
            reconstruction_objective = (
                (1.0 - effective_whiten_weight) * recon_loss
                + effective_whiten_weight * whitened_recon_loss)

        # For heterogeneity, always pick winner based on best loss
        min_indices_het = jnp.argmin(candidate_losses, axis=1)
        rotations_het = rotations_eval[jnp.arange(x.shape[0]), min_indices_het, :][:, None, ...]
        shifts_het = shifts_eval[jnp.arange(x.shape[0]), min_indices_het, :][:, None, ...]

        # Inialize some values for later
        latent_dim = model.encoder_het.mean_x.out_features
        latent = jnp.zeros((x.shape[0], latent_dim), dtype=x.dtype)
        recon_het_loss = jnp.asarray(0.0, dtype=x.dtype)
        variance_loss = jnp.asarray(0.0, dtype=x.dtype)
        covariance_loss = jnp.asarray(0.0, dtype=x.dtype)

        if train_heterogeneity:
            # Decode latent and heterogeneity images
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
                recon_het_loss = candidate_reconstruction_losses(
                    images_het[:, None, ...], x, ctf, model.ctf_type,
                    normalize_target=False).mean()
                variance_loss, covariance_loss, _ = latent_variance_covariance_loss(
                    latent, model.het_min_std)
            else:
                recon_het_loss, het_predicted, het_target = heterogeneity_reconstruction_loss(
                    images_het[:, None, ...], x, ctf, model.ctf_type,
                    model.het_loss_scales, model.het_loss_weights,
                    model.het_mask_radius, model.het_normalize_target,
                    return_prepared=True)

                # Consider the case where whitening is requested
                if apply_loss_whitening:
                    whitened_het_loss = whitened_reconstruction_loss(
                        het_predicted, het_target, whiten_filter).mean()
                    effective_whiten_weight = jnp.clip(
                        jnp.asarray(whiten_weight, dtype=recon_het_loss.dtype), 0.0, 1.0)
                    recon_het_loss = (
                        (1.0 - effective_whiten_weight) * recon_het_loss
                        + effective_whiten_weight * whitened_het_loss)
                variance_loss, covariance_loss, _ = latent_variance_covariance_loss(
                    latent, model.het_min_std,
                    bank=model.latent_memory_bank.get(),
                    bank_count=model.latent_bank_count.get_value())

        # Uniform angular distribution loss
        rotations_flat = rearrange(rotations, "b n w h -> (b n) w h")
        candidate_directions = rotations_flat[:, :, 2]
        winner_directions = rotations_selected[:, 0, :, 2]

        loss_swd = sliced_wasserstein_sphere(candidate_directions, rng=swd_key, n_projections=64)
        loss_uniform = lambda_uniform * loss_swd

        # Uniform coverage of the projection sphere loss
        coverage_loss = jnp.asarray(0.0, dtype=recon_loss.dtype)
        if apply_candidate_coverage:
            coverage_bins = jnp.asarray(
                generate_spherical_rotations(CANDIDATE_COVERAGE_BINS)[:, :, 2],
                dtype=candidate_directions.dtype)
            coverage_loss = candidate_coverage_loss(
                candidate_directions, coverage_bins, model.memory_bank.get(),
                model.candidate_bank_count.get_value(), coverage_key,
                kappa=CANDIDATE_COVERAGE_KAPPA,
                bank_samples=CANDIDATE_BANK_SAMPLES,
                bank_mix=CANDIDATE_BANK_MIX)

        loss = (0.5 * reconstruction_objective + loss_uniform
                + candidate_coverage_weight * coverage_loss)

        if train_heterogeneity:
            loss = (loss + 0.5 * recon_het_loss
                    + model.het_variance_weight * variance_loss
                    + model.het_covariance_weight * covariance_loss)

        if apply_geometry_priors:
            spacing_loss, smoothness_loss = geometry_prior_losses(
                coords, values, neighbor_indices, std_eff)
            loss = (loss + spacing_weight * spacing_loss
                    + smoothness_weight * smoothness_loss)

            if train_heterogeneity:
                het_spacing_loss, het_smoothness_loss = jax.vmap(
                    geometry_prior_losses, in_axes=(0, 0, None, None))(
                        coords_het[:, None, ...], values_het[:, None, ...],
                        neighbor_indices, std_eff)
                loss = (loss + spacing_weight * jnp.mean(het_spacing_loss)
                        + smoothness_weight * jnp.mean(het_smoothness_loss))

        if apply_support:
            loss = loss + support_weight * support_loss(
                coords, values, support_center, support_radius, std_eff)

            if train_heterogeneity:
                loss = loss + support_weight * jnp.mean(jax.vmap(
                    support_loss, in_axes=(0, 0, None, None, None))(
                        coords_het[:, None, ...], values_het[:, None, ...],
                        support_center, support_radius, std_eff))

        metrics = (recon_loss, recon_het_loss)
        return loss, (metrics, candidate_directions, winner_directions, latent)

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

    if model.ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, (params_pose, params_volume, params_het)), has_aux=True)
    (loss, (metrics, candidate_directions, winner_directions, latent)), grads_combined = grad_fn(model, x)

    grads_pose, grads_volume, grads_het = grads_combined.split(params_pose, params_volume, params_het)

    optimizer_pose.update(model, grads_pose)
    optimizer_volume.update(model, grads_volume)
    if train_heterogeneity:
        optimizer_het.update(model, grads_het)

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

    recon_loss = score_candidates(model, x, values, coords, rotations, shifts, ctf)

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
    from hax.programs import fit_volume

    from hax.cli import common_args as ca

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
    parser.add_argument("--candidate_coverage_weight", type=float, default=0.01,
                        help=f"Weight of the bank-aware spherical occupancy KL, which spreads the pose "
                             f"hypotheses over the projection sphere during the first "
                             f"{int(CANDIDATE_COVERAGE_EPOCHS)} epochs of the {bcolors.ITALIC}ab initio{bcolors.ENDC} search. "
                             f"Set it to 0 to disable the loss entirely.")
    parser.add_argument("--heterogeneity_profile", choices=("legacy", "anti_collapse"),
                        default="anti_collapse",
                        help="Heterogeneity training profile. anti_collapse enables staged residual "
                             "training with decoder centering, a masked/normalized multiscale loss and "
                             "bank-backed latent variance/covariance statistics; "
                             "legacy preserves the historical objective.")
    parser.add_argument("--lat_dim", type=int, default=8,
                        help="Dimension of the heterogeneity latent space.")
    parser.add_argument("--het_start_epoch", type=int, default=None,
                        help="First epoch that trains heterogeneity, providing a consensus-only warm-up. "
                             "Profile defaults: legacy heterogeneity=0, anti-collapse=5.")
    parser.add_argument("--consensus_parameterization", choices=("network", "direct"), default=None,
                        help="Consensus Gaussian delta parameterization. Default: direct.")
    parser.add_argument("--whiten_loss_weight", type=float, default=0.5,
                        help="Weight in [0,1] for the noise-whitened consensus reconstruction loss. The "
                             "dataset noise spectrum is estimated once from the particle solvent corners; "
                             "whitening equalises the per-frequency-shell SNR so high-resolution shells receive "
                             "real gradient instead of being drowned by the low-frequency power. 0 disables.")
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
                             "stacking). Active once the pose warm-up finishes. 0 disables.")
    parser.add_argument("--amplitude_smoothness_weight", type=float, default=0.01,
                        help="Weight of the kNN amplitude-smoothness prior (graph Laplacian on Gaussian masses) "
                             "so one iso-surface threshold traces the whole chain instead of beading. Active once "
                             "the pose warm-up finishes. 0 disables.")
    parser.add_argument("--no_point_recycling", action="store_true",
                        help="Disable periodic recycling of amplitude-dead Gaussians next to mass-carrying ones.")
    parser.add_argument("--recycle_every", type=int, default=5,
                        help="Epoch cadence for point recycling after the warm-up")
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
    if args.candidate_coverage_weight < 0.0:
        parser.error("--candidate_coverage_weight must be non-negative")
    if args.support_weight < 0.0:
        parser.error("--support_weight must be non-negative")
    if args.spacing_prior_weight < 0.0 or args.amplitude_smoothness_weight < 0.0:
        parser.error("--spacing_prior_weight and --amplitude_smoothness_weight must be non-negative")
    if args.recycle_every < 1:
        parser.error("--recycle_every must be at least 1")
    if args.num_gaussians is not None and args.num_gaussians < 1:
        parser.error("--num_gaussians must be positive")
    if not 0.0 <= args.whiten_loss_weight <= 1.0:
        parser.error("--whiten_loss_weight must be in [0,1]")
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

    # Estimate particle radius from the images
    extent_radius_px = None
    if args.vol is None and args.mode == "train" and not args.no_extent_estimation:
        n_probe = int(min(256, len(generator.md)))
        probe_indices = np.linspace(0, len(generator.md) - 1, n_probe).astype(int)
        probe = np.stack([np.squeeze(generator.md.getMetaDataImage(int(index)))
                          for index in probe_indices])
        extent_radius_px = estimate_particle_extent(probe)
        if extent_radius_px is not None:
            print(f"{bcolors.OKCYAN}Estimated particle radius from {n_probe} images: "
                  f"{extent_radius_px:.1f} px ({2.0 * extent_radius_px / xsize:.0%} of the box "
                  f"as diameter){bcolors.ENDC}")
        else:
            print(f"{bcolors.WARNING}Could not estimate the particle extent from the images; "
                  f"falling back to the fixed quarter-box initialization{bcolors.ENDC}")

    # Estimate number of gaussians needed (parameter or from particle radius)
    if args.num_gaussians is not None:
        num_gaussians = args.num_gaussians
    elif args.vol is None and extent_radius_px is not None:
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
        if extent_radius_px is not None:
            ball_radius = float(np.clip(extent_radius_px / (0.5 * xsize), 0.1, 0.9))
        else:
            ball_radius = 0.25
        coords = ball_radius * jnp.array(generate_sphere_points(num_gaussians) + np.random.normal(0, 0.1, (num_gaussians, 3)))
        values = jnp.full((num_gaussians,), 0.01)
        sigma = 1.1


    # # If exists, clean MMAP
    # if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

    # Mean point spacing
    implied_spacing = None
    if args.vol is None and args.mode == "train":
        radius_check = extent_radius_px if extent_radius_px is not None else 0.25 * xsize
        implied_spacing = ((4.0 / 3.0) * np.pi * radius_check ** 3 / num_gaussians) ** (1.0 / 3.0)

    sigma_min = implied_spacing / 2.8 if implied_spacing is not None else 0.0
    if implied_spacing is not None:
        sigma_now = float(np.mean(np.asarray(sigma)))
        floor_str = f" (min {sigma_min:.2f})" if sigma_min > 0.0 else ""
        print(f"{bcolors.OKCYAN}Cloud geometry: {num_gaussians} points, implied spacing "
              f"{implied_spacing:.2f} px, splat width {sigma_now:.2f} px{floor_str}{bcolors.ENDC}")

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key, choice_key = jax.random.split(rng, 3)

    # Prepare network (ReconSIREN)
    reconsiren = ReconSIREN(coords, values, xsize, args.sr, ctf_type=args.ctf_type, symmetry_group=args.symmetry_group,
                            refine_current_assignment=args.refine_current_assignment, lat_dim=args.lat_dim, sigma=sigma,
                            bank_size=10000, learn_delta_volume=not args.do_not_learn_volume,
                            num_components=args.num_components,
                            use_anchor_rotations=not args.do_not_use_anchor_rotations,
                            consensus_parameterization=args.consensus_parameterization,
                            heterogeneity_profile=args.heterogeneity_profile,
                            het_start_epoch=args.het_start_epoch,
                            sigma_min=sigma_min,
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
            return model.delta_volume_decoder.decode_volume(sigma=model.get_std())

        # Consensus point cloud for KNN
        @nnx.jit
        def decode_cloud(model):
            return model.delta_volume_decoder()

        # Physical + equalized intermediate maps
        @nnx.jit
        def decode_volume_pair(model, values_pair):
            coords, _ = model.delta_volume_decoder()
            return model.delta_volume_decoder.decode_volume(
                coords_values=(coords, values_pair),
                sigma=model.get_std())

        # Decode volume
        @nnx.jit
        def decode_het_volume(model, x):
            return model.decode_het_volume(x)

        def write_intermediate_volumes(volumes, out_dir):
            """Write the intermediate map(s)"""
            volumes = np.asarray(volumes)
            ImageHandler().write(volumes[0],
                                 os.path.join(out_dir, "reconsiren_map_intermediate.mrc"),
                                 overwrite=True)
            if volumes.shape[0] > 1:
                ImageHandler().write(volumes[1],
                                     os.path.join(out_dir, "reconsiren_map_equalized_intermediate.mrc"),
                                     overwrite=True)

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
        volume_is_direct = reconsiren.delta_volume_decoder.parameterization == "direct"
        volume_lr = VOLUME_LEARNING_RATE if volume_is_direct else 1e-4
        coords_lr = volume_lr if volume_is_direct else None
        amplitude_lr = AMPLITUDE_LEARNING_RATE if volume_is_direct else None

        optimizer_pose = nnx.Optimizer(reconsiren,  optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(args.learning_rate, eps=1e-6)), wrt=params_pose)
        optimizer_volume = nnx.Optimizer(
            reconsiren,
            volume_optimizer_transform(reconsiren.delta_volume_decoder.parameterization,
                                       volume_lr, coords_lr=coords_lr,
                                       amplitude_lr=amplitude_lr),
            wrt=params_volume)
        optimizer_het = nnx.Optimizer(reconsiren, optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(learning_rate=HET_LEARNING_RATE, eps=1e-6)), wrt=params_het)
        het_start_epoch = reconsiren.het_start_epoch
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
        intermediate_equalized_enabled = not args.do_not_learn_volume
        graphdef, state = nnx.split((reconsiren, optimizer_pose, optimizer_volume, optimizer_het))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"),
                                                                                        optimizer_pose, optimizer_volume, optimizer_het)
            print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
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
                    # Gaussian recycling
                    if recycling_enabled and total_steps > 1500 and (
                            epoch_index % args.recycle_every == 0):
                        state, n_recycled = recycle_dead_points_reconsiren(
                            graphdef, state, rng, RECYCLE_DEAD_FRACTION)
                        rng, _ = jax.random.split(rng)
                        n_recycled = int(n_recycled)
                        if n_recycled:
                            print(f"\n{bcolors.OKCYAN}Recycled {n_recycled} dead Gaussians "
                                  f"onto the structure{bcolors.ENDC}")

                    # KNN update
                    if geometry_priors_enabled or support_enabled:
                        reconsiren_cloud, _, _, _ = nnx.merge(graphdef, state)
                        cloud_coords, cloud_values = decode_cloud(reconsiren_cloud)
                        cloud_coords = np.asarray(cloud_coords[0], np.float32)
                        cloud_values = np.asarray(cloud_values[0], np.float32)
                        if geometry_priors_enabled:
                            n_neighbors = int(min(KNN_NEIGHBORS, cloud_coords.shape[0] - 1))
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
                                support_radius_value = float(max(
                                    1.15 * distances[order][quantile_index],
                                    4.0 * float(np.mean(np.asarray(sigma)))))
                                support_center_value = jnp.asarray(center, dtype=jnp.float32)

                    total_loss = 0
                    total_recon_loss = 0
                    total_recon_het_loss = 0
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
                            volume = None
                            if intermediate_equalized_enabled:
                                _, cloud_masses = decode_cloud(reconsiren)
                                equalized_masses = equalize_masses(
                                    cloud_masses[0], EQUALIZED_MAP_GAMMA)
                                if equalized_masses is not None:
                                    values_pair = jnp.stack(
                                        [jnp.asarray(np.asarray(cloud_masses[0], np.float32)),
                                         jnp.asarray(equalized_masses)], axis=0)
                                    volume = decode_volume_pair(reconsiren, values_pair)
                            if volume is None:
                                volume = decode_volume(reconsiren)
                            middle_slize = int(np.round(0.5 * volume.shape[-1]))
                            slice_xy, slice_xz, slice_yz = (min_max_scale(volume[0, middle_slize, :, :]),
                                                            min_max_scale(volume[0, :, middle_slize, :]),
                                                            min_max_scale(volume[0, :, :, middle_slize]))
                            slices = np.stack([slice_xy, slice_xz, slice_yz], axis=0)[..., None]

                        logger.submit(write_intermediate_volumes, volume, args.output_path)
                        logger.submit(writer.add_images, "Predicted volume (slices)", slices,
                                      dataformats="NHWC", global_step=i)

                    if logger.should("landscape", i):
                        with logger.section():
                            reconsiren, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)

                            # Plot winners and candidates distributions
                            angular_banks = (
                                ("Angular distribution winners",
                                 reconsiren.winner_memory_bank.get(),
                                 reconsiren.winner_bank_count.get_value()),
                                ("Angular distribution candidates",
                                 reconsiren.memory_bank.get(),
                                 reconsiren.candidate_bank_count.get_value()),
                            )
                            for title, bank, count in angular_banks:
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
                                writer.add_figure(title, fig, global_step=i)

                            # Predict some heterogeneous volumes
                            n_latent_steps = int(min(steps_per_epoch,
                                                     np.ceil(LATENTS_FOR_CLUSTERING / args.batch_size)))
                            n_latent_steps = n_latent_steps if epoch_index >= het_start_epoch else 0
                            latents = []
                            decoded_centers = []
                            graphdef_aux, state_aux = nnx.split(reconsiren)
                            for _ in range(n_latent_steps):
                                (x_latent, labels_latent) = next(iter_data_loader_train)
                                _, _, latent = predict_angular_assignment_step_reconsiren(graphdef_aux, state_aux,
                                                                                          x_latent, labels_latent,
                                                                                          md_columns, rng)
                                latents.append(np.array(latent))
                            if latents:
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
                uniform_weight = 0.1

                # Consider whitening only once the winners settle
                whiten_weight_step = (args.whiten_loss_weight
                                      if whiten_filter is not None and not use_tau
                                      else 0.0)
                apply_loss_whitening = whiten_weight_step > 0.0

                # Consider connectivity priors only once the winners settle
                apply_geometry_priors_step = (
                    geometry_priors_enabled and knn_indices is not None
                    and not use_tau)
                apply_support_step = (support_enabled
                                      and support_radius_value is not None)

                coverage_steps = CANDIDATE_COVERAGE_EPOCHS * steps_per_epoch
                apply_candidate_coverage = (
                    total_steps < coverage_steps
                    and args.candidate_coverage_weight > 0.0)
                candidate_coverage_weight = (
                    args.candidate_coverage_weight
                    if apply_candidate_coverage else 0.0)
                train_heterogeneity = epoch_index >= het_start_epoch
                loss, metrics, state, rng = train_step_reconsiren(
                    graphdef, state, x, labels, md_columns, rng,
                    lambda_uniform=uniform_weight, tau=tau, use_tau=use_tau,
                    apply_candidate_coverage=apply_candidate_coverage,
                    candidate_coverage_weight=candidate_coverage_weight,
                    apply_loss_whitening=apply_loss_whitening,
                    whiten_weight=whiten_weight_step,
                    whiten_filter=whiten_filter,
                    apply_geometry_priors=apply_geometry_priors_step,
                    spacing_weight=args.spacing_prior_weight,
                    smoothness_weight=args.amplitude_smoothness_weight,
                    neighbor_indices=knn_indices if apply_geometry_priors_step else None,
                    apply_support=apply_support_step,
                    support_center=support_center_value if apply_support_step else None,
                    support_radius=support_radius_value if apply_support_step else 0.0,
                    support_weight=args.support_weight,
                    train_heterogeneity=train_heterogeneity,
                    return_metrics=True,
                    extra_blur=CLOUD_BLUR_WARMUP if use_tau else 0.0)
                recon_loss, recon_het_loss = metrics
                total_loss += loss
                total_recon_loss += recon_loss
                total_recon_het_loss += recon_het_loss

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

                    # Progress bar update  (TQDM)
                    stage = "joint" if train_heterogeneity else "consensus"
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

        # Jitted functions for volume prediction
        decode_volume = jax.jit(lambda: reconsiren.delta_volume_decoder.decode_volume(
            sigma=reconsiren.get_std()))
        decode_het_volume = jax.jit(reconsiren.decode_het_volume)
        decode_het_cloud = jax.jit(reconsiren.decode_het_cloud)

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

        sharpened = sharpen_gaussian_envelope(jnp.asarray(decoded_volume[0]),
                                              reconsiren.get_std(),
                                              reg=SHARPENED_MAP_REG)
        ImageHandler().write(np.array(sharpened),
                             os.path.join(args.output_path, "reconsiren_map_sharpened.mrc"),
                             overwrite=True)

        cloud_coords, cloud_values = reconsiren.delta_volume_decoder()
        equalized_masses = equalize_masses(cloud_values[0], EQUALIZED_MAP_GAMMA)
        if equalized_masses is not None:
            equalized = reconsiren.delta_volume_decoder.decode_volume(
                coords_values=(cloud_coords, jnp.asarray(equalized_masses)[None, ...]),
                sigma=reconsiren.get_std())
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

            sharpened = sharpen_gaussian_envelope(jnp.asarray(decoded[0]),
                                                  reconsiren.get_std(),
                                                  reg=SHARPENED_MAP_REG)
            ImageHandler().write(np.array(sharpened),
                                 os.path.join(args.output_path,
                                              f"reconsiren_hetmap_{idx:02d}_sharpened.mrc"),
                                 overwrite=True)

            het_coords, het_values = decode_het_cloud(center[None, ...])
            equalized_masses = equalize_masses(het_values[0], EQUALIZED_MAP_GAMMA)
            if equalized_masses is not None:
                equalized = splat_cloud_volumes(
                    het_coords + reconsiren.delta_het_decoder.factor,
                    jnp.asarray(equalized_masses)[None, ...],
                    reconsiren.delta_het_decoder.volume_size,
                    sigma=reconsiren.get_std())
                ImageHandler().write(np.array(equalized[0]),
                                     os.path.join(args.output_path,
                                                  f"reconsiren_hetmap_{idx:02d}_equalized.mrc"),
                                     overwrite=True)
            idx += 1

    # If exists, clean MMAP
    # if not args.load_images_to_ram and os.path.isdir(generator.mmap_output_dir):
    #     shutil.rmtree(generator.mmap_output_dir)
