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
    def __init__(self, input_dim, pyramid_levels=4, num_components=18, refine_current_assignment=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 64  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 3))
        self.pyramid_levels = pyramid_levels
        self.num_components = num_components
        self.refine_current_assignment = refine_current_assignment

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

        # Layers to 9D rotation
        self.ensemble_6d_heads = PoseHeadEnsemble(num_members=num_components, is_refine=refine_current_assignment, rngs=rngs)

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
        hidden_layers_conv = [
            Linear(self.input_dim * self.input_dim, self.input_conv_dim * self.input_conv_dim, rngs=rngs,
                   dtype=jnp.bfloat16)]
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
        x = rearrange(x, 'b h w c -> b (h w c)')

        x = nnx.leaky_relu(self.hidden_layers_conv[0](x))  # or nnx.relu

        x = rearrange(x, 'b (h w c) -> b h w c', h=self.input_conv_dim, w=self.input_conv_dim, c=1)

        for layer in self.hidden_layers_conv[1:]:
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

        bamp0 = values * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp1 = values * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp2 = values * (bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp3 = values * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp = jnp.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposi = jnp.concat([bposi, bposi + jnp.array((1, 0)), bposi + jnp.array((1, 1)), bposi + jnp.array((0, 1))], axis=1)

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        images = jax.vmap(scatter_img)(images, bposi, bamp)

        # Gaussian filter (needed by forward interpolation)
        if filter:
            images = dm_pix.gaussian_blur(images[..., None], std, kernel_size=9)[..., 0]

        # Apply CTF
        if ctf_type in ["apply", "wiener", "squared"]:
            ctf = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf = rearrange(ctf, "b n w h -> (b n) w h")
            images = ctfFilter(images, ctf, pad_factor=2)

        images = rearrange(images, "(b n) w h -> b n w h", b=rotations.shape[0], n=rotations.shape[1])

        return images

class ReconSIREN(nnx.Module):

    @save_config
    def __init__(self, coords, values, xsize, sr, bank_size=1024, ctf_type="apply", lat_dim=8, sigma=1.0,
                 symmetry_group="c1", refine_current_assignment=False, learn_delta_volume=True, *, rngs: nnx.Rngs):
        super(ReconSIREN, self).__init__()
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.symmetry_matrices = symmetry_matrices(symmetry_group)
        self.refine_current_assignment = refine_current_assignment
        self.learn_delta_volume = learn_delta_volume
        self.encoder_pose = EncoderPose(self.xsize, refine_current_assignment=refine_current_assignment, rngs=rngs)
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
                                             self.get_std(), ctf_type)

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


@partial(jax.jit, static_argnames=("use_tau",))
def train_step_reconsiren(graphdef, state, x, labels, md, key, tau=0.0001, use_tau=False, lambda_uniform=0.1):
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
        coords_het, values_het = model.delta_het_decoder(sample)

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

        # Broadcast input images to right size
        x_loss = jnp.broadcast_to(x_loss_nb[:, None, ...], (x_loss_nb.shape[0], images_corrected.shape[1], x_loss_nb.shape[1], x_loss_nb.shape[2]))

        x_flat = rearrange(x_loss, "b n w h -> (b n) w h")
        images_corrected_flat = rearrange(images_corrected_loss, "b n w h -> (b n) w h")
        x_flat = standard_normalization(x_flat)

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
                                        jax.lax.stop_gradient(shifts_het), ctf, model.ctf_type, model.get_std())[:, 0, ...]
        images_het_loss = images_het[..., 0] if images_het.shape[-1] == 1 else images_het
        if model.ctf_type == "wiener":
            images_het_loss = wiener2DFilter(images_het_loss, ctf, pad_factor=2)
        elif model.ctf_type == "squared":
            images_het_loss = ctfFilter(images_het_loss, ctf, pad_factor=2)
        # x_loss_nb = standard_normalization(x_loss_nb)

        # Bandpass (TODO: Make optional to membran proteins only)
        # x_loss_nb = bandpass_filter(x_loss_nb, pixel_size_A=model.sr, highpass_A=50.)
        # images_het_loss = bandpass_filter(images_het_loss, pixel_size_A=model.sr, highpass_A=50.)

        recon_het_loss = dm_pix.mse(images_het_loss[..., None], x_loss_nb[..., None]).mean()

        # Index losses and rotations based on extracted indices
        recon_loss = recon_loss[jnp.arange(images_corrected.shape[0]), min_indices].mean()
        recon_loss_all = 0.5 * (recon_loss + recon_het_loss)
        
        # Viewing directions from rotations
        rotations = rearrange(rotations, "b n w h -> (b n) w h")
        directions = rotations[:, :, 2]

        # L1 based denoising
        l1_loss = jnp.mean(jnp.abs(values)) + jnp.mean(jnp.abs(values_het))

        # KL loss VAE
        kl_loss = -0.5 * jnp.sum(1. + 2. * logstd - jnp.square(jnp.exp(logstd)) - jnp.square(latent))

        # Decoupling (TODO: In the future this will be for missing angles like TF implementation)

        # Uniform angular distribution loss
        loss_swd = sliced_wasserstein_sphere(directions, rng=key, n_projections=64)
        loss_repulsion = repulsion_loss(directions, s=2.)
        loss_uniform = lambda_uniform * loss_swd + 0.0 * loss_repulsion

        loss = (recon_loss_all + 0.0 * l1_loss + 0.00000 * kl_loss + 1.0 * loss_uniform)
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
    from hax.metrics import JaxSummaryWriter
    from hax.networks import VolumeAdjustment, train_step_volume_adjustment
    from hax.programs import fit_volume, adjust_weights_to_images
    from hax.programs.gaussian_volume_fitting import get_cosine_reg_strength

    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, type=str,
                        help="Xmipp/Relion metadata file with the images (+ alignments / CTF) to be analyzed")
    parser.add_argument("--vol", required=False, type=str,
                        help="If provided, the neural network will start from this volume when assigning the angles and shifts to the images.")
    parser.add_argument("--mask", required=False, type=str,
                        help=f"ReconSIREN reconstruction mask (the mask provided must be binary - "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: since this is a reconstruction mask, it should be defined such that it covers the "
                             f"volume were the motions of interest are expected to happen)")
    parser.add_argument("--num_gaussians", required=False, type=int, default=10000,
                        help="Before training the network, HetSIREN will try to fit a set of Gaussians in the reference volume to recreate it. "
                            "The default criterium is to automatically determine the number of Gaussians neede to reproduce the reference volume "
                            "with high-fidelity. However, if you prefer to fix the number of Gaussians in advance based on your own criterium (e.g., "
                            "the number of residues in your protein), you can set this parameter. When set, the HetSIREN will fit this fixed number of Gaussians "
                            "so that the reproduce the reference volume as well as possible.")
    parser.add_argument("--load_images_to_ram", action='store_true',
                        help=f"If provided, images will be loaded to RAM. This is recommended if you want the best performance and your dataset fits in your RAM memory. If this flag is not provided, "
                             f"images will be memory mapped. When this happens, the program will trade disk space for performance. Thus, during the execution additional disk space will be used and the performance "
                             f"will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal once the execution has finished.")
    parser.add_argument("--sr", required=True, type=float,
                        help="Sampling rate of the images/volume")
    parser.add_argument("--do_not_learn_volume", action="store_true",
                        help="When this parameter is provided, ReconSIREN will just learn an angular assignment with shifts without learning any map. This is usually useful when a reference volume with "
                             "high resolution is provided (e.g. coming from an atomic model) and no refinement of the map is needed.")
    parser.add_argument("--refine_current_assignment", action="store_true",
                        help=f"If your input metadata has already and angular assignment and shifts, you can provide this option to refine those angles instead of finding an {bcolors.ITALIC}ab initio{bcolors.ENDC} "
                             f"alignment.")
    parser.add_argument("--symmetry_group", type=str, default="c1",
                        help=f"If your protein has any kind of symmetry, you may pass it here so that it is considered while learning the angular assignment and the volume ({bcolors.WARNING}NOTE{bcolors.ENDC}: "
                             f"only {bcolors.ITALIC}c*{bcolors.ENDC} and {bcolors.ITALIC}d*{bcolors.ENDC} symmetry groups are currently supported - the parameter is lower case sensitive - even if symmetry is provided, "
                             f"the network will learn a {bcolors.ITALIC}symmetry broken{bcolors.ENDC} set of angles in c1. Therefore, the angles can be directly used in a reconstruction/refinement.)")
    parser.add_argument("--ctf_type", required=True, type=str, choices=["None", "apply", "wiener", "precorrect"],
                        help="Determines whether to consider the CTF and, in case it is considered, whether it will be applied to the projections (apply) or used to correct the metadata images (wiener - precorrect)")
    parser.add_argument("--mode", required=True, type=str, choices=["train", "predict", "send_to_pickle"],
                        help=f"{bcolors.BOLD}train{bcolors.ENDC}: train a neural network from scratch or from a previous execution if reload is provided\n"
                             f"{bcolors.BOLD}predict{bcolors.ENDC}: predict the latent vectors from the input images ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case)")
    parser.add_argument("--epochs", required=False, type=int, default=50,
                        help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset of images - set to default to 50 - "
                             "as a rule of thumb, consider 50 to 100 epochs enough for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")
    parser.add_argument("--batch_size", required=False, type=int, default=8,
                        help="Determines how many images will be load in the GPU at any moment during training (set by default to 8 - "
                             f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                             f"to monitor and/or measure memory usage and adjust this value - keep also in mind that bigger batch sizes might be less precise when looking for very local motions")
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-4,
                        help=f"The learning rate ({bcolors.ITALIC}lr{bcolors.ENDC}) sets the speed of learning. Think of the model as trying to find the lowest point in a valley; the {bcolors.ITALIC}lr{bcolors.ENDC} "
                             f"is the size of the step it takes on each attempt. A large {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}0.01{bcolors.ENDC}) is like taking huge leaps — it's fast but can be unstable, "
                             f"overshoot the lowest point, or cause {bcolors.ITALIC}NAN{bcolors.ENDC} errors. A small {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}1e-6{bcolors.ENDC}) is like taking tiny "
                             f"shuffles — it's stable but very slow and might get stuck before reaching the bottom. A good default is often {bcolors.ITALIC}0.0001{bcolors.ENDC}. If training fails or errors explode, "
                             f"try making the {bcolors.ITALIC}lr{bcolors.ENDC} 10 times smaller (e.g., {bcolors.ITALIC}0.001{bcolors.ENDC} --> {bcolors.ITALIC}0.0001{bcolors.ENDC}).")
    parser.add_argument("--dataset_split_fraction", required=False, type=list_of_floats, default=[0.8, 0.2],
                        help=f"Here you can provide the fractions to split your data automatically into a training and a validation subset following the format: {bcolors.ITALIC}training_fraction{bcolors.ENDC},"
                             f"{bcolors.ITALIC}validation_fraction{bcolors.ENDC}. While the training subset will be used to train/update the network parameters, the validation subset will only be used to evaluate the "
                             f"accuracy of the network when faced with new data. Therefore, the validation subset will never be used to update the networks parameters. {bcolors.WARNING}NOTE{bcolors.ENDC}: the sum of "
                             f"{bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} must be equal to one.")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, new metadata...)")
    parser.add_argument("--reload", required=False, type=str,
                        help=f"Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data).")
    parser.add_argument("--ssd_scratch_folder", required=False, type=str,
                        help=f"When the parameter {bcolors.UNDERLINE}load_images_to_ram{bcolors.ENDC} is not provided, we strongly recommend to provide here a path to a folder in a SSD disk to read faster the data. If not given, the data will be loaded from "
                             f"the default disk.")
    args, _ = parser.parse_known_args()

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
    if sum(args.dataset_split_fraction) != 1:
        raise ValueError(
            f"The sum of {bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} is not equal one. Please, update the values "
            f"to fulfill this requirement.")

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
                            bank_size=10000, learn_delta_volume=not args.do_not_learn_volume, rngs=nnx.Rngs(model_key))

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

                    if i > 0 and i % 1 == 0:
                        pbar.set_postfix_str(f"{bcolors.WARNING}Generating intermediate results...{bcolors.ENDC}")

                        # Example of predicted data for Tensorboard
                        reconsiren, optimizer_pose, optimizer_volume, optimizer_het = nnx.merge(graphdef, state)
                        volume = decode_volume(reconsiren)
                        middle_slize = int(np.round(0.5 * volume.shape[-1]))
                        ImageHandler().write(np.array(volume),
                                             os.path.join(args.output_path, "reconsiren_map_intermediate.mrc"),
                                             overwrite=True)
                        slice_xy, slice_xz, slice_yz = (min_max_scale(volume[0, middle_slize, :, :]),
                                                        min_max_scale(volume[0, :, middle_slize, :]),
                                                        min_max_scale(volume[0, :, :, middle_slize]))
                        slices = np.stack([slice_xy, slice_xz, slice_yz], axis=0)[..., None]
                        writer.add_images("Predicted volume (slices)", slices, dataformats="NHWC", global_step=i)

                        # Plot angular distribution
                        directions = np.array(reconsiren.memory_bank.get())
                        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
                        beta = jnp.arccos(jnp.clip(z, -1.0, 1.0))
                        alpha = jnp.arctan2(y, x)
                        euler_angles = jnp.stack([alpha, beta], axis=-1)
                        fig, _ = plot_angular_distribution(euler_angles)
                        writer.add_figure("Angular distribution density", fig, global_step=i)

                        # Predict some heterogeneous volumes
                        latents = []
                        graphdef_aux, state_aux = nnx.split(reconsiren)
                        for _ in range(steps_per_epoch):
                            (x, labels) = next(iter_data_loader_train)
                            _, _, latent = predict_angular_assignment_step_reconsiren(graphdef_aux, state_aux, x,
                                                                                      labels, md_columns, rng)
                            latents.append(np.array(latent))
                        latents = np.concatenate(latents, axis=0)
                        kmeans = KMeans(n_clusters=10).fit(latents)
                        centers = kmeans.cluster_centers_
                        idx = 1
                        for center in centers:
                            decoded = decode_het_volume(reconsiren, center[None, ...])
                            ImageHandler().write(np.array(decoded),
                                                 os.path.join(args.output_path, f"reconsiren_hetmap_{idx:02d}.mrc"),
                                                 overwrite=True)
                            idx += 1

                        # Save checkpoint model
                        NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"), epoch=i)

                    i += 1

                if total_steps <= 1500:
                    tau = 1e-3
                    use_tau = True
                else:
                    tau = 0.0
                    use_tau = False
                loss, recon_loss, state, rng = train_step_reconsiren(graphdef, state, x, labels, md_columns, rng, lambda_uniform=0.1, tau=tau, use_tau=use_tau)
                total_loss += loss
                total_recon_loss += recon_loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | recon_loss={total_recon_loss / step:.5f}")

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * steps_per_epoch)) == 0:
                    writer.add_scalar('Training loss (ReconSIREN)',
                                      total_loss / step,
                                      i * steps_per_epoch + step)

                    writer.add_scalars('Reconstruction loss (ReconSIREN)',
                                       {"train": total_recon_loss / step},
                                       i * steps_per_epoch + step)

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

        # Save model
        NeuralNetworkCheckpointer.save(reconsiren, os.path.join(args.output_path, "ReconSIREN"))

        # Remove checkpoint
        shutil.rmtree(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"))

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

        md_pred.write(os.path.join(args.output_path, "predicted_pose_shifts" + os.path.splitext(args.md)[1]))

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
    if not args.load_images_to_ram and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap_grain")):
        shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap_grain"))
