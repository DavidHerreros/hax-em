#!/usr/bin/env python


import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import optax
import dm_pix
from functools import partial
import itertools
import numpy as np

from einops import rearrange

from sklearn.cluster import KMeans

from hax.utils import *
from hax.layers import *
from hax.programs import splat_weights_trilinear, FastVariableBlur3D


def wrap_zyz_angles(angles):
    """
    Wraps ZYZ Euler angles to canonical ranges:
    alpha: [0, 2pi), beta: [0, pi], gamma: [0, 2pi)
    """
    alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]

    # 1. Wrap beta to [0, 2pi) first
    beta_ext = jnp.remainder(beta, 2 * jnp.pi)

    # 2. If beta is in (pi, 2pi), we reflect it:
    # beta' = 2pi - beta, and flip alpha/gamma by pi
    mask = beta_ext > jnp.pi

    beta_wrapped = jnp.where(mask, 2 * jnp.pi - beta_ext, beta_ext)
    alpha_wrapped = jnp.where(mask, alpha + jnp.pi, alpha)
    gamma_wrapped = jnp.where(mask, gamma + jnp.pi, gamma)

    # 3. Finally, wrap alpha and gamma to [0, 2pi)
    alpha_final = jnp.remainder(alpha_wrapped, 2 * jnp.pi)
    gamma_final = jnp.remainder(gamma_wrapped, 2 * jnp.pi)

    return jnp.stack([alpha_final, beta_wrapped, gamma_final], axis=-1)


def sample_uniform_zyz(key, shape):
    k1, k2, k3 = jax.random.split(key, 3)

    # Alpha and Gamma are uniform [0, 2pi]
    alpha = jax.random.uniform(k1, shape, minval=0., maxval=2 * jnp.pi)
    gamma = jax.random.uniform(k2, shape, minval=0., maxval=2 * jnp.pi)

    # Beta must be sampled such that cos(beta) is uniform in [-1, 1]
    cos_beta = jax.random.uniform(k3, shape, minval=-1.0, maxval=1.0)
    beta = jnp.arccos(cos_beta)

    return jnp.stack([alpha, beta, gamma], axis=-1)


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


class EncoderPose(nnx.Module):
    def __init__(self, input_dim, pyramid_levels=4, num_components=18, refine_current_assignment=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 64  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 3))
        self.pyramid_levels = pyramid_levels
        self.num_components = num_components
        self.refine_current_assignment = refine_current_assignment

        # Anchor rotations
        self.anchor_rotations = jnp.array(generate_spherical_rotations(num_components))

        # Hidden layers
        hidden_layers = [Linear(self.input_conv_dim * self.input_conv_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(3):
            hidden_layers.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers = nnx.List(hidden_layers)

        # Layers to 9D rotation
        hidden_6d_rotation = [Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        hidden_6d_rotation.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        hidden_6d_rotation.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        if refine_current_assignment:
            hidden_6d_rotation.append(Linear(1024, self.num_components * 6, kernel_init=nnx.initializers.zeros_init(), rngs=rngs))
        else:
            # identity_6d = jnp.array([1., 0., 0., 0., 1., 0.])[None, ...]
            # identity_6d = jnp.tile(identity_6d, (1, self.num_components))
            kernel_init = jax.nn.initializers.normal(stddev=1e-4)
            # bias_init = lambda key, shape, dtype: identity_6d
            # kernel_init = nnx.initializers.zeros_init()
            # bias_init = nnx.initializers.normal(stddev=1.0)
            bias_init = nnx.initializers.zeros_init()
            hidden_6d_rotation.append(Linear(1024, self.num_components * 6, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs))
        self.hidden_6d_rotation = nnx.List(hidden_6d_rotation)

        # Layers to shifts
        hidden_shifts = [Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        hidden_shifts.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        hidden_shifts.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        if refine_current_assignment:
            hidden_shifts.append(Linear(1024, 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init()))
        else:
            hidden_shifts.append(Linear(1024, 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init()))
        self.hidden_shifts = nnx.List(hidden_shifts)

        # Probability head
        self.prob_head = nnx.Linear(in_features=1024, out_features=num_components, kernel_init=nnx.initializers.zeros_init(),
                                    bias_init=nnx.initializers.constant(1. / num_components), rngs=rngs)

    def __call__(self, x, return_diversity_loss=False, is_training=False, key=None):
        # Resize images
        x = jax.image.resize(x, (x.shape[0], self.input_conv_dim, self.input_conv_dim, 1), method="bilinear")

        # Hidden layers
        x = rearrange(x, 'b h w c -> b (h w c)')
        for layer in self.hidden_layers:
            if layer.in_features == layer.out_features:
                x = nnx.gelu(x + layer(x))
            else:
                x = nnx.gelu(layer(x))

        # First output: rotation matrices
        rotations_6d = nnx.gelu(self.hidden_6d_rotation[0](x))
        for layer in self.hidden_6d_rotation[1:-1]:
            rotations_6d = nnx.gelu(rotations_6d + layer(rotations_6d))
        rotations_6d = self.hidden_6d_rotation[-1](rotations_6d)
        # rotation_9d = self.hidden_6d_rotation[-1](x)  # Keep for reference: before only one layer needed

        rotations_6d = rotations_6d.reshape(x.shape[0] * self.num_components, 6)
        if self.refine_current_assignment:
            identity_6d = jnp.array([1., 0., 0., 0., 1., 0.])[None, ...].repeat(rotations_6d.shape[0], axis=0)
            rotations_6d = identity_6d + rotations_6d

        # Add continuous exploration noise ONLY during training
        # if is_training:
        #     # stddev controls the exploration radius around the anchors
        #     noise = jax.random.normal(key, rotations_6d.shape) * 0.0001
        #     rotations_6d = rotations_6d + noise

        a1, a2 = jnp.split(rotations_6d, 2, axis=-1)
        b1 = a1 / jnp.clip(jnp.linalg.norm(a1, axis=-1, keepdims=True), a_min=1e-6)
        a2_ortho = a2 - jnp.sum(a2 * b1, axis=-1, keepdims=True) * b1
        b2 = a2_ortho / jnp.clip(jnp.linalg.norm(a2_ortho, axis=-1, keepdims=True), a_min=1e-6)
        b3 = jnp.cross(b1, b2, axis=-1)
        rotations = jnp.stack([b1, b2, b3], axis=-1)
        rotations = rotations.reshape(x.shape[0], self.num_components, 3, 3)
        rotations = jnp.einsum('bnhk,nkw->bnhw', rotations, self.anchor_rotations)

        # Third output: in plane shifts
        in_plane_shifts = nnx.gelu(self.hidden_shifts[0](x))
        for layer in self.hidden_shifts[1:-1]:
            in_plane_shifts = nnx.gelu(in_plane_shifts + layer(in_plane_shifts))
        # in_plane_shifts = 0.5 * self.input_dim * self.hidden_shifts[-1](in_plane_shifts)
        in_plane_shifts = self.hidden_shifts[-1](in_plane_shifts)
        # if self.refine_current_assignment:
        #     in_plane_shifts = self.alpha_shifts * in_plane_shifts

        # Broadcast shifts to euler angles shape
        in_plane_shifts = jnp.broadcast_to(in_plane_shifts[:, None, :], (in_plane_shifts.shape[0], self.num_components, 2))

        # Probability
        logits = nnx.sigmoid(self.prob_head(x))

        if return_diversity_loss:
            directions = rotations @ jnp.array([0, 0, 1])
            pairwise_dots = directions @ directions.transpose(0, 2, 1)
            off_diagonal_dots = pairwise_dots * (1.0 - jnp.eye(rotations.shape[1], dtype=pairwise_dots.dtype))
            diversity_loss = jnp.mean(jnp.sum(jnp.square(off_diagonal_dots), axis=(-2, -1)))
            return rotations, in_plane_shifts, logits, diversity_loss
        else:
            return rotations, in_plane_shifts, logits

class EncoderHet(nnx.Module):
    def __init__(self, input_dim, lat_dim=8, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 64  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 3))

        # # Hidden layers
        # hidden_layers_conv = [Conv(1, 128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16)]
        # hidden_layers_conv.append(Conv(128, 128, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(nnx.BatchNorm(num_features=128, momentum=0.9, epsilon=1e-5, dtype=jnp.bfloat16, rngs=rngs))
        #
        # hidden_layers_conv.append(Conv(128, 256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(256, 256, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(nnx.BatchNorm(num_features=256, momentum=0.9, epsilon=1e-5, dtype=jnp.bfloat16, rngs=rngs))
        #
        # hidden_layers_conv.append(Conv(256, 512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(512, 512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(512, 512, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(nnx.BatchNorm(num_features=512, momentum=0.9, epsilon=1e-5, dtype=jnp.bfloat16, rngs=rngs))
        #
        # hidden_layers_conv.append(Conv(512, 1024, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(1024, 1024, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(1024, 1024, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(nnx.BatchNorm(num_features=1024, momentum=0.9, epsilon=1e-5, dtype=jnp.bfloat16, rngs=rngs))
        # self.hidden_layers_conv = nnx.List(hidden_layers_conv)
        #
        # hidden_layers_linear = [Linear(self.out_conv_dim * self.out_conv_dim * 1024, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        # hidden_layers_linear.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_linear.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_linear.append(nnx.BatchNorm(num_features=1024, momentum=0.9, epsilon=1e-5, dtype=jnp.bfloat16, rngs=rngs))
        # # self.hidden_layers_linear.append(Linear(1024, 8, rngs=rngs))
        # self.hidden_layers_linear = nnx.List(hidden_layers_linear)
        #
        # # Layers to latent
        # hidden_latent = [Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16)]
        # hidden_latent.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        # hidden_latent.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        # hidden_latent.append(nnx.BatchNorm(num_features=256, momentum=0.9, epsilon=1e-5, dtype=jnp.bfloat16, rngs=rngs))
        # self.hidden_latent = nnx.List(hidden_latent)
        # self.mean_x = Linear(256, lat_dim, rngs=rngs)
        # self.logstd_x = Linear(256, lat_dim, rngs=rngs)

        # self.input_conv_dim = 32  # Original was 64
        # self.out_conv_dim = int(self.input_conv_dim / (2 ** 4))
        #
        # hidden_layers_conv = [Linear(self.input_dim * self.input_dim, self.input_conv_dim * self.input_conv_dim, rngs=rngs, dtype=jnp.bfloat16)]
        # hidden_layers_conv.append(Conv(1, 4, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(4, 8, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(8, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers_conv.append(Conv(16, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_layers_conv = nnx.List(hidden_layers_conv)
        #
        # hidden_layers_linear = [Linear(16 * self.out_conv_dim * self.out_conv_dim, 256, rngs=rngs, dtype=jnp.bfloat16)]
        # for _ in range(3):
        #     hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        #
        # hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        # for _ in range(2):
        #     hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        #
        # self.hidden_layers_linear = nnx.List(hidden_layers_linear)

        # hidden_layers = [Linear(self.input_dim * self.input_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        # for _ in range(3):
        #     hidden_layers.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        # hidden_layers.append(Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16))
        # for _ in range(2):
        #     hidden_layers.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_layers = nnx.List(hidden_layers)
        #
        # self.mean_x = Linear(256, lat_dim, rngs=rngs)
        # self.logstd_x = Linear(256, lat_dim, rngs=rngs)

        # Mamba vision style encoder
        self.patch_embed = nnx.Conv(1, 1024,
                                    kernel_size=(4, 4),
                                    strides=(4, 4),
                                    rngs=rngs)
        self.layers = nnx.List([MambaBlock(1024, rngs=rngs) for _ in range(6)])
        self.final_norm = nnx.LayerNorm(1024, rngs=rngs)
        self.mean_x = Linear(1024, lat_dim, rngs=rngs)
        self.logstd_x = Linear(1024, lat_dim, rngs=rngs)

    def sample_gaussian(self, mean, logstd, *, rngs):
        return logstd * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, *, rngs=None):
        # # Resize images
        # x = jax.image.resize(x, (x.shape[0], self.input_conv_dim, self.input_conv_dim, 1), method="bilinear")
        #
        # # Convolutional hidden layers
        # for layer in self.hidden_layers_conv:
        #     if not isinstance(layer, nnx.BatchNorm):
        #         if layer.in_features == layer.out_features and 1 in layer.strides:
        #             x = nnx.gelu(x + layer(x))
        #         else:
        #             x = nnx.gelu(layer(x))
        #     else:
        #         x = layer(x)
        #
        # # Linear hidden layers
        # x = rearrange(x, 'b h w c -> b (h w c)')
        # for layer in self.hidden_layers_linear[:-1]:
        #     if not isinstance(layer, nnx.BatchNorm):
        #         if layer.in_features == layer.out_features:
        #             x = nnx.gelu(x + layer(x))
        #         else:
        #             x = nnx.gelu(layer(x))
        #     else:
        #         x = layer(x)
        # x = self.hidden_layers_linear[-1](x)
        #
        # # Latent space (heterogeneity)
        # latent = nnx.gelu(self.hidden_latent[0](x))
        # for layer in self.hidden_latent[1:]:
        #     if not isinstance(layer, nnx.BatchNorm):
        #         latent = nnx.gelu(latent + layer(latent))
        #     else:
        #         latent = layer(latent)
        # mean = self.mean_x(latent)
        # logstd = self.logstd_x(latent)
        # sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean
        #
        # return sample, mean, logstd

        # x = rearrange(x, 'b h w c -> b (h w c)')
        #
        # x = nnx.leaky_relu(self.hidden_layers_conv[0](x))  # or nnx.relu
        #
        # x = rearrange(x, 'b (h w c) -> b h w c', h=self.input_conv_dim, w=self.input_conv_dim, c=1)
        #
        # for layer in self.hidden_layers_conv[1:]:
        #     if layer.in_features != layer.out_features:
        #         x = nnx.leaky_relu(layer(x))  # or nnx.relu
        #     else:
        #         aux = layer(x)
        #         if aux.shape[1] == x.shape[1]:
        #             x = nnx.leaky_relu(x + aux)  # or nnx.relu
        #         else:
        #             x = nnx.leaky_relu(aux)  # or nnx.relu
        #
        # x = rearrange(x, 'b h w c -> b (h w c)')
        #
        # for layer in self.hidden_layers_linear:
        #     if layer.in_features != layer.out_features:
        #         x = nnx.leaky_relu(layer(x))  # or nnx.relu
        #     else:
        #         x = nnx.leaky_relu(x + layer(x))  # or nnx.relu
        #
        # x = rearrange(x, 'b h w c -> b (h w c)')
        #
        # for layer in self.hidden_layers:
        #     x = nnx.leaky_relu(layer(x))  # or nnx.relu
        #
        # mean = self.mean_x(x)
        # logstd = self.logstd_x(x)
        # sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean
        #
        # return sample, mean, logstd

        # Resize images
        x = jax.image.resize(x, (x.shape[0], self.input_conv_dim, self.input_conv_dim, 1), method="bilinear")

        x = self.patch_embed(x)

        x = rearrange(x, "b h w c -> b (h w) c")
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        x = jnp.mean(x, axis=1)

        mean = self.mean_x(x)
        logstd = self.logstd_x(x)
        sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean

        return sample, mean, logstd

class HetVolumeDecoder(nnx.Module):
    def __init__(self, total_voxels, lat_dim, volume_size, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.total_voxels = total_voxels

        # Gaussian std
        # self.std = nnx.Param(1.0)
        self.std = 1.0

        # Indices to (normalized) coords
        self.factor = 0.5 * volume_size

        # Initial Gaussian values
        # Noise scale 0.5 or 0.1
        self.coords = 0.25 * jnp.array(generate_sphere_points(total_voxels) + np.random.normal(0, 0.1, (total_voxels, 3)))
        self.values = jnp.zeros((total_voxels,))

        # kernel_init = nnx.initializers.variance_scaling(scale=1. / 3., mode="fan_out", distribution="uniform")
        # kernel_init = nnx.initializers.glorot_uniform()
        # hidden_coords = [Linear(in_features=lat_dim // 2, out_features=256, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init_first(c=1.))]
        # for _ in range(4):
        #     hidden_coords.append(Linear(in_features=256, out_features=256, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init(c=6.)))
        # hidden_coords.append(Linear(in_features=256, out_features=3 * total_voxels, rngs=rngs, kernel_init=kernel_init))
        #
        # hidden_values = [Linear(in_features=lat_dim // 2, out_features=256, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init_first(c=6.))]
        # for _ in range(4):
        #     hidden_values.append(Linear(in_features=256, out_features=256, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init(c=1.)))
        # hidden_values.append(Linear(in_features=256, out_features=total_voxels, rngs=rngs, kernel_init=kernel_init))

        # self.hidden_coords = nnx.List(hidden_coords)
        # self.hidden_values = nnx.List(hidden_values)

        kernel_init = nnx.initializers.variance_scaling(scale=1. / 3., mode="fan_out", distribution="uniform")
        hidden_layers = [Linear(in_features=lat_dim, out_features=256, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init_first(c=1.))]
        for _ in range(4):
            hidden_layers.append(Linear(in_features=256, out_features=256, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init(c=6.)))
        self.hidden_layers = nnx.List(hidden_layers)
        self.hidden_coords = Linear(in_features=256, out_features=3 * total_voxels, rngs=rngs, kernel_init=kernel_init)
        self.hidden_values = Linear(in_features=256, out_features=total_voxels, rngs=rngs, kernel_init=kernel_init)

    def __call__(self, x):
        # x_coords, x_map = jnp.split(x, indices_or_sections=2, axis=1)

        # # Decode values
        # x_map = jnp.sin(30.0 * self.hidden_values[0](x_map))
        # for layer in self.hidden_values[1:-1]:
        #     x_map = jnp.sin(x_map + 1.0 * layer(x_map))
        # x_map = self.hidden_values[-1](x_map)
        #
        # # Decode coords
        # x_coords = jnp.sin(30.0 * self.hidden_coords[0](x_coords))
        # for layer in self.hidden_coords[1:-1]:
        #     x_coords = jnp.sin(x_coords + 1.0 * layer(x_coords))
        # x_coords = self.hidden_coords[-1](x_coords)
        #
        # # Extract delta_coords and values
        # x_coords = jnp.reshape(x_coords, (x.shape[0], self.total_voxels, 3))
        # delta_coords, delta_values = x_coords, x_map

        # Common encoder
        x = jnp.sin(30. * self.hidden_layers[0](x))
        for layer in self.hidden_layers[1:-1]:
            x = jnp.sin(1. * (x + layer(x)))
        x = self.hidden_layers[-1](x)

        delta_coords = self.hidden_coords(x)
        delta_values = self.hidden_values(x)

        delta_coords = jnp.reshape(delta_coords, (x.shape[0], self.total_voxels, 3))

        # Recover coords (non-normalized)
        delta_coords = self.factor * (self.coords[None, ...] + delta_coords)

        return delta_coords, nnx.relu(self.values[None, ...] + delta_values)

    def decode_volume(self, x, filter=True):
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
            # grids = jax.vmap(lambda x: low_pass_3d(x, std=self.std.get_value()))(grids)
            grids = jax.vmap(lambda x: low_pass_3d(x, std=1.0))(grids)

        return grids

class PhysDecoder:
    def __init__(self, xsize, transport_mass):
        self.xsize = xsize
        self.transport_mass = transport_mass

    def __call__(self, x, values, coords, xsize, rotations, shifts, ctf, ctf_type, std, filter=True):
        # Volume factor
        factor = 0.5 * xsize

        # Broadcast coords and values
        coords = rearrange(jnp.tile(coords[:, None, ...], (1, rotations.shape[1], 1, 1)), "b n c d -> (b n) c d")
        values = rearrange(jnp.tile(values[:, None, ...], (1, rotations.shape[1], 1)), "b n c -> (b n) c")

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

        if self.transport_mass:
            bposf = jnp.floor(c_sampling)
            bposi = bposf.astype(jnp.int32)
            bposf = c_sampling - bposf

            bamp0 = values * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
            bamp1 = values * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
            bamp2 = values * (bposf[:, :, 0]) * (bposf[:, :, 1])
            bamp3 = values * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])
            bamp = jnp.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
            bposi = jnp.concat([bposi, bposi + jnp.array((1, 0)), bposi + jnp.array((1, 1)), bposi + jnp.array((0, 1))], axis=1)
        else:
            bposf = jnp.round(c_sampling)
            bposi = bposf.astype(jnp.int32)

            num = jnp.square(bposf - c_sampling).sum(axis=-1)
            sigma = 1.
            bamp = values * jnp.exp(-num / (2. * sigma ** 2.))

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        images = jax.vmap(scatter_img)(images, bposi, bamp)

        # Gaussian filter (needed by forward interpolation)
        if filter:
            # images = dm_pix.gaussian_blur(images[..., None], std.get_value(), kernel_size=9)[..., 0]
            images = dm_pix.gaussian_blur(images[..., None], std, kernel_size=9)[..., 0]

        # Apply CTF
        if ctf_type in ["apply", "wiener", "squared"]:
            ctf = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf = rearrange(ctf, "b n w h -> (b n) w h")
            images = ctfFilter(images, ctf, pad_factor=2)

        images = rearrange(images, "(b n) w h -> b n w h", b=rotations.shape[0], n=rotations.shape[1])

        return images

class ReconSIRENHetOnly(nnx.Module):

    @save_config
    def __init__(self, reference_volume, reconstruction_mask, xsize, sr, bank_size=2048, ctf_type="apply", lat_dim=8,
                 transport_mass=False, symmetry_group="c1", refine_current_assignment=False,
                 learn_delta_volume=True, *, rngs: nnx.Rngs):
        super(ReconSIRENHetOnly, self).__init__()
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.reference_volume = reference_volume
        self.reconstruction_mask = reconstruction_mask.astype(float)
        self.inds = jnp.asarray(jnp.where(reconstruction_mask > 0.0)).T
        self.symmetry_matrices = symmetry_matrices(symmetry_group)
        self.refine_current_assignment = refine_current_assignment
        self.learn_delta_volume = learn_delta_volume
        self.transport_mass = transport_mass
        reference_values = reference_volume[self.inds[..., 0], self.inds[..., 1], self.inds[..., 2]][None, ...]
        self.encoder_pose = EncoderPose(self.xsize, refine_current_assignment=refine_current_assignment, rngs=rngs)
        self.encoder_het = EncoderHet(self.xsize, lat_dim=lat_dim, rngs=rngs)
        self.delta_het_decoder = HetVolumeDecoder(10000, lat_dim=lat_dim, volume_size=self.xsize, rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize, transport_mass=transport_mass)

        # Hyperparameter tuning
        self.alpha_uniform = nnx.Param(0.1)

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.subset_size = min(2048, bank_size)

        self.memory_bank = nnx.Variable(
            jnp.zeros((self.bank_size, 2))
        )
        self.memory_bank_ptr = nnx.Variable(
            jnp.zeros((1,), dtype=jnp.int32)
        )

    def __call__(self, x, rngs: nnx.Rngs = None, **kwargs):
        # TODO: Return only best angles
        return self.encoder_pose(x)

    def get_alpha_uniform_lamda(self):
        return nnx.relu(self.alpha_uniform.get_value())

    # --- Method for enqueuing to the memory bank ---
    def enqueue(self, keys_to_add):
        """Updates the memory bank and pointer using JIT-compatible operations."""
        ptr = self.memory_bank_ptr.get_value()[0]

        # Define the starting position for the update.
        # It must be a tuple with one index per dimension of the array.
        # Our memory_bank is 2D, so we need (start_row, start_column).
        start_indices = (ptr, 0)

        # Use `lax.dynamic_update_slice` instead of `.at[...].set(...)`
        self.memory_bank.value = jax.lax.dynamic_update_slice(
            self.memory_bank.get_value(), # 1. The original large array to be updated
            keys_to_add,                  # 2. The smaller array containing the new data
            start_indices                 # 3. The dynamic starting position
        )

        # The pointer update logic remains the same, as it's just arithmetic
        current_batch_size = keys_to_add.shape[0]
        self.memory_bank_ptr.value = jnp.array(
            [(ptr + current_batch_size) % self.bank_size]
        )

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

        if self.ctf_type == "precorrect":
            # Wiener filter
            x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

        # Encode images
        rotations, shifts, logits = self.encoder_pose(x, return_diversity_loss=False, is_training=False)
        x = self.encoder_het(x)

        # Get best poses/shifts
        best_index = jnp.argmax(logits, axis=-1)
        rotations = jnp.take_along_axis(rotations, best_index[..., None, None, None], axis=1).squeeze(axis=1)
        shifts = jnp.take_along_axis(shifts, best_index[..., None, None], axis=1).squeeze(axis=1)

        # Decode volume
        coords, values = self.delta_het_decoder.decode_volume(x, filter=True)

        # Generate projections
        images_corrected = self.phys_decoder(x, values, coords, self.xsize, rotations, shifts, ctf, ctf_type, self.delta_het_decoder.std)

        return images_corrected

    def decode_het_volume(self, x, filter=True):
        if x.ndim == 4:
            x = self.encoder_het(x)

        # Decode het volume
        vol = self.delta_het_decoder.decode_volume(x, filter=filter)

        return vol


@partial(jax.jit, static_argnames=("is_train_step",))
def step_reconsiren_het_only(graphdef, state, x, labels, md, key, lambda_uniform=0.0005, is_train_step=False):
    model, optimizer_pose, optimizer_het = nnx.merge(graphdef, state)

    # Random keys
    key, swd_key, uniform_key, choice_key, distributions_key = jax.random.split(key, 5)

    def loss_fn(model, x):
        # Correct CTF in images for encoder if needed
        if model.ctf_type == "apply":
            x_ctf_corrected = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]
        else:
            x_ctf_corrected = x

        # Get euler angles and shifts
        rotations, shifts, logits, diversity_loss = model.encoder_pose(x_ctf_corrected, return_diversity_loss=True, is_training=is_train_step, key=distributions_key)
        sample, latent, logstd = model.encoder_het(x_ctf_corrected, rngs=distributions_key)

        # Decode het volume
        coords_het, values_het = model.delta_het_decoder(sample)

        # Refine current assignment (if provided)
        rotations = jnp.matmul(current_rotations[:, None, :, :], rotations)  # TODO: The two options seem to work?
        shifts = current_shifts[:, None, :] + shifts

        # Random symmetry matrices
        random_indices = jax.random.choice(choice_key, jnp.arange(model.symmetry_matrices.shape[0]), shape=(rotations.shape[0],))
        rotations = jnp.matmul(jnp.transpose(model.symmetry_matrices[random_indices], (0, 2, 1))[:, None, :, :], rotations)

        # Generate projections
        # images_corrected = model.phys_decoder(x, jax.lax.stop_gradient(values_het), jax.lax.stop_gradient(coords_het), model.xsize, rotations, shifts, ctf, model.ctf_type)
        images_corrected = model.phys_decoder(x, values_het, coords_het, model.xsize, rotations, shifts, ctf, model.ctf_type, model.delta_het_decoder.std)

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

        # Project "mask"
        if not model.transport_mass:
            projected_mask = model.phys_decoder(x, jnp.ones_like(values_het), jax.lax.stop_gradient(coords_het),
                                                model.xsize, rotations, shifts, ctf, None, jax.lax.stop_gradient(model.delta_het_decoder.std), False)
            projected_mask = jnp.where(projected_mask > 1, 1.0, projected_mask)
        else:
            projected_mask = jnp.ones_like(images_corrected)

        # Projection mask
        projected_mask = projected_mask[..., 0] if projected_mask.shape[-1] == 1 else projected_mask
        x_loss = x_loss * projected_mask
        images_corrected_loss = images_corrected_loss * projected_mask

        x_flat = rearrange(x_loss, "b n w h -> (b n) w h")
        images_corrected_flat = rearrange(images_corrected_loss, "b n w h -> (b n) w h")
        recon_loss = dm_pix.mse(images_corrected_flat[..., None], x_flat[..., None])
        recon_loss = rearrange(recon_loss, "(b n) -> b n", b=images_corrected_loss.shape[0], n=images_corrected_loss.shape[1])

        # Get minimum indices
        min_indices = jnp.argmin(recon_loss, axis=1)

        # Index losses and rotations based on extracted indices
        recon_loss = recon_loss[jnp.arange(images_corrected.shape[0]), min_indices].mean()
        recon_loss_all = recon_loss.mean()
        # rotations = rotations[jnp.arange(images_corrected.shape[0]), min_indices, :]
        rotations = rearrange(rotations, "b n w h -> (b n) w h")

        # Rotation repulsion loss
        # margin = 0.5
        # flat_poses = rotations.reshape(rotations.shape[0], rotations.shape[1], 9)
        # sim_matrix = jnp.einsum('bni,bmi->bnm', flat_poses, flat_poses)
        # mask = 1.0 - jnp.eye(sim_matrix.shape[1])
        # loss_rot_repulsion = jnp.maximum(0.0, sim_matrix - margin) * mask
        # loss_rot_repulsion = jnp.mean(loss_rot_repulsion)

        # Reconstruction loss (Soft-WTA)
        # log_pi = jax.nn.log_softmax(logits, axis=-1)
        # tau = 10.
        # combined_terms = log_pi - recon_loss / tau
        # recon_loss_all = -jax.nn.logsumexp(combined_terms, axis=-1).mean()

        # Reconstruction loss (best WTA)
        # best_index = jnp.argmax(logits, axis=-1)
        # rotations = jax.lax.stop_gradient(jnp.take_along_axis(rotations, best_index[..., None, None, None], axis=1))
        # shifts = jax.lax.stop_gradient(jnp.take_along_axis(shifts, best_index[..., None, None], axis=1))
        # images_corrected = model.phys_decoder(x, values_het, coords_het, model.xsize, rotations, shifts, ctf, model.ctf_type, model.delta_het_decoder.std).squeeze(axis=1)
        # recon_loss = dm_pix.mse(images_corrected[..., None], x).mean()
        # recon_loss_all += recon_loss

        # Logits entropy loss
        # probs = jax.nn.softmax(logits, axis=-1)
        # log_probs = jax.nn.log_softmax(logits, axis=-1)
        # entropy = -jnp.sum(probs * log_probs, axis=-1)
        # loss_logits = -jnp.mean(entropy)

        # Probability loss to get the heads
        prob_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=min_indices)
        prob_loss = jnp.mean(prob_loss)

        # Rotations to Euler angles (ZYZ)
        euler_angles = wrap_zyz_angles(-euler_from_matrix_batch(rotations))[..., :2]

        # L1 based denoising
        if not model.transport_mass:
            l1_loss = jnp.mean(jnp.abs(values_het)) + jnp.mean(jnp.abs(values_het))
        else:
            l1_loss = 0.0

        # L1 and L2 total variation
        # diff_x, diff_y, diff_z = sparse_finite_3D_differences(values_het, model.inds, model.xsize)
        # l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        # l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        # KL loss VAE
        kl_loss = -0.5 * jnp.sum(1. + 2. * logstd - jnp.square(jnp.exp(logstd)) - jnp.square(latent))

        # Decoupling (TODO: In the future this will be for missing angles like TF implementation)

        # Uniform angular distribution loss
        random_indices = jnr.choice(choice_key, a=jnp.arange(model.bank_size), shape=(model.subset_size,), replace=False)
        memory_bank_subset = model.memory_bank[random_indices]
        memory_bank_subset = jnp.concat([euler_angles, memory_bank_subset], axis=0)
        uniform_distributed_samples = sample_uniform_zyz(uniform_key, memory_bank_subset.shape[0])[..., :2]
        uniform_angular_distribution_loss = sliced_wasserstein_loss(memory_bank_subset, uniform_distributed_samples, key)
        loss_uniform = uniform_angular_distribution_loss + diversity_loss

        # loss = (recon_loss_all + 0.001 * l1_loss + 0.001 * (l1_grad_loss + l2_grad_loss) + 0.000001 * kl_loss +
        #         lambda_uniform * loss_uniform)
        loss = recon_loss_all + 0.000001 * kl_loss + lambda_uniform * loss_uniform + 0.0001 * prob_loss
        # loss = recon_loss_all
        return loss, (recon_loss.mean(), loss_uniform, euler_angles)

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

    if is_train_step:
        # Optimizer parameters
        params_pose = nnx.All(nnx.Param, nnx.PathContains('encoder_pose'))
        params_het = nnx.All(nnx.Param, (nnx.PathContains("encoder_het"), nnx.PathContains('delta_het_decoder')))

        grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, (params_pose, params_het)), has_aux=True)
        (loss, (recon_loss, loss_uniform, euler_angles)), grads_combined = grad_fn(model, x)

        grads_pose, grads_het = grads_combined.split(params_pose, params_het)

        optimizer_pose.update(model, grads_pose)
        optimizer_het.update(model, grads_het)

        # Update memory bank
        model.enqueue(euler_angles)

        state = nnx.state((model, optimizer_pose, optimizer_het))

        return loss, recon_loss, state, key
    else:
        (_, (recon_loss, _, _)) = loss_fn(model, x)

        return recon_loss


@jax.jit
def predict_angular_assignment_step_reconsiren_het_only(graphdef, state, x, labels, md, key):
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

    # Correct CTF in images for encoder if needed
    if model.ctf_type == "apply":
        x_ctf_corrected = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]
    else:
        x_ctf_corrected = x

    # Get euler angles and shifts
    rotations, shifts, logits = model.encoder_pose(x_ctf_corrected, return_diversity_loss=False, is_training=False)
    _, latent, _ = model.encoder_het(x_ctf_corrected, rngs=distributions_key)

    # Get best poses/shifts
    best_index = jnp.argmax(logits, axis=-1)
    rotations = jnp.take_along_axis(rotations, best_index[...,None, None, None], axis=1).squeeze(axis=1)
    shifts = jnp.take_along_axis(shifts, best_index[..., None, None], axis=1).squeeze(axis=1)

    # Refine current assignment (if provided)
    # rotations = jnp.matmul(rotations, current_rotations)
    rotations = jnp.matmul(current_rotations, rotations)  # TODO: The two options seem to work?
    shifts = current_shifts + shifts

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
    from hax.schedulers import CosineAnnealingScheduler
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
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                             precision=np.float16, group_size=1, shard_size=10000)

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

    # # If exists, clean MMAP
    # if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key, choice_key = jax.random.split(rng, 3)

    # Prepare network (ReconSIREN)
    reconsiren = ReconSIRENHetOnly(vol, mask, xsize, args.sr, ctf_type=args.ctf_type, symmetry_group=args.symmetry_group,
                                   transport_mass=True, refine_current_assignment=args.refine_current_assignment, lat_dim=8,
                                   bank_size=2048, learn_delta_volume=not args.do_not_learn_volume, rngs=nnx.Rngs(model_key))

    # Reload network
    if args.reload is not None:
        reconsiren = NeuralNetworkCheckpointer.load(os.path.join(args.reload, "ReconSIREN"))

    # Volume adjustment (only if reference volume is provided)
    if args.vol is not None:
        # Extract mask coords
        inds = np.asarray(np.where(mask > 0.0)).T
        values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
        factor = 0.5 * vol.shape[0]
        coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
        coords = (coords - factor) / factor

        # Define volume adjustment network
        volumeAdjustment = VolumeAdjustment(lat_dim=3, coords=coords, values=values, predicts_value=True, rngs=nnx.Rngs(model_key))

        # Reload volume adjustment network if needed
        if args.reload is not None:
            volumeAdjustment = NeuralNetworkCheckpointer.load(os.path.join(args.reload, "volumeAdjustment"))

    # Train network
    if args.mode == "train":

        reconsiren.train()

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "ReconSIREN_metrics"))

        # Jitted functions for volume prediction
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

        if args.vol is not None:
            if not os.path.isdir(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT")):
                # Optimizers (Volume Adjustment)
                optimizer_vol = nnx.Optimizer(volumeAdjustment, optax.adam(1e-5), wrt=nnx.Param)
                graphdef, state = nnx.split((volumeAdjustment, optimizer_vol))

                # Number epochs (volume adjustment)
                if len(generator.md) >= 10000:
                    num_epochs_vol = 20
                else:
                    num_epochs_vol = 200

                # Training loop (Volume Adjustment)
                print(f"{bcolors.OKCYAN}\n###### Training volume adjustment... ######")

                i = 0
                pbar = tqdm(range(num_epochs_vol * steps_per_epoch), file=sys.stdout, ascii=" >=", colour="green",
                            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

                with closing(iter(data_loader_train)) as iter_data_loader:
                    for total_steps in pbar:
                        (x, labels) = next(iter_data_loader)

                        if total_steps % steps_per_epoch == 0:
                            total_loss = 0

                            # For progress bar (TQDM)
                            step = 1
                            print(f'\nTraining epoch {i + 1}/{num_epochs_vol} |')
                            pbar.set_description(f"Epoch {int(total_steps / steps_per_epoch + 1)}/{args.epochs}")

                            i += 1

                        loss, state = train_step_volume_adjustment(graphdef, state, x, labels, md_columns, args.sr,
                                                                   args.ctf_type, vol.shape[0])
                        total_loss += loss

                        # Progress bar update  (TQDM)
                        pbar.set_postfix_str(f"loss={total_loss / step:.5f}")

                        # Summary writer (training loss)
                        if step % int(np.ceil(0.1 * steps_per_epoch)) == 0:
                            writer.add_scalar('Training loss (volume adjustment)',
                                              total_loss / step,
                                              i * steps_per_epoch + step)

                        step += 1

                volumeAdjustment, optimizer_vol = nnx.merge(graphdef, state)
                values = volumeAdjustment()

                # Place values on grid and replace ReconSIREN reference volume
                grid = jnp.zeros_like(vol)
                grid = grid.at[inds[..., 0], inds[..., 1], inds[..., 2]].set(values)
                reconsiren.reference_volume = grid
                reconsiren.delta_volume_decoder.reference_values = values

                # Save model
                NeuralNetworkCheckpointer.save(reconsiren, os.path.join(args.output_path, "volumeAdjustment"))

        # Learning rate scheduler
        # total_steps = args.epochs * len(data_loader)
        # lr_schedule_pose = CosineAnnealingScheduler.getScheduler(peak_value=4. * args.learning_rate, total_steps=total_steps, warmup_frac=0.1, init_value=args.learning_rate, end_value=0.0)
        # lr_schedule_volume = CosineAnnealingScheduler.getScheduler(peak_value=4. * 1e-3, total_steps=total_steps, warmup_frac=0.1, init_value=1e-3, end_value=0.0)
        # lr_schedule_het = CosineAnnealingScheduler.getScheduler(peak_value=4. * 1e-3, total_steps=total_steps, warmup_frac=0.1, init_value=1e-3, end_value=0.0)

        # Define optimizers
        volume_lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1e-3,
            warmup_steps=1000,  # Delay volume learning
            decay_steps=50000,
            end_value=1e-4
        )
        volume_tx = optax.adamw(learning_rate=volume_lr_schedule, weight_decay=1e-4)

        pose_lr_schedule = optax.cosine_decay_schedule(
            init_value=3e-4,  # Start active immediately
            decay_steps=50000,
            alpha=0.1
        )
        pose_tx = optax.chain(
            optax.clip_by_global_norm(1.0),  # Prevent explosive pose updates
            optax.radam(learning_rate=pose_lr_schedule)
        )

        # Optimizers (ReconSIREN)
        params_pose = nnx.All(nnx.Param, nnx.PathContains('encoder_pose'))
        params_het = nnx.All(nnx.Param, (nnx.PathContains('encoder_het'), nnx.PathContains('delta_het_decoder')))
        optimizer_pose = nnx.Optimizer(reconsiren, optax.adam(1e-4), wrt=params_pose)  # Or rmsprop with 1e-3
        optimizer_het = nnx.Optimizer(reconsiren, optax.adam(1e-4), wrt=params_het)
        graphdef, state = nnx.split((reconsiren, optimizer_pose, optimizer_het))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"),
                                                                                        optimizer_pose, optimizer_het)
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

        # with closing(iter(data_loader_train)) as iter_data_loader_train, closing(iter(data_loader_val)) as iter_data_loader_val:
        with closing(iter(data_loader_train)) as iter_data_loader_train:
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
                        reconsiren, optimizer_pose, optimizer_het = nnx.merge(graphdef, state)

                        # Plot angular distribution
                        euler_angles = np.array(reconsiren.memory_bank.get_value())
                        fig, _ = plot_angular_distribution(euler_angles)
                        writer.add_figure("Angular distribution density", fig, global_step=i)

                        # Predict some heterogeneous volumes
                        latents = []
                        graphdef_aux, state_aux = nnx.split(reconsiren)
                        for _ in range(steps_per_epoch):
                            (x, labels) = next(iter_data_loader_train)
                            _, _, latent = predict_angular_assignment_step_reconsiren_het_only(graphdef_aux, state_aux, x,
                                                                                               labels, md_columns, rng)
                            latents.append(np.array(latent))
                        latents = np.concatenate(latents, axis=0)
                        kmeans = KMeans(n_clusters=30).fit(latents)
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

                # if i < 3:
                #     loss, state, rng = warmup_het_reconsiren(graphdef, state, x, labels, md_columns, rng, is_train_step=True)
                #     recon_loss = loss
                # else:
                # lambda_uniform = get_cosine_reg_strength(total_steps, 10 * steps_per_epoch, 0.001, 1e-4)
                loss, recon_loss, state, rng = step_reconsiren_het_only(graphdef, state, x, labels, md_columns, rng, lambda_uniform=0.0005, is_train_step=True)
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

                # Summary writer (validation loss)
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

        reconsiren, optimizer_pose, optimizer_het = nnx.merge(graphdef, state)

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
            rotations, shifts, latent = predict_angular_assignment_step_reconsiren_het_only(graphdef, state, x, labels, md_columns, rng)

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