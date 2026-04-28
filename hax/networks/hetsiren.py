#!/usr/bin/env python


from functools import partial
import numpy as np

import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import dm_pix

from einops import rearrange

from hax.utils import *
from hax.layers import *
from hax.programs import splat_weights_trilinear, splat_weights, FastVariableBlur3D


def mse(a, b):
    return jnp.mean(jnp.square(a - b), axis=(-3, -2, -1))


class Encoder(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, architecture="convnn", isVae=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 32  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 4))
        self.architecture = architecture
        self.isVae = isVae

        if self.architecture == "mlpnn":
            hidden_layers = [Linear(self.input_dim * self.input_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
            for _ in range(n_layers):
                hidden_layers.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers.append(Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16))
            for _ in range(2):
                hidden_layers.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers = nnx.List(hidden_layers)
            # self.hidden_layers = hidden_layers
            self.latent = Linear(256, lat_dim, rngs=rngs)

        elif self.architecture == "convnn":
            hidden_layers_conv = [Linear(self.input_dim * self.input_dim, self.input_conv_dim * self.input_conv_dim, rngs=rngs, dtype=jnp.bfloat16)]
            hidden_layers_conv.append(Conv(1, 4, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(Conv(4, 8, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(Conv(8, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(Conv(16, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv = nnx.List(hidden_layers_conv)

            hidden_layers_linear = [Linear(16 * self.out_conv_dim * self.out_conv_dim, 256, rngs=rngs, dtype=jnp.bfloat16)]
            for _ in range(3):
                hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

            hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
            for _ in range(2):
                hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

            self.hidden_layers_linear = nnx.List(hidden_layers_linear)

            if isVae:
                self.mean_x = Linear(256, lat_dim, rngs=rngs)
                self.logstd_x = Linear(256, lat_dim, rngs=rngs)
            else:
                self.latent = Linear(256, lat_dim, rngs=rngs)

        else:
            raise ValueError("Architecture not supported. Implemented architectures are: mlpnn / convnn")

    def sample_gaussian(self, mean, logstd, *, rngs):
        return logstd * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, return_last=False, *, rngs=None):
        if self.architecture == "mlpnn":
            x = rearrange(x, 'b h w c -> b (h w c)')

            for layer in self.hidden_layers:
                x = nnx.leaky_relu(layer(x))  # or nnx.relu

        elif self.architecture == "convnn":
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

        if return_last:
            return x
        else:
            if self.isVae:
                mean = self.mean_x(x)
                logstd = self.logstd_x(x)
                sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean
                return sample, mean, logstd
            else:
                latent = self.latent(x)
                return latent

class EncoderTomo(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, isVae=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.isVae = isVae

        hidden_layers = [Linear(self.input_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(n_layers):
            hidden_layers.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers.append(Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16))
        for _ in range(2):
            hidden_layers.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers = nnx.List(hidden_layers)
        self.latent = Linear(256, lat_dim, rngs=rngs)

        if isVae:
            self.mean_x = Linear(256, lat_dim, rngs=rngs)
            self.logstd_x = Linear(256, lat_dim, rngs=rngs)
        else:
            self.latent = Linear(256, lat_dim, rngs=rngs)

    def sample_gaussian(self, mean, logstd, *, rngs):
        return logstd * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, return_last=False, *, rngs=None):
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))

        if return_last:
            return x
        else:
            if self.isVae:
                mean = self.mean_x(x)
                logstd = self.logstd_x(x)
                sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean
                return sample, mean, logstd
            else:
                latent = self.latent(x)
                return latent

class MultiEncoder(nnx.Module):
    def __init__(self, input_dim, d_hid=100, lat_dim=10, n_layers=3, isVae=False, architecture="convnn", isTomoSIREN=False, *, rngs: nnx.Rngs):
        if isTomoSIREN:
            self.encoders = nnx.Dict({"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, architecture=architecture, rngs=rngs),
                                      "encoder_dec": EncoderTomo(d_hid, lat_dim, n_layers=n_layers, rngs=rngs)})
        else:
            self.encoders = nnx.Dict({"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, architecture=architecture, rngs=rngs),
                                      "encoder_dec": Encoder(input_dim, lat_dim, n_layers=n_layers, architecture=architecture, rngs=rngs)})
        self.isVae = isVae
        if isVae:
            self.mean_x = Linear(256, lat_dim, rngs=rngs)
            self.logstd_x = Linear(256, lat_dim, rngs=rngs)
        else:
            self.latent = Linear(256, lat_dim, rngs=rngs)

        # Hidden layers latent space
        hidden_layers_latent = [Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(2):
            hidden_layers_latent.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_latent = nnx.List(hidden_layers_latent)

        # Hidden layer refinement
        hidden_layers_refinement = [Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(2):
            hidden_layers_refinement.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_refinement = nnx.List(hidden_layers_refinement)

        # Rigid registration of volumes
        self.rigid_6d_rotation = nnx.Linear(256, 6, rngs=rngs, kernel_init=nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())
        self.rotations_logsig = nnx.Linear(256, 3, rngs=rngs)
        self.rigid_shifts = nnx.Linear(256, 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init(), bias_init=nnx.initializers.zeros_init())

    def sample_gaussian(self, mean, logstd, *, rngs):
        return logstd * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, encoder_id="encoder_exp", return_last=False, return_alignment_refinement=False, *,
                 rngs=None):
        x = self.encoders[encoder_id](x, return_last=True)

        if return_alignment_refinement:
            x_ref = nnx.leaky_relu(x + self.hidden_layers_refinement[0](x))  # or nnx.relu
            for layer in self.hidden_layers_refinement[1:]:
                x_ref = nnx.leaky_relu(layer(x_ref + x_ref))  # or nnx.relu

            # Estimate rotations for volume registration
            rotations_6d = self.rigid_6d_rotation(x_ref)
            identity_6d = jnp.array([1., 0., 0., 0., 1., 0.])[None, ...].repeat(rotations_6d.shape[0], axis=0)
            rotations_6d = identity_6d + rotations_6d
            rotations_rigid = PoseDistMatrix.mode_rotmat(rotations_6d)
            rotations_logscale = self.rotations_logsig(x_ref)

            # Estimate shifts for volume registration
            shifts_rigid = self.rigid_shifts(x_ref)

        for layer in self.hidden_layers_latent:
            x = nnx.leaky_relu(x + layer(x))  # or nnx.relu

        if self.isVae:
            mean = self.mean_x(x)
            logstd = self.logstd_x(x)
            sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean
            if return_last:
                if return_alignment_refinement:
                    return (sample, mean, logstd), (rotations_rigid, shifts_rigid, rotations_logscale), x
                else:
                    return (sample, mean, logstd), x
            else:
                if return_alignment_refinement:
                    return (sample, mean, logstd), (rotations_rigid, shifts_rigid, rotations_logscale)
                else:
                    return sample, mean, logstd
        else:
            latent = self.latent(x)
            if return_last:
                if return_alignment_refinement:
                    return latent, (rotations_rigid, shifts_rigid, rotations_logscale), x
                else:
                    return latent, x
            else:
                if return_alignment_refinement:
                    return latent, (rotations_rigid, shifts_rigid, rotations_logscale)
                else:
                    return latent


class DeltaVolumeDecoder(nnx.Module):
    def __init__(self, total_voxels, lat_dim, volume_size, coords, reference_values, transport_mass=False, is_implicit=True, hybrid_pe=False, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.reference_values = reference_values[None, ...]
        self.total_voxels = total_voxels
        self.transport_mass = transport_mass
        self.is_implicit = is_implicit
        self.hybrid_pe = hybrid_pe

        # Indices to (normalized) coords
        mins, maxs = coords.min(axis=0), coords.max(axis=0)
        self.scale = 0.5 * max(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2])
        self.centering = jnp.array((0.5 * volume_size, 0.5 * volume_size, 0.5 * volume_size))[None, None, ...]
        self.coords = ((coords[None, ...] - self.centering) / self.scale)

        # Scale value for SIREN2
        inds = jnp.stack([coords[:, 2], coords[:, 1], coords[:, 0]], axis=1)
        coords_for_psi = (inds - 0.5 * volume_size) / (0.5 * volume_size)
        if transport_mass:
            vol_for_psi = splat_weights_trilinear(volume_size, coords_for_psi, reference_values)
        else:
            vol_for_psi = splat_weights(volume_size, coords_for_psi, reference_values)
        vol_for_psi = FastVariableBlur3D((volume_size, volume_size, volume_size))(vol_for_psi[None, ..., None], sigma=1.0)[0, ..., 0]
        psi = calculate_spectral_centroid_3d(vol_for_psi)
        s0 = 50. * (1. - jnp.exp(5. * psi * 32.))
        s1 = 0.4 * psi / 32.

        # Graph from coordinates
        if not jnp.all(reference_values == 0) and transport_mass:
            # Graph from reference
            self.edge_index, self.edge_weights, self.consensus_distances, self.tau, _, _ = build_graph_from_coordinates(self.coords[0], k_spacing=2, k_knn=6, radius_factor=1.5)
            self.edge_weights = jnp.ones_like(self.edge_weights)

        # Delta volume decoder (TODO: Check and fix hypernetwork - compare with TF implementation)
        # self.hidden_linear = [HyperLinear(in_features=lat_dim, out_features=8, in_hyper_features=lat_dim, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16)]
        # for _ in range(3):
        #     self.hidden_linear.append(HyperLinear(in_features=8, out_features=8, in_hyper_features=8, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_linear.append(HyperLinear(in_features=8, out_features=8, in_hyper_features=8, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16))

        if transport_mass:
            if self.is_implicit:
                if not self.hybrid_pe:
                    # Implicit version
                    hidden_coords = [Siren2Linear(in_features=lat_dim // 2 + 3, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=True, w0=30.0, s=s0, use_bias=False)]
                    hidden_coords.append(Siren2Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=s1, use_bias=False))
                    for _ in range(7):
                        hidden_coords.append(Siren2Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=0.0, use_bias=False))
                    hidden_coords.append(nnx.Linear(in_features=32, out_features=3, rngs=rngs, use_bias=False, kernel_init=nnx.initializers.zeros_init()))

                    hidden_values = [Siren2Linear(in_features=lat_dim // 2 + 3, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=True, w0=30.0, s=s0, use_bias=False)]
                    hidden_values.append(Siren2Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=s1, use_bias=False))
                    for _ in range(7):
                        hidden_values.append(Siren2Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=0.0, use_bias=False))
                    hidden_values.append(nnx.Linear(in_features=32, out_features=1, rngs=rngs, use_bias=False, kernel_init=nnx.initializers.zeros_init()))

                else:
                    # Implicit version
                    kernel_init = nnx.initializers.variance_scaling(scale=1. / 3., mode="fan_in", distribution="uniform")
                    hidden_coords = [nnx.Linear(in_features=lat_dim // 2 + 3 * 10 * 2, out_features=32, rngs=rngs, dtype=jnp.float32, use_bias=False, kernel_init=kernel_init)]
                    for _ in range(8):
                        hidden_coords.append( nnx.Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, use_bias=False, kernel_init=kernel_init))
                    hidden_coords.append(nnx.Linear(in_features=32, out_features=3, rngs=rngs, dtype=jnp.float32, use_bias=False, kernel_init=kernel_init))
                    hidden_coords.append(nnx.Linear(in_features=3, out_features=3, rngs=rngs, use_bias=False, kernel_init=kernel_init))

                    hidden_values = [Siren2Linear(in_features=lat_dim // 2 + 3, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=True, w0=30.0, s=s0, use_bias=False)]
                    hidden_values.append(Siren2Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=s1, use_bias=False))
                    for _ in range(7):
                        hidden_values.append(Siren2Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=0.0, use_bias=False))
                    hidden_values.append(Siren2Linear(in_features=32, out_features=1, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=0.0, use_bias=False))
                    hidden_values.append(nnx.Linear(in_features=1, out_features=1, rngs=rngs, use_bias=False, kernel_init=nnx.initializers.zeros_init()))

            else:
                # Standard version
                hidden_coords = [Siren2Linear(in_features=lat_dim // 2, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=True, w0=30.0, s=s0)]
                hidden_coords.append(Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=False, custom_init=True, is_residual=True, w0=1.0, s=s1))
                for _ in range(4):
                    hidden_coords.append(Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=False, custom_init=True, is_residual=True, w0=1.0, s=0.0))
                hidden_coords.append(Linear(in_features=8, out_features=3 * total_voxels, rngs=rngs, kernel_init=nnx.initializers.glorot_uniform(), bias_init=nnx.initializers.zeros_init()))

                hidden_values = [Siren2Linear(in_features=lat_dim // 2, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=True, w0=30.0, s=s0)]
                hidden_values.append(Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=False, custom_init=True, is_residual=True, w0=1.0, s=s1))
                for _ in range(4):
                    hidden_values.append(Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=False, custom_init=True, is_residual=True, w0=1.0, s=0.0))
                hidden_values.append(Linear(in_features=8, out_features=total_voxels, rngs=rngs, kernel_init=nnx.initializers.glorot_uniform(), bias_init=nnx.initializers.zeros_init()))

            self.hidden_values = nnx.List(hidden_values)
            self.hidden_coords = nnx.List(hidden_coords)

        else:
            self.is_implicit = False
            kernel_init = nnx.initializers.glorot_uniform() if not bool(jnp.all(reference_values == 0.0)) else nnx.initializers.zeros_init()
            hidden_values = [Siren2Linear(in_features=lat_dim, out_features=8, rngs=rngs, dtype=jnp.bfloat16, is_first=True, w0=30.0,s=s0)]
            hidden_values.append(Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, custom_init=True, is_residual=True, w0=1.0, s=s1))
            for _ in range(3):
                hidden_values.append(Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, custom_init=True, is_residual=True, w0=1.0, s=0.0))
            hidden_values.append(Linear(in_features=8, out_features=total_voxels, rngs=rngs, kernel_init=kernel_init))

            self.hidden_values = nnx.List(hidden_values)

    def __call__(self, x, c=None):
        if self.transport_mass:
            if self.is_implicit:
                # Positional encoding of coords
                if c is None:
                    c = self.coords[0]
                c = jnp.tile(c[None, ...], (x.shape[0], 1, 1))

                # Adjust latents
                x = jnp.tile(x[:, None, ...], (1, c.shape[1], 1))

                x_coords, x_map = jnp.split(x, indices_or_sections=2, axis=-1)

                # Join coords and latents
                if self.hybrid_pe:
                    c_pe = positional_encoding(c[0], 10, self.scale)
                    c_pe = jnp.tile(c_pe[None, ...], (x.shape[0], 1, 1))
                    x_coords = jnp.concatenate([c_pe, x_coords], axis=-1)
                else:
                    x_coords = jnp.concatenate([c, x_coords], axis=-1)
                x_map = jnp.concatenate([c, x_map], axis=-1)

                # Decode values
                x_map = self.hidden_values[0](x_map)
                for layer in self.hidden_values[1:-1]:
                    x_map = layer(x_map)
                x_map = self.hidden_values[-1](x_map)[..., 0]

                # Decode coords
                if self.hybrid_pe:
                    x_coords = nnx.elu(self.hidden_coords[0](x_coords))
                    for layer in self.hidden_coords[1:-1]:
                        x_coords = nnx.elu(layer(x_coords))
                    x_coords = self.hidden_coords[-1](x_coords)
                else:
                    x_coords = self.hidden_coords[0](x_coords)
                    for layer in self.hidden_coords[1:-1]:
                        x_coords = layer(x_coords)
                    x_coords = self.hidden_coords[-1](x_coords)

            else:
                x_coords, x_map = jnp.split(x, indices_or_sections=2, axis=1)

                # Decode values
                x_map = self.hidden_values[0](x_map)
                for layer in self.hidden_values[1:-1]:
                    x_map = layer(x_map)
                x_map = self.hidden_values[-1](x_map)

                # Decode coords
                x_coords = self.hidden_coords[0](x_coords)
                for layer in self.hidden_coords[1:-1]:
                    x_coords = layer(x_coords)
                x_coords = self.hidden_coords[-1](x_coords)

                x_coords = jnp.reshape(x_coords, (x.shape[0], self.total_voxels, 3))

            delta_coords, delta_values = x_coords, x_map

            # Recover volume values
            values = nnx.relu(self.reference_values + delta_values)

            # Recover coords (non-normalized)
            coords = self.scale * (self.coords + delta_coords)
        else:
            # Decode voxel values
            x_map = self.hidden_values[0](x)
            for layer in self.hidden_values[1:-1]:
                x_map = layer(x_map)
            x_map = self.hidden_values[-1](x_map)

            # Recover volume values
            values = self.reference_values + x_map

            # Recover coords (non-normalized)
            coords = self.scale * self.coords.repeat(x.shape[0], axis=0)

        return coords, values

    def decode_volume(self, x=None, coords_values=None, filter=True, sigma=1.0):
        if x is not None:
            # Decode volume values
            coords, values = self.__call__(x)
        elif coords_values is not None:
            coords, values = coords_values
        else:
            raise ValueError("Please provide either x or coords_value parameter")

        # Displace coordinates
        coords = coords + self.centering

        # Place values on grid
        grids = jnp.zeros((values.shape[0], self.volume_size, self.volume_size, self.volume_size))

        # Scatter volume
        if self.transport_mass:
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
        else:
            bamp = values
            bposi = jnp.floor(coords).astype(jnp.int32)

        def scatter_volume(vol, bpos_i, bamp_i):
            return vol.at[bpos_i[..., 2], bpos_i[..., 1], bpos_i[..., 0]].add(bamp_i)

        grids = jax.vmap(scatter_volume, in_axes=(0, 0, 0))(grids, bposi, bamp)

        # Filter volume
        if filter:
            grids = jax.vmap(low_pass_3d, in_axes=(0, None))(grids, sigma)

        return grids

class PhysDecoder:
    def __init__(self, xsize, sr, transport_mass):
        self.xsize = xsize
        self.transport_mass = transport_mass
        # self.pad_factor = 1 if xsize > 256 else 2
        self.pad_factor = 2

        physical_radius_px = 60.0 / sr
        max_safe_radius_px = xsize // 8
        dilation_radius = jnp.minimum(physical_radius_px, max_safe_radius_px)
        self.dilation_radius = int(jnp.maximum(dilation_radius, 5))

    def __call__(self, x, values, coords, xsize, rotations, shifts, centering, ctf, ctf_type, sigma, bg_weight=1.0, filter=True):
        # Get rotation matrices
        if rotations.ndim == 2:
            rotations = euler_matrix_batch(rotations[:, 0], rotations[:, 1], rotations[:, 2])

        coords = jnp.matmul(coords, rearrange(rotations, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts[:, None, :] + centering[..., :-1]

        # Scatter image
        B = x.shape[0]
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
            bamp = values * jnp.exp(-num[:, None, :] / (2. * sigma ** 2.))
        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        images = jax.vmap(scatter_img)(images, bposi, jnp.mean(bamp, axis=1))

        # Gaussian filter (needed by forward interpolation)
        if filter:
            images = dm_pix.gaussian_blur(images[..., None], sigma, kernel_size=9)[..., 0]

        # Define weighted mask for losses
        images_mask = jnp.where(images > 1e-6, 1.0, 0.0)

        # Apply CTF
        if ctf_type in ["apply" or "wiener" or "squared"]:
            images = ctfFilter(images, ctf, pad_factor=self.pad_factor)

            # Weighted mask (CTF case)
            kernel_size = 2 * self.dilation_radius + 1
            images_mask = images_mask[..., None].astype(x.dtype)
            images_mask_vert = jax.lax.reduce_window(
                images_mask, -jnp.inf, jax.lax.max,
                window_dimensions=(1, kernel_size, 1, 1),
                window_strides=(1, 1, 1, 1), padding='SAME'
            )
            images_mask = jax.lax.reduce_window(
                images_mask_vert, -jnp.inf, jax.lax.max,
                window_dimensions=(1, 1, kernel_size, 1),
                window_strides=(1, 1, 1, 1), padding='SAME'
            )
            images_mask = images_mask[..., 0]

        # Final weighted mask
        images_mask = jnp.where(images_mask == 0.0, bg_weight, 1.0)

        return images, images_mask

class HetSIREN(nnx.Module):

    @save_config
    def __init__(self, lat_dim, reference_volume, reconstruction_mask, coords, values, xsize, sr, d_hid=100, bank_size=1024, ctf_type="apply",
                 sigma=1.0, decoupling=False, isVae=False, transport_mass=False, local_reconstruction=False, architecture="convnn",
                 is_implicit=True, isTomoSIREN=False, *, rngs: nnx.Rngs):
        super(HetSIREN, self).__init__()
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.decoupling = decoupling if not isTomoSIREN else False
        self.isVae = isVae
        self.isTomoSIREN = isTomoSIREN
        self.local_reconstruction = local_reconstruction
        self.reference_volume = reference_volume
        self.reconstruction_mask = reconstruction_mask.astype(float)
        self.coords = jnp.array(coords)
        self.d_hid = d_hid
        self.lat_dim = lat_dim
        self.has_reference_volume = not bool(np.all(reference_volume == 0.0))
        self.encoder = MultiEncoder(self.xsize, d_hid, lat_dim, n_layers=3, isVae=isVae, architecture=architecture, isTomoSIREN=isTomoSIREN, rngs=rngs) \
            if decoupling or isTomoSIREN else Encoder(self.xsize, lat_dim, isVae=isVae, architecture=architecture, rngs=rngs)
        self.delta_volume_decoder = DeltaVolumeDecoder(self.coords.shape[0], lat_dim, self.xsize, self.coords, values, transport_mass=transport_mass, is_implicit=is_implicit, rngs=rngs)

        self.phys_decoder = PhysDecoder(self.xsize, sr, transport_mass=transport_mass)

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.subset_size = min(2048, bank_size)

        self.memory_bank = nnx.Variable(
            jnp.zeros((self.bank_size, lat_dim))
        )
        self.memory_bank_ptr = nnx.Variable(
            jnp.zeros((1,), dtype=jnp.int32)
        )

        # Gaussians size
        # self.sigma = nnx.Param(sigma)
        self.sigma = sigma

    def __call__(self, x, rngs=None, **kwargs):
        if self.isVae:
            if self.decoupling:
                (sample, mean, _), (rotations, shifts, _) = self.encoder(x, "encoder_exp", return_last=False, return_alignment_refinement=True, rngs=rngs)
            else:
                (sample, mean, _), (rotations, shifts, _) = self.encoder(x, return_last=False, return_alignment_refinement=True, rngs=rngs)
            if kwargs.pop("gaussian_sample", False):
                latent = sample
            else:
                latent = mean
        else:
            if self.decoupling:
                latent, (rotations, shifts, _)  = self.encoder(x, "encoder_exp", return_last=False, return_alignment_refinement=True, rngs=rngs)
            else:
                latent, (rotations, shifts, _) = self.encoder(x, return_last=False, return_alignment_refinement=True, rngs=rngs)
        if kwargs.pop("return_alignment_refinement", True):
            return latent, (rotations, shifts)
        else:
            return latent

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

    def decode_image(self, x, labels, md, ctf_type=None, return_latent=False, corrupt_projection_with_ctf=False):
        # Precompute batch alignments
        euler_angles = md["euler_angles"][labels]

        # Precompute batch shifts
        shifts = md["shifts"][labels]

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

        if x.ndim == 4:
            if self.ctf_type == "precorrect":
                # Wiener filter
                x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

            # Encode images
            latents, (rotations_rigid, shifts_rigid) = self(x, return_alignment_refinement=True)
        else:
            latents = x

        # Get rotation matrices
        if euler_angles.ndim == 2:
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        else:
            rotations = euler_angles

        # Decode volumes
        coords, values = self.delta_volume_decoder(latents)

        # Consider alignments if needed
        if x.ndim == 4:
            # Consider refinement and rigid registration alignments
            rotations = jnp.matmul(rotations, rotations_rigid)
            shifts = shifts + shifts_rigid

        # CTF corruption
        if not corrupt_projection_with_ctf:
            ctf_type = None

        # Generate projections
        images_corrected, _ = self.phys_decoder(x, values, coords, self.xsize, rotations, shifts,
                                                self.delta_volume_decoder.centering, ctf, ctf_type, self.sigma)

        if return_latent:
            return images_corrected, latents
        else:
            return images_corrected

    def decode_volume(self, x):
        if x.ndim == 1:
            x = x[None, ...]

        return self.delta_volume_decoder.decode_volume(x=x, filter=True)

    def decode_field(self, x):
        if x.ndim == 4:
            x, _ = self(x)

        coords, values = self.delta_volume_decoder(x)

        inital_coords = self.delta_volume_decoder.scale * self.delta_volume_decoder.coords
        field = coords - inital_coords

        return (field / (0.5 * self.xsize), inital_coords / (0.5 * self.xsize))


@partial(jax.jit, static_argnames=("do_update", "l1_lambda", "graph_lambda"))
def train_step_hetsiren(graphdef, state, x, labels, md, key, do_update=True, l1_lambda=1e-4, graph_lambda=1e-4):
    model, optimizer = nnx.merge(graphdef, state)
    distributions_key, rot_sample_key, choice_key, key = jnr.split(key, 4)

    # TODO: Explore sampling the posterior with M>1
    M = 1

    if M > 1:
        # VMAP functions
        phys_decoder = jax.vmap(model.phys_decoder, in_axes=(None, None, None, None, 1, None, None, None, None, None, None), out_axes=1)
        wiener2DFilter_vmap = jax.vmap(wiener2DFilter, in_axes=(1, None, None), out_axes=1)
        ctfFilter_vmap = jax.vmap(ctfFilter, in_axes=(1, None, None), out_axes=1)
    else:
        phys_decoder = model.phys_decoder
        wiener2DFilter_vmap = wiener2DFilter
        ctfFilter_vmap = ctfFilter

    # sparse_finite_3D_differences_field = jax.vmap(sparse_finite_3D_differences, in_axes=(-1, None, None), out_axes=-1)
    calculate_deformation_regularity_loss_batch = jax.vmap(calculate_deformation_regularity_loss, in_axes=(0, None, None, None))
    calculate_repulsion_loss_batch = jax.vmap(calculate_repulsion_loss, in_axes=(0, None, None))

    def loss_fn(model, x):
        # Check if Tomo mode
        if model.isTomoSIREN:
            (x, subtomogram_label) = x

        # Prepare input images for encoder
        # if model.ctf_type != "precorrect":
        #     x_in = wiener2DFilter(x[..., 0], ctf, pad_factor=pad_factor)[..., None]
        # else:
        #     x_in = x
        # x_in = apply_batch_translations(x_in, shifts)
        x_in = x

        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                (sample, latent, logstd), prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
                (_, latent_1, _), (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out_random = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            else:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid, rotations_logscale) = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key)
        else:
            if model.decoupling:
                latent, (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                latent, prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
                latent_1, (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out_random = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            else:
                latent, (rotations_rigid, shifts_rigid, rotations_logscale) = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key)

        # Decode volumes
        if model.isVae:
            coords, values = model.delta_volume_decoder(sample)
        else:
            coords, values = model.delta_volume_decoder(latent)

        # Get rotation matrices
        if euler_angles.ndim == 2:
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        else:
            rotations = euler_angles

        # Rotation posterior scheduling
        # min_log_scale = jnp.log(0.03) * max(0.0, 1.0 - steps_accum/30000)
        rotations_logscale = jnp.clip(rotations_logscale, jnp.log(0.03), 2.0)

        # Sample new rotations
        if M > 1:
            # Consider refinement and rigid registration alignments (for delta_volume_decoder_rigid output)
            rotations_refined = jnp.matmul(rotations, rotations_rigid)

            rotations_refined, omegas, log_q = sample_topM_R(rot_sample_key, rotations_refined, rotations_logscale, M=M)
        else:
            # Consider refinement and rigid registration alignments (for delta_volume_decoder_rigid output)
            rotations_refined = jnp.matmul(rotations, rotations_rigid)
        shifts_refined = shifts + shifts_rigid

        # Only rigid part: coords and values
        reference_values = model.delta_volume_decoder.reference_values

        # Centering
        centering = model.delta_volume_decoder.centering

        # Generate projections
        if model.has_reference_volume:
            images_corrected, _ = phys_decoder(x, values, jax.lax.stop_gradient(coords), model.xsize, rotations_refined, shifts_refined,
                                               centering, ctf, model.ctf_type, model.sigma, 0.0)
            images_corrected_field, _ = phys_decoder(x, reference_values, coords, model.xsize, rotations_refined, shifts_refined,
                                                     centering, ctf, model.ctf_type, model.sigma, 0.0)
        else:
            images_corrected, _ = phys_decoder(x, values, coords, model.xsize, rotations_refined, shifts_refined,
                                               centering, ctf, model.ctf_type, model.sigma, 0.0)
            images_corrected_field = images_corrected

        if not model.delta_volume_decoder.transport_mass:
            consensus_coords, consensus_values = model.delta_volume_decoder(jnp.zeros_like(latent))
            images_consensus, _ = phys_decoder(x, consensus_values, consensus_coords, model.xsize, rotations_refined, shifts_refined,
                                               centering, ctf, model.ctf_type, model.sigma, 0.0)

        # Projection "mask" in case of no mass transport
        if not model.delta_volume_decoder.transport_mass and model.local_reconstruction:
            _, projected_mask = phys_decoder(x, jnp.ones_like(values), jax.lax.stop_gradient(coords), model.xsize,
                                             rotations_refined, shifts_refined, centering, ctf, None, model.sigma, False, 0.0)
        else:
            projected_mask = jnp.ones_like(x)[..., 0]

        if M > 1:
            projected_mask = projected_mask[:, None, ...]

        # Losses
        images_corrected = jnp.squeeze(images_corrected)
        images_corrected_field = jnp.squeeze(images_corrected_field)
        x = jnp.squeeze(x)
        if not model.delta_volume_decoder.transport_mass:
            images_consensus = jnp.squeeze(images_consensus)

        # Consider CTF if Wiener mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_loss = wiener2DFilter_vmap(images_corrected, ctf, pad_factor)
            images_corrected_field_loss = wiener2DFilter_vmap(images_corrected_field, ctf, pad_factor)
            if not model.delta_volume_decoder.transport_mass:
                images_consensus_loss = wiener2DFilter_vmap(images_consensus, ctf, pad_factor)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_loss = ctfFilter_vmap(images_corrected, ctf, pad_factor)
            images_corrected_field_loss = ctfFilter_vmap(images_corrected_field, ctf, pad_factor)
            if not model.delta_volume_decoder.transport_mass:
                images_consensus_loss = ctfFilter_vmap(images_consensus, ctf, pad_factor)
        else:
            x_loss = x
            images_corrected_loss = images_corrected
            images_corrected_field_loss = images_corrected_field
            if not model.delta_volume_decoder.transport_mass:
                images_consensus_loss = images_consensus

        if M > 1:
            x_loss = x_loss[:, None, ...]

        # Projection mask
        x_loss = x_loss * projected_mask
        images_corrected_loss = images_corrected_loss * projected_mask
        images_corrected_field_loss = images_corrected_field_loss * projected_mask
        if not model.delta_volume_decoder.transport_mass:
            images_consensus_loss = images_consensus_loss * projected_mask

        # recon_loss = dm_pix.mae(images_corrected_loss[..., None], x_loss[..., None]).mean()
        recon_loss = 0.1 * mse(images_corrected_loss[..., None], x_loss[..., None]) + 0.9 * mse(images_corrected_field_loss[..., None], x_loss[..., None])
        if model.delta_volume_decoder.transport_mass:
            recons_loss_all = recon_loss.mean()
        else:
            recons_loss_all = 0.5 * (recon_loss.mean() + mse(images_consensus_loss[..., None], x_loss[..., None]).mean())

        # L1 based denoising
        l1_loss = jnp.mean(jnp.abs(values))

        # L1 and L2 total variation (old version - no sparse)
        # diff_x = volumes[:, 1:, :, :] - volumes[:, :-1, :, :]
        # diff_y = volumes[:, :, 1:, :] - volumes[:, :, :-1, :]
        # diff_z = volumes[:, :, :, 1:] - volumes[:, :, :, :-1]
        # l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        # l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        # Values
        # diff_x, diff_y, diff_z = sparse_finite_3D_differences(values, model.inds, model.xsize)
        # l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        # l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        # Field
        # if model.has_reference_volume and model.delta_volume_decoder.transport_mass:
        #     diff_field_x, diff_field_y, diff_field_z = sparse_finite_3D_differences_field(field, model.inds, model.xsize)
        #     l1_grad_field_loss = jnp.abs(diff_field_x).mean() + jnp.abs(diff_field_z).mean() + jnp.abs(diff_field_y).mean()
        #     l2_grad_field_loss = jnp.square(diff_field_x).mean() + jnp.square(diff_field_z).mean() + jnp.square(diff_field_y).mean()
        # else:
        #     l1_grad_field_loss = 0.0
        #     l2_grad_field_loss = 0.0

        # Centering loss
        if model.has_reference_volume and model.delta_volume_decoder.transport_mass:
            factor = 0.5 * model.xsize
            coords_cm = (coords + centering - factor) / factor
            cm = jnp.average(coords_cm, weights=jnp.broadcast_to(values[..., None], coords.shape), axis=1)
            loss_cm = jnp.linalg.norm(cm, axis=1).mean()
        else:
            loss_cm = 0.0

        # Local distance preservation
        if model.isVae:
            coords_mean, values_mean = model.delta_volume_decoder(latent)
            loss_dp = jnp.abs(values[..., None] * coords / model.delta_volume_decoder.scale
                              - values_mean[..., None] * coords_mean / model.delta_volume_decoder.scale).mean()
        else:
            loss_dp = 0.0

        # Chimeric volume losses
        if model.local_reconstruction:
            outsize_mask = (1. - model.reconstruction_mask).astype(jnp.bool)
            hist_loss = (jnp.square(values.max(axis=1) - model.reference_volume.max(where=outsize_mask, initial=model.reference_volume.max())[None, ...]).mean() +
                         jnp.square(values.min(axis=1) - model.reference_volume.min(where=outsize_mask, initial=model.reference_volume.min())[None, ...]).mean() +
                         jnp.square(values.mean(axis=1) - model.reference_volume.mean(where=outsize_mask)[None, ...]).mean() +
                         jnp.square(values.std(axis=1) - model.reference_volume.std(where=outsize_mask)[None, ...]).mean())
        else:
            hist_loss = 0.0

        # Variational loss
        if model.isVae:
            # KL divergence loss
            kl_loss = -0.5 * jnp.sum(1. + 2. * logstd - jnp.square(jnp.exp(logstd)) - jnp.square(latent))
        else:
            kl_loss = 0.0

        # Variational loss (poses)
        if M > 1:
            w_pose, _ = importance_weights(recons_loss_all, log_q)
            nll = jnp.sum(w_pose * recons_loss_all).mean()
            kl_pose = PoseDistMatrix.kl_to_isotropic_prior(rotations_logscale, prior_log_scale=0.0).mean()
        else:
            nll = recons_loss_all.mean()
            kl_pose = 0.0

        # Graph based loss
        if model.has_reference_volume and model.delta_volume_decoder.transport_mass:
            # Get data
            consensus_distances = model.delta_volume_decoder.consensus_distances
            deformed_positions = coords / model.delta_volume_decoder.scale
            radius_graph = model.delta_volume_decoder.edge_index
            edge_weights = model.delta_volume_decoder.edge_weights
            tau = model.delta_volume_decoder.tau

            # Losses
            loss_def_regularity = calculate_deformation_regularity_loss_batch(deformed_positions, radius_graph,
                                                                              consensus_distances, edge_weights)
            loss_repulsion = calculate_repulsion_loss_batch(deformed_positions, radius_graph, tau)

            # Total loss
            loss_graph = (loss_def_regularity + 0.01 * loss_repulsion).mean()
        else:
            loss_graph = 0.0

        # Decoupling
        if model.decoupling or model.isTomoSIREN:
            if not model.isTomoSIREN:
                rotations_random_matrix = euler_matrix_batch(rotations_random[:, 0], rotations_random[:, 1], rotations_random[:, 2])
                if M > 1:
                    images_corrected = images_corrected[:, 0, ...]
                    rotations_random_refined = jnp.matmul(rotations_random_matrix, jax.lax.stop_gradient(rotations_rigid))
                else:
                    rotations_random_refined = jnp.matmul(rotations_random_matrix, jax.lax.stop_gradient(rotations_rigid))
                shifts_random_refined = jnp.zeros_like(shifts_refined)
                images_random, _ = model.phys_decoder(x, values, coords, model.xsize, rotations_random_refined, shifts_random_refined,
                                                      centering, ctf_random, None, model.sigma, 0.0)
                if model.isVae:
                    (_, latent_1, _), prev_layer_out_random = model.encoder(images_corrected[..., None], "encoder_dec", return_last=True, return_alignment_refinement=False, rngs=distributions_key)
                    (_, latent_2, _) = model.encoder(images_random[..., None], "encoder_dec", return_alignment_refinement=False, rngs=distributions_key)
                else:
                    latent_1, prev_layer_out_random = model.encoder(images_corrected[..., None], "encoder_dec", return_last=True, return_alignment_refinement=False, rngs=distributions_key)
                    latent_2 = model.encoder(images_random[..., None], "encoder_dec", return_alignment_refinement=False, rngs=distributions_key)
                decoupling_loss = (jnp.mean(jnp.square(latent - latent_1), axis=-1).mean() +
                                   jnp.mean(jnp.square(latent - latent_2), axis=-1).mean() +
                                   jnp.mean(jnp.square(prev_layer_out - prev_layer_out_random), axis=-1).mean())
            else:
                decoupling_loss = (jnp.mean(jnp.square(latent - latent_1), axis=-1).mean() +
                                   jnp.mean(jnp.square(prev_layer_out - prev_layer_out_random), axis=-1).mean())

            random_indices = jnr.choice(choice_key, a=jnp.arange(model.bank_size), shape=(model.subset_size,), replace=False)
            memory_bank_subset = model.memory_bank[random_indices]

            dist = jnp.pow(latent[:, None, :] - memory_bank_subset, 2.).sum(axis=-1)
            dist_nn, _ = jax.lax.approx_min_k(dist, k=10, recall_target=0.95)
            dist_fn, _ = jax.lax.approx_max_k(dist, k=10, recall_target=0.95)

            decoupling_loss += 1.0 * triplet_loss(dist_nn, dist_fn, reduction="mean", margin=0.01)

        else:
            decoupling_loss = 0.0

        loss = (nll + 0.000001 * kl_loss + 0.000001 * kl_pose + 0.0001 * decoupling_loss
                + l1_lambda * l1_loss + graph_lambda * loss_graph + 100. * hist_loss + 0.0001 * loss_dp + 0.0001 * loss_cm)
        return loss, (recon_loss.mean(), latent)

    # Check if Tomo mode
    if model.isTomoSIREN:
        (x, subtomogram_label) = x

    # Precompute batch aligments
    euler_angles = md["euler_angles"][labels]

    # Precompute batch shifts
    shifts = md["shifts"][labels]

    # Precompute batch CTFs
    pad_factor = model.phys_decoder.pad_factor
    if model.ctf_type is not None:
        defocusU = md["ctfDefocusU"][labels]
        defocusV = md["ctfDefocusV"][labels]
        defocusAngle = md["ctfDefocusAngle"][labels]
        cs = md["ctfSphericalAberration"][labels]
        kv = md["ctfVoltage"][labels][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         model.sr, [pad_factor * model.xsize, int(pad_factor * 0.5 * model.xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], pad_factor * model.xsize, int(pad_factor * 0.5 * model.xsize + 1)], dtype=x.dtype)

    if model.ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

    # Prepare data for decoupling encoder
    rotations_random = jnr.choice(distributions_key, euler_angles, axis=0, shape=(x.shape[0],), replace=False)
    if model.ctf_type == "apply":
        defocusU = jnr.choice(distributions_key, defocusU, axis=0, shape=(x.shape[0],), replace=False)
        defocusV = jnr.choice(distributions_key, defocusV, axis=0, shape=(x.shape[0],), replace=False)
        defocusAngle = jnr.choice(distributions_key, defocusAngle, axis=0, shape=(x.shape[0],), replace=False)
        ctf_random = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                                model.sr, [pad_factor * model.xsize, int(pad_factor * 0.5 * model.xsize + 1)],
                                x.shape[0], True)
    else:
        ctf_random = jnp.ones([x.shape[0], pad_factor * model.xsize, int(pad_factor * 0.5 * model.xsize + 1)], dtype=x.dtype)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    if model.isTomoSIREN:
        (loss, (recon_loss, latent)), grads = grad_fn(model, (x, subtomogram_label))
    else:
        (loss, (recon_loss, latent)), grads = grad_fn(model, x)

    if do_update:
        optimizer.update(model, grads)

        # Update memory bank
        model.enqueue(latent)

        state = nnx.state((model, optimizer))

        return loss, recon_loss, state, key
    else:
        return loss, recon_loss


@jax.jit
def gradient_for_recon_graph_losses(graphdef, state, x, labels, md, key):
    model, optimizer = nnx.merge(graphdef, state)

    distributions_key, key = jax.random.split(key, 2)

    phys_decoder = model.phys_decoder
    wiener2DFilter_vmap = wiener2DFilter
    ctfFilter_vmap = ctfFilter

    calculate_deformation_regularity_loss_batch = jax.vmap(calculate_deformation_regularity_loss, in_axes=(0, None, None, None))
    calculate_repulsion_loss_batch = jax.vmap(calculate_repulsion_loss, in_axes=(0, None, None))

    def predict_latent_from_images(model, x):
        # Check if Tomo mode
        if model.isTomoSIREN:
            (x, subtomogram_label) = x

        # Prepare input images for encoder
        # if model.ctf_type != "precorrect":
        #     x_in = wiener2DFilter(x[..., 0], ctf, pad_factor=pad_factor)[..., None]
        # else:
        #     x_in = x
        # x_in = apply_batch_translations(x_in, shifts)
        x_in = x

        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (latent, _, _), (rotations_rigid, shifts_rigid, _), _ = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                (latent, _, _), _ = model.encoder(subtomogram_label, "encoder_dec", return_last=True, rngs=distributions_key)
                _, (rotations_rigid, shifts_rigid, _), _ = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            else:
                (latent, _, _), (rotations_rigid, shifts_rigid, _) = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key)
        else:
            if model.decoupling:
                latent, (rotations_rigid, shifts_rigid, _), _ = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                latent, _ = model.encoder(subtomogram_label, "encoder_dec", return_last=True, rngs=distributions_key)
                _, (rotations_rigid, shifts_rigid, _), _ = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            else:
                latent, (rotations_rigid, shifts_rigid, _) = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key)

        return latent, rotations_rigid, shifts_rigid


    def loss_representation_fn(model, x, latent, rotations_rigid, shifts_rigid):
        # Decode volumes
        coords, values = model.delta_volume_decoder(latent)

        # Get rotation matrices
        if euler_angles.ndim == 2:
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        else:
            rotations = euler_angles

        rotations_refined = jnp.matmul(rotations, rotations_rigid)
        shifts_refined = shifts + shifts_rigid

        # Reference values
        reference_values = model.delta_volume_decoder.reference_values

        # Centering
        centering = model.delta_volume_decoder.centering

        # Generate projections
        if model.has_reference_volume:
            images_corrected_field, _ = phys_decoder(x, reference_values, coords, model.xsize, rotations_refined,
                                                     shifts_refined, centering, ctf, model.ctf_type, model.sigma, 0.0)
        else:
            images_corrected_field, _ = phys_decoder(x, values, coords, model.xsize, rotations_refined, shifts_refined,
                                                     centering, ctf, model.ctf_type, model.sigma, 0.0)

        # Projection "mask" in case of no mass transport
        if not model.delta_volume_decoder.transport_mass and model.local_reconstruction:
            _, projected_mask = phys_decoder(x, jnp.ones_like(values), jax.lax.stop_gradient(coords), model.xsize,
                                             rotations_refined, shifts_refined, centering, ctf, None, model.sigma, False, 0.0)
        else:
            projected_mask = jnp.ones_like(x)[..., 0]

        # Losses
        images_corrected_field = jnp.squeeze(images_corrected_field)
        x = jnp.squeeze(x)

        # Consider CTF if Wiener mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_field_loss = wiener2DFilter_vmap(images_corrected_field, ctf, pad_factor)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_field_loss = ctfFilter_vmap(images_corrected_field, ctf, pad_factor)
        else:
            x_loss = x
            images_corrected_field_loss = images_corrected_field

        # Projection mask
        x_loss = x_loss * projected_mask
        images_corrected_field_loss = images_corrected_field_loss * projected_mask

        recon_loss = 0.9 * mse(images_corrected_field_loss[..., None], x_loss[..., None])

        return recon_loss.mean()

    def loss_graph_fn(model, latent):
        # Decode volumes
        coords, values = model.delta_volume_decoder(latent)

        # Graph based loss
        if model.has_reference_volume and model.delta_volume_decoder.transport_mass:
            # Get data
            consensus_distances = model.delta_volume_decoder.consensus_distances
            deformed_positions = coords / model.delta_volume_decoder.scale
            radius_graph = model.delta_volume_decoder.edge_index
            edge_weights = model.delta_volume_decoder.edge_weights
            tau = model.delta_volume_decoder.tau

            # Losses
            loss_def_regularity = calculate_deformation_regularity_loss_batch(deformed_positions, radius_graph,
                                                                              consensus_distances, edge_weights)
            loss_repulsion = calculate_repulsion_loss_batch(deformed_positions, radius_graph, tau)

            # Total loss
            loss_graph = (loss_def_regularity + 0.01 * loss_repulsion).mean()
        else:
            loss_graph = 0.0

        return loss_graph

    # Check if Tomo mode
    if model.isTomoSIREN:
        (x, subtomogram_label) = x

    # Precompute batch aligments
    euler_angles = md["euler_angles"][labels]

    # Precompute batch shifts
    shifts = md["shifts"][labels]

    # Precompute batch CTFs
    pad_factor = model.phys_decoder.pad_factor
    if model.ctf_type is not None:
        defocusU = md["ctfDefocusU"][labels]
        defocusV = md["ctfDefocusV"][labels]
        defocusAngle = md["ctfDefocusAngle"][labels]
        cs = md["ctfSphericalAberration"][labels]
        kv = md["ctfVoltage"][labels][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         model.sr, [pad_factor * model.xsize, int(pad_factor * 0.5 * model.xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], pad_factor * model.xsize, int(pad_factor * 0.5 * model.xsize + 1)], dtype=x.dtype)

    if model.ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

    # Get latent vectors and alignments
    latent, rotations_rigid, shifts_rigid = predict_latent_from_images(model, x)
    latent = jax.lax.stop_gradient(latent)
    rotations_rigid = jax.lax.stop_gradient(rotations_rigid)
    shifts_rigid = jax.lax.stop_gradient(shifts_rigid)
    x = jax.lax.stop_gradient(x)

    grads_data = nnx.grad(loss_representation_fn)(model, x, latent, rotations_rigid, shifts_rigid)
    grads_reg = nnx.grad(loss_graph_fn)(model, latent)

    params_filter = nnx.All(nnx.Param, nnx.PathContains('hidden_coords'))
    grads_data, _ = grads_data.split(params_filter, ...)
    grads_reg, _ = grads_reg.split(params_filter, ...)

    # Calculate Global Norms
    def global_norm(g):
        leaves = jax.tree_util.tree_leaves(g)
        return jnp.sqrt(sum(jnp.sum(jnp.square(l)) for l in leaves))

    # norm_data = optax.global_norm(grads_data)
    # norm_reg = optax.global_norm(grads_reg)
    norm_data = global_norm(grads_data)
    norm_reg = global_norm(grads_reg)

    return norm_data, norm_reg


@jax.jit
def validation_step_hetsiren(graphdef, state, x, labels, md, key):
    model, optimizer = nnx.merge(graphdef, state)

    distributions_key, key = jax.random.split(key, 2)

    def loss_fn(model, x):
        # Check if Tomo mode
        if model.isTomoSIREN:
            (x, subtomogram_label) = x

        # Prepare input images for encoder
        # if model.ctf_type != "precorrect":
        #     x_in = wiener2DFilter(x[..., 0], ctf, pad_factor=pad_factor)[..., None]
        # else:
        #     x_in = x
        # x_in = apply_batch_translations(x_in, shifts)
        x_in = x

        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                (sample, latent, logstd), prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True, rngs=distributions_key)
            else:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid, rotations_logscale) = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key)
        else:
            if model.decoupling:
                latent, (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                latent, prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True, rngs=distributions_key)
            else:
                latent, (rotations_rigid, shifts_rigid, rotations_logscale) = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key)

        # Decode volumes
        if model.isVae:
            coords, values = model.delta_volume_decoder(sample)
        else:
            coords, values = model.delta_volume_decoder(latent)

        # Get rotation matrices
        if euler_angles.ndim == 2:
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        else:
            rotations = euler_angles

        # Consider refinement and rigid registration alignments (for delta_volume_decoder_rigid output)
        rotations_refined = jnp.matmul(rotations, rotations_rigid)
        shifts_refined = shifts + shifts_rigid

        # Centering
        centering = model.delta_volume_decoder.centering

        # Generate projections
        if model.has_reference_volume:
            reference_values = model.delta_volume_decoder.reference_values
            images_corrected, _ = model.phys_decoder(x, values, coords, model.xsize, rotations_refined,
                                                     shifts_refined, centering, ctf, model.ctf_type, model.sigma, 0.0)
            images_corrected_field, _ = model.phys_decoder(x, reference_values, coords, model.xsize, rotations_refined,
                                                           shifts_refined, centering, ctf, model.ctf_type, model.sigma, 0.0)
        else:
            images_corrected, _ = model.phys_decoder(x, values, coords, model.xsize, rotations_refined, shifts_refined,
                                                     centering, ctf, model.ctf_type, model.sigma, 0.0)
            images_corrected_field = images_corrected

        # Projection "mask" in case of no mass transport
        if not model.delta_volume_decoder.transport_mass and model.local_reconstruction:
            _, projected_mask = model.phys_decoder(x, jnp.ones_like(values), coords, model.xsize,
                                                   rotations_refined, shifts_refined, centering, ctf, None, model.sigma,
                                                   False, 0.0)
        else:
            projected_mask = jnp.ones_like(x)[..., 0]

        # Losses
        images_corrected = jnp.squeeze(images_corrected)
        images_corrected_field = jnp.squeeze(images_corrected_field)
        x = jnp.squeeze(x)

        # Consider CTF if Wiener mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_loss = wiener2DFilter(images_corrected, ctf, pad_factor)
            images_corrected_field_loss = wiener2DFilter(images_corrected_field, ctf, pad_factor)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_loss = ctfFilter(images_corrected, ctf, pad_factor)
            images_corrected_field_loss = ctfFilter(images_corrected_field, ctf, pad_factor)
        else:
            x_loss = x
            images_corrected_loss = images_corrected
            images_corrected_field_loss = images_corrected_field

        # Projection mask
        x_loss = x_loss * projected_mask
        images_corrected_loss = images_corrected_loss * projected_mask
        images_corrected_field_loss = images_corrected_field_loss * projected_mask

        recon_loss = 0.1 * mse(images_corrected_loss[..., None], x_loss[..., None]) + 0.9 * mse(images_corrected_field_loss[..., None], x_loss[..., None])

        return recon_loss.mean()

    # Check if Tomo mode
    if model.isTomoSIREN:
        (x, subtomogram_label) = x

    # Precompute batch aligments
    euler_angles = md["euler_angles"][labels]

    # Precompute batch shifts
    shifts = md["shifts"][labels]

    # Precompute batch CTFs
    pad_factor = model.phys_decoder.pad_factor
    if model.ctf_type is not None:
        defocusU = md["ctfDefocusU"][labels]
        defocusV = md["ctfDefocusV"][labels]
        defocusAngle = md["ctfDefocusAngle"][labels]
        cs = md["ctfSphericalAberration"][labels]
        kv = md["ctfVoltage"][labels][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         model.sr, [pad_factor * model.xsize, int(pad_factor * 0.5 * model.xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], pad_factor * model.xsize, int(pad_factor * 0.5 * model.xsize + 1)], dtype=x.dtype)

    if model.ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

    if model.isTomoSIREN:
        loss = loss_fn(model, (x, subtomogram_label))
    else:
        loss = loss_fn(model, x)

    return loss




def main():
    import os
    import sys
    from tqdm import tqdm
    import random
    import numpy as np
    import argparse
    import shutil
    from xmipp_metadata.image_handler import ImageHandler
    import optax
    from contextlib import closing
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, extract_columns, NumpyGenerator
    from hax.networks import train_step_hetsiren
    from hax.metrics import JaxSummaryWriter
    from hax.programs.gaussian_volume_fitting import fit_volume, adjust_weights_to_images, splat_weights_bilinear
    # from hax.schedulers import CosineAnnealingScheduler

    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, type=str,
                        help=f"Xmipp/Relion metadata file with the images (+ alignments / CTF) to be analyzed. {bcolors.WARNING}NOTE{bcolors.ENDC}: If the metadata "
                             f"includes the label {bcolors.ITALIC}subtomo_labels{bcolors.ENDC}, then TomoSIREN network will be trained (i.e., HetSIREN for "
                             f"tomography data). In this case, the parameter {bcolors.ITALIC}decoupling{bcolors.ENDC} will not have any effect as pose/CTF "
                             f"decoupling is already handled by TomoSIREN.")
    parser.add_argument("--vol", required=False, type=str,
                        help="If provided, the neural network will learn how to refine this volume towards the heterogeneous states present in the images")
    parser.add_argument("--mask", required=False, type=str,
                        help=f"HetSIREN reconstruction mask (useful to focus the reconstruction/estimated landscape on a specific region of interest - the mask provided "
                             f"must be binary -  {bcolors.WARNING}NOTE{bcolors.ENDC}: since this is a reconstruction mask, it should be defined such that it covers the "
                             f"volume were the motions of interest are expected to happen)")
    parser.add_argument("--load_images_to_ram", action='store_true',
                        help=f"If provided, images will be loaded to RAM. This is recommended if you want the best performance and your dataset fits in your RAM memory. If this flag is not provided, "
                             f"images will be memory mapped. When this happens, the program will trade disk space for performance. Thus, during the execution additional disk space will be used and the performance "
                             f"will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal once the execution has finished.")
    parser.add_argument("--sr", required=True, type=float,
                        help="Sampling rate of the images/volume")
    parser.add_argument("--lat_dim", required=False, type=int, default=8,
                        help="Dimensionality of the latent space of the network (set by default to 8)")
    parser.add_argument("--transport_mass", action='store_true',
                        help='When set, HetSIREN will be able to "move" the mass inside the mask instead of just reconstructing the volume. This implies that HetSIREN will estimate the motion '
                             'to be applied to the points within the provided mask, instead of considering them fixed in space. This approach is useful when working with large box sizes that '
                             'do not fit in GPU memory, or when a more through analysis of motions is desired. '
                             f'{bcolors.WARNING}NOTE{bcolors.ENDC}: When this option is set and a reference volume is provided, we recommend changing the reference mask to a tight mask computed '
                             f'from the reference volume. This mask now tells the program which regions should be moved. Therefore, consider providing a mask that covers all the protein regions you would like '
                             f'to be analyzed by HetSIREN.')
    parser.add_argument("--num_gaussians", required=False, type=int,
                        help="Before training the network, HetSIREN will try to fit a set of Gaussians in the reference volume to recreate it. "
                            "The default criterium is to automatically determine the number of Gaussians needed to reproduce the reference volume "
                            "with high-fidelity. However, if you prefer to fix the number of Gaussians in advance based on your own criterium (e.g., "
                            "the number of residues in your protein), you can set this parameter. When set, the HetSIREN will fit this fixed number of Gaussians "
                            "so that the reproduce the reference volume as well as possible.")
    parser.add_argument("--sharpening", required=False, action='store_true',
                        help='')
    parser.add_argument("--local_reconstruction", action='store_true',
                        help=f'When set, HetSIREN will turn to local heterogeneous reconstruction/refinement mod, focusing the analysis of heterogeneity to a region of interest enclosed by the provided refernece mask. '
                             f'{bcolors.WARNING}WARNING{bcolors.ENDC}: IF PROVIDED, TRANSPORT MASS WILL BE OVERRIDDEN AND NOT CONSIDERED. '
                             f'{bcolors.WARNING}WARNING{bcolors.ENDC}: IF PROVIDED, HAVING A REFERENCE VOLUME IS MANDATORY. OTHERWISE, THIS PARAMETER WILL BE NEGLECTED. ')
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
                             f"is the size of the step it takes on each attempt. A large {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}0.01{bcolors.ENDC}) is like taking huge leaps ? it's fast but can be unstable, "
                             f"overshoot the lowest point, or cause {bcolors.ITALIC}NAN{bcolors.ENDC} errors. A small {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}1e-6{bcolors.ENDC}) is like taking tiny "
                             f"shuffles ? it's stable but very slow and might get stuck before reaching the bottom. A good default is often {bcolors.ITALIC}0.0001{bcolors.ENDC}. If training fails or errors explode, "
                             f"try making the {bcolors.ITALIC}lr{bcolors.ENDC} 10 times smaller (e.g., {bcolors.ITALIC}0.001{bcolors.ENDC} --> {bcolors.ITALIC}0.0001{bcolors.ENDC}).")
    parser.add_argument("-dataset_split_fraction", required=False, type=list_of_floats, default=[0.8, 0.2],
                        help=f"Here you can provide the fractions to split your data automatically into a training and a validation subset following the format: {bcolors.ITALIC}training_fraction{bcolors.ENDC},"
                             f"{bcolors.ITALIC}validation_fraction{bcolors.ENDC}. While the training subset will be used to train/update the network parameters, the validation subset will only be used to evaluate the "
                             f"accuracy of the network when faced with new data. Therefore, the validation subset will never be used to update the networks parameters. {bcolors.WARNING}NOTE{bcolors.ENDC}: the sum of "
                             f"{bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} must be equal to one.")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, new metadata...)")
    parser.add_argument("--reload", required=False, type=str,
                        help=f"Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data - "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: If a reference volume was provided, HetSIREN also learns a gray level adjustment. In this case, "
                             f"reload must be the path to a folder containing two additional folders called: {bcolors.UNDERLINE}HetSIREN{bcolors.ENDC} and {bcolors.UNDERLINE}Gaussian_volume_fitting{bcolors.ENDC})")
    parser.add_argument("--denoising_strength", required=False, type=float, default=1e-4,
                        help=f"Determines how strongly HetSIREN will learn to remove noise from the resulting volumes. Increasing the value of this parameter will result in a stronger regularization of the noise, but it may affect the protein "
                             f"signal as well. ({bcolors.WARNING}NOTE{bcolors.ENDC}: We recommend setting this parameter in the range 0.0001 to 0.1)")
    parser.add_argument("--implicit_network", action='store_true',
                        help=f'When set, HetSIREN will use an implicit neural network approach to recover conformational states. Implicit neural networks are more memory consuming, but they are also more accurate in the detection of very local motions. '
                             f'If this architecture is selected, we strongly recommend to set as well the parameter {bcolors.ITALIC}total_mass{bcolors.ENDC} to limit the memory consumption of the network and increase its performance.')
    parser.add_argument("--ssd_scratch_folder", required=False, type=str,
                        help=f"When the parameter {bcolors.UNDERLINE}load_images_to_ram{bcolors.ENDC} is not provided, we strongly recommend to provide here a path to a folder in a SSD disk to read faster the data. If not given, the data will be loaded from "
                             f"the default disk.")
    args, _ = parser.parse_known_args()

    # Manually handed parameters
    local_reconstruction = args.local_reconstruction
    transport_mass = args.transport_mass if not local_reconstruction else False

    # Check that training and validation fractions add up to one
    if sum(args.dataset_split_fraction) != 1:
        raise ValueError(f"The sum of {bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} is not equal one. Please, update the values "
                         f"to fulfill this requirement.")

    # Prepare metadata
    if args.sharpening:
        d_hid = 256
        generator = MetaDataGenerator(args.md, mode="tomo", d_hid=d_hid)
    else:
        d_hid = 100
        generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Check if TomoSIREN is needed
    isTomoSIREN = generator.mode == "tomo"

    # Preprocess volume (and mask)
    if args.vol is not None:
        vol = ImageHandler(args.vol).getData()
    else:
        volume_size = generator.md.getMetaDataImage(0).shape[0]
        local_reconstruction = False
        vol = np.zeros((volume_size, volume_size, volume_size))

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    else:
        if args.transport_mass:
            mask = ImageHandler(args.vol).generateMask(boxsize=64)
        else:
            volume_size = generator.md.getMetaDataImage(0).shape[0]
            mask = ImageHandler().createCircularMask(boxSize=volume_size, is3D=True)

    # If exists, clean MMAP
    # if os.path.isdir(os.path.join(mmap_output_dir, "images_mmap_grain")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap_grain"))

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key, choice_key = jax.random.split(rng, 3)

    # Reload network
    if args.reload is not None:
        hetsiren = NeuralNetworkCheckpointer.load(os.path.join(args.reload, "HetSIREN"))

    # Prepare grain dataset
    if not args.load_images_to_ram and args.mode in ["train", "predict"]:
        mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4, precision=np.float16, group_size=1, shard_size=10000)
    else:
        mmap_output_dir = None

    # Train network
    if args.mode == "train":

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "HetSIREN_metrics"))

        # data_loader_train, data_loader_val = generator.return_grain_dataset(batch_size=args.batch_size, shuffle="global_data_loader", num_epochs=None,
        #                                                                     num_workers=-1, num_threads=1, split_fraction=args.dataset_split_fraction,
        #                                                                     load_to_ram=args.load_images_to_ram)
        # steps_per_epoch = int(int(args.dataset_split_fraction[0] * len(generator.md)) / args.batch_size)
        # steps_per_val = int(int(args.dataset_split_fraction[1] * len(generator.md)) / args.batch_size)
        data_loader_train = generator.return_grain_dataset(batch_size=args.batch_size, shuffle="global_data_loader",
                                                           num_epochs=None, num_workers=-1, num_threads=1,
                                                           load_to_ram=args.load_images_to_ram)
        steps_per_epoch = int(len(generator.md) / args.batch_size)

        # Projector help text in Tensorboard
        legend_projector = """
                <h3>WARNING: Images shown in projector</h3>
                <ul>
                    <li>The pose of the images shown in the projector is random and not related to the real pose of your data. 
                    Therefore, DO NOT consider this images as a representation on how poses are classified in the latent space.</li>
                    <li>WARNING: The projector tab does not update its data even if the Tensorboard page is refreshed. To update the latent 
                    spaces shown in the projector, please, restart completely the Tensorboard server and open it again in the browser.</li>
                </ul>
                """
        writer.add_text("Projector warning", legend_projector)

        if not "hetsiren" in locals():
            if args.vol is not None:
                fit_path = os.path.join(args.output_path, "Gaussian_volume_fitting")
                if not os.path.isdir(os.path.join(fit_path)):
                    # Mask preparation
                    if args.transport_mass:
                        mask_fit = mask
                    else:
                        mask_fit = ImageHandler().generateMask(inputFn=vol, boxsize=64)

                    # Consensus volume
                    if args.num_gaussians is not None:
                        model, _, _ = fit_volume(vol, mask=mask_fit, iterations=20000, learning_rate=0.001, n_init=args.num_gaussians, fixed_gaussians=True)
                    else:
                        model, _, _ = fit_volume(vol, mask=mask_fit, iterations=20000, learning_rate=0.01, grad_threshold=1e-5, densify_interval=2000, n_init=2500)

                    # Adjust to images
                    model, _ = adjust_weights_to_images(model, args.md, mmap_output_dir, args.sr, learning_rate=0.01,
                                                        num_epochs=3, is_global=True, ctf_type=args.ctf_type)

                    # Save model
                    NeuralNetworkCheckpointer.save(model, fit_path)

                    # Save volume
                    vol = np.array(model(place_deltas=True))
                    vol_splatted = np.array(model())
                    ImageHandler().write(vol_splatted, os.path.join(args.output_path, "consensus_volume.mrc"), overwrite=True)
                    ImageHandler().write(vol, os.path.join(args.output_path, "consensus_volume_deltas.mrc"), overwrite=True)
                else:
                    model = NeuralNetworkCheckpointer.load(checkpoint_path=fit_path)
                    vol = np.array(model(place_deltas=True))

            if transport_mass:
                # Prepare network (HetSIREN)
                factor = 0.5 * generator.md.getMetaDataImage(0).shape[0]
                coords = np.array(factor * model.means.get_value() + factor)
                coords = np.stack([coords[..., 2], coords[..., 1], coords[..., 0]], axis=1)
                values = np.array(jax.nn.relu(model.weights.get_value()))
                sigma = jax.nn.relu(model.sigma_param.get_value())
            else:
                inds = np.asarray(np.where(mask > 0.0)).T
                coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
                if args.vol is None:
                    values = jnp.zeros((inds.shape[0],))
                    sigma = 1.0
                else:
                    vol = np.array(model())
                    values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
                    sigma = jax.nn.relu(model.sigma_param.get_value())

            hetsiren = HetSIREN(args.lat_dim, vol, mask, coords, values,
                                generator.md.getMetaDataImage(0).shape[0], args.sr, d_hid=d_hid, sigma=sigma,
                                ctf_type=args.ctf_type, decoupling=True, isVae=True, transport_mass=transport_mass,
                                local_reconstruction=local_reconstruction, bank_size=1024,
                                isTomoSIREN=isTomoSIREN, is_implicit=args.implicit_network,
                                rngs=nnx.Rngs(model_key))
        hetsiren.train()

        # Example of training data for Tensorboard
        x_example, _, labels_example = next(iter(data_loader_train))
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Training data batch", x_example, dataformats="NHWC")

        # Learning rate scheduler
        # total_steps = args.epochs * len(data_loader)
        # lr_schedule = CosineAnnealingScheduler.getScheduler(peak_value=args.learning_rate, total_steps=total_steps, warmup_frac=0.1, end_value=0.0, init_value=1e-5)

        # Optimizers (HetSIREN)
        optimizer = nnx.Optimizer(hetsiren, optax.adamw(args.learning_rate), wrt=nnx.Param)
        graphdef, state = nnx.split((hetsiren, optimizer))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "HetSIREN_CHECKPOINT")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "HetSIREN_CHECKPOINT"), optimizer)
            print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
        else:
            resume_epoch = 0

        # Jitted functions to improve performance
        @partial(jax.jit, static_argnames=["ctf_type", "return_latent", "corrupt_projection_with_ctf"])
        def hetsiren_decode_image(graphdef, state, x, labels, md, ctf_type=None, return_latent=False, corrupt_projection_with_ctf=False):
            model, _ = nnx.merge(graphdef, state)
            return model.decode_image(x, labels, md, ctf_type=ctf_type, return_latent=return_latent,
                                      corrupt_projection_with_ctf=corrupt_projection_with_ctf)

        @jax.jit
        def hetsiren_decode_volume(graphdef, state, x):
            model, _ = nnx.merge(graphdef, state)
            return model.decode_volume(x)

        image_resize = jax.jit(jax.image.resize, static_argnames=("shape", "method"))

        # Training loop (HetSIREN)
        print(f"{bcolors.OKCYAN}\n###### Training variability... ######")

        i = 0
        pbar = tqdm(range(resume_epoch * steps_per_epoch, args.epochs * steps_per_epoch), file=sys.stdout, ascii=" >=", colour="green",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        iter_data_loader_train = iter(data_loader_train)
        for total_steps in pbar:
            if isTomoSIREN:
                (x, subtomo_labels, labels) = next(iter_data_loader_train)
            else:
                (x, _, labels) = next(iter_data_loader_train)

            if total_steps % steps_per_epoch == 0:
                total_loss = 0
                total_recon_loss = 0
                total_validation_loss = 0

                # Compute graph lambda
                graph_lambda = 0.9
                # num_warmup_epochs = 3
                # if i < num_warmup_epochs:
                #     graph_lambda = 1.0
                # else:
                #     pbar.set_description(f"{bcolors.WARNING}Computing graph loss lambda{bcolors.ENDC}")
                #     grad_norm_data, grad_norm_reg = 0.0, 0.0
                #     for _ in range(int(0.1 * steps_per_epoch)):
                #         (x_graph, labels_graph) = next(iter_data_loader_train)
                #         grad_norm_data_step, grad_norm_reg_step = gradient_for_recon_graph_losses(graphdef, state, x_graph, labels_graph, md_columns, rng)
                #         grad_norm_data += np.array(grad_norm_data_step)
                #         grad_norm_reg += np.array(grad_norm_reg_step)
                #         pbar.set_postfix_str(f"graph_lambda={0.9 * (grad_norm_data / grad_norm_reg):.5f}")
                #     graph_lambda = 0.9 * (grad_norm_data / grad_norm_reg)

                # For progress bar (TQDM)
                step = 1
                step_validation = 1
                pbar.set_description(f"Epoch {int(total_steps / steps_per_epoch + 1)}/{args.epochs}")

                # Log intermediate results at the begining of the epoch
                # Get first 5 images from batch
                x_for_tb = x[:5]
                labels_for_tb = labels[:5]

                # Decode some images and show them in Tensorboard
                x_pred_intermediate, latents_intermediate = hetsiren_decode_image(graphdef, state, x_for_tb,
                                                                                  labels_for_tb, md_columns,
                                                                                  ctf_type=args.ctf_type,
                                                                                  return_latent=True,
                                                                                  corrupt_projection_with_ctf=True)
                x_pred_intermediate = jax.vmap(min_max_scale)(x_pred_intermediate[..., None])
                writer.add_images("Predicted images batch", x_pred_intermediate, dataformats="NHWC")

                # Decode some states and show them in Tensorboard
                volumes_intermediate = hetsiren_decode_volume(graphdef, state, latents_intermediate)
                writer.add_volumes_slices(volumes_intermediate)

                # Log landscape stored in memory bank
                if i > 0 and i % 5 == 0:
                    choice_key_use, choice_key = jax.random.split(rng, 2)
                    hetsiren_intermediate, _ = nnx.merge(graphdef, state)
                    random_indices = jnr.choice(choice_key_use,
                                                a=jnp.arange(hetsiren_intermediate.bank_size),
                                                shape=(hetsiren_intermediate.subset_size,), replace=False)
                    latents_intermediate = hetsiren_intermediate.memory_bank.get_value()[random_indices]
                    latents_data_loader = NumpyGenerator(latents_intermediate).return_grain_dataset(
                        preShuffle=False, shuffle=False, batch_size=args.batch_size,
                        num_epochs=1, num_workers=0)
                    latents_images = []
                    for (latents, _) in latents_data_loader:
                        random_labels = jnp.asarray(
                            np.random.randint(low=0, high=len(generator.md), size=(latents.shape[0],)),
                            dtype=jnp.int32)
                        x_pred_intermediate = hetsiren_decode_image(graphdef, state, latents, random_labels,
                                                                    md_columns, ctf_type=None, return_latent=False,
                                                                    corrupt_projection_with_ctf=False)
                        x_pred_intermediate = \
                        image_resize(x_pred_intermediate[..., None], (latents.shape[0], 128, 128, 1),
                                     method="bilinear")[..., 0]
                        latents_images.append(np.asarray(x_pred_intermediate))
                    latents_images = np.concatenate(latents_images, axis=0)
                    latent_images_min = latents_images.min(axis=(1, 2), keepdims=True)
                    latent_images_max = latents_images.max(axis=(1, 2), keepdims=True)
                    latents_images = (latents_images - latent_images_min) / (latent_images_max - latent_images_min)
                    writer.add_embedding(latents_intermediate, label_img=latents_images[:, None, ...],
                                         tag="HetSIREN latent space", global_step=i)

                    # Save checkpoint model
                    NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path,
                                                                                              "HetSIREN_CHECKPOINT"),
                                                                epoch=i)

                i += 1
            if isTomoSIREN:
                x_total = (x, subtomo_labels)
            else:
                x_total = x
            loss, recon_loss, state, rng = train_step_hetsiren(graphdef, state, x_total, labels, md_columns, rng,
                                                               l1_lambda=args.denoising_strength,
                                                               graph_lambda=graph_lambda)
            total_loss += loss
            total_recon_loss += recon_loss

            # Summary writer (training loss)
            if step % int(np.ceil(0.1 * steps_per_epoch)) == 0:
                writer.add_scalar('Training loss (HetSIREN)',
                                  total_loss / step,
                                  i * steps_per_epoch + step)

                writer.add_scalars('Reconstruction loss (HetSIREN)',
                                   {"train": total_recon_loss / step},
                                    i * steps_per_epoch + step)

            # # Summary writer (validation loss)
            # if step % int(np.ceil(0.9 * steps_per_epoch)) == 0:
            #     # Run validation step
            #     pbar.set_postfix_str(f"{bcolors.WARNING}Running validation step...{bcolors.ENDC}")
            #     for (x_validation, labels_validation) in data_loader_val:
            #         loss_validation = validation_step_hetsiren(graphdef, state, x_validation, labels_validation, md_columns, rng)
            #         total_validation_loss += loss_validation
            #         step_validation += 1
            #     writer.add_scalars('Reconstruction loss (HetSIREN)',
            #                        {"validation": total_validation_loss / step_validation},
            #                        i * steps_per_epoch + step)

            # Progress bar update  (TQDM)
            if args.transport_mass:
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | recon_loss={total_recon_loss / step:.5f} | graph_lambda={graph_lambda:.5f}")
            else:
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | recon_loss={total_recon_loss / step:.5f}")

            step += 1

        hetsiren, optimizer = nnx.merge(graphdef, state)

        # Example of predicted data for Tensorboard
        x_pred_example = hetsiren_decode_image(graphdef, state, x_example, labels_example, md_columns, ctf_type=args.ctf_type, return_latent=False, corrupt_projection_with_ctf=True)
        x_pred_example = jax.vmap(min_max_scale)(x_pred_example[..., None])
        writer.add_images("Predicted images batch", x_pred_example, dataformats="NHWC")

        # Save model
        NeuralNetworkCheckpointer.save(hetsiren, os.path.join(args.output_path, "HetSIREN"))

        # Remove checkpoint
        shutil.rmtree(os.path.join(args.output_path, "HetSIREN_CHECKPOINT"))

    elif args.mode == "predict":

        hetsiren.eval()

        # Rotations to Xmipp angles
        euler_from_matrix_batch = jax.vmap(jax.jit(euler_from_matrix))

        def xmippEulerFromMatrix(matrix):
            return -jnp.rad2deg(euler_from_matrix_batch(matrix))

        # Prepare data loader
        data_loader = generator.return_grain_dataset(batch_size=args.batch_size, shuffle=False, num_epochs=1,
                                                     num_workers=-1, load_to_ram=args.load_images_to_ram)
        steps_per_epoch = int(np.ceil(len(generator.md) / args.batch_size))

        # Jitted prediction functions
        @nnx.jit
        def predict_fn(model, x):
            return model(x)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting HetSIREN latents... ######")

        # For progress bar (TQDM)
        pbar = tqdm(data_loader, file=sys.stdout, ascii=" >=", colour="green", total=steps_per_epoch,
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        if isTomoSIREN:
            iterable = pbar
        else:
            iterable = ((x, None, labels) for (x, labels) in pbar)

        md_pred = generator.md
        md_pred[:, 'latent_space'] = np.asarray([",".join(np.char.mod('%f', item)) for item in np.zeros((len(md_pred), args.lat_dim))])
        for (x, _, labels) in iterable:
            if isinstance(x, tuple):
                x = x[0]

            latents_batch, (rotations_rigid, shifts_rigid) = predict_fn(hetsiren, x)

            # Precompute batch aligments
            rotations_batch = md_columns["euler_angles"][labels]

            # Precompute batch shifts
            shifts_batch = md_columns["shifts"][labels]

            # Get rotation matrices
            if rotations_batch.ndim == 2:
                rotations_batch = euler_matrix_batch(rotations_batch[:, 0], rotations_batch[:, 1], rotations_batch[:, 2])

            # Consider refinement and rigid registration alignments
            rotations_refined = jnp.matmul(rotations_batch, rotations_rigid)
            shifts_refined = shifts_batch + shifts_rigid

            # Convert rotation to Euler angles in Xmipp format
            euler_angles_refined = xmippEulerFromMatrix(rotations_refined)

            # Convert to Numpy
            euler_angles_refined, shifts_refined = np.array(euler_angles_refined), np.array(shifts_refined)

            # Save to metadata
            md_pred[labels, 'angleRot'] = euler_angles_refined[..., 0]
            md_pred[labels, 'angleTilt'] = euler_angles_refined[..., 1]
            md_pred[labels, 'anglePsi'] = euler_angles_refined[..., 2]
            md_pred[labels, 'shiftX'] = shifts_refined[..., 0]
            md_pred[labels, 'shiftY'] = shifts_refined[..., 1]
            md_pred[labels, 'latent_space'] = np.asarray([",".join(np.char.mod('%f', item)) for item in latents_batch])

        # Save latents in metadata
        md_pred.write(os.path.join(args.output_path, "predicted_latents" + os.path.splitext(args.md)[1]))

    # If exists, clean MMAP
    # if not args.load_images_to_ram and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap_grain")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap_grain"))
