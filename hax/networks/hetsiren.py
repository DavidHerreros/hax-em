#!/usr/bin/env python


from functools import partial
import numpy as np

import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import dm_pix

from einops import rearrange
from sklearn.cluster import KMeans

from hax.utils import *
from hax.layers import *
from hax.programs import splat_weights_trilinear, splat_weights, FastVariableBlur3D
try:  # CryoUni depends on optional torch / cryouni deps; only needed for the "cryouni" architecture
    from hax.pretrained_models import CryoUni, CryoUniHead, CryoUniNNX
except ImportError:
    CryoUni = CryoUniHead = CryoUniNNX = None


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
            self.hidden_layers = nnx.List(hidden_layers)
            # self.hidden_layers = hidden_layers
            self.latent = Linear(1024, lat_dim, rngs=rngs)

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

        elif self.architecture == "cryouni":
            self.cryouni = CryoUni(input_shape=self.input_dim)
            self.cryouni_head = CryoUniHead(image_size=self.input_dim, rngs=rngs)
            self.hidden_layers = Linear(self.cryouni_head.out_shape, 256, rngs=rngs, dtype=jnp.bfloat16)

        else:
            raise ValueError("Architecture not supported. Implemented architectures are: mlpnn / convnn")

        if isVae:
            # self.layer_normalization = nnx.LayerNorm(256, rngs=rngs)
            self.mean_x = Linear(256, lat_dim, rngs=rngs)
            self.logstd_x = Linear(256, lat_dim, rngs=rngs)
        else:
            self.latent = Linear(256, lat_dim, rngs=rngs)

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

        elif self.architecture == "cryouni":
            x = self.cryouni(x)
            x = self.cryouni_head(x["clstokens"], x["patchtokens"])
            x = nnx.leaky_relu(self.hidden_layers(x))

        if return_last:
            return x
        else:
            if self.isVae:
                # x = self.layer_normalization(x)
                mean = self.mean_x(x)
                logstd = self.logstd_x(x)
                # logstd = jnp.clip(logstd, -4.0, 4.0)
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
    def __init__(self, input_dim, lat_dim=10, n_layers=3, isVae=False, architecture="convnn", isTomoSIREN=False, *, rngs: nnx.Rngs):
        if isTomoSIREN:
            self.encoders = nnx.Dict({"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, architecture=architecture, rngs=rngs),
                                      "encoder_dec": EncoderTomo(100, lat_dim, n_layers=n_layers, rngs=rngs)})
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
                 rngs=None, warmup_alpha=1.0):
        x = self.encoders[encoder_id](x, return_last=True)

        if return_alignment_refinement:
            x_ref = nnx.leaky_relu(x + self.hidden_layers_refinement[0](x))  # or nnx.relu
            for layer in self.hidden_layers_refinement[1:]:
                x_ref = nnx.leaky_relu(layer(x_ref + x_ref))  # or nnx.relu

            # Estimate rotations for volume registration. warmup_alpha (in [0, 1])
            # ramps the refinement in from the identity so early training keeps the
            # input poses untouched and only gradually starts refining them.
            rotations_6d = self.rigid_6d_rotation(x_ref)
            identity_6d = jnp.array([1., 0., 0., 0., 1., 0.])[None, ...].repeat(rotations_6d.shape[0], axis=0)
            rotations_6d = identity_6d + warmup_alpha * rotations_6d
            rotations_rigid = PoseDistMatrix.mode_rotmat(rotations_6d)
            rotations_logscale = self.rotations_logsig(x_ref)

            # Estimate shifts for volume registration (ramped in the same way)
            shifts_rigid = warmup_alpha * self.rigid_shifts(x_ref)

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
    def __init__(self, total_voxels, lat_dim, volume_size, coords, reference_values, transport_mass=False, is_implicit=True, hybrid_pe=False,
                 point_transformer=False, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.reference_values = reference_values[None, ...]
        self.total_voxels = total_voxels
        self.transport_mass = transport_mass
        self.is_implicit = is_implicit
        self.hybrid_pe = hybrid_pe
        self.point_transformer = point_transformer

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
            if self.point_transformer:
                self.geom = nnx.data(build_geometry(self.coords[0], gauss_scale="auto", hierarchical_sizes=(32, 128, 512), compute_local_frames=True))

                # self.point_transformer_net = PointTransformerDecoder(out_channels=4, latent_dim=lat_dim, rngs=rngs)

                self.point_transformer_net = PointTransformerDecoder(out_channels=(3, 1), feat_dim=32, nk=256, input_bottleneck=64,
                                                                     hierarchical_sizes=(32, 128, 512), latent_dim=lat_dim, rngs=rngs)

                # self.point_transformer_coords = PointTransformerDecoder(out_channels=3, feat_dim=256, latent_dim=lat_dim, rngs=rngs)
                # self.point_transformer_values = PointTransformerDecoder(out_channels=1, feat_dim=64, latent_dim=lat_dim, rngs=rngs)  # Or 128

            elif self.is_implicit:
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
                    for _ in range(7):
                        hidden_coords.append(nnx.Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, use_bias=False, kernel_init=kernel_init))
                    hidden_coords.append(nnx.Linear(in_features=32, out_features=3, rngs=rngs, dtype=jnp.float32, use_bias=False, kernel_init=kernel_init))
                    hidden_coords.append(nnx.Linear(in_features=3, out_features=3, rngs=rngs, use_bias=False, kernel_init=kernel_init))

                    hidden_values = [nnx.Linear(in_features=lat_dim // 2 + 3 * 10 * 2, out_features=32, rngs=rngs, dtype=jnp.float32, use_bias=False, kernel_init=kernel_init)]
                    for _ in range(7):
                        hidden_values.append(nnx.Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, use_bias=False, kernel_init=kernel_init))
                    hidden_values.append(nnx.Linear(in_features=32, out_features=1, rngs=rngs, dtype=jnp.float32, use_bias=False, kernel_init=kernel_init))
                    hidden_values.append(nnx.Linear(in_features=1, out_features=1, rngs=rngs, use_bias=False, kernel_init=kernel_init))

                self.hidden_values = nnx.List(hidden_values)
                self.hidden_coords = nnx.List(hidden_coords)

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

    def __call__(self, x, c=None, freq_alpha=1.0):
        # freq_alpha in (0, 1] anneals the effective frequency (w0) of the first
        # SIREN layer for coarse-to-fine training; 1.0 = no change. It only
        # applies to the SIREN-based decoders (not point transformer / hybrid PE).
        if self.transport_mass:
            if self.point_transformer:
                # x = self.point_transformer_net(x, self.geom)
                # x_coords, x_map = x[..., :-1], x[..., -1]

                x_coords, x_map = self.point_transformer_net(x, self.geom)
                x_map = x_map[..., 0]

                # x_coords = self.point_transformer_coords(x, self.geom)
                # x_map = self.point_transformer_values(x, self.geom)[..., 0]

            elif self.is_implicit:
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
                    x_map = jnp.concatenate([c_pe, x_map], axis=-1)
                else:
                    x_coords = jnp.concatenate([c, x_coords], axis=-1)
                    x_map = jnp.concatenate([c, x_map], axis=-1)

                # Decode coords
                if self.hybrid_pe:
                    x_coords = nnx.elu(self.hidden_coords[0](x_coords))
                    for layer in self.hidden_coords[1:-1]:
                        x_coords = nnx.elu(layer(x_coords))
                    x_coords = self.hidden_coords[-1](x_coords)

                    # Decode values
                    x_map = nnx.elu(self.hidden_values[0](x_map))
                    for layer in self.hidden_values[1:-1]:
                        x_map = nnx.elu(layer(x_map))
                    x_map = self.hidden_values[-1](x_map)[..., 0]

                else:
                    x_coords = self.hidden_coords[0](x_coords, freq_alpha)
                    for layer in self.hidden_coords[1:-1]:
                        x_coords = layer(x_coords)
                    x_coords = self.hidden_coords[-1](x_coords)

                    # Decode values
                    x_map = self.hidden_values[0](x_map, freq_alpha)
                    for layer in self.hidden_values[1:-1]:
                        x_map = layer(x_map)
                    x_map = self.hidden_values[-1](x_map)[..., 0]

            else:
                x_coords, x_map = jnp.split(x, indices_or_sections=2, axis=1)

                # Decode values
                x_map = self.hidden_values[0](x_map, freq_alpha)
                for layer in self.hidden_values[1:-1]:
                    x_map = layer(x_map)
                x_map = self.hidden_values[-1](x_map)

                # Decode coords
                x_coords = self.hidden_coords[0](x_coords, freq_alpha)
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
            x_map = self.hidden_values[0](x, freq_alpha)
            for layer in self.hidden_values[1:-1]:
                x_map = layer(x_map)
            x_map = self.hidden_values[-1](x_map)

            # Recover volume values
            values = self.reference_values + x_map

            # Recover coords (non-normalized)
            coords = self.scale * self.coords.repeat(x.shape[0], axis=0)

        return coords, values

    def decode_coords_only(self, x, c=None):
        if self.transport_mass:
            if self.point_transformer:
                # x = self.point_transformer_net(x, self.geom)
                # x_coords = x[..., :-1]

                x_coords, _ = self.point_transformer_net(x, self.geom)

                # x_coords = self.point_transformer_coords(x, self.geom)

            elif self.is_implicit:
                # Positional encoding of coords
                if c is None:
                    c = self.coords[0]
                c = jnp.tile(c[None, ...], (x.shape[0], 1, 1))

                # Adjust latents
                x = jnp.tile(x[:, None, ...], (1, c.shape[1], 1))

                # Join coords and latents
                if self.hybrid_pe:
                    c_pe = positional_encoding(c[0], 10, self.scale)
                    c_pe = jnp.tile(c_pe[None, ...], (x.shape[0], 1, 1))
                    x = jnp.concatenate([c_pe, x], axis=-1)
                else:
                    x = jnp.concatenate([c, x], axis=-1)

                # Decode coords
                if self.hybrid_pe:
                    x_coords = nnx.elu(self.hidden_coords[0](x))
                    for layer in self.hidden_coords[1:-1]:
                        x_coords = nnx.elu(layer(x_coords))
                    x_coords = self.hidden_coords[-1](x_coords)

                else:
                    x_coords = self.hidden_coords[0](x)
                    for layer in self.hidden_coords[1:-1]:
                        x_coords = layer(x_coords)
                    x_coords = self.hidden_coords[-1](x_coords)

            else:
                # Decode coords
                x_coords = self.hidden_coords[0](x)
                for layer in self.hidden_coords[1:-1]:
                    x_coords = layer(x_coords)
                x_coords = self.hidden_coords[-1](x_coords)

                x_coords = jnp.reshape(x_coords, (x.shape[0], self.total_voxels, 3))

            # Recover coords (non-normalized)
            coords = self.scale * (self.coords + x_coords)
        else:
            # Recover coords (non-normalized)
            coords = self.scale * self.coords.repeat(x.shape[0], axis=0)

        return coords

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

        # Guard the scatter against NaN/Inf or out-of-range coordinates
        bposi = jnp.clip(bposi, 0, self.volume_size - 1)
        bamp = jnp.nan_to_num(bamp)

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
            bamp = values * jnp.exp(-num / (2. * sigma ** 2.))

        # Guard the scatter against NaN/Inf or out-of-range coordinates
        bposi = jnp.clip(bposi, 0, xsize - 1)
        bamp = jnp.nan_to_num(bamp)

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        images = jax.vmap(scatter_img)(images, bposi, bamp)

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
    def __init__(self, lat_dim, reference_volume, reconstruction_mask, coords, values, xsize, sr, bank_size=1024, ctf_type="apply",
                 sigma=1.0, decoupling=False, isVae=False, transport_mass=False, local_reconstruction=False, architecture="convnn",
                 is_implicit=True, isTomoSIREN=False, train_inverse=False, point_transformer=False, use_frc_loss=False,
                 loss_type=None, spectral_ring_weight=None, *, rngs: nnx.Rngs):
        super(HetSIREN, self).__init__()
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.decoupling = decoupling if not isTomoSIREN else False
        self.isVae = isVae
        self.isTomoSIREN = isTomoSIREN
        self.train_inverse = train_inverse
        self.local_reconstruction = local_reconstruction
        self.reference_volume = reference_volume
        self.reconstruction_mask = reconstruction_mask.astype(float)
        self.coords = jnp.array(coords)
        self.lat_dim = lat_dim
        self.has_reference_volume = not bool(np.all(reference_volume == 0.0))
        self.encoder = MultiEncoder(self.xsize, lat_dim, n_layers=3, isVae=isVae, architecture=architecture, isTomoSIREN=isTomoSIREN, rngs=rngs) \
            if decoupling or isTomoSIREN else Encoder(self.xsize, lat_dim, isVae=isVae, architecture=architecture, rngs=rngs)
        self.delta_volume_decoder = DeltaVolumeDecoder(self.coords.shape[0], lat_dim, self.xsize, self.coords, values, transport_mass=transport_mass, is_implicit=is_implicit, point_transformer=point_transformer, rngs=rngs)
        if self.train_inverse:
            inv_lat_dim = lat_dim if self.delta_volume_decoder.point_transformer else lat_dim // 2
            self.inverse_volume_decoder = PointCloudEncoder(latent_dim=inv_lat_dim, hidden=64, n_blocks=4, dtype=jnp.float32, scale_multiplier=1.0, rngs=rngs)

        self.phys_decoder = PhysDecoder(self.xsize, sr, transport_mass=transport_mass)

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.subset_size = min(2048, bank_size)
        self.memory_bank = MemoryBank(array_init=jnp.zeros((self.bank_size, lat_dim)))

        # Gaussians size
        # self.sigma = nnx.Param(sigma)
        self.sigma = sigma

        # Loss function.
        #   loss_type: one of {"mse", "frc", "spectral"} to select the
        #   reconstruction loss explicitly. If None, defaults to "mse" (or "frc"
        #   for the point transformer decoder, which needs a Fourier-space loss).
        if loss_type is None:
            loss_type = "frc" if use_frc_loss else "mse"
        if self.delta_volume_decoder.point_transformer and loss_type == "mse":
            loss_type = "frc"
        self.loss_type = loss_type

        if loss_type == "frc":
            self.representation_loss_fn = FRCLoss(box_size=xsize, apix=sr, min_resolution_A=30., max_resolution_A=2. * sr)
        elif loss_type == "spectral":
            # SSNR/Wiener-weighted spectral L2 (noise model in the data term).
            # Band runs to Nyquist; the per-ring weight down-weights noisy shells.
            self.representation_loss_fn = SpectralL2Loss(box_size=xsize, apix=sr, ring_weight=spectral_ring_weight,
                                                         min_resolution_A=30., max_resolution_A=2. * sr)
        else:
            self.representation_loss_fn = lambda x, y, freq_alpha=1.0: mse(x[..., None], y[..., None])

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
            # elif self.ctf_type in ["apply", "squared"]:
            #     x = prepare_image_cryocrab(x, ctf)

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
            # rotations = jnp.matmul(rotations_rigid, rotations)
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

        return self.delta_volume_decoder.decode_volume(x=x, filter=True, sigma=self.sigma)

    def decode_field(self, x):
        if x.ndim == 4:
            x, _ = self(x)

        coords, values = self.delta_volume_decoder(x)

        inital_coords = self.delta_volume_decoder.scale * self.delta_volume_decoder.coords
        field = coords - inital_coords

        return (field / (0.5 * self.xsize), inital_coords / (0.5 * self.xsize))

    def cloud_to_latent(self, cloud):
        if self.train_inverse:
            cloud = cloud / self.sr
            cloud = (cloud - self.delta_volume_decoder.centering[0]) / self.delta_volume_decoder.scale
            if cloud.ndim == 2:
                cloud = cloud[None, ...]
            cloud = fps_resample_batched(cloud, 5_000, 0)
            return self.inverse_volume_decoder(cloud)
        else:
            raise UserWarning("The network was not trained with decoder inversion support.")


@partial(jax.jit, static_argnames=("do_update", "l1_lambda", "graph_lambda", "pose_refine_reg",
                                   "decoupling_lambda", "distance_preservation_lambda", "kl_lambda"))
def train_step_hetsiren(graphdef, state, x, labels, md, key, do_update=True, l1_lambda=1e-4, graph_lambda=1e-4,
                        warmup_alpha=1.0, pose_refine_reg=0.1, decoupling_lambda=1e-4, distance_preservation_lambda=1e-4,
                        kl_lambda=1e-3, freq_alpha=1.0):
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
        # if model.ctf_type in ["apply", "squared"]:
        #     x_in = prepare_image_cryocrab(x, ctf)
        # else:
        #     x_in = x
        x_in = x

        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key, warmup_alpha=warmup_alpha)
            elif model.isTomoSIREN:
                (sample, latent, logstd), prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
                (_, latent_1, _), (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out_random = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key, warmup_alpha=warmup_alpha)
            else:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid, rotations_logscale) = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key, warmup_alpha=warmup_alpha)
        else:
            if model.decoupling:
                latent, (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key, warmup_alpha=warmup_alpha)
            elif model.isTomoSIREN:
                latent, prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
                latent_1, (rotations_rigid, shifts_rigid, rotations_logscale), prev_layer_out_random = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key, warmup_alpha=warmup_alpha)
            else:
                latent, (rotations_rigid, shifts_rigid, rotations_logscale) = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key, warmup_alpha=warmup_alpha)

        # Decode volumes (freq_alpha anneals the decoder's first-SIREN-layer w0)
        if model.isVae:
            coords, values = model.delta_volume_decoder(sample, freq_alpha=freq_alpha)
        else:
            coords, values = model.delta_volume_decoder(latent, freq_alpha=freq_alpha)

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
            # rotations_refined = jnp.matmul(rotations_rigid, rotations)
            rotations_refined = jnp.matmul(rotations, rotations_rigid)

            rotations_refined, omegas, log_q = sample_topM_R(rot_sample_key, rotations_refined, rotations_logscale, M=M)
        else:
            # Consider refinement and rigid registration alignments (for delta_volume_decoder_rigid output)
            # rotations_refined = jnp.matmul(rotations_rigid, rotations)
            rotations_refined = jnp.matmul(rotations, rotations_rigid)
        shifts_refined = shifts + shifts_rigid

        # Geodesic anchor: keep the rigid rotation close to the identity so it stays a refinement
        cos_theta_rigid = jnp.clip((jnp.trace(rotations_rigid, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        pose_refine_loss = jnp.mean(1.0 - cos_theta_rigid)

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

        # if not model.delta_volume_decoder.transport_mass:
        #     consensus_coords, consensus_values = model.delta_volume_decoder(jnp.zeros_like(latent))
        #     images_consensus, _ = phys_decoder(x, consensus_values, consensus_coords, model.xsize, rotations_refined, shifts_refined,
        #                                        centering, ctf, model.ctf_type, model.sigma, 0.0)

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
        # if not model.delta_volume_decoder.transport_mass:
        #     images_consensus = jnp.squeeze(images_consensus)

        # Consider CTF if Wiener mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_loss = wiener2DFilter_vmap(images_corrected, ctf, pad_factor)
            images_corrected_field_loss = wiener2DFilter_vmap(images_corrected_field, ctf, pad_factor)
            # if not model.delta_volume_decoder.transport_mass:
            #     images_consensus_loss = wiener2DFilter_vmap(images_consensus, ctf, pad_factor)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_loss = ctfFilter_vmap(images_corrected, ctf, pad_factor)
            images_corrected_field_loss = ctfFilter_vmap(images_corrected_field, ctf, pad_factor)
            # if not model.delta_volume_decoder.transport_mass:
            #     images_consensus_loss = ctfFilter_vmap(images_consensus, ctf, pad_factor)
        else:
            x_loss = x
            images_corrected_loss = images_corrected
            images_corrected_field_loss = images_corrected_field
            # if not model.delta_volume_decoder.transport_mass:
            #     images_consensus_loss = images_consensus

        if M > 1:
            x_loss = x_loss[:, None, ...]

        # Projection mask
        x_loss = x_loss * projected_mask
        images_corrected_loss = images_corrected_loss * projected_mask
        images_corrected_field_loss = images_corrected_field_loss * projected_mask
        # if not model.delta_volume_decoder.transport_mass:
        #     images_consensus_loss = images_consensus_loss * projected_mask

        recon_loss = 0.1 * model.representation_loss_fn(images_corrected_loss, x_loss, freq_alpha) + 0.9 * model.representation_loss_fn(images_corrected_field_loss, x_loss, freq_alpha)
        # if model.delta_volume_decoder.transport_mass:
        recons_loss_all = recon_loss.mean()
        # else:
        #     recons_loss_all = 0.5 * (recon_loss.mean() + mse(images_consensus_loss[..., None], x_loss[..., None]).mean())

        # L1 based denoising
        l1_loss = jnp.mean(jnp.abs(values))

        # L1 denoising for negative values
        values_neg = jnp.where(values < 0.0, -values, 0.0)
        neg_count = jnp.count_nonzero(values < 0.0)
        l1_loss += jnp.sum(values_neg) / jnp.maximum(neg_count, 1)

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
        # if model.has_reference_volume and model.delta_volume_decoder.transport_mass:
        #     factor = 0.5 * model.xsize
        #     coords_cm = (coords + centering - factor) / factor
        #     cm = jnp.average(coords_cm, weights=jnp.broadcast_to(values[..., None], coords.shape), axis=1)
        #     loss_cm = jnp.linalg.norm(cm, axis=1).mean()
        # else:
        #     loss_cm = 0.0

        # Local distance preservation
        if model.isVae:
            coords_mean, values_mean = model.delta_volume_decoder(latent, freq_alpha=freq_alpha)
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
            kl_loss = -0.5 * jnp.mean(1. + 2. * logstd - jnp.square(jnp.exp(logstd)) - jnp.square(latent))
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
            memory_bank_subset = model.memory_bank.get()#[random_indices]

            dist = jnp.pow(latent[:, None, :] - memory_bank_subset, 2.).sum(axis=-1)
            dist_nn, _ = jax.lax.approx_min_k(dist, k=10, recall_target=0.95)
            dist_fn, _ = jax.lax.approx_max_k(dist, k=10, recall_target=0.95)

            decoupling_loss += 1.0 * triplet_loss(dist_nn, dist_fn, reduction="mean", margin=0.01)

        else:
            decoupling_loss = 0.0

        loss = (nll + kl_lambda * kl_loss + 0.000001 * kl_pose + decoupling_lambda * decoupling_loss
                + l1_lambda * l1_loss + graph_lambda * loss_graph + 100. * hist_loss + distance_preservation_lambda * loss_dp
                + pose_refine_reg * pose_refine_loss)
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

    params = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('delta_volume_decoder')))
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, params))
    if model.isTomoSIREN:
        (loss, (recon_loss, latent)), grads = grad_fn(model, (x, subtomogram_label))
    else:
        (loss, (recon_loss, latent)), grads = grad_fn(model, x)

    if do_update:
        grads, _ = grads.split(params, ...)

        optimizer.update(model, grads)

        # Update memory bank
        model.memory_bank.enqueue(latent)

        state = nnx.state((model, optimizer))

        return loss, recon_loss, state, key
    else:
        return loss, recon_loss


@partial(jax.jit, static_argnames=("do_update"))
def train_step_inverse_hetsiren(graphdef, state, x, labels, md, key, do_update=True):
    model, optimizer = nnx.merge(graphdef, state)
    distributions_key, key = jnr.split(key, 2)

    def loss_fn(model, x):
        # Check if Tomo mode
        if model.isTomoSIREN:
            (x, subtomogram_label) = x

        # Prepare input images for encoder
        # if model.ctf_type in ["apply", "squared"]:
        #     x_in = prepare_image_cryocrab(x, ctf)
        # else:
        #     x_in = x
        x_in = x

        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (_, latent, _), _, _ = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                (_, latent, _), _ = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
            else:
                (_, latent, _), _ = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key)
        else:
            if model.decoupling:
                latent,_, _ = model.encoder(x_in, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                latent, _ = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
            else:
                latent, _ = model.encoder(x_in, return_alignment_refinement=True, rngs=distributions_key)

        # Decode volumes
        coords, values = model.delta_volume_decoder(latent)

        subkey1, subkey2, subkey3 = jax.random.split(distributions_key, 3)
        p = jax.random.uniform(subkey1, minval=0.1, maxval=1.0)

        coords_scaled = jax.lax.stop_gradient(coords) / model.delta_volume_decoder.scale
        start_idx = jax.random.randint(subkey3, (), 0, coords.shape[1])
        coords_scaled_resampled = fps_resample_batched(coords_scaled, 5_000, start_idx)

        mask = jax.random.bernoulli(subkey2, p=p, shape=(coords_scaled_resampled.shape[1],))[None, ...]
        inv_latent = model.inverse_volume_decoder(coords_scaled_resampled, mask=mask)

        latent_no_grad = jax.lax.stop_gradient(latent)
        if not model.delta_volume_decoder.point_transformer:
            latent_no_grad = latent_no_grad[..., :model.lat_dim // 2]
        loss = jnp.mean(jnp.square(latent_no_grad - inv_latent), axis=-1).mean()
        # loss = jnp.mean((jnp.sum(inv_latent - latent_no_grad, axis=-1) / (jnp.sum(jnp.abs(latent_no_grad), axis=-1) + 1e-6)) ** 2)

        coords_inv = model.delta_volume_decoder.decode_coords_only(inv_latent)
        # loss += chamfer_distance(jax.lax.stop_gradient(coords), coords_inv)
        loss += chamfer_distance(coords_scaled, coords_inv / model.delta_volume_decoder.scale)

        return loss

    # Check if Tomo mode
    if model.isTomoSIREN:
        (x, subtomogram_label) = x

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

    params_inv = nnx.All(nnx.Param, nnx.PathContains('inverse_volume_decoder'))
    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, params_inv))
    if model.isTomoSIREN:
        loss, grads = grad_fn(model, (x, subtomogram_label))
    else:
        loss, grads = grad_fn(model, x)

    if do_update:
        grads, _ = grads.split(params_inv, ...)
        optimizer.update(model, grads)
        state = nnx.state((model, optimizer))
        return loss, state, key
    else:
        return loss


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
        # if model.ctf_type in ["apply", "squared"]:
        #     x_in = prepare_image_cryocrab(x, ctf)
        # else:
        #     x_in = x
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
        # rotations_refined = jnp.matmul(rotations_rigid, rotations)
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

        recon_loss = 0.9 * model.representation_loss_fn(images_corrected_field_loss, x_loss)

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
        # if model.ctf_type in ["apply", "squared"]:
        #     x_in = prepare_image_cryocrab(x, ctf)
        # else:
        #     x_in = x
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
        # rotations_refined = jnp.matmul(rotations_rigid, rotations)
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

        recon_loss = 0.1 * model.representation_loss_fn(images_corrected_loss, x_loss) + 0.9 * model.representation_loss_fn(images_corrected_field_loss, x_loss)

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
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, extract_columns, NumpyGenerator
    from hax.networks import train_step_hetsiren
    from hax.metrics import JaxSummaryWriter
    from hax.programs import fit_gaussian_splat, fit_weights_to_images, fit_volume, adjust_weights_to_images
    # from hax.schedulers import CosineAnnealingScheduler

    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser,
              help=f"Xmipp/Relion metadata file with the images (+ alignments / CTF) to be analyzed. {bcolors.WARNING}NOTE{bcolors.ENDC}: If the metadata "
                   f"includes the label {bcolors.ITALIC}subtomo_labels{bcolors.ENDC}, then TomoSIREN network will be trained (i.e., HetSIREN for "
                   f"tomography data). In this case, the parameter {bcolors.ITALIC}decoupling{bcolors.ENDC} will not have any effect as pose/CTF "
                   f"decoupling is already handled by TomoSIREN.")
    ca.add_vol(parser,
               help="If provided, the neural network will learn how to refine this volume towards the heterogeneous states present in the images")
    ca.add_mask(parser,
                help=f"HetSIREN reconstruction mask (useful to focus the reconstruction/estimated landscape on a specific region of interest - the mask provided "
                     f"must be binary -  {bcolors.WARNING}NOTE{bcolors.ENDC}: since this is a reconstruction mask, it should be defined such that it covers the "
                     f"volume were the motions of interest are expected to happen)")
    ca.add_load_images_to_ram(parser)
    ca.add_sr(parser)
    ca.add_lat_dim(parser)
    parser.add_argument("--transport_mass", action='store_true',
                        help='When set, HetSIREN will be able to "move" the mass inside the mask instead of just reconstructing the volume. This implies that HetSIREN will estimate the motion '
                             'to be applied to the points within the provided mask, instead of considering them fixed in space. This approach is useful when working with large box sizes that '
                             'do not fit in GPU memory, or when a more through analysis of motions is desired. '
                             f'{bcolors.WARNING}NOTE{bcolors.ENDC}: When this option is set and a reference volume is provided, we recommend changing the reference mask to a tight mask computed '
                             f'from the reference volume. This mask now tells the program which regions should be moved. Therefore, consider providing a mask that covers all the protein regions you would like '
                             f'to be analyzed by HetSIREN.')
    parser.add_argument("--num_gaussians", required=False, type=int,
                        help="Before training the network, HetSIREN will try to fit a set of Gaussians in the reference volume to recreate it. "
                            "The default criterium is to automatically determine the number of Gaussians neede to reproduce the reference volume "
                            "with high-fidelity. However, if you prefer to fix the number of Gaussians in advance based on your own criterium (e.g., "
                            "the number of residues in your protein), you can set this parameter. When set, the HetSIREN will fit this fixed number of Gaussians "
                            "so that the reproduce the reference volume as well as possible.")
    parser.add_argument("--local_reconstruction", action='store_true',
                        help=f'When set, HetSIREN will turn to local heterogeneous reconstruction/refinement mod, focusing the analysis of heterogeneity to a region of interest enclosed by the provided refernece mask. '
                             f'{bcolors.WARNING}WARNING{bcolors.ENDC}: IF PROVIDED, TRANSPORT MASS WILL BE OVERRIDDEN AND NOT CONSIDERED. '
                             f'{bcolors.WARNING}WARNING{bcolors.ENDC}: IF PROVIDED, HAVING A REFERENCE VOLUME IS MANDATORY. OTHERWISE, THIS PARAMETER WILL BE NEGLECTED. ')
    ca.add_ctf_type(parser)
    ca.add_mode(parser)
    ca.add_epochs(parser)
    ca.add_batch_size(parser)
    ca.add_learning_rate(parser)
    ca.add_dataset_split_fraction(parser)
    ca.add_output_path(parser)
    ca.add_reload(parser,
                  help=f"Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data - "
                       f"{bcolors.WARNING}NOTE{bcolors.ENDC}: If a reference volume was provided, HetSIREN also learns a gray level adjustment. In this case, "
                       f"reload must be the path to a folder containing two additional folders called: {bcolors.UNDERLINE}HetSIREN{bcolors.ENDC} and {bcolors.UNDERLINE}Gaussian_volume_fitting{bcolors.ENDC})")
    parser.add_argument("--denoising_strength", required=False, type=float, default=1e-4,
                        help=f"Determines how strongly HetSIREN will learn to remove noise from the resulting volumes. Increasing the value of this parameter will result in a stronger regularization of the noise, but it may affect the protein "
                             f"signal as well. ({bcolors.WARNING}NOTE{bcolors.ENDC}: We recommend setting this parameter in the range 0.0001 to 0.1)")
    parser.add_argument("--implicit_network", action='store_true',
                        help=f'When set, HetSIREN will use an implicit neural network approach to recover conformational states. Implicit neural networks are more memory consuming, but they are also more accurate in the detection of very local motions. '
                             f'If this architecture is selected, we strongly recommend to set as well the parameter {bcolors.ITALIC}total_mass{bcolors.ENDC} to limit the memory consumption of the network and increase its performance.')
    parser.add_argument("--pose_refine_reg", required=False, type=float, default=0.1,
                        help=f"Strength of the geodesic anchor that keeps the per-image rigid pose refinement close to the input poses (penalizes 1 - cos(theta) of the "
                             f"refinement rotation). Larger values keep the refinement smaller/more conservative; set to 0 to disable it. "
                             f"({bcolors.WARNING}NOTE{bcolors.ENDC}: this is the only regularizer acting on the rigid pose refinement in the default M=1 sampling regime)")
    parser.add_argument("--pose_refine_warmup_epochs", required=False, type=float, default=3.0,
                        help=f"Number of initial epochs over which the rigid pose refinement is ramped in from the identity. During this warmup the input poses are kept "
                             f"(almost) untouched so the conformation latent and decoder can settle before poses start being refined. Set to 0 to refine from the first step.")
    parser.add_argument("--train_inverse", action='store_true',
                        help=f"When set, HetSIREN will additionally train an inverse decoder that maps the output Gaussian positions (the deformed point cloud produced by the "
                             f"decoder) back to latent vectors. This provides a direct point-cloud -> latent encoding that complements the image-based encoder. "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: this is only meaningful when mass transport is enabled ({bcolors.ITALIC}--transport_mass{bcolors.ENDC}), since the "
                             f"Gaussian positions only move in that mode; if set without mass transport it is ignored.")
    parser.add_argument("--point_transformer", action='store_true',
                        help=f"When set, HetSIREN will use a point transformer architecture in the decoder (operating directly on the Gaussian point cloud) instead of the default "
                             f"decoder. {bcolors.WARNING}NOTE{bcolors.ENDC}: this option and {bcolors.ITALIC}--implicit_network{bcolors.ENDC} are mutually exclusive; if both are "
                             f"provided, the implicit network takes precedence and this flag is ignored. Like the implicit network, it only takes effect with mass transport "
                             f"({bcolors.ITALIC}--transport_mass{bcolors.ENDC}).")
    parser.add_argument("--kl_lambda", required=False, type=float, default=1e-3,
                        help=f"Weight (beta) of the VAE KL divergence, which regularizes the latent posterior towards a standard normal prior. Increase it for a smoother, "
                             f"more regularized latent space at the cost of reconstruction detail; decrease it (or set to 0) to let the encoder use the latent more freely. "
                             f"({bcolors.WARNING}NOTE{bcolors.ENDC}: because the reduction changed, this is not comparable to any previously used constant)")
    parser.add_argument("--grad_clip_norm", required=False, type=float, default=1.0,
                        help=f"Global-norm gradient clipping threshold for the HetSIREN optimizer. Gradients whose global norm exceeds this value are rescaled down, which "
                             f"prevents occasional gradient spikes. Set to 0 to disable clipping.")
    parser.add_argument("--loss_type", required=False, type=str, default="mse", choices=["mse", "frc", "spectral"],
                        help=f"Reconstruction (representation) loss. {bcolors.ITALIC}mse{bcolors.ENDC} (default): image-space L2 (robust on noise but blurs "
                             f"high frequencies). {bcolors.ITALIC}frc{bcolors.ENDC}: Fourier Ring Correlation (sharp on clean data, but weights noise-dominated shells equally "
                             f"and tends to overfit noise on experimental data). {bcolors.ITALIC}spectral{bcolors.ENDC}: a per-shell SSNR/Wiener-weighted spectral L2 that puts "
                             f"the noise model into the data term - it keeps MSE's robustness but down-weights the noisy high-resolution shells, so it is the recommended choice "
                             f"for noisy experimental data. {bcolors.WARNING}NOTE{bcolors.ENDC}: the point transformer decoder always uses a Fourier-space loss (frc) regardless "
                             f"of this setting; {bcolors.ITALIC}spectral{bcolors.ENDC} estimates its per-shell weights once from a representative batch of your data at startup.")
    parser.add_argument("--freq_anneal_epochs", required=False, type=float, default=0.0,
                        help=f"Coarse-to-fine (spectral) annealing: number of initial epochs over which the effective resolution is ramped in. During this window the decoder's "
                             f"first-SIREN-layer frequency (w0) and the upper limit of the FRC/spectral loss band are scaled up from {bcolors.ITALIC}--freq_anneal_start{bcolors.ENDC} "
                             f"to full resolution. This forces the network to fit low-frequency structure before high-frequency detail, which strongly stabilizes training on noisy "
                             f"data (it prevents the SIREN from overfitting high-frequency noise early). Set to 0 to disable (train at full resolution from the first step). "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: does not affect the point transformer decoder's w0 (it has no SIREN first layer), but still anneals its loss band.")
    parser.add_argument("--freq_anneal_start", required=False, type=float, default=0.2,
                        help=f"Starting fraction (in (0, 1]) for {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC}: freq_alpha begins at this value and ramps linearly to 1.0. "
                             f"Smaller values start coarser (lower resolution). Ignored when {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC} is 0.")
    parser.add_argument("--deformation_lambda", required=False, type=float, default=0.9,
                        help=f"Weight of the graph-based deformation regularization (deformation regularity + repulsion) that keeps the local geometry of the moving Gaussians "
                             f"consistent during mass transport. Lower it (e.g. 0.3-0.5) to allow sharper/larger motions at the risk of less regular deformations, raise it for "
                             f"stiffer, more regular motions. Only active with mass transport and a reference volume; ignored otherwise.")
    parser.add_argument("--decoupling_lambda", required=False, type=float, default=1e-4,
                        help=f"Weight of the decoupling regularization (encourages the latent/conformation representation to be invariant to pose and CTF). Increase it to enforce "
                             f"stronger pose/CTF decoupling, decrease it (or set to 0) to relax it.")
    parser.add_argument("--distance_preservation_lambda", required=False, type=float, default=1e-4,
                        help=f"Weight of the local distance preservation regularization (penalizes deviations of the mass-weighted point positions from the consensus/mean state, "
                             f"keeping local distances consistent). Increase it for stiffer, more locally rigid deformations, decrease it (or set to 0) for more flexibility.")
    ca.add_ssd_scratch_folder(parser)
    args = ca.parse_with_config(parser)

    # Ensure the output path exists for every mode (train creates it via the
    # metrics writer, but predict/send_to_pickle write straight into it).
    os.makedirs(args.output_path, exist_ok=True)

    # Manually handed parameters
    local_reconstruction = args.local_reconstruction
    transport_mass = args.transport_mass if not local_reconstruction else False

    # Decoder architecture: implicit network and point transformer are mutually
    # exclusive. If both are requested, the implicit network takes precedence.
    use_implicit = args.implicit_network
    use_point_transformer = args.point_transformer and not use_implicit
    if args.point_transformer and use_implicit:
        print(f"{bcolors.WARNING}Both --implicit_network and --point_transformer were set; "
              f"using the implicit decoder and ignoring --point_transformer.{bcolors.ENDC}")

    # Inverse decoder (Gaussian positions -> latent) only makes sense when the
    # Gaussians actually move, i.e. with mass transport enabled.
    train_inverse = args.train_inverse
    if train_inverse and not transport_mass:
        print(f"{bcolors.WARNING}--train_inverse requires mass transport (--transport_mass); "
              f"disabling the inverse decoder.{bcolors.ENDC}")
        train_inverse = False

    # Check that training and validation fractions add up to one
    ca.validate_dataset_split_fraction(args.dataset_split_fraction)

    # Prepare metadata
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
                        model, _, _ = fit_volume(vol * mask_fit, mask=mask_fit, iterations=20000, learning_rate=0.001, n_init=args.num_gaussians, fixed_gaussians=True)

                        model, _ = adjust_weights_to_images(model, args.md, mmap_output_dir, args.sr,
                                                            learning_rate=0.01,
                                                            num_epochs=5, is_global=True, ctf_type=args.ctf_type)

                        # Save volume
                        vol_splatted = np.array(model())
                    else:
                        initial_num_gaussians = 5000 if args.num_gaussians is None else args.num_gaussians
                        model = fit_gaussian_splat(vol, mask=mask, max_iterations=20_000, convergence_tol=1e-6, learning_rate=1e-4,
                                                   noise_std_multiplier=0.1, noise_amp_multiplier=0.1,
                                                   initial_num_gaussians=initial_num_gaussians, initial_sigma=0.5, min_sigma=0.3, quiet=True)

                        # Adjust to images
                        model = fit_weights_to_images(model, args.md, mmap_output_dir, args.sr, learning_rate=0.0001,
                                                      num_epochs=5, is_global=True, uses_ctf=args.ctf_type in ["apply", "squared", "wiener"])

                        # Save volume
                        vol_splatted = model.render(grid_shape=vol.shape)

                    # Save model
                    NeuralNetworkCheckpointer.save(model, fit_path)

                    # Save volume
                    ImageHandler().write(vol_splatted, os.path.join(args.output_path, "consensus_volume.mrc"), overwrite=True)
                else:
                    model = NeuralNetworkCheckpointer.load(checkpoint_path=fit_path)

            if transport_mass:
                # Prepare network (HetSIREN)
                factor = 0.5 * generator.md.getMetaDataImage(0).shape[0]
                if args.vol is not None:
                    if args.num_gaussians is not None:
                        coords = np.array(factor * model.means.get_value() + factor)
                        coords = np.stack([coords[..., 2], coords[..., 1], coords[..., 0]], axis=1)
                        values = np.array(jax.nn.relu(model.weights.get_value()))
                        sigma = jax.nn.relu(model.sigma_param.get_value())
                    else:
                        coords = np.array(model.get_positions())
                        coords = np.stack([coords[..., 2], coords[..., 1], coords[..., 0]], axis=1)
                        values = np.array(model.get_amplitudes())
                        sigma = model.get_sigma()
                else:
                    inds = np.asarray(np.where(mask > 0.0)).T
                    coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
                    values = jnp.zeros((inds.shape[0],))
                    sigma = 1.0
            else:
                inds = np.asarray(np.where(mask > 0.0)).T
                coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
                if args.vol is None:
                    values = jnp.zeros((inds.shape[0],))
                    sigma = 1.0
                else:
                    vol = np.array(model.render(grid_shape=vol.shape))
                    values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
                    sigma = model.get_sigma()

            # For the spectral loss, estimate its per-shell SSNR/Wiener weights
            # once from a representative batch of the actual data.
            spectral_ring_weight = None
            if args.loss_type == "spectral":
                x_rw, _ = next(iter(data_loader_train))
                if isTomoSIREN:
                    x_rw = x_rw[0]
                x_rw = jnp.asarray(x_rw, dtype=jnp.float32)
                if x_rw.ndim == 4:
                    x_rw = jnp.squeeze(x_rw, axis=-1)
                spectral_ring_weight = estimate_ring_weights(x_rw, generator.md.getMetaDataImage(0).shape[0])
                print(f"\n{bcolors.OKCYAN}Estimated spectral SSNR ring weights from a representative batch "
                      f"(shape {tuple(np.asarray(spectral_ring_weight).shape)}).{bcolors.ENDC}")

            # A zero KL weight makes the VAE prior term inactive, so drop the
            # variational machinery entirely and use a plain (deterministic)
            # autoencoder latent instead.
            use_vae = args.kl_lambda != 0
            if not use_vae:
                print(f"\n{bcolors.WARNING}--kl_lambda is 0: disabling the VAE (using a deterministic latent).{bcolors.ENDC}")

            hetsiren = HetSIREN(args.lat_dim, vol, mask, coords, values,
                                generator.md.getMetaDataImage(0).shape[0], args.sr, sigma=sigma,
                                ctf_type=args.ctf_type, decoupling=True, isVae=use_vae, transport_mass=transport_mass,
                                local_reconstruction=local_reconstruction, bank_size=10000,
                                isTomoSIREN=isTomoSIREN, is_implicit=use_implicit,
                                point_transformer=use_point_transformer, train_inverse=train_inverse,
                                loss_type=args.loss_type, spectral_ring_weight=spectral_ring_weight,
                                architecture="convnn", rngs=nnx.Rngs(model_key))
        hetsiren.train()

        # Example of training data for Tensorboard
        if hetsiren.isTomoSIREN:
            (x_example, _), labels_example = next(iter(data_loader_train))
        else:
            x_example, labels_example = next(iter(data_loader_train))
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Training data batch", x_example, dataformats="NHWC")

        # Learning rate scheduler
        # total_steps = args.epochs * len(data_loader)
        # lr_schedule = CosineAnnealingScheduler.getScheduler(peak_value=args.learning_rate, total_steps=total_steps, warmup_frac=0.1, end_value=0.0, init_value=1e-5)

        # Optimizers (HetSIREN)
        params = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('delta_volume_decoder')))
        params_inv = nnx.All(nnx.Param, nnx.PathContains('inverse_volume_decoder'))
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            tx = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adamw(args.learning_rate))
        else:
            tx = optax.adamw(args.learning_rate)
        optimizer = nnx.Optimizer(hetsiren, tx, wrt=params)
        optimizer_inv = nnx.Optimizer(hetsiren, optax.adam(1e-4), wrt=params_inv)
        graphdef, state = nnx.split((hetsiren, optimizer))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "HetSIREN_CHECKPOINT")) and not os.path.isdir(os.path.join(args.output_path, "HetSIREN_No_Inv")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "HetSIREN_CHECKPOINT"), optimizer)
            print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
        else:
            resume_epoch = 0

        if not os.path.isdir(os.path.join(args.output_path, "Intermediate_volumes")):
            os.mkdir(os.path.join(args.output_path, "Intermediate_volumes"))

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

        # Jitted prediction functions
        @jax.jit
        def predict_latent(graphdef, state, x):
            model, _ = nnx.merge(graphdef, state)
            return model(x)[0]

        image_resize = jax.jit(jax.image.resize, static_argnames=("shape", "method"))

        # Training loop (HetSIREN)
        iter_data_loader_train = iter(data_loader_train)

        if not os.path.isdir(os.path.join(args.output_path, "HetSIREN_No_Inv")):
            print(f"{bcolors.OKCYAN}\n###### Training variability... ######")

            i = 0
            pbar = tqdm(range(resume_epoch * steps_per_epoch, args.epochs * steps_per_epoch), file=sys.stdout,
                        ascii=" >=", colour="green",
                        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
            for total_steps in pbar:
                (x, labels) = next(iter_data_loader_train)

                if total_steps % steps_per_epoch == 0:
                    total_loss = 0
                    total_recon_loss = 0
                    total_validation_loss = 0

                    # Compute graph lambda
                    if args.vol is not None and hetsiren.delta_volume_decoder.transport_mass:
                        graph_lambda = args.deformation_lambda
                    else:
                        graph_lambda = 0.0
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
                    if hetsiren.isTomoSIREN:
                        x_for_tb = x[0][:5]
                    else:
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

                    if i > 0 and i % 5 == 0:
                        # Predict some heterogeneous volumes
                        latents = []
                        for _ in range(steps_per_epoch):
                            (x, labels) = next(iter_data_loader_train)
                            latent = predict_latent(graphdef, state, x)
                            latents.append(np.array(latent))
                        latents = np.concatenate(latents, axis=0)
                        kmeans = KMeans(n_clusters=20).fit(latents)
                        centers = kmeans.cluster_centers_
                        idx = 1
                        for center in centers:
                            decoded = hetsiren_decode_volume(graphdef, state, center[None, ...])
                            ImageHandler().write(np.array(decoded),
                                                 os.path.join(args.output_path, "Intermediate_volumes", f"hetsiren_{idx:02d}.mrc"),
                                                 overwrite=True)
                            idx += 1

                    # Log landscape stored in memory bank
                    if i > 0 and i % 5 == 0:
                        choice_key_use, choice_key = jax.random.split(rng, 2)
                        hetsiren_intermediate, _ = nnx.merge(graphdef, state)
                        random_indices = jnr.choice(choice_key_use,
                                                    a=jnp.arange(hetsiren_intermediate.bank_size),
                                                    shape=(hetsiren_intermediate.subset_size,), replace=False)
                        latents_intermediate = hetsiren_intermediate.memory_bank.get()[random_indices]
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

                # Ramp the rigid pose refinement in from the identity over the first epochs
                warmup_steps = max(1, int(args.pose_refine_warmup_epochs * steps_per_epoch))
                warmup_alpha = float(min(1.0, total_steps / warmup_steps))

                # Coarse-to-fine frequency annealing: ramp freq_alpha from
                # freq_anneal_start up to 1.0 over the first freq_anneal_epochs.
                # It scales the decoder's first-SIREN-layer w0 and the upper limit
                # of the (FRC/spectral) loss band. 0 epochs -> disabled (== 1.0).
                if args.freq_anneal_epochs and args.freq_anneal_epochs > 0:
                    freq_steps = max(1, int(args.freq_anneal_epochs * steps_per_epoch))
                    freq_alpha = float(args.freq_anneal_start
                                       + (1.0 - args.freq_anneal_start) * min(1.0, total_steps / freq_steps))
                else:
                    freq_alpha = 1.0

                loss, recon_loss, state, rng = train_step_hetsiren(graphdef, state, x, labels, md_columns, rng,
                                                                   l1_lambda=args.denoising_strength,
                                                                   graph_lambda=graph_lambda,
                                                                   warmup_alpha=warmup_alpha,
                                                                   pose_refine_reg=args.pose_refine_reg,
                                                                   decoupling_lambda=args.decoupling_lambda,
                                                                   distance_preservation_lambda=args.distance_preservation_lambda,
                                                                   kl_lambda=args.kl_lambda,
                                                                   freq_alpha=freq_alpha)
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

            # Save model
            NeuralNetworkCheckpointer.save(hetsiren, os.path.join(args.output_path, "HetSIREN_No_Inv"))
        else:
            hetsiren = NeuralNetworkCheckpointer.load(os.path.join(args.output_path, "HetSIREN_No_Inv"))

        graphdef, state = nnx.split((hetsiren, optimizer_inv))

        if hetsiren.train_inverse:
            print(f"{bcolors.OKCYAN}\n###### Training decoder inverse... ######")
            i = 0
            pbar = tqdm(range(int(args.epochs * steps_per_epoch)), file=sys.stdout, ascii=" >=", colour="green",
                        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

            for total_steps in pbar:
                (x, labels) = next(iter_data_loader_train)

                if total_steps % steps_per_epoch == 0:
                    total_loss = 0

                    # For progress bar (TQDM)
                    step = 1
                    step_validation = 1
                    pbar.set_description(f"Epoch {int(total_steps / steps_per_epoch + 1)}/{args.epochs}")


                    # Save checkpoint model
                    NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path,
                                                                                              "HetSIREN_CHECKPOINT"),
                                                                epoch=i)

                    i += 1

                loss, state, rng = train_step_inverse_hetsiren(graphdef, state, x, labels, md_columns, rng)
                total_loss += loss

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * steps_per_epoch)) == 0:
                    writer.add_scalar('Inverse training loss (HetSIREN)',
                                      total_loss / step,
                                      i * steps_per_epoch + step)

                # Progress bar update  (TQDM)
                if args.transport_mass:
                    pbar.set_postfix_str(f"loss={total_loss / step:.5f}")
                else:
                    pbar.set_postfix_str(f"loss={total_loss / step:.5f}")

                step += 1

        hetsiren, optimizer_inv = nnx.merge(graphdef, state)

        # Example of predicted data for Tensorboard
        x_pred_example = hetsiren_decode_image(graphdef, state, x_example, labels_example, md_columns, ctf_type=args.ctf_type, return_latent=False, corrupt_projection_with_ctf=True)
        x_pred_example = jax.vmap(min_max_scale)(x_pred_example[..., None])
        writer.add_images("Predicted images batch", x_pred_example, dataformats="NHWC")

        # Save model
        NeuralNetworkCheckpointer.save(hetsiren, os.path.join(args.output_path, "HetSIREN"))

        # Remove checkpoint (only written every 5 epochs, so it may not exist)
        checkpoint_dir = os.path.join(args.output_path, "HetSIREN_CHECKPOINT")
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

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

        md_pred = generator.md
        md_pred[:, 'latent_space'] = np.asarray([",".join(np.char.mod('%f', item)) for item in np.zeros((len(md_pred), args.lat_dim))])
        for (x, labels) in pbar:
            if isinstance(x, tuple):
                x = x[0]

            # if args.ctf_type in ["apply", "squared"]:
            #     defocusU = md_columns["ctfDefocusU"][labels]
            #     defocusV = md_columns["ctfDefocusV"][labels]
            #     defocusAngle = md_columns["ctfDefocusAngle"][labels]
            #     cs = md_columns["ctfSphericalAberration"][labels]
            #     kv = md_columns["ctfVoltage"][labels][0]
            #     ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
            #                      args.sr, [2 * hetsiren.xsize, int(2 * 0.5 * hetsiren.xsize + 1)],
            #                      x.shape[0], True)
            #     x = prepare_image_cryocrab(x, ctf)

            latents_batch, (rotations_rigid, shifts_rigid) = predict_fn(hetsiren, x)

            # Precompute batch aligments
            rotations_batch = md_columns["euler_angles"][labels]

            # Precompute batch shifts
            shifts_batch = md_columns["shifts"][labels]

            # Get rotation matrices
            if rotations_batch.ndim == 2:
                rotations_batch = euler_matrix_batch(rotations_batch[:, 0], rotations_batch[:, 1], rotations_batch[:, 2])

            # Consider refinement and rigid registration alignments
            # rotations_refined = jnp.matmul(rotations_rigid, rotations_batch)
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
