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
        return jnp.exp(logstd) * jnr.normal(rngs, shape=mean.shape) + mean

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
                logstd = jnp.clip(self.logstd_x(x), -4.0, 4.0)
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
        return jnp.exp(logstd) * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, return_last=False, *, rngs=None):
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))

        if return_last:
            return x
        else:
            if self.isVae:
                mean = self.mean_x(x)
                logstd = jnp.clip(self.logstd_x(x), -4.0, 4.0)
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
        return jnp.exp(logstd) * jnr.normal(rngs, shape=mean.shape) + mean

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
            logstd = jnp.clip(self.logstd_x(x), -4.0, 4.0)
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
                 point_transformer=False, rigid_gauge=True, rigid_gauge_irls=1, *, rngs: nnx.Rngs):
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

        # Pin the decoder's output to a canonical frame to  leave the pose head as the only thing that can explain a pose error
        self.rigid_gauge = rigid_gauge
        self.rigid_gauge_irls = int(rigid_gauge_irls)

        gauge_weights = np.asarray(reference_values, dtype=np.float32)
        if not np.any(gauge_weights != 0.0):
            gauge_weights = np.ones_like(gauge_weights)
        gauge_weights = np.maximum(gauge_weights, 0.0)
        gauge_weights = gauge_weights / max(float(gauge_weights.mean()), 1e-8)

        if transport_mass:
            self.gauge = nnx.static(RigidGauge(mode="displacement",
                                               rest_coords=np.asarray(self.coords[0]),
                                               weights=gauge_weights))
        else:
            self.gauge = nnx.static(RigidGauge(mode="density",
                                               basis=build_density_rigid_basis(np.asarray(coords),
                                                                               np.asarray(reference_values),
                                                                               volume_size),
                                               weights=gauge_weights))

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
            self.edge_index, geom_weights, self.consensus_distances, tau_edge, _, _ = build_graph_from_coordinates(self.coords[0], k_spacing=2, k_knn=6, radius_factor=1.5)

            # Mass-weight the graph
            i_e, j_e = self.edge_index
            node_mass = jnp.asarray(reference_values, dtype=jnp.float32)
            node_mass = jnp.maximum(node_mass, 0.0)
            self.node_mass = node_mass
            edge_mass = jnp.sqrt(node_mass[i_e] * node_mass[j_e] + 1e-12)
            self.edge_weights = geom_weights * edge_mass

            # Repulsion cutoff from the mass-carrying points only, so padding with low-mass
            # Gaussians cannot retune the collision scale inside the protein.
            self.tau = (jnp.sum(self.edge_weights * self.consensus_distances)
                        / (jnp.sum(self.edge_weights) + 1e-8))

        # Delta volume decoder (TODO: Check and fix hypernetwork - compare with TF implementation)
        # self.hidden_linear = [HyperLinear(in_features=lat_dim, out_features=8, in_hyper_features=lat_dim, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16)]
        # for _ in range(3):
        #     self.hidden_linear.append(HyperLinear(in_features=8, out_features=8, in_hyper_features=8, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_linear.append(HyperLinear(in_features=8, out_features=8, in_hyper_features=8, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16))

        if transport_mass:
            if self.point_transformer:
                self.geom = nnx.data(build_geometry(self.coords[0], gauss_scale="auto", hierarchical_sizes=(32, 128, 512), compute_local_frames=True))

                # Motion head based on PT
                self.point_transformer_net = PointTransformerDecoder(out_channels=3, feat_dim=32, nk=256, input_bottleneck=64,
                                                                     hierarchical_sizes=(32, 128, 512), latent_dim=lat_dim // 2,
                                                                     final_init_std=0.0, rngs=rngs)

                # Occupancy head
                hidden_values = [Siren2Linear(in_features=lat_dim // 2 + 3, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=True, w0=30.0, s=s0, use_bias=False)]
                hidden_values.append(Siren2Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=s1, use_bias=False))
                for _ in range(7):
                    hidden_values.append(Siren2Linear(in_features=32, out_features=32, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=0.0, use_bias=False))
                hidden_values.append(nnx.Linear(in_features=32, out_features=1, rngs=rngs, use_bias=False, kernel_init=nnx.initializers.zeros_init()))

                self.hidden_values = nnx.List(hidden_values)

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

    def graph_terms(self):
        """Graph quantities for the elastic priors, relative to the consensus.

        Returns ``(edge_index, edge_weights, consensus_distances, consensus_positions,
        node_mass)``. All of these are fixed buffers built once in ``__init__``.
        """
        return (self.edge_index, self.edge_weights, self.consensus_distances,
                self.coords[0], self.node_mass)

    def apply_rigid_gauge(self, field, rest_override=None, weight_override=None):
        """Pin a per-particle decoder field to the canonical (rigid-gauge) frame.

        ``field`` is ``(B, N, 3)`` -- the displacement of the rest points, in
        normalized units -- with mass transport, and ``(B, N, 1)`` -- the density
        change on the fixed points -- without. Returns the gauged field and the
        fraction of it that was rigid, i.e. how much of what the decoder just
        produced was pose rather than conformation.

        ``rest_override`` / ``weight_override`` let the caller superpose against the
        *corrected* consensus (learnable Δc0 / Δw0) rather than the static original, so
        the per-particle deformation is gauged relative to where the consensus actually
        is. They are detached: the frame is chosen by the consensus, not optimised through.

        The fraction is measured even when the gauge is switched off, so a run can be
        checked for pose absorption without changing its behaviour. When the gauge is
        unavailable (a density gauge with no reference volume) both are skipped.
        """
        if not self.gauge.available:
            return field, jnp.zeros((), dtype=jnp.float32)

        if weight_override is not None:
            weights = jnp.maximum(jax.lax.stop_gradient(weight_override), 0.0)
            weights = jnp.where(jnp.sum(weights) > 1e-8, weights, jnp.ones_like(weights))
        else:
            weights = jnp.asarray(self.gauge.weights)

        if self.gauge.mode == "displacement":
            rest = (jax.lax.stop_gradient(rest_override) if rest_override is not None
                    else jnp.asarray(self.gauge.rest_coords))
            gauged, rigid_fraction = gauge_displacement_field(field.astype(jnp.float32),
                                                              rest, weights, irls_iters=self.rigid_gauge_irls)
        else:
            gauged, rigid_fraction = gauge_density_field(field.astype(jnp.float32),
                                                         jnp.asarray(self.gauge.basis),
                                                         weights, irls_iters=self.rigid_gauge_irls)

        if not self.rigid_gauge:
            return field, rigid_fraction

        return gauged.astype(field.dtype), rigid_fraction

    def __call__(self, x, c=None, freq_alpha=1.0, occ_alpha=1.0, return_diagnostics=False):
        # freq_alpha in (0, 1] anneals the effective frequency (w0) of the first
        # SIREN layer for coarse-to-fine training; 1.0 = no change. It only
        # applies to the SIREN-based decoders (not point transformer / hybrid PE).
        #
        # occ_alpha in [0, 1] ramps the per-particle occupancy change into the density.
        # At 0 the amplitudes are exactly the (consensus) reference, so every particle
        # renders the same density and the reconstruction is explained purely by
        # transport -- large motions organise first. As it ramps to 1 the occupancy is
        # released to explain the residual that motion cannot (compositional change).
        if self.transport_mass:
            if self.point_transformer:
                # Disjoint latent subspaces for the two pathways
                x_pt, x_val = jnp.split(x, indices_or_sections=2, axis=-1)

                # Displacement
                x_coords = self.point_transformer_net(x_pt, self.geom)

                # Occupancy
                if c is None:
                    c = self.coords[0]
                c_tiled = jnp.tile(c[None, ...], (x.shape[0], 1, 1))
                x_map = jnp.concatenate([c_tiled,
                                         jnp.tile(x_val[:, None, ...], (1, c_tiled.shape[1], 1))], axis=-1)
                x_map = self.hidden_values[0](x_map, freq_alpha)
                for layer in self.hidden_values[1:-1]:
                    x_map = layer(x_map)
                x_map = self.hidden_values[-1](x_map)[..., 0]

            elif self.is_implicit:
                # Positional encoding of coords
                if c is None:
                    c = self.coords[0]
                c = jnp.tile(c[None, ...], (x.shape[0], 1, 1))

                # Adjust latents
                x = jnp.tile(x[:, None, ...], (1, c.shape[1], 1))

                # Disjoint latent subspaces for the two pathways
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
                # Disjoint latent subspaces for the two pathways
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

            # TODO: Leaving this to consider a corrected (possibly learnable) consensus
            c0, w0 = self.coords, self.reference_values

            # The deformation is where a rigid motion is cheapest to express here, so that
            # is the field the gauge acts on -- superposed against the corrected consensus
            delta_coords, rigid_fraction = self.apply_rigid_gauge(delta_coords,
                                                                  rest_override=c0[0], weight_override=w0[0])

            # Occupancy
            occ_change = occ_alpha * delta_values
            values = nnx.relu(w0 + occ_change)

            # Recover coords (non-normalized)
            coords = self.scale * (c0 + delta_coords)
        else:
            # Decode voxel values
            x_map = self.hidden_values[0](x, freq_alpha)
            for layer in self.hidden_values[1:-1]:
                x_map = layer(x_map)
            x_map = self.hidden_values[-1](x_map)

            # The positions are fixed in this mode, so the only field that can carry
            # a pose refinement is the density itself.
            x_map, rigid_fraction = self.apply_rigid_gauge(x_map[..., None])
            x_map = x_map[..., 0]

            # Recover volume values
            values = self.reference_values + x_map

            # Fixed-grid mode has no transport, so no occupancy/motion split to make.
            occ_change = jnp.zeros_like(x_map)

            # Recover coords (non-normalized)
            coords = self.scale * self.coords.repeat(x.shape[0], axis=0)

        if return_diagnostics:
            return coords, values, rigid_fraction, occ_change

        return coords, values

    def _raw_delta_coords(self, x, c, freq_alpha=1.0):
        """The coordinate head's un-gauged output at query points ``c``.

        ``c`` is ``(M, 3)`` in the decoder's normalized units, ``x`` is ``(B, lat_dim)``.
        Returns ``(B, M, 3)``. This is the piece of ``__call__``'s implicit branch that is a
        genuine function of (latent, coordinate), factored out so it can be evaluated at
        points other than the decoder's own Gaussians -- which is what makes the field
        transferable to a dense reconstruction grid without any splat/interpolate round trip.
        """
        c_t = jnp.tile(c[None, ...], (x.shape[0], 1, 1))
        x_t = jnp.tile(x[:, None, ...], (1, c_t.shape[1], 1))
        # Only the first half of the latent drives the coordinate head (__call__ splits it).
        x_coords, _ = jnp.split(x_t, indices_or_sections=2, axis=-1)

        if self.hybrid_pe:
            c_pe = positional_encoding(c_t[0], 10, self.scale)
            c_pe = jnp.tile(c_pe[None, ...], (x.shape[0], 1, 1))
            h = jnp.concatenate([c_pe, x_coords], axis=-1)
            h = nnx.elu(self.hidden_coords[0](h))
            for layer in self.hidden_coords[1:-1]:
                h = nnx.elu(layer(h))
            return self.hidden_coords[-1](h)

        h = jnp.concatenate([c_t, x_coords], axis=-1)
        h = self.hidden_coords[0](h, freq_alpha)
        for layer in self.hidden_coords[1:-1]:
            h = layer(h)
        return self.hidden_coords[-1](h)

    def decode_field_at(self, x, c_query, freq_alpha=1.0, chunk_size=None):
        """The gauged deformation field evaluated at arbitrary query coordinates.

        ``c_query`` is ``(M, 3)`` in the decoder's normalized units (i.e. the same frame as
        ``self.coords``); ``x`` is ``(B, lat_dim)``. Returns ``(B, M, 3)``, the displacement
        of those query points, in normalized units.

        The gauge is fitted on the decoder's own Gaussians and then applied to the query
        points, rather than refitted on the query set. Refitting would place the dense field
        in a different frame than the one the network was trained in -- the robust
        reweighting is driven by the point distribution, and a dense grid has a different one
        -- so the two fields would differ by an unaccounted rigid motion.

        ``chunk_size`` splits the query points so the ``(B, M, 32)`` hidden activations stay
        bounded; at M ~ 3e5 and B = 8 a single pass is ~700 MB per layer.
        """
        if not (self.transport_mass and self.is_implicit and not self.point_transformer):
            raise UserWarning("decode_field_at requires an implicit mass-transport decoder "
                              "(--transport_mass, --implicit_network, no point transformer).")

        c0, w0 = self.coords, self.reference_values

        # Fit the rigid gauge once, on the Gaussians, exactly as __call__ does.
        delta_ref = self._raw_delta_coords(x, c0[0], freq_alpha)
        if self.gauge.available and self.rigid_gauge:
            w = jnp.maximum(jax.lax.stop_gradient(w0[0]), 0.0)
            w = jnp.where(jnp.sum(w) > 1e-8, w, jnp.ones_like(w))
            transform = gauge_displacement_transform(delta_ref.astype(jnp.float32),
                                                     jax.lax.stop_gradient(c0[0]), w,
                                                     irls_iters=self.rigid_gauge_irls)
        else:
            transform = None

        def one_chunk(cq):
            d = self._raw_delta_coords(x, cq, freq_alpha)
            if transform is None:
                return d
            return apply_gauge_to_points(cq, d.astype(jnp.float32), transform)

        if chunk_size is None or chunk_size >= c_query.shape[0]:
            return one_chunk(c_query)

        return jnp.concatenate([one_chunk(c_query[i:i + chunk_size])
                                for i in range(0, c_query.shape[0], chunk_size)], axis=1)

    def decode_coords_only(self, x, c=None):
        if self.transport_mass:
            if self.point_transformer:
                # `x` is already the coords half of the latent here (the inverse decoder
                # is built with lat_dim // 2), exactly as in the implicit branch
                x_coords = self.point_transformer_net(x, self.geom)

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

            # Same gauge as __call__, or the inverse decoder would be trained against
            # a different frame than the forward one.
            c0, w0 = self.coords, self.reference_values
            x_coords, _ = self.apply_rigid_gauge(x_coords, rest_override=c0[0],
                                                 weight_override=w0[0])

            # Recover coords (non-normalized)
            coords = self.scale * (c0 + x_coords)
        else:
            # Recover coords (non-normalized)
            coords = self.scale * self.coords.repeat(x.shape[0], axis=0)

        return coords

    def decode_consensus_volume(self, filter=True, sigma=1.0):
        """Render the consensus map itself (the input reference)."""
        c0, w0 = self.coords, self.reference_values
        return self.decode_volume(coords_values=(self.scale * c0, w0), filter=filter, sigma=sigma)

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
                 is_implicit=True, isTomoSIREN=False, train_inverse=False, point_transformer=False,
                 loss_type=None, rigid_gauge=True, rigid_gauge_irls=1, *, rngs: nnx.Rngs, **kwargs):
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
        self.delta_volume_decoder = DeltaVolumeDecoder(self.coords.shape[0], lat_dim, self.xsize, self.coords, values, transport_mass=transport_mass, is_implicit=is_implicit, point_transformer=point_transformer, rigid_gauge=rigid_gauge, rigid_gauge_irls=rigid_gauge_irls, rngs=rngs)
        if self.train_inverse:
            # The inverse decoder recovers the latent that produced a point cloud, so it can
            # only recover the half that drives the coordinates
            inv_lat_dim = lat_dim // 2
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
        #   loss_type: one of {"mse", "frc"} to select the reconstruction loss
        #   explicitly. If None, defaults to "mse" (or "frc" for the point
        #   transformer decoder, which needs a Fourier-space loss).
        if loss_type is None:
            loss_type = "mse"
        self.loss_type = loss_type

        if loss_type == "frc":
            self.representation_loss_fn = FRCLoss(box_size=xsize, apix=sr, min_resolution_A=30., max_resolution_A=2. * sr)
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

    def decode_volume(self, x, motion=True, occupancy=True):
        """Decode a volume from latents, optionally ablating either pathway"""
        if x.ndim == 1:
            x = x[None, ...]

        if motion and occupancy:
            return self.delta_volume_decoder.decode_volume(x=x, filter=True, sigma=self.sigma)

        dec = self.delta_volume_decoder
        coords, values = dec(x)
        if not motion:
            coords = jnp.broadcast_to(dec.scale * dec.coords, coords.shape)
        if not occupancy:
            values = jnp.broadcast_to(dec.reference_values, values.shape)

        return dec.decode_volume(coords_values=(coords, values), filter=True, sigma=self.sigma)

    def decode_field(self, x):
        if x.ndim == 4:
            x, _ = self(x)

        coords, values = self.delta_volume_decoder(x)

        c0 = self.delta_volume_decoder.coords
        inital_coords = self.delta_volume_decoder.scale * c0
        field = coords - inital_coords

        return (field / (0.5 * self.xsize), inital_coords / (0.5 * self.xsize))

    def decode_field_at(self, x, coords_vox, chunk_size=None):
        """The deformation field sampled at arbitrary voxel coordinates.

        ``coords_vox`` is ``(M, 3)`` in (x, y, z) voxel index units of the model's own box --
        i.e. what ``numpy.where`` on a mask gives after reversing the axis order. ``x`` is a
        latent batch ``(B, lat_dim)``, or a batch of images ``(B, H, W, 1)``.

        Returns the displacement in voxels, ``(B, M, 3)``, component order (x, y, z) so it
        adds directly to ``coords_vox``.
        """
        if x.ndim == 4:
            x, _ = self(x)

        dvd = self.delta_volume_decoder
        c_query = (jnp.asarray(coords_vox, jnp.float32) - dvd.centering[0, 0]) / dvd.scale
        field = dvd.decode_field_at(x, c_query, chunk_size=chunk_size)
        return dvd.scale * field

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
                                   "decoupling_lambda", "distance_preservation_lambda", "kl_lambda", "arap_lambda",
                                   "geometric_lambda", "arap_rotation", "pose_refine_max_angle",
                                   "shift_refine_reg", "shift_refine_max",
                                   "occupancy_reg", "occupancy_tv", "occupancy_kappa"))
def train_step_hetsiren(graphdef, state, x, labels, md, key, do_update=True, l1_lambda=1e-4, graph_lambda=1e-4,
                        warmup_alpha=1.0, pose_refine_reg=0.1, decoupling_lambda=1e-4, distance_preservation_lambda=1e-4,
                        kl_lambda=1e-3, freq_alpha=1.0, arap_lambda=0.0, geometric_lambda=0.0, render_sigma=None,
                        arap_rotation="polar", latent_alpha=1.0,
                        pose_refine_max_angle=8.0, shift_refine_reg=0.1, shift_refine_max=4.0,
                        occ_alpha=1.0, occupancy_reg=0.0, occupancy_tv=1.0, occupancy_kappa=0.05,
                        amp_recon_weight=0.1,
                        ):
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

    # Mass-weighted reductions (weighted_mean=True) so the graph priors track the consensus
    # mass rather than the raw edge count; repulsion additionally takes the edge weights.
    calculate_deformation_regularity_loss_batch = jax.vmap(partial(calculate_deformation_regularity_loss, weighted_mean=True),
                                                           in_axes=(0, None, None, None))
    calculate_repulsion_loss_batch = jax.vmap(calculate_repulsion_loss, in_axes=(0, None, None, None))
    calculate_arap_loss_batch = jax.vmap(partial(calculate_arap_loss, rotation_method=arap_rotation, weighted_mean=True),
                                         in_axes=(0, None, None, None, None))

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

        # Decode volumes (freq_alpha anneals the decoder's first-SIREN-layer w0).
        #
        # latent_alpha (in [0, 1]) ramps the latent into the decoder. While it is 0
        # the decoder sees the same (zero) latent for every particle, so it can only
        # produce a single consensus volume -- and a consensus volume cannot absorb a
        # *per-particle* pose error. Every bit of the misalignment therefore has to be explained by the pose head,
        # which is what gives it a clean first shot at the residual. The heterogeneity
        # is then ramped in against poses that have already been refined
        z_dec = latent_alpha * (sample if model.isVae else latent)
        coords, values, rigid_fraction, occ_change = model.delta_volume_decoder(z_dec, freq_alpha=freq_alpha,
                                                                                occ_alpha=occ_alpha,
                                                                                return_diagnostics=True)

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

        # Priors on the pose refinement
        cos_theta_rigid = jnp.clip((jnp.trace(rotations_rigid, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
        cos_deadzone = jnp.cos(jnp.deg2rad(pose_refine_max_angle))
        pose_refine_loss = jnp.mean(nnx.relu((1.0 - cos_theta_rigid) - (1.0 - cos_deadzone)))

        # Priors on the in-plane shifts
        shifts_sq = jnp.sum(jnp.square(shifts_rigid), axis=-1)
        shift_refine_loss = jnp.mean(nnx.relu(shifts_sq - shift_refine_max ** 2))

        # Centering
        centering = model.delta_volume_decoder.centering

        # Render width: the model's fixed base sigma, or an annealed (broader)
        # width for the band-limited representation when the caller schedules 
        # it (render_sigma, coupled to the freq annealing).
        render_sigma_eff = model.sigma if render_sigma is None else render_sigma

        # Generate the projections
        if model.has_reference_volume and model.delta_volume_decoder.transport_mass:
            reference_values = model.delta_volume_decoder.reference_values
            images_corrected, _ = phys_decoder(x, values, jax.lax.stop_gradient(coords), model.xsize,
                                               rotations_refined, shifts_refined,
                                               centering, ctf, model.ctf_type, render_sigma_eff, 0.0)
            images_corrected_field, _ = phys_decoder(x, reference_values, coords, model.xsize,
                                                     rotations_refined, shifts_refined,
                                                     centering, ctf, model.ctf_type, render_sigma_eff, 0.0)
        else:
            # No consensus to pin the amplitudes to, so there is no split to make
            images_corrected, _ = phys_decoder(x, values, coords, model.xsize, rotations_refined, shifts_refined,
                                               centering, ctf, model.ctf_type, render_sigma_eff, 0.0)
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

        # Consider CTF if Wiener mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_loss = wiener2DFilter_vmap(images_corrected, ctf, pad_factor)
            images_corrected_field_loss = wiener2DFilter_vmap(images_corrected_field, ctf, pad_factor)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter(x, ctf, pad_factor=pad_factor)
            images_corrected_loss = ctfFilter_vmap(images_corrected, ctf, pad_factor)
            images_corrected_field_loss = ctfFilter_vmap(images_corrected_field, ctf, pad_factor)
        else:
            x_loss = x
            images_corrected_loss = images_corrected
            images_corrected_field_loss = images_corrected_field

        if M > 1:
            x_loss = x_loss[:, None, ...]

        # Projection mask
        x_loss = x_loss * projected_mask
        images_corrected_loss = images_corrected_loss * projected_mask
        images_corrected_field_loss = images_corrected_field_loss * projected_mask

        # Split the reconstruction between the amplitude pathway (learned values at
        # frozen coords -> images_corrected) and the mass-transport pathway (fixed
        # reference values at learned coords -> images_corrected_field)
        recon_loss = (amp_recon_weight * model.representation_loss_fn(images_corrected_loss, x_loss, freq_alpha)
                      + (1.0 - amp_recon_weight)
                      * model.representation_loss_fn(images_corrected_field_loss, x_loss, freq_alpha))

        # if model.delta_volume_decoder.transport_mass:

        # Keep the per-sample (and, when M>1, per-pose) loss here. It must NOT be
        # collapsed to a scalar before the M>1 importance weighting below, which
        # needs a (B, M) tensor. For M=1 the downstream `.mean()` reproduces the
        # previous scalar exactly, so this is behaviour-preserving.
        recons_loss_all = recon_loss
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

        # Local distance preservation (decoded through the same latent ramp, or it
        # would compare two different decoder inputs)
        if model.isVae:
            coords_mean, values_mean = model.delta_volume_decoder(latent_alpha * latent, freq_alpha=freq_alpha)
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
            # Responsibility (importance) weighting over the M sampled poses.
            # Detach the weights so the gradient flows through the per-pose losses
            # (soft-EM style), not through the softmax that produced the weights.
            w_pose, _ = importance_weights(recons_loss_all, log_q)
            w_pose = jax.lax.stop_gradient(w_pose)
            # Sum over the pose axis (-1), then average over the batch.
            nll = jnp.sum(w_pose * recons_loss_all, axis=-1).mean()
            kl_pose = PoseDistMatrix.kl_to_isotropic_prior(rotations_logscale, prior_log_scale=0.0).mean()
        else:
            nll = recons_loss_all.mean()
            kl_pose = 0.0

        # Graph based loss
        if model.has_reference_volume and model.delta_volume_decoder.transport_mass:
            # Graph terms relative to the (possibly learnable) corrected consensus, so the
            # priors do not fight a consensus correction.
            radius_graph, edge_weights, consensus_distances, consensus_positions, _ = \
                model.delta_volume_decoder.graph_terms()
            deformed_positions = coords / model.delta_volume_decoder.scale
            tau = model.delta_volume_decoder.tau

            # Losses
            loss_def_regularity = calculate_deformation_regularity_loss_batch(deformed_positions, radius_graph,
                                                                              consensus_distances, edge_weights)
            loss_repulsion = calculate_repulsion_loss_batch(deformed_positions, radius_graph, tau, edge_weights)

            # As-rigid-as-possible term: penalise only the non-rigid part of the
            # local deformation (factors out the best local rotation), so hinge /
            # domain motions are not over-stiffened while noise-driven shear is
            # resisted. Rest positions are the corrected consensus; deformed_positions are
            # in the same normalized frame.
            if arap_lambda > 0.0:
                num_points = deformed_positions.shape[1]
                loss_arap = calculate_arap_loss_batch(deformed_positions, consensus_positions,
                                                      radius_graph, edge_weights, num_points).mean()
            else:
                loss_arap = 0.0

            # Total loss
            loss_graph = (loss_def_regularity + 0.01 * loss_repulsion).mean() + arap_lambda * loss_arap
        else:
            loss_graph = 0.0

        # Occupancy prior (transport mode)
        if occupancy_reg > 0.0 and model.delta_volume_decoder.transport_mass and model.has_reference_volume:
            edge_idx, edge_mass, _, _, node_mass = model.delta_volume_decoder.graph_terms()
            node_mass = jax.lax.stop_gradient(node_mass)   # (N,); mass gates the prior, not optimised through it
            edge_mass = jax.lax.stop_gradient(edge_mass)   # (E,)
            occ_l1 = jnp.mean(jnp.sum(jnp.abs(occ_change) / (node_mass[None, :] + occupancy_kappa), axis=-1))

            i_e, j_e = edge_idx
            occ_tv = jnp.mean(jnp.sum(edge_mass[None, :] * jnp.abs(occ_change[:, i_e] - occ_change[:, j_e]), axis=-1)
                              / (jnp.sum(edge_mass) + 1e-8))
            loss_occupancy = occ_l1 + occupancy_tv * occ_tv
        else:
            loss_occupancy = 0.0

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

        # Geometric correction
        if geometric_lambda > 0.0:
            rot_ctx = rotations_refined if M == 1 else rotations_refined[:, 0]
            geom_ctx = (rot_ctx, shifts_refined, ctf)

            def _render_point(z, ctx_i):
                rot_i, shift_i, ctf_i = ctx_i
                z_b = latent_alpha * z[None, :]
                coords_z, values_z = model.delta_volume_decoder(z_b, freq_alpha=freq_alpha)
                img, _ = model.phys_decoder(z_b, values_z, coords_z, model.xsize,
                                            rot_i[None], shift_i[None], centering,
                                            ctf_i[None], model.ctf_type, render_sigma_eff, 0.0)
                return img[0].reshape(-1)  # (xsize*xsize,)

            # Exact (all pixels). For large boxes pass e.g.
            # pixel_subsample=16384, key=choice_key to switch to the unbiased O(k*d) Gram estimato.
            loss_geometric = composed_geometric_correction_loss(
                latent, _render_point, geom_ctx, model.xsize * model.xsize)
        else:
            loss_geometric = 0.0

        # The KL rides the latent ramp: while the decoder ignores the latent, the
        # reconstruction cannot push the posterior away from the prior, so a KL at
        # full strength would just collapse a latent that is not being used yet
        loss = (nll + kl_lambda * latent_alpha * kl_loss + 0.000001 * kl_pose + decoupling_lambda * decoupling_loss
                + l1_lambda * l1_loss + graph_lambda * loss_graph + 100. * hist_loss + distance_preservation_lambda * loss_dp
                + pose_refine_reg * pose_refine_loss + shift_refine_reg * shift_refine_loss
                + geometric_lambda * loss_geometric + occupancy_reg * loss_occupancy)

        # Pose diagnostics. `rigid_fraction` is how much of what the decoder just
        # produced lives in the rigid subspace, i.e. how much pose the decoder is
        # absorbing; it is measured even when the gauge is off. If the refinement
        # angle and shift sit at ~0 while this climbs, the decoder is eating the pose
        #
        # `occ_fraction` is the share of the decoded density change explained by
        # occupancy rather than motion: mean |occ_change| relative to the mean absolute
        # deviation of the amplitudes from the reference. If it climbs while the
        # deformation field goes quiet, the occupancy prior is too weak (compositional
        # change is being explained as amplitude, not caught as motion, or vice versa)
        theta_deg = jnp.rad2deg(jnp.arccos(jnp.clip(cos_theta_rigid, -1.0 + 1e-6, 1.0 - 1e-6)))
        occ_fraction = jnp.mean(jnp.abs(occ_change)) / (jnp.mean(jnp.abs(values)) + 1e-8)

        # The size of the deformation that survives the gauge, in Angstrom -- the only part
        # that reaches the density, and the only part a downstream motion correction can use
        deformation_A = jnp.mean(jnp.linalg.norm(
            coords - model.delta_volume_decoder.scale * model.delta_volume_decoder.coords,
            axis=-1)) * model.sr

        metrics = {"pose_angle_deg": jnp.mean(theta_deg),
                   "shift_norm_px": jnp.mean(jnp.sqrt(shifts_sq + 1e-8)),
                   "rigid_fraction": rigid_fraction,
                   "deformation_A": deformation_A,
                   "occ_fraction": occ_fraction}
        metrics = jax.lax.stop_gradient(metrics)

        return loss, (recon_loss.mean(), latent, metrics)

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
        (loss, (recon_loss, latent, metrics)), grads = grad_fn(model, (x, subtomogram_label))
    else:
        (loss, (recon_loss, latent, metrics)), grads = grad_fn(model, x)

    if do_update:
        grads, _ = grads.split(params, ...)

        optimizer.update(model, grads)

        # Update memory bank
        model.memory_bank.enqueue(latent)

        state = nnx.state((model, optimizer))

        return loss, recon_loss, metrics, state, key
    else:
        return loss, recon_loss, metrics


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

        # Only the coords half of the latent is recoverable from the point cloud
        latent_no_grad = jax.lax.stop_gradient(latent)[..., :model.lat_dim // 2]
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

    calculate_deformation_regularity_loss_batch = jax.vmap(partial(calculate_deformation_regularity_loss, weighted_mean=True),
                                                           in_axes=(0, None, None, None))
    calculate_repulsion_loss_batch = jax.vmap(calculate_repulsion_loss, in_axes=(0, None, None, None))

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
            loss_repulsion = calculate_repulsion_loss_batch(deformed_positions, radius_graph, tau, edge_weights)

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
    from hax.metrics import JaxSummaryWriter, TrainingLogger
    from hax.programs import fit_gaussian_splat, fit_weights_to_images, fit_volume, adjust_weights_to_images
    from hax.schedulers import CosineAnnealingScheduler

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
    ca.add_logging_args(parser)
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
                        help=f"Strength of the hinge that keeps the per-image rigid pose refinement within {bcolors.ITALIC}--pose_refine_max_angle{bcolors.ENDC} of the input poses. "
                             f"It penalizes only the excess beyond that angle, so a genuinely-needed small correction is not pushed back to zero. Set to 0 to disable it.")
    parser.add_argument("--pose_refine_max_angle", required=False, type=float, default=8.0,
                        help=f"Deadzone (in degrees) of the pose-refinement hinge: rotations up to this angle are free, beyond it they are penalized with weight "
                             f"{bcolors.ITALIC}--pose_refine_reg{bcolors.ENDC}. This is what 'stay a refinement' should mean -- do not wander further than a plausible alignment error. "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: setting it to {bcolors.ITALIC}0{bcolors.ENDC} recovers the old one-sided pull towards the identity, which penalizes "
                             f"even the sub-degree corrections you want and biases the model into letting the decoder absorb the pose instead.")
    parser.add_argument("--shift_refine_reg", required=False, type=float, default=0.1,
                        help=f"Strength of the hinge on the per-image in-plane shift refinement, the exact analogue of {bcolors.ITALIC}--pose_refine_reg{bcolors.ENDC} for translations. "
                             f"Set to 0 to disable. ({bcolors.WARNING}NOTE{bcolors.ENDC}: the shift refinement previously had no prior at all, even though a 2D image shift is exactly a "
                             f"3D translation of the decoded object and is therefore the most trivially absorbable part of the refinement)")
    parser.add_argument("--shift_refine_max", required=False, type=float, default=4.0,
                        help=f"Deadzone (in pixels) of the shift-refinement hinge: shifts up to this magnitude are free, beyond it they are penalized with weight "
                             f"{bcolors.ITALIC}--shift_refine_reg{bcolors.ENDC}.")
    parser.add_argument("--pose_refine_warmup_epochs", required=False, type=float, default=0.0,
                        help=f"Number of initial epochs over which the rigid pose refinement is ramped in from the identity. {bcolors.WARNING}NOTE{bcolors.ENDC}: the refinement heads are "
                             f"zero-initialized, so the model already starts at exactly the input poses -- the ramp buys nothing at initialization and only scales the pose head's gradient "
                             f"towards zero during precisely the window in which the decoder is learning to absorb the misalignment. It now defaults to 0 (refine from the first step); the "
                             f"decoder is instead held at a consensus early on, see {bcolors.ITALIC}--latent_warmup_epochs{bcolors.ENDC}.")
    parser.add_argument("--latent_warmup_epochs", required=False, type=float, default=None,
                        help=f"Number of initial epochs over which the latent is ramped INTO the decoder (default: {bcolors.ITALIC}~12%% of --epochs{bcolors.ENDC}, so the schedule scales with the budget). "
                             f"For the first half of this window the decoder sees a zero latent, so it can only "
                             f"produce a single consensus volume -- and a consensus volume cannot absorb a {bcolors.ITALIC}per-particle{bcolors.ENDC} pose error, so the whole misalignment "
                             f"has to be explained by the pose head. Over the second half the latent (and the KL term, which rides the same clock) ramps in linearly, against poses that have "
                             f"already been refined. This is the coarse-to-fine ordering a classical refinement uses -- consensus and poses first, heterogeneity after -- and it costs nothing. "
                             f"Set to 0 to feed the full latent from the first step.")
    parser.add_argument("--rigid_gauge", required=False, type=str, default="core", choices=["core", "mass", "off"],
                        help=f"Pins the decoder's per-particle output to a canonical frame, so a pose error cannot be absorbed by the decoder and has nowhere to go except the pose head "
                             f"(rotating the decoded object and refining the image pose produce the SAME projection, so without this the two are interchangeable -- and since every geometric "
                             f"prior here is rigid-invariant while the pose head is anchored to the identity, the decoder is the cheaper place for the model to put a misalignment). With "
                             f"{bcolors.ITALIC}--transport_mass{bcolors.ENDC} the deformed cloud is rigidly superposed back onto the rest cloud (a weighted Kabsch, one 3x3 solve per particle); "
                             f"without it, the linearized rigid perturbations of the reference density are projected out instead. "
                             f"{bcolors.WARNING}It does NOT remove motion{bcolors.ENDC}: the correction is an isometry of the decoded object, so internal distances, hinge angles and relative "
                             f"domain motions are preserved exactly -- what it fixes is the FRAME the motion is reported in, and that frame is chosen by the weights: "
                             f"{bcolors.ITALIC}core{bcolors.ENDC} (default) reweights against the residual so the frame is set by the least-moving part of the structure (the rigid core), which "
                             f"leaves domain motions entirely in the deformation field. {bcolors.ITALIC}mass{bcolors.ENDC} uses the mass-weighted optimal superposition instead, in which a large "
                             f"domain motion makes the 'stationary' domain appear to counter-rotate and lets the pose head absorb a slice of the conformational change. {bcolors.ITALIC}off"
                             f"{bcolors.ENDC} disables the correction but still reports the diagnostic, so a run can be checked for pose absorption without changing its behaviour. "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: without mass transport the gauge needs a reference density to differentiate, so it is unavailable (and silently a no-op) on "
                             f"fully ab-initio runs with no {bcolors.ITALIC}--vol{bcolors.ENDC}.")
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
    parser.add_argument("--loss_type", required=False, type=str, default="mse", choices=["mse", "frc"],
                        help=f"Reconstruction (representation) loss. {bcolors.ITALIC}mse{bcolors.ENDC} (default): image-space L2 (robust on noise but blurs "
                             f"high frequencies). {bcolors.ITALIC}frc{bcolors.ENDC}: Fourier Ring Correlation (sharp on clean data, but weights noise-dominated shells equally, so "
                             f"on experimental data it must be band-limited to a trusted resolution to avoid overfitting noise - see {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC} "
                             f"and {bcolors.ITALIC}--sigma_anneal_factor{bcolors.ENDC}, which together reproduce the EMAN2/e2gmm recipe "
                             f"of a band-limited FRC over a band-limited Gaussian representation). {bcolors.WARNING}NOTE{bcolors.ENDC}: the point transformer decoder always uses "
                             f"a Fourier-space loss (frc) regardless of this setting.")
    parser.add_argument("--freq_anneal_epochs", required=False, type=float, default=0.0,
                        help=f"Coarse-to-fine annealing: number of initial epochs over which the effective resolution is ramped in. During this window the decoder's "
                             f"first-SIREN-layer frequency (w0) and the upper limit of the FRC loss band are scaled up from {bcolors.ITALIC}--freq_anneal_start{bcolors.ENDC} "
                             f"to full resolution (and, if {bcolors.ITALIC}--sigma_anneal_factor{bcolors.ENDC} > 1, the Gaussian render width is annealed on the same clock). This "
                             f"forces the network to fit low-frequency structure before high-frequency detail, which strongly stabilizes training on noisy data (it prevents the "
                             f"SIREN from overfitting high-frequency noise early). Set to 0 to disable (train at full resolution from the first step). "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: does not affect the point transformer decoder's w0 (it has no SIREN first layer), but still anneals its loss band.")
    parser.add_argument("--freq_anneal_start", required=False, type=float, default=0.2,
                        help=f"Starting fraction (in (0, 1]) for {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC}: freq_alpha begins at this value and ramps linearly to 1.0. "
                             f"Smaller values start coarser (lower resolution). Ignored when {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC} is 0.")
    parser.add_argument("--sigma_anneal_factor", required=False, type=float, default=1.0,
                        help=f"EMAN2/e2gmm-style band-limited representation for the FRC loss. The Gaussian render width used in the training loss starts at "
                             f"{bcolors.ITALIC}sigma_anneal_factor x base_sigma{bcolors.ENDC} (broad, low-resolution blobs) and shrinks back to the base width on the SAME coarse-to-fine "
                             f"clock as {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC}. Broadening the Gaussians makes the density itself intrinsically band-limited early on, so the "
                             f"amplitude-invariant FRC cannot pour gradient into high-resolution noise shells the model is not yet allowed to represent (this is the safeguard that lets "
                             f"EMAN2 use FRC on experimental data). Only affects mass-transport rendering. Requires {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC} > 0 (it shares that "
                             f"clock). Set to 1.0 to disable (default; fixed base width). Typical: 2-2.5 - the render blur uses a fixed 9-px kernel, so a broadened width beyond "
                             f"~2.5x the base sigma is progressively truncated (still broadens, just sub-linearly), which bounds the useful range.")
    parser.add_argument("--schedule_recon_split", action='store_true',
                        help=f"Ramp the reconstruction split over training instead of holding it fixed. When set, the weight of the amplitude term ramps linearly from "
                             f"{bcolors.ITALIC}--recon_split_start{bcolors.ENDC} to {bcolors.ITALIC}--recon_split_final{bcolors.ENDC} over {bcolors.ITALIC}--recon_split_epochs"
                             f"{bcolors.ENDC} (the motion term takes the complement). Starting near 0 makes early training explain the data almost purely by transport, so large "
                             f"motions organise before occupancy is allowed in. Only meaningful with mass transport ({bcolors.ITALIC}--transport_mass{bcolors.ENDC}). Default: off "
                             f"(fixed {bcolors.ITALIC}--recon_split_final{bcolors.ENDC} throughout).")
    parser.add_argument("--recon_split_start", required=False, type=float, default=0.0,
                        help=f"Amplitude-term weight at the START of training when {bcolors.ITALIC}--schedule_recon_split{bcolors.ENDC} is on. {bcolors.ITALIC}0.0{bcolors.ENDC} "
                             f"(default) means the data is explained purely by mass transport at the beginning.")
    parser.add_argument("--recon_split_final", required=False, type=float, default=0.1,
                        help=f"Amplitude-term weight at the END of the ramp (and the fixed weight used throughout when {bcolors.ITALIC}--schedule_recon_split{bcolors.ENDC} is off). "
                             f"The reconstruction loss is {bcolors.ITALIC}w x L(learned amplitudes, frozen coords) + (1-w) x L(reference amplitudes, learned coords){bcolors.ENDC}. "
                             f"The second term can only be reduced by moving mass, so it is what forces motion to be modelled as transport; the first lets occupancy explain the "
                             f"residual transport cannot, without being able to leak gradient into the coordinate head. Default {bcolors.ITALIC}0.1{bcolors.ENDC} (90%% transport). "
                             f"Raise it toward 0.5 if genuine compositional change is being forced into motion; lower it toward 0 for pure transport.")
    parser.add_argument("--recon_split_epochs", required=False, type=float, default=0.0,
                        help=f"Number of initial epochs over which the amplitude-term weight ramps from {bcolors.ITALIC}--recon_split_start{bcolors.ENDC} to "
                             f"{bcolors.ITALIC}--recon_split_final{bcolors.ENDC}. Only used with {bcolors.ITALIC}--schedule_recon_split{bcolors.ENDC}; if left at 0 the ramp "
                             f"falls back to the {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC} clock.")
    parser.add_argument("--occupancy_reg", required=False, type=float, default=0.0,
                        help=f"Weight of the occupancy prior, which lets HetSIREN model {bcolors.ITALIC}compositional{bcolors.ENDC} heterogeneity (a domain/ligand appearing or "
                             f"disappearing between particles) rather than forcing everything through the deformation field. Each Gaussian carries a per-particle amplitude change on top "
                             f"of the reference; this penalises it with a {bcolors.ITALIC}scale-normalised L1{bcolors.ENDC} (cost = fold change where the reference has density, so "
                             f"modulating existing density is cheap, while creating mass where the reference is empty costs {bcolors.ITALIC}|occ|/--occupancy_kappa{bcolors.ENDC} -- possible "
                             f"but expensive). Set to 0 (default) to disable occupancy entirely (pure mass transport). Only meaningful with mass transport ({bcolors.ITALIC}--transport_mass"
                             f"{bcolors.ENDC}) and a reference. {bcolors.WARNING}NOTE{bcolors.ENDC}: watch the {bcolors.ITALIC}occ_fraction{bcolors.ENDC} diagnostic -- if it climbs while the "
                             f"pose/deformation goes quiet, lower this or {bcolors.ITALIC}--occupancy_kappa{bcolors.ENDC}.")
    parser.add_argument("--occupancy_kappa", required=False, type=float, default=0.05,
                        help=f"Creation threshold of the occupancy prior (as a density, in the reference's gray scale; default {bcolors.ITALIC}0.05{bcolors.ENDC}). It is added to the "
                             f"reference amplitude in the L1 denominator, so it sets how much evidence it takes to invent mass where the reference is (near) empty: smaller = harder to create "
                             f"mass, larger = easier. Where the reference already has substantial density it has little effect (the cost is a fold change). Only used when "
                             f"{bcolors.ITALIC}--occupancy_reg{bcolors.ENDC} > 0.")
    parser.add_argument("--occupancy_tv", required=False, type=float, default=1.0,
                        help=f"Relative weight of the spatial-coherence (graph total-variation) term of the occupancy prior (default {bcolors.ITALIC}1.0{bcolors.ENDC}). It penalises "
                             f"occupancy {bcolors.ITALIC}differences{bcolors.ENDC} between neighbouring Gaussians, so a whole domain switching on together pays only at its boundary while "
                             f"scattered noise pays everywhere -- this is what separates a genuine compositional domain from amplitude noise. Raise it for blockier occupancy, lower it toward "
                             f"0 to allow finer-grained occupancy. Only used when {bcolors.ITALIC}--occupancy_reg{bcolors.ENDC} > 0.")
    parser.add_argument("--occupancy_warmup_epochs", required=False, type=float, default=3.0,
                        help=f"Number of initial epochs over which occupancy is ramped in (default {bcolors.ITALIC}3.0{bcolors.ENDC}). While it ramps, the amplitudes stay at the reference "
                             f"so the data is explained purely by transport -- large motions organise first -- and occupancy is then released to explain only the residual that motion cannot "
                             f"(compositional change). Feeding occupancy from step 0 lets it short-circuit the motion (it is a strictly easier way to fit any image), so a non-zero warmup is "
                             f"recommended. Only used when {bcolors.ITALIC}--occupancy_reg{bcolors.ENDC} > 0; set to 0 to use occupancy from the first step.")
    parser.add_argument("--arap_lambda", required=False, type=float, default=0.0,
                        help=f"Weight of the as-rigid-as-possible (ARAP) deformation prior. Unlike the distance-preservation term, ARAP factors out the best local rotation per "
                             f"Gaussian before penalizing the deformation, so locally rigid motions (hinges, domain rotations) are NOT over-stiffened while noise-driven non-rigid "
                             f"shear is still resisted. Only active with mass transport and a reference volume. Set to 0 to disable (default); a small value (e.g. 0.05-0.2) is a good "
                             f"starting point, tuned together with {bcolors.ITALIC}--deformation_lambda{bcolors.ENDC}/{bcolors.ITALIC}--distance_preservation_lambda{bcolors.ENDC}.")
    parser.add_argument("--arap_rotation", required=False, type=str, default="polar", choices=["polar", "svd"],
                        help=f"How the per-Gaussian optimal rotation in the ARAP prior ({bcolors.ITALIC}--arap_lambda{bcolors.ENDC}) is recovered. {bcolors.ITALIC}polar{bcolors.ENDC} "
                             f"(default) uses a matmul-only scaled polar iteration that is ~15-20x faster than the SVD path on GPU (batched 3x3 SVD is a severe XLA:GPU bottleneck) "
                             f"and numerically matches it to ~1e-6. {bcolors.ITALIC}svd{bcolors.ENDC} uses the exact SVD; keep it as a reference/fallback (run both and compare the ARAP "
                             f"loss to self-check). No effect unless {bcolors.ITALIC}--arap_lambda{bcolors.ENDC} > 0.")
    parser.add_argument("--geometric_lambda", required=False, type=float, default=0.0,
                        help=f"Weight of the geometric-correction loss. It penalises the local volume distortion of the heterogeneity decoder via the Gram determinant "
                             f"{bcolors.ITALIC}det(J_D(z)^T J_D(z)){bcolors.ENDC} of its Jacobian (the derivative of the decoded deformation w.r.t. the latent), pushing the decoded "
                             f"latent-to-shape map towards a locally volume-preserving (isometric) embedding. Set to 0 to disable (default). {bcolors.WARNING}NOTE{bcolors.ENDC}: computing "
                             f"the decoder Jacobian is expensive (it runs the decoder once per latent dimension and is differentiated again by the optimizer), so only enable it when the "
                             f"geometric term is actually needed.")
    parser.add_argument("--lr_schedule", action='store_true',
                        help=f"Use a warmup + cosine-decay learning-rate schedule for the HetSIREN optimizer instead of a constant learning rate. The LR ramps linearly from "
                             f"{bcolors.ITALIC}1e-5{bcolors.ENDC} to {bcolors.ITALIC}--learning_rate{bcolors.ENDC} over the first 10%% of training, then follows a cosine decay down to "
                             f"0 by the final epoch. This warms up the SIREN safely (avoiding early high-frequency instabilities) and anneals the step size at the end for a sharper, "
                             f"more stable final model. Resumes correctly from a checkpoint (the schedule follows the optimizer step count). Default: off (constant learning rate).")
    parser.add_argument("--ema_decay", required=False, type=float, default=0.0,
                        help=f"Exponential moving average (Polyak averaging) of the trainable weights. When > 0 (typical: {bcolors.ITALIC}0.999{bcolors.ENDC}), a running average "
                             f"{bcolors.ITALIC}ema = decay*ema + (1-decay)*weights{bcolors.ENDC} is maintained during training and the AVERAGED weights (not the last raw ones) are "
                             f"exported as the final model. This smooths out the noise in the late-training weight trajectory and typically yields a cleaner, higher-resolution "
                             f"reconstruction on noisy data. The averaged forward model is also used as the (frozen) target when training the decoder inverse. Set to 0 to disable "
                             f"(default). {bcolors.WARNING}NOTE{bcolors.ENDC}: the EMA buffer is checkpointed, so a training resume restores the running average exactly (legacy "
                             f"checkpoints without an EMA buffer fall back to re-seeding it from the resumed weights).")
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
    # metrics writer, but predict writes straight into it).
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

    # Prepare grain dataset
    if not args.load_images_to_ram and args.mode in ["train", "predict"]:
        mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4, precision=np.float16, group_size=1, shard_size=10000)
        scratch_dir = generator.mmap_output_dir
    else:
        mmap_output_dir = None
        scratch_dir = None

    # Reconstruct a consensus volume if neither --vol nor --mask were provided -- but only when
    # the model is actually going to be built from them
    auto_reference = (args.vol is None and args.mask is None
                      and args.reload is None and args.mode == "train")
    if auto_reference:
        os.makedirs(args.output_path, exist_ok=True)
        consensus = reconstruct_consensus_volume(generator.md, md_columns, args.sr,
                                                 use_ctf=args.ctf_type not in (None, "None"),
                                                 scratch_dir=scratch_dir)
        consensus_path = os.path.join(args.output_path, "consensus_reconstruction.mrc")
        ImageHandler().write(consensus, consensus_path, overwrite=True)
        args.vol = consensus_path
        print(f"{bcolors.OKGREEN}Consensus volume reconstructed from the input poses -> {consensus_path}"
              f"{bcolors.ENDC}")

    # Preprocess volume (and mask)
    if args.vol is not None:
        vol = ImageHandler(args.vol).getData()
    else:
        volume_size = generator.md.getMetaDataImage(0).shape[0]
        local_reconstruction = False
        vol = np.zeros((volume_size, volume_size, volume_size))

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    elif auto_reference:
        # Derived from the map reconstructed
        mask = consensus_mask(vol)
        ImageHandler().write(mask, os.path.join(args.output_path, "consensus_mask.mrc"), overwrite=True)
    elif args.transport_mass and args.vol is not None:
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


    # Train network
    if args.mode == "train":

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "HetSIREN_metrics"))

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
            fit_gaussians = args.vol is not None and (transport_mass or not auto_reference)
            if fit_gaussians:
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

                        model, _ = adjust_weights_to_images(model, args.md, mmap_output_dir, args.sr,
                                                            learning_rate=0.01,
                                                            num_epochs=5, is_global=True, ctf_type=args.ctf_type)

                        # Save volume
                        vol_splatted = np.array(model())
                    else:
                        initial_num_gaussians = 5000 if args.num_gaussians is None else args.num_gaussians
                        model = fit_gaussian_splat(vol, mask=mask_fit, max_iterations=20_000, convergence_tol=1e-6, learning_rate=1e-4,
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
                elif not fit_gaussians:
                    values = np.asarray(vol)[inds[:, 0], inds[:, 1], inds[:, 2]]
                    sigma = 1.0
                else:
                    vol = np.array(model.render(grid_shape=vol.shape))
                    values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
                    sigma = model.get_sigma()

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
                                loss_type=args.loss_type,
                                rigid_gauge=args.rigid_gauge != "off",
                                rigid_gauge_irls=1 if args.rigid_gauge == "core" else 0,
                                architecture="convnn", rngs=nnx.Rngs(model_key))

            if args.rigid_gauge != "off" and not hetsiren.delta_volume_decoder.gauge.available:
                print(f"\n{bcolors.WARNING}--rigid_gauge {args.rigid_gauge} was requested, but without mass transport the gauge needs a reference "
                      f"density to build its basis and none was given (--vol). The projection is disabled; the pose refinement is still protected by "
                      f"--latent_warmup_epochs and the refinement hinges.{bcolors.ENDC}")
        hetsiren.train()

        # Resolve ``--batch_size auto`` before the data loader and steps_per_epoch
        # (both depend on it) are built. The largest memory-safe batch is estimated
        # analytically from the peak memory of ``train_step_hetsiren`` -- no
        # execution, no OOM probing (see hax.utils.estimate_batch_size). A throwaway
        # optimizer of the same structure as the real one gives the correct
        # parameter/optimizer memory footprint here without needing the LR schedule
        # (which itself depends on steps_per_epoch, hence on the batch size). Any
        # failure falls back to a fixed default so a run never aborts here.
        if args.batch_size == "auto":
            from hax.utils import estimate_batch_size
            estimated = None
            if hetsiren.isTomoSIREN:
                print(f"{bcolors.WARNING}\nAutomatic batch size is not supported in Tomo mode; "
                      f"using a fixed batch size instead.{bcolors.ENDC}")
            else:
                params_probe = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('delta_volume_decoder')))
                params_inv_probe = nnx.All(nnx.Param, nnx.PathContains('inverse_volume_decoder'))
                probe_optimizer = nnx.Optimizer(hetsiren, optax.adamw(args.learning_rate), wrt=params_probe)
                probe_graphdef, probe_state = nnx.split((hetsiren, probe_optimizer))

                # Headroom for buffers this probe does not see but the real run
                # allocates after this point: the EMA weight copy (one extra copy
                # of the trainable params) and the inverse-decoder Adam optimizer
                # (mu + nu, i.e. ~2x its params). The main optimizer is already
                # reflected via bytes_in_use (probe_optimizer is resident now).
                def _nbytes(tree):
                    total = 0
                    for leaf in jax.tree_util.tree_leaves(tree):
                        nb = getattr(leaf, "nbytes", None) or getattr(getattr(leaf, "value", None), "nbytes", None)
                        if nb:
                            total += int(nb)
                    return total
                reserve_bytes = 2 * _nbytes(nnx.state(hetsiren, params_inv_probe))
                if args.ema_decay and args.ema_decay > 0.0:
                    reserve_bytes += _nbytes(nnx.state(hetsiren, params_probe))

                estimated = estimate_batch_size(
                    probe_graphdef, probe_state, train_step_hetsiren, md_columns, rng,
                    (hetsiren.xsize, hetsiren.xsize, 1),
                    reserved_bytes=reserve_bytes,
                    step_kwargs=dict(do_update=True))

                # Release the probe's optimizer buffers so they do not transiently
                # coexist with the real optimizer built below.
                del probe_optimizer, probe_graphdef, probe_state
            args.batch_size = estimated if estimated is not None else 8
            if estimated is None:
                print(f"{bcolors.WARNING}Falling back to --batch_size {args.batch_size}.{bcolors.ENDC}")

        # Training data loader and steps_per_epoch (batch size is now concrete).
        data_loader_train = generator.return_grain_dataset(batch_size=args.batch_size, shuffle="global_data_loader",
                                                           num_epochs=None, num_workers=-1, num_threads=1,
                                                           load_to_ram=args.load_images_to_ram)
        steps_per_epoch = int(len(generator.md) / args.batch_size)

        # Example of training data for Tensorboard
        if hetsiren.isTomoSIREN:
            (x_example, _), labels_example = next(iter(data_loader_train))
        else:
            x_example, labels_example = next(iter(data_loader_train))
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Training data batch", x_example, dataformats="NHWC")

        # Curriculum warmups, resolved to absolute epoch counts. Left unset (the default)
        # they scale as a fraction of --epochs
        _wfloor = 2.0
        latent_warmup_ep = (args.latent_warmup_epochs if args.latent_warmup_epochs is not None
                            else max(_wfloor, 0.12 * args.epochs))          # ~12% forming the consensus / holding the latent
        print(f"{bcolors.OKCYAN}Curriculum (epochs): latent_warmup={latent_warmup_ep:.1f}  "
              f"(of {args.epochs} total){bcolors.ENDC}")

        # Learning rate: constant (default) or a warmup + cosine-decay schedule
        # (--lr_schedule). The schedule is a function of the optimizer step count,
        # so it resumes correctly from a checkpoint.
        total_train_steps = max(1, int(args.epochs * steps_per_epoch))
        if args.lr_schedule:
            lr_value = CosineAnnealingScheduler.getScheduler(
                peak_value=args.learning_rate, total_steps=total_train_steps,
                warmup_frac=0.1, init_value=1e-5, end_value=0.0)
        else:
            lr_value = args.learning_rate

        # Optimizers (HetSIREN)
        params = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('delta_volume_decoder')))
        params_inv = nnx.All(nnx.Param, nnx.PathContains('inverse_volume_decoder'))
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            tx = optax.chain(optax.clip_by_global_norm(args.grad_clip_norm), optax.adamw(lr_value))
        else:
            tx = optax.adamw(lr_value)
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

        # Exponential moving average (Polyak) of the trainable weights. Updated
        # every step. On a resume it is restored exactly from the checkpoint;
        # otherwise it is seeded from the current weights. The tree.map takes a
        # concrete snapshot so the buffer is decoupled from the live module's
        # Variables (nnx.state returns references to them).
        if args.ema_decay and args.ema_decay > 0.0:
            _ema_src, _ = nnx.merge(graphdef, state)
            ema_template = nnx.state(_ema_src, params)
            restored_ema = NeuralNetworkCheckpointer.load_ema(
                os.path.join(args.output_path, "HetSIREN_CHECKPOINT"), ema_template)
            if restored_ema is not None:
                ema_params = restored_ema
                print(f"{bcolors.WARNING}Restored EMA weight buffer from checkpoint.{bcolors.ENDC}")
            else:
                ema_params = jax.tree.map(lambda x: x, ema_template)
        else:
            ema_params = None

        @jax.jit
        def ema_update(graphdef, state, ema_params):
            model, _ = nnx.merge(graphdef, state)
            cur = nnx.state(model, params)
            return jax.tree.map(lambda e, c: args.ema_decay * e + (1.0 - args.ema_decay) * c,
                                ema_params, cur)

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

        def write_intermediate_volumes(volumes, out_dir, prefix):
            """Write the per-cluster intermediate volumes (runs on the logging thread)."""
            os.makedirs(out_dir, exist_ok=True)
            for idx, volume in enumerate(volumes, start=1):
                ImageHandler().write(volume, os.path.join(out_dir, f"{prefix}_{idx:02d}.mrc"), overwrite=True)

        # Logging cadence + background offload of the host-side logging work.
        logger = TrainingLogger(image_every=args.log_images_every,
                                landscape_every=args.log_landscape_every,
                                checkpoint_every=args.log_checkpoint_every,
                                steps_per_epoch=steps_per_epoch,
                                time_budget=args.log_time_budget,
                                background=not args.log_sync).start()

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

                    # Log intermediate results at the begining of the epoch.
                    if logger.should("images", i):
                        with logger.section():
                            # Get first 5 images from batch
                            if hetsiren.isTomoSIREN:
                                x_for_tb = x[0][:5]
                            else:
                                x_for_tb = x[:5]
                            labels_for_tb = labels[:5]

                            # Decode some images and some states
                            x_pred_intermediate, latents_intermediate = hetsiren_decode_image(graphdef, state, x_for_tb,
                                                                                              labels_for_tb, md_columns,
                                                                                              ctf_type=args.ctf_type,
                                                                                              return_latent=True,
                                                                                              corrupt_projection_with_ctf=True)
                            x_pred_intermediate = jax.vmap(min_max_scale)(x_pred_intermediate[..., None])
                            volumes_intermediate = hetsiren_decode_volume(graphdef, state, latents_intermediate)

                        logger.submit(writer.add_images, "Predicted images batch", x_pred_intermediate,
                                      dataformats="NHWC")
                        logger.submit(writer.add_volumes_slices, volumes_intermediate)

                    if logger.should("landscape", i):
                        with logger.section():
                            choice_key_use, choice_key = jax.random.split(choice_key, 2)
                            hetsiren_intermediate, _ = nnx.merge(graphdef, state)
                            latents_intermediate = sample_bank(hetsiren_intermediate.memory_bank.get(),
                                                               choice_key_use,
                                                               hetsiren_intermediate.subset_size,
                                                               n_valid=total_steps * args.batch_size)

                            # Predict some heterogeneous volumes (one per cluster centre)
                            n_clusters = int(min(20, latents_intermediate.shape[0]))
                            kmeans = KMeans(n_clusters=n_clusters).fit(np.asarray(latents_intermediate))
                            decoded_centers = [np.array(hetsiren_decode_volume(graphdef, state, center[None, ...]))
                                               for center in kmeans.cluster_centers_]

                            # Sprite images for the Tensorboard projector
                            latents_images = []
                            for start in range(0, latents_intermediate.shape[0], args.batch_size):
                                latents = latents_intermediate[start:start + args.batch_size]
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
                            latents_intermediate = np.asarray(latents_intermediate)

                        logger.submit(write_intermediate_volumes, decoded_centers,
                                      os.path.join(args.output_path, "Intermediate_volumes"), "hetsiren")
                        logger.submit(writer.add_embedding, latents_intermediate,
                                      label_img=latents_images[:, None, ...],
                                      tag="HetSIREN latent space", global_step=i)

                    # Save checkpoint model (+ EMA buffer when enabled, so a
                    # resume restores the average exactly).
                    if logger.should("checkpoint", i):
                        with logger.section():
                            NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path,
                                                                                                      "HetSIREN_CHECKPOINT"),
                                                                        epoch=i, ema_params=ema_params, wait=False)

                    i += 1

                # Ramp the rigid pose refinement in from the identity over the first epochs
                if args.pose_refine_warmup_epochs > 0:
                    warmup_steps = max(1, int(args.pose_refine_warmup_epochs * steps_per_epoch))
                    warmup_alpha = float(min(1.0, total_steps / warmup_steps))
                else:
                    warmup_alpha = 1.0

                # Ramp the latent into the decoder
                if latent_warmup_ep > 0:
                    latent_steps = max(1, int(latent_warmup_ep * steps_per_epoch))
                    hold_steps = latent_steps // 2
                    latent_alpha = float(min(1.0, max(0.0, total_steps - hold_steps)
                                             / max(1, latent_steps - hold_steps)))
                else:
                    latent_alpha = 1.0

                # Coarse-to-fine frequency annealing: ramp freq_alpha from
                # freq_anneal_start up to 1.0 over the first freq_anneal_epochs.
                # It scales the decoder's first-SIREN-layer w0 and the upper limit
                # of the FRC loss band. 0 epochs -> disabled (== 1.0).
                if args.freq_anneal_epochs and args.freq_anneal_epochs > 0:
                    freq_steps = max(1, int(args.freq_anneal_epochs * steps_per_epoch))
                    freq_alpha = float(args.freq_anneal_start
                                       + (1.0 - args.freq_anneal_start) * min(1.0, total_steps / freq_steps))
                else:
                    freq_alpha = 1.0

                # Band-limited representation: on the same clock, broaden the Gaussian 
                # render width (sigma_anneal_factor x base at freq_anneal_start, back 
                # to base at full resolution) so the density itself is intrinsically 
                # low-pass early and the FRC cannot chase high-resolution noise. 
                # None -> use the model's fixed base sigma.
                if args.sigma_anneal_factor > 1.0 and args.freq_anneal_epochs and args.freq_anneal_epochs > 0:
                    frac = (freq_alpha - args.freq_anneal_start) / max(1e-6, 1.0 - args.freq_anneal_start)
                    frac = min(1.0, max(0.0, frac))
                    sigma_mult = args.sigma_anneal_factor * (1.0 - frac) + frac
                    render_sigma = jnp.float32(hetsiren.sigma * sigma_mult)
                else:
                    render_sigma = None

                # Occupancy curriculum: ramp the per-particle amplitude change in from 0,
                # so early training explains the data by transport (motion) and occupancy
                # is only released later to catch the residual motion cannot
                if args.occupancy_reg > 0.0 and args.occupancy_warmup_epochs > 0:
                    occ_steps = max(1, int(args.occupancy_warmup_epochs * steps_per_epoch))
                    occ_alpha = float(min(1.0, total_steps / occ_steps))
                else:
                    occ_alpha = 1.0

                # Reconstruction split: weight of the amplitude (occupancy) rendering
                # against the mass-transport rendering
                if args.schedule_recon_split:
                    if args.recon_split_epochs and args.recon_split_epochs > 0:
                        split_steps = max(1, int(args.recon_split_epochs * steps_per_epoch))
                    elif args.freq_anneal_epochs and args.freq_anneal_epochs > 0:
                        split_steps = max(1, int(args.freq_anneal_epochs * steps_per_epoch))
                    else:
                        split_steps = 1
                    split_frac = min(1.0, total_steps / split_steps)
                    amp_recon_weight = jnp.float32(args.recon_split_start
                                                   + (args.recon_split_final - args.recon_split_start) * split_frac)
                else:
                    amp_recon_weight = jnp.float32(args.recon_split_final)

                loss, recon_loss, pose_metrics, state, rng = train_step_hetsiren(graphdef, state, x, labels, md_columns, rng,
                                                                   l1_lambda=args.denoising_strength,
                                                                   graph_lambda=graph_lambda,
                                                                   warmup_alpha=warmup_alpha,
                                                                   latent_alpha=latent_alpha,
                                                                   pose_refine_reg=args.pose_refine_reg,
                                                                   pose_refine_max_angle=args.pose_refine_max_angle,
                                                                   shift_refine_reg=args.shift_refine_reg,
                                                                   shift_refine_max=args.shift_refine_max,
                                                                   decoupling_lambda=args.decoupling_lambda,
                                                                   distance_preservation_lambda=args.distance_preservation_lambda,
                                                                   kl_lambda=args.kl_lambda,
                                                                   freq_alpha=freq_alpha,
                                                                   arap_lambda=args.arap_lambda,
                                                                   geometric_lambda=args.geometric_lambda,
                                                                   render_sigma=render_sigma,
                                                                   occ_alpha=occ_alpha,
                                                                   occupancy_reg=args.occupancy_reg,
                                                                   occupancy_tv=args.occupancy_tv,
                                                                   occupancy_kappa=args.occupancy_kappa,
                                                                   amp_recon_weight=amp_recon_weight,
                                                                   arap_rotation=args.arap_rotation)
                total_loss += loss
                total_recon_loss += recon_loss

                # Polyak/EMA update of the trainable weights (only when enabled).
                if ema_params is not None:
                    ema_params = ema_update(graphdef, state, ema_params)

                # Summary writer (training loss) + progress bar.
                if logger.should_log_scalars(step):
                    mean_loss = float(total_loss) / step
                    mean_recon_loss = float(total_recon_loss) / step

                    writer.add_scalar('Training loss (HetSIREN)',
                                      mean_loss,
                                      i * steps_per_epoch + step)

                    writer.add_scalars('Reconstruction loss (HetSIREN)',
                                       {"train": mean_recon_loss},
                                        i * steps_per_epoch + step)

                    # Pose refinement vs. pose absorption
                    pose_angle_deg = float(pose_metrics["pose_angle_deg"])
                    shift_norm_px = float(pose_metrics["shift_norm_px"])
                    rigid_fraction = float(pose_metrics["rigid_fraction"])
                    deformation_A = float(pose_metrics["deformation_A"])
                    occ_fraction = float(pose_metrics["occ_fraction"])

                    writer.add_scalars('Pose refinement (HetSIREN)',
                                       {"rotation (deg)": pose_angle_deg,
                                        "shift (px)": shift_norm_px},
                                       i * steps_per_epoch + step)
                    writer.add_scalar('Decoder rigid fraction (HetSIREN)',
                                      rigid_fraction,
                                      i * steps_per_epoch + step)
                    writer.add_scalar('Decoder deformation (A) (HetSIREN)',
                                      deformation_A,
                                      i * steps_per_epoch + step)
                    writer.add_scalar('Occupancy fraction (HetSIREN)',
                                      occ_fraction,
                                      i * steps_per_epoch + step)

                    # Progress bar update  (TQDM)
                    pose_str = (f"pose={pose_angle_deg:.2f}deg/{shift_norm_px:.2f}px | "
                                f"rigid={rigid_fraction:.3f} | def={deformation_A:.2f}A")
                    occ_str = f" | occ={occ_fraction:.3f}" if args.occupancy_reg > 0 else ""
                    if args.transport_mass:
                        pbar.set_postfix_str(f"loss={mean_loss:.5f} | recon_loss={mean_recon_loss:.5f} | {pose_str}{occ_str} | graph_lambda={graph_lambda:.5f}")
                    else:
                        pbar.set_postfix_str(f"loss={mean_loss:.5f} | recon_loss={mean_recon_loss:.5f} | {pose_str}{occ_str}")

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

                step += 1

            hetsiren, optimizer = nnx.merge(graphdef, state)

            # Export the EMA-averaged weights as the final forward model (also
            # used as the frozen target for the subsequent inverse training).
            if ema_params is not None:
                nnx.update(hetsiren, ema_params)

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
                    if logger.should("checkpoint", i):
                        with logger.section():
                            NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path,
                                                                                                      "HetSIREN_CHECKPOINT"),
                                                                        epoch=i, wait=False)

                    i += 1

                loss, state, rng = train_step_inverse_hetsiren(graphdef, state, x, labels, md_columns, rng)
                total_loss += loss

                # Summary writer (training loss)
                if logger.should_log_scalars(step):
                    mean_loss = float(total_loss) / step
                    writer.add_scalar('Inverse training loss (HetSIREN)',
                                      mean_loss,
                                      i * steps_per_epoch + step)

                    # Progress bar update  (TQDM)
                    pbar.set_postfix_str(f"loss={mean_loss:.5f}")

                step += 1

        hetsiren, optimizer_inv = nnx.merge(graphdef, state)

        # Example of predicted data for Tensorboard
        x_pred_example = hetsiren_decode_image(graphdef, state, x_example, labels_example, md_columns, ctf_type=args.ctf_type, return_latent=False, corrupt_projection_with_ctf=True)
        x_pred_example = jax.vmap(min_max_scale)(x_pred_example[..., None])
        writer.add_images("Predicted images batch", x_pred_example, dataformats="NHWC")

        # Let the background logging thread and the asynchronous checkpoint write finish
        # before the process moves on.
        logger.close()
        NeuralNetworkCheckpointer.wait_for_pending()

        # Save model
        NeuralNetworkCheckpointer.save(hetsiren, os.path.join(args.output_path, "HetSIREN"))

        # Remove checkpoint (only written every 5 epochs, so it may not exist)
        checkpoint_dir = os.path.join(args.output_path, "HetSIREN_CHECKPOINT")
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

    elif args.mode == "predict":

        # Automatic batch sizing targets training memory; for inference just use a
        # fixed default if the user passed ``auto``.
        if args.batch_size == "auto":
            args.batch_size = 8

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
        md_pred.write(os.path.join(args.output_path, "predicted_latents" + os.path.splitext(args.md)[1]),
                      updateImagePaths=True)

    # If exists, clean MMAP
    # if not args.load_images_to_ram and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap_grain")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap_grain"))
