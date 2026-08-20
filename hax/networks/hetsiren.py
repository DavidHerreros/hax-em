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


def match_per_image_contrast(pred, target, mask=None, mode="relative"):
    if mode == "off":
        return pred

    p = jax.lax.stop_gradient(pred)
    t = jax.lax.stop_gradient(target)
    axes = (-2, -1)

    if mask is None:
        sum_pp = jnp.sum(p * p, axis=axes, keepdims=True)
        sum_pt = jnp.sum(p * t, axis=axes, keepdims=True)
    else:
        w = jnp.broadcast_to(mask, p.shape)
        sum_pp = jnp.sum(w * p * p, axis=axes, keepdims=True)
        sum_pt = jnp.sum(w * p * t, axis=axes, keepdims=True)

    # A flat or empty projection carries no scale information; leave those images alone.
    ok = sum_pp > 1e-12 * jnp.mean(sum_pp)
    a = jnp.where(ok, sum_pt / jnp.where(ok, sum_pp, 1.0), 1.0)

    if mode == "relative":
        a = a / jnp.maximum(jnp.mean(a), 1e-6)

    return a * pred


def decimate_lattice(mask, stride=1):
    mask = np.asarray(mask)
    stride = int(stride)
    if stride <= 1:
        return np.asarray(np.where(mask > 0.0)).T

    keep = np.zeros(mask.shape, dtype=bool)
    keep[::stride, ::stride, ::stride] = True
    return np.asarray(np.where((mask > 0.0) & keep)).T


# Constants for the VAE posterior log-sigma head.
LOGSTD_INIT = -3.0
LOGSTD_MIN = -8.0
LOGSTD_MAX = 1.0 
LATENT_SCALE = 0.05
LATENT_SPREAD_EPS = 1e-12


def normalize_latent(mean):
    centred = mean - jnp.mean(mean, axis=0, keepdims=True)
    spread = jnp.sqrt(jnp.mean(jnp.square(centred.astype(jnp.float32))) + LATENT_SPREAD_EPS)
    return mean * (LATENT_SCALE / spread)


def logstd_head(in_features, out_features, *, rngs):
    """A posterior log-sigma head that starts at a fixed, small, input-independent scale."""
    return Linear(in_features, out_features, rngs=rngs,
                  kernel_init=nnx.initializers.zeros_init(),
                  bias_init=nnx.initializers.constant(LOGSTD_INIT))


class Encoder(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, architecture="convnn", isVae=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.out_conv_dim = -(-self.input_dim // (2 ** 4))
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
            hidden_layers_conv = [Conv(1, 4, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16)]
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
            raise ValueError("Architecture not supported. Implemented architectures are: mlpnn / convnn / convstem")

        if isVae:
            # self.layer_normalization = nnx.LayerNorm(256, rngs=rngs)
            self.mean_x = Linear(256, lat_dim, rngs=rngs)
            self.logstd_x = logstd_head(256, lat_dim, rngs=rngs)
        else:
            self.latent = Linear(256, lat_dim, rngs=rngs)

    def sample_gaussian(self, mean, logstd, *, rngs):
        return LATENT_SCALE * jnp.exp(logstd) * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, return_last=False, *, rngs=None):
        if self.architecture == "mlpnn":
            x = rearrange(x, 'b h w c -> b (h w c)')

            for layer in self.hidden_layers:
                x = nnx.leaky_relu(layer(x))  # or nnx.relu

        elif self.architecture == "convnn":
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

        elif self.architecture == "cryouni":
            x = self.cryouni(x)
            x = self.cryouni_head(x["clstokens"], x["patchtokens"])
            x = nnx.leaky_relu(self.hidden_layers(x))

        if return_last:
            return x
        else:
            if self.isVae:
                # x = self.layer_normalization(x)
                mean = normalize_latent(self.mean_x(x))
                logstd = jnp.clip(self.logstd_x(x), LOGSTD_MIN, LOGSTD_MAX)
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
            self.logstd_x = logstd_head(256, lat_dim, rngs=rngs)
        else:
            self.latent = Linear(256, lat_dim, rngs=rngs)

    def sample_gaussian(self, mean, logstd, *, rngs):
        return LATENT_SCALE * jnp.exp(logstd) * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, return_last=False, *, rngs=None):
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))

        if return_last:
            return x
        else:
            if self.isVae:
                mean = normalize_latent(self.mean_x(x))
                logstd = jnp.clip(self.logstd_x(x), LOGSTD_MIN, LOGSTD_MAX)
                sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean
                return sample, mean, logstd
            else:
                latent = self.latent(x)
                return latent

class MultiEncoder(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, isVae=False, architecture="convnn", isTomoSIREN=False, *, rngs: nnx.Rngs):
        if lat_dim % 2 != 0:
            raise ValueError(f"lat_dim must be even -- the latent is read out in two halves "
                             f"(motion / occupancy) and split back apart in the decoder; got {lat_dim}.")

        if isTomoSIREN:
            self.encoders = nnx.Dict({"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, architecture=architecture, rngs=rngs),
                                      "encoder_dec": EncoderTomo(100, lat_dim, n_layers=n_layers, rngs=rngs)})
        else:
            self.encoders = nnx.Dict({"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, architecture=architecture, rngs=rngs),
                                      "encoder_dec": Encoder(input_dim, lat_dim, n_layers=n_layers, architecture=architecture, rngs=rngs)})
        self.isVae = isVae
        if isVae:
            self.mean_field = Linear(256, lat_dim // 2, rngs=rngs)
            self.mean_val = Linear(256, lat_dim // 2, rngs=rngs)
            self.logstd_field = logstd_head(256, lat_dim // 2, rngs=rngs)
            self.logstd_val = logstd_head(256, lat_dim // 2, rngs=rngs)
        else:
            self.latent_field = Linear(256, lat_dim // 2, rngs=rngs)
            self.latent_val = Linear(256, lat_dim // 2, rngs=rngs)

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
        return LATENT_SCALE * jnp.exp(logstd) * jnr.normal(rngs, shape=mean.shape) + mean

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
            mean_field = self.mean_field(x)
            mean_val = self.mean_val(x)

            # Normalized on the concatenated vector
            mean = normalize_latent(jnp.concatenate([mean_field, mean_val], axis=-1))
            logstd_field = jnp.clip(self.logstd_field(x), LOGSTD_MIN, LOGSTD_MAX)
            logstd_val = jnp.clip(self.logstd_val(x), LOGSTD_MIN, LOGSTD_MAX)
            logstd = jnp.concatenate([logstd_field, logstd_val], axis=-1)

            # Sample on the whole vector
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
            latent_field = self.latent_field(x)
            latent_val = self.latent_val(x)
            latent = jnp.concatenate([latent_field, latent_val], axis=-1)
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
                self.geom = nnx.data(build_geometry(self.coords[0], gauss_scale="auto", hierarchical_sizes=(32, 128, 1024), compute_local_frames=True))

                # Motion head based on PT
                self.point_transformer_net = PointTransformerDecoder(out_channels=3, feat_dim=32, nk=128, input_bottleneck=64,
                                                                     hierarchical_sizes=(32, 128, 1024), latent_dim=lat_dim // 2,
                                                                     final_init_std=0.0, predict_at_coarse_level=False, rngs=rngs)

                # Occupancy head
                hidden_values = [Siren2Linear(in_features=lat_dim // 2 + 3, out_features=8, rngs=rngs, dtype=jnp.float32, is_first=True, w0=30.0, s=0.0, use_bias=False)]
                hidden_values.append(Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=0.0, use_bias=False, custom_init=True, is_residual=True))
                for _ in range(3):
                    hidden_values.append(Siren2Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.float32, is_first=False, w0=1.0, s=0.0, use_bias=False, custom_init=True, is_residual=True))
                hidden_values.append(nnx.Linear(in_features=8, out_features=1, rngs=rngs, use_bias=False, kernel_init=nnx.initializers.zeros_init()))

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
        """Graph quantities for the elastic priors relative to the consensus"""
        return (self.edge_index, self.edge_weights, self.consensus_distances,
                self.coords[0], self.node_mass)

    def apply_rigid_gauge(self, field, rest_override=None, weight_override=None):
        """Pin a per-particle decoder field to the canonical (rigid-gauge) frame"""
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

    # Blockwise fixed-grid decode to avoid materializing (B, N) when transport_mass is off
    def fixed_head(self, x, freq_alpha=1.0):
        """The part of the fixed-grid decode that never touches N: ``(B, lat) -> (B, 8)``."""
        h = self.hidden_values[0](x, freq_alpha)
        for layer in self.hidden_values[1:-1]:
            h = layer(h)
        return h

    def fixed_raw_block(self, h, start, block):
        """Raw (ungauged) density change on points [start, start + block): (B, block)"""
        last = self.hidden_values[-1]
        kernel = jax.lax.dynamic_slice_in_dim(last.kernel.value, start, block, axis=1)
        out = jnp.dot(h.astype(kernel.dtype), kernel)
        if last.bias is not None:
            out = out + jax.lax.dynamic_slice_in_dim(last.bias.value, start, block, axis=0)
        return out

    def fixed_block_extras(self, start, block):
        """The per-point constants a fixed-grid block needs: reference, coords, gauge rows"""
        ref = jax.lax.dynamic_slice_in_dim(self.reference_values, start, block, axis=1)
        coords = jax.lax.dynamic_slice_in_dim(self.coords, start, block, axis=1)
        basis = weights = None
        if self.gauge.available:
            basis = jax.lax.dynamic_slice_in_dim(jnp.asarray(self.gauge.basis), start, block, axis=0)
            weights = jax.lax.dynamic_slice_in_dim(jnp.asarray(self.gauge.weights), start, block, axis=0)
        return ref, coords, basis, weights

    def __call__(self, x, c=None, freq_alpha=1.0, occ_alpha=1.0, return_diagnostics=False):
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

            # Rigid motion correction for the field
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

            # Rigid motion correction for the densities
            x_map, rigid_fraction = self.apply_rigid_gauge(x_map[..., None])
            x_map = x_map[..., 0]

            # Recover volume values
            values = self.reference_values + x_map

            # Fixed-grid mode has no transport, so no occupancy/motion split to make.
            occ_change = jnp.zeros_like(x_map)

            # Recover coords (non-normalized).
            coords = jnp.broadcast_to(self.scale * self.coords,
                                      (x.shape[0],) + self.coords.shape[1:])

        if return_diagnostics:
            return coords, values, rigid_fraction, occ_change

        return coords, values

    def raw_delta_coords(self, x, c, freq_alpha=1.0):
        """The coordinate head's un-gauged output at query points c"""
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
        """Evalute the field at arbitrary coordinantes (only valid for implicit networks with mass transport)"""
        if not (self.transport_mass and self.is_implicit and not self.point_transformer):
            raise UserWarning("decode_field_at requires an implicit mass-transport decoder "
                              "(--transport_mass, --implicit_network, no point transformer).")

        c0, w0 = self.coords, self.reference_values

        # Fit the rigid gauge once, on the Gaussians, exactly as __call__ does.
        delta_ref = self.raw_delta_coords(x, c0[0], freq_alpha)
        if self.gauge.available and self.rigid_gauge:
            w = jnp.maximum(jax.lax.stop_gradient(w0[0]), 0.0)
            w = jnp.where(jnp.sum(w) > 1e-8, w, jnp.ones_like(w))
            transform = gauge_displacement_transform(delta_ref.astype(jnp.float32),
                                                     jax.lax.stop_gradient(c0[0]), w,
                                                     irls_iters=self.rigid_gauge_irls)
        else:
            transform = None

        def one_chunk(cq):
            d = self.raw_delta_coords(x, cq, freq_alpha)
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

            c0, w0 = self.coords, self.reference_values
            x_coords, _ = self.apply_rigid_gauge(x_coords, rest_override=c0[0],
                                                 weight_override=w0[0])

            # Recover coords (non-normalized)
            coords = self.scale * (c0 + x_coords)
        else:
            # Recover coords (non-normalized). See __call__: a view, not a copy.
            coords = jnp.broadcast_to(self.scale * self.coords,
                                      (x.shape[0],) + self.coords.shape[1:])

        return coords

    def decode_consensus_volume(self, filter=True, sigma=1.0):
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
            kernel_size = 9
            if isinstance(sigma, (int, float, np.floating, np.integer)):
                kernel_size = max(9, 2 * int(np.ceil(3.0 * float(sigma))) + 1)
            grids = jax.vmap(partial(low_pass_3d, kernel_size=kernel_size), in_axes=(0, None))(grids, sigma)

        return grids

class PhysDecoder:
    def __init__(self, xsize, sr, transport_mass, render_chunk_size=0, lattice_stride=1, base_sigma=1.0,
                 bilinear_scatter=False):
        self.xsize = xsize
        self.transport_mass = transport_mass
        self.bilinear_scatter = bool(transport_mass or bilinear_scatter)
        self.splat_kernel_size = max(9, 2 * int(np.ceil(3.0 * max(float(base_sigma), 1.0))) + 1)
        self.render_chunk_size = int(render_chunk_size or 0)
        self.lattice_stride = max(1, int(lattice_stride))
        self.pad_factor = 1 if xsize > 350 else 2

        physical_radius_px = 60.0 / sr
        max_safe_radius_px = xsize // 8
        dilation_radius = jnp.minimum(physical_radius_px, max_safe_radius_px)
        self.dilation_radius = int(jnp.maximum(dilation_radius, 5))

    def scatter(self, values, coords, xsize, rotations, shifts, centering, batch, dtype):
        coords = jnp.matmul(coords, rearrange(rotations, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts[:, None, :] + centering[..., :-1]

        # Scatter image
        c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
        images = jnp.zeros((batch, xsize, xsize), dtype=dtype)

        if self.bilinear_scatter:
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
            bamp = values * jnp.exp(-num / (2. * 1. ** 2.))

        # Guard the scatter against NaN/Inf or out-of-range coordinates
        bposi = jnp.clip(bposi, 0, xsize - 1)
        bamp = jnp.nan_to_num(bamp)

        # Consider batch dimensions
        bamp = jnp.broadcast_to(bamp, (batch,) + bamp.shape[1:])
        bposi = jnp.broadcast_to(bposi, (batch,) + bposi.shape[1:])

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        return jax.vmap(scatter_img)(images, bposi, bamp)

    def scatter_blocked(self, values, coords, xsize, rotations, shifts, centering, batch, dtype):
        n_points = coords.shape[1]
        block = self.render_chunk_size
        n_blocks, tail = divmod(n_points, block)

        def one_block(values_b, coords_b):
            return self.scatter(values_b, coords_b, xsize, rotations, shifts, centering, batch, dtype)

        images = jnp.zeros((batch, xsize, xsize), dtype=dtype)

        if n_blocks:
            def indexed_block(i):
                start = i * block
                return one_block(jax.lax.dynamic_slice_in_dim(values, start, block, axis=1),
                                 jax.lax.dynamic_slice_in_dim(coords, start, block, axis=1))
            indexed_block = jax.checkpoint(indexed_block)

            def body(acc, i):
                return acc + indexed_block(i), None

            images, _ = jax.lax.scan(body, images, jnp.arange(n_blocks))

        if tail:
            start = n_blocks * block
            images = images + jax.checkpoint(one_block)(values[:, start:], coords[:, start:])
        return images

    def __call__(self, x, values, coords, xsize, rotations, shifts, centering, ctf, ctf_type, sigma, bg_weight=1.0, filter=True):
        # Get rotation matrices
        if rotations.ndim == 2:
            rotations = euler_matrix_batch(rotations[:, 0], rotations[:, 1], rotations[:, 2])

        batch = x.shape[0]
        blocked = 0 < self.render_chunk_size < coords.shape[1]
        scatter = self.scatter_blocked if blocked else self.scatter
        images = scatter(values, coords, xsize, rotations, shifts, centering, batch, x.dtype)

        return self.post_scatter_operations(images, x, ctf, ctf_type, sigma, bg_weight=bg_weight, filter=filter)

    def post_scatter_operations(self, images, x, ctf, ctf_type, sigma, bg_weight=1.0, filter=True):
        kernel_size = self.splat_kernel_size
        if not self.transport_mass:
            sigma = float(max(1.0, 0.5 * self.lattice_stride))
            kernel_size = max(kernel_size, 2 * int(np.ceil(3.0 * sigma)) + 1)

        # Gaussian filter (needed by forward interpolation)
        if filter:
            images = dm_pix.gaussian_blur(images[..., None], sigma, kernel_size=kernel_size)[..., 0]

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
                 rigid_gauge=True, rigid_gauge_irls=1, render_chunk_size=0, lattice_stride=1,
                 bilinear_scatter=False,
                 *, rngs: nnx.Rngs, **kwargs):
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
            # The inverse decoder recovers the latent that produced a point cloud, so it can only recover the half that drives the coordinates
            inv_lat_dim = lat_dim // 2
            self.inverse_volume_decoder = PointCloudEncoder(latent_dim=inv_lat_dim, hidden=64, n_blocks=4, dtype=jnp.float32, scale_multiplier=1.0, rngs=rngs)

        self.phys_decoder = PhysDecoder(self.xsize, sr, transport_mass=transport_mass,
                                        render_chunk_size=render_chunk_size, lattice_stride=lattice_stride,
                                        base_sigma=sigma, bilinear_scatter=bilinear_scatter)

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.subset_size = min(2048, bank_size)
        self.memory_bank = MemoryBank(array_init=jnp.zeros((self.bank_size, lat_dim)))

        # Gaussians size
        # self.sigma = nnx.Param(sigma)
        self.sigma = sigma

        # Loss function. Image-space L2; the freq_alpha argument is accepted and ignored so the
        # call sites stay uniform with the coarse-to-fine w0 annealing.
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

    def fixed_blocks(self, contribution, init):
        block = self.phys_decoder.render_chunk_size
        n_points = self.delta_volume_decoder.coords.shape[1]
        n_blocks, tail = divmod(n_points, block)

        acc = init
        if n_blocks:
            body = jax.checkpoint(lambda i: contribution(i * block, block))
            acc, _ = jax.lax.scan(
                lambda carry, i: (jax.tree.map(jnp.add, carry, body(i)), None),
                acc, jnp.arange(n_blocks))
        if tail:
            rest = jax.checkpoint(lambda s: contribution(s, tail), static_argnums=(0,))
            acc = jax.tree.map(jnp.add, acc, rest(n_blocks * block))
        return acc

    def fixed_gauge_coeffs(self, h, batch, n_points):
        dec = self.delta_volume_decoder
        if not dec.gauge.available:
            return None
        irls = int(dec.rigid_gauge_irls)

        def raw(start, size):
            return dec.fixed_raw_block(h, start, size)[..., None]

        scale = None
        if irls > 0:
            mag_sum = self.fixed_blocks(
                lambda s, c: jnp.sum(density_gauge_magnitude(raw(s, c)), axis=-1, keepdims=True),
                jnp.zeros((batch, 1), jnp.float32))
            scale = density_gauge_scale(mag_sum, n_points)

        def normals(start, size):
            _, _, basis, w0 = dec.fixed_block_extras(start, size)
            field = raw(start, size)
            w = (density_gauge_block_weights(w0[None, :], field, scale) if irls > 0
                 else jnp.broadcast_to(w0[None, :], field.shape[:2]))
            return density_gauge_block_normals(field, basis, w)

        k = jnp.asarray(dec.gauge.basis).shape[-1]
        gram, rhs = self.fixed_blocks(normals, (jnp.zeros((batch, k, k), jnp.float32),
                                                 jnp.zeros((batch, k), jnp.float32)))
        return density_gauge_solve(gram, rhs)

    def fixed_gauge_coeffs(self, h, coeffs, basis, start, size):
        """One block's gauged density change, plus its rigid/field sums of squares."""
        dec = self.delta_volume_decoder
        field = dec.fixed_raw_block(h, start, size)[..., None]
        field_sq = jnp.sum(jnp.square(field)).astype(jnp.float32)
        rigid_sq = jnp.zeros((), jnp.float32)
        if coeffs is not None:
            rigid = density_gauge_apply(coeffs, basis)
            rigid_sq = jnp.sum(jnp.square(rigid)).astype(jnp.float32)
            if dec.rigid_gauge:
                field = (field - rigid).astype(field.dtype)
        return field, rigid_sq, field_sq

    def render_fixed_chunked(self, x, latent, rotations, shifts, centering, ctf, ctf_type,
                             sigma, freq_alpha=1.0, bg_weight=1.0, filter=True,
                             extra_render=None, dp_latent=None):
        dec, pd = self.delta_volume_decoder, self.phys_decoder
        batch, dtype = x.shape[0], x.dtype
        n_points = dec.coords.shape[1]

        def as_matrix(r):
            return euler_matrix_batch(r[:, 0], r[:, 1], r[:, 2]) if r.ndim == 2 else r

        rotations = as_matrix(rotations)
        if extra_render is not None:
            extra_rotations = as_matrix(extra_render["rotations"])

        h = dec.fixed_head(latent, freq_alpha)
        coeffs = self.fixed_gauge_coeffs(h, batch, n_points)

        h_dp = coeffs_dp = None
        if dp_latent is not None:
            h_dp = dec.fixed_head(dp_latent, freq_alpha)
            coeffs_dp = self.fixed_gauge_coeffs(h_dp, batch, n_points)

        # Decode, gauge, scatter and reduce one block at a time
        def render(start, size):
            ref, coords, basis, _ = dec.fixed_block_extras(start, size)
            field, rigid_sq, field_sq = self.fixed_gauge_coeffs(h, coeffs, basis, start, size)

            values = ref + field[..., 0]
            points = jnp.broadcast_to(dec.scale * coords, (batch,) + coords.shape[1:])
            images = pd.scatter(values, points, self.xsize, rotations, shifts, centering,
                                 batch, dtype)

            neg = jnp.where(values < 0.0, -values, 0.0)
            out = [images,
                   jnp.sum(jnp.abs(values)).astype(jnp.float32),
                   jnp.sum(neg).astype(jnp.float32),
                   jnp.count_nonzero(values < 0.0).astype(jnp.float32),
                   rigid_sq, field_sq]

            if extra_render is not None:
                out.append(pd.scatter(values, points, self.xsize, extra_rotations,
                                       extra_render["shifts"], centering, batch, dtype))
            if h_dp is not None:
                # The reference cancels in the difference, so only the fields are needed.
                field_dp, _, _ = self.fixed_gauge_coeffs(h_dp, coeffs_dp, basis, start, size)
                delta = jnp.abs(field[..., 0] - field_dp[..., 0])           # (B, C)
                out.append(jnp.sum(delta * jnp.sum(jnp.abs(coords[0]), axis=-1)[None, :])
                           .astype(jnp.float32))
            return tuple(out)

        init = [jnp.zeros((batch, self.xsize, self.xsize), dtype)] + [jnp.zeros((), jnp.float32)] * 5
        if extra_render is not None:
            init.append(jnp.zeros((batch, self.xsize, self.xsize), dtype))
        if h_dp is not None:
            init.append(jnp.zeros((), jnp.float32))

        acc = self.fixed_blocks(render, tuple(init))
        images, sum_abs, sum_neg, count_neg, rigid_sq, field_sq = acc[:6]
        rest = list(acc[6:])

        rigid_fraction = (rigid_fraction_from_norms(rigid_sq, field_sq)
                          if coeffs is not None else jnp.zeros((), jnp.float32))
        images, images_mask = pd.post_scatter_operations(images, x, ctf, ctf_type, sigma,
                                        bg_weight=bg_weight, filter=filter)

        # values is (B, N): the reference is shared but the decoded field is per-particle
        stats = {"sum_abs": sum_abs, "sum_neg": sum_neg, "count_neg": count_neg,
                 "n_values": float(batch * n_points)}
        if extra_render is not None:
            raw_extra = rest.pop(0)
            stats["images_extra"] = pd.post_scatter_operations(
                raw_extra, x, extra_render["ctf"], extra_render.get("ctf_type"),
                extra_render.get("sigma", sigma),
                bg_weight=extra_render.get("bg_weight", bg_weight),
                filter=extra_render.get("filter", filter))[0]
        if h_dp is not None:
            # mean over (B, N, 3) of |(v - v_dp) * c|
            stats["loss_dp"] = rest.pop(0) / (batch * n_points * 3)
        return images, images_mask, rigid_fraction, stats

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
        """The deformation field sampled at arbitrary voxel coordinates"""
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
                                   "decoupling_lambda", "distance_preservation_lambda", "kl_lambda", "kl_free_bits",
                                   "geometric_lambda", "pose_refine_max_angle",
                                   "shift_refine_reg", "shift_refine_max",
                                   "occupancy_l1", "occupancy_tv", "occupancy_kappa",
                                   "per_image_contrast"))
def train_step_hetsiren(graphdef, state, x, labels, md, key, do_update=True, l1_lambda=1e-4, graph_lambda=1e-4,
                        pose_refine_reg=0.1, decoupling_lambda=1e-4, distance_preservation_lambda=1e-4,
                        kl_lambda=1e-3, kl_free_bits=0.0, freq_alpha=1.0, geometric_lambda=0.0,
                        latent_alpha=1.0,
                        pose_refine_max_angle=8.0, shift_refine_reg=0.1, shift_refine_max=4.0,
                        occ_alpha=1.0, occupancy_l1=0.0, occupancy_tv=0.0, occupancy_kappa=0.05,
                        amp_recon_weight=0.1, per_image_contrast="off",
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
        z_dec = latent_alpha * (sample if model.isVae else latent)

        # Without mass transport the decode fuses with the render
        fused = (model.phys_decoder.render_chunk_size > 0
                 and not model.delta_volume_decoder.transport_mass
                 and not model.local_reconstruction
                 and not model.isTomoSIREN
                 and M == 1)

        # Whether the distance-preservation term is considered
        dp_active = distance_preservation_lambda > 0.0

        coords = values = occ_change = None
        if not fused:
            coords, values, rigid_fraction, occ_change = model.delta_volume_decoder(z_dec, freq_alpha=freq_alpha,
                                                                                    occ_alpha=occ_alpha,
                                                                                    return_diagnostics=True)

        # Get rotation matrices
        if euler_angles.ndim == 2:
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        else:
            rotations = euler_angles

        # Rotation posterior scheduling
        rotations_logscale = jnp.clip(rotations_logscale, jnp.log(0.03), 2.0)

        # Sample new rotations
        if M > 1:
            # Consider refinement and rigid registration alignments
            # rotations_refined = jnp.matmul(rotations_rigid, rotations)
            rotations_refined = jnp.matmul(rotations, rotations_rigid)

            rotations_refined, omegas, log_q = sample_topM_R(rot_sample_key, rotations_refined, rotations_logscale, M=M)
        else:
            # Consider refinement and rigid registration alignments
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

        # Render width
        render_sigma_eff = model.sigma

        # Generate the projections
        if fused:
            extra_render = None
            if model.decoupling:
                rotations_random_refined = jnp.matmul(
                    euler_matrix_batch(rotations_random[:, 0], rotations_random[:, 1], rotations_random[:, 2]),
                    jax.lax.stop_gradient(rotations_rigid))
                extra_render = {"rotations": rotations_random_refined,
                                "shifts": jnp.zeros_like(shifts_refined),
                                "ctf": ctf_random, "ctf_type": None, "sigma": model.sigma,
                                "bg_weight": 0.0}
            images_corrected, _, rigid_fraction, fused_stats = model.render_fixed_chunked(
                x, z_dec, rotations_refined, shifts_refined, centering, ctf, model.ctf_type,
                render_sigma_eff, freq_alpha=freq_alpha, bg_weight=0.0,
                extra_render=extra_render,
                dp_latent=(latent_alpha * latent) if (model.isVae and dp_active) else None)
            images_corrected_field = images_corrected
        elif model.has_reference_volume and model.delta_volume_decoder.transport_mass:
            reference_values = model.delta_volume_decoder.reference_values
            images_corrected, _ = phys_decoder(x, values, jax.lax.stop_gradient(coords), model.xsize,
                                               rotations_refined, shifts_refined,
                                               centering, ctf, model.ctf_type, render_sigma_eff, 0.0)
            images_corrected_field, _ = phys_decoder(x, reference_values, coords, model.xsize,
                                                     rotations_refined, shifts_refined,
                                                     centering, ctf, model.ctf_type, render_sigma_eff, 0.0)
        else:
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

        # Per-particle contrast correction
        if per_image_contrast != "off":
            images_corrected_loss = match_per_image_contrast(images_corrected_loss, x_loss,
                                                             projected_mask, per_image_contrast)
            images_corrected_field_loss = match_per_image_contrast(images_corrected_field_loss, x_loss,
                                                                   projected_mask, per_image_contrast)

        # Split the reconstruction between the amplitude and field components
        recon_loss = (amp_recon_weight * model.representation_loss_fn(images_corrected_loss, x_loss, freq_alpha)
                      + (1.0 - amp_recon_weight)
                      * model.representation_loss_fn(images_corrected_field_loss, x_loss, freq_alpha))
        recons_loss_all = recon_loss

        # L1 based denoising, plus L1 denoising for negative values
        if fused:
            l1_loss = fused_stats["sum_abs"] / fused_stats["n_values"]
            l1_loss += fused_stats["sum_neg"] / jnp.maximum(fused_stats["count_neg"], 1.0)
        else:
            l1_loss = jnp.mean(jnp.abs(values))

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

        # Local distance preservation
        if fused:
            loss_dp = fused_stats.get("loss_dp", 0.0)
        elif model.isVae and dp_active:
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
            latent_n = latent / LATENT_SCALE
            kl_per_dim = -0.5 * (1. + 2. * logstd - jnp.square(jnp.exp(logstd)) - jnp.square(latent_n))
            kl_loss = jnp.mean(jnp.maximum(jnp.mean(kl_per_dim, axis=0), kl_free_bits))
        else:
            kl_loss = 0.0

        # Variational loss (poses)
        if M > 1:
            # Importance weighting over the M sampled poses
            w_pose, _ = importance_weights(recons_loss_all, log_q)
            w_pose = jax.lax.stop_gradient(w_pose)
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

            # Total loss
            loss_graph = (loss_def_regularity + 0.01 * loss_repulsion).mean()
        else:
            loss_graph = 0.0

        # Occupancy prior
        occ_active = (occupancy_l1 > 0.0 or occupancy_tv > 0.0)
        if occ_active and model.delta_volume_decoder.transport_mass and model.has_reference_volume:
            edge_idx, edge_mass, _, _, node_mass = model.delta_volume_decoder.graph_terms()
            node_mass = jax.lax.stop_gradient(node_mass)
            edge_mass = jax.lax.stop_gradient(edge_mass)
            occ_l1 = jnp.mean(jnp.abs(occ_change) / (node_mass[None, :] + occupancy_kappa))

            i_e, j_e = edge_idx
            occ_tv = jnp.mean(jnp.sum(edge_mass[None, :] * jnp.abs(occ_change[:, i_e] - occ_change[:, j_e]), axis=-1)
                              / (jnp.sum(edge_mass) + 1e-8))
            loss_occupancy = occupancy_l1 * occ_l1 + occupancy_tv * occ_tv
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
                if fused:
                    images_random = fused_stats["images_extra"]
                else:
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
                return img[0].reshape(-1)

            loss_geometric = composed_geometric_correction_loss(
                latent, _render_point, geom_ctx, model.xsize * model.xsize)
        else:
            loss_geometric = 0.0

        loss = (nll + kl_lambda * latent_alpha * kl_loss + 0.000001 * kl_pose + decoupling_lambda * decoupling_loss
                + l1_lambda * l1_loss + graph_lambda * loss_graph + 100. * hist_loss + distance_preservation_lambda * loss_dp
                + pose_refine_reg * pose_refine_loss + shift_refine_reg * shift_refine_loss
                + geometric_lambda * loss_geometric + loss_occupancy)

        # Pose diagnostics
        theta_deg = jnp.rad2deg(jnp.arccos(jnp.clip(cos_theta_rigid, -1.0 + 1e-6, 1.0 - 1e-6)))

        if fused:
            occ_fraction = jnp.zeros((), jnp.float32)
            deformation_A = jnp.zeros((), jnp.float32)
            deformation_p99_A = jnp.zeros((), jnp.float32)
        else:
            occ_fraction = jnp.mean(jnp.abs(occ_change)) / (jnp.mean(jnp.abs(values)) + 1e-8)
            disp_A = jnp.linalg.norm(
                coords - model.delta_volume_decoder.scale * model.delta_volume_decoder.coords,
                axis=-1) * model.sr
            deformation_A = jnp.mean(disp_A)
            deformation_p99_A = jnp.percentile(disp_A, 99.0)

        metrics = {"pose_angle_deg": jnp.mean(theta_deg),
                   "shift_norm_px": jnp.mean(jnp.sqrt(shifts_sq + 1e-8)),
                   "rigid_fraction": rigid_fraction,
                   "deformation_A": deformation_A,
                   "deformation_p99_A": deformation_p99_A,
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
    from hax.programs import fit_volume, fit_volume_adaptive, adjust_weights_to_images
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
                        help=f"Before training the network, HetSIREN fits a set of Gaussians to the reference volume to recreate it. "
                             f"By default the count is determined automatically: the fit starts deliberately coarse and grows the number of "
                             f"Gaussians until the map it renders correlates with your reference at {bcolors.ITALIC}FSC >= 0.5{bcolors.ENDC} "
                             f"across every shell the reference populates, so it lands on the fewest Gaussians that cost you no resolution "
                             f"(see {bcolors.ITALIC}--fit_resolution{bcolors.ENDC} to stop somewhere coarser).\n"
                             f"Set this parameter to pin the count in advance from your own criterium instead (e.g. the number of residues in your "
                             f"protein). When set, HetSIREN fits exactly this many Gaussians and reproduces the reference as well as that count allows.\n"
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: the count is not free. The Gaussian width is tied to the point spacing "
                             f"(half of it, which is what tiling costs), and the width in turn bounds the resolution the model can carry -- so too "
                             f"few Gaussians is a low-pass filter on everything the network can express, and too many just costs memory in every "
                             f"training step.")
    parser.add_argument("--fit_resolution", required=False, type=float, default=None,
                        help=f"Resolution (in {bcolors.UNDERLINE}Angstrom{bcolors.ENDC}) the automatic Gaussian fit has to reproduce.")
    parser.add_argument("--lattice_stride", required=False, type=int, default=1,
                        help=f"Sub-sample the fixed-point lattice by this factor along each axis, keeping "
                             f"{bcolors.ITALIC}1 / lattice_stride^3{bcolors.ENDC} of the sample points.\n"
                             f"{bcolors.WARNING}This trades resolution for memory{bcolors.ENDC}.")
    parser.add_argument("--bilinear_scatter", action='store_true',
                        help=f"Deposit each fixed-lattice point over its four neighbouring pixels (bilinear) instead of "
                             f"into the single nearest one.")
    parser.add_argument("--render_chunk_size", required=False, type=int, default=0,
                        help=f"Render the point cloud in blocks of this many points, summing the partial images and "
                             f"recomputing each block in the backward pass instead of keeping it live. "
                             f"Smaller blocks save more memory and compile more slowly.")
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
    parser.add_argument("--encoder_arch", required=False, type=str, default="convnn",
                        choices=["convnn", "mlpnn"],
                        help=f"Encoder that maps each image to its latent (set by default to {bcolors.ITALIC}convnn{bcolors.ENDC}).")
    parser.add_argument("--pose_refine_reg", required=False, type=float, default=0.1,
                        help=f"Strength of the hinge that keeps the per-image rigid pose refinement within {bcolors.ITALIC}--pose_refine_max_angle{bcolors.ENDC} of the input poses. "
                             f"It penalizes only the excess beyond that angle, so a genuinely-needed small correction is not pushed back to zero. Set to 0 to disable it.")
    parser.add_argument("--pose_refine_max_angle", required=False, type=float, default=8.0,
                        help=f"Deadzone (in degrees) of the pose-refinement hinge: rotations up to this angle are free, beyond it they are penalized with weight "
                             f"{bcolors.ITALIC}--pose_refine_reg{bcolors.ENDC}.")
    parser.add_argument("--shift_refine_reg", required=False, type=float, default=0.1,
                        help=f"Strength of the hinge on the per-image in-plane shift refinement, the exact analogue of {bcolors.ITALIC}--pose_refine_reg{bcolors.ENDC} for translations. "
                             f"Set to 0 to disable.")
    parser.add_argument("--shift_refine_max", required=False, type=float, default=4.0,
                        help=f"Deadzone (in pixels) of the shift-refinement hinge: shifts up to this magnitude are free, beyond it they are penalized with weight "
                             f"{bcolors.ITALIC}--shift_refine_reg{bcolors.ENDC}.")
    parser.add_argument("--latent_warmup_epochs", required=False, type=float, default=None,
                        help=f"Number of initial epochs over which the latent is ramped INTO the decoder (default: {bcolors.ITALIC}~12%% of --epochs{bcolors.ENDC}, so the schedule scales with the budget). "
                             f"For the first half of this window the decoder sees a zero latent, so it can only "
                             f"produce a single consensus volume -- and a consensus volume cannot absorb a {bcolors.ITALIC}per-particle{bcolors.ENDC} pose error, so the whole misalignment "
                             f"has to be explained by the pose head. Over the second half the latent (and the KL term, which rides the same clock) ramps in linearly, against poses that have "
                             f"already been refined. This is the coarse-to-fine ordering a classical refinement uses -- consensus and poses first, heterogeneity after -- and it costs nothing. "
                             f"Set to 0 to feed the full latent from the first step.")
    parser.add_argument("--rigid_gauge", required=False, type=str, default="core", choices=["core", "mass", "off"],
                        help=f"Pins the decoder's per-particle output to a canonical frame, so a pose error cannot be absorbed by the decoder and has nowhere to go except the pose head.\n"
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
                             f"more regularized latent space at the cost of reconstruction detail; decrease it (or set to 0) to let the encoder use the latent more freely.")
    parser.add_argument("--kl_free_bits", required=False, type=float, default=0.5,
                        help=f"Free-bits floor for the KL, in nats {bcolors.ITALIC}per latent dimension{bcolors.ENDC} (default {bcolors.ITALIC}0.5{bcolors.ENDC}). A dimension "
                             f"whose KL has fallen below this stops being penalised, which removes the incentive to squeeze the last information out of the latent -- the "
                             f"failure mode ({bcolors.ITALIC}posterior collapse{bcolors.ENDC}) where the encoder stops distinguishing particles and every one of them decodes "
                             f"to the consensus. It is a floor on the {bcolors.ITALIC}penalty{bcolors.ENDC}, not on the information: a dimension the model has no use for is "
                             f"still free to stay unused, so a genuinely low-dimensional landscape is not forced to spread out. Raise it if the latent collapses, lower it "
                             f"(0 reproduces a plain KL) if the latent is noisy. Only used when {bcolors.ITALIC}--kl_lambda{bcolors.ENDC} > 0.")
    parser.add_argument("--grad_clip_norm", required=False, type=float, default=1.0,
                        help=f"Global-norm gradient clipping threshold for the HetSIREN optimizer. Gradients whose global norm exceeds this value are rescaled down, which "
                             f"prevents occasional gradient spikes. Set to 0 to disable clipping.")
    parser.add_argument("--per_image_contrast", required=False, type=str, default="off",
                        choices=["off", "relative", "absolute"],
                        help=f"Fit a gray level per particle inside the reconstruction loss, instead of relying on the single global contrast that "
                             f"{bcolors.ITALIC}--vol{bcolors.ENDC} pins once before training.")
    parser.add_argument("--freq_anneal_epochs", required=False, type=float, default=0.0,
                        help=f"Coarse-to-fine annealing: number of initial epochs over which the effective resolution is ramped in. During this window the decoder's "
                             f"first-SIREN-layer frequency (w0) is scaled up from {bcolors.ITALIC}--freq_anneal_start{bcolors.ENDC} to full resolution. This "
                             f"forces the network to fit low-frequency structure before high-frequency detail, which strongly stabilizes training on noisy data (it prevents the "
                             f"SIREN from overfitting high-frequency noise early). Set to 0 to disable (train at full resolution from the first step). "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: has no effect on the point transformer decoder (it has no SIREN first layer).")
    parser.add_argument("--freq_anneal_start", required=False, type=float, default=0.2,
                        help=f"Starting fraction (in (0, 1]) for {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC}: freq_alpha begins at this value and ramps linearly to 1.0. "
                             f"Smaller values start coarser (lower resolution). Ignored when {bcolors.ITALIC}--freq_anneal_epochs{bcolors.ENDC} is 0.")
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
    parser.add_argument("--occupancy_l1", required=False, type=float, default=0.0,
                        help=f"Weight of the {bcolors.ITALIC}shrinkage{bcolors.ENDC} half of the occupancy prior. Occupancy is what lets HetSIREN model {bcolors.ITALIC}compositional"
                             f"{bcolors.ENDC} heterogeneity (a domain/ligand appearing or disappearing between particles) rather than forcing everything through the deformation field: each "
                             f"Gaussian carries a per-particle amplitude change on top of the reference. This term penalises {bcolors.ITALIC}how much{bcolors.ENDC} of it there is, with a "
                             f"{bcolors.ITALIC}scale-normalised L1{bcolors.ENDC} (cost = fold change where the reference has density, so modulating existing density is cheap, while creating "
                             f"mass where the reference is empty costs {bcolors.ITALIC}|occ|/--occupancy_kappa{bcolors.ENDC} -- possible but expensive). It is a mean over points, so it is on "
                             f"the same scale as {bcolors.ITALIC}--occupancy_tv{bcolors.ENDC}. Only meaningful with mass transport ({bcolors.ITALIC}--transport_mass{bcolors.ENDC}) and a "
                             f"reference. {bcolors.WARNING}NOTE{bcolors.ENDC}: this term only ever shrinks occupancy -- if the {bcolors.ITALIC}occ_fraction{bcolors.ENDC} diagnostic collapses "
                             f"towards 0, this is too high.")
    parser.add_argument("--occupancy_tv", required=False, type=float, default=0.0,
                        help=f"Weight of the {bcolors.ITALIC}spatial-coherence{bcolors.ENDC} half of the occupancy prior (graph total variation). It penalises occupancy "
                             f"{bcolors.ITALIC}differences{bcolors.ENDC} between neighbouring Gaussians, so a whole domain switching on together pays only at its boundary while scattered "
                             f"per-Gaussian noise pays everywhere -- this is what separates a genuine compositional domain from amplitude noise, and it is the term to reach for if occupancy "
                             f"is coming out incoherent. Raise it for blockier occupancy, lower it toward 0 to allow finer-grained occupancy. Weighted independently of "
                             f"{bcolors.ITALIC}--occupancy_l1{bcolors.ENDC}: coherent-but-large occupancy is a perfectly reasonable thing to ask for, so setting this alone (L1 at 0) is a "
                             f"normal configuration. Occupancy is active when either weight is > 0; with both at 0 (default) the amplitudes stay at the reference (pure mass transport).")
    parser.add_argument("--occupancy_kappa", required=False, type=float, default=0.05,
                        help=f"Creation threshold of the occupancy prior (as a density, in the reference's gray scale; default {bcolors.ITALIC}0.05{bcolors.ENDC}). It is added to the "
                             f"reference amplitude in the L1 denominator, so it sets how much evidence it takes to invent mass where the reference is (near) empty: smaller = harder to create "
                             f"mass, larger = easier. Where the reference already has substantial density it has little effect (the cost is a fold change). Only used when "
                             f"{bcolors.ITALIC}--occupancy_l1{bcolors.ENDC} > 0.")
    parser.add_argument("--occupancy_warmup_epochs", required=False, type=float, default=3.0,
                        help=f"Number of initial epochs over which occupancy is ramped in (default {bcolors.ITALIC}3.0{bcolors.ENDC}). While it ramps, the amplitudes stay at the reference "
                             f"so the data is explained purely by transport -- large motions organise first -- and occupancy is then released to explain only the residual that motion cannot "
                             f"(compositional change). Feeding occupancy from step 0 lets it short-circuit the motion (it is a strictly easier way to fit any image), so a non-zero warmup is "
                             f"recommended. Set to 0 to use occupancy from the first step. This ramp is independent of the occupancy priors -- it controls how occupancy enters the density, "
                             f"not how it is penalised -- so it applies whether or not {bcolors.ITALIC}--occupancy_l1{bcolors.ENDC}/{bcolors.ITALIC}--occupancy_tv{bcolors.ENDC} are set.")
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

    # The latent is read out, and decoded, in two halves -- see MultiEncoder.__init__.
    # Checked here as well as in the model so an odd value fails immediately, rather than
    # after the (20k-iteration) Gaussian fit that runs before the model is built.
    if args.lat_dim % 2 != 0:
        raise ValueError(f"{bcolors.FAIL}--lat_dim must be even: the latent is split into a motion half "
                         f"and an occupancy half. Got {args.lat_dim}; use {args.lat_dim + 1}.{bcolors.ENDC}")

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

    box_size = generator.md.getMetaDataImage(0).shape[0]

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    elif not transport_mass and not local_reconstruction:
        # Without mass transport the decoder does not MOVE density, it ADDS it on a fixed
        # lattice -- so the lattice is not a starting point that can migrate, it IS the
        # reconstruction domain. A mask derived from the consensus density would forbid the
        # model from putting mass anywhere the consensus happens to be empty, which is
        # precisely where a conformational change needs to put it, and no amount of training
        # can recover the excluded region. The domain therefore has to be a plain sphere.
        #
        # Density-derived masks are only appropriate WITH mass transport, where the points
        # start on the consensus and are free to move outside it, and in local
        # reconstruction, where restricting to the mask is the stated goal.
        mask = ImageHandler().createCircularMask(boxSize=box_size, is3D=True)
    elif auto_reference:
        # Derived from the map reconstructed
        mask = consensus_mask(vol)
        ImageHandler().write(mask, os.path.join(args.output_path, "consensus_mask.mrc"), overwrite=True)
    elif args.transport_mass and args.vol is not None:
        mask = ImageHandler(args.vol).generateMask(boxsize=64)
    else:
        mask = ImageHandler().createCircularMask(boxSize=box_size, is3D=True)

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

                    # Consensus volume. Both paths produce the same kind of model, so the
                    # contrast match below and the point extraction further down are shared:
                    # --num_gaussians pins the count, otherwise it is searched for.
                    if args.num_gaussians is not None:
                        model, _, _ = fit_volume(vol, mask=mask_fit, iterations=20000, learning_rate=0.001,
                                                 n_init=args.num_gaussians, fixed_gaussians=True)
                    else:
                        model, _ = fit_volume_adaptive(vol, mask_fit, args.sr,
                                                       resolution=args.fit_resolution)

                    model, _ = adjust_weights_to_images(model, args.md, mmap_output_dir, args.sr,
                                                        ctf_type=args.ctf_type)

                    # Save model and volume
                    vol_splatted = np.array(model())
                    NeuralNetworkCheckpointer.save(model, fit_path)

                    # Save volume
                    ImageHandler().write(vol_splatted, os.path.join(args.output_path, "consensus_volume.mrc"), overwrite=True)
                else:
                    model = NeuralNetworkCheckpointer.load(checkpoint_path=fit_path)

            if transport_mass:
                # Prepare network (HetSIREN)
                factor = 0.5 * generator.md.getMetaDataImage(0).shape[0]
                if args.vol is not None:
                    coords = np.array(factor * model.means.get_value() + factor)
                    coords = np.stack([coords[..., 2], coords[..., 1], coords[..., 0]], axis=1)
                    values = np.array(jax.nn.relu(model.weights.get_value()))
                    sigma = float(jax.nn.relu(model.sigma_param.get_value()).mean())
                else:
                    inds = decimate_lattice(mask, args.lattice_stride)
                    coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
                    values = jnp.zeros((inds.shape[0],))
                    sigma = float(max(1.0, 0.5 * args.lattice_stride))
            else:
                inds = decimate_lattice(mask, args.lattice_stride)
                coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
                # A strided lattice needs wider splats to tile the volume, but only half the
                # spacing -- see PhysDecoder.__call__, where the same width is derived and
                # where using the full spacing costs the resolution the data actually has.
                # The Gaussian-fit branch below carries its own fitted width instead.
                lattice_sigma = float(max(1.0, 0.5 * args.lattice_stride))
                if args.vol is None:
                    values = jnp.zeros((inds.shape[0],))
                    sigma = lattice_sigma
                elif not fit_gaussians:
                    values = np.asarray(vol)[inds[:, 0], inds[:, 1], inds[:, 2]]
                    sigma = lattice_sigma
                else:
                    vol = np.array(model())
                    values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
                    sigma = float(jax.nn.relu(model.sigma_param.get_value()).mean())

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
                                rigid_gauge=args.rigid_gauge != "off",
                                rigid_gauge_irls=1 if args.rigid_gauge == "core" else 0,
                                render_chunk_size=args.render_chunk_size,
                                lattice_stride=args.lattice_stride,
                                bilinear_scatter=args.bilinear_scatter,
                                architecture=args.encoder_arch, rngs=nnx.Rngs(model_key))

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
                            # Decode the sanity-check volumes ONE latent at a time, moving each off
                            # the device before the next, exactly as the landscape block below does.
                            # Decoding the whole batch at once holds N volumes plus their low-pass
                            # FFTs live together -- ~3.1 GiB for 5 volumes at box 320, which OOMs an
                            # 8 GB card on top of the resident training state even though the training
                            # step itself fits. One at a time the peak is a single volume (~0.9 GiB),
                            # and the batch size (what actually drives training memory) is irrelevant
                            # here since this always decodes exactly len(latents_intermediate) volumes.
                            volumes_intermediate = np.stack([
                                np.array(hetsiren_decode_volume(graphdef, state, z[None, ...]))[0]
                                for z in latents_intermediate])

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

                            # Sprite images for the Tensorboard projector. Decode in small
                            # fixed chunks, NOT chunks of args.batch_size: the batch is sized
                            # for the training step, and a large fixed-grid lattice (~1.1M
                            # points at box 128) makes a full-batch decode need several GiB,
                            # which OOMs the card here (surfacing as a segfault off the logging
                            # thread). 32 keeps the full projector while bounding the peak.
                            sprite_chunk = min(32, args.batch_size)
                            latents_images = []
                            for start in range(0, latents_intermediate.shape[0], sprite_chunk):
                                latents = latents_intermediate[start:start + sprite_chunk]
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
                # It scales the decoder's first-SIREN-layer w0, so low-frequency
                # structure is fitted first. 0 epochs -> disabled (== 1.0).
                if args.freq_anneal_epochs and args.freq_anneal_epochs > 0:
                    freq_steps = max(1, int(args.freq_anneal_epochs * steps_per_epoch))
                    freq_alpha = float(args.freq_anneal_start
                                       + (1.0 - args.freq_anneal_start) * min(1.0, total_steps / freq_steps))
                else:
                    freq_alpha = 1.0

                # Occupancy curriculum: ramp the per-particle amplitude change in from 0,
                # so early training explains the data by transport (motion) and occupancy
                # is only released later to catch the residual motion cannot.
                #
                # This is a curriculum on how occupancy enters the density, not on how it is
                # penalised, so it is independent of the occupancy priors -- an unregularised
                # occupancy is exactly the case where letting it loose from step 0 does the
                # most damage, so gating the ramp on the priors had it backwards.
                if args.occupancy_warmup_epochs > 0:
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
                                                                   latent_alpha=latent_alpha,
                                                                   pose_refine_reg=args.pose_refine_reg,
                                                                   pose_refine_max_angle=args.pose_refine_max_angle,
                                                                   shift_refine_reg=args.shift_refine_reg,
                                                                   shift_refine_max=args.shift_refine_max,
                                                                   decoupling_lambda=args.decoupling_lambda,
                                                                   distance_preservation_lambda=args.distance_preservation_lambda,
                                                                   kl_lambda=args.kl_lambda,
                                                                   kl_free_bits=args.kl_free_bits,
                                                                   freq_alpha=freq_alpha,
                                                                   geometric_lambda=args.geometric_lambda,
                                                                   occ_alpha=occ_alpha,
                                                                   occupancy_l1=args.occupancy_l1,
                                                                   occupancy_tv=args.occupancy_tv,
                                                                   occupancy_kappa=args.occupancy_kappa,
                                                                   amp_recon_weight=amp_recon_weight,
                                                                   per_image_contrast=args.per_image_contrast)
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
                    deformation_p99_A = float(pose_metrics["deformation_p99_A"])
                    occ_fraction = float(pose_metrics["occ_fraction"])

                    writer.add_scalars('Pose refinement (HetSIREN)',
                                       {"rotation (deg)": pose_angle_deg,
                                        "shift (px)": shift_norm_px},
                                       i * steps_per_epoch + step)
                    writer.add_scalar('Decoder rigid fraction (HetSIREN)',
                                      rigid_fraction,
                                      i * steps_per_epoch + step)
                    # Both curves on one plot: the gap between them IS the localisation of
                    # the motion. p99 >> mean means a small domain is swinging; the two
                    # tracking each other means the whole cloud is moving together, which
                    # (with a rising rigid_fraction) is pose absorption rather than conformation.
                    writer.add_scalars('Decoder deformation (A) (HetSIREN)',
                                       {"mean": deformation_A,
                                        "p99 (local)": deformation_p99_A},
                                       i * steps_per_epoch + step)
                    writer.add_scalar('Occupancy fraction (HetSIREN)',
                                      occ_fraction,
                                      i * steps_per_epoch + step)

                    # Progress bar update  (TQDM)
                    pose_str = (f"pose={pose_angle_deg:.2f}deg/{shift_norm_px:.2f}px | "
                                f"rigid={rigid_fraction:.3f} | "
                                f"def={deformation_A:.2f}/{deformation_p99_A:.1f}A(mean/p99)")
                    # Reported whenever occupancy is live, not only when it is regularised --
                    # an unregularised occupancy is precisely the one worth watching.
                    #
                    # Scientific notation, not %.3f: the values that need reading span several
                    # decades (a head still growing off its zero init sits at ~1e-5, one the L1
                    # has annihilated at ~1e-6, a healthy one at ~1e-1), and %.3f prints every
                    # one of them below 5e-4 as a flat 0.000 -- i.e. it is blind in exactly the
                    # range where the number decides whether the prior is mis-scaled.
                    occ_str = f" | occ={occ_fraction:.2e}" if transport_mass else ""
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

        # Example of predicted data for Tensorboard. Decode only a handful of images, as the
        # in-epoch logging does (x[:5] above): this runs the WHOLE batch through the decoder,
        # and the batch is sized for the training step, not for a diagnostic. With a large
        # lattice (e.g. the fixed-grid spherical mask, ~1.1M points at box 128) a full
        # auto-sized batch here needs several GiB on top of the resident training state and
        # OOMs the card -- which, off the main thread or in a native allocation, surfaces as a
        # segfault rather than a clean error. Five images is all the TensorBoard panel shows.
        x_pred_example = hetsiren_decode_image(graphdef, state, x_example[:5], labels_example[:5], md_columns, ctf_type=args.ctf_type, return_latent=False, corrupt_projection_with_ctf=True)
        x_pred_example = jax.vmap(min_max_scale)(x_pred_example[..., None])
        writer.add_images("Predicted images batch", x_pred_example, dataformats="NHWC")

        # Let the background logging thread and the asynchronous checkpoint write post_scatter_operations
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
