#!/usr/bin/env python


import random
from functools import partial

import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import dm_pix

from einops import rearrange

from hax.utils import *
from hax.layers import *


class Encoder(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, architecture="convnn", isVae=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 32  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 4))
        self.architecture = architecture
        self.isVae = isVae
        self.normal_key = rngs.distributions()

        if self.architecture == "mlpnn":
            self.hidden_layers = [Linear(self.input_dim * self.input_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
            for _ in range(n_layers):
                self.hidden_layers.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers.append(Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16))
            for _ in range(2):
                self.hidden_layers.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
            self.latent = Linear(256, lat_dim, rngs=rngs)

        elif self.architecture == "convnn":
            self.hidden_layers_conv = [Linear(self.input_dim * self.input_dim, self.input_conv_dim * self.input_conv_dim, rngs=rngs, dtype=jnp.bfloat16)]
            self.hidden_layers_conv.append(Conv(1, 4, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(Conv(4, 8, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(Conv(8, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(Conv(16, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

            self.hidden_layers_linear = [Linear(16 * self.out_conv_dim * self.out_conv_dim, 256, rngs=rngs, dtype=jnp.bfloat16)]
            for _ in range(3):
                self.hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

            self.hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
            for _ in range(2):
                self.hidden_layers_linear.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

            if isVae:
                self.mean_x = Linear(256, lat_dim, rngs=rngs)
                self.logstd_x = Linear(256, lat_dim, rngs=rngs)
            else:
                self.latent = Linear(256, lat_dim, rngs=rngs)

        else:
            raise ValueError("Architecture not supported. Implemented architectures are: mlpnn / convnn")

    def sample_gaussian(self, mean, logstd):
        return logstd * jnr.normal(self.normal_key, shape=mean.shape) + mean

    def __call__(self, x, return_last=False):
        if self.architecture == "mlpnn":
            x = rearrange(x, 'b h w c -> b (h w c)')

            for layer in self.hidden_layers:
                x = nnx.relu(layer(x))

        elif self.architecture == "convnn":
            x = rearrange(x, 'b h w c -> b (h w c)')

            x = nnx.relu(self.hidden_layers_conv[0](x))

            x = rearrange(x, 'b (h w c) -> b h w c', h=self.input_conv_dim, w=self.input_conv_dim, c=1)

            for layer in self.hidden_layers_conv[1:]:
                if layer.in_features != layer.out_features:
                    x = nnx.relu(layer(x))
                else:
                    aux = layer(x)
                    if aux.shape[1] == x.shape[1]:
                        x = nnx.relu(x + aux)
                    else:
                        x = nnx.relu(aux)

            x = rearrange(x, 'b h w c -> b (h w c)')

            for layer in self.hidden_layers_linear:
                x = nnx.relu(layer(x))

        if return_last:
            return x
        else:
            if self.isVae:
                mean = self.mean_x(x)
                logstd = self.logstd_x(x)
                sample = self.sample_gaussian(mean, logstd)
                return sample, mean, logstd
            else:
                latent = self.latent(x)
                return latent

class EncoderTomo(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, isVae=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.isVae = isVae
        self.normal_key = rngs.distributions()

        self.hidden_layers = [Linear(self.input_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(n_layers):
            self.hidden_layers.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers.append(Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16))
        for _ in range(2):
            self.hidden_layers.append(Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        self.latent = Linear(256, lat_dim, rngs=rngs)

        if isVae:
            self.mean_x = Linear(256, lat_dim, rngs=rngs)
            self.logstd_x = Linear(256, lat_dim, rngs=rngs)
        else:
            self.latent = Linear(256, lat_dim, rngs=rngs)

    def sample_gaussian(self, mean, logstd):
        return logstd * jnr.normal(self.normal_key, shape=mean.shape) + mean

    def __call__(self, x, return_last=False):
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))

        if return_last:
            return x
        else:
            if self.isVae:
                mean = self.mean_x(x)
                logstd = self.logstd_x(x)
                sample = self.sample_gaussian(mean, logstd)
                return sample, mean, logstd
            else:
                latent = self.latent(x)
                return latent

class MultiEncoder(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, isVae=False, architecture="convnn", isTomoSIREN=False, *, rngs: nnx.Rngs):
        if isTomoSIREN:
            self.encoders = {"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, architecture=architecture, rngs=rngs),
                             "encoder_dec": EncoderTomo(100, lat_dim, n_layers=n_layers, rngs=rngs)}
        else:
            self.encoders = {"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, architecture=architecture, rngs=rngs),
                             "encoder_dec": Encoder(input_dim, lat_dim, n_layers=n_layers, architecture=architecture, rngs=rngs)}
        self.normal_key = rngs.distributions()
        self.isVae = isVae
        if isVae:
            self.mean_x = Linear(256, lat_dim, rngs=rngs)
            self.logstd_x = Linear(256, lat_dim, rngs=rngs)
        else:
            self.latent = Linear(256, lat_dim, rngs=rngs)

    def sample_gaussian(self, mean, logstd):
        return logstd * jnr.normal(self.normal_key, shape=mean.shape) + mean

    def __call__(self, x, encoder_id="encoder_exp", return_last=False):
        x = self.encoders[encoder_id](x, return_last=True)

        if self.isVae:
            mean = self.mean_x(x)
            logstd = self.logstd_x(x)
            sample = self.sample_gaussian(mean, logstd)
            if return_last:
                return [sample, mean, logstd], x
            else:
                return sample, mean, logstd
        else:
            latent = self.latent(x)
            if return_last:
                return latent, x
            else:
                return latent


class DeltaVolumeDecoder(nnx.Module):
    def __init__(self, total_voxels, lat_dim, volume_size, inds, reference_values, transport_mass=False, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.inds = inds
        self.reference_values = reference_values
        self.total_voxels = total_voxels
        self.transport_mass = transport_mass

        # Indices to (normalized) coords
        self.factor = 0.5 * volume_size
        coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)[None, ...]
        self.coords = (coords - self.factor) / self.factor

        # Delta volume decoder (TODO: Check and fix hypernetwork - compare with TF implementation)
        # self.hidden_linear = [HyperLinear(in_features=lat_dim, out_features=8, in_hyper_features=lat_dim, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16)]
        # for _ in range(3):
        #     self.hidden_linear.append(HyperLinear(in_features=8, out_features=8, in_hyper_features=8, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_linear.append(HyperLinear(in_features=8, out_features=8, in_hyper_features=8, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16))

        self.hidden_linear = [Linear(in_features=lat_dim, out_features=8, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init_first(c=1.))]
        for _ in range(3):
            self.hidden_linear.append(Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init(c=6.)))
        self.hidden_linear.append(Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init(c=6.)))

        if transport_mass:
            self.hidden_linear.append(Linear(in_features=8, out_features=4 * total_voxels, rngs=rngs, kernel_init=nnx.initializers.glorot_uniform()))
        else:
            self.hidden_linear.append(Linear(in_features=8, out_features=total_voxels, rngs=rngs, kernel_init=nnx.initializers.glorot_uniform()))


    def __call__(self, x):
        # Decode voxel values
        x = jnp.sin(30.0 * self.hidden_linear[0](x))
        for layer in self.hidden_linear[1:-1]:
            # x = jnp.sin(1.0 * (x + layer(x, x)))
            x = x + jnp.sin(1.0 * layer(x))
        x = self.hidden_linear[-1](x)

        if self.transport_mass:
            # Extract delta_coords and values
            x = jnp.reshape(x, (x.shape[0], self.total_voxels, 4))
            delta_coords, delta_values = x[..., :3], x[..., 3]

            # Recover volume values (TODO: Check if applying ReLu is really needed)
            # values = nnx.relu(self.reference_values + delta_values)
            values = self.reference_values + delta_values

            # Recover coords (non-normalized)
            coords = self.factor * (self.coords + delta_coords)
        else:
            # Recover volume values
            values = self.reference_values + x

            # Recover coords (non-normalized)
            coords = self.factor * self.coords

        return coords, values

    def decode_volume(self, x=None, coords_values=None, filter=True):
        if x is not None:
            # Decode volume values
            coords, values = self.__call__(x)
        elif coords_values is not None:
            coords, values = coords_values
        else:
            raise ValueError("Please provide either x or coords_value parameter")

        # Displace coordinates
        coords = coords + self.factor

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

        grids = jax.vmap(scatter_volume, in_axes=(0, None, 0))(grids, bposi, bamp)

        # Filter volume
        if filter:
            grids = jax.vmap(low_pass_3d)(grids)

        return grids

class PhysDecoder:
    def __init__(self, xsize, transport_mass):
        self.xsize = xsize
        self.transport_mass = transport_mass

    def __call__(self, x, values, coords, xsize, rotations, shifts, ctf, ctf_type, filter=True):
        # Volume factor
        factor = 0.5 * xsize

        # Get rotation matrices
        if rotations.ndim == 2:
            rotations = euler_matrix_batch(rotations[:, 0], rotations[:, 1], rotations[:, 2])
        coords = jnp.matmul(coords, rearrange(rotations, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts[:, None, :] + factor

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

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        images = jax.vmap(scatter_img)(images, bposi, bamp)

        # Gaussian filter (needed by forward interpolation)
        if filter:
            images = dm_pix.gaussian_blur(images[..., None], 1.0, kernel_size=3)[..., 0]

        # Apply CTF
        if ctf_type in ["apply" or "wiener" or "squared"]:
            images = ctfFilter(images, ctf, pad_factor=2)

        return images

class HetSIREN(nnx.Module):
    def __init__(self, lat_dim, reference_volume, reconstruction_mask, xsize, sr, bank_size=10000, ctf_type="apply",
                 decoupling=False, isVae=False, transport_mass=False, local_reconstruction=False, architecture="convnn",
                 isTomoSIREN=False, *, rngs: nnx.Rngs):
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
        self.inds = jnp.asarray(jnp.where(reconstruction_mask > 0.0)).T
        reference_values = reference_volume[self.inds[..., 0], self.inds[..., 1], self.inds[..., 2]][None, ...]
        self.encoder = MultiEncoder(self.xsize, lat_dim, n_layers=3, isVae=isVae, architecture=architecture, isTomoSIREN=isTomoSIREN, rngs=rngs) \
            if decoupling or isTomoSIREN else Encoder(self.xsize, lat_dim, isVae=isVae, architecture=architecture, rngs=rngs)
        self.delta_volume_decoder = DeltaVolumeDecoder(self.inds.shape[0], lat_dim, self.xsize, self.inds, reference_values, transport_mass=transport_mass, rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize, transport_mass=transport_mass)

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.subset_size = min(2048, bank_size)
        self.choice_key = rngs.choice()

        self.memory_bank = nnx.Variable(
            jnp.zeros((self.bank_size, lat_dim))
        )
        self.memory_bank_ptr = nnx.Variable(
            jnp.zeros((1,), dtype=jnp.int32)
        )

    def __call__(self, x, **kwargs):
        if self.isVae:
            if self.decoupling:
                sample, mean, _ = self.encoder(x, "encoder_exp", return_last=False)
            else:
                sample, mean, _ = self.encoder(x, return_last=False)
            if kwargs.pop("gaussian_sample", False):
                latent = sample
            else:
                latent = mean
        else:
            if self.decoupling:
                latent = self.encoder(x, "encoder_exp", return_last=False)
            else:
                latent = self.encoder(x, return_last=False)
        return latent

    # --- Method for enqueuing to the memory bank ---
    def enqueue(self, keys_to_add):
        """Updates the memory bank and pointer using JIT-compatible operations."""
        ptr = self.memory_bank_ptr.value[0]

        # Define the starting position for the update.
        # It must be a tuple with one index per dimension of the array.
        # Our memory_bank is 2D, so we need (start_row, start_column).
        start_indices = (ptr, 0)

        # Use `lax.dynamic_update_slice` instead of `.at[...].set(...)`
        self.memory_bank.value = jax.lax.dynamic_update_slice(
            self.memory_bank.value, # 1. The original large array to be updated
            keys_to_add,             # 2. The smaller array containing the new data
            start_indices            # 3. The dynamic starting position
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
            latents = self(x)
        else:
            latents = x

        # Decode volumes
        coords, values = self.delta_volume_decoder(latents)

        # CTF corruption
        if not corrupt_projection_with_ctf:
            ctf_type = None

        # Generate projections
        images_corrected = self.phys_decoder(x, values, coords, self.xsize, euler_angles, shifts, ctf, ctf_type)

        if return_latent:
            return images_corrected, latents
        else:
            return images_corrected

    def decode_volume(self, x):
        if x.ndim == 1:
            x = x[None, ...]

        return self.delta_volume_decoder.decode_volume(x=x, filter=True)


@jax.jit
def train_step_hetsiren(graphdef, state, x, labels, md):
    model, optimizer = nnx.merge(graphdef, state)
    distributions_key = jnr.PRNGKey(random.randint(0, 2 ** 32 - 1))

    def loss_fn(model, x):
        # Check if Tomo mode
        if model.isTomoSIREN:
            (x, subtomogram_label) = x

        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (sample, latent, logstd), prev_layer_out = model.encoder(x, "encoder_exp", return_last=True)
            elif model.isTomoSIREN:
                (sample, latent, logstd), prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
            else:
                sample, latent, logstd = model.encoder(x)
        else:
            if model.decoupling:
                latent, prev_layer_out = model.encoder(x, "encoder_exp", return_last=True)
            elif model.isTomoSIREN:
                (sample, latent, logstd), prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
            else:
                latent = model.encoder(x)

        # Decode volumes
        if model.isVae:
            coords, values = model.delta_volume_decoder(sample)
        else:
            coords, values = model.delta_volume_decoder(latent)
        # volumes = model.delta_volume_decoder.decode_volume(coords_values=[coords, values])
        # chimeric_volumes =  volumes + ((1. - model.reconstruction_mask) * model.reference_volume)[None, ...]

        # Generate projections
        images_corrected = model.phys_decoder(x, values, coords, model.xsize, euler_angles, shifts, ctf, model.ctf_type)

        # Project "mask"
        if not model.delta_volume_decoder.transport_mass:
            projected_mask = model.phys_decoder(x, jnp.ones_like(values), coords, model.xsize, euler_angles, shifts, ctf, None, False)
            projected_mask = jnp.where(projected_mask > 1, 1.0, projected_mask)
        else:
            projected_mask = jnp.ones_like(x)

        # Losses
        images_corrected = jnp.squeeze(images_corrected)
        x = jnp.squeeze(x)

        # Consider CTF if Wiener mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter(x, ctf, pad_factor=2)
            images_corrected_loss = wiener2DFilter(images_corrected, ctf, pad_factor=2)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter(x, ctf, pad_factor=2)
            images_corrected_loss = ctfFilter(images_corrected, ctf, pad_factor=2)
        else:
            x_loss = x
            images_corrected_loss = images_corrected

        # Projection mask
        projected_mask = jnp.squeeze(projected_mask)
        x_loss = x_loss * projected_mask
        images_corrected_loss = images_corrected_loss * projected_mask

        # recon_loss = dm_pix.mae(images_corrected_loss[..., None], x_loss[..., None]).mean()
        recon_loss = dm_pix.mse(images_corrected_loss[..., None], x_loss[..., None]).mean()

        # L1 based denoising
        if not model.delta_volume_decoder.transport_mass:
            l1_loss = jnp.mean(jnp.abs(values))
        else:
            l1_loss = 0.0

        # L1 and L2 total variation
        # diff_x = volumes[:, 1:, :, :] - volumes[:, :-1, :, :]
        # diff_y = volumes[:, :, 1:, :] - volumes[:, :, :-1, :]
        # diff_z = volumes[:, :, :, 1:] - volumes[:, :, :, :-1]
        # l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        # l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()
        diff_x, diff_y, diff_z = sparse_finite_3D_differences(values, model.inds, model.xsize)
        l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        # Chimeric volume losses

        if model.local_reconstruction:
            outsize_mask = (1. - model.reconstruction_mask).astype(jnp.bool)
            hist_loss = (jnp.square(values.max(axis=1) - model.reference_volume.max(where=outsize_mask, initial=model.reference_volume.max())[None, ...]).mean() +
                         jnp.square(values.min(axis=1) - model.reference_volume.min(where=outsize_mask, initial=model.reference_volume.min())[None, ...]).mean() +
                         jnp.square(values.mean(axis=1) - model.reference_volume.mean(where=outsize_mask)[None, ...]).mean() +
                         jnp.square(values.std(axis=1) - model.reference_volume.std(where=outsize_mask)[None, ...]).mean())
        else:
            hist_loss = 0.0


        if model.isVae:
            # KL divergence loss
            kl_loss = -0.5 * jnp.sum(1 + 2 * logstd - jnp.square(jnp.exp(logstd)) - jnp.square(latent))
        else:
            kl_loss = 0.0

        # Decoupling
        if model.decoupling or model.isTomoSIREN:
            if model.isTomoSIREN:
                if model.isVae:
                    (_, latent_1, _), prev_layer_out_random = model.encoder(x[..., None], "encoder_exp", return_last=True)
                    (_, latent_2, _) = model.encoder(x[..., None], "encoder_exp")
                else:
                    latent_1, prev_layer_out_random = model.encoder(x[..., None], "encoder_exp", return_last=True)
                    latent_2 = model.encoder(x[..., None], "encoder_exp")
            else:
                images_random = model.phys_decoder(x, values, coords, model.xsize, rotations_random, shifts, ctf_random, model.ctf_type)
                if model.isVae:
                    (_, latent_1, _), prev_layer_out_random = model.encoder(images_corrected[..., None], "encoder_dec", return_last=True)
                    (_, latent_2, _) = model.encoder(images_random[..., None], "encoder_dec")
                else:
                    latent_1, prev_layer_out_random = model.encoder(images_corrected[..., None], "encoder_dec", return_last=True)
                    latent_2 = model.encoder(images_random[..., None], "encoder_dec")
            decoupling_loss = (jnp.mean(jnp.square(latent - latent_1), axis=-1).mean() +
                               jnp.mean(jnp.square(latent - latent_2), axis=-1).mean() +
                               jnp.mean(jnp.square(prev_layer_out - prev_layer_out_random), axis=-1).mean())

            random_indices = jnr.choice(model.choice_key, a=jnp.arange(model.bank_size), shape=(model.subset_size,),
                                        replace=False)
            memory_bank_subset = model.memory_bank[random_indices]

            dist = jnp.pow(latent[:, None, :] - memory_bank_subset, 2.).sum(axis=-1)
            dist_nn, _ = jax.lax.approx_min_k(dist, k=10, recall_target=0.95)
            dist_fn, _ = jax.lax.approx_max_k(dist, k=10, recall_target=0.95)

            decoupling_loss += 1.0 * triplet_loss(dist_nn, dist_fn, reduction="mean", margin=0.01)

        else:
            decoupling_loss = 0.0

        loss = recon_loss + 0.0001 * kl_loss + 0.001 * decoupling_loss + l1_loss  + 0.1 * (l1_grad_loss + l2_grad_loss) + 100. * hist_loss
        return loss, (recon_loss, latent)

    # Check if Tomo mode
    if model.isTomoSIREN:
        (x, subtomogram_label) = x

    # Precompute batch aligments
    euler_angles = md["euler_angles"][labels]

    # Precompute batch shifts
    shifts = md["shifts"][labels]

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

    # Prepare data for decoupling encoder
    rotations_random = jnr.choice(distributions_key, euler_angles, axis=0, shape=(x.shape[0],), replace=False)
    if model.ctf_type == "apply":
        defocusU = jnr.choice(distributions_key, defocusU, axis=0, shape=(x.shape[0],), replace=False)
        defocusV = jnr.choice(distributions_key, defocusV, axis=0, shape=(x.shape[0],), replace=False)
        defocusAngle = jnr.choice(distributions_key, defocusAngle, axis=0, shape=(x.shape[0],), replace=False)
        ctf_random = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                                model.sr, [2 * model.xsize, int(2 * 0.5 * model.xsize + 1)],
                                x.shape[0], True)
    else:
        ctf_random = jnp.ones([x.shape[0], 2 * model.xsize, int(2.0 * 0.5 * model.xsize + 1)], dtype=x.dtype)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    if model.isTomoSIREN:
        (loss, (recon_loss, latent)), grads = grad_fn(model, (x, subtomogram_label))
    else:
        (loss, (recon_loss, latent)), grads = grad_fn(model, x)

    optimizer.update(grads)

    # Update memory bank
    model.enqueue(latent)

    state = nnx.state((model, optimizer))

    return loss, recon_loss, state




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
    import torch
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, extract_columns, NumpyGenerator
    from hax.networks import train_step_hetsiren
    from hax.metrics import JaxSummaryWriter
    from hax.networks import VolumeAdjustment, train_step_volume_adjustment

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
                             f"is the size of the step it takes on each attempt. A large {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}0.01{bcolors.ENDC}) is like taking huge leaps — it's fast but can be unstable, "
                             f"overshoot the lowest point, or cause {bcolors.ITALIC}NAN{bcolors.ENDC} errors. A small {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}1e-6{bcolors.ENDC}) is like taking tiny "
                             f"shuffles — it's stable but very slow and might get stuck before reaching the bottom. A good default is often {bcolors.ITALIC}0.0001{bcolors.ENDC}. If training fails or errors explode, "
                             f"try making the {bcolors.ITALIC}lr{bcolors.ENDC} 10 times smaller (e.g., {bcolors.ITALIC}0.001{bcolors.ENDC} --> {bcolors.ITALIC}0.0001{bcolors.ENDC}).")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, new metadata...)")
    parser.add_argument("--reload", required=False, type=str,
                        help=f"Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data - "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: If a reference volume was provided, HetSIREN also learns a gray level adjustment. In this case, "
                             f"reload must be the path to a folder containing two additional folders called: {bcolors.UNDERLINE}HetSIREN{bcolors.ENDC} and {bcolors.UNDERLINE}volumeAdjustment{bcolors.ENDC})")
    args = parser.parse_args()

    # Manually handed parameters
    local_reconstruction = args.local_reconstruction
    transport_mass = args.transport_mass if not local_reconstruction else False

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

    # Data loading approach
    if args.load_images_to_ram:
        mmap = False
        mmap_output_dir = None
    else:
        mmap = True
        mmap_output_dir = args.output_path

    # If exists, clean MMAP
    if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
        shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key, choice_key = jax.random.split(rng, 3)

    # Prepare network (HetSIREN)
    hetsiren = HetSIREN(args.lat_dim, vol, mask, generator.md.getMetaDataImage(0).shape[0], args.sr,
                        ctf_type=args.ctf_type, decoupling=True, isVae=True, transport_mass=transport_mass,
                        local_reconstruction=local_reconstruction, bank_size=len(generator.md), isTomoSIREN=isTomoSIREN,
                        rngs=nnx.Rngs(model_key, choice=choice_key))

    # Reload network
    if args.reload is not None:
        hetsiren = NeuralNetworkCheckpointer.load(hetsiren, os.path.join(args.reload, "HetSIREN"))

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
            volumeAdjustment = NeuralNetworkCheckpointer.load(volumeAdjustment, os.path.join(args.reload, "volumeAdjustment"))

    # Train network
    if args.mode == "train":

        hetsiren.train()

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "HetSIREN_metrics"))

        # Prepare data loader
        data_loader = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=True, preShuffle=True,
                                                  mmap=mmap, mmap_output_dir=mmap_output_dir)

        # Example of training data for Tensorboard
        if hetsiren.isTomoSIREN:
            (x_example, _), labels_example = next(iter(data_loader))
        else:
            x_example, labels_example = next(iter(data_loader))
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Training data batch", x_example, dataformats="NHWC")

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

        if args.vol is not None:
            # Optimizers (Volume Adjustment)
            optimizer_vol = nnx.Optimizer(volumeAdjustment, optax.adam(1e-5))
            graphdef, state = nnx.split((volumeAdjustment, optimizer_vol))

            # Number epochs (volume adjustment)
            if len(generator.md) >= 10000:
                num_epochs_vol = 20
            else:
                num_epochs_vol = 200

            # Training loop (Volume Adjustment)
            print(f"{bcolors.OKCYAN}\n###### Training volume adjustment... ######")
            for i in range(num_epochs_vol):
                total_loss = 0

                # For progress bar (TQDM)
                step = 1
                print(f'\nTraining epoch {i + 1}/{num_epochs_vol} |')
                pbar = tqdm(data_loader, desc=f"Epoch {i + 1}/{num_epochs_vol}", file=sys.stdout, ascii=" >=",
                            colour="green")

                for (x, labels) in pbar:
                    if isinstance(x, tuple):
                        x = x[0]

                    loss, state = train_step_volume_adjustment(graphdef, state, x, labels, md_columns, args.sr,
                                                               args.ctf_type, vol.shape[0])
                    total_loss += loss

                    # Progress bar update  (TQDM)
                    pbar.set_postfix_str(f"loss={total_loss / step:.5f}")

                    # Summary writer (training loss)
                    if step % int(np.ceil(0.1 * len(data_loader))) == 0:
                        writer.add_scalar('Training loss (volume adjustment)',
                                          total_loss / step,
                                          i * len(data_loader) + step)

                    step += 1

            volumeAdjustment, optimizer_vol = nnx.merge(graphdef, state)
            values = volumeAdjustment()

            # Place values on grid and replace HetSIREN reference volume
            grid = jnp.zeros_like(vol)
            grid = grid.at[inds[..., 0], inds[..., 1], inds[..., 2]].set(values)
            hetsiren.reference_volume = grid
            hetsiren.delta_volume_decoder.reference_values = values

        # Optimizers (HetSIREN)
        optimizer = nnx.Optimizer(hetsiren, optax.adam(args.learning_rate))
        graphdef, state = nnx.split((hetsiren, optimizer))

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

        # Training loop (HetSIREN)
        print(f"{bcolors.OKCYAN}\n###### Training variability... ######")
        for i in range(args.epochs):
            total_loss = 0
            total_recon_loss = 0

            # For progress bar (TQDM)
            step = 1
            print(f'\nTraining epoch {i + 1}/{args.epochs} |')
            pbar = tqdm(data_loader, desc=f"Epoch {i + 1}/{args.epochs}", file=sys.stdout, ascii=" >=", colour="green")

            for (x, labels) in pbar:
                loss, recon_loss, state = train_step_hetsiren(graphdef, state, x, labels, md_columns)
                total_loss += loss
                total_recon_loss += recon_loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | recon_loss={total_recon_loss / step:.5f}")

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * len(data_loader))) == 0:
                    writer.add_scalar('Training loss (HetSIREN)',
                                      total_loss / step,
                                      i * len(data_loader) + step)

                    writer.add_scalar('Reconstruction loss (HetSIREN)',
                                      total_recon_loss / step,
                                      i * len(data_loader) + step)
                step += 1

            # Log intermediate results at the end of the epoch
            # Get first 5 images from batch
            if hetsiren.isTomoSIREN:
                x_for_tb = x[0][:5]
            else:
                x_for_tb = x[:5]
            labels_for_tb = labels[:5]

            # Decode some images and show them in Tensorboard
            x_pred_intermediate, latents_intermediate = hetsiren_decode_image(graphdef, state, x_for_tb,
                                                                              labels_for_tb, md_columns,
                                                                              ctf_type=args.ctf_type, return_latent=True,
                                                                              corrupt_projection_with_ctf=True)
            x_pred_intermediate = jax.vmap(min_max_scale)(x_pred_intermediate[..., None])
            writer.add_images("Predicted images batch", x_pred_intermediate, dataformats="NHWC")

            # Decode some states and show them in Tensorboard
            volumes_intermediate = hetsiren_decode_volume(graphdef, state, latents_intermediate)
            writer.add_volumes_slices(volumes_intermediate)

            # Log landscape stored in memory bank
            if i % 5 == 0:
                hetsiren_intermediate, _ = nnx.merge(graphdef, state)
                random_indices = jnr.choice(hetsiren_intermediate.choice_key, a=jnp.arange(hetsiren_intermediate.bank_size), shape=(hetsiren_intermediate.subset_size,), replace=False)
                latents_intermediate = hetsiren_intermediate.memory_bank.value[random_indices]
                latents_data_loader = NumpyGenerator(latents_intermediate).return_tf_dataset(preShuffle=False, shuffle=False, batch_size=args.batch_size)
                latents_images = []
                for (latents, _) in latents_data_loader:
                    random_labels = jnp.asarray(np.random.randint(low=0, high=len(generator.md), size=(latents.shape[0],)), dtype=jnp.int32)
                    x_pred_intermediate = hetsiren_decode_image(graphdef, state, latents, random_labels,
                                                                md_columns, ctf_type=None, return_latent=False,
                                                                corrupt_projection_with_ctf=False)
                    latents_images.append(np.asarray(x_pred_intermediate))
                latents_images = np.concatenate(latents_images, axis=0)
                latent_images_min = latents_images.min(axis=(1, 2), keepdims=True)
                latent_images_max = latents_images.max(axis=(1, 2), keepdims=True)
                latents_images = (latents_images - latent_images_min) / (latent_images_max - latent_images_min)
                latents_images = torch.from_numpy(latents_images)[:, None, ...]
                writer.add_embedding(latents_intermediate, label_img=latents_images, tag="HetSIREN latent space", global_step=i)

        hetsiren, optimizer = nnx.merge(graphdef, state)

        # Example of predicted data for Tensorboard
        x_pred_example = hetsiren_decode_image(graphdef, state, x_example, labels_example, md_columns, ctf_type=args.ctf_type, return_latent=False, corrupt_projection_with_ctf=True)
        x_pred_example = jax.vmap(min_max_scale)(x_pred_example[..., None])
        writer.add_images("Predicted images batch", x_pred_example, dataformats="NHWC")

        # Save model
        NeuralNetworkCheckpointer.save(hetsiren, os.path.join(args.output_path, "HetSIREN"), mode="pickle")
        if args.vol is not None:
            NeuralNetworkCheckpointer.save(volumeAdjustment, os.path.join(args.output_path, "volumeAdjustment"), mode="pickle")

    elif args.mode == "predict":

        hetsiren.eval()

        # Prepare data loader
        data_loader = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=False, preShuffle=False,
                                                  mmap=mmap, mmap_output_dir=mmap_output_dir)

        # Jitted prediction function
        predict_fn = jax.jit(hetsiren.__call__)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting HetSIREN latents... ######")
        latents = []
        # For progress bar (TQDM)
        pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=",
                    colour="green")

        for (x, labels) in pbar:
            if isinstance(x, tuple):
                x = x[0]

            # Wiener filter if precorrect CTF mode
            if args.ctf_type == "precorrect":
                defocusU = md_columns["ctfDefocusU"][labels]
                defocusV = md_columns["ctfDefocusV"][labels]
                defocusAngle = md_columns["ctfDefocusAngle"][labels]
                cs = md_columns["ctfSphericalAberration"][labels]
                kv = md_columns["ctfVoltage"][labels][0]
                ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                                 args.sr, [2 * hetsiren.xsize, int(2 * 0.5 * hetsiren.xsize + 1)],
                                 x.shape[0], True)
                x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

            latents.append(predict_fn(x))
        latents = np.asarray(jnp.concatenate(latents, axis=0))

        # Save latents in metadata
        md = generator.md
        md[:, 'latent_space'] = np.asarray([",".join(np.char.mod('%f', item)) for item in latents])
        md.write(os.path.join(args.output_path, "predicted_latents" + os.path.splitext(args.md)[1]))

    # If exists, clean MMAP
    if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
        shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

if __name__ == "__main__":
    main()
