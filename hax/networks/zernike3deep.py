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
from hax.networks import ImageAdjustment


class Encoder(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, architecture="convnn", isVae=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 32  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 4))
        self.architecture = architecture
        self.isVae = isVae
        self.normal_key = rngs.distributions()

        if self.architecture == "mlpnn":
            self.hidden_layers = [nnx.Linear(self.input_dim * self.input_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
            for _ in range(n_layers):
                self.hidden_layers.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers.append(nnx.Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16))
            for _ in range(2):
                self.hidden_layers.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
            self.latent = nnx.Linear(256, lat_dim, rngs=rngs)

        elif self.architecture == "convnn":
            self.hidden_layers_conv = [nnx.Linear(self.input_dim * self.input_dim, self.input_conv_dim * self.input_conv_dim, rngs=rngs, dtype=jnp.bfloat16)]
            self.hidden_layers_conv.append(nnx.Conv(1, 4, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(nnx.Conv(4, 8, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(nnx.Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(nnx.Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(nnx.Conv(8, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(nnx.Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(nnx.Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv.append(nnx.Conv(16, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

            self.hidden_layers_linear = [nnx.Linear(16 * self.out_conv_dim * self.out_conv_dim, 256, rngs=rngs, dtype=jnp.bfloat16)]
            for _ in range(3):
                self.hidden_layers_linear.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

            self.hidden_layers_linear.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
            for _ in range(2):
                self.hidden_layers_linear.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

            if isVae:
                self.mean_x = nnx.Linear(256, lat_dim, rngs=rngs)
                self.logstd_x = nnx.Linear(256, lat_dim, rngs=rngs)
            else:
                self.latent = nnx.Linear(256, lat_dim, rngs=rngs)

        else:
            raise ValueError("Architecture not supported. Implemented architectures are: mlpnn / convnn")

    def sample_gaussian(self, mean, logstd):
        return logstd * jnr.normal(self.normal_key, shape=mean.shape) + mean

    def __call__(self, x, return_last=False):
        if self.architecture == "mlpnn":
            x = rearrange(x, 'b h w c -> b (h w c)')

            for layer in self.hidden_layers:
                if layer.in_features != layer.out_features:
                    x = nnx.relu(layer(x))
                else:
                    x = nnx.relu(x + layer(x))

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

class MultiEncoder(nnx.Module):
    def __init__(self, input_dim, lat_dim=10, n_layers=3, isVae=False, *, rngs: nnx.Rngs):
        self.encoders = {"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, rngs=rngs),
                         "encoder_dec": Encoder(input_dim, lat_dim, n_layers=n_layers, rngs=rngs)}
        self.normal_key = rngs.distributions()
        self.isVae = isVae
        if isVae:
            self.mean_x = nnx.Linear(256, lat_dim, rngs=rngs)
            self.logstd_x = nnx.Linear(256, lat_dim, rngs=rngs)
        else:
            self.latent = nnx.Linear(256, lat_dim, rngs=rngs)

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


class FlowDecoder(nnx.Module):
    def __init__(self, latent_dim, total_voxels, coords, factor, choice_key, diff_geo=False, second_derivative=False, L1=7, L2=7,
                 *, rngs: nnx.Rngs):
        self.coords = coords
        self.factor = factor
        self.total_voxels = total_voxels
        self.diff_geo = diff_geo
        self.second_derivative = second_derivative
        self.choice_key = choice_key

        # Precompute Zernike3D basis
        self.zernike_degrees = basisDegreeVectors(L1, L2)
        self.sph_coeffs = precomputePolynomialsSph(L2)
        self.zernike_coeffs = precomputePolynomialsZernike(L2, L1)

        # Coefficients layers
        self.hidden_layers_coeff = [Linear(latent_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(3):
            self.hidden_layers_coeff.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.latent_x = Linear(1024, len(self.zernike_degrees), rngs=rngs, kernel_init=jax.nn.initializers.uniform(0.0001))
        self.latent_y = Linear(1024, len(self.zernike_degrees), rngs=rngs, kernel_init=jax.nn.initializers.uniform(0.0001))
        self.latent_z = Linear(1024, len(self.zernike_degrees), rngs=rngs, kernel_init=jax.nn.initializers.uniform(0.0001))

    def decode_coefficients(self, x):
        x = nnx.relu(self.hidden_layers_coeff[0](x))
        for layer in self.hidden_layers_coeff[1:]:
            x = nnx.relu(x + layer(x))
        return self.latent_x(x), self.latent_y(x), self.latent_z(x)

    def __call__(self, latent, inds, xsize):
        factor = 0.5 * xsize
        coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1) - factor

        # Decode coefficients
        latent_x, latent_y, latent_z = self.decode_coefficients(latent)

        # Recover flow from basis
        Z = computeBasis(coords, degrees=self.zernike_degrees, r=factor, groups=None, centers=None,
                         sph_coeffs=self.sph_coeffs, zernike_coeffs=self.zernike_coeffs)
        d_x = jnp.matmul(latent_x, Z)
        d_y = jnp.matmul(latent_y, Z)
        d_z = jnp.matmul(latent_z, Z)
        flow = factor * jnp.stack([d_x, d_y, d_z], axis=-1)
        # flow = jnp.stack([d_x, d_y, d_z], axis=-1)

        #### Differential geometry losses ####
        if self.diff_geo:
            def compute_gradients(coords):
                Z = computeBasis(coords, degrees=self.zernike_degrees, r=factor, groups=None, centers=None,
                                 sph_coeffs=self.sph_coeffs, zernike_coeffs=self.zernike_coeffs)
                d_x = jnp.matmul(latent_x, Z)
                d_y = jnp.matmul(latent_y, Z)
                d_z = jnp.matmul(latent_z, Z)
                return jnp.stack([d_x, d_y, d_z], axis=-1)

            coords_subset = jnr.choice(self.choice_key, coords, shape=(1000,), replace=False)  # Is it better to use shape=(10000,)?
            jacobian_for_single_point = jax.jacfwd(compute_gradients)
            batched_jacobian_fn = jax.vmap(jacobian_for_single_point)
            # jacobians = batched_jacobian_fn(coords_subset / factor)
            jacobians = batched_jacobian_fn(coords_subset / factor)

            jacobians = jnp.eye(3, dtype=jnp.float32)[None, None, ...] + jacobians
            jac_loss = jnp.abs(jnp.linalg.det(jacobians) - 1.)
            jac_loss = jnp.mean(jac_loss)

            if self.second_derivative:
                second_derivative_fn_single = jax.jacfwd(jacobian_for_single_point)
                batched_second_derivative_fn = jax.vmap(second_derivative_fn_single)
                hessians = batched_second_derivative_fn(coords_subset)

                # Beding energy regularization
                dx_xyz = hessians[..., 0, :]
                dy_xyz = hessians[..., 1, :]
                dz_xyz = hessians[..., 2, :]

                dx_xyz = jnp.square(dx_xyz)
                dy_xyz = jnp.square(dy_xyz)
                dz_xyz = jnp.square(dz_xyz)

                be_loss = jnp.mean(dx_xyz[:, :, :, 0]) + jnp.mean(dy_xyz[:, :, :, 1]) + jnp.mean(dz_xyz[:, :, :, 2])
                be_loss +=  2. * jnp.mean(dx_xyz[:, :, :, 1]) + 2. * jnp.mean(dx_xyz[:, :, :, 2]) + jnp.mean(dy_xyz[:, :, :, 2])
            else:
                be_loss = 0.0
        else:
            jac_loss = 0.0
            be_loss = 0.0

        return flow, 0.001 * jac_loss, 0.001 * be_loss, 0.0001 * jnp.sqrt((jnp.square(latent_x) + jnp.square(latent_y) + jnp.square(latent_z)).sum())


class PhysDecoder(nnx.Module):
    def __init__(self, xsize, lat_dim=10, *, rngs: nnx.Rngs):
        self.xsize = xsize

        # Gray level adjustment
        self.imageAdjustment = ImageAdjustment(lat_dim=lat_dim, xsize=xsize, predict_value=False, rngs=rngs)

    def __call__(self, flow, x, inds, values, xsize, rotations, shifts, ctf, ctf_type):
        # Indices to coords
        factor = 0.5 * xsize
        coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)[None, ...]
        coords = coords - factor

        # Apply field
        coords = coords + flow

        # Rotate grid
        if rotations.ndim == 2:
            rotations = euler_matrix_batch(rotations[:, 0], rotations[:, 1], rotations[:, 2])
        coords =  jnp.matmul(coords, rearrange(rotations, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts[:, None, :] + factor

        # Scatter image
        B = flow.shape[0]
        c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
        images = jnp.zeros((B, xsize, xsize), dtype=flow.dtype)

        bamp = values[None, ...]

        bposf = jnp.floor(c_sampling)
        bposi = bposf.astype(jnp.int32)
        bposf = c_sampling - bposf

        bamp0 = bamp * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp1 = bamp * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp2 = bamp * (bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp3 = bamp * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp = jnp.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposi = jnp.concat([bposi, bposi + jnp.array((1, 0)), bposi + jnp.array((1, 1)), bposi + jnp.array((0, 1))], axis=1)

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        images = jax.vmap(scatter_img)(images, bposi, bamp)

        # Gaussian filter (needed by forward interpolation)
        images = dm_pix.gaussian_blur(images[..., None], 1.0, kernel_size=3)[..., 0]

        # Gray level adjustment
        x_nc = jnp.squeeze(x)
        if x_nc.ndim == 2:
            x_nc = x_nc[None, ...]
        a, b = self.imageAdjustment(x_nc)
        if a.ndim == 1:
            images = a[:, None, None] * images + b[:, None, None]
        else:
            images = a * images + b

        # Apply CTF
        if ctf_type in ["apply", "wiener", "squared"]:
            images = ctfFilter(images, ctf, pad_factor=2)

        return images

class Zernike3Deep(nnx.Module):
    def __init__(self, lat_dim, total_voxels, inds, values, xsize, sr, bank_size=10000, ctf_type="apply", diff_geo=False,
                 second_derivative=False, decoupling=False, isVae=False, L1=7, L2=7, *, rngs: nnx.Rngs):
        super(Zernike3Deep, self).__init__()
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.inds = jnp.array(inds)
        self.coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1) - 0.5 * self.xsize
        self.values = jnp.array(values)
        self.decoupling = decoupling
        self.isVae = isVae
        self.encoder = MultiEncoder(self.xsize, lat_dim, n_layers=3, isVae=isVae, rngs=rngs) if decoupling else Encoder(self.xsize, lat_dim, isVae=isVae, rngs=rngs)
        self.flow_decoder = FlowDecoder(lat_dim, total_voxels, self.coords, 0.5 * self.xsize, diff_geo=diff_geo,
                                        second_derivative=second_derivative, choice_key=rngs.choice(), L1=L1, L2=L2, rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize, lat_dim=lat_dim, rngs=rngs)

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
        # Precompute batch aligments
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
                x = wiener2DFilter(x[..., 0], ctf)[..., None]

            # Encode images
            latent = self(x)
        else:
            latent = x

        # Decode flow field
        flow, _, _, _ = self.flow_decoder(latent, self.inds, self.xsize)

        # Check if CTF corruption is needed
        if not corrupt_projection_with_ctf:
            ctf_type = None

        # Generate projections
        images_corrected = self.phys_decoder(flow, x, self.inds, self.values, self.xsize, euler_angles, shifts, ctf, ctf_type)

        if return_latent:
            return images_corrected, latent
        else:
            return images_corrected

    def decode_volume(self, x):
        if x.ndim == 1:
            x = x[None, ...]

        # Get deformation field
        field = self.flow_decoder(x, self.inds, self.xsize)[0]

        # Place values on grids
        grids = jnp.zeros((x.shape[0], self.xsize, self.xsize, self.xsize))
        for idx in range(x.shape[0]):
            c_x = jnp.round(self.inds[..., 2] + field[idx, ..., 0]).astype(jnp.int32)
            c_y = jnp.round(self.inds[..., 1] + field[idx, ..., 1]).astype(jnp.int32)
            c_z = jnp.round(self.inds[..., 0] + field[idx, ..., 2]).astype(jnp.int32)
            grids = grids.at[idx, c_z, c_y, c_x].add(self.values)

        # Low pass filter
        grids = jax.vmap(low_pass_3d)(grids)

        return grids

@jax.jit
def train_step_zernike3deep(graphdef, state, x, labels, md):
    model, optimizer, optimizer_grays = nnx.merge(graphdef, state)
    distributions_key = jnr.PRNGKey(random.randint(0, 2 ** 32 - 1))

    def loss_fn_ot(model, x):
        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (sample, latent, logstd), prev_layer_out = model.encoder(x, "encoder_exp", return_last=True)
            else:
                sample, latent, logstd = model.encoder(x)
        else:
            if model.decoupling:
                latent, prev_layer_out = model.encoder(x, "encoder_exp", return_last=True)
            else:
                latent = model.encoder(x)

        # Decode flow field
        if model.isVae:
            flow, jac_loss, be_loss, coefficient_loss = model.flow_decoder(sample, model.inds, model.xsize)
        else:
            flow, jac_loss, be_loss, coefficient_loss = model.flow_decoder(latent, model.inds, model.xsize)

        # Generate projections
        images_corrected = model.phys_decoder(flow, x, model.inds, model.values, model.xsize, euler_angles, shifts, ctf, model.ctf_type)

        # Losses
        images_corrected = jnp.squeeze(images_corrected)
        x = jnp.squeeze(x)

        # Consider CTF if Wiener or Squared mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter(x, ctf, pad_factor=2)
            images_corrected_loss = wiener2DFilter(images_corrected, ctf, pad_factor=2)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter(x, ctf, pad_factor=2)
            images_corrected_loss = ctfFilter(images_corrected, ctf, pad_factor=2)
        else:
            x_loss = x
            images_corrected_loss = images_corrected

        # # recon_loss = dm_pix.mae(images_corrected[..., None], x_corrected[..., None]).mean()
        recon_loss = dm_pix.mse(images_corrected_loss[..., None], x_loss[..., None]).mean()

        # Field norm loss
        field_norm_loss = jnp.sqrt(jnp.square(flow).sum(axis=-1)).mean() / model.xsize

        if model.isVae:
            # KL divergence loss
            kl_loss = -0.5 * jnp.sum(1 + 2 * logstd - jnp.square(jnp.exp(logstd)) - jnp.square(latent))
        else:
            kl_loss = 0.0

        # Decoupling
        if model.decoupling:
            images_random = model.phys_decoder(flow, x, model.inds, model.values, model.xsize, rotations_random,
                                               shifts, ctf_random, model.ctf_type)
            if model.isVae:
                (_, latent_1, _), prev_layer_out_random = model.encoder(images_corrected[..., None], "encoder_dec",
                                                                       return_last=True)
                (_, latent_2, _) = model.encoder(images_random[..., None], "encoder_dec")
            else:
                latent_1, prev_layer_out_random = model.encoder(images_corrected[..., None], "encoder_dec",
                                                               return_last=True)
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

            # decoupling_loss += 1.0 * contrastive_ce_loss(dist_nn, dist_fn, reduction="mean", temperature=0.001)
            decoupling_loss += 1.0 * triplet_loss(dist_nn, dist_fn, reduction="mean", margin=0.01)

        else:
            decoupling_loss = 0.0

        loss = recon_loss + 0.0001 * kl_loss + (jac_loss + be_loss) + 0.001 * decoupling_loss + 1e-4 * field_norm_loss
        return loss, latent

    params = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('flow_decoder')))
    params_grays = nnx.All(nnx.Param, nnx.PathContains('phys_decoder'))

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

    grad_fn_ot = nnx.value_and_grad(loss_fn_ot, argnums=nnx.DiffState(0, (params, params_grays)), has_aux=True)
    (loss, latent), grads_combined = grad_fn_ot(model, x)

    grads, grads_gray = grads_combined.split(params, params_grays)

    optimizer.update(grads)
    optimizer_grays.update(grads_gray)

    # Update memory bank
    model.enqueue(latent)

    state = nnx.state((model, optimizer, optimizer_grays))

    return loss, state


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
    from hax.networks import train_step_zernike3deep, train_step_volume_adjustment, VolumeAdjustment
    from hax.metrics import JaxSummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, type=str,
                        help="Xmipp metadata file with the images (+ alignments / CTF) to be analyzed")
    parser.add_argument("--vol", required=True, type=str,
                        help="Volume to be warped toward the images")
    parser.add_argument("--mask", required=False, type=str,
                        help="Binary mask computed from volume enclosing all the mass to be moved. If not provided, the volume will be automatically masked for you")
    parser.add_argument("--load_images_to_ram", action='store_true',
                        help=f"If provided, images will be loaded to RAM. This is recommended if you want the best performance and your dataset fits in your RAM memory. If this flag is not provided, "
                             f"images will be memory mapped. When this happens, the program will trade disk space for performance. Thus, during the execution additional disk space will be used and the performance "
                             f"will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal once the execution has finished.")
    parser.add_argument("--sr", required=True, type=float,
                        help="Sampling rate of the images/volume")
    parser.add_argument("--ctf_type", required=True, type=str, choices=["None", "apply", "wiener", "precorrect"],
                        help="Determines whether to consider the CTF and, in case it is considered, whether it will be applied to the projections (apply) or used to correct the metadata images (wiener - precorrect)")
    parser.add_argument("--lat_dim", required=False, type=int, default=8,
                        help="Dimensionality of the latent space of the network (set by default to 8)")
    parser.add_argument("--L1", required=False, type=int, default=7,
                        help="Degree of Zernike3D radial component (increasing this value might help finding more localized motions at the expense of higher memory consumption)")
    parser.add_argument("--L2", required=False, type=int, default=7,
                        help="Degree of Zernike3D angular component (increasing this value might help finding more localized motions at the expense of higher memory consumption)")
    parser.add_argument("--mode", required=True, type=str, choices=["train", "predict", "send_to_pickle"],
                        help=f"{bcolors.BOLD}train{bcolors.ENDC}: train a neural network from scratch or from a previous execution if reload is provided\n"
                             f"{bcolors.BOLD}predict{bcolors.ENDC}: predict the latent vectors from the input images ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case)\n"
                             f"{bcolors.BOLD}send_to_pickle{bcolors.ENDC}: save the network in pickle format. ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case - "
                             f"needed by program {bcolors.UNDERLINE}estimate_latent_covariances{bcolors.ENDC})")
    parser.add_argument("--epochs", required=False, type=int, default=50,
                        help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset of images - set to default to 50 - "
                             "as a rule of thumb, consider 50 to 100 epochs enough for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")
    parser.add_argument("--batch_size", required=False, type=int, default=8,
                        help="Determines how many images will be load in the GPU at any moment during training (set by default to 8 - "
                             f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                             f"to monitor and/or measure memory usage and adjust this value - keep also in mind that bigger batch sizes might be less precise when looking for very local motions")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, new metadata...)")
    parser.add_argument("--reload", required=False, type=str,
                        help=f"Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data - "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: Since Zernike3Deep also learns a gray level adjustment, reload must be the path to a folder containing two additional "
                             f"folders called: {bcolors.UNDERLINE}Zernike3Deep{bcolors.ENDC} and {bcolors.UNDERLINE}volumeAdjustment{bcolors.ENDC})")
    args = parser.parse_args()

    # Preprocess volume (and mask)
    vol = ImageHandler(args.vol).getData()

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    else:
        mask = ImageHandler(args.vol).generateMask(boxsize=64)

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

    inds = np.asarray(np.where(mask > 0.0)).T
    values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]

    factor = 0.5 * vol.shape[0]
    coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
    coords = (coords - factor) / factor

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key, choice_key = jax.random.split(rng, 3)

    # Prepare network (Zernike3Deep)
    zernike3deep = Zernike3Deep(args.lat_dim, inds.shape[0], inds, values, vol.shape[0], args.sr,
                                ctf_type=args.ctf_type, diff_geo=True, second_derivative=True, decoupling=True, isVae=True,
                                L1=args.L1, L2=args.L2, bank_size=len(generator.md), rngs=nnx.Rngs(model_key, choice=choice_key))

    # Prepare network (Volume Adjustment)
    volumeAdjustment = VolumeAdjustment(lat_dim=3, coords=coords, values=values, predicts_value=True, rngs=nnx.Rngs(model_key))


    # Reload network
    if args.reload is not None:
        zernike3deep = NeuralNetworkCheckpointer.load(zernike3deep, os.path.join(args.reload, "Zernike3Deep"))
        volumeAdjustment = NeuralNetworkCheckpointer.load(volumeAdjustment, os.path.join(args.reload, "volumeAdjustment"))

    # Train network
    if args.mode == "train":

        zernike3deep.train()
        volumeAdjustment.train()

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "Zernike3Deep_metrics"))

        # Prepare data loader
        data_loader = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=True, preShuffle=True,
                                                  mmap=mmap, mmap_output_dir=mmap_output_dir)

        # Example of training data for Tensorboard
        x_example, labels_example = next(iter(data_loader))
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Training data batch", x_example, dataformats="NHWC")

        # Projector help text in Tensorboard
        legend_projector = """
                <h3>WARNING: Images shown in projector</h3>
                <ul>
                    <li>The pose of the images shown in the projector is random and not related to the real pose of your data. 
                    Therefore, DO NOT consider this images as a representation on how poses are classified in the latent space.</li>
                </ul>
                """
        writer.add_text("Projector warning", legend_projector)

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
                loss, state = train_step_volume_adjustment(graphdef, state, x, labels, md_columns, args.sr, args.ctf_type, vol.shape[0])
                total_loss += loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f}")

                # Summary writer (training loss)
                if step % int(0.1 * len(data_loader)) == 0:
                    writer.add_scalar('Training loss (volume adjustment)',
                                      total_loss / step,
                                      i * len(data_loader) + step)

                step += 1

        volumeAdjustment, optimizer_vol = nnx.merge(graphdef, state)
        zernike3deep.values = volumeAdjustment()

        # Optimizers (Zernike3Deep)
        params = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('flow_decoder')))
        params_grays = nnx.All(nnx.Param, nnx.PathContains('phys_decoder'))
        optimizer = nnx.Optimizer(zernike3deep, optax.adam(1e-5), wrt=params)
        optimizer_grays = nnx.Optimizer(zernike3deep, optax.adam(1e-5), wrt=params_grays)
        graphdef, state = nnx.split((zernike3deep, optimizer, optimizer_grays))

        # Jitted functions to improve performance
        @partial(jax.jit, static_argnames=["ctf_type", "return_latent", "corrupt_projection_with_ctf"])
        def zernike3deep_decode_image(graphdef, state, x, labels, md, ctf_type=None, return_latent=False,
                                  corrupt_projection_with_ctf=False):
            model, _ = nnx.merge(graphdef, state)
            return model.decode_image(x, labels, md, ctf_type=ctf_type, return_latent=return_latent,
                                      corrupt_projection_with_ctf=corrupt_projection_with_ctf)

        @jax.jit
        def zernike3deep_decode_volume(graphdef, state, x):
            model, _ = nnx.merge(graphdef, state)
            return model.decode_volume(x)

        # Training loop (Zernike3Deep)
        print(f"{bcolors.OKCYAN}\n###### Training variability... ######")
        for i in range(args.epochs):
            total_loss = 0

            # For progress bar (TQDM)
            step = 1
            print(f'\nTraining epoch {i + 1}/{args.epochs} |')
            pbar = tqdm(data_loader, desc=f"Epoch {i + 1}/{args.epochs}", file=sys.stdout, ascii=" >=", colour="green")

            for (x, labels) in pbar:
                loss, state = train_step_zernike3deep(graphdef, state, x, labels, md_columns)
                total_loss += loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f}")

                # Summary writer (training loss)
                if step % int(0.1 * len(data_loader)) == 0:
                    zernike3deep_intermediate, _, _ = nnx.merge(graphdef, state)

                    writer.add_scalar('Training loss (Zernike3Deep)',
                                      total_loss / step,
                                      i * len(data_loader) + step)
                step += 1

            # Log intermediate results at the end of the epoch
            # Get first 5 images from batch
            x_for_tb = x[:5]
            labels_for_tb = labels[:5]

            # Decode some images and show them in Tensorboard
            x_pred_intermediate, latents_intermediate = zernike3deep_decode_image(graphdef, state, x_for_tb,
                                                                                  labels_for_tb, md_columns,
                                                                                  ctf_type=args.ctf_type, return_latent=True,
                                                                                  corrupt_projection_with_ctf=True)
            x_pred_intermediate = jax.vmap(min_max_scale)(x_pred_intermediate[..., None])
            writer.add_images("Predicted images batch", x_pred_intermediate, dataformats="NHWC")

            # Decode some states and show them in Tensorboard
            volumes_intermediate = zernike3deep_decode_volume(latents_intermediate)
            writer.add_volumes_slices(volumes_intermediate)

            # Log landscape stored in memory bank
            if i % 5 == 0:
                hetsiren_intermediate, _ = nnx.merge(graphdef, state)
                random_indices = jnr.choice(hetsiren_intermediate.choice_key,
                                            a=jnp.arange(hetsiren_intermediate.bank_size),
                                            shape=(hetsiren_intermediate.subset_size,), replace=False)
                latents_intermediate = hetsiren_intermediate.memory_bank.value[random_indices]
                latents_data_loader = NumpyGenerator(latents_intermediate).return_tf_dataset(preShuffle=False,
                                                                                             shuffle=False,
                                                                                             batch_size=args.batch_size)
                latents_images = []
                for (latents, _) in latents_data_loader:
                    random_labels = jnp.asarray(np.random.randint(low=0, high=len(generator.md), size=(latents.shape[0],)), dtype=jnp.int32)
                    x_pred_intermediate = zernike3deep_decode_image(graphdef, state, latents, random_labels,
                                                                    md_columns, ctf_type=None, return_latent=False,
                                                                    corrupt_projection_with_ctf=False)
                    latents_images.append(np.asarray(x_pred_intermediate))
                latents_images = np.concatenate(latents_images, axis=0)
                latent_images_min = latents_images.min(axis=(1, 2), keepdims=True)
                latent_images_max = latents_images.max(axis=(1, 2), keepdims=True)
                latents_images = (latents_images - latent_images_min) / (latent_images_max - latent_images_min)
                latents_images = torch.from_numpy(latents_images)[:, None, ...]
                writer.add_embedding(latents_intermediate, label_img=latents_images, tag="Zernike3Deep latent space", global_step=i)

        zernike3deep, optimizer, optimizer_grays = nnx.merge(graphdef, state)

        # Example of predicted data for Tensorboard
        x_pred_example = zernike3deep_decode_image(graphdef, state, x_example, labels_example, md_columns, ctf_type=args.ctf_type, return_latent=False, corrupt_projection_with_ctf=True)
        x_pred_example = jax.vmap(min_max_scale)(x_pred_example[..., None])
        writer.add_images("Predicted images batch", x_pred_example, dataformats="NHWC")

        # Save model
        NeuralNetworkCheckpointer.save(volumeAdjustment, os.path.join(args.output_path, "volumeAdjustment"))
        NeuralNetworkCheckpointer.save(zernike3deep, os.path.join(args.output_path, "Zernike3Deep"))

    elif args.mode == "predict":

        zernike3deep.eval()
        volumeAdjustment.eval()

        # Update zernike3deep values
        zernike3deep.values = volumeAdjustment()

        # Prepare data loader
        data_loader = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=False, preShuffle=False,
                                                  mmap=mmap, mmap_output_dir=mmap_output_dir)

        # Jitted prediction function
        predict_fn = jax.jit(zernike3deep.__call__)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting Zernike3Deep latents... ######")
        latents = []
        for i in range(args.epochs):
            # For progress bar (TQDM)
            pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=",
                        colour="green")

            for (x, labels) in pbar:

                # Wiener filter if precorrect CTF mode
                if args.ctf_type == "precorrect":
                    defocusU = md_columns["ctfDefocusU"][labels]
                    defocusV = md_columns["ctfDefocusV"][labels]
                    defocusAngle = md_columns["ctfDefocusAngle"][labels]
                    cs = md_columns["ctfSphericalAberration"][labels]
                    kv = md_columns["ctfVoltage"][labels][0]
                    ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                                     args.sr, [2 * zernike3deep.xsize, int(2 * 0.5 * zernike3deep.xsize + 1)],
                                     x.shape[0], True)
                    x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

                latents.append(predict_fn(x))
        latents = np.asarray(latents)

        # Save latents in metadata
        md = generator.md
        md[:, 'latent_space'] = np.asarray([",".join(item) for item in latents.astype(str)])
        md.write(os.path.join(args.output_path, "predicted_latents.xmd"))

    elif args.mode == "send_to_pickle":

        # Save mode to pickle
        NeuralNetworkCheckpointer.save(zernike3deep, os.path.join(args.output_path, "Zernike3Deep"), mode="pickle")

    # If exists, clean MMAP
    if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
        shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

if __name__ == "__main__":
    main()
