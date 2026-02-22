#!/usr/bin/env python


from functools import partial

import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import dm_pix

from einops import rearrange

from hax.utils import *
from hax.layers import *
from hax.networks import ImageAdjustment


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
            hidden_layers = [nnx.Linear(self.input_dim * self.input_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
            for _ in range(n_layers):
                hidden_layers.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers.append(nnx.Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16))
            for _ in range(2):
                hidden_layers.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers = nnx.List(hidden_layers)
            self.latent = nnx.Linear(256, lat_dim, rngs=rngs)

        elif self.architecture == "convnn":
            hidden_layers_conv = [nnx.Linear(self.input_dim * self.input_dim, self.input_conv_dim * self.input_conv_dim, rngs=rngs, dtype=jnp.bfloat16)]
            hidden_layers_conv.append(nnx.Conv(1, 4, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(nnx.Conv(4, 8, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(nnx.Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(nnx.Conv(8, 8, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(nnx.Conv(8, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(nnx.Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(nnx.Conv(16, 16, kernel_size=(1, 1), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            hidden_layers_conv.append(nnx.Conv(16, 16, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
            self.hidden_layers_conv = nnx.List(hidden_layers_conv)

            hidden_layers_linear = [nnx.Linear(16 * self.out_conv_dim * self.out_conv_dim, 256, rngs=rngs, dtype=jnp.bfloat16)]
            for _ in range(3):
                hidden_layers_linear.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

            hidden_layers_linear.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
            for _ in range(2):
                hidden_layers_linear.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))

            self.hidden_layers_linear = nnx.List(hidden_layers_linear)

            if isVae:
                self.mean_x = nnx.Linear(256, lat_dim, rngs=rngs)
                self.logstd_x = nnx.Linear(256, lat_dim, rngs=rngs)
            else:
                self.latent = nnx.Linear(256, lat_dim, rngs=rngs)

        else:
            raise ValueError("Architecture not supported. Implemented architectures are: mlpnn / convnn")

    def sample_gaussian(self, mean, logstd, *, rngs):
        return logstd * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, return_last=False, *, rngs=None):
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
    def __init__(self, input_dim, lat_dim=10, n_layers=3, isVae=False, isTomo=False, *, rngs: nnx.Rngs):
        if isTomo:
            self.encoders = nnx.Dict({"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, rngs=rngs),
                                      "encoder_dec": EncoderTomo(100, lat_dim, n_layers=n_layers, rngs=rngs)})
        else:
            self.encoders = nnx.Dict({"encoder_exp": Encoder(input_dim, lat_dim, n_layers=3, rngs=rngs),
                                      "encoder_dec": Encoder(input_dim, lat_dim, n_layers=n_layers, rngs=rngs)})
        self.isVae = isVae
        if isVae:
            self.mean_x = nnx.Linear(256, lat_dim, rngs=rngs)
            self.logstd_x = nnx.Linear(256, lat_dim, rngs=rngs)
        else:
            self.latent = nnx.Linear(256, lat_dim, rngs=rngs)

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
        self.rigid_6d_rotation = nnx.Linear(256, 6, rngs=rngs)
        self.rigid_shifts = nnx.Linear(256, 3, rngs=rngs)

        # Refinement control (residual learning)
        self.alpha_rigid_rotations = nnx.Param(1e-4)
        self.alpha_rigid_shifts = nnx.Param(1e-4)

    def sample_gaussian(self, mean, logstd, *, rngs):
        return logstd * jnr.normal(rngs, shape=mean.shape) + mean

    def __call__(self, x, encoder_id="encoder_exp", return_last=False, return_alignment_refinement=False, *,
                 rngs=None):
        x = self.encoders[encoder_id](x, return_last=True)

        if return_alignment_refinement:
            x_ref = nnx.relu(x + self.hidden_layers_refinement[0](x))
            for layer in self.hidden_layers_refinement[1:]:
                x_ref = nnx.relu(x_ref + layer(x_ref))

            # Estimate rotations for volume registration
            rotations_6d = self.rigid_6d_rotation(x_ref)
            identity_6d = jnp.array([1., 0., 0., 0., 1., 0.])[None, ...].repeat(rotations_6d.shape[0], axis=0)
            rotations_6d = identity_6d + self.alpha_rigid_rotations * rotations_6d
            rotations_rigid = PoseDistMatrix.mode_rotmat(rotations_6d)

            # Estimate shifts for volume registration
            shifts_rigid = self.alpha_rigid_shifts * self.rigid_shifts(x_ref)

        for layer in self.hidden_layers_latent:
            x = nnx.relu(x + layer(x))

        if self.isVae:
            mean = self.mean_x(x)
            logstd = self.logstd_x(x)
            sample = self.sample_gaussian(mean, logstd, rngs=rngs) if rngs is not None else mean
            if return_last:
                if return_alignment_refinement:
                    return (sample, mean, logstd), (rotations_rigid, shifts_rigid), x
                else:
                    return (sample, mean, logstd), x
            else:
                if return_alignment_refinement:
                    return (sample, mean, logstd), (rotations_rigid, shifts_rigid)
                else:
                    return sample, mean, logstd
        else:
            latent = self.latent(x)
            if return_last:
                if return_alignment_refinement:
                    return latent, (rotations_rigid, shifts_rigid), x
                else:
                    return latent, x
            else:
                if return_alignment_refinement:
                    return latent, (rotations_rigid, shifts_rigid)
                else:
                    return latent


class FlowDecoder(nnx.Module):
    def __init__(self, latent_dim, total_voxels, coords, factor, L1=7, L2=7, *, rngs: nnx.Rngs):
        self.coords = coords
        self.factor = factor
        self.total_voxels = total_voxels

        # Precompute Zernike3D basis
        self.zernike_degrees = basisDegreeVectors(L1, L2)
        self.sph_coeffs = nnx.List(precomputePolynomialsSph(L2))
        self.zernike_coeffs = precomputePolynomialsZernike(L2, L1)

        # Graph from coordinates
        self.edge_index, self.edge_weights, self.consensus_distances, self.tau, _, _ = build_graph_from_coordinates(self.coords, k_spacing=4, k_knn=6, radius_factor=1.5)
        self.edge_weights = jnp.ones_like(self.edge_weights)

        # Coefficients layers
        hidden_layers_coeff = [Linear(latent_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(3):
            hidden_layers_coeff.append(Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_coeff = nnx.List(hidden_layers_coeff)
        self.latent_x = Linear(1024, len(self.zernike_degrees), rngs=rngs, kernel_init=jax.nn.initializers.normal(0.0001))
        self.latent_y = Linear(1024, len(self.zernike_degrees), rngs=rngs, kernel_init=jax.nn.initializers.normal(0.0001))
        self.latent_z = Linear(1024, len(self.zernike_degrees), rngs=rngs, kernel_init=jax.nn.initializers.normal(0.0001))

    def decode_coefficients(self, x):
        x = nnx.relu(self.hidden_layers_coeff[0](x))
        for layer in self.hidden_layers_coeff[1:]:
            x = nnx.relu(x + layer(x))
        return self.latent_x(x), self.latent_y(x), self.latent_z(x)

    def __call__(self, latent, coords, xsize):
        factor = 0.5 * xsize

        # Decode coefficients
        latent_x, latent_y, latent_z = self.decode_coefficients(latent)

        # Recover flow from basis
        Z = computeBasis(coords, degrees=self.zernike_degrees, r=1.0, groups=None, centers=None,
                         sph_coeffs=self.sph_coeffs, zernike_coeffs=self.zernike_coeffs)
        d_x = jnp.matmul(latent_x, Z)
        d_y = jnp.matmul(latent_y, Z)
        d_z = jnp.matmul(latent_z, Z)
        flow = factor * jnp.stack([d_x, d_y, d_z], axis=-1)
        # flow = jnp.stack([d_x, d_y, d_z], axis=-1)

        return flow, 0.0001 * jnp.sqrt((jnp.square(latent_x) + jnp.square(latent_y) + jnp.square(latent_z)).sum())


class PhysDecoder(nnx.Module):
    def __init__(self, xsize, lat_dim=10, *, rngs: nnx.Rngs):
        self.xsize = xsize

        # Gray level adjustment
        self.imageAdjustment = ImageAdjustment(lat_dim=lat_dim, xsize=xsize, predict_value=True, rngs=rngs)

    def __call__(self, flow, x, coords, values, xsize, rotations, shifts, ctf, ctf_type, sigma):
        # Indices to coords
        factor = 0.5 * xsize
        coords = coords[None, ...]
        coords = factor * coords

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
        images = dm_pix.gaussian_blur(images[..., None], sigma, kernel_size=9)[..., 0]

        # Apply CTF
        if ctf_type in ["apply", "wiener", "squared"]:
            images = ctfFilter(images, ctf, pad_factor=2)

        # Gray level adjustment
        x_nc = jnp.squeeze(x)
        if x_nc.ndim == 2:
            x_nc = x_nc[None, ...]
        a, b = self.imageAdjustment(x_nc)
        if a.ndim == 1:
            a, b = a[:, None, None], b[:, None, None]

        return images, (a, b)

class Zernike3Deep(nnx.Module):

    @save_config
    def __init__(self, lat_dim, coords, values, xsize, sr, bank_size=1024, ctf_type="apply",
                 sigma=1.0, decoupling=False, isVae=False, L1=7, L2=7, isTomo=False, *, rngs: nnx.Rngs):
        super(Zernike3Deep, self).__init__()
        factor = 0.5 * xsize
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.coords = (jnp.array(coords) - factor) / factor
        self.values = jnp.array(values)
        self.decoupling = decoupling if not isTomo else False
        self.isTomo = isTomo
        self.isVae = isVae
        self.sigma = nnx.Param(sigma)
        self.encoder = MultiEncoder(self.xsize, lat_dim, n_layers=3, isVae=isVae, rngs=rngs, isTomo=isTomo) if decoupling or isTomo else Encoder(self.xsize, lat_dim, isVae=isVae, rngs=rngs)
        self.flow_decoder = FlowDecoder(lat_dim, coords.shape[0], self.coords, factor, L1=L1, L2=L2, rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize, lat_dim=lat_dim, rngs=rngs)

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.subset_size = min(2048, bank_size)

        self.memory_bank = nnx.Variable(
            jnp.zeros((self.bank_size, lat_dim))
        )
        self.memory_bank_ptr = nnx.Variable(
            jnp.zeros((1,), dtype=jnp.int32)
        )

    def __call__(self, x, rngs=None, **kwargs):
        if self.isVae:
            if self.decoupling:
                (sample, mean, _), (rotations, shifts) = self.encoder(x, "encoder_exp", return_last=False, return_alignment_refinement=True, rngs=rngs)
            else:
                (sample, mean, _), (rotations, shifts) = self.encoder(x, return_last=False, return_alignment_refinement=True, rngs=rngs)
            if kwargs.pop("gaussian_sample", False):
                latent = sample
            else:
                latent = mean
        else:
            if self.decoupling:
                latent, (rotations, shifts) = self.encoder(x, "encoder_exp", return_last=False, return_alignment_refinement=True, rngs=rngs)
            else:
                latent, (rotations, shifts) = self.encoder(x, return_last=False, return_alignment_refinement=True, rngs=rngs)
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
            latent, (rotations_rigid, shifts_rigid) = self(x, return_alignment_refinement=True)
        else:
            latent = x

        # Get rotation matrices
        if euler_angles.ndim == 2:
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        else:
            rotations = euler_angles

        # Decode flow field
        flow, _ = self.flow_decoder(latent, self.coords, self.xsize)

        # Consider alignments if needed
        if x.ndim == 4:
            # Consider refinement and rigid registration alignments
            rotations = jnp.matmul(rotations, rotations_rigid)
            shifts = shifts + jnp.matmul(shifts_rigid[:, None, :], rearrange(rotations, "b m n -> b n m"))[:, 0, :2]

        # Check if CTF corruption is needed
        if not corrupt_projection_with_ctf:
            ctf_type = None

        # Generate projections
        images_corrected, _ = self.phys_decoder(flow, x, self.coords, self.values, self.xsize, rotations, shifts, ctf, ctf_type, self.sigma)

        if return_latent:
            return images_corrected, latent
        else:
            return images_corrected

    def decode_volume(self, x, filter=True, sigma=1.0):
        if x.ndim == 1:
            x = x[None, ...]

        # Get deformation field
        field = self.flow_decoder(x, self.coords, self.xsize)[0]

        # Deformed coords
        factor = 0.5 * self.xsize
        coords = (factor * self.coords[None, ...] + field) + factor

        # Place values on grids
        grids = jnp.zeros((x.shape[0], self.xsize, self.xsize, self.xsize))

        # Scatter volume
        bposf = jnp.floor(coords)
        bposi = bposf.astype(jnp.int32)
        bposf = coords - bposf

        bamp0 = self.values[None, ...] * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp1 = self.values[None, ...] * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp2 = self.values[None, ...] * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp3 = self.values[None, ...] * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp4 = self.values[None, ...] * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp5 = self.values[None, ...] * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1]) * (bposf[:, :, 2])
        bamp6 = self.values[None, ...] * (bposf[:, :, 0]) * (bposf[:, :, 1]) * (1.0 - bposf[:, :, 2])
        bamp7 = self.values[None, ...] * (bposf[:, :, 0]) * (bposf[:, :, 1]) * (bposf[:, :, 2])

        bamp = jnp.concat([bamp0, bamp1, bamp2, bamp3, bamp4, bamp5, bamp6, bamp7], axis=1)
        bposi = jnp.concat(
            [bposi, bposi + jnp.array((1, 0, 0)), bposi + jnp.array((0, 1, 0)), bposi + jnp.array((0, 0, 1)),
             bposi + jnp.array((0, 1, 1)), bposi + jnp.array((1, 0, 1)), bposi + jnp.array((1, 1, 0)),
             bposi + jnp.array((1, 1, 1))], axis=1)

        def scatter_volume(vol, bpos_i, bamp_i):
            return vol.at[bpos_i[..., 2], bpos_i[..., 1], bpos_i[..., 0]].add(bamp_i)

        grids = jax.vmap(scatter_volume, in_axes=(0, 0, 0))(grids, bposi, bamp)

        # Filter volume
        if filter:
            grids = jax.vmap(low_pass_3d, in_axes=(0, None))(grids, sigma)

        return grids

@jax.jit
def train_step_zernike3deep(graphdef, state, x, labels, md, key, do_update=True):
    model, optimizer, optimizer_grays = nnx.merge(graphdef, state)
    distributions_key, choice_key, key = jax.random.split(key, 3)

    calculate_deformation_regularity_loss_batch = jax.vmap(calculate_deformation_regularity_loss, in_axes=(0, None, None, None))
    calculate_repulsion_loss_batch = jax.vmap(calculate_repulsion_loss, in_axes=(0, None, None))

    def loss_fn(model, x):
        # Check if Tomo mode
        if model.isTomo:
            (x, subtomogram_label) = x

        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid), prev_layer_out = model.encoder(x, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                (sample, latent, logstd), prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
                (_, latent_1, _), (rotations_rigid, shifts_rigid), prev_layer_out_random = model.encoder(x, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            else:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid) = model.encoder(x, return_alignment_refinement=True, rngs=distributions_key)
        else:
            if model.decoupling:
                latent, (rotations_rigid, shifts_rigid), prev_layer_out = model.encoder(x, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                latent, prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True)
                latent_1, (rotations_rigid, shifts_rigid), prev_layer_out_random = model.encoder(x, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            else:
                latent, (rotations_rigid, shifts_rigid) = model.encoder(x, return_alignment_refinement=True, rngs=distributions_key)

        # Decode flow field
        if model.isVae:
            flow, coefficient_loss = model.flow_decoder(sample, model.coords, model.xsize)
        else:
            flow, coefficient_loss = model.flow_decoder(latent, model.coords, model.xsize)

        # Get rotation matrices
        if euler_angles.ndim == 2:
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        else:
            rotations = euler_angles

        # Refine angular alignment
        rotations_refined = jnp.matmul(rotations, rotations_rigid)
        shifts_refined = shifts + jnp.matmul(shifts_rigid[:, None, :], rearrange(rotations, "b m n -> b n m"))[:, 0, :2]

        # Generate projections
        images_corrected, (a, b) = model.phys_decoder(flow, x, model.coords, model.values, model.xsize, rotations_refined, shifts_refined, ctf, model.ctf_type, model.sigma)
        images_rigid, _ = model.phys_decoder(jnp.zeros_like(flow), x, model.coords, model.values, model.xsize, rotations_refined, shifts_refined, ctf, model.ctf_type, model.sigma)

        # Losses
        images_corrected = jnp.squeeze(images_corrected)
        images_rigid = jnp.squeeze(images_rigid)
        x = jnp.squeeze(x)

        # Consider CTF if Wiener or Squared mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter(x, ctf, pad_factor=2)
            images_corrected_loss = wiener2DFilter(images_corrected, ctf, pad_factor=2)
            images_rigid_loss = wiener2DFilter(images_rigid, ctf, pad_factor=2)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter(x, ctf, pad_factor=2)
            images_corrected_loss = ctfFilter(images_corrected, ctf, pad_factor=2)
            images_rigid_loss = ctfFilter(images_rigid, ctf, pad_factor=2)
        else:
            x_loss = x
            images_corrected_loss = images_corrected
            images_rigid_loss = images_rigid

        # Adjusted image
        # images_corrected_loss = a * images_corrected_loss + b

        recon_loss = mse(images_corrected_loss[..., None], x_loss[..., None]).mean()
        recon_loss_rigid = mse(images_rigid_loss[..., None], x_loss[..., None]).mean()
        # recon_loss = correlation_coefficient_loss(images_corrected_loss, x_loss).mean()
        # recon_loss_rigid = correlation_coefficient_loss(images_rigid_loss, x_loss).mean()
        recons_loss_all = 0.5 * (recon_loss + recon_loss_rigid)

        # Field norm loss
        # field_norm_loss = jnp.sqrt(jnp.square(flow).sum(axis=-1)).mean() / model.xsize

        if model.isVae:
            # KL divergence loss
            kl_loss = -0.5 * jnp.sum(1 + 2 * logstd - jnp.square(jnp.exp(logstd)) - jnp.square(latent))
        else:
            kl_loss = 0.0

        # Graph based loss
        consensus_distances = model.flow_decoder.consensus_distances
        deformed_positions = model.coords + flow / (0.5 * model.xsize)
        radius_graph = model.flow_decoder.edge_index
        edge_weights = model.flow_decoder.edge_weights
        tau = model.flow_decoder.tau
        loss_def_regularity = calculate_deformation_regularity_loss_batch(deformed_positions, radius_graph,
                                                                          consensus_distances, edge_weights)
        loss_repulsion = calculate_repulsion_loss_batch(deformed_positions, radius_graph, tau)
        loss_graph = (loss_def_regularity + 0.01 * loss_repulsion).mean()

        # Decoupling
        if model.decoupling or model.isTomo:
            if not model.isTomo:
                rotations_random_matrix = euler_matrix_batch(rotations_random[:, 0], rotations_random[:, 1], rotations_random[:, 2])
                shifts_random_refined = jnp.zeros_like(shifts_refined)
                images_random, _ = model.phys_decoder(flow, x, model.coords, model.values, model.xsize, rotations_random_matrix,
                                                      shifts_random_refined, ctf_random, model.ctf_type, model.sigma)
                if model.isVae:
                    (_, latent_1, _), prev_layer_out_random = model.encoder(images_corrected[..., None], "encoder_dec",
                                                                           return_last=True, return_alignment_refinement=False, rngs=distributions_key)
                    (_, latent_2, _) = model.encoder(images_random[..., None], "encoder_dec", return_alignment_refinement=False, rngs=distributions_key)
                else:
                    latent_1, prev_layer_out_random = model.encoder(images_corrected[..., None], "encoder_dec",
                                                                   return_last=True, return_alignment_refinement=False, rngs=distributions_key)
                    latent_2 = model.encoder(images_random[..., None], "encoder_dec", return_alignment_refinement=False, rngs=distributions_key)
                decoupling_loss = (jnp.mean(jnp.square(latent - latent_1), axis=-1).mean() +
                                   jnp.mean(jnp.square(latent - latent_2), axis=-1).mean() +
                                   jnp.mean(jnp.square(prev_layer_out - prev_layer_out_random), axis=-1).mean())
            else:
                decoupling_loss = (jnp.mean(jnp.square(latent - latent_1), axis=-1).mean() +
                                   jnp.mean(jnp.square(prev_layer_out - prev_layer_out_random), axis=-1).mean())

            random_indices = jnr.choice(choice_key, a=jnp.arange(model.bank_size), shape=(model.subset_size,),
                                        replace=False)
            memory_bank_subset = model.memory_bank[random_indices]

            dist = jnp.pow(latent[:, None, :] - memory_bank_subset, 2.).sum(axis=-1)
            dist_nn, _ = jax.lax.approx_min_k(dist, k=10, recall_target=0.95)
            dist_fn, _ = jax.lax.approx_max_k(dist, k=10, recall_target=0.95)

            # decoupling_loss += 1.0 * contrastive_ce_loss(dist_nn, dist_fn, reduction="mean", temperature=0.001)
            decoupling_loss += 1.0 * triplet_loss(dist_nn, dist_fn, reduction="mean", margin=0.01)

        else:
            decoupling_loss = 0.0

        loss = recons_loss_all + 0.000001 * kl_loss + 0.000001 * decoupling_loss + 0.9 * loss_graph
        return loss, (recon_loss, latent)

    # Check if Tomo mode
    if model.isTomo:
        (x, subtomogram_label) = x

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

    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, (params, params_grays)), has_aux=True)
    if model.isTomo:
        (loss, (recon_loss, latent)), grads_combined = grad_fn(model, (x, subtomogram_label))
    else:
        (loss, (recon_loss, latent)), grads_combined = grad_fn(model, x)

    grads, grads_gray = grads_combined.split(params, params_grays)

    if do_update:
        optimizer.update(model, grads)
        optimizer_grays.update(model, grads_gray)

        # Update memory bank
        model.enqueue(latent)

        state = nnx.state((model, optimizer, optimizer_grays))

        return loss, recon_loss, state, key
    else:
        return loss, recon_loss


@jax.jit
def validation_step_zernike3deep(graphdef, state, x, labels, md, key):
    model, optimizer, optimizer_grays = nnx.merge(graphdef, state)

    distributions_key, key = jax.random.split(key, 2)

    def loss_fn(model, x):
        # Check if Tomo mode
        if model.isTomo:
            (x, subtomogram_label) = x

        # Encode latent E(z)
        if model.isVae:
            if model.decoupling:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid), prev_layer_out = model.encoder(x, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                (sample, latent, logstd), prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True, rngs=distributions_key)
            else:
                (sample, latent, logstd), (rotations_rigid, shifts_rigid) = model.encoder(x, return_alignment_refinement=True, rngs=distributions_key)
        else:
            if model.decoupling:
                latent, (rotations_rigid, shifts_rigid), prev_layer_out = model.encoder(x, "encoder_exp", return_last=True, return_alignment_refinement=True, rngs=distributions_key)
            elif model.isTomoSIREN:
                latent, prev_layer_out = model.encoder(subtomogram_label, "encoder_dec", return_last=True, rngs=distributions_key)
            else:
                latent, (rotations_rigid, shifts_rigid) = model.encoder(x, return_alignment_refinement=True, rngs=distributions_key)

        # Decode flow field
        if model.isVae:
            flow, coefficient_loss = model.flow_decoder(sample, model.coords, model.xsize)
        else:
            flow, coefficient_loss = model.flow_decoder(latent, model.coords, model.xsize)

        # Get rotation matrices
        if euler_angles.ndim == 2:
            rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        else:
            rotations = euler_angles

        # Consider refinement and rigid registration alignments (for flow_decoder_rigid output)
        rotations_refined = jnp.matmul(rotations, rotations_rigid)
        shifts_refined = shifts + jnp.matmul(shifts_rigid[:, None, :], rearrange(rotations, "b m n -> b n m"))[:, 0, :2]

        # Generate projections
        images_corrected, (a, b) = model.phys_decoder(flow, x, model.coords, model.values, model.xsize, rotations_refined, shifts_refined, ctf, model.ctf_type)

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
            x_loss = a * x + b
            images_corrected_loss = images_corrected

        recon_loss = dm_pix.mse(images_corrected_loss[..., None], x_loss[..., None]).mean()

        loss = recon_loss
        return loss

    # Check if Tomo mode
    if model.isTomo:
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

    if model.isTomo:
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
    from hax.networks import train_step_zernike3deep, train_step_volume_adjustment, VolumeAdjustment
    from hax.metrics import JaxSummaryWriter
    from hax.programs import fit_volume, adjust_weights_to_images
    # from hax.schedulers import CosineAnnealingScheduler

    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, type=str,
                        help="Xmipp/Relion metadata file with the images (+ alignments / CTF) to be analyzed")
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
    parser.add_argument("--num_gaussians", required=False, type=int,
                        help="Before training the network, HetSIREN will try to fit a set of Gaussians in the reference volume to recreate it. "
                             "The default criterium is to automatically determine the number of Gaussians neede to reproduce the reference volume "
                             "with high-fidelity. However, if you prefer to fix the number of Gaussians in advance based on your own criterium (e.g., "
                             "the number of residues in your protein), you can set this parameter. When set, the HetSIREN will fit this fixed number of Gaussians "
                             "so that the reproduce the reference volume as well as possible.")
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
    parser.add_argument("-dataset_split_fraction", required=False, type=list_of_floats, default=[0.8, 0.2],
                        help=f"Here you can provide the fractions to split your data automatically into a training and a validation subset following the format: {bcolors.ITALIC}training_fraction{bcolors.ENDC},"
                             f"{bcolors.ITALIC}validation_fraction{bcolors.ENDC}. While the training subset will be used to train/update the network parameters, the validation subset will only be used to evaluate the "
                             f"accuracy of the network when faced with new data. Therefore, the validation subset will never be used to update the networks parameters. {bcolors.WARNING}NOTE{bcolors.ENDC}: the sum of "
                             f"{bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} must be equal to one.")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, new metadata...)")
    parser.add_argument("--reload", required=False, type=str,
                        help=f"Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data - "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: Since Zernike3Deep also learns a gray level adjustment, reload must be the path to a folder containing two additional "
                             f"folders called: {bcolors.UNDERLINE}Zernike3Deep{bcolors.ENDC} and {bcolors.UNDERLINE}Gaussian_volume_fitting{bcolors.ENDC})")
    args, _ = parser.parse_known_args()

    # Check that training and validation fractions add up to one
    if sum(args.dataset_split_fraction) != 1:
        raise ValueError(f"The sum of {bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} is not equal one. Please, update the values "
                         f"to fulfill this requirement.")

    # Preprocess volume (and mask)
    vol = ImageHandler(args.vol).getData()

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    else:
        mask = ImageHandler(args.vol).generateMask(boxsize=64)

    # # If exists, clean MMAP
    # if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Prepare grain dataset
    if not args.load_images_to_ram and args.mode in ["train", "predict"]:
        mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                             precision=np.float16, group_size=1, shard_size=10000)

    # Check if Tomo is needed
    isTomo = generator.mode == "tomo"

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, choice_key, model_key = jax.random.split(rng, 3)

    # Reload network
    if args.reload is not None:
        zernike3deep = NeuralNetworkCheckpointer.load(os.path.join(args.reload, "Zernike3Deep"))

    # Train network
    if args.mode == "train":

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "Zernike3Deep_metrics"))

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
            if zernike3deep.isTomo:
                (x_example, _), labels_example = next(iter_data_loader)
            else:
                x_example, labels_example = next(iter_data_loader)
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

        if not "zernike3deep" in locals():
            fit_path = os.path.join(args.output_path, "Gaussian_volume_fitting")
            if not os.path.isdir(os.path.join(fit_path)):
                # Consensus volume
                if args.num_gaussians is not None:
                    model, _, _ = fit_volume(vol, mask=mask, iterations=20000, learning_rate=0.001,
                                             n_init=args.num_gaussians, fixed_gaussians=True)
                else:
                    model, _, _ = fit_volume(vol, mask=mask, iterations=20000, learning_rate=0.01, grad_threshold=1e-5,
                                             densify_interval=2000, n_init=2500)

                # Adjust to images
                model, _ = adjust_weights_to_images(model, args.md, mmap_output_dir, args.sr, learning_rate=0.01,
                                                    num_epochs=3, is_global=True)

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

            # Prepare network (Zernike3Deep)
            coords = np.array(model.means.get_value())
            coords = np.stack([coords[..., 2], coords[..., 1], coords[..., 0]], axis=1)
            values = np.array(jax.nn.relu(model.weights.get_value()))
            sigma = jax.nn.relu(model.sigma_param.get_value())
            zernike3deep = Zernike3Deep(args.lat_dim, coords, values, vol.shape[0], args.sr,
                                        ctf_type=args.ctf_type, decoupling=True, isVae=True, sigma=sigma,
                                        L1=args.L1, L2=args.L2, bank_size=1024, isTomo=isTomo,
                                        rngs=nnx.Rngs(model_key))

        zernike3deep.train()

        # Learning rate scheduler
        # total_steps = args.epochs * steps_per_epoch
        # lr_schedule = CosineAnnealingScheduler.getScheduler(peak_value=args.learning_rate, total_steps=total_steps,
        #                                                     warmup_frac=0.1, end_value=0.0, init_value=1e-5)

        # Optimizers (Zernike3Deep)
        params = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('flow_decoder')))
        params_grays = nnx.All(nnx.Param, nnx.PathContains('phys_decoder'))
        optimizer = nnx.Optimizer(zernike3deep, optax.adamw(args.learning_rate), wrt=params)
        optimizer_grays = nnx.Optimizer(zernike3deep, optax.adamw(args.learning_rate), wrt=params_grays)  # TODO: Check if it is better to fix it to 1e-5 always
        graphdef, state = nnx.split((zernike3deep, optimizer, optimizer_grays))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "Zernike3Deep_CHECKPOINT")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "HetSIREN_CHECKPOINT"),
                                                                                        optimizer, optimizer_grays)
            print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
        else:
            resume_epoch = 0

        # Jitted functions to improve performance
        @partial(jax.jit, static_argnames=["ctf_type", "return_latent", "corrupt_projection_with_ctf"])
        def zernike3deep_decode_image(graphdef, state, x, labels, md, ctf_type=None, return_latent=False,
                                  corrupt_projection_with_ctf=False):
            model, _, _ = nnx.merge(graphdef, state)
            return model.decode_image(x, labels, md, ctf_type=ctf_type, return_latent=return_latent,
                                      corrupt_projection_with_ctf=corrupt_projection_with_ctf)

        @jax.jit
        def zernike3deep_decode_volume(graphdef, state, x):
            model, _, _ = nnx.merge(graphdef, state)
            return model.decode_volume(x)

        image_resize = jax.jit(jax.image.resize, static_argnames=("shape", "method"))

        # Training loop (Zernike3Deep)
        print(f"{bcolors.OKCYAN}\n###### Training variability... ######")

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

                    # Log intermediate results at the end of the epoch
                    # Get first 5 images from batch
                    if zernike3deep.isTomo:
                        x_for_tb = x[0][:5]
                    else:
                        x_for_tb = x[:5]
                    labels_for_tb = labels[:5]

                    # Decode some images and show them in Tensorboard
                    x_pred_intermediate, latents_intermediate = zernike3deep_decode_image(graphdef, state, x_for_tb,
                                                                                          labels_for_tb, md_columns,
                                                                                          ctf_type=args.ctf_type,
                                                                                          return_latent=True,
                                                                                          corrupt_projection_with_ctf=True)
                    x_pred_intermediate = jax.vmap(min_max_scale)(x_pred_intermediate[..., None])
                    writer.add_images("Predicted images batch", x_pred_intermediate, dataformats="NHWC")

                    # Decode some states and show them in Tensorboard
                    volumes_intermediate = zernike3deep_decode_volume(latents_intermediate)
                    writer.add_volumes_slices(volumes_intermediate)

                    # Log landscape stored in memory bank
                    if i > 0 and i % 5 == 0:
                        choice_key_use, choice_key = jax.random.split(choice_key, 2)
                        hetsiren_intermediate, _ = nnx.merge(graphdef, state)
                        random_indices = jnr.choice(choice_key_use,
                                                    a=jnp.arange(hetsiren_intermediate.bank_size),
                                                    shape=(hetsiren_intermediate.subset_size,), replace=False)
                        latents_intermediate = hetsiren_intermediate.memory_bank.get_value()[random_indices]
                        latents_data_loader = NumpyGenerator(latents_intermediate).return_tf_dataset(preShuffle=False,
                                                                                                     shuffle=False,
                                                                                                     batch_size=args.batch_size)
                        latents_images = []
                        for (latents, _) in latents_data_loader:
                            random_labels = jnp.asarray(
                                np.random.randint(low=0, high=len(generator.md), size=(latents.shape[0],)),
                                dtype=jnp.int32)
                            x_pred_intermediate = zernike3deep_decode_image(graphdef, state, latents, random_labels,
                                                                            md_columns, ctf_type=None,
                                                                            return_latent=False,
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
                                             tag="Zernike3Deep latent space", global_step=i)

                        # Save checkpoint model
                        NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path, "Zernike3Deep_CHECKPOINT"),  epoch=i)

                loss, recon_loss, state, rng = train_step_zernike3deep(graphdef, state, x, labels, md_columns, rng)
                total_loss += loss
                total_recon_loss += recon_loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | recon_loss={total_recon_loss / step:.5f}")

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * steps_per_epoch)) == 0:
                    zernike3deep_intermediate, _, _ = nnx.merge(graphdef, state)

                    writer.add_scalar('Training loss (Zernike3Deep)',
                                      total_loss / step,
                                      i * steps_per_epoch + step)

                    writer.add_scalars('Image loss (Zernike3Deep)',
                                       {"train": total_recon_loss / step},
                                       i * steps_per_epoch + step)

                # Summary writer (validation loss)
                if step % int(np.ceil(0.5 * steps_per_epoch)) == 0:
                    # Run validation step
                    pbar.set_postfix_str(f"{bcolors.WARNING}Running validation step...{bcolors.ENDC}")
                    for _ in range(steps_per_val):
                        (x_validation, labels_validation) = next(iter_data_loader_val)
                        loss_validation = validation_step_zernike3deep(graphdef, state, x_validation,
                                                                       labels_validation, md_columns, rng)
                        total_validation_loss += loss_validation
                        step_validation += 1

                    writer.add_scalars('Image loss (Zernike3Deep)',
                                       {"validation": total_validation_loss / step_validation},
                                       i * steps_per_epoch + step)

                step += 1

        zernike3deep, optimizer, optimizer_grays = nnx.merge(graphdef, state)

        # Example of predicted data for Tensorboard
        x_pred_example = zernike3deep_decode_image(graphdef, state, x_example, labels_example, md_columns, ctf_type=args.ctf_type, return_latent=False, corrupt_projection_with_ctf=True)
        x_pred_example = jax.vmap(min_max_scale)(x_pred_example[..., None])
        writer.add_images("Predicted images batch", x_pred_example, dataformats="NHWC")

        # Save model
        NeuralNetworkCheckpointer.save(zernike3deep, os.path.join(args.output_path, "Zernike3Deep"))

        # Remove checkpoint
        shutil.rmtree(os.path.join(args.output_path, "Zernike3Deep_CHECKPOINT"))

    elif args.mode == "predict":

        # Rotations to Xmipp angles
        euler_from_matrix_batch = jax.vmap(jax.jit(euler_from_matrix))

        def xmippEulerFromMatrix(matrix):
            return -jnp.rad2deg(euler_from_matrix_batch(matrix))

        # Prepare data loader
        data_loader = generator.return_grain_dataset(batch_size=args.batch_size, shuffle=False, num_epochs=1,
                                                     num_workers=0, load_to_ram=args.load_images_to_ram)
        steps_per_epoch = int(np.ceil(len(generator.md) / args.batch_size))

        # Jitted prediction function
        predict_fn = jax.jit(zernike3deep.__call__)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting Zernike3Deep latents... ######")

        pbar = tqdm(range(steps_per_epoch), desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        latents = []
        euler_angles = []
        shifts = []
        with closing(iter(data_loader)) as iter_data_loader:
            for _ in pbar:
                (x, labels) = next(iter_data_loader)

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

                latents_batch, (rotations_rigid, shifts_rigid) = predict_fn(x)

                # Precompute batch aligments
                rotations_batch = md_columns["euler_angles"][labels]

                # Precompute batch shifts
                shifts_batch = md_columns["shifts"][labels]

                # Get rotation matrices
                if rotations_batch.ndim == 2:
                    rotations_batch = euler_matrix_batch(rotations_batch[:, 0], rotations_batch[:, 1],
                                                         rotations_batch[:, 2])

                # Consider refinement and rigid registration alignments
                rotations_refined = jnp.matmul(rotations_batch, rotations_rigid)
                shifts_refined = shifts_batch + jnp.matmul(shifts_rigid[:, None, :],
                                                           rearrange(rotations_batch, "b m n -> b n m"))[:, 0, :2]

                # Convert rotation to Euler angles in Xmipp format
                euler_angles_refined = xmippEulerFromMatrix(rotations_refined)

                # Convert to Numpy
                euler_angles_refined, shifts_refined = np.array(euler_angles_refined), np.array(shifts_refined)

                latents.append(latents_batch)
                euler_angles.append(euler_angles_refined)
                shifts.append(shifts_refined)
        latents = np.asarray(jnp.concatenate(latents, axis=0))
        euler_angles = np.asarray(jnp.concatenate(euler_angles, axis=0))
        shifts = np.asarray(jnp.concatenate(shifts, axis=0))

        # Save latents in metadata
        md = generator.md
        md[:, 'latent_space'] = np.asarray([",".join(np.char.mod('%f', item)) for item in latents])
        md[:, 'angleRot'] = euler_angles[:, 0]
        md[:, 'angleTilt'] = euler_angles[:, 1]
        md[:, 'anglePsi'] = euler_angles[:, 2]
        md[:, 'shiftX'] = shifts[:, 0]
        md[:, 'shiftY'] = shifts[:, 1]
        md.write(os.path.join(args.output_path, "predicted_latents" +  os.path.splitext(args.md)[1]))

    elif args.mode == "send_to_pickle":

        # Save mode to pickle
        NeuralNetworkCheckpointer.save(zernike3deep, os.path.join(args.output_path, "Zernike3Deep"))

    # If exists, clean MMAP
    if not args.load_images_to_ram and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap_grain")):
        shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap_grain"))
