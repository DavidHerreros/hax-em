#!/usr/bin/env python


import random

import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import dm_pix

from einops import rearrange

from hax.utils import *
from hax.layers import *


class Encoder(nnx.Module):
    def __init__(self, input_dim, pyramid_levels=4, num_components=4, refine_current_assignment=False, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.input_conv_dim = 64  # Original was 64
        self.out_conv_dim = int(self.input_conv_dim / (2 ** 3))
        self.pyramid_levels = pyramid_levels
        self.num_components = num_components
        self.refine_current_assignment = refine_current_assignment

        # Hidden layers
        self.hidden_layers_conv = [nnx.Conv(self.pyramid_levels, 64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16)]
        self.hidden_layers_conv.append(nnx.Conv(64, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

        self.hidden_layers_conv.append(nnx.Conv(64, 128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_conv.append(nnx.Conv(128, 128, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

        self.hidden_layers_conv.append(nnx.Conv(128, 256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_conv.append(nnx.Conv(256, 256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_conv.append(nnx.Conv(256, 256, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

        self.hidden_layers_conv.append(nnx.Conv(256, 512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_conv.append(nnx.Conv(512, 512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_conv.append(nnx.Conv(512, 512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", rngs=rngs, dtype=jnp.bfloat16))

        self.hidden_layers_linear = [nnx.Linear(self.out_conv_dim * self.out_conv_dim * 512, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        self.hidden_layers_linear.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_linear.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_layers_linear.append(Linear(1024, 8, rngs=rngs))

        # Layers to 9D rotation
        self.hidden_9d_rotation = [nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        self.hidden_9d_rotation.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_9d_rotation.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_9d_rotation.append(nnx.Linear(1024, 3, rngs=rngs))
        if refine_current_assignment:
            self.hidden_9d_rotation.append(nnx.Linear(1024, self.num_components * 6, rngs=rngs, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros))
            self.alpha_rotations = nnx.Param(jnp.array(1e-4))
        else:
            self.hidden_9d_rotation.append(nnx.Linear(1024, self.num_components * 9, rngs=rngs))

        # Layers to shifts
        self.hidden_shifts = [nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        self.hidden_shifts.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_shifts.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        if refine_current_assignment:
            self.hidden_shifts.append(nnx.Linear(1024, 2, rngs=rngs, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros))
            self.alpha_shifts = nnx.Param(jnp.array(1e-4))
        else:
            self.hidden_shifts.append(nnx.Linear(1024, 2, rngs=rngs, kernel_init=normal_initializer_mean(mean=0.0, stddev=1e-4)))

    def __call__(self, x, return_diversity_loss=False):
        # Resize images
        x = jax.image.resize(x, (x.shape[0], self.input_conv_dim, self.input_conv_dim, 1), method="bilinear")

        # Pyramid filter
        pyramid_levels_imgs = []
        for i in range(self.pyramid_levels ):
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
                x = nnx.relu(x + layer(x))
            else:
                x = nnx.relu(layer(x))

        # Linear hidden layers
        x = rearrange(x, 'b h w c -> b (h w c)')
        for layer in self.hidden_layers_linear[:-1]:
            if layer.in_features == layer.out_features:
                x = nnx.relu(x + layer(x))
            else:
                x = nnx.relu(layer(x))
        x = self.hidden_layers_linear[-1](x)

        # First output: rotation matrices
        # rotation_9d = nnx.relu(self.hidden_9d_rotation[0](x))
        # for layer in self.hidden_9d_rotation[1:-1]:
        #     rotation_9d = nnx.relu(layer(rotation_9d))
        # rotation_9d = self.hidden_9d_rotation[-1](rotation_9d)  # TODO: THIS WAS WORKING WITH JUST ONE LAYER!! CHECK IN THE FUTURE IN CASE IT IS USEFUL
        rotation_9d = self.hidden_9d_rotation[-1](x)  # TODO: THIS WAS WORKING WITH JUST ONE LAYER!! CHECK IN THE FUTURE IN CASE IT IS USEFUL

        if self.refine_current_assignment:
            rotations_6d = rotation_9d.reshape(x.shape[0] * self.num_components, 6)
            identity_6d = jnp.array([1., 0., 0., 0., 1., 0.])[None, ...].repeat(rotations_6d.shape[0], axis=0)
            rotation_6d = identity_6d + self.alpha_rotations * rotations_6d
            a1, a2 = jnp.split(rotation_6d, 2, axis=-1)
            b1 = a1 / jnp.clip(jnp.linalg.norm(a1, axis=-1, keepdims=True), a_min=1e-6)
            a2_ortho = a2 - jnp.sum(a2 * b1, axis=-1, keepdims=True) * b1
            b2 = a2_ortho / jnp.clip(jnp.linalg.norm(a2_ortho, axis=-1, keepdims=True), a_min=1e-6)
            b3 = jnp.cross(b1, b2, axis=-1)
            rotations = jnp.stack([b1, b2, b3], axis=-1)
            rotations = rotations.reshape(x.shape[0], self.num_components, 3, 3)
        else:
            M = rotation_9d.reshape(x.shape[0] * self.num_components, 9).reshape(-1, 3, 3)
            U, _, V = jnp.linalg.svd(M, full_matrices=False)
            det = jnp.linalg.det(jnp.matmul(U, V))
            correction_diag = jnp.eye(3).reshape((1,) * (det.ndim) + (3, 3))
            correction_diag = jnp.broadcast_to(correction_diag, (M.shape[0], 3, 3))
            correction_diag = correction_diag.at[..., 2, 2].set(det)
            U_corrected = jnp.matmul(U, correction_diag)
            rotations = jnp.matmul(U_corrected, V)
            rotations = rotations.reshape(x.shape[0], self.num_components, 3, 3)

        # Third output: in plane shifts
        in_plane_shifts = nnx.relu(self.hidden_shifts[0](x))
        for layer in self.hidden_shifts[1:-1]:
            in_plane_shifts = nnx.relu(layer(in_plane_shifts))
        # in_plane_shifts = 0.5 * self.input_dim * self.hidden_shifts[-1](in_plane_shifts)
        in_plane_shifts = self.hidden_shifts[-1](in_plane_shifts)
        if self.refine_current_assignment:
            in_plane_shifts = self.alpha_shifts * in_plane_shifts

        # Broadcast shifts to euler angles shape
        in_plane_shifts = jnp.broadcast_to(in_plane_shifts[:, None, :], (in_plane_shifts.shape[0], self.num_components, 2))

        if return_diversity_loss:
            directions = rotations @ jnp.array([0, 0, 1])
            pairwise_dots = directions @ directions.transpose(0, 2, 1)
            off_diagonal_dots = pairwise_dots * (1.0 - jnp.eye(rotations.shape[1], dtype=pairwise_dots.dtype))
            diversity_loss = jnp.mean(jnp.sum(jnp.square(off_diagonal_dots), axis=(-2, -1)))
            return rotations, in_plane_shifts, diversity_loss
        else:
            return rotations, in_plane_shifts


class DeltaVolumeDecoder(nnx.Module):
    def __init__(self, total_voxels, volume_size, inds, reference_values, transport_mass=False, learn_delta_volume=True, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.inds = inds
        self.reference_values = reference_values
        self.total_voxels = total_voxels
        self.transport_mass = transport_mass
        self.learn_delta_volume = learn_delta_volume

        # Indices to (normalized) coords
        self.factor = 0.5 * volume_size
        coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)[None, ...]
        self.coords = (coords - self.factor) / self.factor

        # Delta volume decoder (TODO: Check and fix hypernetwork - compare with TF implementation)
        # self.hidden_linear = [HyperLinear(in_features=lat_dim, out_features=8, in_hyper_features=lat_dim, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16)]
        # for _ in range(3):
        #     self.hidden_linear.append(HyperLinear(in_features=8, out_features=8, in_hyper_features=8, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16))
        # self.hidden_linear.append(HyperLinear(in_features=8, out_features=8, in_hyper_features=8, hidden_hyper_features=8, rngs=rngs, dtype=jnp.bfloat16))

        if transport_mass or learn_delta_volume:
            self.hidden_linear = [Linear(in_features=total_voxels * 3, out_features=8, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init_first(c=1.))]
            for _ in range(3):
                self.hidden_linear.append(Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init(c=1.)))
            self.hidden_linear.append(Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init(c=1.)))

        if transport_mass:
            if learn_delta_volume:
                self.hidden_linear.append(Linear(in_features=8, out_features=4 * total_voxels, rngs=rngs))
            else:
                self.hidden_linear.append(Linear(in_features=8, out_features=3 * total_voxels, rngs=rngs))
        else:
            if learn_delta_volume:
                self.hidden_linear.append(Linear(in_features=8, out_features=total_voxels, rngs=rngs))


    def __call__(self):
        coords = self.coords.flatten()[None, ...]

        # Decode voxel values
        x = jnp.sin(1.0 * self.hidden_linear[0](coords))
        for layer in self.hidden_linear[1:-1]:
            # x = jnp.sin(1.0 * (x + layer(x, x)))
            x = x + jnp.sin(1.0 * layer(x))
        x = self.hidden_linear[-1](x)

        if self.transport_mass:
            if self.learn_delta_volume:
                # Extract delta_coords and values
                x = jnp.reshape(x, (x.shape[0], self.total_voxels, 4))
                delta_coords, delta_values = x[..., :3], x[..., 3]

                # Recover volume values (TODO: Check if applying ReLu is really needed)
                values = nnx.relu(self.reference_values + delta_values)
            else:
                # Extract delta_coords
                delta_coords = jnp.reshape(x, (x.shape[0], self.total_voxels, 3))

                # Recover volume values (TODO: Check if applying ReLu is really needed)
                values = nnx.relu(self.reference_values)

            # Recover coords (non-normalized)
            coords = self.factor * (self.coords + delta_coords)
        else:
            # Recover volume values
            if self.learn_delta_volume:
                values = self.reference_values + x
            else:
                values = self.reference_values

            # Recover coords (non-normalized)
            coords = self.factor * self.coords

        return coords, values

    def decode_volume(self, coords_values=None, filter=True):
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

    def __call__(self, x, values, coords, xsize, rotations, shifts, ctf, ctf_type,  filter=True):
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
        if ctf_type in ["apply", "wiener", "squared"]:
            ctf = jnp.broadcast_to(ctf[:, None, :], (ctf.shape[0], rotations.shape[1], ctf.shape[1], ctf.shape[2]))
            ctf = rearrange(ctf, "b n w h -> (b n) w h")
            images = ctfFilter(images, ctf, pad_factor=2)

        images = rearrange(images, "(b n) w h -> b n w h", b=rotations.shape[0], n=rotations.shape[1])

        return images

class ReconSIREN(nnx.Module):
    def __init__(self, reference_volume, reconstruction_mask, xsize, sr, bank_size=10000, ctf_type="apply", transport_mass=False,
                 symmetry_group="c1", refine_current_assignment=False, learn_delta_volume=True, *, rngs: nnx.Rngs):
        super(ReconSIREN, self).__init__()
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.reference_volume = reference_volume
        self.reconstruction_mask = reconstruction_mask.astype(float)
        self.inds = jnp.asarray(jnp.where(reconstruction_mask > 0.0)).T
        self.symmetry_matrices = symmetry_matrices(symmetry_group)
        self.refine_current_assignment = refine_current_assignment
        self.learn_delta_volume = learn_delta_volume
        reference_values = reference_volume[self.inds[..., 0], self.inds[..., 1], self.inds[..., 2]][None, ...]
        self.encoder = Encoder(self.xsize, refine_current_assignment=refine_current_assignment, rngs=rngs)
        self.delta_volume_decoder = DeltaVolumeDecoder(self.inds.shape[0], self.xsize, self.inds, reference_values,
                                                       transport_mass=transport_mass, learn_delta_volume=learn_delta_volume, rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize, transport_mass=transport_mass)

        # Hyperparameter tuning
        self.alpha_uniform = nnx.Param(1e-4)

        #### Memory bank for latent spaces ####
        self.bank_size = bank_size
        self.subset_size = min(2048, bank_size)
        self.choice_key = rngs.choice()

        self.memory_bank = nnx.Variable(
            jnp.zeros((self.bank_size, 2))
        )
        self.memory_bank_ptr = nnx.Variable(
            jnp.zeros((1,), dtype=jnp.int32)
        )

    def __call__(self, x, **kwargs):
        # TODO: Return only best angles
        return self.encoder(x)

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
        rotations, shifts = self(x)

        # Decode volume
        coords, values = self.delta_volume_decoder()

        # Generate projections
        images_corrected = self.phys_decoder(x, values, coords, self.xsize, rotations, shifts, ctf, ctf_type)

        return images_corrected


@jax.jit
def train_step_reconsiren(graphdef, state, x, labels, md, key):
    model, optimizer_pose, optimizer_volume = nnx.merge(graphdef, state)

    # Random keys
    key, swd_key, uniform_key, choice_key = jax.random.split(key, 4)

    def loss_fn(model, x):
        # Correct CTF in images for encoder if needed
        if model.ctf_type == "apply":
            x_ctf_corrected = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]
        else:
            x_ctf_corrected = x

        # Get euler angles and shifts
        rotations, shifts, diversity_loss = model.encoder(x_ctf_corrected, return_diversity_loss=True)

        # Decode volume
        coords, values = model.delta_volume_decoder()
        # volumes = model.delta_volume_decoder.decode_volume(coords_values=[coords, values])
        # chimeric_volumes =  volumes + ((1. - model.reconstruction_mask) * model.reference_volume)[None, ...]

        # Refine current assignment (if provided)
        # rotations = jnp.matmul(rotations, current_rotations[:, None, :, :])
        rotations = jnp.matmul(current_rotations[:, None, :, :], rotations)  # TODO: The two options seem to work?
        shifts = current_shifts[:, None, :] + shifts

        # Random symmetry matrices
        random_indices = jax.random.choice(choice_key, jnp.arange(model.symmetry_matrices.shape[0]), shape=(rotations.shape[0],))
        rotations = jnp.matmul(jnp.transpose(model.symmetry_matrices[random_indices], (0, 2, 1))[:, None, :, :], rotations)

        # Generate projections
        images_corrected = model.phys_decoder(x, values, coords, model.xsize, rotations, shifts, ctf, model.ctf_type)

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

        # Project "mask"
        if not model.delta_volume_decoder.transport_mass:
            projected_mask = model.phys_decoder(x, jnp.ones_like(values), coords, model.xsize, rotations, shifts, ctf, None, False)
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
        rotations = rotations[jnp.arange(images_corrected.shape[0]), min_indices, :]

        # Rotations to Euler angles (ZYZ)
        euler_angles = euler_from_matrix_batch(rotations)[..., :2]

        # L1 based denoising
        if not model.delta_volume_decoder.transport_mass:
            l1_loss = jnp.mean(jnp.abs(values))
        else:
            l1_loss = 0.0

        # L1 and L2 total variation
        diff_x, diff_y, diff_z = sparse_finite_3D_differences(values, model.inds, model.xsize)
        l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        # Decoupling (TODO: In the future this will be for missing angles like TF implementation)

        # Uniform angular distribution loss
        random_indices = jnr.choice(model.choice_key, a=jnp.arange(model.bank_size), shape=(model.subset_size,),
                                    replace=False)
        memory_bank_subset = model.memory_bank[random_indices]
        uniform_distributed_rot = jax.random.uniform(uniform_key, shape=memory_bank_subset.shape[0], minval=-jnp.pi, maxval=jnp.pi)
        uniform_distributed_tilt = jax.random.uniform(uniform_key, shape=memory_bank_subset.shape[0], minval=0.0, maxval=jnp.pi)
        uniform_distributed_samples = jnp.stack([uniform_distributed_rot, uniform_distributed_tilt], axis=1)
        uniform_angular_distribution_loss = sliced_wasserstein_loss(memory_bank_subset, uniform_distributed_samples, key)

        # loss = recon_loss + l1_loss  + 0.001 * (l1_grad_loss + l2_grad_loss) # + 0.001 * uniform_angular_distribution_loss
        loss = recon_loss + l1_loss + 0.001 * (l1_grad_loss + l2_grad_loss) + nnx.relu(model.alpha_uniform.value) * uniform_angular_distribution_loss + nnx.relu(model.alpha_uniform.value) * diversity_loss
        return loss, (recon_loss, euler_angles)


    # Optimizer parameters
    params_pose = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('alpha_uniform')))
    params_volume = nnx.All(nnx.Param, nnx.PathContains('delta_volume_decoder'))

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

    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, (params_pose, params_volume)), has_aux=True)
    (loss, (recon_loss, euler_angles)), grads_combined = grad_fn(model, x)

    grads_pose, grads_volume = grads_combined.split(params_pose, params_volume)

    optimizer_pose.update(grads_pose)
    optimizer_volume.update(grads_volume)

    # grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    # (loss, (recon_loss, euler_angles)), grads = grad_fn(model, x)
    #
    # optimizer_pose.update(grads)

    # Update memory bank
    model.enqueue(euler_angles)

    state = nnx.state((model, optimizer_pose, optimizer_volume))

    return loss, recon_loss, state, key


@jax.jit
def validation_step_reconsiren(graphdef, state, x, labels, md, key):
    model, optimizer_pose, optimizer_volume = nnx.merge(graphdef, state)

    # Random keys
    key, choice_key = jax.random.split(key, 2)

    def loss_fn(model, x):
        # Correct CTF in images for encoder if needed
        if model.ctf_type == "apply":
            x_ctf_corrected = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]
        else:
            x_ctf_corrected = x

        # Get euler angles and shifts
        rotations, shifts, _ = model.encoder(x_ctf_corrected, return_diversity_loss=True)

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
        images_corrected = model.phys_decoder(x, values, coords, model.xsize, rotations, shifts, ctf, model.ctf_type)

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

        # Project "mask"
        if not model.delta_volume_decoder.transport_mass:
            projected_mask = model.phys_decoder(x, jnp.ones_like(values), coords, model.xsize, rotations, shifts, ctf, None, False)
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
def predict_angular_assignment_step_reconsiren(graphdef, state, x, labels, md):
    model = nnx.merge(graphdef, state)

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
    rotations, shifts, diversity_loss = model.encoder(x_ctf_corrected, return_diversity_loss=True)

    # Decode volume
    coords, values = model.delta_volume_decoder()

    # Refine current assignment (if provided)
    # rotations = jnp.matmul(rotations, current_rotations[:, None, :, :])
    rotations = jnp.matmul(current_rotations[:, None, :, :], rotations)  # TODO: The two options seem to work?
    shifts = current_shifts[:, None, :] + shifts

    # Generate projections
    images_corrected = model.phys_decoder(x, values, coords, model.xsize, rotations, shifts, ctf, model.ctf_type)

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

    # Project "mask"
    if not model.delta_volume_decoder.transport_mass:
        projected_mask = model.phys_decoder(x, jnp.ones_like(values), coords, model.xsize, rotations, shifts, ctf, None, False)
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

    # Index shifts and rotations based on extracted indices
    rotations = rotations[jnp.arange(images_corrected.shape[0]), min_indices, :]
    shifts = shifts[jnp.arange(images_corrected.shape[0]), min_indices, :]

    return rotations, shifts

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
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, extract_columns
    from hax.metrics import JaxSummaryWriter
    from hax.networks import VolumeAdjustment, train_step_volume_adjustment
    from hax.schedulers import CosineAnnealingScheduler

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
    # parser.add_argument("--transport_mass", action='store_true',
    #                     help=f'When set, ReconSIREN will be able to "move" the mass inside the mask instead of just reconstructing the volume. This implies that ReconSIREN will estimate the motion '
    #                          f'to be applied to the points within the provided mask, instead of considering them fixed in space. This approach is useful when working with large box sizes that '
    #                          f'do not fit in GPU memory, or when a more through analysis of motions is desired. {bcolors.WARNING}NOTE{bcolors.ENDC}: We strongly recommend to use this mode, as it '
    #                          f'yields optimal results for most cases. {bcolors.WARNING}NOTE{bcolors.ENDC}: When this option is set and a reference volume is provided, we recommend changing the '
    #                          f'reference mask to a tight mask computed from the reference volume. This mask now tells the program which regions should be moved. Therefore, consider providing a mask '
    #                          f'that covers all the protein regions you would like to be analyzed by HetSIREN.')
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
    parser.add_argument("-dataset_split_fraction", required=False, type=list_of_floats, default=[0.8, 0.2],
                        help=f"Here you can provide the fractions to split your data automatically into a training and a validation subset following the format: {bcolors.ITALIC}training_fraction{bcolors.ENDC},"
                             f"{bcolors.ITALIC}validation_fraction{bcolors.ENDC}. While the training subset will be used to train/update the network parameters, the validation subset will only be used to evaluate the "
                             f"accuracy of the network when faced with new data. Therefore, the validation subset will never be used to update the networks parameters. {bcolors.WARNING}NOTE{bcolors.ENDC}: the sum of "
                             f"{bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} must be equal to one.")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, new metadata...)")
    parser.add_argument("--reload", required=False, type=str,
                        help=f"Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data).")
    args = parser.parse_args()

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

    # Preprocess volume (and mask)
    xsize = generator.md.getMetaDataImage(0).shape[0]
    if args.vol is not None:
        vol = ImageHandler(args.vol).getData()
    else:
        vol = np.zeros((xsize, xsize, xsize))

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    else:
        mask = ImageHandler().createCircularMask(boxSize=xsize, radius=int(0.125 * xsize), is3D=True)

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

    # Prepare network (ReconSIREN)
    reconsiren = ReconSIREN(vol, mask, xsize, args.sr, ctf_type=args.ctf_type, symmetry_group=args.symmetry_group,
                            transport_mass=True, refine_current_assignment=args.refine_current_assignment,
                            bank_size=len(generator.md), learn_delta_volume=not args.do_not_learn_volume, rngs=nnx.Rngs(model_key))

    # Reload network
    if args.reload is not None:
        reconsiren = NeuralNetworkCheckpointer.load(reconsiren, os.path.join(args.reload, "ReconSIREN"))

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

        reconsiren.train()

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "ReconSIREN_metrics"))

        # Jitted functions for volume prediction
        @jax.jit
        def decode_volume(graphdef, state):
            model, _, _ = nnx.merge(graphdef, state)
            return model.delta_volume_decoder.decode_volume()

        # Prepare data loader
        data_loader_full, data_loader, data_loader_validation = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=True, preShuffle=True,
                                                                                            mmap=mmap, mmap_output_dir=mmap_output_dir, split_fraction=args.dataset_split_fraction)

        # Example of training data for Tensorboard
        x_example, labels_example = next(iter(data_loader))
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Training data batch", x_example, dataformats="NHWC")

        if args.vol is not None:
            if not os.path.isdir(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT")):
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
                    pbar = tqdm(data_loader_full, desc=f"Epoch {i + 1}/{num_epochs_vol}", file=sys.stdout, ascii=" >=",
                                colour="green")

                    for (x, labels) in pbar:
                        loss, state = train_step_volume_adjustment(graphdef, state, x, labels, md_columns, args.sr,
                                                                   args.ctf_type, vol.shape[0])
                        total_loss += loss

                        # Progress bar update  (TQDM)
                        pbar.set_postfix_str(f"loss={total_loss / step:.5f}")

                        # Summary writer (training loss)
                        if step % int(np.ceil(0.1 * len(data_loader_full))) == 0:
                            writer.add_scalar('Training loss (volume adjustment)',
                                              total_loss / step,
                                              i * len(data_loader_full) + step)

                        step += 1

                volumeAdjustment, optimizer_vol = nnx.merge(graphdef, state)
                values = volumeAdjustment()

                # Place values on grid and replace ReconSIREN reference volume
                grid = jnp.zeros_like(vol)
                grid = grid.at[inds[..., 0], inds[..., 1], inds[..., 2]].set(values)
                reconsiren.reference_volume = grid
                reconsiren.delta_volume_decoder.reference_values = values

                # Save model
                NeuralNetworkCheckpointer.save(reconsiren, os.path.join(args.output_path, "volumeAdjustment"), mode="pickle")

        # Learning rate scheduler
        total_steps = args.epochs * len(data_loader)
        lr_schedule_pose = CosineAnnealingScheduler.getScheduler(peak_value=4. * args.learning_rate, total_steps=total_steps, warmup_frac=0.1, init_value=args.learning_rate, end_value=0.0)
        lr_schedule_volume = CosineAnnealingScheduler.getScheduler(peak_value=4. * 1e-3, total_steps=total_steps, warmup_frac=0.1, init_value=1e-3, end_value=0.0)

        # Optimizers (ReconSIREN)
        params_pose = nnx.All(nnx.Param, (nnx.PathContains('encoder'), nnx.PathContains('alpha_uniform')))
        params_volume = nnx.All(nnx.Param, nnx.PathContains('delta_volume_decoder'))
        optimizer_pose = nnx.Optimizer(reconsiren, optax.adam(lr_schedule_pose), wrt=params_pose)
        optimizer_volume = nnx.Optimizer(reconsiren, optax.adam(lr_schedule_volume), wrt=params_volume)
        graphdef, state = nnx.split((reconsiren, optimizer_pose, optimizer_volume))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"))
            print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
        else:
            resume_epoch = 0

        # Training loop (ReconSIREN)
        training_volume_log = " / volume" if not args.do_not_learn_volume else ""
        print(f"{bcolors.OKCYAN}\n###### Training angular assignment / shifts{training_volume_log}... ######")
        for i in range(resume_epoch, args.epochs):
            total_loss = 0
            total_recon_loss = 0
            total_validation_loss = 0

            # For progress bar (TQDM)
            step = 1
            step_validation = 1
            print(f'\nTraining epoch {i + 1}/{args.epochs} |')
            pbar = tqdm(data_loader, desc=f"Epoch {i + 1}/{args.epochs}", file=sys.stdout, ascii=" >=", colour="green")

            for (x, labels) in pbar:
                loss, recon_loss, state, rng = train_step_reconsiren(graphdef, state, x, labels, md_columns, rng)
                total_loss += loss
                total_recon_loss += recon_loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | recon_loss={total_recon_loss / step:.5f}")

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * len(data_loader))) == 0:
                    writer.add_scalar('Training loss (ReconSIREN)',
                                      total_loss / step,
                                      i * len(data_loader) + step)

                    writer.add_scalars('Reconstruction loss (ReconSIREN)',
                                       {"train": total_recon_loss / step},
                                       i * len(data_loader) + step)

                # Summary writer (validation loss)
                if step % int(np.ceil(0.5 * len(data_loader))) == 0:
                    # Run validation step
                    print(f"\n{bcolors.WARNING}Running validation step...{bcolors.ENDC}\n")
                    for (x_validation, labels_validation) in data_loader_validation:
                        loss_validation = validation_step_reconsiren(graphdef, state, x_validation, labels_validation,
                                                                   md_columns, rng)
                        total_validation_loss += loss_validation

                        step_validation += 1

                    writer.add_scalars('Reconstruction loss (ReconSIREN)',
                                       {"validation": total_validation_loss / step_validation},
                                       i * len(data_loader) + step)

                step += 1

            if i % 5 == 0:
                # Example of predicted data for Tensorboard
                volume = decode_volume(graphdef, state)
                middle_slize = int(np.round(0.5 * volume.shape[-1]))
                ImageHandler().write(np.array(volume), os.path.join(args.output_path, "reconsiren_map_intermediate.mrc"),
                                     overwrite=True)
                slice_xy, slice_xz, slice_yz = (min_max_scale(volume[0, middle_slize, :, :]),
                                                min_max_scale(volume[0, :, middle_slize, :]),
                                                min_max_scale(volume[0, :, :, middle_slize]))
                slices = np.stack([slice_xy, slice_xz, slice_yz], axis=0)[..., None]
                writer.add_images("Predicted volume (slices)", slices, dataformats="NHWC", global_step=i)

                # Plot angular distribution
                reconsiren_intermediate, _, _ = nnx.merge(graphdef, state)
                euler_angles = np.array(reconsiren_intermediate.memory_bank.value)
                fig, _ = plot_angular_distribution(euler_angles)
                writer.add_figure("Angular distribution density", fig, global_step=i)

                # Save checkpoint model
                NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"), epoch=i)

        reconsiren, optimizer_pose, optimizer_volume = nnx.merge(graphdef, state)

        # Save model
        NeuralNetworkCheckpointer.save(reconsiren, os.path.join(args.output_path, "ReconSIREN"), mode="pickle")

        # Remove checkpoint
        shutil.rmtree(os.path.join(args.output_path, "ReconSIREN_CHECKPOINT"))

    elif args.mode == "predict":  # TODO: Save angles here

        reconsiren.eval()

        # Prepare data loader
        data_loader = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=False, preShuffle=False,
                                                  mmap=mmap, mmap_output_dir=mmap_output_dir)

        # Jitted functions for volume prediction
        decode_volume = jax.jit(reconsiren.delta_volume_decoder.decode_volume)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting angular assignment / shifts... ######")

        # For progress bar (TQDM)
        pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=",
                    colour="green")

        graphdef, state = nnx.split(reconsiren)
        md_pred = generator.md
        for (x, labels) in pbar:
            rotations, shifts = predict_angular_assignment_step_reconsiren(graphdef, state, x, labels, md_columns)

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

        md_pred.write(os.path.join(args.output_path, "predicted_pose_shifts" + os.path.splitext(args.md)[1]))

        # Predict volume
        print(f"{bcolors.OKCYAN}\n###### Predicting volume... ######")

        decoded_volume = decode_volume()
        ImageHandler().write(np.array(decoded_volume), os.path.join(args.output_path, "reconsiren_map.mrc"), overwrite=True)

if __name__ == "__main__":
    main()
