#!/usr/bin/env python


import jax
from jax import numpy as jnp
from jax.scipy.ndimage import map_coordinates
from flax import nnx
import dm_pix

from einops import rearrange

from hax.utils import *
from hax.layers import *
from hax.programs.gaussian_volume_fitting import FastVariableBlur3D


def mse(a, b):
    return jnp.mean(jnp.square(a - b), axis=(-3, -2, -1))


class DeltaVolume(nnx.Module):
    def __init__(self, total_voxels, volume_size, inds, reference_values, num_maps=1, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.inds = inds
        self.reference_values = reference_values
        self.total_voxels = total_voxels
        self.num_maps = num_maps
        # initializer = jax.nn.initializers.glorot_uniform()

        # Indices to (normalized) coords
        self.factor = 0.5 * volume_size
        coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)[None, ...]
        self.coords = (coords - self.factor) / self.factor

        if jnp.all(reference_values) == 0:
            self.lambda_parameter = 1.0
        else:
            self.lambda_parameter = nnx.Param(1e-4)

        # Learnable parameters (keep this in case it is useful in the future)
        # if self.num_maps == 1:
        #     self.params = nnx.Param(initializer(jax.random.key(0), (1, total_voxels * 4,), jnp.float32))
        # else:
        #     self.params = [nnx.Param(initializer(jax.random.key(0), (1, total_voxels * 4,), jnp.float32)),
        #                              nnx.Param(initializer(jax.random.key(1), (1, total_voxels * 4,), jnp.float32))]

        hidden_linear = [Linear(in_features=total_voxels * 3, out_features=8, rngs=rngs, dtype=jnp.bfloat16,
                                     kernel_init=siren_init_first(c=1.))]
        for _ in range(4):
            hidden_linear.append(Linear(in_features=8, out_features=8, rngs=rngs, dtype=jnp.bfloat16, kernel_init=siren_init(c=1.)))
        self.hidden_linear = nnx.List(hidden_linear)

        if self.num_maps == 1:
            self.params = Linear(in_features=8, out_features=4 * total_voxels, rngs=rngs)
        else:
            params = [Linear(in_features=8, out_features=4 * total_voxels, rngs=rngs),
                      Linear(in_features=8, out_features=4 * total_voxels, rngs=rngs)]
            self.params = nnx.List(params)

    def __call__(self):
        # (keep this in case it is useful in the future)
        # if self.num_maps == 2:
        #     params = jnp.concatenate(self.params, axis=0)
        # else:
        #     params = self.params

        coords = self.coords.flatten()[None, ...]

        # Decode voxel values
        x = jnp.sin(1.0 * self.hidden_linear[0](coords))
        for layer in self.hidden_linear[1:]:
            x = jnp.sin(x + 1.0 * layer(x))

        if self.num_maps == 1:
            params = self.params(x)
        else:
            params = jnp.concatenate([self.params[0](x), self.params[1](x)], axis=0)

        # Extract delta_coords and values
        x = jnp.reshape(self.lambda_parameter * params, (self.num_maps, self.total_voxels, 4))
        delta_coords, delta_values = x[..., :3], x[..., 3]

        # Recover volume values (TODO: Check if applying ReLu is really needed)
        values = nnx.relu(self.reference_values + delta_values)

        # Recover coords (non-normalized)
        coords = self.factor * (self.coords + delta_coords)

        return coords, values

    def decode_volume(self, filter=True):
        # Decode volume values
        coords, values = self.__call__()

        # Displace coordinates
        coords = coords + self.factor

        # Place values on grid
        grids = jnp.zeros((self.num_maps, self.volume_size, self.volume_size, self.volume_size))

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
        bposi = jnp.concat(
            [bposi, bposi + jnp.array((1, 0, 0)), bposi + jnp.array((0, 1, 0)), bposi + jnp.array((0, 0, 1)),
             bposi + jnp.array((0, 1, 1)), bposi + jnp.array((1, 0, 1)), bposi + jnp.array((1, 1, 0)),
             bposi + jnp.array((1, 1, 1))], axis=1)

        def scatter_volume(vol, bpos_i, bamp_i):
            return vol.at[bpos_i[..., 2], bpos_i[..., 1], bpos_i[..., 0]].add(bamp_i)

        grids = jax.vmap(scatter_volume, in_axes=(0, 0, 0))(grids, bposi, bamp)

        # Filter volume
        if filter:
            grids = jax.vmap(low_pass_3d)(grids)

        if self.num_maps == 1:
            return grids[0]
        else:
            return grids[0], grids[1]

class PhysDecoder:
    def __init__(self, xsize):
        self.xsize = xsize

    def __call__(self, x, values, coords, xsize, rotations, shifts, ctf, ctf_type, filter=True):
        # Volume factor
        factor = 0.5 * xsize

        # Apply rotation matrices
        coords = jnp.matmul(coords, rearrange(rotations, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts[:, None, :] + factor

        # Scatter image
        B = rotations.shape[0]
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
            images = dm_pix.gaussian_blur(images[..., None], 1.0, kernel_size=3)[..., 0]

        # Apply CTF
        if ctf_type in ["apply", "wiener", "squared"]:
            images = ctfFilter(images, ctf, pad_factor=2)

        return images

class MoDART(nnx.Module):
    def __init__(self, reference_volume, reconstruction_mask, xsize, sr, ctf_type="apply",
                 symmetry_group="c1", reconstruct_halves=False, *, rngs: nnx.Rngs):
        super(MoDART, self).__init__()
        self.xsize = xsize
        self.ctf_type = ctf_type
        self.sr = sr
        self.reference_volume = reference_volume
        self.reconstruction_mask = reconstruction_mask.astype(float)
        self.inds = jnp.asarray(jnp.where(reconstruction_mask > 0.0)).T
        self.symmetry_matrices = symmetry_matrices(symmetry_group)
        self.num_maps = 2 if reconstruct_halves else 1
        reference_values = reference_volume[self.inds[..., 0], self.inds[..., 1], self.inds[..., 2]][None, ...]
        self.delta_volume_decoder = DeltaVolume(self.inds.shape[0], self.xsize, self.inds, reference_values,
                                                num_maps=self.num_maps, rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize)

    def __call__(self, **kwargs):
        return self.delta_volume_decoder.decode_volume(**kwargs)


@jax.jit
def single_step_modart(graphdef, state, x, labels, md, fields_modart, key):
    model, optimizer = nnx.merge(graphdef, state)

    # Random keys
    key, choice_key = jax.random.split(key, 2)

    # Vmap functions
    phys_decoder = jax.vmap(model.phys_decoder, in_axes=(1, 1, 1, None, 1, 1, 1, None), out_axes=1)
    wiener2DFilter_vmap = jax.vmap(wiener2DFilter, in_axes=(1, 1, None), out_axes=1)
    ctfFilter_vmap = jax.vmap(ctfFilter, in_axes=(1, 1, None), out_axes=1)

    def loss_fn(model, x):
        # Decode volume
        coords, values = model.delta_volume_decoder()

        # Random symmetry matrices
        random_indices = jax.random.choice(choice_key, jnp.arange(model.symmetry_matrices.shape[0]), shape=(rotations.shape[0],))
        rotations_sym = jnp.matmul(jnp.transpose(model.symmetry_matrices[random_indices], (0, 2, 1))[:, None, ...], rotations)

        # Generate projections
        images_corrected = phys_decoder(x, values[None, ...], fields_modart + coords[None, ...], model.xsize, rotations_sym, shifts, ctf, model.ctf_type)

        # Losses
        images_corrected_loss = images_corrected[..., 0] if images_corrected.shape[-1] == 1 else images_corrected
        x_loss = x[..., 0] if x.shape[-1] == 1 else x

        # Consider CTF if Wiener/Squared mode (only for loss)
        if model.ctf_type == "wiener":
            x_loss = wiener2DFilter_vmap(x_loss, ctf, 2)
            images_corrected_loss = wiener2DFilter_vmap(images_corrected_loss, ctf, 2)
        elif model.ctf_type == "squared":
            x_loss = ctfFilter_vmap(x_loss, ctf, 2)
            images_corrected_loss = ctfFilter_vmap(images_corrected_loss, ctf, 2)

        recon_loss = mse(images_corrected_loss[..., None], x_loss[..., None])

        # L1 based denoising
        l1_loss = jnp.mean(jnp.abs(values))

        # L1 and L2 total variation
        diff_x, diff_y, diff_z = sparse_finite_3D_differences(values, model.inds, model.xsize)
        l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        loss = recon_loss.mean() + 0.001 * l1_loss + 0.001 * (l1_grad_loss + l2_grad_loss)
        return loss, recon_loss.mean(axis=0)

    # Labels to single batch size
    labels_broadcasted = jnp.reshape(labels, (-1, ))

    # Precompute batch aligments
    euler_angles = md["euler_angles"][labels_broadcasted]
    rotations = euler_matrix_batch(euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2])
    rotations = jnp.reshape(rotations, (labels.shape[0], labels.shape[1], 3, 3))

    # Precompute batch shifts
    shifts = md["shifts"][labels_broadcasted]
    shifts = jnp.reshape(shifts, (labels.shape[0], labels.shape[1], 2))

    # Precompute batch CTFs
    if model.ctf_type is not None:
        defocusU = md["ctfDefocusU"][labels_broadcasted]
        defocusV = md["ctfDefocusV"][labels_broadcasted]
        defocusAngle = md["ctfDefocusAngle"][labels_broadcasted]
        cs = md["ctfSphericalAberration"][labels_broadcasted]
        kv = md["ctfVoltage"][labels_broadcasted][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         model.sr, [2 * model.xsize, int(2 * 0.5 * model.xsize + 1)],
                         labels_broadcasted.shape[0], True)
    else:
        ctf = jnp.ones([labels_broadcasted.shape[0], 2 * model.xsize, int(2.0 * 0.5 * model.xsize + 1)], dtype=x.dtype)
    ctf = jnp.reshape(ctf, (labels.shape[0], labels.shape[1], ctf.shape[-2], ctf.shape[-1]))

    if model.ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter_vmap(x[..., 0], ctf, 2)[..., None]

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, recon_loss), grads = grad_fn(model, x)

    optimizer.update(model, grads)

    state = nnx.state((model, optimizer))

    return loss, recon_loss, state


@jax.jit
def interpolate_image_field(graphdef, state, images, initial_inds_modart):
    xsize = images.shape[1]
    model = nnx.merge(graphdef, state)
    map_coordinates_vmap = jax.vmap(map_coordinates, in_axes=(-1, None, None), out_axes=-1)
    factor = xsize / model.xsize
    initial_scale = 0.5 * model.xsize
    scale = 0.5 * xsize

    # Get coordinates and field
    images_resized = jax.image.resize(images, (images.shape[0], model.xsize, model.xsize, 1), method='bilinear')
    fields, initial_coords = model.decode_field(images_resized)
    # fields_values = jnp.concatenate((fields, values[..., None]), axis=-1)
    initial_coords = jnp.round(initial_scale * initial_coords + initial_scale).astype(jnp.int32)
    fields = scale * fields
    initial_inds_modart = initial_inds_modart / factor

    # Empty grids
    grids = jnp.zeros((images.shape[0], model.xsize, model.xsize, model.xsize, 3))

    # Place values on grid
    def place_on_single_grid(grid, inds, values):
        grid = grid.at[inds[..., 2], inds[..., 1], inds[..., 0], 0].set(values[..., 0])
        grid = grid.at[inds[..., 2], inds[..., 1], inds[..., 0], 1].set(values[..., 1])
        grid = grid.at[inds[..., 2], inds[..., 1], inds[..., 0], 2].set(values[..., 2])
        # grid = grid.at[inds[..., 0], inds[..., 1], inds[..., 2], 0].set(values[..., 3])
        return grid
    place_on_grids = jax.vmap(place_on_single_grid, in_axes=(0, None, 0))
    grids = place_on_grids(grids, initial_coords, fields)

    # Low pass filter grids
    gaussian_filter = FastVariableBlur3D((model.xsize, model.xsize, model.xsize))
    grids = gaussian_filter(grids, sigma=1.0)
    fields_modart = []
    for grid in grids:
        fields_modart.append(map_coordinates_vmap(grid, initial_inds_modart.T, 1))
    fields_modart = jnp.stack(fields_modart, axis=0)

    return fields_modart


def main():
    import os
    import sys
    import shutil
    from tqdm import tqdm
    import random
    import numpy as np
    import argparse
    from xmipp_metadata.image_handler import ImageHandler
    import optax
    from contextlib import closing, ExitStack
    from flax.training.early_stopping import EarlyStopping
    from hax.generators import MetaDataGenerator, extract_columns
    from hax.metrics import JaxSummaryWriter, TrainingLogger
    from hax.networks import VolumeAdjustment, train_step_volume_adjustment
    from hax.schedulers import CosineAnnealingScheduler
    from hax.checkpointer import NeuralNetworkCheckpointer

    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_vol(parser, help="If provided, MoDART will perform a refinement of this volume")
    ca.add_mask(parser,
                help=f"Determines the initial position of the mass available to MoDART to reconstruct a volume. This mask can be tight to the input volume (if provided). "
                     f"{bcolors.WARNING}WARNING{bcolors.ENDC}: The mask provided here MUST be BINARY.")
    ca.add_load_images_to_ram(parser)
    ca.add_sr(parser)
    ca.add_symmetry_group(parser)
    ca.add_ctf_type(parser)
    ca.add_batch_size(parser)
    parser.add_argument("--reconstruct_halves", action="store_true",
                        help="If not provided, MoDART will reconstruct a single volume. Otherwise, MoDART will reconstruct two half maps by splitting the dataset into even/odd parts.")
    parser.add_argument("--motion_correction", type=str,
                        help=f"If provided, MoDART will perform a motion correction while reconstructing the volume to reduce motion blurring. Otherwise, a standard reconstruction is performed. "
                             f"{bcolors.WARNING} NOTE {bcolors.ENDC}: When providing this parameter, you MUST give the path to a trained {bcolors.UNDERLINE} HetSIREN (with transport of mass) "
                             f"{bcolors.ENDC} or {bcolors.UNDERLINE} Zernike3Deep {bcolors.ENDC} neural network.")
    ca.add_output_path(parser)
    ca.add_ssd_scratch_folder(parser)
    ca.add_logging_args(parser, landscape=False, checkpoint=False)
    args = ca.parse_with_config(parser)

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Prepare grain dataset
    if not args.load_images_to_ram:
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

    # If exists, clean MMAP
    # if mmap and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap"))

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key = jax.random.split(rng, 2)

    # MoDART model
    modart = MoDART(vol, mask, xsize, args.sr, ctf_type=args.ctf_type, symmetry_group=args.symmetry_group,
                  reconstruct_halves=args.reconstruct_halves, rngs=nnx.Rngs(model_key))

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

    if args.motion_correction is not None:
        # Reload network to perform motion correction
        model = NeuralNetworkCheckpointer.load(args.motion_correction)
        graphdef_motion_correction, state_motion_correction = nnx.split(model)

    # Prepare summary writer
    writer = JaxSummaryWriter(os.path.join(args.output_path, "MoDART_metrics"))

    # Jitted functions for volume prediction
    @jax.jit
    def get_modart_volume(graphdef, state):
        model, _ = nnx.merge(graphdef, state)
        return model(filter=True)

    def write_intermediate_modart(volumes, step_idx):
        """Write the intermediate map(s) to disk and log their central slices.

        Runs on the logging thread; ``volumes`` are host-side numpy arrays so they are
        safe to hand over. One tuple entry per half (or a single entry otherwise).
        """
        specs = (((volumes[0], "modart_first_half_intermediate.mrc", "Predicted MoDART first half (slices)"),
                  (volumes[1], "modart_second_half_intermediate.mrc", "Predicted MoDART second half (slices)"))
                 if args.reconstruct_halves else
                 ((volumes[0], "modart_map_intermediate.mrc", "Predicted MoDART volume (slices)"),))
        for vol_arr, fn, tag in specs:
            ImageHandler().write(vol_arr, os.path.join(args.output_path, fn), overwrite=True)
            middle_slize = int(np.round(0.5 * vol_arr.shape[-1]))
            slice_xy, slice_xz, slice_yz = (min_max_scale(vol_arr[middle_slize, :, :]),
                                            min_max_scale(vol_arr[:, middle_slize, :]),
                                            min_max_scale(vol_arr[:, :, middle_slize]))
            slices = np.stack([slice_xy, slice_xz, slice_yz], axis=0)[..., None]
            writer.add_images(tag, slices, dataformats="NHWC", global_step=step_idx)

    # Prepare data loader
    if args.reconstruct_halves:
        data_loader_even, data_loader_odd = generator.return_grain_dataset(batch_size=int(0.5 * args.batch_size),
                                                                           shuffle="global", num_epochs=None,
                                                                           num_workers=-1, num_threads=1,
                                                                           split_fraction=[0.5, 0.5],
                                                                           load_to_ram=args.load_images_to_ram)
        steps_per_epoch_even = int(int(0.5 * len(generator.md)) / args.batch_size)
    else:
        data_loader = generator.return_grain_dataset(batch_size=args.batch_size,  shuffle="global", num_epochs=None,
                                                     num_workers=-1, num_threads=1, split_fraction=None,
                                                     load_to_ram=args.load_images_to_ram)
    steps_per_epoch = int(len(generator.md) / args.batch_size)

    # Example of training data for Tensorboard
    example_loader = data_loader_even if args.reconstruct_halves else data_loader
    with closing(iter(example_loader)) as iter_data_loader:
        x_example, labels_example = next(iter_data_loader)
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Example of data batch", x_example, dataformats="NHWC")

    if args.vol is not None:
        checkpoint_volume_adjustment = os.path.join(args.output_path, "VolumeAdjustment")
        if not os.path.isdir(checkpoint_volume_adjustment):
            # Optimizers (Volume Adjustment)
            optimizer_vol = nnx.Optimizer(volumeAdjustment, optax.adam(1e-5), wrt=nnx.Param)
            graphdef, state = nnx.split((volumeAdjustment, optimizer_vol))

            # Number epochs (volume adjustment)
            if len(generator.md) >= 1000000:
                num_epochs_vol = 5
            else:
                num_epochs_vol = 20

            data_loader_vol = generator.return_grain_dataset(batch_size=args.batch_size, shuffle="global", num_epochs=None,
                                                             num_workers=-1, num_threads=1, split_fraction=None,
                                                             load_to_ram=args.load_images_to_ram)

            # Training loop (Volume Adjustment)
            print(f"{bcolors.OKCYAN}\n###### Training volume adjustment... ######")

            i = 0
            pbar = tqdm(range(num_epochs_vol * steps_per_epoch), file=sys.stdout, ascii=" >=",
                        colour="green",
                        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

            with closing(iter(data_loader_vol)) as iter_data_loader_vol:
                for total_steps in pbar:
                    (x, labels) = next(iter_data_loader_vol)

                    if total_steps % steps_per_epoch == 0:
                        total_loss = 0

                        # For progress bar (TQDM)
                        step = 1
                        pbar.set_description(f"Epoch {int(total_steps / steps_per_epoch + 1)}/{num_epochs_vol}")

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
            NeuralNetworkCheckpointer.save(volumeAdjustment, checkpoint_volume_adjustment)
        else:
            volumeAdjustment = NeuralNetworkCheckpointer.load(checkpoint_path=checkpoint_volume_adjustment)

        values = volumeAdjustment()

        # Place values on grid and replace MoDART reference volume
        grid = jnp.zeros_like(vol)
        grid = grid.at[inds[..., 0], inds[..., 1], inds[..., 2]].set(values)
        modart.reference_volume = grid
        modart.delta_volume_decoder.reference_values = values

    # Learning rate scheduler
    total_steps_per_epoch =  steps_per_epoch if not args.reconstruct_halves else steps_per_epoch_even
    total_steps = 20 * total_steps_per_epoch
    lr_schedule = CosineAnnealingScheduler.getScheduler(peak_value=1e-3, total_steps=total_steps, warmup_frac=0.1, init_value=0.0, end_value=0.0)

    # Early stopping
    early_stop = EarlyStopping(min_delta=1e-6, patience=2. * total_steps_per_epoch)

    # Optimizers (MoDART)
    optimizer = nnx.Optimizer(modart, optax.adam(lr_schedule), wrt=nnx.Param)
    graphdef, state = nnx.split((modart, optimizer))

    # Logging cadence + background offload of the host-side logging work.
    logger = TrainingLogger(image_every=args.log_images_every,
                            steps_per_epoch=total_steps_per_epoch,
                            time_budget=args.log_time_budget,
                            background=not args.log_sync).start()

    # Reconstruction loop (MoDART)
    print(f"{bcolors.OKCYAN}\n###### Starting MoDART reconstruction... ######")

    i = 0
    total_steps = 0
    pbar = tqdm(range(steps_per_epoch), file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

    with ExitStack() as stack:
        if args.reconstruct_halves:
            iter_data_loader_even = stack.enter_context(closing(iter(data_loader_even)))
            iter_data_loader_odd = stack.enter_context(closing(iter(data_loader_odd)))
        else:
            iter_data_loader = stack.enter_context(closing(iter(data_loader)))

        while not early_stop.should_stop:

            if total_steps % steps_per_epoch == 0:

                total_loss = 0
                total_recon_loss = 0 if not args.reconstruct_halves else jnp.zeros((2, ))

                # For progress bar (TQDM)
                step = 1
                pbar.reset()
                pbar.set_description(f"Epoch {int(total_steps / steps_per_epoch + 1)}")

                # Intermediate volume
                if logger.should("images", i):
                    with logger.section():
                        modart_volume = get_modart_volume(graphdef, state)
                        if args.reconstruct_halves:
                            volumes = (np.array(modart_volume[0]), np.array(modart_volume[1]))
                        else:
                            volumes = (np.array(modart_volume),)
                    logger.submit(write_intermediate_modart, volumes, i)

                i += 1

            # For progress bar (TQDM)
            if args.reconstruct_halves:
                (x_even, labels_even) = next(iter_data_loader_even)
                (x_odd, labels_odd) = next(iter_data_loader_odd)
                x = jnp.stack([x_even, x_odd], axis=1)
                labels = jnp.stack([labels_even, labels_odd], axis=1)

                if args.motion_correction is not None:
                    x_interpolation = jnp.reshape(x, (-1, x.shape[2], x.shape[3], 1))
                    field_modart = interpolate_image_field(graphdef_motion_correction, state_motion_correction, x_interpolation, modart.inds)
                    field_modart = jnp.reshape(x, (x.shape[0], x.shape[1], field_modart.shape[1], field_modart.shape[2]))
                else:
                    field_modart = jnp.zeros((x.shape[0], modart.inds.shape[0], 3))[:, None, ...]

                loss, recon_loss, state = single_step_modart(graphdef, state, x, labels, md_columns, field_modart, rng)
                total_loss += loss
                total_recon_loss += recon_loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | recon_loss={total_recon_loss.mean() / step:.5f}")

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * steps_per_epoch_even)) == 0:
                    writer.add_scalar('Training loss (MoDART)',
                                      total_loss / step,
                                      i * steps_per_epoch_even + step)

                    writer.add_scalars('Reconstruction loss (MoDART)',
                                       {"First half": total_recon_loss[0] / step, "Second half": total_recon_loss[1] / step},
                                       i * steps_per_epoch_even + step)

                # Update early stopping criteria
                early_stop = early_stop.update(total_loss / step)
                if early_stop.should_stop:
                    print(
                        f"{bcolors.WARNING}End of reconstruction detected. Finishing reconstruction epochs...{bcolors.ENDC}")
                    break

                step += 1
            else:
                (x, labels) = next(iter_data_loader)
                if args.motion_correction is not None:
                    field_modart = interpolate_image_field(graphdef_motion_correction, state_motion_correction, x, modart.inds)
                else:
                    field_modart = jnp.zeros((x.shape[0], modart.inds.shape[0], 3))

                x = x[:, None, ...]
                labels = labels[:, None, ...]
                field_modart = field_modart[:, None, ...]

                loss, recon_loss, state = single_step_modart(graphdef, state, x, labels, md_columns, field_modart, rng)
                total_loss += loss
                total_recon_loss += recon_loss[0]

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | recon_loss={total_recon_loss / step:.5f}")

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * steps_per_epoch)) == 0:
                    writer.add_scalar('Training loss (MoDART)',
                                      total_loss / step,
                                      i * steps_per_epoch + step)

                    writer.add_scalar('Reconstruction loss (MoDART)',
                                      total_recon_loss / step,
                                      i * steps_per_epoch + step)

                # Update early stopping criteria
                early_stop = early_stop.update(total_loss / step)
                if early_stop.should_stop:
                    print(f"{bcolors.WARNING}End of reconstruction detected. Finishing reconstruction epochs...{bcolors.ENDC}")
                    break

                step += 1

            pbar.update()
            total_steps += 1

    # Let the background logging thread finish before writing the final map.
    logger.close()

    # Save final MoDART volume
    modart_volume = get_modart_volume(graphdef, state)
    if args.reconstruct_halves:
        ImageHandler().write(np.array(modart_volume[0]), os.path.join(args.output_path, "modart_first_half.mrc"), overwrite=True)
        ImageHandler().write(np.array(modart_volume[1]), os.path.join(args.output_path, "modart_second_half.mrc"), overwrite=True)
        ImageHandler().write(np.array(0.5 * (modart_volume[0] + modart_volume[1])), os.path.join(args.output_path, "modart_map.mrc"), overwrite=True)
    else:
        ImageHandler().write(np.array(modart_volume), os.path.join(args.output_path, "modart_map.mrc"), overwrite=True)

    # If exists, clean MMAP
    # if not args.load_images_to_ram and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap_grain")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap_grain"))
