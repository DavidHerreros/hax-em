#!/usr/bin/env python


from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx

from einops import rearrange

from hax.utils import *
from hax.layers import *


def mse(a, b):
    return jnp.mean(jnp.square(a - b), axis=(-3, -2, -1))


class DeltaVolume(nnx.Module):
    """The reconstruction: a per-voxel amplitude (and optionally position) correction to a
    reference map, over the voxels the mask selects.
    """

    def __init__(self, total_voxels, volume_size, inds, reference_values, num_maps=1,
                 refine_positions=False, max_shift_voxels=0.5, *, rngs: nnx.Rngs):
        self.volume_size = volume_size
        self.inds = inds
        self.reference_values = reference_values
        self.total_voxels = total_voxels
        self.num_maps = num_maps
        self.refine_positions = refine_positions

        # Indices to (normalized) coords
        self.factor = 0.5 * volume_size
        coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)[None, ...]
        self.coords = (coords - self.factor) / self.factor

        # Scale of the amplitude correction
        ref = np.asarray(reference_values, np.float32)
        rms = float(np.sqrt(np.mean(ref ** 2)))
        self.value_scale = rms if rms > 1e-12 else 1.0

        # Half-voxel cap, expressed in the normalized coordinate units used internally
        self.max_shift = float(max_shift_voxels) / self.factor

        self.delta_values = nnx.Param(jnp.zeros((num_maps, total_voxels), jnp.float32))
        if refine_positions:
            self.delta_coords = nnx.Param(jnp.zeros((num_maps, total_voxels, 3), jnp.float32))

    def __call__(self):
        """Returns ``(coords, values, delta_values, delta_coords)``.

        The two deltas come back so the training step can penalise the *correction* rather
        than the total density -- an L1 on ``values`` shrinks the reference map itself, which
        shows up as a flat low-frequency contrast loss.
        """
        delta_values = self.value_scale * self.delta_values.value
        values = nnx.relu(self.reference_values + delta_values)

        if self.refine_positions:
            delta_coords = self.max_shift * jnp.tanh(self.delta_coords.value)
        else:
            delta_coords = jnp.zeros((self.num_maps, self.total_voxels, 3), jnp.float32)

        # Recover coords (non-normalized)
        coords = self.factor * (self.coords + delta_coords)

        return coords, values, delta_values, delta_coords

    def decode_volume(self, filter=False):
        """Render the reconstruction onto the voxel grid."""
        # Decode volume values
        coords, values, _, _ = self.__call__()

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

        # Zero the weight of any corner that falls outside the box, then clip the index
        bamp = bamp * jnp.all((bposi >= 0) & (bposi <= self.volume_size - 1), axis=-1)
        bposi = jnp.clip(bposi, 0, self.volume_size - 1)
        bamp = jnp.nan_to_num(bamp)

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

    def __call__(self, x, values, coords, xsize, rotations, shifts, ctf, ctf_type):
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

        # Same guard as the volume scatter: mask out-of-box corners before clipping, so a
        # voxel projected off one edge is dropped rather than wrapped onto the other
        bamp = bamp * jnp.all((bposi >= 0) & (bposi <= xsize - 1), axis=-1)
        bposi = jnp.clip(bposi, 0, xsize - 1)
        bamp = jnp.nan_to_num(bamp)

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        images = jax.vmap(scatter_img)(images, bposi, bamp)

        # Apply CTF
        if ctf_type in ["apply", "wiener", "squared"]:
            images = ctfFilter(images, ctf, pad_factor=2)

        return images

class MoDART(nnx.Module):
    def __init__(self, reference_volume, reconstruction_mask, xsize, sr, ctf_type="apply",
                 symmetry_group="c1", reconstruct_halves=False, refine_positions=False,
                 *, rngs: nnx.Rngs):
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
                                                num_maps=self.num_maps, refine_positions=refine_positions,
                                                rngs=rngs)
        self.phys_decoder = PhysDecoder(self.xsize)

    def __call__(self, **kwargs):
        return self.delta_volume_decoder.decode_volume(**kwargs)


@partial(jax.jit, static_argnames=("l1_delta", "tv_lambda", "coord_reg"))
def single_step_modart(graphdef, state, x, labels, md, fields_modart, key,
                       l1_delta=1e-3, tv_lambda=1e-3, coord_reg=1e-2):
    model, optimizer = nnx.merge(graphdef, state)

    # Random keys
    key, choice_key = jax.random.split(key, 2)

    # Vmap functions
    phys_decoder = jax.vmap(model.phys_decoder, in_axes=(1, 1, 1, None, 1, 1, 1, None), out_axes=1)
    wiener2DFilter_vmap = jax.vmap(wiener2DFilter, in_axes=(1, 1, None), out_axes=1)
    ctfFilter_vmap = jax.vmap(ctfFilter, in_axes=(1, 1, None), out_axes=1)

    def loss_fn(model, x):
        # Decode volume
        coords, values, delta_values, delta_coords = model.delta_volume_decoder()

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

        # Sparsity on the correction
        scale = model.delta_volume_decoder.value_scale
        l1_loss = jnp.mean(jnp.abs(delta_values)) / scale

        # L1 and L2 total variation on the density
        diff_x, diff_y, diff_z = sparse_finite_3D_differences(values, model.inds, model.xsize)
        l1_grad_loss = jnp.abs(diff_x).mean() + jnp.abs(diff_z).mean() + jnp.abs(diff_y).mean()
        l2_grad_loss = jnp.square(diff_x).mean() + jnp.square(diff_z).mean() + jnp.square(diff_y).mean()

        # Keep the position refinement honest
        coord_loss = jnp.mean(jnp.square(model.delta_volume_decoder.factor * delta_coords))

        loss = (recon_loss.mean() + l1_delta * l1_loss + tv_lambda * (l1_grad_loss + l2_grad_loss)
                + coord_reg * coord_loss)
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


@partial(jax.jit, static_argnames=("chunk_size",))
def decode_field_modart(graphdef, state, images, coords_vox, chunk_size=None):
    """The per-particle displacement field, evaluated directly at MoDART's own voxels"""
    model = nnx.merge(graphdef, state)
    if images.shape[1] != model.xsize:
        images = jax.image.resize(images, (images.shape[0], model.xsize, model.xsize, 1),
                                  method="bilinear")
    return model.decode_field_at(images, coords_vox, chunk_size=chunk_size)


@jax.jit
def decode_field_sparse(graphdef, state, images):
    """The model's field on its own points, plus those points, both in voxel units."""
    model = nnx.merge(graphdef, state)
    if images.shape[1] != model.xsize:
        images = jax.image.resize(images, (images.shape[0], model.xsize, model.xsize, 1),
                                  method="bilinear")
    field, source_coords = model.decode_field(images)
    scale = 0.5 * model.xsize
    return field * scale, source_coords * scale + scale


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
                        help="If not provided, MoDART will reconstruct a single volume. Otherwise, MoDART will reconstruct two half maps by splitting the dataset into even/odd parts, "
                             f"and report the half-map FSC. {bcolors.WARNING}NOTE{bcolors.ENDC}: without this there is no FSC and therefore no way to tell whether the reconstruction improved anything.")
    parser.add_argument("--motion_correction", type=str,
                        help=f"If provided, MoDART will perform a motion correction while reconstructing the volume to reduce motion blurring. Otherwise, a standard reconstruction is performed. "
                             f"{bcolors.WARNING} NOTE {bcolors.ENDC}: When providing this parameter, you MUST give the path to a trained {bcolors.UNDERLINE} HetSIREN (with transport of mass) "
                             f"{bcolors.ENDC} or {bcolors.UNDERLINE} Zernike3Deep {bcolors.ENDC} neural network.")
    parser.add_argument("--epochs", required=False, type=int, default=20,
                        help=f"Number of reconstruction epochs (default {bcolors.ITALIC}20{bcolors.ENDC}). The learning-rate schedule is annealed over exactly this many, so the "
                             f"fit settles at the end instead of being cut off at full step size.")
    parser.add_argument("--refine_positions", action="store_true",
                        help=f"Let MoDART move the mass as well as re-weight it. Each voxel gets a displacement bounded to half a voxel and penalised by {bcolors.ITALIC}--coord_reg{bcolors.ENDC}. "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: off by default. A free per-voxel displacement lets the model match any image by moving mass rather than by getting the "
                             f"density right, and sub-voxel jitter over every voxel is itself a blur kernel.")
    parser.add_argument("--l1_delta", required=False, type=float, default=1e-3,
                        help=f"Weight of the L1 sparsity prior on the amplitude CORRECTION (default {bcolors.ITALIC}1e-3{bcolors.ENDC}), normalised by the reference RMS. It penalises the "
                             f"departure from the reference map, not the map itself -- an L1 on the total density shrinks the reference and costs a flat slice of low-frequency contrast.")
    parser.add_argument("--tv_lambda", required=False, type=float, default=1e-3,
                        help=f"Weight of the total-variation (spatial smoothness) prior on the density (default {bcolors.ITALIC}1e-3{bcolors.ENDC}).")
    parser.add_argument("--coord_reg", required=False, type=float, default=1e-2,
                        help=f"Weight of the quadratic penalty on the per-voxel displacement (default {bcolors.ITALIC}1e-2{bcolors.ENDC}). Only used with {bcolors.ITALIC}--refine_positions{bcolors.ENDC}.")
    parser.add_argument("--field_chunk", required=False, type=int, default=131072,
                        help=f"Number of reconstruction voxels evaluated per chunk when querying the motion field (default {bcolors.ITALIC}131072{bcolors.ENDC}). Lower it if the field "
                             f"query runs out of memory; it only trades speed for peak VRAM and does not change the result.")
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

    # When neither a reference volume nor a mask is given, reconstruct a consensus volume from
    # the posed particles and take the reconstruction support from it
    auto_reference = args.vol is None and args.mask is None
    if auto_reference:
        os.makedirs(args.output_path, exist_ok=True)
        consensus = reconstruct_consensus_volume(generator.md, md_columns, args.sr,
                                                 use_ctf=args.ctf_type not in (None, "None"),
                                                 scratch_dir=(args.ssd_scratch_folder or args.output_path))
        consensus_path = os.path.join(args.output_path, "consensus_reconstruction.mrc")
        ImageHandler().write(consensus, consensus_path, overwrite=True)
        args.vol = consensus_path
        print(f"{bcolors.OKGREEN}Consensus volume reconstructed from the input poses -> {consensus_path}"
              f"{bcolors.ENDC}")

    if args.vol is not None:
        vol = ImageHandler(args.vol).getData()
    else:
        vol = np.zeros((xsize, xsize, xsize))

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    elif auto_reference:
        mask = consensus_mask(vol, dilate=4)
        ImageHandler().write(mask, os.path.join(args.output_path, "consensus_mask.mrc"), overwrite=True)
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
                  reconstruct_halves=args.reconstruct_halves, refine_positions=args.refine_positions,
                  rngs=nnx.Rngs(model_key))

    # Query points for the motion field
    field_query_coords = jnp.stack([modart.inds[:, 2], modart.inds[:, 1], modart.inds[:, 0]],
                                   axis=1).astype(jnp.float32)

    # Volume adjustment (only if reference volume is provided)
    adjust_volume = args.vol is not None and not auto_reference
    if adjust_volume:
        # Extract mask coords
        inds = np.asarray(np.where(mask > 0.0)).T
        values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]
        factor = 0.5 * vol.shape[0]
        coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
        coords = (coords - factor) / factor

        # Define volume adjustment network
        volumeAdjustment = VolumeAdjustment(lat_dim=3, coords=coords, values=values, predicts_value=True, rngs=nnx.Rngs(model_key))

    knn_field_op = None
    if args.motion_correction is not None:
        # Reload network to perform motion correction
        model = NeuralNetworkCheckpointer.load(args.motion_correction)
        graphdef_motion_correction, state_motion_correction = nnx.split(model)

        # Two ways to get the field onto the reconstruction voxels. An implicit
        # mass-transport HetSIREN is a continuous function of (latent, coordinate), so it can
        # simply be evaluated where we need it -- exact, and no interpolation stage to get
        # wrong. Anything else (Zernike3Deep, a fixed-grid HetSIREN) only defines the field on
        # its own points, so it needs an interpolator; that one is a normalised k-NN average,
        # which is bounded by construction
        dvd = getattr(model, "delta_volume_decoder", None)
        direct_field = bool(getattr(dvd, "transport_mass", False)
                            and getattr(dvd, "is_implicit", False)
                            and not getattr(dvd, "point_transformer", False))
        if direct_field:
            print(f"{bcolors.OKGREEN}Motion field evaluated directly at the {modart.inds.shape[0]} "
                  f"reconstruction voxels.{bcolors.ENDC}")
        else:
            probe = jnp.zeros((1, xsize, xsize, 1), jnp.float32)
            _, source_coords = decode_field_sparse(graphdef_motion_correction,
                                                   state_motion_correction, probe)
            knn_field_op = build_knn_field_operator(np.asarray(source_coords[0]),
                                                    np.asarray(field_query_coords))
            print(f"{bcolors.OKGREEN}Motion field interpolated from {source_coords.shape[1]} model "
                  f"points onto {modart.inds.shape[0]} reconstruction voxels (normalised k-NN)."
                  f"{bcolors.ENDC}")

    # Prepare summary writer
    writer = JaxSummaryWriter(os.path.join(args.output_path, "MoDART_metrics"))

    # Jitted volume prediction
    @jax.jit
    def get_modart_volume(graphdef, state):
        model, _ = nnx.merge(graphdef, state)
        return model(filter=False)

    get_modart_volume_full = get_modart_volume

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
                                                                           split_mode="parity",
                                                                           load_to_ram=args.load_images_to_ram)
    else:
        data_loader = generator.return_grain_dataset(batch_size=args.batch_size,  shuffle="global", num_epochs=None,
                                                     num_workers=-1, num_threads=1, split_fraction=None,
                                                     load_to_ram=args.load_images_to_ram)
    steps_per_epoch = max(1, int(len(generator.md) / args.batch_size))
    total_steps_per_epoch = steps_per_epoch

    # Example of training data for Tensorboard
    example_loader = data_loader_even if args.reconstruct_halves else data_loader
    with closing(iter(example_loader)) as iter_data_loader:
        x_example, labels_example = next(iter_data_loader)
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Example of data batch", x_example, dataformats="NHWC")

    if adjust_volume:
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
        rms_adj = float(np.sqrt(np.mean(np.asarray(values, np.float32) ** 2)))
        modart.delta_volume_decoder.value_scale = rms_adj if rms_adj > 1e-12 else 1.0

    # Learning rate scheduler, annealed over exactly the epochs that will be run.
    total_schedule_steps = args.epochs * total_steps_per_epoch
    lr_schedule = CosineAnnealingScheduler.getScheduler(peak_value=1e-3, total_steps=total_schedule_steps,
                                                        warmup_frac=0.1, init_value=0.0, end_value=0.0)

    # Early stopping
    early_stop = EarlyStopping(min_delta=1e-5, patience=3)

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
    epoch = 0
    log_every = max(1, int(np.ceil(0.1 * total_steps_per_epoch)))
    pbar = tqdm(range(total_steps_per_epoch), file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

    def fetch_batch():
        """One training batch as (images, labels), both with an explicit map axis."""
        if args.reconstruct_halves:
            x_even, labels_even = next(iter_data_loader_even)
            x_odd, labels_odd = next(iter_data_loader_odd)
            return (jnp.stack([x_even, x_odd], axis=1),
                    jnp.stack([labels_even, labels_odd], axis=1))
        x_b, labels_b = next(iter_data_loader)
        return x_b[:, None, ...], labels_b[:, None, ...]

    def motion_field(x_b):
        """The per-particle displacement at the reconstruction voxels, (B, maps, P, 3)."""
        if args.motion_correction is None:
            return jnp.zeros((x_b.shape[0], x_b.shape[1], modart.inds.shape[0], 3), jnp.float32)
        flat = jnp.reshape(x_b, (-1, x_b.shape[2], x_b.shape[3], 1))
        if knn_field_op is None:
            field = decode_field_modart(graphdef_motion_correction, state_motion_correction,
                                        flat, field_query_coords, chunk_size=args.field_chunk)
        else:
            sparse, _ = decode_field_sparse(graphdef_motion_correction,
                                            state_motion_correction, flat)
            field = apply_knn_field(sparse, *knn_field_op)
        return jnp.reshape(field, (x_b.shape[0], x_b.shape[1], field.shape[-2], 3))

    with ExitStack() as stack:
        if args.reconstruct_halves:
            iter_data_loader_even = stack.enter_context(closing(iter(data_loader_even)))
            iter_data_loader_odd = stack.enter_context(closing(iter(data_loader_odd)))
        else:
            iter_data_loader = stack.enter_context(closing(iter(data_loader)))

        while epoch < args.epochs and not early_stop.should_stop:
            total_loss = 0.0
            total_recon_loss = jnp.zeros((modart.num_maps,))
            step = 0
            pbar.reset()
            pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")

            # Intermediate volume
            if logger.should("images", epoch):
                with logger.section():
                    modart_volume = get_modart_volume(graphdef, state)
                    if args.reconstruct_halves:
                        volumes = (np.array(modart_volume[0]), np.array(modart_volume[1]))
                    else:
                        volumes = (np.array(modart_volume),)
                logger.submit(write_intermediate_modart, volumes, epoch)

            for _ in range(total_steps_per_epoch):
                x, labels = fetch_batch()
                field_modart = motion_field(x)

                rng, step_key = jax.random.split(rng)

                loss, recon_loss, state = single_step_modart(
                    graphdef, state, x, labels, md_columns, field_modart, step_key,
                    l1_delta=args.l1_delta, tv_lambda=args.tv_lambda, coord_reg=args.coord_reg)

                total_loss += loss
                total_recon_loss += recon_loss
                step += 1
                total_steps += 1

                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | "
                                     f"recon_loss={total_recon_loss.mean() / step:.5f}")

                if step % log_every == 0:
                    writer.add_scalar('Training loss (MoDART)', total_loss / step,
                                      epoch * total_steps_per_epoch + step)
                    if args.reconstruct_halves:
                        writer.add_scalars('Reconstruction loss (MoDART)',
                                           {"First half": total_recon_loss[0] / step,
                                            "Second half": total_recon_loss[1] / step},
                                           epoch * total_steps_per_epoch + step)
                    else:
                        writer.add_scalar('Reconstruction loss (MoDART)',
                                          total_recon_loss[0] / step,
                                          epoch * total_steps_per_epoch + step)
                pbar.update()

            # One early-stopping update per epoch, on the completed epoch's mean.
            epoch_loss = float(total_loss / max(step, 1))
            epoch += 1
            i = epoch
            early_stop = early_stop.update(epoch_loss)
            if early_stop.should_stop:
                print(f"{bcolors.WARNING}Reconstruction loss stopped improving after {epoch} "
                      f"epochs; finishing.{bcolors.ENDC}")

    # Let the background logging thread finish before writing the final map.
    logger.close()

    # Save final MoDART volume. Same rendering as the previews, no extra filtering: the
    # amplitudes were fitted through this renderer, so this is the map that was actually fitted.
    modart_volume = get_modart_volume_full(graphdef, state)
    if args.reconstruct_halves:
        half_a, half_b = np.array(modart_volume[0]), np.array(modart_volume[1])
        ImageHandler().write(half_a, os.path.join(args.output_path, "modart_first_half.mrc"), overwrite=True)
        ImageHandler().write(half_b, os.path.join(args.output_path, "modart_second_half.mrc"), overwrite=True)
        ImageHandler().write(np.array(0.5 * (half_a + half_b)), os.path.join(args.output_path, "modart_map.mrc"), overwrite=True)

        # The point of reconstructing halves is to be able to answer "did this improve
        # anything?". Report the FSC, and the consensus's own resolution next to it so the
        # ceiling set by the box is visible rather than implied.
        report_half_map_resolution(half_a, half_b, args.sr,
                                   label="MoDART", reference=vol if args.vol is not None else None)
    else:
        ImageHandler().write(np.array(modart_volume), os.path.join(args.output_path, "modart_map.mrc"), overwrite=True)
        print(f"{bcolors.WARNING}No half maps were reconstructed, so there is no FSC and no way to "
              f"tell whether this map is better than its reference. Re-run with "
              f"{bcolors.ITALIC}--reconstruct_halves{bcolors.ENDC}{bcolors.WARNING} to measure it."
              f"{bcolors.ENDC}")

    # If exists, clean MMAP
    # if not args.load_images_to_ram and os.path.isdir(os.path.join(mmap_output_dir, "images_mmap_grain")):
    #     shutil.rmtree(os.path.join(mmap_output_dir, "images_mmap_grain"))
