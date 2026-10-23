#!/usr/bin/env python


from functools import partial

import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx

from hax.utils import *
from hax.utils.reconstruction import _fsc_filter


def warped_detector_coords(vox, field, rotations, shifts, xsize):
    """(row, col) on the 2x oversampled detector of the warped and rotated voxels"""
    coords = jnp.matmul(vox[None, ...] + field, jnp.swapaxes(rotations, -1, -2))
    q = coords[..., :2] - shifts[:, None, :] + 0.5 * xsize
    return 2.0 * jnp.stack([q[..., 1], q[..., 0]], axis=-1)


def bilinear_corners(q, size):
    """Flat indices and weights (B, C, 4) of the detector pixels around each point; corners off the detector weigh zero"""
    base = jnp.floor(q)
    t = q - base
    base = base.astype(jnp.int32)
    idx, w = [], []
    for dr, dc in ((0, 0), (1, 0), (0, 1), (1, 1)):
        r, c = base[..., 0] + dr, base[..., 1] + dc
        inside = (r >= 0) & (r < size) & (c >= 0) & (c < size)
        idx.append(jnp.clip(r, 0, size - 1) * size + jnp.clip(c, 0, size - 1))
        w.append(jnp.where(inside, (t[..., 0] if dr else 1.0 - t[..., 0]) * (t[..., 1] if dc else 1.0 - t[..., 1]), 0.0))
    return jnp.stack(idx, axis=-1), jnp.stack(w, axis=-1)


def knn_field_chunk(fields, idx, weights):
    """Displacement (B, C, 3) at C voxels from the field (B, G, 3) on the model points"""
    field = 0.0
    for j in range(idx.shape[1]):
        field = field + weights[None, :, j, None] * fields[:, idx[:, j], :]
    return field


def splat_deapodization(xsize):
    """Inverse transfer of the bilinear splat on the 2x oversampled detector, on the rfft2 grid of the detector"""
    fy, fx = np.fft.fftfreq(xsize), np.fft.rfftfreq(xsize)
    return jnp.asarray(1.0 / (np.sinc(0.5 * fy)[:, None] ** 2 * np.sinc(0.5 * fx)[None, :] ** 2), jnp.float32)


def detector_pipeline(img_os, deapod, ctf, ctf_type, xsize):
    """Oversampled splat -> detector (Fourier crop, splat kernel divided out) -> CTF"""
    ft = jnp.fft.rfft2(img_os)
    ft = jnp.concatenate([ft[:, :(xsize + 1) // 2, :xsize // 2 + 1], ft[:, ft.shape[1] - xsize // 2:, :xsize // 2 + 1]], axis=1)
    img = jnp.fft.irfft2(ft * deapod, s=(xsize, xsize))
    if ctf_type in ("apply", "wiener"):
        img = ctfFilter(img, ctf, pad_factor=2)
    return img


def warped_corners(c, chunk, vox, knn_idx, knn_w, fields, rotations, shifts, xsize):
    sl = lambda a: jax.lax.dynamic_slice_in_dim(a, c * chunk, chunk, axis=0)
    field = 0.0 if fields is None else knn_field_chunk(fields, sl(knn_idx), sl(knn_w))
    return bilinear_corners(warped_detector_coords(sl(vox), field, rotations, shifts, xsize), 2 * xsize)


@partial(jax.jit, static_argnames=("xsize", "chunk", "ctf_type"))
def project_maps(X, half, vox, knn_idx, knn_w, fields, rotations, shifts, ctf, deapod, xsize, chunk, ctf_type):
    """A X: the projection of each particle's map (X[half]) through its motion, pose and CTF"""
    B, size = rotations.shape[0], 2 * xsize
    offsets = (jnp.arange(B) * size * size)[:, None, None]

    def splat(c, img):
        idx, w = warped_corners(c, chunk, vox, knn_idx, knn_w, fields, rotations, shifts, xsize)
        values = jax.lax.dynamic_slice_in_dim(X, c * chunk, chunk, axis=1)[half]
        return img.at[(idx + offsets).ravel()].add((w * values[..., None]).ravel())

    img_os = jax.lax.fori_loop(0, vox.shape[0] // chunk, splat, jnp.zeros(B * size * size))
    return detector_pipeline(img_os.reshape(B, size, size), deapod, ctf, ctf_type, xsize)


@partial(jax.jit, static_argnames=("xsize", "chunk", "ctf_type", "forward"))
def backproject_residual(X, data, valid, half, vox, knn_idx, knn_w, fields, rotations, shifts, ctf, deapod,
                         xsize, chunk, ctf_type, forward=True):
    """A^T (data - A X), one row per map; the adjoint is the warped back-projection of the residual"""
    B, size = rotations.shape[0], 2 * xsize
    residual = data
    if forward:
        residual = data - project_maps(X, half, vox, knn_idx, knn_w, fields, rotations, shifts, ctf, deapod,
                                       xsize, chunk, ctf_type)
    residual = valid[:, None, None] * residual

    pipeline = lambda img: detector_pipeline(img, deapod, ctf, ctf_type, xsize)
    _, pipeline_t = jax.vjp(pipeline, jnp.zeros((B, size, size)))
    residual = pipeline_t(residual)[0].reshape(B, size * size)
    maps = jax.nn.one_hot(half, X.shape[0]).T

    def backproject(c, out):
        idx, w = warped_corners(c, chunk, vox, knn_idx, knn_w, fields, rotations, shifts, xsize)
        values = jnp.sum(w * jnp.take_along_axis(residual, idx.reshape(B, -1), axis=1).reshape(idx.shape), axis=-1)
        return jax.lax.dynamic_update_slice_in_dim(out, maps @ values, c * chunk, axis=1)

    return jax.lax.fori_loop(0, vox.shape[0] // chunk, backproject, jnp.zeros_like(X))


def half_map_fsc(vol_a, vol_b, support, seed=0):
    """Half-map FSC under a soft support, corrected for the support edge by phase randomisation (Chen et al. 2013)"""
    from scipy.ndimage import binary_dilation, gaussian_filter
    from scipy.fft import fftn, ifftn
    box = vol_a.shape[0]
    n = box // 2 + 1
    w = np.clip(gaussian_filter(binary_dilation(support, iterations=2).astype(np.float64), 2.0), 0.0, 1.0)
    f = np.fft.fftfreq(box)
    shells = np.round(np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2) * box).astype(int).ravel()

    def fsc(a, b):
        A, B = fftn(a * w, workers=-1).ravel(), fftn(b * w, workers=-1).ravel()
        s = lambda v: np.bincount(shells, v, minlength=n)[:n]
        return s(np.real(A * np.conj(B))) / np.sqrt(s(np.abs(A) ** 2) * s(np.abs(B) ** 2) + 1e-30)

    raw = fsc(vol_a, vol_b)
    k0 = int(np.argmax(raw < 0.8)) if np.any(raw < 0.8) else n
    rng = np.random.default_rng(seed)

    def randomise(v):
        F = fftn(v, workers=-1)
        phase = np.angle(fftn(rng.standard_normal(v.shape), workers=-1))
        return np.real(ifftn(np.where(shells.reshape(v.shape) > k0, np.abs(F) * np.exp(1j * phase), F), workers=-1))

    rand = fsc(randomise(vol_a), randomise(vol_b))
    corrected = np.where(np.arange(n) > k0, (raw - rand) / (1.0 - rand + 1e-9), raw)

    # The correction is unstable for a few shells past the randomisation start: bridge them
    gap = np.arange(k0 + 1, min(k0 + 5, n - 1))
    if gap.size:
        corrected[gap] = np.interp(gap, [k0, gap[-1] + 1], corrected[[k0, gap[-1] + 1]])
    return corrected


@partial(jax.jit, static_argnames=("xsize", "ctf_type"))
def batch_geometry(md, labels, rot_refine, shift_refine, sym, sr, xsize, ctf_type):
    """Rotations, shifts and CTFs of a batch, repeated once per symmetry operator"""
    euler_angles = md["euler_angles"][labels]
    rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    rotations = jnp.matmul(rotations, rot_refine)
    shifts = md["shifts"][labels] + shift_refine
    # Copy g is posed R S_g: the projector maps a voxel v to R v
    rotations = jnp.einsum("bij,gjk->gbik", rotations, sym).reshape(-1, 3, 3)
    shifts = jnp.tile(shifts, (sym.shape[0], 1))

    ctf = None
    if ctf_type not in (None, "None"):
        ctf = computeCTF(md["ctfDefocusU"][labels], md["ctfDefocusV"][labels], md["ctfDefocusAngle"][labels],
                         md["ctfSphericalAberration"][labels], md["ctfVoltage"][labels][0], sr,
                         [2 * xsize, xsize + 1], labels.shape[0], True)
    return rotations, shifts, ctf


def symmetry_field_operator(sources, vox, sym):
    """k-NN operator from the model points to the voxels for every symmetry copy (copy-major, then one block per point set).
    Copy g needs the field at S_g u, read from the points rotated by S_g^T"""
    operators = [build_knn_field_operator(np.asarray(source) @ np.asarray(s), vox) for s in sym for source in sources]
    offsets = np.cumsum([0] + [np.asarray(source).shape[0] for _ in sym for source in sources][:-1])
    knn_idx = jnp.concatenate([idx + offset for (idx, _), offset in zip(operators, offsets)], axis=1)
    knn_w = jnp.concatenate([w for _, w in operators], axis=1)
    return knn_idx, knn_w


def symmetry_copy_fields(fields, sym):
    """Fields of the symmetry copies (copy-major, as batch_geometry). Copy g, posed R S_g, moves voxel u by S_g^T f(S_g u),
    so that it lands at R (S_g u + f(S_g u)); each copy's field sits on its own block of symmetry_field_operator"""
    rotated = jnp.einsum("bpi,gij->gbpj", fields, sym)
    blocks = jnp.einsum("gbpj,gh->gbhpj", rotated, jnp.eye(sym.shape[0], dtype=fields.dtype))
    return blocks.reshape(sym.shape[0] * fields.shape[0], sym.shape[0] * fields.shape[1], 3)


@jax.jit
def encode_particles(graphdef, state, images):
    """Latents and refined poses of the motion model"""
    model = nnx.merge(graphdef, state)
    images = jax.image.resize(images, (images.shape[0], model.xsize, model.xsize, 1), method="bilinear")
    latents, (rotations, shifts) = model(images)
    return latents, rotations, shifts


@partial(jax.jit, static_argnames=("xsize",))
def decode_fields(graphdef, state, latents, xsize):
    """Field on the model points and those points, centred voxel units of the reconstruction box"""
    model = nnx.merge(graphdef, state)
    fields, coords = model.decode_field(latents)
    return 0.5 * xsize * fields, 0.5 * xsize * jnp.reshape(coords, (-1, 3))


def main():
    import os
    import sys
    import time
    from tqdm import tqdm
    import argparse
    from xmipp_metadata.image_handler import ImageHandler
    from hax.generators import MetaDataGenerator, extract_columns
    from hax.metrics import JaxSummaryWriter
    from hax.checkpointer import NeuralNetworkCheckpointer

    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_vol(parser, help="If provided, MoDART starts from this volume, after matching its contrast to the images")
    ca.add_mask(parser,
                help=f"Determines the voxels MoDART reconstructs. If not provided, it is derived from --vol, or from a consensus reconstruction of the particles. "
                     f"{bcolors.WARNING}WARNING{bcolors.ENDC}: The mask provided here MUST be BINARY.")
    ca.add_load_images_to_ram(parser)
    ca.add_sr(parser)
    ca.add_symmetry_group(parser)
    ca.add_ctf_type(parser)
    ca.add_batch_size(parser)
    parser.add_argument("--reconstruct_halves", action="store_true",
                        help="Also write the two unfiltered half maps (even/odd particles). They are always reconstructed: their FSC filters the final map.")
    parser.add_argument("--motion_correction", type=str,
                        help=f"If provided, MoDART will perform a motion correction while reconstructing the volume to reduce motion blurring. Otherwise, a standard reconstruction is performed. "
                             f"{bcolors.WARNING} NOTE {bcolors.ENDC}: When providing this parameter, you MUST give the path to a trained {bcolors.UNDERLINE} HetSIREN (with transport of mass) "
                             f"{bcolors.ENDC} or {bcolors.UNDERLINE} Zernike3Deep {bcolors.ENDC} neural network.")
    parser.add_argument("--iterations", required=False, type=int, default=10,
                        help=f"Conjugate gradient iterations, one pass over the particles each (default {bcolors.ITALIC}10{bcolors.ENDC}). More iterations recover finer detail and also fit more noise.")
    parser.add_argument("--regularization", required=False, type=float, default=0.0,
                        help=f"Tikhonov weight, relative to the mean diagonal of the normal equations (default {bcolors.ITALIC}0{bcolors.ENDC}).")
    parser.add_argument("--chunk_size", required=False, type=int, default=65536,
                        help=f"Voxels projected at a time (default {bcolors.ITALIC}65536{bcolors.ENDC}). Lower it if the GPU runs out of memory; it does not change the result.")
    ca.add_output_path(parser)
    ca.add_ssd_scratch_folder(parser)
    ca.add_logging_args(parser, landscape=False, checkpoint=False)
    args = ca.parse_with_config(parser)

    batch_size = 8 if args.batch_size == "auto" else int(args.batch_size)
    ctf_type = None if args.ctf_type in (None, "None") else args.ctf_type
    if ctf_type == "wiener":
        print(f"{bcolors.WARNING}--ctf_type wiener: the CTF is part of the least-squares model, which already inverts it; using apply.{bcolors.ENDC}")
    os.makedirs(args.output_path, exist_ok=True)

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)
    n_particles = len(generator.md)
    xsize = generator.md.getMetaDataImage(0).shape[0]

    # Images: in RAM, or streamed from ArrayRecord shards
    if args.load_images_to_ram:
        images_ram = generator.load_images_to_ram()
    else:
        mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                             precision=np.float16, group_size=1, shard_size=10000)

    def image_batches():
        if args.load_images_to_ram:
            for s in range(0, n_particles, batch_size):
                yield images_ram[s:s + batch_size], np.arange(s, min(s + batch_size, n_particles))
        else:
            loader = generator.return_grain_dataset(batch_size=batch_size, shuffle=False, num_epochs=1, num_workers=2,
                                                    load_to_ram=False)
            for x, labels in loader:
                yield np.asarray(x, np.float32), np.asarray(labels)

    # Volume and mask
    vol = ImageHandler(args.vol).getData() if args.vol is not None else None
    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    else:
        reference = vol
        if reference is None:
            reference = reconstruct_consensus_volume(generator.md, md_columns, args.sr, use_ctf=ctf_type is not None, batch_size=256,
                                                     scratch_dir=args.ssd_scratch_folder or args.output_path)
            ImageHandler().write(reference, os.path.join(args.output_path, "consensus_reconstruction.mrc"), overwrite=True)
        mask = consensus_mask(reference)
        ImageHandler().write(mask, os.path.join(args.output_path, "consensus_mask.mrc"), overwrite=True)
        print(f"{bcolors.OKGREEN}Reconstruction mask derived from the {'input' if vol is not None else 'consensus'} volume: "
              f"{int(mask.sum())} voxels{bcolors.ENDC}")

    # Reconstruction voxels (centred, x-y-z), padded to a whole number of chunks
    inds = np.argwhere(mask > 0.0)
    n_vox = inds.shape[0]
    chunk = int(min(args.chunk_size, int(2 ** np.ceil(np.log2(n_vox)))))
    n_pad = int(np.ceil(n_vox / chunk) * chunk)
    vox = np.full((n_pad, 3), 1e6, np.float32)
    vox[:n_vox] = inds[:, ::-1] - 0.5 * xsize
    vox = jnp.asarray(vox)
    support = jnp.asarray(np.arange(n_pad) < n_vox, jnp.float32)
    iz, iy, ix = (jnp.asarray(inds[:, k]) for k in range(3))
    num_maps = 2
    sym = jnp.asarray(symmetry_matrices(args.symmetry_group), jnp.float32)
    deapod = splat_deapodization(xsize)

    # Motion model: latents and refined poses once, the field from the latents in every pass
    knn_idx = knn_w = None
    latents = rot_refine = shift_refine = None
    if args.motion_correction is not None:
        model = NeuralNetworkCheckpointer.load(args.motion_correction)
        model.eval()
        graphdef_mc, state_mc = nnx.split(model)
        model_xsize = model.xsize
        del model

        print(f"{bcolors.OKCYAN}\n###### Encoding particles with the motion model... ######{bcolors.ENDC}")
        # Stored by label: the streamed loader does not return particles in metadata order
        for x, labels in tqdm(image_batches(), total=int(np.ceil(n_particles / batch_size)), file=sys.stdout,
                              ascii=" >=", colour="green", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            z, r, s = encode_particles(graphdef_mc, state_mc, jnp.asarray(x))
            if latents is None:
                latents = np.zeros((n_particles, z.shape[-1]), np.float32)
                rot_refine = np.zeros((n_particles, 3, 3), np.float32)
                shift_refine = np.zeros((n_particles, 2), np.float32)
            latents[labels], rot_refine[labels], shift_refine[labels] = np.asarray(z), np.asarray(r), np.asarray(s)
        shift_refine = shift_refine * (xsize / model_xsize)

        _, source = decode_fields(graphdef_mc, state_mc, jnp.asarray(latents[:1]), xsize)
        knn_idx, knn_w = symmetry_field_operator([source], np.asarray(vox[:n_vox]), sym)
        knn_idx = jnp.pad(knn_idx, ((0, n_pad - n_vox), (0, 0)))
        knn_w = jnp.pad(knn_w, ((0, n_pad - n_vox), (0, 0)))
        print(f"{bcolors.OKGREEN}Motion field carried from {source.shape[0]} model points to {n_vox} voxels; "
              f"refined poses: {np.rad2deg(np.mean(np.arccos(np.clip((np.trace(rot_refine, axis1=1, axis2=2) - 1) / 2, -1, 1)))):.2f} deg mean, "
              f"{np.sqrt(np.mean(shift_refine ** 2)):.2f} px RMS shift{bcolors.ENDC}")

    def geometry(labels):
        """Per-batch arrays, padded to the batch size and repeated per symmetry operator"""
        n = labels.shape[0]
        labels_p = np.pad(labels, (0, batch_size - n))
        valid = np.tile(np.arange(batch_size) < n, sym.shape[0]).astype(np.float32)
        half = np.tile(labels_p % 2, sym.shape[0])
        if latents is None:
            rr, ss, fields = np.tile(np.eye(3, dtype=np.float32), (batch_size, 1, 1)), np.zeros((batch_size, 2), np.float32), None
        else:
            rr, ss = rot_refine[labels_p], shift_refine[labels_p]
            fields, _ = decode_fields(graphdef_mc, state_mc, jnp.asarray(latents[labels_p]), xsize)
            fields = symmetry_copy_fields(fields, sym)
        rotations, shifts, ctf = batch_geometry(md_columns, jnp.asarray(labels_p), jnp.asarray(rr), jnp.asarray(ss), sym,
                                                args.sr, xsize, ctf_type)
        ctf_rep = None if ctf is None else jnp.tile(ctf, (sym.shape[0], 1, 1))
        return dict(valid=jnp.asarray(valid), half=jnp.asarray(half), fields=fields, rotations=rotations, shifts=shifts,
                    ctf=ctf_rep), ctf

    def data_term(x, ctf):
        x = jnp.asarray(np.pad(x[..., 0], ((0, batch_size - x.shape[0]), (0, 0), (0, 0))))
        if ctf_type == "precorrect":
            x = wiener2DFilter(x, ctf, 2)
        return jnp.tile(x, (sym.shape[0], 1, 1))

    common = dict(vox=vox, knn_idx=knn_idx, knn_w=knn_w, deapod=deapod, xsize=xsize, chunk=chunk, ctf_type=ctf_type)

    # Starting map, contrast matched to the images as in HetSIREN / Zernike3Deep (median of per-image slopes)
    X0 = jnp.zeros((num_maps, n_pad))
    if vol is not None:
        x0 = jnp.pad(jnp.asarray(vol[inds[:, 0], inds[:, 1], inds[:, 2]], jnp.float32), (0, n_pad - n_vox))
        X0 = jnp.tile(x0[None], (num_maps, 1))
        slopes, n_target = [], min(8192, n_particles)
        for x, labels in image_batches():
            g, ctf = geometry(labels)
            p = project_maps(X0, g["half"], fields=g["fields"], rotations=g["rotations"], shifts=g["shifts"], ctf=g["ctf"], **common)
            p, t = np.asarray(p[:labels.shape[0]]), np.asarray(data_term(x, ctf)[:labels.shape[0]])
            p, t = p - p.mean((1, 2), keepdims=True), t - t.mean((1, 2), keepdims=True)
            var_p = np.mean(p * p, (1, 2))
            slopes.append(np.where(var_p > 0, np.mean(p * t, (1, 2)) / np.maximum(var_p, 1e-30), np.nan))
            if sum(len(s) for s in slopes) >= n_target:
                break
        slopes = np.concatenate(slopes)
        scale = max(float(np.median(slopes[np.isfinite(slopes)])), 0.0) if np.any(np.isfinite(slopes)) else 1.0
        X0 = scale * X0
        print(f"{bcolors.OKGREEN}Contrast scale of the starting volume: {scale:.4f} (median of {int(np.isfinite(slopes).sum())} images){bcolors.ENDC}")

    # Preconditioner: |k| / (detector response power + eps), the inverse of the normal operator for even pose coverage
    first_labels = np.arange(min(batch_size, n_particles))
    g, _ = geometry(first_labels)
    delta = jnp.zeros((g["rotations"].shape[0], 2 * xsize, 2 * xsize)).at[:, xsize, xsize].set(1.0)
    power = np.mean(np.abs(np.fft.rfft2(np.asarray(detector_pipeline(delta, deapod, g["ctf"], ctf_type, xsize)))) ** 2, axis=0)
    f2 = np.sqrt(np.fft.fftfreq(xsize)[:, None] ** 2 + np.fft.rfftfreq(xsize)[None, :] ** 2)
    shell2 = np.round(f2 * xsize).astype(int)
    profile = np.bincount(shell2.ravel(), power.ravel()) / np.maximum(np.bincount(shell2.ravel()), 1)
    fz = np.fft.fftfreq(xsize)
    f3 = np.sqrt(fz[:, None, None] ** 2 + fz[None, :, None] ** 2 + np.fft.rfftfreq(xsize)[None, None, :] ** 2)
    response = profile[np.minimum(np.round(f3 * xsize).astype(int), profile.shape[0] - 1)]
    weight = jnp.asarray(np.maximum(f3, 1.0 / xsize) / (response + 0.05 * profile.max()), jnp.float32)

    @jax.jit
    def precondition(v):
        grid = jnp.zeros((num_maps, xsize, xsize, xsize)).at[:, iz, iy, ix].set(v[:, :n_vox])
        grid = jnp.fft.irfftn(jnp.fft.rfftn(grid, axes=(1, 2, 3)) * weight, s=(xsize,) * 3, axes=(1, 2, 3))
        return jnp.pad(grid[:, iz, iy, ix], ((0, 0), (0, n_pad - n_vox)))

    counts = np.bincount(np.arange(n_particles) % num_maps, minlength=num_maps)
    lam = jnp.asarray(args.regularization * counts * sym.shape[0] * float(np.mean(power)), jnp.float32)[:, None]

    def to_volumes(X):
        X = np.asarray(X)
        vols = np.zeros((num_maps, xsize, xsize, xsize), np.float32)
        vols[:, inds[:, 0], inds[:, 1], inds[:, 2]] = X[:, :n_vox]
        return vols

    writer = JaxSummaryWriter(os.path.join(args.output_path, "MoDART_metrics"))
    f = np.fft.fftfreq(xsize)
    shells = np.minimum(np.round(np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2) * xsize).astype(int), xsize // 2)

    def write_maps(X, suffix=""):
        """Half maps (if asked) and their average weighted by sqrt(2 FSC / (1 + FSC)); returns the half-map FSC"""
        from scipy.fft import fftn, ifftn
        vols = to_volumes(X)
        if args.reconstruct_halves:
            ImageHandler().write(vols[0], os.path.join(args.output_path, f"modart_first_half{suffix}.mrc"), overwrite=True)
            ImageHandler().write(vols[1], os.path.join(args.output_path, f"modart_second_half{suffix}.mrc"), overwrite=True)
        fsc = half_map_fsc(vols[0], vols[1], mask > 0.0)
        filtered = np.real(ifftn(fftn(vols.mean(0), workers=-1) * _fsc_filter(fsc)[shells], workers=-1)).astype(np.float32)
        ImageHandler().write(filtered, os.path.join(args.output_path, f"modart_map{suffix}.mrc"), overwrite=True)
        return fsc

    def normal_pass(desc, P):
        """-(A^T A) P over all particles, from the labels alone"""
        out = jnp.zeros_like(P)
        zeros = jnp.zeros((batch_size * sym.shape[0], xsize, xsize))
        for s in tqdm(range(0, n_particles, batch_size), desc=desc, file=sys.stdout, ascii=" >=", colour="green",
                      bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            g, _ = geometry(np.arange(s, min(s + batch_size, n_particles)))
            out = out + backproject_residual(P, zeros, **g, **common)
        return out

    # Right-hand side: A^T (y - A X0)
    print(f"{bcolors.OKCYAN}\n###### Starting MoDART reconstruction (conjugate gradient)... ######{bcolors.ENDC}")
    t0 = time.time()
    R = jnp.zeros((num_maps, n_pad))
    for x, labels in tqdm(image_batches(), total=int(np.ceil(n_particles / batch_size)), desc="Back-projection",
                          file=sys.stdout, ascii=" >=", colour="green", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
        g, ctf = geometry(labels)
        R = R + backproject_residual(X0, data_term(x, ctf), **g, **common, forward=vol is not None)
    R = (R - lam * X0) * support

    # Preconditioned conjugate gradient on (A^T A + lam) X = A^T y, one map per row
    X = X0
    Z = precondition(R) * support
    P = Z
    rz = jnp.sum(R * Z, axis=1)
    r0 = jnp.sqrt(jnp.sum(R * R, axis=1))
    for it in range(args.iterations):
        HP = (lam * P - normal_pass(f"Iteration {it + 1}/{args.iterations}", P)) * support
        curvature = jnp.sum(P * HP, axis=1)
        alpha = jnp.where(curvature > 0, rz / jnp.where(curvature > 0, curvature, 1.0), 0.0)
        X = X + alpha[:, None] * P
        R = R - alpha[:, None] * HP
        Z = precondition(R) * support
        rz_new = jnp.sum(R * Z, axis=1)
        P = Z + jnp.where(rz > 0, rz_new / jnp.where(rz > 0, rz, 1.0), 0.0)[:, None] * P
        rz = rz_new

        rel = np.asarray(jnp.sqrt(jnp.sum(R * R, axis=1)) / r0)
        writer.add_scalar("Relative residual (MoDART)", float(rel.mean()), it + 1)
        write_maps(X, "_intermediate")
        print(f"{bcolors.OKGREEN}Iteration {it + 1}: relative residual {', '.join(f'{v:.3e}' for v in rel)} ({time.time() - t0:.0f} s){bcolors.ENDC}")

    fsc = write_maps(X)
    freq = np.arange(fsc.shape[0]) / (xsize * args.sr)
    np.savetxt(os.path.join(args.output_path, "half_map_fsc.txt"), np.stack([freq, fsc], axis=1), header="1/A FSC")
    below = np.flatnonzero(fsc[1:] < 0.143)
    print(f"{bcolors.OKGREEN}Half-map FSC = 0.143 at {1.0 / freq[below[0] + 1] if below.size else 2.0 * args.sr:.2f} A (mask-corrected); "
          f"modart_map.mrc is the FSC-weighted average{bcolors.ENDC}")
    for name in ("modart_map", "modart_first_half", "modart_second_half"):
        path = os.path.join(args.output_path, f"{name}_intermediate.mrc")
        if os.path.isfile(path):
            os.remove(path)
