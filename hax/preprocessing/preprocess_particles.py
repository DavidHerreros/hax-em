#!/usr/bin/env python
"""Downsample / crop a particle stack and (optionally) correct it for the CTF.

Running the heterogeneity networks on a smaller box is the cheapest way to cut
VRAM and training time, at the price of resolution. This program materialises
that trade-off once, up front, so every downstream run reads an already-small
stack instead of resizing on the fly.

Order of operations (each step is optional):

1. ``--crop_box_size`` -- real-space centred crop/pad. Trims the field of view;
   the sampling rate is untouched, and so are the in-plane shifts (a centred crop
   preserves the box centre they are measured against).
2. ``--new_box_size``  -- Fourier crop/pad. Changes the sampling rate by
   ``box_cropped / box_resized``; the in-plane shifts, being in *pixels*, are
   rescaled by the inverse of that factor.
3. ``--ctf_correction`` -- Wiener deconvolution or phase flipping.

Doing the CTF step *after* the resize is not an approximation: a CTF filter is
diagonal in Fourier space, so it commutes with the spectral truncation that
Fourier cropping performs. Correcting on the small box gives the same result on
the retained band, with a much cheaper FFT, and the CTF oscillations beyond the
new Nyquist frequency are exactly the ones the resize discards anyway.

The three stages -- read, compute, write -- are overlapped: a thread pool reads
batches ahead of the device, the device processes one batch at a time, and a
second thread pool streams results into a memory-mapped output stack. Only a few
batches are ever resident in RAM, so a 10^6-particle stack costs no more memory
than a 10^4-particle one.
"""

import os
import sys
from collections import deque

import jax
import numpy as np
from jax import numpy as jnp

from hax.utils import (bcolors, centered_crop_or_pad, computeCTF, ctfFilter, estimate_batch_size_from_peak_fn,
                       fourier_resample, wiener2DFilter)


# The zero-padding factor hax uses whenever a CTF is evaluated or applied. It
# must match the one baked into ``wiener2DFilter`` / ``ctfFilter`` defaults,
# because ``computeCTF`` is asked for the spectrum of the *padded* image.
PAD_FACTOR = 2

OUTPUT_STACK_NAME = "preprocessed_particles.mrcs"
OUTPUT_MD_STEM = "preprocessed_particles"

# Batch size used when ``--batch_size auto`` cannot be resolved (CPU backend, or a
# device that does not report memory stats).
FALLBACK_BATCH_SIZE = 1024

# Ceiling on the batch ``auto`` may pick. Measured device throughput for this kernel
# plateaus around a batch of 1024 images, and the surrounding loop is bound by disk
# reads anyway, so a larger batch only inflates the read-ahead window (see
# READ_AHEAD_BUDGET_BYTES) and the compile time without going faster.
MAX_AUTO_BATCH_SIZE = 2048

# Fraction of device memory ``auto`` budgets for. XLA's ``memory_analysis`` reports an
# idealized buffer assignment that measured ~10-20% below this kernel's real allocator
# peak, so a 0.7 budget lands near 85% of the card -- uncomfortably tight if anything
# else shares the GPU. Because throughput is flat past ~1024 images, buying that
# headroom back costs essentially no wall time.
AUTO_BATCH_MEMORY_SAFETY = 0.6

# Ceiling on the *host* memory held by batches read ahead of the device. The device
# budget alone does not bound this: a large (or auto-selected) batch multiplied by
# one in-flight batch per reader thread is what would actually exhaust RAM, and it
# scales with the *input* box, which may be much bigger than the output one.
READ_AHEAD_BUDGET_BYTES = 2 * 1024 ** 3


def preprocess_particle_batch(images, defocusU, defocusV, defocusAngle, cs, kv, sr_out,
                              crop_size=None, new_size=None, ctf_correction="none",
                              wiener_epsilon=None):
    """Crop, resize and CTF-correct one batch of particles.

    :param images: ``(B, H, W)`` real images.
    :param defocusU, defocusV, defocusAngle, cs: per-particle CTF terms, ``(B,)``.
    :param kv: acceleration voltage (scalar; hax assumes it is constant).
    :param sr_out: sampling rate of the *output* box, in Angstrom/pixel.
    :param crop_size: real-space box size after cropping, or None to skip.
    :param new_size: box size after Fourier resizing, or None to skip.
    :param ctf_correction: ``none`` | ``wiener`` | ``phase_flip``.
    :param wiener_epsilon: fixed Wiener regularizer; None selects the adaptive,
        per-image spectral-SNR regularizer of :func:`hax.utils.wiener2DFilter`.
    :return: ``(B, box, box)`` preprocessed images.
    """
    x = images.astype(jnp.float32)

    if crop_size is not None:
        x = centered_crop_or_pad(x, (crop_size, crop_size), axes=(-2, -1))

    if new_size is not None:
        x = fourier_resample(x, (new_size, new_size), axes=(-2, -1))

    if ctf_correction == "none":
        return x

    box = x.shape[-1]
    ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv, sr_out,
                     [PAD_FACTOR * box, PAD_FACTOR * box // 2 + 1], x.shape[0], True)

    if ctf_correction == "phase_flip":
        return ctfFilter(x, jnp.where(ctf < 0, -1.0, 1.0), pad_factor=PAD_FACTOR)

    return wiener2DFilter(x, ctf, pad_factor=PAD_FACTOR, epsilon=wiener_epsilon)


_preprocess_jit = jax.jit(
    preprocess_particle_batch,
    static_argnames=("crop_size", "new_size", "ctf_correction", "wiener_epsilon"),
)


def _select_device(preference):
    """Resolve ``--device`` into a concrete JAX device.

    Particle preprocessing is a batched FFT workload over 10^5-10^6 images, where
    the GPU beats a many-core CPU by roughly 4-9x even counting host<->device
    transfers, so ``auto`` prefers it whenever one is visible.
    """
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []

    if preference == "cpu":
        return jax.devices("cpu")[0]
    if preference == "gpu":
        if not gpus:
            raise SystemExit("error: --device gpu: no GPU is visible to JAX. Drop the flag to run on the CPU.")
        return gpus[0]
    return gpus[0] if gpus else jax.devices("cpu")[0]


# hax evaluates CTFs on a ``PAD_FACTOR``-padded box; an odd size makes the padded
# spectrum and the CTF grid disagree in shape, so every hax program (not just this
# one) needs an even box. Reject odd sizes here rather than emit a stack that only
# fails later, deep inside a training run.
_ODD_BOX_HINT = ("hax evaluates CTFs on a zero-padded box, so every box size must be even. "
                 "Pass an even --new_box_size (or --crop_box_size) to fix it.")


def _validate_box(name, size):
    if size is not None and size % 2 != 0:
        raise SystemExit(f"error: --{name}: box size must be even (got {size}). {_ODD_BOX_HINT}")


def _validate_workers(name, count):
    if count < 1:
        raise SystemExit(f"error: --{name} must be >= 1 (got {count}).")


def _auto_batch_size(device, box_in, kv, sr_out, static):
    """Largest memory-safe batch for :func:`preprocess_particle_batch` on ``device``.

    Ahead-of-time compiles the kernel at two batch sizes with purely *abstract*
    inputs and reads XLA's own memory accounting, so nothing is allocated and
    nothing can OOM while probing. Returns None when the estimate is unavailable
    (CPU backend, no memory stats), leaving the caller to fall back.
    """
    def peak_fn(batch_size):
        images = jax.ShapeDtypeStruct((batch_size, box_in, box_in), jnp.float32)
        per_particle = jax.ShapeDtypeStruct((batch_size,), jnp.float32)
        compiled = _preprocess_jit.lower(
            images, per_particle, per_particle, per_particle, per_particle,
            kv, sr_out, **static).compile()
        analysis = compiled.memory_analysis()
        if analysis is None:
            return None
        return int(analysis.argument_size_in_bytes
                   + analysis.temp_size_in_bytes
                   + analysis.output_size_in_bytes)

    with jax.default_device(device):
        return estimate_batch_size_from_peak_fn(
            peak_fn, device=device, max_batch=MAX_AUTO_BATCH_SIZE,
            safety=AUTO_BATCH_MEMORY_SAFETY)


def _read_ahead_depth(num_read_workers, batch_size, shape_in, num_batches):
    """How many batches to keep in flight, bounded by bytes rather than by count."""
    bytes_per_batch = batch_size * int(np.prod(shape_in)) * np.dtype(np.float32).itemsize
    depth = max(1, READ_AHEAD_BUDGET_BYTES // bytes_per_batch)
    # Always keep at least two in flight, or the device waits on every read.
    return max(2, min(num_read_workers, int(depth), num_batches))


def main():
    import argparse

    import mrcfile
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from xmipp_metadata.metadata import XmippMetaData

    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_sr(parser, help=f"Sampling rate of the {bcolors.ITALIC}input{bcolors.ENDC} images, in Angstrom/pixel")
    parser.add_argument("--crop_box_size", required=False, type=int, default=None,
                        help=f"Real-space centred crop applied {bcolors.BOLD}first{bcolors.ENDC}. It trims the field of "
                             f"view without touching the sampling rate, which is useful to remove the empty padding "
                             f"around a particle before downsampling. A value larger than the input box zero-pads instead. "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: make sure the crop is wide enough to still contain the "
                             f"particle once its in-plane shift is applied.")
    parser.add_argument("--new_box_size", required=False, type=int, default=None,
                        help=f"Box size after Fourier resizing, applied {bcolors.BOLD}after{bcolors.ENDC} the crop. This is "
                             f"the parameter that trades resolution for speed and VRAM: halving the box divides the number "
                             f"of pixels (and roughly the memory a network needs) by four, while the new Nyquist frequency "
                             f"becomes {bcolors.ITALIC}2 x sr x box_in / box_out{bcolors.ENDC} Angstrom. The output sampling "
                             f"rate and the in-plane shifts stored in the metadata are updated accordingly.")
    parser.add_argument("--ctf_correction", required=False, type=str, default="none",
                        choices=["none", "wiener", "phase_flip"],
                        help=f"{bcolors.BOLD}none{bcolors.ENDC}: leave the images untouched (default)\n"
                             f"{bcolors.BOLD}wiener{bcolors.ENDC}: regularized CTF inversion, restoring amplitudes and phases\n"
                             f"{bcolors.BOLD}phase_flip{bcolors.ENDC}: flip the sign of the frequencies where the CTF is "
                             f"negative, correcting phases only\n"
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: if you correct the CTF here, the images are already "
                             f"corrected, so downstream programs must be run with {bcolors.UNDERLINE}--ctf_type None{bcolors.ENDC} "
                             f"to avoid correcting them twice.")
    parser.add_argument("--wiener_epsilon", required=False, type=float, default=None,
                        help=f"Fixed regularization constant for the Wiener filter. If not provided, an adaptive, "
                             f"per-image and frequency-dependent regularizer is estimated from the spectral SNR of each "
                             f"image, which is the recommended behaviour.")
    ca.add_batch_size(
        parser, default=FALLBACK_BATCH_SIZE,
        help=f"How many images are processed at once (set by default to {FALLBACK_BATCH_SIZE}). The CTF steps zero-pad each "
             f"image by a factor of {PAD_FACTOR}, so device memory scales with "
             f"{bcolors.ITALIC}batch_size x ({PAD_FACTOR} x box_out)^2{bcolors.ENDC}; lower this value if you "
             f"run out of memory, and monitor it with tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC}. "
             f"Pass {bcolors.ITALIC}auto{bcolors.ENDC} to let hax pick the largest batch that safely fits on your GPU "
             f"(estimated analytically, without running anything, and capped at {MAX_AUTO_BATCH_SIZE}). "
             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: use {bcolors.ITALIC}auto{bcolors.ENDC} to make a large box "
             f"{bcolors.BOLD}fit{bcolors.ENDC}, not to go faster. Unlike a training program, this one is bound by how fast "
             f"images are read from disk, and its throughput is already flat above a batch of ~1024 - so on a box that "
             f"fits comfortably, {bcolors.ITALIC}auto{bcolors.ENDC} only adds a few seconds of extra compilation.")
    parser.add_argument("--device", required=False, type=str, default="auto",
                        choices=["auto", "gpu", "cpu"],
                        help=f"Where the images are processed. {bcolors.BOLD}auto{bcolors.ENDC} (default) uses the GPU when "
                             f"one is visible and falls back to the CPU otherwise. The GPU is several times faster than a "
                             f"many-core CPU here even counting the host/device transfers, so there is normally no reason "
                             f"to force {bcolors.BOLD}cpu{bcolors.ENDC} unless the GPU is busy with another job.")
    parser.add_argument("--num_read_workers", required=False, type=int, default=8,
                        help="Threads reading batches of images from disk ahead of the device. Raise it if the progress "
                             "bar stalls on a slow or networked filesystem.")
    parser.add_argument("--num_write_workers", required=False, type=int, default=4,
                        help="Threads streaming the processed batches into the output stack.")
    parser.add_argument("--relative_image_paths", action='store_true',
                        help=f"Reference the output stack by its file name instead of its absolute path, so the output "
                             f"folder can be moved or shared. Metadata image paths are resolved against the "
                             f"{bcolors.ITALIC}working directory{bcolors.ENDC}, not the metadata file, so with this flag "
                             f"downstream programs must be launched from inside {bcolors.UNDERLINE}output_path{bcolors.ENDC}.")
    ca.add_output_path(parser, help="Path to save the preprocessed stack and its metadata")
    args = ca.parse_with_config(parser)

    if args.crop_box_size is None and args.new_box_size is None and args.ctf_correction == "none":
        raise SystemExit("error: nothing to do: pass --crop_box_size, --new_box_size and/or --ctf_correction.")

    _validate_box("crop_box_size", args.crop_box_size)
    _validate_box("new_box_size", args.new_box_size)
    _validate_workers("num_read_workers", args.num_read_workers)
    _validate_workers("num_write_workers", args.num_write_workers)

    os.makedirs(args.output_path, exist_ok=True)

    md = XmippMetaData(args.md)
    num_particles = len(md)
    if num_particles == 0:
        raise SystemExit(f"error: --md: '{args.md}' contains no particles.")

    shape_in = md.getMetaDataImage(0).shape
    if shape_in[-1] != shape_in[-2]:
        raise SystemExit(f"error: --md: the images are not square (they are {shape_in[-2]}x{shape_in[-1]}); "
                         f"hax works on square boxes.")
    box_in = shape_in[-1]

    # Resolve the box size after each stage, and with it the output sampling rate.
    box_cropped = args.crop_box_size if args.crop_box_size is not None else box_in
    box_out = args.new_box_size if args.new_box_size is not None else box_cropped

    # Catches an odd *input* box that neither flag overrides -- the flags themselves
    # were already checked above.
    if box_out % 2 != 0:
        raise SystemExit(f"error: --md: the images have an odd box size ({box_out}). {_ODD_BOX_HINT}")

    # A centred crop leaves the sampling rate alone; only the Fourier resize changes
    # it. In-plane shifts are stored in pixels, so they follow the inverse factor.
    sr_out = args.sr * box_cropped / box_out
    shift_scale = box_out / box_cropped

    # ``computeCTF`` needs a per-particle defocus. XmippMetaData back-fills missing
    # CTF columns with zeros, so their presence proves nothing -- check the values.
    defocusU = md.getMetaDataColumns("ctfDefocusU").astype(np.float32)
    if args.ctf_correction != "none" and not np.any(defocusU):
        raise SystemExit(f"error: --ctf_correction {args.ctf_correction}: the metadata carries no CTF information "
                         f"(every ctfDefocusU is zero).")
    defocusV = md.getMetaDataColumns("ctfDefocusV").astype(np.float32)
    defocusAngle = md.getMetaDataColumns("ctfDefocusAngle").astype(np.float32)
    cs = md.getMetaDataColumns("ctfSphericalAberration").astype(np.float32)
    voltage = md.getMetaDataColumns("ctfVoltage").astype(np.float32)

    # hax evaluates the electron wavelength from a single scalar voltage, as every
    # other program in the suite does; a mixed-voltage dataset would be mis-modelled.
    kv = np.float32(voltage[0] if len(voltage) else 300.0)
    if args.ctf_correction != "none" and len(voltage) and not np.allclose(voltage, kv):
        print(f"{bcolors.WARNING}WARNING{bcolors.ENDC}: the metadata mixes several acceleration voltages; "
              f"hax models the CTF with a single one ({kv} kV).")

    device = _select_device(args.device)
    stack_path = os.path.join(args.output_path, OUTPUT_STACK_NAME)
    static = dict(crop_size=args.crop_box_size, new_size=args.new_box_size,
                  ctf_correction=args.ctf_correction, wiener_epsilon=args.wiener_epsilon)

    # Resolve ``--batch_size auto`` before anything depends on it. Any failure falls
    # back to a fixed default so a run never aborts here.
    if args.batch_size == "auto":
        estimated = _auto_batch_size(device, box_in, kv, sr_out, static)
        args.batch_size = estimated if estimated is not None else FALLBACK_BATCH_SIZE
        if estimated is None:
            print(f"{bcolors.WARNING}Falling back to --batch_size {args.batch_size}.{bcolors.ENDC}")
    # Never spin up a batch larger than the dataset: it only wastes device memory.
    args.batch_size = min(args.batch_size, num_particles)

    nyquist_in, nyquist_out = 2.0 * args.sr, 2.0 * sr_out

    print(f"{bcolors.OKCYAN}\n###### Preprocessing {num_particles} particles ######{bcolors.ENDC}")
    print(f"     box            : {box_in}"
          + (f" -> crop {box_cropped}" if args.crop_box_size is not None else "")
          + (f" -> resize {box_out}" if args.new_box_size is not None else ""))
    print(f"     sampling rate  : {args.sr:.4f} -> {sr_out:.4f} A/px   (Nyquist {nyquist_in:.2f} -> {nyquist_out:.2f} A)")
    print(f"     in-plane shifts: rescaled by {shift_scale:.4f}")
    print(f"     CTF correction : {args.ctf_correction}")
    print(f"     batch size     : {args.batch_size}")
    print(f"     device         : {device}\n")

    bounds = [(start, min(start + args.batch_size, num_particles))
              for start in range(0, num_particles, args.batch_size)]

    # ``new_mmap`` lays the file out up front so batches can be written straight into
    # their final offsets, out of order and from several threads, without ever
    # holding the whole stack in RAM.
    with mrcfile.new_mmap(stack_path, shape=(num_particles, box_out, box_out),
                          mrc_mode=2, overwrite=True) as mrc:
        mrc.set_image_stack()
        mrc.voxel_size = sr_out
        stack = mrc.data

        def read(start, stop):
            # A single-image batch comes back squeezed to (H, W); restore the batch axis.
            images = md.getMetaDataImage(np.arange(start, stop))
            return np.asarray(images, dtype=np.float32).reshape(stop - start, *shape_in)

        def write(start, stop, batch):
            stack[start:stop] = batch

        with jax.default_device(device), \
                ThreadPoolExecutor(max_workers=args.num_read_workers) as readers, \
                ThreadPoolExecutor(max_workers=args.num_write_workers) as writers, \
                tqdm(total=num_particles, file=sys.stdout, ascii=" >=", colour="green") as pbar:

            # Keep a bounded window of batches in flight: enough to hide the read and
            # write latency behind the device, few enough that RAM stays flat.
            prefetch = _read_ahead_depth(args.num_read_workers, args.batch_size, shape_in, len(bounds))
            reads = deque(
                (start, stop, readers.submit(read, start, stop))
                for start, stop in bounds[:prefetch]
            )
            pending_writes = deque()
            next_batch = prefetch

            while reads:
                start, stop, future = reads.popleft()
                images = future.result()

                if next_batch < len(bounds):
                    nxt = bounds[next_batch]
                    reads.append((nxt[0], nxt[1], readers.submit(read, *nxt)))
                    next_batch += 1

                batch = _preprocess_jit(jnp.asarray(images), defocusU[start:stop], defocusV[start:stop],
                                        defocusAngle[start:stop], cs[start:stop], kv, sr_out, **static)
                pending_writes.append(writers.submit(write, start, stop, np.asarray(batch)))

                while len(pending_writes) >= 2 * args.num_write_workers:
                    pending_writes.popleft().result()
                pbar.update(stop - start)

            for pending in pending_writes:
                pending.result()

    # Point the metadata at the new stack and rescale the shifts it stores. Image paths
    # are resolved against the working directory rather than the metadata file, so an
    # absolute path is the only form that reads back from anywhere; --relative_image_paths
    # trades that for a relocatable output folder.
    reference = OUTPUT_STACK_NAME if args.relative_image_paths else os.path.abspath(stack_path)
    md.setMetaDataColumns(
        [f"{idx:06d}@{reference}" for idx in range(1, num_particles + 1)], "image")
    if shift_scale != 1.0:
        shifts = md.getMetaDataColumns(["shiftX", "shiftY"]).astype(np.float32)
        md.setMetaDataColumns(shifts * shift_scale, ["shiftX", "shiftY"])

    extension = os.path.splitext(args.md)[1].lower()
    extension = extension if extension in (".xmd", ".star") else ".xmd"
    md_path = os.path.join(args.output_path, OUTPUT_MD_STEM + extension)
    md.write(md_path, overwrite=True)

    print(f"\n{bcolors.OKGREEN}Wrote {stack_path} and {md_path}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Downstream programs must now be run with "
          f"{bcolors.UNDERLINE}--sr {sr_out:.4f}{bcolors.ENDC}{bcolors.OKBLUE}.{bcolors.ENDC}")
    if args.ctf_correction != "none":
        print(f"{bcolors.WARNING}The images are already CTF-corrected: run downstream programs with "
              f"{bcolors.UNDERLINE}--ctf_type None{bcolors.ENDC}{bcolors.WARNING} so they are not corrected twice."
              f"{bcolors.ENDC}")
