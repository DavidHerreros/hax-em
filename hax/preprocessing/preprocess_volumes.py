#!/usr/bin/env python
"""Crop and/or downsample reference volumes so they match a preprocessed particle set.

The companion of :mod:`hax.preprocessing.preprocess_particles`: whenever particles
are resized, the reference volume, mask or initial map handed to the same network
has to be resized identically, or the projections will not line up with the images.

Order of operations (each step is optional):

1. ``--crop_box_size`` -- real-space centred crop/pad, trimming the field of view
   at the original sampling rate.
2. ``--new_box_size``  -- Fourier crop/pad, changing the sampling rate by
   ``box_cropped / box_resized``.

Cropping first is both the cheaper order (the FFT then runs on a smaller cube) and
the usual intent: trim the empty solvent padding, *then* downsample what is left.

This program runs on the **CPU**, deliberately. A GPU does the transform a few
times faster, but on the boxes volumes actually come in that is a sub-second
saving on a job that processes a handful of files, and a large box (768^3 and up)
will exhaust a mid-range card's memory during the complex-valued 3D FFT. A
multi-threaded ``scipy.fft`` keeps a 512^3 resize under a second with no such
cliff. The batched, 10^6-image particle path is where the GPU actually pays off.
"""

import os

import numpy as np
from scipy import fft as sfft

from hax.utils import bcolors, centered_crop_or_pad


def preprocess_volume(volume, crop_size=None, new_size=None, workers=-1):
    """Centre-crop and Fourier-resize a single volume.

    :param volume: ``(D, H, W)`` real volume.
    :param crop_size: real-space cube size after cropping, or None to skip.
    :param new_size: cube size after Fourier resizing, or None to skip.
    :param workers: threads handed to ``scipy.fft`` (-1 uses every core).
    :return: the processed volume, as ``float32``.
    """
    volume = np.asarray(volume, dtype=np.float32)
    axes = (0, 1, 2)

    if crop_size is not None:
        volume = centered_crop_or_pad(volume, (crop_size,) * 3, axes)

    if new_size is not None:
        old_shape = volume.shape
        spectrum = sfft.fftshift(sfft.fftn(volume, workers=workers))
        spectrum = centered_crop_or_pad(spectrum, (new_size,) * 3, axes)
        volume = sfft.ifftn(sfft.ifftshift(spectrum), workers=workers).real
        # The inverse transform divides by the new sample count; undo that so the
        # gray levels of the resized map match the original.
        volume = volume * (new_size ** 3 / np.prod(old_shape))

    return np.ascontiguousarray(volume, dtype=np.float32)


def _is_image_stack(path):
    """True when an MRC header positively marks the file as a stack of 2D images.

    A particle stack and a volume are both 3D arrays, so shape alone cannot always
    tell them apart -- a stack of N NxN particles even looks cubic. The MRC header
    does record the difference (``ISPG``/``MZ``), and everything this suite writes
    as a stack sets it, so a run of ``preprocess_particles`` accidentally fed back in
    as a ``--vol`` is caught here instead of being silently resampled as a volume.

    Files that do not carry the flag (older writers, non-MRC formats) are accepted as
    before: their header holds no information to reject them on.
    """
    if os.path.splitext(path)[1].lower() not in (".mrc", ".mrcs", ".map"):
        return False
    try:
        import mrcfile
        with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
            return bool(mrc.is_image_stack())
    except Exception:
        return False


def _output_path(output_dir, source, taken):
    """``<stem>_preprocessed.mrc`` inside ``output_dir``, disambiguated if two inputs
    from different folders happen to share a file name.

    The disambiguating index goes on the *stem*, never after ``_preprocessed``, so
    that every output this program writes ends in the same suffix and a single
    ``*_preprocessed.mrc`` glob finds all of them.
    """
    stem = os.path.splitext(os.path.basename(source))[0]
    candidate = os.path.join(output_dir, f"{stem}_preprocessed.mrc")
    index = 1
    while candidate in taken:
        candidate = os.path.join(output_dir, f"{stem}_{index}_preprocessed.mrc")
        index += 1
    taken.add(candidate)
    return candidate


def main():
    import argparse

    from xmipp_metadata.image_handler import ImageHandler

    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    parser.add_argument("--vol", required=True, type=str, nargs="+",
                        help=f"One or more volumes to preprocess (reference maps, masks, initial volumes...). "
                             f"Pass several paths to process them all with the same settings, which is the way to keep "
                             f"a map and its mask consistent.")
    ca.add_sr(parser, required=False,
              help=f"Sampling rate of the {bcolors.ITALIC}input{bcolors.ENDC} volumes, in Angstrom/pixel. If not "
                   f"provided, it is read from each volume's header.")
    parser.add_argument("--crop_box_size", required=False, type=int, default=None,
                        help=f"Real-space centred crop applied {bcolors.BOLD}first{bcolors.ENDC}. It trims the field of "
                             f"view without touching the sampling rate, which is how you remove the empty solvent padding "
                             f"around a map. A value larger than the input box zero-pads instead.")
    parser.add_argument("--new_box_size", required=False, type=int, default=None,
                        help=f"Cube size after Fourier resizing, applied {bcolors.BOLD}after{bcolors.ENDC} the crop. Use "
                             f"the same value you passed to {bcolors.UNDERLINE}preprocess_particles{bcolors.ENDC} so that "
                             f"the volume and the images share a box size and a sampling rate.")
    parser.add_argument("--num_workers", required=False, type=int, default=-1,
                        help="Threads used by the CPU FFT (-1, the default, uses every available core).")
    ca.add_output_path(parser, help="Path to save the preprocessed volumes")
    args = ca.parse_with_config(parser)

    if args.crop_box_size is None and args.new_box_size is None:
        raise SystemExit("error: nothing to do: pass --crop_box_size, --new_box_size, or both.")

    os.makedirs(args.output_path, exist_ok=True)

    print(f"{bcolors.OKCYAN}\n###### Preprocessing {len(args.vol)} volume(s) ######{bcolors.ENDC}")

    taken = set()
    for source in args.vol:
        if _is_image_stack(source):
            raise SystemExit(f"error: --vol: '{source}' is an image stack, not a volume. "
                             f"Particle stacks are handled by {bcolors.UNDERLINE}preprocess_particles{bcolors.ENDC}.")

        handler = ImageHandler(source)
        volume = np.squeeze(handler.getData())
        if volume.ndim != 3:
            raise SystemExit(f"error: --vol: '{source}' is not a volume (it has {volume.ndim} dimensions).")

        # A header sampling rate of 0 means "unset"; ImageHandler reports 1.0 for it.
        sr_in = args.sr if args.sr is not None else handler.getSamplingRate()
        if not sr_in:
            raise SystemExit(f"error: --sr: '{source}' has no sampling rate in its header; pass --sr explicitly.")

        box_in = volume.shape[0]
        if args.new_box_size is not None and len(set(volume.shape)) != 1:
            raise SystemExit(f"error: --new_box_size: '{source}' is not cubic (shape {volume.shape}); "
                             f"crop it to a cube first with --crop_box_size.")

        box_cropped = args.crop_box_size if args.crop_box_size is not None else box_in
        box_out = args.new_box_size if args.new_box_size is not None else box_cropped
        sr_out = sr_in * box_cropped / box_out

        processed = preprocess_volume(volume, crop_size=args.crop_box_size,
                                      new_size=args.new_box_size, workers=args.num_workers)

        destination = _output_path(args.output_path, source, taken)
        ImageHandler().write(processed, destination, sr=sr_out, overwrite=True)

        print(f"     {os.path.basename(source)}: {volume.shape} @ {sr_in:.4f} A/px "
              f"-> {processed.shape} @ {sr_out:.4f} A/px  ->  {destination}")

    print(f"\n{bcolors.OKGREEN}Wrote {len(args.vol)} volume(s) to {args.output_path}{bcolors.ENDC}")
    print(f"{bcolors.OKBLUE}Remember to pass the new sampling rate to any program reading these volumes."
          f"{bcolors.ENDC}")
