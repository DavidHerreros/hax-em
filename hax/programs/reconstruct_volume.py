#!/usr/bin/env python
"""Consensus volume reconstruction from posed particles.

A thin CLI around :func:`hax.utils.reconstruct_consensus_volume`: the particles already
carry poses, so the map is recovered in a single streaming pass of Wiener-filtered Fourier
gridding -- no iteration, no refinement. The same routine is what HetSIREN calls internally
when it is given neither ``--vol`` nor ``--mask``; this program exposes it on its own, so a
consensus map (and the mask derived from it) can be produced once and reused across runs.

See ``hax/utils/reconstruction.py`` for the geometry and the half-map FSC / gray-scale
calibration this relies on.
"""


def main():
    import os
    import argparse
    import numpy as np
    from xmipp_metadata.image_handler import ImageHandler

    from hax.generators import MetaDataGenerator, extract_columns
    from hax.utils import reconstruct_consensus_volume, consensus_mask, bcolors
    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_sr(parser)
    ca.add_ctf_type(parser,
                    help=f"Whether the images carry a CTF. The reconstruction only needs to know if they do: "
                         f"{bcolors.ITALIC}apply{bcolors.ENDC}, {bcolors.ITALIC}wiener{bcolors.ENDC} and "
                         f"{bcolors.ITALIC}precorrect{bcolors.ENDC} all mean they do (the slices are CTF weighted and the "
                         f"quotient deconvolves them), while {bcolors.ITALIC}None{bcolors.ENDC} means they do not. "
                         f"{bcolors.WARNING}NOTE{bcolors.ENDC}: this must match your data -- weighting CTF-free images by a "
                         f"CTF is not a harmless no-op, it reweights the slices and degrades the map.")
    parser.add_argument("--tau", required=False, type=float, default=0.05,
                        help=f"Wiener floor of the reconstruction quotient (set by default to 0.05). This is only a numerical "
                             f"regularizer to keep the shells with little CTF power from blowing up -- the resolution of the map "
                             f"is set by the data, not by this. Raise it if the map looks noisy at high frequency, lower it if it "
                             f"looks over-smoothed.")
    parser.add_argument("--no_denoise", action='store_true',
                        help=f"Skip the FSC based denoising. By default the particles are split into two independent halves, the "
                             f"FSC between the resulting half maps measures the spectral signal-to-noise, and the combined map is "
                             f"filtered with it, so the shells carrying signal pass untouched and the shells that are only noise are "
                             f"removed (the measured resolution is printed). Pass this flag to get the raw, unfiltered map instead.")
    parser.add_argument("--no_gray_scale_calibration", action='store_true',
                        help=f"Skip the global gray-scale calibration. By default the finished map is forward-projected at a subset "
                             f"of the real poses and least-squares scaled against the input images, so that re-projecting it "
                             f"reproduces their contrast and value range. Pass this flag to keep the raw amplitudes.")
    parser.add_argument("--write_mask", action='store_true',
                        help=f"Also write a binary mask of the protein region derived from the reconstructed map "
                             f"({bcolors.UNDERLINE}consensus_mask.mrc{bcolors.ENDC}). This is the mask HetSIREN/MoDART expect in "
                             f"their {bcolors.ITALIC}--mask{bcolors.ENDC} parameter.")
    parser.add_argument("--mask_threshold", required=False, type=float, default=0.02,
                        help=f"Only used with {bcolors.ITALIC}--write_mask{bcolors.ENDC}: fraction of the map's maximum above which a "
                             f"voxel is considered protein (set by default to 0.02). Raise it for a tighter mask, lower it for a looser one.")
    parser.add_argument("--mask_dilate", required=False, type=int, default=2,
                        help=f"Only used with {bcolors.ITALIC}--write_mask{bcolors.ENDC}: how many voxels to grow the mask by (set by "
                             f"default to 2), so that a deformation estimated later has somewhere to move the mass into.")
    parser.add_argument("--batch_size", required=False, type=int, default=1024,
                        help=f"How many images are read and inserted at once (set by default to 1024). The images are read on a thread "
                             f"pool while the GPU accumulates, so peak RAM tracks this value rather than the number of particles - lower "
                             f"it if you run out of memory, raise it to read fewer, larger chunks.")
    parser.add_argument("--threads", required=False, type=int, default=8,
                        help="Number of reader threads feeding the GPU with image chunks (set by default to 8)")
    ca.add_output_path(parser)
    args = ca.parse_with_config(parser)

    os.makedirs(args.output_path, exist_ok=True)

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Reconstruct the consensus volume. Every CTF mode except None means the stored images
    # carry the CTF (precorrect only Wiener-filters them at train time, it does not alter
    # the data), so that is what decides whether the slices are CTF weighted.
    volume = reconstruct_consensus_volume(generator.md, md_columns, args.sr,
                                          tau=args.tau,
                                          batch_size=args.batch_size,
                                          threads=args.threads,
                                          use_ctf=args.ctf_type not in (None, "None"),
                                          denoise=not args.no_denoise,
                                          calibrate_gray_scale=not args.no_gray_scale_calibration)

    volume_path = os.path.join(args.output_path, "consensus_reconstruction.mrc")
    ImageHandler().write(np.asarray(volume), volume_path, overwrite=True)
    print(f"{bcolors.OKGREEN}Consensus volume reconstructed from the input poses -> {volume_path}{bcolors.ENDC}")

    if args.write_mask:
        mask = consensus_mask(volume, threshold=args.mask_threshold, dilate=args.mask_dilate)
        mask_path = os.path.join(args.output_path, "consensus_mask.mrc")
        ImageHandler().write(np.asarray(mask), mask_path, overwrite=True)
        print(f"{bcolors.OKGREEN}Mask derived from the reconstructed map -> {mask_path}{bcolors.ENDC}")
