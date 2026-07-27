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
                    choices=ca.CTF_TYPE_CHOICES_PREMULTIPLIED,
                    help=f"Whether the images carry a CTF, and whether it has already been multiplied into them. "
                         f"{bcolors.ITALIC}apply{bcolors.ENDC}, {bcolors.ITALIC}wiener{bcolors.ENDC} and "
                         f"{bcolors.ITALIC}precorrect{bcolors.ENDC} all mean the images are the raw observations "
                         f"(the slices are CTF weighted here and the quotient deconvolves them), while "
                         f"{bcolors.ITALIC}None{bcolors.ENDC} means they carry no CTF at all.\n"
                         f"Use {bcolors.ITALIC}premultiplied{bcolors.ENDC} when whatever extracted the particles already "
                         f"multiplied them by their CTF -- this is the default for RELION's 2D tilt-series stacks and for "
                         f"{bcolors.UNDERLINE}WarpTools ts_export_particles{bcolors.ENDC}, which does "
                         f"{bcolors.ITALIC}ImagesFT.Multiply(CTFs){bcolors.ENDC} unless asked not to. The slices then go in "
                         f"as they are and only the denominator uses the CTF.\n"
                         f"{bcolors.WARNING}NOTE{bcolors.ENDC}: this must match your data. Weighting CTF-free images by a "
                         f"CTF is not a harmless no-op, and multiplying pre-multiplied ones a second time leaves the map "
                         f"modulated by an extra CTF -- low frequencies suppressed and the CTF zeros squared.")
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
    parser.add_argument("--no_dose_weighting", action='store_true',
                        help=f"{bcolors.BOLD}(optional){bcolors.ENDC} Do not fold the "
                             f"tilt-series dose/tilt weighting into the Wiener denominator. "
                             f"Tomography images are pre-multiplied by CTF*W, so W belongs "
                             f"there; turning this off leaves a large B-factor in the map. "
                             f"Ignored when the metadata has no preExposure column.")
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
                        help=f"Number of reader threads feeding the GPU with image chunks (set by default to 8). "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: on a spinning disk (HDD) several threads read different "
                             f"regions of the stack at once and the head seeks between them, so a lower value (1 or 2) can be "
                             f"{bcolors.ITALIC}faster{bcolors.ENDC} there than the default.")
    ca.add_output_path(parser)
    ca.add_ssd_scratch_folder(parser,
                              help=f"Path to a folder on a fast (SSD/NVMe) disk. The reconstruction has to read every particle "
                                   f"once, so on a slow disk the read {bcolors.ITALIC}is{bcolors.ENDC} the run: 1M particles at box 320 "
                                   f"is 410 GB, which an HDD needs roughly an hour to deliver, and the GPU spends that hour idle. "
                                   f"Given this parameter, the images are cached as a local {bcolors.ITALIC}float16{bcolors.ENDC} copy "
                                   f"(half the bytes) and streamed from there. {bcolors.WARNING}NOTE{bcolors.ENDC}: building the cache "
                                   f"still costs one full read of the original stack, so it does not make a single cold run faster - it "
                                   f"pays off because the copy is {bcolors.UNDERLINE}reused{bcolors.ENDC}, both by later runs and by the "
                                   f"other hax programs (HetSIREN, Zernike3Deep, MoDART...) pointed at the same scratch folder, which "
                                   f"build and read exactly the same cache.")
    args = ca.parse_with_config(parser)

    os.makedirs(args.output_path, exist_ok=True)

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Cache the stack on the fast disk, if one was given. This is the same array-record copy the
    # network programs build, so pointing them all at one scratch folder pays for it once.
    scratch_dir = None
    if args.ssd_scratch_folder is not None:
        generator.prepare_grain_array_record(mmap_output_dir=args.ssd_scratch_folder, preShuffle=False,
                                             num_workers=4, precision=np.float16, group_size=1,
                                             shard_size=10000)
        scratch_dir = generator.mmap_output_dir

    # Reconstruct the consensus volume. Every CTF mode except None means the stored images
    # carry the CTF (precorrect only Wiener-filters them at train time, it does not alter
    # the data), so that is what decides whether the slices are CTF weighted. `premultiplied`
    # additionally says the CTF is already *in* the pixels, so it belongs in the denominator
    # only.
    volume = reconstruct_consensus_volume(generator.md, md_columns, args.sr,
                                          tau=args.tau,
                                          batch_size=args.batch_size,
                                          threads=args.threads,
                                          use_ctf=args.ctf_type not in (None, "None"),
                                          premultiplied=args.ctf_type == "premultiplied",
                                          denoise=not args.no_denoise,
                                          dose_weighting=not args.no_dose_weighting,
                                          calibrate_gray_scale=not args.no_gray_scale_calibration,
                                          scratch_dir=scratch_dir)

    volume_path = os.path.join(args.output_path, "consensus_reconstruction.mrc")
    ImageHandler().write(np.asarray(volume), volume_path, overwrite=True)
    print(f"{bcolors.OKGREEN}Consensus volume reconstructed from the input poses -> {volume_path}{bcolors.ENDC}")

    if args.write_mask:
        mask = consensus_mask(volume, threshold=args.mask_threshold, dilate=args.mask_dilate)
        mask_path = os.path.join(args.output_path, "consensus_mask.mrc")
        ImageHandler().write(np.asarray(mask), mask_path, overwrite=True)
        print(f"{bcolors.OKGREEN}Mask derived from the reconstructed map -> {mask_path}{bcolors.ENDC}")
