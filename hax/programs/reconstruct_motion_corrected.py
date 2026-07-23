#!/usr/bin/env python
"""Motion-corrected volume reconstruction from posed particles and a trained motion model.

A thin CLI around :func:`hax.utils.reconstruct_motion_corrected_volume`. It is the
non-iterative alternative to ``modart``: instead of fitting ~10^5 free voxel amplitudes by
gradient descent against an image residual, it removes each particle's modelled motion from
its image and then reconstructs by the same CTF-weighted Fourier gridding that produces the
consensus map.

Why that difference matters, rather than being a matter of taste:

* **No unconstrained parameters.** A least-squares fit leaves every direction the forward
  operator cannot see completely undetermined, and Adam gives an undetermined direction a
  full-size step regardless of how little the data says about it -- so those directions do not
  stay at their starting value, they fill with noise. Gridding has no free parameters: each
  Fourier voxel is a CTF-weighted average of the data that touched it, and shells nobody
  measured stay empty.
* **Correct per-frequency weighting for free.** The Wiener quotient ``sum(CTF*F)/sum(CTF^2)``
  is the right estimator at every frequency, including the ones where individual CTFs vanish
  (defocus diversity fills them). Nothing has to be tuned to make that happen.
* **A resolution number that means something.** Half maps come out of the same pass, so the
  FSC is available immediately -- and it is the same even/odd split the consensus uses, so the
  two curves can be compared directly.
* **One pass.** Roughly the cost of the consensus reconstruction, rather than tens of
  thousands of gradient steps.

The correction itself is exact only to the extent that a 3D deformation can be seen in a
projection: a general 3D warp does not act on a central Fourier slice, so what is applied is
its image-plane shadow (see ``_unwarp_images``). For the small, smooth, domain-scale motions
these models estimate that is an excellent approximation; for very large motions along the
view direction it is not, and a domain-decomposed rigid treatment would be the right next step.
"""


def main():
    import os
    import argparse
    import numpy as np
    from xmipp_metadata.image_handler import ImageHandler

    from hax.generators import MetaDataGenerator, extract_columns
    from hax.utils import (reconstruct_motion_corrected_volume, reconstruct_consensus_volume,
                           consensus_mask, report_half_map_resolution, bcolors)
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_sr(parser)
    ca.add_ctf_type(parser,
                    help=f"Whether the images carry a CTF. The reconstruction only needs to know if they do: "
                         f"{bcolors.ITALIC}apply{bcolors.ENDC}, {bcolors.ITALIC}wiener{bcolors.ENDC} and "
                         f"{bcolors.ITALIC}precorrect{bcolors.ENDC} all mean they do, while {bcolors.ITALIC}None"
                         f"{bcolors.ENDC} means they do not.")
    parser.add_argument("--motion_correction", required=True, type=str,
                        help=f"Path to a trained {bcolors.UNDERLINE}HetSIREN with mass transport{bcolors.ENDC} "
                             f"({bcolors.ITALIC}--transport_mass{bcolors.ENDC}) checkpoint. Its encoder supplies each "
                             f"particle's latent and its decoder the displacement field that is removed from the image.")
    parser.add_argument("--tau", required=False, type=float, default=0.05,
                        help=f"Wiener floor of the reconstruction quotient (default {bcolors.ITALIC}0.05{bcolors.ENDC}). A numerical "
                             f"regularizer only -- the resolution is set by the data, not by this.")
    parser.add_argument("--correction", required=False, type=str, default="image",
                        choices=["image", "backprojection"],
                        help=f"WHERE the modelled motion is removed (default {bcolors.ITALIC}image{bcolors.ENDC}).\n"
                             f"  {bcolors.ITALIC}image{bcolors.ENDC} warps each 2D image and then inserts it normally. Fast -- one pass, "
                             f"a couple of operations per pixel -- but a projection ray gets ONE displacement, the density-weighted mean "
                             f"of the motion along it, because a single-valued 2D field cannot carry more. On the ribosome the part of "
                             f"the field that varies ALONG a ray measured ~95% of the part that does not, so this is a real approximation, "
                             f"not a rounding error.\n"
                             f"  {bcolors.ITALIC}backprojection{bcolors.ENDC} bends the backprojection rays in 3D instead: each canonical "
                             f"voxel reads the image where {bcolors.UNDERLINE}its own{bcolors.ENDC} material landed, so two voxels at "
                             f"different depths on one ray read different pixels. This is what DynaMight does (Nat Methods 21, 1855-1862) "
                             f"and it is the geometrically correct version. It costs {bcolors.WARNING}~15-20x more{bcolors.ENDC} "
                             f"(box x mask-fraction gathers per particle instead of one image warp).\n"
                             f"  Both modes share the CTF weighting, the parity halves and the gray scale, so their maps are directly "
                             f"comparable and the difference between them IS the depth-resolved part of the motion.")
    parser.add_argument("--jacobian", action='store_true',
                        help=f"Multiply the resampled density by the Jacobian determinant of the deformation, making the correction "
                             f"conserve {bcolors.UNDERLINE}mass{bcolors.ENDC} rather than values. Resampling alone moves intensities "
                             f"around; where a deformation compresses a region the same material lands on fewer pixels and must be scaled "
                             f"up for the line integral to survive. Second order for the near-isometric deformations the elastic priors "
                             f"produce -- measured at 1.2% (median) and 30% (p99) on the ribosome -- but it grows in proportion to the "
                             f"size of the motion. {bcolors.ITALIC}DynaMight does not do this{bcolors.ENDC}. Off by default so the two "
                             f"correction modes reproduce their published behaviour unless asked.")
    parser.add_argument("--mask_radius", required=False, type=float, default=5.0,
                        help=f"Voxels within this many voxels of a consensus point that carries mass take part in the deformed "
                             f"backprojection (default {bcolors.ITALIC}5.0{bcolors.ENDC}); only used with "
                             f"{bcolors.ITALIC}--correction backprojection{bcolors.ENDC}. Cost is linear in the number of selected voxels, "
                             f"and the model has nothing to say outside its own point cloud. The selected fraction is printed at startup.")
    parser.add_argument("--mask_taper", required=False, type=float, default=2.0,
                        help=f"Width, in voxels, over which the correction is faded out as it approaches the edge of that region "
                             f"(default {bcolors.ITALIC}2.0{bcolors.ENDC}, raised cosine). {bcolors.WARNING}Do not set this to 0{bcolors.ENDC} "
                             f"unless you want to see why it exists: the correction is a difference, so ending it at a hard boundary leaves a "
                             f"step of its full size, and that step rings through the Fourier division. Measured with no taper, "
                             f"{bcolors.UNDERLINE}71.6% of the entire difference between the corrected map and the consensus sat in a +-1.5 "
                             f"voxel shell at the mask edge{bcolors.ENDC} -- 5.1% of the box by volume, where it was 1.43x the size of the "
                             f"density itself. The interpolated field does not fade on its own (its outermost band is the largest of any), so "
                             f"the window has to impose it. Full-strength correction therefore reaches to "
                             f"{bcolors.ITALIC}--mask_radius{bcolors.ENDC} minus this.")
    parser.add_argument("--warp_sigma", required=False, type=float, default=2.0,
                        help=f"Width, in pixels, of the smoothing applied when the scattered per-point displacements are turned into "
                             f"a dense image-plane warp (default {bcolors.ITALIC}2.0{bcolors.ENDC}). It should be comparable to the "
                             f"spacing between the model's points in projection; too small leaves gaps between them, too large washes "
                             f"out the boundary between a moving domain and a static one.")
    parser.add_argument("--min_coverage", required=False, type=float, default=0.05,
                        help=f"Pixels whose splat coverage falls below this fraction of the mean are left UNWARPED rather than having "
                             f"a warp extrapolated into them (default {bcolors.ITALIC}0.05{bcolors.ENDC}). Doing nothing where the model "
                             f"has no support is the safe default; dividing by a near-zero coverage is how the previous 3D field "
                             f"transfer produced displacements thousands of Angstrom long.")
    parser.add_argument("--no_denoise", action='store_true',
                        help=f"Skip the FSC-based denoising of the combined map. The FSC is still measured and reported.")
    parser.add_argument("--no_gray_scale_calibration", action='store_true',
                        help=f"Skip the global gray-scale calibration against the input images.")
    parser.add_argument("--also_consensus", action='store_true',
                        help=f"Also reconstruct the ordinary (uncorrected) consensus from the same particles and print both "
                             f"resolutions. {bcolors.WARNING}Strongly recommended{bcolors.ENDC}: it is the control that says whether "
                             f"the motion correction changed anything, and it is the only way to attribute a difference to the "
                             f"correction rather than to any other part of the pipeline.")
    parser.add_argument("--write_mask", action='store_true',
                        help=f"Also write a binary mask of the protein region derived from the reconstructed map.")
    parser.add_argument("--batch_size", required=False, type=int, default=512,
                        help=f"How many images are read per chunk (default {bcolors.ITALIC}512{bcolors.ENDC}). Peak RAM tracks this "
                             f"rather than the particle count.")
    parser.add_argument("--field_batch", required=False, type=int, default=32,
                        help=f"How many particles are pushed through the motion decoder at once (default {bcolors.ITALIC}32"
                             f"{bcolors.ENDC}). This is the VRAM-sensitive number -- the decoder holds "
                             f"(particles x points x hidden) activations -- so lower it if the field decode runs out of memory. It "
                             f"does not change the result.")
    parser.add_argument("--threads", required=False, type=int, default=8,
                        help=f"Number of reader threads feeding the GPU (default {bcolors.ITALIC}8{bcolors.ENDC}). On a spinning disk "
                             f"a lower value can be faster.")
    ca.add_output_path(parser)
    ca.add_ssd_scratch_folder(parser)
    args = ca.parse_with_config(parser)

    os.makedirs(args.output_path, exist_ok=True)

    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    scratch_dir = None
    if args.ssd_scratch_folder is not None:
        generator.prepare_grain_array_record(mmap_output_dir=args.ssd_scratch_folder, preShuffle=False,
                                             num_workers=4, precision=np.float16, group_size=1,
                                             shard_size=10000)
        scratch_dir = generator.mmap_output_dir

    motion_model = NeuralNetworkCheckpointer.load(args.motion_correction)
    if not getattr(motion_model.delta_volume_decoder, "transport_mass", False):
        raise UserWarning("--motion_correction must point at a HetSIREN trained with "
                          "--transport_mass; a fixed-grid model has no displacement field to apply.")

    use_ctf = args.ctf_type not in (None, "None")

    consensus = None
    if args.also_consensus:
        consensus = reconstruct_consensus_volume(generator.md, md_columns, args.sr, tau=args.tau,
                                                 batch_size=args.batch_size, threads=args.threads,
                                                 use_ctf=use_ctf, denoise=not args.no_denoise,
                                                 calibrate_gray_scale=not args.no_gray_scale_calibration,
                                                 scratch_dir=scratch_dir)
        consensus_path = os.path.join(args.output_path, "consensus_reconstruction.mrc")
        ImageHandler().write(np.asarray(consensus), consensus_path, overwrite=True)
        print(f"{bcolors.OKGREEN}Uncorrected consensus (the control) -> {consensus_path}{bcolors.ENDC}")

    volume, extras = reconstruct_motion_corrected_volume(
        generator.md, md_columns, args.sr, motion_model,
        tau=args.tau, batch_size=args.batch_size, field_batch=args.field_batch,
        threads=args.threads, use_ctf=use_ctf, denoise=not args.no_denoise,
        calibrate_gray_scale=not args.no_gray_scale_calibration, scratch_dir=scratch_dir,
        warp_sigma=args.warp_sigma, min_coverage=args.min_coverage,
        correction=args.correction, jacobian=args.jacobian, mask_radius=args.mask_radius,
        mask_taper=args.mask_taper)

    volume_path = os.path.join(args.output_path, "motion_corrected_reconstruction.mrc")
    ImageHandler().write(np.asarray(volume), volume_path, overwrite=True)
    ImageHandler().write(extras["half_a"], os.path.join(args.output_path, "motion_corrected_half1.mrc"), overwrite=True)
    ImageHandler().write(extras["half_b"], os.path.join(args.output_path, "motion_corrected_half2.mrc"), overwrite=True)
    print(f"{bcolors.OKGREEN}Motion-corrected volume -> {volume_path}{bcolors.ENDC}")

    report_half_map_resolution(extras["half_a"], extras["half_b"], args.sr,
                               label="Motion-corrected", reference=consensus)

    if consensus is not None:
        print(f"{bcolors.OKCYAN}\nCompare the two maps directly: they were reconstructed from the same "
              f"particles, with the same poses, the same CTF weighting and the same gray scale, so any "
              f"difference between them IS the motion correction.{bcolors.ENDC}")

    if args.write_mask:
        mask = consensus_mask(volume, dilate=2)
        mask_path = os.path.join(args.output_path, "reconstruction_mask.mrc")
        ImageHandler().write(np.asarray(mask), mask_path, overwrite=True)
        print(f"{bcolors.OKGREEN}Mask derived from the reconstructed map -> {mask_path}{bcolors.ENDC}")
