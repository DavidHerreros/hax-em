#!/usr/bin/env python
"""Global gray-scale adjustment of a map against a set of particles.

The map is projected at the pose of every particle (with its shift and CTF) and the single scale
that best matches the images is solved in closed form. The output is the input map times that
scale: no network, no Gaussian fit, and no offset (a constant in the map would fill the
background, so the offsets, when fitted, stay with the images).
"""


def main():
    import os
    import argparse
    import numpy as np
    from xmipp_metadata.image_handler import ImageHandler

    from hax.generators import MetaDataGenerator, extract_columns
    from hax.utils import bcolors
    from hax.utils.reconstruction import projection_contrast, global_contrast_from_sums
    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_vol(parser, required=True,
               help="Map to adjust. It must have the same box and sampling rate as the images.")
    ca.add_mask(parser,
                help=f"{bcolors.BOLD}(optional){bcolors.ENDC} Binary mask of the region to match. The fit then only uses "
                     f"the pixels inside the projection of the mask. Without it, the whole image is used. The mask only "
                     f"restricts the fit: the scale is applied to the whole map.")
    ca.add_sr(parser)
    ca.add_ctf_type(parser,
                    choices=ca.CTF_TYPE_CHOICES_PREMULTIPLIED,
                    help=f"Whether the images carry a CTF. {bcolors.ITALIC}apply{bcolors.ENDC}, {bcolors.ITALIC}wiener{bcolors.ENDC} "
                         f"and {bcolors.ITALIC}precorrect{bcolors.ENDC} all mean the images are the raw observations, so the CTF "
                         f"is applied to the projections; {bcolors.ITALIC}premultiplied{bcolors.ENDC} means the images were "
                         f"already multiplied by their CTF; {bcolors.ITALIC}None{bcolors.ENDC} means they carry no CTF at all.")
    parser.add_argument("--fit_offset", action='store_true',
                        help=f"Allow a per-image offset (image = scale * projection + offset) while fitting the scale. Use it "
                             f"when the images are not normalized to a zero-mean background. The offsets are not applied to "
                             f"the map.")
    parser.add_argument("--batch_size", required=False, type=int, default=256,
                        help=f"How many images are read and projected at once (set by default to 256). Lower it if the GPU "
                             f"runs out of memory; it does not change the result.")
    parser.add_argument("--threads", required=False, type=int, default=8,
                        help=f"Number of reader threads feeding the GPU with image chunks (set by default to 8). On a spinning "
                             f"disk (HDD) a lower value (1 or 2) can be faster.")
    ca.add_output_path(parser)
    ca.add_ssd_scratch_folder(parser,
                              help=f"Path to a folder on a fast (SSD/NVMe) disk. The images are cached there as a "
                                   f"{bcolors.ITALIC}float16{bcolors.ENDC} copy and read from it, the same cache the other hax "
                                   f"programs build in that folder.")
    args = ca.parse_with_config(parser)

    os.makedirs(args.output_path, exist_ok=True)

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)
    box = int(generator.md.getMetaDataImage(0).shape[-1])

    volume = ImageHandler(args.vol).getData()
    mask = ImageHandler(args.mask).getData() if args.mask is not None else None
    for name, data in (("--vol", volume), ("--mask", mask)):
        if data is not None and data.shape != (box,) * 3:
            raise ValueError(f"{name} has shape {data.shape}, but the images have a box of {box}: resize it first "
                             f"(e.g. with preprocess_volumes).")

    scratch_dir = None
    if args.ssd_scratch_folder is not None:
        generator.prepare_grain_array_record(mmap_output_dir=args.ssd_scratch_folder, preShuffle=False,
                                             num_workers=4, precision=np.float16, group_size=1,
                                             shard_size=10000)
        scratch_dir = generator.mmap_output_dir

    print(f"{bcolors.OKCYAN}\n###### Fitting the gray scale of the map to {len(generator.md)} images... ######{bcolors.ENDC}")
    sums = projection_contrast(volume, generator.md, md_columns, args.sr, mask=mask,
                               use_ctf=args.ctf_type not in (None, "None"),
                               premultiplied=args.ctf_type == "premultiplied",
                               batch_size=args.batch_size, threads=args.threads, scratch_dir=scratch_dir)
    scale = global_contrast_from_sums(sums, offset=args.fit_offset)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The projections of the map are anti-correlated with the images. Check that the images and "
                         "the map have the same contrast (protein white) and that --ctf_type matches the data.")

    volume_path = os.path.join(args.output_path, "adjusted_volume.mrc")
    ImageHandler().write((scale * volume).astype(np.float32), volume_path, overwrite=True, sr=args.sr)
    np.savetxt(os.path.join(args.output_path, "gray_scale.txt"), [scale], fmt="%.6g",
               header="Scale applied to the input map")
    print(f"{bcolors.OKGREEN}Gray scale of the map: x{scale:.4g}\nAdjusted map -> {volume_path}{bcolors.ENDC}")
