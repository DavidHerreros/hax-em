#!/usr/bin/env python
"""Per-image gray-scale adjustment of the particles against a reference map.

Every image is compared with the map projected at its pose (with its shift and CTF), and its
gray scale -- optionally with an offset -- is solved in closed form, as HetSIREN's per-image
contrast does. The images are then rescaled and written as a new stack. No network is trained.
"""


def main():
    import os
    import sys
    import argparse
    import numpy as np
    import mrcfile
    from tqdm import tqdm
    from xmipp_metadata.image_handler import ImageHandler

    from hax.generators import MetaDataGenerator, extract_columns
    from hax.utils import bcolors
    from hax.utils.reconstruction import projection_contrast, contrast_from_sums, _StackReader, _stream_chunks
    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser)
    ca.add_vol(parser, required=True,
               help="Reference map whose projections set the gray scale of each image. It must have the same box and "
                    "sampling rate as the images.")
    ca.add_mask(parser,
                help=f"{bcolors.BOLD}(optional){bcolors.ENDC} Binary mask of the region to match. The fit then only uses "
                     f"the pixels inside the projection of the mask. Without it, the whole image is used.")
    ca.add_sr(parser)
    ca.add_ctf_type(parser,
                    choices=ca.CTF_TYPE_CHOICES_PREMULTIPLIED,
                    help=f"Whether the images carry a CTF. {bcolors.ITALIC}apply{bcolors.ENDC}, {bcolors.ITALIC}wiener{bcolors.ENDC} "
                         f"and {bcolors.ITALIC}precorrect{bcolors.ENDC} all mean the images are the raw observations, so the CTF "
                         f"is applied to the projections; {bcolors.ITALIC}premultiplied{bcolors.ENDC} means the images were "
                         f"already multiplied by their CTF; {bcolors.ITALIC}None{bcolors.ENDC} means they carry no CTF at all.")
    parser.add_argument("--per_image_contrast", required=False, type=str, default="relative", choices=["relative", "absolute"],
                        help=f"{bcolors.ITALIC}relative{bcolors.ENDC} (default): divide each image by its gray scale relative to "
                             f"the mean of the data set, so the contrast becomes even across particles while the overall gray "
                             f"level is kept. {bcolors.ITALIC}absolute{bcolors.ENDC}: divide each image by its own gray scale, which "
                             f"puts every particle on the gray scale of the map.")
    parser.add_argument("--fit_offset", action='store_true',
                        help=f"Also fit a per-image offset (image = scale * projection + offset) and subtract it. By default only "
                             f"the scale is fitted, which assumes images normalized to a zero-mean background.")
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
    md = generator.md
    md_columns = extract_columns(md)
    n = len(md)
    box = int(md.getMetaDataImage(0).shape[-1])

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

    # Fit the gray scale of every image
    print(f"{bcolors.OKCYAN}\n###### Fitting the gray scale of {n} images... ######{bcolors.ENDC}")
    sums = projection_contrast(volume, md, md_columns, args.sr, mask=mask,
                               use_ctf=args.ctf_type not in (None, "None"),
                               premultiplied=args.ctf_type == "premultiplied",
                               batch_size=args.batch_size, threads=args.threads, scratch_dir=scratch_dir)
    a, _ = contrast_from_sums(sums, offset=args.fit_offset)

    valid = np.isfinite(a) & (a > 0)
    if valid.mean() < 0.5:
        raise ValueError("Most images are anti-correlated with the projections of the map. Check that the images and "
                         "the map have the same contrast (protein white) and that --ctf_type matches the data.")
    median = np.median(a[valid])
    # Bound the scales so that a particle the map does not explain is not amplified into noise
    bounded = np.clip(np.where(valid, a, median), median / 5.0, median * 5.0)
    n_clipped = int(np.sum(valid & (bounded != a)))
    a = bounded

    if args.fit_offset:
        _, _, p, y, w = sums.T
        b = (y - a * p) / np.maximum(w, 1.0)
    else:
        b = np.zeros_like(a)

    reference = a.mean() if args.per_image_contrast == "relative" else 1.0
    adjustment_a = (reference / a).astype(np.float32)
    adjustment_b = (-b * adjustment_a).astype(np.float32)

    print(f"{bcolors.OKGREEN}Gray scale per image: mean {a.mean():.4g}, std {a.std():.4g}, "
          f"range [{a.min():.4g}, {a.max():.4g}]{bcolors.ENDC}")
    if (~valid).any() or n_clipped:
        print(f"{bcolors.WARNING}{int((~valid).sum())} images without a usable fit were given the median scale and "
              f"{n_clipped} were clipped to [median / 5, median * 5].{bcolors.ENDC}")

    # Write the adjusted images
    print(f"{bcolors.OKCYAN}\n###### Writing the adjusted images... ######{bcolors.ENDC}")
    stack_path = os.path.abspath(os.path.join(args.output_path, "adjusted_images.mrcs"))
    reader = _StackReader(md, scratch_dir)
    with mrcfile.new_mmap(stack_path, shape=(n, box, box), mrc_mode=2, overwrite=True) as out:
        out.set_image_stack()
        out.voxel_size = args.sr
        for labels, images in tqdm(_stream_chunks(reader, n, args.batch_size, args.threads),
                                   total=(n + args.batch_size - 1) // args.batch_size,
                                   file=sys.stdout, ascii=" >=", colour="green"):
            out.data[labels] = images * adjustment_a[labels, None, None] + adjustment_b[labels, None, None]

    md[:, "image"] = [f"{idx + 1:06d}@{stack_path}" for idx in range(n)]
    md[:, "adjustment_a"] = adjustment_a
    md[:, "adjustment_b"] = adjustment_b
    md_path = os.path.join(args.output_path, "adjusted_images" + os.path.splitext(args.md)[1])
    md.write(md_path)
    print(f"{bcolors.OKGREEN}Adjusted images -> {stack_path}\nMetadata -> {md_path}{bcolors.ENDC}")
