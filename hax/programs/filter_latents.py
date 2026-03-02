#!/usr/bin/env python


def main():
    import os
    import numpy as np
    import argparse
    from hax.utils import filter_latent_space, bcolors

    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True, type=str,
                        help="Path to the .npy file with the latent space to be filtered")
    parser.add_argument("--thr", required=False, type=float, default=1.0,
                        help="Threshold for the Z-Scores determining which ones will be kept")
    parser.add_argument("--n_neighbours", required=False, type=int, default=10,
                        help="Number of nearest neighbours used to compute the Z-Score for any given latent vector (smaller values are better to capture local features in the latent space)")
    parser.add_argument("--return_ids", action='store_true',
                        help="If provided, the ids of the latent vectors will be retrieved instead of the filtered space")
    parser.add_argument("--batch_size", required=False, type=int, default=1024,
                        help="Determines how many images will be load in the GPU at any moment during training (set by default to 1024 - "
                             f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                             f"to monitor and/or measure memory usage and adjust this value")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the filtered latent space")
    args, _ = parser.parse_known_args()

    # Load latents
    latents = np.load(args.latents)

    # Filter latents
    output = filter_latent_space(latents, args.thr, args.n_neighbours, args.return_ids, args.batch_size)

    # Save space
    np.save(os.path.join(args.output_path, "filtered_latents.npy"), output)
