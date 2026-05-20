#!/usr/bin/env python


import jax
from flax import nnx
from hax.utils import *


def main():
    import os
    import numpy as np
    import argparse
    from xmipp_metadata.image_handler import ImageHandler
    from hax.checkpointer import NeuralNetworkCheckpointer

    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_file", required=True, type=str,
                        help="File with the latent vectors needed to decode the volumes. "
                             f"Valid extensions include {bcolors.ITALIC}.txt{bcolors.ENDC} and {bcolors.ITALIC}.npy{bcolors.ENDC}")
    parser.add_argument("--reload", required=True, type=str,
                        help=f"Path to a folder containing an already saved neural network ({bcolors.WARNING}NOTE{bcolors.ENDC}: "
                             f"Only networks saved in Pickled format can be supplied here.")
    parser.add_argument("--output_path", required=True, type=str,
                        help=f"Path were the decoded volumes will be saved.")
    args, _ = parser.parse_known_args()

    # Read latent vectors
    if args.latents_file.endswith(".txt"):
        latents = np.loadtxt(args.latents_file)
    elif args.latents_file.endswith(".npy"):
        latents = np.load(args.latents_file)
    else:
        raise ValueError(f"The format of {bcolors.ITALIC}latents_file{bcolors.ENDC} is not valid. Pease, provide a file "
                         f"saved as {bcolors.ITALIC}.txt{bcolors.ENDC} or {bcolors.ITALIC}.npy{bcolors.ENDC}")

    # Reload neural network
    network = NeuralNetworkCheckpointer.load(args.reload)

    # Split network (faster JIT)
    graphdef, state = nnx.split(network)

    # Jit function to increase speed
    @jax.jit
    def decode_volume(graphdef, state, x):
        model = nnx.merge(graphdef, state)
        return model.decode_volume(x)

    # Decode and save the volumes
    if latents.ndim > 1:
        volumes = []
        for latent in latents:
            volumes.append(np.array(decode_volume(graphdef, state, latent)))
        # Save volumes to file
        for idx in range(latents.shape[0]):
            ImageHandler().write(volumes[idx], os.path.join(args.output_path, f"decoded_volume_{idx:04d}.mrc"), overwrite=True)
    else:
        volume = decode_volume(graphdef, state, latents)
        ImageHandler().write(volume, os.path.join(args.output_path, f"decoded_volume_00.mrc"), overwrite=True)
