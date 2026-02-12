#!/usr/bin/env python


import random
from functools import partial

import jax
from jax import random as jnr, numpy as jnp

from hax.utils import *


@partial(jax.jit, static_argnames=["model",])
def estimate_latent_covariances(model, x, labels, md):
    # Decode clean projection
    x_clean, latent = model.decode_image(x, labels, md, ctf_type=None, return_latent=True)
    x_clean = x_clean[..., None]

    # CTF
    if model.ctf_type is not None:
        defocusU = md["ctfDefocusU"][labels]
        defocusV = md["ctfDefocusV"][labels]
        defocusAngle = md["ctfDefocusAngle"][labels]
        cs = md["ctfSphericalAberration"][labels]
        kv = md["ctfVoltage"][labels][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         model.sr, [2 * x.shape[1], int(2 * 0.5 * x.shape[1] + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], 2 * x.shape[1], int(2.0 * 0.5 * x.shape[1] + 1)], dtype=x.dtype)

    # Corrupt projection with CTF
    if model.ctf_type == "apply":
        x_clean_ctf = ctfFilter(x_clean[..., 0], ctf, pad_factor=2)[..., None]
    else:
        x_clean_ctf = x_clean

    # Variance map
    residuals = x - x_clean_ctf
    var_map = residuals ** 2
    var_res = jnp.mean(var_map, axis=(1, 2))

    # PSD
    ffts = rfft2_padded(residuals[..., 0], pad_factor=2)[..., None]
    psd2d = jnp.abs(ffts) ** 2.  # power
    mean_psd = jnp.mean(psd2d, axis=(1, 2))
    scale = jnp.sqrt(var_res / (mean_psd + 1e-12))
    amp_map = jnp.sqrt(psd2d) * scale[:, None, None, :]

    # Envelope
    # envelopes = estimate_envelopes(residuals, ctf[..., None], pixel_size=autoencoder.sr, k_min=0.01, k_max=0.5,
    #                                pad_shape=(ctf.shape[1], ctf.shape[1]))
    # x_clean_ctf = ctfFilter(x_clean_ctf[..., 0], envelopes[..., 0], pad_factor=2)[..., None]

    # Covariance
    z_rnd = []
    for _ in range(20):
        key = jnr.PRNGKey(random.randint(0, 2 ** 32 - 1))
        # noise = jax.random.normal(key, x_clean.shape) * jnp.sqrt(var_map)
        noise = jax.random.normal(key, x_clean.shape)
        noise = ctfFilter(noise[..., 0], amp_map[..., 0], pad_factor=2)[..., None]
        x_noisy = x_clean_ctf + noise
        z_rnd.append(model(x_noisy))
    return jnp.stack(z_rnd, axis=1), latent




def main():
    import os
    import sys
    from tqdm import tqdm
    import numpy as np
    import argparse
    from einops import rearrange
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, extract_columns
    from hax.programs import estimate_latent_covariances

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, type=str,
                        help="Xmipp metadata file with the images (+ alignments / CTF) needde to predict the covariances")
    parser.add_argument("--pickled_nn", required=False, type=str,
                        help=f"Path to folder containing a pickled neural network (generated from {bcolors.UNDERLINE}mode send_to_pickle{bcolors.ENDC}")
    parser.add_argument("--batch_size", required=False, type=int, default=64,
                        help="Determines how many images will be load in the GPU at any moment during training (set by default to 8 - "
                             f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                             f"to monitor and/or measure memory usage and adjust this value")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the estimated covariances")
    args = parser.parse_args()

    # Load neural network (note it MUST be saved in pickle mode to make this script general)
    model = NeuralNetworkCheckpointer.load(args.pickled_nn)

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Prepare data loader
    data_loader = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=False, preShuffle=False, prefetch=20)

    # Estimate covariances
    print(f"{bcolors.OKCYAN}\n###### Estimating covariance matrices... ######")
    pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green")
    covariances = []
    latents = []
    for (x, labels) in pbar:
        z_rnd, z = estimate_latent_covariances(model, x, labels, md_columns)
        latents.append(z)
        diff = z_rnd - z_rnd.mean(axis=1)[:, None, :]
        for d in diff:
            covariances.append(jnp.matmul(rearrange(d, "m n -> n m"), d) / d.shape[0])
    covariances = jnp.stack(covariances, axis=0)
    latents = jnp.vstack(latents)

    # Save covariances
    np.save(os.path.join(args.output_path, "covariance_matrices.npy"), covariances)
    np.save(os.path.join(args.output_path, "latents.npy"), latents)

if __name__ == "__main__":
    main()
