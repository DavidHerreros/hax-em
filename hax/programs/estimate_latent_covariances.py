#!/usr/bin/env python


from functools import partial

import jax
from jax import random as jnr, numpy as jnp

from hax.utils import *


@partial(jax.jit, static_argnames=["model", "n_samples"])
def estimate_latent_covariances(model, x, labels, md, key, n_samples=20):
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
    for sample_key in jnr.split(key, n_samples):
        # noise = jax.random.normal(sample_key, x_clean.shape) * jnp.sqrt(var_map)
        noise = jax.random.normal(sample_key, x_clean.shape)
        noise = ctfFilter(noise[..., 0], amp_map[..., 0], pad_factor=2)[..., None]
        x_noisy = x_clean_ctf + noise
        z_rnd.append(model(x_noisy, return_alignment_refinement=False))
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

    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser, help="Xmipp metadata file with the images (+ alignments / CTF) needed to predict the covariances")
    parser.add_argument("--nn_path", required=False, type=str,
                        help=f"Path to folder containing a saved neural network (HetSIREN, Zernike3Deep...)")
    ca.add_batch_size(parser, default=64, help=ca.BATCH_SIZE_HELP_ADJUST)
    ca.add_output_path(parser, help="Path to save the estimated covariances")
    ca.add_load_images_to_ram(parser)
    ca.add_ssd_scratch_folder(parser)
    args = ca.parse_with_config(parser)

    # Load neural network
    model = NeuralNetworkCheckpointer.load(args.nn_path)

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Prepare grain dataset
    if not args.load_images_to_ram:
        mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                             precision=np.float16, group_size=1, shard_size=10000)

    # Prepare data loader
    data_loader = generator.return_grain_dataset(batch_size=args.batch_size, shuffle=False, num_epochs=1,
                                                 num_workers=-1, load_to_ram=args.load_images_to_ram)
    steps_per_epoch = int(np.ceil(len(generator.md) / args.batch_size))

    # Estimate covariances
    print(f"{bcolors.OKCYAN}\n###### Estimating covariance matrices... ######")
    pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green", total=steps_per_epoch,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    covariances = []
    latents = []
    key = jnr.PRNGKey(0)
    for batch_idx, (x, labels) in enumerate(pbar):
        batch_key = jnr.fold_in(key, batch_idx)
        z_rnd, z = estimate_latent_covariances(model, x, labels, md_columns, batch_key)
        latents.append(z)
        diff = z_rnd - z_rnd.mean(axis=1)[:, None, :]
        for d in diff:
            covariances.append(jnp.matmul(rearrange(d, "m n -> n m"), d) / d.shape[0])
    covariances = jnp.stack(covariances, axis=0)
    latents = jnp.vstack(latents)

    # Save covariances
    np.save(os.path.join(args.output_path, "covariance_matrices.npy"), covariances)
    np.save(os.path.join(args.output_path, "latents.npy"), latents)
