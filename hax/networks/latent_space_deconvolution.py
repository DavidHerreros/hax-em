#!/usr/bin/env python


import jax
from jax import random as jnr, numpy as jnp
from flax import nnx

import random
from functools import partial

from hax.utils.miscellaneous import batched_knn
from hax.utils.loggers import bcolors
from hax.utils.decorators import save_config


class Deconvolver(nnx.Module):
    @save_config
    def __init__(self, lat_dim=10, n_layers=3, *, rngs: nnx.Rngs):
        self.lat_dim = lat_dim

        hidden_layers = [nnx.Linear(lat_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(n_layers):
            hidden_layers.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        hidden_layers.append(nnx.Linear(1024, 256, rngs=rngs, dtype=jnp.bfloat16))
        for _ in range(2):
            hidden_layers.append(nnx.Linear(256, 256, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers = nnx.List(hidden_layers)
        self.latent = nnx.Linear(256, lat_dim, rngs=rngs, kernel_init=jax.nn.initializers.uniform(0.0001))

    def __call__(self, x):
        aux = x
        for layer in self.hidden_layers:
            if layer.in_features != layer.out_features:
                aux = nnx.relu(layer(aux))
            else:
                aux = nnx.relu(aux + layer(aux))
        aux = self.latent(aux)
        mean = x + aux

        return mean


@partial(jax.jit, static_argnames=["islog", "fraction", "subsetMode"])
def train_deconv_step(graphdef, state, x, cov, z_space, islog=False, fraction=None, subsetMode="random"):
    model, optimizer_deconv = nnx.merge(graphdef, state)

    key = jnr.PRNGKey(random.randint(0, 2 ** 32 - 1))

    def density_at(z, mean, cov, islog=False):
        if islog:
            lp = jax.scipy.stats.multivariate_normal.logpdf(z, mean, cov)
            # return jax.scipy.special.logsumexp(lp, axis=0) - jnp.log(lp.shape[0])  # scalar
        else:
            lp = jax.scipy.stats.multivariate_normal.pdf(z, mean, cov)
            # return jnp.mean(lp, axis=0)  # scalar
        return lp

    if islog:
        reduced_kernel = lambda x, means, cov, in_log: jax.scipy.special.logsumexp(
            jax.vmap(density_at, in_axes=(None, 0, 0, None))
            (x, means, cov, in_log),
            axis=0) - jnp.log(means.shape[0])
    else:
        reduced_kernel = lambda x, means, cov, in_log: jnp.mean(jax.vmap(density_at, in_axes=(None, 0, 0, None))
                                                                (x, means, cov, in_log),
                                                                axis=0)
    density_at_vmap = jax.vmap(reduced_kernel, in_axes=(0, None, None, None))

    def loss_fn(model, z, d_z, cov, kde_cov, z_space):
        z_pp_space = jax.vmap(model)(z_space)

        # Silverman factor
        weights = jnp.full(z_space.shape[0], 1.0 / z_space.shape[0], dtype=z_space.dtype)
        neff = 1. / jnp.sum(weights ** 2.)
        factor = jnp.power(neff * (z_space.shape[1] + 2.) / 4.0, -1. / (z_space.shape[1] + 4.))

        # Deconvolved landscape covariance
        cov_true = jnp.atleast_2d(jnp.cov(z_pp_space.T, rowvar=True, bias=False, aweights=None))
        cov_true = cov_true * factor ** 2.

        # Fused covariances
        cov_all = kde_cov[None, :, :] + cov

        # Density (all)
        d_zpp = density_at_vmap(z, z_pp_space, cov_all, islog)
        if islog:
            d_zpp = d_zpp - jax.scipy.special.logsumexp(d_zpp)
        else:
            d_zpp = d_zpp / d_zpp.sum()

        # First order derivative regularization (smoothing)
        # M, D = z_pp_space.shape
        # inv_cov = jnp.linalg.inv(cov_true)
        # sign, logdet = jnp.linalg.slogdet(cov_true)
        # log_norm = 0.5 * (D * jnp.log(2.0 * jnp.pi) + logdet)
        # diffs = z[:, None, :] - z_pp_space[None, :, :]
        # exp_arg = jnp.einsum('ijd,dc,ijc->ij', diffs, inv_cov, diffs, precision=jax.lax.Precision.HIGH)
        # K = jnp.exp(-0.5 * exp_arg - log_norm) / M
        # K = K / K.sum()
        # inv_diffs = jnp.einsum('dc,ijd->ijd', inv_cov, diffs, precision=jax.lax.Precision.HIGH)
        # grads = -jnp.einsum('ij,ijd->id', K, inv_diffs, precision=jax.lax.Precision.HIGH)
        # loss_first_order = jnp.mean(grads ** 2.)

        # return jnp.square(d_zpp - d_z).mean() + 0.0001 * loss_first_order
        return jnp.square(d_zpp - d_z).mean()

    if fraction is None:
        fraction = 1000
    elif isinstance(fraction, float):
        fraction = int(z_space.shape[0] * fraction)

    if subsetMode == "random":
        rnd_ids = jax.random.choice(key, jnp.arange(z_space.shape[0], dtype=jnp.int32),
                                    shape=(min(fraction, z_space.shape[0]),), replace=False)
    elif subsetMode == "random_nn":
        rnd_ids, _ = batched_knn(z_space, x, k=1000, block_size=min(100000, z_space.shape[0]))
        rnd_ids = rnd_ids.flatten()
        # rnd_ids = jax.random.choice(key, nn_ids, shape=(min(fraction, len(nn_ids)),), replace=False)
    else:
        raise ValueError("Subset mode not implemented")
    z_space = z_space[rnd_ids]
    cov = cov[rnd_ids]

    kde = jax.scipy.stats.gaussian_kde(z_space.T)

    if islog:
        d = kde.logpdf(x.T)
        d = d - jax.scipy.special.logsumexp(d)
    else:
        d = kde.pdf(x.T)
        d = d / d.sum()

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(model, x, d, cov, kde.covariance, z_space)

    optimizer_deconv.update(model, grads)

    state = nnx.state((model, optimizer_deconv))

    return loss, state



def main():
    import os
    import sys
    from tqdm import tqdm
    import random
    import numpy as np
    from sklearn.decomposition import PCA
    import argparse
    import optax
    from contextlib import closing
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, NumpyGenerator
    from hax.networks import train_deconv_step

    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    ca.add_md(parser,
              help="Xmipp metadata file with the images (+ alignments / CTF) whose latent vectors will be "
                   "recovered on the fly from the provided network and then deconvolved")
    parser.add_argument("--nn_path", required=True, type=str,
                    help=f"Path to folder containing a saved neural network (HetSIREN, Zernike3Deep...). Its encoder is "
                         f"used to recover the latent vectors of the images in {bcolors.UNDERLINE}md{bcolors.ENDC} on the fly")
    parser.add_argument("--covariances", required=True, type=str,
                        help=f"Path to the .npy file with the covariances needed to estimate the deconvolution (output of {bcolors.UNDERLINE}estimate_latent_covariances{bcolors.ENDC} program)")
    ca.add_lat_dim(parser, default=3, help="Dimensionality of the latent space of the deconvolution network")
    ca.add_mode(parser, choices=["train", "predict"],
                help=f"{bcolors.BOLD}train{bcolors.ENDC}: train a neural network from scratch or from a previous execution if reload is provided\n"
                     f"{bcolors.BOLD}predict{bcolors.ENDC}: predict the deconvolved latents from the input latents ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case)")
    parser.add_argument("--deconvolution_strength", required=False, type=float, default=1.0,
                        help="Determines the deconvolution strength (set to 1.0 by default, meaning that the landscape will be deconvolved as expected from the computed covariances - larger values "
                             "will yield a stronger deconvolution compared to the default value, while smaller values will be more conservative")
    ca.add_epochs(parser, default=100,
                  help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset of images - set to default to 100 - "
                       "as a rule of thumb, consider 50 to 100 epochs enough for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")
    ca.add_batch_size(parser, default=1024, help=ca.BATCH_SIZE_HELP_ADJUST)
    ca.add_output_path(parser, help="Path to save the results (trained neural network, deconvolved latents...)")
    ca.add_reload(parser, help=ca.RELOAD_HELP_BASIC)
    ca.add_load_images_to_ram(parser)
    ca.add_ssd_scratch_folder(parser)
    args = parser.parse_args()

    # Ensure the output path exists (the deconvolved latents / model are written into it)
    os.makedirs(args.output_path, exist_ok=True)

    # Load neural network
    model = NeuralNetworkCheckpointer.load(args.nn_path)
    model.eval()

    # Prepare metadata / image data loader
    generator = MetaDataGenerator(args.md)
    if not args.load_images_to_ram:
        mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
        generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                             precision=np.float16, group_size=1, shard_size=10000)

    # No shuffling: the recovered latents stay aligned (by row) with the covariances
    image_loader = generator.return_grain_dataset(batch_size=args.batch_size, shuffle=False, num_epochs=1,
                                                  num_workers=-1, load_to_ram=args.load_images_to_ram)
    steps_per_epoch = int(np.ceil(len(generator.md) / args.batch_size))

    # Recover the latent vectors from the network on the fly (instead of a pre-saved file)
    print(f"{bcolors.OKCYAN}\n###### Recovering latent vectors from the network... ######")

    @nnx.jit
    def predict_latents(model, x):
        # The encoder returns (latent, (rotations, shifts)); only the latent is needed here.
        return model(x, return_alignment_refinement=False)

    pbar = tqdm(image_loader, desc="Progress", file=sys.stdout, ascii=" >=", colour="green",
                total=steps_per_epoch, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    latents = []
    for (x, labels) in pbar:
        if isinstance(x, tuple):   # Tomo mode returns (image, subtomo_label)
            x = x[0]
        latents.append(np.asarray(predict_latents(model, x)))
    latents = np.concatenate(latents, axis=0)

    # Prepare covariances (output of estimate_latent_covariances, same metadata order)
    covariances = args.deconvolution_strength * np.load(args.covariances)

    # Prepare network
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key = jax.random.split(rng, 2)
    deconvolver = Deconvolver(lat_dim=args.lat_dim, rngs=nnx.Rngs(model_key))

    # Reload network
    if args.reload is not None:
        deconvolver = NeuralNetworkCheckpointer.load(os.path.join(args.reload, "deconvolver"))

    # Train network
    if args.mode == "train":

        deconvolver.train()

        # Remove covariances above or below std in PCA
        cov_flat = covariances.reshape((covariances.shape[0], -1))
        cov_pca = PCA(n_components=1).fit_transform(cov_flat)
        cov_std = np.std(cov_pca)
        idx = np.where(np.logical_and(cov_pca < cov_std, cov_pca > -cov_std))[0]
        latents = latents[idx]
        covariances = covariances[idx]

        # Prepare data loader
        data_loader = NumpyGenerator(latents).return_grain_dataset(batch_size=args.batch_size, shuffle=True,
                                                                   preShuffle=True, num_workers=-1, num_epochs=None)
        steps_per_epoch = max(1, int(latents.shape[0] / args.batch_size))

        # Optimizers
        optimizer = nnx.Optimizer(deconvolver, optax.adam(1e-6), wrt=nnx.Param)
        graphdef, state = nnx.split((deconvolver, optimizer))

        # Training loop
        print(f"{bcolors.OKCYAN}\n###### Training deconvolution... ######")

        i = 0
        total_loss = 0
        step = 1
        pbar = tqdm(range(steps_per_epoch, args.epochs * steps_per_epoch), file=sys.stdout, ascii=" >=",
                    colour="green",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        with closing(iter(data_loader)) as iter_data_loader:
            for total_steps in pbar:
                (x, _) = next(iter_data_loader)

                if total_steps % steps_per_epoch == 0:

                    total_loss = 0

                    # For progress bar (TQDM)
                    step = 1
                    pbar.set_description(f"Epoch {int(total_steps / steps_per_epoch + 1)}/{args.epochs}")

                    i += 1

                loss, state = train_deconv_step(graphdef, state, x, covariances, latents, islog=True,
                                                subsetMode="random", fraction=50000)
                total_loss += loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f}")
                step += 1
        deconvolver, optimizer_deconv = nnx.merge(graphdef, state)

        # Save model
        NeuralNetworkCheckpointer.save(deconvolver, os.path.join(args.output_path, "deconvolver"))

    elif args.mode == "predict":

        deconvolver.eval()

        # Prepare data loader
        data_loader = NumpyGenerator(latents).return_grain_dataset(batch_size=args.batch_size, shuffle=False,
                                                                   preShuffle=False, num_workers=0, num_epochs=1)

        # Jitted prediciton function
        predict_fn = jax.jit(deconvolver.__call__)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting deconvolved latents... ######")

        latents_deconv = []
        with closing(iter(data_loader)) as iter_data_loader:
            for (x, _) in iter_data_loader:
                latents_deconv.append(np.asarray(predict_fn(x)))
        latents_deconv = np.concatenate(latents_deconv, axis=0)

        # Save new latents
        np.save(os.path.join(args.output_path, "latents_deconvolved.npy"), latents_deconv)
