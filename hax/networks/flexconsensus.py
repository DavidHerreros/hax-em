#!/usr/bin/env python


import random

import jax
from jax import random as jnr, numpy as jnp
from flax import nnx
import dm_pix

from einops import rearrange

from hax.utils import *
from hax.layers import siren_init_first, siren_init
from hax.networks import ImageAdjustment


def pairwise_distances(embeddings: jnp.ndarray, squared: bool = False) -> jnp.ndarray:
    """
    Compute the 2D matrix of pairwise (squared) Euclidean distances.

    Args:
        embeddings: array of shape (batch_size, embed_dim)
        squared: if True, return squared distances; else return Euclidean distances.

    Returns:
        distances: array of shape (batch_size, batch_size)
    """
    # (batch_size, batch_size)
    dot_product = embeddings @ embeddings.T

    # (batch_size,)
    square_norm = jnp.diag(dot_product)

    # ||a - b||^2 = ||a||^2 - 2<a,b> + ||b||^2
    distances = (
        square_norm[:, None]
        - 2.0 * dot_product
        + square_norm[None, :]
    )

    # Clamp tiny negatives to zero from numerical errors
    distances = jnp.maximum(distances, 0.0)

    if not squared:
        # Avoid inf gradient at 0 by adding tiny epsilon only where needed
        mask = distances == 0.0
        eps = jnp.array(1e-16, dtype=embeddings.dtype)
        distances = distances + mask.astype(embeddings.dtype) * eps
        distances = jnp.sqrt(distances)
        # Put exact zeros back on the masked entries
        distances = distances * (~mask).astype(embeddings.dtype)

    return distances


class Encoder(nnx.Module):
    def __init__(self, input_dim, *, rngs: nnx.Rngs):
        self.input_dim = input_dim

        self.hidden_layers = [nnx.Linear(self.input_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(3):
            self.hidden_layers.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))

    def __call__(self, x):
        for layer in self.hidden_layers:
            if layer.in_features != layer.out_features:
                x = nnx.relu(layer(x))
            else:
                x = nnx.relu(x + layer(x))
        return x


class Decoder(nnx.Module):
    def __init__(self, lat_dim, output_dim, *, rngs: nnx.Rngs):

        self.hidden_layers = [nnx.Linear(lat_dim, 1024, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(3):
            self.hidden_layers.append(nnx.Linear(1024, 1024, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers.append(nnx.Linear(1024, output_dim, rngs=rngs))

    def __call__(self, x):
        for layer in self.hidden_layers[:-1]:
            if layer.in_features != layer.out_features:
                x = nnx.relu(layer(x))
            else:
                x = nnx.relu(x + layer(x))
        out = self.hidden_layers[-1](x)
        return out


class FlexConsensus(nnx.Module):
    def __init__(self, input_spaces_dim, input_spaces_name=None, lat_dim=None, *, rngs: nnx.Rngs):
        super(FlexConsensus, self).__init__()
        self.input_spaces_dim = input_spaces_dim
        lat_dim = min(input_spaces_dim) if lat_dim is None else lat_dim

        # Give default input_spaces_name if not provided
        if input_spaces_name is None:
            self.input_spaces_name = [f"Input_{idx:02d}" for idx in range(len(input_spaces_dim))]
        else:
            self.input_spaces_name = input_spaces_name

        self.encoders = {input_space_name + "_encoder": Encoder(input_space_dim, rngs=rngs) for input_space_dim, input_space_name in zip(input_spaces_dim, self.input_spaces_name)}
        self.decoders = {input_space_name + "_decoder": Decoder(lat_dim, input_space_dim, rngs=rngs) for input_space_dim, input_space_name in zip(input_spaces_dim, self.input_spaces_name)}
        self.consensus_space = nnx.Linear(1024, lat_dim, rngs=rngs)

    def __call__(self, x, space_name_encoder, space_name_decoder=None):
        encoded = self.encoders[space_name_encoder](x)
        if space_name_decoder is not None:
            decoded = self.decoders[space_name_decoder](encoded)
            return encoded, decoded
        else:
            return encoded


@jax.jit
def train_step_flexconsensus(graphdef, state, x):
    model, optimizer = nnx.merge(graphdef, state)

    # Save encoder and decoder losses
    encoder_losses = 0.0
    total_losses = 0.0
    decoder_losses = {input_space_name: 0.0 for input_space_name in model.input_spaces_name}

    # VMAP functions for later
    pairwise_distances_batch = jax.vmap(pairwise_distances)

    def loss_fn(model, x, input_space_idx):
        x_jnp = jnp.array(x)

        # Encode space
        consensus_spaces = [model.consensus_space(model.encoders[input_space_name + "_encoder"](input_space)) for input_space, input_space_name in zip(x, model.input_spaces_name)]  # TODO: Can we find the matching better?

        # Decode spaces
        decoded_spaces = [model.decoders[input_space_name + "_decoder"](consensus_spaces[input_space_idx]) for input_space_name in model.input_spaces_name]

        # Consensus loss (single space)
        diffs = x_jnp[:, None, :, :] - x_jnp[None, :, :, :]
        feature_mse_loss = jnp.mean(diffs ** 2, axis=(2, 3))

        # Distance matrix for Wasserstein distance
        distance_matrices = pairwise_distances_batch(jnp.array(consensus_spaces))
        mean = distance_matrices.mean(axis=(1, 2), keepdims=True)
        std = distance_matrices.std(axis=(1, 2), keepdims=True) + 1e-12
        distance_matrices = (distance_matrices - mean) / std

        # Upper triangular indices
        iu, ju = jnp.triu_indices(distance_matrices.shape[-1], k=1)
        distances = distance_matrices[:, iu, ju]
        distances_sorted = jnp.sort(distances, axis=1)

        # Wasserstein distance
        wdiff = jnp.abs(distances_sorted[:, None, :] - distances_sorted[None, :, :])
        wasserstein_loss = wdiff.mean(axis=-1)

        # Combine losses
        single_space_loss = feature_mse_loss.mean() + wasserstein_loss.mean()

        # Consensus loss (Shannon mapping)
        shannon_mapping_loss = 0.0
        distances_consensus_space = pairwise_distances(consensus_spaces[input_space_idx])[None, ...]
        distances_input_space = pairwise_distances_batch(x_jnp)
        triu_consensus_space = jnp.triu(distances_consensus_space, k=1)
        triu_distances_input_space = jnp.triu(distances_input_space, k=1)
        denominator_without_zeros = jnp.where(triu_distances_input_space == 0, 1.0, triu_distances_input_space)
        shannon_mapping_loss += (jnp.where(triu_distances_input_space == 0., 0.0, jnp.square(triu_distances_input_space - triu_consensus_space) / denominator_without_zeros).sum(axis=(1, 2))
                                 * 1. / triu_distances_input_space.sum(axis=(1, 2))).mean()

        # Consensus loss (Center of mass)
        center_of_mass_loss = jnp.square(consensus_spaces[input_space_idx].mean(axis=0)).sum()

        # Decoder loss
        representation_loss = jnp.mean(jnp.square(x_jnp - jnp.array(decoded_spaces)), axis=-1).mean()

        # Compute total loss
        loss = single_space_loss + shannon_mapping_loss + center_of_mass_loss + representation_loss

        return loss, (single_space_loss + shannon_mapping_loss + center_of_mass_loss, representation_loss)

    for input_space_idx in range(len(model.input_spaces_name)):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, (encoder_loss, decoder_loss)), grads = grad_fn(model, x, input_space_idx)

        optimizer.update(grads)

        # Save losses
        encoder_losses += encoder_loss
        decoder_losses[model.input_spaces_name[input_space_idx]] = decoder_loss
        total_losses += loss

    encoder_losses /= len(x)
    total_losses /= len(x)

    state = nnx.state((model, optimizer))

    return total_losses, encoder_losses, decoder_losses, state




def main():
    import os
    from pathlib import Path
    import sys
    from tqdm import tqdm
    import random
    import numpy as np
    import argparse
    import optax
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import ArrayListGenerator, NumpyGenerator
    from hax.metrics import JaxSummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_space", required=True, type=str, nargs='+',
                        help="Paths to a series of .npy files containing the input spaces to be consensuated. "
                             f"{bcolors.WARNING}NOTE{bcolors.ENDC}: You can pass multiple times this parameter to specify a different file."
                             f'{bcolors.WARNING}NOTE{bcolors.ENDC}: You can pass either the files directly or with the format "{bcolors.UNDERLINE}--input_space NAME:path{bcolors.ENDC}" '
                             f'Using this alternative format allows giving identifiers to each input so it is easier to predict with the network later on. If only the file is given, '
                             f'each input will be named as "Input_#" based on their command line order.'
                             f'{bcolors.WARNING}NOTE{bcolors.ENDC}: If mode parameter is set to predict, we recommend using the NAME:path convention to simplify the prediction process. If not '
                             f'given, the network will attempt to indentify automatically your method.'
                             f"{bcolors.WARNING}WARNING{bcolors.ENDC}: The spaces stored in the files MUST have the SAME number of elements and they MUST be "
                             f"ordered (i.e. the first element in the files is the latent vector obtained from the first image and so on).")
    parser.add_argument("--lat_dim", required=False,
                        help="Dimensionality of the latent space of the network (by default, it is set to the minimum dimension of all the provided input spaces)")
    parser.add_argument("--mode", required=True, type=str, choices=["train", "predict", "send_to_pickle"],
                        help=f"{bcolors.BOLD}train{bcolors.ENDC}: train a neural network from scratch or from a previous execution if reload is provided\n"
                             f"{bcolors.BOLD}predict{bcolors.ENDC}: predict the latent vectors from the input images ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case)")
    parser.add_argument("--epochs", required=False, type=int, default=100,
                        help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset of images - set to default to 50 - "
                             "as a rule of thumb, consider 50 to 100 epochs enough for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")
    parser.add_argument("--batch_size", required=False, type=int, default=1024,
                        help="Determines how many images will be load in the GPU at any moment during training (set by default to 8 - "
                             f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                             f"to monitor and/or measure memory usage and adjust this value - keep also in mind that bigger batch sizes might be less precise when looking for very local motions")
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-5,
                        help=f"The learning rate ({bcolors.ITALIC}lr{bcolors.ENDC}) sets the speed of learning. Think of the model as trying to find the lowest point in a valley; the {bcolors.ITALIC}lr{bcolors.ENDC} "
                             f"is the size of the step it takes on each attempt. A large {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}0.01{bcolors.ENDC}) is like taking huge leaps — it's fast but can be unstable, "
                             f"overshoot the lowest point, or cause {bcolors.ITALIC}NAN{bcolors.ENDC} errors. A small {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}1e-6{bcolors.ENDC}) is like taking tiny "
                             f"shuffles — it's stable but very slow and might get stuck before reaching the bottom. A good default is often {bcolors.ITALIC}0.0001{bcolors.ENDC}. If training fails or errors explode, "
                             f"try making the {bcolors.ITALIC}lr{bcolors.ENDC} 10 times smaller (e.g., {bcolors.ITALIC}0.001{bcolors.ENDC} --> {bcolors.ITALIC}0.0001{bcolors.ENDC}).")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, consensus spaces...)")
    parser.add_argument("--reload", required=False, type=str,
                        help=f"Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data)")
    args = parser.parse_args()

    # If NAME:path convention, split both
    if ":" in args.input_space[0]:
        input_space = []
        input_spaces_name = []
        for name_path in args.input_space:
            name, path = name_path.split(":")
            input_space.append(path)
            input_spaces_name.append(name)
    else:
        input_space = args.input_space
        input_spaces_name = None

    # Get reading function for each file
    reading_fns = []
    for file in input_space:
        if os.path.splitext(file)[-1] == ".npy":
            reading_fns.append(np.load)
        elif os.path.splitext(file)[-1] == ".txt":
            reading_fns.append(np.loadtxt)
        else:
            raise ValueError(f"One of the filed you provided has a non supported extension. Please, provide either a "
                             f"file with {bcolors.ITALIC}.npy{bcolors.ENDC} or {bcolors.ITALIC}.txt{bcolors.ENDC} extension.")

    # Read input spaces
    input_spaces = [reading_fn(file) for file, reading_fn in zip(input_space, reading_fns)]
    input_spaces_dim = [input_space.shape[-1] for input_space in input_spaces]

    # Get latent dimensions
    if args.lat_dim is None:
        lat_dim = 2 ** 32 - 1
        for input_space in input_spaces:
            lat_dim = min(lat_dim, input_space.shape[0])
    else:
        lat_dim = args.lat_dim

    # Random keys
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key = jax.random.split(rng, 2)

    # Prepare network (FlexConsensus)
    flexconsensus = FlexConsensus(input_spaces_dim, input_spaces_name, lat_dim, rngs=nnx.Rngs(model_key))

    # Reload network
    if args.reload is not None:
        flexconsensus = NeuralNetworkCheckpointer.load(flexconsensus, os.path.join(args.reload, "FlexConsensus"))

    # Train network
    if args.mode == "train":

        flexconsensus.train()

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "FlexConsensus_metrics"))

        # Prepare data loader
        data_loader = ArrayListGenerator(input_spaces).return_tf_dataset(batch_size=args.batch_size, shuffle=True, preShuffle=True)

        # Optimizers (FlexConsensus)
        optimizer = nnx.Optimizer(flexconsensus, optax.adam(args.learning_rate))
        graphdef, state = nnx.split((flexconsensus, optimizer))

        # Training loop (FlexConsensus)
        print(f"{bcolors.OKCYAN}\n###### Training consensus... ######")
        for i in range(args.epochs):
            total_loss = 0
            total_encoder_loss = 0
            total_decoder_loss = {input_space_name: 0 for input_space_name in flexconsensus.input_spaces_name}

            # For progress bar (TQDM)
            step = 1
            print(f'\nTraining epoch {i + 1}/{args.epochs} |')
            pbar = tqdm(data_loader, desc=f"Epoch {i + 1}/{args.epochs}", file=sys.stdout, ascii=" >=", colour="green")

            for (x, _) in pbar:
                loss, encoder_loss, decoder_loss, state = train_step_flexconsensus(graphdef, state, x)
                total_loss += loss
                total_encoder_loss += encoder_loss
                total_decoder_loss = {i: total_decoder_loss.get(i, 0) + decoder_loss.get(i, 0) for i in set(total_decoder_loss).union(decoder_loss)}

                # Progress bar update  (TQDM)
                loss_str_decoder_loss = " | ".join([f'{key}_decoder={value / step:.5f}' for key, value in total_decoder_loss.items()])
                pbar.set_postfix_str(f"loss={total_loss / step:.5f} | encoder_loss={encoder_loss / step:.5f} | " + loss_str_decoder_loss)

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * len(data_loader))) == 0:
                    writer.add_scalar('Training loss (FlexConsensus)',
                                      total_loss / step,
                                      i * len(data_loader) + step)
                    writer.add_scalar('Encoder loss (FlexConsensus)',
                                      encoder_loss / step,
                                      i * len(data_loader) + step)
                    for key, value in total_encoder_loss.items():
                        writer.add_scalar(f'Decoder loss ({key})',
                                          value / step,
                                          i * len(data_loader) + step)
                step += 1

        flexconsensus, optimizer = nnx.merge(graphdef, state)

        # Save model
        NeuralNetworkCheckpointer.save(flexconsensus, os.path.join(args.output_path, "FlexConsensus"), mode="pickle")

    elif args.mode == "predict":

        flexconsensus.train()

        # Jitted prediction function
        predict_fn = jax.jit(flexconsensus.__call__)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting consensus latents... ######")
        latents = []
        for idx in range(len(input_spaces)):
            print(f"Predicting input space {idx + 1}/{len(input_spaces)}")

            # Prepare data loader
            data_loader = NumpyGenerator(input_spaces[idx]).return_tf_dataset(batch_size=args.batch_size, shuffle=False, preShuffle=False, prefetch=20)

            if input_spaces_name is not None and input_spaces_name[idx] in flexconsensus.input_spaces_name:
                print(f"Valid identifier {input_spaces_name[idx]} provided for this input")
                consensus_latents = []
                # For progress bar (TQDM)
                pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green")

                for (x, _) in pbar:
                    consensus_latents.append(predict_fn(x, space_name_encoder=input_spaces_name[idx]))
            else:
                print(f"Matching and detecting best possible prediction (due to missing or not valid identifier)")
                representation_error = jnp.inf
                consensus_latents = None
                for input_space_name in flexconsensus.input_spaces_name:
                    consensus_latents_trial = []
                    representation_error_trial = 0
                    print(f"Trying with {input_space_name}")

                    # For progress bar (TQDM)
                    pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green")

                    for (x, _) in pbar:
                        encoded, decoded = predict_fn(x, space_name_encoder=input_space_name, space_name_decoder=input_space_name)
                        consensus_latents_trial.append(encoded)
                        representation_error_trial += jnp.mean(jnp.square(x - decoded), axis=-1).mean()

                    representation_error_trial /= len(data_loader)

                    if representation_error_trial < representation_error:
                        consensus_latents = consensus_latents_trial

            latents.append(np.array(consensus_latents))

        # Save consensus latents
        for latent, input_file in zip(latents, args.input_spaces):
            output_file = os.path.join(args.output_path, str(Path(input_file).stem) + "_consensus.npy")
            np.save(output_file, latent)

if __name__ == "__main__":
    main()
