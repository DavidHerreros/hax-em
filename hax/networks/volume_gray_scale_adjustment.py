#!/usr/bin/env python


from functools import partial

import jax
from jax import numpy as jnp
from flax import nnx
import dm_pix

from einops import rearrange

from hax.utils import *


class VolumeAdjustment(nnx.Module):
    def __init__(self, coords, values, lat_dim=10, predicts_value=True, *, rngs: nnx.Rngs):
        self.coords = coords
        self.values = values
        out_dim = 1 if predicts_value else self.values.shape[0]

        # Gray level adjustment (TODO: with 256 features OK)
        self.hidden_layers_ds = [nnx.Linear(self.coords.shape[0] * 3, 32, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(2):
            self.hidden_layers_ds.append(nnx.Linear(32, 32, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_us = [nnx.Linear(lat_dim, 32, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(2):
            self.hidden_layers_us.append(nnx.Linear(32, 32, rngs=rngs, dtype=jnp.bfloat16))
        self.latent = nnx.Linear(32, lat_dim, rngs=rngs)
        self.a = nnx.Linear(32, out_dim, rngs=rngs, kernel_init=jax.nn.initializers.normal(stddev=0.001), bias_init=nnx.initializers.ones)
        self.b = nnx.Linear(32, out_dim, rngs=rngs, kernel_init=jax.nn.initializers.normal(stddev=0.001))

    def __call__(self, return_ab=False):
        coords = rearrange(self.coords, "(b c) d -> b (c d)", b=1)

        partial_gray = nnx.relu(self.hidden_layers_ds[0](coords))
        for layer in self.hidden_layers_ds[1:]:
            partial_gray = nnx.relu(partial_gray + layer(partial_gray))

        partial_gray = self.latent(partial_gray)

        partial_gray = nnx.relu(self.hidden_layers_us[0](partial_gray))
        for layer in self.hidden_layers_us[1:]:
            partial_gray = nnx.relu(partial_gray + layer(partial_gray))

        a = nnx.relu(self.a(partial_gray))
        b = self.b(partial_gray)
        a, b = jnp.squeeze(a), jnp.squeeze(b)

        if return_ab:
            return a, b
        else:
            values = a * self.values + b
            # values = nnx.relu(values)  # FIXME: Review this

            return values


@partial(jax.jit, static_argnames=["sr", "ctf_type", "xsize"])
def train_step_volume_adjustment(graphdef, state, x, labels, md, sr, ctf_type, xsize):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, x):
        factor = 0.5 * xsize

        # Gray level adjustment
        values = model()

        # Rotate grid
        rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        coords = jnp.matmul(factor * model.coords, rearrange(rotations, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts[:, None, :] + factor

        # Scatter image
        B = x.shape[0]
        c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
        images = jnp.zeros((B, xsize, xsize), dtype=x.dtype)

        # bposf = jnp.round(c_sampling)
        # bposi = bposf.astype(jnp.int32)
        bamp = values[None, ...]

        bposf = jnp.floor(c_sampling)
        bposi = bposf.astype(jnp.int32)
        bposf = c_sampling - bposf

        bamp0 = bamp * (1.0 - bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp1 = bamp * (bposf[:, :, 0]) * (1.0 - bposf[:, :, 1])
        bamp2 = bamp * (bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp3 = bamp * (1.0 - bposf[:, :, 0]) * (bposf[:, :, 1])
        bamp = jnp.concat([bamp0, bamp1, bamp2, bamp3], axis=1)
        bposi = jnp.concat([bposi, bposi + jnp.array((1, 0)), bposi + jnp.array((1, 1)), bposi + jnp.array((0, 1))],
                           axis=1)

        def scatter_img(image, bpos_i, bamp_i):
            return image.at[bpos_i[..., 0], bpos_i[..., 1]].add(bamp_i)

        images = jax.vmap(scatter_img)(images, bposi, bamp)

        # Gaussian filter (needed by forward interpolation)
        images = jnp.squeeze(dm_pix.gaussian_blur(images[..., None], 1.0, kernel_size=3))

        # Prepare data for losses
        images = jnp.squeeze(images)
        x = jnp.squeeze(x)

        # Consider CTF
        if ctf_type == "apply":
            images = ctfFilter(images, ctf, pad_factor=2)
        elif ctf_type == "wiener":
            x = wiener2DFilter(x, ctf, pad_factor=2)
        elif ctf_type == "squared":
            x = ctfFilter(x, ctf, pad_factor=2)
            images = ctfFilter(images, ctf * ctf, pad_factor=2)

        # Loss
        loss = dm_pix.mse(images[..., None], x[..., None]).mean()
        return loss

    # Precompute batch aligments
    euler_angles = md["euler_angles"][labels]

    # Precompute batch shifts
    shifts = md["shifts"][labels]

    # Precompute batch CTFs
    if ctf_type is not None:
        defocusU = md["ctfDefocusU"][labels]
        defocusV = md["ctfDefocusV"][labels]
        defocusAngle = md["ctfDefocusAngle"][labels]
        cs = md["ctfSphericalAberration"][labels]
        kv = md["ctfVoltage"][labels][0]
        ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         sr, [2 * xsize, int(2 * 0.5 * xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], 2 * xsize, int(2.0 * 0.5 * xsize + 1)], dtype=x.dtype)

    if ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(jnp.squeeze(x), ctf)[..., None]

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, x)

    optimizer.update(grads)

    state = nnx.state((model, optimizer))

    return loss, state




def main():
    import os
    import sys
    from tqdm import tqdm
    import random
    import numpy as np
    import argparse
    from xmipp_metadata.image_handler import ImageHandler
    import optax
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, extract_columns
    from hax.networks import train_step_volume_adjustment
    from hax.metrics import JaxSummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=False, type=str,
                        help="Xmipp/Relion metadata file with the images (+ alignments / CTF) to be analyzed")
    parser.add_argument("--vol", required=True, type=str,
                        help="Volume to be adjusted according to the images")
    parser.add_argument("--mask", required=False, type=str,
                        help="This helps focusing the adjustment on a region of interest (if not provided, a inscribed spherical mask will be used)")
    parser.add_argument("--load_images_to_ram", action='store_true',
                        help=f"If provided, images will be loaded to RAM. This is recommended if you want the best performance and your dataset fits in your RAM memory. If this flag is not provided, "
                             f"images will be memory mapped. When this happens, the program will trade disk space for performance. Thus, during the execution additional disk space will be used and the performance "
                             f"will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal once the execution has finished.")
    parser.add_argument("--sr", required=True, type=float,
                        help="Sampling rate of the images/volume")
    parser.add_argument("--ctf_type", required=True, type=str, choices=["None", "apply", "wiener", "precorrect"],
                        help="Determines whether to consider the CTF and, in case it is considered, whether it will be applied to the projections (apply) or used to correct the metadata images (wiener - precorrect)")
    parser.add_argument("--predicts_value", action='store_true',
                        help="If not provided, the adjustment will be estimated per voxel - otherwise, adjustment will be estimated for the whole volume")
    parser.add_argument("--lat_dim", required=False, type=int, default=3,
                        help="Dimensionality of the latent space of the network (set by default to 3)")
    parser.add_argument("--mode", required=True, type=str, choices=["train", "predict"],
                        help=f"{bcolors.BOLD}train{bcolors.ENDC}: train a neural network from scratch or from a previous execution if reload is provided\n"
                             f"{bcolors.BOLD}predict{bcolors.ENDC}: predict the adjustment for the input volume ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case)")
    parser.add_argument("--epochs", required=False, type=int, default=50,
                        help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset of images - set to default to 50 - "
                             "as a rule of thumb, consider 50 to 100 epochs enough for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")
    parser.add_argument("--batch_size", required=False, type=int, default=64,
                        help="Determines how many images will be load in the GPU at any moment during training (set by default to 8 - "
                             f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                             f"to monitor and/or measure memory usage and adjust this value")
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-5,
                        help=f"The learning rate ({bcolors.ITALIC}lr{bcolors.ENDC}) sets the speed of learning. Think of the model as trying to find the lowest point in a valley; the {bcolors.ITALIC}lr{bcolors.ENDC} "
                             f"is the size of the step it takes on each attempt. A large {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}0.01{bcolors.ENDC}) is like taking huge leaps — it's fast but can be unstable, "
                             f"overshoot the lowest point, or cause {bcolors.ITALIC}NAN{bcolors.ENDC} errors. A small {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}1e-6{bcolors.ENDC}) is like taking tiny "
                             f"shuffles — it's stable but very slow and might get stuck before reaching the bottom. A good default is often {bcolors.ITALIC}0.0001{bcolors.ENDC}. If training fails or errors explode, "
                             f"try making the {bcolors.ITALIC}lr{bcolors.ENDC} 10 times smaller (e.g., {bcolors.ITALIC}0.001{bcolors.ENDC} --> {bcolors.ITALIC}0.0001{bcolors.ENDC}).")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, adjusted volume...)")
    parser.add_argument("--reload", required=False, type=str,
                        help="Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data)")
    args = parser.parse_args()

    # Preprocess volume (and mask)
    vol = ImageHandler(args.vol).getData()

    if args.mask is not None:
        mask = ImageHandler(args.mask).getData()
    else:
        mask = ImageHandler().createCircularMask(boxSize=vol.shape[0], is3D=True)

    # Data loading approach
    if args.load_images_to_ram:
        mmap = False
        mmap_output_dir = None
    else:
        mmap = True
        mmap_output_dir = args.output_path

    inds = np.asarray(np.where(mask > 0.0)).T
    values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]

    factor = 0.5 * vol.shape[0]
    coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1)
    coords = (coords - factor) / factor

    # Prepare network
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key = jax.random.split(rng, 2)
    volumeAdjustment = VolumeAdjustment(lat_dim=args.lat_dim, coords=coords, values=values, predicts_value=args.predicts_value, rngs=nnx.Rngs(model_key))

    # Reload network
    if args.reload is not None:
        volumeAdjustment = NeuralNetworkCheckpointer.load(volumeAdjustment, os.path.join(args.reload, "volumeAdjustment"))

    # Train network
    if args.mode == "train":

        volumeAdjustment.train()

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "Volume_adjustment_metrics"))

        # Prepare metadata
        generator = MetaDataGenerator(args.md)
        md_columns = extract_columns(generator.md)
        data_loader = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=True, preShuffle=True,
                                                  mmap=mmap, mmap_output_dir=mmap_output_dir)

        # Example of training data for Tensorboard
        x_example, labels_example = next(iter(data_loader))
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Training data batch", x_example, dataformats="NHWC")

        # Optimizers
        optimizer = nnx.Optimizer(volumeAdjustment, optax.adam(args.learning_rate))
        graphdef, state = nnx.split((volumeAdjustment, optimizer))

        # Training loop
        print(f"{bcolors.OKCYAN}\n###### Training volume adjustment... ######")
        for i in range(args.epochs):
            total_loss = 0

            # For progress bar (TQDM)
            step = 1
            print(f'\nTraining epoch {i + 1}/{args.epochs} |')
            pbar = tqdm(data_loader, desc=f"Epoch {i + 1}/{args.epochs}", file=sys.stdout, ascii=" >=",
                        colour="green")

            for (x, labels) in pbar:
                loss, state = train_step_volume_adjustment(graphdef, state, x, labels, md_columns, args.sr, args.ctf_type, vol.shape[0])
                total_loss += loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f}")

                # Summary writer (training loss)
                if step % np.ceil(int(0.1 * len(data_loader))) == 0:
                    writer.add_scalar('Training loss (volume adjustment)',
                                      total_loss / step,
                                      i * len(data_loader) + step)

                step += 1
        volumeAdjustment, optimizer = nnx.merge(graphdef, state)

        # Save model
        NeuralNetworkCheckpointer.save(volumeAdjustment, os.path.join(args.output_path, "volumeAdjustment"), mode="pickle")

    elif args.mode == "predict":

        volumeAdjustment.eval()

        # Predict new values
        values = np.array(volumeAdjustment())

        # Place new values in volume
        adjusted_vol = np.zeros_like(vol)
        adjusted_vol[inds[:, 0], inds[:, 1], inds[:, 2]] = values

        # Save new volume
        ImageHandler().write(adjusted_vol, os.path.join(args.output_path, "adjusted_volume.mrc"), sr=args.sr)

if __name__ == "__main__":
    main()
