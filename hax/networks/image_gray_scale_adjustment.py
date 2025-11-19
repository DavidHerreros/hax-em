#!/usr/bin/env python


from functools import partial

import jax
from jax import numpy as jnp
from flax import nnx
import dm_pix

from einops import rearrange

from hax.utils import *



class ImageAdjustment(nnx.Module):
    def __init__(self, xsize, lat_dim=10, predict_value=False, *, rngs: nnx.Rngs):
        self.xsize = xsize
        self.predict_value = predict_value

        # Gray level adjustment (TODO: with 256 features OK)
        self.hidden_layers_ds = [nnx.Linear(xsize * xsize, 32, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(2):
            self.hidden_layers_ds.append(nnx.Linear(32, 32, rngs=rngs, dtype=jnp.bfloat16))
        self.hidden_layers_us = [nnx.Linear(lat_dim, 32, rngs=rngs, dtype=jnp.bfloat16)]
        for _ in range(2):
            self.hidden_layers_us.append(nnx.Linear(32, 32, rngs=rngs, dtype=jnp.bfloat16))
        self.latent = nnx.Linear(32, lat_dim, rngs=rngs)

        if predict_value:
            self.a = nnx.Linear(32, 1, rngs=rngs, kernel_init=jax.nn.initializers.normal(stddev=0.001), bias_init=jax.nn.initializers.ones)
            self.b = nnx.Linear(32, 1, rngs=rngs, kernel_init=jax.nn.initializers.normal(stddev=0.001))
        else:
            self.a = nnx.Linear(32, xsize * xsize, rngs=rngs, kernel_init=jax.nn.initializers.normal(stddev=0.001), bias_init=jax.nn.initializers.ones)
            self.b = nnx.Linear(32, xsize * xsize, rngs=rngs, kernel_init=jax.nn.initializers.normal(stddev=0.001))

    def __call__(self, x):
        if x.ndim == 4:
            x = x[..., 0]

        x = rearrange(x, 'b h w -> b (h w)')

        partial_gray = nnx.relu(self.hidden_layers_ds[0](x))
        for layer in self.hidden_layers_ds[1:]:
            partial_gray = nnx.relu(partial_gray + layer(partial_gray))

        partial_gray = self.latent(partial_gray)

        partial_gray = nnx.relu(self.hidden_layers_us[0](partial_gray))
        for layer in self.hidden_layers_us[1:]:
            partial_gray = nnx.relu(partial_gray + layer(partial_gray))

        a = nnx.relu(self.a(partial_gray))
        b = self.b(partial_gray)

        if not self.predict_value:
            a = rearrange(a, "b (w h) -> b w h", w=self.xsize, h=self.xsize)
            b = rearrange(b, "b (w h) -> b w h", w=self.xsize, h=self.xsize)
        else:
            a, b = a[..., 0], b[..., 0]

        return a, b


@partial(jax.jit, static_argnames=["sr", "ctf_type"])
def train_step_image_adjustment(graphdef, state, x, labels, md, sr, ctf_type, coords, values):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, x, coords, values):
        factor = 0.5 * model.xsize

        # Gray level adjustment
        a, b = model(x)

        # Adjust shapes of a and b
        if model.predict_value:
            a = a[:, None, None]
            b = b[:, None, None]

        # Rotate grid
        rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        coords = jnp.matmul(factor * coords, rearrange(rotations, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts[:, None, :] + factor

        # Scatter image
        B = x.shape[0]
        c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
        images = jnp.zeros((B, model.xsize, model.xsize), dtype=x.dtype)

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
        images = dm_pix.gaussian_blur(images[..., None], 1.0, kernel_size=3)[..., 0]

        # Apply gray level correction
        images = a * images + b

        # Prepare data for losses
        x = x[..., 0]

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
                         sr, [2 * model.xsize, int(2 * 0.5 * model.xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], 2 * model.xsize, int(2.0 * 0.5 * model.xsize + 1)], dtype=x.dtype)

    if ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(x[..., 0], ctf)[..., None]

    grad_fn = nnx.value_and_grad(loss_fn)

    loss, grads = grad_fn(model, x, coords, values)

    optimizer.update(grads)

    state = nnx.state((model, optimizer))

    return loss, state


@partial(jax.jit, static_argnames=["sr", "ctf_type"])
def validation_step_image_adjustment(graphdef, state, x, labels, md, sr, ctf_type, coords, values):
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model, x, coords, values):
        factor = 0.5 * model.xsize

        # Gray level adjustment
        a, b = model(x)

        # Adjust shapes of a and b
        if model.predict_value:
            a = a[:, None, None]
            b = b[:, None, None]

        # Rotate grid
        rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        coords = jnp.matmul(factor * coords, rearrange(rotations, "b r c -> b c r"))

        # Apply shifts
        coords = coords[..., :-1] - shifts[:, None, :] + factor

        # Scatter image
        B = x.shape[0]
        c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
        images = jnp.zeros((B, model.xsize, model.xsize), dtype=x.dtype)

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
        images = dm_pix.gaussian_blur(images[..., None], 1.0, kernel_size=3)[..., 0]

        # Apply gray level correction
        images = a * images + b

        # Prepare data for losses
        x = x[..., 0]

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
                         sr, [2 * model.xsize, int(2 * 0.5 * model.xsize + 1)],
                         x.shape[0], True)
    else:
        ctf = jnp.ones([x.shape[0], 2 * model.xsize, int(2.0 * 0.5 * model.xsize + 1)], dtype=x.dtype)

    if ctf_type == "precorrect":
        # Wiener filter
        x = wiener2DFilter(x[..., 0], ctf)[..., None]

    loss = loss_fn(model, x, coords, values)

    return loss



def main():
    import os
    import sys
    from tqdm import tqdm
    import random
    import numpy as np
    import argparse
    import shutil
    from xmipp_metadata.image_handler import ImageHandler
    import optax
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.generators import MetaDataGenerator, extract_columns
    from hax.networks import train_step_image_adjustment
    from hax.metrics import JaxSummaryWriter
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()
    parser.add_argument("--md", required=True, type=str,
                        help="Xmipp/Relion metadata file with the images (+ alignments / CTF) to be analyzed")
    parser.add_argument("--vol", required=True, type=str,
                        help="Volume needed to generate the projections to be adjusted")
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
    parser.add_argument("--predict_value", action='store_true',
                        help="If not provided, the adjustment will be estimated per pixel - otherwise, adjustment will be estimated per projection")
    parser.add_argument("--lat_dim", required=False, type=int, default=3,
                        help="Dimensionality of the latent space of the network (set by default to 3)")
    parser.add_argument("--mode", required=True, type=str, choices=["train", "predict"],
                        help=f"{bcolors.BOLD}train{bcolors.ENDC}: train a neural network from scratch or from a previous execution if reload is provided\n"
                             f"{bcolors.BOLD}predict{bcolors.ENDC}: predict the adjustment for the input images/volume ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case)")
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
    parser.add_argument("-dataset_split_fraction", required=False, type=list_of_floats, default=[0.8, 0.2],
                        help=f"Here you can provide the fractions to split your data automatically into a training and a validation subset following the format: {bcolors.ITALIC}training_fraction{bcolors.ENDC},"
                             f"{bcolors.ITALIC}validation_fraction{bcolors.ENDC}. While the training subset will be used to train/update the network parameters, the validation subset will only be used to evaluate the "
                             f"accuracy of the network when faced with new data. Therefore, the validation subset will never be used to update the networks parameters. {bcolors.WARNING}NOTE{bcolors.ENDC}: the sum of "
                             f"{bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} must be equal to one.")
    parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, new metadata...)")
    parser.add_argument("--reload", required=False, type=str,
                        help="Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data)")
    args = parser.parse_args()

    # Check that training and validation fractions add up to one
    if sum(args.dataset_split_fraction) != 1:
        raise ValueError(
            f"The sum of {bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} is not equal one. Please, update the values "
            f"to fulfill this requirement.")

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

    # Prepare metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Prepare network
    rng = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
    rng, model_key = jax.random.split(rng, 2)
    imageAdjustment = ImageAdjustment(lat_dim=args.lat_dim, xsize=vol.shape[0], predict_value=args.predict_value, rngs=nnx.Rngs(model_key))

    # Reload network
    if args.reload is not None:
        imageAdjustment = NeuralNetworkCheckpointer.load(imageAdjustment, os.path.join(args.reload, "imageAdjustment"))

    # Train network
    if args.mode == "train":

        imageAdjustment.train()

        # Prepare summary writer
        writer = JaxSummaryWriter(os.path.join(args.output_path, "Image_adjustment_metrics"))

        # Prepare data loader
        _, data_loader, data_loader_validation = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=True, preShuffle=True,
                                                                             mmap=mmap, mmap_output_dir=mmap_output_dir, split_fraction=args.dataset_split_fraction)

        # Example of training data for Tensorboard
        x_example, labels_example = next(iter(data_loader))
        x_example = jax.vmap(min_max_scale)(x_example)
        writer.add_images("Training data batch", x_example, dataformats="NHWC")

        # Optimizers
        optimizer = nnx.Optimizer(imageAdjustment, optax.adam(args.learning_rate))
        graphdef, state = nnx.split((imageAdjustment, optimizer))

        # Resume if checkpoint exists
        if os.path.isdir(os.path.join(args.output_path, "imageAdjustment_CHECKPOINT")):
            graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "imageAdjustment_CHECKPOINT"))
            print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
        else:
            resume_epoch = 0

        # Training loop
        print(f"{bcolors.OKCYAN}\n###### Training image adjustment... ######")
        for i in range(resume_epoch, args.epochs):
            total_loss = 0
            total_validation_loss = 0

            # For progress bar (TQDM)
            step = 1
            step_validation = 1
            print(f'\nTraining epoch {i + 1}/{args.epochs} |')
            pbar = tqdm(data_loader, desc=f"Epoch {i + 1}/{args.epochs}", file=sys.stdout, ascii=" >=",
                        colour="green")

            for (x, labels) in pbar:
                loss, state = train_step_image_adjustment(graphdef, state, x, labels, md_columns, args.sr, args.ctf_type, coords, values)
                total_loss += loss

                # Progress bar update  (TQDM)
                pbar.set_postfix_str(f"loss={total_loss / step:.5f}")

                # Summary writer (training loss)
                if step % int(np.ceil(0.1 * len(data_loader))) == 0:
                    writer.add_scalars('Training loss (image adjustment)',
                                       {"train": total_loss / step},
                                       i * len(data_loader) + step)

                # Summary writer (validation loss)
                if step % int(np.ceil(0.5 * len(data_loader))) == 0:
                    # Run validation step
                    print(f"\n{bcolors.WARNING}Running validation step...{bcolors.ENDC}\n")
                    for (x_validation, labels_validation) in data_loader_validation:
                        loss_validation = validation_step_image_adjustment(graphdef, state, x_validation, labels_validation,
                                                                           md_columns, args.sr, args.ctf_type, coords, values)
                        total_validation_loss += loss_validation

                        step_validation += 1

                    writer.add_scalars('Training loss (image adjustment)',
                                       {"validation": total_validation_loss / step_validation},
                                       i * len(data_loader) + step)

                step += 1

            if i % 5:
                # Save checkpoint model
                NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path, "imageAdjustment_CHECKPOINT"), epoch=i)

        imageAdjustment, optimizer = nnx.merge(graphdef, state)

        # Save model
        NeuralNetworkCheckpointer.save(imageAdjustment, os.path.join(args.output_path, "imageAdjustment"), mode="pickle")

        # Remove checkpoint
        shutil.rmtree(os.path.join(args.output_path, "imageAdjustment_CHECKPOINT"))

    elif args.mode == "predict":

        imageAdjustment.eval()

        # Prepare data loader
        data_loader = generator.return_tf_dataset(batch_size=args.batch_size, shuffle=False, preShuffle=False,
                                                  mmap=mmap, mmap_output_dir=mmap_output_dir)

        # Jitted prediction function
        predict_fn = jax.jit(imageAdjustment.__call__)

        # Predict loop
        print(f"{bcolors.OKCYAN}\n###### Predicting image adjustment... ######")
        pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=",
                    colour="green")

        imgs_adjusted = []
        adjustment_a = []
        adjustment_b = []
        for (x, labels) in pbar:
            a, b = predict_fn(x)

            # Adjust shapes of a and b
            if imageAdjustment.predict_value:
                adjustment_a.append(1. / a)
                adjustment_b.append(-b / a)
                a = a[:, None, None]
                b = b[:, None, None]

            adjustment = np.nan_to_num(np.asarray((x[..., 0] - b) / a), nan=0.0, posinf=0.0, neginf=0.0)
            imgs_adjusted.append(adjustment)
        imgs_adjusted = np.concatenate(imgs_adjusted, axis=0)

        # Save new images
        output_images_path = os.path.join(args.output_path, "adjusted_images.mrcs")
        ImageHandler().write(imgs_adjusted, output_images_path, sr=args.sr)
        md = generator.md
        if imageAdjustment.predict_value:
            md[:, "adjustment_a"] = np.concatenate(adjustment_a, axis=0)
            md[:, "adjustment_b"] = np.concatenate(adjustment_b, axis=0)
        for idx in range(len(md)):
            image_id, _ = md[idx, "image"].split('@')
            md[idx, "image"] = "@".join([image_id, output_images_path])
        md.write(os.path.join(args.output_path, "adjusted_images" +  os.path.splitext(args.md)[1]))

if __name__ == "__main__":
    main()
