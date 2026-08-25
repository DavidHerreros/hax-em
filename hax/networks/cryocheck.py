#!/usr/bin/env python


import jax
import jax.numpy as jnp
from flax import nnx
import dm_pix 

import optax

from einops import rearrange

import numpy as np

from hax import * 

from hax.utils.ctf import computeCTF
from hax.utils.fourier_filters import ctfFilter
from hax.utils.euler import euler_matrix_batch
from hax.utils.decorators import save_config

from hax.programs.gaussian_volume_fitting import fit_volume, adjust_weights_to_images



# MLP
class CryoCheck(nnx.Module):
  @save_config
  def __init__(self, n_shells, rngs: nnx.Rngs, hidden=(64, 32)): #a third hidden layer may be added
    self.fc1 = nnx.Linear(n_shells, hidden[0], rngs=rngs)
    self.bn1 = nnx.BatchNorm(hidden[0], rngs=rngs)
    self.fc2 = nnx.Linear(hidden[0], hidden[1], rngs=rngs)
    self.bn2 = nnx.BatchNorm(hidden[1], rngs=rngs)
    self.fc3 = nnx.Linear(hidden[1], 1, rngs=rngs)

  @nnx.jit(static_argnames='eval')
  def __call__(self, x, eval=False, train=False):
    x = nnx.relu(self.bn1(self.fc1(x)))
    x = nnx.relu(self.bn2(self.fc2(x)))
    x = self.fc3(x)
    if eval:
      x = nnx.sigmoid(x)
    return x
  

# Training and Validation
@nnx.jit(static_argnames='train')
def cryoCheck_step(model, optimizer, x, labels,*, train: bool):

    def loss_fn(model, x, labels):
        logits = model(x, eval=False, train=train)  # Get raw logits for loss computation
        # Binary cross entropy
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels)) #avg loss per batch

        return loss

    if train:
      grad_fn = nnx.value_and_grad(loss_fn)

      loss, grads = grad_fn(model, x, labels)

      optimizer.update(model,grads)

    else:
      loss = loss_fn(model, x, labels)

    return loss, model 



# Utils

# Extracting metadata for a batch : euler angles, shifts, ctf
def md_extraction(md_columns, index, vol, args):

    # Precompute batch alignments 
    euler_angles = md_columns["euler_angles"][index] 

    # Precompute batch shifts
    shifts = md_columns["shifts"][index]

    # Precompute batch CTFs
    defocusU = md_columns["ctfDefocusU"][index] 
    defocusV = md_columns["ctfDefocusV"][index]
    defocusAngle = md_columns["ctfDefocusAngle"][index]
    cs = md_columns["ctfSphericalAberration"][index]
    kv = md_columns["ctfVoltage"][0]

    xsize = vol.shape[1]
    batch_size = len(index)

    pad_factor=2
    ctf = computeCTF(defocusU, defocusV, defocusAngle, cs, kv,
                         args.sr, [pad_factor * xsize, int(pad_factor * 0.5 * xsize + 1)],
                         batch_size, True) 
        
    return(euler_angles, shifts, ctf)


# Projecting the batch volume 
def volumeProjection(vol, mask, euler_angles, shifts, ctf):
    """
    Project a 3D volume into a batch of CTF-corrected 2D images, via
    forward voxel scattering (not ray-casting): each masked voxel is
    rotated/shifted according to the given pose and splatted bilinearly
    into the output image, then blurred (to smooth the splatting) and
    CTF-filtered.

    Args:
        vol: (N, N, N) volume, voxel intensities.
        mask: (N, N, N) particle mask, same shape as `vol`.
        euler_angles: (B, 3) rotation angles in degrees.
        shifts: (B, 2) in-plane shifts in pixels.
        ctf: CTF parameters for the batch (from `computeCTF`).

    Returns:
        (B, N, N, 1) projected images, N = vol.shape[0].

    Note: `mask` must match `vol`'s shape exactly, or it will silently
    read the wrong voxels. Rotation center is vol's geometric center.
    """

    inds = np.asarray(np.where(mask > 0.0)).T #z,y,x voxel

    # Voxels intensity
    values = vol[inds[:, 0], inds[:, 1], inds[:, 2]]

    factor = 0.5 * vol.shape[0] 
    coords = jnp.stack([inds[:, 2], inds[:, 1], inds[:, 0]], axis=1) #x,y,z

    coords = (coords - factor)

    # Rotate grid
    rotations = euler_matrix_batch(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]) 
    coords = jnp.matmul(coords[None, ...], rearrange(rotations, "b r c -> b c r")) 
  
    # Apply shifts
    coords = coords[..., :-1] - shifts[:, None, :] + factor
  
    # Scatter image
    B = euler_angles.shape[0]  
    xsize=vol.shape[0] #or vol.shape[1]
    c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
    images = jnp.zeros((B, xsize, xsize), dtype=vol.dtype) 

    # Forward mapping
    bamp = values[None, ...]

    bposf = jnp.floor(c_sampling)
    bposi = bposf.astype(jnp.int32)
    bposf = c_sampling - bposf

    # Split voxels intensity in 4 weights assigned to the four nearest pixels of the targeted one
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
    images = jnp.squeeze(dm_pix.gaussian_blur(images[..., None], 1.0, kernel_size=3), axis=3)

    # Consider CTF
    images = ctfFilter(images, ctf, pad_factor=2)

    return images[..., None] 


# Compute Misalignment 
def compute_min_rotation_angle(mask, pixel_threshold=2.0):
    """
    Minimum rotation angle (degrees) such that the mask voxel farthest from
    its center of mass travels at least `pixel_threshold` pixels of arc.

    arc = R * alpha  ->  alpha_min = pixel_threshold / R_max

    Call once, right after computing `mask`.
    """
    inds = np.asarray(np.where(mask > 0.0)).T          # (N, 3) z,y,x voxel coords
    com = inds.mean(axis=0)                              # geometric centroid
    R_max = np.linalg.norm(inds - com, axis=1).max()

    alpha_min_deg = np.degrees(pixel_threshold / R_max)
    return float(alpha_min_deg), float(R_max)


def generate_misalignment(rngs, euler_angles, shifts, box_size, alpha_min_deg,
                           alpha_max_deg=180.0, shift_min_px=2.0, shift_max_frac=0.10):
    """
    Perturb ground-truth angles and shifts for a "misaligned" example.
    Angle magnitude ~ Uniform[alpha_min_deg, alpha_max_deg], random sign per
    angle (rotation can go either direction).
    Shift magnitude/direction sampled in polar form so every direction is
    equally likely.

    Call identically in training and validation, passing the same
    `alpha_min_deg` (from compute_min_rotation_angle) and `box_size`.

    Returns: rngs (updated key), euler_angles_noisy, shifts_noisy
    """
    B = euler_angles.shape[0]

    rngs, k1, k2, k3, k4 = jax.random.split(rngs, 5)

    angle_mag = jax.random.uniform(k1, euler_angles.shape, minval=alpha_min_deg, maxval=alpha_max_deg)
    angle_sign = jax.random.choice(k2, jnp.array([-1.0, 1.0]), shape=euler_angles.shape)
    euler_angles_noisy = euler_angles + angle_mag * angle_sign

    shift_max_px = shift_max_frac * box_size
    shift_mag = jax.random.uniform(k3, (B,), minval=shift_min_px, maxval=shift_max_px)
    shift_angle = jax.random.uniform(k4, (B,), minval=0.0, maxval=2 * jnp.pi)
    shift_noise = jnp.stack([shift_mag * jnp.cos(shift_angle), shift_mag * jnp.sin(shift_angle)], axis=1)
    shifts_noisy = shifts + shift_noise

    return rngs, euler_angles_noisy, shifts_noisy




def main():

  import os
  import sys
  from tqdm import tqdm
  import random
  import numpy as np
  import matplotlib.pyplot as plt
  import argparse
  import shutil
  from xmipp_metadata.image_handler import ImageHandler
  import optax
  from contextlib import closing
  from hax.utils.loggers import bcolors
  from hax.checkpointer import NeuralNetworkCheckpointer
  from hax.generators import MetaDataGenerator, extract_columns
  from hax.metrics import JaxSummaryWriter
  from hax.utils.frc_jit_clean import compute_fourier_residual
  from hax.utils.standardize_frc import standardize_frc_curve

  def list_of_floats(arg):
        return list(map(float, arg.split(',')))

  parser = argparse.ArgumentParser()
  parser.add_argument("--md", required=True, type=str,
                        help="Xmipp/Relion metadata with the images to be analyzed, provided as a .xmd file")
  parser.add_argument("--vol", required=True, type=str,
                        help="Volume needed to generate the projections provided as a .mrc file")
  parser.add_argument("--load_images_to_ram", action='store_true',
                        help=f"If provided, images will be loaded to RAM. This is recommended if you want the best performance and your dataset fits in your RAM memory. If this flag is not provided, "
                             f"images will be memory mapped. When this happens, the program will trade disk space for performance. Thus, during the execution additional disk space will be used and the performance "
                             f"will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal once the execution has finished.")
  parser.add_argument("--sr", required=True, type=float,
                        help="Sampling rate of the images/volume")
  parser.add_argument("--mode", required=True, type=str, choices=["train", "predict"],
                        help=f"{bcolors.BOLD}train{bcolors.ENDC}: train a neural network from scratch or from a previous execution if reload is provided\n"
                             f"{bcolors.BOLD}predict{bcolors.ENDC}: predict the adjustment for the input volume ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case)")
  parser.add_argument("--epochs", required=False, type=int, default=10,
                        help="Number of epochs to train the network (i.e. how many times to loop over the whole dataset of images - set to default to 10 - "
                             "training is carried out over a configurable numer of epochs that is tipically set to the 10 for single-input images and 20 for combined images")
  parser.add_argument("--batch_size", required=False, type=int, default=32,
                        help="Determines how many images will be load in the GPU at any moment during training (set by default to 32 - "
                             f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                             f"to monitor and/or measure memory usage and adjust this value")
  parser.add_argument("--learning_rate", required=False, type=float, default=1e-4,
                        help=f"The learning rate ({bcolors.ITALIC}lr{bcolors.ENDC}) sets the speed of learning. Think of the model as trying to find the lowest point in a valley; the {bcolors.ITALIC}lr{bcolors.ENDC} "
                             f"is the size of the step it takes on each attempt. A large {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}0.01{bcolors.ENDC}) is like taking huge leaps — it's fast but can be unstable, "
                             f"overshoot the lowest point, or cause {bcolors.ITALIC}NAN{bcolors.ENDC} errors. A small {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}1e-6{bcolors.ENDC}) is like taking tiny "
                             f"shuffles — it's stable but very slow and might get stuck before reaching the bottom. A good default is often {bcolors.ITALIC}0.0001{bcolors.ENDC}. If training fails or errors explode, "
                             f"try making the {bcolors.ITALIC}lr{bcolors.ENDC} 10 times smaller (e.g., {bcolors.ITALIC}0.001{bcolors.ENDC} --> {bcolors.ITALIC}0.0001{bcolors.ENDC}).")
  parser.add_argument("--dataset_split_fraction", required=False, type=list_of_floats, default=[0.8, 0.2],
                        help=f"Here you can provide the fractions to split your data automatically into a training and a validation subset following the format: {bcolors.ITALIC}training_fraction{bcolors.ENDC},"
                             f"{bcolors.ITALIC}validation_fraction{bcolors.ENDC}. While the training subset will be used to train/update the network parameters, the validation subset will only be used to evaluate the "
                             f"accuracy of the network when faced with new data. Therefore, the validation subset will never be used to update the networks parameters. {bcolors.WARNING}NOTE{bcolors.ENDC}: the sum of "
                             f"{bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} must be equal to one.")
  parser.add_argument("--output_path", required=True, type=str,
                        help="Path to save the results (trained neural network, adjusted volume...)")
  parser.add_argument("--reload", required=False, type=str,
                        help="Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data)")
  parser.add_argument("--ssd_scratch_folder", required=False, type=str,
                        help=f"When the parameter {bcolors.UNDERLINE}load_images_to_ram{bcolors.ENDC} is not provided, we strongly recommend to provide here a path to a folder in a SSD disk to read faster the data. If not given, the data will be loaded from "
                             f"the default disk.")
  parser.add_argument("--num_gaussians", required=False, type=int, default=5000,
                        help="Number of Gaussians to fit the input volume to recreate it. This is a crucial step to adjust the grey levels of the input volume to the ones of the images.")
  
  args, _ = parser.parse_known_args()


  # Check that training and validation fractions add up to one
  if sum(args.dataset_split_fraction) != 1:
        raise ValueError(
            f"The sum of {bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} is not equal one. Please, update the values "
            f"to fulfill this requirement.")
        
  # Volume and Mask handling
  vol = ImageHandler(args.vol).getData()
  mask = ImageHandler().generateMask(inputFn=vol, boxsize=vol.shape[0])
  alpha_min_deg, R_max = compute_min_rotation_angle(mask, pixel_threshold=2.0) # it will be further used to compute misalignment; it can be computed once

  # Prepare network
  x_size = vol.shape[0]
  n_shells = x_size // 2 
  rngs = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
  cryoCheck = CryoCheck(n_shells=n_shells, rngs=nnx.Rngs(rngs))

  # Reload network
  if args.reload is not None:
      cryoCheck = NeuralNetworkCheckpointer.load(args.reload)

  # Load metadata
  generator = MetaDataGenerator(args.md)
  md_columns = extract_columns(generator.md)

  if not args.load_images_to_ram and args.mode in ["train", "predict"]:
    mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
    # Prepare grain dataset
    generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                                precision=np.float16, group_size=1, shard_size=10000)  #shard: significa che ho più archivi con 10000 immagini ciascuno e non tutti le immagini in uno solo
  else:
    mmap_output_dir = None
  
  ### Train network ###
  if args.mode == "train":
    
    cryoCheck.train()
    # Prepare summary writer
    writer = JaxSummaryWriter(os.path.join(args.output_path, "cryoCheck_metrics"))

 
    # Gaussian Splatting to adjust grey levels of the input volume
    if args.vol is not None:
      fit_path = os.path.join(args.output_path, "Gaussian_volume_fitting")
      if not os.path.isdir(os.path.join(fit_path)):
      
        model, _, _ = fit_volume(vol, mask=mask, iterations=20000, learning_rate=0.001, n_init=args.num_gaussians, fixed_gaussians=True)
        
        # Adjust to images 
        model, _ = adjust_weights_to_images(model, args.md, mmap_output_dir, args.sr, learning_rate=0.01,
                                            num_epochs=5, is_global=True, ctf_type="apply")

        # Save model
        NeuralNetworkCheckpointer.save(model, fit_path)

        # Save adjusted volume and deltas for visualizationl
        vol_deltas = np.array(model(place_deltas=True))
        vol = np.array(model())
        ImageHandler().write(vol_deltas, os.path.join(args.output_path, "consensus_volume_deltas.mrc"), overwrite=True)
        ImageHandler().write(vol, os.path.join(args.output_path, "consensus_volume.mrc"), overwrite=True)
    
      else:
        model = NeuralNetworkCheckpointer.load(checkpoint_path=fit_path)
        vol = np.array(model())


    # Prepare data loader 
    data_loader_train, data_loader_val = generator.return_grain_dataset(batch_size=args.batch_size, shuffle="global",
                                                                            split_fraction=args.dataset_split_fraction,
                                                                            num_epochs=None,
                                                                            num_workers=-1, num_threads=1,            
                                                                            load_to_ram=args.load_images_to_ram)    
    
    steps_per_epoch = int(int(args.dataset_split_fraction[0] * len(generator.md)) / args.batch_size) 
    steps_per_val = int(int(args.dataset_split_fraction[1] * len(generator.md)) / args.batch_size)

    # Optimizer
    optimizer = nnx.Optimizer(cryoCheck, optax.adamw(args.learning_rate), wrt=nnx.Param)  
    
    # Resume if checkpoint exists
    if os.path.isdir(os.path.join(args.output_path, "cryoCheck_CHECKPOINT")):
      graphdef, state, resume_epoch = NeuralNetworkCheckpointer.load_intermediate(os.path.join(args.output_path, "cryoCheck_CHECKPOINT"), optimizer)
      training_bundle = nnx.merge(graphdef, state)
      cryoCheck = training_bundle[0]
      print(f"{bcolors.WARNING}\nCheckpoint detected: resuming training from epoch {resume_epoch}{bcolors.ENDC}")
    else:
      resume_epoch = 0


    #TRAINING LOOP
    print(f"{bcolors.OKCYAN}\n###### Training CryoCheck... ######") 

    i = 0 
    pbar = tqdm(range(resume_epoch * steps_per_epoch, args.epochs * steps_per_epoch), file=sys.stdout, ascii=" >=",
                    colour="green",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
  
    total_loss = 0

    with closing(iter(data_loader_train)) as iter_data_loader_train, closing(iter(data_loader_val)) as iter_data_loader_val:

      t_score = []
      t_labels = []

      for total_steps in pbar:
        (x, index) = next(iter_data_loader_train) 

        euler_angles, shifts, ctf = md_extraction (md_columns, index, vol, args)

        batch_size = len(index)
        
        # Aligned images
        projection_al = volumeProjection(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles,
                                 shifts=shifts,
                                 ctf=ctf)
        result_al = compute_fourier_residual(projection_al, x, n_shells=n_shells)
        aligned_frc = standardize_frc_curve(result_al.frc_curve, result_al.freqs, ts=args.sr, n_shells_ref=n_shells)[0]
        aligned_frc = jnp.array(aligned_frc)
        #aligned_res = result.residual_map
        aligned_labels = jnp.ones((batch_size,1)) #label for aligned is 1

        # Misaligned images - Data Augmentation
        rngs, euler_angles_noisy, shifts_noisy = generate_misalignment(rngs, euler_angles, shifts, box_size=x_size, alpha_min_deg=alpha_min_deg)
        projection_misal = volumeProjection(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles_noisy,
                                 shifts=shifts_noisy,
                                 ctf=ctf)
        result_mis = compute_fourier_residual(projection_misal, x, n_shells=n_shells)
        misaligned_frc = standardize_frc_curve(result_mis.frc_curve, result_mis.freqs, ts=args.sr, n_shells_ref=n_shells)[0]
        misaligned_frc = jnp.array(misaligned_frc)
        #misaligned_res = result.residual_map
        misaligned_labels = jnp.zeros((batch_size,1)) #label for misaligned is 0
      
       
        frc_curves = jnp.concatenate([aligned_frc, misaligned_frc], axis=0)
        labels = jnp.concatenate([aligned_labels, misaligned_labels], axis=0)
        
        loss, cryoCheck = cryoCheck_step(cryoCheck, optimizer, x=frc_curves, labels=labels, train=True)
        total_loss += loss
        
        ######## roc and confusion matrix #######
        t_score.append(cryoCheck(frc_curves, eval=True)) 
        t_labels.append(labels)


        #VALIDATION STEP at the end of each epoch  
        if (total_steps + 1) % steps_per_epoch == 0:    

          # Compute training scores and metrics 
          #################
          t_score_epoch= jnp.concatenate(t_score, axis=0)
          t_labels_epoch = jnp.concatenate(t_labels, axis=0)

          # Roc Curve and Confusion Matrix - Training
          optimal_threshold_t = writer.add_roc_curve(t_labels_epoch, t_score_epoch, global_step=i, tag="ROC Curve - Training step")
          t_score_heavy = t_score_epoch > optimal_threshold_t
          writer.add_confusion_matrix(t_labels_epoch, t_score_heavy, global_step=i, tag="Confusion Matrix - Training step")
            

          # average training loss at the end of each epoch 
          avg_train_loss = total_loss / steps_per_epoch
          pbar.write(f"\n--- End of Training for Epoch {int((total_steps + 1) / steps_per_epoch)} ---")
          pbar.write(f" Loss: {avg_train_loss:.4f}")

          writer.add_scalars('Training loss (cryocheck)',
                                       {"train": avg_train_loss},
                                       total_steps + 1)
          ###################

          total_loss = 0
          total_validation_loss = 0
          
          # Validation step 
          print(f"{bcolors.WARNING}\n###### Running Validation Step... ######{bcolors.ENDC}")

          cryoCheck.eval()

          val_score = []
          val_labels = []

          for _ in range(steps_per_val):
            
            (x_validation, index_validation) = next(iter_data_loader_val)
           
            euler_angles, shifts, ctf = md_extraction (md_columns, index_validation, vol, args)

            batch_size_v = len(index_validation)


            # Aligned images
            projection_al_v = volumeProjection(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles,
                                 shifts=shifts,
                                 ctf=ctf)
            result_al_v = compute_fourier_residual(projection_al_v, x_validation, n_shells=n_shells)
            aligned_frc_v, freqs_ref = standardize_frc_curve(result_al_v.frc_curve, result_al_v.freqs, ts=args.sr, n_shells_ref=n_shells)
            aligned_frc_v = jnp.array(aligned_frc_v)
            aligned_res_v = result_al_v.residual_map
            aligned_labels_v = jnp.ones((batch_size_v,1))
        
    
            # Misaligned images
            rngs, euler_angles_noisy, shifts_noisy = generate_misalignment(rngs, euler_angles, shifts, box_size=x_size, alpha_min_deg=alpha_min_deg)
            projection_misal_v = volumeProjection(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles_noisy,
                                 shifts=shifts_noisy,
                                 ctf=ctf)
            result_mis_v = compute_fourier_residual(projection_misal_v, x_validation, n_shells=n_shells)
            misaligned_frc_v = standardize_frc_curve(result_mis_v.frc_curve, result_mis_v.freqs, ts=args.sr, n_shells_ref=n_shells)[0]
            misaligned_frc_v = jnp.array(misaligned_frc_v)
            misaligned_res_v = result_mis_v.residual_map
            misaligned_labels_v = jnp.zeros((batch_size_v,1))
          
            frc_curves_validation = jnp.concatenate([aligned_frc_v, misaligned_frc_v],axis=0)
            labels_validation = jnp.concatenate([aligned_labels_v, misaligned_labels_v], axis=0)


            ###########################################################
            # Debugging: save pure projection aligned and misaligned
            if _ == 0:
              particle_id = int(index_validation[0])

              proj_al_2d = jnp.squeeze(projection_al_v[0])
              proj_misal_2d = jnp.squeeze(projection_misal_v[0])

              writer.add_image("Pure_Projection/Aligned", proj_al_2d, global_step=i, dataformats='HW')
              writer.add_image("Pure_Projection/Misaligned", proj_misal_2d, global_step=i, dataformats='HW')

              ImageHandler().write(np.array(proj_al_2d),
                                    os.path.join(args.output_path, "pure_projection_aligned.mrcs"), overwrite=True)
              ImageHandler().write(np.array(proj_misal_2d),
                                    os.path.join(args.output_path, "pure_projection_misaligned.mrcs"), overwrite=True)

              # Metadata aligned and misaligned
              pose_al = {
                  "particle_id": particle_id,
                  "euler_angles_deg": np.array(euler_angles[0]).tolist(),
                  "shifts_px": np.array(shifts[0]).tolist(),
              }
              pose_misal = {
                  "particle_id": particle_id,
                  "euler_angles_deg": np.array(euler_angles_noisy[0]).tolist(),
                  "shifts_px": np.array(shifts_noisy[0]).tolist(),
                  "euler_angles_delta_deg": np.array(euler_angles_noisy[0] - euler_angles[0]).tolist(),
                  "shifts_delta_px": np.array(shifts_noisy[0] - shifts[0]).tolist(),
              }

              import json
              with open(os.path.join(args.output_path, "pose_aligned.json"), "w") as f:
                  json.dump(pose_al, f, indent=2)
              with open(os.path.join(args.output_path, "pose_misaligned.json"), "w") as f:
                  json.dump(pose_misal, f, indent=2)

              # FRC curve
              fig, ax = plt.subplots()
              ax.plot(freqs_ref, np.array(aligned_frc_v[0]), label="Aligned")
              ax.plot(freqs_ref, np.array(misaligned_frc_v[0]), label="Misaligned")
              ax.set_xlabel("Spatial frequency (cycles/Angstrom)")
              ax.set_ylabel("FRC")
              ax.set_title(f"FRC curve - Aligned vs Misaligned (particle id {particle_id})")
              ax.legend()
              ax.set_ylim(-0.2, 1.05)
              writer.add_figure("FRC_Curve/Aligned_vs_Misaligned", fig, global_step=i)
              plt.close(fig)

              np.save(os.path.join(args.output_path, "aligned_frc_curve.npy"), np.array(aligned_frc_v[0]))
              np.save(os.path.join(args.output_path, "misaligned_frc_curve.npy"), np.array(misaligned_frc_v[0]))

            ######################################################################


            loss_validation, cryoCheck = cryoCheck_step(cryoCheck, optimizer, x=frc_curves_validation, labels=labels_validation, train=False)
            total_validation_loss += loss_validation
            
            val_score.append(cryoCheck(frc_curves_validation, eval=True)) #predictions for the validation step
            val_labels.append(labels_validation)
            

          val_score_epoch = jnp.concatenate(val_score, axis=0)
          val_labels_epoch = jnp.concatenate(val_labels, axis=0)

          cryoCheck.train()

          # Roc Curve and Confusion Matrix - Validation
          optimal_threshold = writer.add_roc_curve(val_labels_epoch, val_score_epoch, global_step=i, tag="ROC Curve - Validation step")
          
          # Save optimal threshold value 
          threshold_path = os.path.join(args.output_path, "optimal_threshold_value.txt")
          with open(threshold_path, "w") as f:
              f.write(str(optimal_threshold))

          val_score_heavy = val_score_epoch > optimal_threshold
          writer.add_confusion_matrix(val_labels_epoch, val_score_heavy, global_step=i, tag="Confusion Matrix - Validation step")

          # Average validation loss at the end of each epoch
          avg_val_loss = total_validation_loss / steps_per_val
          pbar.write(f"\n--- End of Validation for Epoch {int((total_steps + 1) / steps_per_epoch)} ---")
          pbar.write(f" Loss validation: {avg_val_loss:.4f}")
          pbar.write(f"-------------------------------------------\n")

          writer.add_scalars('Training loss (cryocheck)',
                                           {"validation": avg_val_loss},
                                           total_steps + 1)

          # Save checkpoint model at each epoch
          graphdef, state = nnx.split((cryoCheck,))
          NeuralNetworkCheckpointer.save_intermediate(graphdef, state, os.path.join(args.output_path, "cryoCheck_CHECKPOINT"),
                                                      epoch=i)  

          t_score = []
          t_labels = []

        

          i += 1

        

    # Save model
    NeuralNetworkCheckpointer.save(cryoCheck, os.path.join(args.output_path, "cryoCheck"))

    # Remove checkpoint
    shutil.rmtree(os.path.join(args.output_path, "cryoCheck_CHECKPOINT"))
      
  
  elif args.mode=="predict":

    cryoCheck.eval()

    # Prepare grain dataset
    data_loader = generator.return_grain_dataset(batch_size=args.batch_size, shuffle=False, num_epochs=1,
                                                     num_workers=-1, load_to_ram=args.load_images_to_ram)
    steps_per_epoch = int(np.ceil(len(generator.md) / args.batch_size))
    
    # Jitted prediction function
    #predict_fn = nnx.jit(cryoCheck.__call__)
    #predict_fn = nnx.jit(lambda x: cryoCheck(x))

    # PREDICTION LOOP
    print(f"{bcolors.OKCYAN}\n###### Predicting CryoCheck... ######") 

    pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green", total=steps_per_epoch,
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    

    labels_prediction = []


    for (x, index) in pbar:

      euler_angles, shifts, ctf = md_extraction (md_columns, index, vol, args)
      
      projection_pred = volumeProjection(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles,
                                 shifts=shifts,
                                 ctf=ctf)
      result = compute_fourier_residual(projection_pred, x, n_shells=n_shells)
      prediction_frc, _ = standardize_frc_curve(result.frc_curve, result.freqs, ts=args.sr, n_shells_ref=n_shells)
      prediction_frc = jnp.array(prediction_frc)
      #predictions = predict_fn(prediction_res,eval=True)
      predictions = cryoCheck(prediction_frc, eval=True)

      labels_prediction.append(np.array(predictions))

    final_predictions = np.concatenate(labels_prediction, axis=0)

    # Retieve optimal threshold value from training step
    threshold_path = os.path.join(args.output_path, "optimal_threshold_value.txt")
    try:
        with open(threshold_path, "r") as f:
            optimal_threshold = float(f.read().strip())
    except FileNotFoundError:
        optimal_threshold = 0.5

    final_predictions_heavy = (final_predictions > optimal_threshold).astype(int)
    
  
    # Save results 
    md=generator.md 
    md[:, "misalignment_score"] = final_predictions
    md[:, "misalignment_score_heavy"] = final_predictions_heavy
    md.write(os.path.join(args.output_path, "md_final_predictions" +  os.path.splitext(args.md)[1]))

