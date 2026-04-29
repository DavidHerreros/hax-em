#!/usr/bin/env python

import jax
import jax.numpy as jnp
from flax import nnx
import dm_pix 

import optax

from einops import rearrange

import numpy as np

from hax import * 

# Bottleneck block that is gonna be repeated [3 4 6 3] times for each layer
class BottleneckBlock(nnx.Module):

  expansion = 4

  def __init__(self, in_channels, out_channels, rngs:nnx.Rngs, stride = 1, downsample=True):   

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride
    self.downsample = downsample

    reduced_channels = out_channels // self.expansion   
    self.reduced_channels = reduced_channels

    # Set of tasks repeated in each layer
    self.conv1 = nnx.Conv(in_channels, reduced_channels, kernel_size=(1,1), use_bias=False, rngs=rngs)      
    self.bn1 = nnx.BatchNorm(reduced_channels, rngs=rngs)  

    self.conv2 = nnx.Conv(reduced_channels, reduced_channels, kernel_size=(3,3), strides=stride, padding= 'SAME', use_bias=False, rngs=rngs) 
    self.bn2 = nnx.BatchNorm(reduced_channels, rngs=rngs)

    self.conv3 = nnx.Conv(reduced_channels, out_channels, kernel_size=(1,1), strides=1, use_bias=False, rngs=rngs) 
    self.bn3 = nnx.BatchNorm(out_channels, rngs=rngs)

    # Downsampling
    if self.downsample:
      self.downsample_conv = nnx.Conv(in_channels, out_channels, kernel_size=(1,1), strides=stride, use_bias=False, rngs=rngs)
      self.downsample_bn = nnx.BatchNorm(out_channels, rngs=rngs)


  def __call__(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = nnx.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = nnx.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample:
      identity = self.downsample_conv(x)
      identity = self.downsample_bn(identity)

    out += identity
    out = nnx.relu(out)

    return out


class ResNet(nnx.Module):
  def __init__ (self, rngs:nnx.Rngs, block=BottleneckBlock, layers=[3, 4, 6, 3], num_classes=1):
    
    self.in_channels = 64
    self.block = block
    self.layers = layers
    self.num_classes = num_classes

    # Initial adaptating convolution 
    self.conv1 = nnx.Conv(1, 64, kernel_size=(7,7), strides=2, padding=3, use_bias=False, rngs=rngs)
    self.bn1 = nnx.BatchNorm(self.in_channels, rngs=rngs)  #perchè solo conv1 ha bisogno della bn

    self.conv2 = nnx.Conv(64, 128, kernel_size=(1,1), strides=1, padding='SAME', use_bias=False, rngs=rngs)
    self.conv3 = nnx.Conv(128, 256, kernel_size=(1,1), strides=1, padding='SAME', use_bias=False, rngs=rngs)
    self.conv4 = nnx.Conv(256, 512, kernel_size=(1,1), strides=1, padding='SAME', use_bias=False, rngs=rngs)

    # Residual layers 
    self.layer1 = self._make_layer(block, self.in_channels, 64, layers[0], rngs=rngs)
    self.layer2 = self._make_layer(block, 128, 128, layers[1], stride=2, rngs=rngs)
    self.layer3 = self._make_layer(block, 256, 256, layers[2], stride=2, rngs=rngs)
    self.layer4 = self._make_layer(block, 512, 512, layers[3], stride=2, rngs=rngs)

    # Linear layer for classification
    self.fc = nnx.Linear(512, 1, rngs=rngs)   

  def _make_layer(self, block, in_channels, out_channels, blocks,  rngs, stride=1):

    downsample = False    
    if stride != 1:
      downsample = True   

    layers = []         
    layers.append(block(in_channels=in_channels, out_channels=out_channels, rngs=rngs, stride=stride, downsample=downsample))        

    for _ in range(1, blocks):
            layers.append(block(in_channels=in_channels, out_channels=out_channels, rngs=rngs))                                      

    return nnx.Sequential(*layers)


  def __call__(self,x):
    if x.ndim == 3:
      # if x is (N,H,W) add a channel dimension
      x=jnp.expand_dims(x, -1)
    elif x.ndim == 2:
      # if x is (N,D), reshape to (N,H,W,1)
      x=x.reshape(x.shape[0],int(jnp.sqrt(x.shape[1])), int(jnp.sqrt(x.shape[1])), 1)
    elif x.ndim == 4:
      # if x is already a (N,H,W,C), ensure C=1
      if x.shape[-1] != 50:                  #50?
        raise ValueError("Expected input with 1 channel, but got {} channels.".format(x.shape[-1]))
    else:
      raise ValueError("Unsupported input dimensions: {}.".format(x.shape[-1]))


    # Initial convolutional layers
    x = self.conv1(x)
    x = self.bn1(x)
    x = nnx.relu(x)

    # Residual layers
    x = self.layer1(x)
    x = self.conv2(x)
    x = self.layer2(x)
    x = self.conv3(x)
    x = self.layer3(x)
    x = self.conv4(x)
    x = self.layer4(x)

    # Pooling and classification
    print(f"Shape prima del pooling: {x.shape}")
    x = jnp.mean(x, axis=(1,2))
    x = self.fc(x)

    return x
  

# Training and Validation
@nnx.jit
def cryoCheck_step(model, optimizer, x, labels, train: bool):

    def loss_fn(model, x, labels):
        logits = model(x)
        # Binary cross entropy
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

        return loss

    if train == True:
      grad_fn = nnx.value_and_grad(loss_fn)

      loss, grads = grad_fn(model, x, labels)

      optimizer.update(grads)

    else:
      loss = loss_fn(model, x, labels)

    return loss, model



# Utils

# Extracting metadata : euler angles, shifts, ctf
def md_extraction(md_columns, index, vol, args):

    euler_angles = md_columns["euler_angles"][index] 

    # Precompute batch shifts
    shifts = md_columns["shifts"][index]

    #precompute batch CTFs
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


# Projecting the volume 
def Preprocessing(vol, mask, euler_angles, shifts, ctf):

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
    xsize=vol.shape[0]
    c_sampling = jnp.stack([coords[..., 1], coords[..., 0]], axis=2)
    images = jnp.zeros((B, xsize, xsize), dtype=vol.dtype) 

    # Forward mapping
    bamp = values[None, ...]

    bposf = jnp.floor(c_sampling)
    bposi = bposf.astype(jnp.int32)
    bposf = c_sampling - bposf

    # Split voxels intensity in 4 weights assigned to the four nearest pixels of a targeted one
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



def main():

  import os
  import sys
  from tqdm import tqdm
  import random
  import numpy as np
  import argparse
  #import shutil
  from xmipp_metadata.image_handler import ImageHandler
  import optax
  from contextlib import closing
  from hax.utils.loggers import bcolors
  from hax.checkpointer import NeuralNetworkCheckpointer
  from hax.generators import MetaDataGenerator, extract_columns
  #from hax.networks import train_step_volume_adjustment
  #from hax.metrics import JaxSummaryWriter

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
  args, _ = parser.parse_known_args()


  # Check that training and validation fractions add up to one
  if sum(args.dataset_split_fraction) != 1:
        raise ValueError(
            f"The sum of {bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} is not equal one. Please, update the values "
            f"to fulfill this requirement.")
        
  # Volume and Mask handling
  vol = ImageHandler(args.vol).getData()
  mask = ImageHandler().generateMask(inputFn=vol, boxsize=64)

  # Prepare network
  rngs = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
  cryoCheck = ResNet(rngs=nnx.Rngs(rngs))

  # Reload network
  if args.reload is not None:
      cryoCheck = NeuralNetworkCheckpointer.load(os.path.join(args.reload, "Trained_CryoCheck"))

  # Train network
  if args.mode == "train":
    
    cryoCheck.train()

    # Load metadata
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Prepare grain dataset
    if not args.load_images_to_ram:
            mmap_output_dir = args.ssd_scratch_folder if args.ssd_scratch_folder is not None else args.output_path
            generator.prepare_grain_array_record(mmap_output_dir=mmap_output_dir, preShuffle=False, num_workers=4,
                                                 precision=np.float16, group_size=1, shard_size=10000)  #shard: significa che ho più archivi con 10000 immagini ciascuno e non tutti le immagini in uno solo
            
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


    #TRAINING LOOP
    print(f"{bcolors.OKCYAN}\n###### Training CryoCheck... ######") 

    i = 0 
    pbar = tqdm(range(args.epochs * steps_per_epoch), file=sys.stdout, ascii=" >=",
                    colour="green",
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
  
  
    with closing(iter(data_loader_train)) as iter_data_loader_train, closing(iter(data_loader_val)) as iter_data_loader_val:
      for total_steps in pbar:
        (x, index) = next(iter_data_loader_train) 

        euler_angles, shifts, ctf = md_extraction (md_columns, index, vol, args)

        batch_size = len(index)
        
        # Aligned images
        aligned_imgs = jnp.abs(Preprocessing(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles,
                                 shifts=shifts,
                                 ctf=ctf) - x)
        alignes_labels = jnp.ones((batch_size,1))

        # Misaligned images - Data Augmentation
        rngs, subkey = jax.random.split(rngs)
        noise = (jax.random.normal(subkey, shape=euler_angles.shape) * 2) + 5
        euler_angles_noisy = euler_angles + noise

        misaligned_imgs = jnp.abs(Preprocessing(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles_noisy,
                                 shifts=shifts,
                                 ctf=ctf) - x)
        misalignes_labels = jnp.zeros((batch_size,1))
      
       
        imgs=jnp.concatenate([aligned_imgs, misaligned_imgs], axis=0)
        labels=jnp.concatenate([alignes_labels, misalignes_labels], axis=0)


        if total_steps % steps_per_epoch == 0:    
          
          total_loss = 0
          total_validation_loss = 0 

          # For progress bar (TQDM)
          #step = 1
          pbar.set_description(f"Epoch {int(total_steps / steps_per_epoch + 1)}/{args.epochs}")

          # Validation step 
          pbar.set_postfix_str(f"{bcolors.WARNING}Running validation step...{bcolors.ENDC}")
          
          for _ in range(steps_per_val):
            (x_validation, index_validation) = next(iter_data_loader_val)

            euler_angles, shifts, ctf = md_extraction (md_columns, index_validation, vol, args)

            # Aligned images
            aligned_vimgs = jnp.abs(Preprocessing(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles,
                                 shifts=shifts,
                                 ctf=ctf) - x_validation)
            
        
            # Misaligned images
            rngs, subkey_v = jax.random.split(rngs)
            noise = (jax.random.normal(subkey_v, shape=euler_angles.shape) * 2) + 5
            euler_angles_noisy = euler_angles + noise

            misaligned_vimgs = jnp.abs(Preprocessing(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles_noisy,
                                 shifts=shifts,
                                 ctf=ctf) - x_validation)
              
            imgs_validation = jnp.concatenate([aligned_vimgs, misaligned_vimgs],axis=0)
              
        
            loss_validation, cryoCheck = cryoCheck_step(cryoCheck, optimizer, x=imgs_validation, labels=labels, train=False)
            total_validation_loss += loss_validation

          i += 1


        loss, cryoCheck = cryoCheck_step(cryoCheck, optimizer, x=imgs, labels=labels, train=True)
        total_loss += loss
  

  
  elif args.mode=="predict":

    cryoCheck.eval()

    # Prepare network
    generator = MetaDataGenerator(args.md)
    md_columns = extract_columns(generator.md)

    # Prepare grain dataset
    data_loader = generator.return_grain_dataset(batch_size=args.batch_size, shuffle=False, num_epochs=1,
                                                     num_workers=-1, load_to_ram=args.load_images_to_ram)
    steps_per_epoch = int(np.ceil(len(generator.md) / args.batch_size))
    
    # Jitted prediction function
    predict_fn = nnx.jit(cryoCheck.__call__)

    # PREDICTION LOOP
    print(f"{bcolors.OKCYAN}\n###### Predicting CryoCheck... ######")

    pbar = tqdm(data_loader, desc=f"Progress", file=sys.stdout, ascii=" >=", colour="green", total=steps_per_epoch,
                    bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    

    labels_prediction = []

    for (x, index) in pbar:

      euler_angles, shifts, ctf = md_extraction (md_columns, index, vol, args)

      prediction_imgs = jnp.abs(Preprocessing(vol=vol,
                                 mask=mask,
                                 euler_angles=euler_angles,
                                 shifts=shifts,
                                 ctf=ctf) - x)
      
      
      predictions = predict_fn(prediction_imgs)

    final_predictions = labels_prediction.append(np.array(predictions))
    
  # Save results 
  md=generator.md #potresti farlo direttamente con md_columns
  md[:, "final predictions"] = np.concatenate(final_predictions, axis=0)
  md.write(os.path.join(args.output_path, "md_final" +  os.path.splitext(args.md)[1]))

