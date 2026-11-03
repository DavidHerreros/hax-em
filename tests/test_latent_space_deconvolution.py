#!/usr/bin/env python
"""
latent_space_deconvolution CLI test scenarios.

The program now recovers the latent vectors **on the fly** from a trained network
(``--nn_path``) applied to the images in ``--md`` (instead of a pre-saved
``latents.npy``), and deconvolves the resulting landscape using per-particle
covariances (``--covariances``, output of ``estimate_latent_covariances``).

The suite therefore builds the full pipeline as ordered scenarios:

    1. train a small HetSIREN              -> the network (--nn_path)
    2. estimate_latent_covariances on it   -> covariance_matrices.npy (--covariances)
    3. latent_space_deconvolution train    -> a trained deconvolver
    4. latent_space_deconvolution predict  -> latents_deconvolved.npy
    5. latent_space_deconvolution train (mmap path + custom strength)

Coverage of latent_space_deconvolution options:

* on-the-fly latent recovery from ``--nn_path`` + ``--md``
* ``--mode train / predict`` (+ ``--reload`` for predict)
* ``--covariances`` / ``--deconvolution_strength`` / ``--lat_dim`` / ``--batch_size``
* ``--load_images_to_ram`` (on) and the mmap path (off) + ``--ssd_scratch_folder``
* output: a saved ``deconvolver`` model + ``latents_deconvolved.npy``
"""

import os

import phantom
from common import Scenario

N_PARTICLES = 192
BOX = 44
SR = 2.0
LAT_DIM = 6          # HetSIREN latent dim == Deconvolver latent dim
EPOCHS_NN = 6        # for the setup network training
EPOCHS_DECONV = 3


def prepare_data(workdir):
    """A single CTF phantom dataset is enough (the deconvolver works on whatever
    latents the network encodes)."""
    data_dir = os.path.join(workdir, "data")
    ctf = phantom.write_dataset(
        os.path.join(data_dir, "ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True, seed=1))
    return {"ctf": ctf}


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]

    def out(name):
        return os.path.join(runs, name)

    def mkdirp(path):
        return lambda: os.makedirs(path, exist_ok=True)

    nn_path = os.path.join(out("setup_hetsiren"), "HetSIREN")
    cov_path = os.path.join(out("setup_covariances"), "covariance_matrices.npy")

    scn = []

    # --- setup 1: train a small HetSIREN to serve as the encoder ---
    scn.append(Scenario(
        name="setup_hetsiren",
        description="[setup] train HetSIREN | ctf=apply (encoder for on-the-fly latents)",
        program="hetsiren",
        args=["--md", ctf["md"], "--sr", str(SR), "--ctf_type", "apply",
              "--mode", "train", "--epochs", str(EPOCHS_NN), "--batch_size", "8",
              "--lat_dim", str(LAT_DIM), "--load_images_to_ram",
              "--output_path", out("setup_hetsiren")],
        expect_files=[nn_path],
        timeout=900))

    # --- setup 2: per-particle covariances from estimate_latent_covariances ---
    scn.append(Scenario(
        name="setup_covariances",
        description="[setup] estimate_latent_covariances -> covariance_matrices.npy",
        program="estimate_latent_covariances",
        args=["--md", ctf["md"], "--nn_path", nn_path, "--batch_size", "16",
              "--load_images_to_ram", "--output_path", out("setup_covariances")],
        expect_files=[cov_path],
        pre=mkdirp(out("setup_covariances")),
        timeout=600))

    # 1) deconvolution training | latents recovered on the fly | RAM
    scn.append(Scenario(
        name="deconv_train_ram",
        description="train | latents recovered on-the-fly from --nn_path | RAM",
        program="latent_space_deconvolution",
        args=["--md", ctf["md"], "--nn_path", nn_path, "--covariances", cov_path,
              "--lat_dim", str(LAT_DIM), "--mode", "train",
              "--epochs", str(EPOCHS_DECONV), "--batch_size", "32",
              "--load_images_to_ram", "--output_path", out("deconv_train_ram")],
        expect_files=[os.path.join(out("deconv_train_ram"), "deconvolver")],
        timeout=600))

    # 2) deconvolution prediction | reload the trained deconvolver
    scn.append(Scenario(
        name="deconv_predict",
        description="predict | --reload trained deconvolver -> latents_deconvolved.npy",
        program="latent_space_deconvolution",
        args=["--md", ctf["md"], "--nn_path", nn_path, "--covariances", cov_path,
              "--lat_dim", str(LAT_DIM), "--mode", "predict",
              "--reload", out("deconv_train_ram"), "--batch_size", "32",
              "--load_images_to_ram", "--output_path", out("deconv_predict")],
        expect_files=[os.path.join(out("deconv_predict"), "latents_deconvolved.npy")],
        timeout=600))

    # 3) deconvolution training | mmap (no RAM) + ssd scratch | custom strength/batch
    scn.append(Scenario(
        name="deconv_train_mmap",
        description="train | mmap+ssd_scratch | deconvolution_strength=2.0 | batch=16",
        program="latent_space_deconvolution",
        args=["--md", ctf["md"], "--nn_path", nn_path, "--covariances", cov_path,
              "--lat_dim", str(LAT_DIM), "--mode", "train",
              "--epochs", str(EPOCHS_DECONV), "--batch_size", "16",
              "--deconvolution_strength", "2.0",
              "--ssd_scratch_folder", out("deconv_train_mmap_scratch"),
              "--output_path", out("deconv_train_mmap")],
        expect_files=[os.path.join(out("deconv_train_mmap"), "deconvolver")],
        timeout=600))

    return scn
