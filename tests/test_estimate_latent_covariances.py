#!/usr/bin/env python
"""
estimate_latent_covariances CLI test scenarios.

``estimate_latent_covariances`` consumes a *previously trained* network
(HetSIREN / Zernike3Deep, saved by ``NeuralNetworkCheckpointer``) and, for each
particle, perturbs the clean reprojection with CTF-coloured noise, re-encodes it
many times, and estimates the latent covariance matrix. It writes
``covariance_matrices.npy`` and ``latents.npy``.

Because the program needs a model on disk, the suite first runs *setup* training
scenarios (a small HetSIREN and a small Zernike3Deep) and then points the
estimate runs at the saved models via ``--nn_path``.

Coverage:

* ``--nn_path``  : a HetSIREN model (ctf=apply) and a Zernike3Deep model (ctf=None)
                   -> exercises both the CTF "apply" branch and the non-apply
                   branch of the covariance estimator, and both model interfaces.
* ``--batch_size``
* ``--load_images_to_ram`` (on) and the mmap path (off) + ``--ssd_scratch_folder``
* output: ``covariance_matrices.npy`` + ``latents.npy``

Note: ``estimate_latent_covariances`` does not create its ``--output_path`` (it
``np.save``s straight into it); the suite pre-creates it.
"""

import os

import phantom
from common import Scenario

N_PARTICLES = 128
BOX = 44
SR = 2.0
EPOCHS = 6
LAT_DIM = 6
NGAUSS = 100


def prepare_data(workdir):
    """CTF phantom (for the HetSIREN model) + continuous-only CTF-free phantom
    (for the Zernike3Deep model)."""
    data_dir = os.path.join(workdir, "data")
    ctf = phantom.write_dataset(
        os.path.join(data_dir, "ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True, seed=1))
    noctf = phantom.write_dataset(
        os.path.join(data_dir, "noctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=False,
                                 compositional=False, seed=0))
    return {"ctf": ctf, "noctf": noctf}


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]
    noctf = data["noctf"]

    def out(name):
        return os.path.join(runs, name)

    def mkdirp(path):
        return lambda: os.makedirs(path, exist_ok=True)

    def cov_files(name):
        return [os.path.join(out(name), "covariance_matrices.npy"),
                os.path.join(out(name), "latents.npy")]

    scn = []

    # --- setup: train a small HetSIREN (ctf=apply) to serve as --nn_path ---
    scn.append(Scenario(
        name="setup_hetsiren_apply",
        description="[setup] train HetSIREN | ctf=apply (produces the model used below)",
        program="hetsiren",
        args=["--md", ctf["md"], "--sr", str(SR), "--ctf_type", "apply",
              "--mode", "train", "--epochs", str(EPOCHS), "--batch_size", "8",
              "--lat_dim", str(LAT_DIM), "--load_images_to_ram",
              "--output_path", out("setup_hetsiren_apply")],
        expect_files=[os.path.join(out("setup_hetsiren_apply"), "HetSIREN")],
        timeout=900))

    hetsiren_nn = os.path.join(out("setup_hetsiren_apply"), "HetSIREN")

    # 1) estimate covariances | HetSIREN model | RAM
    scn.append(Scenario(
        name="estimate_hetsiren_ram",
        description="estimate | HetSIREN (ctf=apply branch) | RAM | batch=16",
        program="estimate_latent_covariances",
        args=["--md", ctf["md"], "--nn_path", hetsiren_nn,
              "--batch_size", "16", "--load_images_to_ram",
              "--output_path", out("estimate_hetsiren_ram")],
        expect_files=cov_files("estimate_hetsiren_ram"),
        pre=mkdirp(out("estimate_hetsiren_ram")),
        timeout=600))

    # 2) estimate covariances | HetSIREN model | mmap (no RAM) + ssd scratch
    scn.append(Scenario(
        name="estimate_hetsiren_mmap",
        description="estimate | HetSIREN | mmap+ssd_scratch | batch=8",
        program="estimate_latent_covariances",
        args=["--md", ctf["md"], "--nn_path", hetsiren_nn,
              "--batch_size", "8",
              "--ssd_scratch_folder", out("estimate_hetsiren_mmap_scratch"),
              "--output_path", out("estimate_hetsiren_mmap")],
        expect_files=cov_files("estimate_hetsiren_mmap"),
        pre=mkdirp(out("estimate_hetsiren_mmap")),
        timeout=600))

    # --- setup: train a small Zernike3Deep (ctf=None) (SLOW: fit_volume) ---
    scn.append(Scenario(
        name="setup_zernike_none",
        description="[setup] train Zernike3Deep | ctf=None (produces the model used below)",
        program="zernike3deep",
        args=["--md", noctf["md"], "--vol", noctf["vol"], "--mask", noctf["mask"],
              "--sr", str(SR), "--ctf_type", "None", "--mode", "train",
              "--epochs", str(EPOCHS), "--batch_size", "8", "--lat_dim", str(LAT_DIM),
              "--L1", "3", "--L2", "3", "--num_gaussians", str(NGAUSS),
              "--load_images_to_ram",
              "--output_path", out("setup_zernike_none")],
        expect_files=[os.path.join(out("setup_zernike_none"), "Zernike3Deep")],
        timeout=1200))

    zernike_nn = os.path.join(out("setup_zernike_none"), "Zernike3Deep")

    # 3) estimate covariances | Zernike3Deep model (non-apply CTF branch) | RAM
    scn.append(Scenario(
        name="estimate_zernike_ram",
        description="estimate | Zernike3Deep (ctf=None branch) | RAM | batch=16",
        program="estimate_latent_covariances",
        args=["--md", noctf["md"], "--nn_path", zernike_nn,
              "--batch_size", "16", "--load_images_to_ram",
              "--output_path", out("estimate_zernike_ram")],
        expect_files=cov_files("estimate_zernike_ram"),
        pre=mkdirp(out("estimate_zernike_ram")),
        timeout=600))

    return scn


# The Zernike3Deep model training runs fit_volume (slow); skip it and its
# dependent estimate run under --quick.
SLOW = {"setup_zernike_none", "estimate_zernike_ram"}
