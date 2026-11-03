#!/usr/bin/env python
"""
decode_states_from_latents CLI test scenarios.

``decode_states_from_latents`` takes a file of latent vectors (``.npy`` or
``.txt``) and a trained network (HetSIREN / Zernike3Deep, ``--reload``), and
decodes each latent into a 3D volume via ``model.decode_volume``, writing
``decoded_volume_XXXX.mrc`` per latent.

The suite first trains small setup networks and feeds them a synthetic set of
latent vectors (centred near 0, like a real centred latent space).

Coverage:

* ``--latents_file`` : ``.npy`` and ``.txt``
* ``--reload``       : a HetSIREN model and a Zernike3Deep model (both expose
                       ``decode_volume``)
* output: one ``decoded_volume_XXXX.mrc`` per latent vector

Note: the program does not create its ``--output_path`` (it writes the .mrc files
straight into it); the suite pre-creates it.
"""

import os
import numpy as np

import phantom
from common import Scenario

N_PARTICLES = 128
BOX = 40
SR = 2.0
LAT_DIM = 6
N_LATENTS = 5
EPOCHS = 6
NGAUSS = 100


def prepare_data(workdir):
    """CTF phantom (HetSIREN) + continuous-only phantom (Zernike3Deep) + a small
    set of latent vectors saved as both .npy and .txt."""
    data_dir = os.path.join(workdir, "data")
    ctf = phantom.write_dataset(
        os.path.join(data_dir, "ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True, seed=1))
    noctf = phantom.write_dataset(
        os.path.join(data_dir, "noctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=False,
                                 compositional=False, seed=0))

    # Latent vectors to decode (small magnitude, like a centred latent space)
    rng = np.random.default_rng(0)
    latents = (0.3 * rng.standard_normal((N_LATENTS, LAT_DIM))).astype(np.float32)
    npy = os.path.join(data_dir, "latents.npy")
    txt = os.path.join(data_dir, "latents.txt")
    np.save(npy, latents)
    np.savetxt(txt, latents)
    return {"ctf": ctf, "noctf": noctf, "latents_npy": npy, "latents_txt": txt}


def data_checks(workdir):
    npy = os.path.join(workdir, "data", "latents.npy")
    lat = np.load(npy)
    return [(
        "latents_shape",
        bool(lat.shape == (N_LATENTS, LAT_DIM)),
        f"latents file shape {lat.shape} (expected {(N_LATENTS, LAT_DIM)})")]


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    ctf, noctf = data["ctf"], data["noctf"]
    npy, txt = data["latents_npy"], data["latents_txt"]

    def out(name):
        return os.path.join(runs, name)

    def mkdirp(path):
        return lambda: os.makedirs(path, exist_ok=True)

    def vols(name):
        return [os.path.join(out(name), f"decoded_volume_{i:04d}.mrc")
                for i in range(N_LATENTS)]

    scn = []

    # --- setup: small HetSIREN ---
    scn.append(Scenario(
        name="setup_hetsiren",
        description="[setup] train HetSIREN | ctf=apply (decoder for the volumes)",
        program="hetsiren",
        args=["--md", ctf["md"], "--sr", str(SR), "--ctf_type", "apply",
              "--mode", "train", "--epochs", str(EPOCHS), "--batch_size", "8",
              "--lat_dim", str(LAT_DIM), "--load_images_to_ram",
              "--output_path", out("setup_hetsiren")],
        expect_files=[os.path.join(out("setup_hetsiren"), "HetSIREN")],
        timeout=900))

    hetsiren_nn = os.path.join(out("setup_hetsiren"), "HetSIREN")

    # 1) decode from a .npy latents file (HetSIREN)
    scn.append(Scenario(
        name="decode_hetsiren_npy",
        description="decode | HetSIREN | latents .npy -> decoded_volume_XXXX.mrc",
        program="decode_states_from_latents",
        args=["--latents_file", npy, "--reload", hetsiren_nn,
              "--output_path", out("decode_hetsiren_npy")],
        expect_files=vols("decode_hetsiren_npy"),
        pre=mkdirp(out("decode_hetsiren_npy")),
        timeout=600))

    # 2) decode from a .txt latents file (HetSIREN)
    scn.append(Scenario(
        name="decode_hetsiren_txt",
        description="decode | HetSIREN | latents .txt -> decoded_volume_XXXX.mrc",
        program="decode_states_from_latents",
        args=["--latents_file", txt, "--reload", hetsiren_nn,
              "--output_path", out("decode_hetsiren_txt")],
        expect_files=vols("decode_hetsiren_txt"),
        pre=mkdirp(out("decode_hetsiren_txt")),
        timeout=600))

    # --- setup: small Zernike3Deep (SLOW: fit_volume) ---
    scn.append(Scenario(
        name="setup_zernike",
        description="[setup] train Zernike3Deep | ctf=None (decoder for the volumes)",
        program="zernike3deep",
        args=["--md", noctf["md"], "--vol", noctf["vol"], "--mask", noctf["mask"],
              "--sr", str(SR), "--ctf_type", "None", "--mode", "train",
              "--epochs", str(EPOCHS), "--batch_size", "8", "--lat_dim", str(LAT_DIM),
              "--L1", "3", "--L2", "3", "--num_gaussians", str(NGAUSS),
              "--load_images_to_ram", "--output_path", out("setup_zernike")],
        expect_files=[os.path.join(out("setup_zernike"), "Zernike3Deep")],
        timeout=1200))

    zernike_nn = os.path.join(out("setup_zernike"), "Zernike3Deep")

    # 3) decode from a .npy latents file (Zernike3Deep)
    scn.append(Scenario(
        name="decode_zernike_npy",
        description="decode | Zernike3Deep | latents .npy -> decoded_volume_XXXX.mrc",
        program="decode_states_from_latents",
        args=["--latents_file", npy, "--reload", zernike_nn,
              "--output_path", out("decode_zernike_npy")],
        expect_files=vols("decode_zernike_npy"),
        pre=mkdirp(out("decode_zernike_npy")),
        timeout=600))

    return scn


# The Zernike3Deep setup runs fit_volume (slow); skip it and its decode under --quick.
SLOW = {"setup_zernike", "decode_zernike_npy"}
