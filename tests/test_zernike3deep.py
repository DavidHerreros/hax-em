#!/usr/bin/env python
"""
Zernike3Deep CLI test scenarios.

Zernike3Deep models conformational variability as a **continuous deformation**
(a Zernike3D displacement field), so the phantom here is generated with
``compositional=False`` — motion only, no mass appearing/disappearing.

Coverage of the ``zernike3deep`` options:

* ``--ctf_type``   : None, apply, wiener, precorrect
* ``--mode``       : train, predict
* ``--vol`` (required) / ``--mask`` (provided and auto-generated)
* ``--L1`` / ``--L2``       (Zernike degrees, varied incl. defaults)
* ``--num_gaussians``       (fixed small count, and the default densify fit)
* ``--lat_dim`` / ``--batch_size`` / ``--learning_rate`` / ``--dataset_split_fraction``
* ``--load_images_to_ram`` (on) and the mmap path (off) + ``--ssd_scratch_folder``
* ``--epochs`` / ``--reload``
"""

import os
import numpy as np

import phantom
from common import Scenario

N_PARTICLES = 160
BOX = 48
SR = 2.0
EPOCHS = 6
NGAUSS = 100          # small fixed Gaussian count -> fast fit_volume


def prepare_data(workdir):
    """Generate continuous-only (motion-only) CTF and CTF-free phantom datasets."""
    data_dir = os.path.join(workdir, "data")
    ctf = phantom.write_dataset(
        os.path.join(data_dir, "ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True,
                                 compositional=False, seed=1))
    noctf = phantom.write_dataset(
        os.path.join(data_dir, "noctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=False,
                                 compositional=False, seed=0))
    return {"ctf": ctf, "noctf": noctf}


def data_checks(workdir):
    """Validate the phantom is continuous-only (motion without mass change)."""
    checks = []

    # Mass conservation: with compositional=False the total density must be
    # identical across conformations (only positions change).
    _, w_lo = phantom.conformation(0.1, BOX, compositional=False, seed=0)
    c_hi, w_hi = phantom.conformation(0.9, BOX, compositional=False, seed=0)
    c_lo, _ = phantom.conformation(0.1, BOX, compositional=False, seed=0)
    mass_lo, mass_hi = float(w_lo.sum()), float(w_hi.sum())
    checks.append((
        "continuous_only_conserves_mass",
        bool(abs(mass_lo - mass_hi) < 1e-4),
        f"total mass t=0.1 -> {mass_lo:.3f}, t=0.9 -> {mass_hi:.3f} (must match)"))

    # Motion is actually present: the mobile blob position changes with t.
    moved = float(np.abs(c_hi - c_lo).max())
    checks.append((
        "continuous_motion_present",
        bool(moved > 1.0),
        f"max atom displacement t=0.1->0.9 = {moved:.2f} px (expected > 1)"))

    # Sanity: a compositional phantom WOULD change mass (guards the switch).
    _, w_comp_lo = phantom.conformation(0.1, BOX, compositional=True, seed=0)
    _, w_comp_hi = phantom.conformation(0.9, BOX, compositional=True, seed=0)
    checks.append((
        "compositional_switch_effective",
        bool(abs(float(w_comp_lo.sum()) - float(w_comp_hi.sum())) > 0.1),
        "compositional=True changes total mass (so False genuinely removes it)"))

    return checks


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]
    noctf = data["noctf"]

    def out(name):
        return os.path.join(runs, name)

    def z3d(name):       # expected Zernike3Deep model dir
        return os.path.join(out(name), "Zernike3Deep")

    base = ["--sr", str(SR), "--epochs", str(EPOCHS), "--load_images_to_ram"]
    scn = []

    # 1) train | ctf=None | mask provided | fixed gaussians | small Zernike degrees
    scn.append(Scenario(
        name="train_none",
        description="train | ctf=None | --vol --mask | num_gaussians=100 | L1=3 L2=3 | RAM",
        program="zernike3deep",
        args=["--md", noctf["md"], "--vol", noctf["vol"], "--mask", noctf["mask"],
              "--ctf_type", "None", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--L1", "3", "--L2", "3", "--num_gaussians", str(NGAUSS),
              "--output_path", out("train_none")] + base,
        expect_files=[z3d("train_none"),
                      os.path.join(out("train_none"), "consensus_volume.mrc")],
        timeout=1200))

    # 2) predict | reload from (1)  (must create its own output dir)
    scn.append(Scenario(
        name="predict_none",
        description="predict | ctf=None | --reload from train_none (creates own output dir)",
        program="zernike3deep",
        args=["--md", noctf["md"], "--vol", noctf["vol"], "--mask", noctf["mask"],
              "--ctf_type", "None", "--mode", "predict",
              "--lat_dim", "6", "--batch_size", "8",
              "--reload", out("train_none"),
              "--output_path", out("predict_none"),
              "--sr", str(SR), "--load_images_to_ram"],
        expect_files=[os.path.join(out("predict_none"), "predicted_latents.xmd")],
        timeout=600))

    # 3) train | ctf=apply | memory-mapped (no RAM) + ssd scratch + custom split/L1/L2
    #    Exercises the record-level train/val split fix on a single-shard dataset.
    scn.append(Scenario(
        name="train_apply_mmap",
        description="train | ctf=apply | mmap+ssd_scratch | L1=5 L2=5 | bs=4 lr=5e-5 split=0.7,0.3",
        program="zernike3deep",
        args=["--md", ctf["md"], "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "8", "--batch_size", "4",
              "--L1", "5", "--L2", "5", "--num_gaussians", str(NGAUSS),
              "--learning_rate", "5e-5", "--dataset_split_fraction", "0.7,0.3",
              "--ssd_scratch_folder", out("train_apply_mmap_scratch"),
              "--sr", str(SR), "--epochs", str(EPOCHS),
              "--output_path", out("train_apply_mmap")],
        expect_files=[z3d("train_apply_mmap")],
        timeout=1200))

    # 4b) short training run (epochs < 5): guards the checkpoint-cleanup fix
    scn.append(Scenario(
        name="train_none_short",
        description="train | ctf=None | epochs=2 (short-run checkpoint-cleanup regression guard)",
        program="zernike3deep",
        args=["--md", noctf["md"], "--vol", noctf["vol"], "--mask", noctf["mask"],
              "--ctf_type", "None", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8", "--epochs", "2",
              "--L1", "3", "--L2", "3", "--num_gaussians", str(NGAUSS),
              "--sr", str(SR), "--load_images_to_ram",
              "--output_path", out("train_none_short")],
        expect_files=[z3d("train_none_short")],
        timeout=900))

    # 4) train | ctf=apply | NO --mask -> auto-generated mask path (+writes mask.mrc)
    scn.append(Scenario(
        name="train_apply_automask",
        description="train | ctf=apply | auto-generated mask (no --mask) | num_gaussians=100",
        program="zernike3deep",
        args=["--md", ctf["md"], "--vol", ctf["vol"],
              "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--L1", "3", "--L2", "3", "--num_gaussians", str(NGAUSS),
              "--output_path", out("train_apply_automask")] + base,
        expect_files=[z3d("train_apply_automask"),
                      os.path.join(out("train_apply_automask"), "mask.mrc")],
        timeout=1200))

    # 5) train | ctf=wiener  (SLOW)
    scn.append(Scenario(
        name="train_wiener",
        description="train | ctf=wiener | --vol --mask | num_gaussians=100 | RAM",
        program="zernike3deep",
        args=["--md", ctf["md"], "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--ctf_type", "wiener", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--L1", "3", "--L2", "3", "--num_gaussians", str(NGAUSS),
              "--output_path", out("train_wiener")] + base,
        expect_files=[z3d("train_wiener")],
        timeout=1200))

    # 6) train | ctf=precorrect  (SLOW)
    scn.append(Scenario(
        name="train_precorrect",
        description="train | ctf=precorrect | --vol --mask | num_gaussians=100 | RAM",
        program="zernike3deep",
        args=["--md", ctf["md"], "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--ctf_type", "precorrect", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--L1", "3", "--L2", "3", "--num_gaussians", str(NGAUSS),
              "--output_path", out("train_precorrect")] + base,
        expect_files=[z3d("train_precorrect")],
        timeout=1200))

    # 7) train | ctf=apply | DEFAULT Zernike degrees + DEFAULT densify fit (n_init=2500)  (SLOW)
    scn.append(Scenario(
        name="train_default_fit",
        description="train | ctf=apply | default L1/L2(=7) + default densify Gaussian fit",
        program="zernike3deep",
        args=["--md", ctf["md"], "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "8", "--batch_size", "8",
              "--output_path", out("train_default_fit")] + base,
        expect_files=[z3d("train_default_fit"),
                      os.path.join(out("train_default_fit"), "consensus_volume.mrc")],
        timeout=2400))

    return scn


# Slow scenarios (extra fit_volume cost); skipped under --quick.
SLOW = {"train_wiener", "train_precorrect", "train_default_fit"}
