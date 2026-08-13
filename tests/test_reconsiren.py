#!/usr/bin/env python
"""
ReconSIREN CLI test scenarios.

Exercises (ideally) *every* ``reconsiren`` option so future regressions surface
as a failing scenario.  ``reconsiren`` performs an *ab initio* estimation of
particle pose, shifts and an initial (heterogeneous) volume with neural
networks.  Coverage:

* ``--ctf_type``               : None, apply, wiener, precorrect
* ``--mode``                   : train, predict
* ``--vol`` / ``--mask``       : reference-volume start (Gaussian fitting path)
* ``--num_gaussians``          : size of the Gaussian point cloud
* optimized/legacy profiles    : independent low-rank heads, direct consensus,
                                  candidate and Gaussian render chunking
* pose exploration             : bank-aware candidate coverage, low-frequency
                                  candidate scoring curriculum
* heterogeneity profiles       : legacy and staged anti-collapse residual training
* ``--do_not_learn_volume``    : pose/shift-only refinement against a reference
* ``--refine_current_assignment`` / ``--symmetry_group``
* ``--load_images_to_ram`` (on) and the memory-mapped path (off) + ``--ssd_scratch_folder``
* ``--batch_size`` / ``--learning_rate`` / ``--dataset_split_fraction`` /
  ``--epochs`` / ``--reload``

The data is the synthetic continuous+compositional phantom from ``phantom.py``.
``epochs = 3`` so the per-epoch intermediate-result path (predicted volume, het
maps, ``ReconSIREN_CHECKPOINT``) and its end-of-training cleanup are exercised.
"""

import os
import numpy as np

import phantom
from common import Scenario

# Phantom size — kept small so the whole suite runs quickly while still being a
# genuine ab-initio heterogeneous reconstruction problem.  ``N_PARTICLES`` stays
# comfortably above the k-means cluster counts used both during training
# (n_clusters=10) and in predict (n_clusters=20).
N_PARTICLES = 160
# BOX = 32 (not 48 like HetSIREN): ReconSIREN's ab-initio train step — full
# volume + het-volume decoding plus the pose memory bank — is markedly more
# memory-hungry, and BOX=48/bs=8 OOMs an 8 GB GPU.  32³ keeps it comfortably
# under budget while remaining a genuine 3D reconstruction problem.
BOX = 32
SR = 2.0
EPOCHS = 3
# Small Gaussian cloud so the sphere initialisation (and the reference-volume
# fitting in the SLOW cases) stays fast.
NUM_GAUSS = 500


def prepare_data(workdir):
    """Generate (once) a CTF and a CTF-free phantom dataset under ``workdir``."""
    data_dir = os.path.join(workdir, "data")
    ctf = phantom.write_dataset(
        os.path.join(data_dir, "ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True, seed=1))
    noctf = phantom.write_dataset(
        os.path.join(data_dir, "noctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=False, seed=0))
    return {"ctf": ctf, "noctf": noctf}


def data_checks(workdir):
    """Validate the phantom itself: 'None' images carry no CTF.

    Returns a list of ``(name, ok, detail)`` tuples.
    """
    checks = []

    # Build a *matched* pair (identical particles) with and without CTF so the
    # only difference is the CTF application.
    clean = phantom.generate_dataset(24, BOX, SR, apply_ctf=False, noise=0.0, seed=7)
    ctfed = phantom.generate_dataset(24, BOX, SR, apply_ctf=True, noise=0.0, seed=7)

    a = clean["images"]
    b = ctfed["images"]

    def _n(x):
        x = x - x.mean(axis=(1, 2), keepdims=True)
        s = x.std(axis=(1, 2), keepdims=True)
        return x / np.where(s > 0, s, 1.0)
    diff = np.mean(np.abs(_n(a) - _n(b)))

    # The CTF must measurably change the images (otherwise 'apply' == 'None').
    checks.append((
        "ctf_changes_images",
        bool(diff > 0.05),
        f"mean|clean-ctf| = {diff:.3f} (expected > 0.05)"))

    # The 'None' dataset images must be CTF-free.
    checks.append((
        "none_images_are_ctf_free",
        bool(clean["apply_ctf"] is False),
        "noctf dataset generated with apply_ctf=False (no ctfFilter applied)"))

    return checks


def scenarios(workdir, data):
    """Return the ordered list of ReconSIREN CLI scenarios."""
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]
    noctf = data["noctf"]

    def out(name):
        return os.path.join(runs, name)

    def rs(name):  # expected ReconSIREN model dir
        return os.path.join(out(name), "ReconSIREN")

    base = ["--sr", str(SR), "--epochs", str(EPOCHS),
            "--num_gaussians", str(NUM_GAUSS), "--load_images_to_ram",
            "--render_chunk_size", "128", "--candidate_chunk_size", "3",
            "--candidate_coverage_epochs", "1",
            "--candidate_coverage_weight", "0.01", "--candidate_coverage_bins", "64",
            "--candidate_coverage_kappa", "24", "--candidate_bank_samples", "64",
            "--candidate_bank_mix", "0.5",
            "--heterogeneity_profile", "anti_collapse", "--het_start_epoch", "1"]

    scn = []

    # 1) train, ctf=None, RAM, sphere init (no reference volume).
    scn.append(Scenario(
        name="train_none_ram",
        description="train | ctf=None | RAM | sphere init (no --vol)",
        program="reconsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "train",
              "--batch_size", "4",
              "--output_path", out("train_none_ram")] + base,
        expect_files=[rs("train_none_ram")],
        timeout=1200))

    # 2) predict, reloading the model trained in (1).
    #    Predict must create its own output dir and write the pose/shift md, the
    #    consensus map and the per-cluster het maps.
    scn.append(Scenario(
        name="predict_none",
        description="predict | ctf=None | --reload from train_none_ram (creates own output dir)",
        program="reconsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "predict",
              "--batch_size", "4", "--num_gaussians", str(NUM_GAUSS),
              "--reload", out("train_none_ram"),
              "--output_path", out("predict_none"),
              "--sr", str(SR), "--load_images_to_ram"],
        expect_files=[os.path.join(out("predict_none"), "predicted_pose_shifts.xmd"),
                      os.path.join(out("predict_none"), "reconsiren_map.mrc")],
        timeout=900))

    # 3) train, ctf=apply, memory-mapped (no RAM) + ssd scratch + custom hyperparams
    scn.append(Scenario(
        name="train_apply_mmap",
        description="train | ctf=apply | mmap+ssd_scratch | bs=4 lr=5e-5 split=0.7,0.3",
        program="reconsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--batch_size", "4", "--learning_rate", "5e-5",
              "--dataset_split_fraction", "0.7,0.3",
              "--num_gaussians", str(NUM_GAUSS),
              "--candidate_coverage_epochs", "1",
              "--candidate_coverage_weight", "0.01",
              "--candidate_coverage_bins", "64",
              "--candidate_coverage_kappa", "24",
              "--candidate_bank_samples", "64",
              "--candidate_bank_mix", "0.5",
              "--heterogeneity_profile", "legacy",
              "--ssd_scratch_folder", out("train_apply_mmap_scratch"),
              "--sr", str(SR), "--epochs", str(EPOCHS),
              "--output_path", out("train_apply_mmap")],
        expect_files=[rs("train_apply_mmap")],
        timeout=1200))

    # 4) train, ctf=wiener
    scn.append(Scenario(
        name="train_wiener",
        description="train | ctf=wiener | RAM",
        program="reconsiren",
        args=["--md", ctf["md"], "--ctf_type", "wiener", "--mode", "train",
              "--batch_size", "4",
              "--output_path", out("train_wiener")] + base,
        expect_files=[rs("train_wiener")],
        timeout=1200))

    # 5) train, ctf=precorrect
    scn.append(Scenario(
        name="train_precorrect",
        description="train | ctf=precorrect | RAM",
        program="reconsiren",
        args=["--md", ctf["md"], "--ctf_type", "precorrect", "--mode", "train",
              "--batch_size", "4",
              "--output_path", out("train_precorrect")] + base,
        expect_files=[rs("train_precorrect")],
        timeout=1200))

    # 6) train, refine an existing assignment under a symmetry group.
    #    The phantom md already carries per-particle angles/shifts, so
    #    --refine_current_assignment has a starting alignment to refine.
    scn.append(Scenario(
        name="train_refine_symmetry",
        description="train | ctf=apply | --refine_current_assignment --symmetry_group c2 | RAM",
        program="reconsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--batch_size", "4",
              "--refine_current_assignment", "--symmetry_group", "c2",
              "--output_path", out("train_refine_symmetry")] + base,
        expect_files=[rs("train_refine_symmetry")],
        timeout=1200))

    # 7) train with reference volume: Gaussian fitting of --vol inside --mask.
    scn.append(Scenario(
        name="train_vol_mask",
        description="train | ctf=apply | --vol --mask (Gaussian volume fitting)",
        program="reconsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--batch_size", "4",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--num_gaussians", "200",
              "--sr", str(SR), "--epochs", str(EPOCHS), "--load_images_to_ram",
              "--output_path", out("train_vol_mask")],
        expect_files=[rs("train_vol_mask"),
                      os.path.join(out("train_vol_mask"), "Gaussian_volume_fitting"),
                      os.path.join(out("train_vol_mask"), "consensus_volume.mrc")],
        timeout=1800))

    # 8) train with reference volume, pose/shift-only (--do_not_learn_volume):
    #    the documented "high-resolution reference, no map refinement" use case.
    scn.append(Scenario(
        name="train_vol_no_learn_volume",
        description="train | ctf=apply | --vol --mask --do_not_learn_volume",
        program="reconsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--batch_size", "4",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--do_not_learn_volume", "--num_gaussians", "200",
              "--sr", str(SR), "--epochs", str(EPOCHS), "--load_images_to_ram",
              "--output_path", out("train_vol_no_learn_volume")],
        expect_files=[rs("train_vol_no_learn_volume")],
        timeout=1800))

    return scn


# Scenarios that need the reference-volume Gaussian fitting (slow); skipped by --quick.
SLOW = {"train_vol_mask", "train_vol_no_learn_volume"}
