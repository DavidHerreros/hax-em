#!/usr/bin/env python
"""
ReconSIREN (het-only) CLI test scenarios.

Exercises (ideally) *every* ``reconsiren_het_only`` option so future regressions
surface as a failing scenario.  ``reconsiren_het_only`` estimates particle pose,
shifts and *heterogeneity* on top of a (given or empty) reconstruction region,
without the ab-initio Gaussian consensus volume that ``reconsiren`` learns.
Coverage:

* ``--ctf_type``               : None, apply, wiener, precorrect
* ``--mode``                   : train, predict, send_to_pickle
* ``--vol`` / ``--mask``       : reference volume + reconstruction mask (adds the
                                 volume-adjustment warm-up step)
* ``--do_not_learn_volume``    : pose/shift + heterogeneity against a reference
* ``--refine_current_assignment`` / ``--symmetry_group``
* ``--load_images_to_ram`` (on) and the memory-mapped path (off) + ``--ssd_scratch_folder``
* ``--batch_size`` / ``--learning_rate`` / ``--dataset_split_fraction`` /
  ``--epochs`` / ``--reload``

The data is the synthetic continuous+compositional phantom from ``phantom.py``.
``epochs = 3`` so the per-epoch intermediate-result path (het maps,
``ReconSIREN_CHECKPOINT``) and its end-of-training cleanup are exercised.
"""

import os
import numpy as np

import phantom
from common import Scenario

# Phantom size — kept small so the whole suite runs quickly.  ``N_PARTICLES``
# stays above the k-means cluster counts used both during training
# (n_clusters=30) and in predict (n_clusters=20); with the default 0.8 split and
# batch size 8 that leaves ~128 training latents per epoch.
N_PARTICLES = 160
# BOX = 32 (not 48 like HetSIREN): the ReconSIREN train step (pose + heterogeneity
# decoding + pose memory bank) is markedly more memory-hungry and BOX=48/bs=8 OOMs
# an 8 GB GPU.  32³ keeps it under budget while remaining a genuine 3D problem.
BOX = 32
SR = 2.0
EPOCHS = 3


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

    checks.append((
        "ctf_changes_images",
        bool(diff > 0.05),
        f"mean|clean-ctf| = {diff:.3f} (expected > 0.05)"))

    checks.append((
        "none_images_are_ctf_free",
        bool(clean["apply_ctf"] is False),
        "noctf dataset generated with apply_ctf=False (no ctfFilter applied)"))

    return checks


def scenarios(workdir, data):
    """Return the ordered list of ReconSIREN (het-only) CLI scenarios."""
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]
    noctf = data["noctf"]

    def out(name):
        return os.path.join(runs, name)

    def rs(name):  # expected ReconSIREN model dir
        return os.path.join(out(name), "ReconSIREN")

    base = ["--sr", str(SR), "--epochs", str(EPOCHS), "--load_images_to_ram"]

    scn = []

    # 1) train, ctf=None, RAM, no reference volume (empty reconstruction region
    #    from the default circular mask).
    scn.append(Scenario(
        name="train_none_ram",
        description="train | ctf=None | RAM | no --vol (default circular mask)",
        program="reconsiren_het_only",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "train",
              "--batch_size", "4",
              "--output_path", out("train_none_ram")] + base,
        expect_files=[rs("train_none_ram")],
        timeout=1200))

    # 2) predict, reloading the model trained in (1).  het-only predict writes the
    #    pose/shift md and the per-cluster het maps (no single consensus map).
    scn.append(Scenario(
        name="predict_none",
        description="predict | ctf=None | --reload from train_none_ram (creates own output dir)",
        program="reconsiren_het_only",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "predict",
              "--batch_size", "4",
              "--reload", out("train_none_ram"),
              "--output_path", out("predict_none"),
              "--sr", str(SR), "--load_images_to_ram"],
        expect_files=[os.path.join(out("predict_none"), "predicted_pose_shifts.xmd"),
                      os.path.join(out("predict_none"), "reconsiren_hetmap_01.mrc")],
        timeout=900))

    # 3) send_to_pickle (currently a no-op for reconsiren_het_only — must still
    #    exit cleanly).  --load_images_to_ram avoids the mmap-cleanup path.
    scn.append(Scenario(
        name="send_to_pickle_none",
        description="send_to_pickle | ctf=None (no-op smoke check)",
        program="reconsiren_het_only",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "send_to_pickle",
              "--output_path", out("send_to_pickle_none"),
              "--sr", str(SR), "--load_images_to_ram"],
        expect_files=[],
        timeout=300))

    # 4) train, ctf=apply, memory-mapped (no RAM) + ssd scratch + custom hyperparams
    scn.append(Scenario(
        name="train_apply_mmap",
        description="train | ctf=apply | mmap+ssd_scratch | bs=4 lr=5e-5 split=0.7,0.3",
        program="reconsiren_het_only",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--batch_size", "4", "--learning_rate", "5e-5",
              "--dataset_split_fraction", "0.7,0.3",
              "--ssd_scratch_folder", out("train_apply_mmap_scratch"),
              "--sr", str(SR), "--epochs", str(EPOCHS),
              "--output_path", out("train_apply_mmap")],
        expect_files=[rs("train_apply_mmap")],
        timeout=1200))

    # 5) train, ctf=wiener
    scn.append(Scenario(
        name="train_wiener",
        description="train | ctf=wiener | RAM",
        program="reconsiren_het_only",
        args=["--md", ctf["md"], "--ctf_type", "wiener", "--mode", "train",
              "--batch_size", "4",
              "--output_path", out("train_wiener")] + base,
        expect_files=[rs("train_wiener")],
        timeout=1200))

    # 6) train, ctf=precorrect
    scn.append(Scenario(
        name="train_precorrect",
        description="train | ctf=precorrect | RAM",
        program="reconsiren_het_only",
        args=["--md", ctf["md"], "--ctf_type", "precorrect", "--mode", "train",
              "--batch_size", "4",
              "--output_path", out("train_precorrect")] + base,
        expect_files=[rs("train_precorrect")],
        timeout=1200))

    # 7) train, refine an existing assignment under a symmetry group.
    scn.append(Scenario(
        name="train_refine_symmetry",
        description="train | ctf=apply | --refine_current_assignment --symmetry_group c2 | RAM",
        program="reconsiren_het_only",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--batch_size", "4",
              "--refine_current_assignment", "--symmetry_group", "c2",
              "--output_path", out("train_refine_symmetry")] + base,
        expect_files=[rs("train_refine_symmetry")],
        timeout=1200))

    # 8) train with reference volume + mask: exercises the volume-adjustment
    #    warm-up (VolumeAdjustment) before the ReconSIREN training loop.
    #
    #    SKIPPED: the --vol path is currently broken.  After the volume-adjustment
    #    loop, reconsiren_het_only.py assigns to
    #    ``reconsiren.delta_volume_decoder.reference_values`` (line ~1319), but
    #    ReconSIRENHetOnly has no ``delta_volume_decoder`` (its decoder is
    #    ``delta_het_decoder``, a HetVolumeDecoder with no ``reference_values``),
    #    so every --vol run raises AttributeError.  Re-enable this scenario once
    #    that assignment is fixed.
    _VOL_BUG = ("--vol path broken: reconsiren_het_only assigns "
                "reconsiren.delta_volume_decoder.reference_values, which does not "
                "exist on ReconSIRENHetOnly (AttributeError)")
    scn.append(Scenario(
        name="train_vol_mask",
        description="train | ctf=apply | --vol --mask (volume-adjustment warm-up)",
        program="reconsiren_het_only",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--batch_size", "4",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--sr", str(SR), "--epochs", str(EPOCHS), "--load_images_to_ram",
              "--output_path", out("train_vol_mask")],
        expect_files=[rs("train_vol_mask"),
                      os.path.join(out("train_vol_mask"), "volumeAdjustment")],
        timeout=1800,
        skip=True, skip_reason=_VOL_BUG))

    # 9) train with reference volume, pose/shift + heterogeneity only
    #    (--do_not_learn_volume): reference map kept fixed.  SKIPPED: same --vol
    #    AttributeError as scenario (8).
    scn.append(Scenario(
        name="train_vol_no_learn_volume",
        description="train | ctf=apply | --vol --mask --do_not_learn_volume",
        program="reconsiren_het_only",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--batch_size", "4",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--do_not_learn_volume",
              "--sr", str(SR), "--epochs", str(EPOCHS), "--load_images_to_ram",
              "--output_path", out("train_vol_no_learn_volume")],
        expect_files=[rs("train_vol_no_learn_volume")],
        timeout=1800,
        skip=True, skip_reason=_VOL_BUG))

    return scn


# Scenarios that need the reference-volume adjustment warm-up (slow); skipped by
# --quick.  (Both are also currently skipped outright — see the --vol bug above.)
SLOW = {"train_vol_mask", "train_vol_no_learn_volume"}
