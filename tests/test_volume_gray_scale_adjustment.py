#!/usr/bin/env python
"""
volume_gray_scale_adjustment CLI test scenarios.

Learns a gray-level adjustment for a reference volume (``--vol``) so its
projections match the experimental images in ``--md``. Train saves a
``volumeAdjustment`` model; predict writes ``adjusted_volume.mrc`` (predict needs
only ``--vol`` + ``--reload``, not images).

Coverage:

* ``--ctf_type``       : None, apply, wiener
* ``--mode``           : train, predict (+ ``--reload``)
* ``--predicts_value`` : on (predict a*vol+b) and off (predict new voxel values)
* ``--vol`` (required) / ``--mask`` provided vs auto-generated
* ``--load_images_to_ram`` (on) and the mmap path (off) + ``--ssd_scratch_folder``
* ``--lat_dim`` / ``--batch_size`` / ``--learning_rate`` / ``--dataset_split_fraction``
* outputs: ``volumeAdjustment`` model; ``adjusted_volume.mrc``

Notes (pre-existing, unpatched): the end-of-training
``rmtree(volumeAdjustment_CHECKPOINT)`` is unconditional but a checkpoint is
written at ``i==1``, so ``epochs >= 2`` is safe (the suite uses 3). ``predict`` /
no-``--mask`` training do not create ``--output_path`` (the suite pre-creates it).
"""

import os

import phantom
from common import Scenario

N_PARTICLES = 96
BOX = 40
SR = 2.0
EPOCHS = 3


def prepare_data(workdir):
    data_dir = os.path.join(workdir, "data")
    ctf = phantom.write_dataset(
        os.path.join(data_dir, "ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True, seed=1))
    return {"ctf": ctf}


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]
    md, vol, mask = ctf["md"], ctf["vol"], ctf["mask"]

    def out(name):
        return os.path.join(runs, name)

    def mkdirp(path):
        return lambda: os.makedirs(path, exist_ok=True)

    def model(name):
        return os.path.join(out(name), "volumeAdjustment")

    def adjusted(name):
        return [os.path.join(out(name), "adjusted_volume.mrc")]

    base = ["--md", md, "--vol", vol, "--sr", str(SR), "--epochs", str(EPOCHS),
            "--batch_size", "16", "--load_images_to_ram"]
    scn = []

    # 1) train | ctf=apply | --mask | --predicts_value
    scn.append(Scenario(
        name="train_apply_pv",
        description="train | ctf=apply | --mask | --predicts_value",
        program="volume_gray_scale_adjustment",
        args=base + ["--mask", mask, "--ctf_type", "apply", "--mode", "train",
                     "--predicts_value", "--output_path", out("train_apply_pv")],
        expect_files=[model("train_apply_pv")],
        timeout=600))

    # 2) predict | reload (1), predicts_value -> a*vol+b
    scn.append(Scenario(
        name="predict_pv",
        description="predict | --reload train_apply_pv | --predicts_value -> adjusted_volume.mrc",
        program="volume_gray_scale_adjustment",
        args=["--vol", vol, "--mask", mask, "--sr", str(SR), "--ctf_type", "apply",
              "--mode", "predict", "--predicts_value",
              "--reload", out("train_apply_pv"), "--output_path", out("predict_pv")],
        expect_files=adjusted("predict_pv"),
        pre=mkdirp(out("predict_pv")),
        timeout=300))

    # 3) train | ctf=None | no predicts_value | custom lr/split/batch
    scn.append(Scenario(
        name="train_none_novpv",
        description="train | ctf=None | predict voxel values | lr=1e-4 split=0.7,0.3",
        program="volume_gray_scale_adjustment",
        args=["--md", md, "--vol", vol, "--mask", mask, "--sr", str(SR),
              "--ctf_type", "None", "--mode", "train", "--epochs", str(EPOCHS),
              "--batch_size", "8", "--learning_rate", "1e-4",
              "--dataset_split_fraction", "0.7,0.3", "--lat_dim", "4",
              "--load_images_to_ram", "--output_path", out("train_none_novpv")],
        expect_files=[model("train_none_novpv")],
        timeout=600))

    # 4) predict | reload (3), no predicts_value -> new voxel values placed in volume
    scn.append(Scenario(
        name="predict_novpv",
        description="predict | --reload train_none_novpv | voxel-values branch -> adjusted_volume.mrc",
        program="volume_gray_scale_adjustment",
        args=["--vol", vol, "--mask", mask, "--sr", str(SR), "--ctf_type", "None",
              "--mode", "predict", "--reload", out("train_none_novpv"),
              "--output_path", out("predict_novpv")],
        expect_files=adjusted("predict_novpv"),
        pre=mkdirp(out("predict_novpv")),
        timeout=300))

    # 5) train | ctf=wiener | mmap (no RAM) + ssd scratch
    scn.append(Scenario(
        name="train_wiener_mmap",
        description="train | ctf=wiener | mmap + ssd_scratch",
        program="volume_gray_scale_adjustment",
        args=["--md", md, "--vol", vol, "--mask", mask, "--sr", str(SR),
              "--ctf_type", "wiener", "--mode", "train", "--epochs", str(EPOCHS),
              "--batch_size", "16", "--predicts_value",
              "--ssd_scratch_folder", out("train_wiener_mmap_scratch"),
              "--output_path", out("train_wiener_mmap")],
        expect_files=[model("train_wiener_mmap")],
        timeout=600))

    return scn
