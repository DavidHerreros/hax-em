#!/usr/bin/env python
"""
image_gray_scale_adjustment CLI test scenarios.

Learns a per-image gray-level adjustment (scale/offset) that matches projections
of a reference volume (``--vol``) to the experimental images in ``--md``. Train
saves an ``imageAdjustment`` model; predict writes ``adjusted_images.mrcs`` + a
metadata file.

Coverage:

* ``--ctf_type``     : None, apply, wiener
* ``--mode``         : train, predict (+ ``--reload``)
* ``--predict_value``: on and off (per-image scalar vs per-pixel adjustment)
* ``--vol`` (required) / ``--mask`` provided vs auto-generated
* ``--load_images_to_ram`` (on) and the mmap path (off) + ``--ssd_scratch_folder``
* ``--lat_dim`` / ``--batch_size`` / ``--learning_rate`` / ``--dataset_split_fraction``
* outputs: ``imageAdjustment`` model; ``adjusted_images.mrcs`` + ``adjusted_images.xmd``

Notes (pre-existing, unpatched): the end-of-training ``rmtree(imageAdjustment_CHECKPOINT)``
is unconditional, but a checkpoint is written at ``i==1`` (epoch 2), so
``epochs >= 2`` is safe (the suite uses 3). ``predict`` and no-``--mask`` training do
not create ``--output_path`` (it is written into before the metrics writer makes
it), so the suite pre-creates it where needed.
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
        return os.path.join(out(name), "imageAdjustment")

    base = ["--md", md, "--vol", vol, "--sr", str(SR), "--epochs", str(EPOCHS),
            "--batch_size", "16", "--load_images_to_ram"]
    scn = []

    # 1) train | ctf=apply | --mask | --predict_value
    scn.append(Scenario(
        name="train_apply_pv",
        description="train | ctf=apply | --mask | --predict_value",
        program="image_gray_scale_adjustment",
        args=base + ["--mask", mask, "--ctf_type", "apply", "--mode", "train",
                     "--predict_value", "--output_path", out("train_apply_pv")],
        expect_files=[model("train_apply_pv")],
        timeout=600))

    # 2) predict | reload (1) -> adjusted images + metadata
    scn.append(Scenario(
        name="predict_pv",
        description="predict | --reload train_apply_pv -> adjusted_images.mrcs + .xmd",
        program="image_gray_scale_adjustment",
        args=["--md", md, "--vol", vol, "--mask", mask, "--sr", str(SR),
              "--ctf_type", "apply", "--mode", "predict", "--predict_value",
              "--batch_size", "16", "--reload", out("train_apply_pv"),
              "--load_images_to_ram", "--output_path", out("predict_pv")],
        expect_files=[os.path.join(out("predict_pv"), "adjusted_images.mrcs"),
                      os.path.join(out("predict_pv"), "adjusted_images.xmd")],
        pre=mkdirp(out("predict_pv")),
        timeout=600))

    # 3) train | ctf=None | no predict_value | custom lr/split/batch
    scn.append(Scenario(
        name="train_none",
        description="train | ctf=None | per-pixel adjustment | lr=1e-4 split=0.7,0.3",
        program="image_gray_scale_adjustment",
        args=["--md", md, "--vol", vol, "--mask", mask, "--sr", str(SR),
              "--ctf_type", "None", "--mode", "train", "--epochs", str(EPOCHS),
              "--batch_size", "8", "--learning_rate", "1e-4",
              "--dataset_split_fraction", "0.7,0.3", "--lat_dim", "4",
              "--load_images_to_ram", "--output_path", out("train_none")],
        expect_files=[model("train_none")],
        timeout=600))

    # 4) train | ctf=wiener | mmap (no RAM) + ssd scratch
    scn.append(Scenario(
        name="train_wiener_mmap",
        description="train | ctf=wiener | mmap + ssd_scratch",
        program="image_gray_scale_adjustment",
        args=["--md", md, "--vol", vol, "--mask", mask, "--sr", str(SR),
              "--ctf_type", "wiener", "--mode", "train", "--epochs", str(EPOCHS),
              "--batch_size", "16", "--predict_value",
              "--ssd_scratch_folder", out("train_wiener_mmap_scratch"),
              "--output_path", out("train_wiener_mmap")],
        expect_files=[model("train_wiener_mmap")],
        timeout=600))

    # 5) train | ctf=apply | NO --mask -> auto-generated mask (writes mask.mrc)
    scn.append(Scenario(
        name="train_automask",
        description="train | ctf=apply | auto-generated mask (no --mask)",
        program="image_gray_scale_adjustment",
        args=base + ["--ctf_type", "apply", "--mode", "train", "--predict_value",
                     "--output_path", out("train_automask")],
        expect_files=[model("train_automask"),
                      os.path.join(out("train_automask"), "mask.mrc")],
        pre=mkdirp(out("train_automask")),
        timeout=600))

    return scn
