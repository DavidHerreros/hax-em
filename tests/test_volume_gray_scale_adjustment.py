#!/usr/bin/env python
"""
volume_gray_scale_adjustment CLI test scenarios.

Fits, in closed form, the single gray scale that makes the projections of a map
(``--vol``) match the images in ``--md``, and writes the map times that scale.
Single pass, no network: no train/predict modes.

Coverage:

* ``--ctf_type``   : apply, precorrect, None
* ``--fit_offset`` : on / off
* ``--mask``       : provided / not provided
* ``--batch_size`` / ``--threads`` (several chunks) and ``--ssd_scratch_folder``
* outputs: ``adjusted_volume.mrc`` + ``gray_scale.txt``

The program creates its own ``--output_path``.
"""

import os

import phantom
from common import Scenario

N_PARTICLES = 96
BOX = 40
SR = 2.0


def prepare_data(workdir):
    data_dir = os.path.join(workdir, "data")
    ctf = phantom.write_dataset(
        os.path.join(data_dir, "ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True, seed=1))
    no_ctf = phantom.write_dataset(
        os.path.join(data_dir, "no_ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=False, seed=2))
    return {"ctf": ctf, "no_ctf": no_ctf}


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]
    no_ctf = data["no_ctf"]

    def out(name):
        return os.path.join(runs, name)

    def adjusted(name):
        return [os.path.join(out(name), "adjusted_volume.mrc"),
                os.path.join(out(name), "gray_scale.txt")]

    base = ["--md", ctf["md"], "--vol", ctf["vol"], "--sr", str(SR)]
    scn = []

    # 1) defaults: scale only, whole image
    scn.append(Scenario(
        name="adjust_apply",
        description="adjust | ctf=apply (defaults)",
        program="volume_gray_scale_adjustment",
        args=base + ["--ctf_type", "apply", "--output_path", out("adjust_apply")],
        expect_files=adjusted("adjust_apply"),
        timeout=600))

    # 2) per-image offsets while fitting + mask
    scn.append(Scenario(
        name="adjust_offset_mask",
        description="adjust | ctf=precorrect | --fit_offset --mask",
        program="volume_gray_scale_adjustment",
        args=base + ["--ctf_type", "precorrect", "--fit_offset", "--mask", ctf["mask"],
                     "--output_path", out("adjust_offset_mask")],
        expect_files=adjusted("adjust_offset_mask"),
        timeout=600))

    # 3) CTF-free images
    scn.append(Scenario(
        name="adjust_none",
        description="adjust | ctf=None | CTF-free phantom",
        program="volume_gray_scale_adjustment",
        args=["--md", no_ctf["md"], "--vol", no_ctf["vol"], "--sr", str(SR), "--ctf_type", "None",
              "--output_path", out("adjust_none")],
        expect_files=adjusted("adjust_none"),
        timeout=600))

    # 4) chunked streaming and the SSD cache
    scn.append(Scenario(
        name="adjust_scratch",
        description="adjust | ctf=apply | --batch_size 32 --threads 2 --ssd_scratch_folder",
        program="volume_gray_scale_adjustment",
        args=base + ["--ctf_type", "apply", "--batch_size", "32", "--threads", "2",
                     "--ssd_scratch_folder", out("adjust_scratch_cache"),
                     "--output_path", out("adjust_scratch")],
        expect_files=adjusted("adjust_scratch"),
        timeout=600))

    return scn


# Every scenario is a single streaming pass over 96 particles: nothing here is slow.
SLOW = set()
