#!/usr/bin/env python
"""
preprocess_particles CLI test scenarios.

``preprocess_particles`` rewrites a particle stack into a smaller, optionally
CTF-corrected one and updates the metadata that goes with it. It has no
train/predict modes and no network: one pass over the stack, streamed from disk to
the device and back.

The phantom is the standard CTF dataset (48-box, every particle with its own pose,
in-plane shift and CTF), because the shifts and CTF columns are exactly what the
program has to transform (shifts) or carry over untouched (CTF, angles).

Coverage:

* ``--crop_box_size``   : real-space centred crop (and zero-pad when larger)
* ``--new_box_size``    : Fourier resize, both down (48->24, 32->16) and up (48->64)
* both together         : the documented crop-then-resize order
* ``--ctf_correction``  : none / wiener / phase_flip, plus a CTF-only run (no resize)
* ``--wiener_epsilon``  : fixed regularizer instead of the adaptive one
* ``--batch_size``      : fixed, and ``auto`` (analytical GPU sizing)
* ``--device``          : cpu (the auto/gpu path is the default everywhere else)
* ``--num_read_workers`` / ``--num_write_workers``
* ``--relative_image_paths`` and the absolute-path default
* Relion ``.star`` input -> ``.star`` output (the extension-mirroring branch)
* outputs: ``preprocessed_particles.mrcs`` + ``preprocessed_particles.{xmd,star}``

Because the runner only checks "exit 0 + expected files", two extra scenarios run
``preprocess_checks.py`` afterwards to verify what actually matters: the sampling
rate, the rescaled shifts, the pixel data against an independent NumPy oracle, and
that the CLI refuses invalid geometry. They run last and depend on the runs above.
"""

import os

import numpy as np

import phantom
from common import Scenario

N_PARTICLES = 64
BOX = 48          # even: hax evaluates CTFs on a zero-padded box
SR = 2.0

_CHECKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess_checks.py")


def prepare_data(workdir):
    """A CTF phantom, a ``.star`` copy of it, and a CTF-free metadata for the guards."""
    data_dir = os.path.join(workdir, "data")
    ctf = phantom.write_dataset(
        os.path.join(data_dir, "ctf"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True, seed=1))

    from xmipp_metadata.metadata import XmippMetaData

    # Same particles, written as Relion .star (absolute image paths survive the
    # round trip), to exercise the output-extension mirroring.
    star = os.path.join(data_dir, "ctf", "particles.star")
    XmippMetaData(ctf["md"]).write(star)

    # A metadata whose CTF columns are all zero: XmippMetaData back-fills missing
    # CTF labels with zeros, so this is what "no CTF information" really looks like
    # and it is what --ctf_correction must refuse.
    no_ctf = os.path.join(data_dir, "ctf", "particles_no_ctf.xmd")
    md = XmippMetaData(ctf["md"])
    for column in ("ctfDefocusU", "ctfDefocusV", "ctfDefocusAngle"):
        md.setMetaDataColumns(np.zeros(N_PARTICLES), column)
    md.write(no_ctf)

    ctf["star"] = star
    ctf["no_ctf"] = no_ctf
    return ctf


def data_checks(workdir):
    """The phantom must carry the two things this program transforms: shifts and CTF."""
    from xmipp_metadata.metadata import XmippMetaData

    md = XmippMetaData(os.path.join(workdir, "data", "ctf", "particles.xmd"))
    shifts = md.getMetaDataColumns(["shiftX", "shiftY"])
    defocus = md.getMetaDataColumns("ctfDefocusU")
    box = md.getMetaDataImage(0).shape[-1]
    return [
        ("shifts_are_nonzero", bool(np.abs(shifts).max() > 0.5),
         f"max |shift| {np.abs(shifts).max():.2f} px — shift rescaling is observable"),
        ("ctf_is_present", bool(defocus.min() > 0.0),
         f"defocusU in [{defocus.min():.0f}, {defocus.max():.0f}] A"),
        ("box_is_even", box % 2 == 0, f"box {box}"),
    ]


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    md, star, no_ctf = data["md"], data["star"], data["no_ctf"]

    def out(name):
        return os.path.join(runs, name)

    def outputs(name, ext="xmd"):
        return [os.path.join(out(name), f"preprocessed_particles.{s}") for s in ("mrcs", ext)]

    base = ["--md", md, "--sr", str(SR)]
    scn = []

    def add(name, description, args, ext="xmd", timeout=600):
        scn.append(Scenario(name=name, description=description, program="preprocess_particles",
                            args=args + ["--output_path", out(name)],
                            expect_files=outputs(name, ext), timeout=timeout))

    # --- geometry ---------------------------------------------------------
    add("resize", "resize | Fourier 48 -> 24 (sr x2, shifts x0.5)",
        base + ["--new_box_size", "24"])
    add("crop", "crop | real-space 48 -> 32 (sr and shifts unchanged)",
        base + ["--crop_box_size", "32"])
    add("crop_resize", "crop+resize | 48 -> crop 32 -> resize 16 (documented order)",
        base + ["--crop_box_size", "32", "--new_box_size", "16"])
    add("upsample", "resize | Fourier up 48 -> 64 (sr x0.75, shifts x4/3)",
        base + ["--new_box_size", "64"])
    add("pad", "crop | zero-pad 48 -> 64 (sr and shifts unchanged)",
        base + ["--crop_box_size", "64"])

    # --- CTF correction ---------------------------------------------------
    add("ctf_only", "ctf | phase_flip with no resize or crop",
        base + ["--ctf_correction", "phase_flip"])
    add("wiener", "ctf | wiener + resize 24 (adaptive spectral-SNR regularizer)",
        base + ["--new_box_size", "24", "--ctf_correction", "wiener"])
    add("phase_flip", "ctf | phase_flip + resize 24",
        base + ["--new_box_size", "24", "--ctf_correction", "phase_flip"])
    add("wiener_eps", "ctf | wiener + fixed --wiener_epsilon 0.1",
        base + ["--new_box_size", "24", "--ctf_correction", "wiener", "--wiener_epsilon", "0.1"])

    # --- performance knobs (must not change the pixels) --------------------
    add("batch_auto", "perf | --batch_size auto (analytical GPU sizing)",
        base + ["--new_box_size", "24", "--batch_size", "auto"])
    add("device_cpu", "perf | --device cpu",
        base + ["--new_box_size", "24", "--device", "cpu"])
    add("workers", "perf | small batch + 2 read / 1 write worker",
        base + ["--new_box_size", "24", "--batch_size", "8",
                "--num_read_workers", "2", "--num_write_workers", "1"])

    # --- output form ------------------------------------------------------
    add("relative", "output | --relative_image_paths (relocatable folder)",
        base + ["--new_box_size", "24", "--relative_image_paths"])
    add("star", "output | Relion .star in -> .star out",
        ["--md", star, "--sr", str(SR), "--new_box_size", "24"], ext="star")

    # --- semantic verification of everything produced above ----------------
    scn.append(Scenario(
        name="verify_outputs",
        description="verify | sampling rate, shift rescaling, NumPy oracle, invariance",
        program="preprocess_particles",
        script=_CHECKS,
        args=["--check", "particles", "--runs", runs, "--box", str(BOX), "--sr", str(SR),
              "--source-md", md],
        timeout=900))

    scn.append(Scenario(
        name="verify_guards",
        description="verify | the CLI refuses odd boxes, no-ops and CTF-less correction",
        program="preprocess_particles",
        script=_CHECKS,
        args=["--check", "particles_guards", "--runs", out("guards"), "--sr", str(SR),
              "--source-md", md, "--no-ctf-md", no_ctf],
        pre=(lambda p=out("guards"): os.makedirs(p, exist_ok=True)),
        timeout=900))

    return scn


# The verification scenarios depend on the runs above, so they are never skipped;
# nothing here is slow enough to warrant --quick pruning.
SLOW = set()
