#!/usr/bin/env python
"""
preprocess_volumes CLI test scenarios.

``preprocess_volumes`` is the companion of ``preprocess_particles``: it crops and/or
Fourier-resizes reference maps and masks so they match a preprocessed particle set.
It runs on the CPU (a multi-threaded ``scipy.fft`` beats the GPU's setup cost and
memory cliff on the boxes volumes come in), takes several volumes per call, and has
no network and no modes.

The phantom supplies a *filled* reference volume and its mask, so gray-level and
centre-of-mass preservation are both measurable on real density rather than on a
sparse point splat.

Coverage:

* ``--crop_box_size``  : real-space centred crop (sampling rate untouched)
* ``--new_box_size``   : Fourier resize, down (48->24, 32->16) and up (48->64)
* both together        : the documented crop-then-resize order
* ``--vol``            : several volumes in one call (a map *and* its mask), and the
                         duplicate-stem naming that keeps every output on one glob
* ``--sr``             : given explicitly, and read from the volume header when omitted
* ``--num_workers``    : CPU FFT threads (must not change the result)
* outputs: ``<stem>_preprocessed.mrc``

As for the particle program, the runner's "exit 0 + files exist" criterion cannot see
whether the density survived, so a final scenario runs ``preprocess_checks.py`` to
verify the geometry, the sampling rate, gray levels, and the output naming that the
GUI's single-volume connector depends on. A second one checks the CLI's guards.
"""

import os

import numpy as np

import phantom
from common import Scenario

N_PARTICLES = 16   # only the volume/mask are used; keep the projection step cheap
BOX = 48
SR = 2.0

_CHECKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess_checks.py")


def prepare_data(workdir):
    """The phantom's reference volume + mask, plus decoys for the guards."""
    data_dir = os.path.join(workdir, "data")
    data = phantom.write_dataset(
        os.path.join(data_dir, "vol"),
        phantom.generate_dataset(N_PARTICLES, BOX, SR, apply_ctf=True, seed=1))

    import mrcfile
    from xmipp_metadata.image_handler import ImageHandler

    # A non-cubic volume: --new_box_size must refuse it rather than silently
    # resample a rectangular box into a cube.
    non_cubic = os.path.join(data_dir, "vol", "non_cubic.mrc")
    ImageHandler().write(np.zeros((BOX // 2, BOX, BOX), dtype=np.float32), non_cubic, sr=SR, overwrite=True)

    # A *cubic* particle stack carrying the MRC image-stack header flag — exactly what
    # preprocess_particles emits, and the shape a plain ndim/cubic check cannot catch.
    # Wiring such a run into a --vol input is an easy mistake, so it must be refused.
    flagged_stack = os.path.join(data_dir, "vol", "flagged_stack.mrcs")
    with mrcfile.new(flagged_stack, overwrite=True) as mrc:
        mrc.set_data(np.zeros((BOX, BOX, BOX), dtype=np.float32))
        mrc.set_image_stack()
        mrc.voxel_size = SR

    data["non_cubic"] = non_cubic
    data["flagged_stack"] = flagged_stack
    return data


def data_checks(workdir):
    """The reference volume must be a filled, centred density (not a sparse splat)."""
    from xmipp_metadata.image_handler import ImageHandler

    vol = np.squeeze(ImageHandler(os.path.join(workdir, "data", "vol", "particles_reference.mrc")).getData())
    occupancy = float((vol > 0.04 * vol.max()).mean())
    return [
        ("volume_is_cubic", len(set(vol.shape)) == 1 and vol.shape[0] == BOX, f"{vol.shape}"),
        ("volume_is_filled", 0.01 < occupancy < 0.9, f"{occupancy:.1%} of voxels above threshold"),
        ("mean_density_is_positive", float(vol.mean()) > 1e-6,
         f"mean {vol.mean():.5f} — gray-level preservation is measurable"),
    ]


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    vol, mask = data["vol"], data["mask"]
    stem = os.path.splitext(os.path.basename(vol))[0]

    def out(name):
        return os.path.join(runs, name)

    def produced(name, *stems):
        return [os.path.join(out(name), f"{s}_preprocessed.mrc") for s in (stems or (stem,))]

    scn = []

    def add(name, description, args, expect=None, timeout=600):
        scn.append(Scenario(name=name, description=description, program="preprocess_volumes",
                            args=args + ["--output_path", out(name)],
                            expect_files=expect or produced(name), timeout=timeout))

    base = ["--vol", vol, "--sr", str(SR)]

    # --- geometry ---------------------------------------------------------
    add("resize", "resize | Fourier 48 -> 24 (sr x2)", base + ["--new_box_size", "24"])
    add("crop", "crop | real-space 48 -> 32 (sr unchanged)", base + ["--crop_box_size", "32"])
    add("crop_resize", "crop+resize | 48 -> crop 32 -> resize 16 (documented order)",
        base + ["--crop_box_size", "32", "--new_box_size", "16"])
    add("upsample", "resize | Fourier up 48 -> 64 (sr x0.75)", base + ["--new_box_size", "64"])

    # --- multiple volumes -------------------------------------------------
    add("pair", "multi | a map and its mask resized together, in one call",
        ["--vol", vol, mask, "--sr", str(SR), "--new_box_size", "24"],
        expect=produced("pair", stem, os.path.splitext(os.path.basename(mask))[0]))
    add("dup", "multi | the same stem twice -> disambiguated, still on one glob",
        ["--vol", vol, vol, "--sr", str(SR), "--new_box_size", "24"],
        expect=[os.path.join(out("dup"), f"{stem}_preprocessed.mrc"),
                os.path.join(out("dup"), f"{stem}_1_preprocessed.mrc")])

    # --- other options ----------------------------------------------------
    add("header_sr", "sr | --sr omitted, read from the volume header",
        ["--vol", vol, "--new_box_size", "24"])
    add("workers", "perf | --num_workers 2 (CPU FFT threads)",
        base + ["--new_box_size", "24", "--num_workers", "2"])

    # --- semantic verification --------------------------------------------
    scn.append(Scenario(
        name="verify_outputs",
        description="verify | geometry, sampling rate, density, NumPy oracle, naming",
        program="preprocess_volumes",
        script=_CHECKS,
        args=["--check", "volumes", "--runs", runs, "--box", str(BOX), "--sr", str(SR),
              "--source-vol", vol],
        timeout=900))

    scn.append(Scenario(
        name="verify_guards",
        description="verify | the CLI refuses a no-op, a non-cubic resize and a particle stack",
        program="preprocess_volumes",
        script=_CHECKS,
        args=["--check", "volumes_guards", "--runs", out("guards"), "--sr", str(SR),
              "--source-vol", vol, "--non-cubic", data["non_cubic"],
              "--stack", data["flagged_stack"]],
        pre=(lambda p=out("guards"): os.makedirs(p, exist_ok=True)),
        timeout=900))

    return scn


SLOW = set()
