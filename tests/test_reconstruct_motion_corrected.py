#!/usr/bin/env python
"""
reconstruct_motion_corrected CLI test scenarios.

``reconstruct_motion_corrected`` is the non-iterative alternative to ``modart``: a single
streaming pass of CTF-weighted Fourier gridding with each particle's modelled motion removed
from its image first. It has no train/predict modes and no free parameters -- it reads the
images + alignments from ``--md`` and a trained mass-transport HetSIREN from
``--motion_correction``, and writes the map plus its two half maps.

Coverage:

* ``--motion_correction``  : required; must be a HetSIREN trained with --transport_mass
* ``--ctf_type``           : apply and None
* ``--also_consensus``     : the uncorrected control map, reconstructed from the same particles
* ``--write_mask``         : mask derived from the reconstructed map
* ``--field_batch``        : the VRAM-sensitive knob for the field decode
* half maps + FSC          : always produced, so the result is measurable
* the ``--ssd_scratch_folder`` cached-read path
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
    return {"ctf": ctf}


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]

    def out(name):
        return os.path.join(runs, name)

    def maps(name, extra=()):
        base = ["motion_corrected_reconstruction.mrc",
                "motion_corrected_half1.mrc", "motion_corrected_half2.mrc"]
        return [os.path.join(out(name), f) for f in list(base) + list(extra)]

    hetsiren_dir = os.path.join(out("setup_hetsiren_transport"), "HetSIREN")
    scn = []

    # --- setup: a HetSIREN WITH mass transport, which is what supplies the field ---
    scn.append(Scenario(
        name="setup_hetsiren_transport",
        description="[setup] train HetSIREN --transport_mass (supplies the motion field)",
        program="hetsiren",
        args=["--md", ctf["md"], "--sr", str(SR), "--ctf_type", "apply",
              "--mode", "train", "--epochs", "6", "--batch_size", "8", "--lat_dim", "6",
              "--transport_mass", "--load_images_to_ram",
              "--output_path", out("setup_hetsiren_transport")],
        expect_files=[hetsiren_dir],
        timeout=1200))

    base = ["--md", ctf["md"], "--sr", str(SR), "--motion_correction", hetsiren_dir,
            "--batch_size", "32", "--field_batch", "8", "--threads", "2"]

    # 1) basic motion-corrected reconstruction | ctf=apply
    scn.append(Scenario(
        name="mcrecon_apply",
        description="motion-corrected reconstruction | ctf=apply",
        program="reconstruct_motion_corrected",
        args=base + ["--ctf_type", "apply", "--output_path", out("mcrecon_apply")],
        expect_files=maps("mcrecon_apply"),
        timeout=900))

    # 2) with the uncorrected control map -- the comparison that makes the result meaningful
    scn.append(Scenario(
        name="mcrecon_control",
        description="motion-corrected | --also_consensus (uncorrected control) | --write_mask",
        program="reconstruct_motion_corrected",
        args=base + ["--ctf_type", "apply", "--also_consensus", "--write_mask",
                     "--output_path", out("mcrecon_control")],
        expect_files=maps("mcrecon_control",
                          extra=("consensus_reconstruction.mrc", "reconstruction_mask.mrc")),
        timeout=1200))

    # 3) ctf=None + no denoising / no gray-scale calibration (raw amplitudes)
    scn.append(Scenario(
        name="mcrecon_none_raw",
        description="motion-corrected | ctf=None | --no_denoise --no_gray_scale_calibration",
        program="reconstruct_motion_corrected",
        args=base + ["--ctf_type", "None", "--no_denoise", "--no_gray_scale_calibration",
                     "--output_path", out("mcrecon_none_raw")],
        expect_files=maps("mcrecon_none_raw"),
        timeout=900))

    # 3b) the deformed-backprojection mode: the correction is applied in 3D, per depth,
    # instead of as a single-valued 2D image warp. Paired with --also_consensus so the run
    # also produces the control the two modes are meant to be compared against.
    scn.append(Scenario(
        name="mcrecon_backprojection",
        description="motion-corrected | --correction backprojection (3D deformed rays)",
        program="reconstruct_motion_corrected",
        args=base + ["--ctf_type", "apply", "--correction", "backprojection",
                     "--also_consensus",
                     "--output_path", out("mcrecon_backprojection")],
        expect_files=maps("mcrecon_backprojection",
                          extra=("consensus_reconstruction.mrc",)),
        timeout=1800))

    # 3c) mass-conserving resampling, on the mode where the Jacobian is most likely to bite
    scn.append(Scenario(
        name="mcrecon_jacobian",
        description="motion-corrected | --correction backprojection --jacobian (mass-conserving)",
        program="reconstruct_motion_corrected",
        args=base + ["--ctf_type", "apply", "--correction", "backprojection", "--jacobian",
                     "--output_path", out("mcrecon_jacobian")],
        expect_files=maps("mcrecon_jacobian"),
        timeout=1800))

    # 3d) the same correction in the cheap 2D mode, with the Jacobian on
    scn.append(Scenario(
        name="mcrecon_image_jacobian",
        description="motion-corrected | --correction image --jacobian",
        program="reconstruct_motion_corrected",
        args=base + ["--ctf_type", "apply", "--correction", "image", "--jacobian",
                     "--output_path", out("mcrecon_image_jacobian")],
        expect_files=maps("mcrecon_image_jacobian"),
        timeout=900))

    # 4) cached-read path through the scratch folder
    scn.append(Scenario(
        name="mcrecon_scratch",
        description="motion-corrected | --ssd_scratch_folder (cached float16 read path)",
        program="reconstruct_motion_corrected",
        args=base + ["--ctf_type", "apply",
                     "--ssd_scratch_folder", out("mcrecon_scratch_cache"),
                     "--output_path", out("mcrecon_scratch")],
        expect_files=maps("mcrecon_scratch"),
        timeout=900))

    return scn


# The HetSIREN setup is the expensive part; the reconstructions themselves are single-pass.
SLOW = {"setup_hetsiren_transport", "mcrecon_control",
        "mcrecon_backprojection", "mcrecon_jacobian"}
