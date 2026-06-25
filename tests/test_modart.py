#!/usr/bin/env python
"""
MoDART CLI test scenarios.

MoDART is an ART-style reconstruction (no train/predict modes): it reconstructs a
single map (or two half maps) from the images + alignments in ``--md``, optionally
refining a reference volume (``--vol``) and/or correcting motion blur using a
trained HetSIREN/Zernike3Deep (``--motion_correction``). It runs until an
early-stopping criterion fires and writes ``modart_map.mrc`` (and half maps).

Coverage:

* ``--ctf_type``         : None, apply, wiener, precorrect
* ``--symmetry_group``   : c1 (default) and c2
* ``--reconstruct_halves``: single map vs two half maps
* ``--vol`` / ``--mask`` : reference-volume refinement (trains a VolumeAdjustment)
* ``--motion_correction``: reconstruction with motion correction from a trained network
* ``--load_images_to_ram`` (on) and the mmap path (off) + ``--ssd_scratch_folder``
* ``--batch_size``
* outputs: ``modart_map.mrc`` (+ ``modart_first/second_half.mrc``)

MoDART creates its own ``--output_path`` (via the metrics writer), so no
pre-creation is needed.
"""

import os

import phantom
from common import Scenario

N_PARTICLES = 96
BOX = 40
SR = 2.0


def prepare_data(workdir):
    """A CTF phantom + its reference volume / mask (used by the --vol scenario)."""
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

    def maps(name, halves=False):
        if halves:
            return [os.path.join(out(name), f) for f in
                    ("modart_first_half.mrc", "modart_second_half.mrc", "modart_map.mrc")]
        return [os.path.join(out(name), "modart_map.mrc")]

    base = ["--md", ctf["md"], "--sr", str(SR), "--batch_size", "8", "--load_images_to_ram"]
    scn = []

    # 1) basic reconstruction | ctf=apply | default (circular) mask | RAM
    scn.append(Scenario(
        name="recon_apply",
        description="reconstruct | ctf=apply | default mask | RAM",
        program="modart",
        args=base + ["--ctf_type", "apply", "--output_path", out("recon_apply")],
        expect_files=maps("recon_apply"),
        timeout=600))

    # 2) ctf=None + c2 symmetry
    scn.append(Scenario(
        name="recon_none_c2",
        description="reconstruct | ctf=None | --symmetry_group c2",
        program="modart",
        args=base + ["--ctf_type", "None", "--symmetry_group", "c2",
                     "--output_path", out("recon_none_c2")],
        expect_files=maps("recon_none_c2"),
        timeout=600))

    # 3) ctf=wiener
    scn.append(Scenario(
        name="recon_wiener",
        description="reconstruct | ctf=wiener",
        program="modart",
        args=base + ["--ctf_type", "wiener", "--output_path", out("recon_wiener")],
        expect_files=maps("recon_wiener"),
        timeout=600))

    # 4) ctf=precorrect
    scn.append(Scenario(
        name="recon_precorrect",
        description="reconstruct | ctf=precorrect",
        program="modart",
        args=base + ["--ctf_type", "precorrect", "--output_path", out("recon_precorrect")],
        expect_files=maps("recon_precorrect"),
        timeout=600))

    # 5) reconstruct_halves -> two half maps + combined map (guards the data_loader fix)
    scn.append(Scenario(
        name="recon_halves",
        description="reconstruct | --reconstruct_halves -> first/second half + map",
        program="modart",
        args=base + ["--ctf_type", "apply", "--reconstruct_halves",
                     "--output_path", out("recon_halves")],
        expect_files=maps("recon_halves", halves=True),
        timeout=600))

    # 6) memory-mapped path (no RAM) + ssd scratch
    scn.append(Scenario(
        name="recon_mmap",
        description="reconstruct | ctf=apply | mmap + ssd_scratch",
        program="modart",
        args=["--md", ctf["md"], "--sr", str(SR), "--batch_size", "8",
              "--ctf_type", "apply",
              "--ssd_scratch_folder", out("recon_mmap_scratch"),
              "--output_path", out("recon_mmap")],
        expect_files=maps("recon_mmap"),
        timeout=600))

    # 7) reference-volume refinement (trains a VolumeAdjustment first)
    scn.append(Scenario(
        name="recon_vol",
        description="reconstruct | --vol --mask (refinement + VolumeAdjustment)",
        program="modart",
        args=base + ["--ctf_type", "apply", "--vol", ctf["vol"], "--mask", ctf["mask"],
                     "--output_path", out("recon_vol")],
        expect_files=maps("recon_vol")
                     + [os.path.join(out("recon_vol"), "VolumeAdjustment")],
        timeout=1200))

    # --- setup for motion correction: a small HetSIREN (provides decode_field) ---
    scn.append(Scenario(
        name="setup_hetsiren",
        description="[setup] train HetSIREN (network used for --motion_correction)",
        program="hetsiren",
        args=["--md", ctf["md"], "--sr", str(SR), "--ctf_type", "apply",
              "--mode", "train", "--epochs", "6", "--batch_size", "8", "--lat_dim", "6",
              "--load_images_to_ram", "--output_path", out("setup_hetsiren")],
        expect_files=[os.path.join(out("setup_hetsiren"), "HetSIREN")],
        timeout=900))

    # 8) reconstruction with motion correction from the trained network
    scn.append(Scenario(
        name="recon_motion",
        description="reconstruct | --motion_correction <HetSIREN>",
        program="modart",
        args=base + ["--ctf_type", "apply",
                     "--motion_correction", os.path.join(out("setup_hetsiren"), "HetSIREN"),
                     "--output_path", out("recon_motion")],
        expect_files=maps("recon_motion"),
        timeout=600))

    return scn


# Slow scenarios (extra training): the reference-volume refinement and the
# HetSIREN setup + motion-correction reconstruction. Skipped under --quick.
SLOW = {"recon_vol", "setup_hetsiren", "recon_motion"}
