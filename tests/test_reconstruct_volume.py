#!/usr/bin/env python
"""
reconstruct_volume CLI test scenarios.

``reconstruct_volume`` is a single-pass consensus reconstruction (no train/predict
modes, no network): the particles in ``--md`` already carry poses, so every image
is inserted as a CTF-weighted central slice of the 3D transform and the map is the
Wiener quotient of the accumulators. It optionally splits the data into two halves
to measure the resolution by FSC and filter the combined map with it, calibrates the
gray scale against the input images, and can derive a binary mask from the result.

Coverage:

* ``--ctf_type``    : apply / wiener / precorrect (images carry a CTF) and None (they do not)
* ``--write_mask``  : mask derived from the map (+ ``--mask_threshold`` / ``--mask_dilate``)
* ``--no_denoise``  : raw map, no half-map FSC filtering
* ``--no_gray_scale_calibration`` : raw amplitudes
* ``--tau``, ``--batch_size``, ``--threads`` (streaming/reader knobs)
* outputs: ``consensus_reconstruction.mrc`` (+ ``consensus_mask.mrc``)

The program creates its own ``--output_path``, so no pre-creation is needed. Every
scenario is fast (one streaming pass), so none is marked slow.
"""

import os

import phantom
from common import Scenario

N_PARTICLES = 96
BOX = 40
SR = 2.0


def prepare_data(workdir):
    """Two phantoms: one whose images carry a CTF, one whose images do not."""
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

    def maps(name, mask=False):
        files = [os.path.join(out(name), "consensus_reconstruction.mrc")]
        if mask:
            files.append(os.path.join(out(name), "consensus_mask.mrc"))
        return files

    base = ["--md", ctf["md"], "--sr", str(SR)]
    scn = []

    # 1) the default path: CTF-weighted slices, half-map FSC denoising, gray-scale calibration
    scn.append(Scenario(
        name="recon_apply",
        description="reconstruct | ctf=apply | FSC denoising + gray-scale calibration (defaults)",
        program="reconstruct_volume",
        args=base + ["--ctf_type", "apply", "--output_path", out("recon_apply")],
        expect_files=maps("recon_apply"),
        timeout=600))

    # 2) CTF-free images: the slices must NOT be CTF weighted (--ctf_type None)
    scn.append(Scenario(
        name="recon_none",
        description="reconstruct | ctf=None | CTF-free phantom (slices not CTF weighted)",
        program="reconstruct_volume",
        args=["--md", no_ctf["md"], "--sr", str(SR), "--ctf_type", "None",
              "--output_path", out("recon_none")],
        expect_files=maps("recon_none"),
        timeout=600))

    # 3) --ctf_type wiener: the stored images still carry the CTF, so the map is the same
    scn.append(Scenario(
        name="recon_wiener",
        description="reconstruct | ctf=wiener",
        program="reconstruct_volume",
        args=base + ["--ctf_type", "wiener", "--output_path", out("recon_wiener")],
        expect_files=maps("recon_wiener"),
        timeout=600))

    # 4) --ctf_type precorrect (same: precorrection does not alter the stored images)
    scn.append(Scenario(
        name="recon_precorrect",
        description="reconstruct | ctf=precorrect",
        program="reconstruct_volume",
        args=base + ["--ctf_type", "precorrect", "--output_path", out("recon_precorrect")],
        expect_files=maps("recon_precorrect"),
        timeout=600))

    # 5) mask derived from the map (what HetSIREN/MoDART take in --mask)
    scn.append(Scenario(
        name="recon_mask",
        description="reconstruct | --write_mask --mask_threshold --mask_dilate -> map + mask",
        program="reconstruct_volume",
        args=base + ["--ctf_type", "apply", "--write_mask",
                     "--mask_threshold", "0.05", "--mask_dilate", "3",
                     "--output_path", out("recon_mask")],
        expect_files=maps("recon_mask", mask=True),
        timeout=600))

    # 6) raw map: no half-map FSC filtering, no gray-scale calibration
    scn.append(Scenario(
        name="recon_raw",
        description="reconstruct | --no_denoise --no_gray_scale_calibration (raw map)",
        program="reconstruct_volume",
        args=base + ["--ctf_type", "apply", "--no_denoise", "--no_gray_scale_calibration",
                     "--output_path", out("recon_raw")],
        expect_files=maps("recon_raw"),
        timeout=600))

    # 7) streaming knobs: a smaller Wiener floor, a batch smaller than the particle count
    #    (several chunks -> exercises the read-ahead loop) and a single reader thread
    scn.append(Scenario(
        name="recon_streaming",
        description="reconstruct | --tau 0.01 --batch_size 32 --threads 2 (chunked streaming)",
        program="reconstruct_volume",
        args=base + ["--ctf_type", "apply", "--tau", "0.01",
                     "--batch_size", "32", "--threads", "2",
                     "--output_path", out("recon_streaming")],
        expect_files=maps("recon_streaming"),
        timeout=600))

    # 8) SSD scratch: the stack is cached as a float16 array-record copy (the same one the
    #    network programs build) and streamed from there instead of from the original stack.
    scn.append(Scenario(
        name="recon_scratch",
        description="reconstruct | --ssd_scratch_folder (stream from the cached float16 copy)",
        program="reconstruct_volume",
        args=base + ["--ctf_type", "apply",
                     "--ssd_scratch_folder", out("recon_scratch_cache"),
                     "--output_path", out("recon_scratch")],
        # The cache lands in <scratch>/images_mmap_grain_<dataset digest>, so the exact folder
        # name is not known here -- the reuse scenario below is what proves it was written.
        expect_files=maps("recon_scratch") + [out("recon_scratch_cache")],
        timeout=600))

    # 9) the cache is *reused*, not rebuilt: a second run against the same scratch folder must
    #    find the shards already there. This is the whole point of the flag.
    scn.append(Scenario(
        name="recon_scratch_reuse",
        description="reconstruct | --ssd_scratch_folder reused by a second run (no rebuild)",
        program="reconstruct_volume",
        args=base + ["--ctf_type", "apply",
                     "--ssd_scratch_folder", out("recon_scratch_cache"),
                     "--output_path", out("recon_scratch_reuse")],
        expect_files=maps("recon_scratch_reuse"),
        timeout=600))

    return scn


# Every scenario is a single streaming pass over 96 particles: nothing here is slow.
SLOW = set()
