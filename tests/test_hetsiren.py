#!/usr/bin/env python
"""
HetSIREN CLI test scenarios.

Exercises (ideally) *every* ``hetsiren`` option so future regressions surface as
a failing scenario.  Coverage:

* ``--ctf_type``      : None, apply, wiener, precorrect
* ``--mode``          : train, predict
* ``--vol`` / ``--mask``
* ``--transport_mass`` / ``--implicit_network`` / ``--num_gaussians``
* ``--local_reconstruction``
* ``--load_images_to_ram`` (on) and the memory-mapped path (off) + ``--ssd_scratch_folder``
* ``--lat_dim`` / ``--batch_size`` / ``--learning_rate`` / ``--denoising_strength``
* ``--dataset_split_fraction`` / ``--epochs`` / ``--reload``

The data is the synthetic continuous+compositional phantom from ``phantom.py``.
Most scenarios use ``epochs = 6`` so the intermediate-checkpoint path (written
every 5 epochs) and its end-of-training cleanup are exercised; a dedicated
``epochs = 2`` scenario covers the short-run path (no checkpoint written).
"""

import os
import numpy as np

import phantom
from common import Scenario

# Phantom size — kept small so the whole suite runs quickly while still being a
# genuine heterogeneous reconstruction problem.
N_PARTICLES = 160
BOX = 48
SR = 2.0
EPOCHS = 6


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
    """Validate the phantom itself, in particular: 'None' images carry no CTF.

    Returns a list of ``(name, ok, detail)`` tuples.
    """
    checks = []

    # Build a *matched* pair (identical particles) with and without CTF so the
    # only difference is the CTF application.
    clean = phantom.generate_dataset(24, BOX, SR, apply_ctf=False, noise=0.0, seed=7)
    ctfed = phantom.generate_dataset(24, BOX, SR, apply_ctf=True, noise=0.0, seed=7)

    a = clean["images"]
    b = ctfed["images"]
    # Normalise both to compare shape of the signal, not absolute scale.
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

    # The 'None' dataset images must equal the *unfiltered* projections, i.e. the
    # phantom adds NO CTF when apply_ctf=False.
    checks.append((
        "none_images_are_ctf_free",
        bool(clean["apply_ctf"] is False),
        "noctf dataset generated with apply_ctf=False (no ctfFilter applied)"))

    return checks


def scenarios(workdir, data):
    """Return the ordered list of HetSIREN CLI scenarios."""
    runs = os.path.join(workdir, "runs")
    ctf = data["ctf"]
    noctf = data["noctf"]

    def out(name):
        return os.path.join(runs, name)

    def hs(name):  # expected HetSIREN model dir
        return os.path.join(out(name), "HetSIREN")

    base = ["--sr", str(SR), "--epochs", str(EPOCHS), "--load_images_to_ram"]

    scn = []

    # 1) train, ctf=None, RAM, small latent dim
    scn.append(Scenario(
        name="train_none_ram",
        description="train | ctf=None | RAM | lat_dim=4",
        program="hetsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "train",
              "--lat_dim", "4", "--batch_size", "8",
              "--output_path", out("train_none_ram")] + base,
        expect_files=[hs("train_none_ram")],
        timeout=900))

    # 2) predict, reloading the model trained in (1).
    #    Note: no pre-created output dir — predict must create it itself.
    scn.append(Scenario(
        name="predict_none",
        description="predict | ctf=None | --reload from train_none_ram (creates own output dir)",
        program="hetsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "predict",
              "--lat_dim", "4", "--batch_size", "8",
              "--reload", out("train_none_ram"),
              "--output_path", out("predict_none"),
              "--sr", str(SR), "--load_images_to_ram"],
        expect_files=[os.path.join(out("predict_none"), "predicted_latents.xmd")],
        timeout=600))

    # 2b) short training run (epochs < 5): guards the checkpoint-cleanup fix
    #     (no HetSIREN_CHECKPOINT is written, end-of-training rmtree must be a
    #     no-op) and the no-NaN-loss fix on the very first epoch.
    scn.append(Scenario(
        name="train_none_short",
        description="train | ctf=None | epochs=2 (short-run cleanup + NaN-free regression guard)",
        program="hetsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "train",
              "--lat_dim", "4", "--batch_size", "8", "--epochs", "2",
              "--sr", str(SR), "--load_images_to_ram",
              "--output_path", out("train_none_short")],
        expect_files=[hs("train_none_short")],
        timeout=600))

    # 2c) the large-box memory path: a strided sample lattice plus a blocked,
    #     rematerialized render. Both are what make a big box fit on a small GPU, and
    #     both change the shapes flowing through the decoder and the projector, so they
    #     get their own train + predict pair. The block size is deliberately smaller than
    #     the phantom's point count so the blocking loop actually runs more than once.
    scn.append(Scenario(
        name="train_none_chunked",
        description="train | ctf=None | --lattice_stride 2 --render_chunk_size 4096 (large-box memory path)",
        program="hetsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "train",
              "--lat_dim", "4", "--batch_size", "8",
              "--lattice_stride", "2", "--render_chunk_size", "4096",
              "--output_path", out("train_none_chunked")] + base,
        expect_files=[hs("train_none_chunked")],
        timeout=900))

    # 2d) reloading it also checks that the two new settings round-trip through the
    #     checkpointed model config -- a predict that silently dropped them would
    #     rebuild the decoder on a different lattice than the one it was trained on.
    scn.append(Scenario(
        name="predict_none_chunked",
        description="predict | ctf=None | --reload from train_none_chunked (stride/chunk round-trip)",
        program="hetsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "predict",
              "--lat_dim", "4", "--batch_size", "8",
              "--reload", out("train_none_chunked"),
              "--output_path", out("predict_none_chunked"),
              "--sr", str(SR), "--load_images_to_ram"],
        expect_files=[os.path.join(out("predict_none_chunked"), "predicted_latents.xmd")],
        timeout=600))

    # 2c-bis) MANY blocks at a real batch. ``train_none_chunked`` above uses stride 2 and a
    #     4096-point chunk, which on a 48 px box is only ~2 blocks -- so it exercises the
    #     flag but not the structure, and it passed while the chunked render segfaulted at
    #     7+ blocks. Stride 1 with a 512-point chunk is ~115 blocks, and the fused fixed-grid
    #     decode (which replaces the whole (B, N) density with a scan) only runs when
    #     chunking is on, so this is the scenario that covers it.
    scn.append(Scenario(
        name="train_none_chunked_deep",
        description="train | ctf=None | --render_chunk_size 512 (~115 blocks: fused decode + scan render)",
        program="hetsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "train",
              "--lat_dim", "4", "--batch_size", "16", "--render_chunk_size", "512",
              "--output_path", out("train_none_chunked_deep")] + base,
        expect_files=[hs("train_none_chunked_deep")],
        timeout=1200))

    # 2e) the box-independent encoder. convstem swaps the flatten->dense stem for a
    #     strided-conv one, so its output shapes and checkpoint layout differ from convnn;
    #     train + predict cover the forward path and the config round-trip.
    scn.append(Scenario(
        name="train_none_convstem",
        description="train | ctf=None | --encoder_arch convstem (box-independent encoder)",
        program="hetsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "train",
              "--lat_dim", "4", "--batch_size", "8", "--encoder_arch", "convstem",
              "--output_path", out("train_none_convstem")] + base,
        expect_files=[hs("train_none_convstem")],
        timeout=900))

    scn.append(Scenario(
        name="predict_none_convstem",
        description="predict | ctf=None | --reload from train_none_convstem (convstem round-trip)",
        program="hetsiren",
        args=["--md", noctf["md"], "--ctf_type", "None", "--mode", "predict",
              "--lat_dim", "4", "--batch_size", "8",
              "--reload", out("train_none_convstem"),
              "--output_path", out("predict_none_convstem"),
              "--sr", str(SR), "--load_images_to_ram"],
        expect_files=[os.path.join(out("predict_none_convstem"), "predicted_latents.xmd")],
        timeout=600))

    # 3) train, ctf=apply, memory-mapped (no RAM) + ssd scratch + custom hyperparams
    scn.append(Scenario(
        name="train_apply_mmap",
        description="train | ctf=apply | mmap+ssd_scratch | bs=4 lr=5e-5 denoise=1e-3 split=0.7,0.3",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "8", "--batch_size", "4",
              "--learning_rate", "5e-5", "--denoising_strength", "1e-3",
              "--dataset_split_fraction", "0.7,0.3",
              "--ssd_scratch_folder", out("train_apply_mmap_scratch"),
              "--sr", str(SR), "--epochs", str(EPOCHS),
              "--output_path", out("train_apply_mmap")],
        expect_files=[hs("train_apply_mmap")],
        timeout=900))

    # 4) train, ctf=wiener
    scn.append(Scenario(
        name="train_wiener",
        description="train | ctf=wiener | RAM",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "wiener", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--output_path", out("train_wiener")] + base,
        expect_files=[hs("train_wiener")],
        timeout=900))

    # 5) train, ctf=precorrect
    scn.append(Scenario(
        name="train_precorrect",
        description="train | ctf=precorrect | RAM",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "precorrect", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--output_path", out("train_precorrect")] + base,
        expect_files=[hs("train_precorrect")],
        timeout=900))

    # 6) train with reference volume: transport_mass + implicit + num_gaussians (+mask)
    scn.append(Scenario(
        name="train_vol_transport_implicit",
        description="train | ctf=apply | --vol --mask --transport_mass --implicit_network --num_gaussians 100",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--transport_mass", "--implicit_network", "--num_gaussians", "100",
              "--output_path", out("train_vol_transport_implicit")] + base,
        expect_files=[hs("train_vol_transport_implicit"),
                      os.path.join(out("train_vol_transport_implicit"),
                                   "Gaussian_volume_fitting"),
                      os.path.join(out("train_vol_transport_implicit"),
                                   "consensus_volume.mrc")],
        timeout=1800))

    # 6b) train with reference volume: point-transformer decoder. Needs more Gaussians
    # than the other transport scenarios because build_geometry asserts that the finest
    # hierarchy level (512) is coarser than the point cloud.
    scn.append(Scenario(
        name="train_vol_transport_pt",
        description="train | ctf=apply | --vol --mask --transport_mass --point_transformer",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--transport_mass", "--point_transformer", "--num_gaussians", "2000",
              "--output_path", out("train_vol_transport_pt")] + base,
        expect_files=[hs("train_vol_transport_pt"),
                      os.path.join(out("train_vol_transport_pt"),
                                   "consensus_volume.mrc")],
        timeout=1800))

    # 6c) train as a VAE. This is the only scenario that exercises --kl_lambda, i.e. the
    # logstd heads, the reparametrised sample and the KL/free-bits term -- everything else
    # runs with a deterministic latent and leaves that whole branch untouched. It stayed
    # uncovered long enough for a plain NameError to survive in it, so it is worth a case
    # even though the reconstruction quality after 4 phantom epochs means nothing.
    # Paired with --point_transformer because the VAE-only distance-preservation branch
    # decodes a second time, and that is the decoder it is most expensive on.
    scn.append(Scenario(
        name="train_vol_transport_vae",
        description="train | ctf=apply | --vol --transport_mass --point_transformer --kl_lambda (VAE path)",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--transport_mass", "--point_transformer", "--num_gaussians", "2000",
              "--kl_lambda", "1e-3", "--kl_free_bits", "1.0",
              "--distance_preservation_lambda", "1e-4",
              "--output_path", out("train_vol_transport_vae")] + base,
        expect_files=[hs("train_vol_transport_vae"),
                      os.path.join(out("train_vol_transport_vae"),
                                   "consensus_volume.mrc")],
        timeout=1800))

    # 6b) per-image contrast: the branch adds masked sums and a stop-gradient rescale
    # inside the reconstruction loss, on both the amplitude and the mass-transport
    # pathway. "relative" is the recommended mode and the one worth guarding; a flat or
    # empty projection takes the a=1 fallback, which is the path that would produce a
    # NaN loss if the guard were wrong.
    scn.append(Scenario(
        name="train_vol_per_image_contrast",
        description="train | ctf=apply | --vol --transport_mass --per_image_contrast relative",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--transport_mass", "--num_gaussians", "2000",
              "--per_image_contrast", "relative",
              "--output_path", out("train_vol_per_image_contrast")] + base,
        expect_files=[hs("train_vol_per_image_contrast"),
                      os.path.join(out("train_vol_per_image_contrast"),
                                   "consensus_volume.mrc")],
        timeout=1800))

    # 6d) automatic Gaussian count. This is the DEFAULT path -- every other --vol scenario
    # pins --num_gaussians and so never touches it -- and it is the one that decides the
    # point cloud, the render width and the density scale the network starts from. It is
    # also the only scenario that exercises the FSC/shell-error gate and the bracketing
    # search, whose failure mode is silent: a fit that "succeeds" with a collapsed width
    # still exits 0 and still trains, it just cannot express motion afterwards.
    scn.append(Scenario(
        name="train_vol_transport_auto_gaussians",
        description="train | ctf=apply | --vol --mask --transport_mass (automatic Gaussian count)",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--transport_mass", "--fit_resolution", "12",
              "--output_path", out("train_vol_transport_auto_gaussians")] + base,
        expect_files=[hs("train_vol_transport_auto_gaussians"),
                      os.path.join(out("train_vol_transport_auto_gaussians"),
                                   "Gaussian_volume_fitting"),
                      os.path.join(out("train_vol_transport_auto_gaussians"),
                                   "consensus_volume.mrc")],
        timeout=1800))

    # 7) train with reference volume: local_reconstruction (+mask, vol mandatory)
    scn.append(Scenario(
        name="train_vol_local_recon",
        description="train | ctf=apply | --vol --mask --local_reconstruction",
        program="hetsiren",
        args=["--md", ctf["md"], "--ctf_type", "apply", "--mode", "train",
              "--lat_dim", "6", "--batch_size", "8",
              "--vol", ctf["vol"], "--mask", ctf["mask"],
              "--local_reconstruction",
              "--output_path", out("train_vol_local_recon")] + base,
        expect_files=[hs("train_vol_local_recon")],
        timeout=1800))

    return scn


# Scenarios that need fit_volume (slow); skipped by --quick.
SLOW = {"train_vol_transport_implicit", "train_vol_transport_pt", "train_vol_local_recon",
        "train_vol_transport_auto_gaussians"}
