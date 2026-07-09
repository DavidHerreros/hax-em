# Hax CLI test-suite

Fast, **data-free** smoke tests for the programs exposed by `hax_project_manager`.
They generate a synthetic CryoEM phantom on the fly and drive each program
through its **real CLI entry point**, so accidental breakages introduced by
future changes surface as a failing scenario.

The suite lives entirely in this `tests/` folder and **does not modify or import
any Hax program module** — every program is executed exactly as a user would run
it from the command line.

## How to run

Use the Python of an environment where `hax` imports and runs (flax 0.12,
jax 0.9, `xmipp_metadata`, working `cuml`):

```bash
# from the repo root
python tests/run_tests.py --gpu 0 hetsiren            # full hetsiren suite
python tests/run_tests.py --gpu 0 --quick hetsiren    # skip the slow fit_volume cases
python tests/run_tests.py --gpu 0 all                 # every implemented program
python tests/run_tests.py --gpu 0 --keep hetsiren     # keep the work dir for inspection
```

## What it does

1. **Phantom generation** (`phantom.py`) — builds a point-cloud molecule whose
   conformation is controlled by a scalar `t ∈ [0, 1]`:
   * **continuous** heterogeneity: a blob slides along X with `t`;
   * **compositional** heterogeneity: a second blob's occupancy grows with `t`.
   Every particle gets a *different* random pose (Xmipp Euler angles), in-plane
   shift and CTF (defocus/astigmatism). Projections use the same geometric
   convention as Hax (`euler_matrix_batch` + the PhysDecoder splat convention),
   and the CTF is applied with Hax's own `computeCTF`/`ctfFilter`.
   * `apply_ctf=True` → CTF-corrupted images (for `apply`/`wiener`/`precorrect`).
   * `apply_ctf=False` → **CTF-free** images (for the `None` mode).
   It also writes a *filled* reference volume + mask (used by `--vol`,
   `--transport_mass`, `--local_reconstruction`).

2. **Data integrity checks** — e.g. confirm that the `None`-mode images carry no
   CTF (matched clean-vs-CTF pair must differ; the `None` dataset is generated
   with `apply_ctf=False`).

   The `compositional` switch can be turned **off** to get *continuous-only*
   (motion without mass change) heterogeneity, as required by deformation-only
   methods such as Zernike3Deep.

3. **CLI scenarios** — each runs a real program command (`hetsiren`,
   `zernike3deep`, …) and passes only if it exits 0 **and** produces the expected
   outputs. A `NaN/Inf` loss is treated as a failure.

### HetSIREN coverage (`test_hetsiren.py`)

| Option / axis | Covered by |
|---|---|
| `--ctf_type None / apply / wiener / precorrect` | `train_none_ram` / `train_apply_mmap` / `train_wiener` / `train_precorrect` |
| `--mode train / predict` | `train_*` / `predict_none` |
| `--vol`, `--mask` | `train_vol_transport_implicit`, `train_vol_local_recon` |
| `--transport_mass`, `--implicit_network`, `--num_gaussians` | `train_vol_transport_implicit` |
| `--local_reconstruction` | `train_vol_local_recon` |
| `--load_images_to_ram` (on) / mmap (off) + `--ssd_scratch_folder` | most scenarios / `train_apply_mmap` |
| `--lat_dim`, `--batch_size`, `--learning_rate`, `--denoising_strength` | varied across scenarios |
| `--dataset_split_fraction`, `--epochs`, `--reload` | `train_apply_mmap` / all (+ `train_none_short`) / `predict_none` |

### Zernike3Deep coverage (`test_zernike3deep.py`)

Zernike3Deep models variability as a **continuous deformation** (a Zernike3D
displacement field), so its phantom is generated with `compositional=False`
(motion only). `--vol` is mandatory, so every `train` runs the Gaussian
`fit_volume`; scenarios use `--num_gaussians 100` to keep that fast, plus one
scenario exercising the default densify fit.

| Option / axis | Covered by |
|---|---|
| `--ctf_type None / apply / wiener / precorrect` | `train_none` / `train_apply_mmap` / `train_wiener` / `train_precorrect` |
| `--mode train / predict` | `train_*` / `predict_none` |
| `--vol` (required) / `--mask` provided vs auto-generated | all / `train_apply_automask` |
| `--L1`, `--L2` (incl. defaults) | `train_none`(3,3), `train_apply_mmap`(5,5), `train_default_fit`(7,7) |
| `--num_gaussians` set vs default densify fit | most / `train_default_fit` |
| `--lat_dim`, `--batch_size`, `--learning_rate`, `--dataset_split_fraction` | varied across scenarios |
| `--load_images_to_ram` (on) / mmap (off) + `--ssd_scratch_folder` | most scenarios / `train_apply_mmap` |
| `--epochs`, `--reload` | all (+ `train_none_short`) / `predict_none` |

Data checks confirm the phantom is genuinely continuous-only: total density is
conserved across conformations while atom positions change (and the
`compositional=True` path *does* change mass, so the switch is effective).

### estimate_latent_covariances coverage (`test_estimate_latent_covariances.py`)

`estimate_latent_covariances` consumes a *trained* network and estimates a latent
covariance matrix per particle by re-encoding CTF-coloured-noise perturbations of
the clean reprojection. The suite therefore first trains small **setup** models
(a HetSIREN, ctf=apply; and a Zernike3Deep, ctf=None) and points the estimate runs
at them via `--nn_path`.

| Option / axis | Covered by |
|---|---|
| `--nn_path` HetSIREN vs Zernike3Deep | `estimate_hetsiren_*` / `estimate_zernike_ram` |
| model CTF "apply" branch vs non-apply branch | `estimate_hetsiren_*` / `estimate_zernike_ram` |
| `--batch_size` | varied (16 / 8) |
| `--load_images_to_ram` (on) / mmap (off) + `--ssd_scratch_folder` | `estimate_hetsiren_ram` / `estimate_hetsiren_mmap` |
| outputs `covariance_matrices.npy` + `latents.npy` | all estimate scenarios |

### latent_space_deconvolution coverage (`test_latent_space_deconvolution.py`)

`latent_space_deconvolution` now recovers the latents **on the fly** from a trained
network (`--nn_path`) applied to the images in `--md` (instead of a pre-saved
`latents.npy`), then deconvolves the landscape using per-particle covariances
(`--covariances`, from `estimate_latent_covariances`). The suite builds the full
pipeline as ordered scenarios: train a HetSIREN → estimate covariances → deconvolve
(train) → deconvolve (predict) → deconvolve (mmap path).

| Option / axis | Covered by |
|---|---|
| on-the-fly latents from `--nn_path` + `--md` | all `deconv_*` |
| `--mode train / predict` (+ `--reload`) | `deconv_train_*` / `deconv_predict` |
| `--covariances`, `--deconvolution_strength`, `--lat_dim`, `--batch_size` | varied across scenarios |
| `--load_images_to_ram` (on) / mmap (off) + `--ssd_scratch_folder` | `deconv_train_ram` / `deconv_train_mmap` |
| output: saved `deconvolver` + `latents_deconvolved.npy` | train / predict scenarios |

### filter_latents coverage (`test_filter_latents.py`)

`filter_latents` is a pure latent-space post-processing step (no network, no
images): it z-scores each vector by its mean distance to its `--n_neighbours`
nearest neighbours and keeps those below `--thr`. The phantom is a synthetic
latent cloud — a Gaussian bulk plus clear outliers — so filtering is deterministic
(the outliers are removed). A data check confirms the outliers are detectable.

| Option / axis | Covered by |
|---|---|
| `--latents` (input .npy) | all |
| `--thr`, `--n_neighbours` | `filter_default` (1.0 / 10) vs `filter_custom` (2.0 / 5) |
| `--return_ids` | `filter_return_ids` |
| `--batch_size` | `filter_custom` |
| output `filtered_latents.npy` | all |

(Minor, unpatched: the program does not create its `--output_path` — the suite
pre-creates it.)

### FlexConsensus coverage (`test_flexconsensus.py`)

FlexConsensus receives N latent spaces with the **same number of points** and an
ordered point-to-point correspondence, learns a common consensus space, and (at
predict) reports per-point consensus / representation errors revealing where the
spaces agree or disagree. The phantom is N synthetic spaces built from a shared,
ordered 1-D coordinate (a circle), each a different random linear embedding
(different dimensionality `4 / 5 / 6`) — agreeing everywhere except a contiguous
region where the first space follows a rotated (disagreeing) trend.

| Option / axis | Covered by |
|---|---|
| `--input_space` (N spaces, `NAME:path`) | all |
| `--lat_dim` default (min input dim) vs explicit | `train_default` / `train_custom` |
| `--mode train / predict` | `train_*` / `predict` |
| `--epochs`, `--batch_size`, `--learning_rate` | varied |
| outputs: `FlexConsensus` model; `*_consensus.npy`, `*_consensus_error.npy`, `*_representation_error.npy` | train / predict |

### decode_states_from_latents coverage (`test_decode_states_from_latents.py`)

`decode_states_from_latents` decodes a file of latent vectors (`.npy` or `.txt`)
into 3D volumes (`decoded_volume_XXXX.mrc`) using a trained network
(`--reload`)'s `decode_volume`. The suite trains small setup networks and feeds
them a synthetic set of latent vectors (centred near 0).

| Option / axis | Covered by |
|---|---|
| `--latents_file` `.npy` vs `.txt` | `decode_hetsiren_npy` / `decode_hetsiren_txt` |
| `--reload` HetSIREN vs Zernike3Deep (both expose `decode_volume`) | `decode_hetsiren_*` / `decode_zernike_npy` |
| output: one `decoded_volume_XXXX.mrc` per latent | all decode scenarios |

(Minor, unpatched: the program does not create its `--output_path` — the suite
pre-creates it; each decoded `.mrc` has a singleton leading axis `(1, box, box,
box)` since `decode_volume` keeps the batch dimension.)

### MoDART coverage (`test_modart.py`)

MoDART is an ART-style reconstruction (no train/predict modes): it reconstructs a
map (or two half maps) from the images + alignments in `--md`, running until
early-stopping. The phantom is the standard CTF dataset (+ its reference volume /
mask for the `--vol` scenario). MoDART creates its own `--output_path` via the
metrics writer, so no pre-creation is needed.

| Option / axis | Covered by |
|---|---|
| `--ctf_type None / apply / wiener / precorrect` | `recon_none_c2` / `recon_apply` / `recon_wiener` / `recon_precorrect` |
| `--symmetry_group` c1 (default) / c2 | most / `recon_none_c2` |
| `--reconstruct_halves` (two half maps) | `recon_halves` |
| `--vol` / `--mask` (refinement + VolumeAdjustment) | `recon_vol` |
| `--motion_correction` (from a trained HetSIREN) | `setup_hetsiren` + `recon_motion` |
| `--load_images_to_ram` (on) / mmap (off) + `--ssd_scratch_folder` | most / `recon_mmap` |
| outputs `modart_map.mrc` (+ `modart_first/second_half.mrc`) | all |

### annotate_space coverage (`test_annotate_space.py`, headless)

`annotate_space` is an interactive PyQt5 + napari desktop app (talks to ChimeraX
over a socket), so it can't be driven through the CLI like the batch programs —
its `main()` blocks on `QApplication.exec_()` and needs a display. Instead the
suite exercises its *headless, logic-bearing* components directly via a helper
script (`annotate_space_headless.py`), run as a subprocess per check. This is the
only program whose scenarios use `Scenario(script=...)` (run the script directly)
rather than the `hax.cli` entry point.

| Component | Headless check |
|---|---|
| Dim reduction — `DimRedQThread` (PCA + UMAP) | `dimred` |
| Clustering — `ClusteringQThread` (KMeans + along-dimension) | `clustering` |
| Socket protocol — `viewer_socket` `Server`<->`Client` round-trip (FromFiles map copy) | `socket` |
| Helpers — `getImagePath` / `getServerProgram` / ... | `utils` |
| **Tier 2** — napari viewer + the app's `MultipleViewerWidget` assembled under `QT_QPA_PLATFORM=offscreen` | `offscreen` |

The `offscreen` check is best-effort: it prints `SKIP` and still exits 0 if the
offscreen backend / napari aren't usable in the environment (so it never fails the
suite spuriously).

**Out of scope** (genuinely needs a GUI / ChimeraX / manual interaction, not
tested): the napari canvas and lasso/point selection, ChimeraX morphing
(`FlexMorphChimeraX`), and screenshots. The real-time map decode itself is the
same network path already covered by `decode_states_from_latents`.

### image_gray_scale_adjustment coverage (`test_image_gray_scale_adjustment.py`)

Learns a per-image gray-level adjustment matching projections of a reference
volume (`--vol`) to the images in `--md`. Train saves an `imageAdjustment` model;
predict writes `adjusted_images.mrcs` + a metadata file.

| Option / axis | Covered by |
|---|---|
| `--ctf_type None / apply / wiener` | `train_none` / `train_apply_pv` / `train_wiener_mmap` |
| `--mode train / predict` (+ `--reload`) | `train_*` / `predict_pv` |
| `--predict_value` on / off | `*_pv` / `train_none` |
| `--vol` (required) / `--mask` provided vs auto-generated | most / `train_automask` |
| `--load_images_to_ram` (on) / mmap (off) + `--ssd_scratch_folder` | most / `train_wiener_mmap` |
| `--lat_dim`, `--batch_size`, `--learning_rate`, `--dataset_split_fraction` | varied |
| outputs `imageAdjustment`; `adjusted_images.mrcs` + `.xmd` | train / predict |

### volume_gray_scale_adjustment coverage (`test_volume_gray_scale_adjustment.py`)

Learns a gray-level adjustment for a reference volume so its projections match the
images in `--md`. Train saves a `volumeAdjustment` model; predict writes
`adjusted_volume.mrc` (predict needs only `--vol` + `--reload`).

| Option / axis | Covered by |
|---|---|
| `--ctf_type None / apply / wiener` | `train_none_novpv` / `train_apply_pv` / `train_wiener_mmap` |
| `--mode train / predict` (+ `--reload`) | `train_*` / `predict_*` |
| `--predicts_value` on (a*vol+b) / off (voxel values) | `predict_pv` / `predict_novpv` |
| `--vol` (required) / `--mask` | all |
| `--load_images_to_ram` (on) / mmap (off) + `--ssd_scratch_folder` | most / `train_wiener_mmap` |
| `--lat_dim`, `--batch_size`, `--learning_rate`, `--dataset_split_fraction` | varied |
| outputs `volumeAdjustment`; `adjusted_volume.mrc` | train / predict |

Both programs share two pre-existing, **unpatched** quirks (worked around by the
suites): the end-of-training `rmtree(..._CHECKPOINT)` is unconditional (but a
checkpoint is written at `i==1`, so `epochs >= 2` is safe), and `predict` /
no-`--mask` training do not create `--output_path`. `volume_gray_scale_adjustment`
additionally has a resume-only copy-paste typo (reads `imageAdjustment_CHECKPOINT`),
not exercised here.

### display_metrics coverage (`test_display_metrics.py`, headless)

`display_metrics` is a convenience launcher: it starts TensorBoard on a `--logdir`
and blocks until Ctrl+C, so it never returns and cannot be driven to completion
through the CLI like the batch programs. As with `annotate_space`, the suite drives
its logic-bearing parts through a helper script (`display_metrics_headless.py`),
one subprocess per check.

| Check | What it exercises |
|---|---|
| `writer` | `JaxSummaryWriter` logs **JAX-array** scalars (auto-converted to numpy by its `__getattribute__` wrapper) + `add_volumes_slices` (jitted MAD/mean, low-pass, color map, central slices); asserts a `tfevents` file is written |
| `launch` | replicates `main()`'s `tensorboard.program` launch in-process and HTTP-probes the returned URL (the server actually serves the run) |
| `cli` | runs the genuine `display_metrics --logdir` entry point through `hax.cli:main`, waits for the "TensorBoard is running at `<url>`" line, HTTP-probes it, then sends `SIGINT` and asserts the clean "Received Ctrl+C" shutdown |

Out of scope (genuinely needs a browser): visually inspecting the TensorBoard UI.

## Adding another program

Create `tests/test_<program>.py` exposing:

* `prepare_data(workdir)` → returns whatever the scenarios need (and generates
  the phantom data on disk);
* `data_checks(workdir)` *(optional)* → list of `(name, ok, detail)`;
* `scenarios(workdir, data)` → list of `common.Scenario`;
* `SLOW` *(optional)* → set of scenario names to skip under `--quick`.

Then add `"<program>"` to `PROGRAMS` in `run_tests.py`.
